"""会话记忆（多轮追问的基础）：两种后端，按配置自动选。

```text
SESSION_BACKEND=auto（默认）+ REDIS_URL 已配 → RedisSessionStore
否则                                        → SessionStore（进程内 + TTL）
```

## 为什么要有 Redis 版本

进程内 dict 在单副本、演示场景够用，但一旦出现这三种情况就出问题：
- **多副本 / 滚动发布**：用户的第 2 句可能落到另一个进程，历史读不到，
  「那其他城市呢？」就变成一句没有上下文的新问题；
- **重启**：会话全丢，用户得从头讲一遍；
- **水平扩容**：副本数越多，命中同一副本的概率越低，追问体验越随机。

这些不是性能问题，是**正确性**问题，所以会话必须外置。

## 数据结构：用 List，不用 String(JSON)

想当然的做法是把整个历史序列化成 JSON 存一个 String，每次读-改-写。它有两个坑：
1. 并发追加会互相覆盖（两个请求同时读到旧值，各自写回，丢一条）；
2. 每次追加都是全量读写，历史越长越慢。

这里用 Redis List：`LPUSH` 追加、`LTRIM` 截断、`EXPIRE` 续期，一条 pipeline 搞定。
三个操作都是 O(1)，而且不需要先读出来（没有读-改-写，也就没有丢更新）。

## 降级不是「假装成功」

Redis 连不上时**自动回落进程内存储**，但会打 warning 并熔断 30 秒——
否则每个请求都要等一次连接超时，一个挂掉的 Redis 能把整个服务拖慢。
回落后历史会短暂分叉（故障前的在 Redis、故障中的在内存），这是可接受的取舍：
宁可体验降级，也不要整个问答不可用。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable

from ..config import settings
from ..core.logging import get_logger

log = get_logger(__name__)

# Redis 故障后的熔断窗口：这段时间内直接走内存，不再尝试连接
_FAIL_BACKOFF_SECONDS = 30.0
_UNSET = object()


class SessionStore:
    """进程内实现：单副本够用，零依赖，也是 Redis 不可用时的兜底。"""

    backend = "memory"

    def __init__(self, max_turns: int | None = None, ttl_minutes: int | None = None) -> None:
        self.max_turns = max_turns or settings.max_history_turns
        self.ttl = (ttl_minutes or settings.session_ttl_minutes) * 60
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._data.items() if now - v["updated"] > self.ttl]
        for k in expired:
            self._data.pop(k, None)

    def history(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        with self._lock:
            self._gc()
            item = self._data.get(session_id)
            return list(item["messages"]) if item else []

    def append(self, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        with self._lock:
            item = self._data.setdefault(
                session_id,
                {"messages": deque(maxlen=self.max_turns * 2), "updated": time.time()},
            )
            item["messages"].append({"role": role, "content": content})
            item["updated"] = time.time()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"sessions": len(self._data), "backend": self.backend}

    def health(self) -> bool:
        return True

    def close(self) -> None:
        with self._lock:
            self._data.clear()


class RedisSessionStore(SessionStore):
    """Redis 实现：多副本共享、重启不丢。

    继承 `SessionStore` 是为了复用 `new_id()` 与内存兜底（`_fallback`），
    读写方法全部覆写。
    """

    backend = "redis"

    def __init__(
        self,
        url: str,
        *,
        max_turns: int | None = None,
        ttl_minutes: int | None = None,
        prefix: str | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(max_turns=max_turns, ttl_minutes=ttl_minutes)
        self.prefix = prefix or settings.session_redis_prefix
        self.url_masked = _mask_url(url)
        self._down_until = 0.0
        # 兜底：Redis 故障期间的写入落在本地，恢复后新历史回 Redis
        self._fallback = SessionStore(max_turns=self.max_turns, ttl_minutes=self.ttl // 60)
        self._client = client if client is not None else _make_client(url)

    # ---------- 内部：带熔断的调用 ----------

    def _call(self, fn: Callable[[], Any]) -> Any:
        if time.time() < self._down_until:
            return _UNSET
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            self._down_until = time.time() + _FAIL_BACKOFF_SECONDS
            log.warning("Redis 会话不可用，临时降级到内存（%.0fs 后重试）: %s",
                        _FAIL_BACKOFF_SECONDS, exc)
            return _UNSET

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    # ---------- 读写 ----------

    def history(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        limit = self.max_turns * 2

        def op() -> list[str]:
            return self._client.lrange(self._key(session_id), 0, limit - 1)

        raw = self._call(op)
        if raw is _UNSET:
            return self._fallback.history(session_id)

        items: list[dict[str, str]] = []
        for text in raw:
            try:
                items.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                # 坏数据不该让整个会话报废：跳过并留痕
                log.warning("会话 %s 里有无法解析的消息，已跳过", session_id)
        items.reverse()               # LPUSH 是新的在前，对外要「旧的在前」
        return items

    def append(self, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        payload = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        limit = self.max_turns * 2 - 1

        def op() -> None:
            pipe = self._client.pipeline()
            pipe.lpush(self._key(session_id), payload)
            pipe.ltrim(self._key(session_id), 0, limit)   # 只留最近 N 轮，别让 prompt 无限长
            pipe.expire(self._key(session_id), self.ttl)  # 每次写入续期
            pipe.execute()

        if self._call(op) is _UNSET:
            self._fallback.append(session_id, role, content)

    def reset(self, session_id: str) -> None:
        def op() -> None:
            self._client.delete(self._key(session_id))

        self._call(op)
        self._fallback.reset(session_id)   # 本地兜底里可能也有这一轮

    def stats(self) -> dict[str, Any]:
        pattern = f"{self.prefix}*"

        def op() -> int:
            count = 0
            for _ in self._client.scan_iter(match=pattern, count=200):
                count += 1
                if count >= 5000:      # 统计不值得拖垮服务，够用就停
                    break
            return count

        total = self._call(op)
        if total is _UNSET:
            return {**self._fallback.stats(), "backend": "memory(fallback)"}
        return {"sessions": int(total), "backend": self.backend}

    def health(self) -> bool:
        return self._call(lambda: bool(self._client.ping())) is not _UNSET

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        self._fallback.close()


def _mask_url(url: str) -> str:
    """日志里不打印密码：redis://:pass@host/0 → redis://:***@host/0"""
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    if ":" in head.rsplit("/", 1)[-1]:
        head = head.rsplit(":", 1)[0] + ":***"
    return f"{head}@{tail}"


def _make_client(url: str) -> Any:
    import redis  # 局部导入：没装 redis 时内存后端仍可用

    return redis.Redis.from_url(
        url,
        decode_responses=True,
        # 显式用 RESP2：redis-py 8.x 默认 RESP3（先发 HELLO 3 握手），
        # 而 Redis 5.x（Windows 移植版、不少老生产环境）没有 HELLO 命令，
        # 会直接报 "unknown command `HELLO`"。我们只用到 List/TTL，RESP2 完全够。
        protocol=2,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        health_check_interval=30,
    )


def build_session_store() -> SessionStore:
    """按配置选后端。选不到就诚实回退，不静默假装在用 Redis。"""
    backend = (settings.session_backend or "auto").strip().lower()
    url = (settings.redis_url or "").strip()

    if backend == "memory" or not url:
        if backend == "redis":
            log.error("SESSION_BACKEND=redis 但没配 REDIS_URL，已回退进程内会话")
        return SessionStore()

    store = RedisSessionStore(url)
    if store.health():
        log.info("会话后端: Redis（%s），前缀 %s", store.url_masked, store.prefix)
        return store
    log.warning("会话后端: Redis 连接失败（%s），本次运行使用进程内会话", store.url_masked)
    store.close()
    return SessionStore()


session_store: SessionStore = build_session_store()
