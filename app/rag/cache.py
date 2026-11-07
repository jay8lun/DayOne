"""答案缓存：同一问题在知识库未变更时，直接复用上一次的回答。

## 为什么在这里缓存，而不是在 LLM 层

RAG 一次问答的成本分布是「condense 改写 → embedding + BM25 检索 → 生成」，
其中生成占大头、检索占次之。缓存必须罩住整条链路才有意义，所以入口放在
`RagService.ask` 的检索之前，命中就直接返回，一次模型都不用调。

## 三条硬约束（不做就会出事）

1. **只缓存第一轮**。追问的语义依赖上文（「那其他城市呢」），同一个句子在不同
   上下文里答案不同，拿它当 key 会串答案。第一轮的问句自包含，安全。
2. **key 里带知识库指纹**。文档一改，`kb_version` 变，旧 key 自然没人再读，
   不需要写任何清理逻辑——这比「先删缓存再重建」可靠得多。
3. **编排 / 记忆 / 工具路径不进缓存**。它们的答案取决于提问者本人的档案和工单，
   张三的 P5 标准不能给李四。

## 降级

Redis 不可用就当作「永远 miss」：不写、不读、不报错，只是失去加速。缓存是优化，
不能变成新的故障点，所以这里没有熔断之外的任何重试。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from ..config import settings
from ..core.logging import get_logger

log = get_logger(__name__)

# KB 指纹 + key 计算是每个请求都要做的，用 mtime 判断是否要重新读状态文件
_KB_CACHE: tuple[tuple[int, int, str], str] | None = None
_DOWN_BACKOFF_SECONDS = 30.0


def _embedder_tag() -> str:
    """当前 embedding 模型的指纹（`provider__model__dim`），和向量集合名用的是同一个。

    **换模型必须让缓存失效**：换 embedding 会建一个新集合，但文档内容哈希一个都没变，
    只按内容算指纹的话，旧缓存会继续命中「用旧模型检索出来」的答案——
    不报错、不告警，看起来完全正常。这类静默错误比崩溃危险。
    """
    try:
        from ..ingest.embedder import get_embedder

        return get_embedder().tag
    except Exception as exc:  # noqa: BLE001 —— 拿不到指纹不该让问答挂掉
        log.debug("拿不到 embedding 指纹，缓存按内容失效: %s", exc)
        return "unknown"


def kb_version() -> str:
    """知识库指纹：文档内容哈希 + embedding 模型指纹，聚合后取前 10 位。

    只 stat 一次文件（纳秒级），内容没变就直接用上次结果——不解析 JSON。
    """
    global _KB_CACHE
    path = settings.resolve(settings.ingest_state_file)
    try:
        st = path.stat()
    except OSError:
        return "nokb"

    # 先确认有知识库再取模型指纹：没知识库时不必初始化 embedder
    tag = _embedder_tag()
    stamp = (st.st_mtime_ns, st.st_size, tag)
    if _KB_CACHE and _KB_CACHE[0] == stamp:
        return _KB_CACHE[1]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        hashes = sorted(str(d.get("hash") or "") for d in (data.get("documents") or {}).values())
        version = hashlib.sha1(("|".join(hashes) + "|" + tag).encode()).hexdigest()[:10]
    except Exception:  # noqa: BLE001 —— 状态文件坏了不该让问答挂掉
        version = "unparsed"
    _KB_CACHE = (stamp, version)
    return version


def normalize(question: str) -> str:
    """归一化：压平空白、统一小写。

    「3 个月」和「3个月」、「P0」和「p0」是同一个问题，不该各占一份缓存。
    这和评测里 `keyword_coverage` 的口径是同一个道理：机器比对要贴近人的直觉。
    """
    return re.sub(r"\s+", "", (question or "")).strip().lower()


class AnswerCache:
    """Redis 优先、进程内兜底的答案缓存。

    进程内那一层不是「降级方案」而是「无 Redis 时的等价实现」：
    单副本调试时也能跑出命中率，换 Redis 只改连接串。
    """

    def __init__(
        self,
        client: Any = None,
        *,
        ttl_seconds: int | None = None,
        prefix: str | None = None,
    ) -> None:
        self._client = client
        self.ttl = int(ttl_seconds or settings.cache_ttl_seconds)
        self.prefix = prefix or settings.cache_redis_prefix
        self._local: dict[str, tuple[float, str]] = {}
        self._down_until = 0.0
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self.backend = "redis" if client is not None else "memory"

    # ---------- key ----------
    def key(self, question: str) -> str:
        digest = hashlib.sha1(normalize(question).encode("utf-8")).hexdigest()
        return f"{self.prefix}{kb_version()}:{digest}"

    @property
    def enabled(self) -> bool:
        return bool(settings.cache_enabled)

    # ---------- 读写 ----------
    def get(self, question: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        k = self.key(question)
        raw: str | None = None

        if self._client is not None and time.time() >= self._down_until:
            try:
                raw = self._client.get(k)
            except Exception as exc:  # noqa: BLE001
                self._down_until = time.time() + _DOWN_BACKOFF_SECONDS
                log.warning("答案缓存不可用，%.0fs 内直接跳过缓存: %s", _DOWN_BACKOFF_SECONDS, exc)

        if raw is None:
            item = self._local.get(k)
            if item and item[0] > time.time():
                raw = item[1]

        if not raw:
            self._misses += 1
            return None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # 坏数据当 miss：不能让一条脏缓存把问题永久卡住
            self._misses += 1
            return None

        self._hits += 1
        data["_cached_at"] = data.get("_cached_at") or 0
        return data

    def set(self, question: str, payload: dict[str, Any]) -> bool:
        """写入缓存。返回是否真的写进去了（便于测试与统计）。"""
        if not self.enabled:
            return False
        try:
            body = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # 检索 trace 里可能有不可序列化的对象：宁可少缓存，也不能让问答 500
            log.debug("答案无法序列化，跳过缓存: %s", exc)
            return False
        if len(body.encode("utf-8")) > settings.cache_max_bytes:
            log.debug("答案过大（%d 字节），跳过缓存", len(body))
            return False

        k = self.key(question)
        # 兜底层无论如何都写：Redis 抖动时还有个本地缓存可用
        self._local[k] = (time.time() + self.ttl, body)
        if len(self._local) > 512:  # 简易上限，避免调试时无限涨
            oldest = sorted(self._local, key=lambda x: self._local[x][0])[:128]
            for gone in oldest:
                self._local.pop(gone, None)

        if self._client is not None and time.time() >= self._down_until:
            try:
                self._client.set(k, body, ex=self.ttl)
            except Exception as exc:  # noqa: BLE001
                self._down_until = time.time() + _DOWN_BACKOFF_SECONDS
                log.warning("写缓存失败，%.0fs 内不再尝试: %s", _DOWN_BACKOFF_SECONDS, exc)

        self._sets += 1
        return True

    # ---------- 观测 ----------
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": self.backend,
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._sets,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "entries_local": len(self._local),
            # 指纹进统计：排查「为什么突然全 miss」时第一眼看它变没变
            "kb_version": kb_version(),
            "ttl_seconds": self.ttl,
            # 命中率按进程统计：多副本时是「本进程看到的」，不是全局
            "scope": "process",
        }

    def reset_stats(self) -> None:
        self._hits = self._misses = self._sets = 0

    def clear(self) -> int:
        """清空本实例的缓存（测试与运维用）。返回删掉的条数。"""
        n = len(self._local)
        self._local.clear()
        if self._client is not None:
            try:
                keys = list(self._client.scan_iter(match=f"{self.prefix}*", count=200))
                if keys:
                    n += len(keys)
                    self._client.delete(*keys)
            except Exception as exc:  # noqa: BLE001
                log.warning("清理缓存失败: %s", exc)
        return n


# --------------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------------- #
_cache: AnswerCache | None = None


def _make_client(url: str) -> Any:
    """和会话共用一套连接参数：RESP2（兼容 Redis 5.x）、短超时。

    socket_timeout 必须短：缓存命中是「省时间」的优化，不能因为等 Redis
    把响应拖慢——超时立刻当 miss 走过去。
    """
    import redis

    return redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=0.3,
        socket_connect_timeout=0.3,
        health_check_interval=30,
        protocol=2,
    )


def get_answer_cache() -> AnswerCache:
    global _cache
    if _cache is not None:
        return _cache

    url = (settings.redis_url or "").strip()
    backend = (settings.session_backend or "auto").strip().lower()
    client = None
    if url and backend != "memory" and settings.cache_enabled:
        try:
            client = _make_client(url)
            client.ping()
        except Exception as exc:  # noqa: BLE001 —— 连不上就用内存层，别挡住启动
            log.info("答案缓存未接 Redis（%s），使用进程内缓存", exc)
            client = None
    _cache = AnswerCache(client)
    return _cache


def reset_answer_cache() -> None:
    global _cache
    _cache = None
