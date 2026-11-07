"""Redis 会话存储的测试。

不连真 Redis：用一个内存假客户端模拟 List / Pipeline / TTL 语义。
要验的是**我们的逻辑**：截断、续期、多实例共享、故障降级、坏数据不炸。

真 Redis 的连通性由 `python scripts/check_redis.py` 和 docker compose 覆盖。
"""

from __future__ import annotations

import asyncio

from app.rag.sessions import RedisSessionStore, SessionStore, _mask_url


class FakePipe:
    def __init__(self, client: "FakeRedis") -> None:
        self.client = client
        self.ops: list[tuple[str, tuple]] = []

    def lpush(self, key, value):
        self.ops.append(("lpush", (key, value)))
        return self

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", (key, start, end)))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", (key, ttl)))
        return self

    def execute(self):
        for name, args in self.ops:
            getattr(self.client, name)(*args)
        self.ops.clear()


class FakeRedis:
    """只实现我们用到的命令，行为对齐 redis-py。"""

    def __init__(self) -> None:
        self.data: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False
        self.calls: list[str] = []

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("fake: redis down")

    def lpush(self, key, value):
        self._check()
        self.calls.append("lpush")
        self.data.setdefault(key, []).insert(0, value)

    def lrange(self, key, start, end):
        self._check()
        values = self.data.get(key, [])
        end = len(values) - 1 if end == -1 else end
        return values[start : end + 1]

    def ltrim(self, key, start, end):
        self._check()
        self.data[key] = self.data.get(key, [])[start : end + 1]

    def expire(self, key, ttl):
        self._check()
        self.ttls[key] = ttl

    def delete(self, key):
        self._check()
        self.data.pop(key, None)
        self.ttls.pop(key, None)

    def pipeline(self):
        return FakePipe(self)

    def scan_iter(self, match=None, count=100):
        self._check()
        prefix = (match or "*").rstrip("*")
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    def ping(self):
        self._check()
        return True

    def close(self):
        pass


def _store(client: FakeRedis, **kw) -> RedisSessionStore:
    return RedisSessionStore("redis://localhost:6379/0", client=client, **kw)


def test_append_and_history_keep_order():
    client = FakeRedis()
    store = _store(client, max_turns=3)
    store.append("s1", "user", "一线城市住宿多少")
    store.append("s1", "assistant", "450 元/晚[[1]]")
    store.append("s1", "user", "那其他城市呢")

    history = store.history("s1")
    assert [h["role"] for h in history] == ["user", "assistant", "user"]
    assert history[0]["content"].startswith("一线城市")


def test_history_is_trimmed_to_max_turns():
    client = FakeRedis()
    store = _store(client, max_turns=2)          # 2 轮 = 4 条
    for i in range(6):
        store.append("s1", "user", f"第{i}句")
    assert len(store.history("s1")) == 4
    # 留下的是最近的，不是最早的
    assert store.history("s1")[-1]["content"] == "第5句"


def test_each_append_refreshes_ttl():
    client = FakeRedis()
    store = _store(client, ttl_minutes=120)
    store.append("s1", "user", "x")
    assert client.ttls["dayone:sess:s1"] == 120 * 60


def test_two_instances_share_one_session():
    """核心价值：换一个进程也能读到同一段历史（多副本 / 滚动发布）。"""
    client = FakeRedis()
    a = _store(client)
    b = _store(client)

    a.append("s1", "user", "一线城市住宿多少")
    a.append("s1", "assistant", "450 元/晚[[1]]")

    assert [h["content"] for h in b.history("s1")][0] == "一线城市住宿多少"


def test_reset_removes_session():
    client = FakeRedis()
    store = _store(client)
    store.append("s1", "user", "x")
    store.reset("s1")
    assert store.history("s1") == []
    assert "dayone:sess:s1" not in client.data


def test_bad_payload_is_skipped_not_fatal():
    client = FakeRedis()
    store = _store(client)
    store.append("s1", "user", "正常消息")
    client.data["dayone:sess:s1"].append("{不是 json")   # 模拟脏数据

    history = store.history("s1")
    assert len(history) == 1 and history[0]["content"] == "正常消息"


def test_failure_falls_back_to_memory_and_recovers():
    client = FakeRedis()
    store = _store(client)

    client.fail = True
    store.append("s1", "user", "故障期间的一句")
    assert store.history("s1")[0]["content"] == "故障期间的一句"   # 落在内存兜底
    assert store.health() is False

    # 熔断窗口内不再尝试连接（避免每个请求都等超时）
    before = len(client.calls)
    store.history("s1")
    assert len(client.calls) == before


def test_stats_reports_backend():
    client = FakeRedis()
    store = _store(client)
    store.append("a", "user", "x")
    store.append("b", "user", "y")
    stats = store.stats()
    assert stats["backend"] == "redis" and stats["sessions"] == 2


def test_memory_backend_still_works():
    """没配 Redis 的老路径不能被改坏。"""
    store = SessionStore(max_turns=2, ttl_minutes=1)
    store.append("s1", "user", "hi")
    assert store.history("s1")[0]["content"] == "hi"
    assert store.stats()["backend"] == "memory"


def test_url_password_is_masked():
    assert _mask_url("redis://:secret@10.0.0.1:6379/0") == "redis://:***@10.0.0.1:6379/0"
    assert _mask_url("redis://127.0.0.1:6379/0") == "redis://127.0.0.1:6379/0"


def test_build_session_store_falls_back_without_url(monkeypatch):
    from app.rag import sessions as mod

    monkeypatch.setattr(mod.settings, "session_backend", "redis")
    monkeypatch.setattr(mod.settings, "redis_url", "")
    store = mod.build_session_store()
    assert store.backend == "memory"
