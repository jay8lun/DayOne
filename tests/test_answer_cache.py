"""答案缓存：key 稳定性、失效、降级，以及「什么时候不许缓存」。

缓存最容易出的不是「没命中」，是**命中了不该命中的**：
追问串答案、编排把个人答案给别人、文档更新后还在答旧标准。
所以这里的负例比正例多。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.rag import cache as cachemod  # noqa: E402
from app.rag.cache import AnswerCache, kb_version, normalize  # noqa: E402
from app.rag.retriever import RetrievalResult, RetrievedChunk  # noqa: E402
from app.rag.sessions import SessionStore  # noqa: E402


# --------------------------------------------------------------------------- #
# 假 Redis：只实现我们用到的那几个命令
# --------------------------------------------------------------------------- #
class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.data[key] = value
        if ex:
            self.ttls[key] = ex
        return True

    def ping(self):
        return True

    def scan_iter(self, match="*", count=200):
        prefix = match.rstrip("*")
        return (k for k in list(self.data) if k.startswith(prefix))

    def delete(self, *keys):
        n = 0
        for k in keys:
            n += 1 if self.data.pop(k, None) is not None else 0
        return n


@pytest.fixture()
def cache(monkeypatch):
    monkeypatch.setattr(cachemod, "kb_version", lambda: "testkb")
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_ttl_seconds", 600)
    return AnswerCache(client=FakeRedis(), ttl_seconds=600, prefix="dayone:ans:")


# --------------------------------------------------------------------------- #
# key
# --------------------------------------------------------------------------- #
def test_normalize_flattens_space_and_case():
    assert normalize("  P0  故障 ") == normalize("p0故障") == "p0故障"
    assert normalize("3 个月") == normalize("3个月")


def test_key_is_stable_across_whitespace(cache):
    a = cache.key("报销单多久内要提交？")
    b = cache.key("报销单  多久内要提交？ ")
    assert a == b
    assert a.startswith("dayone:ans:testkb:")


def test_key_changes_when_kb_version_changes(cache, monkeypatch):
    before = cache.key("报销单多久内要提交？")
    monkeypatch.setattr(cachemod, "kb_version", lambda: "newkb")
    after = cache.key("报销单多久内要提交？")
    assert before != after


def test_kb_version_tracks_state_file(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"documents": {"a": {"hash": "aaa"}, "b": {"hash": "bbb"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ingest_state_file", str(state))
    monkeypatch.setattr(cachemod, "_embedder_tag", lambda: "fixed-tag")
    monkeypatch.setattr(cachemod, "_KB_CACHE", None)

    v1 = kb_version()
    assert v1 and v1 != "nokb"

    # 内容变 → 指纹变（缓存自然失效，不需要写清理逻辑）
    state.write_text(
        json.dumps({"documents": {"a": {"hash": "aaa"}, "b": {"hash": "CHANGED"}}}),
        encoding="utf-8",
    )
    assert kb_version() != v1


def test_kb_version_includes_embedder_tag(tmp_path, monkeypatch):
    """换 embedding 模型也必须让缓存失效。

    换模型会建一个新集合，但文档内容哈希一个都没变——只按内容算指纹的话，
    旧缓存会继续命中「用旧模型检索出来」的答案，不报错也不告警。
    这是本项目真实修过的一个静默 bug。
    """
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"documents": {"a": {"hash": "aaa"}}}), encoding="utf-8")
    monkeypatch.setattr(settings, "ingest_state_file", str(state))

    monkeypatch.setattr(cachemod, "_embedder_tag", lambda: "dashscope__text-embedding-v3__1024")
    monkeypatch.setattr(cachemod, "_KB_CACHE", None)
    before = kb_version()

    monkeypatch.setattr(cachemod, "_embedder_tag", lambda: "local__bge-m3__1024")
    assert kb_version() != before


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def test_set_then_get_roundtrip(cache):
    assert cache.get("报销单多久内要提交？") is None
    assert cache.set("报销单多久内要提交？", {"answer": "60 天内。"}) is True

    hit = cache.get("报销单多久内要提交？")
    assert hit is not None
    assert hit["answer"] == "60 天内。"
    assert cache.stats()["hits"] == 1


def test_expired_local_entry_is_miss(cache):
    """本地层要自己判断过期，不能指望 Redis 的 TTL 兜底。"""
    cache.set("年假几天", {"answer": "5 天"})
    key = cache.key("年假几天")
    cache._local[key] = (time.time() - 1, cache._local[key][1])  # 手动改成已过期
    cache._client.data.pop(key, None)  # Redis 侧也当作已清
    assert cache.get("年假几天") is None


def test_corrupted_payload_counts_as_miss(cache):
    """Redis 里的脏数据应当被当作 miss，而不是让问答直接 500。"""
    key = cache.key("年假几天")
    cache._client.data[key] = "{not json"
    assert cache.get("年假几天") is None


def test_oversized_answer_skipped(monkeypatch):
    monkeypatch.setattr(cachemod, "kb_version", lambda: "kb")
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_max_bytes", 64)
    c = AnswerCache(client=FakeRedis(), ttl_seconds=600, prefix="p:")
    assert c.set("问题", {"answer": "长" * 200}) is False
    assert c.get("问题") is None


def test_unserializable_payload_skipped(cache):
    class Weird:
        pass

    assert cache.set("问题", {"answer": Weird()}) is False


def test_disabled_cache_does_nothing(cache, monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    assert cache.set("问题", {"answer": "x"}) is False
    assert cache.get("问题") is None


# --------------------------------------------------------------------------- #
# 降级
# --------------------------------------------------------------------------- #
def test_redis_failure_get_is_miss_when_local_cold(cache):
    """Redis 挂了且本地没数据：当 miss，绝不能把异常抛进问答链。"""
    cache._client.fail = True
    assert cache.get("没人问过的问题") is None


def test_redis_failure_still_serves_local_layer(cache):
    """Redis 写失败时本地层已经写进去了——同一副本内仍能命中，不白算一次。"""
    cache._client.fail = True
    assert cache.set("问题", {"answer": "x"}) is True
    hit = cache.get("问题")
    assert hit is not None and hit["answer"] == "x"


def test_local_layer_still_works_without_redis(monkeypatch):
    monkeypatch.setattr(cachemod, "kb_version", lambda: "kb")
    monkeypatch.setattr(settings, "cache_enabled", True)
    c = AnswerCache(client=None)  # 没有 Redis
    c.set("年假几天", {"answer": "5 天"})
    assert c.get("年假几天")["answer"] == "5 天"
    assert c.backend == "memory"


def test_stats_shape(cache):
    cache.set("a", {"answer": "1"})
    cache.get("a")
    cache.get("b")
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert s["backend"] == "redis"


# --------------------------------------------------------------------------- #
# 服务层：什么时候可以缓存
# --------------------------------------------------------------------------- #
class FakeGraph:
    async def ainvoke(self, state):
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="消费发生后 60 天内提交报销单，逾期系统自动关闭。",
            metadata={"doc_id": "d1", "doc_title": "报销、差旅与采购流程", "heading_path": "报销基本规则"},
        )
        result = RetrievalResult(
            query=state["question"], chunks=[chunk], bm25_top=12.0,
            coverage=0.8, evidence=0.7, confidence=0.9, trace={"mode": "fake"},
        )
        return {"query": state["question"], "result": result, "sufficient": True, "trace": []}


class FakeLLM:
    provider = "fake"
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        return "报销需在 60 天内提交。[[1]]"

    async def stream(self, messages, **kwargs):
        self.calls += 1
        for piece in ("报销需在 ", "60 天内提交。[[1]]"):
            yield piece


class FakeRetriever:
    """只为满足构造签名：graph 被替换掉了，检索器不会真的被调用。"""


@pytest.fixture()
def service(monkeypatch):
    from app.rag.service import RagService

    monkeypatch.setattr(cachemod, "kb_version", lambda: "svckb")
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "enable_context_budget", False)

    llm = FakeLLM()
    svc = RagService(retriever=FakeRetriever(), llm=llm,
                     sessions=SessionStore(max_turns=6, ttl_minutes=60))
    svc.graph = FakeGraph()

    async def _none(*args, **kwargs):
        return None

    # 编排 / 记忆 / 工具三条旁路都不在本测试范围，否则要造一堆 fixture
    monkeypatch.setattr(svc, "_orchestrated_answer", _none)
    monkeypatch.setattr(svc, "_memory_answer", _none)
    monkeypatch.setattr(svc, "_tool_answer", _none)
    cachemod.reset_answer_cache()
    cachemod._cache = AnswerCache(client=FakeRedis(), ttl_seconds=600, prefix="dayone:ans:")
    return svc, llm


def test_second_identical_question_hits_cache(service):
    svc, llm = service

    async def scenario():
        first = await svc.ask("报销单多久内要提交？", session_id="a")
        calls_after_first = llm.calls
        # 另一个会话（第一轮）问同一问题：知识库没变，直接复用
        second = await svc.ask("报销单多久内要提交？", session_id="b")
        return first, second, calls_after_first

    first, second, calls_after_first = asyncio.run(scenario())
    assert first["answer_source"] == "knowledge_base"
    assert not first.get("cached")
    assert second["cached"] is True
    assert second["answer_source"] == "cache"
    assert second["answer"] == first["answer"]
    assert llm.calls == calls_after_first  # 一次模型都没调
    # 引用、有据率必须原样带回，否则前端角标会空
    assert second["sources"] == first["sources"]
    assert second["grounding_score"] == first["grounding_score"]


def test_followup_is_never_cached(service):
    svc, _ = service

    async def scenario():
        await svc.ask("报销单多久内要提交？", session_id="s")
        # 同一会话再问：有历史 → 不可缓存，必须重新走检索生成
        return await svc.ask("报销单多久内要提交？", session_id="s")

    assert not asyncio.run(scenario()).get("cached")


def test_cache_hit_still_writes_history(service):
    """命中也要落会话历史，否则下一句追问就断链。"""
    svc, _ = service

    async def scenario():
        await svc.ask("报销单多久内要提交？", session_id="x")
        await svc.ask("报销单多久内要提交？", session_id="y")
        return svc.sessions.history("y")

    history = asyncio.run(scenario())
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"


def test_refusal_is_cached_but_recoverable(service):
    """拒答也缓存：重复问同一个不存在的问题很常见，结论稳定。"""
    svc, _ = service

    class EmptyGraph(FakeGraph):
        async def ainvoke(self, state):
            return {
                "query": state["question"],
                "result": RetrievalResult(query=state["question"], chunks=[], trace={}),
                "sufficient": False,
                "trace": [],
            }

    svc.graph = EmptyGraph()

    async def scenario():
        first = await svc.ask("食堂几点开饭？", session_id="r1")
        second = await svc.ask("食堂几点开饭？", session_id="r2")
        return first, second

    first, second = asyncio.run(scenario())
    assert first["refused"] is True
    assert second["cached"] is True and second["refused"] is True


def test_orchestrated_answer_never_cached(service):
    """编排类答案依赖个人档案，写都不许写进缓存。"""
    svc, _ = service
    payload = {"answer": "你是 P5，600 元/晚", "answer_source": "orchestrated"}
    svc._cache_store("我住宿能报多少", payload, cacheable=True)
    assert cachemod.get_answer_cache().get("我住宿能报多少") is None


def test_kb_version_change_invalidates(service, monkeypatch):
    svc, llm = service

    async def first_round():
        await svc.ask("报销单多久内要提交？", session_id="k1")

    asyncio.run(first_round())
    calls = llm.calls

    # 知识库更新（指纹变了）→ 旧缓存不再命中，必须重新生成
    monkeypatch.setattr(cachemod, "kb_version", lambda: "kb-after-update")
    r = asyncio.run(svc.ask("报销单多久内要提交？", session_id="k2"))
    assert not r.get("cached")
    assert llm.calls > calls
