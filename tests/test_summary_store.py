"""多级摘要入库的测试。

这里守的是「入库」带来的三个承诺：
1. 同一段历史**只压一次**——第二次请求不该再调模型；
2. 有新增轮次时**只压增量**，再把增量并进已有摘要；
3. 任何一步失败都**不能丢历史**（降级拼接、抽取式兜底），也不能把主链路搞挂。
"""

from __future__ import annotations

import asyncio

import pytest

from app.context import ContextBuilder
from app.context.summary_store import SummaryStore, reset_summary_store


class CountingLLM:
    """记账的假模型：能看出「这次到底压了几次」。"""

    provider = "counting"
    model = "counting-1"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def complete(self, messages, **kwargs) -> str:
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("stub: 模型挂了")
        if "合并" in prompt:
            return "合并摘要：导师是王琳；一线城市住宿 450 元。"
        return "段摘要：用户问过住宿标准，答 450 元。"


class FakeChunk:
    def __init__(self, text: str = "一线城市住宿 450 元/晚") -> None:
        self.doc_id = "d1"
        self.doc_title = "报销、差旅与采购流程"
        self.heading_path = "差旅 > 住宿标准"
        self.text = text
        self.url = None
        self.chunk_id = "d1-0001"
        self.final_score = 0.9
        self.metadata = {"external_id": "d1"}


def _history(pairs: int) -> list[dict[str, str]]:
    out = []
    for i in range(pairs):
        out.append({"role": "user", "content": f"第{i}个问题：一线城市住宿标准是多少？"})
        out.append({"role": "assistant", "content": f"第{i}个回答：450 元/晚[[1]]。"})
    return out


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolate_store():
    """每个用例结束后把单例置空，避免用例之间串库。"""
    yield
    reset_summary_store(None)


# ---------------- 存储层 ----------------


def test_store_roundtrip_and_delete(tmp_path):
    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        assert await store.get("s1") is None
        await store.put("s1", 4, "摘要A")
        row = await store.get("s1")
        assert row["covered_turns"] == 4 and row["text"] == "摘要A"
        await store.put("s1", 6, "摘要B")
        assert (await store.get("s1"))["covered_turns"] == 6
        assert await store.count() == 1
        await store.delete("s1")
        assert await store.get("s1") is None
        store.close()

    _run(case())


def test_store_evicts_oldest_beyond_limit(tmp_path):
    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        for i in range(3):
            await store.put(f"s{i}", 2, f"摘要{i}", max_sessions=2)
        assert await store.count() == 2
        assert await store.get("s0") is None      # 最旧的被淘汰
        assert await store.get("s2") is not None
        store.close()

    _run(case())


# ---------------- 与上下文组装联动 ----------------


def _build(builder: ContextBuilder, history, session_id="s1"):
    return builder.build(
        "我出差住哪？",
        [FakeChunk()],
        history,
        session_id=session_id,
    )


def test_first_build_writes_summary(tmp_path):
    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        reset_summary_store(store)

        builder = ContextBuilder(llm=CountingLLM())
        await _build(builder, _history(4))          # 8 条，recent 4 条，older 4 条

        row = await store.get("s1")
        assert row is not None
        assert row["covered_turns"] == 4
        assert "450" in row["text"]
        store.close()

    _run(case())


def test_second_build_reuses_stored_summary_without_llm(tmp_path):
    """同一段历史再问一次：不该重新摘要。这是入库最直接的收益。"""

    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        reset_summary_store(store)

        llm = CountingLLM()
        builder = ContextBuilder(llm=llm)
        history = _history(4)
        await _build(builder, history)
        first_calls = len(llm.calls)

        ctx = await _build(builder, history)
        assert len(llm.calls) == first_calls, "没有新增轮次却又调了模型"
        assert any("复用已入库摘要" in layer.note for layer in ctx.layers)
        assert ctx.compressed is True
        store.close()

    _run(case())


def test_incremental_only_summarizes_new_turns(tmp_path):
    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        reset_summary_store(store)

        llm = CountingLLM()
        builder = ContextBuilder(llm=llm)
        await _build(builder, _history(4))          # covered=4
        before = len(llm.calls)

        await _build(builder, _history(6))          # older 变 8 条，只该压 4→8
        new_calls = llm.calls[before:]
        assert len(new_calls) == 2, "应是一次增量摘要 + 一次合并"
        assert any("合并" in c for c in new_calls)

        row = await store.get("s1")
        assert row["covered_turns"] == 8
        assert "导师是王琳" in row["text"]
        store.close()

    _run(case())


def test_llm_failure_falls_back_and_still_persists(tmp_path):
    """模型挂了也要有摘要：抽取式兜底 + 照常入库，只是质量差一点。"""

    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        reset_summary_store(store)

        builder = ContextBuilder(llm=CountingLLM(fail=True))
        ctx = await _build(builder, _history(4))
        assert ctx.compressed is True
        assert any(layer.name == "history.summary" for layer in ctx.layers)
        row = await store.get("s1")
        assert row is not None and row["text"]
        store.close()

    _run(case())


def test_without_session_id_nothing_is_persisted(tmp_path):
    """不传 session_id 时行为回到老样子：临时摘要，不落库。"""

    async def case():
        store = SummaryStore(tmp_path / "s.db")
        await store.init()
        reset_summary_store(store)

        builder = ContextBuilder(llm=CountingLLM())
        ctx = await builder.build("我出差住哪？", [FakeChunk()], _history(4))
        assert ctx.compressed is True
        assert await store.count() == 0
        store.close()

    _run(case())
