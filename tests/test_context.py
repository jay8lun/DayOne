"""上下文预算与压缩的行为测试。

守住三条容易在重构中悄悄坏掉的约束：
1. **参考资料永不压缩**（压了引用回验就废）；
2. **最近一两轮原文不压**（压了指代消解会崩）；
3. 预算不够时，**先丢摘要、再丢更早的原文**，绝不丢当前问题和资料。
"""

from __future__ import annotations

import asyncio

from app.context import ContextBuilder
from app.context.budget import estimate_tokens, truncate_to_tokens


class DummyLLM:
    """假模型：把历史摘要成固定文本，避免测试依赖网络。"""

    provider = "dummy"
    model = "dummy-1"

    def __init__(self, reply: str = "用户问过报销时限；已确认住宿标准 450 元。") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.reply


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.doc_id = "d1"
        self.doc_title = "报销、差旅与采购流程"
        self.heading_path = "差旅 > 住宿标准"
        self.text = text
        self.url = None
        self.chunk_id = "d1-0001"
        self.final_score = 0.9
        self.dense_score = 0.9
        self.bm25_score = 10.0
        self.rerank_score = 0.0
        self.metadata = {"external_id": "d1"}
        self.embedding = None


def _history(turns: int) -> list[dict[str, str]]:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"第{i}个问题：一线城市住宿标准是多少？"})
        out.append({"role": "assistant", "content": f"第{i}个回答：一线城市每晚 450 元[[1]]。"})
    return out


def test_estimate_tokens_roughly_matches_chinese():
    text = "一线城市住宿标准是每晚四百五十元"
    # 中文按 1 字 ≈ 1 token
    assert 12 <= estimate_tokens(text) <= 20
    assert estimate_tokens("") == 0


def test_truncate_keeps_sentence_boundary():
    text = "第一句话。第二句话。第三句话。"
    out = truncate_to_tokens(text, 8)
    assert out.endswith("…")
    assert len(out) < len(text)


def test_chunks_are_never_compressed():
    chunk_text = "一线城市（北京、上海、深圳、广州）住宿标准为每人每晚 450 元，需提供发票。" * 3

    async def case():
        builder = ContextBuilder(llm=DummyLLM())
        ctx = await builder.build(
            "一线城市住宿标准？",
            [FakeChunk(chunk_text)],
            _history(12),                      # 历史很长，逼它压缩
            profile="city=杭州",
        )
        user_msg = ctx.messages[-1]["content"]
        assert chunk_text in user_msg         # 资料原文一字不少
        assert ctx.compressed is True         # 历史确实被压了

    asyncio.run(case())


def test_recent_turns_kept_verbatim():
    async def case():
        builder = ContextBuilder(llm=DummyLLM())
        history = _history(6)
        ctx = await builder.build("那其他城市呢？", [FakeChunk("其他城市 300 元")], history)

        joined = "\n".join(m["content"] for m in ctx.messages)
        assert "【历史摘要】" in joined                 # 旧的被压成摘要
        assert "第5个问题" in joined                    # 最近一轮原文还在
        assert "第0个问题" not in joined                # 最早的不再以原文出现
        assert "第0个回答" not in joined

    asyncio.run(case())


def test_budget_squeeze_drops_summary_before_chunks():
    async def case():
        builder = ContextBuilder(llm=DummyLLM())
        ctx = await builder.build(
            "一线城市住宿标准？",
            [FakeChunk("一线城市每晚 450 元。")],
            _history(20),
            max_tokens=700,                     # 故意把预算压到很紧
        )
        joined = "\n".join(m["content"] for m in ctx.messages)
        assert "一线城市每晚 450 元。" in joined        # 资料仍在
        assert "一线城市住宿标准？" in ctx.messages[-1]["content"]   # 当前问题仍在
        assert any("丢弃" in t for t in ctx.trace)      # 丢弃动作有留痕

    asyncio.run(case())


def test_profile_and_work_injected_and_capped():
    async def case():
        builder = ContextBuilder(llm=DummyLLM())
        ctx = await builder.build(
            "我的材料交到哪了？",
            [FakeChunk("材料清单")],
            [],
            profile="【已知用户信息】\n- city=杭州\n" + "啰嗦内容" * 400,
            work="【当前流程状态】\n- onboarding：当前「提交材料」",
        )
        joined = "\n".join(m["content"] for m in ctx.messages)
        assert "【已知用户信息】" in joined
        assert "onboarding：当前「提交材料」" in joined
        layers = {x.name: x for x in ctx.layers}
        assert layers["profile"].tokens <= 240          # 被截断，不会无限长
        assert layers["chunks"].kept is True

    asyncio.run(case())


def test_no_history_means_no_summary_call():
    async def case():
        llm = DummyLLM()
        builder = ContextBuilder(llm=llm)
        ctx = await builder.build("报销多久内提交？", [FakeChunk("60 天内")], [])
        assert llm.calls == []                          # 没历史就不该花钱调模型
        assert ctx.compressed is False
        assert "【历史摘要】" not in "\n".join(m["content"] for m in ctx.messages)

    asyncio.run(case())


def test_memory_gate_accepts_personal_question():
    """「我在哪一步」该由档案回答；「食堂几点开饭」不该被档案硬答。"""
    from app.rag.service import memory_covers

    memory = (
        "【已知用户信息】\n- city=杭州\n\n"
        "【当前流程状态】\n- onboarding：当前「提交材料」，已交 2 项（身份证复印件、学历证明）"
    )
    assert memory_covers("我的流程走到哪一步了？还要补什么材料？", memory) is True
    assert memory_covers("我的流程进度", memory) is True
    assert memory_covers("公司食堂中午几点开饭？", memory) is False      # 没提自己
    assert memory_covers("我的报销标准是多少？", memory) is False        # 问规则，交给文档
    assert memory_covers("我的流程进度", "") is False                    # 没档案
