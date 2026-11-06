"""多步编排的测试。

编排最容易出两类问题：**串错数据**（第二步拿了别人的名字/编的系统）和**越权代办**
（用户只是问规则，系统却替他提交了申请）。这组测试主要盯这两件事。

为了不依赖网络（检索要 embedding），检索步骤用 stub 片段替换；
真正要验证的是"上游输出怎么变成下游入参""该不该提案"这两层逻辑。
"""

from __future__ import annotations

import asyncio

import pytest

from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.tools import orchestrator as orch
from app.tools.actions import reset_actions
from app.tools.store import ActionStore


class _StubLLM:
    provider = "stub"
    model = "stub"

    async def complete(self, *a, **k):  # pragma: no cover - 编排当前不调模型
        return ""


def _chunk(text: str, title: str = "00-新人第一周指南", heading: str = "入职第 1 天"):
    from app.rag.retriever import RetrievedChunk

    return RetrievedChunk(
        chunk_id="c1", text=text,
        metadata={"doc_title": title, "heading_path": heading, "external_id": "doc-00"},
        final_score=0.9,
    )


GUIDE_TEXT = (
    "在 IT 自助平台 it.yunshu.internal 提交「开发环境开通」工单，勾选："
    "GitLab、Jenkins、KaiRAG 控制台、数据平台只读权限。一般 2 小时内审批完成。"
)


class _DocCtx(orch.RecipeContext):
    """把检索替换成固定片段，其它（记忆/工具/提案）走真实实现。"""

    def __init__(self, *args, text: str = GUIDE_TEXT, memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = memory
        self._text = text

    async def retrieve(self, query: str, k: int = 3):  # type: ignore[override]
        self.step("检索知识库", "doc", True, "stub 片段")
        self.chunks = [_chunk(self._text)]
        return self.chunks


@pytest.fixture()
def memory_svc(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    asyncio.run(store.init())
    yield MemoryService(store)
    store.close()


@pytest.fixture()
def action_svc(tmp_path):
    store = ActionStore(tmp_path / "actions.db")
    asyncio.run(store.init())
    service = reset_actions(store, audit=False)
    yield service
    reset_actions(None)
    store.close()


def _run(coro):
    return asyncio.run(coro)


# ---------------- 剧本 1：记忆 → 工具 ----------------


def test_mentor_chain_uses_memory_output_as_tool_input(memory_svc: MemoryService):
    async def case():
        await memory_svc.remember("u1", "mentor", "王琳")
        ctx = orch.RecipeContext(memory=memory_svc, question="我导师是谁？他分机多少", owner_id="u1", llm=_StubLLM())
        result = await orch.mentor_contact(ctx)

        assert result is not None
        assert "王琳" in result["answer"] and "8007" in result["answer"]
        assert [s.kind for s in ctx.plan] == ["memory", "tool"]
        assert all(s.ok for s in ctx.plan)
        # 第二步的入参必须来自第一步（姓名），不是模型编的
        assert "王琳" in ctx.plan[1].detail

    _run(case())


def test_mentor_missing_is_honest_and_skips_tool(memory_svc: MemoryService):
    async def case():
        ctx = orch.RecipeContext(memory=memory_svc, question="我导师是谁", owner_id="u1", llm=_StubLLM())
        result = await orch.mentor_contact(ctx)

        assert result is not None
        assert "没有记录导师" in result["answer"]
        assert [s.kind for s in ctx.plan] == ["memory"]      # 没名字就不去查目录
        assert ctx.tool_calls == 0

    _run(case())


# ---------------- 剧本 2：文档 → 提案 ----------------


def test_system_extraction_respects_allowlist():
    chunks = [_chunk(GUIDE_TEXT + "另外到 IT 自助平台领门禁卡。")]
    systems = orch._extract_systems(chunks)
    assert "GitLab" in systems and "Jenkins" in systems
    # 「数据平台只读权限」要带上限定词，别申请成写权限
    assert "数据平台只读" in systems
    # 「IT 自助平台」不是要开通的系统，不能混进来
    assert not any("IT 自助平台" in s for s in systems)


def test_question_about_systems_does_not_propose(memory_svc, action_svc):
    async def case():
        ctx = _DocCtx(memory=memory_svc, question="入职第一周要开通哪些系统？", owner_id="u1", llm=_StubLLM())
        result = await orch.first_week_env(ctx)

        assert result is not None
        assert "GitLab" in result["answer"]
        assert not result.get("requires_confirmation")
        assert action_svc.store.path.exists()
        # 关键：只是提问，不许产生待确认申请
        assert await action_svc.store.list_actions("u1") == []

    _run(case())


def test_explicit_request_proposes_with_doc_systems(memory_svc, action_svc):
    async def case():
        ctx = _DocCtx(memory=memory_svc, question="帮我按第一周的清单把开发环境系统都开一下", owner_id="u1", llm=_StubLLM())
        result = await orch.first_week_env(ctx)

        assert result is not None
        assert result["requires_confirmation"] is True
        action = result["pending_action"]
        assert action["action_id"]
        rows = await action_svc.store.list_actions("u1")
        assert len(rows) == 1
        system = rows[0]["args"]["system"]
        assert "GitLab" in system and "数据平台只读" in system
        # 提案阶段不能开单
        assert await action_svc.store.count_tickets("u1") == 0

    _run(case())


def test_no_system_found_means_no_proposal(memory_svc, action_svc):
    async def case():
        ctx = _DocCtx(memory=memory_svc, question="帮我开通一下那个系统", owner_id="u1", llm=_StubLLM(),
                      text="权限申请请走 OA，具体系统以主管确认为准。")
        result = await orch.first_week_env(ctx)

        assert result is None                                  # 判定不适用 → 回落常规链路
        assert await action_svc.store.list_actions("u1") == []

    _run(case())


# ---------------- 剧本 3：文档 + 工具 ----------------


def test_ticket_rule_check_merges_rule_and_status(memory_svc, action_svc):
    async def case():
        await action_svc.propose("u1", "submit_permission_request",
                                 {"system": "VPN", "reason": "调试"})
        # 直接造一张工单（propose 不执行）
        await action_svc.store.create_ticket(owner_id="u1", kind="permission_request",
                                             payload={"system": "VPN"})
        text = "敏感权限默认有效期 30 天，到期自动回收。需要延期的在到期前 3 天重新申请。"
        ctx = _DocCtx(memory=memory_svc, question="敏感权限多久到期？我申请的那个还有效吗", owner_id="u1", llm=_StubLLM(), text=text)
        result = await orch.ticket_rule_check(ctx)

        assert result is not None
        assert "30 天" in result["answer"]           # 规则来自文档
        assert "REQ-" in result["answer"]            # 状态来自工具
        assert [s.kind for s in ctx.plan] == ["doc", "tool"]

    _run(case())


# ---------------- 边界 ----------------


def test_rule_question_is_not_hijacked(memory_svc, action_svc):
    """「申请权限需要什么条件」是规则问题，编排不能抢。"""

    async def case():
        orchestrator = orch.Orchestrator(memory=memory_svc)
        result = await orchestrator.run("申请权限需要什么条件", "u1", _StubLLM())
        assert result is None

    _run(case())


def test_orchestration_never_executes_write_tool(memory_svc, action_svc):
    async def case():
        ctx = orch.RecipeContext(memory=memory_svc, question="x", owner_id="u1", llm=_StubLLM())
        out = await ctx.call_tool("submit_permission_request", {"system": "VPN", "reason": "x"})
        assert out["ok"] is False
        assert "写" in out["error"]
        assert ctx.plan[-1].ok is False
        assert await action_svc.store.count_tickets("u1") == 0

    _run(case())


def test_step_budget_guard():
    ctx = orch.RecipeContext(memory=memory_svc, question="x", owner_id="u1", llm=_StubLLM())
    for i in range(orch.MAX_STEPS):
        ctx.step(f"s{i}", "doc", True)
    with pytest.raises(orch.BudgetExceeded):
        ctx.step("overflow", "doc", True)
