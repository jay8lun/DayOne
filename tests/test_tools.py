"""工具调用的行为测试。

守住四件事：
1. 参数不合法/超时/异常都要变成**一句可读的话**，而不是 500；
2. 写操作不能被自动执行（只读白名单）；
3. 身份由服务端注入，模型不能"替别人查进度"；
4. 结果过长要截断——否则一次查询就能把上下文撑爆。
"""

from __future__ import annotations

import asyncio

import app.memory as memory_pkg
from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.tools import builtin  # noqa: F401  注册内置工具
from app.tools.base import ToolSpec
from app.tools.registry import ToolRegistry
from app.tools.router import looks_actionable, route_tool


class FakeLLM:
    """按预设返回工具调用或纯文本，不联网。"""

    provider = "fake"
    model = "fake-1"

    def __init__(self, tool_calls=None, content: str = "") -> None:
        self._calls = tool_calls or []
        self._content = content
        self.invocations = 0

    async def complete_with_tools(self, messages, tools, **kwargs):
        from app.rag.llm import LLMToolResponse

        self.invocations += 1
        return LLMToolResponse(content=self._content, tool_calls=list(self._calls))

    async def complete(self, messages, **kwargs):
        return "已按工具结果作答"


def _temp_memory(tmp_path) -> MemoryService:
    store = MemoryStore(tmp_path / "mem.db")
    asyncio.run(store.init())
    svc = MemoryService(store)
    memory_pkg._service = svc          # 让 builtin 工具读到测试库
    return svc


def test_schema_shape():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="ping", description="ping", parameters={"type": "object"},
                          handler=lambda: None))
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "ping"


def test_unknown_tool_returns_readable_error():
    reg = ToolRegistry()

    async def case():
        result = await reg.execute("not_exist", {})
        assert result.ok is False
        assert "未注册" in result.error

    asyncio.run(case())


def test_write_tool_requires_confirmation():
    reg = ToolRegistry()
    executed = {"n": 0}

    async def dangerous(**kwargs):
        executed["n"] += 1
        return {"done": True}

    reg.register(ToolSpec(name="submit_ticket", description="提交工单",
                          parameters={"type": "object"}, handler=dangerous, read_only=False))

    async def case():
        blocked = await reg.execute("submit_ticket", {"title": "报销"})
        assert blocked.ok is False and "确认" in blocked.error
        assert executed["n"] == 0            # 没有被偷偷执行

        allowed = await reg.execute("submit_ticket", {"title": "报销"}, allow_write=True)
        assert allowed.ok is True and executed["n"] == 1

    asyncio.run(case())


def test_timeout_and_bad_args_are_caught():
    reg = ToolRegistry()

    async def slow(**kwargs):
        await asyncio.sleep(0.5)
        return {}

    async def strict(owner_id: str):
        return {"owner_id": owner_id}

    reg.register(ToolSpec(name="slow", description="slow", parameters={"type": "object"},
                          handler=slow, timeout_s=0.05))
    reg.register(ToolSpec(name="strict", description="strict", parameters={"type": "object"},
                          handler=strict))

    async def case():
        timed = await reg.execute("slow", {})
        assert timed.ok is False and "超时" in timed.error
        # 模型编了一个不存在的参数名，属于常见幻觉
        bad = await reg.execute("strict", {"wrong_param": "x"})
        assert bad.ok is False and "参数不合法" in bad.error

    asyncio.run(case())


def test_result_is_truncated():
    reg = ToolRegistry()

    async def big(**kwargs):
        return {"rows": ["x" * 100 for _ in range(50)]}

    reg.register(ToolSpec(name="big", description="big", parameters={"type": "object"},
                          handler=big))

    async def case():
        result = await reg.execute("big", {})
        text = result.to_text(max_chars=200)
        assert "已截断" in text and len(text) < 400

    asyncio.run(case())


def test_lookup_colleague_partial_and_department():
    async def case():
        by_name = await builtin.lookup_colleague(name="王琳")
        assert by_name["matches"][0]["ext"] == "8007"

        by_dept = await builtin.lookup_colleague(department="平台研发部")
        assert by_dept["total"] >= 2
        assert all("平台研发部" in m["department"] for m in by_dept["matches"])

        missing = await builtin.lookup_colleague(name="不存在的人")
        assert missing["matches"] == []

    asyncio.run(case())


def test_onboarding_status_reads_work_memory(tmp_path):
    svc = _temp_memory(tmp_path)

    async def case():
        await svc.set_work_step("emp-1", "onboarding", "提交材料", step_index=1)
        await svc.mark_material("emp-1", "onboarding", "身份证复印件")
        await svc.mark_material("emp-1", "onboarding", "学历证明", status="pending")

        result = await builtin.get_my_onboarding_status(owner_id="emp-1")
        flow = result["flows"][0]
        assert flow["step"] == "提交材料"
        assert flow["submitted"] == ["身份证复印件"]
        assert flow["pending"] == ["学历证明"]

        no_identity = await builtin.get_my_onboarding_status(owner_id="")
        assert "error" in no_identity

    asyncio.run(case())


def test_route_tool_skips_non_actionable_question():
    async def case():
        llm = FakeLLM()
        out = await route_tool(llm, "一线城市住宿标准是多少？", "emp-1")
        assert out is None
        assert llm.invocations == 0          # 连模型都不该调

    asyncio.run(case())


def test_route_tool_executes_and_injects_identity(tmp_path):
    svc = _temp_memory(tmp_path)

    async def case():
        await svc.mark_material("emp-9", "onboarding", "学历证明", status="pending")

        from app.rag.llm import ToolCall

        call = ToolCall(id="c1", name="get_my_onboarding_status", arguments="{}")
        out = await route_tool(FakeLLM(tool_calls=[call]), "我的材料还差什么？", "emp-9")
        assert out is not None and out["ok"] is True
        assert "学历证明" in out["text"]
        # 身份由服务端注入，模型传的 args 里没有 owner_id
        assert out["args"]["owner_id"] == "emp-9"

    asyncio.run(case())


def test_route_tool_without_identity_refuses():
    from app.rag.llm import ToolCall

    async def case():
        call = ToolCall(id="c1", name="get_my_onboarding_status", arguments="{}")
        out = await route_tool(FakeLLM(tool_calls=[call]), "我的进度到哪了？", None)
        assert out is not None and out["ok"] is False
        assert "身份" in out["text"]

    asyncio.run(case())


def test_looks_actionable():
    assert looks_actionable("帮我查下王琳的分机号") is True
    assert looks_actionable("我的材料交了吗") is True
    assert looks_actionable("年假可以结转到明年吗") is False


def test_ticket_question_not_swallowed_by_memory_route():
    """「我提交的工单到哪一步」不能走记忆路由。

    实测踩过：这句话同时满足记忆路由的三个条件（提自己 + 问哪一步 + 档案有状态），
    于是被档案截胡，答成了「入职流程到入职培训」。档案里根本没有工单。
    """
    from app.rag.service import memory_covers

    memory = (
        "【已知用户信息】\n- city=杭州\n\n"
        "【当前流程状态】\n- onboarding：当前「入职培训」，已交 1 项（身份证复印件），待补 学历证明"
    )
    assert memory_covers("我的流程走到哪一步了", memory) is True
    assert memory_covers("我提交的权限工单到哪一步了", memory) is False
    assert memory_covers("我的审批进度", memory) is False
