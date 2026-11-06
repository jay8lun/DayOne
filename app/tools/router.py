"""工具路由：判断「这个问题该去查系统，还是查文档」。

顺序很重要（和记忆路由一样）：**动作型问题先走工具，制度型问题走文档**。
否则「帮我查下王琳的分机」会被当成「王琳」这个关键词去检索手册，永远查不到。

两道闸：
1. **启发式前置**：问题里没有「查/找/谁/分机/进度/申请」这类词，直接不进工具调用
   （省一次模型往返，也免得模型在纯制度问题上乱调工具）；
2. **模型选择**：只有过了第一道闸，才把工具清单交给模型，由它决定调哪个、传什么参数。

**写操作在这里被截住**：模型选了写工具，路由层只会调用 `ActionService.propose()`
生成一条待确认操作，绝不执行。执行只发生在用户点确认后的 `/api/tools/confirm`。
"""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..core.logging import get_logger
from ..rag.llm import BaseLLM
from .registry import registry

log = get_logger(__name__)

_ACTION_HINTS = (
    "查", "找", "谁", "在哪", "哪儿", "哪个部门", "分机", "工位", "电话", "座位",
    "进度", "到哪", "第几步", "材料", "导师", "对接人", "负责人", "联系",
    # 写操作相关：没有这些词就不该让模型去「办事」
    "申请", "开通", "提个", "提交", "工单", "报修", "办理", "审批", "权限", "帮我",
)

TOOL_SELECT_PROMPT = """你在为一个企业知识库助手做「是否需要调用工具」的判断。

工具能查的是**系统里的实时数据**（某人的部门/分机/工位、当前用户自己的流程状态）；
公司制度、流程规定、标准金额这类写在手册里的内容**不属于工具范围**，不要调用。

规则：
- 只在确实需要系统数据时调用工具；
- 用户问「我/我的」进度，用 get_my_onboarding_status；
- 用户问某位同事的信息，用 lookup_colleague；
- 用户问自己的工单/申请进展，用 get_my_tickets；
- 用户**明确要求**申请/开通某个系统权限，用 submit_permission_request，并把系统名与理由抽出来；
  只是问「怎么申请」「需要什么条件」时不要调用（那是制度问题）；
- 其他情况一律不调用工具，直接回复「不需要」。
- 不要回答用户的问题本身。"""

TOOL_ANSWER_PROMPT = """根据【工具结果】回答用户的问题。

硬要求：
1. 只能使用工具结果里的信息，不允许补充任何结果之外的细节；
2. 结果里没有的（比如没查到这个人的分机），直接说「系统里查不到」，不要编造；
3. 分机号、工位、部门名必须与结果完全一致，不改写、不推断；
4. 用中文，两句话以内，先给答案再补一句提示（如「可以直接拨 8007」）。"""


def looks_actionable(question: str) -> bool:
    """粗筛：没有动作/查询类信号就不折腾模型了。"""
    q = question.lower()
    return any(h in q for h in _ACTION_HINTS)


async def route_tool(
    llm: BaseLLM,
    question: str,
    owner_id: str | None,
) -> dict[str, Any] | None:
    """需要工具时返回调用结果，否则返回 None（继续走文档检索）。"""
    if not settings.enable_tools or not looks_actionable(question):
        return None
    if not registry.names():
        return None

    try:
        resp = await llm.complete_with_tools(
            [{"role": "system", "content": TOOL_SELECT_PROMPT},
             {"role": "user", "content": question}],
            registry.schemas(read_only_only=False),
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("工具路由失败，回落文档检索: %s", exc)
        return None

    if not resp.wants_tool:
        return None

    call = resp.tool_calls[0]
    spec = registry.get(call.name)
    if spec is None:
        return None
    args = call.args()

    # 身份由服务端注入，不接受模型自己填 —— 否则模型可以"替别人查进度/替别人提申请"
    if spec.needs_owner:
        args["owner_id"] = owner_id or ""
        if not owner_id:
            return {
                "tool": call.name, "args": args, "ok": False,
                "text": "没有拿到你的员工身份，无法查询或办理个人信息。",
            }

    # 写操作：只生成待确认操作，绝不在这里执行。
    # 参数里去掉 owner_id（它不是工具的输入参数，身份在确认执行时由服务端再注入一次）。
    if not spec.read_only:
        if not settings.enable_write_tools:
            return {"tool": call.name, "args": args, "ok": False,
                    "text": "写操作当前已关闭，只能查询。", "requires_confirmation": False}
        from .actions import ActionError, get_actions

        try:
            service = await get_actions()
            action = await service.propose(
                owner_id, call.name, {k: v for k, v in args.items() if k != "owner_id"}
            )
        except ActionError as exc:
            return {"tool": call.name, "args": args, "ok": False, "text": exc.message,
                    "requires_confirmation": False, "error": exc.code}
        return {
            "tool": call.name, "args": args, "ok": True,
            "requires_confirmation": True,
            "action_id": action["id"], "summary": action["summary"],
            "expires_at": action["expires_at"],
            "text": f"需要你确认后才会提交：{action['summary']}",
        }

    result = await registry.execute(call.name, args, owner_id=owner_id)
    return {"tool": call.name, "args": args, "ok": result.ok,
            "text": result.to_text(), "elapsed_ms": result.elapsed_ms}


async def answer_with_tool(llm: BaseLLM, question: str, tool_text: str) -> str:
    messages = [
        {"role": "system", "content": TOOL_ANSWER_PROMPT},
        {"role": "system", "content": f"【工具结果】\n{tool_text}"},
        {"role": "user", "content": question},
    ]
    try:
        return (await llm.complete(messages, temperature=0.1, max_tokens=300)).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("工具结果作答失败: %s", exc)
        return "系统查询完成了，但组织答案时出错了，请稍后再试一次。"
