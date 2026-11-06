"""内置只读工具：入职场景里最常被问、又最不该让新人去翻文档的两件事。

- 「我的入职流程到哪一步了、还差什么材料」→ 读工作记忆（已有结构化数据，不该再检索文档）
- 「某某在哪个部门 / 分机多少 / 坐哪」→ 读组织目录

前两个是**只读**，出错也只是回答不出来，不会造成任何副作用。

`submit_permission_request` 是**第一个写工具**：它不直接执行，只生成一条待确认操作
（`app/tools/actions.py`），用户点确认才真的开单。配套 `get_my_tickets` 让它闭环——
提交完能立刻问「我的工单到哪一步了」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import BASE_DIR
from ..core.logging import get_logger
from .base import ToolSpec
from .registry import registry

log = get_logger(__name__)

_DIRECTORY_FILE = BASE_DIR / "data" / "directory.json"
_cache: list[dict[str, Any]] | None = None


def _load_directory() -> list[dict[str, Any]]:
    global _cache
    if _cache is not None:
        return _cache
    path = Path(_DIRECTORY_FILE)
    if not path.exists():
        log.warning("组织目录不存在: %s", path)
        _cache = []
        return _cache
    try:
        _cache = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.error("组织目录解析失败: %s", exc)
        _cache = []
    return _cache


async def get_my_onboarding_status(owner_id: str = "") -> dict[str, Any]:
    """读工作记忆：当前流程到第几步、材料交了几项、还差什么。"""
    if not owner_id:
        return {"error": "缺少员工身份，无法查询个人流程"}
    from ..memory import get_memory

    store = get_memory().store
    items = await store.list_work(owner_id)
    if not items:
        return {"owner_id": owner_id, "flows": [], "note": "没有查到进行中的流程"}
    flows = []
    for item in items:
        submitted = [m["name"] for m in item["materials"] if m.get("status") == "submitted"]
        pending = [m["name"] for m in item["materials"] if m.get("status") == "pending"]
        flows.append(
            {
                "flow": item["item_key"],
                "step": item["step"],
                "step_index": item["step_index"],
                "submitted": submitted,
                "pending": pending,
                "updated_at": item["updated_at"][:19],
            }
        )
    return {"owner_id": owner_id, "flows": flows}


async def lookup_colleague(name: str = "", department: str = "") -> dict[str, Any]:
    """按姓名或部门查组织目录（姓名支持部分匹配）。"""
    people = _load_directory()
    if not people:
        return {"matches": [], "note": "组织目录不可用"}

    key = (name or "").strip()
    dept = (department or "").strip()
    if not key and not dept:
        return {"matches": [], "note": "请提供姓名或部门"}

    matches = []
    for person in people:
        if key and key not in person.get("name", ""):
            continue
        if dept and dept not in person.get("department", ""):
            continue
        matches.append(person)
    # 最多返回 5 条，避免一次把整个部门倒出来（也顺便控制上下文体积）
    return {"matches": matches[:5], "total": len(matches)}


async def submit_permission_request(
    system: str = "", reason: str = "", urgency: str = "normal", owner_id: str = ""
) -> dict[str, Any]:
    """写操作：开一张权限申请工单（**必须经用户确认后才会被调用**）。"""
    from .actions import get_action_store

    system = (system or "").strip()
    reason = (reason or "").strip()
    if not system or not reason:
        return {"error": "缺少系统名或申请理由，无法提交"}

    store = await get_action_store()
    ticket = await store.create_ticket(
        owner_id=owner_id,
        kind="permission_request",
        payload={"system": system, "reason": reason, "urgency": urgency},
    )
    return {
        "ticket_no": ticket["ticket_no"],
        "system": system,
        "status": "待审批",
        "eta": "1 个工作日内" if urgency != "urgent" else "4 小时内",
        "note": "审批结果会通过企业微信通知你，可以随时问我「我的工单到哪一步」",
    }


async def get_my_tickets(owner_id: str = "") -> dict[str, Any]:
    """查自己提交过的工单状态。"""
    if not owner_id:
        return {"error": "缺少员工身份，无法查询工单"}
    from .actions import get_action_store

    store = await get_action_store()
    rows = await store.list_tickets(owner_id)
    return {
        "owner_id": owner_id,
        "count": len(rows),
        "tickets": [
            {
                "ticket_no": r["ticket_no"],
                "kind": r["kind"],
                "status": r["status"],
                "system": (r.get("payload") or {}).get("system"),
                "created_at": r["created_at"][:19],
            }
            for r in rows[:10]
        ],
    }


def _describe_permission(args: dict[str, Any]) -> str:
    urgency = {"urgent": "加急", "normal": "普通"}.get(
        str(args.get("urgency", "normal")), str(args.get("urgency", "normal"))
    )
    return (
        f"提交权限申请：开通「{args.get('system', '')}」权限（{urgency}），"
        f"理由：{args.get('reason', '')}"
    )


def register_builtin_tools() -> None:
    registry.register(
        ToolSpec(
            name="get_my_onboarding_status",
            description=(
                "查询**当前用户本人**的入职流程状态：走到哪一步、已提交哪些材料、还差哪些。"
                "只在用户问「我/我的」进度时使用；问公司制度怎么规定，不要用这个工具。"
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=get_my_onboarding_status,
            read_only=True,
            needs_owner=True,
            timeout_s=5.0,
        )
    )
    registry.register(
        ToolSpec(
            name="lookup_colleague",
            description=(
                "在组织目录里查询同事：按姓名（可部分匹配）或部门，返回部门、职位、分机、工位。"
                "用于「某某是谁/在哪个部门/分机多少/坐哪」这类问题。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "同事姓名，可为部分姓名，如「王琳」"},
                    "department": {"type": "string", "description": "部门名，如「人力资源」"},
                },
                "required": [],
            },
            handler=lookup_colleague,
            read_only=True,
            timeout_s=5.0,
        )
    )

    registry.register(
        ToolSpec(
            name="submit_permission_request",
            description=(
                "为用户**提交一个系统权限申请工单**（写操作）。"
                "用户明确要「申请/开通/提个工单」某个系统的权限时使用；"
                "问「怎么申请」「需要什么条件」这类制度问题时不要用。"
                "调用后会生成待确认操作，用户确认后才真正提交。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "system": {"type": "string", "description": "要开通的系统名，如「VPN」「GitLab」「生产数据库只读」"},
                    "reason": {"type": "string", "description": "申请理由，如「需要访问测试环境日志排障」"},
                    "urgency": {"type": "string", "enum": ["normal", "urgent"],
                                "description": "紧急程度，默认 normal"},
                },
                "required": ["system", "reason"],
            },
            handler=submit_permission_request,
            read_only=False,
            needs_owner=True,
            timeout_s=8.0,
            describe=_describe_permission,
        )
    )
    registry.register(
        ToolSpec(
            name="get_my_tickets",
            description="查询**当前用户本人**提交过的工单及其状态（如权限申请到哪一步了）。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=get_my_tickets,
            read_only=True,
            needs_owner=True,
            timeout_s=5.0,
        )
    )


register_builtin_tools()
