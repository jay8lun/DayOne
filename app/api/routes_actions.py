"""确认式写操作接口：提议 / 确认 / 取消 / 待办 / 工单。

设计上把「确认」做成独立的一次请求，而不是让 `/api/chat` 带上 `confirm=true`：
确认是一个**有副作用的动作**，必须能用幂等键单独重试、单独审计，
也不该被聊天链路的模型输出影响到（模型不该有能力替用户点确认）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..tools.actions import ActionError, get_actions
from ..tools.registry import registry
from .deps import require_api_key
from .schemas import CancelRequest, ConfirmRequest, ProposeRequest

router = APIRouter(prefix="/api/tools", tags=["tools"], dependencies=[Depends(require_api_key)])


def _http_error(exc: ActionError) -> HTTPException:
    status = {"not_found": 404, "forbidden": 403, "already_executed": 409}.get(exc.code, 400)
    return HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message})


@router.get("", summary="工具清单（含只读标记与是否需要确认）")
async def list_tools() -> dict:
    return {
        "ok": True,
        "count": len(registry.names()),
        "write_enabled": settings.enable_write_tools,
        "confirm_ttl_minutes": settings.tool_confirm_ttl_minutes,
        "tools": registry.describe(),
    }


@router.post("/propose", summary="生成待确认操作（不执行）")
async def propose(body: ProposeRequest) -> dict:
    if not settings.enable_write_tools:
        raise HTTPException(status_code=403, detail={"code": "write_disabled",
                                                     "message": "写操作已关闭（ENABLE_WRITE_TOOLS=false）"})
    try:
        service = await get_actions()
        action = await service.propose(
            body.owner_id, body.tool, body.args, client_key=body.client_key
        )
    except ActionError as exc:
        raise _http_error(exc) from exc
    return {
        "ok": True,
        "requires_confirmation": True,
        "action": action,
        "note": f"{settings.tool_confirm_ttl_minutes} 分钟内有效，确认后才执行",
    }


@router.post("/confirm", summary="确认并执行（幂等：重复确认只执行一次）")
async def confirm(body: ConfirmRequest) -> dict:
    try:
        service = await get_actions()
        result = await service.confirm(body.owner_id, body.action_id)
    except ActionError as exc:
        raise _http_error(exc) from exc
    return result


@router.post("/cancel", summary="取消待确认操作")
async def cancel(body: CancelRequest) -> dict:
    try:
        service = await get_actions()
        result = await service.cancel(body.owner_id, body.action_id)
    except ActionError as exc:
        raise _http_error(exc) from exc
    return result


@router.get("/pending", summary="我的待确认操作（顺带把过期项落库）")
async def pending(owner_id: str = Query("demo-user")) -> dict:
    service = await get_actions()
    rows = await service.pending(owner_id)
    return {"ok": True, "count": len(rows), "items": rows}


@router.get("/tickets", summary="我提交过的工单")
async def tickets(owner_id: str = Query("demo-user")) -> dict:
    service = await get_actions()
    rows = await service.store.list_tickets(owner_id)
    return {"ok": True, "count": len(rows), "items": rows}
