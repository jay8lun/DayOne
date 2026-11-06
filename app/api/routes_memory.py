"""记忆接口：长期事实 / 工作记忆 / 遗忘 / 审计。

这些接口对应「长期运行」要回答的四个问题：
1. 它记住了什么？  ``GET /api/memory/facts``（传 `as_of` 可看「那一刻它认为的事实」）
2. 流程走到哪了？  ``GET /api/memory/work``
3. 为什么记得？    ``GET /api/memory/audit``（每条记忆的来源原话都在）
4. 该忘的忘了吗？  ``POST /api/memory/decay``（支持 dry_run 先看要淘汰谁）
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..memory import get_memory
from .deps import require_api_key
from .schemas import DecayRequest, MaterialRequest, RememberRequest, WorkStepRequest

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_api_key)])


@router.get("/stats", summary="记忆总体统计")
async def stats() -> dict:
    return await get_memory().store.stats()


@router.post("/facts", summary="写入一条记忆（自动处理冲突）")
async def remember(body: RememberRequest) -> dict:
    result = await get_memory().remember(
        body.owner_id, body.predicate, body.value,
        subject=body.subject, kind=body.kind, confidence=body.confidence,
        event_time=body.event_time, source_turn=body.source_turn, session_id=body.session_id,
        mode=body.mode,
    )
    # action 说明冲突是怎么处理的：created / noop / superseded / backfilled / pending
    return {"ok": True, **result}


@router.get("/facts", summary="读取记忆（as_of 支持时间旅行）")
async def facts(
    owner_id: str = Query("demo-user"),
    predicate: str | None = None,
    kind: str | None = Query(None, pattern="^(fact|preference)$"),
    as_of: str | None = Query(None, description="看「这一刻」有效的版本，如 2026-10-20"),
    include_history: bool = False,
) -> dict:
    rows = await get_memory().recall(
        owner_id, predicate=predicate, kind=kind, as_of=as_of, include_history=include_history
    )
    return {"ok": True, "count": len(rows), "items": rows}


@router.get("/profile", summary="拼成可直接进 prompt 的用户画像")
async def profile(owner_id: str = Query("demo-user")) -> dict:
    text = await get_memory().profile_text(owner_id)
    return {"ok": True, "owner_id": owner_id, "profile": text}


@router.get("/work", summary="当前流程状态（做到哪步、材料交了几项）")
async def work(owner_id: str = Query("demo-user"), item_key: str | None = None) -> dict:
    items = await get_memory().store.list_work(owner_id)
    if item_key:
        items = [i for i in items if i["item_key"] == item_key]
    summary = await get_memory().work_summary(owner_id, item_key)
    return {"ok": True, "count": len(items), "items": items, "summary": summary}


@router.post("/work/step", summary="推进流程状态（幂等，默认挡倒退）")
async def work_step(body: WorkStepRequest) -> dict:
    result = await get_memory().set_work_step(
        body.owner_id, body.item_key, body.step, step_index=body.step_index, force=body.force
    )
    return {"ok": True, **result}


@router.post("/work/material", summary="登记材料状态")
async def work_material(body: MaterialRequest) -> dict:
    result = await get_memory().mark_material(
        body.owner_id, body.item_key, body.name, status=body.status
    )
    return {"ok": True, **result}


@router.post("/decay", summary="执行遗忘（活跃→冷归档→墓碑）")
async def decay(body: DecayRequest) -> dict:
    result = await get_memory().decay(
        body.owner_id, half_life_days=body.half_life_days, threshold=body.threshold,
        tombstone_days=body.tombstone_days, dry_run=body.dry_run,
    )
    return {"ok": True, **result}


@router.get("/audit", summary="审计日志（谁在何时写了什么、冲突怎么处理的）")
async def audit(owner_id: str | None = None, limit: int = Query(50, ge=1, le=200)) -> dict:
    rows = await get_memory().store.audits(owner_id, limit)
    return {"ok": True, "count": len(rows), "items": rows}
