"""问答接口：一次性 / SSE 流式 / 重置会话 / 反馈。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..config import settings
from ..core.logging import get_logger
from ..rag.service import get_service
from .deps import require_api_key
from .schemas import ChatRequest, ChatResponse, FeedbackRequest, ResetRequest

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(require_api_key)])
log = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, summary="一次性问答（含来源与引用）")
async def chat(req: ChatRequest) -> ChatResponse:
    service = get_service()
    payload = await service.ask(req.question, req.session_id, req.owner_id)
    return ChatResponse(**payload)


@router.post("/chat/stream", summary="流式问答（SSE：status / sources / delta / done）")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    service = get_service()

    async def event_source():
        try:
            async for event in service.astream(req.question, req.session_id, req.owner_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("SSE 流中断")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲，否则流式会被攒成一坨
        },
    )


@router.post("/chat/reset", summary="清空某个会话的多轮记忆")
async def reset(req: ResetRequest) -> dict:
    await get_service().reset(req.session_id)
    return {"ok": True, "session_id": req.session_id}


@router.post("/feedback", summary="「这个问题没答案」反馈，落盘待补充清单")
async def feedback(req: FeedbackRequest) -> dict:
    path: Path = settings.resolve("data/feedback.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": req.question,
        "session_id": req.session_id,
        "note": req.note,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("收到未命中反馈: %s", req.question[:60])
    return {"ok": True, "total_file": str(path)}
