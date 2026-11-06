"""知识库管理接口：统计、入库、检索调试、适配器探测。

这些接口是「换知识库平台」时的操作面板：
- ``GET /api/kb/adapters`` 看有哪些适配器可用；
- ``POST /api/kb/adapters/{name}/probe`` 不落库、只拉 3 篇看看能不能连通；
- ``POST /api/kb/ingest`` 正式入库（支持增量 / 全量 / 清空重建）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..adapters import available_adapters, create_adapter
from ..config import settings
from ..core.errors import AdapterNotConfigured
from ..ingest.embedder import get_embedder
from ..ingest.pipeline import IngestionPipeline
from ..ingest.store import get_store
from ..rag.service import get_service
from .deps import require_api_key
from .schemas import IngestRequest, SearchRequest

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"], dependencies=[Depends(require_api_key)])


@router.get("/tools", summary="已注册的工具（含只读标记，写操作不会被自动执行）")
async def tools() -> dict:
    from ..tools import registry

    return {"ok": True, "count": len(registry.names()), "tools": registry.describe()}


@router.get("/stats", summary="知识库统计（文档数、分块数、当前 embedding）")
async def stats() -> dict:
    embedder = get_embedder()
    store = get_store(embedder.tag)
    data = await store.stats()
    data["embedding"] = {"provider": embedder.provider, "model": embedder.model, "dim": embedder.dim}
    data["source"] = settings.kb_source

    # 按文档聚合，给前端做「文档清单」
    chunks = await store.all_chunks()
    docs: dict[str, dict] = {}
    for c in chunks:
        meta = c.get("metadata") or {}
        did = str(meta.get("doc_id") or "")
        item = docs.setdefault(
            did,
            {
                "doc_id": did,
                "title": meta.get("doc_title", "未命名"),
                "url": meta.get("url"),
                "chunks": 0,
                "source": meta.get("source", ""),
            },
        )
        item["chunks"] += 1
    data["documents_detail"] = sorted(docs.values(), key=lambda d: d["title"])
    return data


@router.get("/adapters", summary="已注册的知识库适配器")
async def adapters() -> dict:
    return {
        "current": settings.kb_source,
        "registered": available_adapters(),
        "config_hint": "KB_SOURCE / KB_ADAPTER_CONFIG / ADAPTER_PLUGINS",
    }


@router.get("/adapters/{name}/probe", summary="探测某个适配器（只拉目录，不落库）")
async def probe(name: str, limit: int = 3) -> JSONResponse:
    try:
        adapter = create_adapter(name, dict(settings.kb_config) | {"source": name})
    except AdapterNotConfigured as exc:
        return JSONResponse({"ok": False, "error": exc.message}, status_code=400)
    healthy = await adapter.health_check()
    items = []
    if healthy:
        page = await adapter.list_documents(limit=limit)
        items = [
            {"id": i.external_id, "title": i.title, "updated_at": i.updated_at.isoformat() if i.updated_at else None}
            for i in page.items
        ]
    await adapter.aclose()
    return JSONResponse(
        {"ok": healthy, "adapter": adapter.describe(), "sample": items},
        status_code=200 if healthy else 502,
    )


@router.post("/ingest", summary="从知识库源拉取并建立/更新向量库")
async def ingest(req: IngestRequest) -> dict:
    pipeline = IngestionPipeline()
    report = await pipeline.run(force=req.force, limit=req.limit, reset=req.reset)
    # 入库后立即重建检索索引，避免刚上传的文档搜不到
    await get_service().warmup()
    return {"ok": True, "report": report.to_dict()}


@router.post("/search", summary="检索调试：只看召回，不生成")
async def search(req: SearchRequest) -> dict:
    service = get_service()
    result = await service.retriever.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "confidence": result.confidence,
        "coverage": result.coverage,
        "trace": result.trace,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_title": c.doc_title,
                "heading": c.heading_path,
                "score": round(c.final_score, 4),
                "dense": round(c.dense_score, 4),
                "bm25": round(c.bm25_score, 3),
                "text": c.text[:400],
            }
            for c in result.chunks
        ],
    }


@router.post("/reset", summary="清空向量库（危险：需显式调用）")
async def reset_kb() -> dict:
    store = get_store(get_embedder().tag)
    await store.reset()
    await get_service().warmup()
    return {"ok": True, "message": "向量库已清空"}
