"""FastAPI 入口。

启动顺序：日志 -> 插件加载 -> 适配器体检 -> 向量库/索引预热 ->（可选）首次入库。
所有失败都要**降级而不是崩**：一个知识库源连不上，不应该让整个服务起不来。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .adapters.registry import available_adapters, build_adapters, load_plugins
from .api import routes_actions, routes_chat, routes_eval, routes_kb, routes_memory
from .config import BASE_DIR, settings
from .core.errors import AppError
from .core.logging import get_logger, setup_logging
from .ingest.embedder import get_embedder
from .ingest.pipeline import IngestionPipeline
from .ingest.store import get_store
from .memory import get_memory
from .rag.llm import get_llm
from .rag.service import get_service

setup_logging("DEBUG" if settings.debug else "INFO")
log = get_logger(__name__)

STARTED_AT = time.time()
BOOT_INFO: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 插件适配器（公司自研知识库平台放这里，不用改主流程）
    load_plugins(getattr(settings, "adapter_plugins", ""))
    adapters = available_adapters()
    log.info("已注册适配器: %s", ", ".join(adapters))

    # 2) 适配器体检（失败只警告）
    try:
        for adapter in build_adapters():
            ok = await adapter.health_check()
            log.info("适配器 %s 健康检查: %s", adapter.name, "OK" if ok else "FAIL")
            await adapter.aclose()
    except Exception as exc:  # noqa: BLE001
        log.error("适配器初始化异常: %s", exc)

    # 3) 向量库 + 检索索引
    embedder = get_embedder()
    store = get_store(embedder.tag)
    stats = await store.stats()
    log.info(
        "向量库: %s | embedding=%s | chunks=%d",
        store.collection_name,
        embedder.tag,
        stats.get("chunks", 0),
    )
    if stats.get("chunks", 0) == 0 and settings.auto_ingest_on_startup:
        log.info("知识库为空，开始首次入库…")
        try:
            report = await IngestionPipeline(embedder=embedder, store=store).run()
            BOOT_INFO["first_ingest"] = report.to_dict()
        except Exception as exc:  # noqa: BLE001
            log.error("首次入库失败（服务继续启动，可稍后手动入库）: %s", exc)
            BOOT_INFO["first_ingest_error"] = str(exc)
    try:
        await get_service().warmup()
    except Exception as exc:  # noqa: BLE001
        log.error("索引预热失败: %s", exc)

    BOOT_INFO["llm"] = f"{get_llm().provider}/{get_llm().model}"

    # 4) 记忆库（SQLite，本地文件；失败不阻塞启动，只是记忆功能不可用）
    try:
        memory = get_memory()
        await memory.store.init()
        mstats = await memory.store.stats()
        BOOT_INFO["memory"] = mstats
        log.info(
            "记忆库就绪 | 事实 %d 条 / 待确认 %d / 冷归档 %d | 流程 %d",
            mstats.get("facts_active", 0), mstats.get("facts_pending", 0),
            mstats.get("facts_cold", 0), mstats.get("work_items", 0),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("记忆库初始化失败（服务继续启动）: %s", exc)

    # 5) 动作库（待确认操作 / 工单）。和记忆库分开：生命周期与清理策略不同
    try:
        from .tools.actions import get_action_store

        store = await get_action_store()
        pending = await store.list_actions("__probe__", status=None, limit=0)
        BOOT_INFO["actions"] = {"pending_probe": len(pending)}
        log.info("动作库就绪 | %s", settings.tool_actions_db_path)
    except Exception as exc:  # noqa: BLE001
        log.error("动作库初始化失败（写操作将不可用）: %s", exc)

    log.info("服务就绪 | LLM=%s", BOOT_INFO["llm"])
    yield

    log.info("服务关闭")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "通用 RAG 问答模板：换知识库平台只需换适配器。\n\n"
        "- 适配器模式：`app/adapters/`（local_dir / http_json / wecom / 外部插件）\n"
        "- 混合检索：Dense + BM25 + RRF + MMR\n"
        "- 多轮追问：LangGraph 编排的查询改写与检索纠错\n"
        "- 来源标注与引用高亮：`[[n]]` 协议 + 词面回验\n"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log.warning("业务异常 %s: %s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "code": exc.code, "message": exc.message, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("未捕获异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "code": "internal_error", "message": str(exc)},
    )


app.include_router(routes_chat.router)
app.include_router(routes_kb.router)
app.include_router(routes_eval.router)
app.include_router(routes_memory.router)
app.include_router(routes_actions.router)


@app.get("/api/health", tags=["ops"], summary="健康检查（含当前模型与知识库状态）")
async def health() -> dict:
    embedder = get_embedder()
    store = get_store(embedder.tag)
    stats = await store.stats()
    llm = get_llm()
    return {
        "ok": True,
        "app": settings.app_name,
        "version": settings.app_version,
        "uptime_s": int(time.time() - STARTED_AT),
        "llm": {"provider": llm.provider, "model": llm.model, "degraded": llm.provider == "mock"},
        "embedding": {
            "provider": embedder.provider,
            "model": embedder.model,
            "dim": embedder.dim,
            "degraded": embedder.provider == "hashing",
        },
        "knowledge_base": {
            "chunks": stats.get("chunks", 0),
            "documents": stats.get("documents", 0),
            "collection": stats.get("collection"),
            "source": settings.kb_source,
        },
        "adapters": available_adapters(),
        "boot": BOOT_INFO,
    }


# ---------------- 前端 ----------------
WEB_DIR: Path = BASE_DIR / "app" / "web" / "static"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))
