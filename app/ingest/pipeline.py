"""入库流水线：适配器 -> 清洗 -> 切块 -> 向量化 -> 落库（支持增量）。

``ingest_state.json`` 记录每篇文档的内容哈希：
- 哈希不变 → 直接跳过（省钱、省时间）；
- 哈希变化 → 先按 doc_id 删旧 chunk 再写新 chunk（避免残留旧分段被检索到）；
- 源里删掉的文档 → 同步清理向量库，防止「答出已废弃制度」。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..adapters.base import KnowledgeSourceAdapter
from ..adapters.registry import build_adapters
from ..config import settings
from ..core.errors import KnowledgeBaseEmpty
from ..core.logging import get_logger
from .chunker import chunk_document, summarize_chunks
from .embedder import BaseEmbedder, get_embedder
from .store import VectorStore, get_store

log = get_logger(__name__)

EMBED_BATCH = 48


@dataclass
class IngestReport:
    started_at: str = ""
    duration_s: float = 0.0
    adapters: list[str] = field(default_factory=list)
    documents_seen: int = 0
    documents_new: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    chunks_written: int = 0
    embedder: str = ""
    collection: str = ""
    errors: list[dict[str, str]] = field(default_factory=list)
    kb: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
            "adapters": self.adapters,
            "documents_seen": self.documents_seen,
            "documents_new": self.documents_new,
            "documents_updated": self.documents_updated,
            "documents_skipped": self.documents_skipped,
            "documents_deleted": self.documents_deleted,
            "chunks_written": self.chunks_written,
            "embedder": self.embedder,
            "collection": self.collection,
            "errors": self.errors,
            "kb": self.kb,
        }


class IngestionPipeline:
    def __init__(
        self,
        adapters: list[KnowledgeSourceAdapter] | None = None,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or get_store(self.embedder.tag)
        self.adapters = adapters

    # ---------- 状态文件 ----------
    @property
    def state_path(self):
        return settings.resolve(settings.ingest_state_file)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"documents": {}}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("ingest_state 解析失败，按全新入库处理")
            return {"documents": {}}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---------- 主流程 ----------
    async def run(
        self,
        force: bool = False,
        limit: int | None = None,
        reset: bool = False,
    ) -> IngestReport:
        t0 = time.perf_counter()
        report = IngestReport(
            started_at=datetime.now().isoformat(timespec="seconds"),
            embedder=self.embedder.tag,
            collection=self.store.collection_name,
        )
        if reset:
            log.warning("reset=True，清空现有向量库")
            await self.store.reset()
            self._save_state({"documents": {}})

        adapters = self.adapters or build_adapters()
        report.adapters = [a.name for a in adapters]
        state = self._load_state()
        known: dict[str, dict[str, Any]] = state.get("documents", {})
        # 换了 embedding 模型时，集合名会变（带模型指纹），新集合是空的，
        # 但 ingest_state 里还记着「这些文档都同步过了」→ 会导致全部 skip、新库永远是空的。
        # 这里做一次自愈：库空但状态非空，强制重建。
        if known and not reset:
            current = await self.store.stats()
            if int(current.get("chunks", 0)) == 0:
                log.warning(
                    "向量库 %s 为空但入库状态非空（通常是换了 embedding 模型），自动改为全量重建",
                    self.store.collection_name,
                )
                force = True
        seen_ids: set[str] = set()
        pending_chunks = []
        pending_docs: list[tuple[str, str]] = []  # (doc_id, title)

        try:
            for adapter in adapters:
                if not await adapter.health_check():
                    report.errors.append(
                        {"adapter": adapter.name, "error": "health_check 失败，已跳过该源"}
                    )
                    log.error("适配器 %s 健康检查失败，跳过", adapter.name)
                    continue
                try:
                    async for doc in adapter.fetch_all(limit=limit):
                        report.documents_seen += 1
                        seen_ids.add(doc.doc_id)
                        if not doc.content.strip():
                            report.errors.append(
                                {"adapter": adapter.name, "error": f"{doc.title} 正文为空"}
                            )
                            continue
                        prev = known.get(doc.doc_id)
                        if prev and prev.get("hash") == doc.content_hash and not force:
                            report.documents_skipped += 1
                            continue
                        if prev:
                            await self.store.delete_doc(doc.doc_id)
                            report.documents_updated += 1
                        else:
                            report.documents_new += 1

                        chunks = chunk_document(
                            doc,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                            min_chunk_size=settings.min_chunk_size,
                        )
                        pending_chunks.extend(chunks)
                        pending_docs.append((doc.doc_id, doc.title))
                        known[doc.doc_id] = {
                            "hash": doc.content_hash,
                            "title": doc.title,
                            "external_id": doc.external_id,
                            "source": adapter.name,
                            "chunks": len(chunks),
                            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        }
                        if len(pending_chunks) >= EMBED_BATCH * 4:
                            report.chunks_written += await self._flush(pending_chunks, report)
                            pending_chunks = []
                except Exception as exc:  # noqa: BLE001
                    log.exception("适配器 %s 拉取失败", adapter.name)
                    report.errors.append({"adapter": adapter.name, "error": str(exc)})
                finally:
                    await adapter.aclose()

            if pending_chunks:
                report.chunks_written += await self._flush(pending_chunks, report)

            # 源里已删除的文档：清理向量（先不动状态，避免适配器整体挂掉时误删全库）
            failed_all = len(report.errors) > 0 and report.documents_seen == 0
            if not failed_all:
                stale = [did for did in known if did not in seen_ids]
                for doc_id in stale:
                    await self.store.delete_doc(doc_id)
                    known.pop(doc_id, None)
                    report.documents_deleted += 1
                if report.documents_deleted:
                    log.info("清理已下线文档 %d 篇", report.documents_deleted)
        finally:
            self._save_state({"documents": known, "updated_at": datetime.now().isoformat()})

        stats = await self.store.stats()
        report.kb = stats
        report.duration_s = time.perf_counter() - t0
        if stats.get("chunks", 0) == 0:
            raise KnowledgeBaseEmpty("入库后知识库为空，请检查知识库源配置")
        log.info(
            "入库完成: %d 篇(新 %d/更 %d/跳过 %d/删 %d), %d chunks, 耗时 %.1fs",
            report.documents_seen,
            report.documents_new,
            report.documents_updated,
            report.documents_skipped,
            report.documents_deleted,
            report.chunks_written,
            report.duration_s,
        )
        return report

    async def _flush(self, chunks, report: IngestReport) -> int:
        texts = [c.embed_text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(await self.embedder.embed_documents(texts[i : i + EMBED_BATCH]))
        await self.store.upsert(chunks, vectors)
        log.debug("写入 %d chunks", len(chunks))
        return len(chunks)
