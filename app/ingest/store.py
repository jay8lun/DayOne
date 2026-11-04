"""向量库封装（Chroma 持久化）。

要点：
1. **集合名带 provider/model/dim 指纹**，换 embedding 模型时自动换库，
   避免「维度不匹配」把服务搞崩；
2. Chroma 是同步库，所有调用走 ``asyncio.to_thread``，不阻塞事件循环；
3. 这里只管向量；BM25 索引由 ``retriever`` 在启动时从同一份数据构建，保证同源。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from .chunker import Chunk

log = get_logger(__name__)


class VectorStore:
    def __init__(self, embedder_tag: str) -> None:
        self.embedder_tag = embedder_tag
        self.path = str(settings.resolve(settings.chroma_dir))
        self.collection_name = f"{settings.collection_prefix}__{embedder_tag}"
        self._lock = threading.Lock()
        self._client = None
        self._collection = None

    # ---------- 初始化 ----------
    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        with self._lock:
            if self._collection is not None:
                return self._collection
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.path,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:  # 兼容 Chroma 1.x 的新参数形态
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            log.info("向量库就绪: %s (%s)", self.collection_name, self.path)
            return self._collection

    @property
    def collection(self):
        return self._get_collection()

    # ---------- 写 ----------
    def _upsert_sync(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        col = self.collection
        batch = 256
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            col.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + batch],
                documents=[c.text for c in part],
                metadatas=[c.to_metadata() for c in part],
            )

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        await asyncio.to_thread(self._upsert_sync, chunks, embeddings)

    def _delete_doc_sync(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    async def delete_doc(self, doc_id: str) -> None:
        await asyncio.to_thread(self._delete_doc_sync, doc_id)

    # ---------- 读 ----------
    def _query_sync(
        self, vector: list[float], k: int, where: dict | None
    ) -> list[dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=max(1, k),
            where=where or None,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = res.get("embeddings")
        embs = embs[0] if embs is not None and len(embs) else []
        for i, cid in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else 1.0
            out.append(
                {
                    "chunk_id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    # cosine 空间下 distance = 1 - cos，转成 0~1 相似度
                    "similarity": max(0.0, min(1.0, 1.0 - dist)),
                    "distance": dist,
                    "embedding": [float(x) for x in embs[i]] if i < len(embs) else None,
                }
            )
        return out

    async def dense_search(
        self, vector: list[float], k: int = 30, where: dict | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._query_sync, vector, k, where)

    def _all_sync(self) -> list[dict[str, Any]]:
        try:
            res = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            log.error("读取全量 chunk 失败: %s", exc)
            return []
        out = []
        for i, cid in enumerate(res.get("ids") or []):
            out.append(
                {
                    "chunk_id": cid,
                    "text": (res.get("documents") or [])[i] or "",
                    "metadata": (res.get("metadatas") or [])[i] or {},
                }
            )
        return out

    async def all_chunks(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._all_sync)

    def _stats_sync(self) -> dict[str, Any]:
        col = self.collection
        total = col.count()
        sample = col.get(include=["metadatas"], limit=5000)
        metas = sample.get("metadatas") or []
        docs = {m.get("doc_id") for m in metas if isinstance(m, dict)}
        sources: dict[str, int] = {}
        for m in metas:
            if isinstance(m, dict):
                s = str(m.get("source") or "unknown")
                sources[s] = sources.get(s, 0) + 1
        return {
            "collection": self.collection_name,
            "embedder": self.embedder_tag,
            "chunks": total,
            "documents": len(docs),
            "sources": sources,
            "path": self.path,
        }

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._stats_sync)

    def _reset_sync(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log.warning("删除集合失败（可能不存在）: %s", exc)
        self._collection = None

    async def reset(self) -> None:
        await asyncio.to_thread(self._reset_sync)


_store: VectorStore | None = None


def get_store(embedder_tag: str | None = None) -> VectorStore:
    global _store
    if _store is None or (embedder_tag and _store.embedder_tag != embedder_tag):
        from .embedder import get_embedder

        tag = embedder_tag or get_embedder().tag
        _store = VectorStore(tag)
    return _store
