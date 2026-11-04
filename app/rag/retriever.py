"""混合检索：Dense + BM25 -> RRF 融合 -> MMR 去重 -> 可选 LLM 重排。

为什么用 RRF 而不是加权求和？
向量相似度（0~1）与 BM25 分数（0~30+）量纲不同，加权前必须归一化，
而归一化又会引入新的超参。RRF 只用**排名**，对两路分数分布不敏感，
工程上更稳，也不需要在换语料后重新调参：

    score(d) = Σ_i 1 / (k + rank_i(d))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import settings
from ..core.logging import get_logger
from ..ingest.embedder import BaseEmbedder, get_embedder
from ..ingest.store import VectorStore, get_store
from .bm25 import BM25Index
from .llm import BaseLLM, content_terms, tokenize

log = get_logger(__name__)


def bigrams_for_diversity(text: str) -> set[str]:
    """用于去重的轻量表示：中文 2-gram + 英文词。"""
    return set(tokenize(text))


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    term_coverage: float = 0.0
    embedding: list[float] | None = None
    rank_dense: int | None = None
    rank_bm25: int | None = None

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def doc_title(self) -> str:
        return str(self.metadata.get("doc_title", "未命名文档"))

    @property
    def heading_path(self) -> str:
        return str(self.metadata.get("heading_path", ""))

    @property
    def url(self) -> str | None:
        return self.metadata.get("url")

    def to_source(self, n: int) -> dict[str, Any]:
        return {
            "n": n,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "heading": self.heading_path,
            "url": self.url,
            "snippet": self.text if len(self.text) <= 320 else self.text[:320] + "…",
            "score": round(self.final_score, 4),
        }


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    dense_top: float = 0.0
    bm25_top: float = 0.0
    coverage: float = 0.0
    evidence: float = 0.0  # 单条 chunk 的最高词面覆盖度（判断「有没有一段真的在讲这个」）
    confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or get_store(self.embedder.tag)
        self.llm = llm
        self.bm25 = BM25Index(
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            title_weight=settings.bm25_title_weight,
            heading_weight=settings.bm25_heading_weight,
        )
        self._indexed_count = -1
        self._chunk_map: dict[str, tuple[str, dict]] = {}

    async def refresh_index(self, force: bool = False) -> int:
        """从向量库重建 BM25 索引（入库后调用；按 chunk 数量做懒刷新）。"""
        stats = await self.store.stats()
        count = int(stats.get("chunks", 0))
        if not force and count == self._indexed_count and self.bm25.size == count:
            return self.bm25.size
        chunks = await self.store.all_chunks()
        self.bm25.build(chunks)
        self._chunk_map = {
            c["chunk_id"]: (c["text"], c["metadata"] or {}) for c in chunks
        }
        self._indexed_count = count
        log.info("BM25 索引就绪: %d chunks", self.bm25.size)
        return self.bm25.size

    # ---------- 检索 ----------
    async def search(
        self,
        query: str,
        top_k: int | None = None,
        candidates: int | None = None,
        use_mmr: bool | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.retrieve_top_k
        candidates = candidates or settings.retrieve_candidates
        await self.refresh_index()

        # 向量通道开关。评估发现：hashing 兜底向量是纯噪声源，
        # 混进来反而把 MRR 从 0.961 拉到 0.928（见 docs/调优记录.md），
        # 所以在降级 embedding 下自动只用词法通道，不假装自己在做语义检索。
        dense_on = settings.enable_dense and self.embedder.provider != "hashing"
        if dense_on:
            vector = await self.embedder.embed_query(query)
            dense = await self.store.dense_search(vector, k=candidates)
        else:
            dense = []
        lexical = self.bm25.search(query, k=candidates)

        pool: dict[str, RetrievedChunk] = {}
        for rank, item in enumerate(dense, start=1):
            pool[item["chunk_id"]] = RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"] or {},
                dense_score=float(item["similarity"]),
                embedding=item.get("embedding"),
                rank_dense=rank,
            )
        for rank, (cid, score) in enumerate(lexical, start=1):
            if cid in pool:
                pool[cid].bm25_score = float(score)
                pool[cid].rank_bm25 = rank
            else:
                meta_text = await self._fetch_one(cid)
                if meta_text is None:
                    continue
                text, meta = meta_text
                pool[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    metadata=meta,
                    bm25_score=float(score),
                    rank_bm25=rank,
                )

        # RRF 融合（只看排名，不依赖两路分数的量纲）
        k = settings.rrf_k
        q_terms = content_terms(query)
        for c in pool.values():
            score = 0.0
            if c.rank_dense:
                score += 1.0 / (k + c.rank_dense)
            if c.rank_bm25:
                score += 1.0 / (k + c.rank_bm25)
            c.fused_score = score
            # 「词面覆盖度」：问题里的实词有多少出现在这段 chunk 里。
            # 纯 BM25 会被「一长段里恰好堆了很多关键词」的 chunk 骗到，
            # 覆盖度能把「只命中一个高频词」的噪声压下去。
            text_terms = content_terms(f"{c.doc_title} {c.heading_path} {c.text}")
            c.term_coverage = (
                len(q_terms & text_terms) / len(q_terms) if q_terms else 0.0
            )
        max_rrf = max((c.fused_score for c in pool.values()), default=0.0) or 1.0
        # 只有词法通道时，覆盖度这个「补丁」反而会扰动 BM25F 的正确排序，
        # 因此把权重调高（实测 MRR 0.950 → 0.967）。
        rrf_w = settings.rrf_weight if dense_on else max(settings.rrf_weight, 0.85)
        for c in pool.values():
            c.fused_score = round(
                rrf_w * (c.fused_score / max_rrf)
                + (1 - rrf_w) * c.term_coverage,
                6,
            )
        ranked = sorted(pool.values(), key=lambda c: c.fused_score, reverse=True)

        # MMR：企业文档里同一段话常出现在「制度」和「FAQ」两篇里，
        # 不去重会把 top5 全占满，导致引用来源单一。
        if (settings.enable_mmr if use_mmr is None else use_mmr) and len(ranked) > top_k:
            ranked = await self._mmr(query, ranked, candidates)

        # 可选 LLM 重排（贵，默认关；开会显著提升 top3 精度）
        if settings.enable_llm_rerank and self.llm and len(ranked) > top_k:
            ranked = await self._llm_rerank(query, ranked)

        # 归一化 final_score：融合分本身已在 0~1，重排分再掺一点
        for c in ranked:
            c.final_score = (
                round(0.7 * c.fused_score + 0.3 * c.rerank_score, 4)
                if c.rerank_score
                else round(c.fused_score, 4)
            )

        selected = ranked[:top_k]
        result = RetrievalResult(
            query=query,
            chunks=selected,
            dense_top=dense[0]["similarity"] if dense else 0.0,
            bm25_top=lexical[0][1] if lexical else 0.0,
        )
        self._assess(result, query, selected)
        result.trace = {
            "dense_hits": len(dense),
            "bm25_hits": len(lexical),
            "fused_pool": len(pool),
            "returned": len(selected),
            "top_dense_sim": round(result.dense_top, 4),
            "top_bm25": round(result.bm25_top, 3),
            "coverage": round(result.coverage, 3),
            "evidence": round(result.evidence, 3),
            "confidence": round(result.confidence, 3),
        }
        return result

    async def _fetch_one(self, chunk_id: str) -> tuple[str, dict] | None:
        return self._chunk_map.get(chunk_id)

    # ---------- 融合后处理 ----------
    async def _mmr(
        self, query: str, chunks: list[RetrievedChunk], limit: int
    ) -> list[RetrievedChunk]:
        """MMR 去重。

        相似度优先用向量（dense 通道本来就把 embedding 带回来了）；
        没有向量的 chunk（只在 BM25 通道命中）**不重新调 embedding 接口**，
        改用字符 2-gram 的 Jaccard 近似——否则每次查询会多出十几到几十次 API 调用，
        既慢又费钱（这个坑是实测发现的，见 docs/调优记录.md）。
        """
        if len(chunks) <= 2:
            return chunks
        head = chunks[:limit]
        vecs = [
            np.asarray(c.embedding, dtype=np.float32) if c.embedding is not None else None
            for c in head
        ]
        grams = [set(bigrams_for_diversity(c.text)) for c in head]

        def sim(i: int, j: int) -> float:
            if vecs[i] is not None and vecs[j] is not None:
                return float(vecs[i] @ vecs[j])
            gi, gj = grams[i], grams[j]
            if not gi or not gj:
                return 0.0
            return len(gi & gj) / len(gi | gj)

        lam = settings.mmr_lambda
        rel = np.asarray([c.fused_score for c in head], dtype=np.float32)
        selected: list[int] = []
        remaining = list(range(len(head)))
        while remaining:
            best_idx, best_val = remaining[0], -1e9
            for i in remaining:
                redundancy = max((sim(i, j) for j in selected), default=0.0)
                val = lam * float(rel[i]) - (1 - lam) * redundancy
                if val > best_val:
                    best_val, best_idx = val, i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return [head[i] for i in selected] + chunks[limit:]

    async def _llm_rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        import json as _json

        head = chunks[:15]
        listing = "\n".join(
            f"[{i + 1}] {c.doc_title} > {c.heading_path}\n{c.text[:260]}" for i, c in enumerate(head)
        )
        prompt = (
            "你是检索结果打分器。根据用户问题，给下面每个片段打 0-10 分，"
            "能直接回答问题的给 8-10 分，只是相关的给 4-7 分，无关的 0-3 分。\n"
            "只输出 JSON：{\"scores\": [分数, ...]}，顺序与片段一致，不要任何解释。\n\n"
            f"【用户问题】\n{query}\n\n【片段】\n{listing}"
        )
        try:
            raw = await self.llm.complete(  # type: ignore[union-attr]
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format_json=True,
            )
            data = _json.loads(raw)
            scores = [float(x) for x in data.get("scores", [])]
            for i, c in enumerate(head):
                if i < len(scores):
                    c.rerank_score = max(0.0, min(1.0, scores[i] / 10.0))
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 重排失败，回退到 RRF 排序: %s", exc)
            return chunks
        head.sort(key=lambda c: (c.rerank_score, c.fused_score), reverse=True)
        return head + chunks[len(head) :]

    # ---------- 置信度评估 ----------
    def _assess(self, result: RetrievalResult, question: str, chunks: list[RetrievedChunk]) -> None:
        q_terms = content_terms(question)
        if not q_terms or not chunks:
            result.coverage = 0.0
            result.evidence = 0.0
            result.confidence = 0.0
            return
        # coverage：top3 合起来覆盖了多少问题实词（看「信息够不够」）
        corpus_text = " ".join(f"{c.doc_title} {c.heading_path} {c.text}" for c in chunks[:3])
        result.coverage = len(q_terms & content_terms(corpus_text)) / len(q_terms)
        # evidence：有没有**某一段**真正在讲这件事（看「是不是在胡说」）
        result.evidence = max(
            (
                len(q_terms & content_terms(f"{c.doc_title} {c.heading_path} {c.text}"))
                / len(q_terms)
                for c in chunks[:3]
            ),
            default=0.0,
        )

        lexical_norm = 1 / (1 + math.exp(-(max(result.bm25_top, 0.0) - 8.0) / 3.0))
        dense_norm = max(0.0, min(1.0, result.dense_top))
        result.confidence = round(
            0.55 * lexical_norm + 0.25 * result.evidence + 0.20 * dense_norm, 4
        )
