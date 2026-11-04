"""BM25 检索（纯 Python 实现，无额外依赖）。

为什么向量检索之外还要 BM25？
- 制度类问题大量是**精确词命中**（「一等座」「60 天」「L3」），向量容易糊；
- 新人提问常带专有名词（KaiRAG、SkillHub），BM25 对 OOV 词更敏感；
- 两者用 RRF 融合，不调权重也能稳定提升（内部对比见 docs/评估报告.md）。

索引在内存里，从向量库同一份数据构建，保证「同一批 chunk」。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .llm import tokenize


class BM25Index:
    """BM25F（字段加权）：标题、章节路径、正文给不同权重。

    纯 BM25 在制度文档上有个典型问题：某个 chunk 正文里偶然罗列了一串用户问题
    （比如「当用户问『报销能不能报』『住宿超标了吗』」），词频很高，
    就会把真正的《报销流程》挤下去。标题/章节命中的权重放大后，
    「报销」出现在文档标题里的 chunk 会稳定胜出，这正是我们要的语义先验。
    """

    TITLE_WEIGHT = 5.0
    HEADING_WEIGHT = 3.0

    def __init__(self, k1: float = 1.3, b: float = 0.85, title_weight: float | None = None, heading_weight: float | None = None) -> None:
        self.k1 = k1
        self.b = b
        self.title_weight = self.TITLE_WEIGHT if title_weight is None else title_weight
        self.heading_weight = self.HEADING_WEIGHT if heading_weight is None else heading_weight
        self.chunk_ids: list[str] = []
        self.docs_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self._doc_freq: Counter[str] = Counter()
        self.n_docs: int = 0

    # ---------- 构建 ----------
    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.chunk_ids = []
        self.docs_tokens = []
        self.doc_len = []
        self._doc_freq = Counter()
        for item in chunks:
            meta = item.get("metadata") or {}
            tf: Counter[str] = Counter()
            tf.update(tokenize(item.get("text", "")))
            for t, c in Counter(tokenize(meta.get("doc_title", ""))).items():
                tf[t] += c * self.title_weight
            for t, c in Counter(tokenize(meta.get("heading_path", ""))).items():
                tf[t] += c * self.heading_weight
            self.chunk_ids.append(item["chunk_id"])
            self.docs_tokens.append(tf)
            self.doc_len.append(int(sum(tf.values())))
            for t in tf:
                self._doc_freq[t] += 1
        self.n_docs = len(self.chunk_ids)
        total = sum(self.doc_len)
        self.avgdl = total / self.n_docs if self.n_docs else 0.0
        self.idf = {
            t: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for t, df in self._doc_freq.items()
        }

    @property
    def size(self) -> int:
        return self.n_docs

    # ---------- 查询 ----------
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        if not self.n_docs:
            return []
        q_tokens = [t for t in tokenize(query) if t in self.idf]
        if not q_tokens:
            return []
        scores: list[tuple[str, float]] = []
        for idx, tf in enumerate(self.docs_tokens):
            dl = self.doc_len[idx] or 1
            score = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * f * (self.k1 + 1) / denom
            if score > 0:
                scores.append((self.chunk_ids[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
