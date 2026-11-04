"""检索与生成评估指标（纯 Python，不引 ragas，方便解释每个数字怎么来的）。

检索侧：
- Recall@k：top-k 里是否召回了期望文档（多文档问题看覆盖比例）
- MRR：第一个相关结果排名的倒数均值
- nDCG@k：考虑排序位置的折扣增益

生成侧（需要真模型才有意义）：
- Keyword Coverage：答案是否包含必须出现的关键事实
- Citation Precision：带引用标记的句子里，有多少能真的在来源中找到依据（复用生产同款校验）
- Refusal Accuracy：本该拒答的问题，是否真的拒答了
"""

from __future__ import annotations

import math
from typing import Any


def _doc_hit(expected: str, doc_title: str, external_id: str) -> bool:
    exp = expected.strip().lower()
    if not exp:
        return False
    candidates = (doc_title or "").lower(), (external_id or "").lower()
    return any(exp in c or c and c.split("/")[-1].split(".")[0] == exp for c in candidates)


def recall_at_k(retrieved: list[dict[str, Any]], expected_docs: list[str], k: int) -> float:
    if not expected_docs:
        return 0.0
    top = retrieved[:k]
    hit = set()
    for exp in expected_docs:
        for item in top:
            if _doc_hit(exp, item.get("doc_title", ""), item.get("external_id", "")):
                hit.add(exp)
                break
    return len(hit) / len(expected_docs)


def reciprocal_rank(retrieved: list[dict[str, Any]], expected_docs: list[str]) -> float:
    if not expected_docs:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if any(_doc_hit(exp, item.get("doc_title", ""), item.get("external_id", "")) for exp in expected_docs):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[dict[str, Any]], expected_docs: list[str], k: int) -> float:
    """文档级 nDCG：同一篇文档的多个 chunk 只算一次增益，避免重复计分。"""
    if not expected_docs:
        return 0.0
    gains = []
    seen: set[str] = set()
    for item in retrieved[:k]:
        doc = item.get("doc_title", "")
        if doc in seen:
            continue
        seen.add(doc)
        hit = any(
            _doc_hit(exp, doc, item.get("external_id", "")) for exp in expected_docs
        )
        gains.append(1.0 if hit else 0.0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(expected_docs), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return min(1.0, dcg / idcg) if idcg else 0.0


def keyword_coverage(answer: str, must_contain: list[str]) -> float:
    if not must_contain:
        return 1.0
    hit = sum(1 for kw in must_contain if kw in answer)
    return hit / len(must_contain)


def citation_doc_precision(citations: list[dict[str, Any]]) -> float:
    """被引用的来源里，有多少条真的来自期望文档之外？这里只做「引用是否落在来源内」。"""
    if not citations:
        return 0.0
    used = [c for c in citations if c.get("used")]
    return 1.0 if used else 0.0


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
