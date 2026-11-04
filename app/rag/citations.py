"""引用解析与校验（RAG 的「可信度」就在这一步）。

生成模型经常犯两种错：
1. 引用了不存在的编号（[[7]] 但只有 5 条资料）→ 直接删除标记；
2. 标记了编号，但那句话其实不是该资料里的内容（假引用）→ 用词面重叠度打分，
   低于阈值的标记为 ``weak``，前端会以灰色显示，评测里算作「不支持引用」。

这一步不做「重新生成」，因为二次生成会引入更多不确定性；先把问题**暴露**出来，
让人能看到「这句话引用得不可靠」，比偷偷修掉更有价值。
"""

from __future__ import annotations

import re
from typing import Any

from .llm import content_terms
from .retriever import RetrievedChunk

_MARK_RE = re.compile(r"\[\[(\d{1,2})\]\]")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")


def build_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [c.to_source(i + 1) for i, c in enumerate(chunks)]


def _clause_terms(text: str) -> set[str]:
    return content_terms(text)


def verify_and_attach(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    weak_threshold: float = 0.12,
) -> dict[str, Any]:
    """返回清洗后的答案、来源列表、引用明细与「有依据比例」。"""
    sources = build_sources(chunks)
    valid_ids = {s["n"] for s in sources}
    chunk_by_n = {i + 1: c for i, c in enumerate(chunks)}

    unknown: list[int] = []
    used: dict[int, int] = {}
    weak: list[dict[str, Any]] = []
    checked = 0
    supported = 0

    # 逐句检查：句子里出现的编号，是否真的能被对应 chunk 支撑
    raw_sentences = [s for s in _SENT_SPLIT.split(answer) if s.strip()]
    units: list[list] = []  # [正文, [编号...]]
    for sentence in raw_sentences:
        marks = [int(m) for m in _MARK_RE.findall(sentence)]
        body = _MARK_RE.sub("", sentence).strip()
        if not body and marks and units:
            # 模型常把标记写在句号后面（「……提交。[[1]]」），归到上一句，别当成独立句
            units[-1][1].extend(marks)
        else:
            units.append([body, marks])

    for body, marks in units:
        if not marks:
            continue
        body_terms = _clause_terms(body)
        for n in marks:
            if n not in valid_ids:
                unknown.append(n)
                continue
            used[n] = used.get(n, 0) + 1
            chunk = chunk_by_n[n]
            chunk_terms = _clause_terms(f"{chunk.doc_title} {chunk.heading_path} {chunk.text}")
            overlap = len(body_terms & chunk_terms) / (len(body_terms) or 1)
            checked += 1
            if overlap >= weak_threshold:
                supported += 1
            else:
                weak.append({"n": n, "sentence": body[:80], "overlap": round(overlap, 3)})

    cleaned = answer
    # 删除越界编号，避免前端点开是空的
    for n in set(unknown):
        cleaned = re.sub(rf"\[\[{n}\]\]", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()

    citations = []
    for s in sources:
        n = s["n"]
        citations.append(
            {
                **s,
                "used": n in used,
                "hits": used.get(n, 0),
            }
        )

    grounding = round(supported / checked, 3) if checked else 0.0
    return {
        "answer": cleaned,
        "answer_plain": _MARK_RE.sub("", cleaned).strip(),
        "sources": sources,
        "citations": citations,
        "unknown_markers": sorted(set(unknown)),
        "weak_citations": weak,
        "grounding_score": grounding,
        "citation_precision": grounding,  # 同一口径，便于评测引用
        "cited_count": len(used),
    }


def extract_cited_docs(answer: str) -> set[int]:
    return {int(m) for m in _MARK_RE.findall(answer)}
