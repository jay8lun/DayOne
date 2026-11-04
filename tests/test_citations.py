"""引用解析与校验单测：越界编号、假引用、空来源。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.citations import verify_and_attach
from app.rag.retriever import RetrievedChunk


def _chunk(cid: str, text: str, title: str = "报销、差旅与采购流程", heading: str = "报销基本规则"):
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata={"doc_id": "d1", "doc_title": title, "heading_path": heading, "url": "file:///doc.md"},
    )


CHUNKS = [
    _chunk("c1", "消费发生后 60 天内提交报销单，逾期系统自动关闭，需财务总监特批。"),
    _chunk("c2", "一线城市住宿标准为 450 元每晚上限，超出需提前申请。"),
]


def test_valid_citation_kept_and_marked_used():
    answer = "报销需在消费发生后 60 天内提交。[[1]]"
    out = verify_and_attach(answer, CHUNKS)
    assert "[[1]]" in out["answer"]
    assert out["citations"][0]["used"] is True
    assert out["citations"][1]["used"] is False
    assert out["grounding_score"] > 0.5


def test_out_of_range_marker_removed():
    answer = "这句话引用了不存在的资料。[[9]]"
    out = verify_and_attach(answer, CHUNKS)
    assert "[[9]]" not in out["answer"]
    assert out["unknown_markers"] == [9]


def test_unsupported_claim_flagged_weak():
    answer = "公司每年组织两次海外团建。[[1]]"
    out = verify_and_attach(answer, CHUNKS)
    assert out["weak_citations"], "与来源无词面重叠的引用应被标记为弱引用"


def test_answer_without_citation():
    out = verify_and_attach("知识库里没有找到。", CHUNKS)
    assert out["grounding_score"] == 0.0
    assert out["cited_count"] == 0


def test_sources_numbering_matches_chunks():
    out = verify_and_attach("x", CHUNKS)
    assert [s["n"] for s in out["sources"]] == [1, 2]
