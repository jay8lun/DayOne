"""BM25F 与分词单测。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.bm25 import BM25Index
from app.rag.llm import content_terms, tokenize


def _chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "消费发生后 60 天内提交报销单，逾期系统自动关闭。",
            "metadata": {"doc_title": "报销、差旅与采购流程", "heading_path": "报销基本规则"},
        },
        {
            "chunk_id": "c2",
            "text": "远程办公每周最多 2 天，需提前一天在 OA 申请。",
            "metadata": {"doc_title": "考勤、请假与远程办公", "heading_path": "远程办公"},
        },
        {
            "chunk_id": "c3",
            "text": "发布窗口为每周二、周四 14:00—17:00，禁止在周五发布。",
            "metadata": {"doc_title": "发布上线与故障响应 SOP", "heading_path": "发布窗口"},
        },
    ]


def test_tokenize_cjk_bigram():
    tokens = tokenize("报销流程")
    assert "报销" in tokens and "销流" in tokens


def test_bm25_ranks_correct_chunk_first():
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("报销多久内提交", k=3)
    assert hits and hits[0][0] == "c1"


def test_bm25_title_weight_helps():
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("远程办公几天", k=3)
    assert hits[0][0] == "c2"


def test_bm25_empty_index_returns_empty():
    idx = BM25Index()
    idx.build([])
    assert idx.search("任何问题") == []


def test_content_terms_filters_stopwords():
    terms = content_terms("报销是怎么样的？")
    assert "报销" in terms
    assert "怎么" not in terms
