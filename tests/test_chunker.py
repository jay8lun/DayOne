"""切块逻辑单测：标题面包屑、超长切分、碎片合并。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.base import SourceDocument
from app.ingest.chunker import chunk_document, split_sections


def _doc(content: str, title: str = "测试手册") -> SourceDocument:
    return SourceDocument(external_id="test.md", title=title, content=content)


def test_split_sections_keeps_breadcrumb():
    content = "# 手册\n\n## 报销\n\n正文A\n\n### 住宿\n\n正文B\n"
    sections = split_sections(content)
    paths = [s.heading_path for s in sections]
    assert "手册 > 报销" in paths
    assert "手册 > 报销 > 住宿" in paths


def test_h1_not_duplicated_in_heading_path():
    doc = _doc("# 报销手册\n\n## 住宿标准\n\n一线城市 450 元。\n", title="报销手册")
    chunks = chunk_document(doc)
    assert chunks, "应至少产生一个 chunk"
    assert chunks[0].heading_path == "住宿标准"


def test_long_section_is_split_with_overlap():
    body = "## 长章节\n\n" + ("这是用来测试切分的一段话，长度需要超过阈值。" * 40)
    chunks = chunk_document(_doc(body), chunk_size=200, chunk_overlap=40, min_chunk_size=20)
    assert len(chunks) > 3
    assert all(len(c.text) <= 260 for c in chunks)


def test_short_fragment_merged_into_previous():
    content = "## 标题\n\n这是一段足够长的正文内容，用于占满第一个分块的位置。" + "填充内容。" * 40
    content += "\n\n## 小尾巴\n\n短。"
    chunks = chunk_document(_doc(content), chunk_size=300, chunk_overlap=20, min_chunk_size=40)
    assert len(chunks) >= 1
    # 最后一块不应该只剩「短。」这种无信息碎片
    assert len(chunks[-1].text) >= 10


def test_embed_text_includes_doc_title():
    doc = _doc("# T\n\n## S\n\n正文内容足够长用来测试。", title="T")
    chunk = chunk_document(doc)[0]
    assert "《T》" in chunk.embed_text
    assert chunk.heading_path == "S"
