"""结构感知切块。

为什么不用 ``RecursiveCharacterTextSplitter`` 一条命令了事？
企业文档（入职手册、SOP）的价值大量藏在**标题层级**里：
「报销流程 > 差旅 > 住宿标准」和「考勤 > 差旅」语义完全不同。
所以这里做两件事：

1. **按 Markdown 标题切节**，把 ``# 一级 > ## 二级 > ### 三级`` 记成 breadcrumb；
2. 节内再按段落滑窗切，**overlap 只切一次**，并且 chunk 保留原文，
   保证前端「引用高亮」能精确对齐原文，不用二次猜测。

向量化时用 ``breadcrumb + 正文``（提升召回），展示/引用时只用正文（保证高亮准确）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import SourceDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENT_END = "。！？!?；;\n"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str  # 正文（用于展示 + 引用高亮）
    index: int
    heading_path: str = ""  # "报销流程 > 差旅 > 住宿标准"
    url: str | None = None
    updated_at: str | None = None
    space: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """喂给 embedding 的文本：带上下文面包屑，避免「它/该流程」指代丢失。"""
        if not self.heading_path:
            return f"《{self.doc_title}》\n{self.text}"
        return f"《{self.doc_title}》 > {self.heading_path}\n{self.text}"

    def to_metadata(self) -> dict[str, Any]:
        # Chroma metadata 只接受 str/int/float/bool
        meta = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chunk_index": self.index,
            "heading_path": self.heading_path,
            "source": self.source,
        }
        if self.url:
            meta["url"] = self.url
        if self.updated_at:
            meta["updated_at"] = self.updated_at
        if self.space:
            meta["space"] = self.space
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta


@dataclass(slots=True)
class _Section:
    heading_path: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def split_sections(content: str) -> list[_Section]:
    """按标题层级切节，遇到同级/更高级标题就开新节（保留父子面包屑）。"""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    cur = _Section(heading_path="")

    def flush() -> None:
        nonlocal cur
        if cur.text:
            sections.append(cur)
        cur = _Section(heading_path=" > ".join(t for _, t in stack))

    in_fence = False
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = _Section(heading_path=" > ".join(t for _, t in stack))
            continue
        cur.lines.append(line)
    flush()
    return sections


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    """长节滑窗切分，优先在句末/换行处断开，避免把一句话劈两半。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 在窗口后半段找最近的句边界
            window = text[start:end]
            cut = max(
                (window.rfind(ch) for ch in _SENT_END),
                default=-1,
            )
            if cut > chunk_size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_document(
    doc: SourceDocument,
    chunk_size: int = 520,
    chunk_overlap: int = 80,
    min_chunk_size: int = 60,
) -> list[Chunk]:
    """一篇文档 -> N 个 chunk（自带 doc_id / heading 面包屑 / 原文位置）。"""
    chunks: list[Chunk] = []
    updated = doc.updated_at.isoformat() if doc.updated_at else None
    idx = 0
    title_prefix = f"{doc.title} > "
    for section in split_sections(doc.content):
        # H1 通常就是文档标题，面包屑里再重复一遍没有信息量，去掉
        heading = section.heading_path
        if heading.startswith(title_prefix):
            heading = heading[len(title_prefix) :]
        elif heading == doc.title:
            heading = ""
        # 表格/代码块尽量别切碎：整体长度不够就整块留
        for piece in _split_long(section.text, chunk_size, chunk_overlap):
            if len(piece) < min_chunk_size and chunks:
                # 太短的碎片（比如只有一个小标题）并进上一块，别浪费一次向量
                prev = chunks[-1]
                if prev.doc_id == doc.doc_id and len(prev.text) + len(piece) <= chunk_size * 1.5:
                    prev.text = f"{prev.text}\n{piece}"
                    continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{idx:04d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=piece,
                    index=idx,
                    heading_path=heading,
                    url=doc.url,
                    updated_at=updated,
                    space=doc.space,
                    source=str(doc.metadata.get("source", "")),
                    metadata={"external_id": doc.external_id},
                )
            )
            idx += 1
    return chunks


def summarize_chunks(chunks: list[Chunk]) -> dict[str, Any]:
    docs = {c.doc_id for c in chunks}
    lengths = [len(c.text) for c in chunks] or [0]
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
        "max_chunk_chars": max(lengths),
    }
