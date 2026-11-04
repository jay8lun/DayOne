"""内置适配器：本地目录（md/txt/html/pdf/docx）。

这是「最朴素」的适配器，用来验证主流程，同时方便把公司现成的入职手册
直接丢进 data/corpus 就跑起来。它依然只做「读取 → 统一模型」，不绑定业务。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..core.errors import AdapterError
from ..core.logging import get_logger
from .base import DocumentPage, DocumentRef, KnowledgeSourceAdapter, SourceDocument

log = get_logger(__name__)

SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


class LocalDirAdapter(KnowledgeSourceAdapter):
    name = "local_dir"
    description = "本地目录适配器：扫描指定目录下的 md/txt/html/pdf/docx 文件"
    supports_incremental = True
    supports_pagination = False

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        raw = (config or {}).get("path") or settings.kb_local_path
        self.root = settings.resolve(raw)
        self.recursive = bool((config or {}).get("recursive", True))

    # ---------- 目录扫描 ----------
    def _iter_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        pattern = "**/*" if self.recursive else "*"
        files = [
            p
            for p in self.root.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(".")
        ]
        return sorted(files)

    async def health_check(self) -> bool:
        return self.root.exists() and self.root.is_dir()

    async def list_documents(
        self, cursor: str | None = None, limit: int = 100
    ) -> DocumentPage:
        if not self.root.exists():
            raise AdapterError(f"知识库目录不存在：{self.root}")
        items = []
        for p in self._iter_files():
            stat = p.stat()
            items.append(
                DocumentRef(
                    external_id=self._relative_id(p),
                    title=p.stem,
                    updated_at=datetime.fromtimestamp(stat.st_mtime),
                    version=str(int(stat.st_mtime)),
                    url=p.as_uri(),
                )
            )
        start = int(cursor or 0)
        chunk = items[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(items) else None
        return DocumentPage(items=chunk, next_cursor=next_cursor)

    @staticmethod
    def _relative_id(path: Path) -> str:
        return path.as_posix()

    def _resolve_path(self, external_id: str) -> Path:
        p = Path(external_id)
        return p if p.is_absolute() else self.root / p

    async def fetch_document(self, external_id: str) -> SourceDocument:
        path = self._resolve_path(external_id)
        if not path.exists():
            raise AdapterError(f"文档不存在：{external_id}")
        text = await asyncio.to_thread(self._read_text, path)
        stat = path.stat()
        return SourceDocument(
            external_id=external_id,
            title=self._guess_title(path, text),
            content=text,
            url=path.as_uri(),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            space=path.parent.name if path.parent != self.root else "根目录",
            metadata={"suffix": path.suffix.lower(), "path": path.as_posix()},
        )

    # ---------- 解析 ----------
    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            return _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".pdf":
            return _pdf_to_text(path)
        if suffix == ".docx":
            return _docx_to_text(path)
        raise AdapterError(f"不支持的文件类型：{suffix}")

    @staticmethod
    def _guess_title(path: Path, text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^#{1,2}\s+(.+)$", line)
            if m:
                return m.group(1).strip()[:80]
            if len(line) <= 40:
                return line[:80]
            break
        return path.stem


# ---------------- 各格式解析（轻量实现，够用即可） ----------------
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)


def _html_to_text(html: str) -> str:
    html = _SCRIPT_RE.sub(" ", html)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", html, flags=re.I)
    text = _TAG_RE.sub("", html)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        text = text.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pdf_to_text(path: Path) -> str:
    try:
        import pdfplumber  # 已安装在运行环境中
        with pdfplumber.open(str(path)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        return "\n\n".join(f"[第{i + 1}页]\n{t}" for i, t in enumerate(pages)).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF 解析失败 %s: %s", path, exc)
        raise AdapterError(f"PDF 解析失败：{path.name}") from exc


def _docx_to_text(path: Path) -> str:
    try:
        from docx import Document as Docx  # python-docx
        doc = Docx(str(path))
        parts: list[str] = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name or "").lower()
            if style.startswith("heading"):
                level = "".join(c for c in style if c.isdigit()) or "1"
                parts.append(f"{'#' * min(int(level), 6)} {t}")
            else:
                parts.append(t)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append("| " + " | ".join(cells) + " |")
        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        log.warning("DOCX 解析失败 %s: %s", path, exc)
        raise AdapterError(f"DOCX 解析失败：{path.name}") from exc
