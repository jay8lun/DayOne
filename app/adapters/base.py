"""知识库源适配器契约（Adapter Pattern）。

设计原则：
1. **主流程只依赖本文件的抽象**，不 import 任何具体平台 SDK；
2. 换知识库平台 = 新增一个 Adapter 子类（或改 ``KB_ADAPTER_CONFIG``），
   ingest / 检索 / 生成 / API 全部零改动；
3. 适配器只负责「把外部文档变成统一的 SourceDocument」，不做切块、不做向量化，
   保证职责单一、可替换。

接入新平台的两种方式：
- **零代码**：用内置 ``http_json`` 适配器 + JSON 字段映射配置（推荐，见 docs/适配器开发指南.md）；
- **写代码**：继承 ``KnowledgeSourceAdapter``，实现 3 个方法，注册进 registry。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class DocumentRef:
    """文档目录项（轻量，不含正文）。用于增量同步时先比对再决定是否拉取。"""

    external_id: str
    title: str = ""
    updated_at: datetime | None = None
    version: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceDocument:
    """统一文档模型：所有平台最终都收敛成这个结构。"""

    external_id: str
    title: str
    content: str
    url: str | None = None
    updated_at: datetime | None = None
    author: str | None = None
    space: str | None = None  # 所属知识空间 / 目录 / 频道
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """稳定 doc_id：优先用平台 id，避免标题改名导致引用失效。"""
        raw = f"{self.external_id}|{self.title}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.content.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DocumentPage:
    items: list[DocumentRef] = field(default_factory=list)
    next_cursor: str | None = None


class KnowledgeSourceAdapter(ABC):
    """知识库源适配器基类。"""

    #: 适配器唯一标识，配置里用它选择（KB_SOURCE=xxx）
    name: str = "base"
    #: 展示用描述
    description: str = "抽象基类"
    #: 是否支持按 updated_since 增量拉取
    supports_incremental: bool = False
    #: 是否支持游标分页
    supports_pagination: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

    # ---------- 必须实现 ----------
    @abstractmethod
    async def health_check(self) -> bool:
        """连通性检测。失败返回 False，不要抛异常给上层。"""

    @abstractmethod
    async def list_documents(
        self, cursor: str | None = None, limit: int = 100
    ) -> DocumentPage:
        """列出文档目录（不含正文）。"""

    @abstractmethod
    async def fetch_document(self, external_id: str) -> SourceDocument:
        """按平台 id 拉取文档正文。"""

    # ---------- 可选实现 ----------
    async def fetch_all(
        self, limit: int | None = None, updated_since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """默认实现：分页遍历目录 + 逐篇拉正文。子类可覆写以优化（批量接口 / 增量）。"""
        cursor: str | None = None
        seen = 0
        while True:
            page = await self.list_documents(cursor=cursor, limit=50)
            for ref in page.items:
                if updated_since and ref.updated_at and ref.updated_at <= updated_since:
                    continue
                doc = await self.fetch_document(ref.external_id)
                if not doc.title:
                    doc.title = ref.title or ref.external_id
                if not doc.url:
                    doc.url = ref.url
                if not doc.updated_at:
                    doc.updated_at = ref.updated_at
                doc.metadata.setdefault("source", self.name)
                yield doc
                seen += 1
                if limit and seen >= limit:
                    return
            cursor = page.next_cursor
            if not cursor:
                return

    async def aclose(self) -> None:
        """释放连接（HTTP 客户端等）。"""
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "supports_incremental": self.supports_incremental,
            "supports_pagination": self.supports_pagination,
            "config_keys": sorted(self.config.keys()),
        }
