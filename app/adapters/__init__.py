"""适配器层对外出口。"""

from .base import DocumentPage, DocumentRef, KnowledgeSourceAdapter, SourceDocument
from .http_json import HttpJsonAdapter
from .local_dir import LocalDirAdapter
from .registry import (
    available_adapters,
    build_adapters,
    create_adapter,
    get_adapter,
    load_plugins,
    register_adapter,
)
from .wecom import WeComDocAdapter

__all__ = [
    "DocumentPage",
    "DocumentRef",
    "KnowledgeSourceAdapter",
    "SourceDocument",
    "HttpJsonAdapter",
    "LocalDirAdapter",
    "WeComDocAdapter",
    "available_adapters",
    "build_adapters",
    "create_adapter",
    "get_adapter",
    "load_plugins",
    "register_adapter",
]
