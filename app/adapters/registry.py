"""适配器注册表 + 插件加载。

- 内置适配器：local_dir / http_json / wecom
- 外部插件：``ADAPTER_PLUGINS=my_company.adapters.wecom2,my_company.adapters.confluence``
  被 import 的模块里调用 ``register_adapter(MyAdapter)`` 即可自动注册（或提供 ``ADAPTER`` 类变量）。
  插件代码放在 RAG 项目之外也能用，换公司时把插件目录拷过去就行。
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from ..config import settings
from ..core.errors import AdapterNotConfigured
from ..core.logging import get_logger
from .base import KnowledgeSourceAdapter
from .http_json import HttpJsonAdapter
from .local_dir import LocalDirAdapter
from .wecom import WeComDocAdapter

log = get_logger(__name__)

_REGISTRY: dict[str, type[KnowledgeSourceAdapter]] = {}


def register_adapter(cls: type[KnowledgeSourceAdapter]) -> type[KnowledgeSourceAdapter]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls!r} 缺少 name 属性")
    _REGISTRY[cls.name] = cls
    log.debug("注册适配器: %s -> %s", cls.name, cls)
    return cls


def available_adapters() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": name,
            "description": cls.description,
            "supports_incremental": cls.supports_incremental,
            "supports_pagination": cls.supports_pagination,
        }
        for name, cls in sorted(_REGISTRY.items())
    }


def load_plugins(dotted_paths: str) -> None:
    for path in [p.strip() for p in dotted_paths.split(",") if p.strip()]:
        try:
            module = importlib.import_module(path)
        except Exception as exc:  # noqa: BLE001
            log.error("适配器插件加载失败 %s: %s", path, exc)
            continue
        if hasattr(module, "ADAPTER"):
            register_adapter(module.ADAPTER)
        elif hasattr(module, "register"):
            module.register(register_adapter)  # type: ignore[attr-defined]
        else:
            log.warning("插件 %s 既没有 ADAPTER 也没有 register()，已跳过", path)


def create_adapter(name: str, config: dict[str, Any] | None = None) -> KnowledgeSourceAdapter:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise AdapterNotConfigured(
            f"未知知识库适配器：{name}。已注册：{', '.join(sorted(_REGISTRY))}"
        )
    return cls(config or {})


def _raw_configs() -> list[dict[str, Any]]:
    """支持两种写法：

    1) 单源：``{"base_url": "...", "fields": {...}}`` + ``KB_SOURCE=http_json``
    2) 多源：``[{"source": "local_dir", "path": "data/corpus"}, {"source": "http_json", ...}]``
    """
    raw = settings.kb_adapter_config.strip()
    if raw:
        parsed = json.loads(raw)  # config.py 已校验过 JSON 合法性
        if isinstance(parsed, list):
            return [dict(item) for item in parsed]
        return [dict(parsed) | {"source": settings.kb_source}]
    cfg: list[dict[str, Any]] = [{"source": settings.kb_source, "path": settings.kb_local_path}]
    for extra in [s.strip() for s in settings.kb_extra_sources.split(",") if s.strip()]:
        cfg.append({"source": extra, "path": settings.kb_local_path})
    return cfg


def build_adapters() -> list[KnowledgeSourceAdapter]:
    """按配置实例化所有启用的适配器。单个源失败不影响其它源（降级可用）。"""
    adapters: list[KnowledgeSourceAdapter] = []
    for cfg in _raw_configs():
        name = cfg.pop("source", settings.kb_source)
        try:
            adapters.append(create_adapter(name, cfg))
        except Exception as exc:  # noqa: BLE001
            log.error("适配器 %s 初始化失败（已跳过）: %s", name, exc)
    if not adapters:
        raise AdapterNotConfigured("没有任何可用的知识库适配器，请检查 KB_SOURCE / KB_ADAPTER_CONFIG")
    return adapters


def get_adapter(name: str | None = None) -> KnowledgeSourceAdapter:
    """按名字取一个适配器实例（用于 /api/kb/adapters/{name}/probe 探测）。"""
    name = name or settings.kb_source
    for adapter in build_adapters():
        if adapter.name == name:
            return adapter
    return create_adapter(name, {k: v for k, v in settings.kb_config.items()})


register_adapter(LocalDirAdapter)
register_adapter(HttpJsonAdapter)
register_adapter(WeComDocAdapter)
