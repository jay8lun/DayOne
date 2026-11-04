"""适配器层单测：字段映射、时间解析、未知适配器报错。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.adapters.http_json import _dig, _parse_dt
from app.adapters.local_dir import LocalDirAdapter
from app.adapters.registry import available_adapters, create_adapter
from app.core.errors import AdapterNotConfigured


def test_dig_nested_path():
    payload = {"data": {"items": [{"id": 7, "creator": {"name": "林岚"}}]}}
    assert _dig(payload, "data.items.0.id") == 7
    assert _dig(payload, "data.items.0.creator.name") == "林岚"
    assert _dig(payload, "data.missing.deep") is None


def test_parse_dt_formats():
    assert _parse_dt("2026-11-04") is not None
    assert _parse_dt("2026-11-04T10:00:00Z") is not None
    assert _parse_dt(1762000000) is not None
    assert _parse_dt(1762000000000) is not None
    assert _parse_dt("") is None


def test_registry_has_builtin_adapters():
    names = set(available_adapters())
    assert {"local_dir", "http_json", "wecom"} <= names


def test_unknown_adapter_raises():
    with pytest.raises(AdapterNotConfigured):
        create_adapter("not_exists", {})


def test_wecom_reports_not_configured():
    adapter = create_adapter("wecom", {})
    assert adapter.configured is False


def test_local_dir_lists_markdown(tmp_path):
    """不引入 pytest-asyncio，直接用 asyncio.run 跑协程，少一个依赖。"""
    import asyncio

    (tmp_path / "a.md").write_text("# 标题\n\n正文内容", encoding="utf-8")
    adapter = LocalDirAdapter({"path": str(tmp_path)})

    async def scenario():
        assert await adapter.health_check() is True
        page = await adapter.list_documents()
        assert len(page.items) == 1
        doc = await adapter.fetch_document(page.items[0].external_id)
        return doc

    doc = asyncio.run(scenario())
    assert doc.content.startswith("# 标题")
    assert doc.title == "标题"
