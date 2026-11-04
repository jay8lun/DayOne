"""接口级冒烟测试：健康检查 + 检索调试 + 未命中反馈。

说明：这些用例依赖本地已入库的向量库（首次运行 `python scripts/ingest.py`）。
没有数据时会自动跳过，不会把 CI 搞红。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _has_data(client) -> bool:
    r = client.get("/api/health")
    return r.json().get("knowledge_base", {}).get("chunks", 0) > 0


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "llm" in body and "knowledge_base" in body
    assert "local_dir" in body["adapters"]


def test_adapters_registry(client):
    r = client.get("/api/kb/adapters")
    assert r.status_code == 200
    assert {"local_dir", "http_json", "wecom"} <= set(r.json()["registered"])


def test_search_returns_chunks(client):
    if not _has_data(client):
        pytest.skip("向量库为空，先运行 scripts/ingest.py")
    r = client.post("/api/kb/search", json={"query": "报销时限", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"], "应召回至少一条"


def test_chat_cites_sources(client):
    if not _has_data(client):
        pytest.skip("向量库为空，先运行 scripts/ingest.py")
    r = client.post("/api/chat", json={"question": "报销单必须在多久内提交？"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["answer"]
    assert body["sources"], "拒答时也应返回召回到的近似来源"
    assert body["latency_ms"] >= 0


def test_unknown_question_refused(client):
    if not _has_data(client):
        pytest.skip("向量库为空")
    r = client.post("/api/chat", json={"question": "公司食堂中午几点开饭？"})
    body = r.json()
    assert body["refused"] is True
    assert "没有找到" in body["answer"]


def test_feedback_written(client):
    r = client.post("/api/feedback", json={"question": "测试反馈", "session_id": "t1"})
    assert r.status_code == 200
    path = settings.resolve("data/feedback.jsonl")
    assert path.exists()
