"""接口数据契约（Pydantic 模型）。定义在这里，前端/脚本照着这个写就不会错。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: str | None = Field(None, description="会话 id，不传则新建；多轮追问必须复用")
    top_k: int | None = Field(None, ge=1, le=20)


class SourceItem(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str = ""
    url: str | None = None
    snippet: str = ""
    score: float = 0.0


class CitationItem(SourceItem):
    used: bool = False
    hits: int = 0


class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_query: str = ""
    answer: str
    answer_plain: str = ""
    refused: bool = False
    sources: list[SourceItem] = []
    citations: list[CitationItem] = []
    grounding_score: float = 0.0
    citation_precision: float = 0.0
    confidence: float = 0.0
    coverage: float = 0.0
    retrieval: dict[str, Any] = {}
    trace: list[str] = []
    llm: dict[str, str] = {}
    latency_ms: int = 0


class ResetRequest(BaseModel):
    session_id: str


class IngestRequest(BaseModel):
    force: bool = Field(False, description="忽略内容哈希，全量重入库")
    reset: bool = Field(False, description="先清空向量库再入库")
    limit: int | None = Field(None, ge=1, le=5000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    question: str
    session_id: str | None = None
    note: str | None = None


class EvalRequest(BaseModel):
    dataset: str | None = None
    top_k: int | None = None
    limit: int | None = None
    with_generation: bool = Field(
        False, description="是否同时评估生成质量（会调用大模型，较慢且花钱）"
    )
