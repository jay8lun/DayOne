"""接口数据契约（Pydantic 模型）。定义在这里，前端/脚本照着这个写就不会错。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    session_id: str | None = Field(None, description="会话 id，不传则新建；多轮追问必须复用")
    top_k: int | None = Field(None, ge=1, le=20)
    owner_id: str | None = Field(
        None, max_length=64,
        description="员工 id。传了才会注入长期记忆与流程状态（画像/进度），不传就是无状态问答",
    )


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
    context: dict[str, Any] = Field(default_factory=dict, description="上下文预算去向（可观测）")
    answer_source: str = Field(
        "knowledge_base",
        description="答案来源：knowledge_base | memory | tool | orchestrated | cache",
    )
    cached: bool = Field(False, description="是否命中答案缓存（知识库未变更，未重新检索生成）")
    cache_age_s: int = Field(0, description="缓存条目距今多少秒（命中时才有意义）")
    tool: dict[str, Any] = Field(default_factory=dict, description="本次调用到的工具（名称/参数/是否成功）")
    requires_confirmation: bool = Field(False, description="写操作：已生成待确认操作，尚未执行")
    pending_action: dict[str, Any] = Field(default_factory=dict,
                                          description="待确认操作（action_id / summary / expires_at）")
    recipe: str = Field("", description="多步编排命中的剧本 id（空表示未走编排）")
    plan: list[dict[str, Any]] = Field(default_factory=list,
                                       description="编排计划：每步做了什么、成没成")


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


# ---------------- 记忆 ----------------


class RememberRequest(BaseModel):
    """写入一条记忆。`event_time` 是**事件发生**时间，不是写入时间。"""

    owner_id: str = Field("demo-user", description="记忆归属（员工 id）")
    predicate: str = Field(..., min_length=1, max_length=40, description="槽位名，如 city / mentor / seat")
    value: str = Field(..., min_length=1, max_length=200)
    subject: str = Field("user", max_length=40)
    kind: str = Field("fact", pattern="^(fact|preference)$", description="事实（会覆盖）/ 偏好（留序列）")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    event_time: str | None = Field(None, description="事件真正发生的时间，如 2026-10-05；不传视为现在")
    source_turn: str | None = Field(None, max_length=200, description="用户原话，用于回答「你为什么记得」")
    session_id: str | None = None
    mode: str = Field("auto", pattern="^(auto|backfill)$",
                      description="事件时间早于已有记录时：auto=纠正旧值，backfill=补录历史")


class WorkStepRequest(BaseModel):
    owner_id: str = "demo-user"
    item_key: str = Field("onboarding", max_length=40, description="流程标识，如 onboarding / reimbursement")
    step: str = Field(..., max_length=60)
    step_index: int | None = Field(None, ge=0, description="步骤序号，用于挡住倒退")
    force: bool = Field(False, description="确认要回退流程时传 true")


class MaterialRequest(BaseModel):
    owner_id: str = "demo-user"
    item_key: str = "onboarding"
    name: str = Field(..., max_length=60, description="材料名，如 身份证复印件")
    status: str = Field("submitted", pattern="^(submitted|pending|rejected)$")


class DecayRequest(BaseModel):
    owner_id: str | None = Field(None, description="为空则跑全量")
    half_life_days: float | None = Field(None, ge=1, le=3650)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    tombstone_days: float | None = Field(None, ge=0, le=3650)
    dry_run: bool = Field(False, description="只算不写，用来看「会淘汰谁」")


# ---------------- 确认式写操作 ----------------
class ProposeRequest(BaseModel):
    owner_id: str = Field(..., max_length=64, description="员工 id（由前端会话带入）")
    tool: str = Field(..., max_length=60)
    args: dict[str, Any] = Field(default_factory=dict)
    client_key: str | None = Field(
        None, max_length=80,
        description="去重键：同一次意图重复提交（连点）只生成一条待确认操作",
    )


class ConfirmRequest(BaseModel):
    owner_id: str = Field(..., max_length=64)
    action_id: str = Field(..., max_length=40)


class CancelRequest(BaseModel):
    owner_id: str = Field(..., max_length=64)
    action_id: str = Field(..., max_length=40)
