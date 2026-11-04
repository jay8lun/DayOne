"""LangGraph 编排的检索纠错流程（Corrective RAG）。

```text
        ┌──────────┐    ┌──────────┐    ┌────────┐
START → │ condense │ →  │ retrieve │ →  │ grade  │ ─ sufficient ─→ END
        └──────────┘    └──────────┘    └────────┘
                              ↑               │ insufficient
                              └── rewrite ←───┘   (最多 max_corrective_rounds 次)
```

为什么用状态图而不是一串 if：
- 纠错是多轮循环，图能把「重试」显式建模，加节点（比如 rerank / 多路召回）不用改控制流；
- 每一步的中间状态（改写后的 query、覆盖率、置信度）都留在 state 里，方便打日志和评估定位。
生成阶段没放进图里，是因为要单独做 SSE 流式输出，拆开更简单（见 rag/service.py）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from ..config import settings
from ..core.logging import get_logger
from .llm import BaseLLM
from .prompts import CONDENSE_PROMPT, REWRITE_PROMPT
from .retriever import HybridRetriever, RetrievalResult

log = get_logger(__name__)


class RagState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    query: str
    round: int
    result: RetrievalResult
    sufficient: bool
    grade_reason: str
    trace: Annotated[list[str], operator.add]


def _format_history(history: list[dict[str, str]], limit: int = 4) -> str:
    if not history:
        return "（无）"
    lines = []
    for turn in history[-limit:]:
        who = "用户" if turn.get("role") == "user" else "助手"
        content = (turn.get("content") or "").replace("\n", " ")[:200]
        lines.append(f"{who}：{content}")
    return "\n".join(lines)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().strip('"“”').strip()
        if line:
            return line[:200]
    return ""


def build_rag_graph(retriever: HybridRetriever, llm: BaseLLM):
    async def condense(state: RagState) -> dict[str, Any]:
        question = state["question"].strip()
        history = state.get("history") or []
        if not history or not settings.enable_query_rewrite:
            return {"query": question, "round": 0, "trace": ["condense: 无历史，直用原问题"]}
        prompt = CONDENSE_PROMPT.format(history=_format_history(history), question=question)
        try:
            rewritten = _first_line(
                await llm.complete(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("问题改写失败，使用原问题: %s", exc)
            rewritten = ""
        final = rewritten or question
        return {
            "query": final,
            "round": 0,
            "trace": [f"condense: {question[:30]} -> {final[:60]}"],
        }

    async def retrieve(state: RagState) -> dict[str, Any]:
        result = await retriever.search(state["query"])
        return {
            "result": result,
            "trace": [
                f"retrieve(round={state.get('round', 0)}): {len(result.chunks)} 条, "
                f"conf={result.confidence:.2f}, cov={result.coverage:.2f}"
            ],
        }

    async def grade(state: RagState) -> dict[str, Any]:
        result = state["result"]
        sufficient = bool(result.chunks) and (
            result.confidence >= settings.min_confidence
            and (
                result.bm25_top >= settings.min_bm25_anchor
                or result.evidence >= settings.min_evidence
            )
        )
        reason = (
            f"conf={result.confidence:.2f}(阈值{settings.min_confidence}), "
            f"evidence={result.evidence:.2f}(阈值{settings.min_evidence}), "
            f"bm25={result.bm25_top:.1f}(阈值{settings.min_bm25_anchor})"
        )
        return {
            "sufficient": sufficient,
            "grade_reason": reason,
            "trace": [f"grade: {'sufficient' if sufficient else 'insufficient'} ({reason})"],
        }

    async def rewrite(state: RagState) -> dict[str, Any]:
        rnd = state.get("round", 0) + 1
        prompt = REWRITE_PROMPT.format(question=state["question"])
        try:
            new_query = _first_line(
                await llm.complete(
                    [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=120
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("查询改写失败: %s", exc)
            new_query = ""
        new_query = new_query or state["question"]
        return {"query": new_query, "round": rnd, "trace": [f"rewrite#{rnd}: {new_query[:60]}"]}

    def route_after_grade(state: RagState) -> str:
        if state.get("sufficient"):
            return "end"
        if not settings.enable_query_rewrite:
            return "end"
        if state.get("round", 0) >= settings.max_corrective_rounds:
            return "end"
        return "rewrite"

    graph = StateGraph(RagState)
    graph.add_node("condense", condense)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.set_entry_point("condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "end": END})
    graph.add_edge("rewrite", "retrieve")
    return graph.compile()
