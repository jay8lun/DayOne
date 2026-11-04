"""问答服务编排层：串起「取历史 → 检索纠错 → 生成 → 引用校验 → 落历史」。

对外提供两种调用方式：
- :meth:`RagService.ask`：一次性返回完整 JSON（评测、脚本、接口调试用）；
- :meth:`RagService.astream`：产出事件流（SSE 用），先给「来源」再逐字给答案，
  用户不用等整段生成完就能看到引用来自哪篇文档。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from ..config import settings
from ..core.errors import LLMError
from ..core.logging import get_logger
from .citations import build_sources, verify_and_attach
from .graph import build_rag_graph
from .llm import BaseLLM, get_llm
from .prompts import REFUSAL_TEXT, build_answer_messages
from .retriever import HybridRetriever, RetrievalResult
from .sessions import SessionStore, session_store

log = get_logger(__name__)


class RagService:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: BaseLLM | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.retriever = retriever or HybridRetriever(llm=self.llm)
        self.sessions = sessions or session_store
        self.graph = build_rag_graph(self.retriever, self.llm)

    async def warmup(self) -> int:
        return await self.retriever.refresh_index(force=True)

    # ---------- 检索阶段 ----------
    async def prepare(self, question: str, session_id: str | None) -> dict[str, Any]:
        history = self.sessions.history(session_id)
        state = await self.graph.ainvoke(
            {"question": question, "history": history, "trace": []}
        )
        result: RetrievalResult = state["result"]
        return {
            "standalone_query": state.get("query", question),
            "chunks": result.chunks,
            "sources": build_sources(result.chunks),
            "sufficient": bool(state.get("sufficient")),
            "confidence": result.confidence,
            "coverage": result.coverage,
            "grade_reason": state.get("grade_reason", ""),
            "retrieval": result.trace,
            "trace": state.get("trace", []),
            "history_turns": len(history),
        }

    def _base_payload(self, session_id: str, question: str, prep: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "question": question,
            "standalone_query": prep["standalone_query"],
            "refused": False,
            "answer": "",
            "answer_plain": "",
            "sources": [],
            "citations": [],
            "grounding_score": 0.0,
            "citation_precision": 0.0,
            "confidence": prep["confidence"],
            "coverage": prep["coverage"],
            "retrieval": prep["retrieval"],
            "trace": prep["trace"],
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "latency_ms": 0,
        }

    # ---------- 一次性问答 ----------
    async def ask(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        session_id = session_id or self.sessions.new_id()
        prep = await self.prepare(question, session_id)

        if not prep["sufficient"]:
            payload = self._base_payload(session_id, question, prep)
            payload.update(
                {
                    "refused": True,
                    "answer": REFUSAL_TEXT,
                    "answer_plain": REFUSAL_TEXT,
                    "sources": prep["sources"],
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            )
            self._remember(session_id, question, REFUSAL_TEXT)
            return payload

        messages = build_answer_messages(
            prep["standalone_query"], prep["chunks"], self.sessions.history(session_id)
        )
        raw = await self.llm.complete(messages)
        verified = verify_and_attach(raw, prep["chunks"])

        payload = self._base_payload(session_id, question, prep)
        payload.update(verified)
        payload["refused"] = False
        payload["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        self._remember(session_id, question, payload["answer_plain"])
        return payload

    # ---------- 流式问答 ----------
    async def astream(self, question: str, session_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        session_id = session_id or self.sessions.new_id()
        yield {"type": "start", "session_id": session_id, "question": question}

        try:
            yield {"type": "status", "stage": "retrieving", "message": "正在检索知识库…"}
            prep = await self.prepare(question, session_id)
            yield {
                "type": "sources",
                "sources": prep["sources"],
                "confidence": prep["confidence"],
                "coverage": prep["coverage"],
                "standalone_query": prep["standalone_query"],
                "sufficient": prep["sufficient"],
                "trace": prep["trace"],
            }

            if not prep["sufficient"]:
                for i in range(0, len(REFUSAL_TEXT), 12):
                    yield {"type": "delta", "text": REFUSAL_TEXT[i : i + 12]}
                payload = self._base_payload(session_id, question, prep)
                payload.update(
                    {
                        "refused": True,
                        "answer": REFUSAL_TEXT,
                        "answer_plain": REFUSAL_TEXT,
                        "sources": prep["sources"],
                    }
                )
                self._remember(session_id, question, REFUSAL_TEXT)
                payload["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                yield {"type": "done", **payload}
                return

            yield {"type": "status", "stage": "generating", "message": "正在生成回答…"}
            messages = build_answer_messages(
                prep["standalone_query"], prep["chunks"], self.sessions.history(session_id)
            )
            parts: list[str] = []
            async for token in self.llm.stream(messages):
                parts.append(token)
                yield {"type": "delta", "text": token}

            raw = "".join(parts)
            verified = verify_and_attach(raw, prep["chunks"])
            payload = self._base_payload(session_id, question, prep)
            payload.update(verified)
            payload["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            self._remember(session_id, question, payload["answer_plain"])
            yield {"type": "done", **payload}
        except LLMError as exc:
            log.error("问答失败: %s", exc)
            yield {"type": "error", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.exception("问答出现未预期错误")
            yield {"type": "error", "message": f"服务内部错误：{exc}"}

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self.sessions.append(session_id, "user", question)
        self.sessions.append(session_id, "assistant", answer)

    async def reset(self, session_id: str) -> None:
        self.sessions.reset(session_id)


_service: RagService | None = None


def get_service(refresh: bool = False) -> RagService:
    global _service
    if _service is None or refresh:
        _service = RagService()
    return _service
