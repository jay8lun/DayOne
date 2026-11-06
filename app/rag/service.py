"""问答服务编排层：串起「取历史 → 检索纠错 → 生成 → 引用校验 → 落历史」。

对外提供两种调用方式：
- :meth:`RagService.ask`：一次性返回完整 JSON（评测、脚本、接口调试用）；
- :meth:`RagService.astream`：产出事件流（SSE 用），先给「来源」再逐字给答案，
  用户不用等整段生成完就能看到引用来自哪篇文档。
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from typing import Any

from ..config import settings
from ..core.errors import LLMError
from ..core.logging import get_logger
from ..memory import get_memory
from .citations import build_sources, verify_and_attach
from .graph import build_rag_graph
from .llm import BaseLLM, content_terms, get_llm
from .prompts import MEMORY_SYSTEM_PROMPT, REFUSAL_TEXT, build_answer_messages
from .retriever import HybridRetriever, RetrievalResult
from .sessions import SessionStore, session_store

log = get_logger(__name__)


_SELF_HINTS = ("我", "本人", "自己", "咱们")
_STATE_HINTS = ("哪一步", "第几步", "几步", "进度", "到哪", "走到", "还要", "还差",
                "补", "交了", "提交", "状态", "材料", "到哪儿")
# 出现这些词说明用户在问「规则」而不是「我的状态」，必须走文档链路
_RULE_HINTS = ("标准", "多少", "几天", "几号", "怎么", "如何", "规定", "制度", "政策", "能不能")
# 出现这些词说明问的是「我提的单子」，档案里没有工单，必须交给工具层。
# 不加这条，「我提交的工单到哪一步了」会被记忆路由截胡，答成入职流程进度。
_TOOL_OWNED_HINTS = ("工单", "申请单", "审批", "报修", "开通", "单号", "票号")


def memory_covers(question: str, memory_text: str) -> bool:
    """判断这个问题该不该用「个人档案」来答。

    这是**启发式路由**，不是语义理解：同时满足三条才放行——
    ① 提到自己（我/本人）；② 问的是当前状态（进度/哪一步/还要补什么）；
    ③ 档案里确实有状态信息。出现「标准/多少/怎么」这类词说明在问规则，一律交给文档链路；
    出现「工单/审批/开通」这类词说明在问单据，交给工具链路（档案里没有工单）。

    为什么不用模型判断：这一层要的是**稳定可测**。等 P3 加了工具与意图路由节点，
    这里会被「模型判意图 + 工具白名单」取代，届时删掉即可。
    """
    text = (memory_text or "").strip()
    if not text:
        return False
    if not any(h in question for h in _SELF_HINTS):
        return False
    if any(h in question for h in _RULE_HINTS):
        return False
    if any(h in question for h in _TOOL_OWNED_HINTS):
        return False
    if not any(h in question for h in _STATE_HINTS):
        return False
    return any(h in text for h in _STATE_HINTS)


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
            "context": {},
            "answer_source": "knowledge_base",
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "latency_ms": 0,
        }

    # ---------- 一次性问答 ----------
    async def ask(
        self, question: str, session_id: str | None = None, owner_id: str | None = None
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        session_id = session_id or self.sessions.new_id()

        # 多步任务先走编排：它要串「记忆 → 文档 → 工具」，比单层路由更具体。
        # 命中不了会返回 None，下面照旧按 记忆 → 工具 → 文档 走。
        orchestrated = await self._orchestrated_answer(session_id, question, owner_id, t0)
        if orchestrated:
            return orchestrated

        # 先判「是不是在问我本人」。必须在检索**之前**判：
        # 一旦走了改写+检索，历史上下文会把「我的进度」改写成制度问题，档案就被绕过去了。
        early = await self._memory_answer(session_id, question, owner_id, t0)
        if early:
            return early

        tool_payload = await self._tool_answer(session_id, question, owner_id, t0)
        if tool_payload:
            return tool_payload

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

        messages, ctx_info = await self._compose(
            prep["standalone_query"], prep["chunks"], session_id, owner_id
        )
        raw = await self.llm.complete(messages)
        verified = verify_and_attach(raw, prep["chunks"])

        payload = self._base_payload(session_id, question, prep)
        payload.update(verified)
        payload["refused"] = False
        payload["context"] = ctx_info
        payload["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        self._remember(session_id, question, payload["answer_plain"])
        return payload

    # ---------- 流式问答 ----------
    async def astream(
        self, question: str, session_id: str | None = None, owner_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        session_id = session_id or self.sessions.new_id()
        yield {"type": "start", "session_id": session_id, "question": question}

        try:
            orchestrated = await self._orchestrated_answer(session_id, question, owner_id, t0)
            if orchestrated:
                plan = orchestrated.get("plan") or []
                yield {"type": "status", "stage": "orchestrating",
                       "message": f"正在执行 {len(plan)} 步任务…"}
                answer = orchestrated["answer"]
                for i in range(0, len(answer), 12):
                    yield {"type": "delta", "text": answer[i : i + 12]}
                yield {"type": "done", **orchestrated}
                return

            early_memory = await self._memory_blocks(owner_id)
            if early_memory and memory_covers(question, early_memory["text"]):
                yield {"type": "status", "stage": "memory", "message": "正在查你的入职档案…"}
                answer = await self._answer_from_memory(question, early_memory)
                for i in range(0, len(answer), 12):
                    yield {"type": "delta", "text": answer[i : i + 12]}
                payload = self._payload_from_memory(session_id, question, answer, t0)
                self._remember(session_id, question, payload["answer_plain"])
                yield {"type": "done", **payload}
                return

            tool_payload = await self._tool_payload(session_id, question, owner_id, t0)
            if tool_payload:
                stage = "confirm" if tool_payload.get("requires_confirmation") else "tool"
                yield {"type": "status", "stage": stage,
                       "message": ("正在生成待确认操作…" if stage == "confirm"
                                   else f"正在查询系统（{(tool_payload.get('tool') or {}).get('name', '')}）…")}
                answer = tool_payload["answer"]
                for i in range(0, len(answer), 12):
                    yield {"type": "delta", "text": answer[i : i + 12]}
                self._remember(session_id, question, tool_payload["answer_plain"])
                yield {"type": "done", **tool_payload}
                return

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
            messages, ctx_info = await self._compose(
                prep["standalone_query"], prep["chunks"], session_id, owner_id
            )
            parts: list[str] = []
            async for token in self.llm.stream(messages):
                parts.append(token)
                yield {"type": "delta", "text": token}

            raw = "".join(parts)
            verified = verify_and_attach(raw, prep["chunks"])
            payload = self._base_payload(session_id, question, prep)
            payload.update(verified)
            payload["context"] = ctx_info
            payload["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            self._remember(session_id, question, payload["answer_plain"])
            yield {"type": "done", **payload}
        except LLMError as exc:
            log.error("问答失败: %s", exc)
            yield {"type": "error", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.exception("问答出现未预期错误")
            yield {"type": "error", "message": f"服务内部错误：{exc}"}

    async def _orchestrated_answer(
        self, session_id: str, question: str, owner_id: str | None, t0: float
    ) -> dict[str, Any] | None:
        """多步编排：命中剧本时，把 记忆 / 文档 / 工具 串起来做一件事。

        返回里带 `plan`（每步做了什么、成没成），前端与接口都能看到，
        这样「它为什么查了这个」不需要靠猜。
        """
        from ..tools.orchestrator import get_orchestrator

        result = await get_orchestrator().run(question, owner_id, self.llm)
        if not result:
            return None

        answer = result.get("answer") or ""
        chunks = result.get("chunks") or []
        plan = result.get("plan") or []
        payload = self._payload_from_memory(
            session_id, question, answer, t0,
            source="orchestrated",
            trace=[f"orchestrate: {result.get('recipe')} → {len(plan)} 步"]
            + [f"  {p['kind']}: {p['step']} ({'ok' if p['ok'] else 'fail'})" for p in plan],
            retrieval={"mode": "orchestrated", "recipe": result.get("recipe"), "plan": plan},
        )
        payload["plan"] = plan
        payload["recipe"] = result.get("recipe")
        payload["tool"] = {"name": result.get("recipe"), "args": {}, "ok": True}

        if chunks and "[[" in answer:
            payload.update(verify_and_attach(answer, chunks))
        if result.get("requires_confirmation"):
            payload["requires_confirmation"] = True
            payload["pending_action"] = result.get("pending_action") or {}

        self._remember(session_id, question, payload["answer_plain"])
        return payload

    async def _memory_answer(
        self, session_id: str, question: str, owner_id: str | None, t0: float
    ) -> dict[str, Any] | None:
        """个人状态类问题的直通路径：跳过检索，只依据档案作答。"""
        memory = await self._memory_blocks(owner_id)
        if not memory or not memory_covers(question, memory["text"]):
            return None
        answer = await self._answer_from_memory(question, memory)
        payload = self._payload_from_memory(session_id, question, answer, t0)
        self._remember(session_id, question, payload["answer_plain"])
        return payload

    async def _tool_answer(
        self, session_id: str, question: str, owner_id: str | None, t0: float
    ) -> dict[str, Any] | None:
        """动作型问题的直通路径：查系统（只读）或生成待确认操作（写）。"""
        payload = await self._tool_payload(session_id, question, owner_id, t0)
        if payload:
            self._remember(session_id, question, payload["answer_plain"])
        return payload

    async def _tool_payload(
        self, session_id: str, question: str, owner_id: str | None, t0: float
    ) -> dict[str, Any] | None:
        from ..tools.router import answer_with_tool, route_tool

        info = await route_tool(self.llm, question, owner_id)
        if not info:
            return None

        # ① 写操作：答案是**确定性**的（不交给模型润色）。
        # 理由很实在：这里必须准确说出「还没提交、等你确认」，
        # 而模型很容易顺着用户的话说成「已经帮你提交了」。
        if info.get("requires_confirmation"):
            answer = f"{info['text']}\n\n确认后我才会提交；也可以直接说「取消」。"
            payload = self._payload_from_memory(
                session_id, question, answer, t0,
                source="tool",
                trace=[f"route: 写操作 → 生成待确认操作 {info['action_id']}（未执行）"],
                retrieval={"mode": "tool", "tool": info["tool"], "args": info["args"],
                           "requires_confirmation": True},
            )
            payload["tool"] = {"name": info["tool"], "args": info["args"], "ok": True}
            payload["requires_confirmation"] = True
            payload["pending_action"] = {
                "action_id": info["action_id"],
                "tool": info["tool"],
                "summary": info["summary"],
                "expires_at": info["expires_at"],
            }
            return payload

        # ② 工具失败（没权限、查不到、缺参数）：直接说原因，不让模型圆场
        if not info.get("ok", True):
            answer = info.get("text") or "系统查询没有成功，请稍后再试。"
            payload = self._payload_from_memory(
                session_id, question, answer, t0,
                source="tool",
                trace=[f"route: 工具 {info['tool']} 失败：{answer[:40]}"],
                retrieval={"mode": "tool", "tool": info["tool"], "args": info["args"], "ok": False},
            )
            payload["tool"] = {"name": info["tool"], "args": info["args"], "ok": False}
            return payload

        # ③ 只读查询成功：把结果交给模型组织成人话
        answer = await answer_with_tool(self.llm, question, info["text"])
        payload = self._payload_from_memory(
            session_id, question, answer, t0,
            source="tool",
            trace=[f"route: 动作型问题 → 工具 {info['tool']}（{info.get('elapsed_ms', 0)}ms，ok={info['ok']}）"],
            retrieval={"mode": "tool", "tool": info["tool"], "args": info["args"], "ok": info["ok"]},
        )
        payload["tool"] = {"name": info["tool"], "args": info["args"], "ok": info["ok"]}
        return payload

    def _payload_from_memory(
        self,
        session_id: str,
        question: str,
        answer: str,
        t0: float,
        *,
        source: str = "memory",
        trace: list[str] | None = None,
        retrieval: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plain = re.sub(r"\[\[\d+\]\]", "", answer).strip()
        return {
            "session_id": session_id,
            "question": question,
            "standalone_query": question,
            "refused": False,
            "answer": answer,
            "answer_plain": plain,
            "sources": [],
            "citations": [],
            "grounding_score": 0.0,
            "citation_precision": 0.0,
            "confidence": 0.0,
            "coverage": 0.0,
            "retrieval": retrieval or {"mode": "memory", "note": "个人状态问题，未检索文档"},
            "trace": trace or ["route: 命中个人档案，跳过知识库检索"],
            "context": {},
            "answer_source": source,
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    async def _memory_blocks(self, owner_id: str | None) -> dict[str, str] | None:
        """取个人档案（画像 + 流程状态）。没有 owner_id 就是纯无状态问答。"""
        if not owner_id or not settings.memory_inject_context:
            return None
        try:
            memory = get_memory()
            profile = await memory.profile_text(owner_id)
            work = await memory.work_summary(owner_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("读取个人档案失败（按无档案处理）: %s", exc)
            return None
        text = "\n".join(x for x in (profile, work) if x).strip()
        if not text:
            return None
        return {"profile": profile, "work": work, "text": text}

    async def _answer_from_memory(self, question: str, memory: dict[str, str]) -> str:
        messages = [
            {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
            {"role": "system", "content": f"【个人档案】\n{memory['text']}"},
            {"role": "user", "content": question},
        ]
        try:
            return (await self.llm.complete(messages, temperature=0.1)).strip()
        except LLMError as exc:
            log.warning("档案作答失败: %s", exc)
            return "你的档案里暂时读不到这条信息，建议直接问 HR 王琳确认。"

    async def _compose(
        self,
        question: str,
        chunks: list,
        session_id: str,
        owner_id: str | None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """组装送进模型的 messages：预算分层 + 记忆注入。

        关掉 `ENABLE_CONTEXT_BUDGET` 就退回老行为（全塞），方便做 A/B 对比。
        """
        history = self.sessions.history(session_id)
        if not settings.enable_context_budget:
            return build_answer_messages(question, chunks, history), {}

        # 延迟导入：context.builder 依赖 rag.prompts/llm，放在顶层会与 rag 包形成循环导入
        from ..context import get_builder

        profile = work = ""
        if owner_id and settings.memory_inject_context:
            try:
                memory = get_memory()
                profile = await memory.profile_text(owner_id)
                work = await memory.work_summary(owner_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("记忆注入失败（忽略，不影响回答）: %s", exc)

        ctx = await get_builder().build(
            question, chunks, history, profile=profile, work=work
        )
        return ctx.messages, ctx.to_dict()

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
