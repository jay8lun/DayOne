"""上下文组装：预算分层 + 旧的压缩 + 新的保留。

规则（来自 docs/演进设计.md 第 3 节，这里落成代码）：

```text
总预算（context_max_tokens）
├─ 模型输出预留（context_output_reserve）   先扣掉，不能靠挤占它来"塞内容"
├─ 系统规则（SYSTEM_PROMPT）               固定，不压
├─ 用户画像（记忆）                         固定，超了就截断
├─ 流程状态（工作记忆）                     固定，超了就截断
├─ 参考资料（chunks）                       **永不压缩**（压缩会打穿引用回验）
└─ 对话历史                                弹性：最近 N 轮原文 + 更早的摘要
```

顺序很重要：先把不可压的固定层和 chunks 扣掉，剩下的才是历史能用的预算。
反过来做（先塞历史再放资料）就会出现「资料被挤掉、回答没有依据」——
这正是很多"看起来能用"的实现跑长对话就跑飞的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from ..rag.llm import BaseLLM, get_llm
from ..rag.prompts import SYSTEM_PROMPT, build_context
from ..rag.retriever import RetrievedChunk
from .budget import estimate_tokens, truncate_to_tokens

log = get_logger(__name__)

SUMMARY_PROMPT = """把下面的历史对话压缩成一段摘要，供后续回答时参考。

硬要求：
- 只保留**事实**：人名、部门、系统名、数字、期限、结论；
- 保留用户已经确认过的偏好与约束（例如「只要结论」「已经提交过某某」）；
- 不要编造，不要评论，不要"用户询问了…"这种废话；
- 不超过 {limit} 字，直接输出摘要正文。

【历史对话】
{history}
"""

MERGE_PROMPT = """把「已有摘要」和「新增摘要」合并成一段更完整的历史摘要。

硬要求：
- 保留所有事实：人名、部门、系统名、数字、期限，一个都不能丢；
- 同一件事只留最新的说法（用户改了口，就按新的）；
- 去重、去客套话，不要出现"用户询问了…"；
- 不超过 {limit} 字，直接输出合并后的摘要正文。

【已有摘要】
{old}

【新增摘要】
{new}
"""


@dataclass
class Layer:
    """一个预算层的去向，用于可观测性（接口会把它吐出来）。"""

    name: str
    tokens: int
    kept: bool = True
    note: str = ""


@dataclass
class BuiltContext:
    messages: list[dict[str, str]]
    tokens: int
    budget: int
    layers: list[Layer] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "budget": self.budget,
            "compressed": self.compressed,
            "layers": [{"name": x.name, "tokens": x.tokens, "kept": x.kept, "note": x.note}
                       for x in self.layers],
            "trace": self.trace,
        }


class ContextBuilder:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    async def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        *,
        profile: str = "",
        work: str = "",
        max_tokens: int | None = None,
        session_id: str | None = None,
    ) -> BuiltContext:
        budget = max_tokens or settings.context_max_tokens
        usable = max(budget - settings.context_output_reserve, 200)
        layers: list[Layer] = []
        trace: list[str] = []

        # 1) 系统规则：固定层
        system_tokens = estimate_tokens(SYSTEM_PROMPT)
        layers.append(Layer("system", system_tokens, note="固定，不压缩"))

        # 2) 记忆注入：画像 + 流程状态（有就加，超了就截断，不参与压缩博弈）
        blocks: list[str] = []
        for name, text, cap in (
            ("profile", profile, settings.context_profile_tokens),
            ("work", work, settings.context_work_tokens),
        ):
            if not text.strip():
                layers.append(Layer(name, 0, kept=False, note="无内容"))
                continue
            trimmed = truncate_to_tokens(text, cap)
            blocks.append(trimmed)
            layers.append(Layer(name, estimate_tokens(trimmed), note=f"上限 {cap} tokens"))

        # 3) 参考资料：**整块保留**，绝不参与压缩（引用回验依赖原文）
        chunk_text = build_context(chunks)
        chunk_tokens = estimate_tokens(chunk_text)
        layers.append(Layer("chunks", chunk_tokens, note=f"{len(chunks)} 条，不压缩"))

        # 4) 用户问题
        question_tokens = estimate_tokens(question)

        remaining = usable - system_tokens - sum(x.tokens for x in layers if x.kept) - question_tokens
        trace.append(
            f"预算 {budget}（可用 {usable}）→ 已占 system={system_tokens} chunks={chunk_tokens} "
            f"memory={sum(x.tokens for x in layers if x.name in {'profile', 'work'})} 问题={question_tokens} "
            f"，历史可用 {remaining}"
        )

        # 5) 历史：最近 N 轮原文优先，更早的压缩成摘要
        history = history or []
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for block in blocks:
            messages.append({"role": "system", "content": block})

        recent_n = max(settings.context_recent_turns, 1) * 2
        recent = history[-recent_n:]
        older = history[:-recent_n]
        compressed = False

        summary_text = ""
        if older:
            # 入库后只压「新增轮次」，没有新增就直接复用库里的文本（一次模型都不调）
            # older 就是 history 的前缀，它的长度即「已压到第几条」
            summary_text, note = await self._rolling_summary(
                session_id,
                history,
                len(older),
                settings.context_summary_tokens,
            )
            compressed = True
            layers.append(Layer("history.summary", estimate_tokens(summary_text), note=note))

        # 预算不够时，先砍摘要（它信息密度最低且可重建），再砍更早的原文轮次
        recent_tokens = sum(estimate_tokens(t.get("content", "")) for t in recent)
        while summary_text and recent_tokens + estimate_tokens(summary_text) > max(remaining, 0):
            summary_text = ""
            trace.append("预算不足：先丢弃历史摘要")
            for layer in layers:
                if layer.name == "history.summary":
                    layer.kept = False
                    layer.note = "预算不足，已丢弃"

        while len(recent) > 1 and recent_tokens > max(remaining, 0):
            dropped = recent.pop(0)
            recent_tokens -= estimate_tokens(dropped.get("content", ""))
            trace.append(f"预算不足：丢弃更早的一轮（{dropped.get('role')}）")

        if summary_text:
            messages.append({"role": "system", "content": f"【历史摘要】\n{summary_text}"})
        for turn in recent:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                # 历史里的引用标记降级，避免模型把上一轮编号当这一轮
                messages.append({"role": role,
                                 "content": content.replace("[[", "【").replace("]]", "】")})
        layers.append(Layer("history.recent", recent_tokens, note=f"{len(recent)} 条原文"))

        user_block = f"【用户问题】\n{question}\n\n【参考资料】\n{chunk_text}"
        messages.append({"role": "user", "content": user_block})

        total = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if total > budget:
            trace.append(f"警告：估算 {total} tokens 超过预算 {budget}（chunks 不可压，属于预期内溢出）")

        return BuiltContext(messages=messages, tokens=total, budget=budget,
                            layers=layers, trace=trace, compressed=compressed)

    async def _rolling_summary(
        self,
        session_id: str | None,
        history: list[dict[str, str]],
        older_end: int,
        limit: int,
    ) -> tuple[str, str]:
        """L1 增量摘要 → 与 L2 合并 → 入库。返回 (摘要文本, 可观测性说明)。

        `older_end` 是「需要被压缩的历史」在完整 history 里的结束下标（绝对位置）。
        用绝对位置而不是相对长度，是因为 recent 窗口会随轮次变化，
        相对下标会让「压到哪了」在两次请求之间对不上。
        """
        store = None
        if session_id and settings.enable_summary_store:
            try:
                from .summary_store import get_summary_store

                store = await get_summary_store()
            except Exception as exc:  # noqa: BLE001
                log.warning("摘要库不可用，退化为每次重算: %s", exc)

        prev = None
        if store:
            try:
                prev = await store.get(session_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("读取历史摘要失败（忽略，按首次摘要处理）: %s", exc)

        covered = int(prev["covered_turns"]) if prev else 0
        # 会话被裁剪/重置过时 covered 可能越界，夹一下比抛错好
        covered = max(0, min(covered, older_end))
        fresh = history[covered:older_end]

        if prev and not fresh:
            return prev["text"], f"复用已入库摘要（覆盖 {covered} 轮，无新增）"

        new_text = await self._summarize(fresh, limit) if fresh else ""
        if not new_text:
            return (prev["text"] if prev else ""), "摘要生成失败，沿用旧摘要"

        if prev:
            merged = await self._merge(prev["text"], new_text, limit)
            note = f"L2 滚动摘要：{covered}→{older_end} 轮（本次增量 {len(fresh)} 轮）"
        else:
            merged = new_text
            note = f"L1 首次摘要：{older_end} 轮"

        if store:
            try:
                await store.put(
                    session_id,
                    older_end,
                    merged,
                    max_sessions=settings.context_summary_max_sessions,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("摘要入库失败（不影响本次回答）: %s", exc)
        return merged, note

    async def _merge(self, old: str, new: str, limit: int) -> str:
        """把新摘要揉进已有摘要。合并不了就直接拼，绝不因此丢历史。"""
        if not old:
            return new
        prompt = MERGE_PROMPT.format(limit=limit, old=old, new=new)
        try:
            text = await self.llm.complete(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=limit * 2
            )
            text = (text or "").strip()
            if text:
                return truncate_to_tokens(text, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("摘要合并失败，改为拼接: %s", exc)
        return truncate_to_tokens(f"{old}；{new}", limit)

    async def _summarize(self, turns: list[dict[str, str]], limit: int) -> str:
        """压缩旧历史。LLM 失败时退化为抽取式，保证离线也能跑。"""
        lines = []
        for turn in turns:
            who = "用户" if turn.get("role") == "user" else "助手"
            lines.append(f"{who}：{(turn.get('content') or '').replace(chr(10), ' ')[:120]}")
        raw = "\n".join(lines)
        prompt = SUMMARY_PROMPT.format(limit=limit, history=raw)
        try:
            text = await self.llm.complete(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=limit * 2
            )
            text = (text or "").strip()
            if text:
                return truncate_to_tokens(text, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("历史摘要失败，退化为抽取式: %s", exc)
        # 抽取式兜底：保留每轮开头（通常是结论），再去重
        seen: set[str] = set()
        picked = []
        for turn in turns:
            snippet = (turn.get("content") or "").replace("\n", " ").strip()
            key = snippet[:16]
            if not snippet or key in seen:
                continue
            seen.add(key)
            picked.append(snippet[:60])
        return truncate_to_tokens("；".join(picked), limit)


_builder: ContextBuilder | None = None


def get_builder(refresh: bool = False) -> ContextBuilder:
    global _builder
    if _builder is None or refresh:
        _builder = ContextBuilder()
    return _builder
