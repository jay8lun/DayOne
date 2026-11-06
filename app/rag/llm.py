"""LLM 客户端：OpenAI 兼容协议 + 离线 mock。

- ``LLM_PROVIDER=openai``：DeepSeek / 通义千问 / 智谱 / 硅基流动 / 本地 vLLM 都走这个分支；
- ``LLM_PROVIDER=mock``：不联网的**抽取式**基线（从检索结果里挑句子 + 打引用），
  用于「没有 key 也能跑通全链路」「单测不依赖网络」「离线评测下限」。

为什么 mock 不做成「随机返回」？
因为评估要看真实指标。抽取式 mock 至少能让 Recall/Citation 指标跑出有意义的值，
同时提醒你：真实效果必须换上真模型再看一遍。
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..core.errors import LLMError
from ..core.logging import get_logger

log = get_logger(__name__)

Message = dict[str, str]


@dataclass
class ToolCall:
    """模型发起的一次工具调用。参数是**未解析的 JSON 字符串**——
    必须自己解析并校验，不能相信模型给的一定合法。"""

    id: str
    name: str
    arguments: str = "{}"

    def args(self) -> dict[str, Any]:
        try:
            data = json.loads(self.arguments or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class LLMToolResponse:
    """带工具的一轮回复：要么是文本，要么是调用请求（也可能两者都有）。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_.+#\-]+")
_CONTEXT_BLOCK = re.compile(
    r"^\[(\d+)\]\s*《(.+?)》(?:\s*>\s*(.+?))?\s*$\n(.*?)(?=^\[\d+\]\s*《|\Z)",
    re.M | re.S,
)
_STOP = {"的", "了", "是", "在", "和", "与", "有", "我", "你", "他", "她", "它", "吗", "呢", "怎么", "如何", "什么", "哪些", "可以", "需要", "应该", "请问"}


def tokenize(text: str) -> list[str]:
    """中英混排的轻量分词：英文按词，中文按 2-gram（单字兜底）。

    不引入 jieba 是有意的：少一个依赖、少一个模型加载，中文 2-gram 在短查询上
    召回与 BM25 打分已经够用，且不会把「差旅标准」切成奇怪的单字。
    """
    text = text.lower()
    tokens: list[str] = [w for w in _ASCII_WORD.findall(text) if len(w) >= 1]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def content_terms(text: str) -> set[str]:
    """抽取用于「覆盖度」判断的实词（去掉高频虚词）。"""
    return {t for t in tokenize(text) if t not in _STOP and len(t) >= 2}


class BaseLLM(ABC):
    provider = "base"
    model = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = 400,
    ) -> LLMToolResponse:
        """带工具的一轮。默认实现退化为纯文本——**这是有意的**：

        离线 mock、或某些不支持 function calling 的代理，不该让整条链路崩，
        只是「不会调用工具」，其他能力照常。
        """
        text = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        return LLMToolResponse(content=text)


class OpenAICompatLLM(BaseLLM):
    provider = "openai"

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=settings.llm_timeout,
                max_retries=2,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"大模型调用失败：{exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = 400,
    ) -> LLMToolResponse:
        client = self._get_client()
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"工具调用失败：{exc}") from exc

        msg = resp.choices[0].message
        calls = [
            ToolCall(
                id=getattr(tc, "id", "") or f"call_{i}",
                name=getattr(tc.function, "name", "") or "",
                arguments=getattr(tc.function, "arguments", "") or "{}",
            )
            for i, tc in enumerate(getattr(msg, "tool_calls", None) or [])
        ]
        return LLMToolResponse(content=(msg.content or "").strip(), tool_calls=calls)

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"大模型流式调用失败：{exc}") from exc


class MockLLM(BaseLLM):
    """离线抽取式基线：不生成新句子，只从参考资料里挑最相关的几句并标注引用。

    这就保证了「零幻觉」的下限，也让你在没有 key 时能验证：
    检索对不对、引用编号对不对、前端渲染对不对。
    """

    provider = "mock"
    model = "extractive-baseline"

    def _parse_context(self, prompt: str) -> tuple[list[dict[str, Any]], str]:
        blocks: list[dict[str, Any]] = []
        for m in _CONTEXT_BLOCK.finditer(prompt):
            blocks.append(
                {
                    "n": int(m.group(1)),
                    "title": m.group(2).strip(),
                    "heading": (m.group(3) or "").strip(),
                    "body": m.group(4).strip(),
                }
            )
        q = ""
        mq = re.search(r"【用户问题】\s*\n?(.*?)(?:\n【|\Z)", prompt, re.S)
        if mq:
            q = mq.group(1).strip()
        return blocks, q

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？；\n])", text)
        return [p.strip() for p in parts if len(p.strip()) >= 8]

    def _build(self, prompt: str) -> str:
        blocks, question = self._parse_context(prompt)
        if not blocks:
            return "知识库里暂时没有找到相关说明，建议联系 HR 或你的导师确认。"
        q_terms = content_terms(question)
        # 问「多少/几天/标准/上限」时，带数字的句子更可能是答案
        wants_number = any(w in question for w in ("多少", "几", "标准", "上限", "多久", "几天", "比例"))
        scored: list[tuple[float, int, str]] = []
        for b in blocks:
            for sent in self._sentences(b["body"]):
                terms = content_terms(sent)
                if not terms:
                    continue
                hit = len(q_terms & terms) / (len(q_terms) or 1)
                score = hit + 0.12 * (1 - b["n"] * 0.1)
                if wants_number and re.search(r"\d", sent):
                    score += 0.3
                scored.append((score, b["n"], sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for s in scored[:4] if s[0] > 0.02] or scored[:2]
        if not picked:
            return "知识库里暂时没有找到相关说明，建议联系 HR 或你的导师确认。"
        lines = ["根据知识库中的文档："]
        seen: set[str] = set()
        for score, n, sent in picked:
            key = sent[:24]
            if key in seen:
                continue
            seen.add(key)
            body = sent.strip().lstrip("-|• ").rstrip("。；")
            lines.append(f"- {body}[[{n}]]。")
        return "\n".join(lines)

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        # 问题改写类 prompt：没有参考资料。离线模式用「上一轮用户问题 + 本轮问题」拼接，
        # 这是最简单有效的伪指代消解（真实模型应做语义改写）。
        for marker in ("【最新问题】", "【原问题】"):
            if marker in prompt:
                tail = prompt.split(marker, 1)[1]
                question = ""
                for line in tail.splitlines():
                    line = line.strip()
                    if line and not line.startswith("【"):
                        question = line
                        break
                prev = ""
                past = prompt.split("【历史对话】", 1)[1].split("【最新问题】", 1)[0] if "【历史对话】" in prompt else ""
                for line in past.splitlines():
                    if line.strip().startswith("用户："):
                        prev = line.strip()[3:].strip()
                return f"{prev} {question}".strip() if prev else question
        if response_format_json:
            return json.dumps({"query": self._parse_context(prompt)[1][:120]}, ensure_ascii=False)
        return self._build(prompt)

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        text = await self.complete(messages)
        # 按字符块推送，模拟真实流式，前端体验一致
        for i in range(0, len(text), 6):
            yield text[i : i + 6]
            await asyncio.sleep(0.012)


def build_llm() -> BaseLLM:
    provider = settings.llm_provider.lower()
    if provider == "mock":
        log.warning("LLM_PROVIDER=mock：使用离线抽取式基线，回答不会生成新句子")
        return MockLLM()
    if provider in {"auto", "openai"}:
        if settings.llm_api_key:
            return OpenAICompatLLM(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
            )
        if provider == "openai":
            raise LLMError("LLM_PROVIDER=openai 但没有配置 LLM_API_KEY")
        log.warning("未配置 LLM_API_KEY，自动降级为 mock 抽取式基线（仅用于联调）")
        return MockLLM()
    raise LLMError(f"未知 LLM_PROVIDER: {provider}")


_llm: BaseLLM | None = None


def get_llm(refresh: bool = False) -> BaseLLM:
    global _llm
    if _llm is None or refresh:
        _llm = build_llm()
    return _llm
