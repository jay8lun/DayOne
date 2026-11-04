"""Embedding：OpenAI 兼容接口优先，离线 hashing 兜底。

为什么要有兜底？
- 交付/评审时不一定有外网和 key，服务必须能 ``docker compose up`` 就起来；
- 单元测试不该联网。
所以 ``EMBEDDING_PROVIDER=auto`` 时：有 key 走真实 embedding，没 key 自动降级，
并在 ``/api/health`` 里明示当前用的是哪种（**不静默降级**）。

生产建议：``text-embedding-v3`` / ``bge-m3`` / ``text-embedding-3-small``。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

from ..config import settings
from ..core.errors import LLMError
from ..core.logging import get_logger

log = get_logger(__name__)

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_.+#\-]+")


class BaseEmbedder(ABC):
    provider: str = "base"
    model: str = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def tag(self) -> str:
        return f"{self.provider}__{_slug(self.model)}__{self.dim}"

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class OpenAICompatEmbedder(BaseEmbedder):
    """任何兼容 ``POST /v1/embeddings`` 的服务都能用（OpenAI / 通义 / 硅基流动 / Ollama / vLLM）。"""

    provider = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, dim: int) -> None:
        super().__init__(dim)
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        # 不同厂商对单次请求的条数上限不同（如百炼 text-embedding-v3 只允许 10 条），
        # 配置值只是初始值，遇到 400 batch size 错误会自动折半重试。
        self._max_batch = max(1, settings.embedding_batch_size)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=settings.llm_timeout,
                max_retries=3,
            )
        return self._client

    @staticmethod
    def _is_batch_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "batch size" in text or "batch_size" in text or "larger than" in text

    async def _embed_chunk(self, client, texts: list[str]) -> list[list[float]]:
        try:
            resp = await client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:  # noqa: BLE001
            # 厂商限制条数 → 折半重试，直到 1 条为止（1 条还失败就是真错误）
            if len(texts) > 1 and self._is_batch_error(exc):
                half = max(1, len(texts) // 2)
                if half < self._max_batch:
                    self._max_batch = half
                    log.warning("Embedding 单批条数超限，自动降为 %d 条/批", half)
                head = await self._embed_chunk(client, texts[:half])
                tail = await self._embed_chunk(client, texts[half:])
                return head + tail
            raise LLMError(f"Embedding 调用失败：{exc}") from exc
        vecs = sorted(resp.data, key=lambda d: d.index)
        out: list[list[float]] = []
        for item in vecs:
            vec = np.asarray(item.embedding, dtype=np.float32)
            out.append(_l2_normalize(vec).tolist())
        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        out: list[list[float]] = []
        batch = self._max_batch
        for i in range(0, len(texts), batch):
            part = [t[:6000] for t in texts[i : i + batch]]
            out.extend(await self._embed_chunk(client, part))
            await asyncio.sleep(0)  # 让出事件循环，避免大批量时阻塞
        return out


class HashingEmbedder(BaseEmbedder):
    """离线兜底：字符 n-gram 特征哈希 + 次线性 TF + L2 归一化。

    本质是「可以进向量库的 TF 向量」，中文靠 1/2/3-gram 覆盖，无需分词模型。
    质量不如 bge/OpenAI，但**确定性、零依赖、零网络**，适合 demo 与单测。
    """

    provider = "hashing"

    def __init__(self, dim: int | None = None) -> None:
        self.model = "char-ngram-hash"
        super().__init__(dim or settings.hashing_dim)

    def _features(self, text: str) -> dict[str, float]:
        text = text.lower()
        feats: dict[str, float] = {}
        for w in _ASCII_WORD.findall(text):
            if len(w) >= 2:
                feats[f"w:{w}"] = feats.get(f"w:{w}", 0.0) + 1.6
        for run in _CJK_RUN.findall(text):
            for n, weight in ((1, 0.4), (2, 1.4), (3, 0.8)):
                for i in range(len(run) - n + 1):
                    key = f"c{n}:{run[i : i + n]}"
                    feats[key] = feats.get(key, 0.0) + weight
        return feats

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        dim = self.dim
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for feat, count in self._features(text).items():
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % dim
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign * (1.0 + math.log(count))
            out.append(_l2_normalize(vec).tolist())
        return out


def build_embedder() -> BaseEmbedder:
    provider = settings.embedding_provider.lower()
    key = settings.effective_embedding_key
    if provider == "hashing":
        log.warning("使用离线 hashing embedding（演示模式，召回质量低于真实模型）")
        return HashingEmbedder()
    if provider in {"auto", "openai"}:
        if key:
            return OpenAICompatEmbedder(
                api_key=key,
                base_url=settings.effective_embedding_base,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
            )
        if provider == "openai":
            raise LLMError("EMBEDDING_PROVIDER=openai 但没有配置 EMBEDDING_API_KEY / LLM_API_KEY")
        log.warning("未配置 EMBEDDING_API_KEY，自动降级为离线 hashing embedding")
        return HashingEmbedder()
    raise LLMError(f"未知 EMBEDDING_PROVIDER: {provider}")


_embedder: BaseEmbedder | None = None


def get_embedder(refresh: bool = False) -> BaseEmbedder:
    global _embedder
    if _embedder is None or refresh:
        _embedder = build_embedder()
    return _embedder
