"""RAG 核心层：检索 -> 纠错 -> 生成 -> 引用校验。"""

from .bm25 import BM25Index
from .citations import build_sources, verify_and_attach
from .graph import build_rag_graph
from .llm import BaseLLM, MockLLM, OpenAICompatLLM, get_llm
from .prompts import REFUSAL_TEXT, build_answer_messages, build_context
from .retriever import HybridRetriever, RetrievalResult, RetrievedChunk
from .service import RagService, get_service
from .sessions import SessionStore, session_store

__all__ = [
    "BM25Index",
    "build_sources",
    "verify_and_attach",
    "build_rag_graph",
    "BaseLLM",
    "MockLLM",
    "OpenAICompatLLM",
    "get_llm",
    "REFUSAL_TEXT",
    "build_answer_messages",
    "build_context",
    "HybridRetriever",
    "RetrievalResult",
    "RetrievedChunk",
    "RagService",
    "get_service",
    "SessionStore",
    "session_store",
]
