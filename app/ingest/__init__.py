"""入库层：适配器文档 -> chunk -> 向量 -> Chroma。"""

from .chunker import Chunk, chunk_document, split_sections
from .embedder import BaseEmbedder, HashingEmbedder, OpenAICompatEmbedder, get_embedder
from .pipeline import IngestionPipeline, IngestReport
from .store import VectorStore, get_store

__all__ = [
    "Chunk",
    "chunk_document",
    "split_sections",
    "BaseEmbedder",
    "HashingEmbedder",
    "OpenAICompatEmbedder",
    "get_embedder",
    "IngestionPipeline",
    "IngestReport",
    "VectorStore",
    "get_store",
]
