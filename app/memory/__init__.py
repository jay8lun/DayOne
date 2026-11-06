"""记忆子系统：短期会话之外的「跨会话记忆」。

- `clock`：统一时间源（测试可时间旅行，才有办法验证「隔天还记得吗」）
- `store`：SQLite 结构化存储（事实带双时间轴、工作记忆带材料清单、审计留痕）
- `service`：写入闸门、冲突消解、四级遗忘

与 `app/rag/sessions.py` 的分工：
- `sessions` = 短期，本会话最近 N 轮原文，进程内、会丢；
- `memory` = 长期，跨会话、落库、可解释、可遗忘。
"""

from __future__ import annotations

from ..config import settings
from .service import MemoryService
from .store import MemoryStore

_service: MemoryService | None = None


def get_memory(refresh: bool = False) -> MemoryService:
    """进程内单例。DB 路径来自 `MEMORY_DB_PATH`，默认 `data/memory.db`。"""
    global _service
    if _service is None or refresh:
        store = MemoryStore(settings.resolve(settings.memory_db_path))
        _service = MemoryService(store)
    return _service


__all__ = ["MemoryService", "MemoryStore", "get_memory"]
