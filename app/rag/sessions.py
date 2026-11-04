"""会话记忆（多轮追问的基础）。

实现选择：**进程内内存 + TTL**。
- 单实例 / docker compose 演示足够，零依赖；
- 多副本部署时要换成 Redis（接口已经按 kv 抽象，替换只改这一个文件）；
- 只保留最近 N 轮，防止 prompt 无限膨胀把上下文窗口撑爆。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any

from ..config import settings


class SessionStore:
    def __init__(self, max_turns: int | None = None, ttl_minutes: int | None = None) -> None:
        self.max_turns = max_turns or settings.max_history_turns
        self.ttl = (ttl_minutes or settings.session_ttl_minutes) * 60
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._data.items() if now - v["updated"] > self.ttl]
        for k in expired:
            self._data.pop(k, None)

    def history(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        with self._lock:
            self._gc()
            item = self._data.get(session_id)
            return list(item["messages"]) if item else []

    def append(self, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        with self._lock:
            item = self._data.setdefault(
                session_id,
                {"messages": deque(maxlen=self.max_turns * 2), "updated": time.time()},
            )
            item["messages"].append({"role": role, "content": content})
            item["updated"] = time.time()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"sessions": len(self._data)}


session_store = SessionStore()
