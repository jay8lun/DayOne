"""多级摘要入库：把「会话摘要」从一次性计算变成可复用的资产。

## 之前的问题

上下文预算里，超过最近 N 轮的旧对话会在**每次请求**时重新摘要一遍。
代价有三个，都不小：

1. **重复花钱**：同一段历史，问第 5 句和第 6 句各压一次；
2. **结果会漂**：同样的历史，这次压出「导师是王琳」，下次压出「已确定导师」，
   用户会觉得这个助手「记性不稳」；
3. **用完就丢**：会话一结束摘要就没了，第二天接着聊又得从头压。

## 现在的做法：两级、增量、可复用

```text
L1 段摘要：把「上次覆盖位置之后」的新增轮次压成一段
      ↓  合并
L2 滚动摘要：L1 + 已有 L2 → 一段覆盖全部历史的摘要（长度封顶）
      ↓  入库
session_summaries（session_id, covered_turns, text, updated_at）
```

关键设计是 **covered_turns**：记「已经压到第几轮」。
下次只处理 `history[covered:older_end]`，没有新增就直接复用库里的文本，
一次 LLM 都不调。这样既省钱，又让同一段历史永远得到同一份摘要。

写库失败不影响回答：摘要只是上下文优化，不能因为它把主链路搞挂。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id    TEXT PRIMARY KEY,
    covered_turns INTEGER NOT NULL DEFAULT 0,   -- 已压缩到 history 的第几条（绝对下标）
    text          TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_updated ON session_summaries(updated_at);
"""


class SummaryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(SCHEMA)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def get(self, session_id: str) -> dict[str, Any] | None:
        def op() -> dict[str, Any] | None:
            with self._lock:
                row = self._get_conn().execute(
                    "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
                ).fetchone()
            return dict(row) if row else None

        return await asyncio.to_thread(op)

    async def put(
        self, session_id: str, covered_turns: int, text: str, *, max_sessions: int = 500
    ) -> None:
        """写入/更新滚动摘要，并顺手淘汰最旧的记录（表不能无限长）。"""

        def op() -> None:
            now = datetime.now().isoformat(timespec="seconds")
            with self._lock:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO session_summaries (session_id, covered_turns, text, updated_at)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(session_id) DO UPDATE SET"
                    " covered_turns=excluded.covered_turns, text=excluded.text,"
                    " updated_at=excluded.updated_at",
                    (session_id, covered_turns, text, now),
                )
                if max_sessions > 0:
                    conn.execute(
                        "DELETE FROM session_summaries WHERE session_id IN ("
                        "  SELECT session_id FROM session_summaries"
                        "  ORDER BY updated_at DESC LIMIT -1 OFFSET ?)",
                        (max_sessions,),
                    )
                conn.commit()

        await asyncio.to_thread(op)

    async def delete(self, session_id: str) -> None:
        def op() -> None:
            with self._lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
                conn.commit()

        await asyncio.to_thread(op)

    async def count(self) -> int:
        def op() -> int:
            with self._lock:
                return int(
                    self._get_conn().execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
                )

        return await asyncio.to_thread(op)


_store: SummaryStore | None = None


async def get_summary_store(path: str | Path | None = None) -> SummaryStore:
    """进程内单例。测试里可以传自己的临时路径，互不干扰。"""
    global _store
    if _store is None or path is not None:
        from ..config import settings

        target = path or settings.resolve(settings.context_summary_store)
        store = SummaryStore(target)
        await store.init()
        if path is None:
            _store = store
        return store
    return _store


def reset_summary_store(store: SummaryStore | None = None) -> None:
    """测试用：换掉单例，避免用例之间串库。"""
    global _store
    _store = store
