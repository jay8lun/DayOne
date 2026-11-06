"""待确认操作的持久化（`data/actions.db`）。

## 为什么单独一个库，而不是塞进记忆库

记忆库存的是「关于用户的事实」，这里存的是「一次待办的操作和它的执行结果」。
两者生命周期、清理策略、审计要求都不一样：
- 记忆要长期保留、参与衰减；
- 待确认操作是**短期票据**，确认或过期即终结，工单则要长期可查。
混在一张表里，迟早会出现「衰减把待确认的申请忘了」这种事故。

## 幂等键就用 `action_id`

用户双击「确认」、前端重试、消息队列重复投递，都会让同一个确认请求到达两次。
如果每次 `confirm` 都执行一遍，就会开两张单。
所以 `action_id` 本身即幂等键：第一次执行为 `executed`，之后再来直接返回**同一份结果**，
并标记 `replayed=True`——调用方能看出「这是重放，不是又办了一次」。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..memory import clock

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_actions (
    id            TEXT PRIMARY KEY,                 -- 同时是幂等键
    owner_id      TEXT NOT NULL,
    tool          TEXT NOT NULL,
    args          TEXT NOT NULL DEFAULT '{}',
    summary       TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | executed | cancelled | expired
    client_key    TEXT,                             -- 调用方去重键（防连点生成两条提议）
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    executed_at   TEXT,
    result        TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_owner ON pending_actions(owner_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_client
    ON pending_actions(owner_id, client_key) WHERE client_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_no     TEXT PRIMARY KEY,
    owner_id      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    payload       TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'pending',
    action_id     TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_owner ON tickets(owner_id, created_at);
"""


class ActionStore:
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
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------- 待确认操作 ----------
    async def create_action(
        self,
        *,
        owner_id: str,
        tool: str,
        args: dict[str, Any],
        summary: str,
        ttl_minutes: int,
        client_key: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._create_action_sync, owner_id, tool, args, summary, ttl_minutes, client_key
        )

    def _create_action_sync(
        self, owner_id: str, tool: str, args: dict[str, Any], summary: str,
        ttl_minutes: int, client_key: str | None,
    ) -> dict[str, Any]:
        now = clock.now()
        expires = now + timedelta(minutes=max(1, ttl_minutes))
        row = {
            "id": uuid.uuid4().hex[:16],
            "owner_id": owner_id,
            "tool": tool,
            "args": json.dumps(args, ensure_ascii=False),
            "summary": summary,
            "status": "pending",
            "client_key": client_key,
            "created_at": clock.iso(now) or "",
            "expires_at": clock.iso(expires) or "",
            "executed_at": None,
            "result": None,
        }
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO pending_actions (id, owner_id, tool, args, summary, status,"
                " client_key, created_at, expires_at, executed_at, result)"
                " VALUES (:id, :owner_id, :tool, :args, :summary, :status,"
                " :client_key, :created_at, :expires_at, :executed_at, :result)",
                row,
            )
            conn.commit()
        return self._row_to_action(row)

    async def get_action(self, action_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_action_sync, action_id)

    def _get_action_sync(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM pending_actions WHERE id = ?", (action_id,)
            ).fetchone()
        return self._row_to_action(dict(row)) if row else None

    async def find_pending_by_client_key(
        self, owner_id: str, client_key: str
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._find_by_key_sync, owner_id, client_key)

    def _find_by_key_sync(self, owner_id: str, client_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM pending_actions WHERE owner_id = ? AND client_key = ?",
                (owner_id, client_key),
            ).fetchone()
        return self._row_to_action(dict(row)) if row else None

    async def update_action(self, action_id: str, **fields: Any) -> None:
        await asyncio.to_thread(self._update_action_sync, action_id, fields)

    def _update_action_sync(self, action_id: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        if "result" in fields and not isinstance(fields["result"], str):
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        cols = ", ".join(f"{k} = :{k}" for k in fields)
        with self._lock:
            conn = self._get_conn()
            conn.execute(f"UPDATE pending_actions SET {cols} WHERE id = :id",
                         {**fields, "id": action_id})
            conn.commit()

    async def list_actions(
        self, owner_id: str, status: str | None = "pending", limit: int = 20
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_actions_sync, owner_id, status, limit)

    def _list_actions_sync(self, owner_id: str, status: str | None, limit: int) -> list[dict[str, Any]]:
        sql = "SELECT * FROM pending_actions WHERE owner_id = ?"
        args: list[Any] = [owner_id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._get_conn().execute(sql, args).fetchall()
        return [self._row_to_action(dict(r)) for r in rows]

    @staticmethod
    def _row_to_action(row: dict[str, Any]) -> dict[str, Any]:
        row["args"] = json.loads(row.get("args") or "{}")
        if row.get("result"):
            try:
                row["result"] = json.loads(row["result"])
            except json.JSONDecodeError:
                pass
        return row

    # ---------- 工单 ----------
    async def create_ticket(
        self, *, owner_id: str, kind: str, payload: dict[str, Any], action_id: str | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._create_ticket_sync, owner_id, kind, payload, action_id
        )

    def _create_ticket_sync(
        self, owner_id: str, kind: str, payload: dict[str, Any], action_id: str | None
    ) -> dict[str, Any]:
        now = clock.now()
        day = now.strftime("%Y%m%d")
        prefix = f"REQ-{day}-"
        with self._lock:
            conn = self._get_conn()
            seq = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE ticket_no LIKE ?", (prefix + "%",)
            ).fetchone()[0]
            ticket = {
                "ticket_no": f"{prefix}{seq + 1:03d}",
                "owner_id": owner_id,
                "kind": kind,
                "payload": json.dumps(payload, ensure_ascii=False),
                "status": "pending",
                "action_id": action_id,
                "created_at": clock.iso(now) or "",
            }
            conn.execute(
                "INSERT INTO tickets (ticket_no, owner_id, kind, payload, status, action_id, created_at)"
                " VALUES (:ticket_no, :owner_id, :kind, :payload, :status, :action_id, :created_at)",
                ticket,
            )
            conn.commit()
        ticket["payload"] = payload
        return ticket

    async def list_tickets(self, owner_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_tickets_sync, owner_id, limit)

    def _list_tickets_sync(self, owner_id: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM tickets WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.get("payload") or "{}")
            out.append(item)
        return out

    async def count_tickets(self, owner_id: str) -> int:
        return await asyncio.to_thread(self._count_tickets_sync, owner_id)

    def _count_tickets_sync(self, owner_id: str) -> int:
        with self._lock:
            return int(self._get_conn().execute(
                "SELECT COUNT(*) FROM tickets WHERE owner_id = ?", (owner_id,)
            ).fetchone()[0])

    async def purge_owner(self, owner_id: str) -> int:
        """清掉某个 owner 的待确认操作与工单，返回删除行数。

        和记忆库的同名方法一样，只服务**端到端评测**：每条用例都要从
        「没提过申请」的干净状态开始，否则「你最近一张工单是 …」会把上一条
        用例的工单带进来，判定就假了。
        """

        def op() -> int:
            with self._lock:
                conn = self._get_conn()
                deleted = 0
                for table in ("pending_actions", "tickets"):
                    cur = conn.execute(f"DELETE FROM {table} WHERE owner_id = ?", (owner_id,))
                    deleted += cur.rowcount or 0
                conn.commit()
                return deleted

        return await asyncio.to_thread(op)
