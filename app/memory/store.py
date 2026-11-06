"""记忆存储层：SQLite。

为什么不用向量库存这些：
- 事实槽位、流程状态都是「按 key 精确查」，向量检索在这里只会引入噪声
  （问「流程第几步」，语义相似度给出的答案是不可信的）；
- 事实需要事务性更新（有效期、被谁取代），向量库不擅长按键改单条记录。

向量只负责「模糊回忆」（可选，后续接入），结构化事实与流程状态留在 SQL。
两张表：
- `memory_facts`：长期事实 / 偏好，双时间轴（event_time 事件发生、record_time 记录时间）
- `work_items`：工作记忆，一行业务流程（做到哪步、材料交到哪）
另有 `audit_log`：所有写入、冲突、遗忘都留痕，线上排障只有这一个抓手。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id      TEXT    NOT NULL,
    subject       TEXT    NOT NULL,
    predicate     TEXT    NOT NULL,
    value         TEXT    NOT NULL,
    kind          TEXT    NOT NULL DEFAULT 'fact',      -- fact | preference
    confidence    REAL    NOT NULL DEFAULT 1.0,
    event_time    TEXT    NOT NULL,                     -- 事件真正发生的时间
    record_time   TEXT    NOT NULL,                     -- 我们记下它的时间
    valid_from    TEXT    NOT NULL,                     -- 时间轴上从何时有效
    valid_to      TEXT,                                 -- NULL = 至今有效
    source_turn   TEXT,                                 -- 用户原话 / 来源标识
    session_id    TEXT,
    status        TEXT    NOT NULL DEFAULT 'active',    -- active | pending | superseded | cold | tombstone
    superseded_by INTEGER,
    last_used_at  TEXT,
    use_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_facts_slot
    ON memory_facts(owner_id, subject, predicate, status);
CREATE INDEX IF NOT EXISTS idx_facts_owner
    ON memory_facts(owner_id, status);

CREATE TABLE IF NOT EXISTS work_items (
    owner_id      TEXT    NOT NULL,
    item_key      TEXT    NOT NULL,                     -- 流程标识，如 onboarding
    step          TEXT    NOT NULL,
    step_index    INTEGER NOT NULL DEFAULT 0,
    materials     TEXT    NOT NULL DEFAULT '[]',         -- JSON: [{name,status,at}]
    updated_at    TEXT    NOT NULL,
    PRIMARY KEY (owner_id, item_key)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    owner_id   TEXT,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_owner ON audit_log(owner_id, ts);
"""


class MemoryStore:
    """线程安全的轻量 DAL。sqlite3 是同步的，统一丢到线程池执行，避免堵事件循环。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # ---------- 连接 ----------
    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(str(self.path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA)
                conn.commit()
                self._conn = conn
            return self._conn

    async def init(self) -> None:
        await asyncio.to_thread(self._connect)

    def _run(self, fn):
        def wrapper():
            with self._lock:
                return fn(self._connect())

        return asyncio.to_thread(wrapper)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------- facts ----------
    async def insert_fact(self, row: dict[str, Any]) -> dict[str, Any]:
        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            cols = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            cur = conn.execute(
                f"INSERT INTO memory_facts ({cols}) VALUES ({marks})", tuple(row.values())
            )
            conn.commit()
            return dict(conn.execute("SELECT * FROM memory_facts WHERE id=?", (cur.lastrowid,)).fetchone())

        return await self._run(op)

    async def update_fact(self, fact_id: int, **fields: Any) -> None:
        def op(conn: sqlite3.Connection) -> None:
            sets = ", ".join(f"{k}=?" for k in fields)
            conn.execute(f"UPDATE memory_facts SET {sets} WHERE id=?", (*fields.values(), fact_id))
            conn.commit()

        await self._run(op)

    async def facts(
        self,
        owner_id: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        kind: str | None = None,
        status: str | None = "active",
        as_of: str | None = None,
        include_history: bool = False,
    ) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            sql = "SELECT * FROM memory_facts WHERE owner_id=?"
            args: list[Any] = [owner_id]
            if subject:
                sql += " AND subject=?"
                args.append(subject)
            if predicate:
                sql += " AND predicate=?"
                args.append(predicate)
            if kind:
                sql += " AND kind=?"
                args.append(kind)
            if status:
                sql += " AND status=?"
                args.append(status)
            sql += " ORDER BY valid_from DESC, id DESC"
            rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
            if as_of:
                # 时间点查询：那一刻有效的版本（valid_from <= T 且 valid_to 为空或 > T）。
                # 注意：这时**不能**再按「有没有 valid_to」过滤，否则历史版本会被误删。
                return [r for r in rows
                        if r["valid_from"] <= as_of and (not r["valid_to"] or r["valid_to"] > as_of)]
            if not include_history:
                rows = [r for r in rows if not r["valid_to"]]
            return rows

        return await self._run(op)

    async def purge_owner(self, owner_id: str) -> int:
        """删除某个 owner 的全部记忆与审计留痕，返回删除行数。

        只给**评测/演示重置**用：端到端评估会反复以同一批 owner 造档案，
        如果每次都在旧数据上叠加，第二次跑就会出现「上一条用例的导师」，
        指标会随运行次数漂移。生产链路不会调它。
        """

        def op(conn: sqlite3.Connection) -> int:
            deleted = 0
            for table in ("memory_facts", "work_items", "audit_log"):
                cur = conn.execute(f"DELETE FROM {table} WHERE owner_id=?", (owner_id,))
                deleted += cur.rowcount or 0
            conn.commit()
            return deleted

        return await self._run(op)

    async def facts_by_status(self, status: str) -> list[dict[str, Any]]:
        """跨用户扫某一状态的全部记忆（衰减任务用；量级小，SQLite 足够）。"""

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM memory_facts WHERE status=? ORDER BY id", (status,)
            ).fetchall()]

        return await self._run(op)

    async def get_fact(self, fact_id: int) -> dict[str, Any] | None:
        def op(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute("SELECT * FROM memory_facts WHERE id=?", (fact_id,)).fetchone()
            return dict(row) if row else None

        return await self._run(op)

    # ---------- work items ----------
    async def get_work(self, owner_id: str, item_key: str) -> dict[str, Any] | None:
        def op(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                "SELECT * FROM work_items WHERE owner_id=? AND item_key=?", (owner_id, item_key)
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            item["materials"] = json.loads(item["materials"] or "[]")
            return item

        return await self._run(op)

    async def upsert_work(self, owner_id: str, item_key: str, step: str, step_index: int,
                          materials: list[dict[str, Any]], updated_at: str) -> dict[str, Any]:
        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            conn.execute(
                """
                INSERT INTO work_items (owner_id, item_key, step, step_index, materials, updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(owner_id, item_key) DO UPDATE SET
                    step=excluded.step, step_index=excluded.step_index,
                    materials=excluded.materials, updated_at=excluded.updated_at
                """,
                (owner_id, item_key, step, step_index, json.dumps(materials, ensure_ascii=False), updated_at),
            )
            conn.commit()
            return dict(
                conn.execute(
                    "SELECT * FROM work_items WHERE owner_id=? AND item_key=?", (owner_id, item_key)
                ).fetchone()
            )

        return await self._run(op)

    async def list_work(self, owner_id: str) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM work_items WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)
            ).fetchall()]
            for r in rows:
                r["materials"] = json.loads(r["materials"] or "[]")
            return rows

        return await self._run(op)

    # ---------- audit ----------
    async def audit(self, ts: str, owner_id: str | None, action: str, detail: dict[str, Any]) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO audit_log (ts, owner_id, action, detail) VALUES (?,?,?,?)",
                (ts, owner_id, action, json.dumps(detail, ensure_ascii=False)),
            )
            conn.commit()

        await self._run(op)

    async def audits(self, owner_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if owner_id:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE owner_id=? ORDER BY id DESC LIMIT ?",
                    (owner_id, limit),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                item = dict(r)
                item["detail"] = json.loads(item["detail"] or "{}")
                out.append(item)
            return out

        return await self._run(op)

    async def stats(self) -> dict[str, int]:
        def op(conn: sqlite3.Connection) -> dict[str, int]:
            def count(sql: str, *args: Any) -> int:
                return int(conn.execute(sql, args).fetchone()[0])

            return {
                "facts_active": count("SELECT COUNT(*) FROM memory_facts WHERE status='active'"),
                "facts_pending": count("SELECT COUNT(*) FROM memory_facts WHERE status='pending'"),
                "facts_cold": count("SELECT COUNT(*) FROM memory_facts WHERE status='cold'"),
                "facts_superseded": count("SELECT COUNT(*) FROM memory_facts WHERE status='superseded'"),
                "work_items": count("SELECT COUNT(*) FROM work_items"),
                "audit_events": count("SELECT COUNT(*) FROM audit_log"),
            }

        return await self._run(op)
