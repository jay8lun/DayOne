"""记忆的业务规则层：写入闸门、冲突消解、衰减遗忘。

三条设计原则（对应 docs/演进设计.md）：

1. **不是每句话都值得记**：显式指令、结构化事件直接写；LLM 抽取的候选要过置信度闸。
2. **冲突不是「谁新谁赢」**：事实更新要留时间轴，偏好漂移要留序列，抽取错误只能进
   pending 等待确认。三条路分开走，混在一起就必然丢历史或者被错误覆盖。
3. **遗忘是分级而不是删除**：active → cold（归档、不参与召回）→ tombstone（墓碑）。
   流程状态不参与衰减，它该按 TTL 硬过期——把「材料交到哪一步」慢慢淡忘是事故。
"""

from __future__ import annotations

import math
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from . import clock
from .store import MemoryStore

log = get_logger(__name__)

VALID_KINDS = {"fact", "preference"}
VALID_MODES = {"auto", "backfill"}


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ================= 长期事实 / 偏好 =================
    async def remember(
        self,
        owner_id: str,
        predicate: str,
        value: str,
        *,
        subject: str = "user",
        kind: str = "fact",
        confidence: float = 1.0,
        event_time: str | None = None,
        source_turn: str | None = None,
        session_id: str | None = None,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """写入一条记忆，返回 `action` 说明发生了什么。

        action 取值：created / noop / superseded / corrected / backfilled / preference_added / pending

        - `event_time` 是**事件发生**的时间，不是写入时间：用户今天说「我上个月搬到杭州」，
          event_time 应在上个月。两者都存下来才回答得了「什么时候变的」。
        - `mode` 只在「事件时间早于已有记录生效时间」时有意义：
          `auto`（默认）= 纠正，旧记录作废；`backfill` = 补录历史，现在的事实不变。
        """
        predicate = predicate.strip()
        value = value.strip()
        if not predicate or not value:
            raise ValueError("predicate 与 value 不能为空")
        if kind not in VALID_KINDS:
            raise ValueError(f"kind 只支持 {VALID_KINDS}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode 只支持 {VALID_MODES}")
        confidence = max(0.0, min(1.0, float(confidence)))

        recorded = clock.now()
        event = clock.parse_time(event_time, default=recorded) or recorded
        now_iso = clock.iso(recorded)
        event_iso = clock.iso(event) or now_iso

        rows = await self.store.facts(owner_id, subject=subject, predicate=predicate, status="active")
        same = [r for r in rows if not r["valid_to"]]

        # 完全一样的值：幂等，只记一次使用
        for row in same:
            if row["value"] == value:
                await self._touch(row["id"])
                fresh = await self.store.get_fact(row["id"]) or row
                await self.store.audit(now_iso, owner_id, "remember.noop",
                                       {"predicate": predicate, "value": value, "fact_id": row["id"]})
                return {"action": "noop", "fact": fresh}

        base = {
            "owner_id": owner_id, "subject": subject, "predicate": predicate, "value": value,
            "kind": kind, "confidence": confidence, "event_time": event_iso,
            "record_time": now_iso, "source_turn": source_turn, "session_id": session_id,
            "use_count": 0, "last_used_at": now_iso,
        }

        # 偏好：保留序列，不覆盖（口味会变，但「以前喜欢过什么」本身是信息）
        if kind == "preference":
            fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                                 "valid_to": None, "status": "active"})
            await self.store.audit(now_iso, owner_id, "remember.preference",
                                   {"predicate": predicate, "value": value, "fact_id": fact["id"]})
            return {"action": "preference_added", "fact": fact}

        # 置信度不足：进 pending，不覆盖已有事实，等下次用户复述时再定
        if confidence < settings.memory_confidence_min:
            fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                                 "valid_to": None, "status": "pending"})
            await self.store.audit(now_iso, owner_id, "remember.pending",
                                   {"predicate": predicate, "value": value,
                                    "confidence": confidence, "fact_id": fact["id"]})
            return {"action": "pending", "fact": fact,
                    "reason": f"置信度 {confidence:.2f} 低于阈值 {settings.memory_confidence_min}"}

        # 无冲突：直接写入
        if not same:
            fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                                 "valid_to": None, "status": "active"})
            await self.store.audit(now_iso, owner_id, "remember.created",
                                   {"predicate": predicate, "value": value, "fact_id": fact["id"]})
            return {"action": "created", "fact": fact}

        # 有冲突：取时间轴上最新的那条作为「被取代者」
        old = max(same, key=lambda r: r["valid_from"])

        if event_iso < old["valid_from"]:
            # 事件时间早于旧值生效时间：两种完全不同的意图，必须由调用方声明，不能猜。
            #   mode="backfill" → 补录一段历史（现在的事实不变）：适合「我 2019 年在上海工作过」
            #   mode="auto"     → 视为纠正（旧记录作废）：适合「我上个月就搬了，你记错了」
            # 猜错的代价很实在：用户问「我住哪」，系统答一个刚被推翻的旧地址。
            if mode == "backfill":
                fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                                     "valid_to": old["valid_from"],
                                                     "status": "superseded",
                                                     "superseded_by": old["id"]})
                await self.store.audit(now_iso, owner_id, "remember.backfilled",
                                       {"predicate": predicate, "value": value, "fact_id": fact["id"],
                                        "until": old["valid_from"]})
                return {"action": "backfilled", "fact": fact, "current": old}

            fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                                 "valid_to": None, "status": "active"})
            # 旧记录收成零长度区间：它从来没在时间轴上站住过，但记录本身保留（可复盘）
            await self.store.update_fact(old["id"], valid_to=old["valid_from"],
                                         status="superseded", superseded_by=fact["id"])
            await self.store.audit(now_iso, owner_id, "remember.corrected",
                                   {"predicate": predicate, "old": old["value"], "new": value,
                                    "old_id": old["id"], "new_id": fact["id"],
                                    "old_valid_from": old["valid_from"], "event_time": event_iso})
            replaced = await self.store.get_fact(old["id"]) or old
            return {"action": "corrected", "fact": fact, "replaced": replaced}

        # 正常更新：旧记录封口（不删），新记录从事件时间起生效
        fact = await self.store.insert_fact({**base, "valid_from": event_iso,
                                             "valid_to": None, "status": "active"})
        await self.store.update_fact(old["id"], valid_to=event_iso, status="superseded",
                                     superseded_by=fact["id"])
        await self.store.audit(now_iso, owner_id, "remember.superseded",
                               {"predicate": predicate, "old": old["value"], "new": value,
                                "old_id": old["id"], "new_id": fact["id"],
                                "old_valid_to": event_iso})
        # 回读一次，调用方拿到的才是「封口之后」的真实状态
        replaced = await self.store.get_fact(old["id"]) or old
        return {"action": "superseded", "fact": fact, "replaced": replaced}

    async def recall(
        self,
        owner_id: str,
        *,
        predicate: str | None = None,
        kind: str | None = None,
        as_of: str | None = None,
        include_history: bool = False,
    ) -> list[dict[str, Any]]:
        """读取记忆。给 `as_of` 就是「那一刻它认为的事实」（时间旅行）。"""
        as_of_iso = clock.iso(clock.parse_time(as_of)) if as_of else None
        live = {"active", "superseded", "cold"}   # pending 待确认、tombstone 已埋葬都不召回
        if as_of_iso:
            rows = await self.store.facts(owner_id, predicate=predicate, kind=kind,
                                          status=None, as_of=as_of_iso)
            return [r for r in rows if r["status"] in live]
        if include_history:
            rows = await self.store.facts(owner_id, predicate=predicate, kind=kind,
                                          status=None, include_history=True)
            return [r for r in rows if r["status"] in live]
        return await self.store.facts(owner_id, predicate=predicate, kind=kind, status="active")

    async def profile_text(self, owner_id: str) -> str:
        """把已知信息拼成一段可直接塞进 prompt 的画像。"""
        rows = await self.recall(owner_id)
        if not rows:
            return ""
        facts, prefs = [], []
        for r in rows:
            line = f"{r['predicate']}={r['value']}（记录于 {r['record_time'][:10]}"
            if r["source_turn"]:
                line += f"，原话：{r['source_turn'][:30]}"
            line += "）"
            (prefs if r["kind"] == "preference" else facts).append(line)
        out = []
        if facts:
            out.append("【已知用户信息】\n" + "\n".join(f"- {x}" for x in facts))
        if prefs:
            out.append("【用户偏好】\n" + "\n".join(f"- {x}" for x in prefs))
        return "\n".join(out)

    # ================= 工作记忆 =================
    async def set_work_step(
        self,
        owner_id: str,
        item_key: str,
        step: str,
        *,
        step_index: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """推进流程状态。**幂等**：同一状态重复上报只算一次；倒退默认忽略（防止重复消息把流程打回去）。"""
        now_iso = clock.iso(clock.now()) or ""
        current = await self.store.get_work(owner_id, item_key)
        index = step_index if step_index is not None else (current["step_index"] if current else 0)

        if current and current["step"] == step:
            await self.store.audit(now_iso, owner_id, "work.noop",
                                   {"item": item_key, "step": step})
            return {"action": "noop", "work": current}

        if current and index < current["step_index"] and not force:
            await self.store.audit(now_iso, owner_id, "work.stale_ignored",
                                   {"item": item_key, "step": step, "index": index,
                                    "current_index": current["step_index"]})
            return {"action": "stale_ignored", "work": current,
                    "reason": "目标步骤早于当前步骤；确认要回退请传 force=true"}

        materials = current["materials"] if current else []
        work = await self.store.upsert_work(owner_id, item_key, step, index, materials, now_iso)
        work["materials"] = materials
        await self.store.audit(now_iso, owner_id, "work.step",
                               {"item": item_key, "from": current["step"] if current else None,
                                "to": step, "index": index})
        return {"action": "advanced", "work": work}

    async def mark_material(
        self,
        owner_id: str,
        item_key: str,
        name: str,
        *,
        status: str = "submitted",
    ) -> dict[str, Any]:
        """登记材料状态（submitted / pending / rejected）。同名材料以最后一次上报为准。"""
        now_iso = clock.iso(clock.now()) or ""
        current = await self.store.get_work(owner_id, item_key)
        materials = list(current["materials"]) if current else []
        entry = {"name": name, "status": status, "at": now_iso}
        for i, m in enumerate(materials):
            if m.get("name") == name:
                materials[i] = entry
                break
        else:
            materials.append(entry)

        work = await self.store.upsert_work(
            owner_id, item_key,
            current["step"] if current else "started",
            current["step_index"] if current else 0,
            materials, now_iso,
        )
        work["materials"] = materials
        await self.store.audit(now_iso, owner_id, "work.material",
                               {"item": item_key, "name": name, "status": status})
        return {"action": "material_recorded", "work": work}

    async def work_summary(self, owner_id: str, item_key: str | None = None) -> str:
        items = ([await self.store.get_work(owner_id, item_key)] if item_key
                 else await self.store.list_work(owner_id))
        items = [i for i in items if i]
        if not items:
            return ""
        lines = []
        for it in items:
            done = [m["name"] for m in it["materials"] if m.get("status") == "submitted"]
            todo = [m["name"] for m in it["materials"] if m.get("status") == "pending"]
            line = f"- {it['item_key']}：当前「{it['step']}」"
            if done:
                line += f"，已交 {len(done)} 项（{'、'.join(done)}）"
            if todo:
                line += f"，待补 {'、'.join(todo)}"
            lines.append(line)
        return "【当前流程状态】\n" + "\n".join(lines)

    # ================= 遗忘 =================
    async def decay(
        self,
        owner_id: str | None = None,
        *,
        half_life_days: float | None = None,
        threshold: float | None = None,
        tombstone_days: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """衰减淘汰：活跃 → 冷归档 → 墓碑。

        权重 = 置信度 × 时间衰减 × 使用频次。**只作用于长期事实/偏好**，
        工作记忆不参与（它按流程生命周期硬过期，不该被"慢慢淡忘"）。
        """
        half_life = half_life_days or settings.memory_half_life_days
        limit = threshold if threshold is not None else settings.memory_decay_threshold
        keep_days = tombstone_days or settings.memory_tombstone_days
        now_dt = clock.now()
        now_iso = clock.iso(now_dt) or ""

        owners = [owner_id] if owner_id else sorted(
            {r["owner_id"] for r in await self._all_active()}
        )
        cold, buried, kept = [], [], []
        for oid in owners:
            # 先给「已有的冷归档」拍快照：本轮刚降级的记录不该在同一次里被立刻埋葬，
            # 否则一次 decay 就把「归档」这级跳过去了，线上看不出是谁在什么时候被冷落的。
            cold_before = await self.store.facts(oid, status="cold")
            rows = await self.store.facts(oid, status="active")
            for row in rows:
                score = self._weight(row, now_dt, half_life)
                if score < limit:
                    cold.append({"fact_id": row["id"], "predicate": row["predicate"],
                                 "value": row["value"], "weight": round(score, 4)})
                    if not dry_run:
                        await self.store.update_fact(row["id"], status="cold")
                        await self.store.audit(now_iso, oid, "decay.cold",
                                               {"fact_id": row["id"], "weight": round(score, 4)})
                else:
                    kept.append(row["id"])

            for row in cold_before:
                idle = clock.days_between(now_dt, clock.parse_time(row["last_used_at"] or row["record_time"])
                                          or now_dt)
                if idle >= keep_days:
                    buried.append({"fact_id": row["id"], "predicate": row["predicate"]})
                    if not dry_run:
                        await self.store.update_fact(row["id"], status="tombstone")
                        await self.store.audit(now_iso, oid, "decay.tombstone",
                                               {"fact_id": row["id"], "idle_days": round(idle, 1)})

        return {"owners": len(owners), "folded_to_cold": cold, "tombstoned": buried,
                "kept": len(kept), "dry_run": dry_run,
                "params": {"half_life_days": half_life, "threshold": limit, "tombstone_days": keep_days}}

    def _weight(self, row: dict[str, Any], now_dt, half_life: float) -> float:
        last = clock.parse_time(row["last_used_at"] or row["record_time"]) or now_dt
        age = max(clock.days_between(now_dt, last), 0.0)
        recency = math.exp(-age / max(half_life, 1e-6) * math.log(2))
        freq = 1.0 + math.log1p(row["use_count"])
        # 偏好比事实更"耐放"：口味不会因为三个月没提就作废
        kind_bonus = 1.35 if row["kind"] == "preference" else 1.0
        return float(row["confidence"]) * recency * freq * kind_bonus

    async def _all_active(self) -> list[dict[str, Any]]:
        """扫全表拿 active 记忆（记忆量级小；换 PG 后改成 DISTINCT owner 查询即可）。"""
        return await self.store.facts_by_status("active")

    async def _touch(self, fact_id: int) -> None:
        row = await self.store.get_fact(fact_id)
        if not row:
            return
        await self.store.update_fact(
            fact_id,
            use_count=int(row["use_count"]) + 1,
            last_used_at=clock.iso(clock.now()),
        )
