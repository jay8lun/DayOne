"""记忆子系统的行为测试。

重点验证三个「不做就会出事」的地方：
1. 事实更新要**封口不删除**（能回答「什么时候变的」）；
2. 低置信度抽取**不能覆盖**已有事实（宁可问一句，也不要把用户信息改错）；
3. 流程状态**幂等且挡倒退**（重复消息不能把进度打回去）。
"""

from __future__ import annotations

import asyncio

from app.memory.clock import FixedClock, use_clock
from app.memory.service import MemoryService
from app.memory.store import MemoryStore


def _svc(tmp_path) -> MemoryService:
    store = MemoryStore(tmp_path / "mem.db")
    asyncio.run(store.init())
    return MemoryService(store)


def test_fact_update_keeps_history(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            r1 = await svc.remember("u1", "city", "北京", source_turn="我住北京")
        with use_clock(FixedClock("2026-11-05T15:00:00")):
            r2 = await svc.remember("u1", "city", "杭州", source_turn="我上个月搬到杭州了")

        assert r1["action"] == "created"
        assert r2["action"] == "superseded"
        assert r2["replaced"]["value"] == "北京"

        current = await svc.recall("u1")
        assert [x["value"] for x in current] == ["杭州"]

        history = await svc.recall("u1", include_history=True)
        old = [x for x in history if x["value"] == "北京"][0]
        assert old["status"] == "superseded"
        assert old["valid_to"].startswith("2026-11-05")   # 封口时间 = 新事件时间
        assert old["superseded_by"] == r2["fact"]["id"]   # 可追溯被谁取代

    asyncio.run(case())


def test_time_travel_as_of(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.remember("u1", "city", "北京")
        with use_clock(FixedClock("2026-11-05T09:00:00")):
            await svc.remember("u1", "city", "杭州")

        before = await svc.recall("u1", as_of="2026-11-03")
        after = await svc.recall("u1", as_of="2026-11-06")
        assert [x["value"] for x in before] == ["北京"]
        assert [x["value"] for x in after] == ["杭州"]

    asyncio.run(case())


def test_same_value_is_idempotent(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            first = await svc.remember("u1", "seat", "3F-A12")
        with use_clock(FixedClock("2026-11-02T09:00:00")):
            again = await svc.remember("u1", "seat", "3F-A12")
        assert again["action"] == "noop"
        assert again["fact"]["id"] == first["fact"]["id"]
        assert again["fact"]["use_count"] == 1     # 被用了一次，权重升高

    asyncio.run(case())


def test_low_confidence_goes_pending(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.remember("u1", "mentor", "张伟", source_turn="我导师是张伟")
        with use_clock(FixedClock("2026-11-03T09:00:00")):
            weak = await svc.remember("u1", "mentor", "李娜", confidence=0.45,
                                      source_turn="可能是李娜？")
        assert weak["action"] == "pending"

        current = await svc.recall("u1", predicate="mentor")
        assert [x["value"] for x in current] == ["张伟"]     # 旧事实没被污染

        stats = await svc.store.stats()
        assert stats["facts_pending"] == 1

    asyncio.run(case())


def test_backfill_explicit_mode(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-05T09:00:00")):
            await svc.remember("u1", "city", "杭州")
            back = await svc.remember("u1", "city", "上海", event_time="2026-10-01", mode="backfill")
        assert back["action"] == "backfilled"
        assert back["fact"]["valid_to"].startswith("2026-11-05")

        current = await svc.recall("u1")
        assert [x["value"] for x in current] == ["杭州"]      # 现在的事实不变
        past = await svc.recall("u1", as_of="2026-10-15")
        assert [x["value"] for x in past] == ["上海"]         # 历史被补全

    asyncio.run(case())


def test_late_event_time_treated_as_correction(tmp_path):
    """今天记「北京」，用户却说「前天就搬到杭州了」——这是纠正，不是补录。"""

    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-06T10:00:00")):
            await svc.remember("u1", "city", "北京")
            r = await svc.remember("u1", "city", "杭州", event_time="2026-11-04")
        assert r["action"] == "corrected"
        assert r["replaced"]["status"] == "superseded"

        current = await svc.recall("u1")
        assert [x["value"] for x in current] == ["杭州"]      # 用户现在住杭州

    asyncio.run(case())


def test_preference_keeps_sequence(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.remember("u1", "answer_style", "简洁", kind="preference")
        with use_clock(FixedClock("2026-11-04T09:00:00")):
            r = await svc.remember("u1", "answer_style", "带表格", kind="preference")
        assert r["action"] == "preference_added"

        active = await svc.recall("u1", kind="preference")
        assert len(active) == 2          # 偏好留序列，不覆盖
        assert active[0]["value"] == "带表格"   # 按时间倒序，最新在前

    asyncio.run(case())


def test_work_step_idempotent_and_blocks_regression(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            a = await svc.set_work_step("u1", "onboarding", "提交材料", step_index=2)
            b = await svc.set_work_step("u1", "onboarding", "提交材料", step_index=2)
            c = await svc.set_work_step("u1", "onboarding", "入职培训", step_index=1)
            d = await svc.set_work_step("u1", "onboarding", "入职培训", step_index=1, force=True)
        assert a["action"] == "advanced"
        assert b["action"] == "noop"
        assert c["action"] == "stale_ignored"
        assert d["action"] == "advanced"
        assert d["work"]["step"] == "入职培训"

    asyncio.run(case())


def test_material_dedupe(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.mark_material("u1", "onboarding", "身份证复印件")
            await svc.mark_material("u1", "onboarding", "学历证明", status="pending")
        with use_clock(FixedClock("2026-11-02T09:00:00")):
            r = await svc.mark_material("u1", "onboarding", "学历证明")
        mats = {m["name"]: m["status"] for m in r["work"]["materials"]}
        assert len(mats) == 2                      # 同名不重复登记
        assert mats["学历证明"] == "submitted"      # 后来上报覆盖状态

        summary = await svc.work_summary("u1", "onboarding")
        assert "已交 2 项" in summary

    asyncio.run(case())


def test_decay_folds_to_cold_then_tombstone(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-01-01T09:00:00")):
            await svc.remember("u1", "hobby", "打羽毛球", kind="preference", confidence=0.9)

        # 400 天没人提：先冷归档
        with use_clock(FixedClock("2027-02-05T09:00:00")):
            r1 = await svc.decay("u1", half_life_days=30, threshold=0.2)
        assert len(r1["folded_to_cold"]) == 1
        assert r1["kept"] == 0

        # 冷归档放置超过墓碑期：立墓碑（不物理删除，可审计）
        with use_clock(FixedClock("2027-08-05T09:00:00")):
            r2 = await svc.decay("u1", tombstone_days=30)
        assert len(r2["tombstoned"]) == 1

        stats = await svc.store.stats()
        assert stats["facts_cold"] == 0
        assert stats["facts_active"] == 0
        assert stats["facts_superseded"] >= 0

    asyncio.run(case())


def test_decay_dry_run_does_not_write(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-01-01T09:00:00")):
            await svc.remember("u1", "hobby", "爬山", kind="preference")
        with use_clock(FixedClock("2027-01-01T09:00:00")):
            r = await svc.decay("u1", dry_run=True)
        assert len(r["folded_to_cold"]) == 1
        stats = await svc.store.stats()
        assert stats["facts_active"] == 1          # 试算不落库

    asyncio.run(case())


def test_profile_text_for_prompt(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.remember("u1", "city", "杭州", source_turn="我住杭州")
            await svc.remember("u1", "answer_style", "简洁", kind="preference")
        text = await svc.profile_text("u1")
        assert "【已知用户信息】" in text
        assert "【用户偏好】" in text
        assert "city=杭州" in text
        assert "原话" in text

    asyncio.run(case())


def test_audit_records_conflict_decision(tmp_path):
    svc = _svc(tmp_path)

    async def case():
        with use_clock(FixedClock("2026-11-01T09:00:00")):
            await svc.remember("u1", "city", "北京")
        with use_clock(FixedClock("2026-11-05T09:00:00")):
            await svc.remember("u1", "city", "杭州")
        logs = await svc.store.audits("u1")
        actions = [x["action"] for x in logs]
        assert "remember.created" in actions
        assert "remember.superseded" in actions
        detail = [x for x in logs if x["action"] == "remember.superseded"][0]["detail"]
        assert detail["old"] == "北京" and detail["new"] == "杭州"

    asyncio.run(case())
