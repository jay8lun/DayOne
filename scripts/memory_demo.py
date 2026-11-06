"""记忆子系统演示：一条能讲清楚的故事。

跑法：
    python scripts/memory_demo.py

它会演示四件在真实使用中一定会发生的事：
1. 用户换了信息 —— 旧记录封口而不是删除，所以能回答「什么时候变的」；
2. 用户补述更早的事实 —— 补录进历史，不会篡改「现在」；
3. LLM 抽取不够自信 —— 进待确认，不覆盖已有事实；
4. 长期没人提 —— 冷归档再立墓碑；但**流程状态不参与遗忘**。

时间用 FixedClock 驱动，所以「隔天」「半年后」都是瞬间跳到，不用真等。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.clock import FixedClock, use_clock  # noqa: E402
from app.memory.service import MemoryService  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402

DB = Path("data/memory_demo.db")


def line(title: str = "") -> None:
    print("\n" + ("─" * 72 if not title else f"── {title} " + "─" * max(0, 68 - len(title))))


async def main() -> None:
    if DB.exists():
        DB.unlink()
    store = MemoryStore(DB)
    await store.init()
    svc = MemoryService(store)
    user = "employee-1001"

    line("11-01 新人第一天：说了住址和导师")
    with use_clock(FixedClock("2026-11-01T09:30:00")):
        r = await svc.remember(user, "city", "北京", source_turn="我住北京朝阳")
        print(f"  city=北京  -> action={r['action']}")
        await svc.remember(user, "mentor", "张伟", source_turn="我导师是张伟")
        await svc.set_work_step(user, "onboarding", "提交材料", step_index=1)
        await svc.mark_material(user, "onboarding", "身份证复印件")
        await svc.mark_material(user, "onboarding", "学历证明", status="pending")
        print(f"  流程：提交材料；已交 1 项，待补 1 项")

    line("11-05 他说：我上周搬到杭州了（事件时间 11-04）")
    with use_clock(FixedClock("2026-11-05T14:00:00")):
        r = await svc.remember(user, "city", "杭州", event_time="2026-11-04",
                               source_turn="我上周搬到杭州了")
        print(f"  city=杭州  -> action={r['action']}")
        print(f"  被取代：{r['replaced']['value']}（封口于 {r['replaced']['valid_to']}）")

    line("11-06 他补交材料，同时系统抽到一条不确定的信息")
    with use_clock(FixedClock("2026-11-06T10:00:00")):
        await svc.mark_material(user, "onboarding", "学历证明")          # 补交
        r = await svc.remember(user, "seat", "3F-A12", confidence=0.4,
                               source_turn="好像是在 3F 那边？")
        print(f"  seat（置信度 0.40）-> action={r['action']}（不写进事实，等确认）")

    line("现在它知道什么")
    print(await svc.profile_text(user))
    print()
    print(await svc.work_summary(user, "onboarding"))

    line("时间旅行：那一刻它认为你住哪？")
    for day in ("2026-11-02", "2026-11-04T12:00:00", "2026-11-06"):
        rows = await svc.recall(user, predicate="city", as_of=day)
        print(f"  as_of={day:22s} -> {rows[0]['value'] if rows else '（无记录）'}")

    line("7 个月没人提：还在，只是权重在降（不会因为一阵子没提就清空）")
    with use_clock(FixedClock("2027-06-01T09:00:00")):
        preview = await svc.decay(user, dry_run=True)
        print(f"  预计冷归档 {len(preview['folded_to_cold'])} 条，保留 {preview['kept']} 条")

    line("一年半没人提：降到阈值以下 → 冷归档（不删，仍可按时间查回）")
    with use_clock(FixedClock("2028-06-01T09:00:00")):
        preview = await svc.decay(user, dry_run=True)
        print(f"  会被冷归档：{[(x['predicate'], x['weight']) for x in preview['folded_to_cold']]}")
        done = await svc.decay(user)
        print(f"  执行后冷归档 {len(done['folded_to_cold'])} 条")

    line("冷归档再放 90 天：立墓碑（软删除，审计里仍可追到）")
    with use_clock(FixedClock("2028-10-01T09:00:00")):
        final = await svc.decay(user)
        print(f"  立墓碑 {len(final['tombstoned'])} 条")
        print(f"  记忆库统计：{await store.stats()}")

    line("流程状态不参与遗忘")
    work = await svc.store.get_work(user, "onboarding")
    print(f"  {work['item_key']} 仍在「{work['step']}」，材料 {len(work['materials'])} 项 —— 制度性状态该硬过期，不该淡忘")

    line("审计：每一次判断都留痕")
    for item in (await store.audits(user, limit=6))[::-1]:
        print(f"  {item['ts'][:16]}  {item['action']:<22} {item['detail']}")

    print("\n演示库：", DB, "（可随时删除，真实运行用 data/memory.db）\n")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
