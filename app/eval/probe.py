"""闭环探针（closed-loop probes）：隔天、换场景，验证记忆是否真的收敛。

## 为什么要单独做这个

单元测试能证明「规则写对了」，证明不了「系统在时间轴上会收敛」。而记忆系统最难的
恰恰是后者：

> 昨天教它的事，今天换个问法还答得出来吗？该忘的，真的忘干净了吗？
> 说「上个月就搬了」之后，它还会不会张口就报旧地址？

所以每个探针都是一条**带日期的脚本**：在固定时钟上跑到「明天 / 下周 / 一年后」，
跑完对结果断言，报告里写清「哪条断言、期望什么、实际什么」。全部探针各开一套临时库，
互不污染，可反复跑。

## 边界（避免自欺）

- 探针测的是**我们自己的记忆、路由、上下文逻辑**，不调外部模型：
  上下文探针用确定性 stub，否则会出现「今天过、明天不过」的假警报。
- 端到端问答质量（Recall / MRR / 引用有据率）由 `scripts/evaluate.py` 负责，两者不重叠。
- 探针**不允许**「跳过」或「重试到过」：失败就记失败，并写清是哪一条。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..memory.clock import FixedClock, iso, parse_time, use_clock
from ..memory.service import MemoryService
from ..memory.store import MemoryStore

log = get_logger(__name__)

# ============================ 结果结构 ============================


@dataclass
class Check:
    """一条断言。detail 必须是「期望 vs 实际」，否则失败时无法定位。"""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class ProbeReport:
    id: str
    title: str
    why: str
    category: str
    passed: bool = False
    checks: list[Check] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str = ""

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "why": self.why,
            "category": self.category,
            "passed": self.passed,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "checks": [c.to_dict() for c in self.checks],
            "timeline": self.timeline,
        }


# ============================ 探针上下文 ============================


class ProbeContext:
    """给探针用的小工具集：拨时钟、写记忆、读回来、下断言，顺手记时间轴。"""

    def __init__(
        self, service: MemoryService, store: MemoryStore, owner: str, tmpdir: Path
    ) -> None:
        self.service = service
        self.store = store
        self.owner = owner
        self.tmpdir = tmpdir
        self.checks: list[Check] = []
        self.timeline: list[str] = []
        self._now: datetime = FixedClock("2026-11-01T09:00:00").now()

    # ---------- 时间 ----------
    @contextmanager
    def at(self, when: str) -> Iterator[None]:
        """把系统时钟拨到 `when`（只在这段 with 内有效）。"""
        with use_clock(FixedClock(when)):
            self._now = parse_time(when) or self._now
            yield

    def note(self, text: str) -> None:
        self.timeline.append(f"{iso(self._now)[:16].replace('T', ' ')}  {text}")

    # ---------- 写入 ----------
    async def remember(
        self,
        predicate: str,
        value: str,
        *,
        kind: str = "fact",
        confidence: float = 1.0,
        event_time: str | None = None,
        mode: str = "auto",
        source_turn: str | None = None,
    ) -> str:
        result = await self.service.remember(
            self.owner, predicate, value, kind=kind, confidence=confidence,
            event_time=event_time, mode=mode, source_turn=source_turn,
        )
        self.note(f"记住 {predicate}={value} → action={result['action']}")
        return str(result["action"])

    async def step(self, item: str, name: str, index: int) -> str:
        result = await self.service.set_work_step(self.owner, item, name, step_index=index)
        self.note(f"流程推进 {item} → {name}(#{index}) action={result['action']}")
        return str(result["action"])

    async def material(self, item: str, name: str, status: str = "submitted") -> None:
        await self.service.mark_material(self.owner, item, name, status=status)
        self.note(f"材料 {name} → {status}")

    async def decay(self, **kwargs: Any) -> dict[str, Any]:
        result = await self.service.decay(self.owner, **kwargs)
        self.note(
            f"衰减执行 dry_run={result['dry_run']}：冷归档 {len(result['folded_to_cold'])} 条"
            f"（{', '.join(x['predicate'] for x in result['folded_to_cold']) or '无'}），"
            f"埋葬 {len(result['tombstoned'])} 条，保留 {result['kept']} 条"
        )
        return result

    # ---------- 读取 ----------
    async def values(self, predicate: str) -> list[str]:
        rows = await self.service.recall(self.owner, predicate=predicate)
        return [str(r["value"]) for r in rows]

    async def values_as_of(self, predicate: str, when: str) -> list[str]:
        rows = await self.service.recall(self.owner, predicate=predicate, as_of=when)
        return [str(r["value"]) for r in rows]

    async def profile(self) -> str:
        return await self.service.profile_text(self.owner)

    async def work_summary(self) -> str:
        return await self.service.work_summary(self.owner)

    # ---------- 断言 ----------
    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=bool(ok), detail=detail))

    def check_eq(self, name: str, expected: Any, actual: Any) -> None:
        self.check(name, expected == actual, f"期望 {expected!r}，实际 {actual!r}")

    def check_in(self, name: str, needle: str, haystack: str) -> None:
        self.check(name, needle in haystack, f"期望包含 {needle!r}；实际：{haystack[:160]!r}")

    def check_not_in(self, name: str, needle: str, haystack: str) -> None:
        self.check(name, needle not in haystack, f"期望不包含 {needle!r}；实际：{haystack[:160]!r}")


# ============================ 注册表 ============================


@dataclass
class Probe:
    id: str
    title: str
    why: str
    category: str
    fn: Callable[[ProbeContext], Any]


_PROBES: list[Probe] = []


def probe(category: str, pid: str, title: str, why: str):
    def wrap(fn: Callable[[ProbeContext], Any]):
        _PROBES.append(Probe(pid, title, why, category, fn))
        return fn

    return wrap


def registered() -> list[Probe]:
    return list(_PROBES)


# ============================ 探针（记忆：记住 / 冲突 / 遗忘） ============================


@probe(
    "retention", "retention_next_day",
    "隔天还在：昨天教的三件事，今天和两周后都答得出来",
    "新人第二天问「我住哪、导师是谁」，没人会把昨天的对话再重复一遍。"
    "这条探针证明长期事实确实跨会话落盘，而不是只活在当前上下文里。",
)
async def p_retention(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-02T10:00:00"):
        await ctx.remember("city", "杭州", source_turn="我住杭州")
        await ctx.remember("mentor", "王琳", source_turn="我导师是王琳")
        await ctx.remember("seat", "8F-A15", source_turn="我工位在 8 楼 A 区 15")

    with ctx.at("2026-11-03T09:20:00"):
        ctx.check_eq("次日 recall 城市", ["杭州"], await ctx.values("city"))
        ctx.check_eq("次日 recall 导师", ["王琳"], await ctx.values("mentor"))
        ctx.check_eq("次日 recall 工位", ["8F-A15"], await ctx.values("seat"))
        ctx.check_in("次日画像含导师", "王琳", await ctx.profile())

    with ctx.at("2026-11-16T09:00:00"):
        # 两周没说话，但只要没触发 decay，事实就该在
        ctx.check_eq("两周后仍记得城市", ["杭州"], await ctx.values("city"))
        ctx.check_eq("两周后仍记得工位", ["8F-A15"], await ctx.values("seat"))


@probe(
    "routing", "paraphrase_routing",
    "换个问法还认得出：口语化「还差啥」能路由到档案，规则问题不抢答",
    "用户不会照着字段名提问。这条探针用真实的路由判定（memory_covers）跑几种说法，"
    "确保「问我自己」被接住，而「问制度标准」仍然交给文档——两边都不能串。",
)
async def p_paraphrase(ctx: ProbeContext) -> None:
    from ..rag.service import memory_covers

    with ctx.at("2026-11-02T10:00:00"):
        await ctx.remember("city", "杭州")
        await ctx.step("onboarding", "提交材料", 1)
        await ctx.material("onboarding", "身份证复印件")
        await ctx.material("onboarding", "学历证明", status="pending")

    text = (await ctx.profile()) + "\n" + (await ctx.work_summary())
    cases = {
        "我的入职进度到哪了": True,
        "我入职材料还差啥？": True,
        "我还需要补什么材料": True,
        "我要交的材料交完了吗": True,
        "住宿标准是多少？": False,
        "我的报销标准是多少": False,      # 带「我」但在问规则，必须让给文档
        "公司食堂中午几点开饭？": False,
    }
    for q, expect in cases.items():
        ok = memory_covers(q, text)
        ctx.check(
            f"路由「{q}」→ {'档案' if expect else '文档'}",
            ok is expect,
            f"期望 {'命中档案' if expect else '不命中'}，实际 {'命中' if ok else '未命中'}",
        )


@probe(
    "conflict", "conflict_latest_wins",
    "前后矛盾以最新为准，但旧值仍可查：不覆盖、不丢历史",
    "「我搬到杭州了」之后若还答北京，就是记忆系统最致命的那种错。"
    "同时旧值不能删——要能回答「什么时候变的」。",
)
async def p_conflict(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-01T09:00:00"):
        await ctx.remember("city", "北京", source_turn="我住北京")
    with ctx.at("2026-11-04T15:00:00"):
        action = await ctx.remember("city", "杭州", source_turn="我搬到杭州了")

    ctx.check_eq("冲突处理动作", "superseded", action)
    ctx.check_eq("当前值是最新的", ["杭州"], await ctx.values("city"))
    ctx.check_not_in("画像里不再出现旧值", "=北京", await ctx.profile())
    ctx.check_eq("时间旅行：11-02 时应为北京", ["北京"], await ctx.values_as_of("city", "2026-11-02"))

    history = await ctx.service.recall(ctx.owner, predicate="city", include_history=True)
    ctx.check(
        "旧值没有被删除（可在历史里查到）",
        any(r["value"] == "北京" for r in history),
        f"历史记录值：{[r['value'] for r in history]}",
    )
    ctx.check(
        "旧记录被正确封口（valid_to 已写入）",
        all(r["valid_to"] for r in history if r["value"] == "北京"),
        f"北京记录 valid_to={[r['valid_to'] for r in history if r['value'] == '北京']}",
    )


@probe(
    "conflict", "correction_vs_backfill",
    "「你记错了」和「我以前在上海」是两件事，不能混",
    "两种说法的事件时间都早于旧记录，但意图相反：一个是纠正（旧值作废），"
    "一个是补录历史（现在的事实不变）。系统不猜，由调用方声明 mode。",
)
async def p_correction(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-05T09:00:00"):
        await ctx.remember("city", "北京")

    with ctx.at("2026-11-06T09:00:00"):
        corrected = await ctx.remember("city", "杭州", event_time="2026-11-04", mode="auto")
    ctx.check_eq("纠正动作", "corrected", corrected)
    ctx.check_eq("纠正后当前值", ["杭州"], await ctx.values("city"))
    ctx.check_eq(
        "纠正后旧值不再占用时间轴（11-05 也应为杭州）",
        ["杭州"], await ctx.values_as_of("city", "2026-11-05"),
    )

    with ctx.at("2026-11-06T09:10:00"):
        back = await ctx.remember("city", "上海", event_time="2019-03-01", mode="backfill")
    ctx.check_eq("补录动作", "backfilled", back)
    ctx.check_eq("补录不改变当前值", ["杭州"], await ctx.values("city"))
    ctx.check_eq(
        "补录补全了历史（2020 年应为上海）",
        ["上海"], await ctx.values_as_of("city", "2020-01-01"),
    )


@probe(
    "forgetting", "noise_cleanup",
    "该忘的忘干净：长期不提的闲聊降级并最终出画像",
    "记忆不是越多越好。长期无人调用的信息会按「时间 × 频次」衰减，"
    "降到阈值以下进冷归档，久置后立墓碑，此后不再出现在画像里。",
)
async def p_noise_cleanup(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-02T09:00:00"):
        await ctx.remember("hobby", "周末钓鱼", kind="fact")
        await ctx.remember("favorite_food", "麻辣香锅", kind="preference")

    with ctx.at("2027-11-02T09:00:00"):   # 一年没提
        preview = await ctx.decay(dry_run=True)
        ctx.check(
            "试算能预告会清掉谁",
            {x["predicate"] for x in preview["folded_to_cold"]} >= {"hobby", "favorite_food"},
            f"预计冷归档：{[x['predicate'] for x in preview['folded_to_cold']]}",
        )
        profile_after_preview = await ctx.profile()
        ctx.check("dry_run 不落库：画像仍含 hobby", "hobby" in profile_after_preview,
                  f"试算后画像：{profile_after_preview[:60]!r}")
        await ctx.decay()

    ctx.check_eq("冷归档后不再出现在 recall", [], await ctx.values("hobby"))
    ctx.check_not_in("冷归档后不在画像里", "hobby", await ctx.profile())

    with ctx.at("2028-06-01T09:00:00"):   # 冷归档超过 90 天
        result = await ctx.decay()
        ctx.check(
            "久置后立墓碑",
            {x["predicate"] for x in result["tombstoned"]} >= {"hobby", "favorite_food"},
            f"本轮埋葬：{[x['predicate'] for x in result['tombstoned']]}",
        )
    ctx.check_eq("墓碑后彻底读不到", [], await ctx.values("hobby"))


@probe(
    "forgetting", "pinned_survives_decay",
    "关键字段不该被当闲聊清掉：工号、部门、导师常驻",
    "衰减只按「时间 × 频次」，分不清「闲聊」和「关键身份信息」。"
    "不加保护的话，一年没人提工号，它就被清掉了——而工号是每次办手续都要用的。",
)
async def p_pinned(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-02T09:00:00"):
        await ctx.remember("employee_id", "E20261102")
        await ctx.remember("department", "平台研发部")
        await ctx.remember("hobby", "周末钓鱼")

    with ctx.at("2028-11-02T09:00:00"):   # 两年没提
        result = await ctx.decay()

    ctx.check_eq("工号仍在", ["E20261102"], await ctx.values("employee_id"))
    ctx.check_eq("部门仍在", ["平台研发部"], await ctx.values("department"))
    ctx.check_eq("闲聊已被清", [], await ctx.values("hobby"))
    ctx.check(
        "结果里报告了被保护条数",
        int(result.get("pinned", 0)) >= 2,
        f"pinned={result.get('pinned')}（期望 ≥2）",
    )


@probe(
    "work", "work_state_survives_decay",
    "流程状态不参与「淡忘」：一年后仍说得清走到哪一步、缺什么",
    "「材料交到哪一步」是制度性状态，到点该硬过期，但绝不该被慢慢淡忘。"
    "同时重复消息不能把进度打回去——这是线上最常见的污染源。",
)
async def p_work(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-02T09:00:00"):
        await ctx.step("onboarding", "提交材料", 1)
        await ctx.material("onboarding", "身份证复印件")
        await ctx.material("onboarding", "学历证明", status="pending")

    with ctx.at("2026-11-03T09:00:00"):
        await ctx.step("onboarding", "入职培训", 2)
        stale = await ctx.step("onboarding", "提交材料", 1)
        ctx.check_eq("重复的旧进度被忽略（幂等/防倒退）", "stale_ignored", stale)

    with ctx.at("2027-11-02T09:00:00"):
        await ctx.decay()
    summary = await ctx.work_summary()
    ctx.check_in("一年后仍记得当前步骤", "入职培训", summary)
    ctx.check_in("一年后仍记得缺的材料", "学历证明", summary)


@probe(
    "audit", "forgetting_traceable",
    "每个遗忘决策都留痕：谁、什么时候、为什么被清",
    "「它怎么把我说的忘了」必须答得出来。审计里记录 fact_id、权重、闲置天数，"
    "出错时能复盘，而不是只能说一句「系统就是这么设计的」。",
)
async def p_audit(ctx: ProbeContext) -> None:
    with ctx.at("2026-11-02T09:00:00"):
        await ctx.remember("hobby", "周末钓鱼")

    with ctx.at("2027-11-02T09:00:00"):
        first = await ctx.decay()
    ctx.check(
        "第一次衰减只降一级（冷归档，不跳级埋葬）",
        bool(first["folded_to_cold"]) and not first["tombstoned"],
        f"冷归档 {len(first['folded_to_cold'])} 条，埋葬 {len(first['tombstoned'])} 条",
    )

    with ctx.at("2028-01-02T09:00:00"):   # 冷归档后再放 60+ 天
        second = await ctx.decay()
    ctx.check(
        "第二次衰减才立墓碑",
        bool(second["tombstoned"]),
        f"本轮埋葬 {[x['predicate'] for x in second['tombstoned']]}",
    )

    rows = await ctx.store.audits(ctx.owner, limit=100)
    actions = [r["action"] for r in rows]
    ctx.check("审计里有冷归档记录", "decay.cold" in actions, f"实际动作：{sorted(set(actions))}")

    cold = [r for r in rows if r["action"] == "decay.cold"]
    detail = cold[0]["detail"] if cold else {}
    ctx.check(
        "冷归档记录了权重（可解释为什么被清）",
        isinstance(detail.get("weight"), (int, float)),
        f"detail={detail}",
    )
    tomb = [r for r in rows if r["action"] == "decay.tombstone"]
    ctx.check(
        "墓碑记录了闲置天数",
        bool(tomb) and isinstance(tomb[0]["detail"].get("idle_days"), (int, float)),
        f"detail={tomb[0]['detail'] if tomb else '无墓碑记录'}",
    )


@probe(
    "retention", "cross_user_isolation",
    "绝不串人：A 的信息不能出现在 B 的画像里",
    "记忆一旦按人隔离失效，就是数据事故，不是 bug。这条探针用两个员工 id 交叉读写，"
    "验证事实、流程状态、画像三条通道都不越界。",
)
async def p_isolation(ctx: ProbeContext) -> None:
    other = f"{ctx.owner}-b"
    with ctx.at("2026-11-02T09:00:00"):
        await ctx.remember("city", "杭州")
        await ctx.service.remember(other, "city", "北京", source_turn="我住北京")
        await ctx.step("onboarding", "提交材料", 1)
        await ctx.service.set_work_step(other, "onboarding", "入职培训", step_index=2)

    ctx.check_eq("A 只看到自己的城市", ["杭州"], await ctx.values("city"))
    b_rows = await ctx.service.recall(other, predicate="city")
    ctx.check_eq("B 只看到自己的城市", ["北京"], [r["value"] for r in b_rows])

    a_profile = await ctx.profile()
    b_profile = await ctx.service.profile_text(other)
    ctx.check_not_in("A 的画像不含 B 的信息", "北京", a_profile)
    ctx.check_not_in("B 的画像不含 A 的信息", "杭州", b_profile)

    a_work = await ctx.work_summary()
    b_work = await ctx.service.work_summary(other)
    ctx.check_in("A 的流程是「提交材料」", "提交材料", a_work)
    ctx.check_in("B 的流程是「入职培训」", "入职培训", b_work)


@probe(
    "write", "write_action_idempotent",
    "能办事但不能乱办：确认才执行、重复确认只开一张单、别人的单确认不了",
    "「帮我提个权限申请」如果点一次确认就开一张单，用户连点三下就是三张单，"
    "HR 那边会收到三条重复审批。这条探针验证提议不落库、确认幂等、越权被拒、过期作废。",
)
async def p_write(ctx: ProbeContext) -> None:
    from ..tools.actions import ActionError, reset_actions
    from ..tools.builtin import get_my_tickets
    from ..tools.store import ActionStore

    store = ActionStore(ctx.tmpdir / "actions.db")
    await store.init()
    svc = reset_actions(store, audit=False)
    args = {"system": "VPN", "reason": "需要在宿舍调试测试环境"}
    try:
        with ctx.at("2026-11-06T10:00:00"):
            a1 = await svc.propose(ctx.owner, "submit_permission_request", args, client_key="msg-1")
            a2 = await svc.propose(ctx.owner, "submit_permission_request", args, client_key="msg-1")
            ctx.check("连点两次只生成一条待办", a1["id"] == a2["id"] and bool(a2["deduped"]),
                      f"a1={a1['id']} a2={a2['id']} deduped={a2['deduped']}")
            ctx.check("提议阶段不产生工单", await store.count_tickets(ctx.owner) == 0,
                      f"当前工单数 {await store.count_tickets(ctx.owner)}")

            first = await svc.confirm(ctx.owner, a1["id"])
            second = await svc.confirm(ctx.owner, a1["id"])
            ticket_no = (first["result"].get("data") or {}).get("ticket_no", "")
            ctx.check("确认后开出一张工单", ticket_no.startswith("REQ-"), f"ticket_no={ticket_no!r}")
            ctx.check("重复确认是重放而非再开一单", second["replayed"] is True and second["result"] == first["result"],
                      f"replayed={second['replayed']}")
            ctx.check("工单只有一张", await store.count_tickets(ctx.owner) == 1,
                      f"当前工单数 {await store.count_tickets(ctx.owner)}")

            # 越权：别人不能确认我的待办
            foreign = await svc.propose(ctx.owner, "submit_permission_request",
                                        {"system": "生产数据库只读", "reason": "排障"},
                                        client_key="msg-2")
            try:
                await svc.confirm("intruder", foreign["id"])
                ctx.check("他人无法确认我的待办", False, "居然确认成功了")
            except ActionError as exc:
                ctx.check("他人无法确认我的待办", exc.code == "forbidden", f"code={exc.code}")
            ctx.check("越权尝试未产生工单", await store.count_tickets("intruder") == 0,
                      f"intruder 工单数 {await store.count_tickets('intruder')}")

            # 过期：把有效期调到过去，模拟「昨天的确认今天才点」
            await store.update_action(foreign["id"], expires_at="2020-01-01T00:00:00+00:00")
            try:
                await svc.confirm(ctx.owner, foreign["id"])
                ctx.check("过期操作作废", False, "过期了还能确认")
            except ActionError as exc:
                ctx.check("过期操作作废", exc.code == "expired", f"code={exc.code}")

        with ctx.at("2026-11-06T10:05:00"):
            info = await get_my_tickets(owner_id=ctx.owner)
        ctx.check("提交后能立刻查到工单", any(t["ticket_no"] == ticket_no for t in info["tickets"]),
                  f"查到 {[t['ticket_no'] for t in info['tickets']]}")
    finally:
        reset_actions(None)
        store.close()


# ============================ 探针（上下文：压缩不能吃掉证据） ============================


class _StubLLM:
    """确定性 stub：不联网、不随机，用来测**我们的**预算与压缩逻辑。"""

    provider = "stub"
    model = "stub-deterministic"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def complete(self, messages, *, temperature=None, max_tokens=None, **_: Any) -> str:
        if self.fail:
            raise RuntimeError("stub: 故意失败，用于验证抽取式兜底")
        text = messages[-1]["content"] if messages else ""
        # 回显一个「像摘要」的短句：真正的压缩质量由真实模型决定，这里只验流程
        return f"之前聊过：{text[:80]}"


def _fake_chunk(i: int) -> Any:
    from ..rag.retriever import RetrievedChunk

    return RetrievedChunk(
        chunk_id=f"c{i}",
        text=f"第 {i} 条制度原文：一线城市住宿标准 450 元/晚，需在 30 天内提交报销单。",
        metadata={"doc_title": "08-报销差旅与采购流程", "heading_path": "住宿标准",
                  "external_id": "doc-08"},
        final_score=0.9,
    )


def _fake_history(turns: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(turns):
        out.append({"role": "user", "content": f"第 {i + 1} 个问题：报销和发票的规则是什么？"})
        out.append({"role": "assistant",
                    "content": f"第 {i + 1} 个回答：住宿 450 元/晚，发票丢了要写情况说明，30 天内提交。"})
    return out


@probe(
    "context", "chunks_never_compressed",
    "资料永不被压缩：预算再紧，引用原文也必须完整进上下文",
    "引用回验靠「答案句子」与「chunk 原文」的词面重叠。一旦压缩 chunk，"
    "grounding 会假性下降，看起来像模型变差——所以压缩只能压对话，不能压证据。",
)
async def p_chunks_intact(ctx: ProbeContext) -> None:
    from ..config import settings
    from ..context.builder import ContextBuilder
    from ..context.budget import estimate_tokens
    from ..rag.prompts import build_context

    builder = ContextBuilder(llm=_StubLLM())
    chunks = [_fake_chunk(i) for i in range(5)]
    expected = build_context(chunks)

    with use_clock(FixedClock("2026-11-06T09:00:00")):
        tight = await builder.build("住宿标准是多少？", chunks, _fake_history(8), max_tokens=1200)
        loose = await builder.build("住宿标准是多少？", chunks, _fake_history(8), max_tokens=6000)

    for name, built in (("预算紧", tight), ("预算松", loose)):
        text = "\n".join(m["content"] for m in built.messages)
        ctx.check(f"{name}：chunk 原文完整保留", expected in text,
                  f"原文 {len(expected)} 字，上下文中{'命中' if expected in text else '缺失'}")

    chunk_layer = next((x for x in tight.layers if x.name == "chunks"), None)
    ctx.check(
        "预算紧时 chunks 层仍标记为保留",
        bool(chunk_layer and chunk_layer.kept),
        f"layer={chunk_layer}",
    )
    ctx.check(
        "预算紧时被牺牲的是历史（trace 有记录）",
        any("丢弃" in t or "compressed" in t for t in tight.trace) or tight.compressed,
        f"trace={tight.trace}",
    )
    ctx.check(
        "估算 token 数不超过预算太多（允许 chunks 不可压导致的溢出）",
        tight.tokens <= tight.budget * 1.6,
        f"tokens={tight.tokens} / budget={tight.budget}",
    )
    _ = estimate_tokens, settings


@probe(
    "context", "summary_survives_fallback",
    "摘要失败也不能丢历史：模型挂了就走抽取式兜底",
    "摘要要调模型，模型会失败。失败时如果直接丢掉旧历史，用户会突然「失忆」。"
    "这条探针把 stub 打死，验证仍有摘要、且最近两轮原文逐字保留。",
)
async def p_summary_fallback(ctx: ProbeContext) -> None:
    from ..context.builder import ContextBuilder

    history = _fake_history(6)
    builder = ContextBuilder(llm=_StubLLM(fail=True))
    with use_clock(FixedClock("2026-11-06T09:00:00")):
        built = await builder.build("发票丢了怎么办？", [_fake_chunk(0)], history, max_tokens=4000)

    text = "\n".join(m["content"] for m in built.messages)
    ctx.check("摘要仍存在（未因模型失败而消失）", "【历史摘要】" in text, f"消息角色：{[m['role'] for m in built.messages]}")
    ctx.check("标记为已压缩", built.compressed is True, f"compressed={built.compressed}")

    recent_user = history[-2]["content"].strip()
    ctx.check("最近一轮用户原话逐字保留", recent_user in text, f"期望包含 {recent_user!r}")
    ctx.check(
        "旧历史没有静默消失（要么在摘要里，要么 trace 记录了丢弃）",
        "【历史摘要】" in text or any("丢弃" in t for t in built.trace),
        f"trace={built.trace}",
    )


# ============================ 执行器 ============================


async def _run_one(p: Probe, owner: str = "probe-user") -> ProbeReport:
    report = ProbeReport(id=p.id, title=p.title, why=p.why, category=p.category)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"probe-{p.id}-"))
    store = MemoryStore(tmpdir / "memory.db")
    started = time.perf_counter()
    try:
        await store.init()
        ctx = ProbeContext(MemoryService(store), store, owner, tmpdir)
        await p.fn(ctx)
        report.checks = ctx.checks
        report.timeline = ctx.timeline
        report.passed = bool(ctx.checks) and all(c.ok for c in ctx.checks)
    except Exception as exc:  # noqa: BLE001 - 探针崩溃要记下来，不能吞
        report.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        report.passed = False
        log.error("探针 %s 执行失败: %s", p.id, exc)
    finally:
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            store.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return report


async def run_probes(only: list[str] | None = None, owner: str = "probe-user") -> list[ProbeReport]:
    """跑全部（或指定）探针。串行执行：`use_clock` 是全局状态，不能并发。"""
    reports: list[ProbeReport] = []
    for p in _PROBES:
        if only and p.id not in only:
            continue
        reports.append(await _run_one(p, owner))
    return reports


def summarize(reports: list[ProbeReport]) -> dict[str, Any]:
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    checks = [c for r in reports for c in r.checks]
    return {
        "probes": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "checks": len(checks),
        "checks_passed": sum(1 for c in checks if c.ok),
        "elapsed_ms": sum(r.elapsed_ms for r in reports),
        "by_category": {
            cat: {
                "total": len([r for r in reports if r.category == cat]),
                "passed": len([r for r in reports if r.category == cat and r.passed]),
            }
            for cat in sorted({r.category for r in reports})
        },
    }


CATEGORY_LABEL = {
    "retention": "记得住",
    "routing": "路由",
    "conflict": "冲突",
    "forgetting": "遗忘",
    "work": "工作记忆",
    "write": "写操作",
    "audit": "可解释",
    "context": "上下文",
}


def render_markdown(reports: list[ProbeReport], meta: dict[str, Any] | None = None) -> str:
    s = summarize(reports)
    meta = meta or {}
    lines = [
        "# 记忆 / 上下文闭环探针报告",
        "",
        f"- 生成时间：{meta.get('time', '')}",
        f"- 结果：**{s['passed']}/{s['probes']} 条探针通过**，"
        f"断言 {s['checks_passed']}/{s['checks']} 通过，耗时 {s['elapsed_ms']} ms",
        "- 说明：每条探针各开一套临时库，在固定时钟上跑到未来；不调用外部模型。",
        "",
        "## 总览",
        "",
        "| 探针 | 类型 | 结果 | 断言 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        ok = "通过" if r.passed else "**失败**"
        label = CATEGORY_LABEL.get(r.category, r.category)
        good = sum(1 for c in r.checks if c.ok)
        lines.append(f"| `{r.id}` | {label} | {ok} | {good}/{len(r.checks)} | {r.title} |")

    lines += ["", "## 明细", ""]
    for r in reports:
        lines.append(f"### `{r.id}` — {r.title}")
        lines.append("")
        lines.append(f"> 为什么要有这条：{r.why}")
        lines.append("")
        lines.append("时间轴：")
        lines.append("")
        lines.append("```text")
        lines.extend(r.timeline or ["（无）"])
        lines.append("```")
        lines.append("")
        if r.error:
            lines.append("执行异常：")
            lines.append("")
            lines.append("```text")
            lines.append(r.error.strip())
            lines.append("```")
            lines.append("")
        lines.append("| 断言 | 结果 | 期望 vs 实际 |")
        lines.append("| --- | --- | --- |")
        for c in r.checks:
            mark = "✓" if c.ok else "✗"
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {mark} | {detail} |")
        lines.append("")

    fails = [r for r in reports if not r.passed]
    lines += ["## 失败项", ""]
    if not fails:
        lines.append("无。")
    else:
        for r in fails:
            if r.error:
                lines.append(f"- `{r.id}`：执行异常（见上）")
            for c in r.failed:
                lines.append(f"- `{r.id}` / {c.name}：{c.detail}")
    return "\n".join(lines) + "\n"
