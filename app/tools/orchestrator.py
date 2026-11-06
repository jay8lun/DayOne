"""多步编排：把「记忆 / 文档 / 工具」串成一条有界的工作流。

## 为什么不做自由 agent

让模型自己决定「先调什么、再调什么」听起来更强，但在这个场景里代价很大：
- 步骤不可预测 → **没法评测**（同一句话每次走的路径不同）；
- 参数容易编 → 从文档里"抽"出一个不存在的系统名，用户照着提交就是错单；
- 出问题说不清 → 用户问「你为什么查了这个」，答不上来。

所以这里用**有界剧本（recipe）**：每个剧本是一个写死的步骤序列，
匹配条件是「多个信号同时出现」，步骤之间传递真实数据，返回值带完整 `plan`。
可以理解为「把专家脑子里的固定套路固化下来」，而不是让模型即兴发挥。

## 成功口径（先定，再写代码）

1. **可解释**：返回 `plan[]`，每步记 kind / 输入 / 输出 / 是否成功；
2. **参数有出处**：上游步骤的值只能来自记忆、工具或文档，模型不能凭空补；
3. **有界**：步数 ≤ 4、工具调用 ≤ 3；任一步失败立即停，并说明第几步失败；
4. **写操作仍然只提议**：编排里遇到写工具只生成待确认操作，绝不执行；
5. **不抢单意图**：匹配条件要求多个信号同时命中，普通制度问题照旧走文档检索。

## 四个剧本（都能在语料/数据里找到依据）

| 剧本 | 链路 | 依赖关系 |
| --- | --- | --- |
| `mentor_contact` | 记忆（导师=王琳）→ 工具（查组织目录） | 导师名是上一步的输出，作为下一步的入参 |
| `first_week_env` | 文档（第一周要勾哪些系统）→ 提案（生成待确认申请） | 系统清单来自文档片段，不是模型编的 |
| `ticket_rule_check` | 文档（敏感权限有效期）→ 工具（查我的工单） | 规则与个人状态两路数据合并回答 |
| `travel_standard` | 记忆（我的职级）→ 文档（差旅标准表） | 用职级把「制度表格」定位到具体一行，而不是把整张表甩给用户 |
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..core.logging import get_logger
from ..rag.llm import BaseLLM, content_terms

log = get_logger(__name__)

MAX_STEPS = 4
MAX_TOOL_CALLS = 3

# 步骤名要给用户看，不能直接甩内部工具名
_TOOL_STEP_LABELS = {
    "lookup_colleague": "查组织目录",
    "get_my_tickets": "查我的工单",
    "get_my_onboarding_status": "查入职流程",
}


# ============================ 上下文 ============================


@dataclass
class Step:
    name: str
    kind: str            # memory | doc | tool | propose
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.name, "kind": self.kind, "ok": self.ok, "detail": self.detail}


@dataclass
class RecipeContext:
    question: str
    owner_id: str | None
    llm: BaseLLM
    # 可注入：单测/探针用自己的临时库，不去碰全局单例
    memory: Any = None
    plan: list[Step] = field(default_factory=list)
    tool_calls: int = 0
    chunks: list[Any] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    # ---------- 记录 ----------
    def step(self, name: str, kind: str, ok: bool, detail: str = "") -> None:
        if len(self.plan) >= MAX_STEPS:
            raise BudgetExceeded(f"超过最大步数 {MAX_STEPS}")
        self.plan.append(Step(name, kind, ok, detail[:160]))

    # ---------- 读取记忆 ----------
    async def profile(self) -> dict[str, str]:
        if not self.owner_id:
            self.step("读取入职档案", "memory", False, "没有员工身份")
            return {}
        from ..memory import get_memory

        memory = self.memory if self.memory is not None else get_memory()
        rows = await memory.recall(self.owner_id)
        data = {str(r["predicate"]): str(r["value"]) for r in rows}
        summary = "、".join(f"{k}={v}" for k, v in list(data.items())[:4]) or "无记录"
        # ok 表示「这一步执行成功了吗」，不是「有没有查到东西」。
        # 档案里没记录也是一次成功的读取——前端把这种情况画成红色失败，
        # 用户会以为系统坏了；真正的失败只有「没有身份，压根没查」。
        self.step("读取入职档案", "memory", True, summary)
        return data

    # ---------- 调工具（只读） ----------
    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.tool_calls >= MAX_TOOL_CALLS:
            raise BudgetExceeded(f"超过最大工具调用次数 {MAX_TOOL_CALLS}")
        from .registry import registry

        spec = registry.get(name)
        if spec is None:
            self.step(_TOOL_STEP_LABELS.get(name, name), "tool", False, "工具未注册")
            return {"ok": False, "data": {}, "error": "工具未注册"}
        if not spec.read_only:
            # 编排里不允许直接执行写工具：写操作只能走 propose
            self.step(_TOOL_STEP_LABELS.get(name, name), "tool", False, "写工具不能在编排中直接执行")
            return {"ok": False, "data": {}, "error": "写操作需走确认流程"}

        self.tool_calls += 1
        label = _TOOL_STEP_LABELS.get(name, name)
        result = await registry.execute(name, args, owner_id=self.owner_id)
        self.step(label, "tool", result.ok, (result.error or str(result.data))[:120])
        return {"ok": result.ok, "data": result.data or {}, "error": result.error}

    # ---------- 检索文档 ----------
    async def retrieve(self, query: str, k: int = 3) -> list[Any]:
        from ..rag.service import get_service

        service = get_service()
        await service.warmup()
        result = await service.retriever.search(query, top_k=k)
        self.chunks = list(result.chunks)
        brief = "；".join(f"{c.doc_title}>{c.heading_path}" for c in self.chunks[:2])
        self.step("检索知识库", "doc", bool(self.chunks), brief or f"未命中（conf={result.confidence:.2f}）")
        return self.chunks

    # ---------- 生成待确认操作 ----------
    async def propose(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        from .actions import ActionError, get_actions

        service = await get_actions()
        try:
            action = await service.propose(self.owner_id, tool, args)
        except ActionError as exc:
            self.step("生成待确认申请", "propose", False, exc.message)
            return {"ok": False, "error": exc.message}
        self.step("生成待确认申请", "propose", True, action["summary"])
        return {"ok": True, "action": action}


class BudgetExceeded(Exception):
    """编排超出步数/调用上限。"""


# ============================ 剧本 ============================


@dataclass
class Recipe:
    id: str
    title: str
    match: Callable[[str], bool]
    run: Callable[[RecipeContext], Awaitable[dict[str, Any] | None]]


def _has(q: str, *words: str) -> bool:
    return any(w in q for w in words)


# 「问规则」的词：出现这些就说明用户只是想知道怎么规定的，不该替他提交任何东西
_RULE_WORDS = ("条件", "流程", "怎么", "如何", "规定", "制度", "需要什么", "多久", "哪些材料")

# 「明确要我办」的词：没有这些，即便问题里出现了「开通/申请」，也只回答不代办
_ACTION_WORDS = ("帮我", "替我", "提一下", "申请一下", "提交一下", "办一下", "开一下", "直接提")


async def mentor_contact(ctx: RecipeContext) -> dict[str, Any] | None:
    """「我导师是谁，分机多少」：先读档案拿名字，再拿名字去查组织目录。"""
    profile = await ctx.profile()
    mentor = profile.get("mentor", "").strip()
    if not mentor:
        return {
            "answer": "你的档案里没有记录导师，建议直接问 HR 或你的直属主管确认。",
            "source": "orchestrated",
        }

    lookup = await ctx.call_tool("lookup_colleague", {"name": mentor})
    person = {}
    if lookup["ok"]:
        matches = (lookup["data"] or {}).get("matches") or []
        person = matches[0] if matches else {}
    if not person:
        return {
            "answer": f"你的导师是 {mentor}，但组织目录里没查到联系方式，可以先用企业微信搜名字。",
            "source": "orchestrated",
        }

    name = person.get("name") or mentor
    line = f"你的导师是 {name}"
    if person.get("department"):
        line += f"（{person['department']}）"
    contact = []
    ext = person.get("ext") or person.get("extension")
    if ext:
        contact.append(f"分机 {ext}")
    if person.get("seat"):
        contact.append(f"工位 {person['seat']}")
    if contact:
        line += "，" + "，".join(contact)
    return {
        "answer": line + "。有事可以直接找他，不用先发邮件。",
        "source": "orchestrated",
        "people": person,
    }


async def first_week_env(ctx: RecipeContext) -> dict[str, Any] | None:
    """「第一周要开通哪些系统，帮我提一下」：文档取清单 → 生成待确认申请。"""
    chunks = await ctx.retrieve("入职第一周 开发环境开通 哪些系统 工单 勾选", k=3)
    if not chunks:
        return None

    systems = _extract_systems(chunks)
    if not systems:
        # 文档里没有可核对的系统清单时，不生成任何申请（宁可让用户自己看文档）
        ctx.step("从文档提取系统清单", "doc", False, "文档里没有可识别的系统名")
        return None
    ctx.step("从文档提取系统清单", "doc", True, "、".join(systems))

    head = f"入职第 1 天要在 IT 自助平台提交「开发环境开通」工单，勾选：{'、'.join(systems)}[[1]]。"

    # 用户没让我办，就只回答清单。这一条很关键：
    # 「第一周要开通哪些系统」是提问，「帮我按这个提一下」才是委托。
    if not _has(ctx.question, *_ACTION_WORDS):
        return {"answer": head + "\n\n需要我直接帮你提这个申请的话，说一句「帮我提一下」就行。",
                "chunks": ctx.chunks[:1], "source": "orchestrated"}

    if not settings.enable_write_tools:
        return {"answer": head + "\n\n（写操作当前已关闭，请到 IT 自助平台手动提交。）",
                "chunks": ctx.chunks[:1], "source": "orchestrated"}

    proposal = await ctx.propose(
        "submit_permission_request",
        {
            "system": "、".join(systems),
            "reason": "按《新人第一周指南》完成开发环境开通（入职第 1 天）",
        },
    )
    if not proposal["ok"]:
        return {"answer": f"{head}\n\n自动生成申请失败了（{proposal['error']}），你可以直接到 IT 自助平台手动提交。"}

    action = proposal["action"]
    return {
        "answer": (
            f"{head}\n\n"
            f"已按这个清单生成一条待确认申请（**还没有提交**）：{action['summary']}\n"
            "确认后我才会提交；也可以直接说「取消」。"
        ),
        "source": "orchestrated",
        "requires_confirmation": True,
        "pending_action": {
            "action_id": action["id"],
            "tool": "submit_permission_request",
            "summary": action["summary"],
            "expires_at": action["expires_at"],
        },
        "chunks": ctx.chunks[:1],
    }


async def ticket_rule_check(ctx: RecipeContext) -> dict[str, Any] | None:
    """「敏感权限多久到期？我申请的那个还有效吗」：规则 + 个人单据两路合并。"""
    chunks = await ctx.retrieve("敏感权限 默认有效期 到期 自动回收", k=3)
    # 不能「看到有效期就取第一句」：语料里「申请路径：…填写用途与有效期…」也含这三个字，
    # 但它讲的是流程，不是期限。按关键词打分，取最像"规定"的那句。
    best, best_score = "", 0
    for chunk in chunks:
        for sentence in chunk.text.replace("\n", "。").split("。"):
            if not sentence.strip():
                continue
            score = 0
            if "默认有效期" in sentence or "有效期" in sentence:
                score += 1
            if "到期" in sentence:
                score += 1
            if "自动化" in sentence or "回收" in sentence:
                score += 1
            if "路径" in sentence or "填写" in sentence or "审批" in sentence:
                score -= 3
            if score > best_score:
                best, best_score = sentence.strip(" 。"), score
    rule = best if best_score > 0 else ""
    if not rule:
        return None

    tickets = await ctx.call_tool("get_my_tickets", {})
    items = (tickets["data"] or {}).get("tickets") or []
    if not items:
        return {
            "answer": f"知识库里的规定：{rule}[[1]]。\n你目前还没有提交过权限申请。",
            "chunks": chunks[:1],
            "source": "orchestrated",
        }

    latest = items[0]
    status_text = {"pending": "待审批", "approved": "已通过", "rejected": "已驳回"}.get(
        str(latest.get("status")), str(latest.get("status"))
    )
    return {
        "answer": (
            f"知识库里的规定：{rule}[[1]]。\n"
            f"你最近一张工单是 {latest.get('ticket_no')}（{latest.get('system') or '权限申请'}），"
            f"当前状态：{status_text}。"
        ),
        "chunks": chunks[:1],
        "source": "orchestrated",
    }


# ============================ 剧本 4：职级 → 差旅标准 ============================

_TIER1_CITIES = ("北京", "上海", "广州", "深圳")

# 非一线城市不用穷举：只要认出「用户说了个城市、但它不是一线」，就能落到其他城市那一档。
# 认不出来也不要紧，按两档一起答，不会答错。
_OTHER_CITIES = (
    "杭州", "南京", "成都", "武汉", "西安", "苏州", "天津", "重庆", "长沙", "合肥",
    "郑州", "青岛", "厦门", "宁波", "无锡", "济南", "福州", "大连", "沈阳", "昆明",
    "南昌", "贵阳", "石家庄", "太原", "长春", "哈尔滨", "兰州", "南宁", "海口", "珠海",
)


def _travel_city(question: str) -> tuple[str, bool] | None:
    """返回 (城市, 是否一线)；问题里没提城市返回 None。"""
    for city in _TIER1_CITIES:
        if city in question:
            return city, True
    for city in _OTHER_CITIES:
        if city in question:
            return city, False
    return None


def _parse_level(text: str) -> str:
    """从「P5」「p6」「P 7」里取出职级；取不到返回空串（不猜）。"""
    match = re.search(r"[Pp]\s*([1-9])\b", text or "")
    return f"P{match.group(1)}" if match else ""


def _match_level_row(text: str, level: str) -> list[str] | None:
    """在 Markdown 表格里定位职级对应的那一行。

    差旅标准是**表格**，直接整段丢给模型，它经常把 P5 读成 P4 那一行。
    这里按区间精确匹配：P5 命中 `P5—P6`，P8 命中 `P7 及以上`。
    """
    rank = int(level[1:])
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        head = cells[0]
        span = re.search(r"P(\d+)\s*[—\-–~至]\s*P(\d+)", head)
        if span and int(span.group(1)) <= rank <= int(span.group(2)):
            return cells
        above = re.search(r"P(\d+)\s*及以上", head)
        if above and rank >= int(above.group(1)):
            return cells
    return None


def _price(cell: str) -> str:
    found = re.search(r"(\d+)", cell or "")
    return f"{found.group(1)} 元/晚" if found else (cell or "")


def _tier_name(head: str) -> str:
    return head.split("/")[-1].strip() if "/" in head else head.strip()


async def travel_standard(ctx: RecipeContext) -> dict[str, Any] | None:
    """「我 P5，下周去上海出差，住宿能报多少」：档案里的职级 + 文档里的表格行。"""
    profile = await ctx.profile()
    # 职级可能记在档案里，也可能用户这句话里就说了
    level = _parse_level(profile.get("level", "")) or _parse_level(ctx.question)

    chunks = await ctx.retrieve("差旅标准 住宿 一线城市 其他城市 职级 元/晚 特批", k=3)
    if not chunks:
        return None

    if not level:
        # 不知道职级就不硬猜。给对照表 + 一句「告诉我职级我就算准」
        return {
            "answer": (
                "住宿标准按职级分档：\n"
                "- P1—P4：一线城市 450 元/晚，其他城市 350 元/晚[[1]]\n"
                "- P5—P6：一线城市 600 元/晚，其他城市 450 元/晚\n"
                "- P7 及以上：一线城市 800 元/晚，其他城市 600 元/晚\n"
                "告诉我你的职级和目的地，我直接给你具体数字。"
            ),
            "chunks": chunks[:1],
            "source": "orchestrated",
        }

    city = _travel_city(ctx.question)
    for chunk in chunks:
        cells = _match_level_row(getattr(chunk, "text", ""), level)
        if not cells:
            continue
        tier = _tier_name(cells[0])
        if city and city[1]:
            body = (
                f"你是 {level}（{tier}），去{city[0]}（一线城市）出差，"
                f"住宿标准 {_price(cells[1])}[[1]]。"
            )
        elif city:
            body = (
                f"你是 {level}（{tier}），去{city[0]}（其他城市）出差，"
                f"住宿标准 {_price(cells[2])}[[1]]。"
            )
        else:
            body = (
                f"你是 {level}（{tier}）：一线城市 {_price(cells[1])}，"
                f"其他城市 {_price(cells[2])}[[1]]。"
            )
        return {
            "answer": body + "超出标准的住宿要在出差前提交特批，事后申请一律不批。",
            # 只带命中的那一段：引用编号 [[1]] 必须对应它，否则回验会判无据
            "chunks": [chunk],
            "source": "orchestrated",
        }
    return None


_SYSTEM_ALLOWLIST = (
    "GitLab", "Jenkins", "KaiRAG", "数据平台", "VPN", "生产数据库", "模型密钥", "OA",
)


_CHECKOUT_WORDS = ("勾选", "包括", "开通", "清单", "权限如下")


def _extract_systems(chunks: list[Any]) -> list[str]:
    """从文档片段里抽系统名，**只认白名单，且只认「要开通的句子」**。

    两道过滤缺一不可：
    1. 白名单：模型自由抽取会把「IT 自助平台」也当成要开通的系统；
    2. 句子过滤：语料里「申请路径：OA → 权限申请」中的 OA 是**入口**，
       不是要开通的系统。只扫含「勾选/包括/开通/清单」的句子，并把「勾选」之后
       的部分当作清单，能把这层混淆挡掉。

    抽不到就返回空 → 上层不生成任何申请（宁可让用户自己看文档）。
    """
    import re

    found: list[str] = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        for sentence in re.split(r"[。\n]", text):
            if not any(w in sentence for w in _CHECKOUT_WORDS):
                continue
            # 「勾选：A、B、C」→ 只看勾选之后的内容
            segment = sentence.split("勾选", 1)[-1] if "勾选" in sentence else sentence
            for name in _SYSTEM_ALLOWLIST:
                if name not in segment or name in found:
                    continue
                # 「数据平台只读权限」这种带限定的，把限定词一起带上，避免申请成写权限
                idx = segment.find(name)
                suffix = segment[idx + len(name): idx + len(name) + 4]
                found.append(f"{name}只读" if suffix.startswith("只读") else name)
    return found


# ============================ 编排器 ============================

RECIPES: list[Recipe] = [
    Recipe(
        id="mentor_contact",
        title="导师是谁、怎么联系",
        match=lambda q: _has(q, "导师", "buddy", "Buddy", "mentor")
        and _has(q, "谁", "分机", "电话", "工位", "联系", "找"),
        run=mentor_contact,
    ),
    Recipe(
        id="first_week_env",
        title="第一周开通哪些系统并代为提案",
        match=lambda q: _has(q, "开通", "开权限", "申请", "开一下", "都开", "开个", "开权限")
        # 新人不会说「帮我开通系统权限」，他说的是「开发环境都开一下」。
        # 所以第二个条件必须收「开发环境」这类说法，否则真实用户的话术全漏。
        and _has(q, "系统", "权限", "开发环境", "环境开通")
        and _has(q, "哪些", "什么", "清单", "帮我", "提一下", "第一周", "开发环境")
        # 「申请权限需要什么条件」是规则问题，交给文档检索，不在这里代办
        and not _has(q, *_RULE_WORDS),
        run=first_week_env,
    ),
    Recipe(
        id="ticket_rule_check",
        title="权限有效期规则 + 我的单据状态",
        match=lambda q: _has(q, "权限", "工单", "申请")
        and _has(q, "多久", "到期", "有效期", "还有效", "什么时候失效"),
        run=ticket_rule_check,
    ),
    Recipe(
        id="travel_standard",
        title="我的职级能住多少钱的酒店",
        match=lambda q: _has(q, "住宿", "酒店", "房费", "住宿费", "住哪", "住一晚")
        and _has(q, "出差", "住", "酒店", "住宿")
        # 必须是「跟我有关」。问通用标准（「一线城市住宿标准是多少」）应走文档检索：
        # 编排会按个人职级算一个数字，而问的人可能只是想知道制度本身。
        and (_has(q, "我") or bool(re.search(r"[Pp][1-9]", q)))
        # 「同事 P5 去北京能报多少」问的是别人，不能拿「你」的档案去套
        and not _has(q, "同事", "别人", "我朋友")
        # 「我住宿怎么报销」问的是流程，不是标准 → 交回文档链路
        and not _has(q, *_RULE_WORDS),
        run=travel_standard,
    ),
]


class Orchestrator:
    def __init__(self, memory: Any = None) -> None:
        self.memory = memory

    async def run(self, question: str, owner_id: str | None, llm: BaseLLM) -> dict[str, Any] | None:
        if not settings.enable_tools:
            return None
        for recipe in RECIPES:
            try:
                matched = recipe.match(question)
            except Exception:  # noqa: BLE001
                matched = False
            if not matched:
                continue

            ctx = RecipeContext(question=question, owner_id=owner_id, llm=llm, memory=self.memory)
            try:
                result = await recipe.run(ctx)
            except BudgetExceeded as exc:
                log.warning("编排受限（%s）: %s", recipe.id, exc)
                result = {"answer": f"这个任务步骤太多，我先停下（{exc}）。可以拆成两个问题问我。",
                          "source": "orchestrated"}
            except Exception as exc:  # noqa: BLE001
                log.warning("编排 %s 失败: %s", recipe.id, exc)
                result = {"answer": "这个任务执行到一半出错了，我先停下，没有提交任何内容。",
                          "source": "orchestrated"}

            if result is None:
                # 剧本判定「不适用」（比如文档里没有可核对的系统清单）→ 让后面的链路接手
                log.info("编排 %s 判定不适用，回落常规链路", recipe.id)
                continue

            result.setdefault("plan", [s.to_dict() for s in ctx.plan])
            result["recipe"] = recipe.id
            result["elapsed_ms"] = int((time.perf_counter() - ctx.started) * 1000)
            return result
        return None


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
