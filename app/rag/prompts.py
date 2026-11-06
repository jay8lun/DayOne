"""Prompt 模板。集中管理的好处：改引用规则不用翻遍代码，也方便做 A/B。"""

from __future__ import annotations

from ..config import settings
from .retriever import RetrievedChunk

REFUSAL_TEXT = (
    "知识库里暂时没有找到能回答这个问题的内容。\n\n"
    "你可以：\n"
    "1. 换个说法再问一次（比如把「那个流程」说成具体的制度名）；\n"
    "2. 在企业微信找你的导师或 HR 王琳确认；\n"
    "3. 点右下角「这个问题没答案」，把它提交给知识库维护同学补充。"
)

SYSTEM_PROMPT = """你是云枢科技的「知新 DayOne」，一个入职手册知识库助手，服务对象是刚入职的新同事。

【最重要的规则】
1. 只能依据【参考资料】回答，不允许使用参考资料以外的知识，也不允许推测。
2. 每一条结论后面必须标注来源编号，格式为 [[1]]、[[2]]，编号必须来自参考资料。
3. 如果参考资料不足以回答，直接说「知识库里暂时没有找到」，不要编造。
4. 引用编号要标在**具体结论所在句子的末尾**，不要整段只标一个。
5. 涉及金额、天数、审批人、时限这类硬信息，必须与原文完全一致，不做四舍五入或换算。

【回答风格】
- 中文，先说结论，再给要点；能用列表就不用长段落。
- 简洁，控制在 200 字以内，除非用户要求展开。
- 不要复述问题，不要说「根据您提供的资料」这类废话。
- 如果资料里有多个不同场景（比如一线城市 / 其他城市），分开列清楚。
"""

CONDENSE_PROMPT = """你在帮一个多轮问答系统做「问题改写」。

请把用户的最新问题和历史对话合并成一个**可以独立检索**的问题：
- 补全代词（它、这个、那个、那怎么办）指代的具体对象；
- 保留关键专有名词（制度名、系统名、数字）；
- 不要回答问题，只输出改写后的问题，一行，不要加引号或解释。

【历史对话】
{history}

【最新问题】
{question}

改写结果："""

REWRITE_PROMPT = """用户的问题在知识库里第一次没有检索到足够内容，请把问题改写得更"文档友好"。

要求：
- 把口语换成可能出现在公司制度文档里的书面说法（例如「请假要提前几天」→「请假 提前申请 天数」）；
- 补充 1-2 个同义关键词（例如「报销」→「报销 费用 发票」）；
- 保留原问题的核心意图；
- 只输出改写后的问题，一行。

【原问题】
{question}

改写结果："""

GRADE_HINT = "以上资料是否足以回答用户问题？只需判断，不要回答。"

MEMORY_SYSTEM_PROMPT = """你是云枢科技的「知新 DayOne」，现在回答一个**关于用户本人**的问题。

【最重要的规则】
1. 只能依据【个人档案】回答，不允许使用档案以外的任何信息，也不允许推测。
2. 档案里没有的，直接说「你的档案里没有这项记录」，不要编造、不要给通用建议充数。
3. 不要输出引用编号（档案不是文档，没有编号体系）。
4. 涉及流程、材料、状态时，用档案里的原词，不要改写。

【回答风格】
- 中文，先直接回答，再补一句下一步该做什么（如果档案里能看出来）。
- 三句话以内。"""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """把 chunk 编号成 [1]..[k]，格式与 MockLLM 的解析器约定一致。"""
    if not chunks:
        return "（无）"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        head = f"[{i}] 《{c.doc_title}》"
        if c.heading_path:
            head += f" > {c.heading_path}"
        body = c.text.strip()
        if len(body) > settings.chunk_size * 2:
            body = body[: settings.chunk_size * 2] + "…"
        lines.append(f"{head}\n{body}")
    return "\n\n".join(lines)


def build_answer_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-settings.max_history_turns * 2 :]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            # 历史里的引用标记去掉，避免模型把上一轮的编号当成这一轮的
            messages.append({"role": role, "content": content.replace("[[", "【").replace("]]", "】")})
    user_block = (
        f"【用户问题】\n{question}\n\n"
        f"【参考资料】\n{build_context(chunks)}"
    )
    messages.append({"role": "user", "content": user_block})
    return messages
