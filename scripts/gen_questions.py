"""探针问句反生成：从知识库原文反向生成「新人真的会问的问题」。

## 为什么不能手写评估集就完事

手写的问题有两个毛病：
1. **写的人知道答案**，会不自觉地用文档里的词（「chunk_size」而不是「分块大小」），
   于是测出来的是「关键词匹配得好不好」，不是「新人问得对不对」；
2. 覆盖靠自觉，冷门章节永远测不到。

所以这里反过来做：从每一段原文出发，让模型生成「只用这段能答」的问题，
再用**原文反查**把模型编的事实筛掉。产出的是 `golden_onboarding.jsonl` 同格式的
数据，可以直接拿去跑 `scripts/evaluate.py --dataset ...`。

## 两条防幻觉的闸

- 生成的 `must_contain` 必须在原文里**逐字出现**，否则整条丢弃（这是模型最爱编的地方）；
- 问题里不允许出现原文没有的专有名词——不额外查，靠上一条兜住大部分。

```bash
python scripts/gen_questions.py --per-doc 1 --limit 12        # 生成到默认路径
python scripts/gen_questions.py --dry-run                     # 只打印，不写文件
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.ingest.store import get_store  # noqa: E402
from app.rag.llm import get_llm  # noqa: E402

PROMPT = """你在为一份企业入职手册做问答测试集。

下面是一段手册原文（文档：{title}；小节：{heading}）：
---
{text}
---

请生成 {n} 个新人入职第一周**真的会问**的问题，并给出答案必须包含的关键事实。

硬要求：
1. 问题要口语化，像一个新人直接在群里打字问的，不要照抄标题、不要用「本文提到」；
2. 问题必须**只用上面这段原文**就能回答，不能依赖别的章节；
3. 关键事实只能从原文里挑，最多 2 个，优先数字、系统名、专有名词；
4. 不要问「这段讲了什么」这类元问题。

只输出 JSON 数组，不要任何解释：
[{{"question": "...", "must_contain": ["..."]}}]
"""


def _normalize(text: str) -> str:
    """比对用：去掉空白与全角空格，避免「60 天」和「60天」被判成不一致。"""
    return re.sub(r"[\s\u3000]+", "", text or "")


def _parse_json_array(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("question")]


def _sample(chunks: list[dict], per_doc: int, limit: int, seed: int) -> list[dict]:
    """按文档分组抽样：每篇取若干段，保证 12 篇文档都有机会被测到。"""
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        if len((chunk.get("text") or "")) < 80:      # 太短的段落问不出像样的问题
            continue
        title = str((chunk.get("metadata") or {}).get("doc_title") or "未命名")
        by_doc.setdefault(title, []).append(chunk)

    rng = random.Random(seed)
    picked: list[dict] = []
    titles = sorted(by_doc)
    rng.shuffle(titles)
    for t in titles:
        pool = by_doc[t]
        rng.shuffle(pool)
        picked.extend(pool[:per_doc])
        if len(picked) >= limit:
            break
    return picked[:limit]


def _usable_fact(fact: str) -> bool:
    """关键事实必须是「能被逐字复现的短事实」。

    模型很爱把整句规定当事实（「超出标准的住宿需提前申请特批，出差前提交」），
    这种句子连人复述都会换措辞，拿它判分只会让指标虚低、失去意义。

    所以只认两类：带数字的（「30 天」「100」）、带英文标识的（「SKILL.md」「ruff」）。
    纯中文短语一律不要——「出差前提交」换成「提前申请」意思一样，但按字面就判错，
    那不是模型答错，是评分口径错。
    """
    compact = _normalize(fact)
    return bool(compact) and len(compact) <= 12 and bool(re.search(r"\d|[A-Za-z]", compact))


async def _generate_one(chunk: dict, n: int, llm) -> list[dict]:
    meta = chunk.get("metadata") or {}
    text = (chunk.get("text") or "").strip()
    prompt = PROMPT.format(
        title=meta.get("doc_title", ""), heading=meta.get("heading_path", ""), text=text, n=n
    )
    raw = await llm.complete(
        [{"role": "user", "content": prompt}], temperature=0.4, max_tokens=600
    )
    body = _normalize(text)
    out = []
    for item in _parse_json_array(raw):
        question = str(item.get("question") or "").strip()
        facts = [str(f).strip() for f in (item.get("must_contain") or []) if str(f).strip()]
        facts = [f for f in facts if _usable_fact(f)][:2]
        if not question or not facts:
            continue
        # 关键闸：事实必须在原文里逐字出现，否则是模型编的，整条丢掉
        if not all(_normalize(f) in body for f in facts):
            continue
        out.append(
            {
                "question": question,
                "must_contain": facts,
                "expected_docs": [meta.get("doc_title", "")],
                "note": f"反生成自 {meta.get('heading_path', '')}",
            }
        )
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="从知识库反生成探针问句")
    parser.add_argument("--per-doc", type=int, default=1, help="每篇文档取几段")
    parser.add_argument("--limit", type=int, default=12, help="最多生成几条")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="app/eval/datasets/generated_probes.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging("WARNING")
    store = get_store()
    chunks = await store.all_chunks()
    if not chunks:
        print("知识库是空的：先跑 python scripts/ingest.py")
        return 1

    picked = _sample(chunks, args.per_doc, args.limit, args.seed)
    print(f"从 {len(chunks)} 段里抽了 {len(picked)} 段，开始反生成…")

    llm = get_llm()
    rows: list[dict] = []
    seen: set[str] = set()
    for i, chunk in enumerate(picked, start=1):
        try:
            items = await _generate_one(chunk, 1, llm)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(picked)}] 生成失败，跳过：{exc}")
            continue
        for item in items:
            key = re.sub(r"[^\w\u4e00-\u9fa5]", "", item["question"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": f"g{len(rows) + 1:02d}", "type": "single", **item})
        print(f"  [{i}/{len(picked)}] {items[0]['question'] if items else '（被事实校验淘汰）'}")

    if not rows:
        print("一条都没生成出来：检查模型是否可用。")
        return 1

    if args.dry_run:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    path = settings.resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"\n已写入 {path}（{len(rows)} 条）")
    print(f"评测：python scripts/evaluate.py --dataset {args.out} --with-generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
