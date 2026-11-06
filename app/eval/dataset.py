"""评估数据集加载。

一行一个 JSON（jsonl），字段：

```json
{"id": "q01", "question": "报销单多久内要提交？", "expected_docs": ["报销、差旅与采购流程"],
 "must_contain": ["60 天"], "type": "single"}
```

- ``expected_docs``：命中其中任意一篇即算「召回正确」（用标题或文件名模糊匹配，
  避免文档改名/加序号后数据集全废）；
- ``must_contain``：答案必须包含的关键事实（数字、专有名词），用于生成质量打分；
- ``type``：single / multi / followup / refusal，用于分组看指标；
- ``history``：followup 类型专用，先塞进会话再提问。

下面几个字段只给**编排类端到端**用例（``app/eval/datasets/orchestration_e2e.jsonl``）用：

- ``expect_recipe``：应该命中的剧本 id（``mentor_contact`` / ``first_week_env`` /
  ``ticket_rule_check``）；
- ``expect_no_recipe``：**不应该**被编排接走（规则类问题必须落到文档检索；
  个人进度类应该落到记忆/单工具路由）。和 ``expect_recipe`` 互斥；
- ``expect_steps``：plan 里的步骤类型序列，如 ``["memory", "tool"]``。
  这一条盯的是「顺序」——编排最容易错的是把第二步放在第一步之前；
- ``expect_confirmation``：是否应该产出待确认操作（``True`` = 只提案不执行）；
- ``expect_contains`` / ``expect_not_contains``：答案必须 / 不得出现的内容；
- ``setup``：跑用例前造的数据。``{"facts": {"mentor": "王琳"}, "ticket": {...}}``，
  没有它，同一条问题在空档案下与有档案下测的是两回事。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    expected_docs: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    type: str = "single"
    history: list[dict[str, str]] = field(default_factory=list)
    note: str = ""

    # ---- 编排端到端专用 ----
    expect_recipe: str | None = None
    expect_no_recipe: bool = False
    expect_steps: list[str] = field(default_factory=list)
    expect_confirmation: bool | None = None
    expect_contains: list[str] = field(default_factory=list)
    expect_not_contains: list[str] = field(default_factory=list)
    setup: dict = field(default_factory=dict)

    @property
    def has_orchestration_expectation(self) -> bool:
        return bool(
            self.expect_recipe
            or self.expect_no_recipe
            or self.expect_steps
            or self.expect_confirmation is not None
            or self.expect_contains
            or self.expect_not_contains
        )


def load_dataset(path: str | Path) -> list[EvalCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"评估集不存在：{p}")
    cases: list[EvalCase] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        recipe = raw.get("expect_recipe")
        no_recipe = bool(raw.get("expect_no_recipe", False))
        if recipe and no_recipe:
            raise ValueError(f"{p.name} 第 {lineno} 行：expect_recipe 与 expect_no_recipe 互斥")
        cases.append(
            EvalCase(
                id=str(raw.get("id") or f"q{lineno:02d}"),
                question=raw["question"],
                expected_docs=raw.get("expected_docs") or [],
                must_contain=raw.get("must_contain") or [],
                type=raw.get("type", "single"),
                history=raw.get("history") or [],
                note=raw.get("note", ""),
                expect_recipe=recipe,
                expect_no_recipe=no_recipe,
                expect_steps=raw.get("expect_steps") or [],
                expect_confirmation=raw.get("expect_confirmation"),
                expect_contains=raw.get("expect_contains") or [],
                expect_not_contains=raw.get("expect_not_contains") or [],
                setup=raw.get("setup") or {},
            )
        )
    return cases
