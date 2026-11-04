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
        cases.append(
            EvalCase(
                id=str(raw.get("id") or f"q{lineno:02d}"),
                question=raw["question"],
                expected_docs=raw.get("expected_docs") or [],
                must_contain=raw.get("must_contain") or [],
                type=raw.get("type", "single"),
                history=raw.get("history") or [],
                note=raw.get("note", ""),
            )
        )
    return cases
