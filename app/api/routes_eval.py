"""评估接口：跑数据集、看最近一次报告。"""

from __future__ import annotations

import json

import orjson
from fastapi import APIRouter, Depends

from ..config import settings
from ..eval.dataset import load_dataset
from ..eval.runner import Evaluator, render_markdown
from .deps import require_api_key
from .schemas import EvalRequest

router = APIRouter(prefix="/api/eval", tags=["evaluation"], dependencies=[Depends(require_api_key)])

REPORT_PATH = "data/eval/last_report.json"


@router.get("/dataset", summary="查看评估集内容")
async def dataset() -> dict:
    cases = load_dataset(settings.eval_dataset)
    return {
        "path": settings.eval_dataset,
        "count": len(cases),
        "cases": [
            {
                "id": c.id,
                "type": c.type,
                "question": c.question,
                "expected_docs": c.expected_docs,
                "must_contain": c.must_contain,
            }
            for c in cases
        ],
    }


@router.post("/run", summary="运行评估（同步执行，数据量大时请用脚本）")
async def run_eval(req: EvalRequest) -> dict:
    evaluator = Evaluator()
    report = await evaluator.run(
        dataset_path=req.dataset,
        top_k=req.top_k,
        limit=req.limit,
        with_generation=req.with_generation,
    )
    path = settings.resolve(REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    (path.parent / "last_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


@router.get("/last", summary="最近一次评估报告")
async def last_report() -> dict:
    path = settings.resolve(REPORT_PATH)
    if not path.exists():
        return {"ok": False, "message": "还没有跑过评估"}
    return {"ok": True, "report": json.loads(path.read_text(encoding="utf-8"))}
