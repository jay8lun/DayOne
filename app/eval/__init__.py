"""评估层：数据集 / 指标 / 执行器。"""

from .dataset import EvalCase, load_dataset
from .metrics import keyword_coverage, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from .runner import Evaluator, render_markdown

__all__ = [
    "EvalCase",
    "load_dataset",
    "keyword_coverage",
    "mean",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "Evaluator",
    "render_markdown",
]
