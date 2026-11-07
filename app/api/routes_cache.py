"""答案缓存的运维接口：看统计、清空。

`POST /api/cache/clear` 存在的直接原因是一个坑：只清 Redis 里的键，
服务进程内存里那层兜底缓存还在，于是「清完了还在命中」。
所以清空必须走服务自己，把两层一起清掉。

顺带解决了另一个问题：知识库更新后想立刻生效，不用等 10 分钟 TTL 自然过期
（虽然 `kb_version` 变化后旧键已经不会被读到，清一下能省内存）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..rag.cache import get_answer_cache
from .deps import require_api_key

router = APIRouter(prefix="/api/cache", tags=["cache"], dependencies=[Depends(require_api_key)])


@router.get("", summary="缓存统计（命中率 / 条数 / TTL）")
async def cache_stats() -> dict[str, Any]:
    return get_answer_cache().stats()


@router.post("/clear", summary="清空答案缓存（Redis + 进程内两层）")
async def clear_cache() -> dict[str, Any]:
    cache = get_answer_cache()
    removed = cache.clear()
    cache.reset_stats()
    return {"cleared": removed, "backend": cache.backend, "stats": cache.stats()}
