"""接口依赖：服务鉴权。

默认不开启（``SERVICE_API_KEY`` 为空），方便内网演示；
生产环境在 .env 里设一个强随机串，前端在设置里填同一个值即可。
"""

from __future__ import annotations

from fastapi import Header

from ..config import settings
from ..core.errors import AuthError


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.service_api_key.strip()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise AuthError("缺少或错误的 X-API-Key")
