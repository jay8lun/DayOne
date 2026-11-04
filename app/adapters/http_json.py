"""通用 HTTP JSON 适配器：**零代码**对接任意「能返回 JSON 的知识库接口**。

绝大多数企业内部知识库（企微文档、Confluence、飞书、语雀、自建 CMS）
都能抽象成两个接口：

    GET  {base_url}{list_path}?{cursor_param}=&{limit_param}=   ->  {"list": [...] }
    GET  {base_url}/docs/{id}                                   ->  {"content": "..."}

差异只在「字段叫什么、怎么分页、怎么鉴权」。因此把这些差异全部下沉为
``KB_ADAPTER_CONFIG`` 里的 JSON 配置，主流程一行都不用改：

```json
{
  "base_url": "https://kb.corp.example.com/api/v1",
  "list_path": "/documents",
  "detail_path": "/documents/{id}",
  "auth": {"type": "bearer", "token": "xxxx"},
  "items_path": "data.items",
  "next_cursor_path": "data.next_cursor",
  "cursor_param": "cursor",
  "limit_param": "page_size",
  "fields": {
    "id": "doc_id",
    "title": "name",
    "content": "content.markdown",
    "url": "url",
    "updated_at": "updated_at",
    "author": "creator.name",
    "space": "space.name"
  },
  "content_format": "auto"
}
```

如果贵司接口的返回结构完全不一样（比如要 POST + GraphQL），
再继承 :class:`KnowledgeSourceAdapter` 写代码即可，见 docs/适配器开发指南.md。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from ..core.errors import AdapterError
from ..core.logging import get_logger
from .base import DocumentPage, DocumentRef, KnowledgeSourceAdapter, SourceDocument
from .local_dir import _html_to_text

log = get_logger(__name__)

DEFAULT_FIELDS = {
    "id": "id",
    "title": "title",
    "content": "content",
    "url": "url",
    "updated_at": "updated_at",
    "author": "author",
    "space": "space",
}


def _dig(obj: Any, path: str | None) -> Any:
    """按 ``a.b.c`` 取嵌套字段，取不到返回 None（不抛异常，避免个别脏数据中断同步）。"""
    if not path:
        return None
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else None
        else:
            return None
    return cur


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        # 兼容 秒 / 毫秒 时间戳
        ts = float(value)
        if ts > 1e11:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M"):
            try:
                return datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


class HttpJsonAdapter(KnowledgeSourceAdapter):
    name = "http_json"
    description = "通用 REST/JSON 适配器：通过字段映射 + 分页配置对接任意知识库 HTTP 接口"
    supports_incremental = True
    supports_pagination = True

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise AdapterError("http_json 适配器缺少 base_url，请配置 KB_ADAPTER_CONFIG")
        self.list_path = cfg.get("list_path", "/documents")
        self.detail_path = cfg.get("detail_path")
        self.items_path = cfg.get("items_path", "data") or "data"
        self.next_cursor_path = cfg.get("next_cursor_path")
        self.cursor_param = cfg.get("cursor_param", "cursor")
        self.limit_param = cfg.get("limit_param", "limit")
        self.updated_param = cfg.get("updated_param")
        self.fields = {**DEFAULT_FIELDS, **(cfg.get("fields") or {})}
        self.content_format = cfg.get("content_format", "auto")
        self.timeout = float(cfg.get("timeout", 20))
        self._client: httpx.AsyncClient | None = None
        self._list_cache: dict[str, dict] = {}

    # ---------- HTTP ----------
    def _headers(self) -> dict[str, str]:
        auth = self.config.get("auth") or {}
        kind = (auth.get("type") or "none").lower()
        headers = {"Accept": "application/json"}
        if kind == "bearer":
            headers["Authorization"] = f"Bearer {auth.get('token', '')}"
        elif kind == "header" and auth.get("name"):
            headers[auth["name"]] = str(auth.get("value", ""))
        elif kind == "basic" and auth.get("username"):
            import base64
            raw = f"{auth.get('username')}:{auth.get('password', '')}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        headers.update({str(k): str(v) for k, v in (self.config.get("headers") or {}).items()})
        return headers

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def health_check(self) -> bool:
        try:
            client = await self._http()
            resp = await client.get(self.list_path, params={self.limit_param: 1})
            return resp.status_code < 400
        except Exception as exc:  # noqa: BLE001
            log.warning("http_json health_check 失败: %s", exc)
            return False

    # ---------- 目录 ----------
    async def list_documents(self, cursor: str | None = None, limit: int = 50) -> DocumentPage:
        client = await self._http()
        params: dict[str, Any] = {self.limit_param: limit}
        if cursor:
            params[self.cursor_param] = cursor
        try:
            resp = await client.get(self.list_path, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"拉取文档目录失败：{exc}") from exc

        items_raw = _dig(payload, self.items_path) or []
        if isinstance(items_raw, dict):  # 兼容 {"data": {"items": [...]}} 写法
            items_raw = _dig(items_raw, "items") or []
        items: list[DocumentRef] = []
        for raw in items_raw:
            if not isinstance(raw, dict):
                continue
            ext_id = _dig(raw, self.fields["id"])
            if ext_id is None:
                continue
            ext_id = str(ext_id)
            self._list_cache[ext_id] = raw
            items.append(
                DocumentRef(
                    external_id=ext_id,
                    title=str(_dig(raw, self.fields["title"]) or ext_id),
                    updated_at=_parse_dt(_dig(raw, self.fields["updated_at"])),
                    url=_dig(raw, self.fields["url"]),
                    metadata={"raw_keys": sorted(raw.keys())},
                )
            )
        next_cursor = _dig(payload, self.next_cursor_path) if self.next_cursor_path else None
        return DocumentPage(items=items, next_cursor=str(next_cursor) if next_cursor else None)

    # ---------- 正文 ----------
    async def fetch_document(self, external_id: str) -> SourceDocument:
        raw: dict[str, Any] | None = None
        if self.detail_path:
            path = self.detail_path.format(id=external_id, doc_id=external_id)
            client = await self._http()
            try:
                resp = await client.get(path)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise AdapterError(f"拉取文档 {external_id} 失败：{exc}") from exc
            raw = _dig(body, self.config.get("detail_wrap", "data")) or body
            if not isinstance(raw, dict):
                raw = body
        else:
            raw = self._list_cache.get(external_id)
            if raw is None:
                page = await self.list_documents(limit=100)
                raw = self._list_cache.get(external_id)
                if raw is None:
                    raise AdapterError(f"文档 {external_id} 不在目录缓存中，请配置 detail_path")

        content = _dig(raw, self.fields["content"])
        if content is None:
            # 有些接口正文放在 blocks / paragraphs 数组里
            blocks = _dig(raw, "blocks") or _dig(raw, "content.blocks") or []
            if isinstance(blocks, list):
                content = "\n\n".join(
                    str(b.get("text") or b.get("content") or "") if isinstance(b, dict) else str(b)
                    for b in blocks
                )
        content = "" if content is None else str(content)
        if self.content_format == "html" or (
            self.content_format == "auto" and re.search(r"</(p|div|h[1-6])>", content)
        ):
            content = _html_to_text(content)

        title = str(_dig(raw, self.fields["title"]) or external_id)
        return SourceDocument(
            external_id=external_id,
            title=title,
            content=content,
            url=_dig(raw, self.fields["url"]),
            updated_at=_parse_dt(_dig(raw, self.fields["updated_at"])),
            author=_dig(raw, self.fields["author"]),
            space=_dig(raw, self.fields["space"]),
            metadata={"source": self.name},
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
