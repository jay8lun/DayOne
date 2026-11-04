"""企业微信文档适配器（占位实现）。

现状：公司还没开企微知识库接口权限，所以这里**只保留接口形状**，
不写死任何私有字段，等权限下来后按下面 TODO 填 30 行即可。

它存在的意义：
1. 证明「插件化」是真的——新增平台只是新增一个文件，主流程不动；
2. 前端 / 健康检查能显示「已注册但未配置」，不会假装能用；
3. 把「需要哪些权限、哪个接口」写清楚，方便找 IT 提需求。

官方接口参考（申请权限后替换）：
- 获取文档列表:  GET /cgi-bin/wedoc/get_doc_list?access_token=&doc_type=&limit=&cursor=
- 获取文档内容:  GET /cgi-bin/wedoc/get_doc_content?access_token=&docid=
- access_token:  GET /cgi-bin/gettoken?corpid=&corpsecret=
"""

from __future__ import annotations

from .base import DocumentPage, KnowledgeSourceAdapter, SourceDocument
from ..core.errors import AdapterNotConfigured


class WeComDocAdapter(KnowledgeSourceAdapter):
    name = "wecom"
    description = "企业微信文档适配器（待接口权限，当前为占位实现）"
    supports_incremental = False
    supports_pagination = True

    REQUIRED_CONFIG = ("corp_id", "corp_secret")

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.corp_id = (config or {}).get("corp_id", "")
        self.corp_secret = (config or {}).get("corp_secret", "")
        self._token: str | None = None

    @property
    def configured(self) -> bool:
        return all(bool((self.config or {}).get(k)) for k in self.REQUIRED_CONFIG)

    async def health_check(self) -> bool:
        # 未配置就直接 False，不在启动时抛异常阻塞整个服务
        return self.configured

    async def list_documents(self, cursor: str | None = None, limit: int = 100) -> DocumentPage:
        raise AdapterNotConfigured(
            "企业微信文档接口未开通。请向 IT 申请「文档-读取」权限，"
            "拿到 corp_id / corp_secret 后填入 KB_ADAPTER_CONFIG，"
            "并在 app/adapters/wecom.py 中实现 list_documents / fetch_document。"
        )

    async def fetch_document(self, external_id: str) -> SourceDocument:
        raise AdapterNotConfigured("企业微信文档接口未开通，无法拉取正文。")

    # TODO(接入时删除以下注释并实现):
    # 1. async def _get_token(self): GET /cgi-bin/gettoken?corpid=&corpsecret=  （缓存 7200s）
    # 2. list_documents: GET /cgi-bin/wedoc/get_doc_list  -> 映射 docid/title/update_time
    # 3. fetch_document: GET /cgi-bin/wedoc/get_doc_content  -> 正文（markdown/html）
    # 4. 把返回的 docid 作为 external_id，update_time 作为 updated_at，
    #    其余字段塞进 metadata，切块/检索/引用逻辑完全不用改。
