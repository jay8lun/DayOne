"""全局配置：全部通过环境变量 / .env 注入，代码里不写死任何密钥。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """所有配置项均可通过 .env 覆盖，见 .env.example。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "知新 DayOne · 入职手册知识库助手"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    # 服务自身鉴权（可选）。为空表示不校验，适合内网演示。
    service_api_key: str = ""

    # ---------- 大模型（OpenAI 兼容协议：DeepSeek / 通义 / 智谱 / 硅基流动 / vLLM 均可） ----------
    llm_provider: str = "auto"  # auto | openai | mock
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    llm_timeout: float = 60.0

    # ---------- 向量化 ----------
    embedding_provider: str = "auto"  # auto | openai | hashing
    embedding_api_key: str = ""  # 留空则复用 llm_api_key
    embedding_base_url: str = ""  # 留空则复用 llm_base_url
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    # hashing 兜底向量维度（离线可用，质量低于真实 embedding）
    hashing_dim: int = 512

    # ---------- 知识库源适配器 ----------
    kb_source: str = "local_dir"  # local_dir | http_json | wecom
    # 适配器参数统一用 JSON 传，换平台只改这里 / 换适配器，不改主流程
    kb_adapter_config: str = ""
    kb_local_path: str = "data/corpus"
    # 允许多个适配器同时挂载，形如 "local_dir,http_json"
    kb_extra_sources: str = ""
    # 启动时若向量库为空，自动从知识库源入库一次（演示/首次部署友好）
    auto_ingest_on_startup: bool = True
    # 外部适配器插件（逗号分隔的模块路径），换公司时把插件模块挂上去即可
    adapter_plugins: str = ""

    # ---------- 切块 ----------
    chunk_size: int = 520
    chunk_overlap: int = 80
    min_chunk_size: int = 60

    # ---------- 检索 ----------
    retrieve_top_k: int = 5
    retrieve_candidates: int = 30
    # 是否启用向量通道。注意：hashing 兜底向量会被自动判定为「不可信」而关闭（见 retriever）
    enable_dense: bool = True
    rrf_k: int = 60
    # 融合权重：rrf_weight 给「排名融合」，其余给「词面覆盖度」
    # 0.5~0.65 区间实测差异很小（见 docs/调优记录.md），取 0.6 折中
    rrf_weight: float = 0.6
    # BM25F 参数（可用 scripts/tune.py 在自有数据集上扫）
    bm25_k1: float = 1.3
    bm25_b: float = 0.85
    bm25_title_weight: float = 5.0
    bm25_heading_weight: float = 3.0
    enable_mmr: bool = True
    mmr_lambda: float = 0.72
    enable_llm_rerank: bool = False
    enable_query_rewrite: bool = True
    max_corrective_rounds: int = 1
    # 低于该置信度视为「知识库无相关内容」，直接兜底拒答
    min_confidence: float = 0.35
    # 拒答的第二道闸：必须满足其一，才算「真的在讲这件事」
    #   bm25_top  ≥ 该值  → 有强词法命中（实测正常问题最低 9.2、无关问题最高 7.4）
    #   evidence  ≥ 该值  → 单块词面覆盖度足够
    # 注意：这两个数是**模型/语料相关**的，换 embedding 或换知识库后要重标（见 docs/调优记录.md）
    min_bm25_anchor: float = 8.5
    min_evidence: float = 0.30

    # ---------- 会话 ----------
    max_history_turns: int = 6
    session_ttl_minutes: int = 120

    # ---------- 记忆（长期，跨会话） ----------
    memory_db_path: str = "data/memory.db"
    # 写入闸门：LLM 抽取的候选低于该置信度只进 pending，不覆盖已有事实
    memory_confidence_min: float = 0.7
    # 遗忘曲线：半衰期越长越"念旧"；阈值越低越舍不得忘
    # 取 90 天是权衡结果：半年不提，权重还剩 0.25 会被保留；
    # 一年多不提才降到 0.05 以下被冷归档。30 天太激进（三个月不聊就把用户住址忘了）。
    memory_half_life_days: float = 90.0
    memory_decay_threshold: float = 0.15
    # 冷归档放置多久后立墓碑（墓碑可审计，不物理删除）
    memory_tombstone_days: float = 90.0
    # 常驻字段：不参与衰减。衰减只按「时间 × 频次」，分不清闲聊和身份信息——
    # 不保护的话，一年没人提工号就被清掉了，而办手续每次都要用。
    # 逗号分隔的 predicate；换语料/换公司按自家字段名改。
    memory_pinned_predicates: str = "employee_id,department,mentor,job_level,email"
    # 回答时是否注入「已知用户信息 + 当前流程状态」
    memory_inject_context: bool = True

    @property
    def pinned_predicates(self) -> set[str]:
        return {x.strip() for x in self.memory_pinned_predicates.split(",") if x.strip()}

    # ---------- 工具调用（P3） ----------
    # 关掉就完全跳过工具路由（省一次模型往返），文档/记忆链路不受影响
    enable_tools: bool = True
    # 写操作（提工单/申请）走「提议—确认—执行」。关掉则只能只读查询
    enable_write_tools: bool = True
    # 待确认操作的有效期：过期即作废，逼用户重新发起，避免「昨天点的确认今天才生效」
    tool_confirm_ttl_minutes: int = 30
    # 待确认操作与工单的独立库（与记忆库分开：生命周期和清理策略都不同）
    tool_actions_db_path: str = "data/actions.db"

    # ---------- 上下文预算（P2） ----------
    # 关掉就退回「全塞进去」的老行为，方便 A/B
    enable_context_budget: bool = True
    # 输入侧预算（不含模型输出）。按 8k 窗口估的保守值，换大窗口模型可调高
    context_max_tokens: int = 3000
    # 给输出预留，不允许靠挤占它来塞内容
    context_output_reserve: int = 900
    context_profile_tokens: int = 220
    context_work_tokens: int = 160
    # 最近 N 轮原文不压缩（压了会破坏指代消解）
    context_recent_turns: int = 2
    context_summary_tokens: int = 320

    # ---------- 存储 ----------
    chroma_dir: str = "data/chroma"
    collection_prefix: str = "onboarding_kb"
    ingest_state_file: str = "data/ingest_state.json"

    # ---------- 评估 ----------
    eval_dataset: str = "app/eval/datasets/golden_onboarding.jsonl"
    eval_top_k: int = 5

    # ---------- 派生属性 ----------
    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() in {"", "*"}:
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def kb_config(self) -> dict[str, Any]:
        if self.kb_adapter_config.strip():
            try:
                return json.loads(self.kb_adapter_config)
            except json.JSONDecodeError as exc:  # pragma: no cover - 配置错误
                raise ValueError(f"KB_ADAPTER_CONFIG 不是合法 JSON：{exc}") from exc
        return {"path": self.kb_local_path}

    @property
    def effective_embedding_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_embedding_base(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("CHUNK_OVERLAP 不能为负")
        return v

    def resolve(self, p: str | Path) -> Path:
        """相对路径统一相对项目根目录解析。"""
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
