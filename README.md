# 知新 DayOne · 入职手册知识库助手

> 新人知新，问必有据。
> 一套**通用 RAG 模板**：换知识库平台只换适配器，主流程一行不改。

**名字的由来**

- **DayOne**：入职第一天。那是新人最茫然、也最不敢开口问的一天，产品的目标就一句话——**第一天就有答案**。
- **知新**：取自「温故而知新」，也谐音**知心**。敢问的前提是不怕被笑，所以名字先要把这个态度给出来。

---

## 1. 解决什么问题

新人入职最典型的两件事：

| 痛点 | 表现 | 我们的解法 |
| --- | --- | --- |
| **不敢问** | 怕问得初级、怕打扰人，于是自己瞎猜，猜错了返工 | 匿名、随时、秒回；答案带来源编号，可自行核对原文 |
| **重复问** | 同一个「报销几天内提交」，每个新人问一遍，老员工被反复打断 | 把答案沉淀进知识库，一次入库、无限复用；问过的没答案的问题会自动进「待补充清单」 |

所以这个系统有两个硬要求：**答案必须能追溯来源**，**不知道就必须说不知道**。

### 界面

![问答界面](docs/ui-answer.png)

左边是对话（每条结论后面跟着可点的来源编号），右边是本次检索到的文档片段与分数，
点编号会跳到对应片段并高亮关键句；底部一行显示引用数、有据率、置信度与耗时。
（截图为真实模型 `deepseek-v4-flash` + `text-embedding-v3` 的回答）

---

## 2. 效果（真实模型实测，可复现）

35 条人工标注的入职问答（single / multi / followup / refusal 四类），
模型 `deepseek-v4-flash` + `text-embedding-v3`（1024 维）：

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| Recall@5 | **1.000** | 期望文档是否进入前 5 |
| MRR | **0.984** | 第一个正确结果的平均排名 |
| nDCG@5 | **0.989** | 考虑排序位置 |
| 拒答准确率 | **1.000** | 知识库没有的问题 3/3 都拒答了，没有硬编 |
| 关键事实覆盖 | **0.938** | 回答里是否包含原文的金额/天数等硬信息（关键词比对已压平空格，见调优记录） |
| 引用有据率 | **1.000** | 带引用的句子中，能在来源里找到依据的比例 |
| 平均延迟 | 3707 ms | 含检索 + 生成（流式输出，用户先看到来源） |

对比**没有 API Key 的离线兜底模式**（mock 抽取式生成 + hashing 向量，同一数据集）：

| 指标 | 离线基线 | 真实模型 |
| --- | --- | --- |
| Recall@5 | 1.000 | 1.000 |
| MRR | 0.953 | **0.984** |
| nDCG@5 | 0.963 | **0.989** |
| 关键事实覆盖 | 0.750 | **0.938** |
| 引用有据率 | 0.976 | **1.000** |

> 离线基线的作用是「没有 key 也能评审、单测不联网」；接上真模型后检索排序与生成质量都有明显提升。
> 两份报告分别保存在 `docs/评估报告.md`（真实模型）与 `docs/评估报告_离线基线.md`。
> 复现命令：`python scripts/evaluate.py --top-k 5 --with-generation`

---

## 3. 三分钟跑起来

### 3.1 本地（已有 venv，不重复装依赖）

```bash
cd DayOne    # 仓库根目录
# 本项目作者在 D:/kai_study/.venv 这套环境里验证，换机器的把下面 python 换成自己的即可

# 1) 配置（没有 key 也能跑，会自动降级为离线基线）
cp .env.example .env        # 然后按需填 LLM_API_KEY

# 2) 入库（首次会自动执行，也可手动跑）
D:/kai_study/.venv/Scripts/python.exe scripts/ingest.py

# 3) 一键启动：Redis 没跑会先拉起来，再起 Web 服务
D:/kai_study/.venv/Scripts/python.exe scripts/start.py
```

打开 <http://127.0.0.1:8000> 就是聊天界面；接口文档在 <http://127.0.0.1:8000/docs>。

> 本项目已针对 `D:\kai_study\.venv` 验证（Python 3.12），`requirements.txt` 里的版本与该环境一致，
> 直接复用现有环境即可，不需要 `pip install -r requirements.txt`。

#### `scripts/start.py` 做了什么

比起裸跑 uvicorn，它多三步：**探测 Redis → 没跑就拉起来 → 再起服务**，另外把
「配置了什么、实际连上了什么」在启动时就说清楚，省得对着 500 猜半天。

```bash
python scripts/start.py              # 常规：Redis 没跑就拉起，然后起服务
python scripts/start.py --memory     # 不用 Redis，进程内会话（重启丢历史）
python scripts/start.py --redis-only # 只拉 Redis（脱离终端常驻）
python scripts/start.py --detach-redis # Redis 常驻，Web 停了也不退
python scripts/start.py --check      # 只体检：探活 + 找可执行文件 + 打印数据目录
python scripts/start.py --stop-redis # 关掉 Redis（先落盘再退）
python scripts/start.py --reload     # 开发热重载
```

三个刻意的设计：

1. **只有脚本会拉进程，`app/` 里不写自启。** 生产（compose / k8s）里进程归编排层管，
   应用去拉进程会制造孤儿进程、也没法做健康检查。
2. **默认与项目同生命周期。** 脚本拉起 Redis → 起服务 → `Ctrl+C` 时先 `SHUTDOWN SAVE`
   再退出，端口和进程一起收干净。这个项目的 Redis 只服务「开着项目看效果」一个场景，
   不需要常驻；真要常驻加 `--detach-redis`（Windows `DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB`，
   POSIX `start_new_session`）。
   **用户原本就开着的 Redis 不归脚本管**——只关自己拉起来的那个。
3. **拉不起来不硬撑。** `SESSION_BACKEND=auto` 时自动回退进程内会话（功能正常，重启丢上下文）；
   `SESSION_BACKEND=redis` 时直接退出，避免「以为在用 Redis，其实在用内存」。

> **路径必须是纯 ASCII**：Windows 版 Redis 用 ANSI API 做 `chdir`，
> 数据目录里带中文会直接失败 `Can't chdir ... Illegal byte sequence`。
> 所以自动拉起的 `dir` 默认落在 `redis-server.exe` 同级的 `dayone/`，而不是项目内的 `data/`——
> 本项目路径含 `学习日志`，放项目里会起不来。想换位置就设 `REDIS_RUN_DIR=D:/redis-data`。

### 3.2 Docker

```bash
cp .env.example .env        # 填 key
docker compose up -d --build
docker compose logs -f rag
```

### 3.3 换成公司自己的知识库（核心诉求）

三步，不改任何 RAG 代码：

```bash
# ① 选适配器：能返回 JSON 的接口 → http_json；本地文件 → local_dir
KB_SOURCE=http_json

# ② 填字段映射（你的接口字段叫什么，就映射成什么）
KB_ADAPTER_CONFIG={"base_url":"https://kb.corp.example.com/api/v1","list_path":"/documents","detail_path":"/documents/{id}","auth":{"type":"bearer","token":"xxx"},"items_path":"data.items","fields":{"id":"doc_id","title":"name","content":"content.markdown","url":"url","updated_at":"updated_at"}}

# ③ 重新入库
python scripts/ingest.py --force
```

接口结构比较特殊（GraphQL / 私有签名 / 需要拼装多个接口）时，写一个 30 行的适配器类即可，
详见 [docs/适配器开发指南.md](docs/适配器开发指南.md)。

---

## 4. 架构

```text
                    ┌──────────────────────────── 知识库源（可替换） ────────────────────────────┐
                    │  local_dir（md/pdf/docx）   http_json（任意 REST）   wecom（待接口）   你的插件  │
                    └───────────────────────────────────┬──────────────────────────────────────┘
                                                        │  KnowledgeSourceAdapter 抽象
                                    ┌───────────────────▼────────────────────┐
                                    │ 入库流水线 ingest/pipeline.py           │
                                    │ 清洗 → 结构感知切块 → Embedding → Chroma│
                                    │ 内容哈希比对，增量更新 / 删除同步        │
                                    └───────────────────┬────────────────────┘
                                                        │
  用户提问 ──► FastAPI ──► LangGraph 状态机（C-RAG）
                              condense（多轮改写）
                                 ↓
                              retrieve（Dense + BM25F + RRF + MMR + 可选重排）
                                 ↓
                              grade（置信度 / 证据覆盖度判定）
                                 ↓ 不足且未超次数
                              rewrite ──► retrieve
                                 ↓ 足够
                              generate（流式，强制 [[n]] 引用）
                                 ↓
                              verify（越界编号剔除 + 弱引用标记 + 有据率）
                                 ↓
                    SSE: 先推 sources，再逐字推 answer，最后推 done
```

### 模块职责

| 目录 | 职责 | 关键文件 |
| --- | --- | --- |
| `app/adapters/` | 知识库源适配器（插件化），只负责「外部文档 → 统一模型」 | `base.py` `http_json.py` `local_dir.py` `registry.py` |
| `app/ingest/` | 切块、向量化、落库、增量同步 | `chunker.py` `embedder.py` `store.py` `pipeline.py` |
| `app/rag/` | 检索融合、C-RAG 状态机、生成、引用校验、会话 | `retriever.py` `graph.py` `service.py` `citations.py` |
| `app/eval/` | 数据集、指标、评估执行器 | `runner.py` `metrics.py` `datasets/` |
| `app/api/` | 对外 HTTP 接口 | `routes_chat.py` `routes_kb.py` `routes_eval.py` |
| `app/web/static/` | 零构建前端（原生 JS） | `index.html` `app.js` `styles.css` |
| `data/corpus/` | 示例语料：12 篇虚构公司的入职手册 | — |

---

## 5. 五个关键设计决策

### ① 为什么用适配器模式，而不是「支持 PDF / 支持企微」

如果把「读 PDF」「读企微文档」写成 `if source == "wecom"`，每接一个平台就要改主流程，
最后必然变成一坨泥。所以我们把**变化的部分**（平台差异）和**不变的部分**（切块/检索/生成）切开：

- 适配器只做一件事：把外部文档变成 `SourceDocument(external_id, title, content, url, updated_at)`；
- 主流程只认这个结构，不 import 任何平台 SDK；
- 换平台 = 加一个类，注册进 registry；主流程 **0 改动**。

更进一步：内置的 `http_json` 适配器把「字段名、分页、鉴权」全部下沉为 JSON 配置，
**大多数平台连代码都不用写**。

### ② 为什么是混合检索（Dense + BM25F），而不是纯向量

制度类问题有大量**精确词**：「60 天」「450 元」「L3」「P1—P4」。纯向量会把「450」和「600」糊在一起，BM25 不会。
反过来，BM25 对「年假怎么算」这种口语化问法又很弱。两者用 **RRF** 融合（只看排名，不调权重），
再叠加**词面覆盖度**压制「关键词堆砌型」噪声 chunk。

实测（30 条单轮问答题，top_k=5），两种 embedding 质量下各量了一遍：

**离线 hashing 向量**（`EMBEDDING_PROVIDER=hashing`）：

| 检索方式 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| 纯向量 | 0.900 | 0.793 | 0.825 |
| 纯 BM25F | 1.000 | **0.961** | 0.973 |
| 混合（向量未做质量判断） | 1.000 | 0.928 | 0.948 |
| **混合（降级时自动关向量）** | 1.000 | **0.950** | 0.960 |

**真实向量**（`text-embedding-v3`）：

| 检索方式 | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| 纯向量 | 1.000 | **1.000** | 1.000 |
| 纯 BM25F | 1.000 | 0.961 | 0.973 |
| **混合（最终版）** | 1.000 | 0.983 | 0.988 |

两件事值得说：

1. 离线那组里，**盲目融合反而更差**（0.961 → 0.928）：hashing 向量是纯词法噪声，参与投票会把 BM25 排对的文档顶下去。
   所以系统检测到降级 embedding 时会自动只用词法通道——**这一路输入不够格，就别让它投票**。
2. 换成真 embedding 后，向量通道从「噪声」变成「主力」（纯向量 MRR 1.000）。但两路确实互补：
   逐条比 top1 命中，2 条是 dense 对 bm25 错、1 条反过来（q27「账号没权限找谁审批」）。
   真实流量里编号/缩写靠词法、口语化问法靠语义，所以保留混合。

这个「先量再改、变差就撤、换模型重标」的全过程写在 [docs/调优记录.md](docs/调优记录.md)（含 4 个接真模型后才暴露的坑）。

### ③ 为什么要 LangGraph 编排，而不是一次检索硬生成

问「那其他城市呢？」这种追问，如果直接拿原句去检索，召回的是垃圾。所以流程是状态机：

```text
condense → retrieve → grade ─┬─ 足够 ────────────────► generate
                             └─ 不足 → rewrite → retrieve（最多 1 次）
```

`grade` 用**置信度 + 单块证据覆盖度**判断「知识库里到底有没有这件事」，
不足则改写重试，仍不足就明确拒答——而不是让模型自由发挥。这就是 C-RAG（Corrective RAG）的核心。

### ④ 引用怎么做到「可核对」

- Prompt 强制 `[[n]]` 编号，编号只能来自本次检索结果；
- 后处理剔除越界编号（防止编造 `[[7]]`）；
- **逐句回验**：把带编号的句子和对应 chunk 做词面重叠，低于阈值的标记为「弱引用」，前端灰显；
- 输出 `grounding_score`（有据率），评测里直接当指标用。

### ⑤ 没有 API Key 也要能跑

`LLM_PROVIDER=auto` / `EMBEDDING_PROVIDER=auto`：有 key 走真实模型，没 key 自动降级为
**抽取式 mock + hashing 向量**，并在 `/api/health` 和界面状态里**明确标注 degraded**。
好处：评审时 `docker compose up` 就能演示；单测不联网；降级从不静默。

---

## 6. 长期运行：记忆子系统

问答只能回答「手册里怎么说」，回答不了「我住哪」「材料交到哪一步了」。
`app/memory/` 补上这一层，分四类记忆：

| 层 | 存什么 | 存哪 | 生命周期 |
| --- | --- | --- | --- |
| 短期 | 本会话最近 N 轮 | `sessions.py`：Redis（配了 `REDIS_URL`）或进程内 | 会话结束 / TTL |
| 工作记忆 | 流程第几步、材料交了几项 | SQLite `work_items` | 流程结束 |
| 长期事实 | 部门、导师、工位 | SQLite `memory_facts`（双时间轴） | 长期，带有效期 |
| 长期偏好 | 回答风格、常用系统 | 同上（`kind=preference`） | 慢衰减 |

三个关键设计（细节见 [docs/演进设计.md](docs/演进设计.md)）：

- **冲突不是「谁新谁赢」**：事实更新封口不删除（`superseded`）、偏好保留序列、
  抽取置信度低只进 `pending`、事件时间早于旧记录时分「纠正」与「补录历史」两种意图；
- **遗忘是分级不是删除**：`active → cold → tombstone`，`dry_run` 先试算会淘汰谁；
  **工作记忆不参与衰减**（制度性状态该硬过期，不该淡忘）；
- **每个决策留痕**：来源原话、事件时间、置信度、被谁取代，全在 `audit_log` 里。

```bash
# 一条把它们串起来的故事：换地址、补材料、时间旅行、半年后淡忘
python scripts/memory_demo.py
```

接口：`POST /api/memory/facts`（写，返回 `action` 说明冲突怎么处理的）、
`GET /api/memory/facts?as_of=2026-11-03`（时间旅行）、
`GET /api/memory/work`（流程进度）、`POST /api/memory/decay`（遗忘，支持 dry_run）、
`GET /api/memory/audit`（审计）。

### 上下文预算（长对话不爆窗）

长对话会把窗口撑满，这里按「五层预算」分配（`app/context/`）：

```text
总预算 3000 tokens
├─ 输出预留 900          先扣掉，不靠挤占它塞内容
├─ 系统规则              固定
├─ 用户画像 / 流程状态    固定，超了截断
├─ 参考资料 chunks       **永不压缩**（压了引用回验就废）
└─ 对话历史              最近 2 轮原文 + 更早的摘要
```

每次响应都带 `context.layers` 与 `context.trace`，能直接看到"token 花在哪、丢了什么"：

```json
{"layers": [{"name": "chunks", "tokens": 725, "note": "5 条，不压缩"},
            {"name": "history.summary", "tokens": 136, "note": "6 轮压缩为摘要"}],
 "trace": ["预算 3000（可用 2100）→ ... ，历史可用 591"]}
```

**个人状态类问题在检索之前就分流**：问「我的流程到哪一步」不会去查文档，而是直接读档案
（`answer_source=memory`）；一问规则（「住宿标准是多少」）就交回文档链路。
不这么做的话，历史改写会把"我的进度"改写成制度问题，档案永远用不上——这是实测踩出来的。

### 工具调用（能办事，不只是能说）

问「王琳分机多少」「张伟坐哪」「我的流程走到哪一步」，这些答案不在手册里，在系统里。

- 注册表 + schema：`app/tools/registry.py`，主流程只认名字和 schema——和适配器同一套哲学；
- 只读工具：`get_my_onboarding_status`、`lookup_colleague`、`get_my_tickets`；
- 路由顺序 **记忆 → 工具 → 文档**：动作型问题不该被当成关键词去检索手册；
- **身份由服务端注入**（执行器统一注入，模型不能自己填 `owner_id`）；
- **结果必须裁剪**（`ToolResult.to_text`），每次调用写审计。

实测：

| 问题 | 来源 | 结果 |
| --- | --- | --- |
| 帮我查一下王琳的分机号 | tool: lookup_colleague | 8007 / 人力资源部 / 5F-B02 |
| 张伟坐在哪 | tool: lookup_colleague | 8F-A15 / 分机 8121 |
| 入职流程走到哪一步了 | tool: get_my_onboarding_status | 入职培训，缺学历证明 |
| 我提交的工单到哪一步了 | tool: get_my_tickets | REQ-20261106-001 / pending |
| 一线城市住宿标准 | knowledge_base | 450/600/800 元，带 [[1]] 引用 |

#### 写操作：提议 → 确认 → 执行（不会自己动手）

「帮我提个 VPN 权限申请」这类请求，模型可以把参数抽出来，但**不能直接执行**——
一次误开单会真的找人审批，撤不回来。所以分成三步：

```text
用户：帮我申请 VPN 权限，宿舍调试
  ↓ 模型选 submit_permission_request（只生成待确认操作，不执行）
  ↓ 返回 requires_confirmation=true + pending_action{action_id, summary, expires_at}
用户点「确认执行」
  ↓ POST /api/tools/confirm（action_id 即幂等键）
  ↓ 执行 → 开单 → 返回 REQ-20261106-001
再点一次「确认」
  ↓ replayed=true，返回同一张单（不会开第二张）
```

四条边界都写在代码里，也都有测试/探针盯着：

| 边界 | 做法 | 不这么做会怎样 |
| --- | --- | --- |
| 不确认不执行 | 执行器对 `read_only=False` 一律拒绝，只有 `confirm` 内部传 `allow_write` | 模型一句"帮你提交了"就真开了单 |
| 重复确认不多开 | `action_id` 即幂等键，第二次返回 `replayed=true` | 连点三下 → HR 收到三张重复单 |
| 别人的单不能确认 | 服务端比对 `owner_id`，403 且不透露参数 | 拿到 action_id 就能替别人办事 |
| 过期即作废 | 默认 30 分钟 TTL，过期后确认直接拒 | 昨天点的确认今天生效，用户已改主意 |

还有个容易忽略的点：**写操作的答案是确定性拼接的，不交给模型润色**。
模型很容易顺着用户说"已经帮你提交了"，而这时其实还停在待确认。

界面上的样子（`docs/ui-confirm.png`）：卡片里有摘要、`确认执行`、`取消`、有效期。

> 写操作的数据落在独立的 `data/actions.db`（待确认操作 + 工单），
> 不与记忆库混表：两者生命周期和清理策略完全不同。

#### 多步编排：把「档案 / 文档 / 系统」串起来做一件事

前面三层各自都能答一类问题，但真实请求经常要跨层。比如「帮我按第一周的清单把开发环境系统都开一下」需要：
**从文档里取系统清单** → **判断要不要真提申请** → **生成待确认操作**。

这里没有做自由 agent，而是**有界剧本（recipe）**——把固定套路写死，理由是能评测、能解释、参数有出处：

| 剧本 | 链路 | 依赖 |
| --- | --- | --- |
| `mentor_contact` | 档案（导师=王琳）→ 查组织目录 | 导师名是上一步的输出，作为下一步入参 |
| `first_week_env` | 文档（第一周勾选哪些系统）→ 生成待确认申请 | 系统清单来自文档片段，不是模型编的 |
| `ticket_rule_check` | 文档（权限有效期）→ 查我的工单 | 规则与个人状态两路合并 |
| `travel_standard` | 档案（我的职级）→ 文档（差旅标准表） | 用职级把表格定位到**具体一行**，不把整张表甩给用户 |

**先定成功口径，再写代码**（否则"能做"变成"说不清做对没"）：

1. **可解释**：返回 `plan[]`，每步记 kind / 输入 / 输出 / 是否成功，界面直接摊开给你看；
2. **参数有出处**：上游输出只能来自记忆、工具或文档；
3. **有界**：步数 ≤ 4、工具调用 ≤ 3，任一步失败立即停并说明第几步；
4. **写操作只提议**：编排里遇到写工具不执行，只生成待确认操作；
5. **不抢单意图**：匹配要求多个信号同时命中。

实测：

| 问题 | 剧本 | 结果 |
| --- | --- | --- |
| 我导师是谁？他分机多少？ | `mentor_contact` | 王琳（人力资源部），分机 8007，工位 5F-B02 |
| 入职第一周要开通哪些系统？ | `first_week_env` | 只答清单（GitLab、Jenkins、KaiRAG、数据平台只读），**不提案** |
| 帮我按第一周的清单把开发环境系统都开一下 | `first_week_env` | 生成待确认申请 → 确认后 REQ-20261106-002 |
| 敏感权限多久到期？我申请的那个还有效吗 | `ticket_rule_check` | 30 天到期自动回收 + 你的工单待审批 |
| 我下周去上海出差，住宿能报多少？ | `travel_standard` | 你的职级 P5 + 一线城市 → 600 元/晚（附差距条款） |

界面上的样子（`docs/ui-plan.png`）：答案下面有一条「执行了 3 步」的计划条，每步标了来源类型。

两句实话：

- **「提问」和「委托」必须分开**。同样提到「开通系统」，「第一周要开通哪些系统」是提问，
  只答清单；有「帮我/提一下」才生成申请。混在一起就是替用户做主。
- **抽参数只认白名单 + 原句校验**。语料里「申请路径：OA → 权限申请」中的 OA 是入口不是系统，
  自由抽取会把它当成要开通的系统，用户照着提交就是错单。抽不到就不生成申请。

### 闭环探针：怎么证明它「隔天还记得、该忘的忘干净」

记忆系统最容易自欺：功能写完了，但没人验证它在时间轴上会收敛。
`app/eval/probe.py` 里是 **13 条带日期的脚本**——在固定时钟上跑到「明天 / 两周后 / 一年后 / 两年后」，
跑完逐条断言，产出「哪条断言、期望什么、实际什么」：

```bash
python scripts/probe.py                     # 13 条探针，约 0.3 秒
python scripts/probe.py --only noise_cleanup # 单条
# 报告：docs/探针报告.md（含每条的时间轴与断言明细）
```

| 探针 | 验的是什么 |
| --- | --- |
| `retention_next_day` | 昨天教的三件事，次日与两周后都答得出来 |
| `paraphrase_routing` | 「还差啥」「到哪了」仍路由到档案；问规则的不能抢答 |
| `conflict_latest_wins` | 前后矛盾以最新为准，旧值封口不删，时间旅行可查 |
| `correction_vs_backfill` | 「你记错了」与「我以前在上海」不混 |
| `noise_cleanup` | 一年不提 → 冷归档出画像；再久置 → 立墓碑 |
| `pinned_survives_decay` | 工号/部门常驻，不被当闲聊清掉 |
| `work_state_survives_decay` | 流程状态不参与淡忘；重复旧进度被打回 |
| `forgetting_traceable` | 每次降级/埋葬在审计里查得到权重与闲置天数 |
| `cross_user_isolation` | A 的信息不出现在 B 的画像与流程里 |
| `write_action_idempotent` | 提议不落库、重复确认只开一张单、他人不能确认、过期作废 |
| `multi_step_plan` | 上一步的输出真被下一步用上；档案没数据时不乱调；规则问题不被编排截胡 |
| `chunks_never_compressed` | 预算再紧，chunk 原文完整进上下文 |
| `summary_survives_fallback` | 摘要模型挂了也不丢历史（走抽取式兜底） |

两个刻意的设计：

- **探针必须能红**。`tests/test_probe.py` 有一条反向测试：把常驻保护关掉后，
  `pinned_survives_decay` 必须失败——断言永远绿的探针等于没写；
- **跑得快、可进 CI**。每条探针各开一套临时库并自动清理，11 条共 0.3 秒，退出码 0/1。

### 编排端到端评测：证明「多步的事办对了」，而不只是「文档找得到」

检索指标（Recall / MRR）回答不了「两步有没有串对」。`app/eval/datasets/orchestration_e2e.jsonl`
里 16 条用例，每条都写死期望：命中哪个剧本、plan 的步骤**顺序**、该不该产生待确认操作、
答案里必须有什么 / 不能有什么。

```bash
python scripts/evaluate.py --orchestration     # 报告：docs/评估报告_编排.md
```

| 指标 | 结果 | 卡的是什么错 |
| --- | --- | --- |
| 端到端成功率 | 1.000（16/16） | 任何一项不过就算失败 |
| 剧本选择准确率 | 1.000 | 该命中的没命中、不该命中的被接走 |
| 步骤顺序准确率 | 1.000 | 把第二步放到第一步前面（数据串错） |
| 内容准确率 | 1.000 | 必含事实缺失、出现「已提交成功」这类禁词 |
| 误编排率 | 0.000 | 规则问题被编排抢走 |

其中 4 条是**负例**（该不编排）：规则问题、个人进度、单据状态、通用制度标准。
负例比正例重要——编排最容易犯的错不是办不成，是「抢着办」。

### 探针问句反生成：让评估集自己长大

手写的问题有个毛病：写的人知道答案，会不自觉用文档里的词。所以反着做——
从原文生成问题，再用原文反查把模型编的事实筛掉：

```bash
python scripts/gen_questions.py --per-doc 2 --limit 16   # 产出 generated_probes.jsonl
python scripts/evaluate.py --dataset app/eval/datasets/generated_probes.jsonl --with-generation
```

14 条反生成问句（如「咱们为啥非要自己搞个 KaiRAG 啊，直接用 LangChain 不香吗？」）实测
Recall@5 0.929 / 关键事实覆盖 0.929 / 引用有据率 1.000，见 `docs/评估报告_反生成.md`。
比手写集略低是正常的——它更像新人真的会打的字。

两条防幻觉的闸：生成的 `must_contain` 必须在原文里**逐字出现**，且必须是带数字或英文标识的
短事实（长句连人复述都会换措辞，拿它判分只会让指标虚低）。

### 多级摘要入库：同一段历史只压一次

长对话的摘要原来是「每次请求重算一遍」：既重复花钱，又让同一段历史每次压出不同结果，
用户会觉得助手记性不稳。现在改成两级、增量、写盘：

```text
L1 段摘要（只压 covered_turns 之后的新增轮次）
   ↓ 合并
L2 滚动摘要（覆盖全部历史，长度封顶）
   ↓
data/summaries.db  ← 下次没有新增就直接复用，一次模型都不调
```

实测 8 轮对话：第 4 轮 `L1 首次摘要：2 轮` → 第 5—7 轮 `L2 滚动摘要：2→4 / 4→6 / 6→8 轮` →
第 8 轮 `复用已入库摘要（覆盖 8 轮，无新增）`。会话重置时摘要一起清掉，不会「清了历史还带着旧摘要」。

### 会话外置：为什么多副本就必须上 Redis

摘要解决的是「历史太长」，会话存储解决的是「历史存在哪」。进程内 dict 在单副本够用，
但这三种情况会直接坏掉：

- **多副本 / 滚动发布**：第 2 句落到另一个进程，「那其他城市呢」就变成一句没有上下文的新问题；
- **重启**：会话全丢，用户得从头讲；
- **扩容**：副本越多，命中同一进程的概率越低，追问体验越随机。

这不是性能问题，是**正确性**问题。所以 `app/rag/sessions.py` 加了 Redis 实现，
填了 `REDIS_URL` 就自动启用，不填照旧走内存（功能一样，只是重启会丢）：

```bash
# 方式一：docker compose（redis 服务已配好，含 appendonly 持久化）
docker compose up -d redis

# 方式二：已有 Redis，直接填 .env
REDIS_URL=redis://127.0.0.1:6379/0
SESSION_BACKEND=auto        # auto | memory | redis

# 方式三：Windows 免安装版（本机验证用的就是这个，D:\Redis）
#   redis-server.exe redis.windows.conf
#   想开机自启（需管理员）：redis-server.exe --service-install redis.windows-service.conf

# 部署前自检：连通 / 读写顺序 / TTL / 多实例共享，四项全过才返回 0
python scripts/check_redis.py
```

> **Windows 注意**：官方没有 Windows 原生版，用社区移植的 `tporadowski/redis`（5.0.14）
> 或 Memurai。Redis 5.x 没有 `HELLO` 命令，所以客户端显式用 RESP2（`protocol=2`）——
> redis-py 8.x 默认 RESP3，不指定会直接报 `unknown command 'HELLO'`。

**为什么用 List 而不是 String(JSON)**：String 方案每次都是「读出来 → 改 → 写回」，
两个请求同时追加会互相覆盖（丢一条消息），而且历史越长越慢。
List 用 `LPUSH + LTRIM + EXPIRE` 一条 pipeline 完成，全是 O(1)，没有读-改-写就没有丢更新。

**降级不做假**：Redis 连不上时自动回落进程内，但会打 warning 并**熔断 30 秒**——
否则每个请求都要等一次连接超时，一个挂掉的 Redis 能把整个服务拖慢。
故障期间的历史会短暂分叉（故障前在 Redis、故障中在内存），这是刻意取舍：
宁可体验降级，也不要整个问答不可用。

实测（本机 Redis 5.0.14.1）：写入一轮会话后**重启服务**，`boot.sessions` 显示
`{'sessions': 2, 'backend': 'redis'}`，同一 `session_id` 追问「那其他城市呢？」
被正确改写成「其他城市的住宿标准是多少？」并答出 350/450/600 元。
`redis-cli keys 'dayone:sess:*'` 能看到键，`TTL` 是 7200 秒。

---

## 7. 接口一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/api/chat` | 一次性问答，返回答案 + 来源 + 引用 + 指标 |
| POST | `/api/chat/stream` | SSE 流式：`start / status / sources / delta / done` |
| POST | `/api/chat/reset` | 清空某会话的多轮记忆 |
| POST | `/api/feedback` | 「这个问题没答案」，落盘 `data/feedback.jsonl` |
| GET | `/api/health` | 健康检查（模型、降级状态、知识库规模、适配器列表） |
| GET | `/api/kb/stats` | 文档数 / 分块数 / 文档清单 |
| GET | `/api/kb/adapters` | 已注册适配器 |
| GET | `/api/kb/adapters/{name}/probe` | 探测某个适配器能否拉到目录（不落库） |
| POST | `/api/kb/ingest` | 触发入库（`force` / `reset` / `limit`） |
| POST | `/api/kb/search` | 检索调试：只看召回，不生成 |
| GET | `/api/eval/dataset` · POST `/api/eval/run` · GET `/api/eval/last` | 评估集与评估报告 |
| GET | `/api/tools` | 工具清单（`read_only` 标记、写开关、确认有效期） |
| POST | `/api/tools/propose` | 生成待确认操作（**不执行**；`client_key` 防连点） |
| POST | `/api/tools/confirm` | 确认执行（`action_id` 即幂等键，重复确认返回 `replayed=true`） |
| POST | `/api/tools/cancel` | 取消待确认操作 |
| GET | `/api/tools/pending` · GET `/api/tools/tickets` | 我的待办 / 我提交过的工单 |

```bash
# 一次性问答
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"报销单必须在多久内提交？"}' | python -m json.tool

# 流式
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"每周最多可以远程办公几天？"}'

# 检索调试（看 dense / bm25 / 融合分）
curl -s -X POST http://127.0.0.1:8000/api/kb/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"住宿标准","top_k":3}'
```

返回体（节选）：

```json
{
  "answer": "消费发生后 **60 天内**提交报销单，逾期系统自动关闭[[2]]。",
  "sources": [{"n": 2, "doc_title": "报销、差旅与采购流程", "heading": "报销基本规则", "score": 0.83}],
  "citations": [{"n": 2, "used": true, "hits": 1}],
  "confidence": 0.62, "grounding_score": 1.0, "refused": false,
  "standalone_query": "报销单必须在多久内提交？", "latency_ms": 780
}
```

---

## 8. 评估与调优

```bash
# 只评检索（快、不花 token）
python scripts/evaluate.py --top-k 5

# 连生成一起评（需要真模型才有意义）
python scripts/evaluate.py --top-k 5 --with-generation

# 参数扫描（融合权重 / BM25 长度归一 / 标题权重）
python scripts/tune.py --top-k 3
```

数据集格式（一行一条）：`question` / `expected_docs` / `must_contain` / `type` / `history`，
见 `app/eval/datasets/golden_onboarding.jsonl`。新增语料时**同步补用例**，否则调优就是盲调。

---

## 9. 目录结构

```text
DayOne/
├── app/
│   ├── main.py                 # FastAPI 入口、lifespan、异常处理、静态资源
│   ├── config.py               # 全部配置（pydantic-settings，读 .env）
│   ├── adapters/               # ★ 知识库适配器（插件化核心）
│   ├── ingest/                 # 切块 / embedding / Chroma / 增量流水线
│   ├── rag/                    # 混合检索、LangGraph、生成、引用、会话
│   ├── memory/                 # 工作记忆 / 长期事实 / 冲突消解 / 三级遗忘 / 统一时钟
│   ├── context/                # 预算分层 + 多级摘要入库（summary_store.py）
│   ├── tools/                  # 工具注册表 / 路由 / 确认式写操作 / 有界剧本编排
│   ├── eval/                   # 评估集 + 指标 + 执行器 + 闭环探针
│   ├── api/                    # 路由 + 数据契约 + 鉴权
│   ├── core/                   # 日志、异常
│   └── web/static/             # 前端（原生 JS，无构建）
├── data/corpus/                # 12 篇示例入职手册
├── docs/
│   ├── 技术栈说明.md            # ★ 每项技术在哪个文件、干什么用
│   ├── 架构设计.md
│   ├── 适配器开发指南.md
│   ├── 演进设计.md              # 记忆/上下文/工具/编排的分期与取舍
│   ├── 评估报告.md              # 检索 + 生成（脚本生成）
│   ├── 评估报告_编排.md          # 编排端到端（16 条用例）
│   ├── 评估报告_反生成.md        # 反生成问句集（14 条）
│   ├── 探针报告.md              # 13 条闭环探针
│   └── 调优记录.md              # 每轮踩坑的 before/after
├── scripts/                    # ingest / evaluate / probe / gen_questions / check_redis / tune / ask
├── tests/                      # 108 个测试（单测 + 接口冒烟 + 评测逻辑）
├── Dockerfile · docker-compose.yml
├── requirements.txt · .env.example
└── README.md
```

---

## 10. 已实现 / 未实现

**已实现**

- 适配器插件化（内置 local_dir / http_json / wecom 占位 + 外部模块加载 + 多源并行挂载）
- 增量入库（内容哈希比对、下线文档清理、集合名带模型指纹避免维度冲突）
- 混合检索（Dense + BM25F + RRF + 词面覆盖度 + MMR + 可选 LLM 重排）
- 多轮追问（LangGraph 查询改写 + 会话记忆 + TTL）
- 来源标注与引用高亮（`[[n]]` 协议 + 逐句回验 + 前端点击定位高亮）
- **记忆子系统**（工作记忆 / 长期事实 / 双时间轴冲突 / 三级遗忘 / 审计日志 / 统一时钟支持时间旅行）
- **上下文预算**（五层分配 / 资料不压缩 / 旧历史摘要 / 个人状态问题检索前分流）
- **工具调用**（注册表 / 执行器 / 路由 / 只读白名单 / 身份注入 / 审计）
- **确认式写操作**（提议 / 确认 / 幂等 / 归属校验 / TTL / 独立动作库 / 工单查询）
- **多步编排**（4 个有界剧本 / 跨层取参 / 计划可解释 / 步数与调用上限 / 不抢单意图）
- **编排端到端评测**（16 条用例：剧本命中 / 步骤顺序 / 该不该代办 / 误编排率；含 4 条负例）
- **探针问句反生成**（从原文生成问题 + 原文反查筛幻觉，14 条实测报告）
- **多级摘要入库**（L1 段摘要 → L2 滚动摘要 → SQLite；无新增轮次不调模型，会话重置同步清理）
- **会话外置**（Redis List + TTL：多副本共享、重启不丢追问；连不上自动回退进程内并熔断 30s）
- **闭环探针**（13 条：跨天保留 / 冲突 / 遗忘 / 隔离 / 写操作幂等 / 编排串接 / 压缩不吞证据，附报告）
- 拒答兜底、SSE 流式、离线降级、Docker 化（含 Redis 服务）、评估脚本与 119 个测试

**没做（有意为之，避免过度设计）**

- 用户体系与权限隔离（当前是内网单租户；接企微可拿 userid 做过滤）
- 向量库分布式（Chroma 单机够用；上量换 pgvector / Milvus，只改 `store.py`）
- 定时增量同步（现在靠脚本/接口触发；接 cron 或 APScheduler 即可）

---

## 11. 常见问题

**Q：没有 API Key 能演示吗？**
能。会自动用离线基线，界面状态栏会显示「离线基线模式」，不静默降级。

**Q：换个 embedding 模型要不要重建库？**
要。集合名带 `provider__model__dim` 指纹，新模型会自动建新集合，跑一次 `ingest.py --force` 即可，旧集合可直接删。

**Q：为什么回答有时把表格表头也列出来？**
离线抽取式基线只挑句子不做总结，属于基线特性；接真实模型后不会有这个问题。

**Q：知识库里没有的问题，会不会硬编？**
不会。`grade` 节点判定不足会直接走兜底话术，并在返回体里 `refused=true`，同时把召回到的近似文档列出来供你判断。

---

## 12. 技术栈索引

FastAPI / Docker / LangGraph / Chroma / Embedding / RAG 各自「用在哪、干什么」，
见 **[docs/技术栈说明.md](docs/技术栈说明.md)**（含文件路径与函数名）。
