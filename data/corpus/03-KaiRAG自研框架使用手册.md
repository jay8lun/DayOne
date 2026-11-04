# KaiRAG 自研框架使用手册

KaiRAG 是公司自研的检索增强生成框架，封装了文档接入、切块、混合检索、重排、引用标注和评测。新人最常问的是「为什么要自研，直接用 LangChain 不行吗」——答案是：LangChain 提供的是积木，KaiRAG 提供的是**已经拼好并且经过线上验证的流水线**，以及公司统一的引用协议、评测集和权限模型。底层仍然复用 LangChain / LangGraph 的组件。

## 整体结构

```text
数据源适配器 (Adapter)  ->  清洗切块 (Chunker)  ->  向量化 (Embedder)
                                                        |
用户提问 -> 查询改写 -> 混合检索 (Dense + BM25 + RRF) -> 重排 -> 生成
                                                        |
                                                  引用校验 -> 输出
```

## 安装与初始化

```bash
pip install kai-rag==2.4.1      # 内网源：http://pypi.yunshu.internal/simple
```

```python
from kai_rag import KaiRAG

kb = KaiRAG(
    collection="onboarding_kb",
    embedder="bge-m3",          # 默认，1024 维
    retriever="hybrid",          # hybrid | dense | bm25
    top_k=5,
)
print(kb.ask("报销需要哪些材料？"))
```

## 三个必须理解的配置项

### top_k（召回条数）

默认 5。评估发现：top_k=3 时多跳问题召回不足；top_k=8 时噪声明显增加且延迟上升 40%。**入职手册这类文档推荐 4—6**。

### chunk_size（切块大小）

默认 512 字符，overlap 64。中文场景下超过 800 字符的 chunk 会导致生成时「注意力分散」，答案容易串段；低于 200 字符则丢失上下文，出现「该流程」指代不明。

### hybrid 权重

混合检索用 RRF 融合，不直接加权求和（因为向量相似度和 BM25 分数不在同一量纲）。RRF 的 k 默认 60，公式为 `score = Σ 1/(k + rank)`。

## 引用协议（重要）

KaiRAG 要求生成结果必须带编号引用，格式为 `[[1]]`、`[[2]]`，编号对应检索结果顺序。后处理会：

1. 解析编号，校验编号是否越界（越界直接删除）；
2. 校验引用句与被引 chunk 的词面重叠度，低于阈值标记为「弱引用」；
3. 输出结构化 citations 列表，前端据此做来源高亮。

**不允许**在没有检索结果的情况下让模型自由发挥。知识库没有答案时，标准话术是：「知识库里暂时没有找到相关说明，建议联系 HR 或你的导师确认。」

## 常见错误

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `Collection dimension mismatch` | 换了 embedding 模型但没重建集合 | 集合名加模型指纹，或执行 `kai-rag reindex --reset` |
| `No relevant documents` | 语料没入库 / 阈值过高 | 先查 `/api/kb/stats`，再调低 `min_confidence` |
| 回答串段、答非所问 | chunk 过大或 overlap 不当 | 调小 chunk_size 到 400 左右 |
| 引用编号错乱 | prompt 里没约束编号格式 | 用官方 prompt 模板，别自己改引用标记 |

## 与 LangGraph 的关系

KaiRAG 2.x 的检索纠错流程用 LangGraph 编排成状态机：`condense -> retrieve -> grade -> (rewrite -> retrieve)* -> generate`。`grade` 节点判断召回是否足够，不足则改写查询重试一次，仍不足则走兜底话术。这样比「一次检索硬生成」的幻觉率显著降低（内部评测：无答案问题的错误作答率从 31% 降到 9%）。
