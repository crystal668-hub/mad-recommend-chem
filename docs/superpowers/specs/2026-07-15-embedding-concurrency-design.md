# 文本向量化并发与向量写入流水线标准规格

## 1. 文档信息

| 属性 | 值 |
| --- | --- |
| 状态 | Proposed |
| 版本 | 1.0 |
| 日期 | 2026-07-15 |
| 适用入口 | `build_vector_db_batch.py` |
| 核心模块 | `database/embedder.py`、`utils/request_limiter.py`、`database/vector_store.py` |

本文使用以下规范词：

- **必须（MUST）**：实现和验收不可省略。
- **应该（SHOULD）**：除非有明确、记录在案的理由，否则必须实现。
- **可以（MAY）**：可选增强，不影响核心规格验收。

## 2. 目标

将当前按 agent 独立发起 embedding 请求、按小批次同步写入 Chroma 的流程，改造为：

1. 按 provider 能力进行原生批量向量化。
2. 按共享配额组统一管理并发、RPM/RPS 和 TPM/TPS。
3. 使用滚动窗口持续调度请求，避免波次式突发和队头阻塞。
4. 将通过校验的向量送入有界写队列，由单一 Chroma writer 批量写入。
5. 保证 provider 失败、限流或无效响应不会被转换成零向量写入数据库。
6. 保留断点续跑能力，使失败 chunk 在下一次运行中可被自动补齐。

## 3. 背景与当前基线

当前配置对应以下调用拓扑：

| Agent | 逻辑 provider | 实际 embedding endpoint | 当前默认行为 |
| --- | --- | --- | --- |
| agent1 | ZenMux | `agent-team-api.myrimate.cn/v1` | 单文本同步请求 |
| agent2 | Voyage | Voyage API | 单文本同步请求 |
| agent3 | ZenMux | `agent-team-api.myrimate.cn/v1` | 单文本同步请求 |
| agent4 | Aliyun | `agent-team-api.myrimate.cn/v1` | 单文本同步请求 |

当 `embedding_concurrency=N > 1` 时，agent1 和 agent3 各自并发 N 个 batch；agent2
和 agent4 仍保持单请求。由此得到：

```text
myrimate 网关最大在途请求 = 2N + 1
进程最大 embedding 在途请求 = 2N + 2
```

当前异步请求不经过进程级 `GlobalRequestLimiter`，且并发上限按 agent 而不是按共享
endpoint 或账户配额计算。OpenAI SDK 还会在应用层重试之外自动重试，导致单个逻辑请求
可能产生多次网络尝试。

最近一组可用的完整历史构建基线为 392,395 chunks/agent：

| Agent | 历史吞吐 | 历史完成时间 |
| --- | ---: | ---: |
| agent1 | 1.60 chunks/s | 约 68.0 小时 |
| agent2 | 2.21 chunks/s | 约 49.2 小时 |
| agent3 | 1.71 chunks/s | 约 63.6 小时 |
| agent4 | 3.91 chunks/s | 约 27.9 小时 |

该历史数据来自旧 provider 路由，只用于说明逐文本请求的数量级，不作为新 provider 的
绝对性能承诺。新方案必须在同一数据、同一 provider、同一额度下建立新的 A/B 基线。

## 4. 范围

### 4.1 包含范围

- 批量构建脚本的 embedding 调度。
- OpenAI-compatible、Voyage 和 Aliyun embedding adapter。
- provider/account/gateway 级共享配额管理。
- 请求批次、重试、退避、失败记录和结果校验。
- embedding producer 与 Chroma writer 之间的有界流水线。
- 断点续跑和最终构建状态。
- 结构化日志、性能指标、测试和验收。

### 4.2 不包含范围

- 更换 embedding 模型或重新定义各 agent 的模型职责。
- 修改 chunking 语义、chunk ID 或 Chroma collection 命名。
- 将 Chroma 替换为其他向量数据库。
- 在线 RAG 查询请求的批量化。在线查询可复用新的配额管理器，但不属于本期吞吐验收范围。
- 自动购买或提升 provider 配额。

## 5. 术语

- **Embedding profile**：一个 agent 的模型、维度、endpoint、凭据和 provider 配置。
- **Transport**：实际调用协议，例如 `openai_compatible` 或 `voyage_sdk`。
- **Quota group**：共享同一并发/RPM/TPM 预算的一组请求。它不等同于 agent 或 provider 名称。
- **Request batch**：一次 provider API 请求包含的文本集合。
- **Write batch**：一次提交给 Chroma 的文档、向量、metadata 和 ID 集合。
- **In-flight**：已经取得配额并已发出、但尚未完成的网络请求。
- **Logical request**：调度器提交的一次 request batch，包括其全部重试尝试。
- **Network attempt**：一次实际 HTTP/SDK 调用。
- **Valid vector**：数量、维度、数值和索引均通过校验的 embedding。

## 6. 总体架构

```text
加载并切分文档
      |
      v
按 collection 批量检查已有 chunk ID
      |
      v
按 embedding profile 构造 request batch
      |
      v
共享 QuotaScheduler
  - max_inflight
  - RPM/RPS
  - TPM/TPS
  - 自适应降载
      |
      v
ProviderAdapter.embed_documents_batch()
      |
      v
结果排序与严格校验
      |
      +--------> 失败清单（不写 Chroma）
      |
      v
有界 write queue
      |
      v
单一 Chroma writer 合并并批量写入
      |
      v
运行摘要与断点续跑状态
```

### 6.1 所有权边界

- Batcher 只负责形成满足 provider 限制的 request batch。
- QuotaScheduler 只负责调度、公平性和限流，不解析 provider 响应。
- ProviderAdapter 只负责网络调用及协议转换。
- Validator 只负责把响应映射回输入并验证向量。
- Chroma writer 是唯一允许调用 `VectorStore.add_documents()` 的并发角色。

## 7. 配置规格

### 7.1 顶层运行配置

新增 `embedding_runtime` 配置段：

```yaml
embedding_runtime:
  global_max_inflight: 8
  request_timeout_seconds: 60
  write_batch_size: 100
  write_queue_max_batches: 8
  write_flush_interval_ms: 500
  existing_id_check_batch_size: 1000
  failure_manifest_dir: "./outputs/embedding_failures"
  adaptive_concurrency: true

  retry:
    max_attempts: 6
    base_delay_seconds: 1
    max_delay_seconds: 60
    honor_retry_after: true
    jitter: "full"

  quota_groups:
    myrimate:
      initial_inflight: 2
      max_inflight: 4
      requests_per_minute: null
      tokens_per_minute: null
      cooldown_seconds: 30

    voyage_org:
      initial_inflight: 2
      max_inflight: 4
      requests_per_minute: 2000
      tokens_per_minute: 3000000
      cooldown_seconds: 30
```

规则：

- `global_max_inflight` 是进程级保险上限，必须覆盖同步和异步 embedding 网络尝试。
- `quota_groups.*.max_inflight` 是硬上限，运行期自适应并发不得超过它。
- `requests_per_minute` 或 `tokens_per_minute` 为 `null` 时，表示额度未知，不表示无限制；
  此时必须保留保守并发和 429 自适应降载。
- Voyage 数值是公开基础额度的初始示例，部署时必须以当前组织和项目控制台为准。
- myrimate 是自定义网关，实施前必须确认其限流是按 endpoint、账户、API key 还是上游模型聚合。

### 7.2 Agent embedding profile

在现有 agent embedding 配置中增加协议能力和配额归属：

```yaml
llm:
  agent1:
    embedding_provider: "zenmux"
    embedding_transport: "openai_compatible"
    embedding_quota_group: "myrimate"
    embedding_request_batch_size: 32
    embedding_max_batch_items: 2048
    embedding_max_batch_tokens: 300000

  agent2:
    embedding_provider: "voyage"
    embedding_transport: "voyage_sdk"
    embedding_quota_group: "voyage_org"
    embedding_request_batch_size: 128
    embedding_max_batch_items: 128
    embedding_max_batch_tokens: null

  agent3:
    embedding_provider: "zenmux"
    embedding_transport: "openai_compatible"
    embedding_quota_group: "myrimate"
    embedding_request_batch_size: 32
    embedding_max_batch_items: 2048
    embedding_max_batch_tokens: 300000

  agent4:
    embedding_provider: "aliyun"
    embedding_transport: "openai_compatible"
    embedding_quota_group: "myrimate"
    embedding_request_batch_size: 1
    embedding_max_batch_items: 1
    embedding_max_batch_tokens: null
```

规则：

- `embedding_provider` 描述模型/provider 归属。
- `embedding_transport` 决定使用哪个 adapter，不得再通过 provider 名称推断是否支持异步 batch。
- `embedding_quota_group` 决定共享配额，不得默认使用 agent 名称。
- `embedding_request_batch_size` 是应用目标值，实际 batch 还必须受最大 items、最大 tokens 和
  响应体保护阈值约束。
- agent4 只有在 myrimate 网关确认支持 `input=[...]` 后才能把上述两个 batch 配置提升到
  10 或 provider 明确允许的其他值；确认前必须保持 1。
- 凭据不得写入 quota group 名称、日志或 metrics label。

### 7.3 CLI 兼容性

新增以下 CLI override：

- `--embedding-request-batch-size`
- `--embedding-write-batch-size`
- `--embedding-global-max-inflight`
- `--embedding-quota-overrides <path>`

现有 `--embedding-batch-size` 在一个兼容周期内作为
`--embedding-request-batch-size` 的弃用别名，并在日志中输出一次 warning。现有
`--embedding-concurrency` 作为单 agent 参数语义必须弃用，因为它无法表达共享 quota group。

## 8. Request batch 规格

### 8.1 原生 batch

- 支持 batch 的 transport 在 `max_inflight=1` 时也必须提交列表输入。
- 不得以 `concurrency > 1` 作为启用 batch 的条件。
- Voyage 必须调用一次 `voyage_client.embed(texts=batch_texts, ...)`，不得循环调用单文本接口。
- OpenAI-compatible adapter 必须调用一次 `client.embeddings.create(input=batch_texts, ...)`。

### 8.2 Batch 形成规则

Batcher 按输入顺序贪心装填，并同时满足：

```text
items <= embedding_request_batch_size
items <= embedding_max_batch_items
estimated_tokens <= embedding_max_batch_tokens（非 null 时）
estimated_response_bytes <= response_soft_limit_bytes
```

任何单条文本超过模型 token 上限时，不得截断后静默入库。处理策略必须显式配置为：

- `fail`：记录失败并跳过，默认策略。
- `truncate`：仅当 provider/model 规格明确允许，且 metadata 记录截断事实时使用。

### 8.3 建议初始 batch

| Profile | 初始 request batch | 调优候选 | 说明 |
| --- | ---: | --- | --- |
| agent1 ZenMux 3072 维 | 32 | 16/32/64/128 | 响应体较大，不直接使用 provider 最大 2048 |
| agent2 Voyage 1024 维 | 128 | 64/128 | Voyage 官方建议批量请求 |
| agent3 ZenMux 3072 维 | 32 | 16/32/64/128 | 必须确认网关对模型的 batch/token 限制 |
| agent4 Aliyun 1024 维 | 1，确认后 10 | 1/10/25 | 当前经过自定义网关，不能直接套用直连额度 |

ZenMux 公共文档给出的通用 batch 上限是 2048 inputs、总计 300,000 tokens。该上限只用于
合法性约束，不是推荐运行值。高维向量响应体会先于 input 上限成为吞吐和内存瓶颈。

## 9. 配额与调度规格

### 9.1 共享调度器

- 进程内必须只有一个 embedding QuotaScheduler。
- 所有 agent 的 request batch 必须先取得 global slot，再取得 quota-group slot。
- 同一 quota group 内必须共享 in-flight、RPM 和 TPM 计数。
- retry 的每次 network attempt 都必须重新消耗 RPM/TPM 预算，但仍属于同一个 logical request。
- 调度器必须使用滚动窗口；某请求完成后应立即从有资格的队列补充下一个请求。
- 不得再按 `gather(N requests) -> 等待全部 -> sleep N 次` 的波次方式运行。

### 9.2 公平性

同一 quota group 内应使用按 agent 轮询的公平队列：

1. 每个 agent 至多连续提交一个 batch。
2. 存在其他待调度 agent 时，不允许单个 agent 占满全部新释放 slot。
3. 某 agent 队列为空时，其机会可被其他 agent 使用。

该规则防止 agent1/agent3 长任务使 agent4 长时间饥饿。

### 9.3 RPM/TPM

- RPM/RPS 必须按实际 network attempt 计数。
- TPM/TPS 应使用 provider tokenizer；无法获得时可以使用保守估算，但必须记录估算方法。
- 调度前必须同时满足请求数和 token 预算。
- 不得使用固定 `sleep_between_batches` 作为主限流机制。
- 固定 sleep 参数在新调度器启用时必须被忽略并输出弃用 warning。

### 9.4 自适应并发

运行并发从 `initial_inflight` 开始，采用 AIMD 策略：

- 收到 429、明确的 throttling error 或 provider overload 时，将当前并发降为
  `max(1, floor(current / 2))`。
- 进入 `max(Retry-After, cooldown_seconds)` 冷却期。
- 连续 100 个 logical requests 成功且五分钟内无 throttling 时，并发增加 1。
- 当前并发不得超过 `max_inflight`。
- 401、403、无效模型、无效参数等永久错误不得通过降低并发反复重试。

## 10. Retry 规格

### 10.1 单一重试层

- OpenAI/AsyncOpenAI client 必须设置 `max_retries=0`，由应用调度器统一重试。
- 或者完全删除应用层 retry 并只使用 SDK retry。两者只能选择一个。
- 本规格推荐应用层统一重试，以便正确统计 network attempt、RPM、TPM 和 Retry-After。

### 10.2 可重试错误

以下错误可以重试：

- connect/read/write timeout。
- HTTP 408、409、429。
- HTTP 500、502、503、504。
- provider 明确标记为暂时性 overload 的错误。

以下错误默认不可重试：

- HTTP 400、401、403、404、422 中的参数或权限错误。
- 不支持的模型、维度或 batch 大小。
- 输入文本超过硬限制。
- 响应向量维度与 collection 规格不一致。

### 10.3 退避

- `Retry-After` 存在时必须优先采用。
- 否则使用 full-jitter exponential backoff。
- 默认最多 6 次 network attempts，包括首次调用。
- 最终失败必须返回结构化失败，不得返回零向量占位。

## 11. 结果校验与失败语义

### 11.1 向量校验

每个 batch 必须验证：

1. 响应条目数等于非空输入条目数。
2. OpenAI-compatible 响应按 `data.index` 映射回输入，不依赖返回数组顺序。
3. 每个向量维度等于 collection/profile 预期维度。
4. 所有元素都是有限数值，不包含 NaN 或 Infinity。
5. provider 返回的全零向量视为无效；空白输入应在调度前被拒绝，而不是生成零向量。

### 11.2 写入规则

- 只有 Valid vector 可以进入 write queue。
- provider failure、无效响应和校验失败对应的 chunk ID 必须保持未写入状态。
- 不得用 `[0.0] * dimension` 表示 provider failure。
- 已成功写入的 chunk 不因同 batch 其他 chunk 失败而回滚；失败项必须单独记录。

### 11.3 失败清单

每次运行输出 JSONL failure manifest，至少包含：

```json
{
  "agent": "agent3",
  "collection": "xxx_agent3",
  "chunk_id": "...",
  "provider": "zenmux",
  "model": "google/gemini-embedding-2",
  "quota_group": "myrimate",
  "attempts": 6,
  "error_type": "rate_limit",
  "retryable": true,
  "message": "sanitized provider error"
}
```

manifest 不得包含 API key、Authorization header 或完整敏感请求体。

### 11.4 运行状态

构建结果必须使用以下状态：

- `ok`：所有目标 chunk 已存在或成功写入，失败数为 0。
- `partial`：至少一个 chunk 成功，但仍有未写入失败项。
- `error`：配置错误、collection 不可用或 fail-fast 模式下出现不可恢复错误。
- `skipped`：沿用现有 skip 语义。

只要 failure manifest 非空，状态不得报告为 `ok`。

## 12. Chroma 写入流水线

### 12.1 单 writer

- 默认保持一个 Chroma writer，避免持久化后端锁竞争。
- embedding producer 不得直接调用 `VectorStore.add_documents()`。
- write queue 必须有界，默认最多 8 个待写 batch。
- 队列满时 producer 必须等待，形成反压，禁止无界缓存 embedding。

### 12.2 写批次合并

Writer 在满足任一条件时 flush：

- 已累计 `write_batch_size`，默认 100。
- 第一条待写记录已等待 `write_flush_interval_ms`，默认 500 ms。
- 收到正常结束信号。

写入前必须再次确认 documents、embeddings、metadatas 和 ids 长度一致。写入成功后才能增加
`newly_added`。

### 12.3 Resume 优化

- existing ID 检查必须在 embedding 前完成。
- 每个 collection 按 `existing_id_check_batch_size` 批量查询，默认 1000。
- 不得继续按 embedding request batch 大小执行一次 Chroma 查询。
- 失败且未写入的 ID 在下一次 `--resume` 时自然成为 missing ID。

## 13. 生命周期与取消

- 正常结束时先停止 producer，再 drain write queue，最后关闭 provider client 和 writer。
- fail-fast 时停止提交新请求，允许已验证结果完成写入，然后取消未开始请求。
- Ctrl+C 必须触发同样的有序关闭，failure manifest 应包含未完成 logical requests。
- client close、queue drain 和 writer shutdown 必须有超时，禁止进程无限等待。

## 14. 可观测性

### 14.1 结构化事件

至少记录以下事件：

- `embedding.batch.queued`
- `embedding.request.started`
- `embedding.request.completed`
- `embedding.request.retry`
- `embedding.request.throttled`
- `embedding.batch.invalid`
- `embedding.batch.failed`
- `vector_store.write.started`
- `vector_store.write.completed`
- `vector_store.write.failed`
- `vector_db.build.summary`

### 14.2 必备字段

事件应包含：

- agent、provider、model、quota_group。
- logical_request_id 和 attempt。
- input_count、estimated_tokens、vector_dimension。
- effective_inflight、group_max_inflight。
- queue_wait_ms、provider_latency_ms、total_latency_ms。
- status、HTTP/provider error category。
- retry_after_ms（存在时）。
- write_queue_depth 和 write_batch_size（写入事件）。

### 14.3 汇总指标

每个 agent 和 quota group 必须输出：

- chunks/s、requests/s、tokens/s。
- 平均和 p50/p95/p99 request latency。
- 平均 request batch size 和 batch fill ratio。
- network attempts/logical request。
- 429、5xx、timeout 和 invalid vector 比率。
- 当前/峰值 in-flight。
- write queue 峰值深度、writer chunks/s。
- already_present、newly_added、failed、final_count。

## 15. 安全与隐私

- 日志和 failure manifest 不得记录 API key 或 Authorization header。
- 为区分额度账户，可以记录不可逆 credential fingerprint，但只能使用带进程私有 salt 的短 hash。
- 原始文档正文默认不得写入失败日志；只记录 chunk ID、长度和 token 数。
- provider 返回的错误正文必须经过长度限制和敏感字段清洗。

## 16. 测试规格

### 16.1 单元测试

必须覆盖：

- `max_inflight=1` 时仍使用原生 request batch。
- agent1/agent3/agent4 共享 quota group 时，合计并发不超过 group cap。
- global cap 同时覆盖同步和异步 adapter。
- Voyage 一次调用收到完整文本列表。
- item/token 双限制正确切分 batch。
- rolling scheduler 在一个请求完成后立即补位，不等待同波次慢请求。
- RPM/TPM 预算不足时延迟调度。
- Retry-After、full jitter 和最大 attempts。
- SDK retry 被禁用，避免双重重试。
- 429 触发并发减半，成功窗口触发缓慢恢复。
- 响应依据 `data.index` 恢复输入顺序。
- 数量、维度、NaN、Infinity 和全零向量校验。
- 失败项不进入 write queue，运行状态为 `partial`。
- failure manifest 不泄露凭据和正文。
- write queue 有界并能施加反压。
- resume 会重新处理上次失败且未写入的 ID。

### 16.2 集成测试

使用本地 fake embedding server 模拟：

- 正常批量响应。
- 延迟不均匀的响应。
- 429 + Retry-After。
- 间歇性 500/timeout。
- 永久 400/401/422。
- 数量不匹配、顺序打乱、维度错误和全零向量。

使用临时 Chroma 目录验证：

- 只有有效向量被写入。
- 多 producer 下没有并发 writer。
- 中断后 resume 的最终 ID 集合完整且无重复。

### 16.3 负载测试

正式上线前使用至少 5,000 chunks/agent 的固定样本执行：

```text
request_batch_size: 1, 10, 32, 64, 128
quota_group max_inflight: 1, 2, 4, 8
write_batch_size: 10, 100, 500
```

测试必须先分别运行各 provider，再运行四 agent 组合，以识别共享网关额度。生产调优不得使用
会污染正式 collection 的测试数据。

## 17. 验收标准

### 17.1 正确性硬指标

- provider failure 写入的零向量数量必须为 0。
- 成功写入向量的数量、ID、维度与校验结果必须 100% 一致。
- 中断后 resume 的最终 ID 集合必须等于无中断基准集合。
- failure manifest 非空时构建状态必须为 `partial` 或 `error`。
- 不得出现凭据泄露。

### 17.2 并发与限流硬指标

- 任意时刻 quota group 实际 in-flight 不得超过有效上限。
- 任意时刻进程 embedding in-flight 不得超过 global cap。
- 稳态 429 比率应低于 0.5%；出现持续 429 时必须自动降至 1 并发。
- retry network attempts 占比应低于 5%；超过时本次配置不得晋级生产默认值。
- 不得再出现固定间隔的 N 请求突发波形。

### 17.3 性能硬指标

在相同 provider、额度、数据和机器的 A/B 测试中：

- 对已确认支持 batch 且目标 batch 不小于 10 的 profile，外部 logical request 数相对逐文本
  基线至少减少 90%。
- 四 agent 总 wall-clock 至少提升 3 倍。
- 平均有效 batch fill ratio 至少为 75%。
- Chroma writer 不得成为主瓶颈：producer 因 write queue 满而等待的时间占比低于 10%。
- 不得出现 `database is locked` 或丢失 write acknowledgment。

## 18. 最终预期效果

以下分为必须实现的确定性效果和依赖 provider 配额的预测效果。

### 18.1 确定性效果

| 维度 | 当前状态 | 完成后的预期 |
| --- | --- | --- |
| 失败数据 | 重试耗尽后可能写零向量 | provider failure 零向量写入为 0 |
| Batch | 并发为 1 时退回逐文本 | 并发为 1 时仍使用原生 batch |
| 共享网关并发 | agent 独立叠加 | quota group 统一硬限制 |
| Retry | SDK retry 与应用 retry 叠加 | 单一、可观测、遵循 Retry-After 的 retry |
| 调度 | 波次 gather + 固定 sleep | 滚动窗口 + RPM/TPM 平滑调度 |
| 写入 | producer 每小批直接写 | 有界队列 + 单 writer + 100 条合并写 |
| Resume | 零向量 ID 被误判为完成 | 失败 ID 保持 missing，可自动补齐 |
| 运行结果 | 失败可能仍报告 ok | ok/partial/error 与实际数据一致 |

### 18.2 性能预测

基于原生 batch 和保守共享并发，预期：

- ZenMux/OpenAI-compatible 请求数减少 90%-97%。
- Voyage 请求数减少 98% 以上。
- Chroma `add` 调用数相对 10 条写入减少约 90%。
- 四 agent 总构建吞吐获得 3-6 倍保守提升。
- 历史约 68 小时的完整构建，在 provider 配额允许且数据规模相同时，预计降至 12-24 小时。
- 若 myrimate 网关确认支持 batch 64、共享并发 4 且无明显 429，扩展目标为 6-10 倍，
  完整构建预计 7-12 小时。

这些时间是容量预测，不是无条件 SLA。最终 SLA 必须以新 provider 路由下的组合负载测试结果为准。
如果 myrimate 实际额度低于初始假设，系统仍必须优先保证正确性和限流合规，只降低吞吐，不能
通过写零向量或无限重试换取表面完成率。

### 18.3 可选的进一步效果

agent1 的 `text-embedding-3-large` 可以在离线检索质量评估通过后，将维度从 3072 降至
1024。该选项预计减少约 66.7% 的向量响应体、内存和 Chroma 存储，但它属于模型质量/存储
优化，不是本并发规格的验收前提。

## 19. 分阶段交付

### 阶段 1：正确性与可观测性

- 移除失败零向量写入。
- 建立 Valid vector 校验和 failure manifest。
- 禁用双重 retry。
- 增加请求、限流和写入指标。

退出条件：正确性硬指标全部通过。

### 阶段 2：原生 batch

- 并发为 1 时启用 ZenMux/OpenAI-compatible batch。
- Voyage 使用原生 batch。
- 解耦 request batch 和 write batch。

退出条件：已启用 batch 的 profile，其 logical request 数减少至少 90%，且无数据正确性回归。

### 阶段 3：共享调度器

- 引入 global/quota-group 限流。
- 使用滚动窗口、公平队列和自适应并发。
- 删除固定 sleep 主限流路径。

退出条件：组合负载测试中并发和 429 指标达标。

### 阶段 4：写入流水线与生产调优

- 引入有界 write queue 和单 writer 合并写。
- 批量 existing ID 检查。
- 完成 batch/concurrency 参数矩阵测试。

退出条件：总 wall-clock 至少提升 3 倍，writer 等待占比低于 10%。

## 20. 上线前待确认事项

1. myrimate 网关的额度归属：按账户、key、endpoint 还是上游模型计算。
2. myrimate 对 agent3 模型的最大 items、最大 tokens 和响应体限制。
3. agent4 经 myrimate 调用时是否支持列表 `input`，以及是否与 ZenMux 请求共享配额。
4. Voyage 当前组织和项目实际 RPM/TPM，而不是只采用公开基础额度。
5. 生产机器允许的峰值内存和期望 write queue 大小。

以上事项未确认时，允许采用本文保守初始值上线小流量构建，但不得提升到 myrimate 共享并发 4
以上，也不得将预测的 7-12 小时作为 SLA。

## 21. 参考资料

- [ZenMux Embeddings](https://zenmux.ai/docs/guide/advanced/embeddings.html)
- [ZenMux Create an Embedding API](https://zenmux.ai/docs/api/openai/create-embeddings.html)
- [Voyage AI Rate Limits](https://docs.voyageai.com/docs/rate-limits)
- [Alibaba Cloud Model Studio Rate Limiting](https://www.alibabacloud.com/help/en/model-studio/rate-limit)
