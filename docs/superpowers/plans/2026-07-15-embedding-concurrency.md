# Embedding Concurrency Pipeline Implementation Plan

> Source specification:
> `docs/superpowers/specs/2026-07-15-embedding-concurrency-design.md`

## Objective

Replace per-agent wave concurrency and direct Chroma writes with a shared, quota-aware
embedding runtime. The implementation must batch requests even at concurrency one,
apply one retry layer, validate every vector before persistence, retain failed IDs for
resume, and feed a bounded single-writer Chroma pipeline.

## Delivery Strategy

Implementation is split into independently testable layers:

1. Runtime configuration, batching, validation, and failure types.
2. Shared quota scheduler and centralized retry.
3. Multi-provider native batch adapters.
4. Bounded Chroma write pipeline and failure manifest.
5. Batch builder, configuration, CLI, and documentation integration.
6. Regression and local load verification.

Live provider benchmarking is an explicit deployment gate. Automated tests use fake
clients and a local fake provider so they do not consume external API quota.

## Task 1: Runtime Types, Configuration, Batching, And Validation

**Files:**

- Create: `database/embedding_runtime.py`
- Create: `test/test_embedding_runtime.py`

### Step 1: Add failing tests

Cover:

- Runtime defaults and YAML override parsing.
- Legacy request batch and concurrency overrides.
- Batch creation by item and token limits.
- Empty text rejection.
- Response count, dimension, finite-value, and zero-vector validation.
- OpenAI response ordering through `data.index`.
- Failure records exclude document text and credentials.

Run:

```powershell
python -m unittest test.test_embedding_runtime -v
```

Expected: fail because the runtime module does not exist.

### Step 2: Implement runtime data types

Add:

- `RetryPolicy`
- `QuotaPolicy`
- `EmbeddingRuntimeSettings`
- `EmbeddingBatch`
- `EmbeddingFailure`
- `EmbeddingValidationError`
- `build_text_batches()`
- `validate_embeddings()`

Configuration parsing must be defensive and clamp all sizes/concurrency to positive
values. Unknown quota groups are configuration errors.

### Step 3: Run tests

Run the focused test command and confirm it passes.

## Task 2: Shared Quota Scheduler And Central Retry

**Files:**

- Modify: `database/embedding_runtime.py`
- Modify: `test/test_embedding_runtime.py`

### Step 1: Add failing scheduler tests

Cover:

- Global max in-flight across quota groups.
- Shared group max across different agents.
- Rolling refill after the first operation completes.
- RPM and TPM window blocking with an injected clock.
- `Retry-After` precedence.
- Full-jitter exponential fallback.
- Retryable versus permanent errors.
- Adaptive halving on throttle and bounded recovery after success.
- Every network attempt consumes quota.

### Step 2: Implement scheduler

Add one process-run `EmbeddingQuotaScheduler` backed by asyncio primitives. It must:

- Acquire both global and quota-group capacity.
- Maintain request/token rolling windows.
- Execute exactly one network attempt per acquired slot.
- Apply centralized retry and emit attempt statistics.
- Expose immutable summary snapshots for logs and build results.

The scheduler must not depend on Chroma or provider SDK classes.

### Step 3: Run focused tests

Run:

```powershell
python -m unittest test.test_embedding_runtime -v
```

## Task 3: Multi-Provider Native Batch Adapters

**Files:**

- Modify: `database/embedder.py`
- Modify: `test/test_embedder_provider_routing.py`
- Modify: `test/test_embedding.py`

### Step 1: Add failing adapter tests

Cover:

- ZenMux/OpenAI-compatible batch at max in-flight one.
- OpenAI result reordering by response index.
- Voyage receives the complete text list in one SDK call.
- Aliyun/OpenAI-compatible transport can batch when explicitly enabled.
- OpenAI clients are created with SDK retries disabled.
- Provider errors raise and never return zero vectors.
- Blank document input raises before a provider call.
- Query embedding behavior remains compatible.

### Step 2: Implement adapter contract

Add:

```python
async def embed_documents_batch(
    self,
    texts: List[str],
    agent_name: str,
) -> List[List[float]]:
    ...
```

Transport selection must use `embedding_transport`, not provider-name membership.
OpenAI-compatible calls use `AsyncOpenAI(max_retries=0)`. Voyage uses its native list
input through an executor without introducing a second retry layer.

Keep existing single-query APIs for RAG compatibility, but route document batch builds
through the new method.

### Step 3: Run adapter tests

Run:

```powershell
python -m unittest test.test_embedder_provider_routing test.test_embedding_runtime -v
```

## Task 4: Bounded Chroma Writer And Failure Manifest

**Files:**

- Modify: `database/embedding_runtime.py`
- Create: `test/test_embedding_write_pipeline.py`

### Step 1: Add failing writer tests

Cover:

- A bounded queue applies producer backpressure.
- A single writer owns every `add_documents` call.
- Items coalesce to the configured write batch size.
- A flush timeout writes a partial batch.
- `newly_added` changes only after write acknowledgement.
- Write failures become sanitized failure records.
- Normal shutdown drains all valid items.

### Step 2: Implement writer

Add `EmbeddingWritePipeline` with a dedicated single-thread executor. Queue items retain
collection ownership so buffers never mix different agent collections. The writer
flushes on size, timeout, or shutdown and exposes per-agent counters.

Add an atomic JSONL failure-manifest writer. Provider and vector-store failures share
the same sanitized record contract.

### Step 3: Run writer tests

Run:

```powershell
python -m unittest test.test_embedding_write_pipeline -v
```

## Task 5: Batch Builder Integration

**Files:**

- Modify: `build_vector_db_batch.py`
- Create: `test/test_build_vector_db_batch_concurrency.py`
- Modify: `test/test_batch_resume.py`

### Step 1: Add failing integration tests

Use fake embedders and vector stores to cover:

- Existing IDs are queried in large batches before embedding.
- Request batching remains enabled with concurrency one.
- Agent1/agent3/agent4 share the same quota-group cap.
- A slow request does not block refill after another completes.
- Invalid/provider-failed batches never call Chroma.
- Partial results include failed count and manifest path.
- Resume retries failed IDs because they were not persisted.
- Fail-fast stops new scheduling but drains validated writes.

### Step 2: Replace per-agent thread workers

Build one asyncio orchestration run containing:

- One configured embedder per agent.
- Per-agent request batch iterators.
- One shared `EmbeddingQuotaScheduler`.
- One bounded `EmbeddingWritePipeline`.
- Rolling, bounded per-agent request windows.
- Final `ok`/`partial`/`error` summaries.

Remove the wave `asyncio.gather()` loop, direct producer writes, zero-vector fallback,
and fixed-sleep rate limiting from the batch builder.

### Step 3: Run integration tests

Run:

```powershell
python -m unittest test.test_build_vector_db_batch_concurrency test.test_batch_resume -v
```

## Task 6: Configuration And CLI Integration

**Files:**

- Modify: `config/config.yaml`
- Modify: `build_vector_db_batch.py`
- Modify: `README.md`
- Modify: `Developer.md`
- Modify: `test/test_build_vector_db_batch_concurrency.py`

### Step 1: Add configuration tests

Cover:

- Every configured agent has transport, quota group, and request batch settings.
- myrimate is shared by agent1, agent3, and agent4.
- Voyage has a separate organization quota group.
- Legacy CLI flags map to the new settings with one deprecation warning.
- New request/write/global override flags take precedence.

### Step 2: Add production defaults

Use the specification's conservative starting values:

- myrimate initial/max in-flight: 2/4.
- Voyage initial/max in-flight: 2/4, with dashboard-overridable RPM/TPM.
- ZenMux request batch: 32.
- Voyage request batch: 128.
- Agent4 request batch: 1 until gateway list-input support is confirmed.
- Write batch: 100; queue: 8 batches.

Document that unknown quota values are not unlimited and that production defaults must
be promoted only after the load gate passes.

### Step 3: Run CLI/config tests

Run the focused builder tests.

## Task 7: Regression And Acceptance Verification

### Step 1: Run all relevant tests

```powershell
python -m unittest \
  test.test_embedding_runtime \
  test.test_embedding_write_pipeline \
  test.test_embedder_provider_routing \
  test.test_build_vector_db_batch_concurrency \
  test.test_build_vector_db_batch_literature_types \
  test.test_batch_resume \
  test.test_vector_store_add_documents_ids \
  test.test_request_limiter -v
```

### Step 2: Run repository unit-test discovery

```powershell
python -m unittest discover -s test -p "test_*.py"
```

Tests that are explicitly live-provider smoke scripts must remain opt-in and must not be
called by automated discovery.

### Step 3: Local deterministic load test

Use a fake provider with variable latency and injected 429s. Verify:

- Group/global in-flight caps are never exceeded.
- No zero or invalid vector is written.
- Request batch fill ratio is at least 75%.
- Request count falls by at least 90% for batch-enabled profiles.
- Writer backpressure time remains below 10%.
- Adaptive concurrency recovers without a request burst.

### Step 4: Deployment load gate

Run the specification's 5,000 chunks/agent matrix against real providers in temporary
collections. Record the quota source, batch/concurrency selection, throughput, p95,
retry rate, and 429 rate before promoting defaults.

## Commit Sequence

1. `docs: plan embedding concurrency implementation`
2. `feat: add quota-aware embedding runtime`
3. `feat: pipeline batch embeddings into chroma`
4. `docs: document embedding runtime controls`

Every commit follows `AGENTS.md`: tests first, then commit. No implementation commit may
contain known failing tests.
