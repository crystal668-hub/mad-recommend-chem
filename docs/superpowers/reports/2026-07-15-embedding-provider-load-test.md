# Embedding Provider Load Test Report

## Scope

- Date: 2026-07-15
- Sample: first 1,000 chunks from the O5H corpus per agent
- Chunking: size 256, overlap 50
- Storage: isolated Chroma directories under
  `outputs/embedding_load_tests/20260715/`
- Runtime defaults: global in-flight 8, quota-group initial/max in-flight 2/4,
  write batch 100, queue capacity 8
- Provider retries: disabled in SDKs; application retry policy only

A later shared-provider concurrency sweep used the first 300 O5H chunks per agent
for agent1, agent3, and agent4. This smaller sample was approved to limit provider
traffic while testing the shared myrimate ceiling.

The approved sample was reduced from 5,000 to 1,000 chunks per agent. All four
independent runs and the four-agent combined gate completed. An initial agent1 HTTP
401 was traced to a stale `OPENAI_API_KEY` inherited by the test runner taking
precedence over the current repository `.env`; it was not a provider failure. The
successful retry explicitly loaded the current `.env` into the child process without
logging credential values.

## Smoke Results

| Agent | Provider/model | Request batch | Result | Finding |
| --- | --- | ---: | --- | --- |
| agent1 | zenmux / `openai/text-embedding-3-large` | 32 | Pass | 4/4 persisted after correcting test-runner environment precedence |
| agent2 | Voyage / `voyage-3-large` | 128 | Pass | 4/4 vectors persisted |
| agent3 | zenmux / `google/gemini-embedding-2` | 32 | Fail | list input returned only one vector for four texts |
| agent3 | zenmux / `google/gemini-embedding-2` | 1 | Pass | 4/4 vectors persisted |
| agent4 | Aliyun / `text-embedding-v4` | 1 | Pass | 4/4 vectors persisted |

The agent3 result changed its safe default request batch from 32 to 1 in commit
`f3b08c9`. An explicit configuration override can be reconsidered only after the
gateway returns one vector per list input.

## Independent 1,000-Chunk Runs

| Metric | agent1 | agent2 | agent3 | agent4 |
| --- | ---: | ---: | ---: | ---: |
| Status / persisted | ok / 1,000 | ok / 1,000 | ok / 1,000 | ok / 1,000 |
| Failed chunks | 0 | 0 | 0 | 0 |
| Logical requests | 32 | 8 | 1,000 | 1,000 |
| Request batch size | 32 | 128 | 1 | 1 |
| Batch fill ratio | 97.66% | 97.66% | 100% | 100% |
| Request reduction vs scalar | 96.80% | 99.20% | 0% | 0% |
| Embedding throughput | 7.14 chunks/s | 84.20 chunks/s | 2.24 chunks/s | 15.65 chunks/s |
| Network request p95 | 16,538 ms | 3,566 ms | 2,480 ms | 282 ms |
| Logical request p95 | 26,315 ms | 6,547 ms | 3,325 ms | 454 ms |
| Wall clock | 140.14 s | 12.06 s | 445.87 s | 64.02 s |
| Actual peak in-flight | 2 | 2 | 4 | 4 |
| Attempts / retries / throttles | 32 / 0 / 0 | 8 / 0 / 0 | 1,000 / 0 / 0 | 1,000 / 0 / 0 |
| Chroma write calls / avg batch | 31 / 32.26 | 12 / 83.33 | 347 / 2.88 | 10 / 100 |

All independent collections contained exactly 1,000 documents. No invalid or zero
vector reached Chroma, and no failure manifest was produced for the successful runs.

## Agent3 Concurrency Sweep

Agent3 was tested again with the same first 1,000 O5H chunks, request batch 1, and an
isolated Chroma collection for every run. Concurrency 16 and 24 were repeated to
separate a one-run peak from reproducible throughput.

| Concurrency | Throughput (chunks/s) | Network p95 | Network p99 | Retries | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 2.243 | 2,480 ms | 4,710 ms | 0 | Baseline |
| 6 | 3.344 | 2,908 ms | 4,720 ms | 0 | Pass |
| 8 | 4.761 | 2,515 ms | 3,217 ms | 0 | Pass |
| 12 | 5.682 | 4,303 ms | 13,337 ms | 0 | Tail latency increased |
| 16 | 9.254 | 2,606 ms | 3,413 ms | 0 | Pass |
| 16 repeat | 8.539 | 2,639 ms | 11,340 ms | 0 | Pass, reproducible throughput |
| 20 | 8.009 | 5,963 ms | 12,040 ms | 0 | Throughput regression |
| 24 | 10.954 | 4,095 ms | 7,456 ms | 0 | Highest single-run peak |
| 24 repeat | 5.797 | 17,813 ms | 34,095 ms | 0 | Unstable, not promotable |
| 28 | 9.262 | 7,421 ms | 21,455 ms | 0 | Overloaded tail latency |
| 32 | 7.834 | 6,107 ms | 60,009 ms | 17 | Regressed with timeout retries |

Every run ultimately persisted 1,000/1,000 chunks without invalid vectors or 429s.
The endpoint expressed overload through latency and timeouts rather than HTTP 429.
At concurrency 32, 17 retry attempts were required and throughput was 28% below the
single-run concurrency-24 peak.

The highest observed throughput was 10.954 chunks/s at concurrency 24, but its repeat
fell to 5.797 chunks/s. Concurrency 16 is the highest stable tested setting: its two
runs delivered 9.254 and 8.539 chunks/s with no retries, averaging 8.897 chunks/s.
That is approximately four times the concurrency-4 baseline and reduces the linear
agent3-only estimate for 500,000 chunks from about 61.9 hours to about 15.6 hours.

For an agent3-only build, use `initial_inflight: 8`, `max_inflight: 16`, and a global
cap of at least 16. Agent1 and agent4 use the same myrimate quota group, so a higher
shared ceiling requires the combined evidence recorded below. The scheduler should
also reduce effective concurrency on sustained timeouts or retry-rate growth; its
current AIMD decrease is driven primarily by explicit throttling signals.

## Combined Agent1/3/4 Concurrency Sweep

Agent1, agent3, and agent4 were then tested together because all three resolve to the
same myrimate endpoint quota group. Every run used the same first 300 O5H chunks per
agent, isolated Chroma collections, agent1 request batch 32, and singleton requests
for agent3 and agent4. Each run therefore issued 610 network requests and attempted
900 vector writes.

| Shared concurrency | Wall clock | Group p95 | Group p99 | Retries / throttles | Write queue peak | agent1 | agent3 | agent4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 120.831 s | 3,342 ms | 21,179 ms | 0 / 0 | 2/8 | 7.275 chunks/s | 2.507 chunks/s | 4.141 chunks/s |
| 16 | 60.141 s | 3,072 ms | 14,633 ms | 0 / 0 | 3/8 | 18.715 chunks/s | 5.073 chunks/s | 9.573 chunks/s |
| 24 | 46.175 s | 3,625 ms | 16,000 ms | 0 / 0 | 4/8 | 16.729 chunks/s | 6.641 chunks/s | 11.243 chunks/s |
| 24 repeat | 48.558 s | 6,770 ms | 15,379 ms | 0 / 0 | 8/8 | 16.112 chunks/s | 6.308 chunks/s | 11.465 chunks/s |
| 28 | 47.873 s | 7,502 ms | 13,176 ms | 0 / 0 | 6/8 | 16.801 chunks/s | 6.401 chunks/s | 12.761 chunks/s |

Shared concurrency 24 is the highest stable tested combination. Its two runs
completed in 46.175 and 48.558 seconds with zero failures, retries, or throttles.
The repeat preserved wall-clock throughput even though p95 varied from 3.625 to
6.770 seconds. Post-run Chroma validation found exactly 300 unique IDs in each of
the three repeat collections.

Concurrency 28 is not promotable: wall time regressed relative to the first
concurrency-24 run while p95 more than doubled. This is the same latency-first
overload pattern seen in the agent3-only sweep. Concurrency 32 was deliberately not
tested because 28 had already crossed the useful-throughput knee, and the earlier
agent3-only concurrency-32 run required 17 retries with a p99 near 60 seconds.

For the shared myrimate group, use `initial_inflight: 16` and `max_inflight: 24`.
Use a process `global_max_inflight` of at least 24 for agent1/3/4 alone, or at least
28 when Voyage is also allowed its independent four requests. The initial value of
16 lets the limiter ramp toward the tested ceiling instead of beginning at the
tail-latency knee. Preserve agent1 batch 32 and agent3/agent4 batch 1. The repeat
briefly filled the 8-slot write queue, so production telemetry should retain queue
depth and enqueue-wait monitoring; the stable wall time and complete writes do not
justify a writer change from this sample alone. Fall back to shared concurrency 16
if p95 remains above 10 seconds or timeout retries appear across consecutive windows.

Across the two concurrency-24 runs, the three agents completed 900 embedding and
write operations in an average 47.367 seconds, or 18.999 aggregate vector writes per
second. If 500,000 source chunks must be written by all three agents (1.5 million
vectors), the linear embedding-plus-write estimate is about 21.9 hours. Chunking and
startup add overhead not represented by this sample, so this is a capacity estimate,
not an end-to-end service-level guarantee.

## Initial Four-Agent Combined Run

The initial combination ran all four agents concurrently with 1,000 chunks each.

| Metric | Voyage quota group | Shared myrimate quota group |
| --- | ---: | ---: |
| Network attempts | 8 | 2,032 |
| Successes / retries / throttles | 8 / 0 / 0 | 2,032 / 0 / 0 |
| Actual peak in-flight | 2 | 4 |
| Configured max in-flight | 4 | 4 |
| Network request p95 | 6,235 ms | 2,376 ms |
| Effective in-flight at finish | 2 | 4 |

Combined wall clock was 585.51 seconds. Per-agent throughput was 8.04 chunks/s for
agent1, 75.95 chunks/s for agent2, 1.71 chunks/s for agent3, and 2.76 chunks/s for
agent4. The shared myrimate group never exceeded four in-flight attempts and recorded
no 429s or retries.

Chroma validation after shutdown found exactly 1,000 unique IDs in each of the four
collections. There were no duplicate IDs, missing write acknowledgements, or
`database is locked` errors. Peak write queue depth was 3 of 8.

The execution timeline showed an agent-level fairness limitation: agent1 repeatedly
reacquired shared myrimate slots and mostly completed before the singleton agent3 and
agent4 workloads made substantial progress. The group cap remained correct, but the
scheduler is not a strict fair queue across agents.

## Gate Assessment

| Gate | Result | Evidence |
| --- | --- | --- |
| Valid vectors only | Pass | 4,000/4,000 combined chunks persisted; zero failures |
| Group in-flight cap | Pass | myrimate peak 4/4; Voyage peak 2/4 |
| Global in-flight cap | Pass | group peaks sum to at most 6, below global cap 8 |
| Steady 429 rate below 0.5% | Pass | 0/2,040 combined attempts |
| Retry attempts below 5% | Pass | 0/2,040 combined attempts |
| Batch fill at least 75% | Pass | agent1/Voyage 97.66%; singleton profiles 100% |
| Batch-enabled request reduction at least 90% | Pass | agent1 reduced 96.8%; Voyage reduced 99.2% |
| Writer queue not saturated | Pass by queue-depth proxy | peak 3/8; exact wait-time ratio is not instrumented |
| Four-agent combined gate | Pass | four collections contain 1,000 unique IDs each |
| Fair scheduling across agents | Needs improvement | agent1 was favored while sharing the myrimate group |
| Three-times wall-clock improvement | Not assessed | no same-provider scalar A/B baseline was run |

## Decisions And Follow-Up

1. Keep `google/gemini-embedding-2` at request batch 1 until myrimate list-input
   behavior changes.
2. Promote agent3-only builds to initial/max in-flight 8/16 after operational review.
   For combined agent1/3/4 builds, use shared myrimate initial/max in-flight 16/24;
   the ceiling is empirical because the provider has not documented a quota value.
3. Keep Voyage request batch 128 and in-flight initial/max 2/4. The request reduction
   target passed without retries or throttling.
4. Add an agent-aware fair queue or round-robin admission policy inside each shared
   quota group. This is a throughput-distribution improvement; the hard concurrency
   and provider-limit gates already pass.
5. Consider a longer or per-collection write flush interval for slow singleton
   providers. Agent3 averaged only 2.88 chunks per Chroma call, although queue depth
   confirms this did not become the wall-clock bottleneck in this run.
