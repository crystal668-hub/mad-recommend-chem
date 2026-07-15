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

The approved sample was reduced from 5,000 to 1,000 chunks per agent. A four-agent
gate could not be completed because agent1's configured myrimate credential returned
HTTP 401. The test did not repeat known-invalid agent1 requests at load volume.

## Smoke Results

| Agent | Provider/model | Request batch | Result | Finding |
| --- | --- | ---: | --- | --- |
| agent1 | zenmux / `openai/text-embedding-3-large` | 32 | Blocked | myrimate returned HTTP 401 `Invalid token` |
| agent2 | Voyage / `voyage-3-large` | 128 | Pass | 4/4 vectors persisted |
| agent3 | zenmux / `google/gemini-embedding-2` | 32 | Fail | list input returned only one vector for four texts |
| agent3 | zenmux / `google/gemini-embedding-2` | 1 | Pass | 4/4 vectors persisted |
| agent4 | Aliyun / `text-embedding-v4` | 1 | Pass | 4/4 vectors persisted |

The agent3 result changed its safe default request batch from 32 to 1 in commit
`f3b08c9`. An explicit configuration override can be reconsidered only after the
gateway returns one vector per list input.

## Independent 1,000-Chunk Runs

| Metric | agent2 | agent3 | agent4 |
| --- | ---: | ---: | ---: |
| Status / persisted | ok / 1,000 | ok / 1,000 | ok / 1,000 |
| Failed chunks | 0 | 0 | 0 |
| Logical requests | 8 | 1,000 | 1,000 |
| Request batch size | 128 | 1 | 1 |
| Batch fill ratio | 97.66% | 100% | 100% |
| Request reduction vs scalar | 99.20% | 0% | 0% |
| Embedding throughput | 84.20 chunks/s | 2.24 chunks/s | 15.65 chunks/s |
| Network request p95 | 3,566 ms | 2,480 ms | 282 ms |
| Logical request p95 | 6,547 ms | 3,325 ms | 454 ms |
| Wall clock | 12.06 s | 445.87 s | 64.02 s |
| Actual peak in-flight | 2 | 4 | 4 |
| Attempts / retries / throttles | 8 / 0 / 0 | 1,000 / 0 / 0 | 1,000 / 0 / 0 |
| Chroma write calls / avg batch | 12 / 83.33 | 347 / 2.88 | 10 / 100 |

All independent collections contained exactly 1,000 documents. No invalid or zero
vector reached Chroma, and no failure manifest was produced for the successful runs.

## Combined Provider Run

The available-provider combination ran agent2, agent3, and agent4 concurrently with
1,000 chunks each.

| Metric | Voyage quota group | Shared myrimate quota group |
| --- | ---: | ---: |
| Network attempts | 8 | 2,000 |
| Successes / retries / throttles | 8 / 0 / 0 | 2,000 / 0 / 0 |
| Actual peak in-flight | 2 | 4 |
| Configured max in-flight | 4 | 4 |
| Network request p95 | 12,067 ms | 2,253 ms |
| Effective in-flight at finish | 2 | 4 |

Combined wall clock was 492.48 seconds. Per-agent throughput was 31.59 chunks/s for
agent2, 2.03 chunks/s for agent3, and 3.75 chunks/s for agent4. The shared myrimate
group never exceeded four in-flight attempts and recorded no 429s or retries.

Chroma validation after shutdown found exactly 1,000 unique IDs in each of the three
collections. There were no duplicate IDs, missing write acknowledgements, or
`database is locked` errors. Peak write queue depth was 2 of 8.

## Gate Assessment

| Gate | Result | Evidence |
| --- | --- | --- |
| Valid vectors only | Pass | 3,000/3,000 combined chunks persisted; zero failures |
| Group in-flight cap | Pass | myrimate peak 4/4; Voyage peak 2/4 |
| Global in-flight cap | Pass | group peaks sum to at most 6, below global cap 8 |
| Steady 429 rate below 0.5% | Pass for available providers | 0/2,008 combined attempts |
| Retry attempts below 5% | Pass for available providers | 0/2,008 combined attempts |
| Batch fill at least 75% | Pass | Voyage 97.66%; singleton profiles 100% |
| Batch-enabled request reduction at least 90% | Pass | Voyage reduced requests by 99.2% |
| Writer queue not saturated | Pass by queue-depth proxy | peak 2/8; exact wait-time ratio is not instrumented |
| Four-agent combined gate | Blocked | agent1 myrimate credential returns HTTP 401 |
| Three-times wall-clock improvement | Not assessed | no same-provider scalar A/B baseline was run |

## Decisions And Follow-Up

1. Keep `google/gemini-embedding-2` at request batch 1 until myrimate list-input
   behavior changes.
2. Keep myrimate max in-flight at 4. The combined run completed 2,000 attempts with
   zero throttles, but the quota source is still an inferred shared endpoint rather
   than a documented account limit.
3. Keep Voyage request batch 128 and in-flight initial/max 2/4. The request reduction
   target passed without retries or throttling.
4. Do not promote a complete four-agent production gate until agent1 has a valid
   myrimate credential and passes smoke before the 1,000-chunk rerun.
5. Consider a longer or per-collection write flush interval for slow singleton
   providers. Agent3 averaged only 2.88 chunks per Chroma call, although queue depth
   confirms this did not become the wall-clock bottleneck in this run.

