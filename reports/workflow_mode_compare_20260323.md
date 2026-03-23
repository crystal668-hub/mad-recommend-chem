# Workflow Mode Comparison Report

Date: 2026-03-23

Question used for all runs:

`Does Pt/C improve HER activity in 1 M KOH?`

## Scope

This report compares `workflow-mode=ledger` and `workflow-mode=react_reviewed` using the same real ChemQA question and real CLI execution. It separates three categories of failure:

1. Environment-induced failure caused by sandbox network restrictions.
2. Practical runtime failure under a fixed 240 s budget.
3. Workflow-intrinsic failure after the chain progresses deeply enough to expose its own policy or quality limits.

Important constraint:

- The sandboxed runs are not valid evidence for prompt quality because they fail before the first semantic router call completes.
- The strongest intrinsic failure evidence currently comes from `ledger`, because an unrestricted real run progressed into claim mining and peer review.
- `react_reviewed` was verified under a real 240 s run, but was not re-run without an external timeout budget in this pass.

## Commands And Run IDs

Sandboxed controlled runs:

- `timeout 180 python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --workflow-mode ledger --artifact-dir .cache/compare_runs/ledger_run_controlled`
  - run id: `20260323_203124`
- `timeout 180 python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --workflow-mode react_reviewed --artifact-dir .cache/compare_runs/react_run_controlled`
  - run id: `20260323_203124`

Escalated controlled runs:

- `timeout 240 python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --workflow-mode ledger --artifact-dir .cache/compare_runs/ledger_run_controlled_escalated`
  - run id: `20260323_203212`
- `timeout 240 python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --workflow-mode react_reviewed --artifact-dir .cache/compare_runs/react_run_controlled_escalated`
  - run id: `20260323_204225`

Earlier unrestricted real run already present in the workspace:

- `python -m chemqa --question "Does Pt/C improve HER activity in 1 M KOH?" --workflow-mode ledger --artifact-dir .cache/compare_runs/ledger_run`
  - run id: `20260323_202032`

## Executive Summary

Under real execution, the dominant observed problem is no longer "LLM message format cannot be parsed". The dominant failures are now:

- sandbox network blockage at the router semantic stage
- retrieval and evidence-path runtime expansion
- ledger-specific quality explosion that produces too many noisy claims and then hard-fails at peer review policy

Observed outcome by mode:

| Mode | Run | Verified deepest stage | Final answer produced | Verified fail mode |
| --- | --- | --- | --- | --- |
| `ledger` | sandbox controlled | router semantic stage | no | network connection error |
| `react_reviewed` | sandbox controlled | router semantic stage | no | network connection error |
| `ledger` | escalated 240 s | document acquisition in progress after 12 candidates | no | timed out under runtime budget |
| `react_reviewed` | escalated 240 s | proposer cycle 1 search, candidate screening, 3 document acquisitions, 1 evidence read | no | timed out under runtime budget |
| `ledger` | unrestricted real run | claim mining complete, peer review dispatch | no | peer review policy fail-fast because 187 claims exceeded budget 40 |

The two strongest workflow conclusions are:

1. The prompt refactor appears to have removed the previously dominant early parse fragility from the verified paths that were actually executed.
2. `ledger` still has a deeper workflow problem: routing and evidence quality instability can create a claim explosion that correctly trips fail-fast policy before synthesis.

## Evidence By Run

### 1. Sandboxed controlled runs are environment failures, not workflow failures

Artifacts:

- [ledger failure](/home/administrator/ChemQA/.cache/compare_runs/ledger_run_controlled/router/failure.json)
- [react failure](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled/router/failure.json)

Both modes failed with the same persisted reason:

- `stage: semantic`
- `reason: semantic stage failed: Connection error.`

Conclusion:

- These runs are only evidence that the sandbox blocks required network access.
- They do not tell us anything meaningful about prompt format robustness.

### 2. `ledger` under escalated 240 s budget

Artifacts:

- [runtime manifest](/home/administrator/ChemQA/.cache/compare_runs/ledger_run_controlled_escalated/runtime_manifest.json)
- [run log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log)
- [artifact dir](/home/administrator/ChemQA/.cache/compare_runs/ledger_run_controlled_escalated)

Verified stage progression from log:

- `qa_grounding_complete question_type=causal`
- `qa_query_planning_complete query_plans=4`
- `qa_retrieval_candidates_complete paper_candidates=12`
- multiple `document_acquirer_stage_success`
- multiple successful PDF extractions
- at least one OA fetch 403

Representative log evidence:

- run starts at [run.log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log)
- grounding at [run.log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log:3)
- query planning at [run.log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log:5)
- candidate retrieval at [run.log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log:271)
- OA 403 at [run.log](/home/administrator/ChemQA/logs/runs/20260323_203212/run.log:330)

Artifact summary:

- no final answer files
- 162 provider raw JSON files
- 7 indexed paper records
- 3 fulltext PDFs extracted

Observed fail mode:

- external `timeout 240` terminated the run before synthesis or review
- the chain was still inside retrieval and acquisition work, not inside a prompt parse failure branch

Interpretation:

- For this question, `ledger` still spends substantial time expanding and grounding literature before answer synthesis becomes possible.
- The stricter prompt contracts did not block the chain early, but the workflow is still too wide for a 240 s budget.

### 3. `react_reviewed` under escalated 240 s budget

Artifacts:

- [runtime manifest](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/runtime_manifest.json)
- [run log](/home/administrator/ChemQA/logs/runs/20260323_204225/run.log)
- [candidate screening](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/proposer_cycle_1_candidate_screening.json)
- [runtime stage events](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/diagnostics/runtime_stage_events.json)

Verified stage progression:

- `react_call_start`
- cycle 1 review-lane search executed successfully
- candidate screening artifact persisted
- 12 paper candidates found
- 3 papers locked, 2 dropped
- 3 documents acquired
- 1 fulltext PDF extracted and indexed
- evidence extraction started for one locked paper

Representative evidence:

- proposer start at [run.log](/home/administrator/ChemQA/logs/runs/20260323_204225/run.log:3)
- runtime search success after 56.525 s in [runtime stage events](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/diagnostics/runtime_stage_events.json)
- screening result in [candidate screening](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/proposer_cycle_1_candidate_screening.json)

Artifact summary:

- no final answer files
- 62 provider raw JSON files
- 4 query plans
- 12 paper candidates
- 3 paper records
- 1 fulltext PDF extracted
- `llm_screening_used: true`
- no persisted execution warnings

Observed fail mode:

- external `timeout 240` terminated the run after acquisition and early evidence work
- no prompt parse failure, repair-loop failure, or reviewer-format failure was observed in the persisted artifacts

Interpretation:

- `react_reviewed` is narrower than `ledger` under the same budget, but it still did not finish one proposer cycle end-to-end within 240 s for this question.
- The important signal is that it successfully produced structured intermediate artifacts rather than crashing on malformed LLM output.

### 4. Deep intrinsic failure observed in unrestricted `ledger` run

Artifacts:

- [peer review failure](/home/administrator/ChemQA/.cache/compare_runs/ledger_run/peer_review/failure.json)
- [peer review agent run](/home/administrator/ChemQA/.cache/compare_runs/ledger_run/peer_review/agent_run.json)
- [run log](/home/administrator/ChemQA/logs/runs/20260323_202032/run.log)

This is the most important intrinsic failure found in the current workspace.

Verified chain progression:

- router classified the question as `comparison`
- retrieval completed
- `qa_document_acquisition_complete paper_records=12 indexed_fulltexts=5`
- evidence extraction completed with `evidence_items=330`
- claim mining completed with `claims=187`
- peer review dispatch started
- workflow failed fast before reviewer execution

Representative log evidence:

- `qa_grounding_complete question_type=comparison` in [run.log](/home/administrator/ChemQA/logs/runs/20260323_202032/run.log:3)
- `qa_document_acquisition_complete paper_records=12 indexed_fulltexts=5` in [run.log](/home/administrator/ChemQA/logs/runs/20260323_202032/run.log:117)
- `qa_claim_mining_complete claims=187` in [run.log](/home/administrator/ChemQA/logs/runs/20260323_202032/run.log:120)
- `qa_peer_review_dispatch claims=187` in [run.log](/home/administrator/ChemQA/logs/runs/20260323_202032/run.log:121)

Persisted failure:

- `error: peer_review_execution_failed`
- `stage: peer_review_policy`
- `reason: Peer review cannot proceed under fail-fast policy.`
- disable reason: `claim volume exceeded budget (187 > 40)`

Run statistics from failure artifact:

- `claim_count: 187`
- `evidence_count: 330`
- `abstract_only_ratio: 0.763`
- `fulltext_section_count: 6`

Critical quality signal from the persisted claim set:

- The generated claims include obviously low-value or nonsensical comparison entities such as:
  - `1 M KOH vs H2`
  - `Additionally vs Levich`
  - `As vs Fig`
  - `Chemical vs Reagent`

These are visible in [peer review agent run](/home/administrator/ChemQA/.cache/compare_runs/ledger_run/peer_review/agent_run.json).

Interpretation:

- This is not a prompt parse failure.
- This is a workflow quality failure caused by a bad semantic framing plus low-precision evidence-to-claim conversion.
- The fail-fast peer review policy behaved correctly by refusing to pass a 187-claim noisy ledger downstream.

## Comparison Against The Prompt-Format Refactor Goal

The original motivation for the refactor was that agents frequently returned messages that could not be parsed, causing workflow failure.

What was verified in these runs:

- I did not find a persisted failure caused by malformed LLM output in the tested real runs.
- `react_reviewed` successfully persisted structured screening output in [proposer_cycle_1_candidate_screening.json](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/proposer_cycle_1_candidate_screening.json), which is evidence that at least one LLM-controlled stage respected the new contract.
- `ledger` successfully passed router and query planning in the controlled escalated run and progressed all the way to peer review policy in the unrestricted run, again without a verified parse-format failure artifact.

What this means:

- The prompt modularization and stricter output examples likely improved early-stage robustness.
- The current bottlenecks have shifted from "cannot parse model output" to "the workflow does too much, too broadly, and sometimes on the wrong semantic framing."

This is an inference from the tested runs, not a formal proof that all parse failures are gone.

## Root Cause Analysis

### A. Router stability is still a first-class problem

The exact same question was observed as:

- `comparison` in unrestricted `ledger` run
- `causal` in controlled escalated `ledger` run
- `causal` in `react_reviewed` task spec

This instability matters because the `comparison` framing widened retrieval and produced low-quality claim entities.

### B. `ledger` has evidence and claim volume explosion

Verified numbers from unrestricted `ledger` run:

- 12 paper records
- 330 evidence items
- 187 mined claims

With peer review budget configured at 40 claims, this guarantees failure unless claim pruning or quality filtering happens earlier.

### C. `react_reviewed` is narrower, but still slow for this question

Under 240 s it reached:

- search
- screening
- three acquisitions
- one evidence read

but not:

- candidate submission
- reviewer calls
- acceptance decision
- final answer

So the mode is structurally more targeted than `ledger`, but still not fast enough under the tested budget.

### D. External document access remains noisy but not fatal by itself

Observed document-level issues:

- `ledger`: Chinese Chemical Society OA fetch 403
- `react_reviewed`: MDPI OA fetch 403

These did not directly crash the workflow because the system continued after them. They are secondary friction, not the dominant root cause here.

## Conclusions

1. In the verified real runs, I did not observe the previously dominant "LLM output cannot be parsed" failure mode.
2. `react_reviewed` now survives deep enough to produce structured intermediate artifacts, but it still did not finish within 240 s for this question.
3. `ledger` has a deeper workflow-quality failure: route instability plus noisy evidence/claim mining can generate so many low-value claims that peer review is blocked by policy before synthesis.
4. The fail-fast policy is doing the right thing once the ledger becomes unreviewable.

## Recommended Next Fixes

1. Stabilize router semantics for this question family.
   - Add a stricter validation rule that implicit comparison intent cannot dominate unless explicit comparison targets are present.

2. Add pre-peer-review claim pruning in `ledger`.
   - Reject obviously non-answer-bearing claims before peer review budget accounting.
   - Examples to target: figure-reference claims, publisher-site claims, malformed comparator entities.

3. Add stronger answer-bearing relevance filters before evidence extraction.
   - Both modes are still spending too much time on weakly relevant literature.

4. Persist a unified stage summary for `ledger`, similar to `react_reviewed`.
   - `react_reviewed` diagnostics made this analysis much easier.

5. Re-test after the above changes with two budgets:
   - strict budget: 240 s
   - completion budget: no external timeout or a much larger timeout

## Key Files

- [sandbox ledger failure](/home/administrator/ChemQA/.cache/compare_runs/ledger_run_controlled/router/failure.json)
- [sandbox react failure](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled/router/failure.json)
- [ledger 240 s manifest](/home/administrator/ChemQA/.cache/compare_runs/ledger_run_controlled_escalated/runtime_manifest.json)
- [react 240 s manifest](/home/administrator/ChemQA/.cache/compare_runs/react_run_controlled_escalated/runtime_manifest.json)
- [ledger unrestricted peer review failure](/home/administrator/ChemQA/.cache/compare_runs/ledger_run/peer_review/failure.json)
- [ledger unrestricted peer review input](/home/administrator/ChemQA/.cache/compare_runs/ledger_run/peer_review/agent_run.json)
