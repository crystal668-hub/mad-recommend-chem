Artifact contract:

- The coordinator must emit `chemqa_review_protocol.json`.
- The protocol JSON must be sufficient for a post-run collector to rebuild:
  - `candidate_submission.json`
  - `acceptance_decision.json`
  - `submission_trace.json`
  - `submission_cycles.json`
  - `proposer_trajectory.json`
  - `reviewer_trajectories.json`
  - `review_statuses.json`
  - `final_review_items.json`
  - `qa_result.json`

The reconstructed `qa_result.json` is expected to remain externally compatible
with the current `react_reviewed` artifact surface.
