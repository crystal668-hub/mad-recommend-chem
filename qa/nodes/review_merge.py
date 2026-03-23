from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from prompts.qa_prompts import REVIEW_MERGE_SYSTEM_PROMPT, build_review_merge_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.peer_review_errors import PeerReviewExecutionError
from qa.review_utils import build_evidence_lookup, traceable_evidence_items
from qa.retrieval_state import ClaimRecord, ClaimRevisionRecord, ConflictEdge, EvidenceLedger, ReviewFlag


class ReviewMergeNode:
    reviewer_type = "ReviewMergeNode"

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        claim: ClaimRecord,
        evidence_ledger: EvidenceLedger,
        *,
        active_flags: Sequence[ReviewFlag],
        active_conflict_edges: Sequence[ConflictEdge],
        revision_records: Sequence[ClaimRevisionRecord],
        use_llm: bool = True,
    ) -> Tuple[str, str]:
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        supporting_items = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        status, rationale = self._adjudicate_with_llm(
            claim=claim,
            active_flags=active_flags,
            active_conflict_edges=active_conflict_edges,
            revision_records=revision_records,
            use_llm=use_llm,
        )
        self._validate_adjudication(
            claim=claim,
            supporting_items=supporting_items,
            active_flags=active_flags,
            active_conflict_edges=active_conflict_edges,
            status=status,
            rationale=rationale,
        )
        return status, rationale

    __call__ = run

    def _validate_adjudication(
        self,
        *,
        claim: ClaimRecord,
        supporting_items: Sequence[object],
        active_flags: Sequence[ReviewFlag],
        active_conflict_edges: Sequence[ConflictEdge],
        status: str,
        rationale: str,
    ) -> None:
        if not rationale.strip():
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge returned an empty rationale.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            )

        has_warning = any(flag.severity == "warning" for flag in active_flags)
        has_critical_fabrication = any(
            flag.flag_type == "Fabricated_Citation" and flag.severity == "critical"
            for flag in active_flags
        )
        has_true_conflict = any(edge.conflict_type == "true_conflict" for edge in active_conflict_edges)
        has_any_conflict = bool(active_conflict_edges)

        if has_critical_fabrication and status != "rejected":
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge cannot accept or contest a claim with fabricated citations.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content={"status": status, "rationale": rationale},
            )
        if not supporting_items and status != "rejected":
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge cannot accept or contest an untraceable claim.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content={"status": status, "rationale": rationale},
            )
        if (has_warning or has_any_conflict) and status == "accepted":
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge cannot accept a claim while warning-level issues or conflicts remain active.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content={"status": status, "rationale": rationale},
            )
        if has_true_conflict and status not in {"contested", "rejected"}:
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge returned an unsupported status for an unresolved true conflict.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content={"status": status, "rationale": rationale},
            )

    def _adjudicate_with_llm(
        self,
        *,
        claim: ClaimRecord,
        active_flags: Sequence[ReviewFlag],
        active_conflict_edges: Sequence[ConflictEdge],
        revision_records: Sequence[ClaimRevisionRecord],
        use_llm: bool,
    ) -> Tuple[str, str]:
        if not use_llm:
            raise PeerReviewExecutionError(
                stage="review_merge_policy",
                message="Review merge requires LLM adjudication.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            )
        if self.llm is None:
            raise PeerReviewExecutionError(
                stage="review_merge_startup",
                message="Review merge LLM is unavailable.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            )
        messages = [
            {"role": "system", "content": REVIEW_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_review_merge_user_prompt(
                    claim=claim.model_dump(exclude_none=True),
                    active_flags=[flag.model_dump(exclude_none=True) for flag in active_flags],
                    active_conflict_edges=[edge.model_dump(exclude_none=True) for edge in active_conflict_edges],
                    revision_records=[record.model_dump(exclude_none=True) for record in revision_records],
                ),
            },
        ]
        try:
            response_content = invoke_llm(self.llm, messages)
        except Exception as exc:
            raise PeerReviewExecutionError(
                stage="review_merge_invoke",
                message="Review merge LLM call failed.",
                details={"error_type": type(exc).__name__, "error": str(exc)},
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            ) from exc
        parsed = parse_json_object(response_content)
        if not isinstance(parsed, dict):
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge returned a non-object payload.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )
        status = parsed.get("status")
        rationale = str(parsed.get("rationale") or "").strip()
        if status not in {"accepted", "contested", "rejected"} or not rationale:
            raise PeerReviewExecutionError(
                stage="review_merge_invalid_response",
                message="Review merge returned an invalid status payload.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )
        return status, rationale
