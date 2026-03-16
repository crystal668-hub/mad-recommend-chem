from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from prompts.qa_prompts import REVIEW_MERGE_SYSTEM_PROMPT, build_review_merge_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
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

        if any(flag.flag_type == "Fabricated_Citation" and flag.severity == "critical" for flag in active_flags):
            return "rejected", "Rejected because at least one cited evidence reference is not traceable in the ledger."

        if any(flag.flag_type == "Unsupported" for flag in active_flags) and not supporting_items:
            return "rejected", "Rejected because no valid supporting snippet remains after revision."

        true_conflicts = [edge for edge in active_conflict_edges if edge.conflict_type == "true_conflict"]
        if true_conflicts:
            weak_support = any(flag.flag_type in {"Unsupported", "Weak_Evidence"} for flag in active_flags)
            if weak_support or not supporting_items:
                return "rejected", "Rejected because a true conflict remains and the supporting chain is not strong enough to survive review."
            return "contested", "Contested because materially overlapping claims still point in opposite directions."

        if any(edge.conflict_type == "condition_divergence" for edge in active_conflict_edges):
            return "contested", "Contested because directionally opposed claims still differ across condition scopes."

        warnings = [flag for flag in active_flags if flag.severity == "warning"]
        if warnings:
            if use_llm and self._should_use_llm(active_flags=active_flags, active_conflict_edges=active_conflict_edges):
                adjudicated = self._adjudicate_with_llm(
                    claim=claim,
                    active_flags=active_flags,
                    active_conflict_edges=active_conflict_edges,
                    revision_records=revision_records,
                )
                if adjudicated is not None:
                    return adjudicated
            return "contested", "Contested because warning-level review issues remain after revision."

        if not supporting_items:
            return "rejected", "Rejected because the claim is not traceable to a supporting snippet."

        return "accepted", "Accepted because the evidence is traceable, no warning or critical issues remain, and no unresolved true conflict was found."

    __call__ = run

    def _should_use_llm(
        self,
        *,
        active_flags: Sequence[ReviewFlag],
        active_conflict_edges: Sequence[ConflictEdge],
    ) -> bool:
        if self.llm is None:
            return False
        warning_types = {flag.flag_type for flag in active_flags if flag.severity == "warning"}
        if len(warning_types) >= 2:
            return True
        if "Condition_Divergence" in warning_types and "Weak_Evidence" in warning_types:
            return True
        if active_conflict_edges and any(edge.conflict_type == "condition_divergence" for edge in active_conflict_edges):
            return True
        return False

    def _adjudicate_with_llm(
        self,
        *,
        claim: ClaimRecord,
        active_flags: Sequence[ReviewFlag],
        active_conflict_edges: Sequence[ConflictEdge],
        revision_records: Sequence[ClaimRevisionRecord],
    ) -> Optional[Tuple[str, str]]:
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
        if not (hasattr(self.llm, "invoke") or callable(self.llm)):
            return None
        try:
            parsed = parse_json_object(invoke_llm(self.llm, messages))
        except Exception:
            return None
        if parsed is None:
            return None
        status = parsed.get("status")
        rationale = str(parsed.get("rationale") or "").strip()
        if status not in {"accepted", "contested", "rejected"} or not rationale:
            return None
        return status, rationale
