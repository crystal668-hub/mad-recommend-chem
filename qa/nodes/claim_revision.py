from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from prompts.qa_prompts import CLAIM_REVISION_SYSTEM_PROMPT, build_claim_revision_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.peer_review_errors import PeerReviewExecutionError
from qa.review_utils import (
    build_evidence_lookup,
    common_condition_scope,
    compatible_with_scope,
    dominant_condition_scope,
    infer_claim_direction,
    normalize_condition_signature,
    speculative_text,
    traceable_evidence_items,
    valid_evidence_items,
)
from qa.retrieval_state import ClaimRecord, ClaimRevisionRecord, ConflictEdge, EvidenceLedger, ReviewFlag
from qa.state import TaskSpec


class ClaimRevisionNode:
    reviewer_type = "ClaimRevisionNode"

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        claim: ClaimRecord,
        evidence_ledger: EvidenceLedger,
        *,
        task_spec: Optional[TaskSpec] = None,
        review_flags: Sequence[ReviewFlag] = (),
        conflict_edges: Sequence[ConflictEdge] = (),
        use_llm: bool = True,
    ) -> Tuple[ClaimRecord, ClaimRevisionRecord, bool]:
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        valid_support = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        valid_oppose = valid_evidence_items(claim.opposing_evidence_ids, evidence_lookup=evidence_lookup)
        required_axes = list(task_spec.required_condition_axes or []) if task_spec is not None else []

        updated_scope = dict(claim.condition_scope)
        updated_support_ids = [item.evidence_id for item in valid_support]
        updated_oppose_ids = [item.evidence_id for item in valid_oppose]
        notes: List[str] = []

        invalid_ref_present = any(flag.flag_type == "Fabricated_Citation" for flag in review_flags)
        need_scope_repair = any(
            flag.flag_type in {"Missing_Condition", "Incomplete_Condition", "Overgeneralized"}
            for flag in review_flags
        )
        need_conflict_narrowing = any(edge.conflict_type in {"true_conflict", "condition_divergence"} for edge in conflict_edges)
        need_downgrade = any(
            flag.flag_type in {"Mechanism_Speculative", "Weak_Evidence"}
            for flag in review_flags
        )

        if invalid_ref_present:
            notes.append("Removed evidence references that cannot be traced in the ledger.")

        if (need_scope_repair or need_conflict_narrowing) and valid_support:
            common_scope = common_condition_scope(valid_support)
            dominant_scope = dominant_condition_scope(valid_support)
            for axis, value in common_scope.items():
                updated_scope.setdefault(axis, value)
            for axis in required_axes:
                if axis in updated_scope:
                    continue
                if axis in dominant_scope:
                    updated_scope[axis] = dominant_scope[axis]
            if need_conflict_narrowing and not common_scope:
                updated_scope = dominant_scope or updated_scope
            if updated_scope != claim.condition_scope:
                notes.append("Tightened claim scope to match cited evidence conditions.")

        if updated_scope:
            filtered_support = [item.evidence_id for item in valid_support if compatible_with_scope(item, updated_scope)]
            filtered_oppose = [item.evidence_id for item in valid_oppose if compatible_with_scope(item, updated_scope)]
            if filtered_support:
                updated_support_ids = filtered_support
            updated_oppose_ids = filtered_oppose
            if updated_support_ids != [item.evidence_id for item in valid_support] or updated_oppose_ids != [
                item.evidence_id for item in valid_oppose
            ]:
                notes.append("Rebalanced evidence references to the revised condition scope.")

        direction = infer_claim_direction(
            claim,
            supporting_items=valid_evidence_items(updated_support_ids, evidence_lookup=evidence_lookup),
            opposing_items=valid_evidence_items(updated_oppose_ids, evidence_lookup=evidence_lookup),
        )
        revised_support_items = valid_evidence_items(updated_support_ids, evidence_lookup=evidence_lookup)
        revised_text = claim.claim_text

        llm_revision = self._revise_with_llm(
            claim=claim,
            task_spec=task_spec,
            review_flags=review_flags,
            conflict_edges=conflict_edges,
            supporting_items=revised_support_items,
            allowed_condition_scope=updated_scope,
            use_llm=use_llm,
        )
        llm_scope = llm_revision["condition_scope"]
        filtered_support = [item.evidence_id for item in revised_support_items if compatible_with_scope(item, llm_scope)]
        filtered_oppose = [item.evidence_id for item in valid_oppose if compatible_with_scope(item, llm_scope)]
        if revised_support_items and not filtered_support:
            raise PeerReviewExecutionError(
                stage="claim_revision_invalid_response",
                message="Claim revision removed all supporting evidence from the allowed scope.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=llm_revision,
            )
        updated_scope = llm_scope
        updated_support_ids = filtered_support or updated_support_ids
        updated_oppose_ids = filtered_oppose
        revised_text = llm_revision["claim_text"]
        if revised_text != claim.claim_text:
            notes.append("LLM revision rewrote the claim conservatively within the allowed scope.")

        revision_action = "keep"
        if updated_scope != claim.condition_scope:
            revision_action = "narrow"
        elif revised_text != claim.claim_text and need_downgrade:
            revision_action = "downgrade"
        elif invalid_ref_present or updated_support_ids != list(claim.supporting_evidence_ids) or updated_oppose_ids != list(
            claim.opposing_evidence_ids
        ):
            revision_action = "evidence_rebalance"

        if not notes:
            notes.append("No safe conservative revision was available without inventing evidence.")

        revised_claim = claim.model_copy(
            update={
                "claim_text": revised_text,
                "condition_scope": updated_scope,
                "condition_signature": (
                    claim.condition_signature
                    if updated_scope == claim.condition_scope
                    else normalize_condition_signature(updated_scope)
                ),
                "supporting_evidence_ids": updated_support_ids,
                "opposing_evidence_ids": updated_oppose_ids,
                "provenance_notes": self._append_provenance_note(
                    claim.provenance_notes,
                    revision_action=revision_action,
                ),
            }
        )
        revision_record = ClaimRevisionRecord(
            claim_id=claim.claim_id,
            original_claim_text=claim.claim_text,
            revised_claim_text=revised_text,
            revision_action=revision_action,
            updated_condition_scope=updated_scope,
            updated_supporting_evidence_ids=updated_support_ids,
            updated_opposing_evidence_ids=updated_oppose_ids,
            revision_rationale=" ".join(notes),
        )
        substantive_change = (
            revised_text != claim.claim_text
            or updated_scope != claim.condition_scope
            or updated_support_ids != list(claim.supporting_evidence_ids)
            or updated_oppose_ids != list(claim.opposing_evidence_ids)
        )
        return revised_claim, revision_record, substantive_change

    __call__ = run

    def _append_provenance_note(self, prior_note: Optional[str], *, revision_action: str) -> str:
        prefix = (prior_note or "").strip()
        review_note = f"Module 4 revision_action={revision_action}."
        if prefix:
            return f"{prefix} {review_note}"
        return review_note

    def _revise_with_llm(
        self,
        *,
        claim: ClaimRecord,
        task_spec: Optional[TaskSpec],
        review_flags: Sequence[ReviewFlag],
        conflict_edges: Sequence[ConflictEdge],
        supporting_items: Sequence[object],
        allowed_condition_scope: Dict[str, str],
        use_llm: bool,
    ) -> Dict[str, Any]:
        if not use_llm:
            raise PeerReviewExecutionError(
                stage="claim_revision_policy",
                message="Claim revision requires LLM generation.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            )
        if self.llm is None:
            raise PeerReviewExecutionError(
                stage="claim_revision_startup",
                message="Claim revision LLM is unavailable.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            )
        messages = [
            {"role": "system", "content": CLAIM_REVISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_claim_revision_user_prompt(
                    claim=claim.model_dump(exclude_none=True),
                    task_spec=task_spec.model_dump(exclude_none=True) if task_spec is not None else None,
                    review_flags=[flag.model_dump(exclude_none=True) for flag in review_flags],
                    conflict_edges=[edge.model_dump(exclude_none=True) for edge in conflict_edges],
                    supporting_evidence=[
                        {
                            "evidence_id": item.evidence_id,
                            "snippet": item.snippet,
                            "conditions": item.conditions,
                        }
                        for item in supporting_items[:4]
                    ],
                    allowed_condition_scope=allowed_condition_scope,
                ),
            },
        ]
        try:
            response_content = invoke_llm(self.llm, messages)
        except Exception as exc:
            raise PeerReviewExecutionError(
                stage="claim_revision_invoke",
                message="Claim revision LLM call failed.",
                details={"error_type": type(exc).__name__, "error": str(exc)},
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
            ) from exc
        parsed = parse_json_object(response_content)
        if not isinstance(parsed, dict):
            raise PeerReviewExecutionError(
                stage="claim_revision_invalid_response",
                message="Claim revision returned a non-object payload.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )

        claim_text = str(parsed.get("claim_text") or "").strip()
        if not claim_text:
            raise PeerReviewExecutionError(
                stage="claim_revision_invalid_response",
                message="Claim revision returned an empty claim_text.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )
        if claim.main_entity and claim.main_entity.lower() not in claim_text.lower():
            raise PeerReviewExecutionError(
                stage="claim_revision_invalid_response",
                message="Claim revision removed the main entity from the revised claim text.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )

        repaired_scope = self._repair_condition_scope(
            raw_scope=parsed.get("condition_scope"),
            allowed_condition_scope=allowed_condition_scope,
        )
        if repaired_scope is None:
            raise PeerReviewExecutionError(
                stage="claim_revision_invalid_response",
                message="Claim revision returned an invalid condition_scope.",
                reviewer_type=self.reviewer_type,
                claim_id=claim.claim_id,
                response_content=response_content,
            )

        if not repaired_scope and allowed_condition_scope:
            repaired_scope = dict(allowed_condition_scope)
        return {
            "claim_text": claim_text.rstrip(".") + ".",
            "condition_scope": repaired_scope,
        }

    def _repair_condition_scope(
        self,
        *,
        raw_scope: Any,
        allowed_condition_scope: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if raw_scope is None:
            return dict(allowed_condition_scope)
        if not isinstance(raw_scope, dict):
            return None
        repaired: Dict[str, str] = {}
        for axis, value in raw_scope.items():
            axis_text = str(axis or "").strip().lower()
            value_text = str(value or "").strip()
            allowed_value = allowed_condition_scope.get(axis_text)
            if not axis_text or not value_text or allowed_value is None:
                continue
            if value_text.lower() != str(allowed_value).strip().lower():
                continue
            repaired[axis_text] = value_text
        extra_axes = set(repaired).difference(allowed_condition_scope)
        if extra_axes:
            return None
        return repaired
