from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from prompts.qa_prompts import (
    CONTRADICTION_REVIEWER_SYSTEM_PROMPT,
    build_contradiction_reviewer_user_prompt,
)
from qa.llm_utils import invoke_llm, parse_json_object
from qa.review_utils import (
    build_evidence_lookup,
    conflict_pair_key,
    directions_are_opposed,
    infer_claim_direction,
    same_topic,
    scopes_highly_overlap,
    shared_and_differing_axes,
    top_evidence_refs,
    valid_evidence_items,
)
from qa.retrieval_state import ClaimRecord, ConflictEdge, EvidenceLedger, ReviewFlag


class ContradictionReviewer:
    reviewer_type = "ContradictionReviewer"

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        claims: Sequence[ClaimRecord],
        evidence_ledger: EvidenceLedger,
        *,
        review_round: int = 1,
        focus_claim_ids: Optional[Sequence[str]] = None,
        candidate_pair_keys: Optional[Set[Tuple[str, str]]] = None,
    ) -> Tuple[List[ReviewFlag], List[ConflictEdge], Set[Tuple[str, str]]]:
        claim_list = list(claims)
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        focus_ids = set(focus_claim_ids or [])
        reviewed_pairs: Set[Tuple[str, str]] = set()
        flags: List[ReviewFlag] = []
        edges: List[ConflictEdge] = []

        for index, left_claim in enumerate(claim_list):
            for right_claim in claim_list[index + 1 :]:
                pair_key = conflict_pair_key(left_claim.claim_id, right_claim.claim_id)
                if candidate_pair_keys is not None and pair_key not in candidate_pair_keys:
                    continue
                if candidate_pair_keys is None and focus_ids and not (
                    left_claim.claim_id in focus_ids or right_claim.claim_id in focus_ids
                ):
                    continue
                if candidate_pair_keys is None and not focus_ids and not same_topic(left_claim, right_claim):
                    continue

                reviewed_pairs.add(pair_key)
                if not same_topic(left_claim, right_claim):
                    continue

                left_support = valid_evidence_items(left_claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
                left_oppose = valid_evidence_items(left_claim.opposing_evidence_ids, evidence_lookup=evidence_lookup)
                right_support = valid_evidence_items(right_claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
                right_oppose = valid_evidence_items(right_claim.opposing_evidence_ids, evidence_lookup=evidence_lookup)

                left_direction = infer_claim_direction(left_claim, supporting_items=left_support, opposing_items=left_oppose)
                right_direction = infer_claim_direction(right_claim, supporting_items=right_support, opposing_items=right_oppose)
                if not directions_are_opposed(left_direction, right_direction):
                    continue

                shared_axes, differing_axes = shared_and_differing_axes(
                    left_claim.condition_scope,
                    right_claim.condition_scope,
                )
                conflict_type, severity, reason, flag_type = self._adjudicate_conflict_type(
                    left_claim=left_claim,
                    right_claim=right_claim,
                    left_support=left_support,
                    right_support=right_support,
                    shared_axes=shared_axes,
                    differing_axes=differing_axes,
                )
                if conflict_type is None:
                    continue

                evidence_refs = top_evidence_refs([*left_support, *right_support], limit=4)
                edge = ConflictEdge(
                    conflict_id=f"conflict_r{review_round}_{left_claim.claim_id}_{right_claim.claim_id}_{conflict_type}",
                    left_claim_id=left_claim.claim_id,
                    right_claim_id=right_claim.claim_id,
                    conflict_type=conflict_type,
                    severity=severity,
                    reason=reason,
                    shared_axes=shared_axes,
                    differing_axes=differing_axes,
                    evidence_refs=evidence_refs,
                )
                edges.append(edge)
                flags.extend(
                    [
                        self._make_flag(
                            claim_id=left_claim.claim_id,
                            counterpart_claim_id=right_claim.claim_id,
                            review_round=review_round,
                            flag_type=flag_type,
                            severity=severity,
                            note=reason,
                            evidence_refs=evidence_refs,
                        ),
                        self._make_flag(
                            claim_id=right_claim.claim_id,
                            counterpart_claim_id=left_claim.claim_id,
                            review_round=review_round,
                            flag_type=flag_type,
                            severity=severity,
                            note=reason,
                            evidence_refs=evidence_refs,
                        ),
                    ]
                )

        return self._dedupe_flags(flags), edges, reviewed_pairs

    __call__ = run

    def _make_flag(
        self,
        *,
        claim_id: str,
        counterpart_claim_id: str,
        review_round: int,
        flag_type: str,
        severity: str,
        note: str,
        evidence_refs: Sequence[str],
    ) -> ReviewFlag:
        return ReviewFlag(
            flag_id=f"flag_r{review_round}_{claim_id}_{flag_type.lower()}_{counterpart_claim_id}",
            claim_id=claim_id,
            reviewer_type=self.reviewer_type,
            flag_type=flag_type,
            severity=severity,
            note=note,
            evidence_refs=list(evidence_refs),
        )

    def _dedupe_flags(self, flags: Iterable[ReviewFlag]) -> List[ReviewFlag]:
        deduped: List[ReviewFlag] = []
        seen: Set[tuple[str, str, tuple[str, ...]]] = set()
        for flag in flags:
            key = (flag.claim_id, flag.flag_type, tuple(flag.evidence_refs))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(flag)
        return deduped

    def _adjudicate_conflict_type(
        self,
        *,
        left_claim: ClaimRecord,
        right_claim: ClaimRecord,
        left_support: Sequence[object],
        right_support: Sequence[object],
        shared_axes: Sequence[str],
        differing_axes: Sequence[str],
    ) -> Tuple[Optional[str], str, str, str]:
        default = self._deterministic_conflict_type(
            left_claim=left_claim,
            right_claim=right_claim,
            differing_axes=differing_axes,
        )
        if self.llm is None:
            return default

        messages = [
            {"role": "system", "content": CONTRADICTION_REVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_contradiction_reviewer_user_prompt(
                    left_claim=left_claim.model_dump(exclude_none=True),
                    right_claim=right_claim.model_dump(exclude_none=True),
                    left_evidence=[
                        {"evidence_id": item.evidence_id, "snippet": item.snippet, "conditions": item.conditions}
                        for item in left_support[:3]
                    ],
                    right_evidence=[
                        {"evidence_id": item.evidence_id, "snippet": item.snippet, "conditions": item.conditions}
                        for item in right_support[:3]
                    ],
                    shared_axes=shared_axes,
                    differing_axes=differing_axes,
                ),
            },
        ]
        try:
            parsed = parse_json_object(invoke_llm(self.llm, messages))
        except Exception:
            return default
        conflict_type = (parsed or {}).get("conflict_type")
        if conflict_type == "no_conflict":
            return None, "info", "LLM adjudication found no actionable conflict after pair prefiltering.", ""
        if conflict_type in {"true_conflict", "condition_divergence"}:
            if conflict_type == "true_conflict":
                reason = str((parsed or {}).get("reason") or "").strip() or default[2]
                return "true_conflict", "warning", reason, "True_Conflict"
            reason = str((parsed or {}).get("reason") or "").strip() or default[2]
            return "condition_divergence", "info", reason, "Condition_Divergence"
        return default

    def _deterministic_conflict_type(
        self,
        *,
        left_claim: ClaimRecord,
        right_claim: ClaimRecord,
        differing_axes: Sequence[str],
    ) -> Tuple[str, str, str, str]:
        if scopes_highly_overlap(left_claim.condition_scope, right_claim.condition_scope) and not differing_axes:
            return (
                "true_conflict",
                "warning",
                "Claims assert opposite directions under materially overlapping conditions.",
                "True_Conflict",
            )
        return (
            "condition_divergence",
            "info",
            "Claims move in opposite directions but differ in condition scope.",
            "Condition_Divergence",
        )
