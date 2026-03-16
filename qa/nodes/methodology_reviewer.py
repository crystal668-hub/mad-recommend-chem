from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

from prompts.qa_prompts import (
    METHODOLOGY_REVIEWER_SYSTEM_PROMPT,
    build_reviewer_user_prompt,
)
from qa.llm_utils import invoke_llm, parse_json_object
from qa.review_utils import (
    HIGH_RISK_CLAIM_TYPES,
    build_evidence_lookup,
    condition_scope_is_ambiguous,
    contains_absolute_language,
    speculative_text,
    top_evidence_refs,
    traceable_evidence_items,
)
from qa.retrieval_state import ClaimRecord, EvidenceLedger, ReviewFlag
from qa.state import TaskSpec


METHODOLOGY_FLAG_TYPES = (
    "Missing_Condition",
    "Incomplete_Condition",
    "Overgeneralized",
    "Mechanism_Speculative",
    "Metric_Mismatch",
)


class MethodologyReviewer:
    reviewer_type = "MethodologyReviewer"

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        claim: ClaimRecord,
        evidence_ledger: EvidenceLedger,
        *,
        task_spec: Optional[TaskSpec] = None,
        review_round: int = 1,
        focus_flag_types: Optional[Sequence[str]] = None,
        use_llm: bool = True,
    ) -> List[ReviewFlag]:
        focus = set(focus_flag_types or [])
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        supporting_items = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        flags: List[ReviewFlag] = []
        required_axes = list(task_spec.required_condition_axes or []) if task_spec is not None else []
        support_axes = set()
        support_metrics = set()
        for item in supporting_items:
            support_axes.update(item.conditions)
            support_metrics.update(item.metric_mentions)

        if (
            claim.claim_type in HIGH_RISK_CLAIM_TYPES
            or required_axes
            or support_axes
        ) and not claim.condition_scope:
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Missing_Condition",
                    severity="warning",
                    note="Claim lacks an explicit condition scope despite condition-bound supporting evidence.",
                    evidence_refs=top_evidence_refs(supporting_items),
                )
            )
        elif condition_scope_is_ambiguous(
            claim,
            required_axes=required_axes,
            supporting_items=supporting_items,
        ):
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Incomplete_Condition",
                    severity="warning",
                    note="Claim condition scope omits axes that appear in the supporting evidence or task requirements.",
                    evidence_refs=top_evidence_refs(supporting_items),
                )
            )

        if contains_absolute_language(claim.claim_text) or (support_axes and not claim.condition_scope):
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Overgeneralized",
                    severity="warning",
                    note="Claim wording is broader than the condition-bound evidence currently supports.",
                    evidence_refs=top_evidence_refs(supporting_items),
                )
            )

        if claim.metric_family != "unspecified" and support_metrics and claim.metric_family not in support_metrics:
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Metric_Mismatch",
                    severity="warning",
                    note="Claim metric family does not match the metrics explicitly mentioned in supporting evidence.",
                    evidence_refs=top_evidence_refs(supporting_items),
                )
            )

        if claim.claim_type == "mechanism" and (
            speculative_text(claim.claim_text) or any(speculative_text(item.snippet) for item in supporting_items)
        ):
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Mechanism_Speculative",
                    severity="warning",
                    note="Mechanism claim relies on speculative language and should be downgraded unless narrowed further.",
                    evidence_refs=top_evidence_refs(supporting_items),
                )
            )

        if focus:
            flags = [flag for flag in flags if flag.flag_type in focus]
        flags.extend(
            self._review_with_llm(
                claim=claim,
                task_spec=task_spec,
                supporting_items=supporting_items,
                review_round=review_round,
                focus_flag_types=sorted(focus) if focus else None,
                use_llm=use_llm,
            )
        )
        if focus:
            flags = [flag for flag in flags if flag.flag_type in focus]
        return self._dedupe_flags(flags)

    __call__ = run

    def _make_flag(
        self,
        *,
        claim: ClaimRecord,
        review_round: int,
        flag_type: str,
        severity: str,
        note: str,
        evidence_refs: Sequence[str],
    ) -> ReviewFlag:
        return ReviewFlag(
            flag_id=f"flag_r{review_round}_{claim.claim_id}_{flag_type.lower()}",
            claim_id=claim.claim_id,
            reviewer_type=self.reviewer_type,
            flag_type=flag_type,
            severity=severity,
            note=note,
            evidence_refs=list(evidence_refs),
        )

    def _dedupe_flags(self, flags: Sequence[ReviewFlag]) -> List[ReviewFlag]:
        deduped: List[ReviewFlag] = []
        seen: Set[tuple[str, str]] = set()
        for flag in flags:
            key = (flag.claim_id, flag.flag_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(flag)
        return deduped

    def _review_with_llm(
        self,
        *,
        claim: ClaimRecord,
        task_spec: Optional[TaskSpec],
        supporting_items: Sequence[object],
        review_round: int,
        focus_flag_types: Optional[Sequence[str]],
        use_llm: bool,
    ) -> List[ReviewFlag]:
        if self.llm is None or not use_llm:
            return []
        evidence_payload = [
            {
                "evidence_id": item.evidence_id,
                "snippet": item.snippet,
                "conditions": item.conditions,
                "metric_mentions": item.metric_mentions,
            }
            for item in supporting_items[:4]
        ]
        messages = [
            {"role": "system", "content": METHODOLOGY_REVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_reviewer_user_prompt(
                    review_kind="methodology",
                    task_spec=task_spec.model_dump(exclude_none=True) if task_spec is not None else None,
                    claim=claim.model_dump(exclude_none=True),
                    evidence_snippets=evidence_payload,
                    focus_flag_types=focus_flag_types,
                    allowed_flag_types=METHODOLOGY_FLAG_TYPES,
                ),
            },
        ]
        try:
            parsed = parse_json_object(invoke_llm(self.llm, messages))
        except Exception:
            return []
        if not isinstance(parsed, dict):
            return []
        valid_evidence_refs = {item["evidence_id"] for item in evidence_payload}
        repaired: List[ReviewFlag] = []
        for raw_flag in parsed.get("flags") or []:
            if not isinstance(raw_flag, dict):
                continue
            flag_type = raw_flag.get("flag_type")
            if flag_type not in METHODOLOGY_FLAG_TYPES:
                continue
            severity = raw_flag.get("severity")
            if severity not in {"info", "warning", "critical"}:
                severity = "warning"
            note = str(raw_flag.get("note") or "").strip()
            if not note:
                continue
            evidence_refs = [
                evidence_id
                for evidence_id in raw_flag.get("evidence_refs") or []
                if evidence_id in valid_evidence_refs
            ]
            repaired.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type=flag_type,
                    severity=severity,
                    note=note,
                    evidence_refs=evidence_refs or top_evidence_refs(supporting_items),
                )
            )
        return repaired
