from __future__ import annotations

from statistics import mean
from typing import Any, List, Optional, Sequence, Set

from prompts.qa_prompts import CITATION_REVIEWER_SYSTEM_PROMPT, build_reviewer_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.review_utils import build_evidence_lookup, top_evidence_refs, traceable_evidence_items, valid_evidence_items
from qa.retrieval_state import ClaimRecord, EvidenceLedger, ReviewFlag


CITATION_FLAG_TYPES = ("Unsupported", "Weak_Evidence")


class CitationReviewer:
    reviewer_type = "CitationReviewer"

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        claim: ClaimRecord,
        evidence_ledger: EvidenceLedger,
        *,
        review_round: int = 1,
        focus_flag_types: Optional[Sequence[str]] = None,
        use_llm: bool = True,
    ) -> List[ReviewFlag]:
        focus = set(focus_flag_types or [])
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        flags: List[ReviewFlag] = []

        invalid_refs = [
            evidence_id
            for evidence_id in list(claim.supporting_evidence_ids) + list(claim.opposing_evidence_ids)
            if evidence_id not in evidence_lookup
        ]
        if invalid_refs:
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Fabricated_Citation",
                    severity="critical",
                    note="Claim references evidence identifiers that are not present in the ledger.",
                    evidence_refs=invalid_refs,
                )
            )

        supporting_items = valid_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        traceable_support = [item for item in supporting_items if item.snippet.strip()]
        if not traceable_support:
            flags.append(
                self._make_flag(
                    claim=claim,
                    review_round=review_round,
                    flag_type="Unsupported",
                    severity="warning",
                    note="Claim does not retain a traceable supporting snippet in the evidence ledger.",
                    evidence_refs=list(claim.supporting_evidence_ids),
                )
            )
        else:
            weak_evidence_reasons: List[str] = []
            if (
                len(traceable_support) == 1
                and claim.claim_type in {"causal", "mechanism", "comparison", "frontier_summary"}
                and (
                    traceable_support[0].source_layer == "abstract"
                    or traceable_support[0].extraction_confidence < 0.75
                )
            ):
                weak_evidence_reasons.append("Only one supporting snippet remains.")
            if claim.claim_type in {"causal", "mechanism", "comparison"} and all(
                item.source_layer == "abstract" for item in traceable_support
            ):
                weak_evidence_reasons.append("High-risk claim relies on abstract-only support.")
            if mean(item.extraction_confidence for item in traceable_support) < 0.68:
                weak_evidence_reasons.append("Average extraction confidence is low.")
            if weak_evidence_reasons:
                flags.append(
                    self._make_flag(
                        claim=claim,
                        review_round=review_round,
                        flag_type="Weak_Evidence",
                        severity="warning",
                        note=" ".join(weak_evidence_reasons),
                        evidence_refs=top_evidence_refs(traceable_support),
                    )
                )

        flags.extend(
            self._review_with_llm(
                claim=claim,
                traceable_support=traceable_support,
                review_round=review_round,
                focus_flag_types=focus_flag_types,
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
        traceable_support: Sequence[object],
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
                "source_layer": item.source_layer,
                "section_type": item.section_type,
                "extraction_confidence": item.extraction_confidence,
            }
            for item in traceable_support[:4]
        ]
        messages = [
            {"role": "system", "content": CITATION_REVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_reviewer_user_prompt(
                    review_kind="citation",
                    task_spec=None,
                    claim=claim.model_dump(exclude_none=True),
                    evidence_snippets=evidence_payload,
                    focus_flag_types=focus_flag_types,
                    allowed_flag_types=CITATION_FLAG_TYPES,
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
            if flag_type not in CITATION_FLAG_TYPES:
                continue
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
                    severity="warning",
                    note=note,
                    evidence_refs=evidence_refs or top_evidence_refs(traceable_support),
                )
            )
        return repaired
