from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from qa.review_utils import build_evidence_lookup, render_condition_text, traceable_evidence_items, valid_evidence_items
from qa.retrieval_state import ClaimRecord, EvidenceItem, EvidenceLedger, PaperRecord, RetrievalDiagnosticRecord, ReviewSummary
from qa.retrieval_utils import normalize_doi, normalize_text
from qa.state import AnswerSection, TaskSpec
from qa.synthesis_state import (
    CitationRecord,
    ClaimTraceItem,
    ConfidenceRating,
    ContestedClaimRecord,
    SectionClaimPack,
    SectionConfidenceRecord,
    SynthesisInputPack,
)


LIMITATION_SECTION_TOKENS = {
    "caveat",
    "caveats",
    "limitation",
    "limitations",
    "controversy",
    "controversies",
    "uncertainty",
    "uncertainties",
    "open_question",
    "open_questions",
}

SECTION_KEYWORD_MAP = {
    "direct": {"direct", "answer", "summary", "conclusion", "comparison", "recent", "trend"},
    "evidence": {"supporting", "evidence", "representative", "paper", "papers"},
    "conditions": {"condition", "conditions", "scope"},
    "mechanism": {"mechanism", "path", "pathway"},
    "effect": {"effect", "direction", "comparison"},
}


class SynthesisPackBuilder:
    def __init__(self, max_claims_per_section: int = 3) -> None:
        self.max_claims_per_section = max(1, int(max_claims_per_section))

    def run(
        self,
        *,
        task_spec: TaskSpec,
        evidence_ledger: EvidenceLedger,
        review_summaries: Optional[Sequence[ReviewSummary]] = None,
        paper_records: Optional[Sequence[PaperRecord]] = None,
        retrieval_diagnostics: Optional[Sequence[RetrievalDiagnosticRecord]] = None,
        execution_warnings: Optional[Sequence[str]] = None,
    ) -> SynthesisInputPack:
        summaries = list(review_summaries or evidence_ledger.review_summaries or [])
        summary_by_claim = {summary.claim_id: summary for summary in summaries}
        evidence_lookup = build_evidence_lookup(evidence_ledger)
        accepted_claims = sorted(
            (claim for claim in evidence_ledger.claims if claim.status == "accepted"),
            key=lambda claim: (claim.claim_confidence, claim.cluster_size, claim.claim_id),
            reverse=True,
        )
        contested_claims = [claim for claim in evidence_ledger.claims if claim.status == "contested"]
        rejected_claim_ids = {claim.claim_id for claim in evidence_ledger.claims if claim.status == "rejected"}

        citation_catalog = self._build_citation_catalog(
            claims=[*accepted_claims, *contested_claims],
            paper_records=paper_records or [],
            evidence_lookup=evidence_lookup,
        )
        citation_lookup = {citation.citation_id: citation for citation in citation_catalog}
        primary_section = self._primary_answer_section(task_spec.answer_sections)

        section_claims: List[SectionClaimPack] = []
        claim_trace: List[ClaimTraceItem] = []
        for section in task_spec.answer_sections:
            if self._is_limitation_section(section):
                continue
            pack = self._build_section_pack(
                section=section,
                accepted_claims=accepted_claims,
                primary_section=primary_section,
                evidence_lookup=evidence_lookup,
                citation_lookup=citation_lookup,
                summary_by_claim=summary_by_claim,
            )
            if pack is None:
                continue
            section_claims.append(pack)
            accepted_ids = set(pack.accepted_claim_ids)
            claim_trace.extend(
                self._build_claim_trace_items(
                    section_id=pack.section_id,
                    claims=[claim for claim in accepted_claims if claim.claim_id in accepted_ids],
                    evidence_lookup=evidence_lookup,
                    status="accepted",
                )
            )

        insufficient_evidence = self._insufficient_evidence(
            accepted_claims=accepted_claims,
            section_claims=section_claims,
            primary_section=primary_section,
            question_type=task_spec.question_type,
        )
        overall_confidence = self._compute_overall_confidence(
            accepted_claims=accepted_claims,
            contested_claims=contested_claims,
            evidence_lookup=evidence_lookup,
            evidence_ledger=evidence_ledger,
            insufficient_evidence=insufficient_evidence,
        )
        retrieval_diagnostics_summary = self._build_retrieval_diagnostics_summary(retrieval_diagnostics or [])
        cleaned_execution_warnings = self._clean_execution_warnings(execution_warnings or [])

        limitations_section_id = self._limitations_section_id(task_spec.answer_sections)
        limitations_required = bool(
            contested_claims
            or insufficient_evidence
            or overall_confidence.level == "low"
            or retrieval_diagnostics_summary
        )
        contested_records = self._build_contested_claim_records(
            claims=contested_claims,
            evidence_lookup=evidence_lookup,
            summary_by_claim=summary_by_claim,
        )
        if limitations_required:
            claim_trace.extend(
                self._build_claim_trace_items(
                    section_id=limitations_section_id,
                    claims=contested_claims,
                    evidence_lookup=evidence_lookup,
                    status="contested",
                )
            )

        section_confidence = [
            SectionConfidenceRecord(
                section_id=pack.section_id,
                title=pack.title,
                confidence=pack.section_confidence,
            )
            for pack in section_claims
        ]
        if limitations_required:
            section_confidence.append(
                SectionConfidenceRecord(
                    section_id=limitations_section_id,
                    title="Limitations / Controversies",
                    confidence=self._compute_limitations_confidence(
                        contested_records=contested_records,
                        insufficient_evidence=insufficient_evidence,
                        overall_confidence=overall_confidence,
                    ),
                )
            )

        filtered_claim_trace = [
            item
            for item in claim_trace
            if item.claim_id not in rejected_claim_ids and item.citation_ids
        ]

        return SynthesisInputPack(
            question=task_spec.question,
            task_spec=task_spec,
            section_claims=section_claims,
            contested_claims=contested_records,
            citation_catalog=citation_catalog,
            overall_confidence=overall_confidence,
            section_confidence=section_confidence,
            insufficient_evidence=insufficient_evidence,
            claim_trace=filtered_claim_trace,
            retrieval_diagnostics_summary=retrieval_diagnostics_summary,
            execution_warnings=cleaned_execution_warnings,
        )

    __call__ = run

    def _build_citation_catalog(
        self,
        *,
        claims: Sequence[ClaimRecord],
        paper_records: Sequence[PaperRecord],
        evidence_lookup: Dict[str, EvidenceItem],
    ) -> List[CitationRecord]:
        paper_metadata = {paper.paper_id: paper for paper in paper_records}
        claim_ids_by_citation: Dict[str, Set[str]] = defaultdict(set)
        paper_id_by_citation: Dict[str, str] = {}
        for claim in claims:
            for item in self._claim_evidence_items(claim, evidence_lookup=evidence_lookup):
                citation_id = normalize_doi(item.doi) or item.paper_id
                if not citation_id:
                    continue
                claim_ids_by_citation[citation_id].add(claim.claim_id)
                paper_id_by_citation[citation_id] = item.paper_id

        catalog: List[CitationRecord] = []
        for citation_id in sorted(claim_ids_by_citation):
            paper_id = paper_id_by_citation[citation_id]
            paper = paper_metadata.get(paper_id)
            doi = normalize_doi(paper.doi if paper is not None else None) or (
                citation_id if citation_id.startswith("10.") else None
            )
            title = normalize_text(paper.title if paper is not None else "") or paper_id
            venue = normalize_text(paper.venue if paper is not None else "") or None
            year = paper.year if paper is not None else None
            catalog.append(
                CitationRecord(
                    citation_id=citation_id,
                    paper_id=paper_id,
                    doi=doi,
                    title=title,
                    year=year,
                    venue=venue,
                    supporting_claim_ids=sorted(claim_ids_by_citation[citation_id]),
                )
            )
        return catalog

    def _build_section_pack(
        self,
        *,
        section: AnswerSection,
        accepted_claims: Sequence[ClaimRecord],
        primary_section: Optional[AnswerSection],
        evidence_lookup: Dict[str, EvidenceItem],
        citation_lookup: Dict[str, CitationRecord],
        summary_by_claim: Dict[str, ReviewSummary],
    ) -> Optional[SectionClaimPack]:
        selected_claims = self._select_claims_for_section(section=section, accepted_claims=accepted_claims)
        if not selected_claims and primary_section is not None and section.section_id == primary_section.section_id:
            selected_claims = list(accepted_claims[:1]) if accepted_claims else []
        if not selected_claims and section.required and primary_section is not None and section.section_id == primary_section.section_id:
            return SectionClaimPack(
                section_id=section.section_id,
                title=section.title,
                accepted_claim_ids=[],
                claim_summaries=[],
                core_citation_ids=[],
                section_confidence=self._low_confidence(
                    "No accepted claim reached the primary answer slot, so the answer remains explicitly limited."
                ),
            )
        if not selected_claims:
            return None

        claim_summaries = [self._claim_summary(claim) for claim in selected_claims]
        core_citation_ids = self._rank_citation_ids(
            claims=selected_claims,
            evidence_lookup=evidence_lookup,
            citation_lookup=citation_lookup,
        )
        return SectionClaimPack(
            section_id=section.section_id,
            title=section.title,
            accepted_claim_ids=[claim.claim_id for claim in selected_claims],
            claim_summaries=claim_summaries,
            core_citation_ids=core_citation_ids,
            section_confidence=self._compute_section_confidence(
                section=section,
                claims=selected_claims,
                evidence_lookup=evidence_lookup,
                summary_by_claim=summary_by_claim,
            ),
        )

    def _select_claims_for_section(
        self,
        *,
        section: AnswerSection,
        accepted_claims: Sequence[ClaimRecord],
    ) -> List[ClaimRecord]:
        scored: List[Tuple[float, float, int, ClaimRecord]] = []
        for claim in accepted_claims:
            score = self._score_claim_for_section(section=section, claim=claim)
            if score <= 0:
                continue
            scored.append((score, claim.claim_confidence, claim.cluster_size, claim))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3].claim_id), reverse=True)
        limit = self._section_claim_limit(section)
        return [item[3] for item in scored[:limit]]

    def _score_claim_for_section(self, *, section: AnswerSection, claim: ClaimRecord) -> float:
        section_text = " ".join([section.section_id, section.title, section.instruction]).lower().replace("/", " ")
        tokens = {token for token in section_text.replace("-", " ").split() if token}
        score = 0.0

        if self._looks_like_any(tokens, SECTION_KEYWORD_MAP["direct"]):
            score += 2.5
        if self._looks_like_any(tokens, SECTION_KEYWORD_MAP["evidence"]):
            score += 1.2 + min(claim.cluster_size, 3) * 0.4
        if self._looks_like_any(tokens, SECTION_KEYWORD_MAP["conditions"]):
            score += 2.0 if claim.condition_scope else -1.0
        if self._looks_like_any(tokens, SECTION_KEYWORD_MAP["mechanism"]):
            score += 3.0 if claim.claim_type == "mechanism" else -0.5
        if self._looks_like_any(tokens, SECTION_KEYWORD_MAP["effect"]):
            score += 2.2 if claim.claim_type in {"causal", "comparison"} else 0.4
        if "paper" in tokens or "papers" in tokens or "representative" in tokens:
            score += 2.0 if claim.claim_type == "frontier_summary" else 0.6

        if section.required:
            score += 0.5
        if claim.claim_type == "frontier_summary" and "recent" in tokens:
            score += 1.5
        if claim.condition_scope and ("condition" in tokens or "scope" in tokens):
            score += 1.5
        if claim.claim_type == "mechanism" and "mechanism" in tokens:
            score += 1.5
        if claim.claim_type in {"causal", "comparison"} and "direction" in tokens:
            score += 1.2
        if section.section_id == claim.section_id:
            score += 0.8

        score += claim.claim_confidence * 1.4
        score += min(claim.cluster_size, 4) * 0.2
        return score

    def _section_claim_limit(self, section: AnswerSection) -> int:
        lower_id = section.section_id.lower()
        if "evidence" in lower_id or "paper" in lower_id:
            return min(self.max_claims_per_section, 3)
        return min(self.max_claims_per_section, 2)

    def _claim_summary(self, claim: ClaimRecord) -> str:
        summary = normalize_text(claim.claim_text)
        condition_text = render_condition_text(claim.condition_scope)
        if condition_text and condition_text.lower() not in summary.lower():
            return f"{summary} Condition scope: {condition_text}."
        return summary

    def _rank_citation_ids(
        self,
        *,
        claims: Sequence[ClaimRecord],
        evidence_lookup: Dict[str, EvidenceItem],
        citation_lookup: Dict[str, CitationRecord],
    ) -> List[str]:
        counts: Counter[str] = Counter()
        years: Dict[str, int] = {}
        for claim in claims:
            for item in self._claim_evidence_items(claim, evidence_lookup=evidence_lookup):
                citation_id = normalize_doi(item.doi) or item.paper_id
                if not citation_id or citation_id not in citation_lookup:
                    continue
                counts[citation_id] += 1
                year = citation_lookup[citation_id].year or 0
                years[citation_id] = max(years.get(citation_id, 0), year)
        ranked = sorted(
            counts,
            key=lambda citation_id: (counts[citation_id], years.get(citation_id, 0), citation_id),
            reverse=True,
        )
        return ranked[:3]

    def _build_contested_claim_records(
        self,
        *,
        claims: Sequence[ClaimRecord],
        evidence_lookup: Dict[str, EvidenceItem],
        summary_by_claim: Dict[str, ReviewSummary],
    ) -> List[ContestedClaimRecord]:
        contested_records: List[ContestedClaimRecord] = []
        for claim in claims:
            summary = summary_by_claim.get(claim.claim_id)
            citation_ids = self._claim_citation_ids(claim, evidence_lookup=evidence_lookup)
            rationale = normalize_text(summary.merge_rationale if summary is not None else "")
            if not rationale:
                rationale = "Review retained this claim as unresolved or condition-bound rather than fully accepted."
            contested_records.append(
                ContestedClaimRecord(
                    claim_id=claim.claim_id,
                    claim_summary=self._claim_summary(claim),
                    citation_ids=citation_ids,
                    confidence=round(claim.claim_confidence, 2),
                    rationale=rationale,
                )
            )
        return contested_records

    def _build_claim_trace_items(
        self,
        *,
        section_id: str,
        claims: Sequence[ClaimRecord],
        evidence_lookup: Dict[str, EvidenceItem],
        status: str,
    ) -> List[ClaimTraceItem]:
        items: List[ClaimTraceItem] = []
        seen_keys: Set[Tuple[str, str]] = set()
        for claim in claims:
            citation_ids = self._claim_citation_ids(claim, evidence_lookup=evidence_lookup)
            key = (section_id, claim.claim_id)
            if key in seen_keys or not citation_ids:
                continue
            seen_keys.add(key)
            items.append(
                ClaimTraceItem(
                    section_id=section_id,
                    claim_id=claim.claim_id,
                    status=status,
                    citation_ids=citation_ids,
                    confidence=round(claim.claim_confidence, 2),
                )
            )
        return items

    def _claim_citation_ids(self, claim: ClaimRecord, *, evidence_lookup: Dict[str, EvidenceItem]) -> List[str]:
        citation_ids: List[str] = []
        for item in self._claim_evidence_items(claim, evidence_lookup=evidence_lookup):
            citation_id = normalize_doi(item.doi) or item.paper_id
            if citation_id and citation_id not in citation_ids:
                citation_ids.append(citation_id)
        return citation_ids

    def _claim_evidence_items(self, claim: ClaimRecord, *, evidence_lookup: Dict[str, EvidenceItem]) -> List[EvidenceItem]:
        if claim.status == "accepted":
            return traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        supporting = valid_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
        opposing = valid_evidence_items(claim.opposing_evidence_ids, evidence_lookup=evidence_lookup)
        combined: List[EvidenceItem] = []
        for item in [*supporting, *opposing]:
            if item not in combined and normalize_text(item.snippet):
                combined.append(item)
        return combined

    def _compute_section_confidence(
        self,
        *,
        section: AnswerSection,
        claims: Sequence[ClaimRecord],
        evidence_lookup: Dict[str, EvidenceItem],
        summary_by_claim: Dict[str, ReviewSummary],
    ) -> ConfidenceRating:
        if not claims:
            return self._low_confidence("This section has no accepted claim support.")
        evidence_strengths: List[float] = []
        revision_penalty = 0.0
        for claim in claims:
            support_items = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
            if support_items:
                evidence_strengths.append(
                    sum(item.extraction_confidence for item in support_items) / len(support_items)
                )
            summary = summary_by_claim.get(claim.claim_id)
            if summary is not None and summary.review_rounds > 1:
                revision_penalty += 0.03
            if summary is not None and summary.revision_records:
                revision_penalty += 0.02

        avg_claim_conf = sum(claim.claim_confidence for claim in claims) / len(claims)
        avg_evidence_conf = sum(evidence_strengths) / len(evidence_strengths) if evidence_strengths else 0.45
        score = 0.22
        score += min(len(claims), 3) * 0.12
        score += avg_claim_conf * 0.34
        score += avg_evidence_conf * 0.24
        score -= revision_penalty
        if self._is_primary_answer_section(section):
            score += 0.04
        rationale = (
            f"{len(claims)} accepted claim(s) support this section with "
            f"mean claim confidence {avg_claim_conf:.2f} and mean evidence confidence {avg_evidence_conf:.2f}."
        )
        return self._confidence_from_score(score, rationale)

    def _compute_overall_confidence(
        self,
        *,
        accepted_claims: Sequence[ClaimRecord],
        contested_claims: Sequence[ClaimRecord],
        evidence_lookup: Dict[str, EvidenceItem],
        evidence_ledger: EvidenceLedger,
        insufficient_evidence: bool,
    ) -> ConfidenceRating:
        accepted_count = len(accepted_claims)
        contested_count = len(contested_claims)
        rejected_count = sum(1 for claim in evidence_ledger.claims if claim.status == "rejected")
        true_conflicts = sum(1 for edge in evidence_ledger.conflict_edges if edge.conflict_type == "true_conflict")
        condition_divergences = sum(
            1 for edge in evidence_ledger.conflict_edges if edge.conflict_type == "condition_divergence"
        )

        claim_conf = (
            sum(claim.claim_confidence for claim in accepted_claims) / accepted_count
            if accepted_count
            else 0.0
        )
        evidence_strengths: List[float] = []
        for claim in accepted_claims:
            support_items = traceable_evidence_items(claim.supporting_evidence_ids, evidence_lookup=evidence_lookup)
            if support_items:
                evidence_strengths.append(
                    sum(item.extraction_confidence for item in support_items) / len(support_items)
                )
        evidence_conf = sum(evidence_strengths) / len(evidence_strengths) if evidence_strengths else 0.0

        score = 0.16
        score += min(accepted_count, 4) * 0.13
        score += claim_conf * 0.28
        score += evidence_conf * 0.22
        score -= min(contested_count, 4) * 0.08
        score -= min(rejected_count, 4) * 0.10
        score -= true_conflicts * 0.10
        score -= condition_divergences * 0.04
        if insufficient_evidence:
            score -= 0.12

        rationale = (
            f"{accepted_count} accepted, {contested_count} contested, and {rejected_count} rejected claim(s); "
            f"{true_conflicts} true conflict edge(s) remain."
        )
        return self._confidence_from_score(score, rationale)

    def _compute_limitations_confidence(
        self,
        *,
        contested_records: Sequence[ContestedClaimRecord],
        insufficient_evidence: bool,
        overall_confidence: ConfidenceRating,
    ) -> ConfidenceRating:
        if contested_records:
            score = 0.62 if overall_confidence.level != "low" else 0.5
            rationale = f"{len(contested_records)} contested claim(s) are explicitly preserved for auditability."
            return self._confidence_from_score(score, rationale)
        if insufficient_evidence:
            return self._confidence_from_score(
                0.42,
                "The limitations section is required because accepted evidence is too sparse for a fully firm answer.",
            )
        return self._confidence_from_score(0.5, "This section records residual uncertainty conservatively.")

    def _build_retrieval_diagnostics_summary(
        self,
        retrieval_diagnostics: Sequence[RetrievalDiagnosticRecord],
    ) -> str:
        relevant_records = [
            record
            for record in retrieval_diagnostics
            if record.failure_count or record.timeout_count or record.skipped_count
        ]
        if not relevant_records:
            return ""

        pieces: List[str] = []
        for record in relevant_records[:5]:
            scope = self._diagnostic_scope(record)
            counts: List[str] = []
            if record.failure_count:
                counts.append(f"{record.failure_count} failure{'s' if record.failure_count != 1 else ''}")
            if record.timeout_count:
                counts.append(f"{record.timeout_count} timeout{'s' if record.timeout_count != 1 else ''}")
            if record.skipped_count:
                counts.append(f"{record.skipped_count} skipped")
            qualifiers = self._diagnostic_qualifiers(record)
            if counts:
                text = f"{scope} had {', '.join(counts)}"
                if qualifiers:
                    text += f" ({'; '.join(qualifiers)})"
                pieces.append(text)
        if not pieces:
            return ""
        return (
            "External literature retrieval encountered issues: "
            + "; ".join(pieces)
            + ". These limitations may reflect provider or network availability rather than a true absence of relevant studies."
        )

    def _diagnostic_scope(self, record: RetrievalDiagnosticRecord) -> str:
        provider_names = {
            "openalex": "OpenAlex",
            "crossref": "Crossref",
            "semantic_scholar": "Semantic Scholar",
            "unpaywall": "Unpaywall",
            "oa_fetch": "open-access fetch",
        }
        provider_text = provider_names.get(record.provider, record.provider.replace("_", " "))
        if record.stage == "search" and record.lane:
            return f"{provider_text} {record.lane} search"
        if record.stage == "enrichment":
            return f"{provider_text} enrichment"
        if record.stage == "lookup":
            return f"{provider_text} lookup"
        if record.stage == "fetch":
            return provider_text
        return f"{provider_text} {record.stage}"

    def _diagnostic_qualifiers(self, record: RetrievalDiagnosticRecord) -> List[str]:
        joined_messages = " ".join(record.sample_messages).lower()
        qualifiers: List[str] = []
        if "retry exhausted" in joined_messages:
            qualifiers.append("retry exhausted")
        if "provider unavailable" in joined_messages or "marked unavailable" in joined_messages:
            qualifiers.append("provider unavailable")
        if record.skipped_count and ("provider unavailable" in joined_messages or "marked unavailable" in joined_messages):
            qualifiers.append("subsequent calls skipped")
        return qualifiers

    def _clean_execution_warnings(self, warnings: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for warning in warnings:
            text = normalize_text(warning)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _insufficient_evidence(
        self,
        *,
        accepted_claims: Sequence[ClaimRecord],
        section_claims: Sequence[SectionClaimPack],
        primary_section: Optional[AnswerSection],
        question_type: str,
    ) -> bool:
        if not accepted_claims:
            return True
        if primary_section is not None:
            primary_pack = next((pack for pack in section_claims if pack.section_id == primary_section.section_id), None)
            if primary_pack is None or not primary_pack.accepted_claim_ids:
                return True
        if question_type in {"causal", "mechanism", "comparison", "frontier"} and len(accepted_claims) < 2:
            return True
        return False

    def _primary_answer_section(self, sections: Sequence[AnswerSection]) -> Optional[AnswerSection]:
        for section in sections:
            if not self._is_limitation_section(section):
                return section
        return None

    def _limitations_section_id(self, sections: Sequence[AnswerSection]) -> str:
        for section in sections:
            if self._is_limitation_section(section):
                return section.section_id
        return "limitations_controversies"

    def _is_limitation_section(self, section: AnswerSection) -> bool:
        raw_text = f"{section.section_id} {section.title} {section.instruction}".lower()
        normalized_text = raw_text.replace("/", " ").replace("-", " ").replace("_", " ")
        tokens = {token for token in normalized_text.split() if token}
        if "open" in tokens and "question" in tokens:
            return True
        if "open" in tokens and "questions" in tokens:
            return True
        return any(token in LIMITATION_SECTION_TOKENS for token in tokens) or any(
            marker in raw_text for marker in LIMITATION_SECTION_TOKENS
        )

    def _is_primary_answer_section(self, section: AnswerSection) -> bool:
        return not self._is_limitation_section(section)

    def _looks_like_any(self, tokens: Set[str], keywords: Iterable[str]) -> bool:
        keyword_set = set(keywords)
        return any(keyword in tokens for keyword in keyword_set)

    def _confidence_from_score(self, score: float, rationale: str) -> ConfidenceRating:
        bounded = round(max(0.05, min(score, 0.95)), 2)
        if bounded >= 0.72:
            level = "high"
        elif bounded >= 0.45:
            level = "medium"
        else:
            level = "low"
        return ConfidenceRating(level=level, score=bounded, rationale=rationale)

    def _low_confidence(self, rationale: str) -> ConfidenceRating:
        return ConfidenceRating(level="low", score=0.2, rationale=rationale)
