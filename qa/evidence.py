from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from prompts.qa_prompts import (
    CLAIM_MINER_SYSTEM_PROMPT,
    EVIDENCE_EXTRACTOR_SYSTEM_PROMPT,
    build_claim_miner_user_prompt,
    build_evidence_extractor_user_prompt,
)
from qa.handoff import EvidenceExtractorHandoff
from qa.llm_utils import invoke_llm, parse_json_object
from qa.retrieval_state import (
    ClaimRecord,
    EvidenceItem,
    EvidenceLedger,
    PaperRecord,
    SectionIndex,
    SectionTextView,
)
from qa.retrieval_utils import normalize_text, slugify
from qa.state import EntityPack, TaskSpec


CLAIM_ELIGIBLE_ROLES = {"observation", "limitation", "mechanism"}
ROLE_ORDER = ("condition", "observation", "limitation", "mechanism")
SECTION_PRIORITY = {
    "methods": 0,
    "abstract": 1,
    "results": 2,
    "discussion": 3,
    "conclusion": 4,
    "limitations": 5,
    "unknown": 6,
}
METRIC_FAMILY_PRIORITY = [
    "yield",
    "selectivity",
    "current_density",
    "overpotential",
    "conversion",
    "activity",
    "rate",
    "stability",
]
METRIC_PATTERNS = [
    ("yield", re.compile(r"\byield\b|\b\d+(?:\.\d+)?\s*%\s+yield\b", re.I)),
    ("selectivity", re.compile(r"\bselectivity\b|\bfaradaic efficiency\b|\bFE\b", re.I)),
    ("current_density", re.compile(r"\bcurrent density\b|\bmA\s*/\s*cm2\b", re.I)),
    ("overpotential", re.compile(r"\boverpotential\b|\b-?\d+(?:\.\d+)?\s*m?V\b", re.I)),
    ("conversion", re.compile(r"\bconversion\b", re.I)),
    ("activity", re.compile(r"\bactivity\b|\bturnover\b|\bTOF\b|\bTON\b", re.I)),
    ("rate", re.compile(r"\brate\b|\bkinetic\b|\bkinetics\b", re.I)),
    ("stability", re.compile(r"\bstability\b|\bdurability\b|\bdeactivation\b", re.I)),
]
LIMITATION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bdid not\b",
        r"\bfailed to\b",
        r"\bno significant\b",
        r"\bnot observed\b",
        r"\blimited\b",
        r"\blimitation\b",
        r"\bunderperformed\b",
        r"\bdeactivated\b",
        r"\bonly under\b",
        r"\bwhereas\b",
        r"\bbut not\b",
        r"\bpoor(?:ly)?\b",
    )
]
MECHANISM_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bbecause\b",
        r"\bdue to\b",
        r"\bvia\b",
        r"\bthrough\b",
        r"\bmechanis(?:m|tic)\b",
        r"\bpathway\b",
        r"\bintermediate\b",
        r"\boxidative addition\b",
        r"\breductive elimination\b",
        r"\badsorption\b",
        r"\bdesorption\b",
        r"\bhydrolysis\b",
        r"\bisomerization\b",
        r"\brate[- ]determining\b",
        r"\belectron transfer\b",
    )
]
OBSERVATION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bincreas(?:e|ed|es|ing)\b",
        r"\bdecreas(?:e|ed|es|ing)\b",
        r"\bimprov(?:e|ed|es|ing)\b",
        r"\benhanc(?:e|ed|es|ing)\b",
        r"\bshow(?:ed|s)?\b",
        r"\bobserved\b",
        r"\bgave\b",
        r"\byielded\b",
        r"\boutperform(?:ed|s|ing)?\b",
        r"\bbetter than\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\bcorrelat(?:e|ed|es|ing)\b",
        r"\baffect(?:s|ed|ing)?\b",
        r"\binfluence(?:s|d|ing)?\b",
        r"\bcontrol(?:s|led|ling)?\b",
        r"\bsuppress(?:ed|es|ing)?\b",
    )
]
COMPARISON_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompared with\b",
        r"\bbetter than\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\boutperform(?:ed|s|ing)?\b",
    )
]
NEGATIVE_EFFECT_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bdecreas(?:e|ed|es|ing)\b",
        r"\breduc(?:e|ed|es|ing)\b",
        r"\bsuppress(?:ed|es|ing)?\b",
        r"\blower than\b",
        r"\bworse\b",
        r"\bpoor(?:ly)?\b",
    )
]
CHEMICAL_TOKEN_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9]{0,}(?:[-/][A-Za-z0-9]+)*)\b")
SOLVENT_NAMES = [
    "water",
    "toluene",
    "dioxane",
    "dmf",
    "dma",
    "thf",
    "meoh",
    "ethanol",
    "methanol",
    "acetonitrile",
    "dcm",
    "dichloromethane",
    "hexane",
    "benzene",
    "ipa",
]
CONDITION_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "temperature": (
        re.compile(r"\b(?P<value>-?\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?\s*)?C\b", re.I),
        re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*K\b", re.I),
    ),
    "time": (
        re.compile(r"\b(?P<value>\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours|min|mins|minutes))\b", re.I),
    ),
    "ph": (
        re.compile(r"\b(?P<value>pH\s*\d+(?:\.\d+)?)\b", re.I),
    ),
    "pressure": (
        re.compile(r"\b(?P<value>\d+(?:\.\d+)?\s*(?:bar|atm|kPa|MPa|Pa))\b", re.I),
    ),
    "potential": (
        re.compile(r"\b(?P<value>-?\d+(?:\.\d+)?\s*(?:V|mV))\b", re.I),
    ),
    "electrolyte": (
        re.compile(r"\b(?P<value>\d+(?:\.\d+)?\s*M\s+[A-Za-z0-9()/-]+)\b", re.I),
    ),
}
PREFIXED_CONDITION_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "catalyst": (
        re.compile(r"\b(?:catalyst|catalyzed by|catalyzed with|using)\s+(?P<value>[A-Za-z0-9()/.+\-]{2,40})", re.I),
    ),
    "material": (
        re.compile(r"\b(?:electrode|material|support)\s+(?P<value>[A-Za-z0-9()/.+\-]{2,40})", re.I),
    ),
    "substrate": (
        re.compile(r"\bsubstrate\s+(?P<value>[A-Za-z0-9()/.+\-]{2,40})", re.I),
    ),
    "ligand": (
        re.compile(r"\bligand\s+(?P<value>[A-Za-z0-9()/.+\-]{2,40})", re.I),
    ),
    "reagent": (
        re.compile(r"\b(?:with|using)\s+(?P<value>[A-Z][A-Za-z0-9()/.+\-]{1,40})", re.I),
    ),
}
SOURCE_LAYER_BY_SECTION = {"abstract": "abstract"}
ENTITY_TYPE_TO_AXIS = {
    "catalyst": "catalyst",
    "material": "material",
    "substrate": "substrate",
    "solvent": "solvent",
    "ligand": "ligand",
    "reagent": "reagent",
}


@dataclass
class _RoleDecision:
    roles: List[str]
    claim_polarity: str
    entity_mentions: List[str]
    metric_mentions: List[str]
    notes: Optional[str]


def build_condition_signature(condition_scope: Dict[str, str]) -> str:
    normalized_scope = {
        str(key).strip().lower(): normalize_text(value).lower()
        for key, value in sorted((condition_scope or {}).items())
        if str(key).strip() and normalize_text(value)
    }
    return json.dumps(normalized_scope, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class EvidenceExtractor:
    def __init__(
        self,
        handoff: Optional[EvidenceExtractorHandoff] = None,
        llm: Any = None,
    ) -> None:
        self.handoff = handoff or EvidenceExtractorHandoff()
        self.llm = llm

    def run(
        self,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        paper_record: PaperRecord,
        section_index: SectionIndex,
    ) -> List[EvidenceItem]:
        evidence_items: List[EvidenceItem] = []
        abstract_view = self._build_abstract_view(paper_record)
        if abstract_view is not None:
            evidence_items.extend(
                self._extract_from_section(
                    task_spec=task_spec,
                    entity_pack=entity_pack,
                    paper_record=paper_record,
                    section_view=abstract_view,
                )
            )

        evidence_is_weak = not any(item.role in CLAIM_ELIGIBLE_ROLES for item in evidence_items)
        missing_conditions = self._has_missing_conditions(task_spec=task_spec, evidence_items=evidence_items)
        preferred_sections = self.handoff.read_preferred_sections(
            paper_record=paper_record,
            section_index=section_index,
            task_spec=task_spec,
            evidence_is_weak=evidence_is_weak,
            missing_conditions=missing_conditions,
        )
        for section_view in preferred_sections:
            evidence_items.extend(
                self._extract_from_section(
                    task_spec=task_spec,
                    entity_pack=entity_pack,
                    paper_record=paper_record,
                    section_view=section_view,
                )
            )

        deduped_items = self._dedupe_items(evidence_items)
        return self._backfill_conditions(task_spec=task_spec, evidence_items=deduped_items)

    __call__ = run

    def _build_abstract_view(self, paper_record: PaperRecord) -> Optional[SectionTextView]:
        abstract_text = normalize_text(paper_record.abstract)
        if not abstract_text:
            return None
        return SectionTextView(
            paper_id=paper_record.paper_id,
            section_id="sec_abstract",
            section_type="abstract",
            heading="Abstract",
            text=abstract_text,
            fulltext_char_start=0,
            fulltext_char_end=len(abstract_text),
        )

    def _extract_from_section(
        self,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        paper_record: PaperRecord,
        section_view: SectionTextView,
    ) -> List[EvidenceItem]:
        evidence_items: List[EvidenceItem] = []
        local_index = 0
        for span_start, span_end, snippet in self._iter_snippet_spans(section_view.text):
            conditions = self._extract_conditions(snippet=snippet, entity_pack=entity_pack)
            decision = self._classify_roles(
                snippet=snippet,
                task_spec=task_spec,
                entity_pack=entity_pack,
                section_type=section_view.section_type,
            )
            for role in ROLE_ORDER:
                if role == "condition":
                    if not conditions:
                        continue
                    evidence_items.append(
                        self._make_evidence_item(
                            paper_record=paper_record,
                            section_view=section_view,
                            span_start=span_start,
                            span_end=span_end,
                            snippet=snippet,
                            local_index=local_index,
                            role=role,
                            claim_polarity="neutral",
                            conditions=conditions,
                            entity_mentions=decision.entity_mentions,
                            metric_mentions=decision.metric_mentions,
                            note="Rule-based condition extraction.",
                        )
                    )
                    local_index += 1
                    continue
                if role not in decision.roles:
                    continue
                evidence_items.append(
                    self._make_evidence_item(
                        paper_record=paper_record,
                        section_view=section_view,
                        span_start=span_start,
                        span_end=span_end,
                        snippet=snippet,
                        local_index=local_index,
                        role=role,
                        claim_polarity=decision.claim_polarity,
                        conditions=conditions,
                        entity_mentions=decision.entity_mentions,
                        metric_mentions=decision.metric_mentions,
                        note=decision.notes,
                    )
                )
                local_index += 1
        return evidence_items

    def _make_evidence_item(
        self,
        *,
        paper_record: PaperRecord,
        section_view: SectionTextView,
        span_start: int,
        span_end: int,
        snippet: str,
        local_index: int,
        role: str,
        claim_polarity: str,
        conditions: Dict[str, str],
        entity_mentions: Sequence[str],
        metric_mentions: Sequence[str],
        note: Optional[str],
    ) -> EvidenceItem:
        source_layer = SOURCE_LAYER_BY_SECTION.get(section_view.section_type, "fulltext")
        evidence_id = f"{paper_record.paper_id}:{section_view.section_id}:{local_index}"
        return EvidenceItem(
            evidence_id=evidence_id,
            paper_id=paper_record.paper_id,
            doi=paper_record.doi,
            section_id=section_view.section_id,
            section_type=section_view.section_type,
            role=role,
            snippet=snippet,
            source_span={"start": span_start, "end": span_end},
            source_layer=source_layer,
            claim_polarity=claim_polarity,
            conditions=conditions,
            condition_source_refs=[],
            metric_mentions=list(metric_mentions),
            entity_mentions=list(entity_mentions),
            extraction_confidence=self._estimate_confidence(
                role=role,
                section_type=section_view.section_type,
                source_layer=source_layer,
                metric_mentions=metric_mentions,
                entity_mentions=entity_mentions,
                conditions=conditions,
            ),
            extraction_notes=note or f"Heuristic {role} extraction from {section_view.section_type}.",
        )

    def _iter_snippet_spans(self, text: str) -> Iterable[Tuple[int, int, str]]:
        sentence_pattern = re.compile(r"(?s)[^\n.!?;]+(?:[.!?;]+|$)")
        for match in sentence_pattern.finditer(text or ""):
            raw_snippet = match.group(0)
            snippet = raw_snippet.strip()
            if len(snippet) < 12:
                continue
            lead_trim = len(raw_snippet) - len(raw_snippet.lstrip())
            start = match.start() + lead_trim
            end = start + len(snippet)
            yield start, end, snippet

    def _classify_roles(
        self,
        *,
        snippet: str,
        task_spec: TaskSpec,
        entity_pack: EntityPack,
        section_type: str,
    ) -> _RoleDecision:
        heuristic_roles = self._heuristic_roles(snippet=snippet, section_type=section_type)
        llm_payload = self._classify_with_llm(snippet=snippet, task_spec=task_spec, section_type=section_type)
        roles = set(heuristic_roles)
        if isinstance(llm_payload, dict):
            for role in llm_payload.get("roles") or []:
                if role in CLAIM_ELIGIBLE_ROLES:
                    roles.add(role)
        ordered_roles = [role for role in ROLE_ORDER if role in roles and role != "condition"]
        entity_mentions = self._extract_entity_mentions(snippet=snippet, entity_pack=entity_pack)
        metric_mentions = self._extract_metric_mentions(snippet)
        if isinstance(llm_payload, dict):
            entity_mentions = self._merge_unique(entity_mentions, llm_payload.get("entity_mentions"))
            metric_mentions = self._merge_unique(metric_mentions, llm_payload.get("metric_mentions"))
            claim_polarity = (
                llm_payload.get("claim_polarity")
                if llm_payload.get("claim_polarity") in {"support", "oppose", "neutral"}
                else None
            )
            note = normalize_text(llm_payload.get("notes")) or None
        else:
            claim_polarity = None
            note = None
        if claim_polarity is None:
            claim_polarity = self._infer_polarity(snippet=snippet, roles=ordered_roles)
        if not ordered_roles and section_type in {"abstract", "results", "discussion"} and (entity_mentions or metric_mentions):
            ordered_roles = ["observation"]
        if not ordered_roles:
            ordered_roles = []
        note = note or f"Heuristic role extraction from {section_type}."
        return _RoleDecision(
            roles=ordered_roles,
            claim_polarity=claim_polarity,
            entity_mentions=entity_mentions,
            metric_mentions=metric_mentions,
            notes=note,
        )

    def _heuristic_roles(self, *, snippet: str, section_type: str) -> List[str]:
        roles: List[str] = []
        if any(pattern.search(snippet) for pattern in LIMITATION_PATTERNS):
            roles.append("limitation")
        if any(pattern.search(snippet) for pattern in MECHANISM_PATTERNS):
            roles.append("mechanism")
        has_metric = bool(self._extract_metric_mentions(snippet))
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\s*%?\b", snippet))
        if section_type in {"abstract", "results", "discussion", "conclusion", "limitations"} and (
            has_metric
            or has_number
            or any(pattern.search(snippet) for pattern in OBSERVATION_PATTERNS)
            or any(pattern.search(snippet) for pattern in COMPARISON_PATTERNS)
        ):
            roles.append("observation")
        return self._merge_unique([], roles)

    def _infer_polarity(self, *, snippet: str, roles: Sequence[str]) -> str:
        if "limitation" in roles:
            return "oppose"
        if any(pattern.search(snippet) for pattern in NEGATIVE_EFFECT_PATTERNS):
            return "oppose"
        if roles:
            return "support"
        return "neutral"

    def _extract_conditions(self, *, snippet: str, entity_pack: EntityPack) -> Dict[str, str]:
        scope: Dict[str, str] = {}
        lower_snippet = snippet.lower()
        for entity in entity_pack.entities:
            axis = ENTITY_TYPE_TO_AXIS.get(entity.entity_type)
            if not axis or axis in scope:
                continue
            candidates = self._merge_unique(
                [entity.canonical_name],
                [entity.mention, *(entity.aliases or [])],
            )
            for candidate in candidates:
                normalized_candidate = normalize_text(candidate).lower()
                if normalized_candidate and normalized_candidate in lower_snippet:
                    scope[axis] = normalized_candidate
                    break
        for solvent_name in SOLVENT_NAMES:
            if re.search(rf"\b{re.escape(solvent_name)}\b", snippet, re.I):
                scope.setdefault("solvent", solvent_name)
        for axis, patterns in CONDITION_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(snippet)
                if not match:
                    continue
                value = normalize_text(match.group("value")).lower()
                if value:
                    scope.setdefault(axis, value)
                break
        for axis, patterns in PREFIXED_CONDITION_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(snippet)
                if not match:
                    continue
                value = normalize_text(match.group("value")).lower().rstrip(".,;:")
                if value:
                    scope.setdefault(axis, value)
                break
        return scope

    def _extract_entity_mentions(self, *, snippet: str, entity_pack: EntityPack) -> List[str]:
        mentions: List[str] = []
        lower_snippet = snippet.lower()
        for entity in entity_pack.entities:
            candidate_names = self._merge_unique(
                [entity.canonical_name],
                [entity.mention, *(entity.aliases or [])],
            )
            for candidate_name in candidate_names:
                token = normalize_text(candidate_name).lower()
                if token and token in lower_snippet:
                    mentions.append(entity.canonical_name or candidate_name)
                    break
        if mentions:
            return self._merge_unique([], mentions)
        fallback_tokens = [
            token
            for token in CHEMICAL_TOKEN_PATTERN.findall(snippet)
            if len(token) > 1 and not token.isupper()
        ]
        return self._merge_unique([], fallback_tokens[:3])

    def _extract_metric_mentions(self, snippet: str) -> List[str]:
        mentions: List[str] = []
        for family, pattern in METRIC_PATTERNS:
            if pattern.search(snippet):
                mentions.append(family)
        return self._merge_unique([], mentions)

    def _classify_with_llm(
        self,
        *,
        snippet: str,
        task_spec: TaskSpec,
        section_type: str,
    ) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None
        messages = [
            {"role": "system", "content": EVIDENCE_EXTRACTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_evidence_extractor_user_prompt(
                    question_type=task_spec.question_type,
                    section_type=section_type,
                    snippet=snippet,
                ),
            },
        ]
        try:
            raw_response = invoke_llm(self.llm, messages)
        except Exception:
            return None
        return self._repair_llm_classification(parse_json_object(raw_response))

    def _repair_llm_classification(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        roles = [
            role
            for role in payload.get("roles") or []
            if role in CLAIM_ELIGIBLE_ROLES
        ]
        claim_polarity = payload.get("claim_polarity")
        if claim_polarity not in {"support", "oppose", "neutral"}:
            claim_polarity = None
        entity_mentions = self._merge_unique([], payload.get("entity_mentions"))
        metric_mentions = self._merge_unique([], payload.get("metric_mentions"))
        notes = normalize_text(payload.get("notes")) or None
        return {
            "roles": roles,
            "claim_polarity": claim_polarity,
            "entity_mentions": entity_mentions,
            "metric_mentions": metric_mentions,
            "notes": notes,
        }

    def _estimate_confidence(
        self,
        *,
        role: str,
        section_type: str,
        source_layer: str,
        metric_mentions: Sequence[str],
        entity_mentions: Sequence[str],
        conditions: Dict[str, str],
    ) -> float:
        confidence = 0.42
        if role == "condition":
            confidence += 0.18
        if role == "mechanism":
            confidence += 0.14
        if role == "observation":
            confidence += 0.12
        if section_type in {"results", "discussion"}:
            confidence += 0.18
        elif section_type == "abstract":
            confidence += 0.08
        elif section_type == "methods":
            confidence += 0.1
        if source_layer == "fulltext":
            confidence += 0.05
        if metric_mentions:
            confidence += 0.08
        if entity_mentions:
            confidence += 0.05
        if conditions and role != "condition":
            confidence += 0.04
        return round(max(0.05, min(confidence, 0.95)), 2)

    def _has_missing_conditions(self, *, task_spec: TaskSpec, evidence_items: Sequence[EvidenceItem]) -> bool:
        required_axes = set(task_spec.required_condition_axes or [])
        if not required_axes:
            return False
        if not evidence_items:
            return True
        for item in evidence_items:
            if item.role not in CLAIM_ELIGIBLE_ROLES:
                continue
            if not required_axes.issubset(set(item.conditions)):
                return True
        return False

    def _dedupe_items(self, evidence_items: Sequence[EvidenceItem]) -> List[EvidenceItem]:
        ordered: List[EvidenceItem] = []
        seen: Dict[Tuple[str, str, int, int, str], EvidenceItem] = {}
        for item in evidence_items:
            key = (
                item.section_id,
                item.role,
                item.source_span.start,
                item.source_span.end,
                normalize_text(item.snippet).lower(),
            )
            existing = seen.get(key)
            if existing is None or item.extraction_confidence > existing.extraction_confidence:
                seen[key] = item
        for item in evidence_items:
            key = (
                item.section_id,
                item.role,
                item.source_span.start,
                item.source_span.end,
                normalize_text(item.snippet).lower(),
            )
            if seen.get(key) is item:
                ordered.append(item)
        return ordered

    def _backfill_conditions(
        self,
        *,
        task_spec: TaskSpec,
        evidence_items: Sequence[EvidenceItem],
    ) -> List[EvidenceItem]:
        condition_items = [item for item in evidence_items if item.role == "condition" and item.conditions]
        if not condition_items:
            return list(evidence_items)
        required_axes = set(task_spec.required_condition_axes or [])
        enriched_items: List[EvidenceItem] = []
        for item in evidence_items:
            if item.role not in CLAIM_ELIGIBLE_ROLES:
                enriched_items.append(item)
                continue
            if item.section_type == "methods":
                enriched_items.append(item)
                continue
            needs_backfill = (not item.conditions) or bool(required_axes.difference(item.conditions))
            if not needs_backfill:
                enriched_items.append(item)
                continue
            merged_conditions = dict(item.conditions)
            source_refs = list(item.condition_source_refs)
            ranked_sources = sorted(
                condition_items,
                key=lambda candidate: (
                    self._condition_source_rank(target=item, source=candidate),
                    candidate.source_span.start,
                    candidate.evidence_id,
                ),
            )
            for source_item in ranked_sources:
                if source_item.evidence_id == item.evidence_id:
                    continue
                added = False
                for axis, value in source_item.conditions.items():
                    if axis in merged_conditions:
                        continue
                    if required_axes and axis not in required_axes and merged_conditions:
                        continue
                    merged_conditions[axis] = value
                    added = True
                if added and source_item.evidence_id not in source_refs:
                    source_refs.append(source_item.evidence_id)
                if required_axes and required_axes.issubset(set(merged_conditions)):
                    break
            if merged_conditions == item.conditions and source_refs == item.condition_source_refs:
                enriched_items.append(item)
                continue
            notes = item.extraction_notes or "Heuristic extraction."
            if source_refs:
                notes = f"{notes} Condition scope backfilled from {'; '.join(source_refs)}."
            enriched_items.append(
                item.model_copy(
                    update={
                        "conditions": merged_conditions,
                        "condition_source_refs": source_refs,
                        "extraction_notes": notes,
                    }
                )
            )
        return enriched_items

    def _condition_source_rank(self, *, target: EvidenceItem, source: EvidenceItem) -> Tuple[int, int, int]:
        overlap = len(set(target.entity_mentions).intersection(source.entity_mentions))
        same_section_penalty = 1 if target.section_id == source.section_id else 0
        priority = SECTION_PRIORITY.get(source.section_type, 99)
        return priority, same_section_penalty, -overlap

    def _merge_unique(self, baseline: Sequence[str], extra: Optional[Sequence[Any]]) -> List[str]:
        merged: List[str] = []
        seen = set()
        values = list(baseline or [])
        if isinstance(extra, Sequence) and not isinstance(extra, (str, bytes)):
            values.extend(extra)
        for value in values:
            text = normalize_text(value)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged


class ClaimMiner:
    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def run(
        self,
        evidence_items: Sequence[EvidenceItem],
        *,
        task_spec: Optional[TaskSpec] = None,
    ) -> List[ClaimRecord]:
        clusters: Dict[Tuple[str, str, str, str, str], List[EvidenceItem]] = defaultdict(list)
        for item in evidence_items:
            if item.role not in CLAIM_ELIGIBLE_ROLES:
                continue
            if item.extraction_confidence <= 0.0:
                continue
            question_type = task_spec.question_type if task_spec is not None else self._infer_question_type(item)
            main_entity = self._derive_main_entity(item=item, question_type=question_type)
            relation_type = self._derive_relation_type(item=item)
            metric_family = self._derive_metric_family(item=item)
            condition_signature = build_condition_signature(item.conditions)
            cluster_key = (main_entity, relation_type, metric_family, condition_signature, question_type)
            clusters[cluster_key].append(item)

        claim_records: List[ClaimRecord] = []
        prefix_counts: Dict[str, int] = defaultdict(int)
        for cluster_key in sorted(clusters):
            cluster_items = sorted(
                clusters[cluster_key],
                key=lambda item: (
                    item.paper_id,
                    item.section_id,
                    item.source_span.start,
                    item.source_span.end,
                    item.evidence_id,
                ),
            )
            claim_record = self._build_claim(cluster_key=cluster_key, evidence_items=cluster_items, prefix_counts=prefix_counts)
            claim_records.append(claim_record)
        return claim_records

    __call__ = run

    def _build_claim(
        self,
        *,
        cluster_key: Tuple[str, str, str, str, str],
        evidence_items: Sequence[EvidenceItem],
        prefix_counts: Dict[str, int],
    ) -> ClaimRecord:
        main_entity, relation_type, metric_family, condition_signature, question_type = cluster_key
        representative = max(
            evidence_items,
            key=lambda item: (
                item.claim_polarity != "oppose",
                item.extraction_confidence,
                -item.source_span.start,
            ),
        )
        condition_scope = dict(representative.conditions)
        supporting_ids = [item.evidence_id for item in evidence_items if item.claim_polarity != "oppose"]
        opposing_ids = [item.evidence_id for item in evidence_items if item.claim_polarity == "oppose"]
        claim_type = self._select_claim_type(question_type=question_type, representative=representative)
        claim_text = self._generate_claim_text(
            claim_type=claim_type,
            main_entity=main_entity,
            relation_type=relation_type,
            metric_family=metric_family,
            condition_scope=condition_scope,
            representative=representative,
            evidence_items=evidence_items,
        )
        prefix = self._claim_id_prefix(
            main_entity=main_entity,
            relation_type=relation_type,
            metric_family=metric_family,
            condition_signature=condition_signature,
        )
        prefix_counts[prefix] += 1
        claim_id = f"{prefix}:{prefix_counts[prefix]}"
        claim_confidence = self._estimate_claim_confidence(
            evidence_items=evidence_items,
            supporting_count=len(supporting_ids),
            opposing_count=len(opposing_ids),
        )
        provenance_notes = (
            f"Built from {len(evidence_items)} evidence item(s); "
            f"support={len(supporting_ids)} oppose={len(opposing_ids)}; "
            f"question_type={question_type}."
        )
        return ClaimRecord(
            claim_id=claim_id,
            claim_type=claim_type,
            section_id=representative.section_id,
            claim_text=claim_text,
            main_entity=main_entity,
            relation_type=relation_type,
            metric_family=metric_family,
            condition_scope=condition_scope,
            condition_signature=condition_signature,
            supporting_evidence_ids=supporting_ids,
            opposing_evidence_ids=opposing_ids,
            status="draft",
            claim_confidence=claim_confidence,
            cluster_size=len(evidence_items),
            provenance_notes=provenance_notes,
        )

    def _derive_main_entity(self, *, item: EvidenceItem, question_type: str) -> str:
        mentions = item.entity_mentions or []
        if question_type == "comparison" and len(mentions) >= 2:
            return " vs ".join(mentions[:2])
        if mentions:
            return mentions[0]
        if item.conditions.get("catalyst"):
            return item.conditions["catalyst"]
        if item.conditions.get("material"):
            return item.conditions["material"]
        return slugify(item.paper_id, max_length=32)

    def _derive_relation_type(self, *, item: EvidenceItem) -> str:
        snippet = item.snippet
        if item.role == "mechanism" or any(pattern.search(snippet) for pattern in MECHANISM_PATTERNS):
            return "mechanistic_explanation"
        if any(pattern.search(snippet) for pattern in COMPARISON_PATTERNS):
            return "comparison"
        if item.role in {"observation", "limitation"} or any(pattern.search(snippet) for pattern in OBSERVATION_PATTERNS):
            return "effect"
        return "association"

    def _derive_metric_family(self, *, item: EvidenceItem) -> str:
        metric_mentions = item.metric_mentions or []
        for metric_family in METRIC_FAMILY_PRIORITY:
            if metric_family in metric_mentions:
                return metric_family
        return metric_mentions[0] if metric_mentions else "unspecified"

    def _infer_question_type(self, item: EvidenceItem) -> str:
        if any(pattern.search(item.snippet) for pattern in COMPARISON_PATTERNS):
            return "comparison"
        if item.role == "mechanism":
            return "mechanism"
        if item.role == "limitation":
            return "causal"
        return "fact"

    def _select_claim_type(self, *, question_type: str, representative: EvidenceItem) -> str:
        if question_type == "frontier":
            return "frontier_summary"
        if question_type in {"causal", "mechanism", "comparison"}:
            return question_type
        if representative.role == "mechanism":
            return "mechanism"
        return "fact"

    def _generate_claim_text(
        self,
        *,
        claim_type: str,
        main_entity: str,
        relation_type: str,
        metric_family: str,
        condition_scope: Dict[str, str],
        representative: EvidenceItem,
        evidence_items: Sequence[EvidenceItem],
    ) -> str:
        llm_text = self._generate_with_llm(
            claim_type=claim_type,
            main_entity=main_entity,
            relation_type=relation_type,
            metric_family=metric_family,
            condition_scope=condition_scope,
            representative=representative,
            evidence_items=evidence_items,
        )
        if llm_text:
            return llm_text
        condition_text = self._render_condition_text(condition_scope)
        entity_text = main_entity or "This evidence"
        metric_text = metric_family.replace("_", " ") if metric_family != "unspecified" else "performance"
        if claim_type == "comparison":
            base = f"{entity_text} shows a comparison-dependent difference in {metric_text}"
        elif claim_type == "mechanism":
            base = f"{entity_text} is linked to {metric_text} through a mechanistic pathway"
        elif claim_type == "causal":
            base = f"{entity_text} affects {metric_text}"
        elif claim_type == "frontier_summary":
            base = f"Recent evidence connects {entity_text} to {metric_text}"
        else:
            base = f"{entity_text} is associated with {metric_text}"
        if condition_text:
            base = f"{base} under {condition_text}"
        return base.rstrip(".") + "."

    def _generate_with_llm(
        self,
        *,
        claim_type: str,
        main_entity: str,
        relation_type: str,
        metric_family: str,
        condition_scope: Dict[str, str],
        representative: EvidenceItem,
        evidence_items: Sequence[EvidenceItem],
    ) -> Optional[str]:
        if self.llm is None:
            return None
        messages = [
            {"role": "system", "content": CLAIM_MINER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_claim_miner_user_prompt(
                    claim_type=claim_type,
                    main_entity=main_entity,
                    relation_type=relation_type,
                    metric_family=metric_family,
                    condition_scope=condition_scope,
                    representative_snippet=representative.snippet,
                    supporting_evidence=[item.snippet for item in evidence_items[:3]],
                ),
            },
        ]
        try:
            raw_response = invoke_llm(self.llm, messages)
        except Exception:
            return None
        parsed = parse_json_object(raw_response)
        claim_text = normalize_text((parsed or {}).get("claim_text"))
        if claim_text and condition_scope:
            lower_claim_text = claim_text.lower()
            if not any(normalize_text(value).lower() in lower_claim_text for value in condition_scope.values()):
                claim_text = f"{claim_text.rstrip('.')} under {self._render_condition_text(condition_scope)}."
        return claim_text or None

    def _render_condition_text(self, condition_scope: Dict[str, str]) -> str:
        if not condition_scope:
            return ""
        return ", ".join(f"{axis}={value}" for axis, value in sorted(condition_scope.items()))

    def _claim_id_prefix(
        self,
        *,
        main_entity: str,
        relation_type: str,
        metric_family: str,
        condition_signature: str,
    ) -> str:
        digest = hashlib.sha1(
            "|".join([main_entity, relation_type, metric_family, condition_signature]).encode("utf-8")
        ).hexdigest()[:10]
        readable = slugify(main_entity or "claim", max_length=24)
        return f"claim_{readable}_{digest}"

    def _estimate_claim_confidence(
        self,
        *,
        evidence_items: Sequence[EvidenceItem],
        supporting_count: int,
        opposing_count: int,
    ) -> float:
        if not evidence_items:
            return 0.0
        avg_confidence = sum(item.extraction_confidence for item in evidence_items) / len(evidence_items)
        confidence = avg_confidence
        if supporting_count > 1:
            confidence += 0.05
        if opposing_count:
            confidence -= 0.08
        return round(max(0.05, min(confidence, 0.95)), 2)


class EvidenceLedgerBuilder:
    def run(
        self,
        *,
        claims: Sequence[ClaimRecord],
        evidence_items: Sequence[EvidenceItem],
    ) -> EvidenceLedger:
        claim_list = list(claims)
        evidence_list = list(evidence_items)
        return EvidenceLedger(
            claims=claim_list,
            evidence_items=evidence_list,
            claim_index={claim.claim_id: index for index, claim in enumerate(claim_list)},
            evidence_index={item.evidence_id: index for index, item in enumerate(evidence_list)},
            cluster_stats={
                "claim_count": len(claim_list),
                "evidence_count": len(evidence_list),
                "support_edge_count": sum(len(claim.supporting_evidence_ids) for claim in claim_list),
                "oppose_edge_count": sum(len(claim.opposing_evidence_ids) for claim in claim_list),
            },
            ledger_notes=[
                "Module 3 stores draft claims with explicit support and opposition links.",
                "Evidence items remain independent rows and may be reused across claims.",
            ],
        )

    __call__ = run
