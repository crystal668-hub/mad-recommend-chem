from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qa.retrieval_state import ClaimRecord, EvidenceItem, EvidenceLedger
from qa.retrieval_utils import normalize_text


POSITIVE_DIRECTION_PATTERNS = (
    re.compile(r"\bincreas(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\bimprov(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\benhanc(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\bhigher than\b", re.I),
    re.compile(r"\boutperform(?:ed|s|ing)?\b", re.I),
    re.compile(r"\bbetter than\b", re.I),
    re.compile(r"\bpromot(?:e|ed|es|ing)\b", re.I),
)
NEGATIVE_DIRECTION_PATTERNS = (
    re.compile(r"\bdecreas(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\breduc(?:e|ed|es|ing)\b", re.I),
    re.compile(r"\blower than\b", re.I),
    re.compile(r"\bworse\b", re.I),
    re.compile(r"\bsuppress(?:ed|es|ing)?\b", re.I),
    re.compile(r"\bdid not\b", re.I),
    re.compile(r"\bdoes not\b", re.I),
    re.compile(r"\bfailed to\b", re.I),
    re.compile(r"\bnot observed\b", re.I),
    re.compile(r"\bno significant\b", re.I),
)
SPECULATIVE_PATTERNS = (
    re.compile(r"\bmay\b", re.I),
    re.compile(r"\bmight\b", re.I),
    re.compile(r"\blikely\b", re.I),
    re.compile(r"\bpossibly\b", re.I),
    re.compile(r"\bsuggest(?:s|ed)?\b", re.I),
    re.compile(r"\bproposed\b", re.I),
    re.compile(r"\bappears? to\b", re.I),
)
ABSOLUTE_PATTERNS = (
    re.compile(r"\balways\b", re.I),
    re.compile(r"\bnever\b", re.I),
    re.compile(r"\ball\b", re.I),
    re.compile(r"\bproves?\b", re.I),
    re.compile(r"\bdefinitive(?:ly)?\b", re.I),
    re.compile(r"\bguarantee(?:s|d)?\b", re.I),
)
HIGH_RISK_CLAIM_TYPES = {"causal", "mechanism", "comparison"}


def build_evidence_lookup(evidence_ledger: EvidenceLedger) -> Dict[str, EvidenceItem]:
    return {item.evidence_id: item for item in evidence_ledger.evidence_items}


def valid_evidence_items(evidence_ids: Sequence[str], evidence_lookup: Dict[str, EvidenceItem]) -> List[EvidenceItem]:
    return [evidence_lookup[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_lookup]


def traceable_evidence_items(evidence_ids: Sequence[str], evidence_lookup: Dict[str, EvidenceItem]) -> List[EvidenceItem]:
    items = valid_evidence_items(evidence_ids=evidence_ids, evidence_lookup=evidence_lookup)
    return [item for item in items if normalize_text(item.snippet)]


def normalize_condition_signature(condition_scope: Dict[str, str]) -> str:
    normalized_scope = {
        str(key).strip().lower(): normalize_text(value).lower()
        for key, value in sorted((condition_scope or {}).items())
        if str(key).strip() and normalize_text(value)
    }
    return json.dumps(normalized_scope, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def common_condition_scope(evidence_items: Sequence[EvidenceItem]) -> Dict[str, str]:
    if not evidence_items:
        return {}
    scope: Dict[str, str] = {}
    axis_values: Dict[str, set[str]] = {}
    for item in evidence_items:
        for axis, value in (item.conditions or {}).items():
            axis_values.setdefault(axis, set()).add(normalize_text(value).lower())
    for axis, values in axis_values.items():
        if len(values) == 1:
            scope[axis] = next(iter(values))
    return scope


def dominant_condition_scope(evidence_items: Sequence[EvidenceItem]) -> Dict[str, str]:
    if not evidence_items:
        return {}
    dominant: Dict[str, str] = {}
    counters: Dict[str, Counter[str]] = {}
    for item in evidence_items:
        for axis, value in (item.conditions or {}).items():
            counters.setdefault(axis, Counter())[normalize_text(value).lower()] += 1
    for axis, counter in counters.items():
        dominant[axis] = counter.most_common(1)[0][0]
    return dominant


def compatible_with_scope(item: EvidenceItem, condition_scope: Dict[str, str]) -> bool:
    if not condition_scope:
        return True
    for axis, value in condition_scope.items():
        item_value = normalize_text(item.conditions.get(axis)).lower()
        if not item_value or item_value != normalize_text(value).lower():
            return False
    return True


def infer_claim_direction(
    claim: ClaimRecord,
    *,
    supporting_items: Sequence[EvidenceItem],
    opposing_items: Sequence[EvidenceItem],
) -> str:
    score = 0
    texts = [claim.claim_text]
    texts.extend(item.snippet for item in supporting_items)
    texts.extend(item.snippet for item in opposing_items)
    for text in texts:
        negative_match = any(pattern.search(text or "") for pattern in NEGATIVE_DIRECTION_PATTERNS)
        positive_match = any(pattern.search(text or "") for pattern in POSITIVE_DIRECTION_PATTERNS)
        if negative_match and not positive_match:
            score -= 1
        elif positive_match and not negative_match:
            score += 1
        elif negative_match and positive_match:
            score -= 1
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    if supporting_items and opposing_items:
        return "mixed"
    return "neutral"


def directions_are_opposed(left_direction: str, right_direction: str) -> bool:
    pair = {left_direction, right_direction}
    return pair == {"positive", "negative"}


def contains_absolute_language(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in ABSOLUTE_PATTERNS)


def speculative_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SPECULATIVE_PATTERNS)


def render_condition_text(condition_scope: Dict[str, str]) -> str:
    if not condition_scope:
        return ""
    return ", ".join(f"{axis}={value}" for axis, value in sorted(condition_scope.items()))


def build_conservative_claim_text(
    claim: ClaimRecord,
    *,
    condition_scope: Dict[str, str],
    direction: str,
    downgrade: bool,
    speculative: bool,
) -> str:
    entity_text = claim.main_entity or "The evidence"
    metric_text = claim.metric_family.replace("_", " ") if claim.metric_family != "unspecified" else "performance"
    if claim.claim_type == "mechanism":
        if speculative:
            base = f"{entity_text} may influence {metric_text} through a proposed mechanism"
        else:
            base = f"{entity_text} influences {metric_text} through a mechanism"
    elif claim.claim_type == "causal":
        base = f"{entity_text} may affect {metric_text}" if downgrade else f"{entity_text} affects {metric_text}"
    elif claim.claim_type == "comparison":
        base = f"{entity_text} shows a condition-bound difference in {metric_text}"
    elif claim.claim_type == "frontier_summary":
        base = f"Recent evidence connects {entity_text} to {metric_text}"
    else:
        base = f"{entity_text} is associated with {metric_text}"

    if direction == "positive" and claim.claim_type in {"fact", "causal"}:
        base = base.replace(metric_text, f"higher {metric_text}")
    elif direction == "negative" and claim.claim_type in {"fact", "causal"}:
        base = base.replace(metric_text, f"lower {metric_text}")

    if speculative and "may" not in base.lower():
        base = base.replace(" is ", " may be ")
        base = base.replace(" affects ", " may affect ")
        base = base.replace(" shows ", " may show ")

    condition_text = render_condition_text(condition_scope)
    if condition_text:
        base = f"{base} under {condition_text}"
    return base.rstrip(".") + "."


def condition_scope_is_ambiguous(
    claim: ClaimRecord,
    *,
    required_axes: Sequence[str],
    supporting_items: Sequence[EvidenceItem],
) -> bool:
    required = {axis for axis in required_axes if axis}
    claim_axes = set(claim.condition_scope)
    if required and not required.issubset(claim_axes):
        return True
    support_axes = set()
    for item in supporting_items:
        support_axes.update(item.conditions)
    return bool(support_axes and not support_axes.issubset(claim_axes))


def shared_and_differing_axes(
    left_scope: Dict[str, str],
    right_scope: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    shared_axes: List[str] = []
    differing_axes: List[str] = []
    for axis in sorted(set(left_scope).union(right_scope)):
        left_value = normalize_text(left_scope.get(axis)).lower()
        right_value = normalize_text(right_scope.get(axis)).lower()
        if left_value and right_value and left_value == right_value:
            shared_axes.append(axis)
        elif left_value or right_value:
            differing_axes.append(axis)
    return shared_axes, differing_axes


def scopes_highly_overlap(left_scope: Dict[str, str], right_scope: Dict[str, str]) -> bool:
    left_signature = normalize_condition_signature(left_scope)
    right_signature = normalize_condition_signature(right_scope)
    if left_signature == right_signature:
        return True
    shared_axes, differing_axes = shared_and_differing_axes(left_scope, right_scope)
    if not differing_axes and shared_axes:
        return True
    if not left_scope and not right_scope:
        return True
    return False


def same_topic(left_claim: ClaimRecord, right_claim: ClaimRecord) -> bool:
    return (
        normalize_text(left_claim.main_entity).lower() == normalize_text(right_claim.main_entity).lower()
        and normalize_text(left_claim.metric_family).lower() == normalize_text(right_claim.metric_family).lower()
        and normalize_text(left_claim.relation_type).lower() == normalize_text(right_claim.relation_type).lower()
    )


def conflict_pair_key(left_claim_id: str, right_claim_id: str) -> Tuple[str, str]:
    return tuple(sorted((left_claim_id, right_claim_id)))


def top_evidence_refs(evidence_items: Iterable[EvidenceItem], limit: int = 3) -> List[str]:
    refs: List[str] = []
    for item in evidence_items:
        if item.evidence_id in refs:
            continue
        refs.append(item.evidence_id)
        if len(refs) >= limit:
            break
    return refs
