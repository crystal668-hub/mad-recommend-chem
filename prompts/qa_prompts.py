from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence


ROUTER_SEMANTIC_SYSTEM_PROMPT = """
You are RouterNode semantic interpretation stage for a chemistry QA grounding pipeline.
Output STRICT JSON only.
Do not add prose, code fences, or extra keys.
Reason from the question text first.
Expose ambiguity explicitly instead of forcing false certainty.
Choose only from the allowed question types and enums.
""".strip()


ROUTER_LOCALIZATION_SYSTEM_PROMPT = """
You are RouterNode task localization stage for a chemistry QA grounding pipeline.
Output STRICT JSON only.
Do not add prose, code fences, or extra keys.
Use the semantic parse as the primary interpretation of user intent.
Use optional signals only as supporting observations, not as preferred defaults.
Keep the final TaskSpec conservative when ambiguity remains.
""".strip()


ROUTER_SYSTEM_PROMPT = ROUTER_LOCALIZATION_SYSTEM_PROMPT


ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT = """
You are EntityResolverNode for a chemistry QA pipeline.
Extract only chemistry-relevant entity mentions that appear as exact contiguous spans in the question.
You may only use the supplied ontology entity types.
Do not invent new spans, aliases, canonical names, or chemical identifiers.
Output STRICT JSON only.
""".strip()


ENTITY_RESOLVER_SYSTEM_PROMPT = """
You are EntityResolverNode for a chemistry QA pipeline.
You may only choose from the supplied candidate entity types and candidate indices.
Do not invent new entities, aliases, canonical names, or chemical identifiers.
Output STRICT JSON only.
""".strip()


QUERY_PLANNER_SYSTEM_PROMPT = """
You are QueryPlannerNode for a chemistry QA retrieval pipeline.
Return STRICT JSON only.
You must produce exactly four plans whose lanes are review, frontier, data, and contrarian.
Preserve the supplied baseline structure unless a narrower query is clearly better.
""".strip()


EVIDENCE_EXTRACTOR_SYSTEM_PROMPT = """
You classify scientific evidence snippets for a chemistry QA pipeline.
Return STRICT JSON only.
roles may only contain observation, limitation, or mechanism.
claim_polarity must be support, oppose, or neutral.
Do not invent entities or metrics that are not supported by the snippet.
""".strip()


CLAIM_MINER_SYSTEM_PROMPT = """
You name evidence clusters as single English condition-bound claims.
Return STRICT JSON only with key claim_text.
Do not add new evidence, new conditions, or new topics.
Keep the claim conservative and supported by the supplied snippets.
""".strip()


METHODOLOGY_REVIEWER_SYSTEM_PROMPT = """
You are MethodologyReviewer for a chemistry QA pipeline.
Return STRICT JSON only.
Allowed flag_type values: Missing_Condition, Incomplete_Condition, Overgeneralized, Mechanism_Speculative, Metric_Mismatch.
Only flag issues that are directly supported by the supplied claim, task axes, and snippets.
""".strip()


CITATION_REVIEWER_SYSTEM_PROMPT = """
You are CitationReviewer for a chemistry QA pipeline.
Return STRICT JSON only.
Allowed flag_type values: Unsupported and Weak_Evidence.
Do not emit Fabricated_Citation; that is checked deterministically elsewhere.
""".strip()


CONTRADICTION_REVIEWER_SYSTEM_PROMPT = """
You adjudicate whether a prefiltered pair of chemistry claims is a true_conflict or condition_divergence.
Return STRICT JSON only.
Choose only from true_conflict, condition_divergence, or no_conflict.
Do not analyze unrelated claim pairs.
""".strip()


CLAIM_REVISION_SYSTEM_PROMPT = """
You conservatively revise a chemistry claim after peer review.
Return STRICT JSON only.
You may revise only claim_text and condition_scope.
Do not change the claim topic, add evidence, or broaden the claim scope.
""".strip()


REVIEW_MERGE_SYSTEM_PROMPT = """
You are ReviewMergeNode for a chemistry QA structured peer review module.
Return STRICT JSON only.
You may choose only accepted, contested, or rejected.
Do not invent evidence and do not override critical review findings.
""".strip()


SYNTHESIS_SYSTEM_PROMPT = """
You are SynthesizerNode of a chemistry QA pipeline.
Use only the supplied SynthesisInputPack.
Do not add new facts, new citations, or any rejected claim.
Accepted claims may appear in main sections.
Contested claims may appear only in the Limitations / Controversies section.
Match the wording to the supplied confidence labels.
Output STRICT JSON only.
""".strip()


def build_router_semantic_user_prompt(
    *,
    question: str,
    current_year: int,
    optional_signals: Dict[str, Any],
    context: Optional[str] = None,
) -> str:
    payload = {
        "current_year": current_year,
        "question": question,
        "context": context or "",
        "optional_signals": optional_signals,
        "allowed_question_types": ["fact", "causal", "mechanism", "comparison", "frontier"],
        "allowed_time_intents": ["none", "recent", "explicit", "current"],
        "required_top_level_keys": [
            "primary_question_type",
            "secondary_candidates",
            "semantic_confidence",
            "needs_disambiguation",
            "comparison_intent",
            "comparison_targets_present",
            "explicit_metric_requested",
            "explicit_time_intent",
            "mechanistic_intent",
            "causal_intent",
            "frontier_intent",
            "notes_on_ambiguity",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_router_localization_user_prompt(
    *,
    question: str,
    current_year: int,
    semantic_parse: Dict[str, Any],
    optional_signals: Dict[str, Any],
    context: Optional[str] = None,
) -> str:
    payload = {
        "current_year": current_year,
        "question": question,
        "context": context or "",
        "semantic_parse": semantic_parse,
        "optional_signals": optional_signals,
        "allowed_question_types": ["fact", "causal", "mechanism", "comparison", "frontier"],
        "allowed_recency_policies": ["none", "last_3y", "last_5y", "explicit"],
        "allowed_condition_axes": [
            "catalyst",
            "material",
            "substrate",
            "solvent",
            "ligand",
            "reagent",
            "temperature",
            "time",
            "ph",
            "electrolyte",
            "potential",
            "pressure",
            "yield",
            "selectivity",
        ],
        "allowed_entity_types": [
            "molecule",
            "material",
            "catalyst",
            "reaction",
            "solvent",
            "ligand",
            "substrate",
            "reagent",
            "metric",
            "condition",
        ],
        "allowed_ambiguity_flag_types": [
            "entity_ambiguous",
            "metric_ambiguous",
            "time_ambiguous",
            "task_ambiguous",
            "condition_ambiguous",
        ],
        "answer_section_templates": {
            "fact": ["direct_answer", "supporting_evidence", "caveats"],
            "causal": ["direct_answer", "effect_direction", "supporting_evidence", "causal_limitations"],
            "mechanism": ["direct_answer", "supporting_evidence", "mechanism_path", "caveats"],
            "comparison": ["comparison_summary", "evidence_by_option", "conditions", "conclusion"],
            "frontier": ["recent_trends", "representative_papers", "open_questions"],
        },
        "required_top_level_keys": [
            "version",
            "question",
            "normalized_question",
            "question_type",
            "recency_policy",
            "year_from",
            "year_to",
            "answer_sections",
            "required_condition_axes",
            "query_constraints",
            "ambiguity_flags",
            "router_confidence",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_entity_mention_extraction_user_prompt(
    *,
    question: str,
    task_spec: Dict[str, Any],
    allowed_entity_types: Sequence[str],
) -> str:
    payload = {
        "question": question,
        "task_spec": task_spec,
        "allowed_entity_types": list(allowed_entity_types),
        "required_top_level_keys": ["mentions"],
        "mention_schema": {
            "surface_form": "string",
            "candidate_entity_types": ["string"],
            "selected_entity_type": "string|null",
            "confidence": "float",
            "rationale": "string",
        },
        "rules": {
            "surface_form_must_be_exact_contiguous_substring_of_question": True,
            "mentions_must_be_in_question_order": True,
            "candidate_entity_types_must_come_from_allowed_entity_types": True,
            "selected_entity_type_must_be_null_or_from_candidate_entity_types": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_entity_resolver_user_prompt(
    *,
    question: str,
    task_spec: Dict[str, Any],
    mention_payload: Dict[str, Any],
) -> str:
    payload = {
        "question": question,
        "task_spec": task_spec,
        "mention_payload": mention_payload,
        "required_top_level_keys": [
            "selected_index",
            "entity_type",
            "confidence",
            "rationale",
        ],
        "rules": {
            "entity_type_must_come_from_candidates": True,
            "selected_index_must_reference_supplied_candidate_options_or_be_null": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_query_planner_user_prompt(
    *,
    question: str,
    task_spec: Dict[str, Any],
    entity_pack: Dict[str, Any],
    baseline_plans: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "question": question,
        "task_spec": task_spec,
        "entity_pack": entity_pack,
        "baseline_plans": list(baseline_plans),
        "required_top_level_keys": ["plans"],
        "allowed_lanes": ["review", "frontier", "data", "contrarian"],
        "plan_schema": {
            "lane": "string",
            "query_text": "string",
            "must_terms": ["string"],
            "exclude_terms": ["string"],
            "year_from": "int|null",
            "year_to": "int|null",
            "preferred_sources": ["openalex|crossref|semantic_scholar"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_evidence_extractor_user_prompt(
    *,
    question_type: str,
    section_type: str,
    snippet: str,
) -> str:
    payload = {
        "question_type": question_type,
        "section_type": section_type,
        "snippet": snippet,
        "required_top_level_keys": [
            "roles",
            "claim_polarity",
            "entity_mentions",
            "metric_mentions",
            "notes",
        ],
        "allowed_roles": ["observation", "limitation", "mechanism"],
        "allowed_claim_polarity": ["support", "oppose", "neutral"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_claim_miner_user_prompt(
    *,
    claim_type: str,
    main_entity: str,
    relation_type: str,
    metric_family: str,
    condition_scope: Dict[str, str],
    representative_snippet: str,
    supporting_evidence: Sequence[str],
) -> str:
    payload = {
        "claim_type": claim_type,
        "main_entity": main_entity,
        "relation_type": relation_type,
        "metric_family": metric_family,
        "condition_scope": condition_scope,
        "representative_snippet": representative_snippet,
        "supporting_evidence": list(supporting_evidence),
        "required_top_level_keys": ["claim_text"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_reviewer_user_prompt(
    *,
    review_kind: str,
    task_spec: Optional[Dict[str, Any]],
    claim: Dict[str, Any],
    evidence_snippets: Sequence[Dict[str, Any]],
    focus_flag_types: Optional[Sequence[str]],
    allowed_flag_types: Sequence[str],
) -> str:
    payload = {
        "review_kind": review_kind,
        "task_spec": task_spec,
        "claim": claim,
        "evidence_snippets": list(evidence_snippets),
        "focus_flag_types": list(focus_flag_types or []),
        "allowed_flag_types": list(allowed_flag_types),
        "required_top_level_keys": ["flags"],
        "flag_schema": {
            "flag_type": "string",
            "severity": "info|warning|critical",
            "note": "string",
            "evidence_refs": ["evidence_id"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_contradiction_reviewer_user_prompt(
    *,
    left_claim: Dict[str, Any],
    right_claim: Dict[str, Any],
    left_evidence: Sequence[Dict[str, Any]],
    right_evidence: Sequence[Dict[str, Any]],
    shared_axes: Sequence[str],
    differing_axes: Sequence[str],
) -> str:
    payload = {
        "left_claim": left_claim,
        "right_claim": right_claim,
        "left_evidence": list(left_evidence),
        "right_evidence": list(right_evidence),
        "shared_axes": list(shared_axes),
        "differing_axes": list(differing_axes),
        "required_top_level_keys": ["conflict_type", "reason"],
        "allowed_conflict_types": ["true_conflict", "condition_divergence", "no_conflict"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_claim_revision_user_prompt(
    *,
    claim: Dict[str, Any],
    task_spec: Optional[Dict[str, Any]],
    review_flags: Sequence[Dict[str, Any]],
    conflict_edges: Sequence[Dict[str, Any]],
    supporting_evidence: Sequence[Dict[str, Any]],
    allowed_condition_scope: Dict[str, str],
) -> str:
    payload = {
        "claim": claim,
        "task_spec": task_spec,
        "review_flags": list(review_flags),
        "conflict_edges": list(conflict_edges),
        "supporting_evidence": list(supporting_evidence),
        "allowed_condition_scope": allowed_condition_scope,
        "required_top_level_keys": ["claim_text", "condition_scope", "rationale"],
        "rules": {
            "condition_scope_must_be_subset_of_allowed_condition_scope": True,
            "no_new_evidence_ids": True,
            "do_not_change_topic": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_review_merge_user_prompt(
    *,
    claim: Dict[str, Any],
    active_flags: Sequence[Dict[str, Any]],
    active_conflict_edges: Sequence[Dict[str, Any]],
    revision_records: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "claim": claim,
        "active_flags": list(active_flags),
        "active_conflict_edges": list(active_conflict_edges),
        "revision_records": list(revision_records),
        "allowed_statuses": ["accepted", "contested", "rejected"],
        "required_top_level_keys": ["status", "rationale"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_synthesizer_user_prompt(input_pack: Dict[str, Any]) -> str:
    payload = {
        "input_pack": input_pack,
        "required_top_level_keys": [
            "final_answer",
            "sections",
            "limitations_summary",
        ],
        "section_schema": {
            "section_id": "string",
            "content": "string",
            "citation_ids": ["citation_id"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
