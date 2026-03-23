from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from prompts.ledger.contracts import (
    claim_miner_contract,
    claim_revision_contract,
    contradiction_reviewer_contract,
    entity_mention_extraction_contract,
    entity_resolver_contract,
    evidence_extractor_contract,
    query_planner_contract,
    review_merge_contract,
    reviewer_contract,
    router_localization_contract,
    router_semantic_contract,
    synthesis_contract,
)
from prompts.ledger.render import json_block


def _build_strict_json_system_prompt(*, role: str, mission: str, extra_rules: Sequence[str], contract: Dict[str, Any]) -> str:
    lines = [
        role,
        mission,
        "Return STRICT JSON only.",
        "Do not add prose, markdown fences, comments, or extra keys.",
        "",
        "Output contract:",
        json_block(contract),
        "",
        "Correct JSON example:",
        json_block(contract.get("example") or {}),
        "",
        "Common invalid outputs:",
        json_block(contract.get("invalid_examples") or []),
    ]
    if extra_rules:
        lines.extend(["", "Additional rules:"])
        lines.extend(f"- {rule}" for rule in extra_rules)
    return "\n".join(lines).strip()


def build_router_semantic_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are RouterNode semantic interpretation stage for a chemistry QA grounding pipeline.",
        mission="Reason from the question text first and expose ambiguity explicitly instead of forcing false certainty.",
        extra_rules=["Choose only from the allowed question types and enums."],
        contract=router_semantic_contract(),
    )


def build_router_localization_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are RouterNode task localization stage for a chemistry QA grounding pipeline.",
        mission="Use the semantic parse as the primary interpretation of user intent and keep the final TaskSpec conservative when ambiguity remains.",
        extra_rules=["Use optional signals only as supporting observations, not as preferred defaults."],
        contract=router_localization_contract(),
    )


def build_entity_mention_extraction_system_prompt(*, allowed_entity_types: Sequence[str]) -> str:
    return _build_strict_json_system_prompt(
        role="You are EntityResolverNode for a chemistry QA pipeline.",
        mission="Extract only chemistry-relevant entity mentions that appear as exact contiguous spans in the question.",
        extra_rules=[
            "You may only use the supplied ontology entity types.",
            "Do not invent new spans, aliases, canonical names, or chemical identifiers.",
        ],
        contract=entity_mention_extraction_contract(allowed_entity_types=allowed_entity_types),
    )


def build_entity_resolver_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are EntityResolverNode for a chemistry QA pipeline.",
        mission="Resolve an entity mention only by choosing from the supplied candidate entity types and candidate indices.",
        extra_rules=["Do not invent new entities, aliases, canonical names, or chemical identifiers."],
        contract=entity_resolver_contract(),
    )


def build_query_planner_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are QueryPlannerNode for a chemistry QA retrieval pipeline.",
        mission="Produce exactly four search plans aligned to the supplied task and entity grounding.",
        extra_rules=[
            "You must produce exactly four plans whose lanes are review, frontier, data, and contrarian.",
            "Preserve the supplied baseline structure unless a narrower query is clearly better.",
        ],
        contract=query_planner_contract(),
    )


def build_evidence_extractor_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You classify scientific evidence snippets for a chemistry QA pipeline.",
        mission="Label the snippet conservatively using only information present in the snippet.",
        extra_rules=[
            "roles may only contain observation, limitation, or mechanism.",
            "claim_polarity must be support, oppose, or neutral.",
            "Do not invent entities or metrics that are not supported by the snippet.",
        ],
        contract=evidence_extractor_contract(),
    )


def build_claim_miner_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You name evidence clusters as single English condition-bound claims.",
        mission="Summarize the supported claim conservatively without adding new evidence or topics.",
        extra_rules=["Output only the claim_text field."],
        contract=claim_miner_contract(),
    )


def build_methodology_reviewer_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are MethodologyReviewer for a chemistry QA pipeline.",
        mission="Flag only methodology issues that are directly supported by the supplied claim, task axes, and snippets.",
        extra_rules=["Use only the allowed flag_type values."],
        contract=reviewer_contract(
            allowed_flag_types=["Missing_Condition", "Incomplete_Condition", "Overgeneralized", "Mechanism_Speculative", "Metric_Mismatch"]
        ),
    )


def build_citation_reviewer_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are CitationReviewer for a chemistry QA pipeline.",
        mission="Flag unsupported or weakly supported claims using only the supplied evidence references.",
        extra_rules=["Do not emit Fabricated_Citation; that is checked deterministically elsewhere."],
        contract=reviewer_contract(allowed_flag_types=["Unsupported", "Weak_Evidence"]),
    )


def build_contradiction_reviewer_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You adjudicate whether a prefiltered pair of chemistry claims is a true_conflict or condition_divergence.",
        mission="Assess only the supplied pair and do not generalize beyond it.",
        extra_rules=["Choose only from true_conflict, condition_divergence, or no_conflict."],
        contract=contradiction_reviewer_contract(),
    )


def build_claim_revision_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You conservatively revise a chemistry claim after peer review.",
        mission="Revise only the supported claim wording and condition scope without changing the topic.",
        extra_rules=[
            "You may revise only claim_text and condition_scope.",
            "Do not change the claim topic, add evidence, or broaden the claim scope.",
        ],
        contract=claim_revision_contract(),
    )


def build_review_merge_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are ReviewMergeNode for a chemistry QA structured peer review module.",
        mission="Merge review outcomes conservatively and do not override critical review findings.",
        extra_rules=["You may choose only accepted, contested, or rejected."],
        contract=review_merge_contract(),
    )


def build_synthesis_system_prompt() -> str:
    return _build_strict_json_system_prompt(
        role="You are SynthesizerNode of a chemistry QA pipeline.",
        mission="Write the final answer using only the supplied SynthesisInputPack.",
        extra_rules=[
            "Do not add new facts, new citations, or any rejected claim.",
            "Accepted claims may appear in main sections.",
            "Contested claims may appear only in the Limitations / Controversies section.",
            "Match the wording to the supplied confidence labels.",
        ],
        contract=synthesis_contract(),
    )


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
        "output_contract": router_semantic_contract(),
    }
    return json_block(payload)


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
        "output_contract": router_localization_contract(),
    }
    return json_block(payload)


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
        "output_contract": entity_mention_extraction_contract(allowed_entity_types=allowed_entity_types),
    }
    return json_block(payload)


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
        "output_contract": entity_resolver_contract(),
    }
    return json_block(payload)


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
        "output_contract": query_planner_contract(),
    }
    return json_block(payload)


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
        "output_contract": evidence_extractor_contract(),
    }
    return json_block(payload)


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
        "output_contract": claim_miner_contract(),
    }
    return json_block(payload)


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
        "output_contract": reviewer_contract(allowed_flag_types=allowed_flag_types),
    }
    return json_block(payload)


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
        "output_contract": contradiction_reviewer_contract(),
    }
    return json_block(payload)


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
        "output_contract": claim_revision_contract(),
        "rules": {
            "condition_scope_must_be_subset_of_allowed_condition_scope": True,
            "no_new_evidence_ids": True,
            "do_not_change_topic": True,
        },
    }
    return json_block(payload)


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
        "output_contract": review_merge_contract(),
    }
    return json_block(payload)


def build_synthesizer_user_prompt(input_pack: Dict[str, Any]) -> str:
    payload = {"input_pack": input_pack, "output_contract": synthesis_contract()}
    return json_block(payload)

