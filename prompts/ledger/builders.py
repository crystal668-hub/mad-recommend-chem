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
from prompts.ledger.render import render_template


def _build_contract_system_prompt(template_name: str, *, contract: Dict[str, Any], **values: str) -> str:
    return render_template(
        template_name,
        contract_json=json_block(contract),
        example_json=json_block(contract.get("example") or {}),
        invalid_examples_json=json_block(contract.get("invalid_examples") or []),
        **values,
    )


def build_router_semantic_system_prompt() -> str:
    return _build_contract_system_prompt("router_semantic_system.txt", contract=router_semantic_contract())


def build_router_localization_system_prompt() -> str:
    return _build_contract_system_prompt("router_localization_system.txt", contract=router_localization_contract())


def build_entity_mention_extraction_system_prompt(*, allowed_entity_types: Sequence[str]) -> str:
    contract = entity_mention_extraction_contract(allowed_entity_types=allowed_entity_types)
    return _build_contract_system_prompt(
        "entity_mention_extraction_system.txt",
        contract=contract,
        allowed_entity_types_json=json_block(list(allowed_entity_types)),
    )


def build_entity_resolver_system_prompt() -> str:
    return _build_contract_system_prompt("entity_resolver_system.txt", contract=entity_resolver_contract())


def build_query_planner_system_prompt() -> str:
    contract = query_planner_contract()
    return _build_contract_system_prompt(
        "query_planner_system.txt",
        contract=contract,
        allowed_lanes_json=json_block(contract.get("allowed_lanes") or []),
    )


def build_evidence_extractor_system_prompt() -> str:
    contract = evidence_extractor_contract()
    return _build_contract_system_prompt(
        "evidence_extractor_system.txt",
        contract=contract,
        allowed_roles_json=json_block(contract.get("allowed_roles") or []),
        allowed_claim_polarity_json=json_block(contract.get("allowed_claim_polarity") or []),
    )


def build_claim_miner_system_prompt() -> str:
    return _build_contract_system_prompt("claim_miner_system.txt", contract=claim_miner_contract())


def build_methodology_reviewer_system_prompt() -> str:
    contract = reviewer_contract(
        allowed_flag_types=["Missing_Condition", "Incomplete_Condition", "Overgeneralized", "Mechanism_Speculative", "Metric_Mismatch"]
    )
    return _build_contract_system_prompt(
        "methodology_reviewer_system.txt",
        contract=contract,
        allowed_flag_types_json=json_block(contract.get("allowed_flag_types") or []),
    )


def build_citation_reviewer_system_prompt() -> str:
    contract = reviewer_contract(allowed_flag_types=["Unsupported", "Weak_Evidence"])
    return _build_contract_system_prompt(
        "citation_reviewer_system.txt",
        contract=contract,
        allowed_flag_types_json=json_block(contract.get("allowed_flag_types") or []),
    )


def build_contradiction_reviewer_system_prompt() -> str:
    contract = contradiction_reviewer_contract()
    return _build_contract_system_prompt(
        "contradiction_reviewer_system.txt",
        contract=contract,
        allowed_conflict_types_json=json_block(contract.get("allowed_conflict_types") or []),
    )


def build_claim_revision_system_prompt() -> str:
    return _build_contract_system_prompt("claim_revision_system.txt", contract=claim_revision_contract())


def build_review_merge_system_prompt() -> str:
    contract = review_merge_contract()
    return _build_contract_system_prompt(
        "review_merge_system.txt",
        contract=contract,
        allowed_statuses_json=json_block(contract.get("allowed_statuses") or []),
    )


def build_synthesis_system_prompt() -> str:
    return _build_contract_system_prompt("synthesis_system.txt", contract=synthesis_contract())


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
