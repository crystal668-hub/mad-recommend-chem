from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from prompts.basic_node.contracts import (
    claim_miner_contract,
    entity_mention_extraction_contract,
    entity_resolver_contract,
    evidence_extractor_contract,
    query_planner_contract,
    router_localization_contract,
    router_semantic_contract,
)
from prompts.basic_node.render import json_block, render_template


def _build_contract_system_prompt(template_name: str, *, contract: Dict[str, Any], **values: str) -> str:
    return render_template(
        template_name,
        contract_json=json_block(contract),
        example_json=json_block(contract.get("example") or {}),
        invalid_examples_json=json_block(contract.get("invalid_examples") or []),
        **values,
    )


def build_router_semantic_system_prompt() -> str:
    return _build_contract_system_prompt("router_semantic_system.yaml", contract=router_semantic_contract())


def build_router_localization_system_prompt() -> str:
    return _build_contract_system_prompt("router_localization_system.yaml", contract=router_localization_contract())


def build_entity_mention_extraction_system_prompt(*, allowed_entity_types: Sequence[str]) -> str:
    contract = entity_mention_extraction_contract(allowed_entity_types=allowed_entity_types)
    return _build_contract_system_prompt(
        "entity_mention_extraction_system.yaml",
        contract=contract,
        allowed_entity_types_json=json_block(list(allowed_entity_types)),
    )


def build_entity_resolver_system_prompt() -> str:
    return _build_contract_system_prompt("entity_resolver_system.yaml", contract=entity_resolver_contract())


def build_query_planner_system_prompt() -> str:
    contract = query_planner_contract()
    return _build_contract_system_prompt(
        "query_planner_system.yaml",
        contract=contract,
        allowed_lanes_json=json_block(contract.get("allowed_lanes") or []),
    )


def build_evidence_extractor_system_prompt() -> str:
    contract = evidence_extractor_contract()
    return _build_contract_system_prompt(
        "evidence_extractor_system.yaml",
        contract=contract,
        allowed_roles_json=json_block(contract.get("allowed_roles") or []),
        allowed_claim_polarity_json=json_block(contract.get("allowed_claim_polarity") or []),
    )


def build_claim_miner_system_prompt() -> str:
    return _build_contract_system_prompt("claim_miner_system.yaml", contract=claim_miner_contract())


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
