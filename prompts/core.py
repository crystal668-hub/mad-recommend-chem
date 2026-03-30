from __future__ import annotations

from prompts.basic_node import (
    build_claim_miner_system_prompt,
    build_claim_miner_user_prompt,
    build_entity_mention_extraction_system_prompt,
    build_entity_mention_extraction_user_prompt,
    build_entity_resolver_system_prompt,
    build_entity_resolver_user_prompt,
    build_evidence_extractor_system_prompt,
    build_evidence_extractor_user_prompt,
    build_query_planner_system_prompt,
    build_query_planner_user_prompt,
    build_router_localization_system_prompt,
    build_router_localization_user_prompt,
    build_router_semantic_system_prompt,
    build_router_semantic_user_prompt,
)


ROUTER_SEMANTIC_SYSTEM_PROMPT = build_router_semantic_system_prompt()
ROUTER_LOCALIZATION_SYSTEM_PROMPT = build_router_localization_system_prompt()
ROUTER_SYSTEM_PROMPT = ROUTER_LOCALIZATION_SYSTEM_PROMPT
ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT = build_entity_mention_extraction_system_prompt(
    allowed_entity_types=[
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
    ]
)
ENTITY_RESOLVER_SYSTEM_PROMPT = build_entity_resolver_system_prompt()
QUERY_PLANNER_SYSTEM_PROMPT = build_query_planner_system_prompt()
EVIDENCE_EXTRACTOR_SYSTEM_PROMPT = build_evidence_extractor_system_prompt()
CLAIM_MINER_SYSTEM_PROMPT = build_claim_miner_system_prompt()


__all__ = [
    "ROUTER_SEMANTIC_SYSTEM_PROMPT",
    "ROUTER_LOCALIZATION_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_RESOLVER_SYSTEM_PROMPT",
    "QUERY_PLANNER_SYSTEM_PROMPT",
    "EVIDENCE_EXTRACTOR_SYSTEM_PROMPT",
    "CLAIM_MINER_SYSTEM_PROMPT",
    "build_router_semantic_user_prompt",
    "build_router_localization_user_prompt",
    "build_entity_mention_extraction_user_prompt",
    "build_entity_resolver_user_prompt",
    "build_query_planner_user_prompt",
    "build_evidence_extractor_user_prompt",
    "build_claim_miner_user_prompt",
]
