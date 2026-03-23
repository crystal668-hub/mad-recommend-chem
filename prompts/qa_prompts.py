from __future__ import annotations

from prompts.ledger import (
    build_claim_miner_system_prompt,
    build_claim_miner_user_prompt,
    build_claim_revision_system_prompt,
    build_claim_revision_user_prompt,
    build_citation_reviewer_system_prompt,
    build_contradiction_reviewer_system_prompt,
    build_contradiction_reviewer_user_prompt,
    build_entity_mention_extraction_system_prompt,
    build_entity_mention_extraction_user_prompt,
    build_entity_resolver_system_prompt,
    build_entity_resolver_user_prompt,
    build_evidence_extractor_system_prompt,
    build_evidence_extractor_user_prompt,
    build_methodology_reviewer_system_prompt,
    build_query_planner_system_prompt,
    build_query_planner_user_prompt,
    build_review_merge_system_prompt,
    build_review_merge_user_prompt,
    build_reviewer_user_prompt,
    build_router_localization_system_prompt,
    build_router_localization_user_prompt,
    build_router_semantic_system_prompt,
    build_router_semantic_user_prompt,
    build_synthesis_system_prompt,
    build_synthesizer_user_prompt,
)


ROUTER_SEMANTIC_SYSTEM_PROMPT = build_router_semantic_system_prompt()
ROUTER_LOCALIZATION_SYSTEM_PROMPT = build_router_localization_system_prompt()
ROUTER_SYSTEM_PROMPT = ROUTER_LOCALIZATION_SYSTEM_PROMPT
ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT = build_entity_mention_extraction_system_prompt(
    allowed_entity_types=["molecule", "material", "catalyst", "reaction", "solvent", "ligand", "substrate", "reagent", "metric", "condition"]
)
ENTITY_RESOLVER_SYSTEM_PROMPT = build_entity_resolver_system_prompt()
QUERY_PLANNER_SYSTEM_PROMPT = build_query_planner_system_prompt()
EVIDENCE_EXTRACTOR_SYSTEM_PROMPT = build_evidence_extractor_system_prompt()
CLAIM_MINER_SYSTEM_PROMPT = build_claim_miner_system_prompt()
METHODOLOGY_REVIEWER_SYSTEM_PROMPT = build_methodology_reviewer_system_prompt()
CITATION_REVIEWER_SYSTEM_PROMPT = build_citation_reviewer_system_prompt()
CONTRADICTION_REVIEWER_SYSTEM_PROMPT = build_contradiction_reviewer_system_prompt()
CLAIM_REVISION_SYSTEM_PROMPT = build_claim_revision_system_prompt()
REVIEW_MERGE_SYSTEM_PROMPT = build_review_merge_system_prompt()
SYNTHESIS_SYSTEM_PROMPT = build_synthesis_system_prompt()


__all__ = [
    "ROUTER_SEMANTIC_SYSTEM_PROMPT",
    "ROUTER_LOCALIZATION_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_RESOLVER_SYSTEM_PROMPT",
    "QUERY_PLANNER_SYSTEM_PROMPT",
    "EVIDENCE_EXTRACTOR_SYSTEM_PROMPT",
    "CLAIM_MINER_SYSTEM_PROMPT",
    "METHODOLOGY_REVIEWER_SYSTEM_PROMPT",
    "CITATION_REVIEWER_SYSTEM_PROMPT",
    "CONTRADICTION_REVIEWER_SYSTEM_PROMPT",
    "CLAIM_REVISION_SYSTEM_PROMPT",
    "REVIEW_MERGE_SYSTEM_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
    "build_router_semantic_user_prompt",
    "build_router_localization_user_prompt",
    "build_entity_mention_extraction_user_prompt",
    "build_entity_resolver_user_prompt",
    "build_query_planner_user_prompt",
    "build_evidence_extractor_user_prompt",
    "build_claim_miner_user_prompt",
    "build_reviewer_user_prompt",
    "build_contradiction_reviewer_user_prompt",
    "build_claim_revision_user_prompt",
    "build_review_merge_user_prompt",
    "build_synthesizer_user_prompt",
]
