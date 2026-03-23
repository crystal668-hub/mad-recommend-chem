from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def router_semantic_contract() -> Dict[str, Any]:
    return {
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
        "allowed_question_types": ["fact", "causal", "mechanism", "comparison", "frontier"],
        "allowed_time_intents": ["none", "recent", "explicit", "current"],
        "example": {
            "primary_question_type": "comparison",
            "secondary_candidates": ["fact"],
            "semantic_confidence": 0.83,
            "needs_disambiguation": False,
            "comparison_intent": True,
            "comparison_targets_present": True,
            "explicit_metric_requested": False,
            "explicit_time_intent": "none",
            "mechanistic_intent": False,
            "causal_intent": False,
            "frontier_intent": False,
            "notes_on_ambiguity": "",
        },
        "invalid_examples": [
            {"question_type": "comparison"},
            {"primary_question_type": "ranking"},
            "```json\n{\"primary_question_type\": \"comparison\"}\n```",
        ],
    }


def router_localization_contract() -> Dict[str, Any]:
    return {
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
        "allowed_question_types": ["fact", "causal", "mechanism", "comparison", "frontier"],
        "allowed_recency_policies": ["none", "last_3y", "last_5y", "explicit"],
        "example": {
            "version": 1,
            "question": "How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
            "normalized_question": "How does Pt/C compare with NiMo catalysts for HER activity in alkaline media?",
            "question_type": "comparison",
            "recency_policy": "none",
            "year_from": None,
            "year_to": None,
            "answer_sections": ["comparison_summary", "evidence_by_option", "conditions", "conclusion"],
            "required_condition_axes": ["catalyst", "electrolyte"],
            "query_constraints": {
                "must_include_terms": ["Pt/C", "NiMo", "HER"],
                "should_include_terms": ["alkaline media"],
                "exclude_terms": [],
            },
            "ambiguity_flags": [],
            "router_confidence": 0.79,
        },
        "invalid_examples": [
            {"question_type": "comparison"},
            {"answer_sections": "comparison_summary"},
            "Pt/C is better than NiMo.",
        ],
    }


def entity_mention_extraction_contract(*, allowed_entity_types: Sequence[str]) -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["mentions"],
        "allowed_entity_types": list(allowed_entity_types),
        "mention_schema": {
            "surface_form": "string",
            "candidate_entity_types": ["string"],
            "selected_entity_type": "string|null",
            "confidence": "float",
            "rationale": "string",
        },
        "example": {
            "mentions": [
                {
                    "surface_form": "Pt/C",
                    "candidate_entity_types": ["catalyst", "material"],
                    "selected_entity_type": "catalyst",
                    "confidence": 0.88,
                    "rationale": "Exact chemistry mention in the question.",
                }
            ]
        },
        "invalid_examples": [
            {"surface_form": "Pt/C"},
            {"mentions": [{"surface_form": "invented alias"}]},
            "```json\n{\"mentions\": []}\n```",
        ],
    }


def entity_resolver_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["selected_index", "entity_type", "confidence", "rationale"],
        "example": {
            "selected_index": 0,
            "entity_type": "catalyst",
            "confidence": 0.87,
            "rationale": "Best match among supplied candidates.",
        },
        "invalid_examples": [
            {"entity_index": 0},
            {"selected_index": 99, "entity_type": "invented_type"},
            "No candidate fits.",
        ],
    }


def query_planner_contract() -> Dict[str, Any]:
    return {
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
        "example": {
            "plans": [
                {
                    "lane": "review",
                    "query_text": "Pt/C NiMo HER review alkaline media",
                    "must_terms": ["Pt/C", "NiMo", "review"],
                    "exclude_terms": [],
                    "year_from": 2019,
                    "year_to": 2026,
                    "preferred_sources": ["openalex", "semantic_scholar", "crossref"],
                },
                {
                    "lane": "frontier",
                    "query_text": "Pt/C NiMo HER recent advances alkaline media",
                    "must_terms": ["Pt/C", "NiMo", "recent"],
                    "exclude_terms": [],
                    "year_from": 2022,
                    "year_to": 2026,
                    "preferred_sources": ["openalex", "semantic_scholar", "crossref"],
                },
                {
                    "lane": "data",
                    "query_text": "Pt/C NiMo HER benchmark performance alkaline media",
                    "must_terms": ["Pt/C", "NiMo", "benchmark"],
                    "exclude_terms": [],
                    "year_from": 2022,
                    "year_to": 2026,
                    "preferred_sources": ["openalex", "semantic_scholar", "crossref"],
                },
                {
                    "lane": "contrarian",
                    "query_text": "Pt/C NiMo HER limitation negative result alkaline media",
                    "must_terms": ["Pt/C", "NiMo", "limitation"],
                    "exclude_terms": [],
                    "year_from": 2022,
                    "year_to": 2026,
                    "preferred_sources": ["openalex", "semantic_scholar", "crossref"],
                },
            ]
        },
        "invalid_examples": [
            {"plans": [{"lane": "review"}]},
            {"plans": [{"lane": "ranking"}]},
            "```json\n{\"plans\": []}\n```",
        ],
    }


def evidence_extractor_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["roles", "claim_polarity", "entity_mentions", "metric_mentions", "notes"],
        "allowed_roles": ["observation", "limitation", "mechanism"],
        "allowed_claim_polarity": ["support", "oppose", "neutral"],
        "example": {
            "roles": ["observation"],
            "claim_polarity": "support",
            "entity_mentions": ["Pt/C"],
            "metric_mentions": ["overpotential"],
            "notes": "Snippet reports improved HER activity.",
        },
        "invalid_examples": [
            {"role": "observation"},
            {"roles": ["causal"], "claim_polarity": "supports"},
            "Pt/C looks better.",
        ],
    }


def claim_miner_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["claim_text"],
        "example": {
            "claim_text": "Under alkaline HER conditions, Pt/C shows lower overpotential than bare carbon."
        },
        "invalid_examples": [
            {"claim": "Pt/C is good."},
            {"claim_text": "Pt/C is always best for every reaction."},
            "Pt/C works.",
        ],
    }


def reviewer_contract(*, allowed_flag_types: Sequence[str]) -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["flags"],
        "allowed_flag_types": list(allowed_flag_types),
        "flag_schema": {
            "flag_type": "string",
            "severity": "info|warning|critical",
            "note": "string",
            "evidence_refs": ["evidence_id"],
        },
        "example": {
            "flags": [
                {
                    "flag_type": allowed_flag_types[0] if allowed_flag_types else "Unsupported",
                    "severity": "warning",
                    "note": "Claim omits a required electrolyte condition.",
                    "evidence_refs": ["ev-1"],
                }
            ]
        },
        "invalid_examples": [
            {"review_flags": []},
            {"flags": [{"flag_type": "Invented_Type"}]},
            "This claim looks weak.",
        ],
    }


def contradiction_reviewer_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["conflict_type", "reason"],
        "allowed_conflict_types": ["true_conflict", "condition_divergence", "no_conflict"],
        "example": {
            "conflict_type": "condition_divergence",
            "reason": "The claims differ on electrolyte and potential, so they do not directly conflict."
        },
        "invalid_examples": [
            {"label": "conflict"},
            {"conflict_type": "maybe"},
            "These papers disagree.",
        ],
    }


def claim_revision_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["claim_text", "condition_scope", "rationale"],
        "example": {
            "claim_text": "Pt/C improves HER activity in 1 M KOH under the reported conditions.",
            "condition_scope": {"electrolyte": "1 M KOH"},
            "rationale": "Added the missing electrolyte condition flagged in review.",
        },
        "invalid_examples": [
            {"claim": "Pt/C improves HER activity."},
            {"claim_text": "Pt/C is universally optimal.", "condition_scope": {}},
            "No change needed.",
        ],
    }


def review_merge_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["status", "rationale"],
        "allowed_statuses": ["accepted", "contested", "rejected"],
        "example": {"status": "contested", "rationale": "Critical evidence gaps remain unresolved."},
        "invalid_examples": [
            {"decision": "accepted"},
            {"status": "partial"},
            "Looks acceptable.",
        ],
    }


def synthesis_contract() -> Dict[str, Any]:
    return {
        "required_top_level_keys": ["final_answer", "sections", "limitations_summary"],
        "section_schema": {
            "section_id": "string",
            "content": "string",
            "citation_ids": ["citation_id"],
        },
        "example": {
            "final_answer": "Pt/C remains the strongest supported HER benchmark under the cited alkaline conditions.",
            "sections": [
                {
                    "section_id": "direct_answer",
                    "content": "Pt/C shows the strongest evidence-backed HER performance under the cited alkaline conditions.",
                    "citation_ids": ["CIT-1"],
                }
            ],
            "limitations_summary": "Evidence remains limited to the cited conditions and accepted claims.",
        },
        "invalid_examples": [
            {"answer": "Pt/C is best."},
            {"sections": "direct_answer"},
            "Pt/C is best overall.",
        ],
    }

