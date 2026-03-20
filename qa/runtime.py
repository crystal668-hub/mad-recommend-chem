from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.chat_models import build_chat_model_from_config, describe_chat_model_config, resolve_env_var
from qa.evidence import ClaimMiner, EvidenceExtractor, EvidenceLedgerBuilder
from qa.handoff import EvidenceExtractorHandoff
from qa.nodes.citation_reviewer import CitationReviewer
from qa.nodes.claim_revision import ClaimRevisionNode
from qa.nodes.contradiction_reviewer import ContradictionReviewer
from qa.nodes.document_acquirer import DocumentAcquirerNode
from qa.nodes.entity_resolver import EntityResolverNode, PubChemClient
from qa.nodes.methodology_reviewer import MethodologyReviewer
from qa.nodes.query_planner import QueryPlannerNode
from qa.nodes.retriever import RetrieverNode
from qa.nodes.review_merge import ReviewMergeNode
from qa.nodes.router import RouterNode
from qa.nodes.synthesizer import SynthesizerNode
from qa.pipeline import QueryGroundingPipeline
from qa.pdf_extraction import PDFExtractionPipeline
from qa.providers import (
    CrossrefClient,
    HttpTextFetcher,
    OpenAlexClient,
    SemanticScholarClient,
    UnpaywallClient,
)
from qa.retrieval_pipeline import HeterogeneousRetrievalPipeline
from qa.review_pipeline import StructuredPeerReviewPipeline
from qa.react_reviewed_workflow import ReactReviewedWorkflow
from qa.react_reviewed_state import WorkflowMode
from qa.synthesis_pipeline import VerifiedSynthesisPipeline


logger = logging.getLogger("MAD.qa.runtime")

INFERENCE_NODE_MODEL_KEYS = (
    "router",
    "entity_resolver",
    "query_planner",
    "evidence_extractor",
    "claim_miner",
    "methodology_reviewer",
    "citation_reviewer",
    "contradiction_reviewer",
    "claim_revision",
    "review_merge",
    "synthesizer",
    "react_proposer",
    "react_reviewer_search_coverage",
    "react_reviewer_evidence_trace",
    "react_reviewer_reasoning_consistency",
    "react_reviewer_counterevidence",
)

DEFAULT_QA_MODEL_ALIASES = {node_name: "agent1" for node_name in INFERENCE_NODE_MODEL_KEYS}

DEFAULT_QA_PEER_REVIEW_CONFIG: Dict[str, Any] = {
    "max_claims_for_llm_review": 40,
    "max_second_round_claims": 15,
    "disable_llm_review_when_abstract_only": True,
    "fallback_mode": "deterministic_only",
}

DEFAULT_QA_PDF_EXTRACTION_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "primary_backend": "pymupdf",
    "secondary_backend": "docling",
    "ocr_backend": "ocrmypdf",
    "enable_ocr_fallback": True,
    "min_total_chars": 800,
    "min_chars_per_text_page": 80,
    "min_text_page_ratio": 0.5,
    "min_printable_ratio": 0.95,
    "snippet_target_chars": 1000,
    "snippet_overlap_chars": 120,
    "preserve_page_blocks": True,
    "max_ocr_pages": 40,
    "ocr_timeout_seconds": 300,
    "skip_ocr_when_text_already_usable": True,
}

DEFAULT_QA_ENTITY_RESOLUTION_CONFIG: Dict[str, Any] = {
    "seed_file": "./qa/resources/entity_seeds.yaml",
    "emit_seed_suggestions": True,
    "pubchem_enabled": True,
    "pubchem_entity_types": ["molecule", "solvent", "reagent", "ligand", "substrate"],
    "max_pubchem_candidates": 5,
    "mention_extraction_min_confidence": 0.7,
    "llm_disambiguation_enabled": True,
    "disambiguation_min_confidence": 0.7,
    "fail_open_on_provider_error": True,
}

DEFAULT_QA_CONFIG: Dict[str, Any] = {
    "workflow_mode": "react_reviewed",
    "save_output": True,
    "outputs_dir": None,
    "artifact_subdir": "qa_artifacts",
    "enable_peer_review": True,
    "model_timeout_seconds": 45.0,
    "progress_log_every_claims": 10,
    "models": copy.deepcopy(DEFAULT_QA_MODEL_ALIASES),
    "peer_review": copy.deepcopy(DEFAULT_QA_PEER_REVIEW_CONFIG),
    "entity_resolution": copy.deepcopy(DEFAULT_QA_ENTITY_RESOLUTION_CONFIG),
    "react_reviewed": {
        "max_propose_steps_initial": 6,
        "max_propose_steps_revision": 4,
        "proposer_fallback_mode": "fail_fast_only",
        "proposer_repair_attempts": 1,
        "proposer_evidence_policy": "prefer_fulltext",
        "max_review_cycles": 3,
        "reviewer_max_steps": 3,
        "reviewer_max_concurrency": 4,
        "reviewer_max_retrieval_actions": 1,
        "reviewer_retrieval_budget_by_role": {
            "search_coverage": 1,
            "evidence_trace": 0,
            "reasoning_consistency": 0,
            "counterevidence": 2,
        },
        "max_review_items_per_reviewer": 3,
        "max_review_items_per_step_section": 1,
        "stop_on_no_blocking_items": True,
        "require_all_reviewers": True,
        "review_call_retry_limit": 1,
        "review_failure_blocks_acceptance": True,
    },
    "pdf_extraction": copy.deepcopy(DEFAULT_QA_PDF_EXTRACTION_CONFIG),
    "providers": {
        "openalex_mailto": None,
        "crossref_mailto": None,
        "semantic_scholar_api_key": None,
        "unpaywall_email": None,
        "http_timeout": 10.0,
        "fetch_timeout": 15.0,
        "retry_attempts": 2,
        "backoff_base_seconds": 1.0,
        "backoff_max_seconds": 8.0,
    },
}


@dataclass
class QARuntime:
    qa_config: Dict[str, Any]
    grounding_pipeline: QueryGroundingPipeline
    retrieval_pipeline: HeterogeneousRetrievalPipeline
    peer_review_pipeline: Optional[StructuredPeerReviewPipeline]
    synthesis_pipeline: VerifiedSynthesisPipeline
    react_reviewed_workflow: Optional[Any]
    runtime_manifest: Dict[str, Any]


def resolve_qa_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    qa_config = copy.deepcopy(DEFAULT_QA_CONFIG)
    raw_qa_config = dict((config or {}).get("qa", {}) or {})
    raw_models = dict(raw_qa_config.pop("models", {}) or {})
    raw_peer_review = dict(raw_qa_config.pop("peer_review", {}) or {})
    raw_entity_resolution = dict(raw_qa_config.pop("entity_resolution", {}) or {})
    raw_react_reviewed = dict(raw_qa_config.pop("react_reviewed", {}) or {})
    raw_pdf_extraction = dict(raw_qa_config.pop("pdf_extraction", {}) or {})
    raw_providers = dict(raw_qa_config.pop("providers", {}) or {})
    qa_config.update(raw_qa_config)

    outputs_dir = qa_config.get("outputs_dir") or (config or {}).get("paths", {}).get("outputs") or "./outputs"
    artifact_subdir = str(qa_config.get("artifact_subdir") or DEFAULT_QA_CONFIG["artifact_subdir"]).strip()

    model_aliases = copy.deepcopy(DEFAULT_QA_MODEL_ALIASES)
    for node_name, alias in raw_models.items():
        if not str(node_name or "").strip():
            continue
        model_aliases[str(node_name).strip()] = str(alias).strip()

    provider_config = copy.deepcopy(DEFAULT_QA_CONFIG["providers"])
    provider_config.update(raw_providers)
    peer_review_config = copy.deepcopy(DEFAULT_QA_PEER_REVIEW_CONFIG)
    peer_review_config.update(raw_peer_review)
    entity_resolution_config = copy.deepcopy(DEFAULT_QA_ENTITY_RESOLUTION_CONFIG)
    entity_resolution_config.update(raw_entity_resolution)
    react_reviewed_config = copy.deepcopy(DEFAULT_QA_CONFIG["react_reviewed"])
    react_reviewed_config.update(raw_react_reviewed)
    pdf_extraction_config = copy.deepcopy(DEFAULT_QA_PDF_EXTRACTION_CONFIG)
    pdf_extraction_config.update(raw_pdf_extraction)

    qa_config["workflow_mode"] = _coerce_allowed_text(
        qa_config.get("workflow_mode"),
        allowed={"ledger", "react_reviewed"},
        fallback=DEFAULT_QA_CONFIG["workflow_mode"],
    )
    qa_config["save_output"] = bool(qa_config.get("save_output", DEFAULT_QA_CONFIG["save_output"]))
    qa_config["outputs_dir"] = str(outputs_dir)
    qa_config["artifact_subdir"] = artifact_subdir or str(DEFAULT_QA_CONFIG["artifact_subdir"])
    qa_config["enable_peer_review"] = bool(
        qa_config.get("enable_peer_review", DEFAULT_QA_CONFIG["enable_peer_review"])
    )
    qa_config["model_timeout_seconds"] = _coerce_positive_float(
        qa_config.get("model_timeout_seconds"),
        fallback=45.0,
    )
    qa_config["progress_log_every_claims"] = _coerce_positive_int(
        qa_config.get("progress_log_every_claims"),
        fallback=10,
    )
    qa_config["models"] = model_aliases
    qa_config["peer_review"] = {
        "max_claims_for_llm_review": _coerce_positive_int(
            peer_review_config.get("max_claims_for_llm_review"),
            fallback=DEFAULT_QA_PEER_REVIEW_CONFIG["max_claims_for_llm_review"],
        ),
        "max_second_round_claims": _coerce_positive_int(
            peer_review_config.get("max_second_round_claims"),
            fallback=DEFAULT_QA_PEER_REVIEW_CONFIG["max_second_round_claims"],
        ),
        "disable_llm_review_when_abstract_only": bool(
            peer_review_config.get(
                "disable_llm_review_when_abstract_only",
                DEFAULT_QA_PEER_REVIEW_CONFIG["disable_llm_review_when_abstract_only"],
            )
        ),
        "fallback_mode": _coerce_allowed_text(
            peer_review_config.get("fallback_mode"),
            allowed={"deterministic_only"},
            fallback=DEFAULT_QA_PEER_REVIEW_CONFIG["fallback_mode"],
        ),
    }
    qa_config["entity_resolution"] = {
        "seed_file": str(
            entity_resolution_config.get("seed_file") or DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["seed_file"]
        ).strip(),
        "emit_seed_suggestions": bool(
            entity_resolution_config.get(
                "emit_seed_suggestions",
                DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["emit_seed_suggestions"],
            )
        ),
        "pubchem_enabled": bool(
            entity_resolution_config.get("pubchem_enabled", DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["pubchem_enabled"])
        ),
        "pubchem_entity_types": _coerce_text_list(
            entity_resolution_config.get("pubchem_entity_types"),
            fallback=DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["pubchem_entity_types"],
        ),
        "max_pubchem_candidates": _coerce_positive_int(
            entity_resolution_config.get("max_pubchem_candidates"),
            fallback=DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["max_pubchem_candidates"],
        ),
        "mention_extraction_min_confidence": _coerce_probability(
            entity_resolution_config.get("mention_extraction_min_confidence"),
            fallback=DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["mention_extraction_min_confidence"],
        ),
        "llm_disambiguation_enabled": bool(
            entity_resolution_config.get(
                "llm_disambiguation_enabled",
                DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["llm_disambiguation_enabled"],
            )
        ),
        "disambiguation_min_confidence": _coerce_probability(
            entity_resolution_config.get("disambiguation_min_confidence"),
            fallback=DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["disambiguation_min_confidence"],
        ),
        "fail_open_on_provider_error": bool(
            entity_resolution_config.get(
                "fail_open_on_provider_error",
                DEFAULT_QA_ENTITY_RESOLUTION_CONFIG["fail_open_on_provider_error"],
            )
        ),
    }
    qa_config["react_reviewed"] = {
        "max_propose_steps_initial": _coerce_positive_int(
            react_reviewed_config.get("max_propose_steps_initial"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["max_propose_steps_initial"],
        ),
        "max_propose_steps_revision": _coerce_positive_int(
            react_reviewed_config.get("max_propose_steps_revision"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["max_propose_steps_revision"],
        ),
        "proposer_fallback_mode": _coerce_allowed_text(
            react_reviewed_config.get("proposer_fallback_mode"),
            allowed={"fail_fast_only"},
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["proposer_fallback_mode"],
        ),
        "proposer_repair_attempts": _coerce_non_negative_int(
            react_reviewed_config.get("proposer_repair_attempts"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["proposer_repair_attempts"],
        ),
        "proposer_evidence_policy": _coerce_allowed_text(
            react_reviewed_config.get("proposer_evidence_policy"),
            allowed={"prefer_fulltext"},
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["proposer_evidence_policy"],
        ),
        "max_review_cycles": _coerce_positive_int(
            react_reviewed_config.get("max_review_cycles"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["max_review_cycles"],
        ),
        "reviewer_max_steps": _coerce_positive_int(
            react_reviewed_config.get("reviewer_max_steps"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["reviewer_max_steps"],
        ),
        "reviewer_max_concurrency": _coerce_positive_int(
            react_reviewed_config.get("reviewer_max_concurrency"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["reviewer_max_concurrency"],
        ),
        "reviewer_max_retrieval_actions": _coerce_non_negative_int(
            react_reviewed_config.get("reviewer_max_retrieval_actions"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["reviewer_max_retrieval_actions"],
        ),
        "reviewer_retrieval_budget_by_role": {
            role: _coerce_non_negative_int(
                dict(react_reviewed_config.get("reviewer_retrieval_budget_by_role", {}) or {}).get(role),
                fallback=DEFAULT_QA_CONFIG["react_reviewed"]["reviewer_retrieval_budget_by_role"][role],
            )
            for role in DEFAULT_QA_CONFIG["react_reviewed"]["reviewer_retrieval_budget_by_role"]
        },
        "max_review_items_per_reviewer": _coerce_positive_int(
            react_reviewed_config.get("max_review_items_per_reviewer"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["max_review_items_per_reviewer"],
        ),
        "max_review_items_per_step_section": _coerce_positive_int(
            react_reviewed_config.get("max_review_items_per_step_section"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["max_review_items_per_step_section"],
        ),
        "stop_on_no_blocking_items": bool(
            react_reviewed_config.get(
                "stop_on_no_blocking_items",
                DEFAULT_QA_CONFIG["react_reviewed"]["stop_on_no_blocking_items"],
            )
        ),
        "require_all_reviewers": bool(
            react_reviewed_config.get(
                "require_all_reviewers",
                DEFAULT_QA_CONFIG["react_reviewed"]["require_all_reviewers"],
            )
        ),
        "review_call_retry_limit": _coerce_non_negative_int(
            react_reviewed_config.get("review_call_retry_limit"),
            fallback=DEFAULT_QA_CONFIG["react_reviewed"]["review_call_retry_limit"],
        ),
        "review_failure_blocks_acceptance": bool(
            react_reviewed_config.get(
                "review_failure_blocks_acceptance",
                DEFAULT_QA_CONFIG["react_reviewed"]["review_failure_blocks_acceptance"],
            )
        ),
    }
    qa_config["pdf_extraction"] = {
        "enabled": bool(pdf_extraction_config.get("enabled", DEFAULT_QA_PDF_EXTRACTION_CONFIG["enabled"])),
        "primary_backend": _coerce_allowed_text(
            pdf_extraction_config.get("primary_backend"),
            allowed={"pymupdf"},
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["primary_backend"],
        ),
        "secondary_backend": _coerce_allowed_text(
            pdf_extraction_config.get("secondary_backend"),
            allowed={"docling"},
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["secondary_backend"],
        ),
        "ocr_backend": _coerce_allowed_text(
            pdf_extraction_config.get("ocr_backend"),
            allowed={"ocrmypdf"},
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["ocr_backend"],
        ),
        "enable_ocr_fallback": bool(
            pdf_extraction_config.get(
                "enable_ocr_fallback",
                DEFAULT_QA_PDF_EXTRACTION_CONFIG["enable_ocr_fallback"],
            )
        ),
        "min_total_chars": _coerce_non_negative_int(
            pdf_extraction_config.get("min_total_chars"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["min_total_chars"],
        ),
        "min_chars_per_text_page": _coerce_positive_int(
            pdf_extraction_config.get("min_chars_per_text_page"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["min_chars_per_text_page"],
        ),
        "min_text_page_ratio": _coerce_probability(
            pdf_extraction_config.get("min_text_page_ratio"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["min_text_page_ratio"],
        ),
        "min_printable_ratio": _coerce_probability(
            pdf_extraction_config.get("min_printable_ratio"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["min_printable_ratio"],
        ),
        "snippet_target_chars": _coerce_positive_int(
            pdf_extraction_config.get("snippet_target_chars"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["snippet_target_chars"],
        ),
        "snippet_overlap_chars": _coerce_non_negative_int(
            pdf_extraction_config.get("snippet_overlap_chars"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["snippet_overlap_chars"],
        ),
        "preserve_page_blocks": bool(
            pdf_extraction_config.get(
                "preserve_page_blocks",
                DEFAULT_QA_PDF_EXTRACTION_CONFIG["preserve_page_blocks"],
            )
        ),
        "max_ocr_pages": _coerce_positive_int(
            pdf_extraction_config.get("max_ocr_pages"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["max_ocr_pages"],
        ),
        "ocr_timeout_seconds": _coerce_positive_int(
            pdf_extraction_config.get("ocr_timeout_seconds"),
            fallback=DEFAULT_QA_PDF_EXTRACTION_CONFIG["ocr_timeout_seconds"],
        ),
        "skip_ocr_when_text_already_usable": bool(
            pdf_extraction_config.get(
                "skip_ocr_when_text_already_usable",
                DEFAULT_QA_PDF_EXTRACTION_CONFIG["skip_ocr_when_text_already_usable"],
            )
        ),
    }
    qa_config["providers"] = {
        "openalex_mailto": _resolve_provider_text(provider_config.get("openalex_mailto")),
        "crossref_mailto": _resolve_provider_text(provider_config.get("crossref_mailto")),
        "semantic_scholar_api_key": _resolve_provider_text(provider_config.get("semantic_scholar_api_key")),
        "unpaywall_email": _resolve_provider_text(provider_config.get("unpaywall_email")),
        "http_timeout": _coerce_positive_float(provider_config.get("http_timeout"), fallback=10.0),
        "fetch_timeout": _coerce_positive_float(provider_config.get("fetch_timeout"), fallback=15.0),
        "retry_attempts": _coerce_non_negative_int(provider_config.get("retry_attempts"), fallback=2),
        "backoff_base_seconds": _coerce_non_negative_float(
            provider_config.get("backoff_base_seconds"),
            fallback=1.0,
        ),
        "backoff_max_seconds": _coerce_non_negative_float(
            provider_config.get("backoff_max_seconds"),
            fallback=8.0,
        ),
    }
    if qa_config["providers"]["backoff_max_seconds"] < qa_config["providers"]["backoff_base_seconds"]:
        qa_config["providers"]["backoff_max_seconds"] = qa_config["providers"]["backoff_base_seconds"]
    return qa_config


def build_qa_runtime(
    *,
    config: Dict[str, Any],
    config_path: str = "./config/config.yaml",
    grounding_pipeline: Optional[QueryGroundingPipeline] = None,
    retrieval_pipeline: Optional[HeterogeneousRetrievalPipeline] = None,
    peer_review_pipeline: Optional[StructuredPeerReviewPipeline] = None,
    synthesis_pipeline: Optional[VerifiedSynthesisPipeline] = None,
    react_reviewed_workflow: Optional[Any] = None,
) -> QARuntime:
    qa_config = resolve_qa_runtime_config(config)
    llm_configs = dict((config or {}).get("llm", {}) or {})
    warnings: list[str] = []
    llm_manifest: Dict[str, Any] = {}
    llm_cache: Dict[str, Any] = {}
    default_model_timeout = qa_config["model_timeout_seconds"]
    progress_log_every_claims = qa_config["progress_log_every_claims"]
    peer_review_config = dict(qa_config.get("peer_review", {}) or {})

    def _fallback_label(node_name: str) -> str:
        if node_name == "entity_resolver":
            return "entity mention extraction will hard-fail when grounding runs"
        return "node will use deterministic fallback"

    def get_node_llm(node_name: str) -> Any:
        requested_alias = str(qa_config["models"].get(node_name) or "").strip()
        entry = {
            "requested_alias": requested_alias or None,
            "enabled": False,
            "provider": None,
            "model": None,
            "fallback": "deterministic",
            "timeout_seconds": None,
        }
        llm_manifest[node_name] = entry

        if not requested_alias:
            message = f"qa.models.{node_name} is missing; {_fallback_label(node_name)}."
            warnings.append(message)
            logger.warning(message)
            entry["error"] = "missing_model_alias"
            return None

        model_config = llm_configs.get(requested_alias)
        if not isinstance(model_config, dict):
            message = (
                f"qa.models.{node_name} points to unknown llm alias '{requested_alias}'; "
                f"{_fallback_label(node_name)}."
            )
            warnings.append(message)
            logger.warning(message)
            entry["error"] = "unknown_model_alias"
            return None

        provider, model, has_api_key = describe_chat_model_config(model_config)
        entry["provider"] = provider
        entry["model"] = model
        if not has_api_key:
            message = (
                f"qa.models.{node_name} -> llm.{requested_alias} has no usable api_key; "
                f"{_fallback_label(node_name)}."
            )
            warnings.append(message)
            logger.warning(message)
            entry["error"] = "missing_api_key"
            return None

        configured_timeout = model_config.get("timeout") or model_config.get("request_timeout") or default_model_timeout
        entry["timeout_seconds"] = configured_timeout
        if requested_alias in llm_cache:
            entry["enabled"] = True
            return llm_cache[requested_alias]

        effective_model_config = dict(model_config)
        configured_timeout = effective_model_config.get("timeout") or effective_model_config.get("request_timeout")
        if configured_timeout in (None, "") and default_model_timeout is not None:
            effective_model_config["timeout"] = default_model_timeout
            configured_timeout = default_model_timeout
        entry["timeout_seconds"] = configured_timeout

        try:
            llm = build_chat_model_from_config(effective_model_config)
        except Exception as exc:
            message = (
                f"Failed to build chat model for qa.models.{node_name} -> llm.{requested_alias}: {exc}; "
                f"{_fallback_label(node_name)}."
            )
            warnings.append(message)
            logger.warning(message)
            entry["error"] = str(exc)
            return None

        llm_cache[requested_alias] = llm
        entry["enabled"] = True
        return llm

    def get_node_model_config(node_name: str) -> Optional[Dict[str, Any]]:
        requested_alias = str(qa_config["models"].get(node_name) or "").strip()
        entry = llm_manifest.get(node_name)
        if entry is None:
            entry = {
                "requested_alias": requested_alias or None,
                "enabled": False,
                "provider": None,
                "model": None,
                "fallback": "deterministic",
                "timeout_seconds": None,
            }
            llm_manifest[node_name] = entry
        if not requested_alias:
            return None
        model_config = llm_configs.get(requested_alias)
        if not isinstance(model_config, dict):
            return None
        provider, model, has_api_key = describe_chat_model_config(model_config)
        entry["provider"] = provider
        entry["model"] = model
        if not has_api_key:
            entry["error"] = "missing_api_key"
            return None
        effective_model_config = dict(model_config)
        configured_timeout = effective_model_config.get("timeout") or effective_model_config.get("request_timeout")
        if configured_timeout in (None, "") and default_model_timeout is not None:
            effective_model_config["timeout"] = default_model_timeout
            configured_timeout = default_model_timeout
        entry["timeout_seconds"] = configured_timeout
        entry["enabled"] = True
        return effective_model_config

    provider_config = qa_config["providers"]
    http_timeout = provider_config["http_timeout"]
    fetch_timeout = provider_config["fetch_timeout"]
    retry_attempts = provider_config["retry_attempts"]
    backoff_base_seconds = provider_config["backoff_base_seconds"]
    backoff_max_seconds = provider_config["backoff_max_seconds"]
    retry_kwargs = {
        "retry_attempts": retry_attempts,
        "backoff_base_seconds": backoff_base_seconds,
        "backoff_max_seconds": backoff_max_seconds,
    }

    openalex_client = OpenAlexClient(
        timeout=http_timeout,
        mailto=provider_config["openalex_mailto"],
        **retry_kwargs,
    )
    crossref_client = CrossrefClient(
        timeout=http_timeout,
        mailto=provider_config["crossref_mailto"],
        **retry_kwargs,
    )
    semantic_scholar_api_key = provider_config["semantic_scholar_api_key"]
    if not semantic_scholar_api_key:
        message = "qa.providers.semantic_scholar_api_key is not configured; Semantic Scholar will run without an API key."
        warnings.append(message)
        logger.warning(message)
    semantic_scholar_client = SemanticScholarClient(
        timeout=http_timeout,
        api_key=semantic_scholar_api_key,
        **retry_kwargs,
    )
    unpaywall_email = provider_config["unpaywall_email"]
    unpaywall_client = None
    if unpaywall_email:
        unpaywall_client = UnpaywallClient(
            email=unpaywall_email,
            timeout=http_timeout,
            **retry_kwargs,
        )
    else:
        message = "qa.providers.unpaywall_email is not configured; Unpaywall lookup will be skipped."
        warnings.append(message)
        logger.warning(message)

    runtime_manifest = {
        "config_path": config_path,
        "qa": {
            "workflow_mode": qa_config["workflow_mode"],
            "save_output": qa_config["save_output"],
            "outputs_dir": qa_config["outputs_dir"],
            "artifact_subdir": qa_config["artifact_subdir"],
            "enable_peer_review": qa_config["enable_peer_review"],
            "model_timeout_seconds": qa_config["model_timeout_seconds"],
            "progress_log_every_claims": qa_config["progress_log_every_claims"],
            "peer_review": qa_config["peer_review"],
            "entity_resolution": qa_config["entity_resolution"],
            "react_reviewed": qa_config["react_reviewed"],
            "pdf_extraction": qa_config["pdf_extraction"],
        },
        "models": llm_manifest,
        "providers": {
            "openalex": {
                "enabled": True,
                "mailto": provider_config["openalex_mailto"],
                "timeout": http_timeout,
                "retry_attempts": retry_attempts,
                "backoff_base_seconds": backoff_base_seconds,
                "backoff_max_seconds": backoff_max_seconds,
            },
            "crossref": {
                "enabled": True,
                "mailto": provider_config["crossref_mailto"],
                "timeout": http_timeout,
                "retry_attempts": retry_attempts,
                "backoff_base_seconds": backoff_base_seconds,
                "backoff_max_seconds": backoff_max_seconds,
            },
            "semantic_scholar": {
                "enabled": True,
                "api_key_configured": bool(semantic_scholar_api_key),
                "timeout": http_timeout,
                "retry_attempts": retry_attempts,
                "backoff_base_seconds": backoff_base_seconds,
                "backoff_max_seconds": backoff_max_seconds,
            },
            "unpaywall": {
                "enabled": bool(unpaywall_client is not None),
                "email": unpaywall_email,
                "timeout": http_timeout,
                "retry_attempts": retry_attempts,
                "backoff_base_seconds": backoff_base_seconds,
                "backoff_max_seconds": backoff_max_seconds,
            },
            "http_fetcher": {
                "enabled": True,
                "timeout": fetch_timeout,
                "retry_attempts": retry_attempts,
                "backoff_base_seconds": backoff_base_seconds,
                "backoff_max_seconds": backoff_max_seconds,
            },
            "pubchem": {
                "enabled": bool(qa_config["entity_resolution"]["pubchem_enabled"]),
                "timeout": http_timeout,
                "max_candidates": qa_config["entity_resolution"]["max_pubchem_candidates"],
                "entity_types": list(qa_config["entity_resolution"]["pubchem_entity_types"]),
            },
        },
        "warnings": warnings,
        "overrides": {
            "grounding_pipeline": grounding_pipeline is not None,
            "retrieval_pipeline": retrieval_pipeline is not None,
            "peer_review_pipeline": peer_review_pipeline is not None,
            "synthesis_pipeline": synthesis_pipeline is not None,
            "react_reviewed_workflow": react_reviewed_workflow is not None,
        },
    }

    if grounding_pipeline is None:
        entity_resolution_config = dict(qa_config.get("entity_resolution", {}) or {})
        grounding_pipeline = QueryGroundingPipeline(
            router=RouterNode(llm=get_node_llm("router")),
            entity_resolver=EntityResolverNode(
                llm=get_node_llm("entity_resolver"),
                pubchem_client=PubChemClient(timeout=http_timeout),
                seed_path=entity_resolution_config.get("seed_file"),
                pubchem_enabled=entity_resolution_config.get("pubchem_enabled", True),
                pubchem_entity_types=entity_resolution_config.get("pubchem_entity_types"),
                max_pubchem_candidates=entity_resolution_config.get("max_pubchem_candidates", 5),
                mention_extraction_min_confidence=entity_resolution_config.get(
                    "mention_extraction_min_confidence",
                    0.7,
                ),
                llm_disambiguation_enabled=entity_resolution_config.get("llm_disambiguation_enabled", True),
                disambiguation_min_confidence=entity_resolution_config.get("disambiguation_min_confidence", 0.7),
                fail_open_on_provider_error=entity_resolution_config.get("fail_open_on_provider_error", True),
                emit_seed_suggestions=entity_resolution_config.get("emit_seed_suggestions", True),
            ),
        )

    if retrieval_pipeline is None:
        handoff = EvidenceExtractorHandoff()
        retrieval_pipeline = HeterogeneousRetrievalPipeline(
            query_planner=QueryPlannerNode(llm=get_node_llm("query_planner")),
            retriever=RetrieverNode(
                openalex_client=openalex_client,
                crossref_client=crossref_client,
                semantic_scholar_client=semantic_scholar_client,
            ),
            document_acquirer=DocumentAcquirerNode(
                unpaywall_client=unpaywall_client,
                fetcher=HttpTextFetcher(timeout=fetch_timeout, **retry_kwargs),
                pdf_extractor=PDFExtractionPipeline(config=qa_config["pdf_extraction"]),
            ),
            handoff=handoff,
            evidence_extractor=EvidenceExtractor(
                handoff=handoff,
                llm=get_node_llm("evidence_extractor"),
            ),
            claim_miner=ClaimMiner(llm=get_node_llm("claim_miner")),
            ledger_builder=EvidenceLedgerBuilder(),
            peer_review_pipeline=None,
            progress_log_every=progress_log_every_claims,
        )

    if not qa_config["enable_peer_review"]:
        peer_review_pipeline = None
    elif peer_review_pipeline is None:
        peer_review_pipeline = StructuredPeerReviewPipeline(
            methodology_reviewer=MethodologyReviewer(llm=get_node_llm("methodology_reviewer")),
            citation_reviewer=CitationReviewer(llm=get_node_llm("citation_reviewer")),
            contradiction_reviewer=ContradictionReviewer(llm=get_node_llm("contradiction_reviewer")),
            claim_revision_node=ClaimRevisionNode(llm=get_node_llm("claim_revision")),
            review_merge_node=ReviewMergeNode(llm=get_node_llm("review_merge")),
            max_claims_for_llm_review=peer_review_config["max_claims_for_llm_review"],
            max_second_round_claims=peer_review_config["max_second_round_claims"],
            disable_llm_review_when_abstract_only=peer_review_config["disable_llm_review_when_abstract_only"],
            fallback_mode=peer_review_config["fallback_mode"],
            progress_log_every_claims=progress_log_every_claims,
        )

    if synthesis_pipeline is None:
        synthesis_pipeline = VerifiedSynthesisPipeline(
            synthesizer=SynthesizerNode(llm=get_node_llm("synthesizer")),
            progress_log_every=progress_log_every_claims,
        )

    if react_reviewed_workflow is None and qa_config["workflow_mode"] == "react_reviewed":
        react_reviewed_workflow = ReactReviewedWorkflow(
            qa_config=qa_config,
            router=grounding_pipeline.router,
            entity_resolver=grounding_pipeline.entity_resolver,
            query_planner=retrieval_pipeline.query_planner,
            retriever=retrieval_pipeline.retriever,
            document_acquirer=retrieval_pipeline.document_acquirer,
            handoff=retrieval_pipeline.handoff,
            evidence_extractor=retrieval_pipeline.evidence_extractor,
            proposer_model_config=get_node_model_config("react_proposer"),
            reviewer_model_configs={
                "search_coverage": get_node_model_config("react_reviewer_search_coverage") or {},
                "evidence_trace": get_node_model_config("react_reviewer_evidence_trace") or {},
                "reasoning_consistency": get_node_model_config("react_reviewer_reasoning_consistency") or {},
                "counterevidence": get_node_model_config("react_reviewer_counterevidence") or {},
            },
        )

    return QARuntime(
        qa_config=qa_config,
        grounding_pipeline=grounding_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        peer_review_pipeline=peer_review_pipeline,
        synthesis_pipeline=synthesis_pipeline,
        react_reviewed_workflow=react_reviewed_workflow,
        runtime_manifest=runtime_manifest,
    )


def _resolve_provider_text(value: Any) -> Optional[str]:
    resolved = resolve_env_var(str(value).strip()) if value is not None else None
    text = str(resolved or "").strip()
    return text or None


def _coerce_positive_float(value: Any, *, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _coerce_non_negative_float(value: Any, *, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number >= 0 else fallback


def _coerce_non_negative_int(value: Any, *, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number >= 0 else fallback


def _coerce_probability(value: Any, *, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number < 0.0 or number > 1.0:
        return fallback
    return number


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _coerce_allowed_text(value: Any, *, allowed: set[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else fallback


def _coerce_text_list(value: Any, *, fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    values = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    seen = set()
    for item in values:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned or list(fallback)
