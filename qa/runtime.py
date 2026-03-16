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
from qa.providers import (
    CrossrefClient,
    HttpTextFetcher,
    OpenAlexClient,
    SemanticScholarClient,
    UnpaywallClient,
)
from qa.retrieval_pipeline import HeterogeneousRetrievalPipeline
from qa.review_pipeline import StructuredPeerReviewPipeline
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
)

DEFAULT_QA_MODEL_ALIASES = {node_name: "agent1" for node_name in INFERENCE_NODE_MODEL_KEYS}

DEFAULT_QA_CONFIG: Dict[str, Any] = {
    "save_output": True,
    "outputs_dir": None,
    "artifact_subdir": "qa_artifacts",
    "enable_peer_review": True,
    "models": copy.deepcopy(DEFAULT_QA_MODEL_ALIASES),
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
    runtime_manifest: Dict[str, Any]


def resolve_qa_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    qa_config = copy.deepcopy(DEFAULT_QA_CONFIG)
    raw_qa_config = dict((config or {}).get("qa", {}) or {})
    raw_models = dict(raw_qa_config.pop("models", {}) or {})
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

    qa_config["save_output"] = bool(qa_config.get("save_output", DEFAULT_QA_CONFIG["save_output"]))
    qa_config["outputs_dir"] = str(outputs_dir)
    qa_config["artifact_subdir"] = artifact_subdir or str(DEFAULT_QA_CONFIG["artifact_subdir"])
    qa_config["enable_peer_review"] = bool(
        qa_config.get("enable_peer_review", DEFAULT_QA_CONFIG["enable_peer_review"])
    )
    qa_config["models"] = model_aliases
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
) -> QARuntime:
    qa_config = resolve_qa_runtime_config(config)
    llm_configs = dict((config or {}).get("llm", {}) or {})
    warnings: list[str] = []
    llm_manifest: Dict[str, Any] = {}
    llm_cache: Dict[str, Any] = {}

    def get_node_llm(node_name: str) -> Any:
        requested_alias = str(qa_config["models"].get(node_name) or "").strip()
        entry = {
            "requested_alias": requested_alias or None,
            "enabled": False,
            "provider": None,
            "model": None,
            "fallback": "deterministic",
        }
        llm_manifest[node_name] = entry

        if not requested_alias:
            message = f"qa.models.{node_name} is missing; node will use deterministic fallback."
            warnings.append(message)
            logger.warning(message)
            entry["error"] = "missing_model_alias"
            return None

        model_config = llm_configs.get(requested_alias)
        if not isinstance(model_config, dict):
            message = (
                f"qa.models.{node_name} points to unknown llm alias '{requested_alias}'; "
                "node will use deterministic fallback."
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
                "node will use deterministic fallback."
            )
            warnings.append(message)
            logger.warning(message)
            entry["error"] = "missing_api_key"
            return None

        if requested_alias in llm_cache:
            entry["enabled"] = True
            return llm_cache[requested_alias]

        try:
            llm = build_chat_model_from_config(model_config)
        except Exception as exc:
            message = (
                f"Failed to build chat model for qa.models.{node_name} -> llm.{requested_alias}: {exc}; "
                "node will use deterministic fallback."
            )
            warnings.append(message)
            logger.warning(message)
            entry["error"] = str(exc)
            return None

        llm_cache[requested_alias] = llm
        entry["enabled"] = True
        return llm

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
            "save_output": qa_config["save_output"],
            "outputs_dir": qa_config["outputs_dir"],
            "artifact_subdir": qa_config["artifact_subdir"],
            "enable_peer_review": qa_config["enable_peer_review"],
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
        },
        "warnings": warnings,
        "overrides": {
            "grounding_pipeline": grounding_pipeline is not None,
            "retrieval_pipeline": retrieval_pipeline is not None,
            "peer_review_pipeline": peer_review_pipeline is not None,
            "synthesis_pipeline": synthesis_pipeline is not None,
        },
    }

    if grounding_pipeline is None:
        grounding_pipeline = QueryGroundingPipeline(
            router=RouterNode(llm=get_node_llm("router")),
            entity_resolver=EntityResolverNode(
                llm=get_node_llm("entity_resolver"),
                pubchem_client=PubChemClient(timeout=http_timeout),
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
            ),
            handoff=handoff,
            evidence_extractor=EvidenceExtractor(
                handoff=handoff,
                llm=get_node_llm("evidence_extractor"),
            ),
            claim_miner=ClaimMiner(llm=get_node_llm("claim_miner")),
            ledger_builder=EvidenceLedgerBuilder(),
            peer_review_pipeline=None,
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
        )

    if synthesis_pipeline is None:
        synthesis_pipeline = VerifiedSynthesisPipeline(
            synthesizer=SynthesizerNode(llm=get_node_llm("synthesizer")),
        )

    return QARuntime(
        qa_config=qa_config,
        grounding_pipeline=grounding_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        peer_review_pipeline=peer_review_pipeline,
        synthesis_pipeline=synthesis_pipeline,
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
