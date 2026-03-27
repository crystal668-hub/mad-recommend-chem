from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, get_args

from pydantic import Field, field_validator, model_validator

from qa.state import EntityPack, StrictModel, TaskSpec
from qa.state import ConditionAxis, SourceSpan


LaneType = Literal["review", "frontier", "data", "contrarian"]
ProviderName = Literal["openalex", "crossref", "semantic_scholar", "unpaywall"]
DiagnosticProviderName = Literal["openalex", "crossref", "semantic_scholar", "unpaywall", "oa_fetch", "pdf_probe"]
DiagnosticStage = Literal["search", "enrichment", "lookup", "fetch"]
SectionType = Literal[
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "unknown",
]
FulltextStatus = Literal["abstract_only", "fulltext_indexed", "binary_only", "fulltext_unusable", "missing", "error"]
PaperProfileStatus = Literal["ready", "error"]
EvidenceRole = Literal["observation", "condition", "limitation", "mechanism"]
SourceLayer = Literal["abstract", "fulltext"]
ClaimPolarity = Literal["support", "oppose", "neutral"]
ClaimType = Literal["fact", "causal", "mechanism", "comparison", "frontier_summary"]
ClaimStatus = Literal["draft", "accepted", "contested", "rejected"]
ReviewerType = Literal["MethodologyReviewer", "CitationReviewer", "ContradictionReviewer", "ClaimRevisionNode", "ReviewMergeNode"]
ReviewFlagType = Literal[
    "Missing_Condition",
    "Incomplete_Condition",
    "Overgeneralized",
    "Unsupported",
    "Fabricated_Citation",
    "Weak_Evidence",
    "True_Conflict",
    "Condition_Divergence",
    "Mechanism_Speculative",
    "Metric_Mismatch",
]
ReviewSeverity = Literal["info", "warning", "critical"]
ConflictType = Literal["true_conflict", "condition_divergence"]
RevisionAction = Literal["keep", "narrow", "downgrade", "evidence_rebalance"]

CONDITION_AXES = set(get_args(ConditionAxis))


class QueryPlan(StrictModel):
    lane: LaneType
    query_text: str
    must_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)
    year_from: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1900, le=2100)
    preferred_sources: List[ProviderName] = Field(default_factory=list)

    @field_validator("must_terms", "exclude_terms", "preferred_sources", mode="before")
    @classmethod
    def coerce_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @model_validator(mode="after")
    def validate_years(self) -> "QueryPlan":
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


class PaperCandidate(StrictModel):
    paper_id: str
    doi: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    tldr: Optional[str] = None
    fields_of_study: List[str] = Field(default_factory=list)
    is_open_access: Optional[bool] = None
    open_access_pdf_url: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    provider_hits: List[str] = Field(default_factory=list)
    lane_sources: List[LaneType] = Field(default_factory=list)
    retrieval_score: float = Field(default=0.0)
    ranking_features: Dict[str, Any] = Field(default_factory=dict)
    provider_artifacts: Dict[str, str] = Field(default_factory=dict)
    oa_url: Optional[str] = None
    openalex_id: Optional[str] = None
    best_oa_pdf_url: Optional[str] = None
    best_oa_landing_page_url: Optional[str] = None
    oa_eligible: bool = False
    oa_source: Optional[str] = None
    oa_signal_reason: Optional[str] = None
    pdf_probe_verdict: Optional[Literal["strong", "weak"]] = None
    pdf_probe_method: Optional[Literal["head", "range_get"]] = None
    pdf_probe_final_url: Optional[str] = None

    @field_validator("authors", "provider_hits", "lane_sources", "fields_of_study", mode="before")
    @classmethod
    def coerce_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class PaperRecord(StrictModel):
    paper_id: str
    doi: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    provider_sources: List[str] = Field(default_factory=list)
    provider_artifacts: Dict[str, str] = Field(default_factory=dict)
    oa_url: Optional[str] = None
    fulltext_available: bool = False
    fulltext_status: FulltextStatus = "missing"
    fulltext_format: Optional[str] = None
    fulltext_artifact_path: Optional[str] = None
    source_artifact_path: Optional[str] = None
    index_artifact_path: Optional[str] = None
    extraction_report_path: Optional[str] = None
    sections_artifact_path: Optional[str] = None
    snippets_artifact_path: Optional[str] = None
    extraction_warnings: List[str] = Field(default_factory=list)
    fulltext_extractor: Optional[str] = None
    ocr_applied: bool = False

    @field_validator("authors", "provider_sources", "extraction_warnings", mode="before")
    @classmethod
    def coerce_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class PaperProfile(StrictModel):
    paper_id: str
    title: str
    doi: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    source_artifact_path: Optional[str] = None
    profile_status: PaperProfileStatus = "ready"
    profile_xml_artifact_path: Optional[str] = None
    error_message: Optional[str] = None


class Section(StrictModel):
    section_id: str
    section_type: SectionType
    heading: str
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    fulltext_char_start: int = Field(ge=0)
    fulltext_char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "Section":
        if self.fulltext_char_end < self.fulltext_char_start:
            raise ValueError("section end must be >= section start")
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("section page_end must be >= page_start")
        return self


class SectionIndex(StrictModel):
    paper_id: str
    fulltext_status: FulltextStatus
    sections: List[Section] = Field(default_factory=list)


class SectionTextView(StrictModel):
    paper_id: str
    section_id: str
    section_type: SectionType
    heading: str
    text: str
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    fulltext_char_start: int = Field(ge=0)
    fulltext_char_end: int = Field(ge=0)


class RetrievalDiagnosticRecord(StrictModel):
    provider: DiagnosticProviderName
    stage: DiagnosticStage
    lane: Optional[LaneType] = None
    hit_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    timeout_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    empty_count: int = Field(default=0, ge=0)
    sample_messages: List[str] = Field(default_factory=list)

    @field_validator("sample_messages", mode="before")
    @classmethod
    def coerce_messages(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class EvidenceItem(StrictModel):
    evidence_id: str
    paper_id: str
    doi: Optional[str] = None
    section_id: str
    section_type: SectionType
    role: EvidenceRole
    snippet: str
    source_span: SourceSpan
    source_layer: SourceLayer
    claim_polarity: ClaimPolarity = "neutral"
    conditions: Dict[str, str] = Field(default_factory=dict)
    condition_source_refs: List[str] = Field(default_factory=list)
    metric_mentions: List[str] = Field(default_factory=list)
    entity_mentions: List[str] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_notes: Optional[str] = None

    @field_validator("conditions", mode="before")
    @classmethod
    def normalize_conditions(cls, value):
        if not value:
            return {}
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, str] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key or "").strip().lower()
            if key not in CONDITION_AXES:
                continue
            val = str(raw_val or "").strip()
            if not val:
                continue
            normalized[key] = val
        return normalized

    @field_validator("condition_source_refs", "metric_mentions", "entity_mentions", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ClaimRecord(StrictModel):
    claim_id: str
    claim_type: ClaimType
    section_id: str
    claim_text: str
    main_entity: str
    relation_type: str
    metric_family: str
    condition_scope: Dict[str, str] = Field(default_factory=dict)
    condition_signature: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    opposing_evidence_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus = "draft"
    claim_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster_size: int = Field(default=0, ge=0)
    provenance_notes: Optional[str] = None

    @field_validator("condition_scope", mode="before")
    @classmethod
    def normalize_condition_scope(cls, value):
        if not value:
            return {}
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, str] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key or "").strip().lower()
            if key not in CONDITION_AXES:
                continue
            val = str(raw_val or "").strip()
            if val:
                normalized[key] = val
        return normalized

    @field_validator("supporting_evidence_ids", "opposing_evidence_ids", mode="before")
    @classmethod
    def coerce_id_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @model_validator(mode="after")
    def normalize_signature(self) -> "ClaimRecord":
        normalized_signature = json.dumps(self.condition_scope, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if not self.condition_signature:
            self.condition_signature = normalized_signature
        return self


class ReviewFlag(StrictModel):
    flag_id: str
    claim_id: str
    reviewer_type: ReviewerType
    flag_type: ReviewFlagType
    severity: ReviewSeverity
    note: str
    evidence_refs: List[str] = Field(default_factory=list)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def coerce_refs(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ConflictEdge(StrictModel):
    conflict_id: str
    left_claim_id: str
    right_claim_id: str
    conflict_type: ConflictType
    severity: ReviewSeverity
    reason: str
    shared_axes: List[str] = Field(default_factory=list)
    differing_axes: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)

    @field_validator("shared_axes", "differing_axes", "evidence_refs", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ClaimRevisionRecord(StrictModel):
    claim_id: str
    original_claim_text: str
    revised_claim_text: str
    revision_action: RevisionAction
    updated_condition_scope: Dict[str, str] = Field(default_factory=dict)
    updated_supporting_evidence_ids: List[str] = Field(default_factory=list)
    updated_opposing_evidence_ids: List[str] = Field(default_factory=list)
    revision_rationale: str

    @field_validator("updated_condition_scope", mode="before")
    @classmethod
    def normalize_scope(cls, value):
        if not value:
            return {}
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, str] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key or "").strip().lower()
            if key not in CONDITION_AXES:
                continue
            val = str(raw_val or "").strip()
            if val:
                normalized[key] = val
        return normalized

    @field_validator("updated_supporting_evidence_ids", "updated_opposing_evidence_ids", mode="before")
    @classmethod
    def coerce_id_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ReviewSummary(StrictModel):
    claim_id: str
    review_rounds: int = Field(ge=1, le=2)
    review_flags: List[ReviewFlag] = Field(default_factory=list)
    conflict_edge_ids: List[str] = Field(default_factory=list)
    revision_records: List[ClaimRevisionRecord] = Field(default_factory=list)
    final_status: ClaimStatus
    merge_rationale: str

    @field_validator("conflict_edge_ids", mode="before")
    @classmethod
    def coerce_conflict_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class EvidenceLedger(StrictModel):
    version: str = "1.0"
    claims: List[ClaimRecord] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    claim_index: Dict[str, int] = Field(default_factory=dict)
    evidence_index: Dict[str, int] = Field(default_factory=dict)
    cluster_stats: Dict[str, Any] = Field(default_factory=dict)
    ledger_notes: List[str] = Field(default_factory=list)
    review_flags: List[ReviewFlag] = Field(default_factory=list)
    conflict_edges: List[ConflictEdge] = Field(default_factory=list)
    review_summaries: List[ReviewSummary] = Field(default_factory=list)

    @field_validator("ledger_notes", mode="before")
    @classmethod
    def coerce_notes(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @model_validator(mode="after")
    def populate_indexes(self) -> "EvidenceLedger":
        if not self.claim_index:
            self.claim_index = {claim.claim_id: index for index, claim in enumerate(self.claims)}
        if not self.evidence_index:
            self.evidence_index = {item.evidence_id: index for index, item in enumerate(self.evidence_items)}
        default_stats = {
            "claim_count": len(self.claims),
            "evidence_count": len(self.evidence_items),
            "support_edge_count": sum(len(claim.supporting_evidence_ids) for claim in self.claims),
            "oppose_edge_count": sum(len(claim.opposing_evidence_ids) for claim in self.claims),
        }
        if not self.cluster_stats:
            self.cluster_stats = default_stats
        else:
            for key, value in default_stats.items():
                self.cluster_stats.setdefault(key, value)
        return self


class RetrievalState(StrictModel):
    question: str
    context: Optional[str] = None
    task_spec: TaskSpec
    entity_pack: EntityPack
    query_plans: List[QueryPlan] = Field(default_factory=list)
    paper_candidates: List[PaperCandidate] = Field(default_factory=list)
    paper_records: List[PaperRecord] = Field(default_factory=list)
    section_indices: List[SectionIndex] = Field(default_factory=list)
    retrieval_diagnostics: List[RetrievalDiagnosticRecord] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    evidence_ledger: Optional[EvidenceLedger] = None
    execution_warnings: List[str] = Field(default_factory=list)
    artifact_dir: str

    @field_validator("execution_warnings", mode="before")
    @classmethod
    def coerce_execution_warnings(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []
