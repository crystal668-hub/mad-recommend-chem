from __future__ import annotations

from typing import List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class ToolArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyToolInput(ToolArgsModel):
    pass


class PlanQueriesToolInput(ToolArgsModel):
    focus: Literal["initial", "revision"] = Field(
        default="initial",
        description="Planning focus for the current proposer cycle.",
    )


class SearchPapersToolInput(ToolArgsModel):
    query_plan_id: Optional[StrictStr] = Field(
        default=None,
        description="Stable query plan identifier from plan_queries.",
    )
    query_text: Optional[StrictStr] = Field(
        default=None,
        description="Ad hoc search query text when query_plan_id is omitted.",
    )
    lane: StrictStr = Field(
        default="data",
        description="Retrieval lane label for the current search.",
    )
    reason: StrictStr = Field(
        default="",
        description="Short reason for why this search is being run.",
    )


class ScreenPapersToolInput(ToolArgsModel):
    paper_ids: List[StrictStr] = Field(
        default_factory=list,
        description="Optional stable paper identifiers to screen; defaults to the current cycle's searched papers.",
    )
    max_candidates: StrictInt = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of acquired papers to lock after profile-based reranking.",
    )


class ReviewerSearchPapersToolInput(SearchPapersToolInput):
    lane: StrictStr = Field(
        default="contrarian",
        description="Retrieval lane label for the current reviewer search.",
    )


class DownloadDocumentToolInput(ToolArgsModel):
    paper_id: StrictStr = Field(description="Stable paper identifier from search_papers.")


class ParseDocumentToolInput(ToolArgsModel):
    paper_id: StrictStr = Field(description="Stable paper identifier from search_papers.")


class SectionAccessToolInput(ToolArgsModel):
    paper_id: StrictStr = Field(description="Stable paper identifier from parse_document.")
    section_ids: List[StrictStr] = Field(
        default_factory=list,
        description="Optional stable section identifiers to target.",
    )
    preferred_sections: StrictBool = Field(
        default=False,
        description="Whether to prefer the system's recommended sections.",
    )


class FetchCitationContextToolInput(ToolArgsModel):
    paper_id: Optional[StrictStr] = Field(default=None, description="Optional paper identifier.")
    section_id: Optional[StrictStr] = Field(default=None, description="Optional section identifier.")
    evidence_id: Optional[StrictStr] = Field(default=None, description="Optional evidence identifier.")
    citation_id: Optional[StrictStr] = Field(default=None, description="Optional citation identifier.")


class InspectEntityCacheToolInput(ToolArgsModel):
    name: Optional[StrictStr] = Field(default=None, description="Optional entity name filter.")
    entity_type: Optional[StrictStr] = Field(default=None, description="Optional entity type filter.")
    limit: StrictInt = Field(default=10, ge=1, le=100, description="Maximum number of cached entities to return.")


class InspectSubmissionAnchorToolInput(ToolArgsModel):
    section_id: Optional[StrictStr] = Field(default=None, description="Optional submission section identifier.")
    step_number: Optional[StrictInt] = Field(default=None, ge=1, description="Optional proposer step number.")
    review_id: Optional[StrictStr] = Field(default=None, description="Optional review identifier.")


class SearchLiteratureToolInput(ToolArgsModel):
    query: StrictStr = Field(description="Search query for the literature RAG store.")
    top_k: StrictInt = Field(default=5, ge=1, le=50, description="Maximum number of literature chunks to return.")


class FetchLiteratureChunkToolInput(ToolArgsModel):
    source_id: StrictStr = Field(description="Canonical rag:chroma/... source identifier.")


class SearchExperienceToolInput(ToolArgsModel):
    components: List[StrictStr] = Field(
        default_factory=list,
        description="Catalyst or material components used to search prior experience.",
    )
    top_k: StrictInt = Field(default=5, ge=1, le=50, description="Maximum number of experience items to return.")


class AnalyzeToolInput(ToolArgsModel):
    gap_analysis: StrictStr = Field(description="Current gaps, uncertainties, or findings.")
    next_step_plan: StrictStr = Field(description="Next action plan after the analysis step.")


class SubmissionConfidenceRatingToolInput(ToolArgsModel):
    level: Literal["high", "medium", "low"] = Field(description="Discrete confidence level.")
    score: StrictFloat = Field(ge=0.0, le=1.0, description="Continuous confidence score.")
    rationale: StrictStr = Field(description="Short explanation for the confidence score.")


class SubmissionStepRefToolInput(ToolArgsModel):
    trajectory_id: StrictStr = Field(description="Trajectory identifier that produced this step reference.")
    step_number: StrictInt = Field(ge=1, description="1-based step number in the referenced trajectory.")


class SubmissionCitationToolInput(ToolArgsModel):
    citation_id: StrictStr = Field(description="Stable citation identifier used by submission sections.")
    paper_id: StrictStr = Field(description="Stable paper identifier for the cited paper.")
    doi: Optional[StrictStr] = Field(default=None, description="Optional DOI for the cited paper.")
    title: StrictStr = Field(description="Paper title.")
    year: Optional[StrictInt] = Field(default=None, ge=1900, le=2100, description="Optional publication year.")
    venue: Optional[StrictStr] = Field(default=None, description="Optional publication venue.")
    section_ids: List[StrictStr] = Field(
        default_factory=list,
        description="Stable section anchors from this retrieval cycle.",
    )
    evidence_ids: List[StrictStr] = Field(
        default_factory=list,
        description="Stable evidence anchors from this retrieval cycle.",
    )


class SubmissionSectionToolInput(ToolArgsModel):
    section_id: StrictStr = Field(description="Stable answer section identifier from TaskSpec.answer_sections.")
    title: StrictStr = Field(description="Section title.")
    content: StrictStr = Field(description="Section content.")
    citation_ids: List[StrictStr] = Field(
        default_factory=list,
        description="Citation ids supporting this section.",
    )
    step_refs: List[SubmissionStepRefToolInput] = Field(
        default_factory=list,
        description="Trajectory step references supporting this section.",
    )
    issue_refs: List[StrictStr] = Field(
        default_factory=list,
        description="Open review item identifiers attached to this section.",
    )
    section_confidence: SubmissionConfidenceRatingToolInput = Field(
        description="Section-level confidence rating.",
    )


class AnswerSubmissionToolInput(ToolArgsModel):
    submission_id: StrictStr = Field(description="Stable submission identifier for this cycle.")
    question: StrictStr = Field(description="Original user question.")
    version: StrictInt = Field(default=1, ge=1, description="Monotonic submission version.")
    sections: List[SubmissionSectionToolInput] = Field(
        default_factory=list,
        description="Canonical answer sections.",
    )
    citations: List[SubmissionCitationToolInput] = Field(
        default_factory=list,
        description="Canonical submission citation catalog.",
    )
    limitations: List[StrictStr] = Field(
        default_factory=list,
        description="Explicit limitations disclosed by the submission.",
    )
    overall_confidence: SubmissionConfidenceRatingToolInput = Field(
        description="Overall submission confidence rating.",
    )
    trajectory_id: StrictStr = Field(description="Primary trajectory identifier for this submission.")
    step_refs: List[SubmissionStepRefToolInput] = Field(
        default_factory=list,
        description="Top-level trajectory step references.",
    )
    issue_refs: List[StrictStr] = Field(
        default_factory=list,
        description="Open review items carried into the submission.",
    )


class ProposerConcludeToolInput(ToolArgsModel):
    submission: AnswerSubmissionToolInput = Field(
        description=(
            "Final canonical AnswerSubmission payload. This is the only top-level conclude argument: "
            "call conclude with exactly {'submission': {...}} and do not rename the wrapper key or pass a bare payload."
        )
    )


class ReviewItemToolInput(ToolArgsModel):
    review_id: Optional[StrictStr] = Field(default=None, description="Stable review item identifier.")
    reviewer_role: Optional[StrictStr] = Field(default=None, description="Reviewer role producing this item.")
    anchor_kind: Optional[Literal["step_section", "section_only", "global", "missing_section"]] = Field(
        default=None,
        description="Anchor mode used by the review item.",
    )
    severity: Optional[Literal["blocking", "warning", "note"]] = Field(
        default=None,
        description="Severity level for the review item.",
    )
    flaw_type: Optional[StrictStr] = Field(default=None, description="Short flaw taxonomy label.")
    critique: Optional[StrictStr] = Field(default=None, description="Critique text.")
    required_action: Optional[StrictStr] = Field(default=None, description="Action needed to address this issue.")
    evidence_refs: List[StrictStr] = Field(
        default_factory=list,
        description="Optional evidence identifiers or citation references.",
    )
    status: Optional[Literal["open", "addressed", "dismissed"]] = Field(
        default=None,
        description="Optional review status.",
    )
    target_trajectory_id: Optional[StrictStr] = Field(default=None, description="Optional target trajectory id.")
    target_step_number: Optional[StrictInt] = Field(default=None, ge=1, description="Optional target step number.")
    target_section_id: Optional[StrictStr] = Field(default=None, description="Optional target section id.")


class ReviewPayloadToolInput(ToolArgsModel):
    review_items: List[ReviewItemToolInput] = Field(
        default_factory=list,
        description="Canonical reviewer payload with review_items.",
    )


class ReviewerConcludeToolInput(ToolArgsModel):
    review: ReviewPayloadToolInput = Field(description="Final canonical reviewer payload.")


class GenericTextConcludeToolInput(ToolArgsModel):
    conclusion: StrictStr = Field(description="Final free-form answer text.")


def get_generic_conclude_args_schema(_schema_kind: str | None = None) -> Type[ToolArgsModel]:
    return GenericTextConcludeToolInput
