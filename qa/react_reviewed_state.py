from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from qa.state import StrictModel


WorkflowMode = Literal["ledger", "react_reviewed"]
ReviewerRole = Literal[
    "search_coverage",
    "evidence_trace",
    "reasoning_consistency",
    "counterevidence",
]
AnchorKind = Literal["step_section", "section_only", "global", "missing_section"]
ReviewSeverity = Literal["blocking", "warning", "note"]
ReviewItemStatus = Literal["open", "addressed", "dismissed"]
ReviewResponseMode = Literal["addressed", "partially_addressed", "disagree"]
ReviewCompletionStatus = Literal["completed", "incomplete"]
ConfidenceLevel = Literal["high", "medium", "low"]


class SubmissionConfidenceRating(StrictModel):
    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class SubmissionStepRef(StrictModel):
    trajectory_id: str
    step_number: int = Field(ge=1)


class SubmissionCitation(StrictModel):
    citation_id: str
    paper_id: str
    doi: Optional[str] = None
    title: str
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    section_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("section_ids", "evidence_ids", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class SubmissionSection(StrictModel):
    section_id: str
    title: str
    content: str
    citation_ids: List[str] = Field(default_factory=list)
    step_refs: List[SubmissionStepRef] = Field(default_factory=list)
    issue_refs: List[str] = Field(default_factory=list)
    section_confidence: SubmissionConfidenceRating

    @field_validator("citation_ids", "issue_refs", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class SubmissionTraceItem(StrictModel):
    section_id: str
    citation_ids: List[str] = Field(default_factory=list)
    step_refs: List[SubmissionStepRef] = Field(default_factory=list)
    issue_refs: List[str] = Field(default_factory=list)

    @field_validator("citation_ids", "issue_refs", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class AnswerSubmission(StrictModel):
    submission_id: str
    question: str
    version: int = Field(default=1, ge=1)
    sections: List[SubmissionSection] = Field(default_factory=list)
    citations: List[SubmissionCitation] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    overall_confidence: SubmissionConfidenceRating
    trajectory_id: str
    step_refs: List[SubmissionStepRef] = Field(default_factory=list)
    issue_refs: List[str] = Field(default_factory=list)

    @field_validator("limitations", "issue_refs", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ReviewItem(StrictModel):
    review_id: str
    reviewer_role: ReviewerRole
    anchor_kind: AnchorKind
    severity: ReviewSeverity
    flaw_type: str
    critique: str
    required_action: str
    evidence_refs: List[str] = Field(default_factory=list)
    status: ReviewItemStatus = "open"
    target_trajectory_id: Optional[str] = None
    target_step_number: Optional[int] = Field(default=None, ge=1)
    target_section_id: Optional[str] = None

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def coerce_evidence_refs(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @model_validator(mode="after")
    def validate_anchor(self) -> "ReviewItem":
        if self.anchor_kind == "step_section":
            if not self.target_trajectory_id or self.target_step_number is None or not self.target_section_id:
                raise ValueError("step_section reviews require trajectory, step_number, and section_id")
        elif self.anchor_kind == "section_only":
            if not self.target_section_id:
                raise ValueError("section_only reviews require section_id")
        return self


class ReviewResponse(StrictModel):
    review_id: str
    response_mode: ReviewResponseMode
    response_note: str
    new_step_refs: List[SubmissionStepRef] = Field(default_factory=list)
    section_patch_refs: List[str] = Field(default_factory=list)

    @field_validator("section_patch_refs", mode="before")
    @classmethod
    def coerce_patch_refs(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ReviewerRunStatus(StrictModel):
    reviewer_role: ReviewerRole
    status: Literal["completed", "salvaged", "timeout", "invalid_json", "error"]
    message: str = ""
    cycle_number: int = Field(ge=1)
    retrieval_actions_used: int = Field(default=0, ge=0)
    retrieval_budget_limit: int = Field(default=0, ge=0)
    budget_blocked_calls: int = Field(default=0, ge=0)


class SubmissionCycleState(StrictModel):
    cycle_number: int = Field(ge=1)
    current_submission: AnswerSubmission
    proposer_trajectory: Dict[str, object]
    reviewer_trajectories: Dict[str, Dict[str, object]] = Field(default_factory=dict)
    open_review_items: List[ReviewItem] = Field(default_factory=list)
    review_responses: List[ReviewResponse] = Field(default_factory=list)
    reviewer_statuses: List[ReviewerRunStatus] = Field(default_factory=list)
