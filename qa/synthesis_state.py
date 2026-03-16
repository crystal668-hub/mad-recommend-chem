from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator

from qa.state import StrictModel, TaskSpec


ConfidenceLevel = Literal["high", "medium", "low"]


class ConfidenceRating(StrictModel):
    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class SectionConfidenceRecord(StrictModel):
    section_id: str
    title: str
    confidence: ConfidenceRating


class CitationRecord(StrictModel):
    citation_id: str
    paper_id: str
    doi: Optional[str] = None
    title: str
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    venue: Optional[str] = None
    supporting_claim_ids: List[str] = Field(default_factory=list)

    @field_validator("supporting_claim_ids", mode="before")
    @classmethod
    def coerce_claim_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ContestedClaimRecord(StrictModel):
    claim_id: str
    claim_summary: str
    citation_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str

    @field_validator("citation_ids", mode="before")
    @classmethod
    def coerce_citation_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class ClaimTraceItem(StrictModel):
    section_id: str
    claim_id: str
    status: Literal["accepted", "contested"]
    citation_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("citation_ids", mode="before")
    @classmethod
    def coerce_citation_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class SectionClaimPack(StrictModel):
    section_id: str
    title: str
    accepted_claim_ids: List[str] = Field(default_factory=list)
    claim_summaries: List[str] = Field(default_factory=list)
    core_citation_ids: List[str] = Field(default_factory=list)
    section_confidence: ConfidenceRating

    @field_validator("accepted_claim_ids", "claim_summaries", "core_citation_ids", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class SynthesisInputPack(StrictModel):
    question: str
    task_spec: TaskSpec
    section_claims: List[SectionClaimPack] = Field(default_factory=list)
    contested_claims: List[ContestedClaimRecord] = Field(default_factory=list)
    citation_catalog: List[CitationRecord] = Field(default_factory=list)
    overall_confidence: ConfidenceRating
    section_confidence: List[SectionConfidenceRecord] = Field(default_factory=list)
    insufficient_evidence: bool = False
    claim_trace: List[ClaimTraceItem] = Field(default_factory=list)
    retrieval_diagnostics_summary: str = ""
    execution_warnings: List[str] = Field(default_factory=list)

    @field_validator("execution_warnings", mode="before")
    @classmethod
    def coerce_execution_warnings(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class AnswerSectionOutput(StrictModel):
    section_id: str
    title: str
    content: str
    citation_ids: List[str] = Field(default_factory=list)
    section_confidence: ConfidenceRating

    @field_validator("citation_ids", mode="before")
    @classmethod
    def coerce_citation_ids(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []


class QAResult(StrictModel):
    question: str
    language: str = "en"
    final_answer: str
    sections: List[AnswerSectionOutput] = Field(default_factory=list)
    citations: List[CitationRecord] = Field(default_factory=list)
    claim_trace: List[ClaimTraceItem] = Field(default_factory=list)
    overall_confidence: ConfidenceRating
    section_confidence: List[SectionConfidenceRecord] = Field(default_factory=list)
    insufficient_evidence: bool = False
    limitations_summary: str = ""
    retrieval_diagnostics_summary: str = ""
    execution_warnings: List[str] = Field(default_factory=list)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    time_elapsed: float = Field(default=0.0, ge=0.0)

    @field_validator("execution_warnings", mode="before")
    @classmethod
    def coerce_result_warnings(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []
