from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GROUNDING_VERSION = "1.0"

QuestionType = Literal["fact", "causal", "mechanism", "comparison", "frontier"]
RecencyPolicy = Literal["none", "last_3y", "last_5y", "explicit"]
ConditionAxis = Literal[
    "catalyst",
    "material",
    "substrate",
    "solvent",
    "ligand",
    "reagent",
    "temperature",
    "time",
    "ph",
    "electrolyte",
    "potential",
    "pressure",
    "yield",
    "selectivity",
]
AmbiguityFlagType = Literal[
    "entity_ambiguous",
    "metric_ambiguous",
    "time_ambiguous",
    "task_ambiguous",
    "condition_ambiguous",
]
AmbiguitySeverity = Literal["low", "medium", "high"]
EntityType = Literal[
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
EntityStatus = Literal["resolved", "partially_resolved", "unresolved"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpan(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceSpan":
        if self.end < self.start:
            raise ValueError("source span end must be >= start")
        return self


class AnswerSection(StrictModel):
    section_id: str
    title: str
    required: bool
    instruction: str


class AmbiguityFlag(StrictModel):
    flag_type: AmbiguityFlagType
    target: str
    note: str
    severity: AmbiguitySeverity


class QueryConstraints(StrictModel):
    must_include_terms: List[str] = Field(default_factory=list)
    should_include_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)
    preferred_entity_types: List[EntityType] = Field(default_factory=list)
    allow_broad_expansion: bool = True

    @field_validator(
        "must_include_terms",
        "should_include_terms",
        "exclude_terms",
        "preferred_entity_types",
        mode="before",
    )
    @classmethod
    def coerce_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class TaskSpec(StrictModel):
    version: str = GROUNDING_VERSION
    question: str
    normalized_question: str
    question_type: QuestionType
    recency_policy: RecencyPolicy
    year_from: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1900, le=2100)
    answer_sections: List[AnswerSection] = Field(default_factory=list)
    required_condition_axes: List[ConditionAxis] = Field(default_factory=list)
    query_constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    ambiguity_flags: List[AmbiguityFlag] = Field(default_factory=list)
    router_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("required_condition_axes", mode="before")
    @classmethod
    def coerce_axes(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @model_validator(mode="after")
    def validate_years(self) -> "TaskSpec":
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


class EntityRecord(StrictModel):
    entity_id: str
    mention: str
    canonical_name: str
    entity_type: EntityType
    entity_subtype: Optional[str] = None
    formula: Optional[str] = None
    smiles: Optional[str] = None
    inchikey: Optional[str] = None
    pubchem_cid: Optional[int] = None
    aliases: List[str] = Field(default_factory=list)
    query_anchors: List[str] = Field(default_factory=list)
    resolver_source: str
    resolution_confidence: float = Field(ge=0.0, le=1.0)
    status: EntityStatus
    source_text: str
    source_span: SourceSpan

    @field_validator("aliases", "query_anchors", mode="before")
    @classmethod
    def coerce_text_list(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


class ConditionMention(StrictModel):
    condition_id: str
    axis: ConditionAxis
    raw_value: str
    normalized_value: Optional[str] = None
    unit: Optional[str] = None
    operator: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str
    source_span: SourceSpan


class UnresolvedMention(StrictModel):
    mention: str
    candidate_entity_types: List[EntityType] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str
    source_span: SourceSpan


class EntityPack(StrictModel):
    version: str = GROUNDING_VERSION
    entities: List[EntityRecord] = Field(default_factory=list)
    condition_mentions: List[ConditionMention] = Field(default_factory=list)
    unresolved_mentions: List[UnresolvedMention] = Field(default_factory=list)
    entity_ambiguity_flags: List[AmbiguityFlag] = Field(default_factory=list)


class GroundingState(StrictModel):
    question: str
    context: Optional[str] = None
    task_spec: TaskSpec
    entity_pack: EntityPack
