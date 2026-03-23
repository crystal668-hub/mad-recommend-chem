from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prompts.qa_prompts import (
    ROUTER_LOCALIZATION_SYSTEM_PROMPT,
    ROUTER_SEMANTIC_SYSTEM_PROMPT,
    build_router_localization_user_prompt,
    build_router_semantic_user_prompt,
)
from qa.llm_utils import invoke_llm, parse_json_object
from qa.state import AmbiguityFlag, AnswerSection, QueryConstraints, TaskSpec


QUESTION_TYPE_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "causal": (
        re.compile(r"\baffect\b", re.I),
        re.compile(r"\binfluence\b", re.I),
        re.compile(r"\blead to\b", re.I),
        re.compile(r"\bcause\b", re.I),
        re.compile(r"\bpromote\b", re.I),
        re.compile(r"\bsuppress\b", re.I),
        re.compile(r"\bincrease\b", re.I),
        re.compile(r"\bdecrease\b", re.I),
        re.compile(r"\bdoes\s+.+\s+affect\s+.+", re.I),
    ),
    "comparison": (
        re.compile(r"\bvs\.?\b", re.I),
        re.compile(r"\bversus\b", re.I),
        re.compile(r"\bcompared with\b", re.I),
        re.compile(r"\bcompare\b", re.I),
        re.compile(r"\bbetter than\b", re.I),
    ),
    "frontier": (
        re.compile(r"\blatest\b", re.I),
        re.compile(r"\brecent\b", re.I),
        re.compile(r"\bprogress\b", re.I),
        re.compile(r"\badvances?\b", re.I),
        re.compile(r"\bfrontier\b", re.I),
        re.compile(r"\bstate of the art\b", re.I),
    ),
    "mechanism": (
        re.compile(r"\bwhy\b", re.I),
        re.compile(r"\bhow\b", re.I),
        re.compile(r"\bmechanism\b", re.I),
        re.compile(r"\bpathway\b", re.I),
    ),
}

ANSWER_SECTION_TEMPLATES: Dict[str, List[Tuple[str, str, bool, str]]] = {
    "fact": [
        ("direct_answer", "Direct Answer", True, "Answer the question directly and succinctly."),
        ("supporting_evidence", "Supporting Evidence", True, "Summarize the most relevant supporting evidence."),
        ("caveats", "Caveats", False, "Note uncertainties, limits, or conflicting evidence."),
    ],
    "causal": [
        ("direct_answer", "Direct Answer", True, "State whether the effect is supported and in what direction."),
        ("effect_direction", "Effect Direction", True, "Describe increase, decrease, mixed, or null effects."),
        ("supporting_evidence", "Supporting Evidence", True, "Provide the strongest evidence for the claimed effect."),
        ("causal_limitations", "Causal Limitations", False, "Explain limits on causal interpretation."),
    ],
    "mechanism": [
        ("direct_answer", "Direct Answer", True, "State the most supported mechanism-level answer."),
        ("supporting_evidence", "Supporting Evidence", True, "Provide evidence for the proposed mechanism."),
        ("mechanism_path", "Mechanism Path", True, "Lay out the mechanistic pathway or steps."),
        ("caveats", "Caveats", False, "Describe open mechanistic uncertainties."),
    ],
    "comparison": [
        ("comparison_summary", "Comparison Summary", True, "Summarize the comparison at a glance."),
        ("evidence_by_option", "Evidence by Option", True, "Break down evidence for each compared option."),
        ("conditions", "Conditions", True, "State the conditions that materially affect the comparison."),
        ("conclusion", "Conclusion", True, "State the final comparative conclusion with limits."),
    ],
    "frontier": [
        ("recent_trends", "Recent Trends", True, "Summarize the most important recent themes and shifts."),
        ("representative_papers", "Representative Papers", True, "List representative recent papers or directions."),
        ("open_questions", "Open Questions", True, "Highlight unresolved questions and gaps."),
    ],
}

AXIS_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "catalyst": (re.compile(r"\bcatalyst\b", re.I),),
    "material": (re.compile(r"\bmaterial\b", re.I), re.compile(r"\belectrode\b", re.I), re.compile(r"\bcathode\b", re.I), re.compile(r"\banode\b", re.I)),
    "substrate": (re.compile(r"\bsubstrate\b", re.I),),
    "solvent": (re.compile(r"\bsolvent\b", re.I),),
    "ligand": (re.compile(r"\bligand\b", re.I),),
    "reagent": (re.compile(r"\breagent\b", re.I), re.compile(r"\badditive\b", re.I), re.compile(r"\bbase\b", re.I), re.compile(r"\bacid\b", re.I)),
    "temperature": (re.compile(r"\btemperature\b", re.I), re.compile(r"掳\s*C", re.I), re.compile(r"\b\d+(?:\.\d+)?\s*K\b", re.I)),
    "time": (re.compile(r"\btime\b", re.I), re.compile(r"\bduration\b", re.I), re.compile(r"\b\d+(?:\.\d+)?\s*(?:h|hr|hrs|min|minute|minutes)\b", re.I)),
    "ph": (re.compile(r"\bpH\b", re.I),),
    "electrolyte": (re.compile(r"\belectrolyte\b", re.I), re.compile(r"\b\d+(?:\.\d+)?\s*M\s+[A-Za-z0-9()/-]+", re.I)),
    "potential": (re.compile(r"\bpotential\b", re.I), re.compile(r"\bvoltage\b", re.I), re.compile(r"\boverpotential\b", re.I), re.compile(r"\b-?\d+(?:\.\d+)?\s*(?:V|mV)\b", re.I)),
    "pressure": (re.compile(r"\bpressure\b", re.I), re.compile(r"\b\d+(?:\.\d+)?\s*(?:bar|atm|kPa|MPa|Pa)\b", re.I)),
    "yield": (re.compile(r"\byield\b", re.I),),
    "selectivity": (re.compile(r"\bselectivity\b", re.I), re.compile(r"\bFaradaic efficiency\b", re.I), re.compile(r"\bFE\b", re.I)),
}

PREFERRED_ENTITY_PATTERNS: Dict[str, Sequence[re.Pattern[str]]] = {
    "reaction": (re.compile(r"\breaction\b", re.I), re.compile(r"\bCO2RR\b", re.I), re.compile(r"\bHER\b", re.I), re.compile(r"\bOER\b", re.I), re.compile(r"\bORR\b", re.I)),
    "catalyst": (re.compile(r"\bcatalyst\b", re.I),),
    "material": (re.compile(r"\bmaterial\b", re.I), re.compile(r"\belectrode\b", re.I)),
    "molecule": (re.compile(r"\bmolecule\b", re.I), re.compile(r"\bcompound\b", re.I)),
    "solvent": (re.compile(r"\bsolvent\b", re.I),),
    "ligand": (re.compile(r"\bligand\b", re.I),),
    "substrate": (re.compile(r"\bsubstrate\b", re.I),),
    "reagent": (re.compile(r"\breagent\b", re.I), re.compile(r"\badditive\b", re.I)),
    "metric": (re.compile(r"\byield\b", re.I), re.compile(r"\bselectivity\b", re.I), re.compile(r"\bFaradaic efficiency\b", re.I), re.compile(r"\boverpotential\b", re.I)),
}

METRIC_SIGNALS = re.compile(r"\b(yield|selectivity|faradaic efficiency|fe|current density|overpotential|activity|conversion)\b", re.I)
METRIC_AMBIGUITY_SIGNALS = re.compile(r"\b(best|better|performance|efficient|effective|outperform)\b", re.I)
REFERENTIAL_ENTITY_SIGNALS = re.compile(r"\b(this|that|it|they|these|those)\b", re.I)
COMMON_QUERY_STOP_TOKENS = {"What", "Which", "When", "Where", "Does", "Do", "How", "Why", "Can", "Should", "Would", "Recent", "Latest", "Advances"}
VALID_QUESTION_TYPES = set(ANSWER_SECTION_TEMPLATES.keys())
VALID_RECENCY_POLICIES = {"none", "last_3y", "last_5y", "explicit"}
VALID_TIME_INTENTS = {"none", "recent", "explicit", "current"}
VALID_AMBIGUITY_FLAG_TYPES = {"entity_ambiguous", "metric_ambiguous", "time_ambiguous", "task_ambiguous", "condition_ambiguous"}
VALID_AMBIGUITY_SEVERITIES = {"low", "medium", "high"}
QUESTION_TYPE_DEFAULT_PREFERRED_ENTITY_TYPES: Dict[str, Sequence[str]] = {
    "mechanism": ("reaction", "catalyst", "condition"),
    "comparison": ("catalyst", "reaction", "metric", "condition"),
}
LOWERCASE_MUST_INCLUDE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bbare carbon\b", re.I),
    re.compile(r"\balkaline media\b", re.I),
    re.compile(r"\balkaline electrolyte\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*M\s+[A-Za-z0-9()/-]+\b"),
)
EXPLICIT_ELECTROLYTE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*M\s+[A-Za-z0-9()/-]+\b", re.I)
ELECTROLYTE_SPECIES_PATTERN = re.compile(r"\b(?:KOH|NaOH|LiOH|CsOH|RbOH|H2SO4|HClO4|NaCl|KCl)\b", re.I)
COMPARISON_CUES_PATTERN = re.compile(r"\b(?:vs\.?|versus|compared with|compare|better than)\b", re.I)
METRIC_AMBIGUITY_DEFAULT_TERMS: Sequence[str] = ("overpotential", "Tafel slope", "exchange current density")


class RouterExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        question: str,
        normalized_question: str,
        context: Optional[str],
        debug_payload: Dict[str, Any],
    ) -> None:
        super().__init__(reason)
        self.stage = str(stage or "").strip() or "unknown"
        self.reason = str(reason or "").strip() or "router execution failed"
        self.question = str(question or "")
        self.normalized_question = str(normalized_question or "")
        self.context = context
        self.debug_payload = dict(debug_payload or {})

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": "router_execution_failed",
            "stage": self.stage,
            "reason": self.reason,
            "question": self.question,
            "normalized_question": self.normalized_question,
            "context_present": bool((self.context or "").strip()),
        }


class RouterNode:
    def __init__(self, llm: Any = None, current_year: Optional[int] = None) -> None:
        self.llm = llm
        self.current_year = int(current_year or datetime.now().year)
        self.last_run_debug: Dict[str, Any] = {}

    def run(self, question: str, context: Optional[str] = None) -> TaskSpec:
        normalized_question = self._normalize_question(question)
        auxiliary_signals = self._build_auxiliary_signals(
            question=question,
            normalized_question=normalized_question,
            context=context,
        )
        self.last_run_debug = {
            "input": {"question": question, "context": context},
            "normalized_question": normalized_question,
            "auxiliary_signals": auxiliary_signals,
        }

        if self.llm is None:
            self._raise_failure(
                stage="startup",
                question=question,
                normalized_question=normalized_question,
                context=context,
                reason="router LLM is unavailable",
            )

        try:
            semantic_payload = self._run_semantic_stage(
                question=question,
                context=context,
                auxiliary_signals=auxiliary_signals,
            )
        except Exception as exc:
            self._raise_failure(
                stage="semantic",
                question=question,
                normalized_question=normalized_question,
                context=context,
                reason=f"semantic stage failed: {exc}",
            )

        if semantic_payload is None:
            self._raise_failure(
                stage="semantic",
                question=question,
                normalized_question=normalized_question,
                context=context,
                reason="semantic stage returned unusable output",
            )

        try:
            task_spec = self._run_localization_stage(
                question=question,
                normalized_question=normalized_question,
                context=context,
                auxiliary_signals=auxiliary_signals,
                semantic_payload=semantic_payload,
            )
        except Exception as exc:
            self._raise_failure(
                stage="localization",
                question=question,
                normalized_question=normalized_question,
                context=context,
                reason=f"localization stage failed: {exc}",
            )

        if task_spec is None:
            self._raise_failure(
                stage="localization",
                question=question,
                normalized_question=normalized_question,
                context=context,
                reason="localization stage returned unusable output",
            )

        self.last_run_debug["output"] = task_spec.model_dump(exclude_none=True)
        return task_spec

    __call__ = run

    def _run_semantic_stage(
        self,
        *,
        question: str,
        context: Optional[str],
        auxiliary_signals: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        raw_output = self._invoke_semantic_llm(
            question=question,
            context=context,
            auxiliary_signals=auxiliary_signals,
        )
        self.last_run_debug["semantic_stage_raw"] = raw_output
        parsed = parse_json_object(raw_output)
        if parsed is None:
            return None
        repaired = self._repair_semantic_payload(parsed, question=question)
        if repaired is not None:
            self.last_run_debug["semantic_stage"] = repaired
        return repaired

    def _run_localization_stage(
        self,
        *,
        question: str,
        normalized_question: str,
        context: Optional[str],
        auxiliary_signals: Dict[str, Any],
        semantic_payload: Dict[str, Any],
    ) -> Optional[TaskSpec]:
        raw_output = self._invoke_localization_llm(
            question=question,
            context=context,
            auxiliary_signals=auxiliary_signals,
            semantic_payload=semantic_payload,
        )
        self.last_run_debug["localization_stage_raw"] = raw_output
        parsed = parse_json_object(raw_output)
        if parsed is None:
            return None

        baseline_payload = self._build_llm_baseline_payload(
            question=question,
            normalized_question=normalized_question,
            semantic_payload=semantic_payload,
            auxiliary_signals=auxiliary_signals,
        )
        repaired = self._repair_llm_payload(parsed, baseline_payload)
        self.last_run_debug["localization_stage"] = repaired
        if self._should_reject_llm_payload(semantic_payload=semantic_payload, repaired_payload=repaired):
            return None
        return TaskSpec.model_validate(repaired)

    def _raise_failure(
        self,
        *,
        stage: str,
        question: str,
        normalized_question: str,
        context: Optional[str],
        reason: str,
    ) -> None:
        failure_payload = {
            "error": "router_execution_failed",
            "stage": str(stage or "").strip() or "unknown",
            "reason": str(reason or "").strip() or "router execution failed",
        }
        self.last_run_debug["failure"] = failure_payload
        self.last_run_debug.pop("output", None)
        raise RouterExecutionError(
            stage=failure_payload["stage"],
            reason=failure_payload["reason"],
            question=question,
            normalized_question=normalized_question,
            context=context,
            debug_payload=self.last_run_debug,
        )

    def _build_auxiliary_signals(
        self,
        *,
        question: str,
        normalized_question: str,
        context: Optional[str],
    ) -> Dict[str, Any]:
        tentative_question_type, _ = self._detect_question_type(question)
        recency_policy, year_from, year_to, _, time_flags = self._detect_time_window(
            question,
            question_type=tentative_question_type,
            allow_frontier_default=False,
        )
        auxiliary_ambiguity_flags = self._build_auxiliary_ambiguity_flags(
            question=question,
            recency_policy=recency_policy,
            time_flags=time_flags,
        )
        return {
            "normalized_question": normalized_question,
            "context_present": bool((context or "").strip()),
            "explicit_time_window": {
                "recency_policy": recency_policy,
                "year_from": year_from,
                "year_to": year_to,
            },
            "tentative_question_type": tentative_question_type,
            "detected_condition_axes": self._extract_required_condition_axes(question),
            "preferred_entity_types": self._extract_preferred_entity_types(question, question_type=tentative_question_type),
            "must_include_terms": self._extract_must_include_terms(question, year_from, year_to, recency_policy),
            "metric_terms": self._extract_metric_terms(question),
            "comparative_cues_detected": any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["comparison"]),
            "causal_cues_detected": any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["causal"]),
            "mechanistic_cues_detected": any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["mechanism"]),
            "frontier_cues_detected": any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["frontier"]),
            "currentness_cues_detected": bool(re.search(r"\bcurrent|today|now\b", question, re.I)),
            "referential_entity_cues_detected": bool(REFERENTIAL_ENTITY_SIGNALS.search(question)),
            "auxiliary_ambiguity_flags": [flag.model_dump() for flag in auxiliary_ambiguity_flags],
        }

    def _build_llm_baseline_payload(
        self,
        *,
        question: str,
        normalized_question: str,
        semantic_payload: Dict[str, Any],
        auxiliary_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        question_type = semantic_payload["primary_question_type"]
        recency_policy, year_from, year_to, time_flags = self._resolve_llm_time_window(
            question=question,
            question_type=question_type,
            semantic_payload=semantic_payload,
            auxiliary_signals=auxiliary_signals,
        )
        required_condition_axes = list(auxiliary_signals.get("detected_condition_axes") or [])
        preferred_entity_types = self._extract_preferred_entity_types(question, question_type=question_type)
        preferred_entity_types = list(dict.fromkeys([*preferred_entity_types, *list(auxiliary_signals.get("preferred_entity_types") or [])]))
        ambiguity_flags = self._build_ambiguity_flags(question, question_type, recency_policy, time_flags)
        ambiguity_flags = self._extend_with_semantic_ambiguity_flags(
            ambiguity_flags=ambiguity_flags,
            question_type=question_type,
            semantic_payload=semantic_payload,
        )
        query_constraints = QueryConstraints(
            must_include_terms=list(auxiliary_signals.get("must_include_terms") or []),
            should_include_terms=self._extract_should_include_terms(question, question_type),
            exclude_terms=[],
            preferred_entity_types=preferred_entity_types,
            allow_broad_expansion=bool(semantic_payload.get("needs_disambiguation")) or question_type in {"frontier", "comparison"},
        )
        router_confidence = self._estimate_llm_confidence(
            semantic_payload=semantic_payload,
            required_condition_axes=required_condition_axes,
            ambiguity_count=len(ambiguity_flags),
        )
        return {
            "version": "1.0",
            "question": question,
            "normalized_question": normalized_question,
            "question_type": question_type,
            "recency_policy": recency_policy,
            "year_from": year_from,
            "year_to": year_to,
            "answer_sections": [section.model_dump() for section in self._build_answer_sections(question_type)],
            "required_condition_axes": required_condition_axes,
            "query_constraints": query_constraints.model_dump(),
            "ambiguity_flags": [flag.model_dump() for flag in ambiguity_flags],
            "router_confidence": router_confidence,
        }

    def _detect_question_type(self, question: str) -> Tuple[str, float]:
        if re.search(r"^\s*why\s+does\b", question, re.I) and any(
            pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["comparison"]
        ):
            return "causal", 0.9
        if re.search(r"^\s*why\b", question, re.I) or re.search(r"\bmechanism\b|\bpathway\b", question, re.I):
            return "mechanism", 0.88
        if re.search(r"^\s*how\b", question, re.I) and not any(
            pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["comparison"]
        ):
            return "mechanism", 0.82
        precedence = ["causal", "comparison", "frontier", "mechanism"]
        for question_type in precedence:
            if any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS[question_type]):
                return question_type, 0.85
        return "fact", 0.6

    def _detect_time_window(
        self,
        question: str,
        question_type: Optional[str],
        *,
        allow_frontier_default: bool,
    ) -> Tuple[str, Optional[int], Optional[int], float, List[AmbiguityFlag]]:
        text = question
        current_year = self.current_year
        flags: List[AmbiguityFlag] = []
        explicit_range = re.search(r"\b(?:from|between)\s+(20\d{2}|19\d{2})\s+(?:to|and)\s+(20\d{2}|19\d{2})\b", text, re.I)
        if explicit_range:
            year_from = int(explicit_range.group(1))
            year_to = int(explicit_range.group(2))
            if year_from > year_to:
                year_from, year_to = year_to, year_from
            return "explicit", year_from, year_to, 0.9, flags

        dash_range = re.search(r"\b(19\d{2}|20\d{2})\s*[-鈥揮]\s*(19\d{2}|20\d{2})\b", text)
        if dash_range:
            year_from = int(dash_range.group(1))
            year_to = int(dash_range.group(2))
            if year_from > year_to:
                year_from, year_to = year_to, year_from
            return "explicit", year_from, year_to, 0.9, flags

        since_match = re.search(r"\bsince\s+(19\d{2}|20\d{2})\b", text, re.I)
        if since_match:
            year_from = int(since_match.group(1))
            return "explicit", year_from, current_year, 0.85, flags

        specific_year = re.search(r"\b(?:in|during)\s+(19\d{2}|20\d{2})\b", text, re.I)
        if specific_year:
            year = int(specific_year.group(1))
            return "explicit", year, year, 0.8, flags

        if re.search(r"\b(?:last|past)\s+5\s+years\b", text, re.I):
            return "last_5y", current_year - 4, current_year, 0.8, flags

        if re.search(r"\b(?:last|past)\s+3\s+years\b", text, re.I):
            return "last_3y", current_year - 2, current_year, 0.8, flags

        if allow_frontier_default and question_type == "frontier":
            flags.append(
                AmbiguityFlag(
                    flag_type="time_ambiguous",
                    target="time_window",
                    note=f"Recentness cue detected without explicit years; defaulting to {current_year - 2}-{current_year}.",
                    severity="low",
                )
            )
            return "last_3y", current_year - 2, current_year, 0.75, flags

        return "none", None, None, 0.55, flags

    def _resolve_llm_time_window(
        self,
        *,
        question: str,
        question_type: str,
        semantic_payload: Dict[str, Any],
        auxiliary_signals: Dict[str, Any],
    ) -> Tuple[str, Optional[int], Optional[int], List[AmbiguityFlag]]:
        explicit_time_window = dict(auxiliary_signals.get("explicit_time_window") or {})
        recency_policy = str(explicit_time_window.get("recency_policy") or "none")
        year_from = self._safe_int(explicit_time_window.get("year_from"), fallback=None)
        year_to = self._safe_int(explicit_time_window.get("year_to"), fallback=None)
        time_flags = [
            AmbiguityFlag.model_validate(flag)
            for flag in list(auxiliary_signals.get("auxiliary_ambiguity_flags") or [])
            if isinstance(flag, dict) and flag.get("flag_type") == "time_ambiguous"
        ]

        if recency_policy in VALID_RECENCY_POLICIES and recency_policy != "none":
            return recency_policy, year_from, year_to, time_flags

        time_intent = semantic_payload.get("explicit_time_intent")
        if time_intent in {"recent", "current"} or question_type == "frontier":
            time_flags.append(
                AmbiguityFlag(
                    flag_type="time_ambiguous",
                    target="time_window",
                    note=f"Recentness was requested without explicit years; defaulting to {self.current_year - 2}-{self.current_year}.",
                    severity="low",
                )
            )
            return "last_3y", self.current_year - 2, self.current_year, time_flags

        return "none", None, None, time_flags

    def _extract_required_condition_axes(self, question: str) -> List[str]:
        axes: List[str] = []
        for axis, patterns in AXIS_PATTERNS.items():
            if any(pattern.search(question) for pattern in patterns):
                axes.append(axis)
        return axes

    def _build_answer_sections(self, question_type: str) -> List[AnswerSection]:
        return [
            AnswerSection(section_id=section_id, title=title, required=required, instruction=instruction)
            for section_id, title, required, instruction in ANSWER_SECTION_TEMPLATES[question_type]
        ]

    def _extract_preferred_entity_types(self, question: str, *, question_type: Optional[str] = None) -> List[str]:
        preferred: List[str] = []
        for entity_type, patterns in PREFERRED_ENTITY_PATTERNS.items():
            if any(pattern.search(question) for pattern in patterns):
                preferred.append(entity_type)
        for entity_type in QUESTION_TYPE_DEFAULT_PREFERRED_ENTITY_TYPES.get(str(question_type or "").strip(), ()):
            if entity_type not in preferred:
                preferred.append(entity_type)
        return preferred

    def _extract_must_include_terms(
        self,
        question: str,
        year_from: Optional[int],
        year_to: Optional[int],
        recency_policy: str,
    ) -> List[str]:
        terms: List[str] = []
        if recency_policy == "explicit":
            if year_from is not None:
                self._append_unique_term(terms, str(year_from))
            if year_to is not None and year_to != year_from:
                self._append_unique_term(terms, str(year_to))
        chemical_tokens = re.findall(r"\b(?:[A-Z][A-Za-z0-9]{1,}(?:[-/][A-Za-z0-9]+)*)\b", question)
        for token in chemical_tokens:
            if token in COMMON_QUERY_STOP_TOKENS:
                continue
            self._append_unique_term(terms, token)
        for pattern in LOWERCASE_MUST_INCLUDE_PATTERNS:
            for match in pattern.finditer(question):
                self._append_unique_term(terms, match.group(0))
        return terms

    def _extract_should_include_terms(self, question: str, question_type: str) -> List[str]:
        terms: List[str] = []
        if question_type == "frontier":
            self._append_unique_terms(terms, ["recent review", "state of the art"])
        if question_type == "mechanism":
            self._append_unique_terms(terms, ["mechanism", "pathway"])
        if question_type == "causal":
            self._append_unique_terms(terms, ["effect", "trend"])
        for metric_term in self._extract_metric_terms(question):
            self._append_unique_term(terms, metric_term)
        if METRIC_AMBIGUITY_SIGNALS.search(question) and not METRIC_SIGNALS.search(question):
            self._append_unique_terms(terms, METRIC_AMBIGUITY_DEFAULT_TERMS)
        if self._has_alkaline_condition_ambiguity(question):
            self._append_unique_terms(terms, ["alkaline", "KOH"])
        return terms

    @staticmethod
    def _append_unique_term(target: List[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)

    def _append_unique_terms(self, target: List[str], values: Sequence[Any]) -> None:
        for value in list(values or []):
            self._append_unique_term(target, value)

    @staticmethod
    def _has_alkaline_condition_ambiguity(question: str) -> bool:
        return bool(re.search(r"\balkaline\b", question, re.I)) and not bool(EXPLICIT_ELECTROLYTE_PATTERN.search(question))

    def _extract_metric_terms(self, question: str) -> List[str]:
        metric_terms: List[str] = []
        for match in METRIC_SIGNALS.finditer(question):
            metric_term = match.group(0)
            if metric_term not in metric_terms:
                metric_terms.append(metric_term)
        return metric_terms

    def _build_auxiliary_ambiguity_flags(
        self,
        *,
        question: str,
        recency_policy: str,
        time_flags: List[AmbiguityFlag],
    ) -> List[AmbiguityFlag]:
        flags = list(time_flags)
        if METRIC_AMBIGUITY_SIGNALS.search(question) and not METRIC_SIGNALS.search(question):
            flags.append(
                AmbiguityFlag(
                    flag_type="metric_ambiguous",
                    target="metric",
                    note="Comparative or performance language appears without a specific evaluation metric.",
                    severity="medium",
                )
            )
        if REFERENTIAL_ENTITY_SIGNALS.search(question) and not re.search(r"\b[A-Z][A-Za-z0-9/-]{1,}\b", question):
            flags.append(
                AmbiguityFlag(
                    flag_type="entity_ambiguous",
                    target="entity",
                    note="Question includes referential mentions without clear entity names.",
                    severity="low",
                )
            )
        if recency_policy == "none" and re.search(r"\bcurrent|today|now\b", question, re.I):
            flags.append(
                AmbiguityFlag(
                    flag_type="time_ambiguous",
                    target="time_window",
                    note="Currentness cue detected without a retrievable time range.",
                    severity="low",
                )
            )
        return flags

    def _build_ambiguity_flags(
        self,
        question: str,
        question_type: str,
        recency_policy: str,
        time_flags: List[AmbiguityFlag],
    ) -> List[AmbiguityFlag]:
        flags = self._build_auxiliary_ambiguity_flags(
            question=question,
            recency_policy=recency_policy,
            time_flags=time_flags,
        )
        if question_type == "comparison" and not re.search(r"\b(?:vs\.?|versus|better than|compared with)\b", question, re.I):
            flags.append(
                AmbiguityFlag(
                    flag_type="task_ambiguous",
                    target="comparison_scope",
                    note="Comparison intent is present, but the compared options may be underspecified.",
                    severity="low",
                )
            )
        return flags

    def _extend_with_semantic_ambiguity_flags(
        self,
        *,
        ambiguity_flags: Sequence[AmbiguityFlag],
        question_type: str,
        semantic_payload: Dict[str, Any],
    ) -> List[AmbiguityFlag]:
        flags = list(ambiguity_flags)
        notes = list(semantic_payload.get("notes_on_ambiguity") or [])
        severity = "medium" if float(semantic_payload.get("semantic_confidence", 0.0)) < 0.55 else "low"
        for note in notes:
            flags.append(
                AmbiguityFlag(
                    flag_type="task_ambiguous",
                    target="semantic_router",
                    note=str(note),
                    severity=severity,
                )
            )
        if question_type == "comparison" and not semantic_payload.get("comparison_targets_present", False):
            flags.append(
                AmbiguityFlag(
                    flag_type="task_ambiguous",
                    target="comparison_scope",
                    note="Comparison intent was inferred semantically, but explicit comparison targets remain underspecified.",
                    severity="medium",
                )
            )
        return self._dedupe_ambiguity_flags(flags)

    def _estimate_rule_confidence(
        self,
        *,
        question_type_score: float,
        time_score: float,
        required_condition_axes: Sequence[str],
        ambiguity_count: int,
    ) -> float:
        confidence = 0.35
        confidence += 0.25 * question_type_score
        confidence += 0.15 * time_score
        confidence += min(len(required_condition_axes), 3) * 0.05
        confidence -= min(ambiguity_count, 3) * 0.08
        return round(max(0.05, min(confidence, 0.99)), 2)

    def _estimate_llm_confidence(
        self,
        *,
        semantic_payload: Dict[str, Any],
        required_condition_axes: Sequence[str],
        ambiguity_count: int,
    ) -> float:
        confidence = 0.25 + 0.55 * float(semantic_payload.get("semantic_confidence", 0.0))
        confidence += min(len(required_condition_axes), 3) * 0.04
        if semantic_payload.get("comparison_targets_present"):
            confidence += 0.04
        if semantic_payload.get("explicit_metric_requested"):
            confidence += 0.03
        if semantic_payload.get("needs_disambiguation"):
            confidence -= 0.08
        confidence -= min(ambiguity_count, 3) * 0.07
        return round(max(0.05, min(confidence, 0.99)), 2)

    def _invoke_semantic_llm(
        self,
        *,
        question: str,
        context: Optional[str],
        auxiliary_signals: Dict[str, Any],
    ) -> Any:
        messages = [
            {"role": "system", "content": ROUTER_SEMANTIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_router_semantic_user_prompt(
                    question=question,
                    current_year=self.current_year,
                    optional_signals=auxiliary_signals,
                    context=context,
                ),
            },
        ]
        return invoke_llm(self.llm, messages)

    def _invoke_localization_llm(
        self,
        *,
        question: str,
        context: Optional[str],
        auxiliary_signals: Dict[str, Any],
        semantic_payload: Dict[str, Any],
    ) -> Any:
        messages = [
            {"role": "system", "content": ROUTER_LOCALIZATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_router_localization_user_prompt(
                    question=question,
                    current_year=self.current_year,
                    semantic_parse=semantic_payload,
                    optional_signals=auxiliary_signals,
                    context=context,
                ),
            },
        ]
        return invoke_llm(self.llm, messages)

    def _repair_semantic_payload(self, parsed: Dict[str, Any], *, question: str = "") -> Optional[Dict[str, Any]]:
        secondary_candidates = self._repair_question_type_candidates(parsed.get("secondary_candidates"))
        primary_question_type = parsed.get("primary_question_type")
        if primary_question_type not in VALID_QUESTION_TYPES:
            primary_question_type = self._infer_semantic_primary_type(parsed, secondary_candidates)
        if primary_question_type not in VALID_QUESTION_TYPES:
            return None
        secondary_candidates = [candidate for candidate in secondary_candidates if candidate != primary_question_type]
        time_intent = parsed.get("explicit_time_intent")
        if time_intent not in VALID_TIME_INTENTS:
            time_intent = "none"
        repaired = {
            "primary_question_type": primary_question_type,
            "secondary_candidates": secondary_candidates,
            "semantic_confidence": self._repair_confidence(parsed.get("semantic_confidence"), 0.35),
            "needs_disambiguation": self._safe_bool(parsed.get("needs_disambiguation")),
            "comparison_intent": self._safe_bool(parsed.get("comparison_intent")),
            "comparison_targets_present": self._safe_bool(parsed.get("comparison_targets_present")),
            "explicit_metric_requested": self._safe_bool(parsed.get("explicit_metric_requested")),
            "explicit_time_intent": time_intent,
            "mechanistic_intent": self._safe_bool(parsed.get("mechanistic_intent")),
            "causal_intent": self._safe_bool(parsed.get("causal_intent")),
            "frontier_intent": self._safe_bool(parsed.get("frontier_intent")),
            "notes_on_ambiguity": self._repair_text_list(parsed.get("notes_on_ambiguity"), ()),
        }
        repaired["primary_question_type"] = self._coerce_question_type_from_question(
            question=question,
            proposed_type=repaired["primary_question_type"],
        )
        if repaired["semantic_confidence"] < 0.2 and not repaired["notes_on_ambiguity"]:
            repaired["notes_on_ambiguity"] = ["Semantic interpretation confidence is extremely low."]
        return repaired

    def _infer_semantic_primary_type(
        self,
        parsed: Dict[str, Any],
        secondary_candidates: Sequence[str],
    ) -> Optional[str]:
        for key, question_type in (
            ("comparison_intent", "comparison"),
            ("frontier_intent", "frontier"),
            ("mechanistic_intent", "mechanism"),
            ("causal_intent", "causal"),
        ):
            if self._safe_bool(parsed.get(key)):
                return question_type
        return secondary_candidates[0] if secondary_candidates else None

    def _repair_question_type_candidates(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        repaired: List[str] = []
        for item in value:
            candidate = str(item or "").strip()
            if candidate in VALID_QUESTION_TYPES and candidate not in repaired:
                repaired.append(candidate)
        return repaired

    def _repair_llm_payload(self, parsed: Dict[str, Any], baseline_payload: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(baseline_payload)
        repaired["version"] = str(parsed.get("version") or baseline_payload["version"])
        repaired["question"] = str(parsed.get("question") or baseline_payload["question"])
        repaired["normalized_question"] = str(parsed.get("normalized_question") or baseline_payload["normalized_question"])

        question_type = parsed.get("question_type")
        if question_type in VALID_QUESTION_TYPES:
            repaired["question_type"] = self._coerce_question_type_from_question(
                question=repaired["question"],
                proposed_type=question_type,
            )

        recency_policy = parsed.get("recency_policy")
        if recency_policy in VALID_RECENCY_POLICIES:
            repaired["recency_policy"] = recency_policy

        repaired["year_from"] = self._safe_int(parsed.get("year_from"), fallback=baseline_payload["year_from"])
        repaired["year_to"] = self._safe_int(parsed.get("year_to"), fallback=baseline_payload["year_to"])
        if repaired["year_from"] is not None and repaired["year_to"] is not None and repaired["year_from"] > repaired["year_to"]:
            repaired["year_from"], repaired["year_to"] = repaired["year_to"], repaired["year_from"]

        repaired["answer_sections"] = self._repair_answer_sections(parsed.get("answer_sections"), question_type=repaired["question_type"])
        repaired["required_condition_axes"] = self._repair_required_axes(parsed.get("required_condition_axes"), baseline_payload)
        repaired["query_constraints"] = self._repair_query_constraints(parsed.get("query_constraints"), baseline_payload)
        repaired["ambiguity_flags"] = self._repair_ambiguity_flags(parsed.get("ambiguity_flags"), baseline_payload)
        repaired["router_confidence"] = self._repair_confidence(parsed.get("router_confidence"), baseline_payload["router_confidence"])
        return repaired

    def _coerce_question_type_from_question(self, *, question: str, proposed_type: str) -> str:
        normalized = str(proposed_type or "").strip()
        if (
            normalized in {"comparison", "mechanism"}
            and re.search(r"^\s*why\s+does\b", question, re.I)
            and any(pattern.search(question) for pattern in QUESTION_TYPE_PATTERNS["comparison"])
            and not re.search(r"\bmechanism\b|\bpathway\b", question, re.I)
        ):
            return "causal"
        return normalized

    def _repair_answer_sections(self, raw_sections: Any, question_type: str) -> List[Dict[str, Any]]:
        if not isinstance(raw_sections, list):
            return [section.model_dump() for section in self._build_answer_sections(question_type)]
        repaired: List[Dict[str, Any]] = []
        for section in raw_sections:
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("section_id") or "").strip()
            if not section_id:
                continue
            title = str(section.get("title") or section_id.replace("_", " ").title())
            required = bool(section.get("required", True))
            instruction = str(section.get("instruction") or f"Provide content for {title}.")
            repaired.append(AnswerSection(section_id=section_id, title=title, required=required, instruction=instruction).model_dump())
        if repaired:
            return repaired
        return [section.model_dump() for section in self._build_answer_sections(question_type)]

    def _repair_required_axes(self, raw_axes: Any, baseline_payload: Dict[str, Any]) -> List[str]:
        if not isinstance(raw_axes, list):
            return baseline_payload["required_condition_axes"]
        whitelist = set(AXIS_PATTERNS.keys())
        return [axis for axis in raw_axes if axis in whitelist] or baseline_payload["required_condition_axes"]

    def _repair_query_constraints(self, raw_constraints: Any, baseline_payload: Dict[str, Any]) -> Dict[str, Any]:
        baseline = dict(baseline_payload["query_constraints"])
        if not isinstance(raw_constraints, dict):
            return baseline
        repaired = dict(baseline)
        valid_entity_types = {
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
        }
        for key in ("must_include_terms", "should_include_terms", "exclude_terms", "preferred_entity_types"):
            value = raw_constraints.get(key)
            if isinstance(value, list):
                items = [str(item) for item in value if str(item).strip()]
                if key == "preferred_entity_types":
                    repaired[key] = [item for item in items if item in valid_entity_types]
                else:
                    repaired[key] = items
        if isinstance(raw_constraints.get("allow_broad_expansion"), bool):
            repaired["allow_broad_expansion"] = raw_constraints["allow_broad_expansion"]
        return QueryConstraints.model_validate(repaired).model_dump()

    def _repair_ambiguity_flags(self, raw_flags: Any, baseline_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        repaired = list(baseline_payload["ambiguity_flags"])
        if isinstance(raw_flags, list):
            for raw_flag in raw_flags:
                if not isinstance(raw_flag, dict):
                    continue
                flag_type = raw_flag.get("flag_type")
                severity = raw_flag.get("severity")
                if flag_type not in VALID_AMBIGUITY_FLAG_TYPES:
                    continue
                if severity not in VALID_AMBIGUITY_SEVERITIES:
                    severity = "low"
                repaired.append(
                    AmbiguityFlag(
                        flag_type=flag_type,
                        target=str(raw_flag.get("target") or flag_type),
                        note=str(raw_flag.get("note") or "Ambiguity retained from router output."),
                        severity=severity,
                    ).model_dump()
                )
        return self._dedupe_ambiguity_flag_payloads(repaired)

    def _should_reject_llm_payload(
        self,
        *,
        semantic_payload: Dict[str, Any],
        repaired_payload: Dict[str, Any],
    ) -> bool:
        ambiguity_flags = list(repaired_payload.get("ambiguity_flags") or [])
        has_high_severity = any(isinstance(flag, dict) and flag.get("severity") == "high" for flag in ambiguity_flags)
        semantic_confidence = float(semantic_payload.get("semantic_confidence", 0.0))
        router_confidence = float(repaired_payload.get("router_confidence") or 0.0)
        return semantic_confidence < 0.15 and router_confidence < 0.15 and (has_high_severity or len(ambiguity_flags) >= 2)

    def _dedupe_ambiguity_flags(self, ambiguity_flags: Sequence[AmbiguityFlag]) -> List[AmbiguityFlag]:
        deduped: List[AmbiguityFlag] = []
        seen = set()
        for flag in ambiguity_flags:
            key = (flag.flag_type, flag.target, flag.note, flag.severity)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(flag)
        return deduped

    def _dedupe_ambiguity_flag_payloads(self, payloads: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for payload in payloads:
            key = (payload.get("flag_type"), payload.get("target"), payload.get("note"), payload.get("severity"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(payload)
        return deduped

    def _normalize_question(self, question: str) -> str:
        return re.sub(r"\s+", " ", question or "").strip()

    def _safe_int(self, value: Any, fallback: Optional[int]) -> Optional[int]:
        try:
            return int(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def _repair_confidence(self, value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return round(max(0.0, min(numeric, 1.0)), 2)

    def _safe_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"true", "1", "yes"}:
                return True
            if cleaned in {"false", "0", "no", ""}:
                return False
        return False

    def _repair_text_list(self, value: Any, fallback: Sequence[str]) -> List[str]:
        if not isinstance(value, list):
            return list(fallback)
        repaired: List[str] = []
        for item in value:
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in repaired:
                repaired.append(cleaned)
        return repaired or list(fallback)
