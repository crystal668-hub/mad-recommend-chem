from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prompts.qa_prompts import ROUTER_SYSTEM_PROMPT, build_router_user_prompt
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
    "temperature": (re.compile(r"\btemperature\b", re.I), re.compile(r"°\s*C", re.I), re.compile(r"\b\d+(?:\.\d+)?\s*K\b", re.I)),
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

METRIC_SIGNALS = re.compile(
    r"\b(yield|selectivity|faradaic efficiency|fe|current density|overpotential|activity|conversion)\b",
    re.I,
)
METRIC_AMBIGUITY_SIGNALS = re.compile(r"\b(best|better|performance|efficient|effective|outperform)\b", re.I)
REFERENTIAL_ENTITY_SIGNALS = re.compile(r"\b(this|that|it|they|these|those)\b", re.I)
COMMON_QUERY_STOP_TOKENS = {
    "What",
    "Which",
    "When",
    "Where",
    "Does",
    "Do",
    "How",
    "Why",
    "Can",
    "Should",
    "Would",
    "Recent",
    "Latest",
    "Advances",
}


class RouterNode:
    def __init__(self, llm: Any = None, current_year: Optional[int] = None) -> None:
        self.llm = llm
        self.current_year = int(current_year or datetime.now().year)

    def run(self, question: str, context: Optional[str] = None) -> TaskSpec:
        normalized_question = self._normalize_question(question)
        rule_payload = self._build_rule_payload(question=question, normalized_question=normalized_question, context=context)

        if self.llm is None:
            return TaskSpec.model_validate(rule_payload)

        try:
            raw_output = self._invoke_llm(question=question, context=context, rule_payload=rule_payload)
            parsed = parse_json_object(raw_output)
            if parsed is not None:
                repaired = self._repair_llm_payload(parsed, rule_payload)
                return TaskSpec.model_validate(repaired)
        except Exception:
            pass

        repaired_fallback = self._append_fallback_flag(rule_payload)
        return TaskSpec.model_validate(repaired_fallback)

    __call__ = run

    def _build_rule_payload(self, question: str, normalized_question: str, context: Optional[str]) -> Dict[str, Any]:
        question_type, question_type_score = self._detect_question_type(question)
        recency_policy, year_from, year_to, time_score, time_flags = self._detect_time_window(question, question_type)
        required_condition_axes = self._extract_required_condition_axes(question)
        ambiguity_flags = self._build_ambiguity_flags(question, question_type, recency_policy, time_flags)
        preferred_entity_types = self._extract_preferred_entity_types(question)
        query_constraints = QueryConstraints(
            must_include_terms=self._extract_must_include_terms(question, year_from, year_to, recency_policy),
            should_include_terms=self._extract_should_include_terms(question, question_type),
            exclude_terms=[],
            preferred_entity_types=preferred_entity_types,
            allow_broad_expansion=len(ambiguity_flags) > 0 or question_type in {"frontier", "comparison"},
        )
        router_confidence = self._estimate_confidence(
            question_type_score=question_type_score,
            time_score=time_score,
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

    def _append_fallback_flag(self, rule_payload: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(rule_payload)
        ambiguity_flags = list(rule_payload.get("ambiguity_flags") or [])
        ambiguity_flags.append(
            AmbiguityFlag(
                flag_type="task_ambiguous",
                target="router",
                note="Router kept the deterministic rule payload because the LLM output could not be validated.",
                severity="low",
            ).model_dump()
        )
        repaired["ambiguity_flags"] = ambiguity_flags
        return repaired

    def _detect_question_type(self, question: str) -> Tuple[str, float]:
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

    def _detect_time_window(self, question: str, question_type: str) -> Tuple[str, Optional[int], Optional[int], float, List[AmbiguityFlag]]:
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

        dash_range = re.search(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b", text)
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

        if question_type == "frontier":
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

    def _extract_preferred_entity_types(self, question: str) -> List[str]:
        preferred: List[str] = []
        for entity_type, patterns in PREFERRED_ENTITY_PATTERNS.items():
            if any(pattern.search(question) for pattern in patterns):
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
                terms.append(str(year_from))
            if year_to is not None and year_to != year_from:
                terms.append(str(year_to))
        chemical_tokens = re.findall(r"\b(?:[A-Z][A-Za-z0-9]{1,}(?:[-/][A-Za-z0-9]+)*)\b", question)
        for token in chemical_tokens:
            if token in COMMON_QUERY_STOP_TOKENS:
                continue
            if token not in terms:
                terms.append(token)
        return terms

    def _extract_should_include_terms(self, question: str, question_type: str) -> List[str]:
        terms: List[str] = []
        if question_type == "frontier":
            terms.extend(["recent review", "state of the art"])
        if question_type == "mechanism":
            terms.extend(["mechanism", "pathway"])
        if question_type == "causal":
            terms.extend(["effect", "trend"])
        if METRIC_SIGNALS.search(question):
            metric_terms = [match.group(0) for match in METRIC_SIGNALS.finditer(question)]
            for metric_term in metric_terms:
                if metric_term not in terms:
                    terms.append(metric_term)
        return terms

    def _build_ambiguity_flags(
        self,
        question: str,
        question_type: str,
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
        if question_type == "comparison" and not re.search(r"\b(?:vs\.?|versus|better than|compared with)\b", question, re.I):
            flags.append(
                AmbiguityFlag(
                    flag_type="task_ambiguous",
                    target="comparison_scope",
                    note="Comparison intent detected, but the compared options may be underspecified.",
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

    def _estimate_confidence(
        self,
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

    def _invoke_llm(self, question: str, context: Optional[str], rule_payload: Dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_router_user_prompt(
                    question=question,
                    current_year=self.current_year,
                    rule_hints=rule_payload,
                    context=context,
                ),
            },
        ]
        return invoke_llm(self.llm, messages)

    def _repair_llm_payload(self, parsed: Dict[str, Any], rule_payload: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(rule_payload)
        repaired["version"] = str(parsed.get("version") or rule_payload["version"])
        repaired["question"] = str(parsed.get("question") or rule_payload["question"])
        repaired["normalized_question"] = str(parsed.get("normalized_question") or rule_payload["normalized_question"])

        question_type = parsed.get("question_type")
        if question_type in ANSWER_SECTION_TEMPLATES:
            repaired["question_type"] = question_type

        recency_policy = parsed.get("recency_policy")
        if recency_policy in {"none", "last_3y", "last_5y", "explicit"}:
            repaired["recency_policy"] = recency_policy

        repaired["year_from"] = self._safe_int(parsed.get("year_from"), fallback=rule_payload["year_from"])
        repaired["year_to"] = self._safe_int(parsed.get("year_to"), fallback=rule_payload["year_to"])

        repaired["answer_sections"] = self._repair_answer_sections(
            parsed.get("answer_sections"),
            question_type=repaired["question_type"],
        )
        repaired["required_condition_axes"] = self._repair_required_axes(parsed.get("required_condition_axes"), rule_payload)
        repaired["query_constraints"] = self._repair_query_constraints(parsed.get("query_constraints"), rule_payload)
        repaired["ambiguity_flags"] = self._repair_ambiguity_flags(parsed.get("ambiguity_flags"), rule_payload)
        repaired["router_confidence"] = self._repair_confidence(parsed.get("router_confidence"), rule_payload["router_confidence"])
        return repaired

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
            repaired.append(
                AnswerSection(section_id=section_id, title=title, required=required, instruction=instruction).model_dump()
            )
        if repaired:
            return repaired
        return [section.model_dump() for section in self._build_answer_sections(question_type)]

    def _repair_required_axes(self, raw_axes: Any, rule_payload: Dict[str, Any]) -> List[str]:
        if not isinstance(raw_axes, list):
            return rule_payload["required_condition_axes"]
        whitelist = set(AXIS_PATTERNS.keys())
        return [axis for axis in raw_axes if axis in whitelist] or rule_payload["required_condition_axes"]

    def _repair_query_constraints(self, raw_constraints: Any, rule_payload: Dict[str, Any]) -> Dict[str, Any]:
        baseline = dict(rule_payload["query_constraints"])
        if not isinstance(raw_constraints, dict):
            return baseline
        repaired = dict(baseline)
        for key in ("must_include_terms", "should_include_terms", "exclude_terms", "preferred_entity_types"):
            value = raw_constraints.get(key)
            if isinstance(value, list):
                repaired[key] = [str(item) for item in value if str(item).strip()]
        if isinstance(raw_constraints.get("allow_broad_expansion"), bool):
            repaired["allow_broad_expansion"] = raw_constraints["allow_broad_expansion"]
        return QueryConstraints.model_validate(repaired).model_dump()

    def _repair_ambiguity_flags(self, raw_flags: Any, rule_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        repaired = list(rule_payload["ambiguity_flags"])
        if not isinstance(raw_flags, list):
            return repaired
        for raw_flag in raw_flags:
            if not isinstance(raw_flag, dict):
                continue
            flag_type = raw_flag.get("flag_type")
            severity = raw_flag.get("severity")
            if flag_type not in {"entity_ambiguous", "metric_ambiguous", "time_ambiguous", "task_ambiguous", "condition_ambiguous"}:
                continue
            if severity not in {"low", "medium", "high"}:
                severity = "low"
            repaired.append(
                AmbiguityFlag(
                    flag_type=flag_type,
                    target=str(raw_flag.get("target") or flag_type),
                    note=str(raw_flag.get("note") or "Ambiguity retained from router output."),
                    severity=severity,
                ).model_dump()
            )
        return repaired

    def _repair_confidence(self, value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return round(max(0.0, min(numeric, 1.0)), 2)

    def _normalize_question(self, question: str) -> str:
        return re.sub(r"\s+", " ", question or "").strip()

    def _safe_int(self, value: Any, fallback: Optional[int]) -> Optional[int]:
        try:
            return int(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback
