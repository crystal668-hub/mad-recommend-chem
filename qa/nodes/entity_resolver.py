from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import yaml

from prompts.qa_prompts import ENTITY_RESOLVER_SYSTEM_PROMPT, build_entity_resolver_user_prompt
from qa.llm_utils import invoke_llm, parse_json_object
from qa.state import (
    AmbiguityFlag,
    ConditionMention,
    EntityPack,
    EntityRecord,
    SourceSpan,
    TaskSpec,
    UnresolvedMention,
)


ENTITY_AXIS_MAP = {
    "catalyst": "catalyst",
    "material": "material",
    "substrate": "substrate",
    "solvent": "solvent",
    "ligand": "ligand",
    "reagent": "reagent",
}

ENTITY_TYPE_PRIORITY = {
    "reaction": 0,
    "catalyst": 1,
    "material": 1,
    "substrate": 2,
    "solvent": 2,
    "ligand": 2,
    "reagent": 2,
    "molecule": 3,
    "metric": 4,
    "condition": 5,
}

FORMULA_TOKEN_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9]{1,}(?:[-/][A-Za-z0-9]+)*)\b")
ROLE_AFTER_PATTERN = re.compile(
    r"\b(?P<lemma>[A-Za-z0-9][A-Za-z0-9/+\-.]{1,})\s+(?P<role>ligand|solvent|reagent|substrate|catalyst|material)\b",
    re.I,
)
ROLE_BEFORE_PATTERN = re.compile(
    r"\b(?P<role>ligand|solvent|reagent|substrate|catalyst|material)\s+(?P<lemma>[A-Za-z0-9][A-Za-z0-9/+\-.]{1,})\b",
    re.I,
)
METRIC_PATTERNS = [
    re.compile(r"\bFaradaic efficiency\b", re.I),
    re.compile(r"\bFE\b"),
    re.compile(r"\byield\b", re.I),
    re.compile(r"\bselectivity\b", re.I),
    re.compile(r"\boverpotential\b", re.I),
    re.compile(r"\bcurrent density\b", re.I),
]
COMMON_STOP_TOKENS = {
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
    "Progress",
}
SMALL_MOLECULE_TYPES = {"molecule", "reagent", "solvent", "ligand"}
REACTION_ABBREVIATIONS = {"CO2RR", "HER", "OER", "ORR", "EOR", "UOR", "HOR", "HZOR", "O5H"}


@dataclass
class MentionCandidate:
    mention: str
    start: int
    end: int
    alias_hits: List[Dict[str, Any]] = field(default_factory=list)
    entity_type_hint: Optional[str] = None


class PubChemClient:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def lookup_compound(self, query: str) -> Optional[Dict[str, Any]]:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/IUPACName,CanonicalSMILES,InChIKey,CID,MolecularFormula/JSON"
        try:
            response = requests.get(url, params={"name": query}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if not properties:
                return None
            item = properties[0]
            return {
                "canonical_name": item.get("IUPACName") or query,
                "formula": item.get("MolecularFormula"),
                "smiles": item.get("CanonicalSMILES"),
                "inchikey": item.get("InChIKey"),
                "pubchem_cid": item.get("CID"),
                "aliases": [query],
            }
        except Exception:
            return None


class EntityResolverNode:
    def __init__(
        self,
        llm: Any = None,
        pubchem_client: Optional[Any] = None,
        enable_rdkit: Optional[bool] = None,
        entity_alias_path: Optional[str] = None,
        reaction_alias_path: Optional[str] = None,
    ) -> None:
        resource_root = Path(__file__).resolve().parents[1] / "resources"
        self.entity_aliases = self._load_yaml(Path(entity_alias_path) if entity_alias_path else resource_root / "entity_aliases.yaml")
        self.reaction_aliases = self._load_yaml(Path(reaction_alias_path) if reaction_alias_path else resource_root / "reaction_aliases.yaml")
        self.alias_index = self._build_alias_index()
        self.llm = llm
        self.pubchem_client = pubchem_client
        self.enable_rdkit = True if enable_rdkit is None else bool(enable_rdkit)

    def run(self, question: str, task_spec: TaskSpec) -> EntityPack:
        mention_candidates = self._extract_mentions(question)
        typed_mentions = [self._type_mention(question, candidate) for candidate in mention_candidates]
        entities: List[EntityRecord] = []
        unresolved_mentions: List[UnresolvedMention] = []
        ambiguity_flags: List[AmbiguityFlag] = []

        for typed_mention in typed_mentions:
            entity_record, unresolved_mention, ambiguity_flag = self._normalize_mention(
                question=question,
                typed_mention=typed_mention,
                task_spec=task_spec,
            )
            if entity_record is not None:
                entities.append(entity_record)
            if unresolved_mention is not None:
                unresolved_mentions.append(unresolved_mention)
            if ambiguity_flag is not None:
                ambiguity_flags.append(ambiguity_flag)

        entities = self._dedupe_entities(entities)
        unresolved_mentions = self._dedupe_unresolved_mentions(unresolved_mentions)
        ambiguity_flags = self._dedupe_ambiguity_flags(ambiguity_flags)
        condition_mentions = self._extract_condition_mentions(question, entities)
        return EntityPack(
            version="1.0",
            entities=entities,
            condition_mentions=condition_mentions,
            unresolved_mentions=unresolved_mentions,
            entity_ambiguity_flags=ambiguity_flags,
        )

    __call__ = run

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _build_alias_index(self) -> Dict[str, List[Dict[str, Any]]]:
        alias_index: Dict[str, List[Dict[str, Any]]] = {}
        for entity_type, entries in self.entity_aliases.items():
            for payload in entries.values():
                aliases = list(payload.get("aliases") or [])
                canonical_name = payload.get("canonical_name")
                if canonical_name and canonical_name not in aliases:
                    aliases.append(canonical_name)
                for alias in aliases:
                    alias_index.setdefault(str(alias).lower(), []).append(
                        {
                            "entity_type": entity_type,
                            "resolver_source": "local_alias",
                            "data": payload,
                        }
                    )
        for payload in self.reaction_aliases.values():
            aliases = list(payload.get("aliases") or [])
            canonical_name = payload.get("canonical_name")
            if canonical_name and canonical_name not in aliases:
                aliases.append(canonical_name)
            for alias in aliases:
                alias_index.setdefault(str(alias).lower(), []).append(
                    {
                        "entity_type": "reaction",
                        "resolver_source": "reaction_alias",
                        "data": payload,
                    }
                )
        return alias_index

    def _extract_mentions(self, question: str) -> List[MentionCandidate]:
        candidates: List[MentionCandidate] = []
        occupied_spans: List[Tuple[int, int]] = []

        aliases = sorted(self.alias_index.keys(), key=len, reverse=True)
        for alias in aliases:
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.I)
            for match in pattern.finditer(question):
                if self._span_overlaps(match.start(), match.end(), occupied_spans):
                    continue
                occupied_spans.append((match.start(), match.end()))
                candidates.append(
                    MentionCandidate(
                        mention=question[match.start() : match.end()],
                        start=match.start(),
                        end=match.end(),
                        alias_hits=list(self.alias_index.get(alias, [])),
                    )
                )

        for pattern in (ROLE_AFTER_PATTERN, ROLE_BEFORE_PATTERN):
            for match in pattern.finditer(question):
                span = match.span("lemma")
                if self._span_overlaps(span[0], span[1], occupied_spans):
                    continue
                occupied_spans.append(span)
                candidates.append(
                    MentionCandidate(
                        mention=question[span[0] : span[1]],
                        start=span[0],
                        end=span[1],
                        entity_type_hint=match.group("role").lower(),
                    )
                )

        for metric_pattern in METRIC_PATTERNS:
            for match in metric_pattern.finditer(question):
                if self._span_overlaps(match.start(), match.end(), occupied_spans):
                    continue
                occupied_spans.append((match.start(), match.end()))
                candidates.append(
                    MentionCandidate(
                        mention=question[match.start() : match.end()],
                        start=match.start(),
                        end=match.end(),
                        entity_type_hint="metric",
                    )
                )

        for match in FORMULA_TOKEN_PATTERN.finditer(question):
            mention = question[match.start() : match.end()]
            if self._span_overlaps(match.start(), match.end(), occupied_spans):
                continue
            if mention in COMMON_STOP_TOKENS:
                continue
            if not (re.search(r"\d", mention) or "/" in mention or "-" in mention or mention.isupper()):
                continue
            occupied_spans.append((match.start(), match.end()))
            candidates.append(MentionCandidate(mention=mention, start=match.start(), end=match.end()))

        candidates.sort(key=lambda item: (item.start, -(item.end - item.start)))
        return candidates

    def _type_mention(self, question: str, candidate: MentionCandidate) -> Dict[str, Any]:
        candidate_entity_types = sorted({hit["entity_type"] for hit in candidate.alias_hits})
        if candidate.entity_type_hint and candidate.entity_type_hint not in candidate_entity_types:
            candidate_entity_types.append(candidate.entity_type_hint)

        if not candidate_entity_types:
            heuristic_types = self._heuristic_entity_types(question, candidate)
            candidate_entity_types.extend(heuristic_types)

        return {
            "mention": candidate.mention,
            "start": candidate.start,
            "end": candidate.end,
            "source_text": candidate.mention,
            "alias_hits": candidate.alias_hits,
            "entity_type_hint": candidate.entity_type_hint,
            "candidate_entity_types": candidate_entity_types,
        }

    def _heuristic_entity_types(self, question: str, candidate: MentionCandidate) -> List[str]:
        mention = candidate.mention
        window_start = max(0, candidate.start - 24)
        window_end = min(len(question), candidate.end + 24)
        context_window = question[window_start:window_end].lower()
        upper_mention = mention.upper()
        lower_mention = mention.lower()

        if any(pattern.fullmatch(mention) for pattern in METRIC_PATTERNS):
            return ["metric"]
        if upper_mention in REACTION_ABBREVIATIONS or lower_mention.endswith("reaction"):
            return ["reaction"]
        if candidate.entity_type_hint:
            return [candidate.entity_type_hint]
        if "catalyst" in context_window or "/C" in mention or "-LDH" in mention:
            return ["catalyst"]
        if mention.startswith("NMC") or any(token in lower_mention for token in ("graphene", "oxide", "perovskite", "mof")):
            return ["material"]
        if "ligand" in context_window:
            return ["ligand"]
        if "solvent" in context_window:
            return ["solvent"]
        if "substrate" in context_window:
            return ["substrate"]
        if "reagent" in context_window or "additive" in context_window or "base" in context_window or "acid" in context_window:
            return ["reagent"]
        if re.search(r"\d", mention) or mention.isupper():
            return ["molecule", "reagent", "solvent", "ligand"]
        return []

    def _normalize_mention(
        self,
        *,
        question: str,
        typed_mention: Dict[str, Any],
        task_spec: TaskSpec,
    ) -> Tuple[Optional[EntityRecord], Optional[UnresolvedMention], Optional[AmbiguityFlag]]:
        mention = typed_mention["mention"]
        source_span = SourceSpan(start=typed_mention["start"], end=typed_mention["end"])
        alias_hits = typed_mention["alias_hits"]
        llm_decision = self._resolve_candidate_with_llm(
            question=question,
            typed_mention=typed_mention,
            task_spec=task_spec,
        )
        chosen_alias_hit = self._select_alias_hit(
            alias_hits,
            typed_mention,
            task_spec,
            llm_decision=llm_decision,
        )

        if chosen_alias_hit is not None:
            entity_type = chosen_alias_hit["entity_type"]
            payload = dict(chosen_alias_hit["data"])
            smiles = self._canonicalize_smiles(payload.get("smiles"))
            entity_record = EntityRecord(
                entity_id=self._make_entity_id(entity_type, mention, source_span),
                mention=mention,
                canonical_name=str(payload.get("canonical_name") or mention),
                entity_type=entity_type,
                entity_subtype=payload.get("entity_subtype"),
                formula=payload.get("formula"),
                smiles=smiles,
                inchikey=payload.get("inchikey"),
                pubchem_cid=payload.get("pubchem_cid"),
                aliases=list(payload.get("aliases") or []),
                query_anchors=self._build_query_anchors(mention, payload),
                resolver_source=chosen_alias_hit["resolver_source"],
                resolution_confidence=0.95 if chosen_alias_hit["resolver_source"] != "reaction_alias" else 0.98,
                status="resolved",
                source_text=typed_mention["source_text"],
                source_span=source_span,
            )
            ambiguity_flag = None
            if len({hit["entity_type"] for hit in alias_hits}) > 1:
                ambiguity_flag = AmbiguityFlag(
                    flag_type="entity_ambiguous",
                    target=mention,
                    note=f"Mention matched multiple entity types; resolved to {entity_type} using contextual preference.",
                    severity="low",
                )
            return entity_record, None, ambiguity_flag

        candidate_entity_types = list(typed_mention["candidate_entity_types"])
        llm_entity_type = llm_decision.get("entity_type") if isinstance(llm_decision, dict) else None
        if llm_entity_type in candidate_entity_types:
            candidate_entity_types = [llm_entity_type] + [
                entity_type for entity_type in candidate_entity_types if entity_type != llm_entity_type
            ]
        if len(candidate_entity_types) == 1 and candidate_entity_types[0] in SMALL_MOLECULE_TYPES:
            pubchem_payload = self._lookup_pubchem(mention)
            if pubchem_payload is not None:
                smiles = self._canonicalize_smiles(pubchem_payload.get("smiles"))
                entity_type = candidate_entity_types[0]
                return (
                    EntityRecord(
                        entity_id=self._make_entity_id(entity_type, mention, source_span),
                        mention=mention,
                        canonical_name=str(pubchem_payload.get("canonical_name") or mention),
                        entity_type=entity_type,
                        entity_subtype=None,
                        formula=pubchem_payload.get("formula"),
                        smiles=smiles,
                        inchikey=pubchem_payload.get("inchikey"),
                        pubchem_cid=pubchem_payload.get("pubchem_cid"),
                        aliases=list(pubchem_payload.get("aliases") or [mention]),
                        query_anchors=self._build_query_anchors(mention, pubchem_payload),
                        resolver_source="pubchem",
                        resolution_confidence=0.88,
                        status="resolved",
                        source_text=typed_mention["source_text"],
                        source_span=source_span,
                    ),
                    None,
                    None,
                )

        if len(candidate_entity_types) == 1 and candidate_entity_types[0] in {"material", "catalyst", "reaction", "metric", "condition", "substrate"}:
            entity_type = candidate_entity_types[0]
            confidence = 0.76 if entity_type in {"material", "catalyst", "reaction"} else 0.62
            status = "resolved" if entity_type == "reaction" else "partially_resolved"
            payload = {
                "canonical_name": mention,
                "aliases": [mention],
                "formula": None,
                "smiles": None,
                "inchikey": None,
                "pubchem_cid": None,
            }
            entity_record = EntityRecord(
                entity_id=self._make_entity_id(entity_type, mention, source_span),
                mention=mention,
                canonical_name=mention,
                entity_type=entity_type,
                entity_subtype=None,
                formula=None,
                smiles=None,
                inchikey=None,
                pubchem_cid=None,
                aliases=[mention],
                query_anchors=self._build_query_anchors(mention, payload),
                resolver_source=f"heuristic_{entity_type}",
                resolution_confidence=confidence,
                status=status,
                source_text=typed_mention["source_text"],
                source_span=source_span,
            )
            ambiguity_flag = None
            if status != "resolved":
                ambiguity_flag = AmbiguityFlag(
                    flag_type="entity_ambiguous",
                    target=mention,
                    note=f"{entity_type} mention was kept without structural normalization.",
                    severity="low",
                )
            return entity_record, None, ambiguity_flag

        unresolved_mention = UnresolvedMention(
            mention=mention,
            candidate_entity_types=candidate_entity_types,
            reason=self._build_unresolved_reason(candidate_entity_types, typed_mention),
            confidence=0.35 if candidate_entity_types else 0.2,
            source_text=typed_mention["source_text"],
            source_span=source_span,
        )
        ambiguity_flag = AmbiguityFlag(
            flag_type="entity_ambiguous",
            target=mention,
            note=unresolved_mention.reason,
            severity="medium",
        )
        return None, unresolved_mention, ambiguity_flag

    def _select_alias_hit(
        self,
        alias_hits: Sequence[Dict[str, Any]],
        typed_mention: Dict[str, Any],
        task_spec: TaskSpec,
        *,
        llm_decision: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not alias_hits:
            return None
        if len(alias_hits) == 1:
            return alias_hits[0]

        llm_alias_index = llm_decision.get("alias_hit_index") if isinstance(llm_decision, dict) else None
        if isinstance(llm_alias_index, int) and 0 <= llm_alias_index < len(alias_hits):
            llm_alias_hit = alias_hits[llm_alias_index]
            llm_entity_type = llm_decision.get("entity_type")
            if llm_entity_type in {None, llm_alias_hit["entity_type"]}:
                return llm_alias_hit

        llm_entity_type = llm_decision.get("entity_type") if isinstance(llm_decision, dict) else None
        if llm_entity_type:
            llm_filtered = [hit for hit in alias_hits if hit["entity_type"] == llm_entity_type]
            if len(llm_filtered) == 1:
                return llm_filtered[0]
            if len(llm_filtered) > 1:
                alias_hits = llm_filtered

        preferred_types = set(task_spec.query_constraints.preferred_entity_types)
        for axis in task_spec.required_condition_axes:
            if axis in ENTITY_AXIS_MAP.values():
                preferred_types.add(axis)
        type_hint = typed_mention.get("entity_type_hint")
        if type_hint:
            preferred_types.add(type_hint)

        filtered = [hit for hit in alias_hits if hit["entity_type"] in preferred_types]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            alias_hits = filtered

        canonical_names = {str(hit["data"].get("canonical_name") or "") for hit in alias_hits}
        if len(canonical_names) == 1:
            return sorted(alias_hits, key=lambda hit: ENTITY_TYPE_PRIORITY.get(hit["entity_type"], 99))[0]
        return None

    def _resolve_candidate_with_llm(
        self,
        *,
        question: str,
        typed_mention: Dict[str, Any],
        task_spec: TaskSpec,
    ) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None
        alias_hits = list(typed_mention.get("alias_hits") or [])
        candidate_entity_types = list(typed_mention.get("candidate_entity_types") or [])
        if len(alias_hits) <= 1 and len(candidate_entity_types) <= 1:
            return None

        mention_payload = {
            "mention": typed_mention.get("mention"),
            "source_text": typed_mention.get("source_text"),
            "entity_type_hint": typed_mention.get("entity_type_hint"),
            "candidate_entity_types": candidate_entity_types,
            "alias_hits": [
                {
                    "index": index,
                    "entity_type": hit.get("entity_type"),
                    "resolver_source": hit.get("resolver_source"),
                    "canonical_name": str((hit.get("data") or {}).get("canonical_name") or ""),
                    "aliases": list((hit.get("data") or {}).get("aliases") or []),
                }
                for index, hit in enumerate(alias_hits)
            ],
        }
        messages = [
            {"role": "system", "content": ENTITY_RESOLVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_entity_resolver_user_prompt(
                    question=question,
                    task_spec=task_spec.model_dump(exclude_none=True),
                    mention_payload=mention_payload,
                ),
            },
        ]
        try:
            parsed = parse_json_object(invoke_llm(self.llm, messages))
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        allowed_entity_types = {
            *candidate_entity_types,
            *(hit.get("entity_type") for hit in alias_hits if hit.get("entity_type")),
        }
        entity_type = parsed.get("entity_type")
        if entity_type not in allowed_entity_types:
            entity_type = None

        alias_hit_index = parsed.get("alias_hit_index")
        try:
            alias_hit_index = int(alias_hit_index) if alias_hit_index is not None else None
        except (TypeError, ValueError):
            alias_hit_index = None
        if alias_hit_index is not None and not (0 <= alias_hit_index < len(alias_hits)):
            alias_hit_index = None

        if entity_type is None and alias_hit_index is not None:
            entity_type = alias_hits[alias_hit_index].get("entity_type")

        if entity_type is None and alias_hit_index is None:
            return None
        return {
            "entity_type": entity_type,
            "alias_hit_index": alias_hit_index,
        }

    def _lookup_pubchem(self, mention: str) -> Optional[Dict[str, Any]]:
        if self.pubchem_client is None:
            return None
        lookup_method = getattr(self.pubchem_client, "lookup_compound", None)
        if callable(lookup_method):
            return lookup_method(mention)
        if callable(self.pubchem_client):
            return self.pubchem_client(mention)
        return None

    def _canonicalize_smiles(self, smiles: Optional[str]) -> Optional[str]:
        if not smiles or not self.enable_rdkit:
            return smiles
        try:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return smiles
            return Chem.MolToSmiles(molecule, canonical=True)
        except Exception:
            return smiles

    def _build_query_anchors(self, mention: str, payload: Dict[str, Any]) -> List[str]:
        anchors: List[str] = []
        for candidate in [mention, payload.get("canonical_name"), payload.get("formula"), *(payload.get("aliases") or [])]:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text and text not in anchors:
                anchors.append(text)
        return anchors

    def _build_unresolved_reason(self, candidate_entity_types: Sequence[str], typed_mention: Dict[str, Any]) -> str:
        if not candidate_entity_types:
            return "No supported entity type could be assigned from the mention extraction stage."
        if len(candidate_entity_types) > 1:
            return f"Mention could map to multiple entity types: {', '.join(candidate_entity_types)}."
        entity_type = candidate_entity_types[0]
        if entity_type in SMALL_MOLECULE_TYPES:
            return "Small-molecule normalization did not resolve the mention with local aliases or PubChem."
        return f"Mention was typed as {entity_type} but could not be fully normalized."

    def _extract_condition_mentions(self, question: str, entities: Sequence[EntityRecord]) -> List[ConditionMention]:
        conditions: List[ConditionMention] = []
        seen: set[Tuple[str, int, int]] = set()

        for entity in entities:
            axis = ENTITY_AXIS_MAP.get(entity.entity_type)
            if axis is None:
                continue
            key = (axis, entity.source_span.start, entity.source_span.end)
            if key in seen:
                continue
            seen.add(key)
            conditions.append(
                ConditionMention(
                    condition_id=f"cond_{len(conditions) + 1}",
                    axis=axis,
                    raw_value=entity.mention,
                    normalized_value=entity.canonical_name,
                    unit=None,
                    operator=None,
                    confidence=entity.resolution_confidence,
                    source_text=entity.source_text,
                    source_span=entity.source_span,
                )
            )

        for pattern, axis, unit_normalizer in self._numeric_condition_patterns():
            for match in pattern.finditer(question):
                span = (match.start(), match.end())
                key = (axis, span[0], span[1])
                if key in seen:
                    continue
                seen.add(key)
                value = match.group("value") if "value" in match.groupdict() else None
                unit = match.group("unit") if "unit" in match.groupdict() else None
                operator = match.group("op") if "op" in match.groupdict() else None
                normalized_value = self._normalize_condition_value(axis, value, unit, unit_normalizer, match)
                conditions.append(
                    ConditionMention(
                        condition_id=f"cond_{len(conditions) + 1}",
                        axis=axis,
                        raw_value=question[match.start() : match.end()],
                        normalized_value=normalized_value,
                        unit=unit_normalizer(unit) if callable(unit_normalizer) else unit,
                        operator=operator,
                        confidence=0.85,
                        source_text=question[match.start() : match.end()],
                        source_span=SourceSpan(start=match.start(), end=match.end()),
                    )
                )

        return conditions

    def _numeric_condition_patterns(self):
        return [
            (
                re.compile(r"(?P<op><=|>=|<|>|≈|~)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>°\s*C|℃|K)\b", re.I),
                "temperature",
                lambda unit: "°C" if unit and "C" in unit.upper() else unit,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|≈|~)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hrs|hour|hours|min|minute|minutes|s|sec|second|seconds)\b", re.I),
                "time",
                lambda unit: unit.lower() if unit else unit,
            ),
            (
                re.compile(r"\bpH\s*(?P<op><=|>=|<|>|=)?\s*(?P<value>\d+(?:\.\d+)?)\b", re.I),
                "ph",
                lambda unit: None,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|≈|~)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>V|mV)\b(?:\s*(?:vs\.?|versus)\s*(?P<ref>[A-Za-z0-9/+\-.]+))?", re.I),
                "potential",
                lambda unit: unit,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|≈|~)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bar|atm|kPa|MPa|Pa)\b", re.I),
                "pressure",
                lambda unit: unit,
            ),
            (
                re.compile(r"\b(?:yield)\s*(?:of|=|at)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\b", re.I),
                "yield",
                lambda unit: "%" if unit and unit.lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\s*(?:yield)\b", re.I),
                "yield",
                lambda unit: "%" if unit and unit.lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?:selectivity|Faradaic efficiency|FE)\s*(?:of|=|at)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\b", re.I),
                "selectivity",
                lambda unit: "%" if unit and unit.lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\s*(?:selectivity|Faradaic efficiency|FE)\b", re.I),
                "selectivity",
                lambda unit: "%" if unit and unit.lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>M)\s+(?P<species>[A-Za-z0-9()/-]+)\b"),
                "electrolyte",
                lambda unit: unit,
            ),
        ]

    def _normalize_condition_value(
        self,
        axis: str,
        value: Optional[str],
        unit: Optional[str],
        unit_normalizer,
        match: re.Match[str],
    ) -> Optional[str]:
        if axis == "electrolyte":
            species = match.group("species") if "species" in match.groupdict() else None
            if value and species:
                return f"{value} {unit_normalizer(unit)} {species}"
        if value is None:
            return None
        normalized_unit = unit_normalizer(unit) if callable(unit_normalizer) else unit
        if normalized_unit:
            return f"{value} {normalized_unit}"
        return value

    def _dedupe_entities(self, entities: Sequence[EntityRecord]) -> List[EntityRecord]:
        deduped: List[EntityRecord] = []
        seen: set[Tuple[str, str, int, int]] = set()
        for entity in entities:
            key = (entity.entity_type, entity.canonical_name, entity.source_span.start, entity.source_span.end)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped

    def _dedupe_unresolved_mentions(self, unresolved_mentions: Sequence[UnresolvedMention]) -> List[UnresolvedMention]:
        deduped: List[UnresolvedMention] = []
        seen: set[Tuple[str, int, int]] = set()
        for unresolved_mention in unresolved_mentions:
            key = (
                unresolved_mention.mention,
                unresolved_mention.source_span.start,
                unresolved_mention.source_span.end,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(unresolved_mention)
        return deduped

    def _dedupe_ambiguity_flags(self, ambiguity_flags: Sequence[AmbiguityFlag]) -> List[AmbiguityFlag]:
        deduped: List[AmbiguityFlag] = []
        seen: set[Tuple[str, str, str]] = set()
        for ambiguity_flag in ambiguity_flags:
            key = (ambiguity_flag.flag_type, ambiguity_flag.target, ambiguity_flag.note)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ambiguity_flag)
        return deduped

    def _make_entity_id(self, entity_type: str, mention: str, source_span: SourceSpan) -> str:
        normalized_mention = re.sub(r"[^A-Za-z0-9]+", "_", mention).strip("_").lower() or "mention"
        return f"{entity_type}_{normalized_mention}_{source_span.start}_{source_span.end}"

    def _span_overlaps(self, start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
        for existing_start, existing_end in spans:
            if start < existing_end and end > existing_start:
                return True
        return False
