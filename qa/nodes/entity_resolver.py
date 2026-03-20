from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests
import yaml

from prompts.qa_prompts import (
    ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_RESOLVER_SYSTEM_PROMPT,
    build_entity_mention_extraction_user_prompt,
    build_entity_resolver_user_prompt,
)
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

ALLOWED_ENTITY_TYPES = (
    "reaction",
    "metric",
    "condition",
    "material",
    "catalyst",
    "molecule",
    "solvent",
    "reagent",
    "ligand",
    "substrate",
)
DEFAULT_PUBCHEM_ENTITY_TYPES = {"molecule", "solvent", "reagent", "ligand", "substrate"}
SEED_SUGGESTION_ENTITY_TYPES = {"reaction", "metric", "condition", "material", "catalyst"}


@dataclass
class ResolutionEntry:
    entry_id: str
    entity_type: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    query_anchors: List[str] = field(default_factory=list)
    formula: Optional[str] = None
    smiles: Optional[str] = None
    inchikey: Optional[str] = None
    pubchem_cid: Optional[int] = None
    resolver_source: str = "heuristic"
    resolution_confidence: float = 0.0
    status: str = "resolved"
    entity_subtype: Optional[str] = None
    lookup_keys: List[str] = field(default_factory=list)

    def to_record(
        self,
        *,
        mention: str,
        source_text: str,
        source_span: SourceSpan,
        entity_id: str,
    ) -> EntityRecord:
        return EntityRecord(
            entity_id=entity_id,
            mention=mention,
            canonical_name=self.canonical_name,
            entity_type=self.entity_type,
            entity_subtype=self.entity_subtype,
            formula=self.formula,
            smiles=self.smiles,
            inchikey=self.inchikey,
            pubchem_cid=self.pubchem_cid,
            aliases=list(self.aliases),
            query_anchors=list(self.query_anchors),
            resolver_source=self.resolver_source,
            resolution_confidence=max(0.0, min(float(self.resolution_confidence), 1.0)),
            status=self.status,
            source_text=source_text,
            source_span=source_span,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "entity_subtype": self.entity_subtype,
            "formula": self.formula,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "pubchem_cid": self.pubchem_cid,
            "aliases": list(self.aliases),
            "query_anchors": list(self.query_anchors),
            "resolver_source": self.resolver_source,
            "resolution_confidence": self.resolution_confidence,
            "status": self.status,
            "lookup_keys": list(self.lookup_keys),
        }


@dataclass
class RunResolutionIndex:
    entries: List[ResolutionEntry] = field(default_factory=list)
    lookup: Dict[str, List[ResolutionEntry]] = field(default_factory=dict)
    identity_lookup: Dict[Tuple[str, str, str], ResolutionEntry] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def lookup_entries(self, value: str) -> List[ResolutionEntry]:
        key = self._normalize_lookup_key(value)
        if not key:
            return []
        return list(self.lookup.get(key, []))

    def record_hit(self, *, mention: str, entry: ResolutionEntry) -> None:
        self.events.append(
            {
                "event": "hit",
                "mention": mention,
                "entry_id": entry.entry_id,
                "entity_type": entry.entity_type,
                "canonical_name": entry.canonical_name,
            }
        )

    def register(self, entry: ResolutionEntry, *, mention: Optional[str] = None) -> ResolutionEntry:
        identity_key = self._identity_key(entry)
        existing = self.identity_lookup.get(identity_key)
        if existing is not None:
            self._merge_entry(existing, entry)
            self._register_lookup(existing, mention)
            self.events.append(
                {
                    "event": "merge",
                    "mention": mention or "",
                    "entry_id": existing.entry_id,
                    "entity_type": existing.entity_type,
                    "canonical_name": existing.canonical_name,
                }
            )
            return existing

        if not entry.entry_id:
            entry.entry_id = f"res_{len(self.entries) + 1}"
        self.entries.append(entry)
        self.identity_lookup[identity_key] = entry
        self._register_lookup(entry, mention)
        self.events.append(
            {
                "event": "store",
                "mention": mention or "",
                "entry_id": entry.entry_id,
                "entity_type": entry.entity_type,
                "canonical_name": entry.canonical_name,
            }
        )
        return entry

    def to_payload(self) -> Dict[str, Any]:
        return {
            "entries": [entry.to_payload() for entry in self.entries],
            "cache_events": list(self.events),
        }

    def _register_lookup(self, entry: ResolutionEntry, mention: Optional[str]) -> None:
        for value in [mention, entry.canonical_name, entry.formula, *(entry.aliases or []), *(entry.query_anchors or [])]:
            key = self._normalize_lookup_key(value)
            if not key:
                continue
            if key not in entry.lookup_keys:
                entry.lookup_keys.append(key)
            current = self.lookup.setdefault(key, [])
            if entry not in current:
                current.append(entry)

    def _merge_entry(self, current: ResolutionEntry, incoming: ResolutionEntry) -> None:
        current.aliases = self._merge_unique_text(current.aliases, incoming.aliases)
        current.query_anchors = self._merge_unique_text(current.query_anchors, incoming.query_anchors)
        current.lookup_keys = self._merge_unique_text(current.lookup_keys, incoming.lookup_keys)
        current.resolution_confidence = max(current.resolution_confidence, incoming.resolution_confidence)
        if current.status != "resolved" and incoming.status == "resolved":
            current.status = incoming.status
        for field_name in ("formula", "smiles", "inchikey", "pubchem_cid", "entity_subtype"):
            if getattr(current, field_name) in (None, "") and getattr(incoming, field_name) not in (None, ""):
                setattr(current, field_name, getattr(incoming, field_name))
        if current.resolver_source == "heuristic" and incoming.resolver_source != "heuristic":
            current.resolver_source = incoming.resolver_source

    def _identity_key(self, entry: ResolutionEntry) -> Tuple[str, str, str]:
        if entry.inchikey:
            return ("inchikey", entry.entity_type, str(entry.inchikey).upper())
        if entry.pubchem_cid is not None:
            return ("pubchem_cid", entry.entity_type, str(entry.pubchem_cid))
        return ("canonical_name", entry.entity_type, self._normalize_lookup_key(entry.canonical_name))

    @staticmethod
    def _merge_unique_text(existing: Sequence[str], extra: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in [*(existing or []), *(extra or [])]:
            text = str(value or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    @staticmethod
    def _normalize_lookup_key(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()


@dataclass
class EntityResolutionRunResult:
    entity_pack: EntityPack
    resolution_index: Dict[str, Any]
    provider_calls: List[Dict[str, Any]]
    seed_suggestions: List[Dict[str, Any]]

    def artifact_payloads(self) -> Dict[str, Any]:
        return {
            "entity_resolver/entity_pack.json": self.entity_pack.model_dump(exclude_none=True),
            "entity_resolver/resolution_index.json": self.resolution_index,
            "entity_resolver/provider_calls.json": list(self.provider_calls),
            "entity_resolver/seed_suggestions.json": list(self.seed_suggestions),
        }


class PubChemClient:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def search_candidates(self, query: str, max_candidates: int = 5) -> List[Dict[str, Any]]:
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            return []
        cids_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(cleaned_query, safe='')}/cids/JSON"
        )
        cid_payload = self._get_json(cids_url)
        cid_values = list((cid_payload or {}).get("IdentifierList", {}).get("CID", []) or [])
        cid_values = [str(value).strip() for value in cid_values if str(value).strip()][: max(1, int(max_candidates or 1))]
        if not cid_values:
            fallback = self._lookup_single(cleaned_query)
            return [fallback] if fallback is not None else []

        property_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{','.join(cid_values)}/property/IUPACName,CanonicalSMILES,InChIKey,CID,MolecularFormula/JSON"
        )
        property_payload = self._get_json(property_url)
        properties = list((property_payload or {}).get("PropertyTable", {}).get("Properties", []) or [])
        if not properties:
            fallback = self._lookup_single(cleaned_query)
            return [fallback] if fallback is not None else []
        return [self._normalize_property_item(item, cleaned_query) for item in properties if isinstance(item, dict)]

    def lookup_compound(self, query: str) -> Optional[Dict[str, Any]]:
        candidates = self.search_candidates(query=query, max_candidates=1)
        return candidates[0] if candidates else None

    def _lookup_single(self, query: str) -> Optional[Dict[str, Any]]:
        property_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/"
            "IUPACName,CanonicalSMILES,InChIKey,CID,MolecularFormula/JSON"
        )
        payload = self._get_json(property_url, params={"name": query})
        properties = list((payload or {}).get("PropertyTable", {}).get("Properties", []) or [])
        if not properties:
            return None
        return self._normalize_property_item(properties[0], query)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_property_item(item: Dict[str, Any], query: str) -> Dict[str, Any]:
        return {
            "canonical_name": item.get("IUPACName") or query,
            "formula": item.get("MolecularFormula"),
            "smiles": item.get("CanonicalSMILES"),
            "inchikey": item.get("InChIKey"),
            "pubchem_cid": item.get("CID"),
            "aliases": [query],
        }


class EntityResolverNode:
    def __init__(
        self,
        llm: Any = None,
        pubchem_client: Optional[Any] = None,
        enable_rdkit: Optional[bool] = None,
        seed_path: Optional[str] = None,
        pubchem_enabled: bool = True,
        pubchem_entity_types: Optional[Sequence[str]] = None,
        max_pubchem_candidates: int = 5,
        mention_extraction_min_confidence: float = 0.7,
        llm_disambiguation_enabled: bool = True,
        disambiguation_min_confidence: float = 0.7,
        fail_open_on_provider_error: bool = True,
        emit_seed_suggestions: bool = True,
    ) -> None:
        resource_root = Path(__file__).resolve().parents[1] / "resources"
        default_seed_path = resource_root / "entity_seeds.yaml"
        self.seed_entries = self._load_seed_entries(seed_path=Path(seed_path) if seed_path else default_seed_path)
        self.seed_index = self._build_seed_index()
        self.llm = llm
        self.pubchem_client = pubchem_client
        self.enable_rdkit = True if enable_rdkit is None else bool(enable_rdkit)
        self.pubchem_enabled = bool(pubchem_enabled)
        self.pubchem_entity_types = {
            str(value).strip()
            for value in (pubchem_entity_types or DEFAULT_PUBCHEM_ENTITY_TYPES)
            if str(value).strip()
        }
        self.max_pubchem_candidates = max(1, int(max_pubchem_candidates or 1))
        self.mention_extraction_min_confidence = max(0.0, min(float(mention_extraction_min_confidence), 1.0))
        self.llm_disambiguation_enabled = bool(llm_disambiguation_enabled)
        self.disambiguation_min_confidence = max(0.0, min(float(disambiguation_min_confidence), 1.0))
        self.fail_open_on_provider_error = bool(fail_open_on_provider_error)
        self.emit_seed_suggestions = bool(emit_seed_suggestions)

    def run(self, question: str, task_spec: TaskSpec) -> EntityPack:
        return self.resolve_detailed(question=question, task_spec=task_spec).entity_pack

    __call__ = run

    def _load_seed_entries(
        self,
        *,
        seed_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        payload = self._load_yaml(seed_path)
        return {
            str(entity_type): dict(entries or {})
            for entity_type, entries in dict(payload or {}).items()
            if isinstance(entries, dict)
        }

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}

    def _build_seed_index(self) -> Dict[str, List[Dict[str, Any]]]:
        alias_index: Dict[str, List[Dict[str, Any]]] = {}
        for entity_type, entries in self.seed_entries.items():
            for entry_id, payload in entries.items():
                if not isinstance(payload, dict):
                    continue
                aliases = list(payload.get("aliases") or [])
                canonical_name = payload.get("canonical_name")
                if canonical_name and canonical_name not in aliases:
                    aliases.append(canonical_name)
                for alias in aliases:
                    key = str(alias or "").strip().lower()
                    if not key:
                        continue
                    alias_index.setdefault(key, []).append(
                        {
                            "entry_id": str(entry_id),
                            "entity_type": str(entity_type),
                            "resolver_source": "seed",
                            "data": dict(payload),
                        }
                    )
        return alias_index

    def _extract_mentions_with_llm(self, question: str, task_spec: TaskSpec) -> List[Dict[str, Any]]:
        if self.llm is None:
            raise ValueError("EntityResolverNode requires a configured LLM for mention extraction.")
        messages = [
            {"role": "system", "content": ENTITY_MENTION_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_entity_mention_extraction_user_prompt(
                    question=question,
                    task_spec=task_spec.model_dump(exclude_none=True),
                    allowed_entity_types=ALLOWED_ENTITY_TYPES,
                ),
            },
        ]
        try:
            parsed = parse_json_object(invoke_llm(self.llm, messages))
        except Exception as exc:
            raise ValueError(f"Entity mention extraction failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Entity mention extraction failed: expected a JSON object.")
        mentions = parsed.get("mentions")
        if not isinstance(mentions, list):
            raise ValueError("Entity mention extraction failed: 'mentions' must be a JSON array.")
        extracted_mentions: List[Dict[str, Any]] = []
        occupied_spans: List[Tuple[int, int]] = []
        cursor = 0
        for index, item in enumerate(mentions):
            validated_mention = None
            try:
                validated_mention = self._validate_and_align_extracted_mention(
                    question=question,
                    task_spec=task_spec,
                    mention_payload=item,
                    occupied_spans=occupied_spans,
                    cursor=cursor,
                    mention_index=index,
                )
            except ValueError:
                continue
            extracted_mentions.append(validated_mention)
            cursor = extracted_mentions[-1]["end"]
        return extracted_mentions

    def _validate_and_align_extracted_mention(
        self,
        *,
        question: str,
        task_spec: TaskSpec,
        mention_payload: Any,
        occupied_spans: List[Tuple[int, int]],
        cursor: int,
        mention_index: int,
    ) -> Dict[str, Any]:
        if not isinstance(mention_payload, dict):
            raise ValueError(f"Entity mention extraction failed: mention #{mention_index + 1} must be an object.")
        surface_form = str(mention_payload.get("surface_form") or "").strip()
        if not surface_form:
            raise ValueError(f"Entity mention extraction failed: mention #{mention_index + 1} has empty surface_form.")
        raw_candidate_types = mention_payload.get("candidate_entity_types")
        if not isinstance(raw_candidate_types, list) or not raw_candidate_types:
            raise ValueError(
                f"Entity mention extraction failed: mention '{surface_form}' must include candidate_entity_types."
            )
        candidate_entity_types: List[str] = []
        for value in raw_candidate_types:
            entity_type = str(value or "").strip()
            if entity_type not in ALLOWED_ENTITY_TYPES:
                raise ValueError(
                    f"Entity mention extraction failed: mention '{surface_form}' has invalid entity type '{entity_type}'."
                )
            if entity_type not in candidate_entity_types:
                candidate_entity_types.append(entity_type)
        raw_selected_type = mention_payload.get("selected_entity_type")
        selected_entity_type = str(raw_selected_type or "").strip() if raw_selected_type is not None else ""
        if selected_entity_type and selected_entity_type not in candidate_entity_types:
            raise ValueError(
                f"Entity mention extraction failed: mention '{surface_form}' selected_entity_type must be null "
                "or come from candidate_entity_types."
            )
        try:
            confidence = float(mention_payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Entity mention extraction failed: mention '{surface_form}' must include numeric confidence."
            ) from exc
        confidence = max(0.0, min(confidence, 1.0))
        rationale = str(mention_payload.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(f"Entity mention extraction failed: mention '{surface_form}' must include rationale.")
        start, end = self._align_surface_form(
            question=question,
            surface_form=surface_form,
            cursor=cursor,
            occupied_spans=occupied_spans,
        )
        mention_text = question[start:end]
        alias_hits = list(self.seed_index.get(self._normalize_lookup_text(mention_text), []))
        if not candidate_entity_types and alias_hits:
            candidate_entity_types = sorted({str(hit.get("entity_type") or "") for hit in alias_hits if hit.get("entity_type")})
        return {
            "mention": mention_text,
            "start": start,
            "end": end,
            "source_text": mention_text,
            "alias_hits": alias_hits,
            "candidate_entity_types": candidate_entity_types,
            "selected_entity_type": selected_entity_type or None,
            "mention_extraction_confidence": confidence,
            "mention_extraction_rationale": rationale,
            "task_question": task_spec.question,
        }

    def _align_surface_form(
        self,
        *,
        question: str,
        surface_form: str,
        cursor: int,
        occupied_spans: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        question_lower = question.lower()
        surface_lower = surface_form.lower()
        preferred_start = max(0, int(cursor or 0))
        candidate_starts: List[int] = []
        search_from = preferred_start
        while True:
            start = question_lower.find(surface_lower, search_from)
            if start < 0:
                break
            candidate_starts.append(start)
            search_from = start + 1
        search_from = 0
        while True:
            start = question_lower.find(surface_lower, search_from)
            if start < 0:
                break
            if start not in candidate_starts:
                candidate_starts.append(start)
            search_from = start + 1
        if not candidate_starts:
            raise ValueError(
                f"Entity mention extraction failed: surface_form '{surface_form}' could not be aligned as an exact span."
            )
        saw_overlap = False
        for start in candidate_starts:
            end = start + len(surface_form)
            if question[start:end].lower() != surface_lower:
                continue
            if self._span_overlaps(start, end, occupied_spans):
                saw_overlap = True
                continue
            occupied_spans.append((start, end))
            return start, end
        if saw_overlap:
            raise ValueError(
                f"Entity mention extraction failed: surface_form '{surface_form}' overlaps a previous extracted span."
            )
        raise ValueError(
            f"Entity mention extraction failed: surface_form '{surface_form}' could not be aligned as an exact span."
        )

    def resolve_detailed(self, question: str, task_spec: TaskSpec) -> EntityResolutionRunResult:
        typed_mentions = self._extract_mentions_with_llm(question, task_spec)
        entities: List[EntityRecord] = []
        unresolved_mentions: List[UnresolvedMention] = []
        ambiguity_flags: List[AmbiguityFlag] = []
        resolution_index = RunResolutionIndex()
        provider_calls: List[Dict[str, Any]] = []
        seed_suggestion_tracker: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for typed_mention in typed_mentions:
            entity_record, unresolved_mention, ambiguity_flag = self._normalize_mention(
                question=question,
                typed_mention=typed_mention,
                task_spec=task_spec,
                resolution_index=resolution_index,
                provider_calls=provider_calls,
                seed_suggestion_tracker=seed_suggestion_tracker,
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
        entity_pack = EntityPack(
            version="1.0",
            entities=entities,
            condition_mentions=condition_mentions,
            unresolved_mentions=unresolved_mentions,
            entity_ambiguity_flags=ambiguity_flags,
        )
        return EntityResolutionRunResult(
            entity_pack=entity_pack,
            resolution_index=resolution_index.to_payload(),
            provider_calls=provider_calls,
            seed_suggestions=self._finalize_seed_suggestions(seed_suggestion_tracker),
        )

    def _normalize_mention(
        self,
        *,
        question: str,
        typed_mention: Dict[str, Any],
        task_spec: TaskSpec,
        resolution_index: RunResolutionIndex,
        provider_calls: List[Dict[str, Any]],
        seed_suggestion_tracker: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> Tuple[Optional[EntityRecord], Optional[UnresolvedMention], Optional[AmbiguityFlag]]:
        mention = str(typed_mention.get("mention") or "").strip()
        source_span = SourceSpan(start=int(typed_mention["start"]), end=int(typed_mention["end"]))
        candidate_entity_types = list(typed_mention.get("candidate_entity_types") or [])
        selected_entity_type = str(typed_mention.get("selected_entity_type") or "").strip()
        mention_extraction_confidence = max(
            0.0,
            min(float(typed_mention.get("mention_extraction_confidence") or 0.0), 1.0),
        )
        alias_hits = list(typed_mention.get("alias_hits") or [])

        if not selected_entity_type or mention_extraction_confidence < self.mention_extraction_min_confidence:
            unresolved_reason = self._build_extraction_unresolved_reason(
                selected_entity_type=selected_entity_type,
                candidate_entity_types=candidate_entity_types,
                confidence=mention_extraction_confidence,
            )
            unresolved_mention = UnresolvedMention(
                mention=mention,
                candidate_entity_types=candidate_entity_types,
                reason=unresolved_reason,
                confidence=mention_extraction_confidence,
                source_text=typed_mention["source_text"],
                source_span=source_span,
            )
            ambiguity_flag = AmbiguityFlag(
                flag_type="entity_ambiguous",
                target=mention,
                note=unresolved_reason,
                severity="medium",
            )
            self._note_seed_suggestion(
                tracker=seed_suggestion_tracker,
                mention=mention,
                entity_type=selected_entity_type or (candidate_entity_types[0] if len(candidate_entity_types) == 1 else "ambiguous"),
                proposed_canonical_name=mention,
                aliases_seen=[mention],
                resolver_source="unresolved",
                reason="low_confidence_extraction" if selected_entity_type else "unresolved_extraction",
                source_span=source_span,
            )
            return None, unresolved_mention, ambiguity_flag

        cached_entry = self._lookup_runtime_entry(
            resolution_index=resolution_index,
            mention=mention,
            candidate_entity_types=[selected_entity_type, *candidate_entity_types],
            task_spec=task_spec,
            type_hint=selected_entity_type,
        )
        if cached_entry is not None:
            resolution_index.record_hit(mention=mention, entry=cached_entry)
            return (
                cached_entry.to_record(
                    mention=mention,
                    source_text=typed_mention["source_text"],
                    source_span=source_span,
                    entity_id=self._make_entity_id(cached_entry.entity_type, mention, source_span),
                ),
                None,
                None,
            )

        alias_llm_selection = self._resolve_llm_selection(
            question=question,
            task_spec=task_spec,
            typed_mention=typed_mention,
            candidate_options=[
                {
                    "index": index,
                    "entity_type": hit.get("entity_type"),
                    "resolver_source": hit.get("resolver_source"),
                    "canonical_name": str((hit.get("data") or {}).get("canonical_name") or ""),
                    "aliases": list((hit.get("data") or {}).get("aliases") or []),
                    "formula": (hit.get("data") or {}).get("formula"),
                    "pubchem_cid": (hit.get("data") or {}).get("pubchem_cid"),
                }
                for index, hit in enumerate(alias_hits)
            ],
            allowed_entity_types=sorted(
                {
                    str(hit.get("entity_type") or "").strip()
                    for hit in alias_hits
                    if str(hit.get("entity_type") or "").strip()
                }
            )
            or [selected_entity_type],
        )
        chosen_seed_hit = self._select_seed_hit(
            alias_hits=alias_hits,
            typed_mention=typed_mention,
            task_spec=task_spec,
            llm_selection=alias_llm_selection,
        )
        if chosen_seed_hit is not None:
            seed_payload = dict(chosen_seed_hit.get("data") or {})
            pubchem_entity_type = self._select_pubchem_entity_type(
                str(chosen_seed_hit.get("entity_type") or selected_entity_type)
            )
            if pubchem_entity_type is not None:
                pubchem_candidates = self._lookup_pubchem_candidates(
                    mention=mention,
                    provider_calls=provider_calls,
                )
                selected_candidate, _ambiguity_note, _selected_via_llm, _selected_confidence = self._select_pubchem_candidate(
                    question=question,
                    task_spec=task_spec,
                    typed_mention=typed_mention,
                    entity_type=pubchem_entity_type,
                    candidates=pubchem_candidates,
                )
                if selected_candidate is not None:
                    merged_seed_payload = dict(selected_candidate)
                    merged_seed_payload.update({key: value for key, value in seed_payload.items() if value not in (None, "", [])})
                    merged_seed_payload["aliases"] = self._merge_unique_text(
                        list(selected_candidate.get("aliases") or []),
                        list(seed_payload.get("aliases") or []),
                    )
                    seed_payload = merged_seed_payload
            entry = self._build_resolution_entry(
                entity_type=str(chosen_seed_hit["entity_type"] or selected_entity_type),
                payload=seed_payload,
                mention=mention,
                resolver_source="llm_selected" if alias_llm_selection is not None else "seed",
                resolution_confidence=self._selection_confidence(alias_llm_selection, fallback=0.96),
                status="resolved",
            )
            entry = resolution_index.register(entry, mention=mention)
            ambiguity_flag = None
            if len({hit["entity_type"] for hit in alias_hits}) > 1:
                ambiguity_flag = AmbiguityFlag(
                    flag_type="entity_ambiguous",
                    target=mention,
                    note=f"Mention matched multiple entity types; resolved to {entry.entity_type}.",
                    severity="low",
                )
            return (
                entry.to_record(
                    mention=mention,
                    source_text=typed_mention["source_text"],
                    source_span=source_span,
                    entity_id=self._make_entity_id(entry.entity_type, mention, source_span),
                ),
                None,
                ambiguity_flag,
            )

        pubchem_entity_type = self._select_pubchem_entity_type(selected_entity_type)
        if pubchem_entity_type is not None:
            pubchem_candidates = self._lookup_pubchem_candidates(
                mention=mention,
                provider_calls=provider_calls,
            )
            selected_candidate, ambiguity_note, selected_via_llm, selected_confidence = self._select_pubchem_candidate(
                question=question,
                task_spec=task_spec,
                typed_mention=typed_mention,
                entity_type=pubchem_entity_type,
                candidates=pubchem_candidates,
            )
            if selected_candidate is not None:
                entry = self._build_resolution_entry(
                    entity_type=pubchem_entity_type,
                    payload=selected_candidate,
                    mention=mention,
                    resolver_source="llm_selected" if selected_via_llm else "pubchem",
                    resolution_confidence=selected_confidence,
                    status="resolved",
                )
                entry = resolution_index.register(entry, mention=mention)
                return (
                    entry.to_record(
                        mention=mention,
                        source_text=typed_mention["source_text"],
                        source_span=source_span,
                        entity_id=self._make_entity_id(entry.entity_type, mention, source_span),
                    ),
                    None,
                    None,
                )
            if ambiguity_note:
                unresolved_mention = UnresolvedMention(
                    mention=mention,
                    candidate_entity_types=candidate_entity_types,
                    reason=ambiguity_note,
                    confidence=0.3,
                    source_text=typed_mention["source_text"],
                    source_span=source_span,
                )
                ambiguity_flag = AmbiguityFlag(
                    flag_type="entity_ambiguous",
                    target=mention,
                    note=ambiguity_note,
                    severity="medium",
                )
                self._note_seed_suggestion(
                    tracker=seed_suggestion_tracker,
                    mention=mention,
                    entity_type=pubchem_entity_type,
                    proposed_canonical_name=mention,
                    aliases_seen=[mention],
                    resolver_source="pubchem",
                    reason="ambiguous_pubchem_match",
                    source_span=source_span,
                )
                return None, unresolved_mention, ambiguity_flag

        unresolved_mention = UnresolvedMention(
            mention=mention,
            candidate_entity_types=[selected_entity_type, *[item for item in candidate_entity_types if item != selected_entity_type]],
            reason=self._build_unresolved_reason([selected_entity_type]),
            confidence=mention_extraction_confidence,
            source_text=typed_mention["source_text"],
            source_span=source_span,
        )
        ambiguity_flag = AmbiguityFlag(
            flag_type="entity_ambiguous",
            target=mention,
            note=unresolved_mention.reason,
            severity="medium",
        )
        self._note_seed_suggestion(
            tracker=seed_suggestion_tracker,
            mention=mention,
            entity_type=selected_entity_type or "ambiguous",
            proposed_canonical_name=mention,
            aliases_seen=self._suggested_aliases(typed_mention),
            resolver_source="unresolved",
            reason="unresolved",
            source_span=source_span,
        )
        return None, unresolved_mention, ambiguity_flag

    def _lookup_runtime_entry(
        self,
        *,
        resolution_index: RunResolutionIndex,
        mention: str,
        candidate_entity_types: Sequence[str],
        task_spec: TaskSpec,
        type_hint: Optional[str],
    ) -> Optional[ResolutionEntry]:
        cached_entries = resolution_index.lookup_entries(mention)
        if not cached_entries:
            return None
        return self._select_resolution_entry(
            entries=cached_entries,
            candidate_entity_types=candidate_entity_types,
            task_spec=task_spec,
            type_hint=type_hint,
        )

    def _select_resolution_entry(
        self,
        *,
        entries: Sequence[ResolutionEntry],
        candidate_entity_types: Sequence[str],
        task_spec: TaskSpec,
        type_hint: Optional[str],
    ) -> Optional[ResolutionEntry]:
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0]
        preferred_types = set(candidate_entity_types)
        preferred_types.update(task_spec.query_constraints.preferred_entity_types)
        for axis in task_spec.required_condition_axes:
            if axis in ENTITY_AXIS_MAP.values():
                preferred_types.add(axis)
        if type_hint:
            preferred_types.add(type_hint)
        filtered = [entry for entry in entries if entry.entity_type in preferred_types]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            entries = filtered
        canonical_names = {entry.canonical_name for entry in entries}
        if len(canonical_names) == 1:
            return sorted(entries, key=lambda entry: ENTITY_TYPE_PRIORITY.get(entry.entity_type, 99))[0]
        return None

    def _select_seed_hit(
        self,
        *,
        alias_hits: Sequence[Dict[str, Any]],
        typed_mention: Dict[str, Any],
        task_spec: TaskSpec,
        llm_selection: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not alias_hits:
            return None
        if len(alias_hits) == 1:
            return alias_hits[0]
        selected_index = llm_selection.get("selected_index") if isinstance(llm_selection, dict) else None
        if isinstance(selected_index, int) and 0 <= selected_index < len(alias_hits):
            return alias_hits[selected_index]
        llm_entity_type = str(llm_selection.get("entity_type") or "").strip() if llm_selection else ""
        if llm_entity_type:
            filtered = [hit for hit in alias_hits if hit.get("entity_type") == llm_entity_type]
            if len(filtered) == 1:
                return filtered[0]
            if len(filtered) > 1:
                alias_hits = filtered
        preferred_types = {str(typed_mention.get("selected_entity_type") or "").strip()}
        preferred_types.update(task_spec.query_constraints.preferred_entity_types)
        preferred_types.update(typed_mention.get("candidate_entity_types") or [])
        filtered = [hit for hit in alias_hits if hit.get("entity_type") in preferred_types]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            alias_hits = filtered
        canonical_names = {str((hit.get("data") or {}).get("canonical_name") or "") for hit in alias_hits}
        if len(canonical_names) == 1:
            return sorted(alias_hits, key=lambda hit: ENTITY_TYPE_PRIORITY.get(str(hit.get("entity_type") or ""), 99))[0]
        return None

    def _resolve_llm_selection(
        self,
        *,
        question: str,
        task_spec: TaskSpec,
        typed_mention: Dict[str, Any],
        candidate_options: Sequence[Dict[str, Any]],
        allowed_entity_types: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if self.llm is None or not self.llm_disambiguation_enabled:
            return None
        if len(candidate_options) <= 1 and len(set(allowed_entity_types)) <= 1:
            return None
        mention_payload = {
            "mention": typed_mention.get("mention"),
            "source_text": typed_mention.get("source_text"),
            "selected_entity_type": typed_mention.get("selected_entity_type"),
            "candidate_entity_types": list(typed_mention.get("candidate_entity_types") or []),
            "candidate_options": list(candidate_options),
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

        entity_type = str(parsed.get("entity_type") or "").strip()
        if entity_type not in set(allowed_entity_types):
            entity_type = ""

        selected_index = parsed.get("selected_index")
        try:
            selected_index = int(selected_index) if selected_index is not None else None
        except (TypeError, ValueError):
            selected_index = None
        if selected_index is not None and not (0 <= selected_index < len(candidate_options)):
            selected_index = None
        if entity_type == "" and selected_index is not None:
            selected_option = candidate_options[selected_index]
            option_entity_type = str(selected_option.get("entity_type") or "").strip()
            if option_entity_type in set(allowed_entity_types):
                entity_type = option_entity_type

        try:
            confidence = float(parsed.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        if confidence < self.disambiguation_min_confidence:
            return None
        if not entity_type and selected_index is None:
            return None
        return {
            "selected_index": selected_index,
            "entity_type": entity_type or None,
            "confidence": confidence,
            "rationale": str(parsed.get("rationale") or "").strip(),
        }

    def _select_pubchem_entity_type(self, entity_type: str) -> Optional[str]:
        normalized = str(entity_type or "").strip()
        return normalized if normalized in self.pubchem_entity_types else None

    def _lookup_pubchem_candidates(
        self,
        *,
        mention: str,
        provider_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.pubchem_enabled or self.pubchem_client is None:
            return []
        payload = {
            "provider": "pubchem",
            "query": mention,
            "max_candidates": self.max_pubchem_candidates,
            "status": "empty",
            "candidate_count": 0,
        }
        try:
            search_method = getattr(self.pubchem_client, "search_candidates", None)
            if callable(search_method):
                candidates = search_method(mention, self.max_pubchem_candidates)
            else:
                lookup_method = getattr(self.pubchem_client, "lookup_compound", None)
                if callable(lookup_method):
                    single = lookup_method(mention)
                    candidates = [single] if single is not None else []
                elif callable(self.pubchem_client):
                    single = self.pubchem_client(mention)
                    candidates = [single] if single is not None else []
                else:
                    candidates = []
            candidates = [dict(item) for item in list(candidates or []) if isinstance(item, dict)]
            payload["candidate_count"] = len(candidates)
            payload["status"] = "hit" if candidates else "empty"
            provider_calls.append(payload)
            return candidates
        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = str(exc)
            provider_calls.append(payload)
            if self.fail_open_on_provider_error:
                return []
            raise

    def _select_pubchem_candidate(
        self,
        *,
        question: str,
        task_spec: TaskSpec,
        typed_mention: Dict[str, Any],
        entity_type: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], bool, float]:
        if not candidates:
            return None, None, False, 0.0
        exact_matches = self._exact_pubchem_matches(str(typed_mention.get("mention") or ""), candidates)
        if len(exact_matches) == 1:
            return exact_matches[0], None, False, 0.94
        if len(candidates) == 1:
            return dict(candidates[0]), None, False, 0.9
        llm_selection = self._resolve_llm_selection(
            question=question,
            task_spec=task_spec,
            typed_mention=typed_mention,
            candidate_options=[
                {
                    "index": index,
                    "entity_type": entity_type,
                    "resolver_source": "pubchem",
                    "canonical_name": str(candidate.get("canonical_name") or ""),
                    "aliases": list(candidate.get("aliases") or []),
                    "formula": candidate.get("formula"),
                    "smiles": candidate.get("smiles"),
                    "inchikey": candidate.get("inchikey"),
                    "pubchem_cid": candidate.get("pubchem_cid"),
                }
                for index, candidate in enumerate(candidates)
            ],
            allowed_entity_types=[entity_type],
        )
        if llm_selection is None:
            return (
                None,
                "PubChem returned multiple candidates and LLM disambiguation did not reach the configured confidence threshold.",
                False,
                0.0,
            )
        selected_index = llm_selection.get("selected_index")
        if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
            return dict(candidates[selected_index]), None, True, self._selection_confidence(llm_selection, fallback=0.78)
        return (
            None,
            "PubChem returned multiple candidates and LLM disambiguation did not select a valid candidate.",
            False,
            0.0,
        )

    def _exact_pubchem_matches(self, mention: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_mention = self._normalize_lookup_text(mention)
        matches: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate_names = [
                str(candidate.get("canonical_name") or ""),
                str(candidate.get("formula") or ""),
                *(str(value) for value in list(candidate.get("aliases") or [])),
            ]
            normalized_names = {self._normalize_lookup_text(value) for value in candidate_names if str(value).strip()}
            if normalized_mention and normalized_mention in normalized_names:
                matches.append(dict(candidate))
        return matches

    def _build_resolution_entry(
        self,
        *,
        entity_type: str,
        payload: Dict[str, Any],
        mention: str,
        resolver_source: str,
        resolution_confidence: float,
        status: str,
    ) -> ResolutionEntry:
        smiles = self._canonicalize_smiles(payload.get("smiles"))
        canonical_name = str(payload.get("canonical_name") or mention).strip() or mention
        aliases = self._merge_unique_text([mention], list(payload.get("aliases") or []))
        query_anchors = self._build_query_anchors(mention, payload)
        return ResolutionEntry(
            entry_id="",
            entity_type=entity_type,
            canonical_name=canonical_name,
            entity_subtype=payload.get("entity_subtype"),
            formula=payload.get("formula"),
            smiles=smiles,
            inchikey=payload.get("inchikey"),
            pubchem_cid=payload.get("pubchem_cid"),
            aliases=aliases,
            query_anchors=query_anchors,
            resolver_source=resolver_source,
            resolution_confidence=max(0.0, min(float(resolution_confidence), 1.0)),
            status=status,
        )

    def _selection_confidence(self, selection: Optional[Dict[str, Any]], *, fallback: float) -> float:
        if not isinstance(selection, dict):
            return fallback
        try:
            confidence = float(selection.get("confidence"))
        except (TypeError, ValueError):
            return fallback
        return max(0.0, min(confidence, 1.0))

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
            text = str(candidate or "").strip()
            if text and text not in anchors:
                anchors.append(text)
        return anchors

    def _build_unresolved_reason(self, candidate_entity_types: Sequence[str]) -> str:
        if not candidate_entity_types:
            return "No supported entity type could be assigned from the mention extraction stage."
        if len(candidate_entity_types) > 1:
            return f"Mention could map to multiple entity types: {', '.join(candidate_entity_types)}."
        entity_type = candidate_entity_types[0]
        if entity_type in self.pubchem_entity_types:
            return "Normalization did not resolve the mention with the seed layer or PubChem."
        return f"Mention was typed as {entity_type} but could not be fully normalized."

    def _build_extraction_unresolved_reason(
        self,
        *,
        selected_entity_type: str,
        candidate_entity_types: Sequence[str],
        confidence: float,
    ) -> str:
        if not selected_entity_type:
            if candidate_entity_types:
                return (
                    "Mention extraction did not produce a unique selected_entity_type from: "
                    f"{', '.join(candidate_entity_types)}."
                )
            return "Mention extraction did not assign a supported entity type."
        return (
            f"Mention extraction selected {selected_entity_type} with confidence "
            f"{confidence:.2f}, below the configured threshold."
        )

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
                re.compile("(?P<op><=|>=|<|>|\\u2248|~)?\\s*(?P<value>-?\\d+(?:\\.\\d+)?)\\s*(?P<unit>\\u00B0\\s*C|\\u2103|K)\\b", re.I),
                "temperature",
                lambda unit: "C" if unit and "C" in str(unit).upper() else unit,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|~)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hrs|hour|hours|min|minute|minutes|s|sec|second|seconds)\b", re.I),
                "time",
                lambda unit: str(unit).lower() if unit else unit,
            ),
            (
                re.compile(r"\bpH\s*(?P<op><=|>=|<|>|=)?\s*(?P<value>\d+(?:\.\d+)?)\b", re.I),
                "ph",
                lambda unit: None,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|~)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>V|mV)\b(?:\s*(?:vs\.?|versus)\s*(?P<ref>[A-Za-z0-9/+\-.]+))?", re.I),
                "potential",
                lambda unit: unit,
            ),
            (
                re.compile(r"(?P<op><=|>=|<|>|~)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bar|atm|kPa|MPa|Pa)\b", re.I),
                "pressure",
                lambda unit: unit,
            ),
            (
                re.compile(r"\b(?:yield)\s*(?:of|=|at)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\b", re.I),
                "yield",
                lambda unit: "%" if unit and str(unit).lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\s*(?:yield)\b", re.I),
                "yield",
                lambda unit: "%" if unit and str(unit).lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?:selectivity|Faradaic efficiency|FE)\s*(?:of|=|at)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\b", re.I),
                "selectivity",
                lambda unit: "%" if unit and str(unit).lower() == "percent" else unit,
            ),
            (
                re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\s*(?:selectivity|Faradaic efficiency|FE)\b", re.I),
                "selectivity",
                lambda unit: "%" if unit and str(unit).lower() == "percent" else unit,
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
        merged: List[EntityRecord] = []
        seen: Dict[Tuple[str, str, str], int] = {}
        for entity in entities:
            key = self._entity_identity_key(entity)
            existing_index = seen.get(key)
            if existing_index is None:
                seen[key] = len(merged)
                merged.append(entity)
                continue
            current = merged[existing_index]
            merged[existing_index] = current.model_copy(
                update={
                    "aliases": self._merge_unique_text(current.aliases, entity.aliases),
                    "query_anchors": self._merge_unique_text(current.query_anchors, entity.query_anchors),
                    "resolution_confidence": max(current.resolution_confidence, entity.resolution_confidence),
                    "status": "resolved" if "resolved" in {current.status, entity.status} else current.status,
                }
            )
        return merged

    def _entity_identity_key(self, entity: EntityRecord) -> Tuple[str, str, str]:
        if entity.inchikey:
            return ("inchikey", entity.entity_type, str(entity.inchikey).upper())
        if entity.pubchem_cid is not None:
            return ("pubchem_cid", entity.entity_type, str(entity.pubchem_cid))
        return ("canonical_name", entity.entity_type, self._normalize_lookup_text(entity.canonical_name))

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

    def _note_seed_suggestion(
        self,
        *,
        tracker: Dict[Tuple[str, str, str], Dict[str, Any]],
        mention: str,
        entity_type: str,
        proposed_canonical_name: str,
        aliases_seen: Sequence[str],
        resolver_source: str,
        reason: str,
        source_span: SourceSpan,
    ) -> None:
        if not self.emit_seed_suggestions:
            return
        key = (
            str(entity_type or "").strip() or "unknown",
            self._normalize_lookup_text(proposed_canonical_name or mention),
            str(reason or "").strip() or "unspecified",
        )
        current = tracker.setdefault(
            key,
            {
                "mention": mention,
                "entity_type": str(entity_type or "").strip() or "unknown",
                "proposed_canonical_name": str(proposed_canonical_name or mention).strip() or mention,
                "aliases_seen": [],
                "resolver_source": resolver_source,
                "occurrence_count": 0,
                "reason": reason,
                "source_spans": [],
            },
        )
        current["occurrence_count"] += 1
        current["aliases_seen"] = self._merge_unique_text(current["aliases_seen"], list(aliases_seen or [mention]))
        current["source_spans"].append({"start": source_span.start, "end": source_span.end})

    def _finalize_seed_suggestions(
        self,
        tracker: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        suggestions = list(tracker.values())
        suggestions.sort(
            key=lambda item: (
                -int(item.get("occurrence_count") or 0),
                str(item.get("entity_type") or ""),
                str(item.get("proposed_canonical_name") or ""),
            )
        )
        return suggestions

    def _suggested_aliases(self, typed_mention: Dict[str, Any]) -> List[str]:
        aliases = [str(typed_mention.get("mention") or "").strip()]
        for hit in list(typed_mention.get("alias_hits") or []):
            aliases.extend(str(value).strip() for value in list((hit.get("data") or {}).get("aliases") or []))
        return self._merge_unique_text([], aliases)

    def _make_entity_id(self, entity_type: str, mention: str, source_span: SourceSpan) -> str:
        normalized_mention = re.sub(r"[^A-Za-z0-9]+", "_", mention).strip("_").lower() or "mention"
        return f"{entity_type}_{normalized_mention}_{source_span.start}_{source_span.end}"

    def _span_overlaps(self, start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
        for existing_start, existing_end in spans:
            if start < existing_end and end > existing_start:
                return True
        return False

    @staticmethod
    def _normalize_lookup_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    @staticmethod
    def _merge_unique_text(existing: Sequence[str], extra: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in [*(existing or []), *(extra or [])]:
            text = str(value or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged
