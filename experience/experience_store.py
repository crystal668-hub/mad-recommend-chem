"""
Experience store.

This project supports two types of "experience":
1) Dynamic experiences saved as JSON (append-only) from debate runs.
2) Read-only "experience packs" stored as YAML files (one file or a directory of files).

YAML pack formats supported:
- Explicit list: {"experiences": [ {...}, {...} ]}
- Top-level list: [ {...}, {...} ]
- Hydra-style agent config: {"agent": {"instructions": "...[G0]. ..."}}
  In this case we extract guideline blocks like:
    [G0]. Title: content...
    [G1]. ...
  and expose them as global experiences (no components).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "must",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "our",
    "should",
    "so",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "those",
    "to",
    "use",
    "using",
    "we",
    "when",
    "will",
    "with",
    "without",
    "you",
    "your",
}


def _normalize_component(value: str) -> str:
    return str(value or "").strip().lower()


def _ensure_list_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    # Accept "Pt, Pd" strings as a convenience.
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[,，;；]", value) if p.strip()]
        return parts
    return [str(value).strip()] if str(value).strip() else []


class ExperienceStore:
    """
    Experience store with optional YAML pack loading.

    Args:
        storage_path:
            - ".json" file path: dynamic store (read/write)
            - ".yaml"/".yml" file path OR directory: treated as a read-only pack root
        packs_path:
            Optional extra pack root (file or directory). If omitted, defaults to the
            built-in `./experience` directory (the folder containing this file).
    """

    def __init__(
        self,
        storage_path: str,
        max_experiences: int = 1000,
        relevance_threshold: float = 0.8,
        packs_path: Optional[str] = None,
        guideline_top_k: int = 3,
        always_include_guidelines: bool = True,
        guideline_search_mode: str = "keyword",
        load_builtin_packs: bool = True,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.max_experiences = int(max_experiences)
        self.relevance_threshold = float(relevance_threshold)
        self.guideline_top_k = max(0, int(guideline_top_k))
        self.always_include_guidelines = bool(always_include_guidelines)

        gsm = str(guideline_search_mode or "").strip().lower() or "rank"
        self.guideline_search_mode = gsm if gsm in {"keyword", "rank"} else "rank"

        self._json_path: Optional[Path] = None
        if self.storage_path.suffix.lower() == ".json":
            self._json_path = self.storage_path

        # Pack roots:
        # - If storage_path is YAML or a directory, treat it as a pack root.
        # - Always load the built-in packs (folder containing this file).
        # - Allow the caller to specify an extra packs_path.
        pack_roots: List[Path] = []
        if self.storage_path.is_dir() or self.storage_path.suffix.lower() in {".yaml", ".yml"}:
            pack_roots.append(self.storage_path)

        if bool(load_builtin_packs):
            builtin_packs = Path(__file__).resolve().parent
            pack_roots.append(builtin_packs)

        if packs_path:
            pack_roots.append(Path(packs_path))

        self._pack_roots = self._dedupe_paths(pack_roots)

        self._dynamic_experiences: List[Dict[str, Any]] = self._load_dynamic_experiences()
        self._pack_experiences: List[Dict[str, Any]] = self._load_pack_experiences(self._pack_roots)

        # Public view (for compatibility with older code/docs).
        self.experiences: List[Dict[str, Any]] = self._pack_experiences + self._dynamic_experiences

        # Guideline keyword index (for `guideline_search_mode="keyword"`).
        self._guideline_tokens_by_id: Dict[str, set[str]] = {}
        self._guideline_idf: Dict[str, float] = {}
        self._build_guideline_index()

    # -------------------------
    # Loading
    # -------------------------

    @staticmethod
    def _dedupe_paths(paths: List[Path]) -> List[Path]:
        seen: set[str] = set()
        out: List[Path] = []
        for p in paths:
            try:
                key = str(p.resolve())
            except Exception:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _load_dynamic_experiences(self) -> List[Dict[str, Any]]:
        if self._json_path is None:
            return []
        if not self._json_path.exists():
            return []

        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Mark as writable entries.
                return [self._coerce_dynamic_entry(x) for x in data if isinstance(x, dict)]
            return []
        except Exception:
            # Corrupt or unreadable store: treat as empty (do not crash the runtime).
            return []

    def _coerce_dynamic_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(entry)
        out.setdefault("_readonly", False)
        out["components"] = _ensure_list_str(out.get("components"))
        return out

    def _load_pack_experiences(self, pack_roots: List[Path]) -> List[Dict[str, Any]]:
        experiences: List[Dict[str, Any]] = []
        for root in pack_roots:
            for yaml_file in self._iter_yaml_files(root):
                experiences.extend(self._load_yaml_pack_file(yaml_file))
        return experiences

    @staticmethod
    def _iter_yaml_files(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        if root.is_file():
            if root.suffix.lower() in {".yaml", ".yml"}:
                return [root]
            return []

        files: List[Path] = []
        try:
            files.extend(sorted(root.glob("*.yaml")))
            files.extend(sorted(root.glob("*.yml")))
        except Exception:
            return []
        return files

    def _load_yaml_pack_file(self, path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return []

        items: List[Dict[str, Any]] = []

        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            if isinstance(data.get("experiences"), list):
                items = [x for x in data.get("experiences", []) if isinstance(x, dict)]
            else:
                # Hydra-like pack: parse agent.instructions guidelines into experiences.
                agent = data.get("agent") if isinstance(data.get("agent"), dict) else None
                instructions = agent.get("instructions") if isinstance(agent, dict) else None
                if isinstance(instructions, str) and instructions.strip():
                    return self._extract_guideline_experiences(instructions, source_file=path)
                return []
        else:
            return []

        return [self._coerce_pack_entry(x, source_file=path, idx=i) for i, x in enumerate(items)]

    def _coerce_pack_entry(self, entry: Dict[str, Any], source_file: Path, idx: int) -> Dict[str, Any]:
        out = dict(entry)
        out.setdefault("_readonly", True)
        out.setdefault("kind", out.get("kind") or "case")
        out.setdefault("source_file", str(source_file))
        out.setdefault("source_pack", source_file.name)
        out.setdefault("id", f"pack:{source_file.name}:{idx}")
        out["components"] = _ensure_list_str(out.get("components"))
        return out

    def _extract_guideline_experiences(self, instructions: str, source_file: Path) -> List[Dict[str, Any]]:
        guidelines = self._parse_guidelines(instructions)
        experiences: List[Dict[str, Any]] = []
        for rank, (gid, text) in enumerate(guidelines, 1):
            title, content = self._split_title_content(text)
            experiences.append(
                {
                    "id": f"pack:{source_file.name}:{gid}",
                    "kind": "guideline",
                    "_readonly": True,
                    "source_file": str(source_file),
                    "source_pack": source_file.name,
                    "guideline_id": gid,
                    "rank": rank,
                    "title": title,
                    "content": content,
                    "components": [],
                    "reaction_type": "GLOBAL",
                }
            )
        return experiences

    @staticmethod
    def _parse_guidelines(instructions: str) -> List[Tuple[str, str]]:
        """
        Extract blocks that start with "[G0].", "[G1].", ... from an instructions string.
        Returns: [(guideline_id, text), ...]
        """
        lines = instructions.splitlines()
        current_id: Optional[str] = None
        current_text: List[str] = []
        out: List[Tuple[str, str]] = []

        pat = re.compile(r"^\s*\[(G\d+)\]\.\s*(.*)\s*$")
        for line in lines:
            m = pat.match(line)
            if m:
                if current_id is not None:
                    out.append((current_id, " ".join(current_text).strip()))
                current_id = m.group(1)
                current_text = [m.group(2).strip()]
                continue

            if current_id is None:
                continue

            cont = line.strip()
            if cont:
                current_text.append(cont)

        if current_id is not None:
            out.append((current_id, " ".join(current_text).strip()))

        return out

    @staticmethod
    def _split_title_content(text: str) -> Tuple[str, str]:
        if not text:
            return "Guideline", ""
        if ":" in text:
            title, rest = text.split(":", 1)
            title = title.strip() or "Guideline"
            rest = rest.strip()
            return title, rest
        return "Guideline", text.strip()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """
        Lightweight tokenizer for guideline keyword matching.

        Notes:
        - ASCII-focused: keeps a-z/0-9 tokens (e.g., "co2rr", "fe", "j0").
        - Drops stopwords and purely-numeric tokens to reduce noise.
        """
        s = str(text or "").lower()
        raw_tokens = [t for t in _TOKEN_SPLIT_RE.split(s) if t]
        out: set[str] = set()
        for tok in raw_tokens:
            if tok in _TOKEN_STOPWORDS:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 2:
                continue
            out.add(tok)
        return out

    def _build_guideline_index(self) -> None:
        """
        Precompute per-guideline token sets and corpus IDF weights.

        This enables `guideline_search_mode="keyword"` to rank [G*] blocks
        by overlap with the current task/query text.
        """
        guidelines = [e for e in (self.experiences or []) if (e.get("kind") == "guideline")]
        tokens_by_id: Dict[str, set[str]] = {}
        df: Dict[str, int] = {}

        for g in guidelines:
            gid = str(g.get("id") or "").strip()
            if not gid:
                continue
            blob = " ".join(
                [
                    str(g.get("guideline_id") or ""),
                    str(g.get("title") or ""),
                    str(g.get("content") or ""),
                ]
            )
            toks = self._tokenize(blob)
            tokens_by_id[gid] = toks
            for t in toks:
                df[t] = int(df.get(t, 0)) + 1

        n_docs = len(tokens_by_id)
        idf: Dict[str, float] = {}
        if n_docs:
            for t, d in df.items():
                try:
                    idf[t] = 1.0 + math.log((float(n_docs) + 1.0) / (float(d) + 1.0))
                except Exception:
                    idf[t] = 1.0

        self._guideline_tokens_by_id = tokens_by_id
        self._guideline_idf = idf

    def _rank_guidelines(self, guidelines: List[Dict[str, Any]], query_text: Optional[str]) -> List[Dict[str, Any]]:
        if not guidelines:
            return []

        # Default: stable rank order (G0, G1, ...)
        def _rank_key(x: Dict[str, Any]) -> int:
            try:
                return int(x.get("rank", 10_000))
            except Exception:
                return 10_000

        if self.guideline_search_mode != "keyword" or not (query_text or "").strip():
            out = [dict(g) for g in guidelines]
            out.sort(key=_rank_key)
            return out

        q_tokens = self._tokenize(query_text or "")
        if not q_tokens:
            out = [dict(g) for g in guidelines]
            out.sort(key=_rank_key)
            return out

        max_score = sum(float(self._guideline_idf.get(t, 1.0)) for t in q_tokens)
        if max_score <= 0:
            out = [dict(g) for g in guidelines]
            out.sort(key=_rank_key)
            return out

        scored: List[Dict[str, Any]] = []
        for g in guidelines:
            gid = str(g.get("id") or "").strip()
            g_tokens = self._guideline_tokens_by_id.get(gid)
            if g_tokens is None:
                blob = " ".join(
                    [
                        str(g.get("guideline_id") or ""),
                        str(g.get("title") or ""),
                        str(g.get("content") or ""),
                    ]
                )
                g_tokens = self._tokenize(blob)

            overlap = q_tokens.intersection(g_tokens)
            score = sum(float(self._guideline_idf.get(t, 1.0)) for t in overlap)
            sim = float(score) / float(max_score) if max_score else 0.0
            g_out = dict(g)
            g_out["similarity"] = sim
            scored.append(g_out)

        scored.sort(key=lambda x: (-float(x.get("similarity", 0.0)), _rank_key(x)))
        return scored

    # -------------------------
    # CRUD (dynamic JSON only)
    # -------------------------

    def _save_dynamic(self) -> None:
        if self._json_path is None:
            return
        try:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(self._dynamic_experiences, f, ensure_ascii=False, indent=2)
        except Exception:
            # Best-effort: do not crash.
            return

    def add_experience(self, experience: Dict[str, Any]) -> None:
        """Append an experience to the dynamic JSON store (if configured)."""
        exp = dict(experience or {})
        exp.setdefault("_readonly", False)
        exp.setdefault("id", self._generate_experience_id())
        exp.setdefault("timestamp", datetime.now().isoformat())
        exp["components"] = _ensure_list_str(exp.get("components"))

        self._dynamic_experiences.append(exp)
        if len(self._dynamic_experiences) > self.max_experiences:
            self._dynamic_experiences = self._dynamic_experiences[-self.max_experiences :]

        self._save_dynamic()
        self.experiences = self._pack_experiences + self._dynamic_experiences

    @staticmethod
    def _generate_experience_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"exp_{timestamp}"

    # -------------------------
    # Query
    # -------------------------

    def query_experiences(
        self,
        components: List[str],
        top_k: int = 3,
        min_similarity: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query experiences by component overlap (Jaccard similarity).

        Behavior:
        - Case experiences: matched by component overlap (above threshold).
        - Guideline experiences: global [G*] blocks, optionally ranked by keyword overlap with `query_text`.
        - If `always_include_guidelines` is enabled, we always mix in up to `guideline_top_k`
          guidelines (bounded by `top_k`) before adding case experiences.
        """
        comps_norm = [_normalize_component(c) for c in (components or []) if _normalize_component(c)]
        top_k = max(1, int(top_k))

        if not self.experiences:
            return []

        threshold = float(min_similarity) if min_similarity is not None else self.relevance_threshold

        matched_cases: List[Dict[str, Any]] = []
        scored_cases_all: List[Dict[str, Any]] = []
        global_guides: List[Dict[str, Any]] = []

        for exp in self.experiences:
            if exp.get("kind") == "guideline":
                global_guides.append(dict(exp))
                continue

            exp_components = _ensure_list_str(exp.get("components"))
            if not exp_components:
                continue

            # If the model didn't provide query components, avoid returning arbitrary
            # component-specific cases; fall back to global guideline experiences.
            if not comps_norm:
                continue

            similarity = self._calculate_similarity(comps_norm, exp_components)
            exp_with_score = dict(exp)
            exp_with_score["similarity"] = similarity
            scored_cases_all.append(exp_with_score)
            if similarity >= threshold:
                matched_cases.append(exp_with_score)

        matched_cases.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
        ranked_guides = self._rank_guidelines(global_guides, query_text=query_text)

        # Strategy: optionally enforce a guideline prefix.
        if self.always_include_guidelines and ranked_guides:
            guide_k = min(int(self.guideline_top_k), int(top_k))
            guide_k = max(0, guide_k)
            case_k = max(0, int(top_k) - guide_k)

            results: List[Dict[str, Any]] = []
            results.extend(ranked_guides[:guide_k])
            results.extend(matched_cases[:case_k])

            # Fill remainder: prefer more guidelines, then more matched cases.
            if len(results) < top_k:
                for g in ranked_guides[guide_k:]:
                    if len(results) >= top_k:
                        break
                    results.append(g)
                for c in matched_cases[case_k:]:
                    if len(results) >= top_k:
                        break
                    results.append(c)

            # Last-resort: if we still haven't filled and we have scored cases (below threshold),
            # use the best ones to fill remaining slots.
            if len(results) < top_k and comps_norm and scored_cases_all:
                scored_cases_all.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
                seen_ids = {str(r.get("id") or "") for r in results}
                for c in scored_cases_all:
                    if len(results) >= top_k:
                        break
                    cid = str(c.get("id") or "")
                    if cid and cid in seen_ids:
                        continue
                    results.append(c)
                    if cid:
                        seen_ids.add(cid)

            return results[:top_k]

        # Legacy behavior: matched cases first, then guidelines as fallback.
        results = list(matched_cases[:top_k])
        if len(results) < top_k:
            results.extend(ranked_guides[: (top_k - len(results))])

        # Last-resort fallback: if nothing passed threshold but we have scored candidates,
        # return the best matches anyway.
        if not results and comps_norm and scored_cases_all:
            scored_cases_all.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
            return scored_cases_all[:top_k]

        return results[:top_k]

    @staticmethod
    def _calculate_similarity(components1: List[str], components2: List[str]) -> float:
        set1 = set(_normalize_component(c) for c in (components1 or []) if _normalize_component(c))
        set2 = set(_normalize_component(c) for c in (components2 or []) if _normalize_component(c))

        if not set1 and not set2:
            return 1.0

        union = set1 | set2
        if not union:
            return 0.0

        intersection = set1 & set2
        return len(intersection) / len(union)

    # -------------------------
    # Compatibility helpers
    # -------------------------

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self.experiences or [])
        reaction_counts: Dict[str, int] = {}
        for exp in self.experiences or []:
            rt = exp.get("reaction_type") or "unknown"
            reaction_counts[rt] = reaction_counts.get(rt, 0) + 1
        return {"total_experiences": total, "reaction_type_distribution": reaction_counts}

    def search_by_reaction_type(self, reaction_type: str) -> List[Dict[str, Any]]:
        return [e for e in (self.experiences or []) if (e.get("reaction_type") == reaction_type)]

    def delete_experience(self, experience_id: str) -> bool:
        """Delete from the dynamic store only."""
        before = len(self._dynamic_experiences)
        self._dynamic_experiences = [e for e in self._dynamic_experiences if e.get("id") != experience_id]
        if len(self._dynamic_experiences) == before:
            return False
        self._save_dynamic()
        self.experiences = self._pack_experiences + self._dynamic_experiences
        return True

    def clear_all(self) -> None:
        """Clear the dynamic store only."""
        self._dynamic_experiences = []
        self._save_dynamic()
        self.experiences = list(self._pack_experiences)

    def export_to_file(self, export_path: str) -> None:
        """Export the full experience view (packs + dynamic) as JSON."""
        try:
            out_path = Path(export_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.experiences, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def import_from_file(self, import_path: str, merge: bool = True) -> None:
        """Import into the dynamic store only."""
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                imported = json.load(f)
            if not isinstance(imported, list):
                return
            imported_dicts = [self._coerce_dynamic_entry(x) for x in imported if isinstance(x, dict)]
            if merge:
                self._dynamic_experiences.extend(imported_dicts)
            else:
                self._dynamic_experiences = imported_dicts
            if len(self._dynamic_experiences) > self.max_experiences:
                self._dynamic_experiences = self._dynamic_experiences[-self.max_experiences :]
            self._save_dynamic()
            self.experiences = self._pack_experiences + self._dynamic_experiences
        except Exception:
            return
