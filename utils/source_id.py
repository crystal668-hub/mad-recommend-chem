"""
Source ID utilities.

We use stable, verifiable identifiers for evidence citations returned from the local
Chroma-backed RAG database.

Canonical format (one chunk = one evidence item):
  rag:chroma/<collection>/doi:<doc_id>#chunk:<chunk_id>
Where:
  - collection: Chroma collection name
  - doc_id: DOI string stored in metadata (e.g. "10.1021/....") or "unknown"
  - chunk_id: integer chunk index within the document
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_CHROMA_SOURCE_ID_RE = re.compile(
    r"^rag:chroma/(?P<collection>[^/]+)/doi:(?P<doc_id>[^#]+)#chunk:(?P<chunk_id>\d+)$"
)


@dataclass(frozen=True)
class ChromaSourceRef:
    collection: str
    doc_id: str
    chunk_id: int

    def to_source_id(self) -> str:
        return build_chroma_source_id(self.collection, self.doc_id, self.chunk_id)


def build_chroma_source_id(collection: str, doc_id: str, chunk_id: int) -> str:
    collection = (collection or "unknown").strip()
    doc_id = (doc_id or "unknown").strip()
    return f"rag:chroma/{collection}/doi:{doc_id}#chunk:{int(chunk_id)}"


def parse_chroma_source_id(source_id: str) -> Optional[ChromaSourceRef]:
    if not source_id:
        return None
    match = _CHROMA_SOURCE_ID_RE.match(source_id.strip())
    if not match:
        return None
    return ChromaSourceRef(
        collection=match.group("collection"),
        doc_id=match.group("doc_id"),
        chunk_id=int(match.group("chunk_id")),
    )


def is_valid_chroma_source_id(source_id: str) -> bool:
    return parse_chroma_source_id(source_id) is not None


def normalize_doc_id(doc_id: str) -> str:
    """
    Best-effort normalization for `doc_id` used inside Chroma source_ids.

    Motivation:
    - Our DOI extraction can pick up Markdown emphasis wrappers (e.g. `**10.xxxx/...**`),
      leaving trailing `**` in metadata.doc_id. That cascades into source_id strings like:
        rag:chroma/<collection>/doi:10.xxxx/**...**#chunk:12
    - Agents often cite the DOI without the Markdown wrappers, causing evidence verification
      to fail due to a raw string mismatch.

    This function:
    - strips common Markdown wrappers from both ends: `*`, `_`, and backticks
    - strips common punctuation/brackets from both ends
    - normalizes DOI casing to lowercase (safe for our pipelines)
    """
    d = str(doc_id or "").strip()
    if not d:
        return d

    # Remove surrounding wrappers that frequently appear in Markdown/prose.
    d = d.strip().strip("<>").strip().strip("()[]{}")
    d = d.strip("*_`")
    d = d.strip().strip("*_`")

    # Strip common trailing punctuation artifacts.
    d = d.rstrip(".,;:!?\"'")
    d = d.strip()

    return d.lower()


def normalize_chroma_source_id(source_id: str) -> str:
    """
    Normalize a canonical `rag:chroma/...` source_id so logically equivalent citations match.

    Example:
      rag:chroma/c/doi:10.1002/adma.202109108**#chunk:23
      -> rag:chroma/c/doi:10.1002/adma.202109108#chunk:23
    """
    s = str(source_id or "").strip()
    if not s:
        return s

    # First try a direct parse (fast path).
    ref = parse_chroma_source_id(s)
    if ref is None:
        # Best-effort salvage for prose punctuation like "...#chunk:3)."
        # Keep conservative: only strip ASCII prose punctuation/brackets.
        s2 = s.strip("`\"'").rstrip(").,;:!?\"']}")
        ref = parse_chroma_source_id(s2)
        if ref is None:
            return s.strip()

    return build_chroma_source_id(
        collection=ref.collection,
        doc_id=normalize_doc_id(ref.doc_id),
        chunk_id=ref.chunk_id,
    )
