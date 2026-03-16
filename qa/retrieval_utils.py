from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, List, Optional


DOI_PREFIX_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
WHITESPACE_PATTERN = re.compile(r"\s+")
TITLE_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")
PLACEHOLDER_TEXT_PATTERNS = (
    re.compile(r"^\s*redirecting\s*$", re.I),
    re.compile(r"^\s*loading\s*$", re.I),
    re.compile(r"\benable javascript\b", re.I),
    re.compile(r"\baccess denied\b", re.I),
    re.compile(r"\btemporarily unavailable\b", re.I),
)
GARBLED_TEXT_MARKERS = ("JFIF", "ICC_PROFILE", "8BIM")
TEXTUAL_CONTENT_TYPE_PREFIXES = ("text/",)
TEXTUAL_CONTENT_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
}
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "application/octet-stream": "bin",
}


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = DOI_PREFIX_PATTERN.sub("", str(value).strip())
    cleaned = cleaned.strip().rstrip(".;,)")
    return cleaned.lower() or None


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(value)).strip()


def normalize_title(value: Optional[str]) -> str:
    normalized = normalize_text(value).lower()
    normalized = TITLE_NORMALIZE_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def stable_paper_id(doi: Optional[str], title: str, year: Optional[int] = None) -> str:
    normalized_doi = normalize_doi(doi)
    token = normalized_doi or f"{normalize_title(title)}|{year or ''}"
    prefix = "doi" if normalized_doi else "title"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def title_similarity(left: Optional[str], right: Optional[str]) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(a=left_norm, b=right_norm).ratio()


def year_in_window(year: Optional[int], year_from: Optional[int], year_to: Optional[int]) -> bool:
    if year is None:
        return False
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True


def slugify(value: str, max_length: int = 80) -> str:
    normalized = normalize_title(value).replace(" ", "_")
    if not normalized:
        return "artifact"
    return normalized[:max_length].strip("_") or "artifact"


def is_textual_content_type(content_type: Optional[str]) -> bool:
    normalized = normalize_text(content_type).lower()
    if not normalized:
        return False
    if normalized in TEXTUAL_CONTENT_TYPES:
        return True
    return any(normalized.startswith(prefix) for prefix in TEXTUAL_CONTENT_TYPE_PREFIXES)


def guess_binary_extension(content_type: Optional[str]) -> str:
    normalized = normalize_text(content_type).lower()
    return CONTENT_TYPE_EXTENSIONS.get(normalized, "bin")


def printable_text_ratio(value: Optional[str]) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    return printable / float(len(text))


def looks_like_placeholder_text(value: Optional[str]) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    return any(pattern.search(text) for pattern in PLACEHOLDER_TEXT_PATTERNS)


def looks_like_garbled_text(value: Optional[str]) -> bool:
    text = str(value or "")
    if not text:
        return False
    if any(marker in text for marker in GARBLED_TEXT_MARKERS):
        return True
    if "\x00" in text:
        return True
    if printable_text_ratio(text) < 0.85:
        alpha_words = re.findall(r"[A-Za-z]{3,}", text)
        if len(text) > 200 and len(alpha_words) < 8:
            return True
    return False


def flatten_author_names(raw_authors: Any) -> List[str]:
    if raw_authors is None:
        return []
    if isinstance(raw_authors, list):
        names: List[str] = []
        for item in raw_authors:
            if isinstance(item, str):
                cleaned = normalize_text(item)
                if cleaned:
                    names.append(cleaned)
                continue
            if isinstance(item, dict):
                author = item.get("author") if isinstance(item.get("author"), dict) else item
                name = (
                    author.get("display_name")
                    or author.get("name")
                    or " ".join(
                        part for part in [author.get("given"), author.get("family")] if part
                    ).strip()
                )
                cleaned = normalize_text(name)
                if cleaned:
                    names.append(cleaned)
        return names
    if isinstance(raw_authors, str):
        cleaned = normalize_text(raw_authors)
        return [cleaned] if cleaned else []
    return []


def first_author_key(authors: Iterable[str]) -> str:
    first = next(iter(authors), "")
    if not first:
        return ""
    tokens = normalize_text(first).lower().split()
    return tokens[-1] if tokens else ""
