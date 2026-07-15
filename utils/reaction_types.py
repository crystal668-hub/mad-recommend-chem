from __future__ import annotations

import re
from typing import List, Optional


REACTION_TYPE_LABELS: List[str] = [
    "CO2RR",
    "EOR",
    "HER",
    "HOR",
    "HZOR",
    "O5H",
    "OER",
    "ORR",
    "UOR",
]

CATEGORY_TYPE_LABELS: List[str] = [
    "Antibacterial",
    "Thermoelectric",
    "antiferromagnetism",
    "conductivity",
    "ferrimagnetism",
    "ferromagnetism",
    "hydrogenation of furfural",
    "photocatalytic H2O2 production",
    "photothermal conversion efficiency",
    "thermal conductivity",
]

SUPPORTED_REACTION_TYPE_LABELS: List[str] = REACTION_TYPE_LABELS + CATEGORY_TYPE_LABELS


def _reaction_type_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").strip().lower())


_CANONICAL_BY_KEY = {
    _reaction_type_key(label): label
    for label in SUPPORTED_REACTION_TYPE_LABELS
}


def canonical_reaction_type(value: object) -> Optional[str]:
    """
    Normalize old reaction acronyms and new literature category labels.

    Returns labels exactly as stored in Chroma metadata["reaction_type"].
    Unknown non-empty values are stripped and returned unchanged.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    canonical = _CANONICAL_BY_KEY.get(_reaction_type_key(raw))
    if canonical:
        return canonical
    return raw


def reaction_type_matches(value: object, target: object) -> bool:
    left = canonical_reaction_type(value)
    right = canonical_reaction_type(target)
    if not left or not right:
        return False
    return _reaction_type_key(left) == _reaction_type_key(right)


def is_supported_reaction_type(value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    return _reaction_type_key(raw) in _CANONICAL_BY_KEY
