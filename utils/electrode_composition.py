"""
Electrode composition utilities.

This project historically treated `components` as a list of 5 metal element symbols, e.g.:
  ["Pt", "Pd", "Ru", "Ir", "Rh"]

We now also support an *electrode composition* specification where each metal has a
relative percentage, e.g.:
  ["Ni(69.00%)", "Co(19.07%)", "Fe(11.48%)", "Cu(0.40%)", "Zn(0.05%)"]

These helpers keep the rest of the system stable by:
- extracting element symbols for guards / experience retrieval / parsing
- formatting a stable "Electrode composition (relative %): ..." string for prompts
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import List, Optional, Tuple


_TOKEN_PAREN_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s*[\(\uff08]\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*[\)\uff09]\s*$"
)
_TOKEN_DIRECT_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*$")
_TOKEN_SYMBOL_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*$")


def parse_component_token(token: str) -> Tuple[str, Optional[float]]:
    """
    Parse a single component token.

    Accepted examples:
    - "Ni"
    - "Ni(69.00%)"
    - "Ni 69.00%"
    - "Ni（69.00%）"  (full-width parentheses)
    """
    s = str(token or "").strip()
    if not s:
        raise ValueError("Empty component token")

    m = _TOKEN_PAREN_RE.match(s)
    if m:
        return m.group(1), float(m.group(2))

    m = _TOKEN_DIRECT_RE.match(s)
    if m:
        return m.group(1), float(m.group(2))

    m = _TOKEN_SYMBOL_RE.match(s)
    if m:
        return m.group(1), None

    raise ValueError(f"Invalid component token: {token!r}. Expected like 'Ni' or 'Ni(69.00%)'.")


def parse_components_with_percent(components: List[str]) -> Tuple[List[str], Optional[List[float]]]:
    """
    Parse a list of component tokens into (element_symbols, percents|None).

    Rules:
    - If ANY token includes a percentage, then ALL tokens must include a percentage.
    - Percentages are returned as floats (raw values, not yet normalized to sum=100).
    """
    syms: List[str] = []
    pcts: List[Optional[float]] = []

    any_pct = False
    for t in (components or []):
        sym, pct = parse_component_token(str(t))
        syms.append(sym)
        pcts.append(pct)
        if pct is not None:
            any_pct = True

    if any_pct:
        if any(p is None for p in pcts):
            raise ValueError(
                "Mixed component formats: some metals include percentages and some do not. "
                "Provide either ALL as 'Ni(69.00%)' style, or NONE."
            )
        return syms, [float(p) for p in pcts if p is not None]

    return syms, None


def _allocate_units_by_weights(weights: List[float], total_units: int, min_units_each: int = 0) -> List[int]:
    """
    Allocate `total_units` integer units across N buckets proportional to `weights`,
    with an optional per-bucket minimum.

    Uses a largest-remainder method so the sum is EXACT and allocation is deterministic.
    """
    n = len(weights or [])
    if n <= 0:
        return []

    total_units = int(total_units)
    min_units_each = max(0, int(min_units_each))
    if total_units < n * min_units_each:
        raise ValueError("total_units is smaller than n * min_units_each")

    remaining = total_units - n * min_units_each

    w_pos = [max(0.0, float(w)) for w in (weights or [])]
    wsum = sum(w_pos)
    if wsum <= 0.0:
        w_pos = [1.0] * n
        wsum = float(n)

    raw = [(w / wsum) * remaining for w in w_pos]
    floors = [int(x) for x in raw]  # raw is non-negative; int() = floor
    fracs = [raw[i] - floors[i] for i in range(n)]
    rem = remaining - sum(floors)

    if rem > 0:
        # Tie-break by index to keep stable ordering across platforms/Python versions.
        idxs = sorted(range(n), key=lambda i: (fracs[i], -i), reverse=True)
        for i in idxs[:rem]:
            floors[i] += 1

    out = [min_units_each + floors[i] for i in range(n)]
    diff = total_units - sum(out)
    if diff != 0:
        # Best-effort fix-up (should be extremely rare).
        out[-1] += diff
    return out


def normalize_relative_percentages(
    values: List[float],
    decimals: int = 2,
    min_percent_each: float = 0.0,
) -> List[float]:
    """
    Normalize arbitrary non-negative values into percentages summing to 100.00 (at the given decimals).

    Implementation detail:
    - Work in integer "units" of 10^-decimals percent, so we can guarantee the sum exactly equals 100.00.
    """
    if not values:
        return []

    decimals = int(decimals)
    decimals = max(0, decimals)
    scale = 10**decimals
    total_units = 100 * scale
    min_units_each = int(round(float(min_percent_each) * scale))
    min_units_each = max(0, min_units_each)

    units = _allocate_units_by_weights(values, total_units=total_units, min_units_each=min_units_each)
    return [u / scale for u in units]


def generate_relative_percentages(
    n: int,
    seed: Optional[str] = None,
    decimals: int = 2,
    alpha: float = 0.7,
    min_percent_each: float = 0.01,
) -> List[float]:
    """
    Generate a deterministic pseudo-random relative percentage vector of length n.

    Notes:
    - We use a Dirichlet-like construction via Gamma(alpha, 1.0).
    - Allocation is done in integer units so the sum is exactly 100.00.
    """
    n = int(n)
    if n <= 0:
        return []

    alpha = float(alpha)
    if alpha <= 0.0:
        alpha = 1.0

    seed_s = str(seed or "").strip()
    if not seed_s:
        seed_s = f"n={n}"
    digest = hashlib.sha256(seed_s.encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16)
    rng = random.Random(seed_int)

    weights = [max(0.0, float(rng.gammavariate(alpha, 1.0))) for _ in range(n)]
    # Ensure not all-zero.
    if sum(weights) <= 0.0:
        weights = [1.0] * n

    return normalize_relative_percentages(weights, decimals=decimals, min_percent_each=min_percent_each)


def format_electrode_composition(elements: List[str], percents: List[float], decimals: int = 2) -> str:
    """
    Format as: "Ni(69.00%), Co(19.07%), ..."
    """
    if not elements:
        return ""
    if len(elements) != len(percents or []):
        raise ValueError("elements and percents must have the same length")
    d = max(0, int(decimals))
    parts = [f"{str(el).strip()}({float(p):.{d}f}%)" for el, p in zip(elements, percents)]
    return ", ".join(parts)


def build_electrode_composition(
    elements: List[str],
    percents: Optional[List[float]] = None,
    seed: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """
    Build a normalized + formatted electrode composition string.

    If `percents` is None, generate a deterministic pseudo-random composition.
    If `percents` is provided, treat them as relative weights and normalize to sum=100.00.
    """
    els = [str(x).strip() for x in (elements or []) if str(x).strip()]
    if not els:
        return ""

    if percents:
        p = normalize_relative_percentages(list(percents), decimals=decimals, min_percent_each=0.0)
        return format_electrode_composition(els, p, decimals=decimals)

    seed_s = str(seed or "").strip()
    if not seed_s:
        seed_s = "components:" + "|".join(els)
    p = generate_relative_percentages(len(els), seed=seed_s, decimals=decimals, min_percent_each=0.01)
    return format_electrode_composition(els, p, decimals=decimals)

