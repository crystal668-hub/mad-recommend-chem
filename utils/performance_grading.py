import re
from typing import Any, Dict, Optional, Tuple


# Grade labels (kept consistent across reactions)
GRADE_OUTSTANDING = "Outstanding"
GRADE_GOOD = "Good"
GRADE_FAIR = "Fair"
GRADE_POOR = "Poor"
GRADE_TERRIBLE = "Terrible"


# Per-reaction metric metadata (normalized output unit + display name)
_METRIC_META: Dict[str, Dict[str, str]] = {
    "HER": {"metric_name": "\u03b710", "unit": "mV"},
    "OER": {"metric_name": "\u03b710", "unit": "mV"},
    "ORR": {"metric_name": "E1/2", "unit": "V"},
    "HOR": {"metric_name": "j0", "unit": "mA cm-2"},
    "UOR": {"metric_name": "E@10", "unit": "V"},
    "EOR": {"metric_name": "Mass activity", "unit": "A mgmetal-1"},
    "HZOR": {"metric_name": "E@10", "unit": "mV"},
    "O5H": {"metric_name": "FE", "unit": "%"},
    "CO2RR": {"metric_name": "Partial current density", "unit": "mA cm-2"},
}


# Threshold tables. Boundaries overlap in the original definitions; we resolve ties by
# checking from the better grade to the worse grade (as specified in the plan).
_LOWER_IS_BETTER: Dict[str, Tuple[Tuple[float, str], ...]] = {
    "HER": (
        (50.0, GRADE_OUTSTANDING),
        (100.0, GRADE_GOOD),
        (200.0, GRADE_FAIR),
        (250.0, GRADE_POOR),
    ),
    "OER": (
        (200.0, GRADE_OUTSTANDING),
        (250.0, GRADE_GOOD),
        (300.0, GRADE_FAIR),
        (350.0, GRADE_POOR),
    ),
    "UOR": (
        (1.3, GRADE_OUTSTANDING),
        (1.35, GRADE_GOOD),
        (1.45, GRADE_FAIR),
        (1.6, GRADE_POOR),
    ),
    "HZOR": (
        (-100.0, GRADE_OUTSTANDING),
        (0.0, GRADE_GOOD),
        (50.0, GRADE_FAIR),
        (100.0, GRADE_POOR),
    ),
}

_HIGHER_IS_BETTER: Dict[str, Tuple[Tuple[float, str], ...]] = {
    "ORR": (
        (0.92, GRADE_OUTSTANDING),
        (0.85, GRADE_GOOD),
        (0.75, GRADE_FAIR),
        (0.65, GRADE_POOR),
    ),
    "HOR": (
        (3.0, GRADE_OUTSTANDING),
        (2.5, GRADE_GOOD),
        (1.5, GRADE_FAIR),
        (0.65, GRADE_POOR),
    ),
    "EOR": (
        (31.5, GRADE_OUTSTANDING),
        (25.0, GRADE_GOOD),
        (0.65, GRADE_FAIR),
        (0.5, GRADE_POOR),
    ),
    "O5H": (
        (95.0, GRADE_OUTSTANDING),
        (90.0, GRADE_GOOD),
        (85.0, GRADE_FAIR),
        (80.0, GRADE_POOR),
    ),
    "CO2RR": (
        (1.0, GRADE_OUTSTANDING),
        (0.2, GRADE_GOOD),
        (0.08, GRADE_FAIR),
        (0.04, GRADE_POOR),
    ),
}


def normalize_text(text: str) -> str:
    """
    Normalize common Unicode variants to make parsing resilient.

    Keep this conservative: only normalize characters that frequently appear in copied
    scientific text and break regex matching on Windows terminals.
    """
    s = str(text or "")
    if not s:
        return ""

    # Dashes/minus variants -> ASCII hyphen-minus
    for ch in ["\u2212", "\u2013", "\u2014", "\u207b"]:
        s = s.replace(ch, "-")

    # Superscript digits -> ASCII digits (helps "cm\u207b\u00b2", "mg\u207b\u00b9", etc.)
    s = s.replace("\u00b9", "1").replace("\u00b2", "2").replace("\u00b3", "3")

    # Unify common exponent notations (best-effort)
    s = re.sub(r"(?i)cm\s*\^\s*-?\s*2\b", "cm-2", s)
    s = re.sub(r"(?i)mg\s*\^\s*-?\s*1\b", "mg-1", s)

    return s


def extract_reaction_type(claim: str) -> Optional[str]:
    s = normalize_text(claim)
    m = re.search(r"(?i)\bReaction\s*Type\s*:\s*([A-Za-z0-9]+)", s)
    if not m:
        return None
    rt = str(m.group(1) or "").strip().upper()
    return rt if rt in _METRIC_META else None


def extract_last_performance_metrics_text(claim: str) -> Optional[str]:
    """
    Extract the last "Performance Metrics ...: <text>" segment from a claim.

    - Works for both "Performance Metrics:" and "Performance Metrics (xxx):"
    - Picks the last occurrence to handle claims that include multiple metric lines.
    """
    s = normalize_text(claim)
    matches = list(
        re.finditer(
            r"(?i)Performance\s*Metrics[^\n\r:]*:\s*([^\n\r]*)",
            s,
        )
    )
    if not matches:
        return None
    raw = str(matches[-1].group(1) or "").strip()
    return raw or None


def grade_value(reaction_type: str, value: float) -> str:
    rt = str(reaction_type or "").strip().upper()
    if rt in _LOWER_IS_BETTER:
        for thr, grade in _LOWER_IS_BETTER[rt]:
            if value <= float(thr):
                return grade
        return GRADE_TERRIBLE

    if rt in _HIGHER_IS_BETTER:
        for thr, grade in _HIGHER_IS_BETTER[rt]:
            if value >= float(thr):
                return grade
        return GRADE_TERRIBLE

    # Unknown reaction type: cannot grade.
    return GRADE_TERRIBLE


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _extract_value_unit_near_10ma(text: str) -> Optional[Tuple[float, str]]:
    """
    Extract a (value, unit) pair for V/mV metrics, preferring matches near "10 mA".
    """
    t = normalize_text(text)
    if not t:
        return None

    num = r"([-+]?\d+(?:\.\d+)?)"
    pm = r"(?:\+/-|\u00b1)"
    unit = r"(mV|V)"

    # Value before/after the "10 mA" anchor, with optional "+/-" uncertainty.
    pats = [
        rf"{num}\s*{pm}\s*\d+(?:\.\d+)?\s*{unit}\b[^\n\r]{{0,160}}\b10\s*mA\b",
        rf"\b10\s*mA\b[^\n\r]{{0,160}}{num}\s*{pm}\s*\d+(?:\.\d+)?\s*{unit}\b",
        rf"{num}\s*{unit}\b[^\n\r]{{0,160}}\b10\s*mA\b",
        rf"\b10\s*mA\b[^\n\r]{{0,160}}{num}\s*{unit}\b",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if not m:
            continue
        val = _to_float(m.group(1))
        u = str(m.group(2) or "").strip()
        if val is None or not u:
            continue
        return val, u
    return None


def _extract_first_value_with_unit(text: str, unit_pat: str) -> Optional[Tuple[float, str]]:
    t = normalize_text(text)
    m = re.search(rf"([-+]?\d+(?:\.\d+)?)\s*{unit_pat}\b", t, flags=re.IGNORECASE)
    if not m:
        return None
    val = _to_float(m.group(1))
    if val is None:
        return None
    u = str(m.group(2) or "").strip() if m.lastindex and m.lastindex >= 2 else ""
    return val, u


def _extract_current_density(text: str, prefer_label: Optional[str] = None) -> Optional[Tuple[float, str]]:
    """
    Extract a current density like "<val> mA cm-2" or "<val> A/cm^2".
    Returns (value, unit) where unit is "mA" or "A" (caller normalizes).
    """
    t = normalize_text(text)
    if not t:
        return None

    num = r"([-+]?\d+(?:\.\d+)?)"
    unit = r"(mA|A)"
    cm2 = r"cm\s*(?:\^\s*)?-?\s*2\b"

    if prefer_label:
        # Accept small variations like j0 / j₀.
        label = str(prefer_label)
        m = re.search(rf"(?i){label}\s*[:=]\s*{num}\s*{unit}\s*(?:/|\s*){cm2}", t)
        if m:
            v = _to_float(m.group(1))
            u = str(m.group(2) or "").strip()
            if v is not None and u:
                return v, u

    # Generic current density pattern.
    m = re.search(rf"(?i){num}\s*{unit}\s*(?:/|\s*){cm2}", t)
    if not m:
        return None
    v = _to_float(m.group(1))
    u = str(m.group(2) or "").strip()
    if v is None or not u:
        return None
    return v, u


def _extract_mass_activity(text: str) -> Optional[Tuple[float, str]]:
    """
    Extract mass activity like "<val> mA/mg" or "<val> A mg-1".
    Returns (value, unit) where unit is "mA" or "A" (caller normalizes to A/mg).
    """
    t = normalize_text(text)
    if not t:
        return None

    num = r"([-+]?\d+(?:\.\d+)?)"
    unit = r"(mA|A)"

    # Prefer "mass activity" labeled forms.
    m = re.search(rf"(?i)mass\s*activity[^\n\r]{{0,60}}?{num}\s*{unit}\s*(?:/|\s*)mg", t)
    if not m:
        # Generic A/mg style.
        m = re.search(rf"(?i){num}\s*{unit}\s*(?:/|\s*)mg", t)
    if not m:
        return None

    v = _to_float(m.group(1))
    u = str(m.group(2) or "").strip()
    if v is None or not u:
        return None
    return v, u


def parse_metric_value(rt: str, raw_metric_text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse and normalize a metric point estimate for a given reaction type.

    Returns:
        (metric_value, metric_unit) in the normalized unit expected by the grading rules.
    """
    reaction = str(rt or "").strip().upper()
    t = normalize_text(raw_metric_text)
    if not reaction or not t:
        return None, None

    # ---- V/mV metrics ----
    if reaction in {"HER", "OER", "UOR", "HZOR", "ORR"}:
        # Prefer "10 mA" anchored extraction for @10 mA metrics.
        val_unit = None
        if reaction in {"HER", "OER", "UOR", "HZOR"}:
            val_unit = _extract_value_unit_near_10ma(t)

        if val_unit is None:
            # ORR: prefer E1/2/half-wave tagged values.
            if reaction == "ORR":
                m = re.search(
                    r"(?i)(?:E\s*1/2|E1/2|E_\s*1/2|half[-\s]*wave(?:\s*potential)?)\s*[:=]?\s*"
                    r"([-+]?\d+(?:\.\d+)?)\s*(mV|V)\b",
                    t,
                )
                if m:
                    v = _to_float(m.group(1))
                    u = str(m.group(2) or "").strip()
                    if v is not None and u:
                        val_unit = (v, u)

        if val_unit is None:
            # Fallback: first mV/V occurrence.
            m = re.search(r"(?i)([-+]?\d+(?:\.\d+)?)\s*(mV|V)\b", t)
            if m:
                v = _to_float(m.group(1))
                u = str(m.group(2) or "").strip()
                if v is not None and u:
                    val_unit = (v, u)

        if val_unit is None:
            return None, None

        v, u = val_unit
        u_norm = u.lower()
        if reaction in {"HER", "OER", "HZOR"}:
            # Normalize to mV.
            if u_norm == "v":
                return v * 1000.0, "mV"
            return v, "mV"

        # ORR/UOR normalize to V.
        if u_norm == "mv":
            return v / 1000.0, "V"
        return v, "V"

    # ---- Current density metrics ----
    if reaction in {"HOR", "CO2RR"}:
        # Prefer j0-labeled extraction for HOR.
        prefer = r"j0|j\u2080" if reaction == "HOR" else None
        res = _extract_current_density(t, prefer_label=prefer)
        if res is None:
            return None, None
        v, u = res
        if str(u).strip().lower() == "a":
            return v * 1000.0, "mA cm-2"
        return v, "mA cm-2"

    # ---- Mass activity ----
    if reaction == "EOR":
        res = _extract_mass_activity(t)
        if res is None:
            return None, None
        v, u = res
        if str(u).strip().lower() == "ma":
            return v / 1000.0, "A mgmetal-1"
        return v, "A mgmetal-1"

    # ---- Faradaic efficiency ----
    if reaction == "O5H":
        # Prefer explicit percent.
        m = re.search(r"(?i)([-+]?\d+(?:\.\d+)?)\s*%", t)
        if m:
            v = _to_float(m.group(1))
            return (v, "%") if v is not None else (None, None)

        # Prefer FE-labeled numeric.
        m = re.search(r"(?i)(?:\bFE\b|faradaic\s*efficiency)[^0-9\-+]{0,20}([-+]?\d+(?:\.\d+)?)", t)
        if not m:
            m = re.search(r"(?i)([-+]?\d+(?:\.\d+)?)", t)
        if not m:
            return None, None

        v = _to_float(m.group(1))
        if v is None:
            return None, None
        if 0.0 <= v <= 1.0:
            return v * 100.0, "%"
        return v, "%"

    return None, None


def evaluate_claim(claim: str, reaction_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Evaluate a final claim and return a normalized metric + grade.

    Returns None if we cannot extract a single point estimate with a known unit.
    """
    s = normalize_text(claim)
    rt_given = str(reaction_type or "").strip().upper()
    rt = rt_given if rt_given and rt_given != "UNKNOWN" else (extract_reaction_type(s) or "")
    rt = rt.strip().upper()
    if not rt or rt not in _METRIC_META:
        return None

    raw_metric_text = extract_last_performance_metrics_text(s)
    if not raw_metric_text:
        return None

    metric_value, metric_unit = parse_metric_value(rt, raw_metric_text)
    if metric_value is None or not metric_unit:
        return None

    grade = grade_value(rt, float(metric_value))
    meta = _METRIC_META.get(rt, {})
    return {
        "reaction_type": rt,
        "metric_name": meta.get("metric_name", ""),
        "metric_value": float(metric_value),
        "metric_unit": str(metric_unit),
        "grade": grade,
        "raw_metric_text": str(raw_metric_text),
    }


def metric_direction(reaction_type: str) -> Optional[str]:
    """
    Return the optimization direction for the reaction's normalized metric.

    Returns:
        "lower" if lower metric values are better,
        "higher" if higher metric values are better,
        None if unknown reaction type / not gradable.
    """
    rt = str(reaction_type or "").strip().upper()
    if rt in _LOWER_IS_BETTER:
        return "lower"
    if rt in _HIGHER_IS_BETTER:
        return "higher"
    return None


def grade_rank(grade: str) -> int:
    """
    Map a grade label to an integer rank for sorting.

    Outstanding > Good > Fair > Poor > Terrible. Unknown grades map to -1.
    """
    g = str(grade or "").strip().lower()
    if not g:
        return -1

    mapping = {
        GRADE_OUTSTANDING.lower(): 4,
        GRADE_GOOD.lower(): 3,
        GRADE_FAIR.lower(): 2,
        GRADE_POOR.lower(): 1,
        GRADE_TERRIBLE.lower(): 0,
    }
    return int(mapping.get(g, -1))
