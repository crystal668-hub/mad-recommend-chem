"""
Reaction ranking utilities.

This module ranks multiple reaction-type debate results using:
1) Grade (Outstanding > Good > Fair > Poor > Terrible)
2) A simple tie-breaker based on the metric direction:
   - lower-is-better: smaller metric_value wins (implemented as -metric_value)
   - higher-is-better: larger metric_value wins

Note:
Metrics across different reaction types are not directly comparable; the tie-breaker
exists only to produce deterministic ordering within the same grade.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from utils.performance_grading import grade_rank, metric_direction


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _extract_summary_fields(item: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[float], Optional[str]]:
    """
    Extract (reaction_type, grade, metric_value, metric_unit) from a per-reaction summary dict.
    """
    if not isinstance(item, dict):
        return "", None, None, None

    pe = item.get("performance_evaluation")
    rt = str(item.get("reaction_type") or "").strip().upper()

    grade = None
    metric_value = None
    metric_unit = None
    if isinstance(pe, dict):
        grade = pe.get("grade")
        metric_value = pe.get("metric_value")
        metric_unit = pe.get("metric_unit")
        if not rt:
            rt = str(pe.get("reaction_type") or "").strip().upper()

    if grade is None:
        grade = item.get("grade")
    if metric_value is None:
        metric_value = item.get("metric_value")
    if metric_unit is None:
        metric_unit = item.get("metric_unit")

    return rt, (str(grade).strip() if grade is not None else None), _to_float(metric_value), (str(metric_unit).strip() if metric_unit else None)


def _sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
    rt, grade, metric_value, _metric_unit = _extract_summary_fields(item)

    g_rank = grade_rank(grade or "")

    # Missing metrics should sort last within the same grade.
    if metric_value is None:
        return g_rank, float("-inf")

    direction = metric_direction(rt) if rt else None
    if direction == "lower":
        tie = -float(metric_value)
    else:
        # "higher" and unknown directions: keep as-is.
        tie = float(metric_value)

    return g_rank, tie


def rank_reactions(items: List[Dict[str, Any]], top_k: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Rank reaction summaries and return (ranking, top_k_items).
    """
    safe_items: List[Dict[str, Any]] = [it for it in (items or []) if isinstance(it, dict)]

    ranking = sorted(safe_items, key=_sort_key, reverse=True)
    try:
        k = int(top_k)
    except Exception:
        k = 2
    k = max(0, k)
    return ranking, ranking[:k]

