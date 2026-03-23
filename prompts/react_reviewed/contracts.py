from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def build_submission_prompt_scaffold(
    *,
    question: str,
    cycle_number: int,
    answer_sections: Sequence[Dict[str, Any]],
    issue_refs: Sequence[str],
) -> Dict[str, Any]:
    section_items = []
    for section in answer_sections:
        section_items.append(
            {
                "section_id": section["section_id"],
                "title": section["title"],
                "content": "<grounded section content>",
                "citation_ids": ["<citation_id_from_citations>"],
                "step_refs": [{"trajectory_id": "<trajectory_id>", "step_number": 1}],
                "issue_refs": list(issue_refs),
                "section_confidence": {
                    "level": "medium",
                    "score": 0.5,
                    "rationale": "<brief section confidence rationale>",
                },
            }
        )
    return {
        "submission_id": f"submission_cycle_{cycle_number}",
        "question": question,
        "version": cycle_number,
        "sections": section_items,
        "citations": [
            {
                "citation_id": "CIT-1",
                "paper_id": "<paper_id_from_search_papers>",
                "doi": None,
                "title": "<paper_title_from_tools>",
                "year": None,
                "venue": None,
                "section_ids": ["<section_id_from_tools_or_sec_abstract>"],
                "evidence_ids": ["<evidence_id_from_tools>"],
            }
        ],
        "limitations": ["<explicit limitation grounded in the current run>"],
        "overall_confidence": {
            "level": "medium",
            "score": 0.5,
            "rationale": "<overall confidence rationale grounded in retrieved evidence>",
        },
        "trajectory_id": "<trajectory_id>",
        "step_refs": [{"trajectory_id": "<trajectory_id>", "step_number": 1}],
        "issue_refs": list(issue_refs),
    }


def build_submission_prompt_contract(
    *,
    question: str,
    cycle_number: int,
    answer_sections: Sequence[Dict[str, Any]],
    issue_refs: Sequence[str],
) -> Dict[str, Any]:
    submission_template = build_submission_prompt_scaffold(
        question=question,
        cycle_number=cycle_number,
        answer_sections=answer_sections,
        issue_refs=issue_refs,
    )
    return {
        "tool_name": "conclude",
        "tool_call_rule": (
            "Call conclude with exactly {\"submission\": {...}}. Do not send a bare payload and do not use "
            "alternate top-level keys such as payload, answer_sections, or review."
        ),
        "canonical_submission_keys": [
            "submission_id",
            "question",
            "version",
            "sections",
            "citations",
            "limitations",
            "overall_confidence",
            "trajectory_id",
            "step_refs",
            "issue_refs",
        ],
        "required_section_ids": [str(section["section_id"]) for section in answer_sections],
        "section_object_keys": [
            "section_id",
            "title",
            "content",
            "citation_ids",
            "step_refs",
            "issue_refs",
            "section_confidence",
        ],
        "citation_object_keys": [
            "citation_id",
            "paper_id",
            "doi",
            "title",
            "year",
            "venue",
            "section_ids",
            "evidence_ids",
        ],
        "step_ref_keys": ["trajectory_id", "step_number"],
        "confidence_object_keys": ["level", "score", "rationale"],
        "tool_call_example": {"submission": submission_template},
        "repair_json_example": {"kind": "submission", "payload": submission_template},
        "invalid_examples": [
            {"submission_id": "submission_cycle_1", "sections": []},
            {"payload": {"submission_id": "submission_cycle_1"}},
            {"answer_sections": [{"section_id": "direct_answer"}]},
            "```json\n{\"submission\": {...}}\n```",
        ],
    }


def build_review_prompt_contract(
    *,
    reviewer_role: str,
    target_section_id: Optional[str],
    target_trajectory_id: str,
) -> Dict[str, Any]:
    review_item_template: Dict[str, Any] = {
        "review_id": f"{reviewer_role}_1",
        "reviewer_role": reviewer_role,
        "anchor_kind": "section_only" if target_section_id else "global",
        "severity": "warning",
        "flaw_type": "<short_flaw_type>",
        "critique": "<what is wrong and why>",
        "required_action": "<what must change>",
        "evidence_refs": ["<evidence_or_citation_ref>"],
        "status": "open",
    }
    if target_section_id:
        review_item_template["target_section_id"] = target_section_id
    else:
        review_item_template["target_trajectory_id"] = target_trajectory_id
    review_payload = {"review_items": [review_item_template]}
    return {
        "tool_name": "conclude",
        "tool_call_rule": (
            "Call conclude with exactly {\"review\": {\"review_items\": [...]}}. Do not send a bare array, "
            "do not pass {\"review_items\": [...]} directly, and do not rename the wrapper key."
        ),
        "canonical_review_keys": ["review_items"],
        "review_item_object_keys": [
            "review_id",
            "reviewer_role",
            "anchor_kind",
            "severity",
            "flaw_type",
            "critique",
            "required_action",
            "evidence_refs",
            "status",
            "target_trajectory_id",
            "target_step_number",
            "target_section_id",
        ],
        "allowed_anchor_kinds": ["step_section", "section_only", "global", "missing_section"],
        "allowed_severity": ["blocking", "warning", "note"],
        "tool_call_example": {"review": review_payload},
        "repair_json_example": {"kind": "review_items", "payload": review_payload["review_items"]},
        "invalid_examples": [
            [{"review_id": "rev-1"}],
            {"review_items": [{"review_id": "rev-1"}]},
            {"payload": [{"review_id": "rev-1"}]},
            "```json\n{\"review\": {\"review_items\": [...]}}\n```",
        ],
    }
