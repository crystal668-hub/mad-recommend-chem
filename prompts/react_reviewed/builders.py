from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from prompts.react_reviewed.render import json_block, json_preview, render_template


def _render_proposer_runtime_guidance_block(runtime_guidance: Optional[Dict[str, Any]]) -> str:
    if not isinstance(runtime_guidance, dict) or not runtime_guidance:
        return ""

    def _ensure_sentence(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return cleaned
        if cleaned.endswith((".", "!", "?")):
            return cleaned
        return cleaned + "."

    budget_snapshot = dict(runtime_guidance.get("budget_snapshot") or {})
    recommended_next_tools = [
        str(item).strip()
        for item in list(runtime_guidance.get("recommended_next_tools") or [])
        if str(item).strip()
    ]
    avoid_actions = [
        str(item).strip()
        for item in list(runtime_guidance.get("avoid_actions") or [])
        if str(item).strip()
    ]

    lines = [
        "",
        "Runtime budget snapshot:",
        (
            f"- Action step {budget_snapshot.get('step_number', '?')} of {budget_snapshot.get('max_steps', '?')}; "
            f"remaining steps: {budget_snapshot.get('remaining_steps', '?')}."
        ),
        (
            f"- Query planned: {budget_snapshot.get('query_planned', False)}; "
            f"search rounds: {budget_snapshot.get('search_rounds_used', 0)}; "
            f"download rounds: {budget_snapshot.get('download_rounds_used', 0)}; "
            f"screenings: {budget_snapshot.get('screen_rounds_used', 0)}."
        ),
        (
            f"- Locked papers: {budget_snapshot.get('locked_paper_ids', [])}; "
            f"parsed locked papers: {budget_snapshot.get('parsed_locked_paper_ids', [])}; "
            f"evidence anchors: {budget_snapshot.get('evidence_anchor_count', 0)}."
        ),
        (
            f"- Screening required now: {budget_snapshot.get('screening_required', False)}; "
            f"recovery search/download available: {budget_snapshot.get('recovery_search_download_available', False)}."
        ),
        f"Current stage: {runtime_guidance.get('current_stage', 'unknown')}.",
        "Exit criteria: " + _ensure_sentence(str(runtime_guidance.get("exit_criteria") or "")),
    ]
    if recommended_next_tools:
        lines.append("Recommended next tools: " + ", ".join(recommended_next_tools) + ".")
    if avoid_actions:
        lines.append("Avoid this step: " + " ".join(_ensure_sentence(item) for item in avoid_actions))
    return "\n".join(lines).rstrip()


def build_proposer_user_prompt(
    *,
    cycle_number: int,
    question: str,
    context: Optional[str],
    task_spec: Dict[str, Any],
    entity_pack: Dict[str, Any],
    prior_submission: Optional[Dict[str, Any]],
    prior_proposer_trajectory: Optional[Dict[str, Any]],
    open_review_items: Sequence[Dict[str, Any]],
    conclude_call_contract: Dict[str, Any],
    retrieval_policy: Dict[str, Any],
    limit: int = 12000,
) -> str:
    return json_preview(
        {
            "cycle_number": cycle_number,
            "question": question,
            "context": context,
            "task_spec": task_spec,
            "entity_pack": entity_pack,
            "prior_submission": prior_submission,
            "prior_proposer_trajectory": prior_proposer_trajectory,
            "open_review_items": list(open_review_items),
            "conclude_call_contract": conclude_call_contract,
            "retrieval_policy": retrieval_policy,
        },
        limit=limit,
    )


def build_reviewer_user_prompt(
    *,
    cycle_number: int,
    reviewer_role: str,
    submission: Dict[str, Any],
    proposer_trajectory: Dict[str, Any],
    conclude_call_contract: Dict[str, Any],
    limit: int = 12000,
) -> str:
    return json_preview(
        {
            "cycle_number": cycle_number,
            "reviewer_role": reviewer_role,
            "submission": submission,
            "proposer_trajectory": proposer_trajectory,
            "conclude_call_contract": conclude_call_contract,
        },
        limit=limit,
    )


def build_proposer_thought_prompt() -> str:
    return "CURRENT PHASE: THOUGHT\nState the next retrieval or revision intent in 1-2 short sentences.\nDo not output JSON."


def build_reviewer_thought_prompt() -> str:
    return "CURRENT PHASE: THOUGHT\nState the next audit target in one short sentence."


def build_proposer_system_prompt(*, conclude_contract: Dict[str, Any], proposer_candidate_target: int) -> str:
    return render_template(
        "proposer_system.yaml",
        proposer_candidate_target=str(proposer_candidate_target),
        tool_call_rule=str(conclude_contract.get("tool_call_rule") or ""),
        conclude_contract_json=json_block(conclude_contract),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_proposer_action_prompt(
    *,
    tool_names: Sequence[str],
    retrieval_tools: Sequence[str],
    conclude_contract: Dict[str, Any],
    proposer_candidate_target: int,
    runtime_guidance: Optional[Dict[str, Any]] = None,
) -> str:
    return render_template(
        "proposer_action.yaml",
        tool_names=", ".join(tool_names),
        retrieval_tools=", ".join(retrieval_tools),
        proposer_candidate_target=str(proposer_candidate_target),
        runtime_guidance_block=_render_proposer_runtime_guidance_block(runtime_guidance),
        tool_call_rule=str(conclude_contract.get("tool_call_rule") or ""),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_reviewer_system_prompt(
    *,
    reviewer_role: str,
    role_note: str,
    max_retrieval_actions: int,
    conclude_contract: Dict[str, Any],
) -> str:
    return render_template(
        "reviewer_system.yaml",
        reviewer_role=reviewer_role,
        role_note=role_note,
        max_retrieval_actions=str(max_retrieval_actions),
        tool_call_rule=str(conclude_contract.get("tool_call_rule") or ""),
        conclude_contract_json=json_block(conclude_contract),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_reviewer_action_prompt(
    *,
    tool_names: Sequence[str],
    retrieval_budget: int,
    conclude_contract: Dict[str, Any],
) -> str:
    return render_template(
        "reviewer_action.yaml",
        tool_names=", ".join(tool_names),
        retrieval_budget=str(retrieval_budget),
        tool_call_rule=str(conclude_contract.get("tool_call_rule") or ""),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_screening_system_prompt(*, max_candidates: int) -> str:
    screening_example = {
        "locked_paper_ids": ["paper-1"],
        "dropped_paper_ids": ["paper-2"],
        "decisions": [
            {"paper_id": "paper-1", "decision": "lock", "reason": "Strong question match with better acquisition signals."},
            {"paper_id": "paper-2", "decision": "drop", "reason": "Generic metadata and weaker acquisition signals."},
        ],
    }
    invalid_examples = [
        {"paper_ids": ["paper-1"]},
        {"locked_papers": ["paper-1"], "dropped_papers": ["paper-2"]},
        "```json\n{\"locked_paper_ids\": [\"paper-1\"]}\n```",
    ]
    return render_template(
        "screening_system.yaml",
        max_candidates=str(max_candidates),
        screening_example_json=json_block(screening_example),
        invalid_examples_json=json_block(invalid_examples),
    )


def build_proposer_repair_system_prompt(*, conclude_contract: Dict[str, Any]) -> str:
    return render_template(
        "proposer_repair_system.yaml",
        repair_example_json=json_block(conclude_contract.get("repair_json_example") or {}),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_reviewer_repair_system_prompt(*, conclude_contract: Dict[str, Any]) -> str:
    return render_template(
        "reviewer_repair_system.yaml",
        repair_example_json=json_block(conclude_contract.get("repair_json_example") or {}),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )
