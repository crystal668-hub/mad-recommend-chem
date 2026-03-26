from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from prompts.react_reviewed.render import json_block, json_preview, render_template


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


def build_proposer_system_prompt(*, conclude_contract: Dict[str, Any]) -> str:
    return render_template(
        "proposer_system.yaml",
        tool_call_rule=str(conclude_contract.get("tool_call_rule") or ""),
        conclude_contract_json=json_block(conclude_contract),
        tool_call_example_json=json_block(conclude_contract.get("tool_call_example") or {}),
        invalid_examples_json=json_block(conclude_contract.get("invalid_examples") or []),
    )


def build_proposer_action_prompt(*, tool_names: Sequence[str], retrieval_tools: Sequence[str], conclude_contract: Dict[str, Any]) -> str:
    return render_template(
        "proposer_action.yaml",
        tool_names=", ".join(tool_names),
        retrieval_tools=", ".join(retrieval_tools),
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
