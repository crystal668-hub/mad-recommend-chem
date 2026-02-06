"""
ReAct reasoning data structures.

This module intentionally contains only trajectory/step records used by:
- the LangChain-based ReAct runtime (`agents/react_agent.py`)
- debate coordinators that need step-level targeting and evidence verification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ActionType(Enum):
    """Available ReAct actions."""

    SEARCH_LITERATURE = "search_literature"
    SEARCH_EXPERIENCE = "search_experience"
    ANALYZE = "analyze"
    CONCLUDE = "conclude"


@dataclass
class ToolCallRecord:
    """
    A single tool invocation within a ReAct ACTION step.

    We record these so one ACTION step can contain multiple tool calls while still
    remaining a single ReActStep (one thought -> one action phase -> N tool calls).
    """

    tool_name: str
    tool_call_id: Optional[str]
    tool_args: Dict[str, Any]
    observation: str
    observation_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_args": self.tool_args,
            "observation": self.observation,
            "observation_data": self.observation_data,
        }


@dataclass
class ReActStep:
    """
    One ReAct reasoning step: Thought -> Action -> Observation.
    """

    step_number: int
    thought: str
    action: Union[ActionType, str]
    action_input: Dict[str, Any]
    observation: str

    # Multi-tool support (authoritative per-call details live here).
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    # Backward-compatible single-tool fields.
    tool_call_id: Optional[str] = None
    observation_data: Optional[Any] = None

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def action_name(self) -> str:
        if self.tool_calls:
            names: List[str] = []
            for call in self.tool_calls:
                if call.tool_name and call.tool_name not in names:
                    names.append(call.tool_name)
            return "|".join(names)
        return self.action.value if isinstance(self.action, ActionType) else str(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "action": self.action_name,
            "action_input": self.action_input,
            "tool_calls": [c.to_dict() for c in (self.tool_calls or [])],
            "observation": self.observation,
            "tool_call_id": self.tool_call_id,
            "observation_data": self.observation_data,
            "timestamp": self.timestamp,
        }


@dataclass
class ReActTrajectory:
    """Records the complete reasoning process for one agent call."""

    query: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_steps: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def add_step(self, step: ReActStep) -> None:
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def finalize(self, final_answer: str) -> None:
        self.final_answer = final_answer
        self.end_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def get_trajectory_summary(self) -> str:
        summary_parts = [
            "===== ReAct Trajectory =====",
            f"Original Query: {self.query}",
            f"Total Steps: {self.total_steps}",
            "",
        ]

        for step in self.steps:
            summary_parts.append(f" STEP {step.step_number}:")
            summary_parts.append(f" THOUGHT:\n {step.thought}")
            summary_parts.append(f" ACTION:\n {step.action_name or '(none)'}")
            if getattr(step, "tool_calls", None):
                for c in step.tool_calls:
                    try:
                        args_str = json.dumps(getattr(c, "tool_args", {}), ensure_ascii=False)
                    except Exception:
                        args_str = str(getattr(c, "tool_args", {}))
                    summary_parts.append(f"  - {getattr(c, 'tool_name', '')} args={args_str}")
            summary_parts.append(f" OBSERVATION:\n {step.observation}")
            summary_parts.append("")

        if self.final_answer:
            summary_parts.append(f" FINAL ANSWER:\n {self.final_answer}")
        return "\n".join(summary_parts)
