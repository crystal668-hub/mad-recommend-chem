"""
LangGraph-style Debate Coordinator (no external dependency required).

Implements the agreed debate protocol:
1) Propose (one proposal per model/agent)
2) Repeat rounds of (Review -> Rebuttal -> Rule adjudication)
   - Reviews/Rebuttals MAY be parametric (no evidence). Evidence is OPTIONAL but must be verifiable if provided.
   - Valid reviews (with or without evidence) affect consensus/penalties; this prevents false consensus when
     agents raise substantive parametric critiques but cannot retrieve matching chunks within budget.
   - If a proposal fails to respond to valid reviews for N consecutive rounds -> defeated.
   - Agents can also voluntarily withdraw their proposal.

Notes:
- We keep agent "memoryless" by storing all debate context in coordinator state.
- "Verifiable source_id" is enforced by requiring cited source_id to appear in the
  agent's own retrieval results within the same message trajectory (except the special
  marker `source_id="llm"` for parametric/internal knowledge).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

from agents.react_agent import ReActAgent
from agents.react_reasoning import ReActTrajectory
from prompts.debate_phase_prompts import (
    build_initial_debate_prompt,
    DEBATE_PROPOSE_SYSTEM_PROMPT,
    DEBATE_REVIEW_SYSTEM_PROMPT,
    DEBATE_REBUTTAL_SYSTEM_PROMPT,
)
from utils.electrode_composition import parse_components_with_percent
from utils.logger import get_run_id, make_debate_id, write_debate_artifacts
from utils.source_id import is_valid_chroma_source_id, normalize_chroma_source_id

logger = logging.getLogger("MAD.debate.langgraph")


# =========================
# Pydantic Schemas
# =========================


class EvidenceItem(BaseModel):
    source_id: str
    quote: Optional[str] = None


class ProposalOutput(BaseModel):
    reaction_type: str
    electrode_composition: str
    catalyst_metal_elements: List[str]
    products: str
    performance_metrics: str
    confidence: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    rationale: str = ""


class ReviewItem(BaseModel):
    target_proposal_id: str
    target_step_number: int
    flaw_type: str
    critique: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    reviews: List[ReviewItem]


class RebuttalItem(BaseModel):
    target_review_id: str
    response_mode: str  # defend | revise | withdraw | no_response
    response: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class RebuttalOutput(BaseModel):
    rebuttals: List[RebuttalItem]
    revised_claim: Optional[str] = None


# =========================
# Result Structures
# =========================


@dataclass
class DebateReview:
    review_id: str
    round_number: int
    from_proposal_id: str
    target_proposal_id: str
    target_step_number: int
    flaw_type: str
    critique: str
    evidence: List[Dict[str, Any]]
    valid: bool
    invalid_reason: Optional[str] = None


@dataclass
class DebateRebuttal:
    rebuttal_id: str
    round_number: int
    from_proposal_id: str
    target_review_id: str
    response_mode: str
    response: str
    evidence: List[Dict[str, Any]]
    valid: bool
    invalid_reason: Optional[str] = None


@dataclass
class ProposalState:
    proposal_id: str
    agent_name: str
    status: str = "active"  # active | withdrawn | defeated
    no_response_streak: int = 0
    call_error_streak: int = 0
    last_call_error: Optional[str] = None

    claim: str = ""
    propose_response: Optional[str] = None
    propose_trajectory: Optional[ReActTrajectory] = None

    # Per-round bound threads
    received_reviews: List[DebateReview] = field(default_factory=list)
    sent_reviews: List[DebateReview] = field(default_factory=list)
    sent_rebuttals: List[DebateRebuttal] = field(default_factory=list)


@dataclass
class GraphDebateResult:
    """Result format compatible with main.py saving + summary."""

    consensus_reached: bool
    final_products: Optional[str]
    final_performance: Optional[str]
    reasoning_trajectory: str
    debate_rounds: int
    debate_history: List[Dict[str, Any]]
    time_elapsed: float
    surviving_proposals: List[Dict[str, Any]]
    defeated_proposals: List[Dict[str, Any]]
    withdrawn_proposals: List[Dict[str, Any]]
    # If the debate ends with multiple survivors, we may deterministically resolve a stalemate.
    winner_proposal_id: Optional[str] = None
    resolution_method: Optional[str] = None
    resolution_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_reached": self.consensus_reached,
            "final_products": self.final_products,
            "final_performance": self.final_performance,
            "reasoning_trajectory": self.reasoning_trajectory,
            "debate_rounds": self.debate_rounds,
            "debate_history": self.debate_history,
            "time_elapsed": self.time_elapsed,
            "surviving_proposals": self.surviving_proposals,
            "defeated_proposals": self.defeated_proposals,
            "withdrawn_proposals": self.withdrawn_proposals,
            "winner_proposal_id": self.winner_proposal_id,
            "resolution_method": self.resolution_method,
            "resolution_details": self.resolution_details,
        }


# =========================
# Coordinator
# =========================


class LangGraphDebateCoordinator:
    """
    Debate coordinator that follows a LangGraph-style node/edge protocol.
    """

    def __init__(self, agents: List[ReActAgent], config: Dict[str, Any]):
        self.agents = agents
        self.config = config or {}

        # Protocol params
        self.max_rounds = int(self.config.get("max_rounds", 3))
        self.no_response_threshold = int(self.config.get("no_response_threshold", 1))
        self.max_reviews_per_target = int(self.config.get("max_reviews_per_target", 2))

        # Runtime controls (to bound wall-clock time)
        self.max_concurrency = int(self.config.get("max_concurrency", max(1, len(self.agents))))
        # Backward-compatible: `timeout` exists in config.yaml; treat as the default per-phase wall-clock budget.
        default_timeout = float(self.config.get("timeout", 1200))

        raw_propose_timeout = self.config.get("propose_timeout", None)
        if raw_propose_timeout is None:
            raw_propose_timeout = self.config.get("round_timeout", None)
        if raw_propose_timeout is None:
            raw_propose_timeout = default_timeout
        self.propose_timeout_seconds = float(raw_propose_timeout)
        # Deprecated alias: historically used for the PROPOSE wall-clock budget.
        self.round_timeout_seconds = float(self.propose_timeout_seconds)
        self.review_timeout_seconds = float(self.config.get("review_timeout", default_timeout))
        self.rebuttal_timeout_seconds = float(self.config.get("rebuttal_timeout", default_timeout))
        # Per-agent-call request timeout (best-effort; enforced by the LLM client if supported).
        self.call_timeout_seconds = float(self.config.get("call_timeout", default_timeout))

        # Auto-withdraw policy: repeated timeouts/invalid_json should not drag the whole debate.
        self.auto_withdraw_on_call_errors = bool(self.config.get("auto_withdraw_on_call_errors", True))
        self.auto_withdraw_call_error_threshold = int(self.config.get("auto_withdraw_call_error_threshold", 2))
        self.auto_withdraw_call_error_threshold = max(1, self.auto_withdraw_call_error_threshold)
        raw_types = self.config.get("auto_withdraw_call_error_types", ["timeout", "invalid_json"])
        types: List[str] = []
        if isinstance(raw_types, list):
            for t in raw_types:
                s = str(t).strip().lower()
                if s:
                    types.append(s)
        else:
            s = str(raw_types).strip().lower()
            if s:
                types.append(s)
        self.auto_withdraw_call_error_types = types or ["timeout", "invalid_json"]
        self.auto_withdraw_status = str(self.config.get("auto_withdraw_status", "withdrawn")).strip() or "withdrawn"

        # Dynamic ReAct step budgets per phase 
        self.propose_max_react_steps = int(self.config.get("propose_max_react_steps", 5))
        self.review_max_react_steps = int(self.config.get("review_max_react_steps", 3))
        self.rebuttal_max_react_steps = int(self.config.get("rebuttal_max_react_steps", 4))
        self._current_debate_id: Optional[str] = None

    # -------------------------
    # Public API
    # -------------------------

    def start_debate(
        self,
        components: List[str],
        initial_prompt: Optional[str] = None,
        reaction_type: Optional[str] = None,
    ) -> GraphDebateResult:
        start_time = time.time()
        debate_id = make_debate_id("langgraph", components, reaction_type)
        self._current_debate_id = debate_id

        if initial_prompt is None:
            initial_prompt = build_initial_debate_prompt(components, reaction_type)

        logger.info(
            "langgraph_debate_start",
            extra={
                "event": "langgraph.debate.start",
                "debate_id": debate_id,
                "components": components,
                "reaction_type": reaction_type,
                "max_rounds": self.max_rounds,
                "no_response_threshold": self.no_response_threshold,
                "max_reviews_per_target": self.max_reviews_per_target,
                "max_concurrency": self.max_concurrency,
                # New: PROPOSE phase budget (preferred key: propose_timeout).
                "propose_timeout_seconds": self.propose_timeout_seconds,
                # Deprecated alias (kept for existing log consumers).
                "round_timeout_seconds": self.round_timeout_seconds,
                "review_timeout_seconds": self.review_timeout_seconds,
                "rebuttal_timeout_seconds": self.rebuttal_timeout_seconds,
                "call_timeout_seconds": self.call_timeout_seconds,
                "propose_max_react_steps": self.propose_max_react_steps,
                "review_max_react_steps": self.review_max_react_steps,
                "rebuttal_max_react_steps": self.rebuttal_max_react_steps,
            },
        )

        # 1) Propose
        proposals = self._run_propose_phase(components, reaction_type, initial_prompt)
        logger.info(
            "langgraph_propose_done",
            extra={
                "event": "langgraph.propose.done",
                "debate_id": debate_id,
                "proposal_ids": sorted(list(proposals.keys())),
            },
        )

        # 2) Review/Rebuttal loop
        debate_history: List[Dict[str, Any]] = []
        for pid, p in proposals.items():
            debate_history.append(
                {
                    "type": "propose",
                    "proposal_id": pid,
                    "agent_name": p.agent_name,
                    "claim": p.claim,
                    "trajectory": p.propose_trajectory.to_dict() if p.propose_trajectory else None,
                }
            )
        consensus_reached = False

        for round_number in range(1, self.max_rounds + 1):
            review_deadline = time.time() + self.review_timeout_seconds
            active_ids = [pid for pid, p in proposals.items() if p.status == "active"]
            logger.info(
                "langgraph_round_start",
                extra={
                    "event": "langgraph.round.start",
                    "debate_id": debate_id,
                    "round": round_number,
                    "active_proposals": sorted(active_ids),
                },
            )
            if len(active_ids) <= 1:
                consensus_reached = True
                break

            round_reviews, review_calls = self._run_review_round(
                round_number, proposals, components, reaction_type, deadline_ts=review_deadline
            )
            rebuttal_deadline = time.time() + self.rebuttal_timeout_seconds
            round_rebuttals, rebuttal_calls = self._run_rebuttal_round(
                round_number, proposals, round_reviews, components, reaction_type, deadline_ts=rebuttal_deadline
            )

            # Adjudicate and update proposal states
            round_changed, round_consensus = self._rule_adjudicate(
                proposals=proposals,
                round_number=round_number,
                round_reviews=round_reviews,
                round_rebuttals=round_rebuttals,
                round_review_calls=review_calls,
            )

            status_counts: Dict[str, int] = {}
            for p in proposals.values():
                status_counts[p.status] = status_counts.get(p.status, 0) + 1

            logger.info(
                "langgraph_round_end",
                extra={
                    "event": "langgraph.round.end",
                    "debate_id": debate_id,
                    "round": round_number,
                    "changed": round_changed,
                    "consensus": round_consensus,
                    "valid_reviews": sum(1 for r in round_reviews if r.valid),
                    "valid_rebuttals": sum(1 for r in round_rebuttals if r.valid),
                    "status_counts": status_counts,
                },
            )

            debate_history.extend(review_calls)
            debate_history.extend(rebuttal_calls)
            debate_history.extend(self._format_round_history(round_number, round_reviews, round_rebuttals, proposals))

            if round_consensus:
                consensus_reached = True
                break

        elapsed = time.time() - start_time

        surviving = [p for p in proposals.values() if p.status == "active"]
        defeated = [p for p in proposals.values() if p.status == "defeated"]
        withdrawn = [p for p in proposals.values() if p.status == "withdrawn"]

        # If multiple proposals survive, optionally resolve the stalemate deterministically
        # (no additional LLM calls) so downstream consumers get a single final claim.
        winner_proposal_id: Optional[str] = None
        resolution_method: Optional[str] = None
        resolution_details: Optional[Dict[str, Any]] = None

        final_products, final_performance = self._best_effort_final_fields(surviving)
        resolve_stalemate = _coerce_bool(self.config.get("resolve_stalemate", False))
        stalemate_method = str(self.config.get("stalemate_method", "score") or "score").strip().lower()
        if resolve_stalemate and len(surviving) > 1:
            if stalemate_method != "score":
                logger.warning(
                    "langgraph_stalemate_method_unsupported",
                    extra={
                        "event": "langgraph.stalemate.unsupported_method",
                        "debate_id": debate_id,
                        "method": stalemate_method,
                    },
                )
            else:
                expected_rt = (reaction_type or "UNKNOWN").strip().upper() or "UNKNOWN"
                expected_electrode = _extract_electrode_composition_from_prompt(initial_prompt)
                try:
                    expected_elements, _ = parse_components_with_percent(components or [])
                except Exception:
                    expected_elements = [str(c).strip() for c in (components or []) if str(c).strip()]

                percent_tol = _coerce_float(self.config.get("stalemate_percent_tolerance", 0.05), default=0.05)
                range_strategy = str(self.config.get("stalemate_range_strategy", "conservative") or "conservative").strip().lower()

                winner_proposal_id, final_products, final_performance, resolution_details = self._resolve_stalemate_score(
                    surviving=surviving,
                    expected_reaction_type=expected_rt,
                    expected_electrode_composition=expected_electrode,
                    expected_elements=expected_elements,
                    percent_tolerance=percent_tol,
                    range_strategy=range_strategy,
                )
                resolution_method = "score"

                # Emit a structured event so we can trace why a winner was picked.
                debate_history.append(
                    {
                        "type": "stalemate_resolution",
                        "round": None,
                        "method": resolution_method,
                        "winner_proposal_id": winner_proposal_id,
                        "details": resolution_details,
                    }
                )

        logger.info(
            "langgraph_debate_end",
            extra={
                "event": "langgraph.debate.end",
                "debate_id": debate_id,
                "consensus_reached": consensus_reached,
                "time_elapsed": elapsed,
                "surviving": [p.proposal_id for p in surviving],
                "defeated": [p.proposal_id for p in defeated],
                "withdrawn": [p.proposal_id for p in withdrawn],
            },
        )

        result = GraphDebateResult(
            consensus_reached=consensus_reached,
            final_products=final_products,
            final_performance=final_performance,
            reasoning_trajectory=self._build_reasoning_trajectory(proposals),
            debate_rounds=min(self.max_rounds, self._infer_completed_rounds(debate_history)),
            debate_history=debate_history,
            time_elapsed=elapsed,
            surviving_proposals=[self._proposal_to_dict(p) for p in surviving],
            defeated_proposals=[self._proposal_to_dict(p) for p in defeated],
            withdrawn_proposals=[self._proposal_to_dict(p) for p in withdrawn],
            winner_proposal_id=winner_proposal_id,
            resolution_method=resolution_method,
            resolution_details=resolution_details,
        )

        # Structured artifacts:
        # - transcript jsonl (per event)
        # - full debate json (single file)
        try:
            payload = {
                "debate_id": debate_id,
                "run_id": get_run_id() or None,
                "engine": "langgraph",
                "reaction_type": reaction_type,
                "components": components,
                "result": result.to_dict(),
            }
            paths = write_debate_artifacts(
                debate_id=debate_id,
                engine="langgraph",
                payload=payload,
                transcript_events=debate_history,
            )
            logger.info(
                "langgraph_artifacts_written",
                extra={
                    "event": "langgraph.artifacts.written",
                    "debate_id": debate_id,
                    "full_path": paths.get("full_path"),
                    "transcript_path": paths.get("transcript_path"),
                },
            )
        except Exception:
            logger.exception(
                "langgraph_artifacts_write_failed",
                extra={"event": "langgraph.artifacts.error", "debate_id": debate_id},
            )

        self._current_debate_id = None
        return result

    # -------------------------
    # Phase: Propose
    # -------------------------

    def _run_propose_phase(
        self,
        components: List[str],
        reaction_type: Optional[str],
        prompt: str,
    ) -> Dict[str, ProposalState]:
        proposals: Dict[str, ProposalState] = {}

        for agent in self.agents:
            proposals[agent.agent_id] = ProposalState(proposal_id=agent.agent_id, agent_name=agent.name)

        # Parallelize per-agent propose calls; bound wall-clock time with a phase deadline.
        deadline_ts = time.time() + self.propose_timeout_seconds
        futures: Dict[Future, str] = {}
        call_starts: Dict[str, float] = {}
        ex = ThreadPoolExecutor(max_workers=self.max_concurrency)
        try:
            for agent in self.agents:
                proposal_id = agent.agent_id
                call_starts[proposal_id] = time.time()
                futures[
                    ex.submit(
                        self._run_react_call,
                        agent,
                        prompt,
                        components,  # pass explicitly so agents can enforce component-level guards
                        DEBATE_PROPOSE_SYSTEM_PROMPT,
                        self.propose_max_react_steps,
                        self.call_timeout_seconds,
                    )
                ] = proposal_id

            remaining = max(0.0, deadline_ts - time.time())
            done, not_done = wait(list(futures.keys()), timeout=remaining)

            for fut in done:
                proposal_id = futures[fut]
                agent = next(a for a in self.agents if a.agent_id == proposal_id)
                call_elapsed = time.time() - call_starts.get(proposal_id, time.time())
                response = None
                trajectory = None
                err = None
                try:
                    response, trajectory = fut.result()
                except Exception as e:
                    err = str(e)

                raw = (response.content or "").strip() if response else ""
                proposals[proposal_id].propose_response = raw
                proposals[proposal_id].propose_trajectory = trajectory

                # PROPOSE phase now uses STRICT JSON. Parse + normalize into a compact text claim for downstream prompts.
                parsed, parsed_ok = _parse_json_dict_output(raw)
                proposal_out, schema_ok = _coerce_proposal_output(
                    parsed=parsed,
                    prompt=prompt,
                    components=components,
                    reaction_type=reaction_type,
                    trajectory=trajectory,
                )
                proposal_out, mechanism_sections_ok, auto_confidence_downgraded = _enforce_proposal_mechanism_sections(
                    proposal_out
                )
                proposals[proposal_id].claim = _render_proposal_claim(proposal_out)

                # If the agent failed outright, mark it withdrawn so it doesn't block the debate.
                if err:
                    proposals[proposal_id].status = "withdrawn"

                retrieved_ids = _collect_retrieved_source_ids(trajectory)
                logger.info(
                    "langgraph_propose_call",
                    extra={
                        "event": "langgraph.propose.call",
                        "debate_id": self._current_debate_id,
                        "proposal_id": proposal_id,
                        "agent_name": agent.name,
                        "time_elapsed": call_elapsed,
                        "error": err,
                        "steps": len(getattr(trajectory, "steps", []) or []) if trajectory else 0,
                        "retrieved_source_id_count": len(retrieved_ids),
                        "retrieved_source_ids_preview": sorted(list(retrieved_ids))[:10],
                        "parsed_ok": parsed_ok,
                        "schema_ok": schema_ok,
                        "mechanism_sections_ok": mechanism_sections_ok,
                        "auto_confidence_downgraded": auto_confidence_downgraded,
                        "claim_len": len((proposals[proposal_id].claim or "").strip()),
                    },
                )

            for fut in not_done:
                proposal_id = futures[fut]
                agent = next(a for a in self.agents if a.agent_id == proposal_id)
                fut.cancel()
                proposals[proposal_id].propose_response = ""
                proposals[proposal_id].propose_trajectory = None
                proposals[proposal_id].claim = ""
                proposals[proposal_id].status = "withdrawn"
                logger.warning(
                    "langgraph_propose_timeout",
                    extra={
                        "event": "langgraph.propose.timeout",
                        "debate_id": self._current_debate_id,
                        "proposal_id": proposal_id,
                        "agent_name": agent.name,
                        "timeout_seconds": self.propose_timeout_seconds,
                        "time_elapsed": time.time() - call_starts.get(proposal_id, time.time()),
                    },
                )
        finally:
            # Don't block on slow/stuck threads; per-call request timeouts should stop them eventually.
            ex.shutdown(wait=False, cancel_futures=True)

        return proposals

    def _apply_auto_withdraw_policy(
        self,
        proposals: Dict[str, ProposalState],
        from_id: str,
        phase: str,
        err: Optional[str],
        round_number: int,
        call_history: List[Dict[str, Any]],
    ) -> None:
        """
        Auto-withdraw an agent's proposal after repeated call errors (e.g., timeout/invalid_json).

        This helps debates converge faster by preventing a consistently failing agent from consuming
        the full wall-clock timeout budget every round.
        """
        if not self.auto_withdraw_on_call_errors:
            return
        p = proposals.get(from_id)
        if p is None or p.status != "active":
            return

        err_norm = str(err or "").strip().lower()
        if err_norm in set(self.auto_withdraw_call_error_types):
            p.call_error_streak = int(getattr(p, "call_error_streak", 0) or 0) + 1
            p.last_call_error = err_norm
        else:
            p.call_error_streak = 0
            p.last_call_error = err_norm or None

        if p.call_error_streak < self.auto_withdraw_call_error_threshold:
            return

        p.status = self.auto_withdraw_status
        call_history.append(
            {
                "type": "auto_withdraw",
                "phase": str(phase or ""),
                "round": int(round_number),
                "from_proposal_id": from_id,
                "agent_name": p.agent_name,
                "reason": err_norm or None,
                "streak": p.call_error_streak,
                "threshold": self.auto_withdraw_call_error_threshold,
                "new_status": p.status,
            }
        )
        logger.warning(
            "langgraph_auto_withdraw",
            extra={
                "event": "langgraph.auto_withdraw",
                "debate_id": self._current_debate_id,
                "round": round_number,
                "phase": phase,
                "from_proposal_id": from_id,
                "agent_name": p.agent_name,
                "reason": err_norm,
                "streak": p.call_error_streak,
                "threshold": self.auto_withdraw_call_error_threshold,
                "new_status": p.status,
            },
        )

    # -------------------------
    # Phase: Review
    # -------------------------

    def _assign_review_targets(self, round_number: int, active_ids: List[str]) -> Dict[str, List[str]]:
        """
        Assign review responsibilities so every active proposal is reviewed at least once per round.

        We deterministically assign each reviewer exactly one target (a rotation over active_ids).
        The rotation offset changes by round to vary pairings while preventing self-review.
        """
        n = len(active_ids)
        if n <= 1:
            return {}

        # Offset cycles through 1..n-1 (never 0) so a reviewer is never assigned to itself.
        offset = ((round_number - 1) % (n - 1)) + 1

        assignments: Dict[str, List[str]] = {}
        for i, reviewer_id in enumerate(active_ids):
            target_id = active_ids[(i + offset) % n]
            if target_id == reviewer_id:
                # Should be impossible given the offset selection, but keep a safe fallback.
                target_id = active_ids[(i + 1) % n]
            assignments[reviewer_id] = [target_id]

        return assignments

    def _run_review_round(
        self,
        round_number: int,
        proposals: Dict[str, ProposalState],
        components: List[str],
        reaction_type: Optional[str],
        deadline_ts: Optional[float] = None,
    ) -> Tuple[List[DebateReview], List[Dict[str, Any]]]:
        reviews: List[DebateReview] = []
        call_history: List[Dict[str, Any]] = []

        active_ids = [pid for pid, p in proposals.items() if p.status == "active"]
        assignments = self._assign_review_targets(round_number, active_ids)

        # Pre-build prompts (cheap) then execute calls in parallel.
        task_inputs: Dict[str, Tuple[ReActAgent, str, List[str]]] = {}
        for agent in self.agents:
            from_id = agent.agent_id
            if from_id not in active_ids:
                continue

            # Only assign a subset of targets per reviewer; across all reviewers we cover all proposals.
            targets = assignments.get(from_id, [])
            if not targets:
                continue

            review_prompt = self._build_review_prompt(round_number, from_id, proposals, targets, reaction_type)
            task_inputs[from_id] = (agent, review_prompt, targets)

        phase_deadline = deadline_ts if deadline_ts is not None else (time.time() + self.round_timeout_seconds)
        futures: Dict[Future, str] = {}
        call_starts: Dict[str, float] = {}
        ex = ThreadPoolExecutor(max_workers=self.max_concurrency)
        try:
            for from_id, (agent, prompt, _targets) in task_inputs.items():
                call_starts[from_id] = time.time()
                futures[
                    ex.submit(
                        self._run_react_call,
                        agent,
                        prompt,
                        components,
                        DEBATE_REVIEW_SYSTEM_PROMPT,
                        self.review_max_react_steps,
                        self.call_timeout_seconds,
                    )
                ] = from_id

            remaining = max(0.0, phase_deadline - time.time())
            done, not_done = wait(list(futures.keys()), timeout=remaining)

            for fut in done:
                from_id = futures[fut]
                agent, _prompt, targets = task_inputs[from_id]
                response = None
                trajectory = None
                err = None
                try:
                    response, trajectory = fut.result()
                except Exception as e:
                    err = str(e)

                retrieved_ids = _collect_retrieved_source_ids(trajectory)
                raw_output = (response.content if response else "") or ""

                parsed, parsed_ok = _parse_json_output(raw_output, expected_key="reviews")
                validated, schema_ok = _validate_review_output(parsed)

                # Prevent false consensus: an invalid review output should be treated as an error
                # for the purposes of "0 valid reviews => consensus".
                if err is None and (not parsed_ok or not schema_ok):
                    err = "invalid_json"

                call_history.append(
                    {
                        "type": "review_call",
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "targets": targets,
                        "raw_output": raw_output,
                        "trajectory": trajectory.to_dict() if trajectory else None,
                        "retrieved_source_ids": sorted(retrieved_ids),
                        "parsed_ok": parsed_ok,
                        "schema_ok": schema_ok,
                        "error": err,
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
                )
                self._apply_auto_withdraw_policy(
                    proposals=proposals,
                    from_id=from_id,
                    phase="review",
                    err=err,
                    round_number=round_number,
                    call_history=call_history,
                )

                selected = validated.reviews[: len(targets) * self.max_reviews_per_target]
                valid_count = 0
                invalid_count = 0
                for item_idx, item in enumerate(selected):
                    review_id = f"rev_r{round_number}_{from_id}_{item_idx}"
                    review = self._validate_review_item(
                        review_id=review_id,
                        round_number=round_number,
                        from_id=from_id,
                        item=item,
                        proposals=proposals,
                        retrieved_source_ids=retrieved_ids,
                    )
                    if review.valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                    reviews.append(review)
                    proposals[from_id].sent_reviews.append(review)
                    proposals[review.target_proposal_id].received_reviews.append(review)

                logger.info(
                    "langgraph_review_call",
                    extra={
                        "event": "langgraph.review.call",
                        "debate_id": self._current_debate_id,
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "targets": targets,
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                        "error": err,
                        "retrieved_source_id_count": len(retrieved_ids),
                        "parsed_ok": parsed_ok,
                        "schema_ok": schema_ok,
                        "parsed_reviews": len(validated.reviews),
                        "emitted_reviews": len(selected),
                        "valid_reviews": valid_count,
                        "invalid_reviews": invalid_count,
                    },
                )

            # Timeouts -> no review for that agent this round.
            for fut in not_done:
                from_id = futures[fut]
                agent, _prompt, targets = task_inputs[from_id]
                fut.cancel()
                call_history.append(
                    {
                        "type": "review_call",
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "targets": targets,
                        "raw_output": "",
                        "trajectory": None,
                        "retrieved_source_ids": [],
                        "parsed_ok": False,
                        "schema_ok": False,
                        "error": "timeout",
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
                )
                self._apply_auto_withdraw_policy(
                    proposals=proposals,
                    from_id=from_id,
                    phase="review",
                    err="timeout",
                    round_number=round_number,
                    call_history=call_history,
                )
                logger.warning(
                    "langgraph_review_timeout",
                    extra={
                        "event": "langgraph.review.timeout",
                        "debate_id": self._current_debate_id,
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "agent_name": agent.name,
                    },
                )
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        return reviews, call_history

    # -------------------------
    # Phase: Rebuttal
    # -------------------------

    def _run_rebuttal_round(
        self,
        round_number: int,
        proposals: Dict[str, ProposalState],
        round_reviews: List[DebateReview],
        components: List[str],
        reaction_type: Optional[str],
        deadline_ts: Optional[float] = None,
    ) -> Tuple[List[DebateRebuttal], List[Dict[str, Any]]]:
        rebuttals: List[DebateRebuttal] = []
        call_history: List[Dict[str, Any]] = []

        active_ids = [pid for pid, p in proposals.items() if p.status == "active"]
        valid_reviews_by_target: Dict[str, List[DebateReview]] = {}
        for r in round_reviews:
            if not r.valid:
                continue
            # Rebut against all valid reviews (evidence is optional).
            valid_reviews_by_target.setdefault(r.target_proposal_id, []).append(r)

        # Pre-build prompts then execute calls in parallel.
        task_inputs: Dict[str, Tuple[ReActAgent, str, List[DebateReview]]] = {}
        for agent in self.agents:
            from_id = agent.agent_id
            if from_id not in active_ids:
                continue

            target_reviews = valid_reviews_by_target.get(from_id, [])
            if not target_reviews:
                continue

            rebuttal_prompt = self._build_rebuttal_prompt(round_number, from_id, proposals[from_id], target_reviews, reaction_type)
            task_inputs[from_id] = (agent, rebuttal_prompt, target_reviews)

        phase_deadline = deadline_ts if deadline_ts is not None else (time.time() + self.round_timeout_seconds)
        futures: Dict[Future, str] = {}
        call_starts: Dict[str, float] = {}
        ex = ThreadPoolExecutor(max_workers=self.max_concurrency)
        try:
            for from_id, (agent, prompt, _target_reviews) in task_inputs.items():
                call_starts[from_id] = time.time()
                futures[
                    ex.submit(
                        self._run_react_call,
                        agent,
                        prompt,
                        components,
                        DEBATE_REBUTTAL_SYSTEM_PROMPT,
                        self.rebuttal_max_react_steps,
                        self.call_timeout_seconds,
                    )
                ] = from_id

            remaining = max(0.0, phase_deadline - time.time())
            done, not_done = wait(list(futures.keys()), timeout=remaining)

            for fut in done:
                from_id = futures[fut]
                agent, _prompt, target_reviews = task_inputs[from_id]
                response = None
                trajectory = None
                err = None
                try:
                    response, trajectory = fut.result()
                except Exception as e:
                    err = str(e)

                retrieved_ids = _collect_retrieved_source_ids(trajectory)
                raw_output = (response.content if response else "") or ""

                parsed, parsed_ok = _parse_json_output(raw_output, expected_key="rebuttals")
                validated, schema_ok = _validate_rebuttal_output(parsed)

                # Helpful for debugging + consistent error reporting across phases.
                if err is None and (not parsed_ok or not schema_ok):
                    err = "invalid_json"

                call_record: Dict[str, Any] = {
                    "type": "rebuttal_call",
                    "round": round_number,
                    "from_proposal_id": from_id,
                    "target_review_ids": [r.review_id for r in target_reviews],
                    "raw_output": raw_output,
                    "trajectory": trajectory.to_dict() if trajectory else None,
                    "retrieved_source_ids": sorted(retrieved_ids),
                    "parsed_ok": parsed_ok,
                    "schema_ok": schema_ok,
                    "error": err,
                    "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    # Audits (filled after validation/patching below).
                    "revised_claim_present": bool(validated.revised_claim and validated.revised_claim.strip()),
                    "revised_claim_patched": False,
                    "revised_claim_patch_source_ids": [],
                    "revised_claim_metric_withheld": False,
                    "revised_claim_metric_restored": False,
                    "revised_claim_metric_auto_noted": False,
                }
                call_history.append(call_record)
                self._apply_auto_withdraw_policy(
                    proposals=proposals,
                    from_id=from_id,
                    phase="rebuttal",
                    err=err,
                    round_number=round_number,
                    call_history=call_history,
                )

                valid_count = 0
                invalid_count = 0
                valid_evidence_sids: List[str] = []
                _seen_sid: Set[str] = set()
                for item_idx, item in enumerate(validated.rebuttals):
                    rebuttal_id = f"reb_r{round_number}_{from_id}_{item_idx}"
                    rebuttal = self._validate_rebuttal_item(
                        rebuttal_id=rebuttal_id,
                        round_number=round_number,
                        from_id=from_id,
                        item=item,
                        valid_review_ids={r.review_id for r in target_reviews},
                        retrieved_source_ids=retrieved_ids,
                    )
                    if rebuttal.valid:
                        valid_count += 1
                        for ev in rebuttal.evidence or []:
                            sid = ev.get("source_id")
                            if sid and sid not in _seen_sid:
                                _seen_sid.add(str(sid))
                                valid_evidence_sids.append(str(sid))
                    else:
                        invalid_count += 1
                    rebuttals.append(rebuttal)
                    proposals[from_id].sent_rebuttals.append(rebuttal)

                    # If agent explicitly withdraws, reflect immediately.
                    if rebuttal.valid and rebuttal.response_mode == "withdraw":
                        proposals[from_id].status = "withdrawn"

                # Optional claim revision (evidence is optional; if present and revised_claim lacks source_ids, patch an Evidence line).
                did_revise_claim = False
                revised_claim_patched = False
                patch_sids: List[str] = []
                if validated.revised_claim and validated.revised_claim.strip():
                    revised_claim_text = validated.revised_claim.strip()
                    prev_claim_text = proposals[from_id].claim or ""

                    # Normalize literal "\n" escapes and ensure revised claims remain quantitative.
                    revised_claim_text = _normalize_claim_newlines(revised_claim_text)
                    revised_claim_text, metric_flags = _soft_enforce_revised_claim_metrics(
                        revised_claim_text,
                        prev_claim_text,
                    )
                    call_record["revised_claim_metric_withheld"] = bool(metric_flags.get("withheld_detected"))
                    call_record["revised_claim_metric_restored"] = bool(metric_flags.get("restored_from_prev"))
                    call_record["revised_claim_metric_auto_noted"] = bool(metric_flags.get("auto_note_added"))

                    # Prefer source_ids the agent actually used as evidence; fall back to any retrieved ids.
                    patch_candidates = [sid for sid in valid_evidence_sids if is_valid_chroma_source_id(sid)]
                    if not patch_candidates:
                        patch_candidates = [sid for sid in sorted(retrieved_ids) if is_valid_chroma_source_id(sid)]

                    if patch_candidates and not any(sid in revised_claim_text for sid in patch_candidates):
                        patch_sids = patch_candidates[:3]
                        revised_claim_text = revised_claim_text.rstrip() + "\nEvidence: " + "; ".join(patch_sids)
                        revised_claim_patched = True

                    proposals[from_id].claim = revised_claim_text
                    did_revise_claim = True

                call_record["revised_claim_present"] = bool(validated.revised_claim and validated.revised_claim.strip())
                call_record["revised_claim_patched"] = revised_claim_patched
                call_record["revised_claim_patch_source_ids"] = patch_sids

                logger.info(
                    "langgraph_rebuttal_call",
                    extra={
                        "event": "langgraph.rebuttal.call",
                        "debate_id": self._current_debate_id,
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "target_review_ids": [r.review_id for r in target_reviews],
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                        "error": err,
                        "retrieved_source_id_count": len(retrieved_ids),
                        "parsed_ok": parsed_ok,
                        "schema_ok": schema_ok,
                        "parsed_rebuttals": len(validated.rebuttals),
                        "valid_rebuttals": valid_count,
                        "invalid_rebuttals": invalid_count,
                        "revised_claim": did_revise_claim,
                        "revised_claim_metric_withheld": bool(call_record.get("revised_claim_metric_withheld")),
                        "revised_claim_metric_restored": bool(call_record.get("revised_claim_metric_restored")),
                        "revised_claim_metric_auto_noted": bool(call_record.get("revised_claim_metric_auto_noted")),
                    },
                )

            # Timeouts -> no rebuttals for that agent this round.
            for fut in not_done:
                from_id = futures[fut]
                agent, _prompt, target_reviews = task_inputs[from_id]
                fut.cancel()
                call_history.append(
                    {
                        "type": "rebuttal_call",
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "target_review_ids": [r.review_id for r in target_reviews],
                        "raw_output": "",
                        "trajectory": None,
                        "retrieved_source_ids": [],
                        "parsed_ok": False,
                        "schema_ok": False,
                        "error": "timeout",
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
                )
                self._apply_auto_withdraw_policy(
                    proposals=proposals,
                    from_id=from_id,
                    phase="rebuttal",
                    err="timeout",
                    round_number=round_number,
                    call_history=call_history,
                )
                logger.warning(
                    "langgraph_rebuttal_timeout",
                    extra={
                        "event": "langgraph.rebuttal.timeout",
                        "debate_id": self._current_debate_id,
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "agent_name": agent.name,
                    },
                )
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        return rebuttals, call_history

    # -------------------------
    # Rule Adjudication
    # -------------------------

    def _rule_adjudicate(
        self,
        proposals: Dict[str, ProposalState],
        round_number: int,
        round_reviews: List[DebateReview],
        round_rebuttals: List[DebateRebuttal],
        round_review_calls: List[Dict[str, Any]],
    ) -> Tuple[bool, bool]:
        """
        Returns:
            (changed, consensus)
        """
        changed = False

        active_ids = [pid for pid, p in proposals.items() if p.status == "active"]

        # Valid reviews (with OR without evidence) affect consensus/penalties.
        # Evidence remains OPTIONAL, but if present must be verifiable (validated elsewhere).
        reviews_by_target: Dict[str, List[DebateReview]] = {}
        for r in round_reviews:
            if not r.valid:
                continue
            if r.target_proposal_id not in active_ids:
                continue
            reviews_by_target.setdefault(r.target_proposal_id, []).append(r)

        valid_rebuttals_by_review: Dict[str, List[DebateRebuttal]] = {}
        for reb in round_rebuttals:
            if reb.valid:
                valid_rebuttals_by_review.setdefault(reb.target_review_id, []).append(reb)

        # Consensus: no valid reviews among active proposals
        total_valid_reviews = sum(len(v) for v in reviews_by_target.values())
        if total_valid_reviews == 0:
            # Reset streaks as nobody is being challenged this round.
            for pid in active_ids:
                if proposals[pid].no_response_streak != 0:
                    proposals[pid].no_response_streak = 0
                    changed = True
            # "0 valid reviews => consensus" ONLY if the review phase calls completed cleanly.
            # Otherwise we can get false consensus due to timeouts,
            # exceptions, or invalid JSON being interpreted as "no reviews".
            review_call_has_error = any(str((c or {}).get("error") or "").strip() for c in (round_review_calls or []))
            if review_call_has_error:
                return changed, False
            return changed, True

        for pid in active_ids:
            p = proposals[pid]
            if p.status != "active":
                continue

            target_reviews = reviews_by_target.get(pid, [])
            if not target_reviews:
                if p.no_response_streak != 0:
                    p.no_response_streak = 0
                    changed = True
                continue
            # To record "no response" for a review
            unresolved = []
            for r in target_reviews:
                responses = valid_rebuttals_by_review.get(r.review_id, [])
                if not responses:
                    unresolved.append(r.review_id)
                    continue

                # Any valid non-no_response response is considered a response.
                if not any(resp.response_mode in {"defend", "revise", "withdraw"} for resp in responses):
                    unresolved.append(r.review_id)

            if unresolved:
                p.no_response_streak += 1
                changed = True
            else:
                if p.no_response_streak != 0:
                    p.no_response_streak = 0
                    changed = True

            if p.no_response_streak >= self.no_response_threshold:
                p.status = "defeated"
                changed = True

        # Continue debating if more than one active proposal remains.
        remaining_active = [p for p in proposals.values() if p.status == "active"]
        consensus = len(remaining_active) <= 1
        return changed, consensus

    # -------------------------
    # Validation helpers
    # -------------------------

    def _validate_review_item(
        self,
        review_id: str,
        round_number: int,
        from_id: str,
        item: ReviewItem,
        proposals: Dict[str, ProposalState],
        retrieved_source_ids: Set[str],
    ) -> DebateReview:
        target_id = item.target_proposal_id
        target_step = int(item.target_step_number)

        valid = True
        invalid_reason = None

        target = proposals.get(target_id)
        if not target or target.status != "active":
            valid = False
            invalid_reason = "target_proposal_not_active"
        else:
            steps = (target.propose_trajectory.steps if target.propose_trajectory else [])
            step_numbers = {s.step_number for s in steps}
            if target_step not in step_numbers:
                valid = False
                invalid_reason = "target_step_not_found"

        evidence = [e.model_dump() for e in item.evidence]
        if valid and evidence:
            # If evidence is provided, it must be verifiable: cite at least ONE canonical source_id that was
            # actually retrieved in THIS agent call's trajectory.
            retrieved_norm = {
                normalize_chroma_source_id(sid) for sid in (retrieved_source_ids or set()) if sid
            }
            sids = [e.get("source_id") for e in evidence if e.get("source_id")]
            verifiable = []
            for sid in sids:
                if _is_llm_source_id(sid):
                    verifiable.append(sid)
                    continue
                sid_norm = normalize_chroma_source_id(sid)
                if sid_norm in retrieved_norm and is_valid_chroma_source_id(sid_norm):
                    verifiable.append(sid)
            if not verifiable:
                valid = False
                invalid_reason = "evidence_not_verifiable_in_trajectory"

        return DebateReview(
            review_id=review_id,
            round_number=round_number,
            from_proposal_id=from_id,
            target_proposal_id=target_id,
            target_step_number=target_step,
            flaw_type=item.flaw_type,
            critique=item.critique,
            evidence=evidence,
            valid=valid,
            invalid_reason=invalid_reason,
        )

    def _validate_rebuttal_item(
        self,
        rebuttal_id: str,
        round_number: int,
        from_id: str,
        item: RebuttalItem,
        valid_review_ids: Set[str],
        retrieved_source_ids: Set[str],
    ) -> DebateRebuttal:
        valid = True
        invalid_reason = None

        if item.target_review_id not in valid_review_ids:
            valid = False
            invalid_reason = "unknown_target_review_id"

        mode = (item.response_mode or "").strip().lower()
        if mode not in {"defend", "revise", "withdraw", "no_response"}:
            valid = False
            invalid_reason = "invalid_response_mode"

        evidence = [e.model_dump() for e in item.evidence]
        if valid and mode in {"defend", "revise"}:
            if not str(item.response or "").strip():
                valid = False
                invalid_reason = "empty_response"

        # Evidence is optional for rebuttals, but if provided it must be verifiable.
        if valid and evidence:
            retrieved_norm = {
                normalize_chroma_source_id(sid) for sid in (retrieved_source_ids or set()) if sid
            }
            sids = [e.get("source_id") for e in evidence if e.get("source_id")]
            verifiable = []
            for sid in sids:
                if _is_llm_source_id(sid):
                    verifiable.append(sid)
                    continue
                sid_norm = normalize_chroma_source_id(sid)
                if sid_norm in retrieved_norm and is_valid_chroma_source_id(sid_norm):
                    verifiable.append(sid)
            if not verifiable:
                valid = False
                invalid_reason = "evidence_not_verifiable_in_trajectory"

        return DebateRebuttal(
            rebuttal_id=rebuttal_id,
            round_number=round_number,
            from_proposal_id=from_id,
            target_review_id=item.target_review_id,
            response_mode=mode,
            response=item.response,
            evidence=evidence,
            valid=valid,
            invalid_reason=invalid_reason,
        )

    # -------------------------
    # Prompt builders
    # -------------------------

    def _build_review_prompt(
        self,
        round_number: int,
        reviewer_id: str,
        proposals: Dict[str, ProposalState],
        target_ids: List[str],
        reaction_type: Optional[str],
    ) -> str:
        rt = (reaction_type or "UNKNOWN").strip() or "UNKNOWN"
        parts = [
            f"REVIEW phase (Round {round_number}).",
            f"Target reaction: {rt}",
            "You are assigned to review ONLY the target proposal(s) listed below (do not review yourself).",
            f"Write up to {self.max_reviews_per_target} review item(s) per target proposal.",
            "Evidence is preferred but OPTIONAL.",
            "If you provide evidence, cite at least one verifiable source_id retrieved in THIS call.",
            "If you critique from parametric knowledge, set evidence to a list: \"evidence\": [{\"source_id\": \"llm\"}].",
            "If you cannot find a useful critique within the step budget, return an empty reviews list.",
            "Target a specific step_number that exists in the target's trajectory.",
            "Return STRICT JSON only (follow the schema in the system prompt).",
        ]

        for tid in target_ids:
            t = proposals[tid]
            parts.append("\n--- TARGET PROPOSAL ---")
            parts.append(f"target_proposal_id: {t.proposal_id}")
            parts.append(f"claim:\n{t.claim}\n")
            parts.append("trajectory_steps:")
            steps = t.propose_trajectory.steps if t.propose_trajectory else []
            for s in steps[:6]:
                # Keep short; reviewer can target step_number.
                obs = s.observation or ""
                obs = obs[:300] + ("...(truncated)" if len(obs) > 300 else "")
                parts.append(f"- step_number={s.step_number} action={s.action_name} observation_snippet={obs}")

        return "\n".join(parts)

    def _build_rebuttal_prompt(
        self,
        round_number: int,
        proposal_id: str,
        proposal: ProposalState,
        target_reviews: List[DebateReview],
        reaction_type: Optional[str],
    ) -> str:
        rt = (reaction_type or "UNKNOWN").strip() or "UNKNOWN"
        parts = [
            f"REBUTTAL phase (Round {round_number}).",
            f"Target reaction: {rt}",
            "Respond to EACH review below by its review_id.",
            "If you defend or revise, evidence is preferred but OPTIONAL. If you provide evidence, cite at least one verifiable source_id retrieved in THIS call.",
            "If you revise and you retrieved verifiable evidence, you SHOULD cite at least one source_id (either in evidence or appended as an Evidence line in revised_claim).",
            "Return STRICT JSON only (follow the schema in the system prompt).",
            "\n--- YOUR PROPOSAL ---",
            f"proposal_id: {proposal.proposal_id}",
            f"claim:\n{proposal.claim}",
            "\n--- REVIEWS AGAINST YOU (valid only) ---",
        ]

        for r in target_reviews:
            parts.append(
                f"- review_id={r.review_id} from={r.from_proposal_id} "
                f"target_step={r.target_step_number} flaw_type={r.flaw_type}\n"
                f"  critique: {r.critique}\n"
                f"  evidence_source_ids: {[e.get('source_id') for e in (r.evidence or [])]}"
            )

        return "\n".join(parts)

    # -------------------------
    # Agent call wrapper
    # -------------------------

    def _run_react_call(
        self,
        agent: ReActAgent,
        query: str,
        components: Optional[List[str]] = None,
        system_prompt_override: Optional[str] = None,
        max_steps_override: Optional[int] = None,
        llm_timeout_seconds: Optional[float] = None,
    ):
        # Treat every call as an isolated, memoryless run.
        return agent.generate_response_with_react(
            query=query,
            components=components,
            context=None,
            system_prompt_override=system_prompt_override,
            max_steps_override=max_steps_override,
            llm_timeout_seconds=llm_timeout_seconds,
        )

    # -------------------------
    # Output helpers
    # -------------------------

    @staticmethod
    def _proposal_to_dict(p: ProposalState) -> Dict[str, Any]:
        return {
            "proposal_id": p.proposal_id,
            "agent_name": p.agent_name,
            "status": p.status,
            "no_response_streak": p.no_response_streak,
            "claim": p.claim,
        }

    @staticmethod
    def _format_round_history(
        round_number: int,
        reviews: List[DebateReview],
        rebuttals: List[DebateRebuttal],
        proposals: Dict[str, ProposalState],
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for r in reviews:
            history.append(
                {
                    "type": "review",
                    "round": round_number,
                    **r.__dict__,
                }
            )
        for reb in rebuttals:
            history.append(
                {
                    "type": "rebuttal",
                    "round": round_number,
                    **reb.__dict__,
                }
            )
        for pid, p in proposals.items():
            history.append(
                {
                    "type": "proposal_state",
                    "round": round_number,
                    "proposal_id": pid,
                    "status": p.status,
                    "no_response_streak": p.no_response_streak,
                }
            )
        return history

    @staticmethod
    def _build_reasoning_trajectory(proposals: Dict[str, ProposalState]) -> str:
        lines = ["=== Debate Reasoning Trajectory (LangGraph Mode) ==="]
        for pid, p in proposals.items():
            lines.append(f"\n[Proposal] {pid} status={p.status}")
            if p.propose_trajectory:
                lines.append(p.propose_trajectory.get_trajectory_summary())
        return "\n".join(lines)

    def _resolve_stalemate_score(
        self,
        surviving: List[ProposalState],
        expected_reaction_type: str,
        expected_electrode_composition: str,
        expected_elements: List[str],
        percent_tolerance: float = 0.05,
        range_strategy: str = "conservative",
    ) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        """
        Deterministically pick a single "winner" proposal among multiple survivors.

        Design constraints:
        - No additional LLM calls.
        - Prefer strict element/percentage fidelity (when parseable).
        - Prefer verifiable evidence and single-point performance metrics.
        - Use claim self-reported confidence only as a late tie-breaker.
        """
        expected_rt = (expected_reaction_type or "UNKNOWN").strip().upper() or "UNKNOWN"
        expected_electrode = (expected_electrode_composition or "").strip()
        expected_elems = [str(x).strip() for x in (expected_elements or []) if str(x).strip()]
        tol = max(0.0, float(percent_tolerance or 0.0))

        expected_map = _try_parse_percent_composition_map(expected_electrode)

        candidates: List[Dict[str, Any]] = []
        for p in (surviving or []):
            pid = p.proposal_id
            claim = (p.claim or "").strip()

            claim_electrode = _extract_electrode_composition_from_claim(claim)
            claim_map = _try_parse_percent_composition_map(claim_electrode)
            composition_ok = _composition_matches_expected(
                expected_map=expected_map,
                claim_map=claim_map,
                expected_elements=expected_elems,
                claim_elements=_try_parse_elements_from_composition_text(claim_electrode),
                percent_tolerance=tol,
            )

            claim_rt = _extract_reaction_type_from_claim(claim)
            reaction_type_ok = bool(claim_rt and claim_rt.strip().upper() == expected_rt)

            metric_text = _extract_perf_point_from_claim(claim)
            metric_present = bool(metric_text and str(metric_text).strip())
            metric_single_point_ok = bool(metric_present and not _metric_text_has_range_or_uncertainty(metric_text))

            evidence_ids = _extract_verifiable_source_ids_from_text(claim)
            evidence_count = len(evidence_ids)

            conf_label = _extract_confidence_from_claim(claim)
            conf_rank = _confidence_rank(conf_label)

            sort_key = (
                -int(composition_ok),
                -int(reaction_type_ok),
                -int(metric_present),
                -int(metric_single_point_ok),
                -int(evidence_count),
                -int(conf_rank),
                str(pid or ""),
            )

            candidates.append(
                {
                    "proposal_id": pid,
                    "sort_key": list(sort_key),
                    "composition_ok": composition_ok,
                    "reaction_type_ok": reaction_type_ok,
                    "metric_present": metric_present,
                    "metric_single_point_ok": metric_single_point_ok,
                    "verifiable_evidence_count": evidence_count,
                    "confidence_label": conf_label,
                    "confidence_rank": conf_rank,
                    "extracted": {
                        "reaction_type": claim_rt,
                        "electrode_composition": claim_electrode,
                        "performance_metrics": metric_text,
                        "evidence_source_ids": sorted(evidence_ids),
                    },
                }
            )

        # Deterministic: stable sort by our key; proposal_id is the final tie-breaker.
        candidates_sorted = sorted(candidates, key=lambda x: tuple(x.get("sort_key") or []))
        winner = candidates_sorted[0] if candidates_sorted else None
        winner_id = str((winner or {}).get("proposal_id") or "")

        winner_claim = ""
        for p in (surviving or []):
            if str(p.proposal_id) == winner_id:
                winner_claim = (p.claim or "").strip()
                break

        # Products: CO2RR only; otherwise N/A.
        final_products: Optional[str]
        if expected_rt == "CO2RR":
            final_products = _extract_products_from_claim(winner_claim) or "(missing)"
        else:
            final_products = "N/A"

        # Performance metrics: enforce single-point estimate when possible.
        metric_text = _extract_perf_point_from_claim(winner_claim) or "(missing)"
        metric_text, metric_note = _select_single_point_metric_text(
            metric_text=metric_text,
            reaction_type=expected_rt,
            strategy=range_strategy,
        )

        conf_label = _extract_confidence_from_claim(winner_claim)
        if not conf_label:
            conf_label = "low"
        # If we failed strict composition matching, force low confidence.
        winner_comp_ok = bool((winner or {}).get("composition_ok"))
        if not winner_comp_ok:
            conf_label = "low"

        evidence_ids = _extract_verifiable_source_ids_from_text(winner_claim)
        evidence_list = sorted(evidence_ids)[:5]
        evidence_str = "; ".join(evidence_list) if evidence_list else "llm"

        final_claim_lines = [
            f"Reaction Type: {expected_rt}",
            f"Electrode composition (exactly as provided): {expected_electrode or '(missing)'}",
            f"Metal catalyst elements (explicit): {', '.join(expected_elems) if expected_elems else '(missing)'}",
            f"Products: {final_products or '(missing)'}",
            f"Performance Metrics: {metric_text} (Confidence: {conf_label})",
            f"Evidence: {evidence_str}",
        ]

        auto_notes: List[str] = []
        if metric_note:
            auto_notes.append(metric_note)
        if not evidence_list:
            auto_notes.append("AUTO-NOTE: No verifiable rag:chroma source_id found in winning claim; Evidence set to llm.")
        if not winner_comp_ok:
            auto_notes.append(
                "AUTO-NOTE: No surviving proposal matched the expected electrode composition exactly; "
                "final output forced to expected composition with low confidence."
            )

        if auto_notes:
            final_claim_lines.append("")
            final_claim_lines.extend(auto_notes)

        details: Dict[str, Any] = {
            "expected": {
                "reaction_type": expected_rt,
                "electrode_composition": expected_electrode,
                "elements": expected_elems,
                "percent_tolerance": tol,
                "range_strategy": str(range_strategy or ""),
            },
            "candidates": candidates_sorted,
            "winner_proposal_id": winner_id,
        }

        return winner_id, final_products, "\n".join(final_claim_lines).strip(), details

    @staticmethod
    def _infer_completed_rounds(debate_history: List[Dict[str, Any]]) -> int:
        rounds = {h.get("round") for h in debate_history if h.get("round") is not None}
        rounds = {r for r in rounds if isinstance(r, int) and r > 0}
        return max(rounds) if rounds else 0

    @staticmethod
    def _best_effort_final_fields(surviving: List[ProposalState]) -> Tuple[Optional[str], Optional[str]]:
        # Compatibility: if only one proposal survives, try to show its claim as "final".
        if len(surviving) == 1:
            # We don't strictly parse products/performance here; keep as raw claim.
            return None, surviving[0].claim or None
        return None, None


# =========================
# Parsing helpers (robust JSON)
# =========================


def _parse_json_output(text: str, expected_key: str) -> Tuple[Dict[str, Any], bool]:
    """
    Best-effort extraction of a JSON object from LLM output.

    Returns:
        (parsed, parsed_ok)
    """
    if not text:
        return {expected_key: []}, False

    # Prefer fenced ```json blocks
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    # Fallback: first {...} object with a simple heuristic
    obj = _extract_first_json_object(text)
    if obj is not None:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    return {expected_key: []}, False


def _parse_json_dict_output(text: str) -> Tuple[Dict[str, Any], bool]:
    """
    Best-effort extraction of a JSON object (dict) from LLM output.

    Returns:
        (parsed_dict, parsed_ok)
    """
    if not text:
        return {}, False

    # Prefer fenced ```json blocks
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    obj = _extract_first_json_object(text)
    if obj is not None:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

    return {}, False


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _validate_review_output(parsed: Dict[str, Any]) -> Tuple[ReviewOutput, bool]:
    try:
        return ReviewOutput.model_validate(parsed), True
    except ValidationError:
        # Accept empty on invalid format to avoid crashing the debate.
        return ReviewOutput(reviews=[]), False


def _validate_rebuttal_output(parsed: Dict[str, Any]) -> Tuple[RebuttalOutput, bool]:
    try:
        return RebuttalOutput.model_validate(parsed), True
    except ValidationError:
        return RebuttalOutput(rebuttals=[], revised_claim=None), False


def _extract_electrode_composition_from_prompt(prompt: str) -> str:
    """
    Extract the electrode composition line from the initial PROPOSE prompt.
    """
    q = str(prompt or "")
    if not q:
        return ""
    for pat in [
        r"Electrode composition\s*\(relative %\)\s*:\s*([^\n\r]+)",
        r"Electrode composition\s*\(exactly as provided\)\s*:\s*([^\n\r]+)",
    ]:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return str(m.group(1) or "").strip()
    return ""


def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    # Last resort: non-empty strings -> True.
    return True


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _extract_electrode_composition_from_claim(claim: str) -> str:
    """
    Extract the electrode composition line from a rendered claim.

    Supports:
    - "Electrode composition (exactly as provided): ..."
    - "Electrode composition (relative %): ..."
    - "Electrode composition: ..." (fallback)
    """
    s = str(claim or "")
    if not s:
        return ""
    for pat in [
        r"Electrode composition\s*\(exactly as provided\)\s*:\s*([^\n\r]+)",
        r"Electrode composition\s*\(relative %\)\s*:\s*([^\n\r]+)",
        r"Electrode composition\s*:\s*([^\n\r]+)",
    ]:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return str(m.group(1) or "").strip()
    return ""


def _try_parse_percent_composition_map(composition_text: str) -> Optional[Dict[str, float]]:
    """
    Parse "Ni(69.00%), Co(19.07%), ..." into a normalized {element: percent} map.

    Returns None when the text is missing or not parseable as a full percent composition.
    """
    s = str(composition_text or "").strip()
    if not s:
        return None
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if not tokens:
        return None
    try:
        syms, pcts = parse_components_with_percent(tokens)
    except Exception:
        return None
    if not pcts:
        return None
    total = float(sum([float(x) for x in pcts]))
    if total <= 0.0:
        return None
    out: Dict[str, float] = {}
    for sym, pct in zip(syms, pcts):
        key = str(sym).strip()
        if not key:
            continue
        out[key] = (float(pct) / total) * 100.0
    return out or None


def _try_parse_elements_from_composition_text(composition_text: str) -> List[str]:
    """
    Best-effort element symbol extraction from a composition line.

    Examples:
    - "Co(57.08%), Ni(23.16%), ..." -> ["Co","Ni",...]
    - "CoNiFeCu (atomic ratio 1:1:1:0.5)" -> ["Co","Ni","Fe","Cu"]
    """
    s = str(composition_text or "").strip()
    if not s:
        return []
    found = re.findall(r"([A-Z][a-z]?)", s)
    els: List[str] = []
    seen: Set[str] = set()
    for x in found:
        sym = str(x).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        els.append(sym)
    return els


def _composition_matches_expected(
    expected_map: Optional[Dict[str, float]],
    claim_map: Optional[Dict[str, float]],
    expected_elements: List[str],
    claim_elements: List[str],
    percent_tolerance: float,
) -> bool:
    """
    Strict composition check:
    - If we can parse expected percents, we require claim percents too and compare per-element within tolerance.
    - If expected percents are not parseable, fall back to strict element-set equality when possible.
    """
    tol = max(0.0, float(percent_tolerance or 0.0))

    exp_map = expected_map or None
    if exp_map is not None:
        if not claim_map:
            return False
        if set(exp_map.keys()) != set(claim_map.keys()):
            return False
        for el, exp_pct in exp_map.items():
            got = float(claim_map.get(el, -9999.0))
            if abs(float(exp_pct) - got) > tol:
                return False
        return True

    exp_set = {str(x).strip() for x in (expected_elements or []) if str(x).strip()}
    claim_set = {str(x).strip() for x in (claim_elements or []) if str(x).strip()}
    return bool(exp_set) and exp_set == claim_set


def _extract_reaction_type_from_claim(claim: str) -> str:
    s = str(claim or "")
    if not s:
        return ""
    m = re.search(r"(?im)^\s*Reaction\s*Type\s*:\s*([A-Za-z0-9]+)\s*$", s)
    return str(m.group(1) or "").strip() if m else ""


def _extract_products_from_claim(claim: str) -> str:
    s = str(claim or "")
    if not s:
        return ""
    m = re.search(r"(?im)^\s*Products\s*:\s*([^\n\r]*)", s)
    return str(m.group(1) or "").strip() if m else ""


def _extract_verifiable_source_ids_from_text(text: str) -> Set[str]:
    s = str(text or "")
    if not s:
        return set()
    # Best-effort: collect all "rag:chroma/..." tokens, then normalize + validate.
    raw = re.findall(r"rag:chroma/[^\s;]+", s)
    out: Set[str] = set()
    for sid in raw:
        sid_norm = normalize_chroma_source_id(str(sid))
        if is_valid_chroma_source_id(sid_norm):
            out.add(sid_norm)
    return out


_CONFIDENCE_RANK: Dict[str, int] = {
    "low": 0,
    "medium-low": 1,
    "medium": 2,
    "medium-high": 3,
    "high": 4,
}


def _extract_confidence_from_claim(claim: str) -> str:
    """
    Extract and normalize a confidence label from a claim.

    Accepted labels: low | medium-low | medium | medium-high | high
    Defaults to "low" if missing/unknown.
    """
    s = str(claim or "")
    if not s:
        return "low"
    m = re.search(r"(?i)\bconfidence\s*:\s*([A-Za-z][A-Za-z _-]*)", s)
    if not m:
        return "low"
    raw = str(m.group(1) or "").strip().lower()
    raw = re.sub(r"[\s_]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw)
    raw = raw.strip("-").strip()
    # Strip common trailing punctuation/brackets.
    raw = raw.rstrip(").,;:!?\"'").strip()
    return raw if raw in _CONFIDENCE_RANK else "low"


def _confidence_rank(label: str) -> int:
    return int(_CONFIDENCE_RANK.get(str(label or "").strip().lower(), 0))


_LOWER_IS_BETTER_RTS: Set[str] = {"HER", "OER", "UOR", "HZOR"}
_HIGHER_IS_BETTER_RTS: Set[str] = {"ORR", "HOR", "EOR", "O5H", "CO2RR"}


def _metric_text_has_range_or_uncertainty(metric_text: str) -> bool:
    s = str(metric_text or "")
    if not s:
        return False
    if re.search(r"(?i)\+/-|±", s):
        return True
    # Range separators: "a-b", "a to b", "~", and common unicode dashes.
    sep = r"(?:\bto\b|[\u81f3\u5230\u6bcf]|~|\uFF5E|\u2013|\u2014|\u2212|(?<!\^)-)"
    if re.search(rf"(\d+(?:\.\d+)?)\s*{sep}\s*(\d+(?:\.\d+)?)", s, flags=re.IGNORECASE):
        return True
    return False


def _select_single_point_metric_text(metric_text: str, reaction_type: str, strategy: str = "conservative") -> Tuple[str, Optional[str]]:
    """
    Convert a metric text that contains a numeric range/uncertainty into a single point estimate.

    Returns:
        (metric_text_single_point, auto_note_or_None)
    """
    text = str(metric_text or "").strip()
    if not text:
        return "", None

    rt = str(reaction_type or "").strip().upper()
    strat = str(strategy or "conservative").strip().lower() or "conservative"

    def _is_lower_better() -> bool:
        return rt in _LOWER_IS_BETTER_RTS

    def _is_higher_better() -> bool:
        return rt in _HIGHER_IS_BETTER_RTS

    def _choose_conservative(a: float, b: float) -> float:
        if strat == "mean":
            return (a + b) / 2.0
        if _is_lower_better():
            return max(a, b)
        if _is_higher_better():
            return min(a, b)
        return (a + b) / 2.0

    # +/- uncertainty: "291.5 +/- 0.5 mV"
    pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+/-|±)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if pm:
        base_s, delta_s = pm.group(1), pm.group(2)
        try:
            base = float(base_s)
            delta = abs(float(delta_s))
        except Exception:
            base, delta = None, None
        if base is not None and delta is not None:
            # Conservative: move in the "worse" direction.
            if _is_lower_better():
                chosen = base + delta
            elif _is_higher_better():
                chosen = base - delta
            else:
                chosen = base
            dec = max(len(base_s.split(".", 1)[1]) if "." in base_s else 0, len(delta_s.split(".", 1)[1]) if "." in delta_s else 0)
            chosen_s = f"{chosen:.{dec}f}" if dec > 0 else str(int(round(chosen)))
            new_text = text[: pm.start()] + chosen_s + text[pm.end() :]
            note = "AUTO-NOTE: Detected '+/-' uncertainty; conservative point estimate selected for final output."
            return new_text.strip(), note

    # Range: "305-345 mV" / "305 to 345 mV" / "305~345 mV" / etc.
    sep = r"(?:\bto\b|[\u81f3\u5230\u6bcf]|~|\uFF5E|\u2013|\u2014|\u2212|(?<!\^)-)"
    rng = re.search(rf"(\d+(?:\.\d+)?)\s*{sep}\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if rng:
        a_s, b_s = rng.group(1), rng.group(2)
        try:
            a = float(a_s)
            b = float(b_s)
        except Exception:
            a, b = None, None
        if a is not None and b is not None:
            chosen = _choose_conservative(a, b)
            # Preserve formatting when the chosen value is exactly one endpoint.
            chosen_s: str
            if abs(chosen - a) <= 1e-12:
                chosen_s = a_s
            elif abs(chosen - b) <= 1e-12:
                chosen_s = b_s
            else:
                dec = max(len(a_s.split(".", 1)[1]) if "." in a_s else 0, len(b_s.split(".", 1)[1]) if "." in b_s else 0)
                chosen_s = f"{chosen:.{dec}f}" if dec > 0 else str(int(round(chosen)))
            new_text = text[: rng.start()] + chosen_s + text[rng.end() :]
            note = (
                "AUTO-NOTE: Detected metric range; "
                + ("mean value selected for final output." if strat == "mean" else "conservative bound selected for final output.")
            )
            return new_text.strip(), note

    return text, None


def _coerce_proposal_output(
    parsed: Dict[str, Any],
    prompt: str,
    components: List[str],
    reaction_type: Optional[str],
    trajectory: Optional[ReActTrajectory],
) -> Tuple[ProposalOutput, bool]:
    """
    Normalize a (possibly partial/invalid) PROPOSE STRICT JSON dict into a ProposalOutput.

    We deliberately avoid failing the whole propose phase on minor schema issues; instead we
    patch from coordinator-known task context (reaction_type/components/prompt) and the
    call trajectory (retrieved evidence ids).
    """
    src = parsed if isinstance(parsed, dict) else {}

    # Canonical elements come from the coordinator (task definition); treat the model as best-effort only.
    try:
        elem_syms, _percents = parse_components_with_percent(components or [])
    except Exception:
        elem_syms = [str(c).strip() for c in (components or []) if str(c).strip()]
    elem_syms = [e for e in elem_syms if str(e).strip()]

    rt_from_model = src.get("reaction_type") if isinstance(src.get("reaction_type"), str) else None
    rt_task = str(reaction_type or "").strip()
    rt_task_u = rt_task.upper() if rt_task else ""
    rt_model = str(rt_from_model or "").strip()
    rt_model_u = rt_model.upper() if rt_model else ""

    # Prefer the coordinator-known task reaction type when available; do not let a model-emitted "UNKNOWN"
    # override the real task definition (prevents "UNKNOWN" claims after strict-json fallbacks).
    if rt_task_u and rt_task_u != "UNKNOWN":
        rt = rt_task_u
    else:
        rt = (rt_model_u or rt_task_u or "UNKNOWN").strip() or "UNKNOWN"

    electrode = str(src.get("electrode_composition") or "").strip()
    if not electrode:
        electrode = _extract_electrode_composition_from_prompt(prompt)
    if not electrode and elem_syms:
        electrode = ", ".join(elem_syms)

    # Products only apply to CO2RR; otherwise keep "N/A" stable.
    products = str(src.get("products") or "").strip()
    if rt == "CO2RR":
        if not products:
            products = "(missing)"
    else:
        products = products or "N/A"

    performance_metrics = str(src.get("performance_metrics") or "").strip()
    confidence = str(src.get("confidence") or "").strip() or "low"

    # Always force explicit element listing to avoid "format ping-pong" across agents.
    catalyst_elems = []
    if elem_syms:
        catalyst_elems = elem_syms
    else:
        raw_list = src.get("catalyst_metal_elements")
        if isinstance(raw_list, list):
            for x in raw_list:
                s = str(x).strip()
                if s:
                    catalyst_elems.append(s)

    # Evidence is optional; if missing, seed from retrieved ids (or fall back to llm marker).
    evidence: List[Dict[str, Any]] = []
    raw_evidence = src.get("evidence")
    if isinstance(raw_evidence, list):
        for it in raw_evidence:
            if isinstance(it, str) and it.strip():
                evidence.append({"source_id": it.strip()})
            elif isinstance(it, dict) and it.get("source_id"):
                evidence.append({"source_id": str(it.get("source_id")).strip(), "quote": it.get("quote")})

    if not evidence:
        retrieved = sorted(list(_collect_retrieved_source_ids(trajectory))) if trajectory else []
        retrieved = [str(s).strip() for s in retrieved if str(s).strip()]
        if retrieved:
            evidence = [{"source_id": sid} for sid in retrieved[:3]]
        else:
            evidence = [{"source_id": "llm"}]

    rationale = str(src.get("rationale") or "").strip()

    candidate = {
        "reaction_type": rt,
        "electrode_composition": electrode,
        "catalyst_metal_elements": catalyst_elems,
        "products": products,
        "performance_metrics": performance_metrics,
        "confidence": confidence,
        "evidence": evidence,
        "rationale": rationale,
    }
    try:
        return ProposalOutput.model_validate(candidate), True
    except ValidationError:
        # Last-resort: keep the proposal parseable even if the model returns odd types.
        return (
            ProposalOutput(
                reaction_type=rt,
                electrode_composition=electrode,
                catalyst_metal_elements=catalyst_elems,
                products=products,
                performance_metrics=performance_metrics,
                confidence=confidence,
                evidence=[EvidenceItem(source_id="llm")],
                rationale=rationale,
            ),
            False,
        )


_AUTO_NOTE_MISSING_MECH_SECTIONS = (
    "AUTO-NOTE: Missing required Mismatch/Mechanism/Adjustment sections for cited evidence; confidence downgraded to low."
)

_AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD = (
    "AUTO-NOTE: Revised claim withheld quantitative Performance Metrics; treated as low confidence (metric restored if previously available)."
)


def _normalize_claim_newlines(text: str) -> str:
    """
    Normalize common literal newline escape sequences inside claim text.

    Some agents embed "\\n" in free-text `revised_claim` to avoid JSON literal newlines.
    Coordinator-side patching is line-based, so normalize to real newlines for readability + parsing.
    """
    t = str(text or "")
    if not t:
        return ""
    # Handle double-escaped backslashes (e.g., "\\\\n") conservatively.
    t = t.replace("\\\\r\\\\n", "\\r\\n").replace("\\\\n", "\\n").replace("\\\\r", "\\r")
    return t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def _find_perf_metrics_line(text: str) -> Tuple[Optional[int], Optional[str]]:
    for idx, raw in enumerate((text or "").splitlines()):
        if re.match(r"^\s*Performance Metrics\s*:", raw, flags=re.IGNORECASE):
            return idx, raw
    return None, None


def _perf_line_is_withheld(line: str) -> bool:
    s = str(line or "").strip().lower()
    return ("n/a" in s) or ("not asserted" in s) or ("unknown" in s) or ("tbd" in s)


def _extract_perf_point_from_claim(claim_text: str) -> Optional[str]:
    """
    Extract the best-effort point-estimate text from a claim's Performance Metrics line,
    stripping any existing confidence tag.
    """
    claim_text = _normalize_claim_newlines(claim_text)
    _idx, line = _find_perf_metrics_line(claim_text)
    if not line:
        return None
    m = re.match(r"(?i)^\s*Performance Metrics\s*:\s*(.*)$", line)
    if not m:
        return None
    metric = (m.group(1) or "").strip()
    if not metric:
        return None
    # Remove any "(Confidence: ...)" segment while preserving other parentheses.
    metric = re.sub(r"\(\s*conf(?:idence)?\.?\s*:[^)]*\)", "", metric, flags=re.IGNORECASE).strip()
    if not metric:
        return None
    if metric.strip().lower() in {"(missing)", "missing"}:
        return None
    if _perf_line_is_withheld(metric):
        return None
    return metric


def _soft_enforce_revised_claim_metrics(revised_claim: str, prev_claim: str) -> Tuple[str, Dict[str, bool]]:
    """
    Soft-enforce that revised claims keep a quantitative Performance Metrics line.

    Non-blocking design:
    - If the revised claim withholds metrics (N/A/unknown/TBD) or omits the line, restore from previous claim
      when possible, set Confidence to low, and append an AUTO-NOTE.
    - If the revised claim already contains a numeric estimate, do nothing.
    """
    flags: Dict[str, bool] = {
        "withheld_detected": False,
        "restored_from_prev": False,
        "inserted_placeholder": False,
        "auto_note_added": False,
    }

    text = str(revised_claim or "").strip()
    if not text:
        return text, flags

    idx, line = _find_perf_metrics_line(text)
    withheld = False
    if line is None:
        withheld = True
    elif _perf_line_is_withheld(line):
        withheld = True

    if not withheld:
        return text, flags

    flags["withheld_detected"] = True

    prev_metric = _extract_perf_point_from_claim(prev_claim or "")
    if prev_metric:
        flags["restored_from_prev"] = True
        new_perf = f"Performance Metrics: {prev_metric} (Confidence: low)"
    else:
        flags["inserted_placeholder"] = True
        new_perf = "Performance Metrics: (missing) (Confidence: low)"

    lines = text.splitlines()
    if idx is None:
        insert_at = None
        for i, raw in enumerate(lines):
            if re.match(r"^\s*Products\s*:", raw, flags=re.IGNORECASE):
                insert_at = i + 1
                break
        if insert_at is None:
            insert_at = len(lines)
        lines.insert(insert_at, new_perf)
    else:
        lines[idx] = new_perf

    patched = "\n".join(lines).strip()
    if _AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD not in patched:
        sep = "\n" if patched else ""
        patched = (patched + sep + _AUTO_NOTE_REVISED_CLAIM_METRIC_WITHHELD).strip()
        flags["auto_note_added"] = True

    return patched, flags


def _enforce_proposal_mechanism_sections(p: ProposalOutput) -> Tuple[ProposalOutput, bool, bool]:
    """
    Enforce a lightweight contract when the proposal cites verifiable evidence.

    Design goal:
    - Do NOT force the agent to spend extra ReAct steps fixing formatting.
    - Deterministically downgrade confidence + annotate rationale for later review.
    """
    out = p.model_copy(deep=True)
    has_verifiable = any(not _is_llm_source_id(e.source_id) for e in (out.evidence or []))
    if not has_verifiable:
        return out, True, False

    rationale = str(out.rationale or "")
    # Normalize literal "\n" escapes (models sometimes double-escape) so the check isn't format-fragile.
    check_text = rationale.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    required_pats = [
        r"(?i)\bMismatch\s*:",
        r"(?i)\bMechanism\s*:",
        r"(?i)\bAdjustment\s*:",
    ]
    sections_ok = all(re.search(pat, check_text) is not None for pat in required_pats)

    auto_downgraded = False
    if not sections_ok:
        if str(out.confidence or "").strip().lower() != "low":
            out.confidence = "low"
            auto_downgraded = True
        # Idempotent append for multi-pass safety.
        if _AUTO_NOTE_MISSING_MECH_SECTIONS not in rationale:
            sep = "\n" if rationale.strip() else ""
            out.rationale = (rationale.strip() + sep + _AUTO_NOTE_MISSING_MECH_SECTIONS).strip()
        else:
            out.rationale = rationale.strip()
    else:
        out.rationale = rationale.strip()

    return out, sections_ok, auto_downgraded


def _render_proposal_claim(p: ProposalOutput, max_rationale_chars: int = 1200) -> str:
    rt = (p.reaction_type or "").strip() or "UNKNOWN"
    electrode = (p.electrode_composition or "").strip() or "(missing)"
    elems = ", ".join([str(x).strip() for x in (p.catalyst_metal_elements or []) if str(x).strip()])
    if not elems:
        elems = "(missing)"
    products = (p.products or "").strip() or "N/A"
    perf = (p.performance_metrics or "").strip() or "(missing)"
    conf = (p.confidence or "").strip() or "low"

    sids = [str(e.source_id).strip() for e in (p.evidence or []) if getattr(e, "source_id", None)]
    sids = [sid for sid in sids if sid]
    evidence_line = "Evidence: " + ("; ".join(sids[:5]) if sids else "llm")

    rationale = (p.rationale or "").strip()
    if max_rationale_chars and len(rationale) > int(max_rationale_chars):
        rationale = rationale[: int(max_rationale_chars)].rstrip() + "...(truncated)"

    lines = [
        f"Reaction Type: {rt}",
        f"Electrode composition (exactly as provided): {electrode}",
        f"Metal catalyst elements (explicit): {elems}",
        f"Products: {products}",
        f"Performance Metrics: {perf} (Confidence: {conf})",
        evidence_line,
    ]
    if rationale:
        lines.extend(["", "Rationale:", rationale])
    return "\n".join([ln for ln in lines if ln is not None]).strip()


def _collect_retrieved_source_ids(trajectory: Optional[ReActTrajectory]) -> Set[str]:
    if trajectory is None:
        return set()
    sids: Set[str] = set()
    for step in trajectory.steps:
        # New: one ACTION step can contain multiple tool calls.
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") not in {"search_literature", "fetch_literature_chunk"}:
                continue
            data = getattr(call, "observation_data", None) or []
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)

        # Backward-compatible fallback (legacy single-tool steps).
        if getattr(step, "action_name", "") in {"search_literature", "fetch_literature_chunk"} and isinstance(getattr(step, "observation_data", None), list):
            data = getattr(step, "observation_data", None) or []
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)
    return sids

def _is_llm_source_id(source_id: str) -> bool:
    """
    Special-case evidence marker for parametric/internal knowledge.

    This is intentionally NOT a canonical rag:chroma/... source id.
    """
    return str(source_id or "").strip().lower() == "llm"
