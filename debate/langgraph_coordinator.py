"""
LangGraph-style Debate Coordinator (no external dependency required).

Implements the agreed debate protocol:
1) Propose (one proposal per model/agent)
2) Repeat rounds of (Review -> Rebuttal -> Rule adjudication)
   - A Review must (a) target a specific ReAct step_number AND (b) cite verifiable source_id.
   - If a proposal fails to respond (with verifiable evidence) for 2 consecutive rounds -> defeated.
   - Agents can also voluntarily withdraw their proposal.

Notes:
- We keep agent "memoryless" by storing all debate context in coordinator state.
- "Verifiable source_id" is enforced by requiring cited source_id to appear in the
  agent's own retrieval results within the same message trajectory.
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
from utils.logger import get_run_id, make_debate_id, write_debate_artifacts
from utils.source_id import is_valid_chroma_source_id

logger = logging.getLogger("MAD.debate.langgraph")


# =========================
# Pydantic Schemas
# =========================


class EvidenceItem(BaseModel):
    source_id: str
    quote: Optional[str] = None


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
# Result/Data Structures
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
        self.no_response_threshold = int(self.config.get("no_response_threshold", 2))
        self.max_reviews_per_target = int(self.config.get("max_reviews_per_target", 1))

        # Runtime controls (to bound wall-clock time)
        self.max_concurrency = int(self.config.get("max_concurrency", max(1, len(self.agents))))
        # Backward-compatible: `timeout` exists in config.yaml; treat as the default per-phase wall-clock budget.
        default_timeout = float(self.config.get("timeout", 300))
        self.round_timeout_seconds = float(self.config.get("round_timeout", default_timeout))
        self.review_timeout_seconds = float(self.config.get("review_timeout", default_timeout))
        self.rebuttal_timeout_seconds = float(self.config.get("rebuttal_timeout", default_timeout))
        # Per-agent-call request timeout (best-effort; enforced by the LLM client if supported).
        self.call_timeout_seconds = float(self.config.get("call_timeout", default_timeout))

        # Dynamic ReAct step budgets per phase (defaults requested)
        self.propose_max_react_steps = int(self.config.get("propose_max_react_steps", 8))
        self.review_max_react_steps = int(self.config.get("review_max_react_steps", 3))
        self.rebuttal_max_react_steps = int(self.config.get("rebuttal_max_react_steps", 3))
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

            # If nothing changes and there are no valid reviews, we are done.
            if not round_changed and not any(r.valid for r in round_reviews):
                consensus_reached = True
                break

        elapsed = time.time() - start_time

        surviving = [p for p in proposals.values() if p.status == "active"]
        defeated = [p for p in proposals.values() if p.status == "defeated"]
        withdrawn = [p for p in proposals.values() if p.status == "withdrawn"]

        final_products, final_performance = self._best_effort_final_fields(surviving)

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
        deadline_ts = time.time() + self.round_timeout_seconds
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

                proposals[proposal_id].propose_response = response.content if response else ""
                proposals[proposal_id].propose_trajectory = trajectory
                proposals[proposal_id].claim = (response.content or "").strip() if response else ""

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
                        "claim_len": len((response.content or "").strip()) if response else 0,
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
                        "timeout_seconds": self.round_timeout_seconds,
                        "time_elapsed": time.time() - call_starts.get(proposal_id, time.time()),
                    },
                )
        finally:
            # Don't block on slow/stuck threads; per-call request timeouts should stop them eventually.
            ex.shutdown(wait=False, cancel_futures=True)

        return proposals

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

            review_prompt = self._build_review_prompt(round_number, from_id, proposals, targets)
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
                call_history.append(
                    {
                        "type": "review_call",
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "targets": targets,
                        "raw_output": (response.content if response else ""),
                        "trajectory": trajectory.to_dict() if trajectory else None,
                        "retrieved_source_ids": sorted(retrieved_ids),
                        "error": err,
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
                )

                parsed = _parse_json_output(response.content if response else "", expected_key="reviews")
                validated = _validate_review_output(parsed)

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
                        "error": "timeout",
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
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

            rebuttal_prompt = self._build_rebuttal_prompt(round_number, from_id, proposals[from_id], target_reviews)
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
                call_history.append(
                    {
                        "type": "rebuttal_call",
                        "round": round_number,
                        "from_proposal_id": from_id,
                        "target_review_ids": [r.review_id for r in target_reviews],
                        "raw_output": (response.content if response else ""),
                        "trajectory": trajectory.to_dict() if trajectory else None,
                        "retrieved_source_ids": sorted(retrieved_ids),
                        "error": err,
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
                )

                parsed = _parse_json_output(response.content if response else "", expected_key="rebuttals")
                validated = _validate_rebuttal_output(parsed)

                # Optional claim revision
                did_revise_claim = False
                if validated.revised_claim and validated.revised_claim.strip():
                    proposals[from_id].claim = validated.revised_claim.strip()
                    did_revise_claim = True

                valid_count = 0
                invalid_count = 0
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
                    else:
                        invalid_count += 1
                    rebuttals.append(rebuttal)
                    proposals[from_id].sent_rebuttals.append(rebuttal)

                    # If agent explicitly withdraws, reflect immediately.
                    if rebuttal.valid and rebuttal.response_mode == "withdraw":
                        proposals[from_id].status = "withdrawn"

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
                        "parsed_rebuttals": len(validated.rebuttals),
                        "valid_rebuttals": valid_count,
                        "invalid_rebuttals": invalid_count,
                        "revised_claim": did_revise_claim,
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
                        "error": "timeout",
                        "time_elapsed": time.time() - call_starts.get(from_id, time.time()),
                    }
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
    ) -> Tuple[bool, bool]:
        """
        Returns:
            (changed, consensus)
        """
        changed = False

        active_ids = [pid for pid, p in proposals.items() if p.status == "active"]

        valid_reviews_by_target: Dict[str, List[DebateReview]] = {}
        for r in round_reviews:
            if r.valid and r.target_proposal_id in active_ids:
                valid_reviews_by_target.setdefault(r.target_proposal_id, []).append(r)

        valid_rebuttals_by_review: Dict[str, List[DebateRebuttal]] = {}
        for reb in round_rebuttals:
            if reb.valid:
                valid_rebuttals_by_review.setdefault(reb.target_review_id, []).append(reb)

        # Consensus: no valid reviews among active proposals
        total_valid_reviews = sum(len(v) for v in valid_reviews_by_target.values())
        if total_valid_reviews == 0:
            # Reset streaks as nobody is being challenged this round.
            for pid in active_ids:
                if proposals[pid].no_response_streak != 0:
                    proposals[pid].no_response_streak = 0
                    changed = True
            return changed, True

        for pid in active_ids:
            p = proposals[pid]
            if p.status != "active":
                continue

            target_reviews = valid_reviews_by_target.get(pid, [])
            if not target_reviews:
                if p.no_response_streak != 0:
                    p.no_response_streak = 0
                    changed = True
                continue

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
        if valid:
            if not evidence:
                valid = False
                invalid_reason = "missing_evidence"
            else:
                # Evidence must be verifiable: cite at least ONE canonical source_id that was
                # actually retrieved in THIS agent call's trajectory.
                sids = [e.get("source_id") for e in evidence if e.get("source_id")]
                verifiable = [
                    sid
                    for sid in sids
                    if sid in retrieved_source_ids and is_valid_chroma_source_id(sid)
                ]
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
            if not evidence:
                valid = False
                invalid_reason = "missing_evidence"
            else:
                sids = [e.get("source_id") for e in evidence if e.get("source_id")]
                verifiable = [
                    sid
                    for sid in sids
                    if sid in retrieved_source_ids and is_valid_chroma_source_id(sid)
                ]
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
    ) -> str:
        parts = [
            f"REVIEW phase (Round {round_number}).",
            "You are assigned to review ONLY the target proposal(s) listed below (do not review yourself).",
            f"Write up to {self.max_reviews_per_target} review item(s) per target proposal, and include AT LEAST 1 item per target.",
            "Target a specific step_number that exists in the target's trajectory.",
            "Use `search_rag` and cite at least one verifiable source_id.",
            "Return STRICT JSON only (follow the schema in the system prompt).",
        ]

        for tid in target_ids:
            t = proposals[tid]
            parts.append("\n--- TARGET PROPOSAL ---")
            parts.append(f"target_proposal_id: {t.proposal_id}")
            parts.append(f"claim:\n{t.claim}\n")
            parts.append("trajectory_steps:")
            steps = t.propose_trajectory.steps if t.propose_trajectory else []
            for s in steps[:8]:
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
    ) -> str:
        parts = [
            f"REBUTTAL phase (Round {round_number}).",
            "Respond to EACH review below by its review_id.",
            "If you defend or revise, use `search_rag` and cite at least one verifiable source_id.",
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


def _parse_json_output(text: str, expected_key: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from LLM output.
    """
    if not text:
        return {expected_key: []}

    # Prefer fenced ```json blocks
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: first {...} object with a simple heuristic
    obj = _extract_first_json_object(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except Exception:
            pass

    return {expected_key: []}


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


def _validate_review_output(parsed: Dict[str, Any]) -> ReviewOutput:
    try:
        return ReviewOutput.model_validate(parsed)
    except ValidationError:
        # Accept empty on invalid format to avoid crashing the debate.
        return ReviewOutput(reviews=[])


def _validate_rebuttal_output(parsed: Dict[str, Any]) -> RebuttalOutput:
    try:
        return RebuttalOutput.model_validate(parsed)
    except ValidationError:
        return RebuttalOutput(rebuttals=[], revised_claim=None)


def _collect_retrieved_source_ids(trajectory: Optional[ReActTrajectory]) -> Set[str]:
    if trajectory is None:
        return set()
    sids: Set[str] = set()
    for step in trajectory.steps:
        # New: one ACTION step can contain multiple tool calls.
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") != "search_rag":
                continue
            data = getattr(call, "observation_data", None) or []
            if not isinstance(data, list):
                continue
            for item in data:
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)

        # Backward-compatible fallback (legacy single-tool steps).
        if getattr(step, "action_name", "") == "search_rag" and isinstance(getattr(step, "observation_data", None), list):
            data = getattr(step, "observation_data", None) or []
            for item in data:
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)
    return sids
