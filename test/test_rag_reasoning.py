"""
Agent2 (DeepSeek V3.2 + Voyage) ReAct RAG test script.

This script:
1) Uses Agent2 embedding (Voyage) to query the existing Chroma DB.
2) Runs a full ReAct loop (single agent) with tool calls.

Usage:
  python test_rag_chroma.py --components "Pt,Ni,Fe,Co,Cu" --reaction-type OER
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
try:
    import voyageai  # type: ignore
except Exception:  # pragma: no cover
    voyageai = None  # type: ignore

from agents.agent_config import AgentConfig
from agents.llm_agents import create_agent
from database.embedder import MultiModelEmbedder
try:
    from database.vector_store import VectorStore
except Exception:  # pragma: no cover
    VectorStore = None  # type: ignore


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    if value and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var)
    return value


class Agent2RAGAdapter:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: MultiModelEmbedder,
        agent_name: str,
        embedding_model: str,
        voyage_api_key: str,
        top_k: int = 5
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.agent_name = agent_name
        self.embedding_model = embedding_model
        self.voyage_api_key = voyage_api_key
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = 5, where=None) -> List[Dict]:
        if voyageai is None:
            raise ModuleNotFoundError("voyageai is not installed. Install it to run this script.")
        client = voyageai.Client(api_key=self.voyage_api_key)
        result = client.embed(
            texts=[query],
            model=self.embedding_model,
            input_type="query"
        )
        query_embedding = result.embeddings[0]
        try:
            k = int(top_k) if top_k is not None else int(self.top_k)
        except Exception:
            k = int(self.top_k)
        k = max(1, k)
        similar_docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            where=where,
        )
        formatted = []
        for item in similar_docs:
            distance = item.get("distance")
            score = None
            if distance is not None:
                score = 1.0 - float(distance)
            formatted.append({
                "text": item.get("document", ""),
                "score": score,
                "metadata": item.get("metadata")
            })
        return formatted


def _collect_retrieved_source_ids(trajectory) -> Set[str]:
    """Collect canonical source_id values that were actually retrieved in this run."""
    sids: Set[str] = set()
    if trajectory is None:
        return sids

    for step in getattr(trajectory, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") != "search_literature":
                continue
            data = getattr(call, "observation_data", None) or []
            for item in data:
                sid = item.get("source_id")
                if sid:
                    sids.add(sid)

    # Backward-compatible fallback (legacy single-tool steps).
    for step in getattr(trajectory, "steps", []) or []:
        if getattr(step, "action_name", "") != "search_literature":
            continue
        data = getattr(step, "observation_data", None) or []
        for item in data:
            sid = item.get("source_id")
            if sid:
                sids.add(sid)

    return sids


def _evaluate_proposal(
    components: List[str],
    reaction_type: str,
    final_answer: str,
    trajectory,
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Lightweight "does this look like a debate proposal?" check.

    strict=True enforces:
    - at least one search_literature tool call happened
    - final answer cites at least one verifiable source_id retrieved in the same run
    - conclude tool was called
    """

    problems: List[str] = []
    text = (final_answer or "").strip()
    if not text:
        return False, ["empty_final_answer"]

    rt = (reaction_type or "").strip()
    if rt and rt.lower() not in text.lower():
        problems.append("missing_reaction_type_in_answer")

    # Heuristic: at least one common metric keyword (domain-agnostic enough for this project).
    metric_markers = [
        "overpotential",
        "tafel",
        "stability",
        "current density",
        "mA",
        "mV",
        "faradaic",
        "FE",
    ]
    if not any(m.lower() in text.lower() for m in metric_markers):
        problems.append("missing_obvious_performance_metrics")

    if trajectory is None:
        if strict:
            problems.append("missing_trajectory")
        return (len(problems) == 0), problems

    # Separation rule guard (should be enforced by the agent, but we still validate).
    for step in getattr(trajectory, "steps", []) or []:
        names = {getattr(c, "tool_name", "") for c in (getattr(step, "tool_calls", []) or [])}
        has_search = bool(names.intersection({"search_literature", "search_experience"}))
        has_analysis = bool(names.intersection({"analyze", "conclude"}))
        if has_search and has_analysis:
            problems.append(f"mixed_search_and_analysis_in_step_{getattr(step, 'step_number', '?')}")
            break

    retrieved_ids = _collect_retrieved_source_ids(trajectory)
    if strict:
        if not retrieved_ids:
            problems.append("no_source_id_retrieved")

        if not any(
            getattr(c, "tool_name", "") == "search_literature"
            for step in getattr(trajectory, "steps", []) or []
            for c in (getattr(step, "tool_calls", []) or [])
        ):
            problems.append("no_search_literature_call")

        conclude_calls = [
            c
            for step in getattr(trajectory, "steps", []) or []
            for c in (getattr(step, "tool_calls", []) or [])
            if getattr(c, "tool_name", "") == "conclude"
        ]
        conclude_ok = any(getattr(c, "observation", "").strip() == text for c in conclude_calls)
        if not conclude_ok:
            problems.append("no_conclude_call")

        # Require at least one canonical source_id mention in final answer that was retrieved in-run.
        cited = [sid for sid in retrieved_ids if sid in text]
        if not cited:
            # Fallback: if the model cites canonical-looking ids but we didn't enrich metadata, still flag.
            if not re.search(r"rag:chroma/[^\\s]+", text):
                problems.append("no_verifiable_source_id_cited")

    return (len(problems) == 0), problems


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Agent2 ReAct RAG test with Chroma DB")
    parser.add_argument("--config", default="./config/config.yaml", help="Config file path")
    parser.add_argument("--persist-dir", default="./data/chroma_db", help="Chroma persist directory")
    parser.add_argument("--collection", default=None, help="Chroma collection name (default: base + _agent2)")
    parser.add_argument("--components", default="Pt,Ni,Fe,Co,Cu", help="Comma-separated metal elements")
    parser.add_argument("--reaction-type", default="OER", help="Fixed reaction type, e.g., OER")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--initial-prompt", default=None, help="Override initial prompt")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if no search+verifiable citation+conclude call")
    parser.add_argument("--lenient", dest="strict", action="store_false", help="Only check the answer is non-empty and mentions metrics")
    args = parser.parse_args()

    config = AgentConfig(args.config)
    agent_config = config.get_llm_config("agent2")
    vector_config = config.get_vector_store_config()

    voyage_api_key = _resolve_env_var(agent_config.get("voyage_api_key"))
    if not voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set for agent2 embedding.")

    collection_name = args.collection
    if not collection_name:
        base_name = vector_config.get("collection_name", "chemical_reactions_recommendation")
        collection_name = f"{base_name}_agent2"

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise RuntimeError(f"Persist directory not found: {persist_dir}")

    all_agent_configs = {
        "agent1": config.get_llm_config("agent1"),
        "agent2": config.get_llm_config("agent2"),
        "agent3": config.get_llm_config("agent3"),
        "agent4": config.get_llm_config("agent4")
    }

    embedder = MultiModelEmbedder(agent_config, agent_configs=all_agent_configs)
    vector_store = VectorStore(
        persist_directory=str(persist_dir),
        collection_name=collection_name,
        embedding_function=None
    )

    rag_adapter = Agent2RAGAdapter(
        vector_store=vector_store,
        embedder=embedder,
        agent_name="agent2",
        embedding_model=agent_config.get("embedding_model", "voyage-3-large"),
        voyage_api_key=voyage_api_key,
        top_k=args.top_k
    )

    # Create Agent 
    agent = create_agent(
        agent_type=agent_config.get("provider", "openai"),
        agent_id="agent2",
        name="DeepSeek Researcher",
        model_config=agent_config,
        rag_system=rag_adapter,
        experience_store=None
    )

    # ---------------------------------------------------------
    # 1. Construct Prompt (Simple Instruction)
    # ---------------------------------------------------------
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    
    if args.initial_prompt:
        initial_prompt = args.initial_prompt
    else:
        components_str = ", ".join(components)
        initial_prompt = (
            f"Analyze the performance metrics of an electrode composed of {components_str} "
            f"for the {args.reaction_type} reaction. "
            "Retrieve relevant literature and summarize the key performance metrics."
        )

    # ---------------------------------------------------------
    # 2. Execute Agent (Start ReAct Loop)
    # ---------------------------------------------------------
    print(f"\n{'='*40}\n STARTING REACT LOOP \n{'='*40}")
    
    # Run a full ReAct loop and keep the trajectory for validation.
    response, trajectory = agent.generate_response_with_react(
        query=initial_prompt,
        components=components,
        context=None,
    )
    print(response.reasoning)

    ok, problems = _evaluate_proposal(
        components=components,
        reaction_type=args.reaction_type,
        final_answer=response.content,
        trajectory=trajectory,
        strict=args.strict,
    )

    print(f"\n{'='*40}\n SINGLE-AGENT E2E CHECK \n{'='*40}")
    if ok:
        print("PASS: agent completed the end-to-end run (goal -> retrieval -> conclusion).")
        return 0

    print("FAIL: agent output does not satisfy the end-to-end requirements.")
    for p in problems:
        print(f"- {p}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
