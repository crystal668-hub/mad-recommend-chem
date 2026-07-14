""" 
E2E LangGraph debate test (4 agents) using an existing Chroma DB.

What this script does:
- Binds agent1-agent4 to collections in the existing ./data/chroma_db (no rebuild).
- Runs a LangGraph-style debate (propose -> review -> rebuttal -> adjudication).
- Ensures agents use the `search_literature` tool (which queries Chroma) to retrieve evidence.

Typical usage:
  python .\test\test_debate_langgraph_rag_chroma.py --reaction-type OER --components "Pt,Pd,Ni,Fe,Co"
  python .\test\test_debate_langgraph_rag_chroma.py --reaction-type OER --components "Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)"

Notes:
- This script calls external LLM + embedding APIs; make sure your .env has the needed keys.
- It will FAIL fast if the expected Chroma collections are not present.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Ensure repo root is importable when running this script from arbitrary cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from agents.agent_config import AgentConfig
from agents.llm_agents import create_agent
from debate.langgraph_coordinator import LangGraphDebateCoordinator
from prompts.debate_phase_prompts import build_initial_debate_prompt
from utils.electrode_composition import parse_components_with_percent, build_electrode_composition
from utils.helpers import parse_component_string
from utils.logger import setup_logging
from utils.source_id import is_valid_chroma_source_id


@dataclass
class AgentBinding:
    agent_id: str
    agent_name: str
    provider_hint: str
    collection_name: str


class ChromaRAGAdapter:
    """Minimal RAG adapter compatible with ReActAgent.search_literature.

    Supports:
    - `top_k` (n_results)
    - `where` metadata filter by 'reaction_type'
    """

    def __init__(
        self,
        collection: Any,
        embedder: Any,
        agent_name: str,
        collection_name: str,
        top_k: int = 5,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.agent_name = agent_name
        self.collection_name = collection_name
        self.top_k = int(top_k)

    def retrieve(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        try:
            k = int(top_k) if top_k is not None else int(self.top_k)
        except Exception:
            k = int(self.top_k)
        k = max(1, k)

        # Use the SAME embedding profile used to build this collection.
        query_embedding = self.embedder.embed_text(query, agent_name=self.agent_name)

        # Query Chroma with explicit embeddings (avoid any default embedding_function mismatch).
        try:
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except TypeError:
            # Backward compatibility for older Chroma versions (varying support for include/where).
            try:
                raw = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=where,
                )
            except TypeError:
                raw = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                )

        docs = (raw.get("documents") or [[]])[0] or []
        metas = (raw.get("metadatas") or [[]])[0] or []
        dists = (raw.get("distances") or [[]])[0] or []

        results: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None

            score = None
            if dist is not None:
                try:
                    score = 1.0 - float(dist)
                except Exception:
                    score = None

            results.append({
                "text": doc or "",
                "score": score,
                "metadata": meta or {},
            })

        return results


def _missing_required_envs(config: AgentConfig, embedding_agents_needed: Set[str]) -> List[str]:
    needed: List[str] = []

    # LLM keys
    for agent_id in ["agent1", "agent2", "agent3", "agent4"]:
        cfg = config.get_llm_config(agent_id)
        api_key = cfg.get("api_key")
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env = api_key[2:-1]
            if not os.environ.get(env):
                needed.append(env)

    # Extra embedding key for Voyage (only required if we query with agent2's embedding profile).
    if "agent2" in (embedding_agents_needed or set()):
        cfg2 = config.get_llm_config("agent2")
        vkey = cfg2.get("voyage_api_key")
        if vkey and vkey.startswith("${") and vkey.endswith("}"):
            env = vkey[2:-1]
            if not os.environ.get(env):
                needed.append(env)

    # Optional separate embedding API keys (e.g., agent chats via OpenRouter but embeds via Aliyun).
    for agent_id in (embedding_agents_needed or set()):
        if agent_id == "agent2":
            continue  # handled above (Voyage)
        cfg_e = config.get_llm_config(agent_id)
        ekey = cfg_e.get("embedding_api_key") or cfg_e.get("emb_api_key")
        if ekey and ekey.startswith("${") and ekey.endswith("}"):
            env = ekey[2:-1]
            if not os.environ.get(env):
                needed.append(env)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for x in needed:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _list_collection_names(client) -> Set[str]:
    names: Set[str] = set()
    try:
        cols = client.list_collections() or []
    except Exception:
        return names

    for c in cols:
        n = getattr(c, "name", None)
        if n:
            names.add(str(n))
    return names


def _pick_collection_name(base: str, agent_id: str, existing: Set[str]) -> str:
    # Primary naming (matches build_vector_db.py): <base>_<agentN>
    candidates: List[str]

    if agent_id == "agent1":
        candidates = [f"{base}_agent1", f"{base}_openai"]
    elif agent_id == "agent2":
        candidates = [f"{base}_agent2", f"{base}_deepseek"]
    elif agent_id == "agent3":
        candidates = [f"{base}_agent3", f"{base}_gemini", f"{base}_google"]
    elif agent_id == "agent4":
        candidates = [f"{base}_agent4", f"{base}_qwen"]
    else:
        candidates = [f"{base}_{agent_id}"]

    for name in candidates:
        if name in existing:
            return name

    # If none exists, return the primary candidate (the caller will error with a helpful message).
    return candidates[0]


def _infer_embedding_agent_from_collection(collection_name: str) -> Optional[str]:
    """
    Best-effort inference for which embedding profile likely built a collection.

    Supports both naming styles:
    - build_vector_db.py: <base>_agentN
    - older code: <base>_<provider>
    """
    name = (collection_name or "").strip()
    if not name or "_" not in name:
        return None
    suffix = name.rsplit("_", 1)[-1].strip().lower()

    if suffix in {"agent1", "agent2", "agent3", "agent4"}:
        return suffix

    provider_map = {
        "openai": "agent1",
        "deepseek": "agent2",
        "gemini": "agent3",
        "google": "agent3",
        "qwen": "agent4",
    }
    return provider_map.get(suffix)


def _iter_tool_calls_from_trajectory_dict(traj: Optional[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    if not traj:
        return []
    steps = traj.get("steps") or []
    for step in steps:
        for call in (step.get("tool_calls") or []):
            if isinstance(call, dict):
                yield call


def _collect_search_literature_stats(history: List[Dict[str, Any]]) -> Tuple[int, Set[str]]:
    total_calls = 0
    source_ids: Set[str] = set()

    for evt in history or []:
        traj = evt.get("trajectory")
        if not isinstance(traj, dict):
            continue
        for call in _iter_tool_calls_from_trajectory_dict(traj):
            if call.get("tool_name") != "search_literature":
                continue
            total_calls += 1
            data = call.get("observation_data") or []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("source_id")
                    if sid:
                        source_ids.add(str(sid))

    return total_calls, source_ids


def main() -> int:
    load_dotenv()

    # Lazy imports so unit-test discovery doesn't hard-fail when optional E2E deps are missing.
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional dependency `chromadb` required for this E2E script.\n"
            "Install project requirements (including chromadb extras) before running this script."
        ) from e

    try:
        from database.embedder import MultiModelEmbedder
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import MultiModelEmbedder (embedding dependencies may be missing, e.g. voyageai).\n"
            "Install project requirements before running this script."
        ) from e

    parser = argparse.ArgumentParser(description="E2E LangGraph debate test with 4 agents + existing Chroma DB")
    parser.add_argument("--config", default="./config/config.yaml", help="Config file path")
    parser.add_argument("--persist-dir", default=None, help="Chroma persist directory (default: from config)")
    parser.add_argument("--base-collection", default=None, help="Base collection name (default: from config)")
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Bind ALL agents to this single existing Chroma collection (overrides --base-collection auto-pick). "
            "Useful when you only built one collection (e.g. *_agent2)."
        ),
    )
    parser.add_argument(
        "--embedding-agent",
        default=None,
        choices=["agent1", "agent2", "agent3", "agent4"],
        help=(
            "When using --collection, choose which agent embedding profile to use for query embeddings "
            "(default: inferred from collection name, else agent2)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval for search_literature")
    parser.add_argument(
        "--components",
        default="Pt,Pd,Ni,Fe,Co",
        help=(
            "Exactly 5 metal components. Accepts symbols only (e.g., Pt,Pd,Ni,Fe,Co) "
            "or symbols with relative percentages (e.g., Ni(69.00%), Co(19.07%), ...). "
            "Separators: comma / Chinese comma / semicolon / Chinese semicolon / '、'."
        ),
    )
    parser.add_argument("--reaction-type", default="OER", help="Reaction type, e.g., OER")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True, help="Fail if no search_literature calls / no verifiable source_id")
    parser.add_argument("--lenient", dest="strict", action="store_false", help="Only check debate completes")
    args = parser.parse_args()

    # Resolve config path relative to repo root (so this script is runnable from arbitrary cwd).
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    cfg = AgentConfig(str(config_path))

    # Force logs into <repo_root>/logs regardless of cwd.
    log_cfg = (cfg.config or {}).setdefault("logging", {})
    log_file = str(log_cfg.get("log_file", "./logs/system.log"))
    run_dir = str(log_cfg.get("run_dir", "./logs/runs"))
    if log_file and not Path(log_file).is_absolute():
        log_cfg["log_file"] = str((PROJECT_ROOT / log_file).resolve())
    if run_dir and not Path(run_dir).is_absolute():
        log_cfg["run_dir"] = str((PROJECT_ROOT / run_dir).resolve())

    setup_logging(cfg.config, run_id=f"test_langgraph_{int(time.time())}")
    vector_cfg = cfg.get_vector_store_config() or {}

    persist_dir = Path(args.persist_dir or vector_cfg.get("persist_directory", "./data/chroma_db"))
    base_collection = args.base_collection or vector_cfg.get("collection_name", "chemical_reactions_recommendation")

    if not persist_dir.exists():
        raise RuntimeError(f"Persist directory not found: {persist_dir}")

    # Open Chroma DB and bind each agent to its (existing) collection.
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )
    existing = _list_collection_names(client)

    shared_collection = (args.collection or "").strip() or None
    shared_embedding_agent = (args.embedding_agent or "").strip() or None
    if shared_collection:
        if shared_collection not in existing:
            sample = sorted(existing)[:20]
            raise RuntimeError(
                "Expected Chroma collection not found.\n"
                f"- missing: {shared_collection}\n"
                f"- persist_dir: {persist_dir}\n"
                f"- available (first 20): {sample}\n"
            )
        if not shared_embedding_agent:
            shared_embedding_agent = _infer_embedding_agent_from_collection(shared_collection) or "agent2"

    embedding_agents_needed: Set[str]
    if shared_collection:
        embedding_agents_needed = {shared_embedding_agent}
    else:
        embedding_agents_needed = {"agent1", "agent2", "agent3", "agent4"}

    missing_envs = _missing_required_envs(cfg, embedding_agents_needed)
    if missing_envs:
        raise RuntimeError(
            "Missing required environment variables for API access: " + ", ".join(missing_envs)
        )

    # Build embedder with full agent profiles.
    all_agent_configs = {
        "agent1": cfg.get_llm_config("agent1"),
        "agent2": cfg.get_llm_config("agent2"),
        "agent3": cfg.get_llm_config("agent3"),
        "agent4": cfg.get_llm_config("agent4"),
    }
    embedder = MultiModelEmbedder(all_agent_configs["agent1"], agent_configs=all_agent_configs)

    agent_specs: List[Tuple[str, str, str]] = [
        ("agent1", "GPT Researcher", "openai"),
        ("agent2", "DeepSeek Researcher", "deepseek"),
        ("agent3", "Gemini Researcher", "gemini"),
        ("agent4", "Qwen Researcher", "qwen"),
    ]

    bindings: List[AgentBinding] = []
    for agent_id, agent_name, provider_hint in agent_specs:
        col = shared_collection or _pick_collection_name(base_collection, agent_id, existing)
        if col not in existing:
            sample = sorted(existing)[:20]
            raise RuntimeError(
                "Expected Chroma collection not found.\n"
                f"- missing: {col}\n"
                f"- persist_dir: {persist_dir}\n"
                f"- available (first 20): {sample}\n\n"
                "Hint: build the collection first, e.g. run build_vector_db.py for that agent."
            )
        bindings.append(AgentBinding(agent_id=agent_id, agent_name=agent_name, provider_hint=provider_hint, collection_name=col))

    # Instantiate agents.
    agents = []
    for b in bindings:
        model_cfg = cfg.get_llm_config(b.agent_id)

        collection = client.get_collection(name=b.collection_name)
        count = collection.count()
        if count <= 0:
            raise RuntimeError(f"Chroma collection is empty: {b.collection_name}")

        embed_as = shared_embedding_agent or b.agent_id
        rag_adapter = ChromaRAGAdapter(
            collection=collection,
            embedder=embedder,
            agent_name=embed_as,
            collection_name=b.collection_name,
            top_k=args.top_k,
        )

        agent = create_agent(
            agent_type=model_cfg.get("provider", b.provider_hint),
            agent_id=b.agent_id,
            name=b.agent_name,
            model_config=model_cfg,
            rag_system=rag_adapter,
            experience_store=None,
        )

        agents.append(agent)
        extra = f"; embed_as={embed_as}" if shared_collection else ""
        print(f"[OK] Bound {b.agent_id} -> collection '{b.collection_name}' (docs={count}{extra})")

    # Run debate (LangGraph coordinator).
    debate_cfg = cfg.get_debate_config() or {}
    coordinator = LangGraphDebateCoordinator(agents=agents, config=debate_cfg)

    raw_components = parse_component_string(args.components)
    if not raw_components:
        raise RuntimeError("No components provided")

    elements, percents = parse_components_with_percent(raw_components)
    electrode_composition = build_electrode_composition(elements, percents=percents, seed="|".join(elements))
    initial_prompt = build_initial_debate_prompt(
        elements,
        reaction_type=args.reaction_type,
        electrode_composition=electrode_composition,
    )

    print("\n" + "=" * 60)
    print("STARTING LANGGRAPH DEBATE")
    print("=" * 60)
    print(f"Reaction: {args.reaction_type}")
    print(f"Electrode composition (relative %): {electrode_composition}")
    print(f"Metal catalyst elements: {', '.join(elements)}")
    print(f"Persist dir: {persist_dir}")
    if shared_collection:
        print(f"Shared collection: {shared_collection}")
        print(f"Query embedding profile: {shared_embedding_agent}")
    else:
        print(f"Base collection: {base_collection}")

    result = coordinator.start_debate(components=elements, initial_prompt=initial_prompt, reaction_type=args.reaction_type)
    result_dict = result.to_dict()
    result_dict["electrode_composition"] = electrode_composition

    # Collect tool usage stats.
    total_search_literature_calls, source_ids = _collect_search_literature_stats(result_dict.get("debate_history") or [])
    valid_source_ids = {sid for sid in source_ids if is_valid_chroma_source_id(sid)}

    print("\n" + "=" * 60)
    print("DEBATE RESULT (SUMMARY)")
    print("=" * 60)
    print(f"Consensus: {result_dict.get('consensus_reached')}")
    print(f"Rounds: {result_dict.get('debate_rounds')}")
    print(f"Final products: {result_dict.get('final_products')}")
    print(f"Final performance: {result_dict.get('final_performance')}")
    print(f"search_literature calls (total): {total_search_literature_calls}")
    print(f"Retrieved source_id (unique): {len(source_ids)}")
    print(f"Valid source_id (unique): {len(valid_source_ids)}")

    if not args.strict:
        print("\nLENIENT: not enforcing search_literature/source_id requirements.")
        return 0

    problems: List[str] = []

    if total_search_literature_calls <= 0:
        problems.append("no_search_literature_calls")

    if not valid_source_ids:
        problems.append("no_valid_source_id_retrieved")

    if problems:
        print("\nFAIL: debate completed but did not meet strict evidence requirements.")
        for p in problems:
            print(f"- {p}")
        return 2

    print("\nPASS: debate completed end-to-end with verifiable RAG evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
