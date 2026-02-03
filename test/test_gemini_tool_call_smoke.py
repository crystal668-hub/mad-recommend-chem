"""Smoke test: Gemini (agent3) tool-calling compatibility.

Purpose:
- Ensure Gemini route can emit tool calls that our adapter recognizes (tool_calls and legacy function_call).
- Ensure the agent produces a non-empty final answer and retrieved evidence contains valid source_id.

Usage:
  python .\test\test_gemini_tool_call_smoke.py --reaction-type OER --components "Pt,Cu,Ni,Fe,Co"

Notes:
- Requires valid API keys in environment (.env or exported), since it calls the configured LLM.
- Uses an in-memory dummy RAG adapter (no Chroma required).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure repo root is importable when running this script from arbitrary cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_config import AgentConfig
from agents.llm_agents import create_agent
from utils.logger import setup_logging
from utils.source_id import is_valid_chroma_source_id


@dataclass
class DummyRAGAdapter:
    """Minimal RAG adapter compatible with ReActAgent.search_rag (expects .retrieve(query)->List[Dict])."""

    collection_name: str = "dummy_collection"
    top_k: int = 5

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        # Include doc_id + chunk_id so ReActAgent can enrich each item with `source_id`.
        return [
            {
                "text": "Dummy evidence: OER activity can be enhanced via alloying and electronic structure tuning.",
                "score": 0.99,
                "metadata": {"doc_id": "10.0000/dummy-oer", "chunk_id": 1},
            },
            {
                "text": "Dummy evidence: Overpotential depends on adsorption energies and surface reconstruction.",
                "score": 0.97,
                "metadata": {"doc_id": "10.0000/dummy-oer", "chunk_id": 2},
            },
        ][: int(self.top_k)]


def _force_logs_into_repo(cfg_obj: AgentConfig) -> None:
    # Ensure logs always go to <repo_root>/logs regardless of cwd.
    log_cfg = (cfg_obj.config or {}).setdefault("logging", {})

    log_file = str(log_cfg.get("log_file", "./logs/system.log"))
    run_dir = str(log_cfg.get("run_dir", "./logs/runs"))

    if log_file and not Path(log_file).is_absolute():
        log_cfg["log_file"] = str((PROJECT_ROOT / log_file).resolve())
    if run_dir and not Path(run_dir).is_absolute():
        log_cfg["run_dir"] = str((PROJECT_ROOT / run_dir).resolve())


def _collect_tool_calls(trajectory) -> List[Tuple[int, str]]:
    calls: List[Tuple[int, str]] = []
    for step in getattr(trajectory, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            name = getattr(call, "tool_name", None) or ""
            if name:
                calls.append((int(getattr(step, "step_number", 0) or 0), str(name)))
    return calls


def _collect_source_ids(trajectory) -> List[str]:
    sids: List[str] = []
    for step in getattr(trajectory, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            if getattr(call, "tool_name", "") != "search_rag":
                continue
            data = getattr(call, "observation_data", None) or []
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                sid = item.get("source_id")
                if sid:
                    sids.append(str(sid))
    return sids


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Smoke test Gemini tool-calling compatibility")
    parser.add_argument("--config", default="./config/config.yaml", help="Config file path")
    parser.add_argument("--reaction-type", default="OER")
    parser.add_argument("--components", default="Pt,Cu,Ni,Fe,Co")
    parser.add_argument("--run-id", default=None, help="Optional run_id for logs")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    cfg = AgentConfig(str(config_path))
    _force_logs_into_repo(cfg)

    run_id = args.run_id or f"gemini_tool_call_smoke_{int(time.time())}"
    setup_logging(cfg.config, run_id=run_id)

    model_cfg = cfg.get_llm_config("agent3")
    provider = model_cfg.get("provider", "google")

    agent = create_agent(
        agent_type=provider,
        agent_id="agent3",
        name="Gemini Researcher",
        model_config=model_cfg,
        rag_system=DummyRAGAdapter(collection_name="dummy_collection", top_k=5),
        experience_store=None,
    )

    prompt = (
        "This is a tool-calling smoke test.\n"
        f"Target reaction: {args.reaction_type}\n"
        f"Components: {args.components}\n\n"
        "Requirements:\n"
        "1) In ACTION phase, call `search_rag` at least once (do NOT answer without tool calls).\n"
        "2) After observing results, call `conclude` with a short answer that cites at least one `source_id`.\n"
    )

    response, trajectory = agent.generate_response_with_react(query=prompt, components=None, context=None)

    tool_calls = _collect_tool_calls(trajectory)
    source_ids = _collect_source_ids(trajectory)
    valid_source_ids = [sid for sid in source_ids if is_valid_chroma_source_id(sid)]

    print("=" * 60)
    print("GEMINI TOOL-CALL SMOKE RESULT")
    print("=" * 60)
    print(f"run_id: {run_id}")
    print(f"final_answer_preview: {(response.content or '')[:500]}")
    print(f"steps: {len(getattr(trajectory, 'steps', []) or [])}")
    print(f"tool_calls: {len(tool_calls)}")
    if tool_calls:
        print("tool_calls_preview:")
        for step_no, name in tool_calls[:20]:
            print(f"- step={step_no} tool={name}")
    print(f"search_rag_source_id_total: {len(source_ids)}")
    print(f"search_rag_source_id_valid: {len(valid_source_ids)}")
    if valid_source_ids:
        print("valid_source_id_preview:")
        for sid in valid_source_ids[:5]:
            print(f"- {sid}")

    ok = True
    if not (response.content or "").strip():
        print("FAIL: empty final answer")
        ok = False
    if (response.content or "").strip().lower() == "no conclusion generated.":
        print("FAIL: forced conclude produced empty content")
        ok = False
    if not any(name == "search_rag" for _step, name in tool_calls):
        print("FAIL: no search_rag tool call recorded")
        ok = False
    if not valid_source_ids:
        print("FAIL: no valid source_id found in search_rag observation_data")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
