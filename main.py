"""
===================================
Multi-Agent Debate System 
===================================

Functionality:
1. Initialize system components 
2. Execute multi-agent debate
3. Generate result reports

===================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, List, Optional, Dict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from utils import (
    load_config,
    setup_logging,
    DebateLogger,
    parse_component_string,
    validate_components,
    print_header,
    print_section,
    dict_to_table,
    format_duration,
    save_json,
    ensure_dir
)
from utils.reaction_types import (
    SUPPORTED_REACTION_TYPE_LABELS,
    REACTION_TYPE_LABELS,
    canonical_reaction_type,
    is_supported_reaction_type,
)
from database import RAGSystem
from database.embedder import MultiModelEmbedder
from agents import AgentConfig
from agents.llm_agents import create_agent
from experience import ExperienceStore
from debate.langgraph_coordinator import LangGraphDebateCoordinator


class MADSystem:
    """
    Multi-Agent Debate System Main Class
    Integrates all components and provides a unified interface for running the system.
    Analyzes metal catalyst elements to predict reaction types with lowest overpotential.
    """
    
    def __init__(self, config_path: str = "./config/config.yaml", default_engine: str = "langgraph"):
        """
        Initialize MADSystem with configuration and default debate engine
        
        Args:
            config_path: Path to configuration file
            default_engine: Default debate engine to use "langgraph"
        """
        print_header("Multi-Agent Debate System")
        print("Initializing system...")
        
        self.config = load_config(config_path)
        
        # Initialize logging
        self.logger = setup_logging(self.config)
        self.logger.info("System initialization started")
        
        # Initialize components
        self.rag_systems = {}
        self.agents = []
        self.experience_store = None
        self.debate_coordinator = None
        self.langgraph_debate_coordinator = None
        self.default_engine = default_engine
        
        # Initialization flag
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize all system components
        """
        if self._initialized:
            self.logger.warning("System already initialized, skipping re-initialization")
            return
        
        try:
            # 1. Initialize RAG systems (required)
            self._init_rag_systems()

            # 2. Initialize Experience Store (for `search_experience`)
            self._init_experience_store()
            
            # 3. Initialize Agents
            self._init_agents()
            
            # 4. Initialize debate coordinators
            self._init_debate_coordinators()
            
            self._initialized = True
            self.logger.info("System initialization completed")
            print("\n✓ System initialized successfully\n")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}", exc_info=True)
            print(f"\n✗ System initialization failed: {str(e)}\n")
            raise
    
    def _init_experience_store(self) -> None:
        """Initialize experience store (YAML packs + optional JSON dynamic store)."""
        exp_cfg = self.config.get("experience", {}) or {}
        storage_path = exp_cfg.get("storage_path", "./data/experience_db.json")
        packs_path = exp_cfg.get("packs_path", "./experience")
        max_exps = int(exp_cfg.get("max_experiences", 1000))
        threshold = float(exp_cfg.get("relevance_threshold", 0.8))
        guideline_top_k = int(exp_cfg.get("guideline_top_k", 3))
        always_include_guidelines = bool(exp_cfg.get("always_include_guidelines", True))
        guideline_search_mode = exp_cfg.get("guideline_search_mode", "keyword")
        load_builtin_packs = bool(exp_cfg.get("load_builtin_packs", True))

        self.experience_store = ExperienceStore(
            storage_path=storage_path,
            packs_path=packs_path,
            max_experiences=max_exps,
            relevance_threshold=threshold,
            guideline_top_k=guideline_top_k,
            always_include_guidelines=always_include_guidelines,
            guideline_search_mode=guideline_search_mode,
            load_builtin_packs=load_builtin_packs,
        )
        self.logger.info(
            "ExperienceStore initialized: storage_path=%s packs_path=%s total=%s",
            storage_path,
            packs_path,
            len(getattr(self.experience_store, "experiences", []) or []),
        )
    
    def _init_rag_systems(self) -> None:
        """Initialize RAG retrieval adapters (Chroma/VectorStore-backed)."""
        print("Initializing RAG systems...")
        rag_config = self.config.get('rag', {})
        vector_config = self.config.get('vector_store', {})

        persist_dir = vector_config.get('persist_directory', './data/chroma_db')
        base_collection_name = vector_config.get('collection_name', 'chemical_reactions_recommendation')
        distance_metric = vector_config.get('distance_metric', 'cosine')

        top_k = rag_config.get('top_k', 5)
        similarity_threshold = rag_config.get('similarity_threshold', None)

        # Build a shared embedder that knows each agent's embedding profile.
        agent_config = AgentConfig(self.config)
        agent_keys = ["agent1", "agent2", "agent3", "agent4"]
        all_agent_configs = {k: agent_config.get_llm_config(k) for k in agent_keys}
        embedder = MultiModelEmbedder(all_agent_configs["agent1"], agent_configs=all_agent_configs)

        # One collection per agent (matches `build_vector_db.py`: <base>_<agentX>).
        self.rag_systems = {}
        for agent_key in agent_keys:
            collection_name = f"{base_collection_name}_{agent_key}"
            print(f"  Creating RAG adapter for {agent_key}, using collection: {collection_name}")

            rag_system = RAGSystem(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embedder=embedder,
                agent_name=agent_key,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                distance_metric=distance_metric,
            )
            self.rag_systems[agent_key] = rag_system

        self.logger.info("RAG systems initialization completed")
    
    def _init_agents(self) -> None:
        """Initialize Agents"""
        print("Initializing Agents...")

        agent_config = AgentConfig(self.config)

        agent_specs = [
            ("agent1", "GPT Researcher", "agent1"),
            ("agent2", "DeepSeek Researcher", "agent2"),
            ("agent3", "Gemini Researcher", "agent3"),
            ("agent4", "Qwen Researcher", "agent4")
        ]

        self.agents = []
        for agent_key, agent_name, provider_key in agent_specs:
            model_config = agent_config.get_llm_config(agent_key)
            rag_system = self.rag_systems.get(provider_key)

            agent = create_agent(
                agent_type=model_config.get("provider", "openai"),
                agent_id=agent_key,
                name=agent_name,
                model_config=model_config,
                rag_system=rag_system,
                experience_store=self.experience_store
            )

            self.agents.append(agent)
            print(f"✓ Successfully created Agent: {agent_name} ({provider_key})")

        if not self.agents:
            raise RuntimeError("No Agents were successfully created")

        self.logger.info(f"Successfully initialized {len(self.agents)} Agents")
    
    def _init_debate_coordinators(self) -> None:
        """Initialize debate coordinator(s) (LangGraph-style only)."""
        print("Initializing debate coordinator...")

        debate_config = self.config.get('debate', {})

        # LangGraph-style coordinator (implemented in-repo; no external dependency required).
        self.langgraph_debate_coordinator = LangGraphDebateCoordinator(
            agents=self.agents,
            config=debate_config
        )

        # Default engine points to LangGraph coordinator
        self.debate_coordinator = self.langgraph_debate_coordinator
        
    
    def run_debate(
        self,
        components: List[str],
        reaction_type: Optional[str] = None,
        save_result: bool = True,
        engine: Optional[str] = None
    ) -> dict:
        """
        Run debate
        
        Args:
            components: List of metal catalyst elements
            save_result: Whether to save results
        
        Returns:
            dict: Debate results
        """
        if not self._initialized:
            raise RuntimeError("System not initialized, please call initialize() first")

        # Normalize input into:
        # - `elements`: metal symbols used throughout the system (guards/experience/RAG)
        # - `electrode_composition`: formatted "Ni(69.00%), ..." string used in the debate prompt
        from utils.electrode_composition import parse_components_with_percent, build_electrode_composition
        from prompts.debate_phase_prompts import build_initial_debate_prompt

        elements, percents = parse_components_with_percent([str(c) for c in (components or [])])
        electrode_composition = build_electrode_composition(elements, percents=percents, seed="|".join(elements))

        # Validate components before starting debate
        is_valid, error_msg = validate_components(elements)
        if not is_valid:
            raise ValueError(f"Component validation failed: {error_msg}")
        
        reaction_type = canonical_reaction_type(reaction_type) if reaction_type else None

        if reaction_type:
            self.logger.info(
                "Starting debate with reaction type %s and catalysts: %s (electrode=%s)",
                reaction_type,
                elements,
                electrode_composition,
            )
        else:
            self.logger.info("Starting debate with metal catalysts: %s (electrode=%s)", elements, electrode_composition)
        
        # Create debate logger
        debate_logger = DebateLogger()
        debate_logger.log_debate_start(elements, self.config.get('debate', {}))
        
        # Only LangGraph-style engine is supported.
        selected_engine = "langgraph"
        if engine and str(engine).strip().lower() != "langgraph":
            raise ValueError(f"Unsupported debate engine: {engine}. Only 'langgraph' is supported.")
        coordinator = self.langgraph_debate_coordinator or self.debate_coordinator

        if coordinator is None:
            raise RuntimeError(f"Debate coordinator for engine '{selected_engine}' is not initialized")

        # Run debate
        initial_prompt = build_initial_debate_prompt(
            elements,
            reaction_type=reaction_type,
            electrode_composition=electrode_composition,
        )
        result = coordinator.start_debate(elements, initial_prompt=initial_prompt, reaction_type=reaction_type)
        result_dict = result.to_dict()
        result_dict["engine"] = selected_engine
        result_dict["electrode_composition"] = electrode_composition

        # Performance grading (only when there is a single final conclusion).
        # - Prefer `final_performance` if present.
        # - Otherwise, if there is exactly one surviving proposal, grade its claim.
        # - Otherwise, leave as N/A (no field).
        try:
            from utils.performance_grading import evaluate_claim

            final_claim = None
            fp = result_dict.get("final_performance")
            if isinstance(fp, str) and fp.strip():
                final_claim = fp
            else:
                surviving = result_dict.get("surviving_proposals") or []
                if isinstance(surviving, list) and len(surviving) == 1:
                    claim = (surviving[0] or {}).get("claim") if isinstance(surviving[0], dict) else None
                    if isinstance(claim, str) and claim.strip():
                        final_claim = claim

            perf_eval = evaluate_claim(final_claim, reaction_type=reaction_type) if final_claim else None
            if perf_eval:
                result_dict["performance_evaluation"] = perf_eval
        except Exception as e:
            # Never break debate completion on grading errors.
            self.logger.warning("Performance evaluation failed: %s", str(e))
        
        # Log debate end
        debate_logger.log_debate_end(result_dict)
        
        # Save results
        if save_result:
            self._save_result(
                result,
                elements,
                electrode_composition=electrode_composition,
                engine=selected_engine,
                performance_evaluation=result_dict.get("performance_evaluation"),
            )
        
        self.logger.info("Debate completed")
        
        return result_dict

    def run_rank_reactions(
        self,
        components: List[str],
        reaction_types: Optional[List[str]] = None,
        top_k: int = 2,
        max_parallel_reactions: int = 1,
        save_each_reaction: bool = False,
    ) -> Dict:
        """
        Run debates for multiple reaction types (default: config.yaml chemistry.reaction_types)
        and return a ranked summary with Top-K reactions.

        Ranking policy (decision-complete per user plan):
        - Primary: Grade (Outstanding > Good > Fair > Poor > Terrible)
        - Tie-break: metric direction (lower-is-better uses -metric_value; higher-is-better uses metric_value)

        Notes:
        - By default we DO NOT save per-reaction `outputs/result_*.json` unless `save_each_reaction=True`.
        - When max_parallel_reactions > 1, we run each reaction in an *isolated* coordinator
          (fresh agents/coordinator per reaction) to avoid thread-safety issues.
        """
        if not self._initialized:
            raise RuntimeError("System not initialized, please call initialize() first")

        # Normalize input once (also gives us a stable electrode composition for the summary payload).
        from utils.electrode_composition import parse_components_with_percent, build_electrode_composition

        elements, percents = parse_components_with_percent([str(c) for c in (components or [])])
        electrode_composition = build_electrode_composition(elements, percents=percents, seed="|".join(elements))

        is_valid, error_msg = validate_components(elements)
        if not is_valid:
            raise ValueError(f"Component validation failed: {error_msg}")

        # Resolve reaction types: explicit arg > config.yaml > built-in fallback.
        rt_list: List[str] = []
        if reaction_types:
            rt_list = [canonical_reaction_type(x) for x in reaction_types if str(x).strip()]
            rt_list = [x for x in rt_list if x]
        else:
            cfg_rts = (((self.config or {}).get("chemistry", {}) or {}).get("reaction_types")) or []
            if isinstance(cfg_rts, list) and cfg_rts:
                rt_list = [canonical_reaction_type(x) for x in cfg_rts if str(x).strip()]
                rt_list = [x for x in rt_list if x]
        if not rt_list:
            rt_list = list(REACTION_TYPE_LABELS)
        # De-duplicate while preserving order (avoid re-running the same reaction twice).
        seen_rt: set[str] = set()
        rt_dedup: List[str] = []
        for rt in rt_list:
            s = canonical_reaction_type(rt)
            if not s or s in seen_rt:
                continue
            seen_rt.add(s)
            rt_dedup.append(s)
        rt_list = rt_dedup

        # Clamp and sanitize.
        try:
            k_final = int(top_k)
        except Exception:
            k_final = 2
        k_final = max(0, k_final)
        try:
            max_par = int(max_parallel_reactions)
        except Exception:
            max_par = 1
        max_par = max(1, max_par)

        debate_config = (self.config.get("debate", {}) or {}).copy()

        def _extract_perf_fields(result_dict: Dict) -> Dict[str, Any]:
            pe = result_dict.get("performance_evaluation") if isinstance(result_dict, dict) else None
            out: Dict[str, Any] = {"performance_evaluation": pe if isinstance(pe, dict) else None}
            if isinstance(pe, dict):
                out["metric_value"] = pe.get("metric_value")
                out["metric_unit"] = pe.get("metric_unit")
                out["grade"] = pe.get("grade")
            else:
                out["metric_value"] = None
                out["metric_unit"] = None
                out["grade"] = None
            return out

        def _run_debate_with_coordinator(
            coordinator: LangGraphDebateCoordinator,
            reaction_type: str,
            save_result: bool,
        ) -> Dict[str, Any]:
            from prompts.debate_phase_prompts import build_initial_debate_prompt

            rt = canonical_reaction_type(reaction_type) or "UNKNOWN"

            if rt and rt != "UNKNOWN":
                self.logger.info(
                    "Starting debate (isolated) with reaction type %s and catalysts: %s (electrode=%s)",
                    rt,
                    elements,
                    electrode_composition,
                )
            else:
                self.logger.info(
                    "Starting debate (isolated) with catalysts: %s (electrode=%s)",
                    elements,
                    electrode_composition,
                )

            debate_logger = DebateLogger()
            debate_logger.log_debate_start(elements, debate_config)

            initial_prompt = build_initial_debate_prompt(
                elements,
                reaction_type=rt,
                electrode_composition=electrode_composition,
            )
            result = coordinator.start_debate(elements, initial_prompt=initial_prompt, reaction_type=rt)

            result_dict = result.to_dict()
            result_dict["engine"] = "langgraph"
            result_dict["electrode_composition"] = electrode_composition

            # Attach performance grading if we can extract a final claim.
            try:
                from utils.performance_grading import evaluate_claim

                final_claim = None
                fp = result_dict.get("final_performance")
                if isinstance(fp, str) and fp.strip():
                    final_claim = fp
                else:
                    surviving = result_dict.get("surviving_proposals") or []
                    if isinstance(surviving, list) and len(surviving) == 1:
                        claim = (surviving[0] or {}).get("claim") if isinstance(surviving[0], dict) else None
                        if isinstance(claim, str) and claim.strip():
                            final_claim = claim

                perf_eval = evaluate_claim(final_claim, reaction_type=rt) if final_claim else None
                if perf_eval:
                    result_dict["performance_evaluation"] = perf_eval
            except Exception as e:
                self.logger.warning("Performance evaluation failed: %s", str(e))

            debate_logger.log_debate_end(result_dict)

            if save_result:
                self._save_result(
                    result,
                    elements,
                    electrode_composition=electrode_composition,
                    engine="langgraph",
                    performance_evaluation=result_dict.get("performance_evaluation"),
                )

            return result_dict

        def _build_isolated_coordinator() -> LangGraphDebateCoordinator:
            """
            Build a fresh coordinator (fresh agents) for safe parallel per-reaction execution.
            """
            # Reduce inner concurrency when outer reaction-level parallelism is enabled to mitigate API rate limits.
            local_cfg = dict(debate_config or {})
            if max_par > 1:
                inner_cap = 1 if max_par > 3 else 2
                try:
                    current_inner = int(local_cfg.get("max_concurrency", 4))
                except Exception:
                    current_inner = 4
                local_cfg["max_concurrency"] = max(1, min(current_inner, inner_cap))

            # Create fresh RAG systems to avoid sharing a Chroma collection object across threads.
            rag_config = self.config.get("rag", {}) or {}
            vector_config = self.config.get("vector_store", {}) or {}
            persist_dir = vector_config.get("persist_directory", "./data/chroma_db")
            base_collection_name = vector_config.get("collection_name", "electrochemistry_literature")
            distance_metric = vector_config.get("distance_metric", "cosine")
            top_k_rag = rag_config.get("top_k", 5)
            similarity_threshold = rag_config.get("similarity_threshold", None)

            # Prefer reusing the already-initialized embedder (it knows per-agent embedding profiles).
            embedder = None
            try:
                if self.rag_systems:
                    embedder = next(iter(self.rag_systems.values())).embedder
            except Exception:
                embedder = None
            if embedder is None:
                agent_config = AgentConfig(self.config)
                agent_keys = ["agent1", "agent2", "agent3", "agent4"]
                all_agent_configs = {k: agent_config.get_llm_config(k) for k in agent_keys}
                embedder = MultiModelEmbedder(all_agent_configs["agent1"], agent_configs=all_agent_configs)

            rag_systems_local: Dict[str, Any] = {}
            for agent_key in ["agent1", "agent2", "agent3", "agent4"]:
                collection_name = f"{base_collection_name}_{agent_key}"
                rag_systems_local[agent_key] = RAGSystem(
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    embedder=embedder,
                    agent_name=agent_key,
                    top_k=top_k_rag,
                    similarity_threshold=similarity_threshold,
                    distance_metric=distance_metric,
                )

            agent_config = AgentConfig(self.config)
            agent_specs = [
                ("agent1", "GPT Researcher", "agent1"),
                ("agent2", "DeepSeek Researcher", "agent2"),
                ("agent3", "Gemini Researcher", "agent3"),
                ("agent4", "Qwen Researcher", "agent4"),
            ]

            agents_local = []
            for agent_key, agent_name, provider_key in agent_specs:
                model_config = agent_config.get_llm_config(agent_key)
                rag_system = rag_systems_local.get(provider_key)
                agent = create_agent(
                    agent_type=model_config.get("provider", "openai"),
                    agent_id=agent_key,
                    name=agent_name,
                    model_config=model_config,
                    rag_system=rag_system,
                    experience_store=self.experience_store,
                )
                agents_local.append(agent)

            return LangGraphDebateCoordinator(agents=agents_local, config=local_cfg)

        def _run_one_reaction(rt: str) -> Dict[str, Any]:
            """
            Run one reaction (with optional retry) and return a summary dict.
            """
            import time

            reaction = canonical_reaction_type(rt) or "UNKNOWN"
            attempts = 2  # best-effort retry for transient 429/rate-limit type failures
            last_err = None

            for attempt in range(attempts):
                try:
                    if max_par <= 1:
                        res = self.run_debate(
                            components,
                            reaction_type=reaction,
                            save_result=bool(save_each_reaction),
                            engine="langgraph",
                        )
                    else:
                        coord = _build_isolated_coordinator()
                        res = _run_debate_with_coordinator(
                            coord,
                            reaction_type=reaction,
                            save_result=bool(save_each_reaction),
                        )

                    perf_fields = _extract_perf_fields(res if isinstance(res, dict) else {})
                    return {
                        "reaction_type": reaction,
                        "consensus_reached": bool((res or {}).get("consensus_reached")) if isinstance(res, dict) else False,
                        "final_products": (res or {}).get("final_products") if isinstance(res, dict) else None,
                        "final_performance": (res or {}).get("final_performance") if isinstance(res, dict) else None,
                        "debate_rounds": (res or {}).get("debate_rounds") if isinstance(res, dict) else None,
                        "time_elapsed": (res or {}).get("time_elapsed") if isinstance(res, dict) else None,
                        **perf_fields,
                        "error": None,
                    }
                except Exception as e:
                    last_err = e
                    msg = str(e) or e.__class__.__name__
                    # Retry only for likely transient rate-limit failures.
                    msg_l = msg.lower()
                    retriable = any(
                        s in msg_l
                        for s in [
                            "429",
                            "rate limit",
                            "ratelimit",
                            "too many requests",
                            "temporarily unavailable",
                            "timeout",
                            "read timed out",
                            "connect timeout",
                        ]
                    )
                    if attempt < attempts - 1 and retriable:
                        # Exponential backoff (bounded) with a tiny deterministic jitter.
                        sleep_s = min(30.0, 2.0 * (2**attempt))
                        time.sleep(sleep_s)
                        continue
                    break

            err_text = str(last_err) if last_err is not None else "unknown error"
            return {
                "reaction_type": reaction,
                "consensus_reached": False,
                "final_products": None,
                "final_performance": None,
                "debate_rounds": None,
                "time_elapsed": None,
                "performance_evaluation": None,
                "metric_value": None,
                "metric_unit": None,
                "grade": None,
                "error": err_text,
            }

        # ---- Execute all reactions (sequential or parallel) ----
        summaries: List[Dict[str, Any]] = []
        if max_par <= 1:
            for rt in rt_list:
                summaries.append(_run_one_reaction(rt))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_par) as ex:
                future_to_rt = {ex.submit(_run_one_reaction, rt): rt for rt in rt_list}
                for fut in as_completed(future_to_rt):
                    try:
                        summaries.append(fut.result())
                    except Exception as e:
                        # Should be rare since _run_one_reaction catches, but keep robust.
                        summaries.append(
                            {
                                "reaction_type": canonical_reaction_type(future_to_rt.get(fut)) or "UNKNOWN",
                                "error": str(e),
                                "performance_evaluation": None,
                                "metric_value": None,
                                "metric_unit": None,
                                "grade": None,
                            }
                        )

            # Preserve original reaction_types order as a stable secondary key for equal ranks.
            order = {canonical_reaction_type(rt) or str(rt).strip(): i for i, rt in enumerate(rt_list)}
            summaries.sort(key=lambda d: order.get(canonical_reaction_type(d.get("reaction_type")) or "", 10**9))

        # ---- Rank and save summary ----
        from utils.reaction_ranking import rank_reactions
        from utils.helpers import generate_timestamp

        ranking, top_items = rank_reactions(summaries, top_k=k_final)

        timestamp = generate_timestamp()
        output_dir = ensure_dir(self.config.get("paths", {}).get("outputs", "./outputs"))
        out_path = output_dir / f"rank_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "engine": "langgraph",
            "components": elements,
            "electrode_composition": electrode_composition,
            "ranking": ranking,
            "top_k": top_items,
        }
        save_json(payload, out_path)
        self.logger.info("Reaction ranking saved: %s", str(out_path))

        return payload
    
    def _extract_and_save_experience(self, debate_result, components: List[str], reaction_type: Optional[str] = None) -> None:
        """Experience extraction interface reserved (currently disabled)"""
        return
    
    def _save_result(
        self,
        result,
        components: List[str],
        electrode_composition: Optional[str] = None,
        engine: Optional[str] = None,
        performance_evaluation: Optional[Dict] = None,
    ) -> None:
        """Save debate results to file"""
        output_dir = ensure_dir(self.config.get('paths', {}).get('outputs', './outputs'))
        
        from utils.helpers import create_experiment_id, generate_timestamp
        exp_id = create_experiment_id(components)
        timestamp = generate_timestamp()
        
        filename = f"result_{timestamp}.json"
        filepath = output_dir / filename
        
        result_payload = result.to_dict()
        if performance_evaluation:
            result_payload["performance_evaluation"] = performance_evaluation

        result_data = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "engine": engine,
            "components": components,
            "electrode_composition": (electrode_composition or None),
            "result": result_payload
        }
        
        save_json(result_data, filepath)
        self.logger.info(f"Results saved: {filepath}")
    
    def print_system_status(self) -> None:
        """Print system status"""
        print_header("System Status")

        exp_total = 0
        try:
            if self.experience_store:
                exp_total = int(self.experience_store.get_statistics().get("total_experiences", 0))
        except Exception:
            exp_total = 0

        status = {
            "Initialization Status": "Initialized" if self._initialized else "Not Initialized",
            "Number of Agents": len(self.agents) if self.agents else 0,
            "Experience Store Size": exp_total,
            "RAG Systems": "Loaded" if self.rag_systems else "Not Loaded"
        }
        
        print(dict_to_table(status, headers=("Item", "Status")))
        
        if self.experience_store:
            stats = self.experience_store.get_statistics()
            print_section("Experience Store Statistics", dict_to_table(stats, headers=("Metric", "Value")))

def main():
    """Main program entry point"""
    valid_reaction_types = list(SUPPORTED_REACTION_TYPE_LABELS)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System"
    )
    parser.add_argument(
        '--components',
        type=str,
        help=(
            'Metal components. You may provide symbols only, e.g., "Pt,Pd,Ru,Ir,Rh", '
            'or symbols with relative percentages, e.g., "Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)".'
        )
    )
    parser.add_argument(
        '--reaction-type',
        type=str,
        help='Specify reaction/category type'
    )
    parser.add_argument(
        '--rank-reactions',
        action='store_true',
        help='Run debates for multiple reaction types and rank Top-K reactions by grade'
    )
    parser.add_argument(
        '--top-k-reactions',
        type=int,
        default=2,
        help='When --rank-reactions is enabled, output the best K reaction types (default: 2)'
    )
    parser.add_argument(
        '--reaction-types',
        type=str,
        default=None,
        help='Comma-separated subset of reaction types to rank, e.g. "OER,HER,ORR". Defaults to config.yaml chemistry.reaction_types.'
    )
    parser.add_argument(
        '--max-parallel-reactions',
        type=int,
        default=3,
        help='Reaction-level parallelism when --rank-reactions is enabled (default: 3). Higher may trigger API rate limits.'
    )
    parser.add_argument(
        '--save-each-reaction',
        action='store_true',
        help='When --rank-reactions is enabled, also save per-reaction outputs/result_*.json files'
    )
    parser.add_argument(
        '--engine',
        type=str,
        choices=["langgraph"],
        default="langgraph",
        help='Select Debate Engine (only langgraph is supported currently)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )
    
    args = parser.parse_args()
    
    try:
        # Create system instance
        system = MADSystem(config_path=args.config, default_engine=args.engine)
        
        # Initialize system
        system.initialize()
        
        # Show system status
        if args.status:
            system.print_system_status()
            return
        
        # Run debate
        if args.components:
            components = parse_component_string(args.components)
            print(f"\nAnalyzing metal elements: {', '.join(components)}\n")

            if args.rank_reactions:
                if args.reaction_type:
                    raise ValueError("`--reaction-type` cannot be used together with `--rank-reactions`.")

                subset = None
                if args.reaction_types:
                    subset = [canonical_reaction_type(s) for s in str(args.reaction_types).split(",") if s.strip()]
                    subset = [s for s in subset if s]
                    bad = [s for s in subset if not is_supported_reaction_type(s)]
                    if bad:
                        raise ValueError(f"Unknown reaction types in --reaction-types: {bad}. Choices: {valid_reaction_types}")

                payload = system.run_rank_reactions(
                    components,
                    reaction_types=subset,
                    top_k=int(args.top_k_reactions),
                    max_parallel_reactions=int(args.max_parallel_reactions),
                    save_each_reaction=bool(args.save_each_reaction),
                )

                ranking = payload.get("ranking") or []
                top_items = payload.get("top_k") or []

                def _fmt_metric(it: Dict) -> str:
                    v = it.get("metric_value")
                    u = it.get("metric_unit")
                    if v is None or not u:
                        return "N/A"
                    return f"{v} {u}"

                print_header("Reaction Ranking Summary")
                print("Rank | RT   | Grade        | Metric              | Consensus | Error")
                print("-----|------|--------------|---------------------|----------|------")
                for idx, it in enumerate(ranking, start=1):
                    rt = canonical_reaction_type(it.get("reaction_type")) or "?"
                    grade = str(it.get("grade") or "N/A")
                    metric = _fmt_metric(it)
                    consensus = "Yes" if bool(it.get("consensus_reached")) else "No"
                    err = str(it.get("error") or "")
                    if len(err) > 60:
                        err = err[:57] + "..."
                    print(f"{idx:>4} | {rt:<4} | {grade:<12} | {metric:<19} | {consensus:<8} | {err}")

                print_header(f"Top {len(top_items)} Reaction Types")
                for idx, it in enumerate(top_items, start=1):
                    rt = canonical_reaction_type(it.get("reaction_type")) or "?"
                    grade = str(it.get("grade") or "N/A")
                    metric = _fmt_metric(it)
                    print(f"{idx}. {rt} | Grade: {grade} | Metric: {metric}")

                out_dir = system.config.get("paths", {}).get("outputs", "./outputs")
                ts = payload.get("timestamp") or ""
                if ts:
                    print(f"\nSaved ranking summary: {out_dir}\\rank_{ts}.json")

            else:
                reaction_type = canonical_reaction_type(args.reaction_type) if args.reaction_type else None
                if reaction_type and not is_supported_reaction_type(reaction_type):
                    raise ValueError(f"Unknown reaction type: {args.reaction_type}. Choices: {valid_reaction_types}")

                result = system.run_debate(components, reaction_type=reaction_type, engine=args.engine)

                # Print result summary
                print_header("Debate Result Summary")

                perf_eval = result.get("performance_evaluation") if isinstance(result, dict) else None
                metric_norm = "N/A"
                grade = "N/A"
                if isinstance(perf_eval, dict):
                    v = perf_eval.get("metric_value")
                    u = perf_eval.get("metric_unit")
                    g = perf_eval.get("grade")
                    if v is not None and u:
                        metric_norm = f"{v} {u}"
                    if g:
                        grade = str(g)

                summary = {
                    "Consensus Reached": "Yes" if result["consensus_reached"] else "No",
                    "Products": result.get("final_products") or "Not Determined",
                    "Performance": result.get("final_performance") or "Not Estimated",
                    "Metric (normalized)": metric_norm,
                    "Grade": grade,
                    "Debate Rounds": result["debate_rounds"],
                    "Time Elapsed": format_duration(result["time_elapsed"]),
                }

                print(dict_to_table(summary, headers=("Item", "Result")))
        else:
            print("\nNo metal elements specified.")
            print("Usage: python main.py --components \"Ni(69.00%), Co(19.07%), Fe(11.48%), Cu(0.40%), Zn(0.05%)\"")
         
            system.print_system_status()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
