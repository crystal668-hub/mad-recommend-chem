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
import warnings
from pathlib import Path
from typing import List, Optional, Dict

# Silence noisy optional-dependency warnings from third-party libs (autogen/flaml).
# These warnings are not actionable for typical runs and pollute CLI output/logs.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"flaml(\..*)?",
    message=r"flaml\.automl is not available\..*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"autogen\.oai\.gemini(\..*)?",
    message=r"\s*All support for the `google\.generativeai` package has ended\..*",
)

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
from database import RAGSystem
from database.embedder import MultiModelEmbedder
from agents import AgentConfig
from agents.llm_agents import create_agent
from experience import ExperienceStore
try:
    from debate.autogen_coordinator import AutoGenDebateCoordinator
except Exception:  # optional dependency (pyautogen -> autogen)
    AutoGenDebateCoordinator = None  
from debate.langgraph_coordinator import LangGraphDebateCoordinator
 
class MockExperienceStore:
    """Mock experience store placeholder (no-op)."""

    def query_experiences(self, components: List[str], top_k: int = 3):
        return []

    def add_experience(self, experience: Dict) -> None:
        return

    def get_statistics(self) -> Dict:
        return {"total_experiences": 0}


class MADSystem:
    """
    Multi-Agent Debate System Main Class
    Integrates all components and provides a unified interface for running the system.
    Analyzes metal catalyst elements to predict reaction types with lowest overpotential.
    """
    
    def __init__(self, config_path: str = "./config/config.yaml", default_engine: str = "langgraph"):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        print_header("Multi-Agent Metal Catalyst Overpotential Prediction System")
        print("Initializing system...")
        
        self.config = load_config(config_path)
        
        # Initialize logging
        self.logger = setup_logging(self.config)
        self.logger.info("System initialization started")
        
        # Initialize components
        self.rag_systems = {}
        self.agents = []
        self.experience_store = MockExperienceStore()
        self.debate_coordinator = None
        self.autogen_debate_coordinator = None
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

        try:
            self.experience_store = ExperienceStore(
                storage_path=storage_path,
                packs_path=packs_path,
                max_experiences=max_exps,
                relevance_threshold=threshold,
            )
            self.logger.info(
                "ExperienceStore initialized: storage_path=%s packs_path=%s total=%s",
                storage_path,
                packs_path,
                len(getattr(self.experience_store, "experiences", []) or []),
            )
        except Exception as e:
            # Keep system usable even if experience loading fails.
            self.logger.warning("ExperienceStore init failed; fallback to MockExperienceStore: %s", str(e))
            self.experience_store = MockExperienceStore()
    
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
        """Initialize AutoGen debate coordinators"""
        print("Initializing AutoGen debate coordinators...")
        
        debate_config = self.config.get('debate', {})
        
        if AutoGenDebateCoordinator is not None:
            self.autogen_debate_coordinator = AutoGenDebateCoordinator(
                agents=self.agents,
                config=debate_config
            )
        else:
            self.autogen_debate_coordinator = None

        # Default LangGraph-style coordinator
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
        
        # Validate components
        is_valid, error_msg = validate_components(components)
        if not is_valid:
            raise ValueError(f"Component validation failed: {error_msg}")
        
        if reaction_type:
            self.logger.info(f"Starting debate with reaction type {reaction_type} and catalysts: {components}")
        else:
            self.logger.info(f"Starting debate with metal catalysts: {components}")
        
        # Create debate logger
        debate_logger = DebateLogger()
        debate_logger.log_debate_start(components, self.config.get('debate', {}))
        
        # Select debate engine (default: langgraph)
        selected_engine = (engine or self.default_engine or "langgraph").strip().lower()
        if selected_engine == "langgraph":
            coordinator = self.langgraph_debate_coordinator or self.debate_coordinator
        elif selected_engine == "autogen":
            coordinator = self.autogen_debate_coordinator
        else:
            raise ValueError(f"Unknown debate engine: {selected_engine}")

        if coordinator is None:
            raise RuntimeError(f"Debate coordinator for engine '{selected_engine}' is not initialized")

        # Run debate
        result = coordinator.start_debate(components, reaction_type=reaction_type)
        result_dict = result.to_dict()
        result_dict["engine"] = selected_engine
        
        # Log debate end
        debate_logger.log_debate_end(result_dict)
        
        # Save results
        if save_result:
            self._save_result(result, components, engine=selected_engine)
        
        self.logger.info("Debate completed")
        
        return result_dict
    
    def _extract_and_save_experience(self, debate_result, components: List[str], reaction_type: Optional[str] = None) -> None:
        """Experience extraction interface reserved (currently disabled)"""
        return
    
    def _save_result(self, result, components: List[str], engine: Optional[str] = None) -> None:
        """Save debate results to file"""
        output_dir = ensure_dir(self.config.get('paths', {}).get('outputs', './outputs'))
        
        from utils.helpers import create_experiment_id, generate_timestamp
        exp_id = create_experiment_id(components)
        timestamp = generate_timestamp()
        
        filename = f"result_{timestamp}.json"
        filepath = output_dir / filename
        
        result_data = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "engine": engine,
            "components": components,
            "result": result.to_dict()
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
    valid_reaction_types = [
        "CO2RR", "EOR", "HER", "HOR", "HZOR", "O5H", "OER", "ORR", "UOR"
    ]
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System"
    )
    parser.add_argument(
        '--components',
        type=str,
        help='Metal components, separated by commas or Chinese commas, e.g., "Pt,Pd,Ru,Ir,Rh"'
    )
    parser.add_argument(
        '--reaction-type',
        type=str,
        choices=valid_reaction_types,
        help='Specify Reaction Type (choose one): CO2RR/EOR/HER/HOR/HZOR/O5H/OER/ORR/UOR'
    )
    parser.add_argument(
        '--engine',
        type=str,
        choices=["langgraph", "autogen"],
        default="langgraph",
        help='Select Debate Engine: langgraph(default) or autogen'
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
            print(f"\nAnalyzing metal catalyst elements: {', '.join(components)}\n")
            
            result = system.run_debate(components, reaction_type=args.reaction_type, engine=args.engine)
            
            # Print result summary
            print_header("Debate Result Summary")
            
            summary = {
                "Consensus Reached": "Yes" if result['consensus_reached'] else "No",
                "Products": result.get('final_products') or "Not Determined",
                "Performance": result.get('final_performance') or "Not Estimated",
                "Debate Rounds": result['debate_rounds'],
                "Time Elapsed": format_duration(result['time_elapsed'])
            }
            
            print(dict_to_table(summary, headers=("Item", "Result")))
        else:
            print("\nNo metal catalyst elements specified.")
            print("Usage: python main.py --components 'Pt,Pd,Ru,Ir,Rh'")
         
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
