"""
===================================
AutoGen Debate Coordinator
功能：使用AutoGen GroupChat完全管理多Agent辩论流程
===================================
"""

import logging
import re
import time
import warnings
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Silence noisy optional-dependency warnings from third-party libs (autogen/flaml).
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

from autogen import ConversableAgent, GroupChat, GroupChatManager

from agents.react_agent import ReActAgent
from agents.react_reasoning import ReActTrajectory
from prompts.system_prompts import UNIFIED_SYSTEM_PROMPT
from prompts.debate_phase_prompts import build_initial_debate_prompt
from utils.logger import get_run_id, make_debate_id, write_debate_artifacts

logger = logging.getLogger("MAD.debate.autogen")

class ReActAutoGenAgent(ConversableAgent):
    """
    统一的ReAct风格AutoGen Agent封装
    使用统一System Prompt
    """

    def __init__(
        self,
        name: str,
        llm_config: Dict,
        system_message: str,
        base_agent: ReActAgent,
        initial_prompt: str = ""
    ) -> None:
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        self.base_agent = base_agent
        self.base_agent.system_prompt = system_message
        self.initial_prompt = initial_prompt

    def generate_reply(self, messages: Optional[List[Dict]] = None, sender: Optional[ConversableAgent] = None, **kwargs):
        messages = messages or []
        last_message = self._get_last_debate_message(messages)
        phase_instruction = self._get_latest_phase_instruction(messages, self.name)

        if not last_message:
            last_message = self._get_fallback_prompt(messages)

        combined_parts = []
        if phase_instruction:
            combined_parts.append(phase_instruction)
        if self.initial_prompt:
            combined_parts.append(f"Initial Question:\n{self.initial_prompt}")
        if last_message:
            combined_parts.append(f"Last message:\n{last_message}")
        combined_query = "\n\n".join(combined_parts)

        if hasattr(self.base_agent, "generate_response_with_react"):
            response, trajectory = self.base_agent.generate_response_with_react(
                query=combined_query,
                components=None,
                context=None
            )
        else:
            response = self.base_agent.generate_response(combined_query)
            trajectory = None

        content = response.content if response else ""
        if self._should_withdraw(content):
            return {
                "role": "assistant",
                "content": "WITHDRAW"
            }
        if trajectory:
            content = f"{content}\n\n{self._format_trajectory(trajectory)}"

        return {
            "role": "assistant",
            "content": content
        }

    @staticmethod
    def _get_last_debate_message(messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") in {"assistant", "user"} and msg.get("content"):
                return msg.get("content", "")
        return ""

    @staticmethod
    def _get_latest_phase_instruction(messages: List[Dict], agent_name: Optional[str] = None) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "system" and msg.get("content"):
                if agent_name:
                    msg_name = msg.get("name")
                    if msg_name and msg_name != agent_name:
                        continue
                return msg.get("content", "")
        return ""

    @staticmethod
    def _get_fallback_prompt(messages: List[Dict]) -> str:
        for msg in reversed(messages):
            if msg.get("content"):
                return msg.get("content", "")
        return ""

    @staticmethod
    def _should_withdraw(content: str) -> bool:
        if not content:
            return False
        lower = content.lower()
        return "not a valid model id" in lower or "llm调用失败" in content

    @staticmethod
    def _format_trajectory(trajectory: ReActTrajectory) -> str:
        lines = ["=== ReAct Trajectory ==="]
        for step in trajectory.steps:
            lines.append(f"Step {step.step_number}:")
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action_name}")
            lines.append(f"Observation: {step.observation}")
            lines.append("")
        if trajectory.final_answer:
            lines.append(f"Final Answer: {trajectory.final_answer}")
        return "\n".join(lines)


@dataclass
class DebateResult:
    """
    辩论结果数据类
    封装辩论的最终结果和过程信息
    """
    consensus_reached: bool  # 是否达成共识
    final_products: Optional[str]  # 最终产物结论
    final_performance: Optional[str]  # 最终性能指标
    reasoning_trajectory: str  # 推理轨迹
    debate_rounds: int  # 辩论轮数
    debate_history: List[Dict]  # 完整辩论历史
    time_elapsed: float  # 耗时
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "consensus_reached": self.consensus_reached,
            "final_products": self.final_products,
            "final_performance": self.final_performance,
            "reasoning_trajectory": self.reasoning_trajectory,
            "debate_rounds": self.debate_rounds,
            "debate_history": self.debate_history,
            "time_elapsed": self.time_elapsed
        }


class AutoGenDebateCoordinator:
    """
    AutoGen辩论协调器
    完全依赖AutoGen GroupChat管理辩论流程，无需传统的Debate Manager
    """
    
    def __init__(
        self,
        agents: List[ReActAgent],
        config: Dict
    ):
        """
        初始化AutoGen辩论协调器
        
        Args:
            agents: Agent列表
            config: 辩论配置
        """
        self.agents = agents
        self.config = config
        
        # 辩论参数
        self.max_rounds = config.get('max_rounds', 10)
        self.consensus_threshold = config.get('consensus_threshold', 3)
        
        # AutoGen组件
        self.autogen_agents = []
        self.group_chat = None
        self.manager = None
        self.flow_controller = None
    
    def start_debate(
        self,
        components: List[str],
        initial_prompt: Optional[str] = None,
        reaction_type: Optional[str] = None
    ) -> DebateResult:
        """
        开始辩论
        
        Args:
            components: 金属催化剂元素列表
            initial_prompt: 初始提示（可选）
        
        Returns:
            DebateResult: 辩论结果
        """
        debate_id = make_debate_id("autogen", components, reaction_type)
        logger.info(
            "autogen_debate_start",
            extra={
                "event": "autogen.debate.start",
                "debate_id": debate_id,
                "components": components,
                "reaction_type": reaction_type,
                "max_rounds": self.max_rounds,
                "consensus_threshold": self.consensus_threshold,
            },
        )

        print("=" * 60)
        print("Starting Multi-Agent Debate (AutoGen Mode)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Build initial debate prompt
        if initial_prompt is None:
            components_str = ", ".join(components)
            if initial_prompt is None:
                initial_prompt = build_initial_debate_prompt(components, reaction_type)
        self.initial_prompt = initial_prompt
        
        # 为每个Agent准备RAG增强提示
        self._prepare_agents_with_rag(components, initial_prompt)

        # 创建AutoGen agents和GroupChat
        self._create_autogen_group_chat()
        
        try:
            # 使用第一个agent发起对话
            first_agent = self.autogen_agents[0]
            first_agent.initiate_chat(
                self.manager,
                message=initial_prompt,
                max_turns=self.max_rounds * len(self.agents)
            )
            
            # 提取辩论历史
            debate_history = self.group_chat.messages
            
            # 分析结果
            result = self._extract_consensus_from_history(debate_history, components)
            
            elapsed_time = time.time() - start_time
            result.time_elapsed = elapsed_time

            logger.info(
                "autogen_debate_end",
                extra={
                    "event": "autogen.debate.end",
                    "debate_id": debate_id,
                    "time_elapsed": elapsed_time,
                    "consensus_reached": getattr(result, "consensus_reached", None),
                    "debate_rounds": getattr(result, "debate_rounds", None),
                },
            )

            # Artifacts (structured):
            try:
                payload = {
                    "debate_id": debate_id,
                    "run_id": get_run_id() or None,
                    "engine": "autogen",
                    "reaction_type": reaction_type,
                    "components": components,
                    "result": result.to_dict(),
                }
                paths = write_debate_artifacts(
                    debate_id=debate_id,
                    engine="autogen",
                    payload=payload,
                    transcript_events=debate_history,
                )
                logger.info(
                    "autogen_artifacts_written",
                    extra={
                        "event": "autogen.artifacts.written",
                        "debate_id": debate_id,
                        "full_path": paths.get("full_path"),
                        "transcript_path": paths.get("transcript_path"),
                    },
                )
            except Exception:
                logger.exception(
                    "autogen_artifacts_write_failed",
                    extra={"event": "autogen.artifacts.error", "debate_id": debate_id},
                )
            
            print("\n" + "=" * 60)
            print(f"Debate Ended (Time: {elapsed_time:.2f}s)")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            logger.exception("autogen_debate_error", extra={"event": "autogen.debate.error"})
            print(f"Error during debate: {str(e)}")
            elapsed_time = time.time() - start_time
            try:
                payload = {
                    "debate_id": debate_id,
                    "run_id": get_run_id() or None,
                    "engine": "autogen",
                    "reaction_type": reaction_type,
                    "components": components,
                    "error": str(e),
                }
                write_debate_artifacts(
                    debate_id=debate_id,
                    engine="autogen",
                    payload=payload,
                    transcript_events=[],
                )
            except Exception:
                pass
            
            return DebateResult(
                consensus_reached=False,
                final_products=None,
                final_performance=None,
                reasoning_trajectory=f"Debate failed: {str(e)}",
                debate_rounds=0,
                debate_history=[],
                time_elapsed=elapsed_time
            )
    
    
    def _prepare_agents_with_rag(self, components: List[str], initial_prompt: str) -> None:
        """
        为每个Agent准备RAG增强的系统消息
        
        Args:
            components: 金属催化剂元素列表
            initial_prompt: 初始提示
        """
        self.enhanced_prompts = {}
        self.rag_contexts = {}
        
        for agent in self.agents:
            # 使用Agent自己的RAG系统增强提示
            retrieved_contexts = []
            knowledge_results = agent.retrieve_knowledge(initial_prompt)
            if knowledge_results:
                for result in knowledge_results:
                    text = result.get("text") if isinstance(result, dict) else str(result)
                    if text:
                        retrieved_contexts.append(text)
            enhanced_prompt = agent.format_prompt_with_rag(
                query=initial_prompt,
                retrieved_contexts=retrieved_contexts
            )

            self.enhanced_prompts[agent.agent_id] = enhanced_prompt
            self.rag_contexts[agent.agent_id] = "\n".join(retrieved_contexts) if retrieved_contexts else ""
    
    def _create_autogen_group_chat(self) -> None:
        """
        创建AutoGen GroupChat和相关组件
        """
        # 转换为AutoGen agents
        self.autogen_agents = []

        for agent in self.agents:
            # 获取LLM配置
            llm_config = self._get_llm_config_for_autogen(agent)

            autogen_agent = ReActAutoGenAgent(
                name=agent.name.replace(" ", "_"),
                llm_config=llm_config,
                system_message=UNIFIED_SYSTEM_PROMPT,
                base_agent=agent,
                initial_prompt=getattr(self, "initial_prompt", "")
            )

            self.autogen_agents.append(autogen_agent)

        print(f"Created {len(self.autogen_agents)} AutoGen agents")

        # 初始化辩论流程控制器
        rag_contexts_by_name = {}
        for agent in self.agents:
            agent_name = agent.name.replace(" ", "_")
            rag_contexts_by_name[agent_name] = self.rag_contexts.get(agent.agent_id, "")

        self.flow_controller = DebateFlowController(
            agent_names=[agent.name.replace(" ", "_") for agent in self.autogen_agents],
            max_rounds=self.max_rounds,
            rag_contexts=rag_contexts_by_name,
            initial_prompt=getattr(self, "initial_prompt", "")
        )

        # 创建GroupChat
        self.group_chat = GroupChat(
            agents=self.autogen_agents,
            messages=[],
            max_round=self.max_rounds * len(self.agents),
            speaker_selection_method=self.flow_controller.select_speaker,
            allow_repeat_speaker=True
        )

        # 创建GroupChatManager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            is_termination_msg=self.flow_controller.is_termination_msg
        )
    
    def _get_llm_config_for_autogen(self, agent: ReActAgent) -> Dict:
        """
        为每个 ConversableAgent 生成 AutoGen 格式的 LLM 配置
        
        Args:
            agent: 自定义Agent
        
        Returns:
            Dict: AutoGen格式的LLM配置
        """
        model_config = agent.model_config
        
        config = {
            "model": model_config.get('model', 'gpt-4'),
            "api_key": model_config.get('api_key'),
            "base_url": model_config.get('base_url', 'https://openrouter.ai/api/v1'),
            "temperature": model_config.get('temperature', 0.9),
            "max_tokens": model_config.get('max_tokens', 2000)
        }
        
        return config
    
    def _extract_consensus_from_history(
        self,
        chat_history: List[Dict],
        components: List[str]
    ) -> DebateResult:
        """
        从对话历史中提取共识结果
        
        Args:
            chat_history: AutoGen的对话历史
            components: 金属催化剂元素列表
        
        Returns:
            DebateResult: 辩论结果
        """
        if not chat_history:
            return DebateResult(
                consensus_reached=False,
                final_products=None,
                final_performance=None,
                reasoning_trajectory="No debate history available",
                debate_rounds=0,
                debate_history=[],
                time_elapsed=0
            )
        
        active_agent_names = None
        if self.flow_controller:
            active_agent_names = [
                name for name, is_active in self.flow_controller.active_agents.items() if is_active
            ]

        # 统计最后一轮所有agent的意见
        num_agents = len(self.agents)
        recent_responses = chat_history[-num_agents:] if len(chat_history) >= num_agents else chat_history
        
        product_votes = {}
        performance_map = {}
        
        # 解析每个响应
        for msg in recent_responses:
            content = msg.get('content', '')
            sender = msg.get('name', '')
            if active_agent_names is not None and sender not in active_agent_names:
                continue
            
            products = self._extract_products(content)
            performance = self._extract_performance(content)

            if products:
                normalized = products.strip().lower()
                if normalized not in product_votes:
                    product_votes[normalized] = []
                    performance_map[normalized] = []
                product_votes[normalized].append(sender)
                if performance:
                    performance_map[normalized].append(performance)
        
        # 确定共识
        consensus_reached = False
        final_products = None
        final_performance = None

        if product_votes:
            final_products_norm = max(product_votes.items(), key=lambda x: len(x[1]))[0]
            num_supporters = len(product_votes[final_products_norm])

            required_consensus = int((len(active_agent_names) if active_agent_names is not None else num_agents) * 0.75)
            required_consensus = max(2, required_consensus)
            consensus_reached = num_supporters >= required_consensus

            final_products = final_products_norm
            if performance_map.get(final_products_norm):
                final_performance = performance_map[final_products_norm][-1]
        
        # 构建推理轨迹
        reasoning_trajectory = self._build_reasoning_trajectory(chat_history)
        
        # 计算辩论轮数
        debate_rounds = len(chat_history) // num_agents
        
        # 转换历史格式
        formatted_history = []
        for msg in chat_history:
            content = msg.get('content', '')
            products = self._extract_products(content)
            performance = self._extract_performance(content)
            formatted_history.append({
                'agent': msg.get('name', 'Unknown'),
                'content': content,
                'role': msg.get('role', 'assistant'),
                'products': products,
                'performance': performance
            })
        
        return DebateResult(
            consensus_reached=consensus_reached,
            final_products=final_products,
            final_performance=final_performance,
            reasoning_trajectory=reasoning_trajectory,
            debate_rounds=debate_rounds,
            debate_history=formatted_history,
            time_elapsed=0  # Will be set by caller
        )
    
    def _extract_products(self, content: str) -> Optional[str]:
        """提取产物描述"""
        if not content:
            return None
        patterns = [r"\*\*Products\*\*\s*:\s*(.+)", r"Products\s*:\s*(.+)", r"产物\s*[:：]\s*(.+)"]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_performance(self, content: str) -> Optional[str]:
        """提取性能指标"""
        if not content:
            return None
        patterns = [r"\*\*Performance\*\*\s*:\s*(.+)", r"Performance\s*:\s*(.+)", r"性能\s*[:：]\s*(.+)"]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _build_reasoning_trajectory(self, chat_history: List[Dict]) -> str:
        """
        构建推理轨迹 - 提取完整的LLM推理链条
        
        Args:
            chat_history: 对话历史
        
        Returns:
            str: 完整推理轨迹文本（包含LLM的完整thinking过程）
        """
        trajectory_parts = ["=== Debate Reasoning Trajectory ===\n"]
        
        for i, msg in enumerate(chat_history, 1):
            agent_name = msg.get('name', 'Unknown')
            content = msg.get('content', '')
            
            # 提取关键信息
            products = self._extract_products(content)
            performance = self._extract_performance(content)
            
            # 保留完整的LLM推理内容
            trajectory_parts.append(f"\n[Round {(i-1)//len(self.agents) + 1}] {agent_name}:")
            if products:
                trajectory_parts.append(f"  Products: {products}")
            if performance:
                trajectory_parts.append(f"  Performance: {performance}")
            trajectory_parts.append(f"  Complete Reasoning:\n{content}")
            trajectory_parts.append("-" * 80)  # 添加分隔线以便区分不同agent的推理
        
        return "\n".join(trajectory_parts)


class DebateFlowController:
    """
    去中心化、多轮淘汰制辩论流程控制器
    通过 GroupChat 的 speaker_selection_method 和 is_termination_msg 管理流程
    """

    def __init__(self, agent_names: List[str], max_rounds: int, rag_contexts: Optional[Dict[str, str]] = None, initial_prompt: str = "") -> None:
        self.agent_names = list(agent_names)
        self.max_rounds = max_rounds
        self.phase = "construction"
        self.active_agents = {name: True for name in self.agent_names}
        self.turn_index = 0
        self.defense_turns = 0
        self.last_messages = {name: "" for name in self.agent_names}
        self.rag_contexts = rag_contexts or {}
        self.initial_prompt = initial_prompt

        self.phase_spoken = {
            "construction": set(),
            "critique": set()
        }

        self.phase_limits = {
            "construction": len(self.agent_names),
            "critique": len(self.agent_names)
        }

        self.phase_instructions = {
    "construction": (
        "Based on the given conditions, perform ReAct retrieval and independently propose what you consider the most likely product and key performance metrics."
        "You must explicitly present the 'Thought -> Action -> Observation' trajectory."
    ),
    "critique": (
        "Read the ReAct trajectories from other researchers in previous rounds. Point out issues such as outdated data, mechanistic contradictions, or insufficient evidence."
        "Do not repeat your own viewpoints; focus exclusively on attacking."
    ),
    "defense": (
        "In response to the attacks received, perform ReAct again to search for new evidence. If you can strongly refute the criticism, update your claim;"
        "if you cannot refute it, you must reply **'WITHDRAW'**."
    )
}


    def _active_agent_list(self) -> List[str]:
        return [name for name in self.agent_names if self.active_agents.get(name)]

    def _advance_phase_if_needed(self) -> None:
        if self.phase == "construction" and self._phase_completed("construction"):
            self.phase = "critique"
            self.turn_index = 0
            self.phase_spoken["construction"].clear()
        elif self.phase == "critique" and self._phase_completed("critique"):
            self.phase = "defense"
            self.turn_index = 0
            self.phase_spoken["critique"].clear()

    def _next_active_speaker(self) -> Optional[str]:
        active_list = self._active_agent_list()
        if not active_list:
            return None

        for _ in range(len(self.agent_names)):
            candidate = self.agent_names[self.turn_index % len(self.agent_names)]
            self.turn_index += 1
            if self.active_agents.get(candidate):
                return candidate
        return active_list[0]

    def _inject_phase_instruction(self, messages: List[Dict], speaker: str) -> None:
        instruction = self.phase_instructions.get(self.phase)
        if not instruction:
            return
        rag_context = self.rag_contexts.get(speaker)
        if rag_context:
            instruction = f"{instruction}\n\n## Reference / Context\n{rag_context}"
        if self.initial_prompt:
            instruction = f"{instruction}\n\n## Initial Question\n{self.initial_prompt}"
        messages.append({
            "role": "system",
            "name": speaker,
            "content": f"【Debate Phase: {self.phase.upper()}】{instruction}"
        })

    def select_speaker(self, last_speaker: ConversableAgent, groupchat: GroupChat) -> Optional[ConversableAgent]:
        self._advance_phase_if_needed()

        next_speaker = self._next_active_speaker()
        if next_speaker is None:
            return None

        self._inject_phase_instruction(groupchat.messages, next_speaker)
        selected_agent = groupchat.agent_by_name(next_speaker)
        if selected_agent is None:
            return groupchat.next_agent(last_speaker)
        return selected_agent

    def is_termination_msg(self, msg: Dict) -> bool:
        content = (msg.get("content") or "").strip()
        sender = msg.get("name")

        if sender:
            self.last_messages[sender] = content
            self._record_phase_turn(sender)

        if "WITHDRAW" in content.upper() and sender:
            self.active_agents[sender] = False
            self.phase_spoken["construction"].discard(sender)
            self.phase_spoken["critique"].discard(sender)

        active_list = self._active_agent_list()
        if len(active_list) <= 1:
            return True

        if self.phase == "defense":
            if sender in active_list:
                self.defense_turns += 1
            if self._active_views_converged():
                return True
            if self.defense_turns >= self.max_rounds * len(self.agent_names):
                return True

        return False

    def _record_phase_turn(self, sender: str) -> None:
        if self.phase in {"construction", "critique"} and sender in self._active_agent_list():
            self.phase_spoken[self.phase].add(sender)

    def _phase_completed(self, phase: str) -> bool:
        active = set(self._active_agent_list())
        return active and self.phase_spoken.get(phase) == active

    def _active_views_converged(self) -> bool:
        active_list = self._active_agent_list()
        if len(active_list) <= 1:
            return True

        products = []
        for name in active_list:
            product = self._extract_products(self.last_messages.get(name, ""))
            if not product:
                return False
            products.append(product.strip().lower())

        return len(set(products)) == 1

    @staticmethod
    def _extract_products(content: str) -> Optional[str]:
        if not content:
            return None
        patterns = [r"\*\*Products\*\*\s*:\s*(.+)", r"Products\s*:\s*(.+)", r"产物\s*[:：]\s*(.+)"]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
