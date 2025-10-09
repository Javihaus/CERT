"""
AutoGen integration for CERT SDK.

Provides instrumentation for AutoGen agents, group chats, and multi-agent
conversations with automatic CERT metrics tracking.
"""

from typing import Any, Dict, List, Optional, Callable
import time

from cert.integrations.base import CERTIntegration, PipelineMetrics
from cert.providers.base import ProviderInterface
from cert.models import ModelBaseline
from cert.analysis.quality import QualityScorer


class CERTAutoGen(CERTIntegration):
    """
    CERT integration for AutoGen.

    Instruments AutoGen agents, group chats, and conversations with automatic
    CERT metrics collection including:
    - Agent execution tracking
    - Quality scoring
    - Coordination effect measurement
    - Pipeline health monitoring

    Example:
        >>> import cert
        >>> from cert.integrations.autogen import CERTAutoGen
        >>> import autogen
        >>>
        >>> # Create your AutoGen agents
        >>> llm_config = {"config_list": [{"model": "gpt-4", "api_key": "..."}]}
        >>>
        >>> researcher = autogen.AssistantAgent(
        ...     name="Researcher",
        ...     llm_config=llm_config,
        ...     system_message="You are a research expert."
        ... )
        >>>
        >>> writer = autogen.AssistantAgent(
        ...     name="Writer",
        ...     llm_config=llm_config,
        ...     system_message="You are a professional writer."
        ... )
        >>>
        >>> user_proxy = autogen.UserProxyAgent(
        ...     name="UserProxy",
        ...     code_execution_config=False,
        ... )
        >>>
        >>> # Wrap with CERT instrumentation
        >>> cert_integration = CERTAutoGen(
        ...     provider=cert.create_provider(api_key="...", model_name="gpt-4o"),
        ... )
        >>>
        >>> # Create instrumented group chat
        >>> groupchat = cert_integration.create_instrumented_groupchat(
        ...     agents=[researcher, writer],
        ...     max_round=10,
        ... )
        >>>
        >>> manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        >>>
        >>> # Run with automatic metrics collection
        >>> user_proxy.initiate_chat(manager, message="Your task")
        >>>
        >>> # View metrics
        >>> cert_integration.print_metrics()
    """

    def __init__(
        self,
        provider: Optional[ProviderInterface] = None,
        baseline: Optional[ModelBaseline] = None,
        scorer: Optional[QualityScorer] = None,
        track_all_executions: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize AutoGen integration.

        Args:
            provider: CERT provider for baseline comparison.
            baseline: Model baseline for comparison.
            scorer: Quality scorer for output evaluation.
            track_all_executions: Track all agent executions (default: True).
            verbose: Print execution details (default: False).
        """
        super().__init__(provider, baseline, scorer, track_all_executions)
        self.verbose = verbose
        self._conversation_history: List[Dict[str, Any]] = []
        self._agent_registry: Dict[str, Any] = {}

    def wrap_agent(
        self,
        agent: Any,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        calculate_quality: bool = True,
    ) -> Any:
        """
        Wrap an AutoGen agent with CERT instrumentation.

        This wraps the agent's generate_reply method to automatically
        track executions and calculate metrics.

        Args:
            agent: AutoGen AssistantAgent or UserProxyAgent instance.
            agent_id: Unique identifier (defaults to agent.name).
            agent_name: Human-readable name (defaults to agent.name).
            calculate_quality: Calculate quality scores for outputs (default: True).

        Returns:
            Instrumented agent with same interface.

        Example:
            >>> instrumented = cert_integration.wrap_agent(
            ...     agent=my_agent,
            ...     calculate_quality=True,
            ... )
        """
        # Use agent's name if not provided
        agent_id = agent_id or agent.name
        agent_name = agent_name or agent.name

        # Store agent in registry
        self._agent_registry[agent_id] = {
            "agent": agent,
            "name": agent_name,
            "calculate_quality": calculate_quality,
        }

        # Store original generate_reply method
        original_generate_reply = agent.generate_reply

        def instrumented_generate_reply(messages=None, sender=None, config=None):
            """Instrumented generate_reply method."""
            # Extract input
            input_text = ""
            if messages:
                last_msg = messages[-1] if isinstance(messages, list) else messages
                if isinstance(last_msg, dict):
                    input_text = last_msg.get("content", "")
                else:
                    input_text = str(last_msg)

            if self.verbose:
                print(f"\n[CERT] {agent_name} generating reply...")

            # Execute with timing
            start_time = time.time()
            reply = original_generate_reply(messages=messages, sender=sender, config=config)
            duration_ms = (time.time() - start_time) * 1000

            # Extract output
            output_text = ""
            if isinstance(reply, str):
                output_text = reply
            elif isinstance(reply, dict):
                output_text = reply.get("content", str(reply))
            else:
                output_text = str(reply)

            if self.verbose:
                print(f"[CERT] {agent_name} completed in {duration_ms:.0f}ms")

            # Record execution
            self.record_execution(
                agent_id=agent_id,
                agent_name=agent_name,
                input_text=input_text,
                output_text=output_text,
                duration_ms=duration_ms,
                metadata={"sender": sender.name if sender and hasattr(sender, "name") else None},
            )

            # Calculate quality if requested
            if calculate_quality and input_text and output_text:
                quality = self.calculate_quality(input_text, output_text)
                self.metrics.intermediate_qualities.append(quality)
                self.metrics.output_quality = quality

                if self.verbose:
                    print(f"[CERT] Quality score: {quality:.3f}")

            # Store in conversation history
            self._conversation_history.append({
                "agent": agent_name,
                "input": input_text,
                "output": output_text,
                "timestamp": time.time(),
            })

            return reply

        # Replace the method
        agent.generate_reply = instrumented_generate_reply

        return agent

    def create_instrumented_groupchat(
        self,
        agents: List[Any],
        max_round: int = 10,
        **groupchat_kwargs
    ) -> Any:
        """
        Create an AutoGen GroupChat with automatic CERT instrumentation.

        This wraps all agents in the group chat and tracks the entire
        conversation flow.

        Args:
            agents: List of AutoGen Agent instances.
            max_round: Maximum number of conversation rounds (default: 10).
            **groupchat_kwargs: Additional arguments passed to GroupChat constructor.

        Returns:
            Instrumented GroupChat instance.

        Example:
            >>> groupchat = cert_integration.create_instrumented_groupchat(
            ...     agents=[researcher, writer, critic],
            ...     max_round=15,
            ... )
            >>> manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
            >>> user_proxy.initiate_chat(manager, message="Your task")
        """
        try:
            import autogen
        except ImportError:
            raise ImportError(
                "AutoGen is not installed. Install it with: pip install pyautogen"
            )

        # Reset metrics for new conversation
        self.reset_metrics()

        # Wrap all agents
        wrapped_agents = []
        for agent in agents:
            wrapped = self.wrap_agent(agent)
            wrapped_agents.append(wrapped)

        # Create group chat
        groupchat = autogen.GroupChat(
            agents=wrapped_agents,
            messages=[],
            max_round=max_round,
            **groupchat_kwargs
        )

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"CERT: AutoGen GroupChat Created")
            print(f"{'='*70}")
            print(f"Agents: {[a.name for a in wrapped_agents]}")
            print(f"Max rounds: {max_round}")
            print(f"{'='*70}")

        return groupchat

    def wrap_conversation(
        self,
        initiator: Any,
        recipient: Any,
        message: str,
    ) -> Callable:
        """
        Wrap an AutoGen two-agent conversation with CERT instrumentation.

        Args:
            initiator: The agent initiating the conversation (usually UserProxyAgent).
            recipient: The recipient agent (usually AssistantAgent).
            message: The initial message to start the conversation.

        Returns:
            Callable that executes the conversation and returns result.

        Example:
            >>> conversation = cert_integration.wrap_conversation(
            ...     initiator=user_proxy,
            ...     recipient=assistant,
            ...     message="Solve this problem..."
            ... )
            >>> result = conversation()
            >>> cert_integration.print_metrics()
        """
        # Wrap both agents
        wrapped_initiator = self.wrap_agent(initiator)
        wrapped_recipient = self.wrap_agent(recipient)

        def execute_conversation():
            """Execute the wrapped conversation."""
            self.reset_metrics()

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"CERT: Starting AutoGen Conversation")
                print(f"{'='*70}")
                print(f"Initiator: {wrapped_initiator.name}")
                print(f"Recipient: {wrapped_recipient.name}")

            start_time = time.time()

            # Initiate chat
            result = wrapped_initiator.initiate_chat(
                wrapped_recipient,
                message=message,
            )

            duration_ms = (time.time() - start_time) * 1000

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"CERT: Conversation Complete ({duration_ms:.0f}ms)")
                print(f"{'='*70}")

            # Calculate coordination effect
            if len(self.metrics.intermediate_qualities) >= 2:
                gamma = self.calculate_coordination_effect()
                if self.verbose and gamma:
                    print(f"Coordination Effect: γ = {gamma:.3f}")

            # Calculate pipeline health
            health = self.calculate_pipeline_health()
            if self.verbose and health:
                print(f"Pipeline Health: H = {health:.3f}")

            return result

        return execute_conversation

    def wrap_pipeline(self, pipeline: Any) -> Any:
        """
        Wrap an AutoGen pipeline (GroupChat or conversation) with CERT instrumentation.

        For AutoGen, use create_instrumented_groupchat() or wrap_conversation()
        instead, as AutoGen doesn't have a direct "pipeline" concept.

        Args:
            pipeline: AutoGen GroupChat instance.

        Returns:
            The pipeline (agents should be wrapped individually).
        """
        # AutoGen doesn't have a pipeline concept
        # Instrumentation happens at the agent level
        return pipeline

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history with metrics.

        Returns:
            List of conversation turns with agent names, inputs, outputs, and timestamps.
        """
        return self._conversation_history

    def analyze_conversation(self) -> Dict[str, Any]:
        """
        Analyze the conversation and return detailed metrics.

        Returns:
            Dictionary with conversation analysis including:
            - Number of turns
            - Agent participation
            - Average quality per agent
            - Coordination patterns
        """
        if not self._conversation_history:
            return {"error": "No conversation history available"}

        # Count agent participation
        agent_turns = {}
        agent_qualities = {}

        for i, turn in enumerate(self._conversation_history):
            agent = turn["agent"]
            agent_turns[agent] = agent_turns.get(agent, 0) + 1

            # Get quality for this turn if available
            if i < len(self.metrics.intermediate_qualities):
                quality = self.metrics.intermediate_qualities[i]
                if agent not in agent_qualities:
                    agent_qualities[agent] = []
                agent_qualities[agent].append(quality)

        # Calculate average quality per agent
        agent_avg_quality = {
            agent: sum(qualities) / len(qualities)
            for agent, qualities in agent_qualities.items()
        }

        return {
            "total_turns": len(self._conversation_history),
            "agent_participation": agent_turns,
            "agent_average_quality": agent_avg_quality,
            "overall_quality": self.metrics.output_quality,
            "coordination_effect": self.metrics.coordination_effect,
            "health_score": self.metrics.health_score,
        }
