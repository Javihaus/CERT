"""
LangChain/LangGraph integration for CERT SDK.

Provides instrumentation for LangChain agents and LangGraph multi-agent pipelines
with automatic CERT metrics tracking.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from cert.analysis.quality import QualityScorer
from cert.integrations.base import CERTIntegration
from cert.models import ModelBaseline
from cert.providers.base import ProviderInterface


class CERTLangChain(CERTIntegration):
    """
    CERT integration for LangChain/LangGraph.

    Instruments LangChain agents and LangGraph pipelines with automatic
    CERT metrics collection including:
    - Agent execution tracking
    - Quality scoring
    - Coordination effect measurement
    - Pipeline health monitoring

    Example:
        >>> import cert
        >>> from cert.integrations.langchain import CERTLangChain
        >>> from langgraph.prebuilt import create_react_agent
        >>>
        >>> # Create your LangChain agent
        >>> agent = create_react_agent(model, tools)
        >>>
        >>> # Wrap with CERT instrumentation
        >>> cert_integration = CERTLangChain(
        ...     provider=cert.create_provider(api_key="...", model_name="gpt-4o"),
        ... )
        >>> instrumented_agent = cert_integration.wrap_agent(
        ...     agent=agent,
        ...     agent_id="researcher",
        ...     agent_name="Research Agent",
        ... )
        >>>
        >>> # Run with automatic metrics collection
        >>> result = instrumented_agent.invoke({"messages": [input]})
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
        Initialize LangChain integration.

        Args:
            provider: CERT provider for baseline comparison.
            baseline: Model baseline for comparison.
            scorer: Quality scorer for output evaluation.
            track_all_executions: Track all agent executions (default: True).
            verbose: Print execution details (default: False).
        """
        super().__init__(provider, baseline, scorer, track_all_executions)
        self.verbose = verbose
        self._agent_registry: Dict[str, Any] = {}

    def wrap_agent(
        self,
        agent: Any,
        agent_id: str,
        agent_name: str,
        calculate_quality: bool = True,
    ) -> Any:
        """
        Wrap a LangChain agent with CERT instrumentation.

        This wraps the agent's invoke/stream methods to automatically
        track executions and calculate metrics.

        Args:
            agent: LangChain agent or AgentExecutor instance.
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable name for the agent.
            calculate_quality: Calculate quality scores for outputs (default: True).

        Returns:
            Instrumented agent with same interface.

        Example:
            >>> instrumented = cert_integration.wrap_agent(
            ...     agent=my_agent,
            ...     agent_id="writer",
            ...     agent_name="Content Writer",
            ... )
            >>> result = instrumented.invoke({"messages": [input]})
        """
        # Store agent in registry
        self._agent_registry[agent_id] = {
            "agent": agent,
            "name": agent_name,
            "calculate_quality": calculate_quality,
        }

        # Create wrapper
        class InstrumentedAgent:
            def __init__(wrapper_self, base_agent, integration, agent_id, agent_name, calc_quality):
                wrapper_self._agent = base_agent
                wrapper_self._integration = integration
                wrapper_self._agent_id = agent_id
                wrapper_self._agent_name = agent_name
                wrapper_self._calculate_quality = calc_quality

            def invoke(wrapper_self, *args, **kwargs):
                """Instrumented invoke method."""
                return wrapper_self._integration._instrument_execution(
                    agent=wrapper_self._agent,
                    agent_id=wrapper_self._agent_id,
                    agent_name=wrapper_self._agent_name,
                    method="invoke",
                    calculate_quality=wrapper_self._calculate_quality,
                    args=args,
                    kwargs=kwargs,
                )

            def stream(wrapper_self, *args, **kwargs):
                """Instrumented stream method."""
                # For streaming, we collect the full output
                return wrapper_self._integration._instrument_stream(
                    agent=wrapper_self._agent,
                    agent_id=wrapper_self._agent_id,
                    agent_name=wrapper_self._agent_name,
                    calculate_quality=wrapper_self._calculate_quality,
                    args=args,
                    kwargs=kwargs,
                )

            def __getattr__(wrapper_self, name):
                """Forward other attributes to original agent."""
                return getattr(wrapper_self._agent, name)

        return InstrumentedAgent(agent, self, agent_id, agent_name, calculate_quality)

    def _instrument_execution(
        self,
        agent: Any,
        agent_id: str,
        agent_name: str,
        method: str,
        calculate_quality: bool,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Instrument a single agent execution."""
        # Extract input
        input_text = self._extract_input(args, kwargs)

        if self.verbose:
            print(f"\n[CERT] Starting {agent_name} execution...")

        # Execute with timing
        start_time = time.time()
        result = getattr(agent, method)(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000

        # Extract output
        output_text = self._extract_output(result)

        if self.verbose:
            print(f"[CERT] {agent_name} completed in {duration_ms:.0f}ms")

        # Record execution
        self.record_execution(
            agent_id=agent_id,
            agent_name=agent_name,
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            metadata={"method": method},
        )

        # Calculate quality if requested
        if calculate_quality and input_text and output_text:
            quality = self.calculate_quality(input_text, output_text)
            self.metrics.intermediate_qualities.append(quality)

            # If this is the last agent, store as output quality
            self.metrics.output_quality = quality

            if self.verbose:
                print(f"[CERT] Quality score: {quality:.3f}")

        return result

    def _instrument_stream(
        self,
        agent: Any,
        agent_id: str,
        agent_name: str,
        calculate_quality: bool,
        args: tuple,
        kwargs: dict,
    ):
        """Instrument streaming execution."""
        input_text = self._extract_input(args, kwargs)

        if self.verbose:
            print(f"\n[CERT] Starting {agent_name} streaming execution...")

        start_time = time.time()
        collected_output = []

        # Stream and collect
        for chunk in agent.stream(*args, **kwargs):
            collected_output.append(chunk)
            yield chunk

        duration_ms = (time.time() - start_time) * 1000

        # Extract full output from collected chunks
        output_text = self._extract_output_from_chunks(collected_output)

        if self.verbose:
            print(f"\n[CERT] {agent_name} completed in {duration_ms:.0f}ms")

        # Record execution
        self.record_execution(
            agent_id=agent_id,
            agent_name=agent_name,
            input_text=input_text,
            output_text=output_text,
            duration_ms=duration_ms,
            metadata={"method": "stream"},
        )

        # Calculate quality
        if calculate_quality and input_text and output_text:
            quality = self.calculate_quality(input_text, output_text)
            self.metrics.intermediate_qualities.append(quality)
            self.metrics.output_quality = quality

            if self.verbose:
                print(f"[CERT] Quality score: {quality:.3f}")

    def wrap_pipeline(self, pipeline: Any) -> Any:
        """
        Wrap a LangGraph pipeline with CERT instrumentation.

        For LangGraph, agents should be wrapped individually before
        being added to the graph. This method is provided for compatibility
        but typically you'll use wrap_agent() for each node in your graph.

        Args:
            pipeline: LangGraph graph or compiled graph.

        Returns:
            The pipeline (agents should be wrapped individually).
        """
        # For LangGraph, we instrument individual agents
        # The graph itself doesn't need wrapping
        return pipeline

    def _extract_input(self, args: tuple, kwargs: dict) -> str:
        """Extract input text from agent arguments."""
        if args and isinstance(args[0], dict):
            # Pattern 1: LangChain messages format: {"messages": [...]}
            messages = args[0].get("messages", [])
            if messages:
                # Get the last user message
                if isinstance(messages[-1], dict):
                    return messages[-1].get("content", "")
                if hasattr(messages[-1], "content"):
                    return messages[-1].content
                return str(messages[-1])

            # Pattern 2: LCEL chain format: {"input": "..."}
            if "input" in args[0]:
                return str(args[0]["input"])

        # Fallback
        return str(args[0]) if args else ""

    def _extract_output(self, result: Any) -> str:
        """Extract output text from agent result."""
        # Pattern 1: LangChain messages format: {"messages": [...]}
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                # Get the last assistant message
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    return last_msg.get("content", "")
                if hasattr(last_msg, "content"):
                    return last_msg.content
                return str(last_msg)

        # Pattern 2: LCEL chain result: AIMessage object
        if hasattr(result, "content"):
            return str(result.content)

        # Fallback
        return str(result)

    def _extract_output_from_chunks(self, chunks: List[Any]) -> str:
        """Extract output text from streaming chunks."""
        # Collect all content from chunks
        content_parts = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                messages = chunk.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        content = last_msg.get("content", "")
                    elif hasattr(last_msg, "content"):
                        content = last_msg.content
                    else:
                        content = str(last_msg)

                    if content:
                        content_parts.append(content)

        return " ".join(content_parts) if content_parts else ""

    def create_multi_agent_pipeline(
        self,
        agents: List[Dict[str, Any]],
    ) -> Callable:
        """
        Create a simple multi-agent pipeline with automatic coordination tracking.

        This is a helper for creating simple sequential pipelines where
        each agent processes the output of the previous agent.

        Args:
            agents: List of agent configs, each with:
                - "agent": The LangChain agent instance
                - "agent_id": Unique identifier
                - "agent_name": Display name
                - "calculate_quality": Whether to calculate quality (default: True)

        Returns:
            Pipeline function that takes input and returns final output.

        Example:
            >>> pipeline = cert_integration.create_multi_agent_pipeline([
            ...     {"agent": researcher, "agent_id": "researcher", "agent_name": "Researcher"},
            ...     {"agent": writer, "agent_id": "writer", "agent_name": "Writer"},
            ...     {"agent": editor, "agent_id": "editor", "agent_name": "Editor"},
            ... ])
            >>>
            >>> result = pipeline({"messages": [input_message]})
            >>> cert_integration.print_metrics()
        """
        # Wrap all agents
        wrapped_agents = []
        for agent_config in agents:
            wrapped = self.wrap_agent(
                agent=agent_config["agent"],
                agent_id=agent_config["agent_id"],
                agent_name=agent_config["agent_name"],
                calculate_quality=agent_config.get("calculate_quality", True),
            )
            wrapped_agents.append(wrapped)

        def pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the pipeline."""
            self.reset_metrics()

            result = input_data
            for wrapped_agent in wrapped_agents:
                result = wrapped_agent.invoke(result)

            # Calculate coordination effect
            self.calculate_coordination_effect()

            # Calculate pipeline health
            self.calculate_pipeline_health()

            return result

        return pipeline
