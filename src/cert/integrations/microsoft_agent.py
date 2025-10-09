"""
Microsoft Agent Framework integration for CERT SDK.

Provides instrumentation for Microsoft Agent Framework agents and workflows
with automatic CERT metrics tracking.
"""

from typing import Any, Dict, List, Optional
import time

from cert.integrations.base import CERTIntegration, PipelineMetrics
from cert.providers.base import ProviderInterface
from cert.models import ModelBaseline
from cert.analysis.quality import QualityScorer


class CERTMicrosoftAgent(CERTIntegration):
    """
    CERT integration for Microsoft Agent Framework.

    Instruments Microsoft Agent Framework agents and workflows with automatic
    CERT metrics collection including:
    - Agent execution tracking
    - Quality scoring
    - Coordination effect measurement
    - Pipeline health monitoring

    Example:
        >>> import cert
        >>> from cert.integrations.microsoft_agent import CERTMicrosoftAgent
        >>> from agent_framework import Agent, Workflow
        >>>
        >>> # Create your Microsoft agents
        >>> research_agent = Agent(
        ...     model_provider="openai",
        ...     tools=[...],
        ... )
        >>>
        >>> write_agent = Agent(
        ...     model_provider="openai",
        ...     tools=[...],
        ... )
        >>>
        >>> # Create workflow
        >>> workflow = Workflow()
        >>> workflow.add_agent(research_agent)
        >>> workflow.add_agent(write_agent)
        >>>
        >>> # Wrap with CERT instrumentation
        >>> cert_integration = CERTMicrosoftAgent(
        ...     provider=cert.create_provider(api_key="...", model_name="gpt-4o"),
        ... )
        >>> instrumented_workflow = cert_integration.wrap_workflow(workflow)
        >>>
        >>> # Run with automatic metrics collection
        >>> result = instrumented_workflow.execute(task="Your task")
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
        Initialize Microsoft Agent Framework integration.

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
        Wrap a Microsoft Agent Framework agent with CERT instrumentation.

        This wraps the agent's execute/run methods to automatically
        track executions and calculate metrics.

        Args:
            agent: Microsoft Agent instance.
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable name for the agent.
            calculate_quality: Calculate quality scores for outputs (default: True).

        Returns:
            Instrumented agent with same interface.

        Example:
            >>> instrumented = cert_integration.wrap_agent(
            ...     agent=my_agent,
            ...     agent_id="researcher",
            ...     agent_name="Research Agent",
            ... )
            >>> result = instrumented.execute(task="Research AI trends")
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

            def execute(wrapper_self, *args, **kwargs):
                """Instrumented execute method."""
                return wrapper_self._integration._instrument_execution(
                    agent=wrapper_self._agent,
                    agent_id=wrapper_self._agent_id,
                    agent_name=wrapper_self._agent_name,
                    method="execute",
                    calculate_quality=wrapper_self._calculate_quality,
                    args=args,
                    kwargs=kwargs,
                )

            def run(wrapper_self, *args, **kwargs):
                """Instrumented run method."""
                return wrapper_self._integration._instrument_execution(
                    agent=wrapper_self._agent,
                    agent_id=wrapper_self._agent_id,
                    agent_name=wrapper_self._agent_name,
                    method="run",
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

    def wrap_workflow(self, workflow: Any) -> Any:
        """
        Wrap a Microsoft Agent Framework Workflow with CERT instrumentation.

        This wraps the workflow's execute method to automatically track
        agent executions and calculate metrics.

        Args:
            workflow: Microsoft Agent Framework Workflow instance.

        Returns:
            Instrumented workflow with same interface.

        Example:
            >>> instrumented_workflow = cert_integration.wrap_workflow(my_workflow)
            >>> result = instrumented_workflow.execute(task="Your task")
            >>> cert_integration.print_metrics()
        """
        # Create wrapper
        class InstrumentedWorkflow:
            def __init__(wrapper_self, base_workflow, integration):
                wrapper_self._workflow = base_workflow
                wrapper_self._integration = integration

            def execute(wrapper_self, *args, **kwargs):
                """Instrumented execute method."""
                return wrapper_self._integration._instrument_workflow_execution(
                    workflow=wrapper_self._workflow,
                    args=args,
                    kwargs=kwargs,
                )

            def __getattr__(wrapper_self, name):
                """Forward other attributes to original workflow."""
                return getattr(wrapper_self._workflow, name)

        return InstrumentedWorkflow(workflow, self)

    def _instrument_workflow_execution(
        self,
        workflow: Any,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Instrument workflow execution."""
        self.reset_metrics()

        # Extract initial input
        initial_input = self._extract_input(args, kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"CERT: Starting Microsoft Agent Workflow")
            print(f"{'='*70}")

        # Execute with timing
        start_time = time.time()
        result = workflow.execute(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"CERT: Workflow Complete ({duration_ms:.0f}ms)")
            print(f"{'='*70}")

        # Extract final output
        final_output = self._extract_output(result)

        # Calculate overall quality
        if initial_input and final_output:
            output_quality = self.calculate_quality(initial_input, final_output)
            self.metrics.output_quality = output_quality

            if self.verbose:
                print(f"Final Output Quality: {output_quality:.3f}")

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

    def wrap_pipeline(self, pipeline: Any) -> Any:
        """
        Wrap a Microsoft Agent Framework pipeline (Workflow) with CERT instrumentation.

        This is an alias for wrap_workflow() for consistency with the base interface.

        Args:
            pipeline: Microsoft Agent Framework Workflow instance.

        Returns:
            Instrumented workflow.
        """
        return self.wrap_workflow(pipeline)

    def _extract_input(self, args: tuple, kwargs: dict) -> str:
        """Extract input text from arguments."""
        # Common patterns for Microsoft Agent Framework
        if kwargs.get('task'):
            return str(kwargs['task'])
        elif kwargs.get('input'):
            return str(kwargs['input'])
        elif kwargs.get('message'):
            return str(kwargs['message'])
        elif args:
            return str(args[0])

        return ""

    def _extract_output(self, result: Any) -> str:
        """Extract output text from result."""
        # Microsoft Agent Framework result patterns
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try common keys
            for key in ['output', 'result', 'response', 'text', 'message']:
                if key in result:
                    return str(result[key])
        elif hasattr(result, 'output'):
            return str(result.output)
        elif hasattr(result, 'result'):
            return str(result.result)
        elif hasattr(result, 'response'):
            return str(result.response)

        # Fallback
        return str(result)

    def create_multi_agent_workflow(
        self,
        agents: List[Dict[str, Any]],
    ) -> Any:
        """
        Create a simple multi-agent workflow with automatic coordination tracking.

        This is a helper for creating simple sequential workflows where
        each agent processes the output of the previous agent.

        Args:
            agents: List of agent configs, each with:
                - "agent": The Microsoft Agent instance
                - "agent_id": Unique identifier
                - "agent_name": Display name
                - "calculate_quality": Whether to calculate quality (default: True)

        Returns:
            Workflow function that takes input and returns final output.

        Example:
            >>> workflow = cert_integration.create_multi_agent_workflow([
            ...     {"agent": researcher, "agent_id": "researcher", "agent_name": "Researcher"},
            ...     {"agent": writer, "agent_id": "writer", "agent_name": "Writer"},
            ...     {"agent": editor, "agent_id": "editor", "agent_name": "Editor"},
            ... ])
            >>>
            >>> result = workflow(task="Your task")
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

        def workflow(**kwargs) -> Any:
            """Execute the workflow."""
            self.reset_metrics()

            result = None
            task = kwargs.get('task', kwargs.get('input', ''))

            for wrapped_agent in wrapped_agents:
                # Pass previous output as input to next agent
                if result:
                    task = result

                result = wrapped_agent.execute(task=task)

            # Calculate coordination effect
            self.calculate_coordination_effect()

            # Calculate pipeline health
            self.calculate_pipeline_health()

            return result

        return workflow
