"""
CrewAI integration for CERT SDK.

Provides instrumentation for CrewAI agents and crews with automatic
CERT metrics tracking.
"""

import time
from typing import Any, Dict, List, Optional

from cert.analysis.quality import QualityScorer
from cert.integrations.base import CERTIntegration
from cert.models import ModelBaseline
from cert.providers.base import ProviderInterface


class CERTCrewAI(CERTIntegration):
    """
    CERT integration for CrewAI.

    Instruments CrewAI agents and crews with automatic CERT metrics
    collection including:
    - Agent execution tracking
    - Quality scoring
    - Coordination effect measurement
    - Pipeline health monitoring

    Example:
        >>> import cert
        >>> from cert.integrations.crewai import CERTCrewAI
        >>> from crewai import Agent, Task, Crew
        >>>
        >>> # Create your CrewAI agents
        >>> researcher = Agent(
        ...     role='Researcher',
        ...     goal='Gather information',
        ...     backstory='Expert researcher'
        ... )
        >>>
        >>> writer = Agent(
        ...     role='Writer',
        ...     goal='Create content',
        ...     backstory='Professional writer'
        ... )
        >>>
        >>> # Define tasks
        >>> research_task = Task(
        ...     description='Research AI trends',
        ...     agent=researcher,
        ...     expected_output='Research summary'
        ... )
        >>>
        >>> write_task = Task(
        ...     description='Write article based on research',
        ...     agent=writer,
        ...     expected_output='Article'
        ... )
        >>>
        >>> # Create crew
        >>> crew = Crew(
        ...     agents=[researcher, writer],
        ...     tasks=[research_task, write_task]
        ... )
        >>>
        >>> # Wrap with CERT instrumentation
        >>> cert_integration = CERTCrewAI(
        ...     provider=cert.create_provider(api_key="...", model_name="gpt-4o"),
        ... )
        >>> instrumented_crew = cert_integration.wrap_crew(crew)
        >>>
        >>> # Run with automatic metrics collection
        >>> result = instrumented_crew.kickoff()
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
        Initialize CrewAI integration.

        Args:
            provider: CERT provider for baseline comparison.
            baseline: Model baseline for comparison.
            scorer: Quality scorer for output evaluation.
            track_all_executions: Track all agent executions (default: True).
            verbose: Print execution details (default: False).
        """
        super().__init__(provider, baseline, scorer, track_all_executions)
        self.verbose = verbose
        self._original_crew = None
        self._task_outputs: List[str] = []

    def wrap_agent(self, agent: Any, agent_id: str, agent_name: str) -> Any:
        """
        Wrap a CrewAI agent with CERT instrumentation.

        Note: In CrewAI, agents are typically instrumented at the crew level
        rather than individually. Use wrap_crew() instead for full instrumentation.

        Args:
            agent: CrewAI Agent instance.
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable name for the agent.

        Returns:
            The agent (instrumentation happens at crew level).
        """
        # CrewAI agents are instrumented at the crew level
        # Store agent info for tracking
        if not hasattr(self, "_agent_info"):
            self._agent_info = {}

        self._agent_info[agent_id] = {
            "agent": agent,
            "name": agent_name or agent.role,
        }

        return agent

    def wrap_crew(self, crew: Any) -> Any:
        """
        Wrap a CrewAI Crew with CERT instrumentation.

        This wraps the crew's kickoff() method to automatically track
        task executions and calculate metrics.

        Args:
            crew: CrewAI Crew instance.

        Returns:
            Instrumented crew with same interface.

        Example:
            >>> instrumented_crew = cert_integration.wrap_crew(my_crew)
            >>> result = instrumented_crew.kickoff()
            >>> cert_integration.print_metrics()
        """
        self._original_crew = crew

        # Create wrapper
        class InstrumentedCrew:
            def __init__(wrapper_self, base_crew, integration):
                wrapper_self._crew = base_crew
                wrapper_self._integration = integration

            def kickoff(wrapper_self, inputs: Optional[Dict[str, Any]] = None):
                """Instrumented kickoff method."""
                return wrapper_self._integration._instrument_kickoff(
                    crew=wrapper_self._crew,
                    inputs=inputs,
                )

            def __getattr__(wrapper_self, name):
                """Forward other attributes to original crew."""
                return getattr(wrapper_self._crew, name)

        return InstrumentedCrew(crew, self)

    def _instrument_kickoff(
        self,
        crew: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Instrument crew execution."""
        self.reset_metrics()
        self._task_outputs = []

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("CERT: Starting CrewAI Crew Execution")
            print(f"{'=' * 70}")
            print(f"Agents: {len(crew.agents)}")
            print(f"Tasks: {len(crew.tasks)}")

        # Extract initial input
        initial_input = ""
        if inputs:
            initial_input = str(inputs)
        elif crew.tasks:
            initial_input = crew.tasks[0].description

        # Execute with timing
        start_time = time.time()

        # Hook into task callbacks to track individual task executions
        original_tasks = []
        for i, task in enumerate(crew.tasks):
            original_tasks.append(task)
            task = self._wrap_task(task, i)

        # Run the crew
        result = crew.kickoff(inputs=inputs)

        duration_ms = (time.time() - start_time) * 1000

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"CERT: Crew Execution Complete ({duration_ms:.0f}ms)")
            print(f"{'=' * 70}")

        # Extract final output
        final_output = self._extract_result(result)

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

    def _wrap_task(self, task: Any, task_index: int) -> Any:
        """Wrap a task to track execution."""
        # Store original callback
        original_callback = getattr(task, "callback", None)

        def cert_callback(task_output):
            """CERT instrumented callback."""
            # Extract task information
            agent_name = (
                task.agent.role if hasattr(task, "agent") and task.agent else f"Agent_{task_index}"
            )
            agent_id = f"agent_{task_index}"

            # Extract output
            output_text = self._extract_task_output(task_output)
            self._task_outputs.append(output_text)

            # Record execution
            self.record_execution(
                agent_id=agent_id,
                agent_name=agent_name,
                input_text=task.description,
                output_text=output_text,
                duration_ms=0,  # CrewAI doesn't expose individual task timing
                metadata={
                    "task_index": task_index,
                    "expected_output": task.expected_output,
                },
            )

            # Calculate quality
            if task.description and output_text:
                quality = self.calculate_quality(task.description, output_text)
                self.metrics.intermediate_qualities.append(quality)

                if self.verbose:
                    print(f"  Task {task_index + 1} ({agent_name}): Quality = {quality:.3f}")

            # Call original callback if exists
            if original_callback:
                return original_callback(task_output)

            return task_output

        # Set our callback
        task.callback = cert_callback

        return task

    def wrap_pipeline(self, pipeline: Any) -> Any:
        """
        Wrap a CrewAI pipeline (Crew) with CERT instrumentation.

        This is an alias for wrap_crew() for consistency with the base interface.

        Args:
            pipeline: CrewAI Crew instance.

        Returns:
            Instrumented crew.
        """
        return self.wrap_crew(pipeline)

    def _extract_result(self, result: Any) -> str:
        """Extract output text from crew result."""
        # CrewAI result can be various types
        if isinstance(result, str):
            return result
        if hasattr(result, "raw"):
            return str(result.raw)
        if hasattr(result, "output"):
            return str(result.output)
        if isinstance(result, dict):
            # Try common keys
            for key in ["output", "result", "final_output", "text"]:
                if key in result:
                    return str(result[key])

        # Fallback
        return str(result)

    def _extract_task_output(self, task_output: Any) -> str:
        """Extract output text from task output."""
        if isinstance(task_output, str):
            return task_output
        if hasattr(task_output, "raw"):
            return str(task_output.raw)
        if hasattr(task_output, "output"):
            return str(task_output.output)
        if isinstance(task_output, dict):
            for key in ["output", "result", "text"]:
                if key in task_output:
                    return str(task_output[key])

        return str(task_output)

    def create_instrumented_crew(
        self, agents: List[Any], tasks: List[Any], verbose: bool = False, **crew_kwargs
    ) -> Any:
        """
        Create a CrewAI Crew with automatic CERT instrumentation.

        This is a convenience method for creating and instrumenting a crew
        in one step.

        Args:
            agents: List of CrewAI Agent instances.
            tasks: List of CrewAI Task instances.
            verbose: Enable verbose output from CrewAI (default: False).
            **crew_kwargs: Additional arguments passed to Crew constructor.

        Returns:
            Instrumented Crew instance.

        Example:
            >>> instrumented_crew = cert_integration.create_instrumented_crew(
            ...     agents=[researcher, writer, editor],
            ...     tasks=[research_task, write_task, edit_task],
            ... )
            >>> result = instrumented_crew.kickoff()
        """
        try:
            from crewai import Crew
        except ImportError:
            raise ImportError("CrewAI is not installed. Install it with: pip install crewai")

        # Create crew
        crew = Crew(agents=agents, tasks=tasks, verbose=verbose, **crew_kwargs)

        # Wrap with instrumentation
        return self.wrap_crew(crew)
