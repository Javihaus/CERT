"""
Base integration interface for multi-agent frameworks.

This module defines the core abstractions for integrating CERT metrics
into existing multi-agent frameworks like LangChain, CrewAI, and Microsoft Agent Framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cert.analysis.quality import QualityScorer
from cert.core.metrics import (
    coordination_effect,
    normalized_context_effect,
    pipeline_health_score,
)
from cert.models import ModelBaseline
from cert.providers.base import ProviderInterface


@dataclass
class AgentExecution:
    """Record of a single agent execution."""

    agent_id: str
    agent_name: str
    input_text: str
    output_text: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    # Execution tracking
    executions: List[AgentExecution] = field(default_factory=list)
    total_duration_ms: float = 0.0

    # Quality metrics
    input_quality: Optional[float] = None
    output_quality: Optional[float] = None
    intermediate_qualities: List[float] = field(default_factory=list)

    # Coordination metrics
    coordination_effect: Optional[float] = None  # Raw γ
    coordination_effect_norm: Optional[float] = None  # Normalized γ_norm
    n_agents: int = 0  # Number of agents in pipeline
    independent_performances: List[float] = field(default_factory=list)
    coordinated_performance: Optional[float] = None

    # Health metrics
    health_score: Optional[float] = None
    prediction_error: Optional[float] = None
    observability_coverage: float = 1.0  # Fraction of pipeline instrumented

    # Baseline comparison
    baseline_consistency: Optional[float] = None
    baseline_performance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "num_executions": len(self.executions),
            "total_duration_ms": self.total_duration_ms,
            "output_quality": self.output_quality,
            "coordination_effect": self.coordination_effect,
            "coordination_effect_norm": self.coordination_effect_norm,
            "n_agents": self.n_agents,
            "health_score": self.health_score,
            "observability_coverage": self.observability_coverage,
        }


class CERTIntegration(ABC):
    """
    Base class for CERT framework integrations.

    Provides common functionality for instrumenting multi-agent frameworks
    with CERT observability and metrics.
    """

    def __init__(
        self,
        provider: Optional[ProviderInterface] = None,
        baseline: Optional[ModelBaseline] = None,
        scorer: Optional[QualityScorer] = None,
        track_all_executions: bool = True,
    ):
        """
        Initialize CERT integration.

        Args:
            provider: CERT provider for baseline comparison.
            baseline: Model baseline for comparison.
            scorer: Quality scorer for output evaluation.
            track_all_executions: Track all agent executions (default: True).
        """
        self.provider = provider
        self.baseline = baseline
        self.scorer = scorer or QualityScorer()
        self.track_all_executions = track_all_executions

        # Metrics storage
        self.metrics = PipelineMetrics()
        self._execution_stack: List[AgentExecution] = []

    @abstractmethod
    def wrap_agent(self, agent: Any, agent_id: str, agent_name: str) -> Any:
        """
        Wrap a framework-specific agent with CERT instrumentation.

        Args:
            agent: Framework-specific agent instance.
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable name for the agent.

        Returns:
            Instrumented agent.
        """
        pass

    @abstractmethod
    def wrap_pipeline(self, pipeline: Any) -> Any:
        """
        Wrap a framework-specific pipeline with CERT instrumentation.

        Args:
            pipeline: Framework-specific pipeline instance.

        Returns:
            Instrumented pipeline.
        """
        pass

    def record_execution(
        self,
        agent_id: str,
        agent_name: str,
        input_text: str,
        output_text: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an agent execution.

        Args:
            agent_id: Unique agent identifier.
            agent_name: Agent name.
            input_text: Input to the agent.
            output_text: Output from the agent.
            duration_ms: Execution duration in milliseconds.
            metadata: Optional metadata.
        """
        execution = AgentExecution(
            agent_id=agent_id,
            agent_name=agent_name,
            input_text=input_text,
            output_text=output_text,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        if self.track_all_executions:
            self.metrics.executions.append(execution)
            self.metrics.total_duration_ms += duration_ms

    def calculate_quality(self, prompt: str, response: str) -> float:
        """
        Calculate quality score for a response.

        Args:
            prompt: Input prompt.
            response: Generated response.

        Returns:
            Quality score (0-1).
        """
        components = self.scorer.score(prompt, response)
        return components.composite_score

    def calculate_coordination_effect(self) -> Optional[float]:
        """
        Calculate coordination effect (γ) and normalized effect (γ_norm) from recorded executions.

        Returns:
            Raw coordination effect γ if calculable, None otherwise.
            Also stores γ_norm in metrics.coordination_effect_norm.
        """
        if len(self.metrics.executions) < 2:
            return None

        # Get final output quality (coordinated)
        if not self.metrics.output_quality:
            return None

        # Get individual agent qualities
        if len(self.metrics.intermediate_qualities) < 2:
            return None

        # Calculate raw γ
        gamma = coordination_effect(
            coordinated_performance=self.metrics.output_quality,
            independent_performances=self.metrics.intermediate_qualities,
        )

        # Calculate normalized γ_norm
        n_agents = len(self.metrics.executions)
        gamma_norm = normalized_context_effect(gamma, n_agents)

        # Store both
        self.metrics.coordination_effect = gamma
        self.metrics.coordination_effect_norm = gamma_norm
        self.metrics.n_agents = n_agents

        return gamma

    def calculate_pipeline_health(
        self,
        prediction_error: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate pipeline health score using normalized γ_norm.

        Args:
            prediction_error: Optional prediction error (ε).

        Returns:
            Health score (0-1) if calculable, None otherwise.

        Note:
            Uses normalized γ_norm (not raw γ) for health calculation
            per Equation 8 of the paper.
        """
        # Need coordination effect (calculates both γ and γ_norm)
        if self.metrics.coordination_effect is None:
            gamma = self.calculate_coordination_effect()
            if gamma is None:
                return None

        # Get normalized gamma
        gamma_norm = self.metrics.coordination_effect_norm
        if gamma_norm is None:
            return None

        # Use provided or stored prediction error
        epsilon = prediction_error or self.metrics.prediction_error
        if epsilon is None:
            # Default: use baseline comparison if available
            if self.baseline and self.metrics.output_quality:
                epsilon = abs(self.metrics.output_quality - self.baseline.mean_performance)
            else:
                epsilon = 0.1  # Conservative default

        # Calculate health using γ_norm (not raw γ)
        health = pipeline_health_score(
            epsilon=epsilon,
            gamma_norm=gamma_norm,
            observability_coverage=self.metrics.observability_coverage,
        )

        self.metrics.health_score = health
        self.metrics.prediction_error = epsilon
        return health

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected metrics.

        Returns:
            Dictionary with metric summary.
        """
        summary = self.metrics.to_dict()

        # Add execution details
        if self.metrics.executions:
            summary["agents"] = [
                {
                    "name": ex.agent_name,
                    "duration_ms": ex.duration_ms,
                }
                for ex in self.metrics.executions
            ]

        # Add baseline comparison
        if self.baseline:
            summary["baseline"] = {
                "consistency": self.baseline.consistency,
                "mean_performance": self.baseline.mean_performance,
                "std_performance": self.baseline.std_performance,
            }

        return summary

    def reset_metrics(self) -> None:
        """Reset all metrics for a new pipeline run."""
        self.metrics = PipelineMetrics()
        self._execution_stack = []

    def print_metrics(self) -> None:
        """Print formatted metrics summary."""
        print("\n" + "=" * 70)
        print("CERT Pipeline Metrics")
        print("=" * 70)

        print("\nExecution Summary:")
        print(f"  Total agents:     {len(self.metrics.executions)}")
        print(f"  Total duration:   {self.metrics.total_duration_ms:.0f}ms")

        if self.metrics.executions:
            print("\n  Agent Executions:")
            for i, ex in enumerate(self.metrics.executions, 1):
                print(f"    {i}. {ex.agent_name}: {ex.duration_ms:.0f}ms")

        if self.metrics.output_quality is not None:
            print("\nQuality Metrics:")
            print(f"  Output quality:   {self.metrics.output_quality:.3f}")

        if self.metrics.coordination_effect is not None:
            print("\nContext Propagation Metrics:")
            print(f"  γ (raw):          {self.metrics.coordination_effect:.3f}")
            if self.metrics.coordination_effect_norm is not None:
                print(f"  γ_norm:           {self.metrics.coordination_effect_norm:.3f}")
            if self.metrics.n_agents > 0:
                print(f"  n_agents:         {self.metrics.n_agents}")
            if self.metrics.coordination_effect > 1.0:
                improvement = (self.metrics.coordination_effect - 1.0) * 100
                print(f"  Improvement:      +{improvement:.1f}%")

        if self.metrics.health_score is not None:
            print("\nPipeline Health:")
            print(f"  Health score:     {self.metrics.health_score:.3f}")
            if self.metrics.health_score > 0.8:
                status = "✓ PRODUCTION READY"
            elif self.metrics.health_score > 0.6:
                status = "⚠ NEEDS MONITORING"
            else:
                status = "⚠ NEEDS INVESTIGATION"
            print(f"  Status:           {status}")

        if self.baseline:
            print("\nBaseline Comparison:")
            print(f"  Model:            {self.baseline.model_id}")
            print(f"  Baseline C:       {self.baseline.consistency:.3f}")
            print(f"  Baseline μ:       {self.baseline.mean_performance:.3f}")

        print("=" * 70 + "\n")
