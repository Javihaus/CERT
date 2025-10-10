"""
Cost estimation utilities for CERT measurements.

Provides transparent cost estimation before running expensive calibration measurements.
"""

from typing import Dict, Optional
from dataclasses import dataclass


# Cost per 1M tokens (as of 2025-10-10)
# Input/Output costs in USD
MODEL_COSTS = {
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "provider": "OpenAI"
    },
    "gpt-4o-mini": {
        "input_per_1m": 0.150,
        "output_per_1m": 0.600,
        "provider": "OpenAI"
    },
    "gpt-5": {
        "input_per_1m": 5.00,  # Estimated
        "output_per_1m": 15.00,  # Estimated
        "provider": "OpenAI"
    },
    "chatgpt-5": {  # Alias
        "input_per_1m": 5.00,
        "output_per_1m": 15.00,
        "provider": "OpenAI"
    },
    "claude-3-5-haiku": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "provider": "Anthropic"
    },
    "claude-sonnet-4.5": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "provider": "Anthropic"
    },
    "claude-3-5-sonnet": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "provider": "Anthropic"
    },
    "gemini-3.5-pro": {
        "input_per_1m": 1.25,
        "output_per_1m": 5.00,
        "provider": "Google"
    },
    "grok-3": {
        "input_per_1m": 2.00,
        "output_per_1m": 10.00,
        "provider": "xAI"
    },
}


@dataclass
class CostEstimate:
    """Cost estimate for CERT measurements."""

    total_cost_usd: float
    input_cost_usd: float
    output_cost_usd: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    n_api_calls: int
    model_name: str
    measurement_type: str
    warning: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable cost estimate."""
        lines = [
            f"CERT Cost Estimate - {self.measurement_type}",
            f"{'=' * 60}",
            f"Model: {self.model_name}",
            f"API Calls: {self.n_api_calls}",
            f"",
            f"Tokens:",
            f"  Input:  {self.input_tokens:,} tokens (${self.input_cost_usd:.4f})",
            f"  Output: {self.output_tokens:,} tokens (${self.output_cost_usd:.4f})",
            f"  Total:  {self.total_tokens:,} tokens",
            f"",
            f"Estimated Cost: ${self.total_cost_usd:.4f}",
        ]

        if self.warning:
            lines.extend([
                f"",
                f"⚠️  WARNING: {self.warning}",
            ])

        return "\n".join(lines)


def estimate_calibration_cost(
    model_name: str,
    n_trials: int = 20,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 300,
) -> CostEstimate:
    """
    Estimate cost for baseline calibration measurement.

    Calibration mode runs comprehensive measurements to establish
    baseline performance characteristics.

    Args:
        model_name: Model identifier (e.g., "gpt-4o", "claude-sonnet-4.5").
        n_trials: Number of trials for statistical significance (default: 20).
        avg_input_tokens: Average input tokens per trial (default: 500).
        avg_output_tokens: Average output tokens per trial (default: 300).

    Returns:
        CostEstimate with detailed breakdown.

    Example:
        >>> estimate = estimate_calibration_cost("gpt-4o", n_trials=20)
        >>> print(estimate)
        >>> if estimate.total_cost_usd > 5.00:
        ...     print("Cost too high, reducing trials")
    """
    # Get model costs
    if model_name not in MODEL_COSTS:
        # Use average costs as fallback
        costs = {
            "input_per_1m": 2.00,
            "output_per_1m": 10.00,
            "provider": "Unknown"
        }
        warning = f"Unknown model '{model_name}'. Using average pricing."
    else:
        costs = MODEL_COSTS[model_name]
        warning = None

    # Calculate token usage
    # Calibration includes:
    # - Consistency measurement: n_trials runs
    # - Performance measurement: n_trials runs
    # - Semantic distance calculations (uses embeddings, minimal cost)
    total_api_calls = n_trials * 2

    input_tokens = total_api_calls * avg_input_tokens
    output_tokens = total_api_calls * avg_output_tokens
    total_tokens = input_tokens + output_tokens

    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * costs["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * costs["output_per_1m"]
    total_cost = input_cost + output_cost

    # Add warning for expensive measurements
    if total_cost > 10.00 and not warning:
        warning = f"High cost measurement (${total_cost:.2f}). Consider reducing n_trials."

    return CostEstimate(
        total_cost_usd=total_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        n_api_calls=total_api_calls,
        model_name=model_name,
        measurement_type="Baseline Calibration",
        warning=warning,
    )


def estimate_monitoring_cost(
    model_name: str,
    sample_rate: float = 0.1,
    requests_per_day: int = 1000,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 300,
) -> CostEstimate:
    """
    Estimate cost for continuous production monitoring.

    Monitoring mode samples a fraction of production traffic for
    ongoing health tracking with low overhead.

    Args:
        model_name: Model identifier.
        sample_rate: Fraction of requests to sample (0.0-1.0, default: 0.1).
        requests_per_day: Expected daily request volume (default: 1000).
        avg_input_tokens: Average input tokens per request (default: 500).
        avg_output_tokens: Average output tokens per request (default: 300).

    Returns:
        CostEstimate with daily monitoring costs.

    Example:
        >>> # Estimate monitoring cost
        >>> daily_cost = estimate_monitoring_cost(
        ...     "gpt-4o",
        ...     sample_rate=0.1,
        ...     requests_per_day=10000
        ... )
        >>> print(f"Monthly cost: ${daily_cost.total_cost_usd * 30:.2f}")
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")

    # Get model costs
    if model_name not in MODEL_COSTS:
        costs = {
            "input_per_1m": 2.00,
            "output_per_1m": 10.00,
            "provider": "Unknown"
        }
        warning = f"Unknown model '{model_name}'. Using average pricing."
    else:
        costs = MODEL_COSTS[model_name]
        warning = None

    # Calculate sampled usage
    sampled_requests = int(requests_per_day * sample_rate)

    # Monitoring overhead is minimal:
    # - Semantic distance calculation (embeddings): ~negligible
    # - Quality scoring: no additional LLM calls
    # Main cost is the normal production traffic that gets measured
    input_tokens = sampled_requests * avg_input_tokens
    output_tokens = sampled_requests * avg_output_tokens
    total_tokens = input_tokens + output_tokens

    # Calculate daily costs
    input_cost = (input_tokens / 1_000_000) * costs["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * costs["output_per_1m"]
    total_cost = input_cost + output_cost

    # Add context about monitoring overhead
    overhead_pct = (sample_rate * 100)
    measurement_type = f"Production Monitoring (Daily, {overhead_pct:.1f}% sample rate)"

    return CostEstimate(
        total_cost_usd=total_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        n_api_calls=sampled_requests,
        model_name=model_name,
        measurement_type=measurement_type,
        warning=warning,
    )


def estimate_pipeline_cost(
    model_names: list[str],
    n_agents: int,
    n_trials: int = 20,
    avg_input_tokens_per_agent: int = 500,
    avg_output_tokens_per_agent: int = 300,
) -> CostEstimate:
    """
    Estimate cost for multi-agent pipeline calibration.

    Args:
        model_names: List of model names in the pipeline.
        n_agents: Number of agents in the pipeline.
        n_trials: Number of trial runs for calibration.
        avg_input_tokens_per_agent: Average input tokens per agent.
        avg_output_tokens_per_agent: Average output tokens per agent.

    Returns:
        CostEstimate for full pipeline calibration.

    Example:
        >>> # 3-agent pipeline with GPT-4o
        >>> cost = estimate_pipeline_cost(
        ...     model_names=["gpt-4o"] * 3,
        ...     n_agents=3,
        ...     n_trials=20
        ... )
        >>> print(cost)
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    warnings = []

    # Calculate cost for each agent
    for i, model_name in enumerate(model_names):
        if model_name not in MODEL_COSTS:
            costs = {
                "input_per_1m": 2.00,
                "output_per_1m": 10.00,
            }
            warnings.append(f"Agent {i+1}: Unknown model '{model_name}'")
        else:
            costs = MODEL_COSTS[model_name]

        # Each agent runs in each trial
        input_tokens = n_trials * avg_input_tokens_per_agent
        output_tokens = n_trials * avg_output_tokens_per_agent

        agent_cost = (
            (input_tokens / 1_000_000) * costs["input_per_1m"] +
            (output_tokens / 1_000_000) * costs["output_per_1m"]
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += agent_cost

    total_tokens = total_input_tokens + total_output_tokens
    total_api_calls = n_agents * n_trials

    # Calculate component costs for display
    input_cost = total_cost * (total_input_tokens / total_tokens) if total_tokens > 0 else 0
    output_cost = total_cost * (total_output_tokens / total_tokens) if total_tokens > 0 else 0

    warning = None
    if warnings:
        warning = "; ".join(warnings)
    elif total_cost > 20.00:
        warning = f"High cost pipeline (${total_cost:.2f}). Consider reducing n_trials."

    return CostEstimate(
        total_cost_usd=total_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_tokens=total_tokens,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        n_api_calls=total_api_calls,
        model_name=f"{n_agents}-agent pipeline",
        measurement_type=f"Pipeline Calibration ({n_agents} agents)",
        warning=warning,
    )


def list_model_costs() -> Dict[str, Dict[str, float]]:
    """
    Get current pricing for all supported models.

    Returns:
        Dictionary of model costs.

    Example:
        >>> costs = list_model_costs()
        >>> for model, pricing in costs.items():
        ...     print(f"{model}: ${pricing['input_per_1m']:.2f}/M in")
    """
    return MODEL_COSTS.copy()


def compare_model_costs(
    model_names: list[str],
    n_trials: int = 20,
    avg_tokens: int = 500,
) -> str:
    """
    Compare calibration costs across multiple models.

    Args:
        model_names: List of model names to compare.
        n_trials: Number of trials for calibration.
        avg_tokens: Average tokens per trial.

    Returns:
        Formatted comparison table.

    Example:
        >>> comparison = compare_model_costs(
        ...     ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4.5"],
        ...     n_trials=20
        ... )
        >>> print(comparison)
    """
    estimates = []
    for model_name in model_names:
        est = estimate_calibration_cost(model_name, n_trials, avg_tokens, avg_tokens)
        estimates.append((model_name, est))

    # Sort by cost
    estimates.sort(key=lambda x: x[1].total_cost_usd)

    # Build table
    lines = [
        "Model Cost Comparison",
        "=" * 70,
        f"{'Model':<25} {'Cost':<12} {'Tokens':<15} {'Provider':<15}",
        "-" * 70,
    ]

    for model_name, est in estimates:
        provider = MODEL_COSTS.get(model_name, {}).get("provider", "Unknown")
        lines.append(
            f"{model_name:<25} ${est.total_cost_usd:<11.4f} {est.total_tokens:>14,} {provider:<15}"
        )

    lines.append("-" * 70)
    lines.append(f"All costs for n_trials={n_trials}, avg_tokens={avg_tokens}")

    return "\n".join(lines)
