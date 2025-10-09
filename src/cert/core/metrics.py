"""
Core CERT metric calculations implementing the framework from:
"CERT: Instrumentation and Metrics for Production LLM Coordination"

This module implements the five core CERT metrics with exact formulas from the paper:
1. Behavioral Consistency C(Ai, p) - Equation 1
2. Empirical Performance Distribution (μi,C, σi,C) - Equation 2
3. Coordination Effect γ - Equation 3
4. Prediction Error ε - Equation 6
5. Pipeline Health Score Hcoord - Equation 7
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def behavioral_consistency(
    semantic_distances: NDArray[np.float64],
) -> float:
    """
    Calculate Behavioral Consistency C(Ai, p) from Equation 1.

    For agent Ai and prompt p, measures how consistently an agent responds
    to identical input using semantic distance variability.

    Mathematical definition:
    $$C(A_i, p) = 1 - \\frac{\\text{Std}\\{d(r_j, r_k)\\}}{\\text{Mean}\\{d(r_j, r_k)\\}}$$

    where d(rj, rk) is the semantic distance between responses j and k.

    Args:
        semantic_distances: Array of pairwise semantic distances between responses.
                          Should contain all pairs d(rj, rk) for 1 ≤ j < k ≤ n.

    Returns:
        Behavioral consistency score C ∈ [0, 1].
        Higher values (approaching 1) indicate stable, predictable behavior.
        Lower values indicate high variability requiring enhanced monitoring.

    Example:
        >>> distances = np.array([0.12, 0.15, 0.13, 0.14, 0.16])
        >>> consistency = behavioral_consistency(distances)
        >>> print(f"C = {consistency:.3f}")  # High consistency

    Note:
        Requires at least 2 distance measurements. Returns 0.0 if mean distance is 0.
    """
    if len(semantic_distances) < 2:
        raise ValueError("Need at least 2 distance measurements")

    mean_distance = np.mean(semantic_distances)

    # Avoid division by zero
    if mean_distance == 0:
        return 0.0

    std_distance = np.std(semantic_distances, ddof=1)
    coefficient_variation = std_distance / mean_distance

    consistency = 1.0 - coefficient_variation

    # Clip to [0, 1] range
    return float(np.clip(consistency, 0.0, 1.0))


def empirical_performance_distribution(
    success_scores: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Calculate Empirical Performance Distribution (μi,C, σi,C) from Equation 2.

    For agent Ai and task category C, computes the mean and standard deviation
    of performance across multiple trials.

    Mathematical definition:
    $$\\mu_{i,C} = \\frac{1}{m} \\sum_{j=1}^{m} s_j$$
    $$\\sigma^2_{i,C} = \\frac{1}{m-1} \\sum_{j=1}^{m} (s_j - \\mu_{i,C})^2$$

    where sj represents a success indicator or quality score for trial j.

    Args:
        success_scores: Array of success indicators or quality scores from m trials.
                       Typically normalized to [0, 1] range.

    Returns:
        Tuple of (mean μ, std σ) representing the empirical performance distribution.

    Example:
        >>> scores = np.array([0.85, 0.82, 0.88, 0.79, 0.86])
        >>> mu, sigma = empirical_performance_distribution(scores)
        >>> print(f"μ = {mu:.3f}, σ = {sigma:.3f}")

    Note:
        Uses sample standard deviation (ddof=1) as per Equation 2.
    """
    if len(success_scores) < 2:
        raise ValueError("Need at least 2 performance measurements")

    mu = float(np.mean(success_scores))
    sigma = float(np.std(success_scores, ddof=1))

    return mu, sigma


def coordination_effect(
    coordinated_performance: float,
    independent_performances: List[float],
) -> float:
    """
    Calculate Coordination Effect γ from Equation 3.

    For agents Ai, Aj collaborating on task t via coordination pattern P,
    measures whether coordination produces synergistic (γ > 1) or
    detrimental (γ < 1) effects.

    Mathematical definition:
    $$\\gamma^P_{i,j}(t) = \\frac{\\mathbb{E}[P^{\\text{coordinated}}_{ij}(t)]}{\\mathbb{E}[P^{\\text{independent}}_i(t)] \\cdot \\mathbb{E}[P^{\\text{independent}}_j(t)]}$$

    Args:
        coordinated_performance: Expected performance when agents coordinate.
        independent_performances: List of expected independent performances for each agent.

    Returns:
        Coordination effect γ.
        γ > 1: Synergistic coordination (agents benefit from interaction)
        γ = 1: No coordination effect (independent operation)
        γ < 1: Detrimental interaction (coordination degrades performance)

    Example:
        >>> # Two-agent coordination with synergistic effect
        >>> gamma = coordination_effect(0.75, [0.60, 0.65])
        >>> print(f"γ = {gamma:.3f}")  # γ > 1 indicates positive coordination

    Note:
        For n-agent sequential pipelines, the denominator is the product of
        all n independent agent performances.
    """
    if not independent_performances:
        raise ValueError("Need at least one independent performance value")

    if any(p <= 0 for p in independent_performances):
        raise ValueError("Independent performances must be positive")

    independent_product = float(np.prod(independent_performances))

    if independent_product == 0:
        raise ValueError("Product of independent performances cannot be zero")

    gamma = coordinated_performance / independent_product

    return float(gamma)


def performance_baseline(
    agent_means: NDArray[np.float64],
    gamma_mean: float,
    alpha: float = 0.93,
) -> float:
    """
    Calculate Performance Baseline Pbaseline(A, t) from Equation 5.

    For a sequential coordination pipeline A = {A1, ..., An} executing task t,
    predicts expected performance incorporating coordination effects and
    information degradation.

    Mathematical definition:
    $$P_{\\text{baseline}}(\\mathcal{A}, t) = \\bar{\\mu} \\cdot \\left(1 + \\frac{\\bar{\\gamma} - 1}{\\sqrt{n-1}}\\right) \\cdot \\phi(n)$$

    where:
    - $\\bar{\\mu} = \\frac{1}{n} \\sum_{i=1}^{n} \\mu_i$ is mean agent performance
    - $\\bar{\\gamma}$ is mean coordination effect
    - $\\phi(n) = \\alpha^{n-1}$ models information degradation with α ∈ [0.9, 0.95]

    Args:
        agent_means: Array of mean baseline performances μi for each agent.
        gamma_mean: Mean coordination effect γ̄ across agent pairs.
        alpha: Information degradation parameter (default 0.93 from paper).
               Controls how information quality degrades through pipeline.

    Returns:
        Predicted baseline performance for the pipeline.

    Example:
        >>> # Five-agent pipeline
        >>> means = np.array([0.60, 0.65, 0.62, 0.68, 0.64])
        >>> gamma_bar = 1.35
        >>> baseline = performance_baseline(means, gamma_bar, alpha=0.93)
        >>> print(f"Predicted baseline: {baseline:.3f}")

    Note:
        - Paper empirically determined α = 0.93 but this should be configurable
        - For two-agent pipelines (n=2), the sqrt term becomes 1.0
        - Information degradation φ(n) accounts for context loss in long chains
    """
    n = len(agent_means)

    if n < 2:
        raise ValueError("Need at least 2 agents for pipeline baseline")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    mu_bar = float(np.mean(agent_means))

    # Coordination adjustment factor
    if n == 2:
        coord_adjustment = 1.0 + (gamma_mean - 1.0)
    else:
        coord_adjustment = 1.0 + (gamma_mean - 1.0) / np.sqrt(n - 1)

    # Information degradation
    phi_n = alpha ** (n - 1)

    baseline = mu_bar * coord_adjustment * phi_n

    return float(baseline)


def prediction_error(
    observed_performance: float,
    baseline_performance: float,
) -> float:
    """
    Calculate Prediction Error εpred(t) from Equation 6.

    Measures the relative error between observed pipeline performance
    and the predicted baseline.

    Mathematical definition:
    $$\\epsilon_{\\text{pred}}(t) = \\frac{|\\mathbb{E}[S(\\mathcal{A}, t)] - P_{\\text{baseline}}(\\mathcal{A}, t)|}{P_{\\text{baseline}}(\\mathcal{A}, t)}$$

    Args:
        observed_performance: Actual measured performance E[S(A, t)].
        baseline_performance: Predicted performance Pbaseline(A, t) from Equation 5.

    Returns:
        Relative prediction error ε as a fraction.
        Lower values indicate better prediction accuracy.

    Example:
        >>> epsilon = prediction_error(observed=0.72, baseline=0.65)
        >>> print(f"ε = {epsilon:.1%}")  # 10.8% prediction error

    Note:
        - Returns absolute relative error (always positive)
        - Divide by 100 to convert to percentage
        - Values near 0 indicate accurate predictions
    """
    if baseline_performance <= 0:
        raise ValueError("Baseline performance must be positive")

    epsilon = abs(observed_performance - baseline_performance) / baseline_performance

    return float(epsilon)


def pipeline_health_score(
    epsilon: float,
    gamma_mean: float,
    observability_coverage: float,
) -> float:
    """
    Calculate Pipeline Health Score Hcoord(t) from Equation 7.

    Composite operational metric integrating prediction accuracy,
    coordination strength, and observability coverage into a single
    health indicator for runtime monitoring.

    Mathematical definition:
    $$H_{\\text{coord}}(t) = \\frac{1}{1 + \\epsilon_{\\text{pred}}(t)} \\times \\min(1, \\bar{\\gamma}(t)) \\times C_{\\text{obs}}(t)$$

    Args:
        epsilon: Prediction error εpred from Equation 6.
        gamma_mean: Mean coordination effect γ̄ (capped at 1.0 for stability).
        observability_coverage: Fraction of instrumented interactions Cobs.

    Returns:
        Pipeline health score H ∈ [0, 1].
        Higher scores indicate healthier, more predictable pipelines.

    Example:
        >>> # Healthy pipeline: low error, positive coordination, high coverage
        >>> health = pipeline_health_score(epsilon=0.05, gamma_mean=1.35, observability_coverage=0.95)
        >>> print(f"H = {health:.2f}")  # H ≈ 0.90 (healthy)

        >>> # Unhealthy pipeline: high error, poor coordination
        >>> health = pipeline_health_score(epsilon=0.50, gamma_mean=0.85, observability_coverage=0.75)
        >>> print(f"H = {health:.2f}")  # H ≈ 0.42 (requires attention)

    Operational Interpretation:
        - H > 0.8: Healthy pipeline, standard monitoring
        - 0.6 < H ≤ 0.8: Acceptable, monitor for degradation
        - 0.4 < H ≤ 0.6: Degraded, investigate issues
        - H ≤ 0.4: Critical, immediate attention required

    Note:
        - Analogous to service-level health scores in distributed systems
        - The min(1, γ̄) term prevents over-weighting strong coordination
        - All three factors must be healthy for high overall score
    """
    if not 0.0 <= observability_coverage <= 1.0:
        raise ValueError(f"Observability coverage must be in [0, 1], got {observability_coverage}")

    if epsilon < 0:
        raise ValueError(f"Prediction error must be non-negative, got {epsilon}")

    # Prediction accuracy component
    accuracy_factor = 1.0 / (1.0 + epsilon)

    # Coordination component (capped at 1.0)
    coordination_factor = min(1.0, gamma_mean)

    # Composite health score
    health = accuracy_factor * coordination_factor * observability_coverage

    return float(health)


def performance_variability(
    observed_std: float,
    baseline_std: float,
) -> float:
    """
    Calculate Performance Variability Ω.

    Quantifies behavioral range relative to baseline predictions.
    Measures how much actual execution variance exceeds/falls below
    baseline model assumptions.

    Definition:
    $$\\Omega = \\frac{\\sigma_{\\text{observed}}}{\\sigma_{\\text{baseline}}}$$

    Args:
        observed_std: Standard deviation of observed pipeline performance.
        baseline_std: Expected standard deviation from baseline model.

    Returns:
        Variability ratio Ω.
        Ω > 1: Higher variance than expected (less predictable)
        Ω ≈ 1: Variance matches baseline model
        Ω < 1: Lower variance than expected (more stable)

    Operational Interpretation:
        From Table 5 in paper:
        - Ω < 2.0: Low variability, stable execution (e.g., Gemini: 1.729)
        - 2.0 ≤ Ω < 2.5: Moderate variability (e.g., GPT-4: 2.320)
        - Ω ≥ 2.5: High variability, requires intensive monitoring (e.g., Claude: 2.852)

    Note:
        Provides distinct operational value from prediction error ε.
        GPT-4 shows this separation: near-zero ε but moderate Ω, indicating
        the baseline model captures mean behavior but underestimates variance.
    """
    if baseline_std <= 0:
        raise ValueError("Baseline std must be positive")

    omega = observed_std / baseline_std

    return float(omega)
