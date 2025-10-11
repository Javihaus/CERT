"""
Core CERT metric calculations implementing the framework from:
"CERT: Instrumentation and Metrics for Production LLM Sequential Processing"

This module implements the five core CERT metrics with exact formulas from the paper:
1. Behavioral Consistency C(Ai, p) - Equation 1
2. Empirical Performance Distribution (μi,C, σi,C) - Equation 2
3. Context Propagation Effect γ - Equation 3
4. Prediction Error ε - Equation 6
5. Pipeline Health Score Hpipe - Equation 7
"""

from typing import List, Tuple

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
    Calculate Context Propagation Effect γ from Equation 3.

    Measures performance changes when models process accumulated context in
    sequential pipelines compared to independent execution. This quantifies
    how attention mechanisms behave when later models see outputs from
    earlier models.

    **What this measures:**
    - Statistical characterization of sequential processing behavior
    - Performance change from context accumulation (attention mechanism effects)
    - Operational metric for architecture selection

    **What this does NOT measure:**
    - ❌ Agent coordination, collaboration, or planning
    - ❌ Intelligence or reasoning capabilities
    - ❌ WHY context helps (black box measurement)

    Mathematical definition:
    $$\\gamma^P_{i,j}(t) = \\frac{\\mathbb{E}[P^{\\text{sequential}}_{ij}(t)]}{\\mathbb{E}[P^{\\text{baseline}}(t)]}$$

    Args:
        coordinated_performance: Performance when models process sequentially
                                (later models see accumulated context).
        independent_performances: List of independent performance values for each model.
                                 The baseline is computed as the mean of these values.

    Returns:
        Context propagation effect γ.
        γ > 1: Sequential context accumulation improves performance
        γ = 1: No benefit from accumulated context
        γ < 1: Context accumulation degrades performance (attention dilution,
               context window issues)

    Example:
        >>> # Two-model sequential pipeline with context benefit
        >>> gamma = coordination_effect(0.75, [0.60, 0.65])
        >>> print(f"γ = {gamma:.3f}")  # γ > 1 indicates context helps

        >>> # Interpretation
        >>> if gamma > 1.2:
        ...     print("Strong context propagation - sequential architecture recommended")
        ... elif gamma > 1.0:
        ...     print("Moderate benefit from sequential processing")
        ... else:
        ...     print("Sequential processing degrades performance")

    Operational Interpretation (from paper validation):
        - Gemini: γ=1.137 (high baseline, weak context effect)
        - GPT-4:  γ=1.562 (moderate baseline, strong context effect)
        - Grok:   γ=1.625 (moderate baseline, strong context effect)
        - Claude: γ=1.462 (low baseline, strong context effect)

    Note:
        - Baseline is computed as the mean of independent performances
        - "coordination_effect" function name retained for API compatibility
        - This is engineering characterization, not coordination science
    """
    if not independent_performances:
        raise ValueError("Need at least one independent performance value")

    if any(p <= 0 for p in independent_performances):
        raise ValueError("Independent performances must be positive")

    baseline_expected = float(np.mean(independent_performances))

    if baseline_expected == 0:
        raise ValueError("Baseline expected performance cannot be zero")

    gamma = coordinated_performance / baseline_expected

    return float(gamma)


def normalized_context_effect(gamma: float, n: int) -> float:
    """
    Calculate Normalized Context Propagation Effect γ_norm from Equation 4.

    For pipeline of length n, the normalized context propagation effect enables
    comparison across different pipeline lengths by computing the geometric mean.

    Mathematical definition:
    $$\\gamma_{\\text{norm}} = \\gamma^{1/n}$$

    This geometric mean normalization preserves the probabilistic interpretation
    while making measurements comparable across pipelines of varying length.

    Args:
        gamma: Raw context propagation effect γ from Equation 3.
        n: Number of agents in the pipeline.

    Returns:
        Normalized context propagation effect γ_norm.
        Interpretation remains the same as γ:
        - γ_norm > 1: Positive per-step context benefit
        - γ_norm = 1: No context effect
        - γ_norm < 1: Context degradation

    Example:
        >>> # Two-model pipeline
        >>> gamma_2 = 1.462
        >>> gamma_norm_2 = normalized_context_effect(gamma_2, n=2)
        >>> print(f"γ_norm = {gamma_norm_2:.3f}")  # √1.462 ≈ 1.209

        >>> # Five-model pipeline
        >>> gamma_5 = 13.46
        >>> gamma_norm_5 = normalized_context_effect(gamma_5, n=5)
        >>> print(f"γ_norm = {gamma_norm_5:.3f}")  # 13.46^(1/5) ≈ 1.685

        >>> # Comparison: raw γ shows exponential scaling, γ_norm stays stable
        >>> print(f"Raw γ increased {gamma_5/gamma_2:.1f}x")  # 9.2x
        >>> print(f"Normalized increased {gamma_norm_5/gamma_norm_2:.1f}x")  # 1.4x

    Operational Thresholds (Table 9 from paper):
        - γ_norm > 1.5:   Strong per-step context benefit → Sequential recommended
        - γ_norm 1.2-1.5: Moderate context propagation → Deploy with monitoring
        - γ_norm 1.0-1.2: Weak context effect → Consider evaluation
        - γ_norm < 1.0:   Context degradation → Investigate structure

    Note:
        - For n=2: γ_norm = √γ
        - For n=5: γ_norm = γ^(1/5)
        - Normalization eliminates exponential baseline scaling
        - CV reduces from 39.2% (raw γ) to 4.8% (γ_norm) in 5-agent validation
        - Use γ_norm for pipeline health score (Equation 8)
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")

    if n < 2:
        raise ValueError(f"Pipeline length must be at least 2, got {n}")

    gamma_norm = gamma ** (1.0 / n)

    return float(gamma_norm)


def performance_baseline(
    agent_means: NDArray[np.float64],
    alpha: float = 0.93,
) -> float:
    """
    Calculate Performance Baseline Pbaseline(A, t) from Equation 6.

    For a sequential processing pipeline A = {A1, ..., An} executing task t,
    the expected performance baseline under probabilistic independence is the
    product of individual model performances multiplied by an information
    degradation factor.

    Mathematical definition:
    $$P_{\\text{baseline}}(\\mathcal{A}, t) = \\left(\\prod_{i=1}^{n} \\mu_i\\right) \\cdot \\phi(n)$$

    where:
    - $\\prod_{i=1}^{n} \\mu_i$ is the independence baseline (product of individual performances)
    - $\\phi(n) = \\alpha^{n-1}$ models information degradation with α ∈ [0.90, 0.95]

    Remark 4.8 (Information Degradation): The degradation factor φ(n) accounts for
    cumulative information loss through repeated encoding/decoding cycles in sequential
    transformer processing. Each model compresses context into fixed-dimensional
    representations, losing information with each transformation. Empirical validation
    suggests α ≈ 0.93 provides best fit across architectures.

    Args:
        agent_means: Array of mean baseline performances μi for each model.
        alpha: Information degradation parameter (default 0.93 from paper validation).
               Controls how information quality degrades through pipeline.
               Should be in range [0.90, 0.95] based on empirical data.

    Returns:
        Predicted baseline performance for the pipeline.

    Example:
        >>> # Claude 5-agent baseline from paper (Table 4)
        >>> means = np.array([0.595, 0.595, 0.595, 0.595, 0.595])
        >>> baseline = performance_baseline(means, alpha=0.93)
        >>> print(f"Predicted: {baseline:.4f}")  # ≈ 0.0555
        >>> # ∏(0.595) × 0.93^4 = 0.074 × 0.748 ≈ 0.0555

        >>> # Gemini 5-agent baseline from paper
        >>> means = np.array([0.831, 0.831, 0.831, 0.831, 0.831])
        >>> baseline = performance_baseline(means, alpha=0.93)
        >>> print(f"Predicted: {baseline:.4f}")  # ≈ 0.1982

    Note:
        - Uses product baseline (not mean) per Definition 4.7
        - Degradation factor α empirically determined from validation data
        - For n=2: degradation = α^1 = 0.93
        - For n=5: degradation = α^4 ≈ 0.748
        - This is the denominator in γ calculation (Equation 3)
    """
    n = len(agent_means)

    if n < 2:
        raise ValueError("Need at least 2 agents for pipeline baseline")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    if any(m <= 0 for m in agent_means):
        raise ValueError("All agent means must be positive")

    # Independence baseline: product of individual performances
    product_baseline = float(np.prod(agent_means))

    # Information degradation factor
    phi_n = alpha ** (n - 1)

    # Final baseline with degradation
    baseline = product_baseline * phi_n

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
    gamma_norm: float,
    observability_coverage: float,
) -> float:
    """
    Calculate Pipeline Health Score Hpipe(t) from Equation 8.

    Composite operational metric integrating prediction accuracy,
    normalized context propagation strength, and observability coverage into a
    single health indicator for runtime monitoring.

    Mathematical definition:
    $$H_{\\text{pipe}}(t) = \\frac{1}{1 + \\epsilon_{\\text{pred}}(t)} \\times \\min(1, \\gamma_{\\text{norm}}(t)) \\times C_{\\text{obs}}(t)$$

    NOTE: Uses normalized γ_norm (not raw γ) to ensure health scores remain
    comparable across pipelines of different lengths.

    Args:
        epsilon: Prediction error εpred from Equation 7.
        gamma_norm: Normalized context propagation effect γ_norm from Equation 4.
                    Use normalized_context_effect() to compute from raw γ.
        observability_coverage: Fraction of instrumented interactions Cobs.

    Returns:
        Pipeline health score H ∈ [0, 1].
        Higher scores indicate healthier, more predictable pipelines.

    Example:
        >>> # Healthy pipeline: low error, strong normalized context effect, high coverage
        >>> health = pipeline_health_score(epsilon=0.05, gamma_norm=1.35, observability_coverage=0.95)
        >>> print(f"H = {health:.2f}")  # H ≈ 0.90 (healthy)

        >>> # Degraded pipeline: high error, weak context effect
        >>> health = pipeline_health_score(epsilon=0.50, gamma_norm=0.85, observability_coverage=0.75)
        >>> print(f"H = {health:.2f}")  # H ≈ 0.42 (requires attention)

        >>> # Five-model Claude pipeline from paper (Table 4)
        >>> # γ=13.46, γ_norm=1.685, ε=1246%, Cobs=0.87
        >>> health = pipeline_health_score(epsilon=12.46, gamma_norm=1.685, observability_coverage=0.87)
        >>> print(f"H = {health:.2f}")  # H = 0.43

    Operational Interpretation (Table 5 from paper):
        - H > 0.8: Healthy pipeline, standard monitoring
        - 0.6 < H ≤ 0.8: Acceptable, monitor for degradation
        - 0.4 < H ≤ 0.6: Degraded, investigate issues
        - H ≤ 0.4: Critical, immediate attention required

    Five-Model Pipeline Rankings (Table 5):
        1. Gemini 3.5 Pro: H=0.62 (highest baseline + moderate γ_norm)
        2. GPT-4: H=0.49 (balanced baseline and context effect)
        3. Grok 3: H=0.47 (strong context effect, moderate baseline)
        4. Claude Haiku 3.5: H=0.43 (lowest baseline, highest γ_norm)

    Note:
        - Analogous to service-level health scores in distributed systems
        - The min(1, γ_norm) term prevents over-weighting strong context effects
        - All three factors must be healthy for high overall score
        - Using γ_norm ensures comparability across different pipeline lengths
        - This is an engineering metric, not a measure of intelligence
    """
    if not 0.0 <= observability_coverage <= 1.0:
        raise ValueError(f"Observability coverage must be in [0, 1], got {observability_coverage}")

    if epsilon < 0:
        raise ValueError(f"Prediction error must be non-negative, got {epsilon}")

    if gamma_norm <= 0:
        raise ValueError(f"Gamma_norm must be positive, got {gamma_norm}")

    # Prediction accuracy component
    accuracy_factor = 1.0 / (1.0 + epsilon)

    # Context propagation component (capped at 1.0)
    # Uses normalized γ_norm instead of raw γ
    context_factor = min(1.0, gamma_norm)

    # Composite health score
    health = accuracy_factor * context_factor * observability_coverage

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
