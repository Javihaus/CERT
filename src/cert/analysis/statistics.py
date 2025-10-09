"""
Statistical analysis utilities for CERT framework.

Implements statistical significance tests used in the paper:
- Welch's t-test for coordination effect validation
- Cohen's d for effect size calculation
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import stats


def welch_t_test(
    sample1: NDArray[np.float64],
    sample2: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Perform Welch's t-test for unequal variances.

    Used in the paper to validate statistical significance of coordination
    effects when comparing coordinated vs. independent performance.

    Welch's t-test is more robust than standard t-test when sample sizes
    or variances differ between groups.

    Args:
        sample1: First sample (e.g., independent agent performance).
        sample2: Second sample (e.g., coordinated pipeline performance).

    Returns:
        Tuple of (t_statistic, p_value).
        - t_statistic: The calculated t-value
        - p_value: Two-tailed p-value for significance testing

    Interpretation:
        - p < 0.001: Highly significant (used for Claude, GPT-4, Grok in paper)
        - p < 0.01: Significant (used for Gemini in paper)
        - p < 0.05: Marginally significant
        - p ≥ 0.05: Not statistically significant

    Example:
        >>> independent = np.array([0.60, 0.58, 0.62, 0.59, 0.61])
        >>> coordinated = np.array([0.75, 0.72, 0.78, 0.74, 0.76])
        >>> t_stat, p_value = welch_t_test(independent, coordinated)
        >>> print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
        >>> if p_value < 0.001:
        ...     print("Highly significant coordination effect")
    """
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Need at least 2 samples in each group")

    # Welch's t-test (does not assume equal variances)
    t_statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)

    return float(t_statistic), float(p_value)


def cohen_d(
    sample1: NDArray[np.float64],
    sample2: NDArray[np.float64],
) -> float:
    """
    Calculate Cohen's d effect size.

    Measures the standardized difference between two means, providing
    a scale-independent measure of effect magnitude.

    Used in the paper to quantify coordination effect sizes:
    - Claude, GPT-4, Grok: d > 0.9 (large effects)
    - Gemini: d ≈ 0.66 (moderate effect)

    Formula:
    $$d = \\frac{\\bar{x}_2 - \\bar{x}_1}{s_{\\text{pooled}}}$$

    where $s_{\\text{pooled}} = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$

    Args:
        sample1: First sample (e.g., independent performance).
        sample2: Second sample (e.g., coordinated performance).

    Returns:
        Cohen's d effect size.

    Interpretation:
        - |d| < 0.2: Negligible effect
        - 0.2 ≤ |d| < 0.5: Small effect
        - 0.5 ≤ |d| < 0.8: Medium effect
        - |d| ≥ 0.8: Large effect

    Example:
        >>> independent = np.array([0.60, 0.58, 0.62, 0.59, 0.61])
        >>> coordinated = np.array([0.75, 0.72, 0.78, 0.74, 0.76])
        >>> effect_size = cohen_d(independent, coordinated)
        >>> print(f"Cohen's d = {effect_size:.2f}")
        >>> if effect_size > 0.8:
        ...     print("Large effect size")
    """
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Need at least 2 samples in each group")

    n1 = len(sample1)
    n2 = len(sample2)

    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)

    std1 = np.std(sample1, ddof=1)
    std2 = np.std(sample2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    # Cohen's d
    d = (mean2 - mean1) / pooled_std

    return float(d)


def coefficient_of_variation(data: NDArray[np.float64]) -> float:
    """
    Calculate coefficient of variation (CV).

    Measures relative variability as a percentage. Used in paper's Table 6
    to assess cross-architecture generalization.

    Formula:
    $$CV = \\frac{\\sigma}{\\mu} \\times 100\\%$$

    Args:
        data: Array of measurements.

    Returns:
        Coefficient of variation as a percentage.

    Interpretation (from paper's Table 6):
        - CV < 5%: Excellent generalization (e.g., Consistency C: 3.1%)
        - 5% ≤ CV < 15%: Moderate generalization (e.g., Coordination γ: 13%)
        - CV ≥ 15%: Architecture-dependent (e.g., Prediction ε: 42-67%)

    Example:
        >>> # Behavioral consistency across architectures
        >>> consistency_values = np.array([0.831, 0.831, 0.863, 0.895])
        >>> cv = coefficient_of_variation(consistency_values)
        >>> print(f"CV = {cv:.1f}%")  # 3.1% - excellent generalization
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")

    mean = np.mean(data)
    if mean == 0:
        return 0.0

    std = np.std(data, ddof=1)
    cv = (std / mean) * 100

    return float(cv)


def moving_average(
    data: NDArray[np.float64],
    window_size: int,
) -> NDArray[np.float64]:
    """
    Calculate moving average for time series smoothing.

    Useful for dashboard visualization of metric trends over time.

    Args:
        data: Time series data.
        window_size: Size of the moving average window.

    Returns:
        Smoothed time series (same length as input, with initial values preserved).

    Example:
        >>> metrics_history = np.array([0.7, 0.72, 0.68, 0.71, 0.69, 0.73])
        >>> smoothed = moving_average(metrics_history, window_size=3)
        >>> print(smoothed)
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    if len(data) < window_size:
        # Not enough data for window, return original
        return data

    # Compute moving average using convolution
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(data, weights, mode="same")

    # Fix edge effects by using cumsum for beginning
    for i in range(min(window_size - 1, len(data))):
        smoothed[i] = np.mean(data[: i + 1])

    return smoothed


def confidence_interval(
    data: NDArray[np.float64],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for the mean.

    Args:
        data: Sample data.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).

    Example:
        >>> performance_samples = np.array([0.75, 0.72, 0.78, 0.74, 0.76])
        >>> mean, lower, upper = confidence_interval(performance_samples)
        >>> print(f"Mean: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points")

    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean

    # t-distribution critical value
    confidence_level = confidence
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df)

    margin_of_error = t_critical * std_err

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return float(mean), float(lower_bound), float(upper_bound)
