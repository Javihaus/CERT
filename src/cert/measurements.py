"""
High-level measurement functions for agent evaluation.

These functions provide simple, one-line interfaces to measure
agent consistency, performance, and coordination effects.
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from cert.analysis.quality import QualityScorer
from cert.analysis.semantic import SemanticAnalyzer
from cert.core.metrics import (
    behavioral_consistency,
    empirical_performance_distribution,
)
from cert.models import ModelBaseline
from cert.providers.base import ProviderInterface


async def measure_consistency(
    provider: ProviderInterface,
    prompt: str = "Analyze the key factors in effective business strategy implementation.",
    n_trials: int = 10,
    baseline: Optional[ModelBaseline] = None,
    verbose: bool = True,
) -> float:
    """
    Measure behavioral consistency of an agent.

    This is a high-level function that handles everything:
    - Generates multiple responses to the same prompt
    - Calculates semantic distances
    - Computes consistency score
    - Optionally compares to baseline

    Args:
        provider: Initialized provider instance.
        prompt: Prompt to use for consistency measurement.
        n_trials: Number of trials (default: 10, paper uses 20).
        baseline: Optional baseline to compare against.
        verbose: Print progress and results (default: True).

    Returns:
        Consistency score (C) between 0 and 1.

    Example:
        >>> import cert
        >>> provider = cert.create_provider(api_key="...", model_name="gpt-4o")
        >>> consistency = await cert.measure_consistency(provider, n_trials=10)
        >>> print(f"Consistency: {consistency:.3f}")
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print("Measuring Behavioral Consistency")
        print(f"{'=' * 70}")
        print(f"\nGenerating {n_trials} responses...")
        print(f"Prompt: '{prompt}'")

    # Generate responses
    responses = []
    for i in range(n_trials):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )
        responses.append(response)
        if verbose:
            print(f"  Response {i + 1}/{n_trials} ({len(response)} chars)")

    # Calculate consistency
    if verbose:
        print("\nCalculating semantic distances...")

    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)
    consistency = behavioral_consistency(distances)

    # Show results
    if verbose:
        print(f"\n{'=' * 70}")
        print("Results:")
        print(f"{'=' * 70}")
        print(f"Consistency: C = {consistency:.3f}")

        if baseline:
            print(f"Baseline:    C = {baseline.consistency:.3f}")
            diff = consistency - baseline.consistency
            if abs(diff) < 0.05:
                status = "✓ Within expected range"
            elif diff > 0:
                status = "↑ Higher than baseline"
            else:
                status = "↓ Lower than baseline"
            print(f"Difference:  {diff:+.3f} ({status})")

        print(f"{'=' * 70}\n")

    return consistency


async def measure_performance(
    provider: ProviderInterface,
    prompts: Optional[list] = None,
    baseline: Optional[ModelBaseline] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Measure performance distribution of an agent.

    This is a high-level function that handles everything:
    - Generates responses to multiple prompts
    - Scores quality using multidimensional metrics
    - Computes mean and std performance
    - Optionally compares to baseline

    Args:
        provider: Initialized provider instance.
        prompts: List of prompts (default: standard analytical prompts).
        baseline: Optional baseline to compare against.
        verbose: Print progress and results (default: True).

    Returns:
        Tuple of (mean_performance, std_performance).

    Example:
        >>> import cert
        >>> provider = cert.create_provider(api_key="...", model_name="gpt-4o")
        >>> mu, sigma = await cert.measure_performance(provider)
        >>> print(f"Performance: μ={mu:.3f}, σ={sigma:.3f}")
    """
    # Default prompts from paper
    if prompts is None:
        prompts = [
            "Analyze the key factors in business strategy.",
            "Evaluate the main considerations for project management.",
            "Assess the critical elements in organizational change.",
            "Identify the primary aspects of market analysis.",
            "Examine the essential components of risk assessment.",
        ]

    if verbose:
        print(f"\n{'=' * 70}")
        print("Measuring Performance Distribution")
        print(f"{'=' * 70}")
        print(f"\nGenerating responses for {len(prompts)} prompts...")

    # Generate and score responses
    scorer = QualityScorer()
    quality_scores = []

    for i, prompt in enumerate(prompts):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )

        components = scorer.score(prompt, response)
        quality_scores.append(components.composite_score)

        if verbose:
            print(f"  Prompt {i + 1}: Q = {components.composite_score:.3f}")

    # Calculate distribution
    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    # Show results
    if verbose:
        print(f"\n{'=' * 70}")
        print("Results:")
        print(f"{'=' * 70}")
        print(f"Performance: μ = {mu:.3f}, σ = {sigma:.3f}")

        if baseline:
            print(
                f"Baseline:    μ = {baseline.mean_performance:.3f}, σ = {baseline.std_performance:.3f}"
            )
            mu_diff = mu - baseline.mean_performance
            print(f"Difference:  μ {mu_diff:+.3f}")

        print(f"{'=' * 70}\n")

    return mu, sigma


async def measure_agent(
    provider: ProviderInterface,
    n_consistency_trials: int = 10,
    baseline: Optional[ModelBaseline] = None,
    verbose: bool = True,
) -> dict:
    """
    Measure both consistency and performance in one call.

    This is the simplest way to evaluate an agent - one function
    that measures everything.

    Args:
        provider: Initialized provider instance.
        n_consistency_trials: Number of trials for consistency (default: 10).
        baseline: Optional baseline to compare against.
        verbose: Print progress and results (default: True).

    Returns:
        Dictionary with keys: 'consistency', 'mean_performance', 'std_performance'.

    Example:
        >>> import cert
        >>> provider = cert.create_provider(api_key="...", model_name="gpt-4o")
        >>> results = await cert.measure_agent(provider)
        >>> print(f"Consistency: {results['consistency']:.3f}")
        >>> print(f"Performance: μ={results['mean_performance']:.3f}")
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print("CERT Agent Measurement")
        print(f"{'=' * 70}")
        print("\nMeasuring agent behavioral consistency and performance...")
        print(f"Estimated time: {(n_consistency_trials + 5) * 2} seconds")
        print(f"{'=' * 70}\n")

    # Measure consistency
    consistency = await measure_consistency(
        provider=provider,
        n_trials=n_consistency_trials,
        baseline=baseline,
        verbose=verbose,
    )

    # Measure performance
    mu, sigma = await measure_performance(
        provider=provider,
        baseline=baseline,
        verbose=verbose,
    )

    # Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"Consistency:   C = {consistency:.3f}")
        print(f"Performance:   μ = {mu:.3f}, σ = {sigma:.3f}")

        if baseline:
            print("\nCompared to baseline:")
            print(
                f"  Consistency:   {baseline.consistency:.3f} ({consistency - baseline.consistency:+.3f})"
            )
            print(
                f"  Performance:   {baseline.mean_performance:.3f} ({mu - baseline.mean_performance:+.3f})"
            )

        print(f"{'=' * 70}\n")

    return {
        "consistency": consistency,
        "mean_performance": mu,
        "std_performance": sigma,
    }


async def measure_custom_baseline(
    provider: ProviderInterface,
    prompts: List[str],
    n_consistency_trials: int = 20,
    domain_keywords: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Measure custom baseline for domain-specific applications.

    This function is for advanced use cases where you need custom
    prompts and domain-specific quality scoring (e.g., Healthcare,
    Legal, Finance).

    Args:
        provider: Initialized provider instance.
        prompts: List of domain-specific prompts for your use case.
        n_consistency_trials: Number of trials for consistency (default: 20).
        domain_keywords: Set of domain-specific keywords for quality scoring.
        verbose: Print progress and results (default: True).

    Returns:
        Tuple of (consistency, mean_performance, std_performance).

    Example:
        >>> import cert
        >>> provider = cert.create_provider(api_key="...", model_name="gpt-4-turbo")
        >>>
        >>> healthcare_prompts = [
        ...     "Analyze patient care quality improvement factors.",
        ...     "Evaluate clinical decision support systems.",
        ...     # ... more prompts
        ... ]
        >>>
        >>> healthcare_keywords = {
        ...     "patient", "clinical", "diagnosis", "treatment",
        ...     "healthcare", "medical", "protocol", "safety"
        ... }
        >>>
        >>> c, mu, sigma = await cert.measure_custom_baseline(
        ...     provider=provider,
        ...     prompts=healthcare_prompts,
        ...     domain_keywords=healthcare_keywords,
        ... )
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Custom Domain Baseline Measurement")
        print("=" * 70)
        print(f"\nModel: {provider.config.model_name}")
        print(f"Consistency trials: {n_consistency_trials}")
        print(f"Performance prompts: {len(prompts)}")
        if domain_keywords:
            print(f"Domain keywords: {len(domain_keywords)}")

    # Step 1: Measure Behavioral Consistency
    if verbose:
        print("\n[1/2] Measuring behavioral consistency...")

    consistency_prompt = prompts[0]  # Use first prompt
    responses = []

    for i in range(n_consistency_trials):
        response = await provider.generate_response(
            prompt=consistency_prompt,
            temperature=0.7,
        )
        responses.append(response)
        if verbose and (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{n_consistency_trials}")

    # Calculate consistency
    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)
    consistency = behavioral_consistency(distances)

    if verbose:
        print(f"  ✓ Behavioral Consistency: C = {consistency:.3f}")

    # Step 2: Measure Performance Distribution
    if verbose:
        print("\n[2/2] Measuring performance distribution...")

    # Create custom scorer if domain keywords provided
    if domain_keywords:
        scorer = QualityScorer(domain_keywords=domain_keywords)
        if verbose:
            print(f"  Using {len(domain_keywords)} domain-specific keywords")
    else:
        scorer = QualityScorer()
        if verbose:
            print("  Using default analytical keywords")

    quality_scores = []
    for i, prompt in enumerate(prompts):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )

        components = scorer.score(prompt, response)
        quality_scores.append(components.composite_score)

        if verbose:
            print(f"    Prompt {i + 1}/{len(prompts)}: Q = {components.composite_score:.3f}")

    # Calculate distribution
    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    if verbose:
        print(f"  ✓ Performance: μ = {mu:.3f}, σ = {sigma:.3f}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("Custom Baseline Results")
        print("=" * 70)
        print(f"Consistency:   C = {consistency:.3f}")
        print(f"Mean:          μ = {mu:.3f}")
        print(f"Std Dev:       σ = {sigma:.3f}")
        print("=" * 70 + "\n")

    return consistency, mu, sigma
