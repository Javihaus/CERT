"""
Basic CERT SDK Usage - Using Validated Models

This example shows the recommended workflow:
1. List available validated models from the registry
2. Select a model you have access to
3. Use pre-validated baselines directly

For custom models or domain-specific tasks, see advanced_usage.py
"""

import asyncio
from cert.models import ModelRegistry, get_model_baseline
from cert.providers import OpenAIProvider, AnthropicProvider, GoogleProvider, XAIProvider
from cert.providers.base import ProviderConfig
from cert.analysis.semantic import SemanticAnalyzer
from cert.analysis.quality import QualityScorer
from cert.core.metrics import (
    behavioral_consistency,
    empirical_performance_distribution,
    coordination_effect,
)
import numpy as np


def select_model_interactive():
    """
    Interactive model selection from validated registry.

    Shows available models and helps user choose one they have access to.
    """
    print("=" * 70)
    print("CERT SDK - Model Selection")
    print("=" * 70)
    print("\nAvailable validated models from CERT paper:\n")

    # Group models by provider
    providers = {}
    for model in ModelRegistry.list_models():
        if model.provider not in providers:
            providers[model.provider] = []
        providers[model.provider].append(model)

    # Display models by provider
    model_options = []
    idx = 1
    for provider_name in sorted(providers.keys()):
        print(f"\n{provider_name.upper()} Models:")
        for model in providers[provider_name]:
            print(f"  [{idx}] {model.model_family}")
            print(f"      model_id: {model.model_id}")
            print(f"      Baseline: C={model.consistency:.3f}, μ={model.mean_performance:.3f}, σ={model.std_performance:.3f}")
            if model.coordination_2agent:
                print(f"      2-agent γ: {model.coordination_2agent:.3f}")
            model_options.append(model)
            idx += 1

    print("\n" + "=" * 70)
    print("\nSelect a model you have API access to (1-{}):".format(len(model_options)))

    choice = int(input("> "))
    selected_model = model_options[choice - 1]

    print(f"\n✓ Selected: {selected_model.model_family} ({selected_model.model_id})")
    print(f"  Using validated baseline from paper:")
    print(f"  C={selected_model.consistency:.3f}, μ={selected_model.mean_performance:.3f}, σ={selected_model.std_performance:.3f}\n")

    return selected_model


async def measure_behavioral_consistency_simple(provider, model_baseline, n_trials=10):
    """
    Measure behavioral consistency using validated model.

    Uses pre-validated baseline as reference point.

    Args:
        provider: Initialized provider (OpenAI, Anthropic, etc.)
        model_baseline: Validated baseline from registry
        n_trials: Number of trials (paper used 20, we use 10 for speed)
    """
    print(f"\n{'='*70}")
    print(f"Measuring Behavioral Consistency")
    print(f"{'='*70}")

    # Standard prompt from paper's experiments
    prompt = "Analyze the key factors in effective business strategy implementation."

    print(f"\nGenerating {n_trials} responses to measure consistency...")
    print(f"Prompt: '{prompt}'")

    # Generate responses
    responses = []
    for i in range(n_trials):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,  # Standard temperature
        )
        responses.append(response)
        print(f"  Response {i+1}/{n_trials} generated ({len(response)} chars)")

    # Calculate semantic distances
    print(f"\nCalculating semantic distances...")
    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)

    # Calculate consistency
    consistency = behavioral_consistency(distances)

    # Compare to paper baseline
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Measured Consistency: C = {consistency:.3f}")
    print(f"Paper Baseline:       C = {model_baseline.consistency:.3f}")

    diff = consistency - model_baseline.consistency
    if abs(diff) < 0.05:
        status = "✓ Within expected range"
    elif diff > 0:
        status = "↑ Higher than baseline (more consistent)"
    else:
        status = "↓ Lower than baseline (less consistent)"

    print(f"Difference:           {diff:+.3f} ({status})")

    return consistency


async def measure_performance_distribution_simple(provider, model_baseline, n_trials=10):
    """
    Measure performance distribution using validated model.

    Uses quality scoring from paper (Equation 8).

    Args:
        provider: Initialized provider
        model_baseline: Validated baseline from registry
        n_trials: Number of trials (paper used 15, we use 10 for speed)
    """
    print(f"\n{'='*70}")
    print(f"Measuring Performance Distribution")
    print(f"{'='*70}")

    # Standard prompts from paper's experiments
    prompts = [
        "Analyze the key factors in business strategy.",
        "Evaluate the main considerations for project management.",
        "Assess the critical elements in organizational change.",
        "Identify the primary aspects of market analysis.",
        "Examine the essential components of risk assessment.",
    ]

    print(f"\nGenerating responses for {len(prompts)} prompts...")

    # Generate and score responses
    scorer = QualityScorer()
    quality_scores = []

    for i, prompt in enumerate(prompts[:n_trials//2 + 1]):  # Use subset for speed
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )

        # Score using paper's quality metrics
        components = scorer.score(prompt, response)
        quality_scores.append(components.composite_score)

        print(f"  Prompt {i+1}: Q = {components.composite_score:.3f}")
        print(f"    (semantic: {components.semantic_relevance:.3f}, "
              f"coherence: {components.linguistic_coherence:.3f}, "
              f"density: {components.content_density:.3f})")

    # Calculate distribution
    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    # Compare to paper baseline
    print(f"\n{'='*70}")
    print("Results:")
    print(f"{'='*70}")
    print(f"Measured Performance: μ = {mu:.3f}, σ = {sigma:.3f}")
    print(f"Paper Baseline:       μ = {model_baseline.mean_performance:.3f}, σ = {model_baseline.std_performance:.3f}")

    mu_diff = mu - model_baseline.mean_performance
    print(f"Mean difference:      {mu_diff:+.3f}")

    return mu, sigma


async def demonstrate_coordination_prediction(model_baseline):
    """
    Demonstrate coordination effect prediction using validated baseline.

    Shows how to use paper's baselines for pipeline prediction without
    needing to run actual coordination experiments.

    Args:
        model_baseline: Validated baseline from registry
    """
    print(f"\n{'='*70}")
    print(f"Coordination Effect Prediction (from Paper)")
    print(f"{'='*70}")

    if model_baseline.coordination_2agent:
        print(f"\nValidated 2-agent coordination effect from paper:")
        print(f"  γ = {model_baseline.coordination_2agent:.3f}")

        # Calculate expected coordination performance
        independent_perf = model_baseline.mean_performance
        coordinated_perf = independent_perf * independent_perf * model_baseline.coordination_2agent

        print(f"\nPrediction for 2-agent sequential pipeline:")
        print(f"  Independent performance: {independent_perf:.3f}")
        print(f"  Expected coordinated:    {coordinated_perf:.3f}")
        print(f"  Improvement:             {(coordinated_perf/independent_perf - 1)*100:+.1f}%")
    else:
        print("\n⚠ 2-agent coordination baseline not available for this model.")
        print("  You can measure it using coordination experiments.")


async def main():
    """
    Main demonstration flow:
    1. Select validated model from registry
    2. Initialize provider with API key
    3. Run measurements using validated baselines
    """
    print("\n" + "="*70)
    print("CERT SDK - Basic Usage with Validated Models")
    print("="*70)

    # Step 1: Select model from validated registry
    model_baseline = select_model_interactive()

    # Step 2: Get API key
    print(f"\nEnter your {model_baseline.provider} API key:")
    api_key = input("> ").strip()

    # Step 3: Initialize provider
    config = ProviderConfig(
        api_key=api_key,
        model_name=model_baseline.model_id,
        temperature=0.7,
        max_tokens=1024,
    )

    # Map provider name to class
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "xai": XAIProvider,
    }

    ProviderClass = provider_map[model_baseline.provider]
    provider = ProviderClass(config)

    print(f"\n✓ Provider initialized: {provider}")

    # Step 4: Run measurements
    print(f"\n{'='*70}")
    print("Starting CERT Measurements")
    print(f"{'='*70}")
    print("\nThis will:")
    print("1. Measure behavioral consistency (10 trials)")
    print("2. Measure performance distribution (5 prompts)")
    print("3. Show coordination predictions from paper")
    print("\nEstimated time: 2-3 minutes")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    # Measure consistency
    consistency = await measure_behavioral_consistency_simple(
        provider, model_baseline, n_trials=10
    )

    # Measure performance
    mu, sigma = await measure_performance_distribution_simple(
        provider, model_baseline, n_trials=5
    )

    # Show coordination predictions
    await demonstrate_coordination_prediction(model_baseline)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"\nModel: {model_baseline.model_family} ({model_baseline.model_id})")
    print(f"\nYour Measurements:")
    print(f"  Consistency:   C = {consistency:.3f}")
    print(f"  Performance:   μ = {mu:.3f}, σ = {sigma:.3f}")
    print(f"\nPaper Baselines:")
    print(f"  Consistency:   C = {model_baseline.consistency:.3f}")
    print(f"  Performance:   μ = {model_baseline.mean_performance:.3f}, σ = {model_baseline.std_performance:.3f}")

    if model_baseline.coordination_2agent:
        print(f"  2-agent γ:     {model_baseline.coordination_2agent:.3f}")

    print(f"\n{'='*70}")
    print("✓ Basic measurements complete!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  - Run more trials for statistical significance (20+ recommended)")
    print("  - Measure coordination effects with multi-agent pipelines")
    print("  - See advanced_usage.py for custom models and domain-specific tasks")
    print(f"  - See examples/two_agent_coordination.py for coordination measurement")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nMeasurement cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Check that your API key is valid")
        print("  - Ensure you have access to the selected model")
        print("  - Verify your network connection")
