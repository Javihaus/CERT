#!/usr/bin/env python3
"""
Measure Baseline for GPT-5 (or any new OpenAI model)

This script measures the complete baseline metrics for a new model
and registers it in the CERT ModelRegistry.

Usage:
    python measure_gpt5_baseline.py

Requirements:
    - OpenAI API key
    - 10-15 minutes for full measurement
    - ~$2-5 in API costs (depending on model pricing)
"""

import asyncio
import numpy as np
from getpass import getpass

import cert
from cert.models import ModelRegistry
from cert.providers import OpenAIProvider
from cert.providers.base import ProviderConfig
from cert.analysis.semantic import SemanticAnalyzer
from cert.analysis.quality import QualityScorer
from cert.core.metrics import (
    behavioral_consistency,
    empirical_performance_distribution,
    coordination_effect,
)


# Analytical prompts from CERT paper (used for all baseline measurements)
ANALYTICAL_PROMPTS = [
    "Analyze the key factors in effective team communication.",
    "Evaluate the main considerations for project risk management.",
    "Assess the critical elements of successful strategic planning.",
    "Identify the primary aspects of organizational change management.",
    "Examine the essential components of decision-making frameworks.",
    "Analyze the challenges in cross-functional collaboration.",
    "Evaluate process optimization strategies in complex systems.",
    "Assess quality assurance methodologies and their effectiveness.",
    "Identify barriers to innovation in established organizations.",
    "Examine factors influencing stakeholder engagement.",
    "Analyze the relationship between leadership style and outcomes.",
    "Evaluate resource allocation strategies in constrained environments.",
    "Assess the impact of communication channels on information flow.",
    "Identify success metrics for collaborative initiatives.",
    "Examine the role of feedback mechanisms in performance improvement.",
]


async def measure_consistency(provider, prompt, n_trials=20):
    """
    Measure Behavioral Consistency C(Ai, p) from Equation 1.

    Generate n_trials responses to the same prompt and measure
    consistency using semantic distance variability.

    Args:
        provider: CERT provider instance
        prompt: Prompt to use for consistency measurement
        n_trials: Number of trials (paper uses 20)

    Returns:
        Consistency score C ∈ [0, 1]
    """
    print(f"\n{'='*70}")
    print("STEP 1: Measuring Behavioral Consistency")
    print(f"{'='*70}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Trials: {n_trials}")
    print()

    responses = []

    for i in range(n_trials):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,  # Paper uses 0.7
        )
        responses.append(response)

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{n_trials} responses generated")

    print(f"\n  Calculating semantic distances...")
    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)

    consistency = behavioral_consistency(distances)

    print(f"\n  ✓ Behavioral Consistency: C = {consistency:.3f}")
    print(f"    Mean distance: {np.mean(distances):.3f}")
    print(f"    Std distance:  {np.std(distances):.3f}")

    return consistency


async def measure_performance(provider, prompts):
    """
    Measure Empirical Performance Distribution (μ, σ) from Equation 2.

    Generate responses to multiple prompts and score quality.

    Args:
        provider: CERT provider instance
        prompts: List of prompts for performance measurement

    Returns:
        Tuple of (mean μ, std σ)
    """
    print(f"\n{'='*70}")
    print("STEP 2: Measuring Performance Distribution")
    print(f"{'='*70}")
    print(f"Prompts: {len(prompts)}")
    print()

    scorer = QualityScorer()
    quality_scores = []

    for i, prompt in enumerate(prompts):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )

        components = scorer.score(prompt, response)
        quality_scores.append(components.composite_score)

        print(f"  Prompt {i+1}/{len(prompts)}: Q = {components.composite_score:.3f}")

    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    print(f"\n  ✓ Performance Distribution:")
    print(f"    Mean (μ):     {mu:.3f}")
    print(f"    Std Dev (σ):  {sigma:.3f}")
    print(f"    Min quality:  {min(quality_scores):.3f}")
    print(f"    Max quality:  {max(quality_scores):.3f}")

    return mu, sigma, quality_scores


async def measure_coordination_2agent(provider, prompts, n_pairs=5):
    """
    Measure Context Propagation Effect γ for 2-model pipelines.

    Simulates 2-agent sequential processing and compares to independent execution.

    Args:
        provider: CERT provider instance
        prompts: List of prompts
        n_pairs: Number of prompt pairs to test

    Returns:
        Mean γ across all pairs
    """
    print(f"\n{'='*70}")
    print("STEP 3: Measuring Context Propagation Effect (2-agent)")
    print(f"{'='*70}")
    print(f"Testing {n_pairs} prompt pairs")
    print()

    scorer = QualityScorer()
    gamma_values = []

    for i in range(n_pairs):
        prompt1 = prompts[i * 2]
        prompt2 = prompts[i * 2 + 1]

        print(f"\n  Pair {i+1}/{n_pairs}:")
        print(f"    Prompt 1: {prompt1[:50]}...")
        print(f"    Prompt 2: {prompt2[:50]}...")

        # Independent execution
        response1_indep = await provider.generate_response(prompt=prompt1, temperature=0.7)
        response2_indep = await provider.generate_response(prompt=prompt2, temperature=0.7)

        quality1 = scorer.score(prompt1, response1_indep).composite_score
        quality2 = scorer.score(prompt2, response2_indep).composite_score

        print(f"      Independent Q1: {quality1:.3f}, Q2: {quality2:.3f}")

        # Sequential execution (agent 2 sees agent 1's output)
        response1_seq = await provider.generate_response(prompt=prompt1, temperature=0.7)

        # Agent 2 prompt includes agent 1's output as context
        prompt2_with_context = f"Building on this analysis:\n\n{response1_seq}\n\nNow: {prompt2}"
        response2_seq = await provider.generate_response(prompt=prompt2_with_context, temperature=0.7)

        quality_seq = scorer.score(prompt2, response2_seq).composite_score

        print(f"      Sequential Q:   {quality_seq:.3f}")

        # Calculate γ (Equation 3)
        gamma = coordination_effect(
            coordinated_performance=quality_seq,
            independent_performances=[quality1, quality2]
        )

        gamma_values.append(gamma)
        print(f"      γ = {gamma:.3f}")

    mean_gamma = np.mean(gamma_values)

    print(f"\n  ✓ Context Propagation Effect (2-agent):")
    print(f"    Mean γ:  {mean_gamma:.3f}")
    print(f"    Std γ:   {np.std(gamma_values):.3f}")
    print(f"    Min γ:   {min(gamma_values):.3f}")
    print(f"    Max γ:   {max(gamma_values):.3f}")

    return mean_gamma


async def main():
    """Main measurement script."""

    print("="*70)
    print("CERT Baseline Measurement for New Model")
    print("="*70)

    # Step 0: Get model information
    print("\n1. Enter OpenAI API key:")
    api_key = getpass("> ")

    print("\n2. Enter model name (e.g., 'gpt-5', 'gpt-4o', 'gpt-4o-mini'):")
    model_name = input("> ").strip()

    print("\n3. Enter human-readable model family (e.g., 'GPT-5', 'GPT-4o'):")
    model_family = input("> ").strip()

    print("\n4. Check if model already exists in registry:")
    if ModelRegistry.is_validated(model_name):
        existing = ModelRegistry.get_model(model_name)
        print(f"\n⚠️  WARNING: Model '{model_name}' already exists in registry!")
        print(f"   Current baseline: C={existing.consistency:.3f}, μ={existing.mean_performance:.3f}")
        print("\n   Continue to overwrite? (yes/no)")
        if input("> ").strip().lower() != "yes":
            print("\nCancelled.")
            return

    # Initialize provider
    config = ProviderConfig(
        api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
    )

    provider = OpenAIProvider(config)

    print(f"\n✓ Initialized provider for {model_name}")

    # Estimate cost and time
    print(f"\n{'='*70}")
    print("Measurement Plan")
    print(f"{'='*70}")
    print("This will measure:")
    print("  • Behavioral Consistency:      20 trials")
    print("  • Performance Distribution:    15 prompts")
    print("  • Context Propagation (2-agent): 5 pairs (20 calls)")
    print()
    print("Total API calls: ~55")
    print("Estimated time:  10-15 minutes")
    print("Estimated cost:  $2-5 (depends on model pricing)")
    print()
    print("Continue? (yes/no)")

    if input("> ").strip().lower() != "yes":
        print("\nCancelled.")
        return

    # Step 1: Measure Consistency
    consistency = await measure_consistency(
        provider=provider,
        prompt=ANALYTICAL_PROMPTS[0],  # Use first prompt
        n_trials=20,
    )

    # Step 2: Measure Performance
    mu, sigma, quality_scores = await measure_performance(
        provider=provider,
        prompts=ANALYTICAL_PROMPTS,
    )

    # Step 3: Measure Context Propagation (2-agent)
    gamma_2agent = await measure_coordination_2agent(
        provider=provider,
        prompts=ANALYTICAL_PROMPTS,
        n_pairs=5,
    )

    # Summary
    print(f"\n{'='*70}")
    print("MEASUREMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nModel: {model_name} ({model_family})")
    print(f"\nBaseline Metrics:")
    print(f"  Consistency (C):        {consistency:.3f}")
    print(f"  Mean Performance (μ):   {mu:.3f}")
    print(f"  Std Performance (σ):    {sigma:.3f}")
    print(f"  Context Effect γ (2-agent): {gamma_2agent:.3f}")

    # Compare to paper baselines
    print(f"\n{'='*70}")
    print("Comparison to Paper Baselines")
    print(f"{'='*70}")

    paper_models = {
        "Claude 3 Haiku": {"C": 0.831, "μ": 0.595, "γ": 1.462},
        "GPT-4o": {"C": 0.831, "μ": 0.638, "γ": 1.562},
        "Grok 3": {"C": 0.863, "μ": 0.658, "γ": 1.625},
        "Gemini 3.5 Pro": {"C": 0.895, "μ": 0.831, "γ": 1.137},
    }

    print(f"\n{'Model':<20} {'C':>8} {'μ':>8} {'γ':>8}")
    print("-" * 50)
    for name, metrics in paper_models.items():
        print(f"{name:<20} {metrics['C']:>8.3f} {metrics['μ']:>8.3f} {metrics['γ']:>8.3f}")

    print(f"\n{model_family + ' (NEW)':<20} {consistency:>8.3f} {mu:>8.3f} {gamma_2agent:>8.3f}")

    # Interpretation
    print(f"\n{'='*70}")
    print("Interpretation")
    print(f"{'='*70}")

    if consistency > 0.85:
        print(f"✓ HIGH consistency (C={consistency:.3f}): Very stable, predictable behavior")
    elif consistency > 0.80:
        print(f"✓ GOOD consistency (C={consistency:.3f}): Stable behavior with minor variance")
    else:
        print(f"⚠ MODERATE consistency (C={consistency:.3f}): Higher variance, needs more monitoring")

    if mu > 0.70:
        print(f"✓ HIGH baseline performance (μ={mu:.3f}): Strong individual task performance")
    elif mu > 0.60:
        print(f"✓ GOOD baseline performance (μ={mu:.3f}): Solid performance")
    else:
        print(f"⚠ MODERATE baseline performance (μ={mu:.3f}): Room for improvement")

    if gamma_2agent > 1.5:
        print(f"✓ STRONG context propagation (γ={gamma_2agent:.3f}): Benefits greatly from sequential processing")
    elif gamma_2agent > 1.2:
        print(f"✓ MODERATE context propagation (γ={gamma_2agent:.3f}): Benefits from context accumulation")
    elif gamma_2agent > 1.0:
        print(f"⚠ WEAK context propagation (γ={gamma_2agent:.3f}): Minimal benefit from sequential processing")
    else:
        print(f"⚠ NEGATIVE context propagation (γ={gamma_2agent:.3f}): Context may be degrading performance")

    # Register in ModelRegistry
    print(f"\n{'='*70}")
    print("Register in ModelRegistry?")
    print(f"{'='*70}")
    print("\nThis will add the model to the in-memory registry.")
    print("To persist permanently, you'll need to update src/cert/models.py")
    print("\nRegister now? (yes/no)")

    if input("> ").strip().lower() == "yes":
        baseline = ModelRegistry.register_custom_baseline(
            model_id=model_name,
            provider="openai",
            model_family=model_family,
            consistency=consistency,
            mean_performance=mu,
            std_performance=sigma,
            coordination_2agent=gamma_2agent,
        )

        print(f"\n✓ Registered: {baseline}")
        print(f"\nYou can now use this baseline:")
        print(f"  baseline = ModelRegistry.get_model('{model_name}')")

    # Generate code for permanent registration
    print(f"\n{'='*70}")
    print("To Add Permanently to src/cert/models.py")
    print(f"{'='*70}")
    print("\nAdd this to the _VALIDATED_MODELS dictionary:\n")

    print(f'''    # {model_family}
    "{model_name}": ModelBaseline(
        model_id="{model_name}",
        provider="openai",
        model_family="{model_family}",
        consistency={consistency:.3f},
        mean_performance={mu:.3f},
        std_performance={sigma:.3f},
        coordination_2agent={gamma_2agent:.3f},
        coordination_5agent=None,  # Measure separately if needed
        paper_section="Custom Measurement",
        validation_date="{cert.__version__}",
    ),''')

    print("\n" + "="*70)
    print("Measurement Complete!")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nMeasurement cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
