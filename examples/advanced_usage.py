"""
Advanced CERT SDK Usage - Custom Models and Domain-Specific Tasks

This example shows advanced features:
1. Using models not in the validated registry
2. Measuring custom baselines for your specific use case
3. Domain-specific quality scoring (Healthcare, Legal, Finance, etc.)
4. Custom prompts and evaluation criteria

For basic usage with validated models, see basic_usage.py
"""

import asyncio
from cert.models import ModelRegistry, ModelBaseline
from cert.providers import OpenAIProvider, AnthropicProvider
from cert.providers.base import ProviderConfig
from cert.analysis.semantic import SemanticAnalyzer
from cert.analysis.quality import QualityScorer
from cert.core.metrics import (
    behavioral_consistency,
    empirical_performance_distribution,
)
import numpy as np


async def measure_custom_baseline_full(provider, prompts, n_consistency_trials=20, domain_keywords=None):
    """
    Measure complete custom baseline for any model.

    This is the recommended approach when:
    - Using a model not in the validated registry
    - Working with domain-specific tasks (Healthcare, Legal, Finance, etc.)
    - Need baselines tailored to your specific use case

    Args:
        provider: Initialized provider with any model
        prompts: List of domain-specific prompts for your use case
        n_consistency_trials: Number of trials for consistency (20+ recommended)
        domain_keywords: Optional set of domain-specific keywords for quality scoring

    Returns:
        Tuple of (consistency, mean_performance, std_performance)
    """
    print("\n" + "="*70)
    print("Custom Baseline Measurement")
    print("="*70)
    print(f"\nModel: {provider.config.model_name}")
    print(f"Consistency trials: {n_consistency_trials}")
    print(f"Performance prompts: {len(prompts)}")

    # Step 1: Measure Behavioral Consistency
    print("\n[1/2] Measuring behavioral consistency...")
    print(f"  Generating {n_consistency_trials} responses to same prompt...")

    consistency_prompt = prompts[0]  # Use first prompt
    responses = []

    for i in range(n_consistency_trials):
        response = await provider.generate_response(
            prompt=consistency_prompt,
            temperature=0.7,
        )
        responses.append(response)
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{n_consistency_trials}")

    # Calculate consistency
    analyzer = SemanticAnalyzer()
    distances = analyzer.pairwise_distances(responses)
    consistency = behavioral_consistency(distances)

    print(f"  ✓ Behavioral Consistency: C = {consistency:.3f}")

    # Step 2: Measure Performance Distribution
    print("\n[2/2] Measuring performance distribution...")
    print(f"  Generating and scoring responses for {len(prompts)} prompts...")

    # Create custom scorer if domain keywords provided
    if domain_keywords:
        scorer = QualityScorer(domain_keywords=domain_keywords)
        print(f"  Using {len(domain_keywords)} domain-specific keywords")
    else:
        scorer = QualityScorer()
        print(f"  Using default analytical keywords")

    quality_scores = []
    for i, prompt in enumerate(prompts):
        response = await provider.generate_response(
            prompt=prompt,
            temperature=0.7,
        )

        components = scorer.score(prompt, response)
        quality_scores.append(components.composite_score)

        print(f"    Prompt {i+1}/{len(prompts)}: Q = {components.composite_score:.3f}")

    # Calculate distribution
    mu, sigma = empirical_performance_distribution(np.array(quality_scores))

    print(f"  ✓ Performance: μ = {mu:.3f}, σ = {sigma:.3f}")

    # Summary
    print("\n" + "="*70)
    print("Custom Baseline Results")
    print("="*70)
    print(f"Consistency:   C = {consistency:.3f}")
    print(f"Mean:          μ = {mu:.3f}")
    print(f"Std Dev:       σ = {sigma:.3f}")
    print("="*70)

    return consistency, mu, sigma


async def example_healthcare_baseline():
    """
    Example: Measuring baseline for healthcare-specific tasks.

    Demonstrates:
    - Custom model (claude-3-5-sonnet not in registry)
    - Domain-specific prompts (medical/healthcare)
    - Custom quality keywords for healthcare domain
    """
    print("\n" + "="*70)
    print("Example: Healthcare Domain Custom Baseline")
    print("="*70)

    # Step 1: Check if model is in registry
    model_id = "claude-3-5-sonnet-20241022"

    if ModelRegistry.is_validated(model_id):
        print(f"\n✓ {model_id} is in validated registry")
        baseline = ModelRegistry.get_model(model_id)
    else:
        print(f"\n⚠ {model_id} is NOT in validated registry")
        print("  We need to measure a custom baseline")

    # Step 2: Define healthcare-specific prompts
    healthcare_prompts = [
        "Analyze the key factors in patient care quality improvement.",
        "Evaluate the main considerations for clinical decision support systems.",
        "Assess the critical elements in healthcare data privacy and security.",
        "Identify the primary aspects of telemedicine implementation.",
        "Examine the essential components of medical staff coordination.",
        "Analyze the challenges in healthcare resource allocation.",
        "Evaluate diagnostic workflow optimization strategies.",
        "Assess patient safety protocols and risk management.",
        "Identify barriers to electronic health record adoption.",
        "Examine factors in healthcare cost reduction.",
    ]

    # Step 3: Define healthcare-specific keywords for quality scoring
    healthcare_keywords = {
        # Clinical terms
        "patient", "clinical", "diagnosis", "treatment", "care",
        "medical", "physician", "nurse", "provider", "practitioner",
        # Healthcare operations
        "hospital", "clinic", "facility", "healthcare", "health",
        "quality", "safety", "protocol", "procedure", "guideline",
        # Technology
        "ehr", "emr", "telemedicine", "telehealth", "digital",
        "system", "technology", "data", "record", "information",
        # Management
        "workflow", "process", "management", "coordination", "efficiency",
        "resource", "allocation", "optimization", "improvement",
        # Compliance
        "hipaa", "compliance", "privacy", "security", "regulation",
        "policy", "standard", "certification", "accreditation",
    }

    print(f"\nHealthcare Domain Configuration:")
    print(f"  - {len(healthcare_prompts)} domain-specific prompts")
    print(f"  - {len(healthcare_keywords)} healthcare keywords for quality scoring")
    print(f"  - Model: {model_id}")

    # Step 4: Get API key and initialize provider
    print(f"\nEnter your Anthropic API key:")
    api_key = input("> ").strip()

    config = ProviderConfig(
        api_key=api_key,
        model_name=model_id,
        temperature=0.7,
        max_tokens=1024,
    )

    provider = AnthropicProvider(config)

    # Step 5: Measure custom baseline
    print("\nMeasuring healthcare-specific baseline...")
    print("Note: This will take 5-10 minutes for full measurement")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    consistency, mu, sigma = await measure_custom_baseline_full(
        provider=provider,
        prompts=healthcare_prompts,
        n_consistency_trials=20,  # Full 20 trials as in paper
        domain_keywords=healthcare_keywords,
    )

    # Step 6: Register custom baseline
    print("\n" + "="*70)
    print("Registering Custom Baseline")
    print("="*70)

    custom_baseline = ModelRegistry.register_custom_baseline(
        model_id=model_id,
        provider="anthropic",
        model_family="Claude 3.5 Sonnet (Healthcare)",
        consistency=consistency,
        mean_performance=mu,
        std_performance=sigma,
    )

    print(f"\n✓ Registered: {custom_baseline}")
    print(f"\nThis baseline is now available for use:")
    print(f"  ModelRegistry.get_model('{model_id}')")

    # Step 7: Compare to paper baselines (if similar model available)
    print("\n" + "="*70)
    print("Comparison to Paper Baselines")
    print("="*70)

    haiku_baseline = ModelRegistry.get_model("claude-3-haiku-20240307")
    if haiku_baseline:
        print(f"\nClaude 3 Haiku (from paper):")
        print(f"  C = {haiku_baseline.consistency:.3f}")
        print(f"  μ = {haiku_baseline.mean_performance:.3f}")
        print(f"  σ = {haiku_baseline.std_performance:.3f}")

        print(f"\nClaude 3.5 Sonnet (Healthcare custom):")
        print(f"  C = {consistency:.3f} ({consistency - haiku_baseline.consistency:+.3f})")
        print(f"  μ = {mu:.3f} ({mu - haiku_baseline.mean_performance:+.3f})")
        print(f"  σ = {sigma:.3f} ({sigma - haiku_baseline.std_performance:+.3f})")

        print(f"\nNote: Differences expected due to:")
        print(f"  - Different model (3.5 Sonnet vs 3 Haiku)")
        print(f"  - Different domain (Healthcare vs General Analytical)")

    return custom_baseline


async def example_legal_baseline():
    """
    Example: Measuring baseline for legal domain tasks.

    Quick demonstration of domain-specific customization.
    """
    print("\n" + "="*70)
    print("Example: Legal Domain Custom Baseline")
    print("="*70)

    # Legal-specific prompts
    legal_prompts = [
        "Analyze the key factors in contract negotiation strategy.",
        "Evaluate risk management in corporate governance.",
        "Assess compliance requirements for data protection.",
        "Identify critical elements in intellectual property protection.",
        "Examine due diligence processes in mergers and acquisitions.",
    ]

    # Legal-specific keywords
    legal_keywords = {
        "legal", "law", "regulation", "compliance", "contract",
        "agreement", "litigation", "court", "judge", "attorney",
        "liability", "obligation", "rights", "statute", "jurisdiction",
        "evidence", "precedent", "case", "ruling", "counsel",
        "due diligence", "intellectual property", "patent", "copyright",
        "governance", "regulatory", "policy", "framework", "standard",
    }

    print(f"\nLegal Domain Configuration:")
    print(f"  - {len(legal_prompts)} legal-specific prompts")
    print(f"  - {len(legal_keywords)} legal keywords")
    print(f"\nTo measure full baseline:")
    print(f"  consistency, mu, sigma = await measure_custom_baseline_full(")
    print(f"      provider=your_provider,")
    print(f"      prompts=legal_prompts,")
    print(f"      n_consistency_trials=20,")
    print(f"      domain_keywords=legal_keywords,")
    print(f"  )")


def show_custom_quality_scoring():
    """
    Demonstrate custom quality scoring weights for specific domains.
    """
    print("\n" + "="*70)
    print("Custom Quality Scoring Configuration")
    print("="*70)

    print("\nDefault scoring (from paper - analytical tasks):")
    print("  Semantic Relevance:    30%")
    print("  Linguistic Coherence:  30%")
    print("  Content Density:       40%")

    print("\nCustom example for creative writing:")
    print("  QualityScorer(")
    print("      semantic_weight=0.2,   # Less critical")
    print("      coherence_weight=0.5,  # Very important")
    print("      density_weight=0.3,    # Moderate")
    print("  )")

    print("\nCustom example for technical documentation:")
    print("  QualityScorer(")
    print("      semantic_weight=0.4,   # Very important")
    print("      coherence_weight=0.2,  # Less critical")
    print("      density_weight=0.4,    # Very important")
    print("  )")


async def main():
    """
    Advanced usage demonstration menu.
    """
    print("\n" + "="*70)
    print("CERT SDK - Advanced Usage Examples")
    print("="*70)

    print("\nThis demonstrates:")
    print("  1. Using custom models (not in validated registry)")
    print("  2. Domain-specific baseline measurement")
    print("  3. Custom quality scoring for different domains")

    print("\nSelect example:")
    print("  [1] Healthcare Domain - Full custom baseline measurement")
    print("  [2] Legal Domain - Configuration example")
    print("  [3] Custom Quality Scoring - Show configuration options")
    print("  [0] Exit")

    choice = input("\n> ").strip()

    if choice == "1":
        await example_healthcare_baseline()
    elif choice == "2":
        await example_legal_baseline()
    elif choice == "3":
        show_custom_quality_scoring()
    elif choice == "0":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Please run again.")

    print("\n" + "="*70)
    print("Advanced Features Summary")
    print("="*70)
    print("\n✓ Custom baselines for any model")
    print("✓ Domain-specific prompts and keywords")
    print("✓ Configurable quality scoring weights")
    print("✓ In-memory baseline registration")

    print("\nFor production use:")
    print("  - Measure baselines with 20+ consistency trials")
    print("  - Use 15+ prompts for performance distribution")
    print("  - Follow paper's experimental methodology (Annex)")
    print("  - Consider contributing validated baselines to registry")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
