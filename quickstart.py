#!/usr/bin/env python3
"""
CERT SDK Quickstart - See it work in 30 seconds

No API keys needed - shows hardcoded baselines from paper.
"""


def main():
    print("=" * 70)
    print("CERT SDK - Pipeline Health Check")
    print("=" * 70)

    # Import the core library
    from cert.core.metrics import pipeline_health_score
    from cert.models import ModelRegistry

    # Show available validated models
    print("\nValidated models from paper:\n")
    models = ModelRegistry.list_models()

    for model in models[:5]:  # Show first 5
        print(
            f"  {model.model_family:20} C={model.consistency:.3f}  "
            f"mu={model.mean_performance:.3f}  sigma={model.std_performance:.3f}"
        )

    print(f"\n  ... and {len(models) - 5} more")

    # Simulate pipeline health check
    print("\n" + "=" * 70)
    print("Example: 2-agent pipeline health check")
    print("=" * 70)

    # Scenario: GPT-4 -> Claude pipeline
    print("\nPipeline: GPT-4o -> Claude 3.5 Haiku")

    gpt4 = ModelRegistry.get_model("gpt-4o")
    claude = ModelRegistry.get_model("claude-3-5-haiku-20241022")

    print(f"\nGPT-4o baseline:    C={gpt4.consistency:.3f}, mu={gpt4.mean_performance:.3f}")
    print(f"Claude baseline:    C={claude.consistency:.3f}, mu={claude.mean_performance:.3f}")

    # Calculate expected context propagation effect
    if claude.coordination_2agent:
        gamma = claude.coordination_2agent  # Context propagation effect from paper
    else:
        gamma = 1.15  # Typical context effect value from paper

    # Calculate pipeline health (Equation 7 from paper)
    epsilon = 0.05  # Low prediction error
    observability = 0.9  # Good observability

    health = pipeline_health_score(
        epsilon=epsilon, gamma_mean=gamma, observability_coverage=observability
    )

    print(f"\nContext propagation effect: gamma={gamma:.3f}")
    print(f"Pipeline health:            H={health:.3f}")

    if health > 0.8:
        status = "HEALTHY"
        color = "\033[92m"  # Green
    elif health > 0.6:
        status = "DEGRADED"
        color = "\033[93m"  # Yellow
    else:
        status = "UNHEALTHY"
        color = "\033[91m"  # Red

    reset = "\033[0m"

    print(f"\nStatus: {color}{status}{reset}")

    # Show what this means
    print("\n" + "=" * 70)
    print("What this means:")
    print("=" * 70)
    print("  • gamma > 1.0: Sequential context accumulation improves performance")
    print("  • H > 0.8: Pipeline is stable for production")
    print(f"  • C={gpt4.consistency:.3f}: GPT-4o shows consistent behavior")
    print("\nWhat gamma measures:")
    print("  - How attention mechanisms process extended context")
    print("  - Performance change when models see accumulated output")
    print("  - NOT agent coordination or intelligence")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("  1. Run: python examples/basic_usage.py")
    print("  2. Measure your own models with actual API calls")
    print("  3. Monitor production pipelines with CERT metrics")

    print("\n")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nInstall CERT SDK first:")
        print("  pip install -e .")
        print("\nOr install dependencies:")
        print("  pip install numpy scipy sentence-transformers")
