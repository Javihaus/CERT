"""
Model registry utilities for easy browsing and selection.

Provides convenient functions to view available models, their baselines,
and provider information.
"""

from typing import Dict, List, Optional
from cert.models import ModelRegistry, ModelBaseline


def list_models(provider: Optional[str] = None) -> List[ModelBaseline]:
    """
    Get list of validated models, optionally filtered by provider.

    Args:
        provider: Optional provider name to filter by (e.g., "openai", "google").
                 If None, returns all models.

    Returns:
        List of ModelBaseline objects.

    Example:
        >>> from cert.utils import list_models
        >>>
        >>> # Get all models
        >>> all_models = list_models()
        >>>
        >>> # Get only OpenAI models
        >>> openai_models = list_models(provider="openai")
    """
    return ModelRegistry.list_models(provider=provider)


def print_models(provider: Optional[str] = None, detailed: bool = True) -> None:
    """
    Print formatted list of available models with their baselines.

    This is the main utility for quickly viewing available models and their
    validated baselines from the CERT paper.

    Args:
        provider: Optional provider name to filter by (e.g., "openai", "google").
                 If None, shows all models.
        detailed: If True, shows full baseline details including coordination effects.
                 If False, shows compact view.

    Example:
        >>> from cert.utils import print_models
        >>>
        >>> # Show all models
        >>> print_models()
        >>>
        >>> # Show only OpenAI models
        >>> print_models(provider="openai")
        >>>
        >>> # Compact view
        >>> print_models(detailed=False)
    """
    print("=" * 70)
    print("CERT SDK - Available Validated Models")
    print("=" * 70)
    print()

    # Group models by provider
    providers: Dict[str, List[ModelBaseline]] = {}
    for model in ModelRegistry.list_models(provider=provider):
        if model.provider not in providers:
            providers[model.provider] = []
        providers[model.provider].append(model)

    if not providers:
        print(f"⚠ No models found" + (f" for provider '{provider}'" if provider else ""))
        print("\nAvailable providers:", ", ".join(ModelRegistry.list_providers()))
        return

    # Display models by provider
    for provider_name in sorted(providers.keys()):
        print(f"\n{provider_name.upper()} Models:")
        print("-" * 70)

        for model in sorted(providers[provider_name], key=lambda m: m.model_id):
            print(f"  • {model.model_family}")
            print(f"    model_id: {model.model_id}")
            print(f"    Baseline: C={model.consistency:.3f}, μ={model.mean_performance:.3f}, σ={model.std_performance:.3f}")

            if detailed:
                if model.coordination_2agent:
                    print(f"    2-agent γ: {model.coordination_2agent:.3f}")
                if model.coordination_5agent:
                    print(f"    5-agent γ: {model.coordination_5agent:.3f}")

            print()

    print("=" * 70)
    print(f"Total: {sum(len(models) for models in providers.values())} validated models")
    print("=" * 70)
    print("\nTo use a model:")
    print("  from cert.models import ModelRegistry")
    print("  baseline = ModelRegistry.get_model('model-id-here')")
    print("\nFor custom models, see examples/advanced_usage.ipynb")


def get_model_info(model_id: str) -> None:
    """
    Print detailed information about a specific model.

    Args:
        model_id: Model identifier (e.g., "gpt-4o", "grok-3").

    Example:
        >>> from cert.utils import get_model_info
        >>>
        >>> get_model_info("gpt-4o")
    """
    baseline = ModelRegistry.get_model(model_id)

    if not baseline:
        print(f"⚠ Model '{model_id}' not found in registry")
        print("\nAvailable models:")
        for model in ModelRegistry.list_models():
            print(f"  - {model.model_id}")
        return

    print("=" * 70)
    print(f"Model: {baseline.model_family}")
    print("=" * 70)
    print(f"\nModel ID:     {baseline.model_id}")
    print(f"Provider:     {baseline.provider}")
    print(f"Family:       {baseline.model_family}")
    print(f"\nValidated Baselines (from CERT paper {baseline.paper_section}):")
    print(f"  Consistency (C):        {baseline.consistency:.3f}")
    print(f"  Mean Performance (μ):   {baseline.mean_performance:.3f}")
    print(f"  Std Performance (σ):    {baseline.std_performance:.3f}")

    if baseline.coordination_2agent:
        print(f"\nCoordination Effects:")
        print(f"  2-agent γ:              {baseline.coordination_2agent:.3f}")
        if baseline.coordination_5agent:
            print(f"  5-agent γ:              {baseline.coordination_5agent:.3f}")

    print(f"\nValidation Date:          {baseline.validation_date}")
    print("=" * 70)

    # Interpretation
    print("\nInterpretation:")
    if baseline.consistency > 0.85:
        print(f"  ✓ High consistency ({baseline.consistency:.3f}) - Very predictable behavior")
    elif baseline.consistency > 0.7:
        print(f"  ~ Moderate consistency ({baseline.consistency:.3f}) - Acceptable with monitoring")
    else:
        print(f"  ⚠ Low consistency ({baseline.consistency:.3f}) - May need prompt engineering")

    if baseline.mean_performance > 0.7:
        print(f"  ✓ High performance ({baseline.mean_performance:.3f})")
    elif baseline.mean_performance > 0.6:
        print(f"  ~ Good performance ({baseline.mean_performance:.3f})")
    else:
        print(f"  ~ Moderate performance ({baseline.mean_performance:.3f})")

    if baseline.coordination_2agent:
        if baseline.coordination_2agent > 1.3:
            print(f"  ✓ Strong coordination effect ({baseline.coordination_2agent:.3f}x) - Agents work well together")
        elif baseline.coordination_2agent > 1.0:
            print(f"  ~ Positive coordination ({baseline.coordination_2agent:.3f}x) - Some benefit from multi-agent")
        else:
            print(f"  ⚠ Weak coordination ({baseline.coordination_2agent:.3f}x) - Limited multi-agent benefit")

    print("\nUsage:")
    print(f"  from cert.providers import {baseline.provider.title()}Provider")
    print(f"  from cert.providers.base import ProviderConfig")
    print(f"  ")
    print(f"  config = ProviderConfig(")
    print(f"      api_key='your-api-key',")
    print(f"      model_name='{baseline.model_id}',")
    print(f"  )")
    print(f"  provider = {baseline.provider.title()}Provider(config)")


def compare_models(*model_ids: str) -> None:
    """
    Compare multiple models side-by-side.

    Args:
        *model_ids: Variable number of model IDs to compare.

    Example:
        >>> from cert.utils import compare_models
        >>>
        >>> compare_models("gpt-4o", "grok-3", "gemini-3.5-pro")
    """
    if len(model_ids) < 2:
        print("⚠ Please provide at least 2 model IDs to compare")
        return

    baselines = []
    for model_id in model_ids:
        baseline = ModelRegistry.get_model(model_id)
        if not baseline:
            print(f"⚠ Model '{model_id}' not found in registry")
            return
        baselines.append(baseline)

    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print()

    # Header
    print(f"{'Metric':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.model_family[:20]:>20}", end="")
    print()
    print("-" * 70)

    # Model IDs
    print(f"{'Model ID':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.model_id[:20]:>20}", end="")
    print()

    # Provider
    print(f"{'Provider':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.provider:>20}", end="")
    print()
    print()

    # Baselines
    print(f"{'Consistency (C)':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.consistency:>20.3f}", end="")
    print()

    print(f"{'Mean Perf (μ)':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.mean_performance:>20.3f}", end="")
    print()

    print(f"{'Std Perf (σ)':<25}", end="")
    for baseline in baselines:
        print(f"{baseline.std_performance:>20.3f}", end="")
    print()

    print(f"{'2-agent γ':<25}", end="")
    for baseline in baselines:
        if baseline.coordination_2agent:
            print(f"{baseline.coordination_2agent:>20.3f}", end="")
        else:
            print(f"{'N/A':>20}", end="")
    print()

    print(f"{'5-agent γ':<25}", end="")
    for baseline in baselines:
        if baseline.coordination_5agent:
            print(f"{baseline.coordination_5agent:>20.3f}", end="")
        else:
            print(f"{'N/A':>20}", end="")
    print()

    print()
    print("=" * 70)

    # Recommendations
    print("\nRecommendations:")

    # Best consistency
    best_consistency = max(baselines, key=lambda b: b.consistency)
    print(f"  Most consistent: {best_consistency.model_family} (C={best_consistency.consistency:.3f})")

    # Best performance
    best_performance = max(baselines, key=lambda b: b.mean_performance)
    print(f"  Best performance: {best_performance.model_family} (μ={best_performance.mean_performance:.3f})")

    # Best coordination
    with_coordination = [b for b in baselines if b.coordination_2agent]
    if with_coordination:
        best_coordination = max(with_coordination, key=lambda b: b.coordination_2agent or 0)
        print(f"  Best coordination: {best_coordination.model_family} (γ={best_coordination.coordination_2agent:.3f})")
