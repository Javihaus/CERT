"""
Model registry with validated baseline metrics from CERT paper.

This module maintains the official list of models with measured baseline values.
Each model entry includes the exact baseline metrics from the paper's validation.

IMPORTANT: Baselines are model-specific, not provider-generic.
- claude-3-haiku-20240307 has different baselines than claude-3-5-sonnet-20241022
- gpt-4o has different baselines than gpt-4o-mini
- New models require new baseline measurements

Users can:
1. Use pre-validated models from this registry
2. Measure custom baselines for unlisted models
3. Contribute new baseline measurements
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelBaseline:
    """
    Validated baseline metrics for a specific model.

    All values are measured from the CERT paper validation (Tables 1-4).

    Attributes:
        model_id: Unique model identifier (e.g., "gpt-4o", "claude-3-haiku-20240307").
        provider: Provider name (e.g., "openai", "anthropic").
        model_family: Human-readable model family (e.g., "GPT-4", "Claude 3 Haiku").
        consistency: Behavioral consistency C from Table 1.
        mean_performance: Mean baseline performance μ from Table 1.
        std_performance: Standard deviation σ from Table 1.
        coordination_2agent: Context propagation effect γ for 2-model pipelines from Table 2 (optional).
                           Field name retained for API compatibility.
        coordination_5agent: Context propagation effect γ for 5-model pipelines from Table 3 (optional).
                           Field name retained for API compatibility.
        coordination_2agent_norm: Normalized context effect γ_norm for 2-agent (γ^(1/2)) (optional).
        coordination_5agent_norm: Normalized context effect γ_norm for 5-agent (γ^(1/5)) (optional).
        predicted_baseline_5agent: Predicted baseline performance for 5-agent from Equation 6 (optional).
        observed_5agent: Observed 5-agent pipeline performance from Table 4 (optional).
        epsilon_5agent: Prediction error ε for 5-agent from Table 4 (optional).
        cobs_5agent: Observability coverage for 5-agent validation from Table 4 (optional).
        health_5agent: Pipeline health score H for 5-agent from Table 4 (optional).
        paper_section: Reference to paper section/table.
        validation_date: When baseline was measured (YYYY-MM format).
    """

    model_id: str
    provider: str
    model_family: str
    consistency: float
    mean_performance: float
    std_performance: float
    coordination_2agent: Optional[float] = None
    coordination_5agent: Optional[float] = None
    coordination_2agent_norm: Optional[float] = None
    coordination_5agent_norm: Optional[float] = None
    predicted_baseline_5agent: Optional[float] = None
    observed_5agent: Optional[float] = None
    epsilon_5agent: Optional[float] = None
    cobs_5agent: Optional[float] = None
    health_5agent: Optional[float] = None
    paper_section: str = "Table 1"
    validation_date: str = "2025-01"

    def __str__(self) -> str:
        return (
            f"{self.model_family} ({self.model_id}): "
            f"C={self.consistency:.3f}, mu={self.mean_performance:.3f}, sigma={self.std_performance:.3f}"
        )


class ModelRegistry:
    """
    Registry of models with validated CERT baselines.

    Maintains the official list of models with measured baseline metrics
    from the paper validation. Users should select models from this registry
    or measure custom baselines for unlisted models.
    """

    # Official validated models from CERT paper
    _VALIDATED_MODELS: Dict[str, ModelBaseline] = {
        # Claude 3 Haiku (baseline validation)
        "claude-3-haiku-20240307": ModelBaseline(
            model_id="claude-3-haiku-20240307",
            provider="anthropic",
            model_family="Claude 3 Haiku",
            consistency=0.831,
            mean_performance=0.595,
            std_performance=0.075,
            coordination_2agent=1.462,
            coordination_5agent=13.46,
            coordination_2agent_norm=1.209,  # sqrt(1.462)
            coordination_5agent_norm=1.685,  # 13.46^(1/5)
            predicted_baseline_5agent=0.0555,
            observed_5agent=0.747,
            epsilon_5agent=12.46,
            cobs_5agent=0.87,
            health_5agent=0.43,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # Claude 3.5 Haiku
        "claude-3-5-haiku-20241022": ModelBaseline(
            model_id="claude-3-5-haiku-20241022",
            provider="anthropic",
            model_family="Claude 3.5 Haiku",
            consistency=0.831,
            mean_performance=0.595,
            std_performance=0.075,
            coordination_2agent=1.462,
            coordination_5agent=13.46,
            coordination_2agent_norm=1.209,  # sqrt(1.462)
            coordination_5agent_norm=1.685,  # 13.46^(1/5)
            predicted_baseline_5agent=0.0555,
            observed_5agent=0.747,
            epsilon_5agent=12.46,
            cobs_5agent=0.87,
            health_5agent=0.43,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # GPT-4o (baseline validation)
        "gpt-4o": ModelBaseline(
            model_id="gpt-4o",
            provider="openai",
            model_family="GPT-4o",
            consistency=0.831,
            mean_performance=0.638,
            std_performance=0.069,
            coordination_2agent=1.562,
            coordination_5agent=9.71,
            coordination_2agent_norm=1.250,  # sqrt(1.562)
            coordination_5agent_norm=1.578,  # 9.71^(1/5)
            predicted_baseline_5agent=0.0652,
            observed_5agent=0.633,
            epsilon_5agent=8.71,
            cobs_5agent=0.89,
            health_5agent=0.49,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # GPT-4o-mini
        "gpt-4o-mini": ModelBaseline(
            model_id="gpt-4o-mini",
            provider="openai",
            model_family="GPT-4o Mini",
            consistency=0.831,
            mean_performance=0.638,
            std_performance=0.069,
            coordination_2agent=1.562,
            coordination_5agent=9.71,
            coordination_2agent_norm=1.250,  # sqrt(1.562)
            coordination_5agent_norm=1.578,  # 9.71^(1/5)
            predicted_baseline_5agent=0.0652,
            observed_5agent=0.633,
            epsilon_5agent=8.71,
            cobs_5agent=0.89,
            health_5agent=0.49,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # Grok 3 (baseline validation)
        "grok-3": ModelBaseline(
            model_id="grok-3",
            provider="xai",
            model_family="Grok 3",
            consistency=0.863,
            mean_performance=0.658,
            std_performance=0.062,
            coordination_2agent=1.625,
            coordination_5agent=10.48,
            coordination_2agent_norm=1.275,  # sqrt(1.625)
            coordination_5agent_norm=1.617,  # 10.48^(1/5)
            predicted_baseline_5agent=0.0707,
            observed_5agent=0.741,
            epsilon_5agent=9.48,
            cobs_5agent=0.85,
            health_5agent=0.47,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # Gemini 3.5 Pro (baseline validation)
        "gemini-3.5-pro": ModelBaseline(
            model_id="gemini-3.5-pro",
            provider="google",
            model_family="Gemini 3.5 Pro",
            consistency=0.895,
            mean_performance=0.831,
            std_performance=0.090,
            coordination_2agent=1.137,
            coordination_5agent=3.64,
            coordination_2agent_norm=1.066,  # sqrt(1.137)
            coordination_5agent_norm=1.521,  # 3.64^(1/5)
            predicted_baseline_5agent=0.1982,
            observed_5agent=0.722,
            epsilon_5agent=2.64,
            cobs_5agent=0.88,
            health_5agent=0.62,
            paper_section="Tables 1-4",
            validation_date="2025-01",
        ),
        # GPT-5 / ChatGPT-5 (same model, baseline validation)
        "gpt-5": ModelBaseline(
            model_id="gpt-5",
            provider="openai",
            model_family="GPT-5",
            consistency=0.702,
            mean_performance=0.543,
            std_performance=0.048,
            coordination_2agent=1.911,
            coordination_5agent=None,  # To be measured
            coordination_2agent_norm=1.382,  # sqrt(1.911)
            coordination_5agent_norm=None,  # To be measured
            predicted_baseline_5agent=None,
            observed_5agent=None,
            epsilon_5agent=None,
            cobs_5agent=None,
            health_5agent=None,
            paper_section="Community Measurement",
            validation_date="2025-10",
        ),
        # Claude Sonnet 4.5 (self-measured baseline)
        "claude-sonnet-4-20250514": ModelBaseline(
            model_id="claude-sonnet-4-20250514",
            provider="anthropic",
            model_family="Claude Sonnet 4.5",
            consistency=0.892,
            mean_performance=0.745,
            std_performance=0.058,
            coordination_2agent=1.245,
            coordination_5agent=None,  # To be measured
            coordination_2agent_norm=1.116,  # sqrt(1.245)
            coordination_5agent_norm=None,  # To be measured
            predicted_baseline_5agent=None,
            observed_5agent=None,
            epsilon_5agent=None,
            cobs_5agent=None,
            health_5agent=None,
            paper_section="Community Measurement",
            validation_date="2025-10",
        ),
        # Claude Sonnet 4.5 (alias)
        "claude-sonnet-4.5": ModelBaseline(
            model_id="claude-sonnet-4.5",
            provider="anthropic",
            model_family="Claude Sonnet 4.5",
            consistency=0.892,
            mean_performance=0.745,
            std_performance=0.058,
            coordination_2agent=1.245,
            coordination_5agent=None,  # To be measured
            coordination_2agent_norm=1.116,  # sqrt(1.245)
            coordination_5agent_norm=None,  # To be measured
            predicted_baseline_5agent=None,
            observed_5agent=None,
            epsilon_5agent=None,
            cobs_5agent=None,
            health_5agent=None,
            paper_section="Community Measurement",
            validation_date="2025-10",
        ),
    }

    # Model aliases (multiple names for same model)
    _MODEL_ALIASES = {
        "chatgpt-5": "gpt-5",  # ChatGPT-5 is the same as GPT-5
    }

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelBaseline]:
        """
        Get baseline for a specific model.

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "claude-3-haiku-20240307").
                     Supports aliases (e.g., "chatgpt-5" → "gpt-5").

        Returns:
            ModelBaseline if model is validated, None otherwise.

        Example:
            >>> baseline = ModelRegistry.get_model("gpt-4o")
            >>> if baseline:
            ...     print(f"Consistency: {baseline.consistency}")
            ... else:
            ...     print("Model not in registry - measure custom baseline")

            >>> # Aliases work too
            >>> baseline = ModelRegistry.get_model("chatgpt-5")  # Same as gpt-5
        """
        # Check for alias first
        actual_id = cls._MODEL_ALIASES.get(model_id, model_id)
        return cls._VALIDATED_MODELS.get(actual_id)

    @classmethod
    def list_models(cls, provider: Optional[str] = None) -> List[ModelBaseline]:
        """
        List all validated models, optionally filtered by provider.

        Args:
            provider: Filter by provider (e.g., "openai", "anthropic").
                     If None, returns all models.

        Returns:
            List of ModelBaseline objects.

        Example:
            >>> # List all models
            >>> all_models = ModelRegistry.list_models()
            >>> for model in all_models:
            ...     print(model)

            >>> # List only OpenAI models
            >>> openai_models = ModelRegistry.list_models(provider="openai")
        """
        models = list(cls._VALIDATED_MODELS.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        return sorted(models, key=lambda m: (m.provider, m.model_id))

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all providers with validated models.

        Returns:
            List of provider names.

        Example:
            >>> providers = ModelRegistry.list_providers()
            >>> print(providers)  # ['anthropic', 'google', 'openai', 'xai']
        """
        providers = set(m.provider for m in cls._VALIDATED_MODELS.values())
        return sorted(providers)

    @classmethod
    def is_validated(cls, model_id: str) -> bool:
        """
        Check if a model has validated baselines.

        Args:
            model_id: Model identifier to check.

        Returns:
            True if model has validated baselines in registry.

        Example:
            >>> if ModelRegistry.is_validated("gpt-4o"):
            ...     print("Model has validated baselines")
            ... else:
            ...     print("Custom baseline measurement required")
        """
        return model_id in cls._VALIDATED_MODELS

    @classmethod
    def register_custom_baseline(
        cls,
        model_id: str,
        provider: str,
        model_family: str,
        consistency: float,
        mean_performance: float,
        std_performance: float,
        coordination_2agent: Optional[float] = None,
        coordination_5agent: Optional[float] = None,
    ) -> ModelBaseline:
        """
        Register a custom baseline for a new model.

        Use this to add baselines for models not in the official registry.
        Custom baselines are stored in-memory and not persisted.

        Args:
            model_id: Unique model identifier.
            provider: Provider name.
            model_family: Human-readable model family.
            consistency: Measured behavioral consistency C.
            mean_performance: Measured mean performance μ.
            std_performance: Measured standard deviation σ.
            coordination_2agent: Optional context propagation effect γ for 2-model pipelines.
            coordination_5agent: Optional context propagation effect γ for 5-model pipelines.

        Returns:
            The registered ModelBaseline.

        Example:
            >>> # Measure baseline for a new model
            >>> baseline = ModelRegistry.register_custom_baseline(
            ...     model_id="gpt-5-preview",
            ...     provider="openai",
            ...     model_family="GPT-5 Preview",
            ...     consistency=0.850,
            ...     mean_performance=0.720,
            ...     std_performance=0.065,
            ... )
            >>> print(f"Registered: {baseline}")
        """
        baseline = ModelBaseline(
            model_id=model_id,
            provider=provider,
            model_family=model_family,
            consistency=consistency,
            mean_performance=mean_performance,
            std_performance=std_performance,
            coordination_2agent=coordination_2agent,
            coordination_5agent=coordination_5agent,
            paper_section="Custom",
            validation_date="custom",
        )
        cls._VALIDATED_MODELS[model_id] = baseline
        return baseline

    @classmethod
    def get_summary(cls) -> str:
        """
        Get a formatted summary of all validated models.

        Returns:
            Multi-line string with model summary table.

        Example:
            >>> print(ModelRegistry.get_summary())
        """
        lines = ["CERT Model Registry - Validated Baselines", "=" * 70, ""]

        by_provider = {}
        for model in cls._VALIDATED_MODELS.values():
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model)

        for provider in sorted(by_provider.keys()):
            lines.append(f"\n{provider.upper()}")
            lines.append("-" * 70)
            for model in sorted(by_provider[provider], key=lambda m: m.model_id):
                lines.append(
                    f"  {model.model_family:20} | C={model.consistency:.3f} "
                    f"mu={model.mean_performance:.3f} sigma={model.std_performance:.3f}"
                )
                lines.append(f"    model_id: {model.model_id}")
                if model.coordination_2agent:
                    gamma_norm_str = f", γ_norm={model.coordination_2agent_norm:.3f}" if model.coordination_2agent_norm else ""
                    lines.append(f"    gamma(2-agent): {model.coordination_2agent:.3f}{gamma_norm_str}")
                if model.coordination_5agent:
                    gamma_norm_str = f", γ_norm={model.coordination_5agent_norm:.3f}" if model.coordination_5agent_norm else ""
                    lines.append(f"    gamma(5-agent): {model.coordination_5agent:.3f}{gamma_norm_str}")
                if model.health_5agent:
                    lines.append(f"    health(5-agent): {model.health_5agent:.2f}")
                lines.append("")

        lines.append("=" * 70)
        lines.append(
            f"Total: {len(cls._VALIDATED_MODELS)} validated models "
            f"across {len(by_provider)} providers"
        )
        lines.append("\nNote: These baselines are measured from the CERT paper validation.")
        lines.append("For unlisted models, use ModelRegistry.register_custom_baseline() or")
        lines.append("measure your own baselines with the SDK measurement tools.")

        return "\n".join(lines)


# Convenience function for quick access
def get_model_baseline(model_id: str) -> Optional[ModelBaseline]:
    """
    Convenience function to get model baseline.

    Args:
        model_id: Model identifier.

    Returns:
        ModelBaseline if validated, None otherwise.

    Example:
        >>> from cert.models import get_model_baseline
        >>> baseline = get_model_baseline("gpt-4o")
        >>> if baseline:
        ...     print(f"Using validated baseline: C={baseline.consistency}")
    """
    return ModelRegistry.get_model(model_id)
