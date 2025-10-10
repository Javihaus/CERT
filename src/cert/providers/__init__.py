"""Provider integrations for LLM APIs."""

from typing import Optional

from cert.providers.anthropic import AnthropicProvider
from cert.providers.base import (
    APIError,
    ProviderBaseline,
    ProviderConfig,
    ProviderError,
    ProviderInterface,
    RateLimitError,
)
from cert.providers.google import GoogleProvider
from cert.providers.openai import OpenAIProvider
from cert.providers.xai import XAIProvider


def create_provider(
    api_key: str,
    model_name: str,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs,
) -> ProviderInterface:
    """
    Create a provider instance with simplified initialization.

    Args:
        api_key: API key for the provider.
        model_name: Model identifier (e.g., "gpt-4o", "grok-3").
        provider: Provider name (e.g., "openai", "google", "xai", "anthropic").
                 If None, auto-detected from model_name.
        temperature: Sampling temperature (default: 0.7).
        max_tokens: Maximum tokens in response (default: 1024).
        **kwargs: Additional provider-specific parameters.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider cannot be determined or is not supported.

    Example:
        >>> from cert.providers import create_provider
        >>>
        >>> # Auto-detect provider from model name
        >>> provider = create_provider(
        ...     api_key="your-api-key",
        ...     model_name="gpt-4o",
        ...     temperature=0.7,
        ...     max_tokens=1024,
        ... )
        >>>
        >>> # Explicitly specify provider
        >>> provider = create_provider(
        ...     api_key="your-api-key",
        ...     model_name="gpt-4o",
        ...     provider="openai",
        ... )
    """
    # Auto-detect provider from model name if not specified
    if provider is None:
        from cert.models import ModelRegistry

        baseline = ModelRegistry.get_model(model_name)
        if baseline:
            provider = baseline.provider
        # Fallback: try to infer from model name patterns
        elif model_name.startswith(("gpt-", "gpt")):
            provider = "openai"
        elif model_name.startswith(("grok", "grok-")):
            provider = "xai"
        elif model_name.startswith(("gemini", "gemini-")):
            provider = "google"
        elif model_name.startswith(("claude", "claude-")):
            provider = "anthropic"
        else:
            raise ValueError(
                f"Cannot auto-detect provider for model '{model_name}'. "
                f"Please specify provider explicitly: provider='openai', 'google', 'xai', or 'anthropic'"
            )

    # Map provider name to class
    provider_map = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "xai": XAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_lower = provider.lower()
    if provider_lower not in provider_map:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported providers: {', '.join(provider_map.keys())}"
        )

    # Create config
    config = ProviderConfig(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    # Initialize provider
    ProviderClass = provider_map[provider_lower]
    return ProviderClass(config)


__all__ = [
    "APIError",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenAIProvider",
    "ProviderBaseline",
    "ProviderConfig",
    "ProviderError",
    "ProviderInterface",
    "RateLimitError",
    "XAIProvider",
    "create_provider",
]
