"""
xAI Grok provider integration for CERT framework.

Implements the ProviderInterface for xAI's Grok models with validated
baselines from the paper.

Note: xAI uses OpenAI-compatible API, so this implementation wraps
the OpenAI client with xAI-specific configuration.
"""

import asyncio
from typing import Any, List, Optional

from openai import AsyncOpenAI, OpenAIError
from openai import RateLimitError as OpenAIRateLimitError

from cert.models import ModelRegistry
from cert.providers.base import (
    APIError,
    ProviderBaseline,
    ProviderConfig,
    ProviderInterface,
    RateLimitError,
)


class XAIProvider(ProviderInterface):
    """
    xAI Grok provider implementation.

    Supports Grok 3 model with validated baseline.
    Uses OpenAI-compatible API interface.

    Example:
        >>> from cert.providers import XAIProvider
        >>> from cert.providers.base import ProviderConfig
        >>>
        >>> config = ProviderConfig(
        ...     api_key="your-xai-api-key",
        ...     model_name="grok-3",
        ...     temperature=0.7,
        ... )
        >>> provider = XAIProvider(config)
        >>>
        >>> # Check if model has validated baseline
        >>> baseline = provider.get_baseline()
        >>> if baseline:
        ...     print(f"Using {baseline.model_id}: C={baseline.consistency}")
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize xAI Grok provider.

        Args:
            config: Provider configuration with API key and model name.

        Raises:
            ValueError: If model_name is not a supported Grok model.
        """
        super().__init__(config)

        # xAI uses OpenAI-compatible API
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://api.x.ai/v1",  # xAI endpoint
            timeout=config.timeout,
        )

        # Validate model is a Grok model
        if not self._is_grok_model(config.model_name):
            raise ValueError(
                f"Model {config.model_name} is not a recognized Grok model. "
                f"Supported: grok-3, grok-2, grok-beta"
            )

    def _is_grok_model(self, model_name: str) -> bool:
        """Check if model name is a valid Grok model."""
        grok_models = {
            "grok-3",
            "grok-2",
            "grok-beta",
        }
        return any(model_name.startswith(m) for m in grok_models)

    def get_baseline(self) -> Optional[ProviderBaseline]:
        """
        Get validated baseline for the configured Grok model.

        Returns baseline from ModelRegistry if available.

        Validated models:
        - grok-3: C=0.863, μ=0.658, σ=0.062

        Returns:
            ModelBaseline if model is validated, None otherwise.

        Example:
            >>> provider = XAIProvider(config)
            >>> baseline = provider.get_baseline()
            >>> if baseline:
            ...     print(f"Model {baseline.model_id} is validated")
            ... else:
            ...     print("Model not in registry - measure custom baseline")
        """
        return ModelRegistry.get_model(self.config.model_name)

    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a single response from Grok model.

        Args:
            prompt: Input prompt/task description.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens in response (uses config default if None).
            **kwargs: Additional parameters (follows OpenAI format).

        Returns:
            Generated response text.

        Raises:
            APIError: For API-related failures.
            RateLimitError: When rate limit is exceeded.

        Example:
            >>> response = await provider.generate_response(
            ...     prompt="Analyze the key factors in business strategy",
            ...     temperature=0.7,
            ...     max_tokens=1024,
            ... )
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            response = await self._make_request_with_retry(
                self._generate_single,
                prompt=prompt,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs,
            )
            return response
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"xAI rate limit exceeded: {e}")
        except OpenAIError as e:
            raise APIError(f"xAI API error: {e}")

    async def _generate_single(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Internal method to generate a single response."""
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def batch_generate(
        self,
        prompts: List[str],
        n_samples: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> List[List[str]]:
        """
        Generate multiple responses for multiple prompts.

        Uses asyncio.gather for concurrent API calls to maximize throughput.

        Args:
            prompts: List of input prompts.
            n_samples: Number of response samples per prompt.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens (uses config default if None).
            **kwargs: Additional parameters.

        Returns:
            List of response lists, where result[i][j] is the j-th response
            to the i-th prompt.

        Raises:
            APIError: For API-related failures.
            RateLimitError: When rate limit is exceeded.

        Example:
            >>> # Generate 5 samples for each of 3 prompts
            >>> responses = await provider.batch_generate(
            ...     prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
            ...     n_samples=5,
            ... )
            >>> # responses[0] contains 5 responses to "Prompt 1"
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        # Create all tasks
        tasks = []
        for prompt in prompts:
            for _ in range(n_samples):
                tasks.append(
                    self.generate_response(
                        prompt=prompt,
                        temperature=temp,
                        max_tokens=max_tok,
                        **kwargs,
                    )
                )

        # Execute concurrently
        try:
            all_responses = await asyncio.gather(*tasks)
        except (OpenAIRateLimitError, APIError, RateLimitError):
            raise

        # Reshape into [prompts][samples] structure
        result = []
        idx = 0
        for _ in prompts:
            prompt_responses = all_responses[idx : idx + n_samples]
            result.append(prompt_responses)
            idx += n_samples

        return result

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Note: xAI does not currently provide embeddings API. This method
        raises NotImplementedError. Use SemanticAnalyzer from
        cert.analysis.semantic for semantic distance calculations.

        Args:
            text: Input text to embed.

        Raises:
            NotImplementedError: Always raised.

        Example:
            >>> # Use SemanticAnalyzer instead
            >>> from cert.analysis.semantic import SemanticAnalyzer
            >>> analyzer = SemanticAnalyzer()
            >>> embedding = analyzer.get_embedding(text)
        """
        raise NotImplementedError(
            "xAI does not provide embeddings API. "
            "Use cert.analysis.semantic.SemanticAnalyzer for semantic distance calculations."
        )

    def __repr__(self) -> str:
        """String representation."""
        baseline = self.get_baseline()
        baseline_str = ", validated" if baseline else ", not validated"
        return f"XAIProvider(model={self.config.model_name}{baseline_str})"
