"""
OpenAI provider integration for CERT framework.

Implements the ProviderInterface for OpenAI's GPT models with validated
baselines from the paper.
"""

from typing import List, Optional, Any
import asyncio
from openai import AsyncOpenAI, OpenAIError, RateLimitError as OpenAIRateLimitError
from cert.providers.base import (
    ProviderInterface,
    ProviderConfig,
    ProviderBaseline,
    APIError,
    RateLimitError,
)
from cert.models import ModelRegistry


class OpenAIProvider(ProviderInterface):
    """
    OpenAI provider implementation.

    Supports GPT-4o and GPT-4o-mini models with validated baselines.

    Example:
        >>> from cert.providers import OpenAIProvider
        >>> from cert.providers.base import ProviderConfig
        >>>
        >>> config = ProviderConfig(
        ...     api_key="your-api-key",
        ...     model_name="gpt-4o",
        ...     temperature=0.7,
        ... )
        >>> provider = OpenAIProvider(config)
        >>>
        >>> # Check if model has validated baseline
        >>> baseline = provider.get_baseline()
        >>> if baseline:
        ...     print(f"Using {baseline.model_id}: C={baseline.consistency}")
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: Provider configuration with API key and model name.

        Raises:
            ValueError: If model_name is not a supported OpenAI model.
        """
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
        )

        # Validate model is an OpenAI model
        if not self._is_openai_model(config.model_name):
            raise ValueError(
                f"Model {config.model_name} is not a recognized OpenAI model. "
                f"Supported: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo"
            )

    def _is_openai_model(self, model_name: str) -> bool:
        """Check if model name is a valid OpenAI model."""
        openai_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        }
        return any(model_name.startswith(m) for m in openai_models)

    def get_baseline(self) -> Optional[ProviderBaseline]:
        """
        Get validated baseline for the configured OpenAI model.

        Returns baseline from ModelRegistry if available. For models not
        in the registry (e.g., gpt-3.5-turbo), returns None.

        Validated models:
        - gpt-4o: C=0.831, μ=0.638, σ=0.069
        - gpt-4o-mini: C=0.831, μ=0.638, σ=0.069

        Returns:
            ModelBaseline if model is validated, None otherwise.

        Example:
            >>> provider = OpenAIProvider(config)
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
        Generate a single response from OpenAI model.

        Args:
            prompt: Input prompt/task description.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens in response (uses config default if None).
            **kwargs: Additional OpenAI-specific parameters (e.g., top_p, frequency_penalty).

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
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except OpenAIError as e:
            raise APIError(f"OpenAI API error: {e}")

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
            **kwargs: Additional OpenAI-specific parameters.

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
        Get embedding vector using OpenAI's embedding model.

        Uses text-embedding-3-small by default (1536 dimensions).

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector (1536 dimensions).

        Raises:
            APIError: For API-related failures.

        Example:
            >>> embedding = await provider.get_embedding("Sample text")
            >>> len(embedding)  # 1536
        """
        try:
            response = await self._make_request_with_retry(
                self._get_embedding_internal,
                text=text,
            )
            return response
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except OpenAIError as e:
            raise APIError(f"OpenAI API error: {e}")

    async def _get_embedding_internal(self, text: str) -> List[float]:
        """Internal method to get embedding."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def __repr__(self) -> str:
        """String representation."""
        baseline = self.get_baseline()
        baseline_str = f", validated" if baseline else ", not validated"
        return f"OpenAIProvider(model={self.config.model_name}{baseline_str})"
