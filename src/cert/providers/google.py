"""
Google Gemini provider integration for CERT framework.

Implements the ProviderInterface for Google's Gemini models with validated
baselines from the paper.
"""

from typing import List, Optional, Any
import asyncio
import google.generativeai as genai
from cert.providers.base import (
    ProviderInterface,
    ProviderConfig,
    ProviderBaseline,
    APIError,
    RateLimitError,
)
from cert.models import ModelRegistry


class GoogleProvider(ProviderInterface):
    """
    Google Gemini provider implementation.

    Supports Gemini 3.5 Pro model with validated baseline.

    Example:
        >>> from cert.providers import GoogleProvider
        >>> from cert.providers.base import ProviderConfig
        >>>
        >>> config = ProviderConfig(
        ...     api_key="your-api-key",
        ...     model_name="gemini-3.5-pro",
        ...     temperature=0.7,
        ... )
        >>> provider = GoogleProvider(config)
        >>>
        >>> # Check if model has validated baseline
        >>> baseline = provider.get_baseline()
        >>> if baseline:
        ...     print(f"Using {baseline.model_id}: C={baseline.consistency}")
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize Google Gemini provider.

        Args:
            config: Provider configuration with API key and model name.

        Raises:
            ValueError: If model_name is not a supported Gemini model.
        """
        super().__init__(config)
        genai.configure(api_key=config.api_key)

        # Validate model is a Gemini model
        if not self._is_gemini_model(config.model_name):
            raise ValueError(
                f"Model {config.model_name} is not a recognized Gemini model. "
                f"Supported: gemini-3.5-pro, gemini-pro, gemini-1.5-pro"
            )

        self.model = genai.GenerativeModel(config.model_name)

    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if model name is a valid Gemini model."""
        gemini_models = {
            "gemini-3.5-pro",
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        }
        return any(model_name.startswith(m) for m in gemini_models)

    def get_baseline(self) -> Optional[ProviderBaseline]:
        """
        Get validated baseline for the configured Gemini model.

        Returns baseline from ModelRegistry if available.

        Validated models:
        - gemini-3.5-pro: C=0.895, μ=0.831, σ=0.090

        Returns:
            ModelBaseline if model is validated, None otherwise.

        Example:
            >>> provider = GoogleProvider(config)
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
        Generate a single response from Gemini model.

        Args:
            prompt: Input prompt/task description.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens in response (uses config default if None).
            **kwargs: Additional Gemini-specific parameters.

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
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Google rate limit exceeded: {e}")
            else:
                raise APIError(f"Google API error: {e}")

    async def _generate_single(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Internal method to generate a single response."""
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs,
        )

        # Google API is synchronous, run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                prompt,
                generation_config=generation_config,
            ),
        )

        return response.text

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

        Uses asyncio.gather for concurrent execution.

        Args:
            prompts: List of input prompts.
            n_samples: Number of response samples per prompt.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens (uses config default if None).
            **kwargs: Additional Gemini-specific parameters.

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
        except (APIError, RateLimitError):
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
        Get embedding vector using Gemini's embedding model.

        Uses text-embedding-004 model (768 dimensions).

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector (768 dimensions).

        Raises:
            APIError: For API-related failures.

        Example:
            >>> embedding = await provider.get_embedding("Sample text")
            >>> len(embedding)  # 768
        """
        try:
            response = await self._make_request_with_retry(
                self._get_embedding_internal,
                text=text,
            )
            return response
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Google rate limit exceeded: {e}")
            else:
                raise APIError(f"Google API error: {e}")

    async def _get_embedding_internal(self, text: str) -> List[float]:
        """Internal method to get embedding."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="semantic_similarity",
            ),
        )
        return result["embedding"]

    def __repr__(self) -> str:
        """String representation."""
        baseline = self.get_baseline()
        baseline_str = f", validated" if baseline else ", not validated"
        return f"GoogleProvider(model={self.config.model_name}{baseline_str})"
