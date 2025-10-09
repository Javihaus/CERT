"""
Abstract base interface for LLM provider integrations.

Defines the standard interface that all provider implementations must follow,
including retry logic, rate limiting, and error handling.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import aiohttp


@dataclass
class ProviderConfig:
    """
    Configuration for provider initialization.

    Attributes:
        api_key: API key for authentication.
        model_name: Specific model to use (e.g., "gpt-4o", "claude-3-5-sonnet").
        max_retries: Maximum number of retry attempts (default 3).
        timeout: Request timeout in seconds (default 60).
        rate_limit_rpm: Requests per minute limit (default 60).
        temperature: Default sampling temperature (default 0.7).
        max_tokens: Default maximum tokens in response (default 1024).
    """

    api_key: str
    model_name: str
    max_retries: int = 3
    timeout: int = 60
    rate_limit_rpm: int = 60
    temperature: float = 0.7
    max_tokens: int = 1024


# Import ModelBaseline from models registry
# ProviderBaseline is now an alias for ModelBaseline for backward compatibility
try:
    from cert.models import ModelBaseline as ProviderBaseline
except ImportError:
    # Fallback if models.py not available
    @dataclass
    class ProviderBaseline:
        """Baseline metrics (deprecated - use cert.models.ModelBaseline)."""
        consistency: float
        mean_performance: float
        std_performance: float
        model_id: str = ""


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    pass


class APIError(ProviderError):
    """Raised for API-related errors."""

    pass


class ProviderInterface(ABC):
    """
    Abstract base class for LLM provider integrations.

    All provider implementations (OpenAI, Anthropic, Google, xAI) must
    implement this interface to ensure consistent behavior across the SDK.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize provider with configuration.

        Args:
            config: Provider configuration including API key and parameters.
        """
        self.config = config
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / config.rate_limit_rpm

    @abstractmethod
    def get_baseline(self) -> Optional[ProviderBaseline]:
        """
        Get baseline metrics for the configured model.

        Returns validated baseline from ModelRegistry if available.
        Returns None if model is not in the validated registry.

        Model-Specific Baselines (from CERT paper Tables 1-3):
        - claude-3-haiku-20240307: C=0.831, μ=0.595, σ=0.075
        - gpt-4o: C=0.831, μ=0.638, σ=0.069
        - grok-3: C=0.863, μ=0.658, σ=0.062
        - gemini-3.5-pro: C=0.895, μ=0.831, σ=0.090

        IMPORTANT: Baselines are MODEL-specific, not provider-generic.
        claude-3-5-sonnet will have different baselines than claude-3-haiku.

        Returns:
            ModelBaseline with measured values from paper, or None if not validated.

        Example:
            >>> provider = OpenAIProvider(config)
            >>> baseline = provider.get_baseline()
            >>> if baseline:
            ...     print(f"Using validated baseline for {baseline.model_id}")
            ... else:
            ...     print("Model not validated - measure custom baseline")
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a single response from the LLM.

        Args:
            prompt: Input prompt/task description.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens in response (uses config default if None).
            **kwargs: Provider-specific additional parameters.

        Returns:
            Generated response text.

        Raises:
            APIError: For API-related failures.
            RateLimitError: When rate limit is exceeded.
        """
        pass

    @abstractmethod
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

        Useful for behavioral consistency measurements requiring n trials.

        Args:
            prompts: List of input prompts.
            n_samples: Number of response samples per prompt.
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Maximum tokens (uses config default if None).
            **kwargs: Provider-specific parameters.

        Returns:
            List of response lists, where result[i][j] is the j-th response
            to the i-th prompt.

        Raises:
            APIError: For API-related failures.
            RateLimitError: When rate limit is exceeded.
        """
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Used for semantic distance calculations in behavioral consistency
        and quality scoring.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector (typically 1024-1536 dimensions).

        Note:
            Some providers may use separate embedding models
            (e.g., text-embedding-ada-002 for OpenAI).
        """
        pass

    async def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting between requests.

        Uses simple fixed-window rate limiting based on requests per minute.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, APIError)),
    )
    async def _make_request_with_retry(
        self,
        request_fn,
        *args,
        **kwargs,
    ) -> Any:
        """
        Make API request with automatic retry logic.

        Uses exponential backoff: 2s, 4s, 8s between retries.

        Args:
            request_fn: Async function to call.
            *args: Positional arguments for request_fn.
            **kwargs: Keyword arguments for request_fn.

        Returns:
            Result from request_fn.

        Raises:
            APIError: After max retries exceeded.
        """
        await self._enforce_rate_limit()
        return await request_fn(*args, **kwargs)

    async def calculate_semantic_distance(self, text1: str, text2: str) -> float:
        """
        Calculate semantic distance between two texts using provider embeddings.

        Default implementation uses cosine distance. Providers can override
        for optimized implementations.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Semantic distance in [0, 2] range.
        """
        emb1 = await self.get_embedding(text1)
        emb2 = await self.get_embedding(text2)

        # Compute cosine similarity
        import numpy as np

        emb1_arr = np.array(emb1)
        emb2_arr = np.array(emb2)

        # Normalize
        emb1_norm = emb1_arr / np.linalg.norm(emb1_arr)
        emb2_norm = emb2_arr / np.linalg.norm(emb2_arr)

        # Cosine similarity
        cosine_sim = float(np.dot(emb1_norm, emb2_norm))

        # Convert to distance
        distance = 1.0 - cosine_sim

        return distance

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.config.model_name})"
