"""
Response quality scoring implementing the multidimensional framework from CERT paper.

Implements Equation 8: Q(p,r) composite quality score used for baseline
and coordination evaluation.
"""

import re
from typing import List, NamedTuple, Optional, Set

import numpy as np

from cert.analysis.semantic import SemanticAnalyzer


class QualityComponents(NamedTuple):
    """
    Individual components of the quality score Q(p,r).

    Attributes:
        semantic_relevance: Cosine similarity between prompt and response embeddings (0-1).
        linguistic_coherence: Readability and analytical style score (0-1).
        content_density: Normalized presence of domain-specific keywords (0-1).
        composite_score: Weighted combination following Equation 8.
    """

    semantic_relevance: float
    linguistic_coherence: float
    content_density: float
    composite_score: float


class QualityScorer:
    """
    Multidimensional quality scorer for LLM responses.

    Implements the quality scoring framework from Section 4.3:
    Q(p,r) = 0.3 × Semantic Relevance + 0.3 × Linguistic Coherence + 0.4 × Content Density
    """

    def __init__(
        self,
        semantic_weight: float = 0.3,
        coherence_weight: float = 0.3,
        density_weight: float = 0.4,
        domain_keywords: Optional[Set[str]] = None,
    ):
        """
        Initialize quality scorer with configurable weights.

        Args:
            semantic_weight: Weight for semantic relevance (default 0.3 from paper).
            coherence_weight: Weight for linguistic coherence (default 0.3 from paper).
            density_weight: Weight for content density (default 0.4 from paper).
            domain_keywords: Set of domain-specific keywords for density calculation.
                           If None, uses default analytical keywords.
        """
        # Validate weights sum to 1.0
        total_weight = semantic_weight + coherence_weight + density_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}. "
                f"({semantic_weight} + {coherence_weight} + {density_weight})"
            )

        self.semantic_weight = semantic_weight
        self.coherence_weight = coherence_weight
        self.density_weight = density_weight

        # Initialize semantic analyzer for relevance scoring
        self.semantic_analyzer = SemanticAnalyzer()

        # Default domain keywords for analytical tasks (from paper's experiments)
        self.domain_keywords = domain_keywords or {
            # Strategic analysis
            "strategy",
            "strategic",
            "analysis",
            "framework",
            "approach",
            "methodology",
            # Business concepts
            "market",
            "competitive",
            "advantage",
            "value",
            "performance",
            "optimization",
            "efficiency",
            # Analytical terms
            "evaluate",
            "assess",
            "consider",
            "examine",
            "investigate",
            "determine",
            "measure",
            "quantify",
            # Implementation
            "implement",
            "execute",
            "deploy",
            "integrate",
            "manage",
            "coordinate",
            # Risk and governance
            "risk",
            "compliance",
            "governance",
            "control",
            "monitor",
            "assurance",
        }

    def semantic_relevance(self, prompt: str, response: str) -> float:
        """
        Calculate semantic relevance using cosine similarity.

        Measures whether the response addresses the requested task by computing
        embedding similarity between prompt and response.

        Args:
            prompt: The input prompt/task description.
            response: The agent's response.

        Returns:
            Semantic relevance score in [0, 1].
            Higher values indicate better alignment with prompt.

        Note:
            Uses 1 - distance to convert distance to similarity.
            Distance ∈ [0, 2] → Similarity ∈ [-1, 1] → Clipped to [0, 1]
        """
        distance = self.semantic_analyzer.semantic_distance(prompt, response)

        # Convert distance to similarity: similarity = 1 - distance
        # Since distance ∈ [0, 2], similarity ∈ [-1, 1]
        # We clip negative values to 0 for relevance scoring
        similarity = max(0.0, 1.0 - distance)

        return similarity

    def linguistic_coherence(self, response: str) -> float:
        """
        Calculate linguistic coherence via readability metrics.

        Measures readability and analytical style through average sentence
        and word length, ensuring responses are well-structured and accessible.

        Based on readability research (Flesch, Kincaid et al.) adapted for
        analytical content evaluation.

        Args:
            response: The text to evaluate.

        Returns:
            Coherence score in [0, 1].
            Scores near 0.5-0.7 indicate optimal analytical style.

        Note:
            - Optimal sentence length: 15-25 words (professional writing)
            - Optimal word length: 4-6 characters (balanced complexity)
            - Scores normalized to [0, 1] with peaks in optimal ranges
        """
        if not response or not response.strip():
            return 0.0

        # Sentence splitting (rough approximation)
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Word splitting
        words = response.split()
        if not words:
            return 0.0

        # Average sentence length (in words)
        avg_sentence_length = len(words) / len(sentences)

        # Average word length (in characters)
        avg_word_length = np.mean([len(word) for word in words])

        # Normalize to [0, 1] with optimal ranges
        # Optimal sentence length: 15-25 words
        if 15 <= avg_sentence_length <= 25:
            sentence_score = 1.0
        elif avg_sentence_length < 15:
            sentence_score = avg_sentence_length / 15.0
        else:  # > 25
            sentence_score = max(0.0, 1.0 - (avg_sentence_length - 25) / 25.0)

        # Optimal word length: 4-6 characters
        if 4 <= avg_word_length <= 6:
            word_score = 1.0
        elif avg_word_length < 4:
            word_score = avg_word_length / 4.0
        else:  # > 6
            word_score = max(0.0, 1.0 - (avg_word_length - 6) / 6.0)

        # Combine sentence and word metrics
        coherence = (sentence_score + word_score) / 2.0

        return coherence

    def content_density(self, response: str) -> float:
        """
        Calculate content density through domain-specific keyword presence.

        Measures analytical richness by quantifying the normalized presence
        of domain-specific keywords, reflecting informational completeness.

        Args:
            response: The text to evaluate.

        Returns:
            Content density score in [0, 1].
            Higher values indicate more domain-specific content.

        Note:
            Normalized by response length to avoid bias toward longer responses.
        """
        if not response or not response.strip():
            return 0.0

        # Convert to lowercase for matching
        response_lower = response.lower()

        # Count keyword occurrences
        keyword_count = 0
        for keyword in self.domain_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(keyword) + r"\b"
            keyword_count += len(re.findall(pattern, response_lower))

        # Normalize by response word count
        words = response.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Density as keyword frequency (keywords per 100 words)
        density_raw = (keyword_count / word_count) * 100

        # Normalize to [0, 1] with saturation at ~20% keyword density
        # (beyond this is likely keyword stuffing)
        density_normalized = min(1.0, density_raw / 20.0)

        return density_normalized

    def score(self, prompt: str, response: str) -> QualityComponents:
        """
        Calculate composite quality score Q(p,r) following Equation 8.

        Q(p,r) = 0.3 × Semantic Relevance + 0.3 × Linguistic Coherence + 0.4 × Content Density

        Args:
            prompt: The input prompt/task description.
            response: The agent's response.

        Returns:
            QualityComponents with individual scores and composite Q(p,r).

        Example:
            >>> scorer = QualityScorer()
            >>> components = scorer.score(
            ...     prompt="Analyze the key factors in business strategy.",
            ...     response="Strategic analysis requires evaluating market position..."
            ... )
            >>> print(f"Q(p,r) = {components.composite_score:.3f}")
            >>> print(f"  Semantic: {components.semantic_relevance:.3f}")
            >>> print(f"  Coherence: {components.linguistic_coherence:.3f}")
            >>> print(f"  Density: {components.content_density:.3f}")
        """
        semantic = self.semantic_relevance(prompt, response)
        coherence = self.linguistic_coherence(response)
        density = self.content_density(response)

        # Composite score (Equation 8)
        composite = (
            self.semantic_weight * semantic
            + self.coherence_weight * coherence
            + self.density_weight * density
        )

        return QualityComponents(
            semantic_relevance=semantic,
            linguistic_coherence=coherence,
            content_density=density,
            composite_score=composite,
        )

    def batch_score(self, prompt: str, responses: List[str]) -> List[QualityComponents]:
        """
        Score multiple responses to the same prompt.

        Efficient for baseline measurements requiring multiple trials.

        Args:
            prompt: The input prompt/task description.
            responses: List of agent responses.

        Returns:
            List of QualityComponents for each response.

        Example:
            >>> scorer = QualityScorer()
            >>> scores = scorer.batch_score(
            ...     prompt="Analyze business strategy.",
            ...     responses=["Response 1...", "Response 2...", "Response 3..."]
            ... )
            >>> quality_scores = [s.composite_score for s in scores]
            >>> # Use for empirical performance distribution
            >>> from cert.core.metrics import empirical_performance_distribution
            >>> mu, sigma = empirical_performance_distribution(np.array(quality_scores))
        """
        return [self.score(prompt, response) for response in responses]


# Module-level convenience function
_default_scorer: Optional[QualityScorer] = None


def get_default_scorer() -> QualityScorer:
    """Get or create the default global quality scorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = QualityScorer()
    return _default_scorer


def quality_score(prompt: str, response: str) -> float:
    """
    Convenience function for composite quality score using default scorer.

    Args:
        prompt: The input prompt/task description.
        response: The agent's response.

    Returns:
        Composite quality score Q(p,r) in [0, 1].

    Example:
        >>> q = quality_score("Analyze the market.", "The market shows growth...")
        >>> print(f"Quality: {q:.3f}")
    """
    scorer = get_default_scorer()
    components = scorer.score(prompt, response)
    return components.composite_score
