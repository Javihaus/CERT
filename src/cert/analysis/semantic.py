"""
Semantic similarity analysis using sentence transformers.

Implements semantic distance calculations for measuring behavioral consistency
and response variability as specified in CERT framework.
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from functools import lru_cache


class SemanticAnalyzer:
    """
    Semantic analysis using sentence transformers for embedding-based similarity.

    Uses pre-trained sentence-BERT models to compute semantic embeddings
    and distances between text responses.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic analyzer with a sentence transformer model.

        Args:
            model_name: HuggingFace model name. Default is fast, lightweight model.
                       Options: "all-MiniLM-L6-v2" (default, fast)
                               "all-mpnet-base-v2" (slower, more accurate)
                               "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
        """
        self.model = SentenceTransformer(model_name)
        self._embedding_cache: dict = {}

    def get_embedding(self, text: str, use_cache: bool = True) -> NDArray[np.float32]:
        """
        Get semantic embedding for text.

        Args:
            text: Input text to embed.
            use_cache: Whether to cache embeddings (reduces API costs).

        Returns:
            Normalized embedding vector.
        """
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if use_cache:
            self._embedding_cache[text] = embedding

        return embedding

    def semantic_distance(self, text1: str, text2: str) -> float:
        """
        Calculate semantic distance between two texts using cosine distance.

        Semantic distance = 1 - cosine_similarity
        Distance ∈ [0, 2], where:
        - 0 = identical semantic meaning
        - 1 = orthogonal (no similarity)
        - 2 = opposite meaning

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Semantic distance in [0, 2] range.

        Example:
            >>> analyzer = SemanticAnalyzer()
            >>> d = analyzer.semantic_distance(
            ...     "The company grew revenue by 15%",
            ...     "Revenue increased by fifteen percent"
            ... )
            >>> print(f"Distance: {d:.3f}")  # Small distance for similar meaning
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity (both embeddings are normalized)
        cosine_sim = float(np.dot(emb1, emb2))

        # Convert to distance
        distance = 1.0 - cosine_sim

        return distance

    def pairwise_distances(self, texts: List[str]) -> NDArray[np.float64]:
        """
        Compute all pairwise semantic distances for behavioral consistency.

        For n texts, computes distances d(i,j) for all i < j pairs.
        This is the input format required by behavioral_consistency() metric.

        Args:
            texts: List of n response texts.

        Returns:
            Array of n*(n-1)/2 pairwise distances for all unique pairs.

        Example:
            >>> analyzer = SemanticAnalyzer()
            >>> responses = ["Response 1", "Response 2", "Response 3"]
            >>> distances = analyzer.pairwise_distances(responses)
            >>> print(f"Computed {len(distances)} pairwise distances")
            >>> # Use with CERT metric:
            >>> from cert.core.metrics import behavioral_consistency
            >>> consistency = behavioral_consistency(distances)
        """
        n = len(texts)
        if n < 2:
            raise ValueError("Need at least 2 texts for pairwise distances")

        # Get all embeddings (cached)
        embeddings = np.array([self.get_embedding(text) for text in texts])

        # Compute pairwise distances for upper triangle (i < j)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                cosine_sim = float(np.dot(embeddings[i], embeddings[j]))
                distance = 1.0 - cosine_sim
                distances.append(distance)

        return np.array(distances)

    def batch_semantic_distance(
        self,
        reference_texts: List[str],
        comparison_texts: List[str],
    ) -> NDArray[np.float64]:
        """
        Compute semantic distances between two sets of texts.

        Useful for comparing coordinated responses against independent baselines.

        Args:
            reference_texts: Reference set of texts.
            comparison_texts: Comparison set of texts (must be same length).

        Returns:
            Array of pairwise distances between corresponding texts.

        Example:
            >>> analyzer = SemanticAnalyzer()
            >>> independent = ["Response A1", "Response A2"]
            >>> coordinated = ["Response B1", "Response B2"]
            >>> distances = analyzer.batch_semantic_distance(independent, coordinated)
        """
        if len(reference_texts) != len(comparison_texts):
            raise ValueError("Reference and comparison sets must have same length")

        ref_embeddings = np.array([self.get_embedding(t) for t in reference_texts])
        cmp_embeddings = np.array([self.get_embedding(t) for t in comparison_texts])

        # Element-wise cosine similarity
        cosine_sims = np.sum(ref_embeddings * cmp_embeddings, axis=1)

        # Convert to distances
        distances = 1.0 - cosine_sims

        return distances

    def clear_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()


# Module-level convenience functions
_default_analyzer: Optional[SemanticAnalyzer] = None


def get_default_analyzer() -> SemanticAnalyzer:
    """Get or create the default global analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SemanticAnalyzer()
    return _default_analyzer


def semantic_distance(text1: str, text2: str) -> float:
    """
    Convenience function for semantic distance using default analyzer.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Semantic distance in [0, 2] range.
    """
    analyzer = get_default_analyzer()
    return analyzer.semantic_distance(text1, text2)


def batch_semantic_distance(
    reference_texts: List[str],
    comparison_texts: List[str],
) -> NDArray[np.float64]:
    """
    Convenience function for batch semantic distance using default analyzer.

    Args:
        reference_texts: Reference set of texts.
        comparison_texts: Comparison set of texts.

    Returns:
        Array of pairwise distances.
    """
    analyzer = get_default_analyzer()
    return analyzer.batch_semantic_distance(reference_texts, comparison_texts)
