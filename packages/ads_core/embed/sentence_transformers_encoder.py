"""SentenceTransformers-based encoder for semantic embeddings.

This encoder uses the sentence-transformers library for high-quality text embeddings.
Install with: pip install -e ".[embed]"

Model versioning:
- model_id includes the model name for cache separation
- For full reproducibility, pin the model version via local path or revision parameter
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from ads_core.embed.encoder import Encoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class SentenceTransformersEncoder(Encoder):
    """Encoder using SentenceTransformers library.

    Attributes:
        model_name: HuggingFace model name or local path
        device: Device to run on (None for auto, "cpu", "cuda", etc.)
        normalize: Whether to L2-normalize embeddings (default True)

    Example:
        >>> encoder = SentenceTransformersEncoder("all-MiniLM-L6-v2")
        >>> vecs = encoder.encode(["Hello world", "Test sentence"])
        >>> vecs.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """Initialize the encoder.

        Args:
            model_name: Model name from HuggingFace or local path.
                       Common choices:
                       - "all-MiniLM-L6-v2" (fast, 384d)
                       - "all-mpnet-base-v2" (better quality, 768d)
                       - "multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)
            device: Device to use. None for auto-detection.
            normalize: L2-normalize embeddings (recommended for cosine similarity).
            batch_size: Batch size for encoding large text lists.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install -e '.[embed]' or pip install sentence-transformers"
            )

        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name_or_path=model_name, device=device)

        # Get model dimension
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def model_id(self) -> str:
        """Unique identifier for this model configuration.

        Format: sbert:<model_name>
        The model_name is sanitized for filesystem use in the cache.
        """
        return f"sbert:{self._model_name}"

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            np.ndarray of shape [len(texts), dim] with float32 dtype.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        show_bar = len(texts) > 100
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_bar,
            batch_size=self._batch_size,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (convenience method).

        Args:
            text: String to encode.

        Returns:
            np.ndarray of shape [dim] with float32 dtype.
        """
        return self.encode([text])[0]


def get_available_models() -> List[str]:
    """Get list of recommended SentenceTransformer models.

    Returns:
        List of model names that work well for ADS.
    """
    return [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality (384d)
        "sentence-transformers/all-mpnet-base-v2",  # Best quality (768d)
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # Q&A optimized
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # Paraphrase detection
    ]
