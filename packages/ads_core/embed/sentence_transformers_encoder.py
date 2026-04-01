"""SentenceTransformers-based encoder for semantic embeddings.

This encoder uses the sentence-transformers library for high-quality text embeddings.
Install with: pip install -e ".[embed]"

Model versioning:
- model_id includes the model name for cache separation
- For full reproducibility, pin the model version via local path or revision parameter

MPS (Apple Silicon) support:
- Large models (E5-large, BGE-large) can deadlock on MPS with large batches
- Chunked encoding (chunk_size parameter) prevents Metal GPU timeout
"""
from __future__ import annotations

import gc
import logging
import sys
from typing import List, Optional

import numpy as np

from ads_core.embed.encoder import Encoder

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


def _detect_device(requested: Optional[str]) -> str:
    """Detect best available device, with MPS safety checks."""
    if requested is not None:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class SentenceTransformersEncoder(Encoder):
    """Encoder using SentenceTransformers library.

    Supports chunked encoding to avoid MPS/GPU memory deadlocks with large models.

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
        chunk_size: int = 5000,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install -e '.[embed]' or pip install sentence-transformers"
            )

        self._model_name = model_name
        self._normalize = normalize
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._device = _detect_device(device)

        # Reduce batch size on MPS for large models to avoid deadlock
        if self._device == "mps" and any(k in model_name.lower() for k in ("large", "e5", "bge")):
            self._batch_size = min(self._batch_size, 16)
            logger.info("MPS + large model detected, reducing batch_size to %d", self._batch_size)

        self._model = SentenceTransformer(model_name_or_path=model_name, device=self._device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def model_id(self) -> str:
        return f"sbert:{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings with chunked processing.

        For large text lists, encodes in chunks of `chunk_size` to prevent
        MPS/GPU memory deadlocks. Between chunks, forces garbage collection.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        n = len(texts)
        if n <= self._chunk_size:
            return self._encode_batch(texts)

        # Chunked encoding for large inputs
        logger.info("Chunked encoding: %d texts in %d-text chunks", n, self._chunk_size)
        parts = []
        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)
            chunk_num = start // self._chunk_size + 1
            total_chunks = (n + self._chunk_size - 1) // self._chunk_size
            logger.info("Encoding chunk %d/%d (%d texts)", chunk_num, total_chunks, end - start)
            part = self._encode_batch(texts[start:end])
            parts.append(part)
            # Force memory cleanup between chunks (critical for MPS)
            gc.collect()
            if self._device == "mps":
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass

        return np.vstack(parts)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a single batch/chunk of texts."""
        show_bar = len(texts) > 100
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=show_bar,
            batch_size=self._batch_size,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
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
