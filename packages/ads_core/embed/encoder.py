from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Encoder(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return array [n, d]."""
        raise NotImplementedError


class StubEncoder(Encoder):
    """Deterministic tiny encoder for smoke tests (no ML deps)."""

    def __init__(self, d: int = 64, model_id: str = "stub-encoder-v0"):
        self._d = d
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._d), dtype=np.float32)
        for i, t in enumerate(texts):
            bb = t.encode("utf-8")
            for j, ch in enumerate(bb):
                out[i, j % self._d] += ((ch % 31) - 15) / 15.0
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out / norms
