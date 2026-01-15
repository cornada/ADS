"""Disk-based embedding cache for deterministic and efficient embedding computation.

The cache is keyed by:
- model_id: identifies the encoder (e.g., "stub-encoder-v0", "sbert:all-MiniLM-L6-v2")
- sha256(text): hash of the input text

This ensures:
1. Reproducibility: same text + model = same embedding
2. Efficiency: avoid recomputation across runs
3. Versioning: different models have separate caches
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import numpy as np

from ads_core.utils.hashing import sha256_text


@dataclass
class CacheStats:
    """Statistics from a cache lookup operation."""

    hits: int = 0
    misses: int = 0
    total: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class DiskEmbeddingCache:
    """Disk-based embedding cache keyed by sha256(text) and model_id.

    Directory structure:
        root/
            <safe_model_id>/
                index.json    # {sha256_hash: row_index}
                vecs.npy      # numpy array [n, d]
                meta.json     # cache metadata (model_id, created_at, etc.)
    """

    root: Path
    model_id: str
    _last_stats: Optional[CacheStats] = field(default=None, repr=False)

    def _cache_dir(self) -> Path:
        return self.root / self._safe_model_id()

    def _index_path(self) -> Path:
        return self._cache_dir() / "index.json"

    def _vecs_path(self) -> Path:
        return self._cache_dir() / "vecs.npy"

    def _meta_path(self) -> Path:
        return self._cache_dir() / "meta.json"

    def _safe_model_id(self) -> str:
        """Convert model_id to filesystem-safe string."""
        return self.model_id.replace("/", "_").replace(":", "_").replace(" ", "_")

    @property
    def last_stats(self) -> Optional[CacheStats]:
        """Statistics from the last get_or_compute call."""
        return self._last_stats

    def exists(self) -> bool:
        """Check if cache exists for this model."""
        return self._index_path().exists() and self._vecs_path().exists()

    def load_index(self) -> Dict[str, int]:
        """Load the index mapping hash -> row index."""
        p = self._index_path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def load_vecs(self) -> Optional[np.ndarray]:
        """Load the cached vectors."""
        p = self._vecs_path()
        if not p.exists():
            return None
        return np.load(p)

    def load_meta(self) -> Dict[str, Any]:
        """Load cache metadata."""
        p = self._meta_path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    # Keep backward compatibility
    def load(self) -> Dict[str, int]:
        """Alias for load_index (backward compatibility)."""
        return self.load_index()

    def save(self, index: Dict[str, int], vecs: np.ndarray) -> None:
        """Save index and vectors to disk."""
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")
        np.save(self._vecs_path(), vecs)

        # Update metadata
        meta = self.load_meta()
        meta.update({
            "model_id": self.model_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_vectors": int(vecs.shape[0]),
            "vector_dim": int(vecs.shape[1]) if vecs.ndim > 1 else 0,
        })
        if "created_at" not in meta:
            meta["created_at"] = meta["updated_at"]

        self._meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def get_or_compute(
        self,
        texts: List[str],
        encode_fn: Callable[[List[str]], np.ndarray],
        verbose: bool = False,
    ) -> np.ndarray:
        """Get embeddings from cache or compute and cache them.

        Args:
            texts: List of texts to embed
            encode_fn: Function that takes List[str] and returns np.ndarray [n, d]
            verbose: If True, print cache stats

        Returns:
            np.ndarray of shape [len(texts), d]
        """
        index = self.load_index()
        vecs = self.load_vecs()

        needed = []
        needed_keys = []
        hit_indices: List[int] = []  # indices into texts that are cache hits

        # Identify cache hits and misses
        for i, t in enumerate(texts):
            k = sha256_text(t)
            if k in index and vecs is not None:
                hit_indices.append(i)
            else:
                needed.append(t)
                needed_keys.append(k)

        # Record stats
        stats = CacheStats(
            hits=len(hit_indices),
            misses=len(needed),
            total=len(texts),
        )
        self._last_stats = stats

        if verbose:
            print(f"[cache] model={self.model_id} hits={stats.hits} misses={stats.misses} rate={stats.hit_rate:.1%}")

        # Compute new embeddings if needed
        if needed:
            new_vecs = encode_fn(needed)

            if vecs is None:
                vecs = new_vecs
                start = 0
            else:
                start = vecs.shape[0]
                vecs = np.concatenate([vecs, new_vecs], axis=0)

            # Update index with new entries
            for i, k in enumerate(needed_keys):
                index[k] = start + i

            # Persist to disk
            self.save(index, vecs)

        # Build output in original order
        out = []
        for t in texts:
            k = sha256_text(t)
            row_idx = index[k]
            out.append(vecs[row_idx])

        return np.stack(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)

    def clear(self) -> None:
        """Remove all cached data for this model."""
        cache_dir = self._cache_dir()
        if cache_dir.exists():
            for p in cache_dir.glob("*"):
                p.unlink()
            cache_dir.rmdir()

    def get_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        meta = self.load_meta()
        index = self.load_index()
        return {
            "model_id": self.model_id,
            "cache_dir": str(self._cache_dir()),
            "exists": self.exists(),
            "num_entries": len(index),
            **meta,
        }
