from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ads_core.eval.distances import cosine_similarity


def topk_by_similarity(query_v: np.ndarray, items: Dict[str, np.ndarray], k: int = 20) -> List[Tuple[str, float]]:
    scores = []
    for item_id, v in items.items():
        scores.append((item_id, cosine_similarity(query_v, v)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
