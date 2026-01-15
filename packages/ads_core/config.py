from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ADSConfig:
    seed: int
    dataset_id: str
    embedding_model_id: str
    objectives: List[str]
    lens_mode: str
    autonomy_tau: float
    out_dir: str
    extra: Dict[str, Any]
