from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os

from ads_core.pipeline.toy_pipeline import run_toy

router = APIRouter()

class EvaluateToyRequest(BaseModel):
    seed: int = 42
    embedding_kind: str = "stub"  # stub | sbert
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lens_mode: str = "identity"   # identity | diagonal | learned
    objectives: List[str] = ["market", "mission", "university", "learner"]
    autonomy_tau: float = 0.25

@router.post("/toy")
def evaluate_toy(req: EvaluateToyRequest):
    data_dir = Path(os.getenv("ADS_DATA_DIR", "./data"))
    run_dir = data_dir / "api_runs" / f"toy_{req.embedding_kind}_{req.lens_mode}_{req.seed}"
    embedding_cfg = {"kind": req.embedding_kind, "model_name": req.model_name}
    lenses_cfg = {"mode": req.lens_mode}
    out = run_toy(
        out_dir=run_dir,
        seed=req.seed,
        embedding_cfg=embedding_cfg,
        lenses_cfg=lenses_cfg,
        objectives=req.objectives,
        autonomy_tau=req.autonomy_tau,
        topk=5,
    )
    return {
        "run_dir": str(out.run_dir),
        "results_json": str(out.results_json),
        "pareto_json": str(out.pareto_json),
    }
