from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
from ads_core.eval.pareto import pareto_front

router = APIRouter()

class ParetoRequest(BaseModel):
    items: List[Dict[str, float]]
    keys: List[str]

@router.post("/front")
def compute_pareto(req: ParetoRequest):
    idx = pareto_front(req.items, req.keys)
    return {"pareto_indices": idx}
