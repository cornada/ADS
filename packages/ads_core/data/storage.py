from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from datetime import datetime
import json

from ads_core.data.schemas import Artifact, EmbeddingRecord


@dataclass
class LocalArtifactStore:
    root: Path

    def artifacts_path(self) -> Path:
        return self.root / "artifacts.jsonl"

    def embeddings_path(self) -> Path:
        return self.root / "embeddings.jsonl"

    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def write_meta(self, meta: Dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path().write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_artifacts(self, artifacts: Iterable[Artifact]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.artifacts_path().open("a", encoding="utf-8") as f:
            for a in artifacts:
                f.write(a.model_dump_json() + "\n")

    def load_artifacts(self) -> List[Artifact]:
        p = self.artifacts_path()
        if not p.exists():
            return []
        out: List[Artifact] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(Artifact.model_validate_json(line))
        return out

    def save_embeddings(self, records: Iterable[EmbeddingRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.embeddings_path().open("a", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

    def load_embeddings(self, model_id: Optional[str] = None) -> List[EmbeddingRecord]:
        p = self.embeddings_path()
        if not p.exists():
            return []
        out: List[EmbeddingRecord] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = EmbeddingRecord.model_validate_json(line)
                if model_id is None or rec.model_id == model_id:
                    out.append(rec)
        return out
