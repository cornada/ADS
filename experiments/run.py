from __future__ import annotations

from pathlib import Path
import os
import json
from datetime import datetime, timezone

# Python 3.14 compatibility fix for Hydra's LazyCompletionHelp
# See: https://github.com/facebookresearch/hydra/issues/2876
import sys
if sys.version_info >= (3, 14):
    import argparse

    _original_check_help = argparse.ArgumentParser._check_help

    def _patched_check_help(self, action):
        """Patched to handle Hydra's LazyCompletionHelp which lacks __contains__."""
        if action.help is not None and not isinstance(action.help, str):
            # Hydra's LazyCompletionHelp doesn't implement __contains__, skip validation
            return
        return _original_check_help(self, action)

    argparse.ArgumentParser._check_help = _patched_check_help

import hydra
from omegaconf import DictConfig, OmegaConf

from ads_core.pipeline.toy_pipeline import run_toy
from ads_core.pipeline.dataset_pipeline import run_dataset
from ads_core.report.pareto_report import build_pareto_report


def _resolve_out_dir(cfg: DictConfig) -> Path:
    base = Path(cfg.outputs.base_dir)
    # Create a readable run id (Hydra also sets an output dir; we keep ours inside base_dir)
    run_id = cfg.outputs.run_id
    if run_id == "auto":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base / run_id


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Persist resolved config
    out_dir = _resolve_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    # Get strict_data from config (default False for backward compatibility)
    strict_data = bool(cfg.dataset.get("strict_data", False))

    (out_dir / "meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hydra_cwd": os.getcwd(),
        "strict_data": strict_data,
    }, indent=2), encoding="utf-8")

    if cfg.dataset.id == "toy":
        # Get stability config (optional)
        stability_runs = int(cfg.get("stability", {}).get("n_runs", 0))
        stability_noise = float(cfg.get("stability", {}).get("noise_scale", 0.01))

        outputs = run_toy(
            out_dir=out_dir,
            seed=int(cfg.seed),
            embedding_cfg=dict(cfg.embedding),
            lenses_cfg=dict(cfg.lenses),
            objectives=list(cfg.objectives.enabled),
            autonomy_tau=float(cfg.constraints.autonomy_drift_tau),
            topk=int(cfg.planning.topk),
            stability_runs=stability_runs,
            stability_noise=stability_noise,
        )
        artifacts = build_pareto_report(outputs.run_dir)
        print("[OK] Run dir:", outputs.run_dir)
        print("[OK] Paper artifacts:", {k: str(v) for k, v in artifacts.items()})
        if outputs.ethics_json:
            print("[OK] Ethics report:", outputs.ethics_json)
        if outputs.stability_json:
            print("[OK] Stability report:", outputs.stability_json)
    elif cfg.dataset.id in ("mit", "ucb", "asu"):
        # Use generic dataset pipeline for real datasets
        data_dir = Path(cfg.dataset.get("data_dir", ".")) if cfg.dataset.get("data_dir") else None

        outputs = run_dataset(
            dataset_id=cfg.dataset.id,
            out_dir=out_dir,
            seed=int(cfg.seed),
            embedding_cfg=dict(cfg.embedding),
            lenses_cfg=dict(cfg.lenses),
            objectives=list(cfg.objectives.enabled),
            autonomy_tau=float(cfg.constraints.autonomy_drift_tau),
            data_dir=data_dir,
            strict_data=strict_data,
        )
        artifacts = build_pareto_report(outputs.run_dir)
        print(f"[OK] Dataset: {outputs.dataset_id}")
        print(f"[OK] Artifacts: {outputs.artifact_count}")
        print(f"[OK] Pareto options: {outputs.pareto_count}")
        print(f"[OK] Run dir: {outputs.run_dir}")
        print(f"[OK] Paper artifacts: {dict((k, str(v)) for k, v in artifacts.items())}")
        if outputs.is_synthetic:
            print("[WARNING] Data is SYNTHETIC (from fixtures). Use +profile=paper for real data.")
    else:
        raise NotImplementedError(f"Unknown dataset: {cfg.dataset.id}. Supported: toy, mit, ucb, asu")


if __name__ == "__main__":
    main()
