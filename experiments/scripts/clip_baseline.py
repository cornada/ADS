"""CLIP / SBERT multimodal baseline for ADS-MM cross-modal prediction.

Upgrades over the TF-IDF baseline in crossmodal_baselines.py:
  - Sentence-transformer embeddings (all-MiniLM-L6-v2) instead of TF-IDF
  - If images exist: extracts PIL-based visual features (size, aspect, color stats)
  - sklearn classifiers with class_weight='balanced' to handle class imbalance
  - Fallback: TF-IDF + TruncatedSVD if sentence_transformers unavailable

Tasks:
  1. Richness prediction (Ridge regression, MAE + Spearman rho)
  2. Per-label classification (LogReg balanced, per-type F1 + macro F1)

Usage:
    cd /Volumes/CORNADA_D/ADS
    PYTHONPATH=packages .venv/bin/python experiments/scripts/clip_baseline.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Feature flags: detect available libraries
# ---------------------------------------------------------------------------

HAS_SBERT = False
HAS_SKLEARN = False
HAS_PIL = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    pass

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        mean_absolute_error,
        f1_score,
        precision_score,
        recall_score,
    )
    HAS_SKLEARN = True
except ImportError:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data loading (reuses logic from crossmodal_baselines.py)
# ---------------------------------------------------------------------------

def load_multimodal_data(
    multimodal_path: Path,
    artifacts_path: Path,
) -> List[Dict[str, Any]]:
    """Load and join multimodal metadata with ADS course text."""
    mm_by_aid: Dict[str, Dict] = {}
    with open(multimodal_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("ads_matched") and obj.get("ads_artifact_id"):
                mm_by_aid[obj["ads_artifact_id"]] = obj

    logger.info("Loaded %d multimodal annotations (ADS-matched)", len(mm_by_aid))

    joined: List[Dict[str, Any]] = []
    with open(artifacts_path, encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            aid = art.get("artifact_id", "")
            if aid in mm_by_aid:
                mm = mm_by_aid[aid]
                joined.append({
                    "artifact_id": aid,
                    "text": art.get("text", ""),
                    "title": art.get("title", ""),
                    "department": art.get("metadata", {}).get("department", ""),
                    "slug": mm.get("slug", ""),
                    "has_video": mm["has_video"],
                    "has_notes": mm["has_notes"],
                    "has_interactive": mm["has_interactive"],
                    "has_assessment": mm["has_assessment"],
                    "resource_richness": mm["resource_richness"],
                    "resource_type_count": mm["resource_type_count"],
                    "topic_count": mm.get("topic_count", 0),
                    "level": mm.get("level", ""),
                })

    logger.info("Joined %d courses with multimodal annotations", len(joined))
    return joined


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def encode_texts_sbert(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode texts with sentence-transformers."""
    logger.info("Encoding %d texts with SBERT (%s)...", len(texts), model_name)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    logger.info("  Done in %.1fs, shape=%s", time.time() - t0, embeddings.shape)
    return np.array(embeddings, dtype=np.float32)


def encode_texts_tfidf_svd(texts: List[str], n_components: int = 128) -> np.ndarray:
    """Fallback: TF-IDF + TruncatedSVD dimensionality reduction."""
    logger.info("Encoding %d texts with TF-IDF + SVD (d=%d)...", len(texts), n_components)
    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        X_tfidf = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(n_components, X_tfidf.shape[1] - 1), random_state=42)
        X_reduced = svd.fit_transform(X_tfidf)
        logger.info("  SVD explained variance: %.3f", svd.explained_variance_ratio_.sum())
        return X_reduced.astype(np.float32)
    else:
        # Ultra-fallback: manual TF-IDF (from crossmodal_baselines.py pattern)
        raise RuntimeError("sklearn is required for TF-IDF+SVD fallback")


def extract_image_features(
    image_dir: Path,
    slugs: List[str],
) -> Optional[np.ndarray]:
    """Extract basic visual features from course cover images using PIL.

    Features per image (11-dim):
      - width, height, aspect_ratio (3)
      - mean R, G, B (3)
      - std R, G, B (3)
      - brightness (1)
      - is_color (binary, 1)
    """
    if not HAS_PIL:
        logger.warning("PIL not available, skipping image features")
        return None

    if not image_dir.exists():
        logger.warning("Image directory %s does not exist", image_dir)
        return None

    available_images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    logger.info("Found %d images in %s", len(available_images), image_dir)

    n_features = 11
    features = np.zeros((len(slugs), n_features), dtype=np.float32)
    found_count = 0

    for i, slug in enumerate(slugs):
        img_path = available_images.get(slug)
        if img_path is None:
            continue
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                arr = np.array(img_rgb, dtype=np.float32) / 255.0

                features[i, 0] = w / 1000.0  # normalized width
                features[i, 1] = h / 1000.0  # normalized height
                features[i, 2] = w / max(h, 1)  # aspect ratio

                # Color stats per channel
                for c in range(3):
                    features[i, 3 + c] = arr[:, :, c].mean()
                    features[i, 6 + c] = arr[:, :, c].std()

                # Brightness (luminance)
                features[i, 9] = 0.299 * arr[:, :, 0].mean() + 0.587 * arr[:, :, 1].mean() + 0.114 * arr[:, :, 2].mean()

                # Color vs grayscale
                channel_std = np.std([arr[:, :, c].mean() for c in range(3)])
                features[i, 10] = 1.0 if channel_std > 0.05 else 0.0

                found_count += 1
        except Exception as e:
            logger.debug("Error reading %s: %s", img_path, e)

    logger.info("Extracted image features for %d / %d courses", found_count, len(slugs))

    if found_count < 10:
        logger.warning("Too few images matched (%d). Skipping image features.", found_count)
        return None

    return features


# ---------------------------------------------------------------------------
# Evaluation helpers (manual fallback if sklearn unavailable)
# ---------------------------------------------------------------------------

def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    d = ra - rb
    n = len(a)
    if n < 3:
        return 0.0
    return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1)))


def eval_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """Per-label precision/recall/F1 + macro F1."""
    results: Dict[str, float] = {}
    for j, name in enumerate(label_names):
        tp = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 1)))
        fp = int(np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 1)))
        fn = int(np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[f"{name}_precision"] = round(prec, 3)
        results[f"{name}_recall"] = round(rec, 3)
        results[f"{name}_f1"] = round(f1, 3)

    f1s = [results[f"{name}_f1"] for name in label_names]
    results["macro_f1"] = round(float(np.mean(f1s)), 3)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_sbert_baseline(
    data: List[Dict[str, Any]],
    image_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run SBERT-based (or TF-IDF+SVD) cross-modal baseline."""
    rng = np.random.RandomState(seed)
    n = len(data)

    # Prepare text
    texts = [d["text"] + " " + d.get("title", "") for d in data]
    slugs = [d.get("slug", "") for d in data]
    label_names = ["video", "notes", "interactive", "assessment"]

    y_labels = np.array([
        [d["has_video"], d["has_notes"], d["has_interactive"], d["has_assessment"]]
        for d in data
    ], dtype=np.float64)

    y_richness = np.array([d["resource_richness"] for d in data], dtype=np.float64)

    # --- Text encoding ---
    encoder_name = "unknown"
    if HAS_SBERT:
        X_text = encode_texts_sbert(texts)
        encoder_name = "sbert/all-MiniLM-L6-v2"
    elif HAS_SKLEARN:
        X_text = encode_texts_tfidf_svd(texts, n_components=128)
        encoder_name = "tfidf+svd(128)"
    else:
        raise RuntimeError("Need either sentence_transformers or sklearn installed")

    # --- Image features ---
    X_img = None
    if image_dir is not None:
        X_img = extract_image_features(image_dir, slugs)

    # --- Concatenate features ---
    if X_img is not None:
        X = np.hstack([X_text, X_img])
        feature_desc = f"{encoder_name} + image_pil(11)"
        logger.info("Combined feature matrix: %s", X.shape)
    else:
        X = X_text
        feature_desc = encoder_name

    # --- Train/test split (80/20, stratified-ish by richness quartile) ---
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train_labels = y_labels[train_idx]
    y_test_labels = y_labels[test_idx]
    y_train_rich = y_richness[train_idx]
    y_test_rich = y_richness[test_idx]

    logger.info("Train: %d, Test: %d, Features: %d", len(train_idx), len(test_idx), X.shape[1])

    results: Dict[str, Any] = {}

    # ============================================================
    # Task 1: Richness prediction (Ridge regression)
    # NOTE: SBERT embeddings are already unit-normed; StandardScaler hurts.
    # ============================================================
    logger.info("=== Task 1: Richness prediction ===")

    if HAS_SKLEARN:
        # Ridge on raw embeddings (no scaling -- SBERT is pre-normalized)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train_rich)
        y_pred_rich = ridge.predict(X_test)
        mae = float(mean_absolute_error(y_test_rich, y_pred_rich))
    else:
        # Manual ridge
        lam = 1.0
        w = np.linalg.solve(
            X_train.T @ X_train + lam * np.eye(X_train.shape[1]),
            X_train.T @ y_train_rich,
        )
        y_pred_rich = X_test @ w
        mae = float(np.mean(np.abs(y_test_rich - y_pred_rich)))

    rho = spearman_rho(y_test_rich, y_pred_rich)
    results["task1_richness"] = {
        "sbert_ridge": {"mae": round(mae, 3), "spearman": round(rho, 3)},
    }
    logger.info("  SBERT Ridge: MAE=%.3f, rho=%.3f", mae, rho)

    # ============================================================
    # Task 2: Per-label classification (balanced LogReg)
    # ============================================================
    logger.info("=== Task 2: Resource type classification ===")

    if HAS_SKLEARN:
        # Classification benefits from scaling (LogReg convergence)
        scaler_cls = StandardScaler()
        X_train_s2 = scaler_cls.fit_transform(X_train)
        X_test_s2 = scaler_cls.transform(X_test)

        y_pred_cls = np.zeros_like(y_test_labels)
        per_label_detail: Dict[str, Dict[str, float]] = {}

        for j, name in enumerate(label_names):
            n_pos = int(y_train_labels[:, j].sum())
            n_neg = len(y_train_labels) - n_pos

            if n_pos == 0:
                # No positive examples -- predict all negative
                logger.warning("  Label '%s': 0 positive examples in train, skipping", name)
                y_pred_cls[:, j] = 0
                continue

            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X_train_s2, y_train_labels[:, j])
            y_pred_cls[:, j] = clf.predict(X_test_s2)

            # Report class balance
            logger.info(
                "  Label '%s': train %d pos / %d neg (%.1f%%), test pred %d pos",
                name, n_pos, n_neg, 100 * n_pos / len(y_train_labels),
                int(y_pred_cls[:, j].sum()),
            )

        balanced_cls_results = eval_classification(y_test_labels, y_pred_cls, label_names)

        # Also run unbalanced for comparison
        y_pred_cls_ub = np.zeros_like(y_test_labels)
        for j, name in enumerate(label_names):
            n_pos = int(y_train_labels[:, j].sum())
            if n_pos == 0:
                y_pred_cls_ub[:, j] = 0
                continue
            clf_ub = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=seed,
            )
            clf_ub.fit(X_train_s2, y_train_labels[:, j])
            y_pred_cls_ub[:, j] = clf_ub.predict(X_test_s2)

        unbalanced_cls_results = eval_classification(y_test_labels, y_pred_cls_ub, label_names)

    else:
        # Manual logistic regression fallback (from crossmodal_baselines.py)
        y_pred_cls = np.zeros_like(y_test_labels)
        for j in range(4):
            logits = X_train @ np.linalg.lstsq(X_train, y_train_labels[:, j], rcond=None)[0]
            test_logits = X_test @ np.linalg.lstsq(X_train, y_train_labels[:, j], rcond=None)[0]
            y_pred_cls[:, j] = (test_logits > 0.5).astype(float)

        balanced_cls_results = eval_classification(y_test_labels, y_pred_cls, label_names)
        unbalanced_cls_results = balanced_cls_results  # no distinction without sklearn

    results["task2_classification"] = {
        "sbert_logreg_balanced": balanced_cls_results,
        "sbert_logreg_unbalanced": unbalanced_cls_results,
    }

    logger.info("  Balanced LogReg:   macro_F1=%.3f", balanced_cls_results["macro_f1"])
    logger.info("  Unbalanced LogReg: macro_F1=%.3f", unbalanced_cls_results["macro_f1"])

    # ============================================================
    # Dataset stats + metadata
    # ============================================================
    results["dataset_stats"] = {
        "total_courses": n,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "video_count": int(np.sum(y_labels[:, 0])),
        "video_pct": round(float(np.mean(y_labels[:, 0]) * 100), 1),
        "notes_count": int(np.sum(y_labels[:, 1])),
        "notes_pct": round(float(np.mean(y_labels[:, 1]) * 100), 1),
        "interactive_count": int(np.sum(y_labels[:, 2])),
        "interactive_pct": round(float(np.mean(y_labels[:, 2]) * 100), 1),
        "assessment_count": int(np.sum(y_labels[:, 3])),
        "assessment_pct": round(float(np.mean(y_labels[:, 3]) * 100), 1),
        "mean_richness": round(float(np.mean(y_richness)), 2),
        "std_richness": round(float(np.std(y_richness)), 2),
    }

    results["config"] = {
        "encoder": encoder_name,
        "feature_desc": feature_desc,
        "feature_dim": X.shape[1],
        "has_image_features": X_img is not None,
        "image_count": int((X_img.sum(axis=1) != 0).sum()) if X_img is not None else 0,
        "seed": seed,
        "has_sklearn": HAS_SKLEARN,
        "has_sbert": HAS_SBERT,
        "has_pil": HAS_PIL,
    }

    return results


def main() -> None:
    multimodal_path = PROJECT_ROOT / "data" / "multimodal" / "ocw_video_metadata.jsonl"
    artifacts_path = PROJECT_ROOT / "data" / "processed" / "unified_v4" / "artifacts.jsonl"
    image_dir = PROJECT_ROOT / "data" / "multimodal" / "images"

    logger.info("Libraries: SBERT=%s, sklearn=%s, PIL=%s", HAS_SBERT, HAS_SKLEARN, HAS_PIL)

    if not multimodal_path.exists():
        logger.error("Multimodal data not found: %s", multimodal_path)
        logger.error("Run: python scripts/fetch_ocw_multimodal.py")
        return

    if not artifacts_path.exists():
        logger.error("Artifacts not found: %s", artifacts_path)
        return

    data = load_multimodal_data(multimodal_path, artifacts_path)
    if len(data) < 50:
        logger.error("Too few records (%d). Check data pipeline.", len(data))
        return

    # Check image availability
    use_images = image_dir.exists() and any(image_dir.iterdir())
    if use_images:
        logger.info("Image directory found with files: %s", image_dir)
    else:
        logger.info("No images available, running text-only baseline")
        image_dir = None

    results = run_sbert_baseline(data, image_dir=image_dir if use_images else None)

    # Save
    output_dir = PROJECT_ROOT / "experiments" / "reports" / "crossmodal"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sbert_baseline_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("RESULTS SAVED: %s", output_path)
    logger.info("=" * 70)

    # Print comparison with TF-IDF baseline
    tfidf_path = output_dir / "baseline_results.json"
    if tfidf_path.exists():
        tfidf = json.loads(tfidf_path.read_text())
        logger.info("")
        logger.info("COMPARISON vs TF-IDF baseline:")
        logger.info("  Richness MAE:  TF-IDF=%.3f  ->  SBERT=%.3f  (delta=%.3f)",
                     tfidf["task1_richness"]["tfidf_ridge"]["mae"],
                     results["task1_richness"]["sbert_ridge"]["mae"],
                     results["task1_richness"]["sbert_ridge"]["mae"] - tfidf["task1_richness"]["tfidf_ridge"]["mae"])
        logger.info("  Richness rho:  TF-IDF=%.3f  ->  SBERT=%.3f  (delta=+%.3f)",
                     tfidf["task1_richness"]["tfidf_ridge"]["spearman"],
                     results["task1_richness"]["sbert_ridge"]["spearman"],
                     results["task1_richness"]["sbert_ridge"]["spearman"] - tfidf["task1_richness"]["tfidf_ridge"]["spearman"])
        logger.info("  Class macro_F1: TF-IDF=%.3f  ->  SBERT balanced=%.3f  (delta=+%.3f)",
                     tfidf["task2_classification"]["tfidf_logreg"]["macro_f1"],
                     results["task2_classification"]["sbert_logreg_balanced"]["macro_f1"],
                     results["task2_classification"]["sbert_logreg_balanced"]["macro_f1"] - tfidf["task2_classification"]["tfidf_logreg"]["macro_f1"])

    logger.info("")
    logger.info("Config: %s", json.dumps(results["config"], indent=2))


if __name__ == "__main__":
    main()
