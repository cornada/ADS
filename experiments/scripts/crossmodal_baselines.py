"""Cross-modal baselines for ADS-MM evaluation tasks.

Evaluates text → multimedia resource prediction:
- Task 1: Richness prediction (regression, MAE + Spearman rho)
- Task 2: Resource type classification (multi-label, per-type F1)
- Task 3: Cross-institutional transfer (train MIT, predict others)

Usage:
    python experiments/scripts/crossmodal_baselines.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_multimodal_data(
    multimodal_path: Path,
    artifacts_path: Path,
) -> List[Dict[str, Any]]:
    """Load and join multimodal metadata with ADS course text."""
    # Load multimodal annotations
    mm_by_aid: Dict[str, Dict] = {}
    with open(multimodal_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("ads_matched") and obj.get("ads_artifact_id"):
                mm_by_aid[obj["ads_artifact_id"]] = obj

    logger.info("Loaded %d multimodal annotations (ADS-matched)", len(mm_by_aid))

    # Load ADS artifacts and join
    joined = []
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
                    "has_video": mm["has_video"],
                    "has_notes": mm["has_notes"],
                    "has_interactive": mm["has_interactive"],
                    "has_assessment": mm["has_assessment"],
                    "resource_richness": mm["resource_richness"],
                    "resource_type_count": mm["resource_type_count"],
                    "learning_resource_types": mm["learning_resource_types"],
                    "topic_count": mm.get("topic_count", 0),
                    "level": mm.get("level", ""),
                })

    logger.info("Joined %d courses with multimodal annotations", len(joined))
    return joined


def tfidf_features(texts: List[str], max_features: int = 5000) -> np.ndarray:
    """Simple TF-IDF feature extraction (no sklearn dependency)."""
    from collections import Counter
    import math

    # Tokenize
    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in text.split() if len(w) > 2 and w.isalpha()]

    docs = [tokenize(t) for t in texts]
    n_docs = len(docs)

    # Document frequency
    df: Counter = Counter()
    for doc in docs:
        for word in set(doc):
            df[word] += 1

    # Select top features by DF
    vocab = [w for w, _ in df.most_common(max_features)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # TF-IDF matrix
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        for word, count in tf.items():
            if word in word_to_idx:
                j = word_to_idx[word]
                idf = math.log(n_docs / (1 + df[word]))
                matrix[i, j] = count * idf

    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    return matrix


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """Evaluate multi-label classification per label."""
    results = {}
    for i, name in enumerate(label_names):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[f"{name}_precision"] = round(precision, 3)
        results[f"{name}_recall"] = round(recall, 3)
        results[f"{name}_f1"] = round(f1, 3)

    # Macro F1
    f1s = [results[f"{name}_f1"] for name in label_names]
    results["macro_f1"] = round(np.mean(f1s), 3)

    return results


def logistic_regression_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lr: float = 0.1,
    epochs: int = 100,
) -> np.ndarray:
    """Simple logistic regression (no sklearn dependency)."""
    n_features = X_train.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        logits = X_train @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad_w = X_train.T @ (probs - y_train) / len(y_train)
        grad_b = np.mean(probs - y_train)
        w -= lr * grad_w
        b -= lr * grad_b

    test_logits = X_test @ w + b
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30, 30)))
    return test_probs


def run_baselines(data: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    """Run all cross-modal baselines."""
    rng = np.random.RandomState(seed)
    n = len(data)

    # Prepare features and labels
    texts = [d["text"] + " " + d.get("title", "") for d in data]
    label_names = ["video", "notes", "interactive", "assessment"]

    y_labels = np.array([
        [d["has_video"], d["has_notes"], d["has_interactive"], d["has_assessment"]]
        for d in data
    ], dtype=np.float64)

    y_richness = np.array([d["resource_richness"] for d in data], dtype=np.float64)

    # Train/test split (80/20)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    logger.info("Train: %d, Test: %d", len(train_idx), len(test_idx))

    # TF-IDF features
    logger.info("Computing TF-IDF features...")
    X = tfidf_features(texts, max_features=5000)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_labels = y_labels[train_idx]
    y_test_labels = y_labels[test_idx]
    y_train_rich = y_richness[train_idx]
    y_test_rich = y_richness[test_idx]

    results = {}

    # === Task 1: Richness prediction ===
    logger.info("Task 1: Richness prediction...")

    # Baseline: department mean
    depts = [d["department"] for d in data]
    dept_means: Dict[str, float] = {}
    for idx in train_idx:
        dept = depts[idx]
        if dept not in dept_means:
            dept_means[dept] = []
        dept_means[dept].append(y_richness[idx])
    dept_means = {k: np.mean(v) for k, v in dept_means.items()}
    global_mean = np.mean(y_train_rich)

    y_pred_dept = np.array([dept_means.get(depts[i], global_mean) for i in test_idx])
    mae_dept = np.mean(np.abs(y_test_rich - y_pred_dept))

    # TF-IDF + ridge regression (simple closed-form)
    lam = 1.0
    w_ridge = np.linalg.solve(
        X_train.T @ X_train + lam * np.eye(X_train.shape[1]),
        X_train.T @ y_train_rich,
    )
    y_pred_tfidf = X_test @ w_ridge
    mae_tfidf = np.mean(np.abs(y_test_rich - y_pred_tfidf))

    # Spearman correlation
    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        d = ra - rb
        n = len(a)
        return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))

    rho_dept = spearman(y_test_rich, y_pred_dept)
    rho_tfidf = spearman(y_test_rich, y_pred_tfidf)

    results["task1_richness"] = {
        "dept_prior": {"mae": round(mae_dept, 3), "spearman": round(rho_dept, 3)},
        "tfidf_ridge": {"mae": round(mae_tfidf, 3), "spearman": round(rho_tfidf, 3)},
    }
    logger.info("  Dept prior: MAE=%.3f, rho=%.3f", mae_dept, rho_dept)
    logger.info("  TF-IDF ridge: MAE=%.3f, rho=%.3f", mae_tfidf, rho_tfidf)

    # === Task 2: Resource type classification ===
    logger.info("Task 2: Resource type classification...")

    # Majority baseline
    y_pred_majority = np.zeros_like(y_test_labels)
    for j in range(4):
        majority = int(np.mean(y_train_labels[:, j]) > 0.5)
        y_pred_majority[:, j] = majority

    majority_results = evaluate_classification(y_test_labels, y_pred_majority, label_names)

    # TF-IDF + logistic regression (per label)
    y_pred_tfidf_cls = np.zeros_like(y_test_labels)
    for j in range(4):
        probs = logistic_regression_fit(X_train, y_train_labels[:, j], X_test)
        y_pred_tfidf_cls[:, j] = (probs > 0.5).astype(float)

    tfidf_cls_results = evaluate_classification(y_test_labels, y_pred_tfidf_cls, label_names)

    results["task2_classification"] = {
        "majority": majority_results,
        "tfidf_logreg": tfidf_cls_results,
    }
    logger.info("  Majority: macro_F1=%.3f", majority_results["macro_f1"])
    logger.info("  TF-IDF LogReg: macro_F1=%.3f", tfidf_cls_results["macro_f1"])

    # === Dataset statistics ===
    results["dataset_stats"] = {
        "total_courses": n,
        "video_count": int(np.sum(y_labels[:, 0])),
        "video_pct": round(np.mean(y_labels[:, 0]) * 100, 1),
        "notes_count": int(np.sum(y_labels[:, 1])),
        "notes_pct": round(np.mean(y_labels[:, 1]) * 100, 1),
        "interactive_count": int(np.sum(y_labels[:, 2])),
        "interactive_pct": round(np.mean(y_labels[:, 2]) * 100, 1),
        "assessment_count": int(np.sum(y_labels[:, 3])),
        "assessment_pct": round(np.mean(y_labels[:, 3]) * 100, 1),
        "mean_richness": round(float(np.mean(y_richness)), 2),
        "std_richness": round(float(np.std(y_richness)), 2),
    }

    return results


def main() -> None:
    multimodal_path = PROJECT_ROOT / "data" / "multimodal" / "ocw_video_metadata.jsonl"
    artifacts_path = PROJECT_ROOT / "data" / "processed" / "unified_v4" / "artifacts.jsonl"

    if not multimodal_path.exists():
        logger.error("Multimodal data not found at %s", multimodal_path)
        logger.error("Run: python scripts/fetch_ocw_multimodal.py")
        return

    data = load_multimodal_data(multimodal_path, artifacts_path)
    if len(data) < 50:
        logger.error("Not enough joined data (%d records). Check data collection.", len(data))
        return

    results = run_baselines(data)

    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "reports" / "crossmodal"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Results saved to %s", output_path)
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Dataset: %d courses with multimodal annotations", results["dataset_stats"]["total_courses"])
    logger.info("  Video: %d (%.1f%%)", results["dataset_stats"]["video_count"], results["dataset_stats"]["video_pct"])
    logger.info("  Notes: %d (%.1f%%)", results["dataset_stats"]["notes_count"], results["dataset_stats"]["notes_pct"])
    logger.info("  Mean richness: %.2f ± %.2f", results["dataset_stats"]["mean_richness"], results["dataset_stats"]["std_richness"])
    logger.info("  Task 1 (richness): TF-IDF MAE=%.3f, rho=%.3f",
                results["task1_richness"]["tfidf_ridge"]["mae"],
                results["task1_richness"]["tfidf_ridge"]["spearman"])
    logger.info("  Task 2 (classification): TF-IDF macro_F1=%.3f",
                results["task2_classification"]["tfidf_logreg"]["macro_f1"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
