"""Analyze the unified didactic space in the latent embedding layer.

Questions:
1. Do courses from different universities occupy the same latent space?
2. Do courses and occupations form meaningful clusters?
3. Is there a unified structure — or disconnected islands per institution?
4. Does multimodal metadata correlate with latent position?

Uses: SBERT embeddings (all-MiniLM-L6-v2) on canonical dataset text.

Output: experiments/reports/latent_space/analysis.json + figures
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CANONICAL_PATH = PROJECT_ROOT / "data" / "canonical" / "ads_mm_canonical.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "reports" / "latent_space"
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_canonical() -> List[Dict]:
    """Load canonical dataset."""
    records = []
    with open(CANONICAL_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def embed_texts(texts: List[str], batch_size: int = 256) -> np.ndarray:
    """Embed texts with SBERT (MiniLM)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings)
    except ImportError:
        logger.warning("sentence-transformers not available, using TF-IDF fallback")
        return _tfidf_embed(texts)


def _tfidf_embed(texts: List[str], dim: int = 128) -> np.ndarray:
    """Fallback: TF-IDF + SVD."""
    from collections import Counter
    import math

    def tokenize(t):
        return [w.lower() for w in t.split() if len(w) > 2 and w.isalpha()]

    docs = [tokenize(t) for t in texts]
    df = Counter()
    for doc in docs:
        for w in set(doc):
            df[w] += 1

    vocab = [w for w, _ in df.most_common(5000)]
    w2i = {w: i for i, w in enumerate(vocab)}
    n = len(docs)

    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        for w, c in tf.items():
            if w in w2i:
                matrix[i, w2i[w]] = c * math.log(n / (1 + df[w]))

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms

    # SVD reduction
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :dim] * S[:dim]


def compute_inter_intra_distances(embeddings: np.ndarray, labels: List[str]) -> Dict:
    """Compute inter-cluster and intra-cluster distances."""
    unique_labels = sorted(set(labels))
    label_indices = defaultdict(list)
    for i, lab in enumerate(labels):
        label_indices[lab].append(i)

    # Centroids
    centroids = {}
    for lab in unique_labels:
        idx = label_indices[lab]
        centroids[lab] = embeddings[idx].mean(axis=0)

    # Intra-cluster: mean distance to centroid
    intra = {}
    for lab in unique_labels:
        idx = label_indices[lab]
        if len(idx) < 2:
            continue
        dists = np.linalg.norm(embeddings[idx] - centroids[lab], axis=1)
        intra[lab] = float(np.mean(dists))

    # Inter-cluster: pairwise centroid distances
    inter = {}
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1:]:
            d = float(np.linalg.norm(centroids[l1] - centroids[l2]))
            inter[f"{l1}↔{l2}"] = d

    return {"intra": intra, "inter": inter, "centroids": {k: v.tolist() for k, v in centroids.items()}}


def analyze_space(records: List[Dict], embeddings: np.ndarray) -> Dict:
    """Run all latent space analyses."""
    results = {}
    n = len(records)

    # === 1. Institution clustering ===
    logger.info("Analysis 1: Institution clustering")
    institutions = [r.get("institution", "UNKNOWN") for r in records]
    inst_dist = compute_inter_intra_distances(embeddings, institutions)

    # Is there a unified space or islands?
    inter_vals = list(inst_dist["inter"].values())
    intra_vals = list(inst_dist["intra"].values())
    mean_inter = np.mean(inter_vals) if inter_vals else 0
    mean_intra = np.mean(intra_vals) if intra_vals else 0
    separation_ratio = mean_inter / max(mean_intra, 1e-6)

    results["institution_clustering"] = {
        "mean_intra_distance": round(mean_intra, 4),
        "mean_inter_distance": round(mean_inter, 4),
        "separation_ratio": round(separation_ratio, 4),
        "interpretation": (
            "UNIFIED SPACE (courses overlap across institutions)"
            if separation_ratio < 2.0
            else "SEPARATED ISLANDS (institutions form distinct clusters)"
        ),
        "intra_distances": inst_dist["intra"],
        "top_inter_distances": dict(sorted(inst_dist["inter"].items(), key=lambda x: x[1])[:5]),
        "bottom_inter_distances": dict(sorted(inst_dist["inter"].items(), key=lambda x: -x[1])[:5]),
    }

    # === 2. Type clustering (COURSE vs JOB_ROLE vs MISSION) ===
    logger.info("Analysis 2: Type clustering")
    types = [r.get("type", "?") for r in records]
    type_dist = compute_inter_intra_distances(embeddings, types)

    results["type_clustering"] = {
        "intra": type_dist["intra"],
        "inter": type_dist["inter"],
        "interpretation": "How separated are courses, jobs, and missions in latent space?",
    }

    # === 3. Cross-national nearest neighbors ===
    logger.info("Analysis 3: Cross-national nearest neighbors")
    # For 100 random MIT courses, find nearest neighbors — are they from MIT or other universities?
    mit_idx = [i for i, r in enumerate(records) if r.get("institution") == "MIT" and r.get("type") == "COURSE"]
    rng = np.random.RandomState(42)
    sample_mit = rng.choice(mit_idx, size=min(200, len(mit_idx)), replace=False)

    cross_national_hits = Counter()
    for i in sample_mit:
        dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
        dists[i] = float("inf")
        nn_idx = np.argsort(dists)[:10]  # top 10 neighbors
        for j in nn_idx:
            cross_national_hits[records[j].get("institution", "?")] += 1

    total_hits = sum(cross_national_hits.values())
    results["cross_national_nn"] = {
        "sample_size": len(sample_mit),
        "k": 10,
        "nn_institution_distribution": {k: round(v / total_hits, 3) for k, v in cross_national_hits.most_common()},
        "interpretation": "What fraction of MIT's nearest neighbors come from other institutions?",
    }

    # === 4. Course-Job alignment ===
    logger.info("Analysis 4: Course-Job alignment")
    course_idx = [i for i, r in enumerate(records) if r.get("type") == "COURSE"]
    job_idx = [i for i, r in enumerate(records) if r.get("type") == "JOB_ROLE"]

    if course_idx and job_idx:
        course_centroid = embeddings[course_idx].mean(axis=0)
        job_centroid = embeddings[job_idx].mean(axis=0)
        course_job_dist = float(np.linalg.norm(course_centroid - job_centroid))

        # Per-institution course centroids vs job centroid
        inst_job_alignment = {}
        for inst in set(institutions):
            inst_course_idx = [i for i in course_idx if records[i].get("institution") == inst]
            if len(inst_course_idx) < 10:
                continue
            inst_centroid = embeddings[inst_course_idx].mean(axis=0)
            dist = float(np.linalg.norm(inst_centroid - job_centroid))
            inst_job_alignment[inst] = round(dist, 4)

        results["course_job_alignment"] = {
            "overall_course_job_distance": round(course_job_dist, 4),
            "per_institution": dict(sorted(inst_job_alignment.items(), key=lambda x: x[1])),
            "interpretation": "Distance between course and job centroids in latent space",
        }

    # === 5. Multimodal correlation ===
    logger.info("Analysis 5: Multimodal position correlation")
    mm_idx = [i for i, r in enumerate(records) if r.get("multimodal", {}).get("ocw_slug")]
    if mm_idx:
        mm_emb = embeddings[mm_idx]
        richness = np.array([records[i].get("multimodal", {}).get("resource_richness", 0) for i in mm_idx])
        has_video = np.array([int(records[i].get("multimodal", {}).get("has_video", False)) for i in mm_idx])

        # PCA of multimodal subset — does richness correlate with principal components?
        mm_centered = mm_emb - mm_emb.mean(axis=0)
        U, S, Vt = np.linalg.svd(mm_centered, full_matrices=False)

        # Correlation of richness with first 5 PCs
        pc_richness_corr = {}
        for k in range(min(5, U.shape[1])):
            corr = float(np.corrcoef(U[:, k], richness)[0, 1])
            pc_richness_corr[f"PC{k+1}"] = round(corr, 4)

        results["multimodal_correlation"] = {
            "n_courses_with_multimodal": len(mm_idx),
            "pc_richness_correlation": pc_richness_corr,
            "video_centroid_shift": round(float(np.linalg.norm(
                mm_emb[has_video == 1].mean(axis=0) - mm_emb[has_video == 0].mean(axis=0)
            )), 4) if has_video.sum() > 0 else None,
            "interpretation": "Does latent position predict multimedia availability?",
        }

    # === 6. MISIS activity correlation ===
    logger.info("Analysis 6: MISIS activity in latent space")
    misis_idx = [i for i, r in enumerate(records)
                 if r.get("activity", {}).get("hours") and r.get("type") == "COURSE"]
    if misis_idx:
        misis_emb = embeddings[misis_idx]
        has_labs = np.array([int(records[i]["activity"].get("has_labs", False)) for i in misis_idx])

        if has_labs.sum() > 0 and has_labs.sum() < len(has_labs):
            lab_shift = float(np.linalg.norm(
                misis_emb[has_labs == 1].mean(axis=0) - misis_emb[has_labs == 0].mean(axis=0)
            ))
        else:
            lab_shift = None

        results["misis_activity"] = {
            "n_courses_with_activity": len(misis_idx),
            "lab_centroid_shift": round(lab_shift, 4) if lab_shift else None,
            "interpretation": "Do courses with labs occupy different regions than lecture-only?",
        }

    return results


def generate_figures(records: List[Dict], embeddings: np.ndarray) -> None:
    """Generate UMAP/PCA visualization figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # PCA for speed (UMAP optional)
    logger.info("Computing PCA projection...")
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca2d = U[:, :2] * S[:2]

    institutions = [r.get("institution", "?") for r in records]
    types = [r.get("type", "?") for r in records]

    # === Figure 1: By institution ===
    fig, ax = plt.subplots(figsize=(12, 8))
    inst_colors = {}
    cmap = plt.cm.tab20
    unique_insts = sorted(set(institutions))
    for i, inst in enumerate(unique_insts):
        inst_colors[inst] = cmap(i / max(len(unique_insts) - 1, 1))

    for inst in unique_insts:
        idx = [i for i, x in enumerate(institutions) if x == inst]
        ax.scatter(pca2d[idx, 0], pca2d[idx, 1], c=[inst_colors[inst]],
                   label=f"{inst} ({len(idx)})", alpha=0.3, s=5)

    ax.legend(fontsize=8, markerscale=3, loc="upper right")
    ax.set_title("ADS Latent Space: PCA by Institution", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "latent_by_institution.pdf", dpi=200)
    plt.close()
    logger.info("Saved latent_by_institution.pdf")

    # === Figure 2: By type (COURSE / JOB_ROLE / MISSION) ===
    fig, ax = plt.subplots(figsize=(10, 7))
    type_colors = {"COURSE": "#3498DB", "JOB_ROLE": "#E74C3C", "MISSION": "#2ECC71", "COMPETENCY": "#F39C12"}
    for atype in ["COURSE", "JOB_ROLE", "MISSION", "COMPETENCY"]:
        idx = [i for i, t in enumerate(types) if t == atype]
        if idx:
            ax.scatter(pca2d[idx, 0], pca2d[idx, 1],
                       c=type_colors.get(atype, "#999"),
                       label=f"{atype} ({len(idx)})", alpha=0.4, s=8 if atype == "COURSE" else 20)

    ax.legend(fontsize=10, markerscale=2)
    ax.set_title("ADS Latent Space: Courses vs Jobs vs Missions", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "latent_by_type.pdf", dpi=200)
    plt.close()
    logger.info("Saved latent_by_type.pdf")

    # === Figure 3: Multimodal overlay (MIT only) ===
    mit_idx = [i for i, r in enumerate(records)
               if r.get("institution") == "MIT" and r.get("type") == "COURSE"]
    if mit_idx:
        fig, ax = plt.subplots(figsize=(10, 7))
        richness = [records[i].get("multimodal", {}).get("resource_richness", 0) for i in mit_idx]
        sc = ax.scatter(pca2d[mit_idx, 0], pca2d[mit_idx, 1],
                        c=richness, cmap="YlOrRd", alpha=0.6, s=15, vmin=0, vmax=4)
        fig.colorbar(sc, ax=ax, label="Resource Richness (0=text, 4=multimedia)")
        ax.set_title("MIT Courses: Latent Position vs Multimedia Richness", fontsize=13)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        fig.savefig(FIGURE_DIR / "latent_mit_multimodal.pdf", dpi=200)
        plt.close()
        logger.info("Saved latent_mit_multimodal.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    records = load_canonical()
    logger.info("Loaded %d canonical records", len(records))

    # Filter to records with text
    records = [r for r in records if r.get("text", "").strip()]
    logger.info("Records with text: %d", len(records))

    # Stratified subsample for speed if too large
    import os
    max_records = int(os.environ.get("ADS_MAX_RECORDS", "0")) or len(records)
    if max_records < len(records):
        rng = np.random.RandomState(42)
        # Stratified: keep proportional representation per institution
        from collections import defaultdict
        by_inst = defaultdict(list)
        for i, r in enumerate(records):
            by_inst[r.get("institution", "?")].append(i)
        sampled_idx = []
        for inst, indices in by_inst.items():
            n_sample = max(1, int(len(indices) * max_records / len(records)))
            sampled_idx.extend(rng.choice(indices, size=min(n_sample, len(indices)), replace=False).tolist())
        rng.shuffle(sampled_idx)
        records = [records[i] for i in sampled_idx[:max_records]]
        logger.info("Subsampled to %d records (stratified)", len(records))

    # Embed
    texts = [r.get("text", "")[:512] for r in records]  # Truncate for speed
    logger.info("Embedding %d texts...", len(texts))
    embeddings = embed_texts(texts)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Analyze
    results = analyze_space(records, embeddings)

    # Save results
    out_path = OUTPUT_DIR / "analysis.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    # Generate figures
    generate_figures(records, embeddings)

    # Print summary
    print("\n" + "=" * 70)
    print("LATENT SPACE ANALYSIS SUMMARY")
    print("=" * 70)

    ic = results.get("institution_clustering", {})
    print(f"\n1. INSTITUTION CLUSTERING:")
    print(f"   Separation ratio: {ic.get('separation_ratio', '?')}")
    print(f"   → {ic.get('interpretation', '?')}")

    tc = results.get("type_clustering", {})
    print(f"\n2. TYPE CLUSTERING (inter-distances):")
    for k, v in tc.get("inter", {}).items():
        print(f"   {k}: {v:.4f}")

    cn = results.get("cross_national_nn", {})
    print(f"\n3. CROSS-NATIONAL NEAREST NEIGHBORS (MIT sample, k=10):")
    for k, v in cn.get("nn_institution_distribution", {}).items():
        print(f"   {k}: {v:.1%}")

    cj = results.get("course_job_alignment", {})
    print(f"\n4. COURSE-JOB ALIGNMENT:")
    print(f"   Overall distance: {cj.get('overall_course_job_distance', '?')}")
    for k, v in list(cj.get("per_institution", {}).items())[:5]:
        print(f"   {k}: {v}")

    mc = results.get("multimodal_correlation", {})
    print(f"\n5. MULTIMODAL CORRELATION:")
    print(f"   Video centroid shift: {mc.get('video_centroid_shift', '?')}")
    print(f"   PC-richness correlation: {mc.get('pc_richness_correlation', {})}")

    ma = results.get("misis_activity", {})
    print(f"\n6. MISIS ACTIVITY:")
    print(f"   Lab centroid shift: {ma.get('lab_centroid_shift', '?')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
