#!/usr/bin/env python3
"""
Fetch OCW course cover images and generate image statistics + grid figure.

Usage:
    python scripts/fetch_ocw_images.py --max-courses 10   # test run
    python scripts/fetch_ocw_images.py                     # full run (~1700 images)
    python scripts/fetch_ocw_images.py --stats-only        # skip download, just stats + grid
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Volumes/CORNADA_D/ADS")
METADATA = BASE / "data" / "multimodal" / "ocw_video_metadata.jsonl"
IMAGE_DIR = BASE / "data" / "multimodal" / "images"
MANIFEST = BASE / "data" / "multimodal" / "ocw_image_manifest.jsonl"
GRID_OUT = BASE / "ACM_MM_Dataset" / "Sources" / "figures" / "fig5_image_grid.pdf"

OCW_BASE = "https://ocw.mit.edu"
DELAY = 0.3  # seconds between requests

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def load_metadata(path: Path, max_courses: int | None = None):
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("image_src"):
                records.append(rec)
            if max_courses and len(records) >= max_courses:
                break
    return records


def download_images(records: list[dict], image_dir: Path, manifest_path: Path):
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for i, rec in enumerate(records):
        slug = rec["ocw_slug"]
        image_src = rec["image_src"]
        ext = Path(image_src).suffix or ".jpg"
        out_path = image_dir / f"{slug}{ext}"

        entry = {
            "ocw_slug": slug,
            "image_path": str(out_path),
            "image_size_bytes": 0,
            "download_success": False,
        }

        if out_path.exists() and out_path.stat().st_size > 0:
            entry["image_size_bytes"] = out_path.stat().st_size
            entry["download_success"] = True
            manifest_entries.append(entry)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(records)}] (cached) {slug}")
            continue

        url = OCW_BASE + image_src
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ADS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            entry["image_size_bytes"] = len(data)
            entry["download_success"] = True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  FAIL [{i+1}] {slug}: {exc}")

        manifest_entries.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] downloaded")

        time.sleep(DELAY)

    # Write manifest
    with open(manifest_path, "w") as f:
        for e in manifest_entries:
            f.write(json.dumps(e) + "\n")

    ok = sum(1 for e in manifest_entries if e["download_success"])
    print(f"\nDone: {ok}/{len(records)} images downloaded.")
    print(f"Manifest: {manifest_path}")
    return manifest_entries


# ---------------------------------------------------------------------------
# Statistics + grid figure
# ---------------------------------------------------------------------------

def compute_stats_and_grid(manifest_path: Path, grid_out: Path):
    """Compute image stats and generate a 4x5 grid of sample images."""
    try:
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: Pillow and matplotlib required for stats/grid.")
        print("  pip install Pillow matplotlib")
        return

    entries = []
    with open(manifest_path) as f:
        for line in f:
            e = json.loads(line)
            if e["download_success"]:
                entries.append(e)

    if not entries:
        print("No successful downloads in manifest.")
        return

    # --- Basic statistics ---
    sizes = [e["image_size_bytes"] for e in entries]
    mean_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    exts = {}
    for e in entries:
        ext = Path(e["image_path"]).suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1

    print(f"\n--- Image Statistics ---")
    print(f"  Total images: {len(entries)}")
    print(f"  Mean size:    {mean_size/1024:.1f} KB")
    print(f"  Min size:     {min_size/1024:.1f} KB")
    print(f"  Max size:     {max_size/1024:.1f} KB")
    print(f"  Total size:   {sum(sizes)/1024/1024:.1f} MB")
    print(f"  Format dist:  {exts}")

    # --- Grid of 20 sample images ---
    import random
    random.seed(42)
    sample = random.sample(entries, min(20, len(entries)))

    rows, cols = 4, 5
    if len(sample) < 20:
        cols = min(len(sample), 5)
        rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 9.6))
    fig.suptitle("MIT OCW Course Cover Images (Sample)", fontsize=14, fontweight="bold", y=0.98)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.axis("off")
        if idx < len(sample):
            img_path = sample[idx]["image_path"]
            slug = sample[idx]["ocw_slug"]
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                # Short label: course number
                short = slug.split("-")[0] + "-" + slug.split("-")[1] if "-" in slug else slug[:20]
                ax.set_title(short, fontsize=7, pad=2)
            except Exception as exc:
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    grid_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(grid_out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrid saved: {grid_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch OCW course cover images")
    parser.add_argument("--max-courses", type=int, default=None,
                        help="Limit number of courses to download (default: all)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip download, only compute stats and grid from existing manifest")
    args = parser.parse_args()

    if not args.stats_only:
        print(f"Loading metadata from {METADATA} ...")
        records = load_metadata(METADATA, args.max_courses)
        print(f"Courses with images: {len(records)}")
        print(f"Downloading to {IMAGE_DIR} ...\n")
        download_images(records, IMAGE_DIR, MANIFEST)

    if MANIFEST.exists():
        compute_stats_and_grid(MANIFEST, GRID_OUT)
    else:
        print(f"No manifest at {MANIFEST} -- run download first.")


if __name__ == "__main__":
    main()
