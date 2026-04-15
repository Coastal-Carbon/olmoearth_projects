"""Few-shot mangrove classification from OlmoEarth embeddings.

Samples a small number of labeled pixels (e.g., 20 per class) from ESA
WorldCover within a single region (Ca Mau, Vietnam) and trains a logistic
regression to classify every pixel. Demonstrates that rich embeddings
enable accurate wall-to-wall maps from very few labels.

Generates a 1x3 figure: S2 RGB (with training dots) | WorldCover ref | predicted.

Example:
    $ python -m segmentation.analyze \
        --data-dir data/segmentation/ca_mau \
        --out figures/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from common.embedding_utils import load_embeddings
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WORLDCOVER_MANGROVE = 95
WORLDCOVER_WATER = 80

CLASS_COLORS = {
    0: np.array([0.75, 0.70, 0.55]),  # other
    1: np.array([0.09, 0.55, 0.35]),  # mangrove
    2: np.array([0.20, 0.35, 0.75]),  # water
}


def _reproject_band(
    src_path: Path,
    match: rasterio.DatasetReader,
    band: int = 1,
) -> np.ndarray:
    """Reproject a single band to the grid of *match*."""
    with rasterio.open(src_path) as ref:
        dst = np.zeros((match.height, match.width), dtype=np.float32)
        reproject(
            source=rasterio.band(ref, band),
            destination=dst,
            src_transform=ref.transform,
            src_crs=ref.crs,
            dst_transform=match.transform,
            dst_crs=match.crs,
            resampling=Resampling.nearest,
        )
    return dst


def _load_s2_rgb(
    path: Path,
    match_ds: rasterio.DatasetReader | None = None,
) -> np.ndarray | None:
    """Load S2 RGB and stretch to [0, 1] for display."""
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        if match_ds is not None and ds.shape != (match_ds.height, match_ds.width):
            rgb = np.zeros((3, match_ds.height, match_ds.width), dtype=np.float32)
            for b in range(1, 4):
                reproject(
                    source=rasterio.band(ds, b),
                    destination=rgb[b - 1],
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=match_ds.transform,
                    dst_crs=match_ds.crs,
                    resampling=Resampling.bilinear,
                )
        else:
            rgb = ds.read([1, 2, 3]).astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)
    nans = np.isnan(rgb).any(axis=-1) | (rgb.sum(axis=-1) == 0)
    lo, hi = np.nanpercentile(rgb[~nans], [2, 98])
    rgb = np.clip((rgb - lo) / max(float(hi - lo), 1e-6), 0, 1)
    rgb[nans] = 0.15
    return rgb


def _find_embeddings(directory: Path) -> Path:
    """Locate the embedding .tif in *directory* (may be nested from ZIP extraction)."""
    candidates = list(directory.rglob("*.tif"))
    embed_candidates = [
        p for p in candidates if "embed" in p.stem.lower() or "result" in p.stem.lower()
    ]
    if embed_candidates:
        return embed_candidates[0]
    tifs = [p for p in candidates if p.stem not in ("s2_rgb", "worldcover")]
    if tifs:
        return tifs[0]
    raise FileNotFoundError(f"No embedding .tif found in {directory}")


def _wc_to_rgb(wc: np.ndarray) -> np.ndarray:
    """WorldCover labels to display RGB."""
    wc_int = np.round(wc).astype(np.int64)
    display = np.full((*wc.shape, 3), 0.12, dtype=np.float32)
    display[wc_int == WORLDCOVER_MANGROVE] = CLASS_COLORS[1]
    display[wc_int == WORLDCOVER_WATER] = CLASS_COLORS[2]
    other = (
        (wc_int != 0) & (wc_int != WORLDCOVER_MANGROVE) & (wc_int != WORLDCOVER_WATER)
    )
    display[other] = CLASS_COLORS[0]
    return display


def _class_to_rgb(class_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Predicted class map to display RGB."""
    display = np.full((*class_map.shape, 3), 0.12, dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        display[class_map == cls] = color
    display[~valid] = [0.12, 0.12, 0.12]
    return display


# ---------------------------------------------------------------------------
# Public API (used by both CLI and notebooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotResult:
    """Intermediate results from :func:`train_fewshot`."""

    class_map: np.ndarray
    valid: np.ndarray
    wc: np.ndarray
    f1: float
    miou: float
    train_rows: np.ndarray
    train_cols: np.ndarray
    train_labels: np.ndarray
    n_per_class: int
    s2_rgb: np.ndarray | None


def train_fewshot(
    data_dir: Path,
    n_per_class: int = 20,
    seed: int = 42,
) -> FewShotResult:
    """Train a few-shot classifier on a single region.

    Returns a :class:`FewShotResult` with the dense prediction and all
    arrays needed for figure generation.
    """
    embed_path = _find_embeddings(data_dir)
    print(f"Embeddings: {embed_path}")

    print("Loading embeddings...")
    embed, valid, ds = load_embeddings(embed_path)
    c, h, w = embed.shape
    print(f"  {c} bands, {h}x{w}, valid: {valid.sum():,}")

    print("Loading WorldCover...")
    wc = _reproject_band(data_dir / "worldcover.tif", ds)
    wc_int = np.round(wc).astype(np.int64)
    ref_ok = valid & (wc_int != 0)

    labels = np.zeros_like(wc_int)
    labels[wc_int == WORLDCOVER_MANGROVE] = 1
    labels[wc_int == WORLDCOVER_WATER] = 2

    m_idx = np.flatnonzero(((wc_int == WORLDCOVER_MANGROVE) & ref_ok).ravel())
    w_idx = np.flatnonzero(((wc_int == WORLDCOVER_WATER) & ref_ok).ravel())
    o_idx = np.flatnonzero(
        (
            ref_ok & (wc_int != WORLDCOVER_MANGROVE) & (wc_int != WORLDCOVER_WATER)
        ).ravel()
    )
    print(f"  mangrove={m_idx.size:,}, water={w_idx.size:,}, other={o_idx.size:,}")

    rng = np.random.default_rng(seed)
    n = n_per_class
    print(f"Sampling {n} pixels per class ({3 * n} total)...")

    m_sample = rng.choice(m_idx, size=min(n, m_idx.size), replace=False)
    w_sample = rng.choice(w_idx, size=min(n, w_idx.size), replace=False)
    o_sample = rng.choice(o_idx, size=min(n, o_idx.size), replace=False)

    train_idx = np.concatenate([o_sample, m_sample, w_sample])
    train_labels = np.concatenate(
        [
            np.zeros(len(o_sample), dtype=np.int64),
            np.ones(len(m_sample), dtype=np.int64),
            np.full(len(w_sample), 2, dtype=np.int64),
        ]
    )
    train_rows = train_idx // w
    train_cols = train_idx % w

    flat = embed.reshape(c, -1).T

    print("Training logistic regression...")
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000, solver="lbfgs", class_weight="balanced"
                ),
            ),
        ]
    )
    clf.fit(flat[train_idx], train_labels)

    print("Dense prediction...")
    valid_idx = np.flatnonzero(valid.ravel())
    class_flat = np.full(h * w, 255, dtype=np.uint8)
    chunk = 100_000
    for i in range(0, valid_idx.size, chunk):
        sl = valid_idx[i : i + chunk]
        class_flat[sl] = clf.predict(flat[sl]).astype(np.uint8)
    class_map = class_flat.reshape(h, w)

    eval_mask = ref_ok.ravel().copy()
    eval_mask[train_idx] = False
    eval_idx = np.flatnonzero(eval_mask)

    y_true = labels.ravel()[eval_idx]
    y_pred = class_map.ravel().astype(np.int64)[eval_idx]

    f1 = f1_score(y_true, y_pred, average="weighted")
    miou = jaccard_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    print(f"\nEval ({eval_idx.size:,} px, excluding {len(train_idx)} training px):")
    print(f"  accuracy={acc:.3f}  F1={f1:.3f}  mIoU={miou:.3f}")

    ds.close()

    s2_rgb = _load_s2_rgb(data_dir / "s2_rgb.tif")

    return FewShotResult(
        class_map=class_map,
        valid=valid,
        wc=wc,
        f1=f1,
        miou=miou,
        train_rows=train_rows,
        train_cols=train_cols,
        train_labels=train_labels,
        n_per_class=n,
        s2_rgb=s2_rgb,
    )


def make_fewshot_figure(result: FewShotResult) -> Figure:
    """Create a 1x3 few-shot classification figure.

    Returns the matplotlib Figure (caller decides whether to save or show).
    """
    total = 3 * result.n_per_class
    has_s2 = result.s2_rgb is not None
    ncols = 3 if has_s2 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.06)

    col = 0
    if has_s2:
        axes[col].imshow(result.s2_rgb, interpolation="nearest")
        axes[col].scatter(
            result.train_cols,
            result.train_rows,
            c="#FF00FF",
            s=12,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
        axes[col].set_title(
            f"Sentinel-2 RGB ({total} labeled pixels)",
            fontsize=13,
            fontweight="bold",
            pad=6,
        )
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        col += 1

    axes[col].imshow(_wc_to_rgb(result.wc), interpolation="nearest")
    axes[col].set_title("WorldCover reference", fontsize=13, fontweight="bold", pad=6)
    axes[col].set_xticks([])
    axes[col].set_yticks([])
    col += 1

    pred_rgb = _class_to_rgb(result.class_map, result.valid)
    axes[col].imshow(pred_rgb, interpolation="nearest")
    axes[col].set_title(
        f"Predicted (F1 {result.f1:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=6,
    )
    axes[col].set_xticks([])
    axes[col].set_yticks([])

    legend_patches = [
        Patch(facecolor=CLASS_COLORS[1], label="Mangrove"),
        Patch(facecolor=CLASS_COLORS[2], label="Water"),
        Patch(facecolor=CLASS_COLORS[0], label="Other"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f"Few-shot classification from {total} labeled pixels",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with region data (embeddings.tif, worldcover.tif, s2_rgb.tif).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=20,
        help="Number of labeled pixels per class (default: 20, i.e., 60 total).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    """Train few-shot classifier and save the figure."""
    args = parse_args()
    result = train_fewshot(args.data_dir, args.n_per_class, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    fig = make_fewshot_figure(result)
    total = 3 * args.n_per_class
    fig_path = args.out / f"fewshot_{total}labels.png"
    fig.savefig(
        fig_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
    )
    plt.close(fig)
    print(f"Wrote {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
