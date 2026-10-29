



import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

try:
    import cv2
except ImportError:
    sys.exit("opencv is required:  pip install opencv-python-headless")


def laplacian_var(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def estimate_snr(gray: np.ndarray) -> float:
    h, w = gray.shape
    k = max(8, min(h, w) // 16)
    corners = np.concatenate([
        gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(), gray[-k:, -k:].ravel(),
    ])
    noise = float(corners.std())
    brain = gray[gray > np.percentile(gray, 25)]
    signal = float(brain.mean()) if brain.size else float(gray.mean())
    return signal / noise if noise > 1e-6 else float("inf")


def brain_area_ratio(gray_u8: np.ndarray) -> float:
    _, th = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((th > 0).mean())


def mask_stats(mask_path: str | None, img_shape: tuple[int, int]) -> dict:
    empty = {
        "mask_exists": False, "mask_h": np.nan, "mask_w": np.nan,
        "mask_px": 0, "mask_area_ratio": 0.0, "mask_is_empty": True,
        "mask_n_components": 0, "mask_bbox_h": 0, "mask_bbox_w": 0,
        "mask_max_val": 0, "mask_shape_matches_img": False,
    }
    if not mask_path or not Path(mask_path).exists():
        return empty

    m = np.array(Image.open(mask_path).convert("L"))
    binm = (m > 127).astype(np.uint8)
    px = int(binm.sum())

    n_comp, _ = cv2.connectedComponents(binm)
    n_comp = max(0, n_comp - 1)                

    bbox_h = bbox_w = 0
    if px > 0:
        ys, xs = np.where(binm > 0)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)

    return {
        "mask_exists": True,
        "mask_h": m.shape[0], "mask_w": m.shape[1],
        "mask_px": px,
        "mask_area_ratio": px / m.size,
        "mask_is_empty": px == 0,
        "mask_n_components": n_comp,
        "mask_bbox_h": bbox_h, "mask_bbox_w": bbox_w,
        "mask_max_val": int(m.max()),
        "mask_shape_matches_img": m.shape[:2] == img_shape,
    }


def process_row(row: pd.Series) -> dict:
    img = np.array(Image.open(row["img_path"]).convert("L"))
    gray_u8 = img.astype(np.uint8)
    g = img.astype(np.float64)

    out = {
        "uid": row["uid"],
        "img_h": img.shape[0], "img_w": img.shape[1],
        "img_mean": float(g.mean()),
        "img_std": float(g.std()),
        "img_min": float(g.min()), "img_max": float(g.max()),
        "img_p1": float(np.percentile(g, 1)),
        "img_p99": float(np.percentile(g, 99)),
        "lap_var": laplacian_var(gray_u8),
        "snr_est": estimate_snr(g),
        "brain_ratio": brain_area_ratio(gray_u8),
    }
    out.update(mask_stats(row.get("seg_mask_path"), img.shape[:2]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N images (quick test)")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = single-threaded. Larger number = parallel")
    args = ap.parse_args()

    cfg.set_seed()

    idx_path = cfg.DIR_INDEX / "index.csv"
    if not idx_path.exists():
        sys.exit("index.csv not found. Run 01_build_index.py first.")
    df = pd.read_csv(idx_path)
    if args.limit:
        df = df.head(args.limit)

    cfg.banner(f"Processing {len(df)} images")

    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        rows = list(df.to_dict("records"))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            recs = list(tqdm(ex.map(process_row, [pd.Series(r) for r in rows]),
                             total=len(rows)))
    else:
        recs = [process_row(r) for _, r in tqdm(df.iterrows(), total=len(df))]

    stats = pd.DataFrame(recs)
    merged = df.merge(stats, on="uid", how="left")

    tumor = merged[merged["class_name"] != "no_tumor"]
    if len(tumor):
        q = tumor["mask_area_ratio"].quantile([0.25, 0.5, 0.75]).values
        def size_bin(r):
            if r <= 0:
                return "no_lesion"
            if r < q[0]:
                return "very_small"
            if r < q[1]:
                return "small"
            if r < q[2]:
                return "medium"
            return "large"
        merged["lesion_size_group"] = merged["mask_area_ratio"].apply(size_bin)
    else:
        merged["lesion_size_group"] = "unknown"

    for col, name in (("lap_var", "sharpness_group"), ("snr_est", "snr_group")):
        try:
            merged[name] = pd.qcut(
                merged[col].replace([np.inf, -np.inf], np.nan),
                3, labels=["low", "mid", "high"], duplicates="drop",
            ).astype(str)
        except Exception:
            merged[name] = "unknown"

    out = cfg.DIR_INDEX / "image_stats.csv"
    merged.to_csv(out, index=False)

    cfg.banner("Image sizes")
    print(merged.groupby(["img_h", "img_w"]).size().to_string())

    cfg.banner("Masks")
    print(f"  mask present      : {int(merged['mask_exists'].sum())}")
    print(f"  mask empty        : {int(merged['mask_is_empty'].sum())}")
    print(f"  mask size ≠ image size : "
          f"{int((~merged['mask_shape_matches_img'] & merged['mask_exists']).sum())}")

    print("\n  Empty masks by class:")
    print(merged.groupby("class_name")["mask_is_empty"].agg(["sum", "count"]).to_string())
    print("\n  Expected: all no_tumor empty, the rest non-empty.")
    print("  If a tumor class has empty masks, note those samples —")
    print("  they must be reported explicitly in the paper.")

    cfg.banner("Lesion area by class (fraction of the whole image)")
    print(
        merged[merged["class_name"] != "no_tumor"]
        .groupby("class_name")["mask_area_ratio"]
        .describe(percentiles=[0.05, 0.5, 0.95])
        .to_string()
    )

    cfg.banner("Lesion area by plane")
    print(
        merged[merged["class_name"] != "no_tumor"]
        .groupby("plane_name")["mask_area_ratio"]
        .agg(["count", "mean", "median"])
        .to_string()
    )

    cfg.banner("Derived subgroups")
    for c in ("lesion_size_group", "sharpness_group", "snr_group"):
        print(f"\n{c}:")
        print(merged[c].value_counts().to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        t = merged[merged["mask_area_ratio"] > 0]
        for c in cfg.CLASSES[:-1]:
            s = t[t["class_name"] == c]["mask_area_ratio"]
            if len(s):
                ax[0, 0].hist(s, bins=50, alpha=0.55, label=c)
        ax[0, 0].set_title("Lesion area ratio by class")
        ax[0, 0].set_xlabel("mask area / image area")
        ax[0, 0].legend()

        ax[0, 1].hist(merged["lap_var"].dropna(), bins=60, color="steelblue")
        ax[0, 1].set_title("Laplacian variance (sharpness proxy)")
        ax[0, 1].set_xlabel("var")

        snr = merged["snr_est"].replace([np.inf, -np.inf], np.nan).dropna()
        ax[1, 0].hist(snr, bins=60, color="seagreen")
        ax[1, 0].set_title("Estimated SNR")

        ct = pd.crosstab(merged["class_name"], merged["plane_name"])
        ct.plot(kind="bar", stacked=True, ax=ax[1, 1])
        ax[1, 1].set_title("Class x plane")
        ax[1, 1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        p = cfg.DIR_FIGS / "audit_overview.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"\nFigure saved → {p}")
    except ImportError:
        print("\nmatplotlib is not installed, figures skipped.")

    cfg.banner("Done")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
