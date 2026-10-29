


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    p = cfg.DIR_EMBED / "dino_embeddings.npy"
    if not p.exists():
        sys.exit("Embeddings not found. Run 03_fingerprints.py --stage embed.")
    E = np.load(p)
    U = np.load(cfg.DIR_EMBED / "dino_uids.npy", allow_pickle=True)
    order = pd.Series(range(len(U)), index=U)
    E = E[df["uid"].map(order).to_numpy()]
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def condensed_distance(E: np.ndarray) -> np.ndarray:
    S = E @ E.T
    np.clip(S, -1, 1, out=S)
    D = np.maximum(1.0 - S, 0.0)
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def summarize(df: pd.DataFrame, labels: np.ndarray) -> dict:
    d = df.assign(g=labels)
    g = d.groupby("g").agg(
        n=("uid", "size"),
        n_split=("official_split", "nunique"),
        n_class=("class_name", "nunique"),
        n_plane=("plane_name", "nunique"),
    )
    cross = g[g.n_split == 2]
    leaked = d[d.g.isin(cross.index) & (d.official_split == "test")]
    n_test = max((d.official_split == "test").sum(), 1)
    return {
        "n_clusters": len(g),
        "max_cluster": int(g.n.max()),
        "mean_cluster": round(float(g.n.mean()), 4),
        "multi_clusters": int((g.n > 1).sum()),
        "cross_split_clusters": len(cross),
        "leaked_test": len(leaked),
        "leaked_test_pct": round(100 * len(leaked) / n_test, 2),
        "label_conflict_clusters": int((g.n_class > 1).sum()),
        "multi_plane_clusters": int((g.n_plane > 1).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine similarity threshold (from the 05c output)")
    ap.add_argument("--linkage", default="complete",
                    choices=["complete", "average", "single", "ward"],
                    help="linkage method. Default complete (no chaining)")
    ap.add_argument("--max-study-slices", type=int, default=5)
    ap.add_argument("--sample-for-review", type=int, default=200)
    args = ap.parse_args()

    cfg.set_seed()
    df = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv").reset_index(drop=True)
    E = load_embeddings(df)
    print(f"{len(df)} images | embedding dim {E.shape[1]}")

    cfg.banner("1/5  Distance matrix")
    cd = condensed_distance(E)

    cfg.banner("2/5  Linkage method sensitivity analysis")
    thr_probe = args.threshold if args.threshold else 0.98026
    sens = []
    for meth in ("complete", "average", "single"):
        Zm = linkage(cd, method=meth)
        lab = fcluster(Zm, 1 - thr_probe, criterion="distance")
        s = summarize(df, lab)
        s["linkage"] = meth
        s["threshold"] = thr_probe
        s["respects_doc_max"] = s["max_cluster"] <= args.max_study_slices
        sens.append(s)
    sdf = pd.DataFrame(sens)[
        ["linkage", "threshold", "n_clusters", "max_cluster", "mean_cluster",
         "multi_clusters", "leaked_test", "leaked_test_pct", "respects_doc_max"]]
    sdf.to_csv(cfg.DIR_GROUPS / "linkage_sensitivity.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"\n  → Include this table in the Supplementary. It shows that the choice of"
          f" complete was not arbitrary.")

    Z = linkage(cd, method=args.linkage)
    print(f"\n  Selected method: {args.linkage}")

    cfg.banner("3/5  Threshold calibration")
    grid = [0.999, 0.995, 0.99, 0.985, 0.98026, 0.98, 0.975, 0.97, 0.96, 0.95]
    rows = []
    for t in grid:
        lab = fcluster(Z, 1 - t, criterion="distance")
        s = summarize(df, lab)
        s["threshold"] = t
        rows.append(s)
    cal = pd.DataFrame(rows)[
        ["threshold", "n_clusters", "mean_cluster", "max_cluster", "multi_clusters",
         "cross_split_clusters", "leaked_test", "leaked_test_pct",
         "label_conflict_clusters", "multi_plane_clusters"]]
    cal.to_csv(cfg.DIR_GROUPS / "threshold_calibration.csv", index=False)
    print(cal.to_string(index=False))

    if args.threshold is None:
        ok = cal[cal.max_cluster <= args.max_study_slices]
        thr = float(ok.threshold.min()) if len(ok) else 0.99
        print(f"\n  Automatic threshold: {thr}")
        print("  ⚠ Preferably provide the threshold from 05c rather than using the automatic one.")
    else:
        thr = args.threshold
        print(f"\n  Provided threshold: {thr}")
        row = cal[np.isclose(cal.threshold, thr)]
        if len(row) and row.iloc[0].max_cluster <= args.max_study_slices:
            print(f"  ✓ max cluster = {int(row.iloc[0].max_cluster)} "
                  f"≤ {args.max_study_slices} (per the BRISC documentation)")
        elif len(row):
            print(f"  ⚠ max cluster = {int(row.iloc[0].max_cluster)} "
                  f"> {args.max_study_slices}. Check the linkage method.")

    cfg.banner("4/5  Final clustering")
    labels = fcluster(Z, 1 - thr, criterion="distance")

    tmp = df.assign(g=labels)
    newg, nxt, n_split_ops = np.zeros(len(df), dtype=np.int64), 1, 0
    for g, idx in tmp.groupby("g").groups.items():
        idx = np.asarray(list(idx))
        cls = df.loc[idx, "class_name"].to_numpy()
        if len(set(cls)) == 1:
            newg[idx] = nxt; nxt += 1
        else:
            n_split_ops += 1
            for c in sorted(set(cls)):
                newg[idx[cls == c]] = nxt; nxt += 1
    labels = newg
    print(f"  {n_split_ops} multi-class clusters were split")

    df["pseudo_patient_id"] = [f"pp_{v:05d}" for v in labels]
    fin = summarize(df, labels)
    print()
    for k, v in fin.items():
        print(f"  {k:26s}: {v}")

    grp = df.groupby("pseudo_patient_id")
    df["pp_size"] = grp["uid"].transform("size")
    df["pp_spans_official_split"] = grp["official_split"].transform("nunique") == 2

    print("\n  Pseudo-patient size distribution:")
    print(df.drop_duplicates("pseudo_patient_id").pp_size
          .value_counts().sort_index().to_string())

    print("\n  Test leakage by class:")
    lk = df[(df.official_split == "test") & df.pp_spans_official_split]
    for c in cfg.CLASSES:
        a = int((lk.class_name == c).sum())
        b = int(((df.official_split == "test") & (df.class_name == c)).sum())
        print(f"    {c:12s} {a:3d}/{b:3d} = {100*a/max(b,1):5.1f}%")

    out = cfg.DIR_GROUPS / "pseudo_patients.csv"
    df[["uid", "official_split", "class_name", "plane_name",
        "pseudo_patient_id", "pp_size", "pp_spans_official_split"]].to_csv(out, index=False)
    print(f"\n  → {out}")

    cfg.banner("5/5  Figure")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(cal.threshold, cal.max_cluster, "o-")
        ax[0].axhline(args.max_study_slices, ls="--", c="r",
                      label=f"documented max ({args.max_study_slices})")
        ax[0].axvline(thr, ls=":", c="g", label=f"chosen {thr}")
        ax[0].invert_xaxis(); ax[0].set_xlabel("cosine threshold")
        ax[0].set_ylabel("max cluster size"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"(a) Calibration [{args.linkage} linkage]")

        ax[1].plot(cal.threshold, cal.leaked_test_pct, "s-", c="firebrick")
        ax[1].axvline(thr, ls=":", c="g"); ax[1].invert_xaxis()
        ax[1].set_xlabel("cosine threshold"); ax[1].set_ylabel("% test contaminated")
        ax[1].set_title("(b) Cross-split contamination")

        s = df.drop_duplicates("pseudo_patient_id").pp_size.value_counts().sort_index()
        ax[2].bar(s.index.astype(str), s.values, color="steelblue")
        ax[2].set_xlabel("slices per pseudo-patient"); ax[2].set_ylabel("count")
        ax[2].set_title(f"(c) Sizes @ {thr}")
        plt.tight_layout()
        p = cfg.DIR_FIGS / "fig2_pseudo_patient_calibration.png"
        plt.savefig(p, dpi=180); plt.close()
        print(f"  → {p}")
    except ImportError:
        print("  matplotlib not available.")


if __name__ == "__main__":
    main()
