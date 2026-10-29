
import os
from importlib import import_module
import cv2
import numpy as np
import pandas as pd
from PIL import Image

cfg = import_module("00_config")
ext = import_module("20_external_validation")


def g128(p):
    a = np.asarray(Image.open(p).convert("L")).astype(np.float32)
    a = cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA)
    return (a - a.mean()) / (a.std() + 1e-6)


def main():
    df = pd.read_csv(ext.EXT / "external_index.csv", keep_default_na=False)
    uids, _, bdh = ext.brisc_hashes()
    folds = pd.read_csv(cfg.DIR_GROUPS / "folds.csv").set_index("uid")
    bbits = np.unpackbits(bdh, axis=1).astype(np.uint8)
    rows = []
    for _, r in df[df.sequence != "t1"].iterrows():                                   
        g = np.asarray(Image.open(r.img_path).convert("L"))
        d = (bbits != np.unpackbits(ext.dhash(g)).astype(np.uint8)).sum(1)
        j = int(d.argmin())
        bu = uids[j]
        ncc = float((g128(r.img_path) * g128(folds.loc[bu, "img_path"])).mean())
        rows.append(dict(uid=r.uid, source=r.source, label=r.label, img_path=r.img_path,
                         brisc_uid=bu, brisc_path=folds.loc[bu, "img_path"],
                         brisc_class=folds.loc[bu, "class_name"], dhash_dist=int(d[j]), ncc=ncc))
    R = pd.DataFrame(rows)
    R.to_csv(cfg.DIR_RESULTS / "external_dup_check.csv", index=False)

    for s, g in R.groupby("source"):
        print(f"\n  {s}: n={len(g)}")
        for lo, hi in [(0, 4), (5, 10), (11, 20), (21, 999)]:
            k = g[(g.dhash_dist >= lo) & (g.dhash_dist <= hi)]
            if len(k):
                print(f"    dHash {lo:>2}-{hi:<3}: {len(k):4d} images | NCC median={k.ncc.median():.3f} "
                      f"| NCC≥0.95: {int((k.ncc >= 0.95).sum())}")
    m = R[R.source == "mendeley"]
    if len(m):
        hi = m[m.ncc >= 0.95]
        same = (hi.label.map(lambda i: cfg.CLASSES[int(i)]) == hi.brisc_class).mean() if len(hi) else float("nan")
        print(f"\n  Mendeley with NCC ≥ 0.95 (near-certain duplicates): {len(hi)} of {len(m)}"
              f" | same class as the BRISC pair: {same:.1%}")
        print(f"  Mendeley with dHash ≤ 10 but NCC < 0.95 (probably false alarm): "
              f"{int(((m.dhash_dist <= 10) & (m.ncc < 0.95)).sum())}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for s, g in R.groupby("source"):
        ax[0].hist(g.dhash_dist, bins=range(0, 130, 3), alpha=0.6, label=s)
        ax[1].hist(g.ncc, bins=np.linspace(-0.2, 1, 49), alpha=0.6, label=s)
    ax[0].axvline(10.5, color="k", ls="--", lw=0.8); ax[0].set_xlabel("min dHash distance to BRISC")
    ax[1].axvline(0.95, color="k", ls="--", lw=0.8); ax[1].set_xlabel("NCC with nearest BRISC image")
    ax[0].legend(); fig.tight_layout()
    fig.savefig(cfg.DIR_FIGS / "external_dup_hist.png", dpi=150); plt.close(fig)

    pick = []
    for lo, hi in [(0, 4), (5, 10), (11, 20), (21, 999)]:
        k = m[(m.dhash_dist >= lo) & (m.dhash_dist <= hi)]
        pick += list(k.sample(min(4, len(k)), random_state=0).itertuples()) if len(k) else []
    if pick:
        fig, ax = plt.subplots(len(pick), 2, figsize=(4.4, 2.2 * len(pick)))
        ax = np.atleast_2d(ax)
        for i, r in enumerate(pick):
            for j, (p, t) in enumerate(((r.img_path, f"Mendeley ({cfg.CLASSES[int(r.label)]})"),
                                        (r.brisc_path, f"BRISC ({r.brisc_class})"))):
                ax[i, j].imshow(Image.open(p).convert("L"), cmap="gray"); ax[i, j].axis("off")
                ax[i, j].set_title(t + (f"\nd={r.dhash_dist} ncc={r.ncc:.2f}" if j == 0 else ""), fontsize=7)
        fig.tight_layout()
        fig.savefig(cfg.DIR_FIGS / "external_dup_pairs.png", dpi=130); plt.close(fig)
    print(f"\n  → {cfg.DIR_FIGS / 'external_dup_pairs.png'}\n  → {cfg.DIR_FIGS / 'external_dup_hist.png'}")


if __name__ == "__main__":
    main()
