\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

                                                           
BANDS = [
    (0.9995, 1.001, "B0_exact"),
    (0.9970, 0.9995, "B1"),
    (0.9940, 0.9970, "B2"),
    (0.9910, 0.9940, "B3"),
    (0.9880, 0.9910, "B4"),
    (0.9850, 0.9880, "B5"),
    (0.9800, 0.9850, "B6"),
    (0.9700, 0.9800, "B7_control"),
    (0.9000, 0.9700, "B8_control"),
]


def make_pair_image(pa: str, pb: str, out: Path, label: str) -> bool:
    try:
        a = Image.open(pa).convert("L").resize((320, 320))
        b = Image.open(pb).convert("L").resize((320, 320))
    except Exception:
        return False
    canvas = Image.new("L", (320 * 2 + 12, 320), color=255)
    canvas.paste(a, (0, 0))
    canvas.paste(b, (332, 0))
    canvas.save(out / f"{label}.png")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=30)
    ap.add_argument("--make-images", action="store_true", default=True)
    args = ap.parse_args()

    cfg.set_seed()
    df = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv")
    E = np.load(cfg.DIR_EMBED / "dino_embeddings.npy")
    U = np.load(cfg.DIR_EMBED / "dino_uids.npy", allow_pickle=True)
    order = pd.Series(range(len(U)), index=U)
    E = E[df["uid"].map(order).to_numpy()]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    cfg.banner("Extracting pairs in each band")
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    cls = df["class_name"].to_numpy()
    recs = []

                                                                          
    step = 500
    pairs = []
    for i in range(0, len(E), step):
        S = E[i:i+step] @ E.T
        for r in range(S.shape[0]):
            S[r, i + r] = -1
        j = S.argmax(1)
        v = S.max(1)
        for r in range(S.shape[0]):
            pairs.append((i + r, int(j[r]), float(v[r])))
    P = pd.DataFrame(pairs, columns=["a", "b", "sim"])
                       
    P["key"] = P.apply(lambda x: tuple(sorted((x.a, x.b))), axis=1)
    P = P.drop_duplicates("key")
                                                                
    P = P[cls[P.a.to_numpy()] == cls[P.b.to_numpy()]]
    print(f"  {len(P)} same-class nearest-neighbor pairs")

    for lo, hi, name in BANDS:
        sub = P[(P.sim >= lo) & (P.sim < hi)]
        if len(sub) == 0:
            print(f"  {name:12s} [{lo:.4f},{hi:.4f})  empty")
            continue
        n = min(args.per_band, len(sub))
        pick = sub.sample(n, random_state=int(rng.integers(1e9)))
        print(f"  {name:12s} [{lo:.4f},{hi:.4f})  available={len(sub):5d}  sampled={n}")
        for _, r in pick.iterrows():
            ra, rb = df.iloc[int(r.a)], df.iloc[int(r.b)]
            recs.append({
                "band": name, "similarity": round(r.sim, 6),
                "uid_a": ra.uid, "uid_b": rb.uid,
                "class": ra.class_name,
                "plane_a": ra.plane_name, "plane_b": rb.plane_name,
                "split_a": ra.official_split, "split_b": rb.official_split,
                "path_a": ra.img_path, "path_b": rb.img_path,
                "answer_same_study": "",
            })

    rev = pd.DataFrame(recs)
                                                 
    rev = rev.sample(frac=1, random_state=cfg.GLOBAL_SEED).reset_index(drop=True)
    rev.insert(0, "review_id", [f"R{i:04d}" for i in range(len(rev))])

    out = cfg.DIR_GROUPS / "review_stratified.csv"
                                                       
    blind = rev.drop(columns=["band", "similarity"])
    blind.to_csv(out, index=False)
    rev.to_csv(cfg.DIR_GROUPS / "review_key.csv", index=False)                   

    print(f"\n  {len(rev)} pairs → {out}")
    print(f"  Key (do not open until the review is finished) → "
          f"{cfg.DIR_GROUPS/'review_key.csv'}")

    if args.make_images:
        d = cfg.DIR_GROUPS / "review_images"
        d.mkdir(exist_ok=True)
        ok = 0
        for _, r in rev.iterrows():
            if make_pair_image(r.path_a, r.path_b, d, r.review_id):
                ok += 1
        print(f"  {ok} paired images → {d}")

    cfg.banner("Review instructions")
    print("""
  For each image Rxxxx.png:
    Are these two images from the same MRI study (same patient, same session)?

    yes  = same patient — either an identical image or an adjacent slice from the same series
    no   = two different patients who happen to look alike
    unsure = I am not sure

  Fill in the answer_same_study column in review_stratified.csv.
  The file is blinded — you can see neither the similarity band nor which split the images came from.
  The order is also randomized. This is intentional so that your estimate stays unbiased.

  If possible, have a second person review independently as well so that inter-rater kappa
  can be reported. Top-tier journals require this.
""")


if __name__ == "__main__":
    main()
