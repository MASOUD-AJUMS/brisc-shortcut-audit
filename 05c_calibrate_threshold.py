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
\
\
\
   

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
                                                                         
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-precision", type=float, default=0.80,
                    help="minimum acceptable precision for accepting a band")
    ap.add_argument("--reviewer2", default=None,
                    help="second reviewer file for inter-rater kappa")
    args = ap.parse_args()

    cfg.set_seed()
    rp = cfg.DIR_GROUPS / "review_stratified.csv"
    kp = cfg.DIR_GROUPS / "review_key.csv"
    if not rp.exists() or not kp.exists():
        sys.exit("Review file not found. Run 05b_review_sheets.py first.")

    rev = pd.read_csv(rp)
    key = pd.read_csv(kp)[["review_id", "band", "similarity"]]
    m = rev.merge(key, on="review_id")

    ans = m["answer_same_study"].astype(str).str.strip().str.lower()
    m["ans"] = ans
    filled = m[ans.isin(["yes", "no", "unsure"])]
    if len(filled) == 0:
        sys.exit("No answers filled in. Complete the answer_same_study column.")
    print(f"{len(filled)}/{len(m)} pairs reviewed "
          f"({100*len(filled)/len(m):.0f}%)")
    print(f"  yes={int((filled.ans=='yes').sum())}  "
          f"no={int((filled.ans=='no').sum())}  "
          f"unsure={int((filled.ans=='unsure').sum())}")

                                                            
    if args.reviewer2:
        try:
            from sklearn.metrics import cohen_kappa_score
            r2 = pd.read_csv(args.reviewer2)[["review_id", "answer_same_study"]]
            r2.columns = ["review_id", "ans2"]
            j = filled.merge(r2, on="review_id")
            j = j[j.ans2.astype(str).str.lower().isin(["yes", "no", "unsure"])]
            k = cohen_kappa_score(j.ans, j.ans2.str.lower())
            print(f"\n  Cohen's kappa between the two reviewers: {k:.3f}  (n={len(j)})")
            print("  >0.8 excellent | 0.6-0.8 good | <0.6 poor, clarify the definition")
        except Exception as e:
            print(f"  kappa computation failed: {e}")

                                                                   
    cfg.banner("Precision by similarity band")
    conf = filled[filled.ans != "unsure"]
    rows = []
    for b, s in conf.groupby("band"):
        k_ = int((s.ans == "yes").sum())
        n_ = len(s)
        p, lo, hi = wilson(k_, n_)
        rows.append({"band": b, "n_reviewed": n_, "n_yes": k_,
                     "precision": p, "ci_low": lo, "ci_high": hi,
                     "sim_min": s.similarity.min(), "sim_max": s.similarity.max(),
                     "n_unsure": int((filled[filled.band == b].ans == "unsure").sum())})
    bp = pd.DataFrame(rows).sort_values("sim_min", ascending=False)
    bp.to_csv(cfg.DIR_GROUPS / "band_precision.csv", index=False)
    print(bp.round(4).to_string(index=False))

                                                                     
    cfg.banner("Threshold selection")
    ok = bp[bp.precision >= args.min_precision]
    if len(ok) == 0:
        print(f"  No band reached precision {args.min_precision}.")
        print("  Report only exact duplicates (md5).")
        thr = 0.9995
    else:
        thr = float(ok.sim_min.min())
        print(f"  Loosest band with precision ≥ {args.min_precision}: "
              f"{ok.sort_values('sim_min').iloc[0].band}")
        print(f"  Selected threshold: {thr:.5f}")
        print("  Justification for the paper: “The threshold was chosen such that the grouping precision "
              f"in the blinded radiological review was at least {args.min_precision:.0%}.”")

                                                                  
    cfg.banner("Leakage estimate with precision correction")
    df = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv")
    E = np.load(cfg.DIR_EMBED / "dino_embeddings.npy")
    U = np.load(cfg.DIR_EMBED / "dino_uids.npy", allow_pickle=True)
    o = pd.Series(range(len(U)), index=U)
    E = E[df["uid"].map(o).to_numpy()]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    te = (df.official_split == "test").to_numpy()
    S = E[te] @ E[~te].T
    nn1 = S.max(1)
    te_uid = df.loc[te, "uid"].to_numpy()

    prec = dict(zip(bp.band, bp.precision))
    lo_d = dict(zip(bp.band, bp.ci_low))
    hi_d = dict(zip(bp.band, bp.ci_high))
    band_edges = (key.groupby("band")
                  .agg(lo=("similarity", "min"), hi=("similarity", "max"))
                  .sort_values("lo", ascending=False))

    est, est_lo, est_hi, raw = 0.0, 0.0, 0.0, 0
    print(f"  {'band':12s} {'raw count':>10s} {'prec.':>8s} {'corrected':>10s}")
    for b, r in band_edges.iterrows():
        if r.lo < thr:
            continue
        n = int(((nn1 >= r.lo) & (nn1 <= r.hi + 1e-9)).sum())
        p = prec.get(b, np.nan)
        if np.isnan(p):
            continue
        raw += n
        est += n * p
        est_lo += n * lo_d.get(b, p)
        est_hi += n * hi_d.get(b, p)
        print(f"  {b:12s} {n:10d} {p:8.3f} {n*p:10.1f}")

    n_test = int(te.sum())
    print(f"\n  raw count (no correction)   : {raw} ({100*raw/n_test:.1f}%)")
    print(f"  corrected estimate          : {est:.0f} ({100*est/n_test:.1f}%)")
    print(f"  95% confidence interval     : "
          f"[{est_lo:.0f}, {est_hi:.0f}] "
          f"([{100*est_lo/n_test:.1f}%, {100*est_hi/n_test:.1f}%])")

    n_exact = int((nn1 >= 0.9995).sum())
    print(f"\n  hard floor (md5 pixel copy) : {n_exact} ({100*n_exact/n_test:.1f}%)")
    print("  ← This number is proven and needs no correction.")

    pd.DataFrame([{
        "threshold": thr, "n_test": n_test,
        "exact_lower_bound": n_exact,
        "raw_count": raw, "corrected_estimate": est,
        "ci_low": est_lo, "ci_high": est_hi,
        "min_precision_required": args.min_precision,
    }]).to_csv(cfg.DIR_RESULTS / "leakage_estimate.csv", index=False)

                                                           
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        x = range(len(bp))
        ax[0].errorbar(x, bp.precision,
                       yerr=[bp.precision - bp.ci_low, bp.ci_high - bp.precision],
                       fmt="o-", capsize=4)
        ax[0].axhline(args.min_precision, ls="--", c="r",
                      label=f"accept ≥ {args.min_precision}")
        ax[0].set_xticks(list(x)); ax[0].set_xticklabels(bp.band, rotation=45,
                                                         ha="right", fontsize=8)
        ax[0].set_ylabel("precision (blinded review)"); ax[0].set_ylim(0, 1.05)
        ax[0].legend(fontsize=8); ax[0].set_title("(a) Grouping precision by band")

        grid = np.linspace(0.97, 1.0, 200)
        cnt = [(nn1 >= g).sum() / n_test * 100 for g in grid]
        ax[1].plot(grid, cnt)
        ax[1].axvline(thr, ls=":", c="g", label=f"calibrated {thr:.4f}")
        ax[1].axhline(100 * n_exact / n_test, ls="--", c="firebrick",
                      label=f"proven exact {100*n_exact/n_test:.1f}%")
        ax[1].set_xlabel("cosine threshold"); ax[1].set_ylabel("% test flagged")
        ax[1].legend(fontsize=8); ax[1].set_title("(b) Leakage vs threshold")
        plt.tight_layout()
        p = cfg.DIR_FIGS / "fig3_threshold_calibration.png"
        plt.savefig(p, dpi=180); plt.close()
        print(f"\n  → {p}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
