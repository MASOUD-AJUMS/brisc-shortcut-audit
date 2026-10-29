

import argparse, glob, json, os, re, itertools, warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.environ.get("BRISC_OUT", "/content/drive/MyDrive/BRISC_2025_Downloaded/outputs")
RES = f"{OUT}/results"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
DEFAULT = ["cls_resnet50_base", "seg_unet_r50_base", "mt_base_base", "mt_aspp_base", "mt_ctam_base",
           "mt_base_ev", "mt_ctam_ev"]
EPS = 1e-12


def ensemble(run, folds=range(5)):
    Ps, segs, uids, y = [], [], None, None
    T = {}
    tp = f"{RES}/temperature_P3.json"
    if os.path.exists(tp):
        T = json.load(open(tp))
    Pts = []
    for f in folds:
        p = f"{RES}/preds_P3_{run}_f{f}_test.npz"
        sp = f"{RES}/segper_P3_{run}_f{f}_test.csv"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            u = z["uids"].astype(str)
            if uids is None:
                uids, y = u, z["y"].astype(int)
            order = pd.Series(np.arange(len(u)), index=u).reindex(uids).values
            P = z["probs"].astype(np.float64)[order]
            Ps.append(P)
            t = T.get(f"{run}_f{f}")
            if t:
                L = np.log(np.clip(P, EPS, 1)) / t
                L -= L.max(1, keepdims=True); E = np.exp(L)
                Pts.append(E / E.sum(1, keepdims=True))
        if os.path.exists(sp):
            d = pd.read_csv(sp)
            if os.path.exists(p):
                d.index = z["uids"].astype(str)
                segs.append(d[["dice", "hd95", "gt_empty"]].reindex(uids))
            else:
                segs.append(d[["dice", "hd95", "gt_empty"]])
    out = {"uids": uids, "y": y}
    if Ps:
        out["P"] = np.mean(Ps, 0)
        out["n_folds_cls"] = len(Ps)
        if len(Pts) == len(Ps):
            out["P_ts"] = np.mean(Pts, 0)
    if segs:
        out["dice"] = np.nanmean(np.stack([s["dice"].values.astype(float) for s in segs]), 0)
        out["hd95"] = np.nanmean(np.stack([s["hd95"].values.astype(float) for s in segs]), 0)
        out["lesion"] = ~segs[0]["gt_empty"].astype(bool).values
        out["n_folds_seg"] = len(segs)
    return out


def bal_acc(P, y):
    pred = P.argmax(1)
    return np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])


def macro_auc(P, y):
    from sklearn.metrics import roc_auc_score
    return np.mean([roc_auc_score(y == c, P[:, c]) for c in range(P.shape[1]) if 0 < (y == c).sum() < len(y)])


def ece(P, y, n_bins=15):
    conf, pred = P.max(1), P.argmax(1)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return sum(abs((pred[idx == b] == y[idx == b]).mean() - conf[idx == b].mean()) * (idx == b).mean()
               for b in range(n_bins) if (idx == b).any())


def delong(y_bin, s1, s2):
    pos, neg = s1[y_bin == 1], s1[y_bin == 0]
    m, n = len(pos), len(neg)

    def comps(s):
        X, Y = s[y_bin == 1], s[y_bin == 0]
        V10 = np.array([(np.sum(x > Y) + 0.5 * np.sum(x == Y)) / n for x in X])
        V01 = np.array([(np.sum(X > yv) + 0.5 * np.sum(X == yv)) / m for yv in Y])
        return V10.mean(), V10, V01
    a1, V10a, V01a = comps(s1)
    a2, V10b, V01b = comps(s2)
    S10 = np.cov(np.vstack([V10a, V10b]))
    S01 = np.cov(np.vstack([V01a, V01b]))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    if var <= 0:
        return a1, a2, 0.0, 1.0
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * stats.norm.sf(abs(z))


def boot_diff(f, A, B, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = f(A, y) - f(B, y)
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    ds = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        ds.append(f(A[idx], y[idx]) - f(B[idx], y[idx]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return obs, *np.percentile(ds, [2.5, 97.5]), min(max(p, 1 / n), 1.0)


def mcnemar(c1, c2):
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    models = a.models or [m for m in DEFAULT if glob.glob(f"{RES}/*_P3_{m}_f*_test.*")]
    E = {m: ensemble(m) for m in models}
    print("  models:", {m: (e.get("n_folds_cls", 0), e.get("n_folds_seg", 0)) for m, e in E.items()},
          "(number of folds cls, seg)")

    cls_m = [m for m in models if "P" in E[m]]
    ref = E[cls_m[0]]
    for m in cls_m[1:]:
        assert (E[m]["uids"] == ref["uids"]).all(), f"uid order differs in {m}"
    y = ref["y"]
    rows, prow = [], []
    for m1, m2 in itertools.combinations(cls_m, 2):
        A, B = E[m1]["P"], E[m2]["P"]
        r = dict(model_a=m1, model_b=m2)
        r["macro_auc_a"], r["macro_auc_b"] = macro_auc(A, y), macro_auc(B, y)
        d, lo, hi, p = boot_diff(macro_auc, A, B, y, a.boot)
        r.update(macro_auc_diff=d, macro_auc_ci_lo=lo, macro_auc_ci_hi=hi, p_macro_auc=p)
        r["bal_acc_a"], r["bal_acc_b"] = bal_acc(A, y), bal_acc(B, y)
        d, lo, hi, p = boot_diff(bal_acc, A, B, y, a.boot, seed=1)
        r.update(bal_acc_diff=d, bal_acc_ci_lo=lo, bal_acc_ci_hi=hi, p_bal_acc=p)
        b, c, p = mcnemar(A.argmax(1) == y, B.argmax(1) == y)
        r.update(mcnemar_a_only=b, mcnemar_b_only=c, p_mcnemar=p)
        if "P_ts" in E[m1] and "P_ts" in E[m2]:
            d, lo, hi, p = boot_diff(ece, E[m1]["P_ts"], E[m2]["P_ts"], y, a.boot, seed=2)
            r.update(ece_ts_a=ece(E[m1]["P_ts"], y), ece_ts_b=ece(E[m2]["P_ts"], y),
                     ece_ts_diff=d, ece_ts_ci_lo=lo, ece_ts_ci_hi=hi, p_ece_ts=p)
        rows.append(r)
        for c_, cn in enumerate(CLASSES):
            a1, a2, z, p = delong((y == c_).astype(int), A[:, c_], B[:, c_])
            prow.append(dict(model_a=m1, model_b=m2, cls=cn, auc_a=a1, auc_b=a2, z=z, p=p))
    C = pd.DataFrame(rows)
    for col in [c for c in C.columns if c.startswith("p_")]:
        C[col.replace("p_", "holm_")] = holm(C[col].fillna(1).values)
    D = pd.DataFrame(prow)
    if len(D):
        D["holm"] = holm(D.p.values)
    C.to_csv(f"{RES}/stats_P3_cls_pairs.csv", index=False)
    D.to_csv(f"{RES}/stats_P3_delong_perclass.csv", index=False)

    seg_m = [m for m in models if "dice" in E[m]]
    srows = []
    for m1, m2 in itertools.combinations(seg_m, 2):
        les = E[m1]["lesion"] & E[m2]["lesion"]
        for met in ("dice", "hd95"):
            x1, x2 = E[m1][met][les], E[m2][met][les]
            ok = ~np.isnan(x1) & ~np.isnan(x2)
            w = stats.wilcoxon(x1[ok], x2[ok], zero_method="wilcox") if ok.sum() > 10 else None
            srows.append(dict(model_a=m1, model_b=m2, metric=met, n=int(ok.sum()),
                              median_a=np.median(x1[ok]), median_b=np.median(x2[ok]),
                              mean_a=x1[ok].mean(), mean_b=x2[ok].mean(),
                              median_paired_diff=np.median(x1[ok] - x2[ok]),
                              p=w.pvalue if w else np.nan))
    S = pd.DataFrame(srows)
    if len(S):
        S["holm"] = np.nan
        for met in S.metric.unique():
            k = S.metric == met
            S.loc[k, "holm"] = holm(S.loc[k, "p"].fillna(1).values)
    S.to_csv(f"{RES}/stats_P3_seg_pairs.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 70 + "\nClassification (five-fold ensemble, n=%d) — Holm-corrected p" % len(y) + "\n" + "=" * 70)
    show = ["model_a", "model_b", "bal_acc_diff", "holm_bal_acc", "macro_auc_diff", "holm_macro_auc",
            "mcnemar_a_only", "mcnemar_b_only", "holm_mcnemar"]
    show += [c for c in ("ece_ts_diff", "holm_ece_ts") if c in C.columns]
    print(C[show].round(4).to_string(index=False))
    if len(S):
        print("\n" + "=" * 70 + "\nSegmentation (Wilcoxon, lesion-containing images) — Holm-corrected p\n" + "=" * 70)
        print(S[["model_a", "model_b", "metric", "n", "mean_a", "mean_b", "median_paired_diff", "holm"]]
              .round(4).to_string(index=False))
    print(f"\n  → {RES}/stats_P3_cls_pairs.csv , stats_P3_seg_pairs.csv , stats_P3_delong_perclass.csv")


if __name__ == "__main__":
    main()
