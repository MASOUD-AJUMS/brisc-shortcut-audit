

import os, re, glob, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")
import pandas as pd

OUT = os.environ.get("BRISC_OUT", "/content/drive/MyDrive/BRISC_2025_Downloaded/outputs")
RES, GRP, FIG = f"{OUT}/results", f"{OUT}/groups", f"{OUT}/figures"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
SUBSETS = ["clean", "contaminated", "exact_dup"]
EPS = 1e-7


def pixel_md5(f):
    import hashlib
    from PIL import Image
    cache = f"{RES}/pixel_md5_cache.csv"
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        if set(f["uid"]) <= set(c["uid"]):
            return c.set_index("uid")["md5"]
    print("  Computing pixel MD5 for", len(f), "images (once, a few minutes)...")
    out = {}
    for i, (u, p) in enumerate(zip(f["uid"], f["img_path"])):
        a = np.asarray(Image.open(p).convert("L"))
        out[u] = hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(f)}")
    s = pd.Series(out, name="md5")
    s.rename_axis("uid").reset_index().to_csv(cache, index=False)
    return s


def load_subsets():
    f = pd.read_csv(f"{GRP}/folds.csv")
    t = f[f["P1_role"] == "test"].copy()
    t["subset"] = np.where(t["P2_role"] == "test", "clean", "contaminated")
    md5 = pixel_md5(f)
    f["md5"] = f["uid"].map(md5)
    train_md5 = set(f.loc[f["P1_role"] != "test", "md5"])
    t["exact_dup"] = t["uid"].map(md5).isin(train_md5).values
    bad = t["exact_dup"] & (t["subset"] == "clean")
    if bad.any():
        print(f"  ⚠ {bad.sum()} exact copies in the clean subset — inconsistency; please check")
    if t["exact_dup"].sum() != 101:
        print(f"  ⚠ number of exact copies is {int(t['exact_dup'].sum())} (previous audit: 101) — please check")
    print(f"  P1 test: {len(t)} | clean={int((t.subset=='clean').sum())} "
          f"contaminated={int((t.subset=='contaminated').sum())} exact_dup={int(t.exact_dup.sum())}")
    return t.set_index("uid")


def find_models():
    out = {}
    for p in glob.glob(f"{RES}/preds_P1_*_f*_test.npz"):
        m = re.match(r"preds_P1_(.+)_f(\d+)_test\.npz", os.path.basename(p))
        if m:
            out.setdefault(m.group(1), []).append(int(m.group(2)))
    return {k: sorted(v) for k, v in out.items()}


def load_preds(model, fold):
    z = np.load(f"{RES}/preds_P1_{model}_f{fold}_test.npz", allow_pickle=True)
    d = pd.DataFrame(z["probs"], index=[str(u) for u in z["uids"]]).assign(y=z["y"])
    return d.groupby(level=0).mean() if d.index.duplicated().any() else d


def load_dice(model, fold, proto="P1"):
    p = f"{RES}/segper_{proto}_{model}_f{fold}_test.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    dcol = next((c for c in d.columns if c.lower() in ("dice", "lesion_dice")), None) \
        or next((c for c in d.columns if "dice" in c.lower()), None)
    if dcol is None:
        print(f"  ⚠ {os.path.basename(p)}: dice column not found; columns = {list(d.columns)}")
        return None
    ucol = next((c for c in d.columns if d[c].astype(str).str.startswith("brisc2025_").mean() > 0.9), None)
    if ucol is not None:
        return d.groupby(ucol)[dcol].mean().astype(float)
    z = np.load(f"{RES}/preds_{proto}_{model}_f{fold}_test.npz", allow_pickle=True)
    if len(d) != len(z["uids"]):
        print(f"  ⚠ {os.path.basename(p)}: {len(d)} rows ≠ {len(z['uids'])} in preds — dice ignored")
        return None
    if "gt_empty" in d.columns:
        agree = (d["gt_empty"].astype(bool).values == (z["y"] == CLASSES.index("no_tumor"))).mean()
        if agree < 1.0:
            print(f"  ⚠ {os.path.basename(p)}: ordering consistency with preds not confirmed (gt_empty agreement={agree:.3f}) — dice ignored")
            return None
    s = pd.Series(d[dcol].astype(float).values, index=[str(u) for u in z["uids"]])
    return s.groupby(level=0).mean()


def cls_metrics(P, y):
    pred = P.argmax(1)
    ptrue = P[np.arange(len(y)), y]
    sens = [np.mean(pred[y == c] == c) for c in range(4) if (y == c).any()]
    conf = P.max(1)
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 14)
    ece = sum(abs((pred[idx == b] == y[idx == b]).mean() - conf[idx == b].mean()) * (idx == b).mean()
              for b in range(15) if (idx == b).any())
    return dict(n=len(y), n_err=int((pred != y).sum()), acc=float((pred == y).mean()),
                bal_acc=float(np.mean(sens)), mean_ptrue=float(ptrue.mean()),
                nll=float(-np.log(ptrue + EPS).mean()), ece=float(ece))


def subset_mask(meta, name):
    return meta["exact_dup"].values if name == "exact_dup" else (meta["subset"] == name).values


def strat_perm_test(score, contaminated, strata, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = score[contaminated].mean() - score[~contaminated].mean()
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    lab = contaminated.copy()
    cnt = 0
    for _ in range(n_perm):
        for g in groups:
            lab[g] = rng.permutation(lab[g])
        d = score[lab].mean() - score[~lab].mean()
        cnt += abs(d) >= abs(obs) - 1e-12
    return float(obs), (cnt + 1) / (n_perm + 1)


def class_std_diff(score, contaminated, y):
    diffs, w = [], []
    for c in range(4):
        a, b = score[(y == c) & contaminated], score[(y == c) & ~contaminated]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean()); w.append(len(b))
    return float(np.average(diffs, weights=w)) if diffs else np.nan



def load_p3_ensemble(model):
    fs = sorted(glob.glob(f"{RES}/preds_P3_{model}_f*_test.npz"))
    if not fs:
        return None
    ds = []
    for p in fs:
        z = np.load(p, allow_pickle=True)
        d = pd.DataFrame(z["probs"], index=[str(u) for u in z["uids"]]).assign(y=z["y"])
        ds.append(d.groupby(level=0).mean())
    idx = ds[0].index
    P = np.mean([d.reindex(idx)[[0, 1, 2, 3]].values for d in ds], axis=0)
    return pd.DataFrame(P, index=idx).assign(y=ds[0]["y"].values)


def boot_ci_did(s1, s3, cont, strata, n_boot=5000, seed=1):
    rng = np.random.default_rng(seed)
    cells = [np.where((strata == c) & (cont == k))[0] for c in np.unique(strata) for k in (True, False)]
    cells = [g for g in cells if len(g)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in cells])
        vals.append(class_std_diff(s1[idx], cont[idx], strata[idx]) - class_std_diff(s3[idx], cont[idx], strata[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])


def load_p3_dice(model, uids):
    fs = sorted(glob.glob(f"{RES}/preds_P3_{model}_f*_test.npz"))
    ds = [load_dice(model, int(re.search(r"_f(\d+)_test", f).group(1)), proto="P3") for f in fs]
    if not ds or any(d is None for d in ds):
        return None
    return np.nanmean([d.reindex(uids).values for d in ds], axis=0)


def did_analysis(model, meta, ens, y, trows, dice_ens=None):
    p3 = load_p3_ensemble(model)
    if p3 is None:
        print("  (DiD: preds_P3 not available for this model — skipped)"); return
    common = [u for u in meta.index if u in p3.index]
    if len(common) < 30:
        print(f"  (DiD: only {len(common)} common images — skipped)"); return
    pos = meta.index.get_indexer(common)
    yy = y[pos]
    if not (p3.loc[common, "y"].values.astype(int) == yy).all():
        print("  ⚠ DiD: P1 and P3 labels do not match — skipped"); return
    cont = (meta["subset"].values[pos] == "contaminated")
    pt1 = ens[pos, yy]
    pt3 = p3.loc[common, [0, 1, 2, 3]].values[np.arange(len(yy)), yy]
    scores = [("NLL", -np.log(pt1 + EPS), -np.log(pt3 + EPS), np.ones(len(yy), bool)),
              ("1-p_true", 1 - pt1, 1 - pt3, np.ones(len(yy), bool))]
    if dice_ens is not None:
        d3 = load_p3_dice(model, common)
        if d3 is not None:
            d1 = dice_ens[pos]
            ok = (yy != CLASSES.index("no_tumor")) & ~np.isnan(d1) & ~np.isnan(d3)
            scores.append(("Dice", d1, d3, ok))
        else:
            print("  (DiD Dice: segper_P3 not usable — cls only)")
    print(f"  DiD on {len(common)} common images with the P3 test (contaminated={int(cont.sum())}):")
    for name, s1, s3, ok in scores:
        s1, s3, c, st = s1[ok], s3[ok], cont[ok], yy[ok]
        e1, e3 = class_std_diff(s1, c, st), class_std_diff(s3, c, st)
        obs, p = strat_perm_test(s1 - s3, c, st)
        _, p3only = strat_perm_test(s3, c, st)
        lo, hi = boot_ci_did(s1, s3, c, st)
        trows.append(dict(model=model, test=f"DiD_{name}_P1minusP3_perm_stratified", n=int(ok.sum()),
                          n_cont=int(c.sum()), effect=obs, effect_class_std=e1 - e3,
                          ci95_lo=lo, ci95_hi=hi, diff_P1=e1, diff_P3=e3, p=p, p_P3_alone=p3only))
        print(f"     {name:8s} (cont − clean)  P1 model: {e1:+.4f} | P3 model: {e3:+.4f} (p={p3only:.3g})"
              f"  →  DiD = {e1 - e3:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  p={p:.4g}  (n={int(ok.sum())})")


def analyze(model, folds, meta, rows, prows, trows):
    print(f"\n{'='*70}\n{model}  (folds: {folds})\n{'='*70}")
    per_fold, dice_f = [], []
    for f in folds:
        d = load_preds(model, f)
        missing = set(meta.index) - set(d.index)
        if missing:
            print(f"  ⚠ fold {f}: {len(missing)} test uids not in preds");
        per_fold.append(d.reindex(meta.index))
        dd = load_dice(model, f)
        if dd is not None:
            dice_f.append(dd.reindex(meta.index))

    y = per_fold[0]["y"].values.astype(int)
    ens = np.mean([d[[0, 1, 2, 3]].values for d in per_fold], axis=0)
    has_dice = len(dice_f) == len(folds)
    dice_ens = np.nanmean([d.values for d in dice_f], axis=0) if has_dice else None
    lesion = (y != CLASSES.index("no_tumor"))

    for s in SUBSETS:
        m = subset_mask(meta, s)
        if m.sum() == 0:
            continue
        fold_m = [cls_metrics(d[[0, 1, 2, 3]].values[m], y[m]) for d in per_fold]
        r = dict(model=model, subset=s, n=int(m.sum()))
        for k in ["n_err", "acc", "bal_acc", "mean_ptrue", "nll", "ece"]:
            v = [fm[k] for fm in fold_m]
            r[f"{k}_mean"], r[f"{k}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        e = cls_metrics(ens[m], y[m])
        r.update({f"ens_{k}": e[k] for k in ["n_err", "acc", "bal_acc", "mean_ptrue", "nll"]})
        if has_dice:
            lm = m & lesion
            v = [np.nanmean(d.values[lm]) for d in dice_f]
            r["dice_mean"], r["dice_sd"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        rows.append(r)
        print(f"  {s:13s} n={r['n']:4d}  err/fold={r['n_err_mean']:.1f}  bal_acc={r['bal_acc_mean']:.4f}"
              f"  p_true={r['mean_ptrue_mean']:.4f}  nll={r['nll_mean']:.4f}"
              + (f"  dice={r['dice_mean']:.4f}" if has_dice else ""))

    for c, cn in enumerate(CLASSES):
        for s in SUBSETS:
            m = subset_mask(meta, s) & (y == c)
            if m.sum() == 0:
                continue
            pred_f = [d[[0, 1, 2, 3]].values[m].argmax(1) for d in per_fold]
            pt = ens[m, c]
            r = dict(model=model, cls=cn, subset=s, n=int(m.sum()),
                     sens_mean=float(np.mean([np.mean(p == c) for p in pred_f])),
                     n_err_mean=float(np.mean([np.sum(p != c) for p in pred_f])),
                     ens_mean_ptrue=float(pt.mean()), ens_nll=float(-np.log(pt + EPS).mean()))
            if has_dice and cn != "no_tumor":
                r["dice_ens"] = float(np.nanmean(dice_ens[m]))
            prows.append(r)

    from scipy.stats import fisher_exact
    cont = (meta["subset"] == "contaminated").values
    err = ens.argmax(1) != y
    a, b = int(err[cont].sum()), int(cont.sum() - err[cont].sum())
    c_, d_ = int(err[~cont].sum()), int((~cont).sum() - err[~cont].sum())
    _, p_f = fisher_exact([[a, b], [c_, d_]])
    trows.append(dict(model=model, test="fisher_errors_ens", contaminated=f"{a}/{a+b}",
                      clean=f"{c_}/{c_+d_}", effect=np.nan, p=p_f))

    nll = -np.log(ens[np.arange(len(y)), y] + EPS)
    obs, p = strat_perm_test(nll, cont, y)
    trows.append(dict(model=model, test="nll_diff_perm_stratified_by_class", effect=obs,
                      effect_class_std=class_std_diff(nll, cont, y), p=p))
    print(f"  Fisher error: contaminated {a}/{a+b}  clean {c_}/{c_+d_}  p={p_f:.3g}")
    print(f"  NLL (contaminated − clean): {obs:+.4f}  | class-standardized: "
          f"{class_std_diff(nll, cont, y):+.4f}  | permutation p={p:.4g}")

    if has_dice:
        lm = lesion & ~np.isnan(dice_ens)
        obs, p = strat_perm_test(dice_ens[lm], cont[lm], y[lm])
        trows.append(dict(model=model, test="dice_diff_perm_stratified_by_class", effect=obs,
                          effect_class_std=class_std_diff(dice_ens[lm], cont[lm], y[lm]), p=p))
        print(f"  Dice (contaminated − clean): {obs:+.4f}  | class-standardized: "
              f"{class_std_diff(dice_ens[lm], cont[lm], y[lm]):+.4f}  | permutation p={p:.4g}")
        ex_l = meta["exact_dup"].values[lm]; nd_l = cont[lm] & ~ex_l
        if ex_l.any():
            dl = dice_ens[lm]
            print(f"  Dice: exact_dup={np.nanmean(dl[ex_l]):.4f}  near_dup={np.nanmean(dl[nd_l]):.4f}  "
                  f"clean={np.nanmean(dl[~cont[lm]]):.4f}")
            m2 = ex_l | nd_l
            obs2, p2 = strat_perm_test(dl[m2], ex_l[m2], y[lm][m2])
            trows.append(dict(model=model, test="dice_exactdup_vs_neardup_perm", effect=obs2, p=p2))
            print(f"  Dice (exact_dup − near_dup): {obs2:+.4f}  | permutation p={p2:.4g}")

    if meta["exact_dup"].any():
        ex = meta["exact_dup"].values
        obs, p = strat_perm_test(nll[ex | ~cont], ex[ex | ~cont], y[ex | ~cont])
        trows.append(dict(model=model, test="nll_exactdup_vs_clean_perm", effect=obs, p=p))
        print(f"  NLL (exact_dup − clean): {obs:+.4f}  | permutation p={p:.4g}")
        nd = cont & ~ex
        obs, p = strat_perm_test(nll[ex | nd], ex[ex | nd], y[ex | nd])
        trows.append(dict(model=model, test="nll_exactdup_vs_neardup_perm", effect=obs, p=p))
        print(f"  NLL (exact_dup − near_dup): {obs:+.4f}  | permutation p={p:.4g}")

    did_analysis(model, meta, ens, y, trows, dice_ens if has_dice else None)
    plot(model, ens, y, cont, meta["exact_dup"].values)


def plot(model, ens, y, cont, ex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for c, ax in enumerate(axes):
        m = y == c
        pt = -np.log10(1 - ens[m, c] + 1e-6)                                                 
        data = [pt[~cont[m]], pt[cont[m] & ~ex[m]], pt[ex[m]]]
        labels = [f"clean\n(n={len(data[0])})", f"near-dup\n(n={len(data[1])})", f"exact\n(n={len(data[2])})"]
        keep = [i for i, d in enumerate(data) if len(d)]
        ax.boxplot([data[i] for i in keep], showfliers=True, widths=0.6)
        ax.set_xticks(range(1, len(keep) + 1)); ax.set_xticklabels([labels[i] for i in keep], fontsize=8)
        ax.set_title(CLASSES[c], fontsize=10)
    axes[0].set_ylabel(r"$-\log_{10}(1-p_{\mathrm{true}})$  (higher = more confident)")
    fig.suptitle(f"Confidence on true class, official BRISC test — {model} (5-fold ensemble)", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/fig_leakage_effect_P1_{model}.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  → {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    meta = load_subsets()
    avail = find_models()
    models = args.models or sorted(avail)
    if not models:
        print("No preds_P1_*_test.npz found."); return
    rows, prows, trows = [], [], []
    for m in models:
        if m not in avail:
            print(f"  ⚠ {m}: preds not found"); continue
        analyze(m, avail[m], meta, rows, prows, trows)
    pd.DataFrame(rows).to_csv(f"{RES}/leakage_effect_P1_summary.csv", index=False)
    pd.DataFrame(prows).to_csv(f"{RES}/leakage_effect_P1_perclass.csv", index=False)
    pd.DataFrame(trows).to_csv(f"{RES}/leakage_effect_P1_tests.csv", index=False)
    print(f"\n→ {RES}/leakage_effect_P1_{{summary,perclass,tests}}.csv")
    pc = pd.DataFrame(prows)
    if len(pc):
        print("\nBy class (ensemble):")
        print(pc.pivot_table(index=["model", "cls"], columns="subset",
                             values=["n", "ens_mean_ptrue"]).round(4).to_string())


if __name__ == "__main__":
    main()
