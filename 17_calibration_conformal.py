


import argparse, glob, json, os, re
import numpy as np
import pandas as pd

OUT = os.environ.get("BRISC_OUT", "/content/drive/MyDrive/BRISC_2025_Downloaded/outputs")
RES, FIG = f"{OUT}/results", f"{OUT}/figures"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
NT = 3
EPS = 1e-12
ALPHAS = (0.01, 0.02, 0.05, 0.10)
EXCLUDE = {"mt_ctam_final"}                                        


def load(model_tag, fold, split):
    p = f"{RES}/preds_P3_{model_tag}_f{fold}_{split}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z["probs"].astype(np.float64), z["y"].astype(int), z["uids"].astype(str)


def to_logits(P):
    return np.log(np.clip(P, EPS, 1.0))


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean())


def fit_temperature(P, y):
    L = to_logits(P)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    losses = [nll(softmax(L / t), y) for t in grid]
    t0 = grid[int(np.argmin(losses))]
    fine = t0 * np.exp(np.linspace(-0.05, 0.05, 101))
    return float(fine[int(np.argmin([nll(softmax(L / t), y) for t in fine]))])


def ece_mce(P, y, n_bins=15):
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            g = abs(acc[m].mean() - conf[m].mean())
            ece += g * m.mean(); mce = max(mce, g)
    return float(ece), float(mce)


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(1).mean())


def calib_metrics(P, y):
    e, m = ece_mce(P, y)
    return dict(acc=float((P.argmax(1) == y).mean()), ece=e, mce=m, brier=brier(P, y), nll=nll(P, y))


def qhat(scores, alpha):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


def aps_scores(P, y=None, rng=None):
    order = np.argsort(-P, 1)
    Ps = np.take_along_axis(P, order, 1)
    cum = np.cumsum(Ps, 1)
    u = rng.random(len(P))[:, None] if rng is not None else np.ones((len(P), 1))
    s_sorted = cum - Ps * u                                                 
    S = np.empty_like(P)
    np.put_along_axis(S, order, s_sorted, 1)
    return S[np.arange(len(P)), y] if y is not None else S


def conformal_sets(Pc, yc, Pt, alpha, method, seed=0):
    rng = np.random.default_rng(seed)
    if method == "LAC":
        q = qhat(1 - Pc[np.arange(len(yc)), yc], alpha)
        return (1 - Pt) <= q
    if method == "APS":
        q = qhat(aps_scores(Pc, yc), alpha)
        return aps_scores(Pt) <= q
    if method == "Mondrian":
        sets = np.zeros_like(Pt, bool)
        for c in range(Pt.shape[1]):
            sc = 1 - Pc[yc == c, c]
            if len(sc) == 0:
                sets[:, c] = True; continue
            sets[:, c] = (1 - Pt[:, c]) <= qhat(sc, alpha)
        return sets
    raise ValueError(method)


def set_metrics(sets, y):
    cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    tumor_any = sets[:, :NT].any(1)
    r = dict(coverage=float(cov.mean()), mean_size=float(size.mean()),
             singleton=float((size == 1).mean()), empty=float((size == 0).mean()),
             detect_ambig=float((tumor_any & sets[:, NT]).mean()))
    for c, n in enumerate(CLASSES):
        if (y == c).any():
            r[f"cov_{n}"] = float(cov[y == c].mean())
            r[f"size_{n}"] = float(size[y == c].mean())
    return r


def discover():
    runs = set()
    for p in glob.glob(f"{RES}/preds_P3_*_f*_val.npz"):
        m = re.match(r"preds_P3_(.+)_f(\d+)_val\.npz", os.path.basename(p))
        if m:
            runs.add(m.group(1))
    return sorted(runs)


def reliability_figure(ens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(ens)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.3), squeeze=False)
    for j, (name, (Praw, Pcal, y)) in enumerate(ens.items()):
        a = ax[0, j]
        for P, lab, mk in ((Praw, "raw", "o-"), (Pcal, "temp-scaled", "s-")):
            conf, acc = P.max(1), (P.argmax(1) == y)
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
            xs, ys = [], []
            for b in range(10):
                m = idx == b
                if m.sum() >= 5:
                    xs.append(conf[m].mean()); ys.append(acc[m].mean())
            e, _ = ece_mce(P, y)
            a.plot(xs, ys, mk, ms=3, lw=1, label=f"{lab} (ECE={e:.3f})")
        a.plot([0, 1], [0, 1], "k--", lw=0.6)
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.set_title(name, fontsize=9); a.set_xlabel("confidence")
        if j == 0:
            a.set_ylabel("accuracy")
        a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/fig_reliability_P3.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  → {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="model:tag")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    runs = [r.replace(":", "_") for r in a.runs] if a.runs else [r for r in discover() if r not in EXCLUDE]
    print(f"  {len(runs)} models: {runs}")
    crow, frow, temps, ens = [], [], {}, {}
    for mt in runs:
        Pr_list, Pc_list, y_ref = [], [], None
        for f in a.folds:
            v, t = load(mt, f, "val"), load(mt, f, "test")
            if v is None or t is None:
                continue
            (Pv, yv, _), (Pt, yt, ut) = v, t
            T = fit_temperature(Pv, yv)
            Pv_c, Pt_c = softmax(to_logits(Pv) / T), softmax(to_logits(Pt) / T)
            temps[f"{mt}_f{f}"] = T
            before, after = calib_metrics(Pt, yt), calib_metrics(Pt_c, yt)
            crow.append(dict(run=mt, fold=f, T=T, **{f"{k}_raw": v_ for k, v_ in before.items()},
                             **{f"{k}_ts": v_ for k, v_ in after.items()}))
            for alpha in ALPHAS:
                for meth in ("LAC", "APS", "Mondrian"):
                    sm = set_metrics(conformal_sets(Pv_c, yv, Pt_c, alpha, meth, seed=f), yt)
                    frow.append(dict(run=mt, fold=f, alpha=alpha, method=meth, **sm))
            if y_ref is None:
                y_ref, u_ref = yt, ut
            if list(ut) == list(u_ref):
                Pr_list.append(Pt); Pc_list.append(Pt_c)
        if Pr_list:
            ens[mt] = (np.mean(Pr_list, 0), np.mean(Pc_list, 0), y_ref)
            c = [r for r in crow if r["run"] == mt]
            print(f"  {mt:24s} T={np.mean([r['T'] for r in c]):.2f}  "
                  f"ECE {np.mean([r['ece_raw'] for r in c]):.4f} → {np.mean([r['ece_ts'] for r in c]):.4f}  "
                  f"NLL {np.mean([r['nll_raw'] for r in c]):.4f} → {np.mean([r['nll_ts'] for r in c]):.4f}")
    if not crow:
        print("No preds_P3_*_val/test found."); return

    C, Fm = pd.DataFrame(crow), pd.DataFrame(frow)
    C.to_csv(f"{RES}/calib_P3_perfold.csv", index=False)
    Fm.to_csv(f"{RES}/conformal_P3_perfold.csv", index=False)
    cs = C.drop(columns="fold").groupby("run").agg(["mean", "std"])
    cs.columns = [f"{x}_{y}" for x, y in cs.columns]
    cs.to_csv(f"{RES}/calib_P3_summary.csv")
    fs = Fm.drop(columns="fold").groupby(["run", "alpha", "method"]).agg(["mean", "std"])
    fs.columns = [f"{x}_{y}" for x, y in fs.columns]
    fs.to_csv(f"{RES}/conformal_P3_summary.csv")
    json.dump(temps, open(f"{RES}/temperature_P3.json", "w"), indent=1)

    print("\n" + "=" * 70 + "\nconformal (mean across folds)\n" + "=" * 70)
    show = ["coverage", "mean_size", "singleton", "empty", "detect_ambig", "cov_glioma", "cov_meningioma",
            "cov_pituitary", "cov_no_tumor"]
    t = Fm.groupby(["run", "alpha", "method"])[show].mean().round(3)
    print(t.to_string())
    reliability_figure(ens)
    print(f"\n  → {RES}/calib_P3_summary.csv , conformal_P3_summary.csv , temperature_P3.json")


if __name__ == "__main__":
    main()
