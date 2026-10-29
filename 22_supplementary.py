import argparse, json, os, re, traceback, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

F = import_module("21_make_figures")
cfg, RES, GRP = F.cfg, F.RES, F.GRP
C, INK, INK2, INK3, SHAM, W2 = F.C, F.INK, F.INK2, F.INK3, F.SHAM, F.W2
MCOL, MLAB, MHAT, CLAB = F.MCOL, F.MLAB, F.MHAT, F.CLAB
ptitle, xgrid, ygrid, rcsv = F.ptitle, F.xgrid, F.ygrid, F.rcsv
CLASSES, NT, S = F.CLASSES, F.NT, F.S
OUTS = cfg.DIR_FIGS / "supp"
OUTS.mkdir(parents=True, exist_ok=True)
EXT = cfg.OUT_ROOT / "external"
SHORT = {"glioma": "Gli", "meningioma": "Men", "pituitary": "Pit", "no_tumor": "None"}
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f7fc", "#9cc3ef", C["blue"], "#0d3f80"])
CLS_RUNS = ["cls_resnet50_base", "mt_base_base", "mt_ctam_base"]
ALL_CLS = ["cls_resnet50_base", "mt_base_base", "mt_aspp_base", "mt_ctam_base", "mt_base_ev", "mt_ctam_ev"]
SEG_RUNS = ["seg_unet_r50_base", "mt_base_base", "mt_aspp_base", "mt_ctam_base", "mt_base_ev", "mt_ctam_ev"]


def save(fig, n):
    p = OUTS / f"FigureS{n}"
    fig.savefig(f"{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(f"{p}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  ✓ Figure S{n} → {p}.png")


def cm_panel(a, y, p, rows=None, title="", show_y=True):
    rows = rows if rows is not None else list(range(len(CLASSES)))
    M = np.zeros((len(rows), len(CLASSES)), int)
    for i, r in enumerate(rows):
        for j in range(len(CLASSES)):
            M[i, j] = int(((y == r) & (p == j)).sum())
    P = M / np.maximum(M.sum(1, keepdims=True), 1)
    a.imshow(P, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(CLASSES)):
            a.text(j, i, f"{P[i, j] * 100:.0f}\n({M[i, j]})" if M[i, j] else "0", ha="center", va="center",
                   fontsize=7.6, color="white" if P[i, j] > 0.55 else INK, fontweight="bold" if rows[i] == j else "normal",
                   linespacing=0.95)
    a.set_xticks(range(len(CLASSES))); a.set_xticklabels([SHORT[c] for c in CLASSES], fontsize=8.6)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([SHORT[CLASSES[r]] for r in rows] if show_y else [], fontsize=8.6)
    a.set_xlabel("Predicted", fontsize=9.5)
    if show_y:
        a.set_ylabel("True", fontsize=9.5)
    for s_ in a.spines.values():
        s_.set_visible(False)
    a.tick_params(length=0)
    ptitle(a, "", title, fontsize=10.5)


def s1():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.75), layout="constrained")
    for k, (a, run) in enumerate(zip(axs, CLS_RUNS)):
        _, Pc, y = F._ens(run)
        cm_panel(a, y, Pc.argmax(1), title=MLAB[run], show_y=k == 0)
    save(fig, 1)


def s2():
    from sklearn.metrics import roc_curve, roc_auc_score
    ens = {r: F._ens(r) for r in CLS_RUNS}
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.55), layout="constrained", sharey=True)
    for c, a in enumerate(axs):
        for r, (_, Pc, y) in ens.items():
            fpr, tpr, _ = roc_curve(y == c, Pc[:, c])
            auc = roc_auc_score(y == c, Pc[:, c])
            a.plot(np.clip(fpr, 1e-3, 1), tpr, color=MCOL[r], lw=1.8, label=f"{MLAB[r]}  {auc:.4f}")
        a.set_xscale("log"); a.set_xlim(1e-3, 1); a.set_ylim(0.6, 1.005)
        a.set_xlabel("FPR (log)", fontsize=9.5)
        if c == 0:
            a.set_ylabel("TPR")
        xgrid(a); ygrid(a)
        a.legend(loc="lower right", fontsize=6.8, handlelength=1.1, borderaxespad=0.1, title="AUC", title_fontsize=7)
        ptitle(a, "", CLAB[CLASSES[c]], fontsize=10.5)
    save(fig, 2)


def _ext_ens():
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    idx["dup"] = idx.dup_near.astype(str).str.lower().isin(["true", "1"])
    P = pd.read_csv(EXT / "external_preds.csv", low_memory=False)
    P = P[P.pred.notna() & (P.tag == "base")]
    pc = [f"p_{c}" for c in CLASSES]
    e = P.groupby(["model", "uid"])[pc].mean().reset_index().merge(
        idx[["uid", "source", "sequence", "kind", "label", "dup", "case", "pair_id", "img_path", "mask_path"]], on="uid")
    e["yhat"] = e[pc].values.argmax(1)
    return e


def s3():
    e = _ext_ens()
    mods = [("cls_resnet50", "cls_resnet50_base"), ("mt_base", "mt_base_base"), ("mt_ctam", "mt_ctam_base")]
    fig = plt.figure(figsize=(W2, 4.6), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.45, 0.75], hspace=0.05)
    top, bot = sf[0].subplots(1, 3), sf[1].subplots(1, 3)
    sf[0].suptitle(f"Four-class dataset, non-duplicate images (n = {int(((e.source == 'mendeley') & ~e.dup).sum() / max(e.model.nunique(), 1))})", x=0.0, ha="left", fontsize=11, fontweight="bold")
    _b = e[(e.source == "brats") & (e.sequence == "t1c")]
    print(f"  [S3] BraTS T1c: {_b.case.nunique()} patients, {_b[_b.kind == 'tumor'].case.nunique()} with glioma slices, "
          f"{_b.uid.nunique()} slices")
    sf[1].suptitle("BraTS 2021 T1c slices", x=0.0, ha="left", fontsize=11, fontweight="bold")
    for k, (m, key) in enumerate(mods):
        g = e[(e.model == m) & (e.source == "mendeley") & ~e.dup]
        cm_panel(top[k], g.label.values.astype(int), g.yhat.values, title=MLAB[key], show_y=k == 0)
        g = e[(e.model == m) & (e.source == "brats") & (e.sequence == "t1c")]
        cm_panel(bot[k], g.label.values.astype(int), g.yhat.values, rows=[0, NT], title="", show_y=k == 0)
    save(fig, 3)


def _model(name, run, dev):
    import torch
    models_mod = import_module("08_models")
    m = models_mod.build_model(name, pretrained=False).to(dev).eval()
    m.load_state_dict(torch.load(cfg.DIR_CKPT / f"P3_{run}_f0.pt", map_location=dev, weights_only=False)["model"])
    return m


def _to_t():
    import torch
    import albumentations as A
    clahe = A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
    mean = np.array(cfg.IMAGENET_MEAN, np.float32) * 255; std = np.array(cfg.IMAGENET_STD, np.float32) * 255

    def f(I, g):
        top, left, h, w = g
        crop = clahe(image=np.repeat(I[top:top + h, left:left + w, None], 3, 2))["image"]
        full = np.zeros((S, S, 3), np.uint8); full[top:top + h, left:left + w] = crop
        return torch.from_numpy(((full.astype(np.float32) - mean) / std).transpose(2, 0, 1))
    return f


def _features(model, paths, dev, bs=32):
    import torch
    layer, pick = ((model.encoder, lambda o: o[-1]) if hasattr(model, "encoder") else (model.net.layer4, lambda o: o))
    st = {}
    h = layer.register_forward_hook(lambda m, i, o: st.__setitem__("a", pick(o)))
    tt, out = _to_t(), []
    with torch.no_grad():
        for s_ in range(0, len(paths), bs):
            X = []
            for p in paths[s_:s_ + bs]:
                I, _, g = F.load_square(p)
                X.append(tt(I, g))
            model(torch.stack(X).to(dev))
            out.append(st["a"].float().mean((2, 3)).cpu().numpy())
    h.remove()
    return np.concatenate(out)


def s4(n_brats=150, seed=0):
    import torch
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = F.folds_df()
    te = f[f["P3_role"] == "test"] if "P3_role" in f else f

    seq = {}
    sp = RES / "seq_audit_labels_filled.csv"
    if sp.exists():
        sa = pd.read_csv(sp)
        sa = sa[[c for c in sa.columns if c in ("uid", "sequence")]].dropna()
        for u, s_ in zip(sa.uid, sa.sequence.astype(str).str.upper().str.strip()):
            seq[u] = "T1c" if (s_.startswith("T1C") or "CONTRAST" in s_ or "+C" in s_) and "NON" not in s_ else "noncontrast"
    aud = f[f.uid.isin(seq) & (f.class_name == "no_tumor") & ~f.uid.isin(te.uid)]
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    b = idx[(idx.source == "brats") & (idx.kind == "tumor")]
    both = b.groupby("pair_id").sequence.nunique()
    pids = both[both == 2].index.values
    pids = np.random.default_rng(seed).choice(pids, min(n_brats, len(pids)), replace=False)
    b = b[b.pair_id.isin(pids)]
    items = ([(p, "brisc", c, seq.get(u, "")) for p, c, u in zip(te.img_path, te.class_name, te.uid)]
             + [(p, "audit", "no_tumor", seq[u]) for p, u in zip(aud.img_path, aud.uid)]
             + [(p, "brats_" + s_, "glioma", s_) for p, s_ in zip(b.img_path, b.sequence)])
    D = pd.DataFrame(items, columns=["path", "src", "cls", "seq"])
    fig, axs = plt.subplots(2, 3, figsize=(W2, 5.0), layout="constrained")
    for r, (name, run) in enumerate([("mt_base", "mt_base_base"), ("cls_resnet50", "cls_resnet50_base")]):
        Z = _features(_model(name, run, dev), D.path.tolist(), dev)
        Z = PCA(50, random_state=seed).fit_transform((Z - Z.mean(0)) / (Z.std(0) + 1e-6))
        E = TSNE(2, perplexity=30, init="pca", random_state=seed, learning_rate="auto").fit_transform(Z)
        D["x"], D["y"] = E[:, 0], E[:, 1]
        bg = dict(s=5, color="#dcdbd6", lw=0, zorder=1)
        a = axs[r, 0]
        for c in CLASSES:
            h = D[(D.src == "brisc") & (D.cls == c)]
            a.scatter(h.x, h.y, s=6, lw=0, color=F.CLS_COL[c], label=CLAB[c], alpha=0.85)
        a2 = axs[r, 1]
        a2.scatter(D.x, D.y, **bg)
        h = D[D.src == "brats_t1c"]; a2.scatter(h.x, h.y, s=9, lw=0, color=C["blue"], label="BraTS glioma, T1c", zorder=3)
        h = D[D.src == "brats_t1"]; a2.scatter(h.x, h.y, s=9, lw=0, color=C["orange"], label="Same slices, noncontrast T1", zorder=3)
        a3 = axs[r, 2]
        a3.scatter(D.x, D.y, **bg)
        nt = D[(D.cls == "no_tumor") & D.src.isin(["brisc", "audit"])]
        h = nt[nt.seq == "noncontrast"]; a3.scatter(h.x, h.y, s=12, lw=0, color=C["orange"], label="No tumor, noncontrast", zorder=3)
        h = nt[nt.seq == "T1c"]; a3.scatter(h.x, h.y, s=16, lw=0.6, ec="white", color=C["blue"], label="No tumor, T1c", zorder=4)
        h = D[(D.src == "brisc") & (D.cls != "no_tumor")]
        for ax_ in axs[r]:
            ax_.set_xticks([]); ax_.set_yticks([])
            for s_ in ax_.spines.values():
                s_.set_visible(False)
        axs[r, 0].set_ylabel(MLAB[run], fontsize=11, fontweight="bold")
        if r == 0:
            ptitle(axs[0, 0], "a", "BRISC test, by class", fontsize=10.5)
            ptitle(axs[0, 1], "b", "BraTS paired slices", fontsize=10.5)
            ptitle(axs[0, 2], "c", "Audited no-tumor images", fontsize=10.5)
        else:
            for ax_ in axs[r]:
                ax_.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38 if ax_ is axs[r, 0] else -0.27),
                           ncol=2, fontsize=7.6, markerscale=2.2, handletextpad=0.2, columnspacing=0.6)
        del Z
    save(fig, 4)


def s5():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.9), layout="constrained")
    for ax_, keys, col_pat, yl, lt in ((a, ALL_CLS, "val_balanced_accuracy", "Validation balanced accuracy", "a"),
                                       (b, SEG_RUNS, "lesion_dice", "Validation lesion Dice", "b")):
        for run in keys:
            H = []
            for f_ in range(5):
                p = RES / f"history_P3_{run}_f{f_}.csv"
                if not p.exists():
                    continue
                h = pd.read_csv(p)
                col = next((c for c in h.columns if c.startswith("val") and col_pat in c and not c.endswith("_std")), None)
                if col:
                    H.append(h.set_index("epoch")[col])
            if not H:
                continue
            M = pd.concat(H, axis=1)
            m, s_ = M.mean(1), M.std(1)
            ax_.plot(m.index, m.values, color=MCOL[run], lw=1.6, ls="--" if MHAT[run] else "-", label=MLAB[run])
            ax_.fill_between(m.index, m - s_, m + s_, color=MCOL[run], alpha=0.12, lw=0)
        ax_.set_xlabel("Epoch"); ax_.set_ylabel(yl); ygrid(ax_)
        lo = 0.8 if lt == "a" else 0.6
        ax_.set_ylim(lo, 1.0 if lt == "a" else 0.95)
        ax_.legend(fontsize=7.6, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
        ptitle(ax_, lt, "Classification" if lt == "a" else "Segmentation")
    save(fig, 5)


def s6(n_int=4, seed=5, q=(0.25, 0.50, 0.75)):


    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = F.folds_df()
    te = f[(f["P3_role"] == "test") & (f.class_name != "no_tumor")] if "P3_role" in f else f[f.class_name != "no_tumor"]
    rng = np.random.default_rng(seed)
    cases = []
    for c in ["glioma", "meningioma", "pituitary", None][:n_int]:
        sub = te if c is None else te[te.class_name == c]
        for _, r in sub.sample(frac=1, random_state=int(rng.integers(1e6))).iterrows():
            I, M, g = F.load_square(r.img_path, F.mask_path_of(r))
            if M.mean() >= 0.008 and r.uid not in [x[0] for x in cases]:
                cases.append((r.uid, "BRISC " + CLAB[r.class_name].lower(), I, M, g, None)); break
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    P = pd.read_csv(EXT / "external_preds.csv", low_memory=False)
    P = P[(P.model == "mt_base") & (P.tag == "base") & P.dice.notna()]
    d_ens = P.groupby("uid").dice.mean()
    bt = idx[(idx.source == "brats") & (idx.sequence == "t1c") & (idx.kind == "tumor") & idx.uid.isin(d_ens.index)].copy()
    bt["d"] = d_ens.reindex(bt.uid).values
    bt = bt.sort_values("d").reset_index(drop=True)
    for qq in q:
        r = bt.iloc[min(int(round(qq * (len(bt) - 1))), len(bt) - 1)]
        I, M, g = F.load_square(r.img_path, r.mask_path)
        cases.append((r.uid, f"BraTS P{int(qq * 100)}", I, M, g, r.d))
    tt = _to_t()
    X = torch.stack([tt(c[2], c[4]) for c in cases]).to(dev)
    preds = {}
    for name, run in [("mt_base", "mt_base_base"), ("seg_unet_r50", "seg_unet_r50_base")]:
        acc, nf = 0, 0
        for fo in range(5):
            ck = cfg.DIR_CKPT / f"P3_{run}_f{fo}.pt"
            if not ck.exists():
                continue
            models_mod = import_module("08_models")
            m = models_mod.build_model(name, pretrained=False).to(dev).eval()
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
            with torch.no_grad():
                acc = acc + torch.sigmoid(m(X)["logits_seg"].float()).cpu().numpy()[:, 0]
            nf += 1
            del m
        preds[run] = (acc / max(nf, 1)) > 0.5
    fig, axs = plt.subplots(2, len(cases), figsize=(W2, W2 / len(cases) * 2 + 0.55),
                            gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for j, (u, lab, I, M, g, dref) in enumerate(cases):
        for i, run in enumerate(["mt_base_base", "seg_unet_r50_base"]):
            a = axs[i, j]
            Pm = preds[run][j].astype(bool)
            F.img_ax(a, I)
            F.contour(a, M, C["yellow"], lw=1.3)
            F.contour(a, Pm, MCOL[run], lw=1.3, ls=(0, (2, 1.2)))
            d = 2 * (Pm & M).sum() / max(Pm.sum() + M.sum(), 1)
            F.tag(a, f"Dice {d:.2f}", "bl", 7.2)
            if i == 0:
                a.set_title(lab.replace(" ", "\n", 1), loc="center", fontsize=8.6, fontweight="bold", pad=3)
                if dref is not None:
                    print(f"  [S6] {u}: saved ensemble Dice {dref:.3f} | shown {d:.3f}")
            if j == 0:
                a.set_ylabel(MLAB[run], fontsize=10, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=C["yellow"], lw=2, label="Reference"),
                        Line2D([], [], color=MCOL["mt_base_base"], lw=2, ls="--", label="MT-Base"),
                        Line2D([], [], color=MCOL["seg_unet_r50_base"], lw=2, ls="--", label="U-Net")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    save(fig, 6)


def _int_dice(run):
    ds = []
    for f_ in range(5):
        p, z = RES / f"segper_P3_{run}_f{f_}_test.csv", RES / f"preds_P3_{run}_f{f_}_test.npz"
        if p.exists() and z.exists():
            d = pd.read_csv(p)
            d.index = np.load(z, allow_pickle=True)["uids"].astype(str)
            ds.append(d.loc[d.gt_empty.astype(str).str.lower() == "false", "dice"])
    return pd.concat(ds, axis=1).mean(1).dropna().values if ds else np.array([])


def s7():
    e = _ext_ens_seg()
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 3.1), layout="constrained")
    runs = [r for r in SEG_RUNS]
    for ax_, data, ttl, lt in ((a, {r: _int_dice(r) for r in runs}, "Patient-grouped test set", "a"),
                               (b, e, "BraTS T1c glioma slices", "b")):
        keys = [r for r in runs if r in data and len(data[r])]
        y = np.arange(len(keys))
        bp = ax_.boxplot([data[r] for r in keys], positions=y, vert=False, widths=0.55, patch_artist=True,
                         showfliers=False, medianprops=dict(color="white", lw=1.8),
                         whiskerprops=dict(color=INK3), capprops=dict(color=INK3))
        for patch, r in zip(bp["boxes"], keys):
            patch.set_facecolor(MCOL[r]); patch.set_edgecolor(MCOL[r])
            if MHAT[r]:
                patch.set_hatch("///"); patch.set_edgecolor("white")
        for yi, r in zip(y, keys):
            ax_.text(1.02, yi, f"{np.median(data[r]):.3f}", va="center", fontsize=8.4, fontweight="bold")
        ax_.set_yticks(y); ax_.set_yticklabels([MLAB[r] for r in keys], fontsize=9)
        ax_.invert_yaxis(); ax_.set_xlim(0, 1.13); ax_.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_.set_xlabel("Per-image lesion Dice (fold mean)"); xgrid(ax_)
        ax_.tick_params(axis="y", length=0); ax_.spines["left"].set_visible(False)
        ptitle(ax_, lt, ttl)
    save(fig, 7)


def _ext_ens_seg():
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    P = pd.read_csv(EXT / "external_preds.csv", low_memory=False)
    P = P[P.dice.notna()].merge(idx[["uid", "source", "sequence", "kind"]], on="uid")
    P = P[(P.source == "brats") & (P.sequence == "t1c") & (P.kind == "tumor")]
    g = P.groupby(["model", "tag", "uid"]).dice.mean().reset_index()
    return {f"{m}_{t}": v.dice.values for (m, t), v in g.groupby(["model", "tag"])}


def s8():
    cf = rcsv("conformal_P3_summary.csv")
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.6), layout="constrained", sharex=True)
    for j, meth in enumerate(["LAC", "APS", "Mondrian"]):
        for run in ["cls_resnet50_base", "mt_base_base", "mt_ctam_base"]:
            h = cf[(cf.run == run) & (cf.method == meth)].sort_values("alpha")
            if not len(h):
                continue
            kw = dict(color=MCOL[run], lw=1.8, marker="o", ms=5, mec="white", mew=1, label=MLAB[run])
            xi = [ [0.01, 0.02, 0.05, 0.10].index(round(v, 2)) for v in h.alpha ]
            axs[0, j].plot(xi, h.coverage_mean, **kw)
            axs[1, j].plot(xi, h.mean_size_mean, **kw)
        al = np.array([0.01, 0.02, 0.05, 0.10])
        axs[0, j].plot(range(4), 1 - al, color=INK3, ls="--", lw=1, label="Nominal 1 − α")
        axs[0, j].set_ylim(0.88, 1.0); axs[1, j].set_ylim(0.85, None)
        axs[1, j].axhline(1, color=INK3, lw=0.8, ls=":")
        axs[1, j].set_xlabel("α"); axs[1, j].set_xticks(range(4)); axs[1, j].set_xticklabels(["0.01", "0.02", "0.05", "0.10"])
        for i in range(2):
            ygrid(axs[i, j])
        ptitle(axs[0, j], "abc"[j], meth, fontsize=10.5)
    axs[0, 0].set_ylabel("Marginal coverage"); axs[1, 0].set_ylabel("Mean set size")
    axs[0, 0].legend(fontsize=7.6, loc="lower left")
    save(fig, 8)


def s9():
    dc = rcsv("external_dup_check.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.8), layout="constrained")
    for src, col, lab in (("brats", C["blue"], "BraTS 2021"), ("mendeley", C["orange"], "Four-class dataset")):
        h = dc[dc.source == src]
        w = np.ones(len(h)) / len(h) * 100
        a.hist(h.dhash_dist, bins=np.arange(0, 131, 4), weights=w, color=col, alpha=0.75, label=f"{lab} (n = {len(h)})")
        b.hist(h.ncc, bins=np.linspace(-0.2, 1, 49), weights=w, color=col, alpha=0.75, label=lab)
    a.axvline(10.5, color=INK, lw=1, ls="--"); a.text(12, a.get_ylim()[1] * 0.92, "≤ 10: duplicate", fontsize=8.2)
    b.axvline(0.95, color=INK, lw=1, ls="--"); b.text(0.93, b.get_ylim()[1] * 0.92, "0.95", fontsize=8.2, ha="right")
    a.set_xlabel("Hamming distance to nearest BRISC image"); b.set_xlabel("NCC with nearest BRISC image")
    a.set_ylabel("Images (%)"); ygrid(a); ygrid(b)
    b.legend(*a.get_legend_handles_labels(), fontsize=8.4, loc="upper left")
    ptitle(a, "a", "Difference hash (256 bit)"); ptitle(b, "b", "Pixel verification")
    save(fig, 9)


def s10():
    h = rcsv("inpainter_v2_history.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.6), layout="constrained")
    a.plot(h.epoch, h.val_psnr_hole, color=C["blue"], lw=1.8); a.set_ylabel("PSNR in hole (dB)")
    b.plot(h.epoch, h.val_seam_gap, color=C["orange"], lw=1.8); b.set_ylabel("Seam intensity gap")
    for ax_, lt, t in ((a, "a", "Reconstruction quality"), (b, "b", "Boundary continuity")):
        ax_.set_xlabel("Epoch"); ygrid(ax_); ptitle(ax_, lt, t)
    save(fig, 10)


def md_table(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.{floatfmt}f}"))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    return "\n".join([head, "|" + "---|" * len(df.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])


def tables():
    from sklearn.metrics import roc_auc_score
    L = ["# Supplementary tables (auto-generated)\n"]
    rows = []
    for run in ALL_CLS:
        try:
            _, Pc, y = F._ens(run)
        except Exception:
            continue
        if y is None or np.ndim(Pc) != 2:
            continue
        p = Pc.argmax(1)
        for c, cn in enumerate(CLASSES):
            tp = ((p == c) & (y == c)).sum(); fn = ((p != c) & (y == c)).sum()
            fp = ((p == c) & (y != c)).sum(); tn = ((p != c) & (y != c)).sum()
            sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            rows.append(dict(Model=MLAB[run], Class=CLAB[cn], n=int((y == c).sum()), Sensitivity=sens,
                             Specificity=spec, Precision=prec, F1=2 * prec * sens / max(prec + sens, 1e-9),
                             AUC=roc_auc_score(y == c, Pc[:, c])))
    L += ["## Table S4. Per-class performance on the P3 test set (fold ensemble, temperature-scaled)\n",
          md_table(pd.DataFrame(rows), 4), ""]
    for name, f_, title in (("stats_P3_delong_perclass.csv", None, "Table S5. Per-class AUC comparisons (DeLong; Holm-adjusted)"),
                            ("stats_P3_cls_pairs.csv", None, "Table S6. Pairwise classification comparisons (fold ensemble)"),
                            ("stats_P3_seg_pairs.csv", None, "Table S7. Pairwise segmentation comparisons (Wilcoxon; Holm-adjusted)")):
        try:
            d = rcsv(name)
            for c in ("model_a", "model_b"):
                if c in d:
                    d[c] = d[c].map(lambda v: MLAB.get(v, v))
            L += [f"## {title}\n", md_table(d, 4), ""]
        except Exception as e:
            L.append(f"({name} skipped: {e})\n")
    try:
        cf = rcsv("conformal_P3_summary.csv")
        cols = ["run", "method", "alpha", "coverage_mean", "mean_size_mean", "singleton_mean", "empty_mean", "detect_ambig_mean"] + \
               [f"cov_{c}_mean" for c in CLASSES]
        cf = cf[[c for c in cols if c in cf]].sort_values(["run", "method", "alpha"])
        cf["run"] = cf.run.map(lambda v: MLAB.get(v, v))
        L += ["## Table S8. Conformal prediction on the P3 test set, all methods and α (mean over folds)\n", md_table(cf, 3), ""]
    except Exception as e:
        L.append(f"(Table S8 skipped: {e})\n")
    try:
        es = rcsv("external_summary.csv")
        keep = ["model"] + [c for c in es.columns if c.startswith("mend_sens") or c.startswith("brats_t1_")]
        es = es[es.tag == "base"][[c for c in keep if c in es]]
        L += ["## Table S9. External per-class sensitivity (four-class collection, non-duplicate images) and BraTS noncontrast-T1 results (mean / SD over folds)\n",
              md_table(es, 3), ""]
        cp = rcsv("external_contrast_patient.csv")
        L += ["## Table S10. Contrast-removal effect at patient level (BraTS; cluster bootstrap, Wilcoxon)\n", md_table(cp, 4), ""]
    except Exception as e:
        L.append(f"(Tables S9–S10 skipped: {e})\n")
    try:
        ls = rcsv("linkage_sensitivity.csv", GRP)
        L += ["## Table S11. Pseudo-patient linkage sensitivity\n", md_table(ls, 3), ""]
    except Exception as e:
        L.append(f"(Table S11 skipped: {e})\n")
    L += ["## Table S12. Training configuration\n",
          "| Setting | Value |", "|---|---|",
          "| Input | longest side 224 px, CLAHE (8×8 tiles), zero-pad to 224×224, ImageNet normalisation |",
          "| Optimiser | AdamW, lr 3×10⁻⁴, weight decay 10⁻⁴, one-cycle schedule |",
          "| Batch / epochs | 8 / ≤ 40, early stopping patience 10 (mean of val BA and lesion Dice) |",
          "| Precision / clipping | mixed precision / gradient norm 1.0 |",
          "| Segmentation loss | Dice + BCE + 0.2 Sobel boundary; deep supervision 0.3; empty-mask 0.2 |",
          "| Classification loss | CE (label smoothing 0.05) + 0.1 SupCon (τ = 0.1) |",
          "| Task weighting | homoscedastic uncertainty (Kendall et al., 2018) |",
          "| Evidence-mass loss | weight 0.1 |", "| Counterfactual loss | relative hinge, weight 0.1, 5 warm-up epochs |",
          "| Encoder | EfficientNetV2-S (in21k → in1k) |", "| Hardware / seeds | NVIDIA T4 / 42 + fold |", ""]
    p = OUTS / "supplementary_tables.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ tables → {p}")


FIGS = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10}
NEEDS_MODEL = {4, 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    todo = [n for n in (a.only or list(FIGS)) if not (a.no_model and n in NEEDS_MODEL)]
    errs = {}
    for n in todo:
        try:
            FIGS[n]()
        except Exception as e:
            errs[n] = traceback.format_exc(limit=3)
            print(f"  ✗ Figure S{n}: {type(e).__name__}: {e}")
    if not a.only:
        try:
            tables()
        except Exception:
            errs["tables"] = traceback.format_exc(limit=3)
    if errs:
        print("\n========== Errors (send this section) ==========")
        for k, v in errs.items():
            print(f"--- {k}\n{v}")
    print(f"\n  output folder: {OUTS}")


if __name__ == "__main__":
    main()
