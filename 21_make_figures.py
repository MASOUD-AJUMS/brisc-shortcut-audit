import argparse, glob, json, os, re, traceback
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

cfg = import_module("00_config")
RES, GRP = cfg.DIR_RESULTS, cfg.DIR_GROUPS
OUTF = cfg.DIR_FIGS / "paper"
OUTF.mkdir(parents=True, exist_ok=True)
CLASSES = list(cfg.CLASSES)
NT = CLASSES.index("no_tumor")
S = cfg.IMG_SIZE


INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURF = "#e6e5e0", "#ffffff"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948")
SHAM = "#b9b8b2"

MODELS = [
    ("cls_resnet50_base", "ResNet-50", C["orange"], None),
    ("seg_unet_r50_base", "U-Net", C["violet"], None),
    ("mt_base_base", "MT-Base", C["blue"], None),
    ("mt_aspp_base", "MT-ASPP", C["yellow"], None),
    ("mt_ctam_base", "MT-CTAM", C["aqua"], None),
    ("mt_base_ev", "MT-Base + ev.", C["blue"], "////"),
    ("mt_ctam_ev", "MT-CTAM + ev.", C["aqua"], "////"),
]
MLAB = {k: l for k, l, _, _ in MODELS}
MCOL = {k: c for k, _, c, _ in MODELS}
MHAT = {k: h for k, _, _, h in MODELS}
CLAB = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary", "no_tumor": "No tumor"}
CLS_COL = {"glioma": C["blue"], "meningioma": C["orange"], "pituitary": C["aqua"], "no_tumor": C["violet"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "font.weight": "normal",
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelsize": 10.5, "axes.labelweight": "bold", "axes.labelcolor": INK, "axes.edgecolor": INK3,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 9.5, "legend.frameon": False, "legend.handlelength": 1.2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "lines.linewidth": 2, "lines.markersize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
W2 = 7.2


def ygrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True); ax.set_axisbelow(True)


def ptitle(ax, l, t, **kw):

    ax.set_title(f"{l}   {t}" if l else t, **kw)


def save(fig, n):
    p = OUTF / f"Figure{n}"
    fig.savefig(f"{p}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(f"{p}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  ✓ Figure {n} → {p}.png")


def rcsv(name, d=RES, **kw):
    p = d / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, **kw)


def folds_df():
    return rcsv("folds.csv", GRP)


def present(keys, avail):
    return [k for k in keys if k in set(avail)]


def bar_end_labels(ax, bars, fmt, inside=False, dy=0.01, color=INK, size=9):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + (-dy if inside else dy), fmt(h), ha="center",
                va="top" if inside else "bottom", fontsize=size, color=color, fontweight="bold")


def load_square(img_path, mask_path=None):

    img = np.array(Image.open(img_path).convert("L"))
    m = np.zeros(img.shape, np.uint8)
    if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
        m = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
    h, w = img.shape
    s = S / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    ir = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mr = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (S - nh) // 2, (S - nw) // 2
    I = np.zeros((S, S), np.uint8); M = np.zeros((S, S), np.uint8)
    I[top:top + nh, left:left + nw] = ir; M[top:top + nh, left:left + nw] = mr
    return I, M.astype(bool), (top, left, nh, nw)


def mask_path_of(r):
    for k in ("seg_mask_path", "mask_path"):
        v = r.get(k) if hasattr(r, "get") else None
        if isinstance(v, str) and v and v != "nan":
            return v
    return None


def contour(ax, m, color, lw=1.4, ls="-"):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        c = np.vstack([c[:, 0], c[:1, 0]])
        ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, ls=ls)


def img_ax(ax, I):
    ax.imshow(I, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tag(ax, s, loc="bl", size=8.5):
    x, ha = (0.04, "left") if loc[1] == "l" else (0.96, "right")
    y, va = (0.04, "bottom") if loc[0] == "b" else (0.96, "top")
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))


def plane_of(uid):
    m = re.search(r"_(ax|co|sa)_", uid)
    return {"ax": "Axial", "co": "Coronal", "sa": "Sagittal"}.get(m.group(1) if m else "", "")


def fig1():
    fig = plt.figure(figsize=(W2, 3.75))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
    cols = [
        ("1  Data audit", C["blue"], [("BRISC2025", "6,000 T1c slices"), ("Exact duplicates", "pixel MD5, pHash"),
                                      ("Pseudo-patients", "DINOv2 + radiologist"),
                                      ("Shortcut & sequence", "non-anatomical probe")]),
        ("2  Protocols", C["orange"], [("P1  official", "5,000 / 1,000"), ("P2  purged test", "502 clean / 498 cont."),
                                       ("P3  grouped", "4,795 / 1,205, 5 folds")]),
        ("3  Models", C["aqua"], [("ResNet-50", "classification only"), ("U-Net", "segmentation only"),
                                  ("Multi-task", "MT-Base, ASPP, CTAM"), ("Lesion-centered loss", "evidence, counterfactual")]),
        ("4  Evaluation", C["violet"], [("Leakage effect", "DiD vs. P3 models"), ("Lesion reliance", "Grad-CAM + edits"),
                                        ("Calibration", "temperature, conformal"), ("External validation", "BraTS T1c/T1, four-class")]),
    ]
    x0, cw, gap = 0.8, 22.7, 2.4
    top, hh, bh, bg = 45.5, 5.6, 8.6, 1.9
    body_h = 4 * bh + 3 * bg
    for i, (title, col, items) in enumerate(cols):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, top), cw, hh, boxstyle="round,pad=0,rounding_size=1.1", fc=col, ec="none"))
        ax.text(x + cw / 2, top + hh / 2, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        n = len(items)
        y = top - 2.4 - (body_h - (n * bh + (n - 1) * bg)) / 2 - bh
        for head, sub in items:
            ax.add_patch(FancyBboxPatch((x, y), cw, bh, boxstyle="round,pad=0,rounding_size=1.1",
                                        fc="#f6f6f4", ec=col, lw=1.3))
            ax.text(x + cw / 2, y + bh * 0.64, head, ha="center", va="center", fontsize=9.2, fontweight="bold", color=INK)
            ax.text(x + cw / 2, y + bh * 0.29, sub, ha="center", va="center", fontsize=8.2, color=INK2)
            y -= bh + bg
        if i < len(cols) - 1:
            xa = x + cw + 0.25
            ax.annotate("", xy=(xa + gap - 0.5, top + hh / 2), xytext=(xa, top + hh / 2),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55", color=INK3, lw=1.5))
    save(fig, 1)


def fig2():
    f = folds_df()
    te = f[f["P3_role"] == "test"] if "P3_role" in f else f
    te = te.assign(plane=te.uid.map(plane_of))
    planes = ["Axial", "Coronal", "Sagittal"]
    fig, axs = plt.subplots(4, 3, figsize=(W2 * 0.62, W2 * 0.62 * 4 / 3 * 1.02),
                            gridspec_kw=dict(wspace=0.03, hspace=0.03))
    rng = np.random.default_rng(7)
    for i, c in enumerate(CLASSES):
        for j, pl in enumerate(planes):
            a = axs[i, j]
            sub = te[(te.class_name == c) & (te.plane == pl)]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if not len(sub):
                continue

            cand = sub.sample(min(40, len(sub)), random_state=int(rng.integers(1e6)))
            pick = None
            for _, r in cand.iterrows():
                I, M, _ = load_square(r.img_path, mask_path_of(r))
                if c == "no_tumor" or M.mean() >= 0.01:
                    pick = (I, M); break
            if pick is None:
                r = cand.iloc[0]; pick = load_square(r.img_path, mask_path_of(r))[:2]
            img_ax(a, pick[0])
            if pick[1].any():
                contour(a, pick[1], C["yellow"], lw=1.3)
            if i == 0:
                a.set_title(pl, loc="center", fontsize=11, fontweight="bold", pad=5)
            if j == 0:
                a.set_ylabel(CLAB[c], fontsize=11, fontweight="bold", labelpad=6)
    save(fig, 2)


def band_precision_panel(a):
    bp = rcsv("band_precision.csv", GRP)
    cols = {c.lower(): c for c in bp.columns}
    pc = next((cols[c] for c in cols if "prec" in c and "lo" not in c and "hi" not in c), None)
    lo = next((cols[c] for c in cols if c == "ci_low" or ("lo" in c or "lower" in c) and c not in ("band_lo", "sim_min")
               and ("ci" in c or "wil" in c or "lower" in c)), None)
    hi = next((cols[c] for c in cols if c == "ci_high" or ("hi" in c or "upper" in c) and c not in ("band_hi", "sim_max")
               and ("ci" in c or "wil" in c or "upper" in c)), None)
    lab = None
    if {"sim_min", "sim_max"} <= set(cols):
        lab = ["identical" if lo_ >= 0.9999 else f"{lo_:.3f}–{hi_:.3f}"
               for lo_, hi_ in zip(bp[cols["sim_min"]], bp[cols["sim_max"]])]
    elif {"band_lo", "band_hi"} <= set(cols):
        lab = [f"{l:g}–{h:g}" for l, h in zip(bp[cols["band_lo"]].round(3), bp[cols["band_hi"]].round(3))]
    else:
        lab = bp[bp.columns[0]].astype(str).tolist()
    if pc is None:
        raise KeyError(f"band_precision.csv columns: {list(bp.columns)}")
    y = bp[pc].values.astype(float)
    x = np.arange(len(y))
    a.plot(x, y, "-", color=C["blue"], lw=2, zorder=2)
    a.scatter(x, y, s=46, color=C["blue"], zorder=3, edgecolor="white", linewidth=1.5)
    if lo and hi:
        a.vlines(x, bp[lo], bp[hi], color=C["blue"], lw=1.6, alpha=0.45, zorder=1)
    a.axhline(1.0, color=INK3, lw=0.8, ls=":")
    a.set_xticks(x); a.set_xticklabels(lab, rotation=40, ha="right", fontsize=8.5)
    a.set_ylim(-0.03, 1.06); a.set_ylabel("Same-study precision")
    a.set_xlabel("DINOv2 cosine similarity band"); a.axvline(len(y) - 2.5, color=C["red"], lw=1, ls="--"); ygrid(a)


def fig3():
    f = folds_df()
    fig = plt.figure(figsize=(W2, 6.6), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1, 0.95], hspace=0.06)
    a, b = sf[0].subplots(1, 2, gridspec_kw=dict(width_ratios=[1.15, 1]))
    c, d = sf[1].subplots(1, 2, gridspec_kw=dict(width_ratios=[0.85, 1.15]))


    t = f[f["P1_role"] == "test"].copy()
    md5 = rcsv("pixel_md5_cache.csv").set_index("uid")["md5"]
    train_md5 = set(f.loc[f["P1_role"] != "test", "uid"].map(md5))
    t["exact"] = t.uid.map(md5).isin(train_md5)
    t["shared"] = t["P2_role"] != "test"
    g = t.groupby("class_name").agg(exact=("exact", "mean"), shared=("shared", "mean")).reindex(CLASSES)
    x = np.arange(4); w = 0.38
    b1 = a.bar(x - w / 2 - 0.01, g.exact * 100, w, color=C["orange"], label="Pixel-identical")
    b2 = a.bar(x + w / 2 + 0.01, g.shared * 100, w, color=C["blue"], label="Shares pseudo-patient")
    for bars in (b1, b2):
        bar_end_labels(a, bars, lambda v: f"{v:.0f}", dy=1.5, size=8.5)
    a.set_xticks(x); a.set_xticklabels(["Glioma", "Menin-\ngioma", "Pituitary", "No\ntumor"], fontsize=9)
    a.set_ylabel("Official test images (%)"); a.set_ylim(0, 105); ygrid(a)
    a.legend(loc="upper right", fontsize=8.6)
    ptitle(a, "a", "Test-set contamination")


    try:
        band_precision_panel(b)
    except Exception as e:
        b.text(0.5, 0.5, f"band_precision.csv\n{e}", ha="center", va="center", fontsize=8, transform=b.transAxes)
    ptitle(b, "b", "Same-study precision by band")


    sb = rcsv("shortcut_baseline.csv")
    sb = sb[sb.eval_on == "test_full"]
    best = sb.sort_values("balanced_accuracy").groupby("feature_set").tail(1)
    best = best.assign(code=best.feature_set.str.extract(r"^(S\d+)")[0].fillna(best.feature_set)).sort_values("code")
    names = {"size": "Size", "intensity": "Intensity", "sharp": "Sharpness", "background": "Background",
             "all": "All nine", "brain": "Brain area", "snr": "SNR"}
    def nm(s):
        for k, v in names.items():
            if k in s.lower():
                return v
        return re.sub(r"[^\x00-\x7F]+", "", s).strip("_ ") or s
    labs = [nm(s) for s in best.feature_set]
    y = np.arange(len(best))
    c.barh(y, best.balanced_accuracy * 100, 0.62, color=C["violet"])
    for yi, v in zip(y, best.balanced_accuracy * 100):
        c.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=8.8, fontweight="bold")
    c.axvline(25, color=INK3, lw=1, ls="--")
    c.text(25, -0.75, "chance", fontsize=8.3, color=INK2, ha="center", va="bottom")
    c.set_yticks(y); c.set_yticklabels(labs, fontsize=9); c.set_ylim(len(y) - 0.5, -0.95)
    c.set_xlim(0, 112); c.set_xticks([0, 25, 50, 75, 100]); c.set_xlabel("Balanced accuracy (%)"); xgrid(c)
    c.tick_params(axis="y", length=0); c.spines["left"].set_visible(False)
    ptitle(c, "c", "Shortcut probe")


    sq = rcsv("seq_audit_summary_reader2.csv").set_index("class_name").reindex(CLASSES)
    order = [("T1c", "T1c", C["blue"]), ("T1", "T1", C["orange"]), ("FLAIR", "FLAIR", C["aqua"]),
             ("T2", "T2", C["yellow"]), ("other", "Other", SHAM)]
    left = np.zeros(4); y = np.arange(4)
    for k, lab, col in order:
        v = sq[k].fillna(0).values * 100 if k in sq else np.zeros(4)
        d.barh(y, v, 0.62, left=left, color=col, label=lab, edgecolor="white", linewidth=1.5)
        for yi, (l0, vi) in enumerate(zip(left, v)):
            if vi >= 9:
                d.text(l0 + vi / 2, yi, f"{vi:.0f}", ha="center", va="center", fontsize=8.8,
                       fontweight="bold", color="white" if col not in (C["yellow"], SHAM) else INK)
        left += v
    d.set_yticks(y); d.set_yticklabels([CLAB[c] for c in CLASSES], fontsize=9.5); d.invert_yaxis()
    d.set_xlim(0, 100); d.set_xlabel("Images per class (%)")
    d.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.01, 0.99), fontsize=8.6, handlelength=0.9,
             columnspacing=0.9, handletextpad=0.4, borderaxespad=0)
    d.spines["left"].set_visible(False); d.tick_params(axis="y", length=0)
    ptitle(d, "d", "Blinded sequence audit", pad=22)
    save(fig, 3)


def fig4():
    model = "mt_base_base"
    pc = rcsv("leakage_effect_P1_perclass.csv"); pc = pc[pc.model == model]
    sm = rcsv("leakage_effect_P1_summary.csv"); sm = sm[sm.model == model]
    ts = rcsv("leakage_effect_P1_tests.csv"); ts = ts[(ts.model == model) & ts.test.str.startswith("DiD_")]
    SUB = [("clean", "Clean", C["blue"], "o"), ("contaminated", "Contaminated", C["orange"], "s"),
           ("exact_dup", "Exact duplicate", C["red"], "D")]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 3.05), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[1.15, 0.95, 1.1]))
    cls = [k for k in CLASSES if k in set(pc.cls)]
    for k, (s_, lab, col, mk) in enumerate(SUB):
        h = pc[pc.subset == s_].set_index("cls").reindex(cls)
        if h.ens_mean_ptrue.notna().any():
            a.scatter(h.ens_mean_ptrue, np.arange(len(cls)) + (k - 1) * 0.22, s=50, marker=mk, color=col,
                      edgecolor="white", linewidth=1.1, label=lab, zorder=3)
    a.set_yticks(range(len(cls))); a.set_yticklabels([CLAB[c] for c in cls], fontsize=9.5); a.invert_yaxis()
    a.set_xlabel("Mean true-class probability"); xgrid(a)
    a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    ptitle(a, "a", "Confidence")

    sm2 = sm.set_index("subset")
    rows = [(lab, sm2.loc[s_, "dice_mean"], sm2.loc[s_].get("dice_sd", np.nan), col) for s_, lab, col, _ in SUB
            if s_ in sm2.index and pd.notna(sm2.loc[s_, "dice_mean"])]
    y = np.arange(len(rows))
    b.barh(y, [r[1] for r in rows], 0.6, color=[r[3] for r in rows])
    for yi, r in zip(y, rows):
        b.text(r[1] + 0.02, yi, f"{r[1]:.3f}", va="center", fontsize=9, fontweight="bold")
    b.set_yticks(y); b.set_yticklabels([r[0].replace(" ", "\n") for r in rows], fontsize=9.5); b.invert_yaxis()
    b.set_xlim(0, 1.2); b.set_xticks([0, 0.5, 1.0]); b.set_xlabel("Lesion Dice"); xgrid(b)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    ptitle(b, "b", "Segmentation")

    lab = {"NLL": "NLL", "1-p_true": "1 − p(true)", "Dice": "Lesion Dice"}
    ts = ts.assign(score=ts.test.str.extract(r"DiD_(.+?)_P1minusP3")[0])
    ts = ts.set_index("score").reindex([k for k in lab if k in set(ts.score)])
    eff = ts["effect_class_std"] if "effect_class_std" in ts else ts["effect"]
    y = np.arange(len(ts))
    for yi, (e, lo, hi, p) in enumerate(zip(eff, ts.ci95_lo, ts.ci95_hi, ts.p)):
        c.hlines(yi, lo, hi, color=C["blue"], lw=2.4)
        c.scatter([e], [yi], s=62, color=C["blue"], edgecolor="white", linewidth=1.4, zorder=3)
        c.text(e, yi - 0.2, ("P < .001" if p < 0.001 else f"P = {p:.3f}".replace("0.", ".", 1)), ha="center", va="bottom", fontsize=8.4, color=INK2, zorder=5,
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    c.axvline(0, color=INK3, lw=1, ls="--")
    c.set_yticks(y); c.set_yticklabels([lab[k] for k in ts.index], fontsize=9.5)
    c.set_ylim(len(y) - 0.5, -0.75)
    c.set_xlabel("DiD (P1 − P3 model)"); xgrid(c)
    c.tick_params(axis="y", length=0); c.spines["left"].set_visible(False)
    ptitle(c, "c", "Beyond difficulty")
    fig.legend(handles=[Line2D([], [], marker=mk, ls="", ms=7, color=col, label=lab_) for _, lab_, col, mk in SUB],
               loc="outside upper center", ncol=3, fontsize=9.2, columnspacing=1.6)
    save(fig, 4)


def fig5(k=3, seed=4):
    p = RES / "cf_edits_P3.npz"
    if not p.exists():
        p = RES / "cf_EDITS_P3.npz"
    E = np.load(p, allow_pickle=True)
    uids = E["uids"].astype(str)
    per = rcsv("cf_P3_perimage.csv")
    per = per[(per.model == "mt_base") & (per.tag == "base")].groupby("uid").mean(numeric_only=True)
    rng = np.random.default_rng(seed)
    area = E["area"]
    les = np.where(E["ok_A"] & E["ok_S"] & (area >= np.nanpercentile(area, 40)))[0]
    nt = np.where(E["ok_B"] & E["ok_Bs"])[0]
    pl = rng.choice(les, k, replace=False); pn = rng.choice(nt, k, replace=False)
    H = W2 / 6 * k * 1.0 + 0.95
    fig, axs = plt.subplots(k, 6, figsize=(W2, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 0.95 / H, wspace=0.03, hspace=0.03)
    heads = ["Original", "A: lesion\nremoved", "S: sham\nremoval", "Original", "B: lesion\ninserted",
             "Bs: sham\ninsertion"]
    for i in range(k):
        u1, u2 = uids[pl[i]], uids[pn[i]]
        r1 = per.loc[u1] if u1 in per.index else None
        r2 = per.loc[u2] if u2 in per.index else None
        for j, (key, idx, r, col) in enumerate([("img_orig", pl[i], r1, "p_true_orig"), ("img_A", pl[i], r1, "p_true_A"),
                                                ("img_S", pl[i], r1, "p_true_S"), ("img_orig", pn[i], r2, "p_nt_orig"),
                                                ("img_B", pn[i], r2, "p_nt_B"), ("img_Bs", pn[i], r2, "p_nt_Bs")]):
            a = axs[i, j]
            img_ax(a, E[key][idx])
            if j in (0, 1):
                contour(a, E["masks"][idx], C["yellow"], lw=1.2, ls="-" if j == 0 else (0, (2, 1.5)))
            if r is not None and col in r and pd.notna(r[col]):
                tag(a, f"{r[col]:.2f}", size=8)
            if i == 0:
                a.set_title(heads[j], loc="center", fontsize=9.2, fontweight="bold", pad=4, va="bottom")

    fig.canvas.draw()
    x = (axs[0, 2].get_position().x1 + axs[0, 3].get_position().x0) / 2
    y1 = axs[0, 0].get_position().y1
    fig.add_artist(Line2D([x, x], [axs[-1, 0].get_position().y0, y1 + 0.55 / H], transform=fig.transFigure,
                          color=INK3, lw=1.2))
    for (c0, c1), t in (((0, 2), "Lesion removal  ·  p(true class)"), ((3, 5), "Lesion insertion  ·  p(no tumor)")):
        xm = (axs[0, c0].get_position().x0 + axs[0, c1].get_position().x1) / 2
        fig.text(xm, y1 + 0.6 / H, t, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    save(fig, 5)


CF_RUNS = [("cls_resnet50", "base"), ("mt_base", "base"), ("mt_ctam", "base"), ("mt_base", "ev"), ("mt_ctam", "ev")]


def fig6():
    bs = rcsv("cf_P3_by_size.csv"); sm = rcsv("cf_P3_summary.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 3.15), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[1.1, 1.05, 0.8]))
    sizes = ["small", "medium", "large"]
    runs = [(m, t) for m, t in CF_RUNS if ((sm.model == m) & (sm.tag == t)).any()]
    off = np.linspace(-0.14, 0.14, len(runs))
    for k, (m, t) in enumerate(runs):
        key = f"{m}_{t}"
        h = bs[(bs.model == m) & (bs.tag == t)].set_index("size").reindex(sizes)
        a.errorbar(np.arange(3) + off[k], h.LRS_mean, yerr=h.LRS_std, color=MCOL[key], ls="--" if t == "ev" else "-",
                   lw=1.8, marker="o", ms=6, mec="white", mew=1.1, elinewidth=1.1, label=MLAB[key])
    a.set_xticks(range(3)); a.set_xticklabels(["Small", "Medium", "Large"])
    a.set_xlabel("Lesion size tertile"); a.set_ylabel("Lesion reliance score"); ygrid(a)
    a.set_ylim(0, None)
    a.legend(fontsize=8, loc="lower right", handlelength=1.9, labelspacing=0.3)
    ptitle(a, "a", "LRS by lesion size")

    y = np.arange(len(runs)); h_ = 0.36
    labs = [MLAB[f"{m}_{t}"] for m, t in runs]
    for ax_, (e, s_, l1, l2, lt, ttl, xl) in ((b, ("flip_A", "flip_S", "Lesion removed (A)", "Sham (S)", "b",
                                                  "Lesion removal", "Images flipped (%)")),
                                              (c, ("tumor_B", "tumor_Bs", "Lesion inserted (B)", "Sham (Bs)", "c",
                                                   "Lesion insertion", "Images flipped (%)"))):
        ve, vs, se = [], [], []
        for m, t in runs:
            r = sm[(sm.model == m) & (sm.tag == t)].iloc[0]
            ve.append(r[f"{e}_mean"] * 100); vs.append(r[f"{s_}_mean"] * 100); se.append(r[f"{e}_std"] * 100)
        ax_.barh(y - h_ / 2 - 0.01, ve, h_, xerr=se, color=INK2, error_kw=dict(ecolor=INK3, lw=1), label=l1)
        ax_.barh(y + h_ / 2 + 0.01, vs, h_, color=SHAM, label=l2)
        top = max(np.add(ve, se))
        for yi, v, sd in zip(y, ve, se):
            ax_.text(v + sd + top * 0.03, yi - h_ / 2, f"{v:.1f}", va="center", fontsize=8.4, fontweight="bold")
        ax_.set_yticks(y); ax_.set_yticklabels(labs if lt == "b" else [], fontsize=9)
        ax_.set_ylim(len(y) - 0.4, -1.25)
        ax_.set_xlim(0, top * 1.28); ax_.set_xlabel(xl); xgrid(ax_)
        ax_.tick_params(axis="y", length=0); ax_.spines["left"].set_visible(False)
        ax_.legend(loc="upper right", fontsize=8, ncol=1, handlelength=1, borderaxespad=0.2, labelspacing=0.25)
        ptitle(ax_, lt, ttl)
    save(fig, 6)


def fig7(n=4, seed=11):
    import torch
    import albumentations as A
    models_mod = import_module("08_models")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clahe = A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
    mean = np.array(cfg.IMAGENET_MEAN, np.float32) * 255; std = np.array(cfg.IMAGENET_STD, np.float32) * 255

    def to_t(I, g):
        top, left, h, w = g
        crop = clahe(image=np.repeat(I[top:top + h, left:left + w, None], 3, 2))["image"]
        full = np.zeros((S, S, 3), np.uint8); full[top:top + h, left:left + w] = crop
        return torch.from_numpy(((full.astype(np.float32) - mean) / std).transpose(2, 0, 1))[None]

    def headm(I):
        _, th = cv2.threshold(cv2.GaussianBlur(I, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        f = np.zeros_like(th)
        if cs:
            cv2.drawContours(f, [max(cs, key=cv2.contourArea)], -1, 255, -1)
        return f > 0

    def gradcam(name, run, X, Y):
        ck = cfg.DIR_CKPT / f"P3_{run}_f0.pt"
        model = models_mod.build_model(name, pretrained=False).to(dev).eval()
        model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
        layer, pick = ((model.encoder, lambda o: o[-1]) if hasattr(model, "encoder")
                       else (model.net.layer4, lambda o: o))
        st = {}
        h = layer.register_forward_hook(lambda m, i, o: st.__setitem__("a", pick(o)))
        out = []
        for x, y in zip(X, Y):
            x = x.to(dev).requires_grad_(False)
            model.zero_grad(set_to_none=True)
            o = model(x)
            act = st["a"]
            g = torch.autograd.grad(o["logits_cls"][0, y], act)[0]
            cam = torch.relu((g.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, (S, S), mode="bilinear", align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy(); cam = cam / (cam.max() + 1e-8)
            out.append(cam)
        h.remove(); del model
        return out

    f = folds_df()
    te = f[(f["P3_role"] == "test") & (f.class_name != "no_tumor")] if "P3_role" in f else f[f.class_name != "no_tumor"]
    rng = np.random.default_rng(seed)
    picks = []
    for c in ["glioma", "meningioma", "pituitary", None]:
        sub = te if c is None else te[te.class_name == c]
        for _, r in sub.sample(frac=1, random_state=int(rng.integers(1e6))).iterrows():
            if r.uid in [p[0].uid for p in picks]:
                continue
            I, M, g = load_square(r.img_path, mask_path_of(r))
            if M.mean() >= 0.008:
                picks.append((r, I, M, g)); break
    picks = picks[:n]
    X = [to_t(I, g) for _, I, _, g in picks]
    Y = [int(CLASSES.index(r.class_name)) for r, *_ in picks]
    rows = [("ResNet-50", gradcam("cls_resnet50", "cls_resnet50_base", X, Y)),
            ("MT-Base", gradcam("mt_base", "mt_base_base", X, Y))]
    fig, axs = plt.subplots(3, len(picks), figsize=(W2 * 0.82, W2 * 0.82 / len(picks) * 3 + 0.2),
                            gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for j, (r, I, M, g) in enumerate(picks):
        hm = headm(I)
        img_ax(axs[0, j], I); contour(axs[0, j], M, C["yellow"], lw=1.3)
        axs[0, j].set_title(CLAB[r.class_name], loc="center", fontsize=10, fontweight="bold", pad=4)
        for i, (lab, cams) in enumerate(rows, start=1):
            cam = cams[j]
            a = axs[i, j]
            img_ax(a, I)
            a.imshow(np.ma.masked_less(cam, 0.08), cmap="magma", alpha=0.62, vmin=0, vmax=1)
            contour(a, M, C["yellow"], lw=1.0)
            contour(a, hm, "white", lw=0.7, ls=(0, (2, 2)))
            tot = cam.sum() + 1e-8
            tag(a, f"in lesion {cam[M].sum() / tot:.2f}", "tl", 7.4)
            tag(a, f"outside head {cam[~hm].sum() / tot:.2f}", "bl", 7.4)
            if j == 0:
                a.set_ylabel(lab, fontsize=10.5, fontweight="bold")
    axs[0, 0].set_ylabel("Image", fontsize=10.5, fontweight="bold")
    save(fig, 7)


def _reliab(P, y, nb=10):
    conf, acc = P.max(1), (P.argmax(1) == y)
    bins = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, nb - 1)
    xs, ys, ns = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(acc[m].mean()); ns.append(m.sum())
    ece = sum(abs(acc[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
              for b in range(nb) if (idx == b).any())
    return np.array(xs), np.array(ys), ece


def _ens(run):
    T = json.load(open(RES / "temperature_P3.json")) if (RES / "temperature_P3.json").exists() else {}
    Pr, Pc, y, uids = [], [], None, None
    for fz in range(5):
        p = RES / f"preds_P3_{run}_f{fz}_test.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        u = z["uids"].astype(str)
        if uids is None:
            uids, y = u, z["y"].astype(int)
        o = pd.Series(np.arange(len(u)), index=u).reindex(uids).values
        P = z["probs"].astype(np.float64)[o]
        L = np.log(np.clip(P, 1e-12, 1)) / T.get(f"{run}_f{fz}", 1.0)
        E = np.exp(L - L.max(1, keepdims=True))
        Pr.append(P); Pc.append(E / E.sum(1, keepdims=True))
    return np.mean(Pr, 0), np.mean(Pc, 0), y


def fig8():
    fig = plt.figure(figsize=(W2, 6.0), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.05, 0.95], hspace=0.06)
    top = sf[0].subplots(1, 2)
    data = {run: _ens(run) for run in ["cls_resnet50_base", "mt_base_base"]}
    curves, allx = {}, []
    for run, (Pr, Pc, y) in data.items():
        curves[run] = [(_reliab(P, y), lab, col, mk) for P, lab, col, mk in
                       ((Pr, "Raw", SHAM, "o"), (Pc, "Temperature-scaled", MCOL[run], "s"))]
        for (xs, ys, _), *_ in curves[run]:
            allx += list(xs) + list(ys)
    lo = max(0.0, np.floor((min(allx) - 0.03) * 20) / 20) if allx else 0.0
    for k, (a, run) in enumerate(zip(top, data)):
        for (xs, ys, e), lab, col, mk in curves[run]:
            a.plot(xs, ys, "-", color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.1, label=f"{lab}  (ECE {e:.3f})",
                   clip_on=False, zorder=3)
        a.plot([lo, 1], [lo, 1], color=INK3, lw=1, ls="--")
        a.set_xlim(lo, 1.0); a.set_ylim(lo, 1.0); a.set_aspect("equal", adjustable="box")
        a.set_xlabel("Confidence"); a.set_ylabel("Accuracy" if k == 0 else "")
        ygrid(a); xgrid(a)
        a.legend(loc="lower right", fontsize=8.2, borderaxespad=0.2)
        ptitle(a, "a" if k == 0 else "", MLAB[run])
    cf = rcsv("conformal_P3_summary.csv")
    cf = cf[(cf.method == "Mondrian") & np.isclose(cf.alpha, 0.01)].set_index("run")
    runs = [k for k, *_ in MODELS if k in cf.index]
    b, c = sf[1].subplots(1, 2, gridspec_kw=dict(width_ratios=[1.12, 0.88]))
    y = np.arange(len(runs))
    for a, col, f_, xl, lt, ttl in ((b, "mean_size", lambda v: f"{v:.3f}", "Mean set size", "b", "Set size, α = 0.01"),
                                    (c, "detect_ambig", lambda v: f"{v * 100:.1f}", "Ambiguous sets (%)",
                                     "c", "Detection ambiguity")):
        v = cf.loc[runs, f"{col}_mean"].values
        sc = 100 if col == "detect_ambig" else 1
        base = 1.0 if sc == 1 else 0.0
        bars = a.barh(y, v * sc - base, 0.62, left=base, color=[MCOL[r] for r in runs])
        for br, r in zip(bars, runs):
            if MHAT[r]:
                br.set_hatch("///"); br.set_edgecolor("white")
        span = max(v * sc) - base
        for yi, vi in zip(y, v):
            a.text(vi * sc + span * 0.03, yi, f_(vi), va="center", fontsize=8.6, fontweight="bold")
        a.set_yticks(y); a.set_yticklabels([MLAB[r] for r in runs] if lt == "b" else [], fontsize=9.2)
        a.invert_yaxis(); a.set_xlim(base, base + span * 1.25); a.set_xlabel(xl); xgrid(a)
        a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
        ptitle(a, lt, ttl)
    save(fig, 8)


def fig9():
    cp = rcsv("external_contrast_pairs.csv")
    di = rcsv("external_dup_impact.csv")
    dc = rcsv("external_dup_check.csv")
    fig = plt.figure(figsize=(W2, 5.5), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.2, 0.72], hspace=0.05)
    a1, a2, b = sf[0].subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 0.78, 1.3]))
    runs = [("mt_base", "MT-Base"), ("mt_ctam", "MT-CTAM"), ("cls_resnet50", "ResNet-50")]
    for a, kind, ttl, lt in ((a1, "tumor", "Glioma slices", "a"), (a2, "normal", "Tumor-free", "")):
        for yi, (m, lab) in enumerate(runs):
            r = cp[(cp.model == m) & (cp.tag == "base") & (cp.kind == kind)]
            if not len(r):
                continue
            r = r.iloc[0]
            v1, v0 = r.tumor_rate_t1c * 100, r.tumor_rate_t1 * 100
            a.plot([v0, v1], [yi, yi], color=GRID, lw=6, solid_capstyle="round", zorder=1)
            a.scatter([v1], [yi], s=78, color=C["blue"], zorder=3, edgecolor="white", linewidth=1.3)
            a.scatter([v0], [yi], s=78, color=C["orange"], zorder=3, edgecolor="white", linewidth=1.3)
            a.text(50, yi + 0.3, f"{int(r.t1c_only)} vs {int(r.t1_only)} discordant", ha="center", va="top",
                   fontsize=7.8, color=INK2)
        a.set_yticks(range(len(runs))); a.set_yticklabels([l for _, l in runs] if lt else [], fontsize=9.5)
        a.set_ylim(len(runs) - 0.35, -1.0)
        a.set_xlim(-5, 105); a.set_xticks([0, 50, 100]); xgrid(a)
        a.set_xlabel("Called tumor (%)")
        a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
        ptitle(a, lt, ttl)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=C["blue"], label="T1c"),
                       Line2D([], [], marker="o", ls="", ms=7, color=C["orange"], label="Noncontrast T1")],
              loc="upper left", ncol=2, fontsize=8.6, columnspacing=0.8, handletextpad=0.2, borderaxespad=0.1)

    SUBS = [("clean", "Non-duplicate"), ("naive_all", "All (naive)"),
            ("dup_unseen(test)", "Dup., BRISC test"), ("dup_seen_in_training", "Dup., BRISC train")]
    g = di.groupby(["model", "subset"]).bal_acc.agg(["mean", "std"]).reset_index()
    mods = [("cls_resnet50", "ResNet-50", MCOL["cls_resnet50_base"]), ("mt_base", "MT-Base", MCOL["mt_base_base"]),
            ("mt_ctam", "MT-CTAM", MCOL["mt_ctam_base"])]
    for k, (m, lab, col) in enumerate(mods):
        h = g[g.model == m].set_index("subset").reindex([s_ for s_, _ in SUBS])
        b.errorbar(h["mean"] * 100, np.arange(len(SUBS)) + (k - 1) * 0.2, xerr=h["std"] * 100, fmt="o", color=col,
                   ms=6.5, mec="white", mew=1.1, elinewidth=1.3, label=lab)
    lo = np.floor((g["mean"].min() * 100 - 6) / 5) * 5
    b.set_xlim(lo, 101.5)
    b.set_yticks(range(len(SUBS))); b.set_yticklabels([l for _, l in SUBS], fontsize=9)
    b.set_ylim(len(SUBS) - 0.5, -0.5)
    b.axhspan(-0.45, 0.45, color=C["blue"], alpha=0.07, lw=0)
    b.set_xlabel("Balanced accuracy (%)"); xgrid(b)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.legend(loc="lower left", fontsize=8.4, labelspacing=0.25, handletextpad=0.2, borderaxespad=0.2)
    ptitle(b, "b", "Four-class dataset")

    m = dc[(dc.source == "mendeley") & (dc.dhash_dist <= 4)]
    ex = []
    for cn in ["glioma", "meningioma", "pituitary"]:
        k = m[m.brisc_class == cn]
        if len(k):
            ex.append(k.sample(1, random_state=3).iloc[0])
    ex += [r for _, r in m.sample(min(3, len(m)), random_state=5).iterrows()][:max(0, 3 - len(ex))]
    sfs = sf[1].subfigures(1, len(ex), wspace=0.04)
    sf[1].suptitle("c   External images and their nearest BRISC2025 image", x=0.0, ha="left",
                   fontsize=11.5, fontweight="bold")
    for i, r in enumerate(ex):
        axp = np.atleast_1d(sfs[i] if len(ex) > 1 else sfs).ravel()[0].subplots(1, 2, gridspec_kw=dict(wspace=0.03))
        for j, (p, t) in enumerate(((r.img_path, "External"), (r.brisc_path, "BRISC2025"))):
            I, _, _ = load_square(p)
            img_ax(axp[j], I)
            axp[j].set_title(t, loc="center", fontsize=9, fontweight="bold", pad=3)
        tag(axp[0], CLAB[r.brisc_class], "tl", 7.4)
        tag(axp[1], f"NCC {r.ncc:.3f}", "bl", 7.4)
    save(fig, 9)


def fmt(m, s, d=3, pct=False):
    if pd.isna(m):
        return "—"
    k = 100 if pct else 1
    return f"{m * k:.{d}f} ± {s * k:.{d}f}" if pd.notna(s) else f"{m * k:.{d}f}"


def tables():
    L = ["# Auto-generated tables (from results/*.csv)\n"]

    rows = []
    for key, lab, *_ in MODELS:
        p = RES / f"summary_P3_{key}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        g = lambda c: (d[c].mean(), d[c].std()) if c in d else (np.nan, np.nan)
        ba, auc, dice = g("test_balanced_accuracy"), g("test_macro_auc"), g("test_lesion_dice")
        hd, emp, ece = g("test_lesion_hd95"), g("test_empty_mask_accuracy"), g("test_ece")
        rows.append(f"| {lab} | {fmt(*ba, 2, True)} | {fmt(*auc, 4)} | {fmt(*dice, 4)} | {fmt(*hd, 2)} | "
                    f"{fmt(*emp, 2, True)} | {fmt(*ece, 4)} |")
    L += ["## Table 2. P3 test performance (mean ± SD over folds)\n",
          "| Model | BA (%) | Macro AUC | Lesion Dice | HD95 | Empty-mask acc. (%) | ECE |",
          "|---|---|---|---|---|---|---|"] + rows + [""]

    try:
        cf = rcsv("cf_P3_summary.csv"); at = rcsv("attribution_P3_summary.csv")
        rows = []
        for m, t in CF_RUNS:
            r = cf[(cf.model == m) & (cf.tag == t)]
            q = at[(at.model == m) & (at.tag == t)]
            if not len(r):
                continue
            r = r.iloc[0]; q = q.iloc[0] if len(q) else None
            rows.append(f"| {MLAB[f'{m}_{t}']} | {fmt(r.LRS_mean, r.LRS_std)} | {fmt(r.flip_A_mean, r.flip_A_std, 1, True)} | "
                        f"{r.flip_S_mean * 100:.1f} | {r.tumor_B_mean * 100:.1f} / {r.tumor_Bs_mean * 100:.1f} | "
                        + (f"{q.cam_EIT_mean:.3f} | {q.enrichment_mean:.2f} | {q.bg_mass_mean:.3f} |" if q is not None else "— | — | — |"))
        L += ["## Table 4. Lesion reliance (P3)\n",
              "| Model | LRS | flip_A (%) | flip_S (%) | Insertion → tumor (%) B / Bs | CAM-in-lesion | Enrichment | Background mass |",
              "|---|---|---|---|---|---|---|---|"] + rows + [""]
    except Exception as e:
        L.append(f"(Table 4 skipped: {e})\n")

    try:
        ca = rcsv("calib_P3_summary.csv").set_index("run")
        cf = rcsv("conformal_P3_summary.csv")
        cf = cf[(cf.method == "Mondrian") & np.isclose(cf.alpha, 0.01)].set_index("run")
        rows = []
        for key, lab, *_ in MODELS:
            if key not in ca.index:
                continue
            r = ca.loc[key]; q = cf.loc[key] if key in cf.index else None
            rows.append(f"| {lab} | {r.T_mean:.2f} | {r.ece_raw_mean:.4f} → {r.ece_ts_mean:.4f} | "
                        f"{r.nll_raw_mean:.3f} → {r.nll_ts_mean:.3f} | "
                        + (f"{q.coverage_mean:.3f} | {q.mean_size_mean:.3f} | {q.detect_ambig_mean * 100:.1f} |"
                           if q is not None else "— | — | — |"))
        L += ["## Table 5. Calibration and Mondrian conformal sets (α = 0.01)\n",
              "| Model | T | ECE raw → scaled | NLL raw → scaled | Coverage | Mean set size | Detection ambiguity (%) |",
              "|---|---|---|---|---|---|---|"] + rows + [""]
    except Exception as e:
        L.append(f"(Table 5 skipped: {e})\n")

    try:
        es = rcsv("external_summary.csv"); cp = rcsv("external_contrast_pairs.csv"); di = rcsv("external_dup_impact.csv")
        es = es[es.tag == "base"].set_index("model")
        mods = [m for m in ["cls_resnet50", "mt_base", "mt_ctam", "seg_unet_r50"] if m in es.index]
        lab = {"cls_resnet50": "ResNet-50", "mt_base": "MT-Base", "mt_ctam": "MT-CTAM", "seg_unet_r50": "U-Net"}
        def cell(m, k, d=3, pct=False):
            return fmt(es.loc[m].get(f"{k}_mean", np.nan), es.loc[m].get(f"{k}_std", np.nan), d, pct)
        def pair(m, kind):
            r = cp[(cp.model == m) & (cp.kind == kind)]
            if not len(r):
                return "—", "—", "—"
            r = r.iloc[0]
            return (f"{r.tumor_rate_t1c * 100:.1f}", f"{r.tumor_rate_t1 * 100:.1f}",
                    f"{int(r.t1c_only)} / {int(r.t1_only)} (p = {r.mcnemar_p:.2g})")
        naive = di[di.subset == "naive_all"].groupby("model")[["bal_acc", "macro_auc", "coverage"]].agg(["mean", "std"])
        R = [("Tumor call T1c, glioma slices (%)", lambda m: pair(m, "tumor")[0]),
             ("Tumor call noncontrast T1, same slices (%)", lambda m: pair(m, "tumor")[1]),
             ("Discordant T1c-only / T1-only", lambda m: pair(m, "tumor")[2]),
             ("Tumor call T1c, tumor-free slices (%)", lambda m: pair(m, "normal")[0]),
             ("Tumor call noncontrast T1, tumor-free (%)", lambda m: pair(m, "normal")[1]),
             ("BraTS glioma sensitivity (%)", lambda m: cell(m, "brats_t1c_glioma_sens", 1, True)),
             ("BraTS specificity (%)", lambda m: cell(m, "brats_t1c_normal_spec", 1, True)),
             ("BraTS tumor-core Dice", lambda m: cell(m, "brats_t1c_dice")),
             ("BraTS HD95 (px)", lambda m: cell(m, "brats_t1c_hd95", 2)),
             ("BraTS coverage", lambda m: cell(m, "brats_t1c_coverage", 2)),
             ("BraTS set size", lambda m: cell(m, "brats_t1c_set_size", 2)),
             ("Mendeley BA (non-duplicate)", lambda m: cell(m, "mend_bal_acc")),
             ("Mendeley macro AUC", lambda m: cell(m, "mend_macro_auc")),
             ("Mendeley coverage", lambda m: cell(m, "mend_coverage", 2)),
             ("Mendeley set size", lambda m: cell(m, "mend_set_size", 2)),
             ("Mendeley detection ambiguity (%)", lambda m: cell(m, "mend_detect_ambig", 1, True)),
             ("Mendeley BA, naive (with duplicates)",
              lambda m: fmt(*naive.loc[m, "bal_acc"].values) if m in naive.index else "—"),
             ("Mendeley macro AUC, naive",
              lambda m: f"{naive.loc[m, ('macro_auc', 'mean')]:.3f}" if m in naive.index else "—")]
        L += ["## Table 6. External validation\n", "| | " + " | ".join(lab[m] for m in mods) + " |",
              "|---|" + "---|" * len(mods)]
        for name, fn in R:
            L.append(f"| {name} | " + " | ".join(str(fn(m)) for m in mods) + " |")
        L.append("")
    except Exception as e:
        L.append(f"(Table 6 skipped: {e})\n")
    p = OUTF / "tables_auto.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ tables → {p}")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8, 9: fig9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    todo = a.only or list(FIGS)
    if a.no_model and 7 in todo:
        todo.remove(7)
    errs = {}
    for n in todo:
        try:
            FIGS[n]()
        except Exception as e:
            errs[n] = traceback.format_exc(limit=3)
            print(f"  ✗ Figure {n}: {type(e).__name__}: {e}")
    if not a.only:
        try:
            tables()
        except Exception as e:
            errs["tables"] = traceback.format_exc(limit=3)
    if errs:
        print("\n========== Errors (send this section) ==========")
        for k, v in errs.items():
            print(f"--- {k}\n{v}")
    print(f"\n  output folder: {OUTF}")


if __name__ == "__main__":
    main()
