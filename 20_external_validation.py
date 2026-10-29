


import argparse, glob, hashlib, json, os, random, re, sys
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
from PIL import Image

cfg = import_module("00_config")
S = cfg.IMG_SIZE
EXT = cfg.OUT_ROOT / "external"
CLASSES = cfg.CLASSES
NT = cfg.CLASS2IDX["no_tumor"]


def _case_id(path):
    b = os.path.basename(path)
    b = re.sub(r"\.nii(\.gz)?$", "", b)
    return re.sub(r"_(t1ce|t1c|t1|t2|flair|seg)$", "", b, flags=re.I)


def find_brats(a):
    cases = {}
    if a.brats_root:
        for p in glob.glob(f"{a.brats_root}/**/*.nii*", recursive=True):
            low = os.path.basename(p).lower()
            k = ("t1c" if re.search(r"_t1c(e)?\.nii", low) else "t1" if re.search(r"_t1\.nii", low)
                 else "seg" if re.search(r"_seg\.nii", low) else None)
            if k:
                cases.setdefault(_case_id(p), {})[k] = p
    else:
        for key, d in (("t1c", a.t1c_dir), ("t1", a.t1_dir), ("seg", a.seg_dir)):
            if d:
                for p in glob.glob(f"{d}/*.nii*"):
                    cases.setdefault(_case_id(p), {})[key] = p
    return {c: v for c, v in cases.items() if "t1c" in v and "seg" in v}


def _to_u8(vol, brain):
    v = vol[brain]
    lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (0, 1)
    return (np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def _slice2d(vol, z):
    return np.ascontiguousarray(np.rot90(vol[:, :, z]))                             


def prep_brats(a, rows):
    import nibabel as nib
    cases = find_brats(a)
    print(f"  BraTS: {len(cases)} patients with T1c and mask | with T1: {sum('t1' in v for v in cases.values())}")
    if not cases:
        return
    keys = sorted(cases)
    random.Random(0).shuffle(keys)
    keys = keys[:a.max_cases] if a.max_cases else keys
    out = EXT / "brats"
    for d in ("t1c", "t1", "mask"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for n, c in enumerate(keys):
        v = cases[c]
        t1c = nib.load(v["t1c"]).get_fdata()
        seg = np.rint(nib.load(v["seg"]).get_fdata()).astype(int)
        t1 = nib.load(v["t1"]).get_fdata() if "t1" in v else None
        if t1 is not None and t1.shape != t1c.shape:
            t1 = None
        brain = t1c > 0
        u1c = _to_u8(t1c, brain)
        u1 = _to_u8(t1, t1 > 0) if t1 is not None else None
        labs = set(np.unique(seg)) - {0}
        tc_labels = [1, 4] if 4 in labs else [1, 3]                                                      
        tc = np.isin(seg, tc_labels)
        wt = seg > 0
        tc_area = tc.sum((0, 1)); wt_any = wt.any((0, 1)); br_area = brain.sum((0, 1))
        rng = random.Random(f"slice-{c}")
        cand = [z for z in range(seg.shape[2]) if tc_area[z] >= 300]
        tumor_z = []
        for z in sorted(cand, key=lambda z: -tc_area[z]):
            if all(abs(z - t) >= 5 for t in tumor_z):
                tumor_z.append(z)
            if len(tumor_z) >= 3:
                break
        tz = np.where(wt_any)[0]
        normal_c = [z for z in range(seg.shape[2]) if not wt_any[z] and br_area[z] >= 0.5 * br_area.max()
                    and (len(tz) == 0 or np.min(np.abs(tz - z)) >= 10)]
        normal_z = rng.sample(normal_c, min(2, len(normal_c)))
        for z, kind in [(z, "tumor") for z in tumor_z] + [(z, "normal") for z in normal_z]:
            sid = f"brats_{c}_z{z:03d}"
            mp = out / "mask" / f"{sid}.png"
            Image.fromarray((_slice2d(tc, z) * 255).astype(np.uint8) if kind == "tumor"
                            else np.zeros(_slice2d(tc, z).shape, np.uint8)).save(mp)
            for seq, U in (("t1c", u1c), ("t1", u1)):
                if U is None:
                    continue
                ip = out / seq / f"{sid}.png"
                Image.fromarray(_slice2d(U, z)).save(ip)
                rows.append(dict(uid=f"{sid}_{seq}", pair_id=sid, source="brats", sequence=seq, kind=kind,
                                 label=CLASSES.index("glioma") if kind == "tumor" else NT,
                                 img_path=str(ip), mask_path=str(mp), case=c))
        if (n + 1) % 50 == 0:
            print(f"    {n + 1}/{len(keys)}")


MAP = [("glio", "glioma"), ("menin", "meningioma"), ("pitu", "pituitary"),
       ("healthy", "no_tumor"), ("normal", "no_tumor"), ("no_tumor", "no_tumor"), ("notumor", "no_tumor"),
       ("no tumor", "no_tumor")]


def prep_mendeley(a, rows):
    if not a.mendeley_dir:
        return
    root = a.mendeley_dir
    for z in glob.glob(f"{root}/**/*.zip", recursive=True):                        
        tgt = os.path.splitext(z)[0]
        if not os.path.isdir(tgt):
            import zipfile
            zipfile.ZipFile(z).extractall(tgt)
    imgs = [p for p in glob.glob(f"{root}/**/*", recursive=True)
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
    unk = set()
    for p in imgs:
        parts = [s.lower() for s in os.path.relpath(p, root).split(os.sep)[:-1]]
        cls = None
        for part in reversed(parts):
            cls = next((c for k, c in MAP if k in part), None)
            if cls:
                break
        if cls is None:
            unk.add(os.path.dirname(os.path.relpath(p, root))); continue
        rows.append(dict(uid="mend_" + hashlib.md5(p.encode()).hexdigest()[:12], pair_id="", source="mendeley",
                         sequence="mixed", kind="tumor" if cls != "no_tumor" else "normal",
                         label=CLASSES.index(cls), img_path=p, mask_path="", case=""))
    n = pd.Series([r["label"] for r in rows if r["source"] == "mendeley"]).map(lambda i: CLASSES[i]).value_counts()
    print(f"  Mendeley: {len(imgs)} images | mapping: {n.to_dict()}")
    if unk:
        print(f"  ⚠ folders without a recognized class (ignored): {sorted(unk)[:6]}")


def dhash(gray, n=16):
    r = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA)
    return np.packbits((r[:, 1:] > r[:, :-1]).ravel())


def pix_md5(gray):
    return hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest()


def brisc_hashes():
    cache = cfg.DIR_RESULTS / "brisc_dhash_md5.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["uids"], z["md5"], z["dh"]
    f = pd.read_csv(cfg.DIR_GROUPS / "folds.csv")
    print(f"  hashing {len(f)} BRISC images (once)...")
    md5, dh = [], []
    for p in f.img_path:
        g = np.asarray(Image.open(p).convert("L"))
        md5.append(pix_md5(g)); dh.append(dhash(g))
    np.savez(cache, uids=f.uid.values, md5=np.array(md5), dh=np.stack(dh))
    return f.uid.values, np.array(md5), np.stack(dh)


def dedup(df, max_ham=10):
    _, bmd5, bdh = brisc_hashes()
    bset = set(bmd5)
    bbits = np.unpackbits(bdh, axis=1).astype(np.uint8)
    exact, near, dist = [], [], []
    for p in df.img_path:
        g = np.asarray(Image.open(p).convert("L"))
        exact.append(pix_md5(g) in bset)
        bits = np.unpackbits(dhash(g)).astype(np.uint8)
        d = int((bbits != bits).sum(1).min())
        dist.append(d); near.append(d <= max_ham)
    df["dup_exact"], df["dup_near"], df["dhash_min_dist"] = exact, near, dist
    for s, g in df.groupby("source"):
        print(f"  {s}: exact copies={int(g.dup_exact.sum())} | near (Hamming≤{max_ham})={int(g.dup_near.sum())} of {len(g)}")
    return df


def _tf():
    import albumentations as A
    t7 = import_module("07b_transforms_fix")
    return A.Compose([A.LongestMaxSize(max_size=S, interpolation=cv2.INTER_LINEAR),
                      A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0),
                      t7.pad_if_needed(S),
                      A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)])


def load_batch(df, tf):
    import torch
    X, M = [], []
    for _, r in df.iterrows():
        g = np.asarray(Image.open(r.img_path).convert("L"))
        m = (np.asarray(Image.open(r.mask_path).convert("L")) > 127).astype(np.uint8) if r.mask_path else \
            np.zeros(g.shape, np.uint8)
        o = tf(image=np.stack([g] * 3, -1), mask=m)
        X.append(torch.from_numpy(o["image"].transpose(2, 0, 1)).float())
        M.append(o["mask"].astype(bool))
    return torch.stack(X), np.stack(M)


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt, binary_erosion
    if not pred.any() or not gt.any():
        return np.nan
    bp = pred ^ binary_erosion(pred); bg = gt ^ binary_erosion(gt)
    dp = distance_transform_edt(~bg)[bp]; dg = distance_transform_edt(~bp)[bg]
    return float(np.percentile(np.concatenate([dp, dg]), 95))


def mondrian_q(Pv, yv, alpha=0.01):
    q = np.ones(len(CLASSES))
    for c in range(len(CLASSES)):
        sc = np.sort(1 - Pv[yv == c, c])
        if len(sc):
            k = min(int(np.ceil((len(sc) + 1) * (1 - alpha))), len(sc))
            q[c] = sc[k - 1]
    return q


def infer(df, runs, folds, bs):
    import torch, torch.nn.functional as F
    models_mod = import_module("08_models")
    dev = cfg.get_device()
    T = json.load(open(cfg.DIR_RESULTS / "temperature_P3.json")) if (cfg.DIR_RESULTS / "temperature_P3.json").exists() else {}
    tf = _tf()
    out = []
    for name, tag in runs:
        for f in folds:
            run = f"P3_{name}_{tag}_f{f}"
            ck = cfg.DIR_CKPT / f"{run}.pt"
            if not ck.exists():
                print(f"  ⚠ {run}: checkpoint not found"); continue
            c = torch.load(ck, map_location=dev, weights_only=False)
            mj = cfg.DIR_RESULTS / f"metrics_{run}.json"
            if mj.exists() and json.load(open(mj)).get("best_epoch") != c.get("epoch"):
                print(f"  ✗ {run}: checkpoint does not match metrics — skipped"); continue
            model = models_mod.build_model(name, pretrained=False).to(dev).eval()
            model.load_state_dict(c["model"])
            t = T.get(f"{name}_{tag}_f{f}", 1.0)
            q = None
            vz = cfg.DIR_RESULTS / f"preds_{run}_val.npz"
            if vz.exists():
                z = np.load(vz, allow_pickle=True)
                L = np.log(np.clip(z["probs"], 1e-12, 1)) / t
                Pv = np.exp(L - L.max(1, keepdims=True)); Pv /= Pv.sum(1, keepdims=True)
                q = mondrian_q(Pv, z["y"].astype(int))
            for s in range(0, len(df), bs):
                sub = df.iloc[s:s + bs]
                X, M = load_batch(sub, tf)
                with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    o = model(X.to(dev))
                rec = dict(uid=sub.uid.values, model=name, tag=tag, fold=f)
                if o.get("logits_cls") is not None:
                    lo = o["logits_cls"].float()
                    P = F.softmax(lo, 1).cpu().numpy()
                    Pt = F.softmax(lo / t, 1).cpu().numpy()
                    for k, cn in enumerate(CLASSES):
                        rec[f"p_{cn}"] = Pt[:, k]
                    rec["pred"] = Pt.argmax(1)
                    if q is not None:
                        sets = (1 - Pt) <= q[None]
                        rec["set_size"] = sets.sum(1)
                        rec["covered"] = sets[np.arange(len(sub)), sub.label.values]
                        rec["set_ambig"] = sets[:, :NT].any(1) & sets[:, NT]
                if o.get("logits_seg") is not None:
                    pm = (torch.sigmoid(o["logits_seg"].float()) > 0.5).cpu().numpy()[:, 0]
                    d, h, emp = [], [], []
                    for pi, gi in zip(pm, M):
                        if gi.any():
                            inter = (pi & gi).sum()
                            d.append(2 * inter / (pi.sum() + gi.sum())); h.append(hd95(pi, gi)); emp.append(np.nan)
                        else:
                            d.append(np.nan); h.append(np.nan); emp.append(float(not pi.any()))
                    rec.update(dice=d, hd95=h, empty_correct=emp)
                out.append(pd.DataFrame(rec))
            print(f"  ✓ {run}")
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(df, P):
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    P = P.merge(df[["uid", "pair_id", "source", "sequence", "kind", "label", "dup_exact", "dup_near"]], on="uid")
    P = P[~(P.dup_exact | P.dup_near)]
    rows = []
    for (m, t, f), g in P.groupby(["model", "tag", "fold"]):
        r = dict(model=m, tag=t, fold=f)
        has_cls = "pred" in g and g.pred.notna().any()
        for seq in ("t1c", "t1"):
            b = g[(g.source == "brats") & (g.sequence == seq)]
            tu, no = b[b.kind == "tumor"], b[b.kind == "normal"]
            if has_cls and len(tu):
                r[f"brats_{seq}_tumor_detect"] = (tu.pred != NT).mean()
                r[f"brats_{seq}_glioma_sens"] = (tu.pred == CLASSES.index("glioma")).mean()
                if len(no):
                    r[f"brats_{seq}_normal_spec"] = (no.pred == NT).mean()
                if "covered" in b:
                    r[f"brats_{seq}_coverage"] = b.covered.mean()
                    r[f"brats_{seq}_set_size"] = b.set_size.mean()
            if "dice" in b and len(tu):
                r[f"brats_{seq}_dice"] = tu.dice.mean()
                r[f"brats_{seq}_hd95"] = tu.hd95.mean()
                if len(no):
                    r[f"brats_{seq}_empty_acc"] = no.empty_correct.mean()
        me = g[g.source == "mendeley"]
        if has_cls and len(me):
            y, pr = me.label.values.astype(int), me.pred.values.astype(int)
            r["mend_bal_acc"] = np.mean([np.mean(pr[y == c] == c) for c in np.unique(y)])
            for c, cn in enumerate(CLASSES):
                if (y == c).any():
                    r[f"mend_sens_{cn}"] = np.mean(pr[y == c] == c)
            Pm = me[[f"p_{c}" for c in CLASSES]].values
            try:
                r["mend_macro_auc"] = np.mean([roc_auc_score(y == c, Pm[:, c]) for c in np.unique(y)])
            except ValueError:
                pass
            if "covered" in me:
                r["mend_coverage"] = me.covered.mean(); r["mend_set_size"] = me.set_size.mean()
                r["mend_detect_ambig"] = me.set_ambig.mean()
        rows.append(r)
    per = pd.DataFrame(rows)
    cols = [c for c in per.columns if c not in ("model", "tag", "fold")]
    summ = per.groupby(["model", "tag"])[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    per.to_csv(cfg.DIR_RESULTS / "external_perfold.csv", index=False)
    summ.to_csv(cfg.DIR_RESULTS / "external_summary.csv")

    tests = []
    b = P[(P.source == "brats") & P.pred.notna()] if "pred" in P else pd.DataFrame()
    if len(b):
        e = b.groupby(["model", "tag", "pair_id", "sequence", "kind"])[[f"p_{c}" for c in CLASSES]].mean().reset_index()
        e["tumor"] = e[[f"p_{c}" for c in CLASSES]].values.argmax(1) != NT
        for (m, t, kind), g in e.groupby(["model", "tag", "kind"]):
            w = g.pivot_table(index="pair_id", columns="sequence", values="tumor", aggfunc="first").dropna()
            if {"t1c", "t1"} <= set(w.columns) and len(w):
                a_, b_ = w.t1c.astype(bool), w.t1.astype(bool)
                n10, n01 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
                p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
                tests.append(dict(model=m, tag=t, kind=kind, n_pairs=len(w), tumor_rate_t1c=a_.mean(),
                                  tumor_rate_t1=b_.mean(), t1c_only=n10, t1_only=n01, mcnemar_p=p))
    T = pd.DataFrame(tests)
    T.to_csv(cfg.DIR_RESULTS / "external_contrast_pairs.csv", index=False)

    pd.set_option("display.width", 220)
    cfg.banner("External validation (mean ± SD across folds; duplicates removed)")
    keys = [k for k in ["brats_t1c_tumor_detect", "brats_t1c_glioma_sens", "brats_t1c_normal_spec", "brats_t1c_dice",
                        "brats_t1c_hd95", "brats_t1c_empty_acc", "brats_t1_tumor_detect", "brats_t1_normal_spec",
                        "brats_t1c_coverage", "brats_t1c_set_size", "mend_bal_acc", "mend_macro_auc",
                        "mend_sens_glioma", "mend_sens_meningioma", "mend_sens_pituitary", "mend_sens_no_tumor",
                        "mend_coverage", "mend_set_size", "mend_detect_ambig"] if f"{k}_mean" in summ]
    show = pd.DataFrame({k: summ[f"{k}_mean"].round(3).astype(str) + "±" + summ[f"{k}_std"].round(3).astype(str)
                         for k in keys}).T
    print(show.to_string())
    if len(T):
        cfg.banner("Contrast test: same BraTS slice, T1c vs non-contrast T1 (ensemble, McNemar)")
        print(T.round(4).to_string(index=False))
    figure(P)
    print(f"\n  → {cfg.DIR_RESULTS / 'external_summary.csv'}\n  → {cfg.DIR_RESULTS / 'external_contrast_pairs.csv'}")


def figure(P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = P[(P.source == "brats") & ("pred" in P) & P.get("pred", pd.Series(dtype=float)).notna()]
    if not len(b):
        return
    g = (b.assign(tumor=b.pred != NT)
         .groupby(["model", "tag", "fold", "sequence", "kind"]).tumor.mean().reset_index()
         .groupby(["model", "tag", "sequence", "kind"]).tumor.agg(["mean", "std"]).reset_index())
    g["name"] = g.model + (g.tag.where(g.tag != "base", "").radd(":").replace(":", ""))
    names = sorted(g.name.unique())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, kind, title in ((ax[0], "tumor", "Glioma slices: predicted 'tumour'"),
                           (ax[1], "normal", "Tumour-free slices: predicted 'tumour'")):
        x = np.arange(len(names))
        for k, (seq, lab) in enumerate((("t1c", "T1c (contrast)"), ("t1", "T1 (no contrast)"))):
            h = g[(g.kind == kind) & (g.sequence == seq)].set_index("name").reindex(names)
            a.bar(x + (k - 0.5) * 0.38, h["mean"], 0.38, yerr=h["std"], label=lab, capsize=2)
        a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        a.set_title(title, fontsize=9); a.set_ylim(0, 1.05)
    ax[0].set_ylabel("fraction"); ax[0].legend(fontsize=8)
    fig.tight_layout()
    p = cfg.DIR_FIGS / "fig_external_contrast.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  → {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats-root"); ap.add_argument("--t1c-dir"); ap.add_argument("--t1-dir"); ap.add_argument("--seg-dir")
    ap.add_argument("--mendeley-dir")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--runs", nargs="*", default=["mt_base:base", "mt_ctam:base", "cls_resnet50:base", "seg_unet_r50:base"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage", default="all", choices=["prep", "dedup", "infer", "report", "all"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    EXT.mkdir(parents=True, exist_ok=True)
    idx_p, pred_p = EXT / "external_index.csv", EXT / "external_preds.csv"

    if a.stage in ("prep", "all") and (a.rebuild or not idx_p.exists()):
        cfg.banner("prep")
        rows = []
        prep_brats(a, rows); prep_mendeley(a, rows)
        if not rows:
            sys.exit("  ✗ no data found — check the paths.")
        pd.DataFrame(rows).to_csv(idx_p, index=False)
    df = pd.read_csv(idx_p, keep_default_na=False)
    df["label"] = df.label.astype(int)
    print(f"  Index: {len(df)} images | " + ", ".join(f"{k}={v}" for k, v in df.groupby(['source', 'sequence', 'kind']).size().items()))

    if a.stage in ("dedup", "all") and (a.rebuild or "dup_exact" not in df.columns):
        cfg.banner("dedup against BRISC")
        df = dedup(df); df.to_csv(idx_p, index=False)
    for c in ("dup_exact", "dup_near"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
        else:
            df[c] = False

    if a.stage in ("infer", "all"):
        cfg.banner("infer")
        runs = [tuple(r.split(":")) for r in a.runs]
        prev = pd.read_csv(pred_p) if pred_p.exists() and not a.rebuild else pd.DataFrame()
        todo = [(m, t) for m, t in runs if not len(prev) or not ((prev.model == m) & (prev.tag == t)).any()]
        if todo:
            new = infer(df, todo, a.folds, a.batch_size)
            prev = pd.concat([prev, new], ignore_index=True)
            prev.to_csv(pred_p, index=False)
        else:
            print("  ✓ all models have already been run")

    if a.stage in ("report", "all"):
        report(df, pd.read_csv(pred_p))


if __name__ == "__main__":
    main()
