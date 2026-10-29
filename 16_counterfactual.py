



import argparse, glob, json, os, random, sys
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A

cfg = import_module("00_config")
data_mod = import_module("07_data")
models_mod = import_module("08_models")
inp = import_module("12b_inpainter_v2")
data_mod.build_transforms = import_module("07b_transforms_fix").build_transforms

S = cfg.IMG_SIZE
_resize = A.Compose([A.LongestMaxSize(max_size=S, interpolation=cv2.INTER_LINEAR)])
_clahe = A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0)
MEAN = np.array(cfg.IMAGENET_MEAN, np.float32) * 255
STD = np.array(cfg.IMAGENET_STD, np.float32) * 255
NT = cfg.CLASS2IDX["no_tumor"]
EDITS = ["orig", "A", "S", "B", "Bs", "C"]


def load_raw(row):
    img = np.array(Image.open(row["img_path"]).convert("L"))
    m = np.zeros(img.shape, np.uint8)
    p = row.get("seg_mask_path")
    if isinstance(p, str) and p and p != "nan" and os.path.exists(p):
        m = (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)
    o = _resize(image=img, mask=m)
    ir, mr = o["image"], o["mask"]
    h, w = ir.shape
    top, left = int((S - h) / 2.0), int((S - w) / 2.0)
    I = np.zeros((S, S), np.uint8); M = np.zeros((S, S), np.uint8)
    I[top:top + h, left:left + w] = ir; M[top:top + h, left:left + w] = mr
    return I, M.astype(bool), (top, left, h, w)


def to_tensor(img224, geom):
    top, left, h, w = geom
    crop = np.repeat(img224[top:top + h, left:left + w, None], 3, axis=2)
    crop = _clahe(image=crop)["image"]
    full = np.zeros((S, S, 3), np.uint8)
    full[top:top + h, left:left + w] = crop
    x = (full.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def head_mask(img224, erode=0):
    _, th = cv2.threshold(cv2.GaussianBlur(img224, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        th = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(th)
    cv2.drawContours(f, cnts, -1, 255, -1)
    if erode:
        f = cv2.erode(f, np.ones((erode, erode), np.uint8))
    return f > 0


def dilate(m, px):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def place_shape(mask, allowed, forbid, rng, tries=300):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None, None
    y0, x0 = ys.min(), xs.min()
    hh, ww = ys.max() - y0 + 1, xs.max() - x0 + 1
    for _ in range(tries):
        ny, nx = rng.randrange(0, S - hh + 1), rng.randrange(0, S - ww + 1)
        sm = np.zeros_like(mask)
        sm[ys - y0 + ny, xs - x0 + nx] = True
        if (sm & forbid).any():
            continue
        if (sm & allowed).sum() >= 0.98 * sm.sum():
            return sm, (ny - y0, nx - x0)
    return None, None


def paste(src, src_mask, dst, offset):
    dy, dx = offset
    ys, xs = np.where(src_mask)
    cy, cx = int((ys.min() + ys.max()) / 2 + dy), int((xs.min() + xs.max()) / 2 + dx)
    m = (dilate(src_mask, 2).astype(np.uint8) * 255)
    try:
        out = cv2.seamlessClone(np.repeat(src[..., None], 3, 2), np.repeat(dst[..., None], 3, 2),
                                m, (cx, cy), cv2.NORMAL_CLONE)
        return out[..., 0]
    except cv2.error:
        return None


def build_edits(te, path, seed=0):
    cfg.banner(f"Building edits for {len(te)} P3 test images")
    N = len(te)
    arr = {e: np.zeros((N, S, S), np.uint8) for e in EDITS}
    ok = {e: np.zeros(N, bool) for e in EDITS}
    geoms = np.zeros((N, 4), np.int32)
    area = np.full(N, np.nan, np.float32)
    donor_cls = np.full(N, -1, np.int32)
    raws, masks = [], []
    for i, (_, r) in enumerate(te.iterrows()):
        I, M, g = load_raw(r)
        raws.append(I); masks.append(M); geoms[i] = g
    labels = te["class_idx"].values
    tumor_idx = [i for i in range(N) if labels[i] != NT and masks[i].sum() >= 30]
    for i in range(N):
        rng = random.Random(f"{seed}-{te['uid'].iloc[i]}")
        I, M = raws[i], masks[i]
        arr["orig"][i] = I; ok["orig"][i] = True
        hm = head_mask(I)
        c = I.copy(); c[~hm] = 0
        arr["C"][i] = c; ok["C"][i] = True
        if labels[i] != NT and M.sum() > 0:
            area[i] = M.sum() / max(hm.sum(), 1)
            mA = dilate(M, 3) & hm
            arr["A"][i] = inp.inpaint(I, mA); ok["A"][i] = True
            sm, _ = place_shape(mA, head_mask(I, erode=15), dilate(M, 10), rng)
            if sm is not None:
                arr["S"][i] = inp.inpaint(I, sm); ok["S"][i] = True
        elif labels[i] == NT and tumor_idx:
            for _ in range(20):
                d = rng.choice(tumor_idx)
                frac = masks[d].sum() / max(head_mask(raws[d]).sum(), 1)
                if 0.003 <= frac <= 0.10:
                    break
            Md = masks[d]
            allowed = head_mask(I, erode=15)
            sm, off = place_shape(Md, allowed, np.zeros_like(Md), rng)
            if sm is None:
                continue
            b = paste(raws[d], Md, I, off)
            hs, off_h = place_shape(Md, head_mask(raws[d], erode=15), dilate(Md, 10), rng)
            if b is None or hs is None:
                continue
            healthy = np.zeros_like(Md); healthy[hs] = True
            off_bs = (off[0] - off_h[0], off[1] - off_h[1])
            bs = paste(raws[d], healthy, I, off_bs)
            if bs is None:
                continue
            arr["B"][i], arr["Bs"][i] = b, bs
            ok["B"][i] = ok["Bs"][i] = True
            donor_cls[i] = labels[d]
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{N}")
    np.savez_compressed(path, uids=te["uid"].values.astype(str), labels=labels, geoms=geoms,
                        area=area, donor_cls=donor_cls,
                        **{f"img_{e}": arr[e] for e in EDITS}, **{f"ok_{e}": ok[e] for e in EDITS},
                        masks=np.stack(masks))
    for e in EDITS:
        print(f"  {e:4s}: {int(ok[e].sum())} images")
    return np.load(path, allow_pickle=True)


def sanity_check(E, te, n=24):
    t7 = import_module("07b_transforms_fix")
    ref_tf = A.Compose([A.LongestMaxSize(max_size=S, interpolation=cv2.INTER_LINEAR),
                        A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0),
                        t7.pad_if_needed(S),
                        A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)])
    dmax = 0.0
    path_of = te.set_index("uid")["img_path"]                                              
    for i in range(n):
        img = np.array(Image.open(path_of[str(E["uids"][i])]).convert("L"))
        ref = torch.from_numpy(ref_tf(image=np.stack([img] * 3, -1))["image"].transpose(2, 0, 1))
        dmax = max(dmax, (to_tensor(E["img_orig"][i], E["geoms"][i]) - ref).abs().max().item())
    print(f"  preprocessing consistency with 07b (fixed clip): max|Δ| = {dmax:.2e}  "
          + ("✓" if dmax < 1e-3 else "✗ mismatch — results are not reliable"))
    return dmax


@torch.no_grad()
def predict(model, imgs, geoms, idx, device, bs=32):
    out = np.full((len(imgs), cfg.N_CLASSES), np.nan, np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        x = torch.stack([to_tensor(imgs[k], geoms[k]) for k in j]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            lo = model(x)["logits_cls"]
        out[j] = F.softmax(lo.float(), 1).cpu().numpy()
    return out


def discover_runs():
    runs = set()
    for p in glob.glob(str(cfg.DIR_RESULTS / "metrics_P3_*_f*.json")):
        try:
            d = json.load(open(p))
            if not d["model"].startswith("seg_"):
                runs.add((d["model"], d["config_tag"]))
        except Exception:
            pass
    return sorted(runs)


def run_model(name, tag, fold, E, device):
    ck = cfg.DIR_CKPT / f"P3_{name}_{tag}_f{fold}.pt"
    if not ck.exists():
        print(f"  ⚠ {ck.name} not found"); return None
    model = models_mod.build_model(name, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    geoms, y = E["geoms"], E["labels"]
    P = {e: predict(model, E[f"img_{e}"], geoms, np.where(E[f"ok_{e}"])[0], device) for e in EDITS}
    del model; torch.cuda.empty_cache()
    N = len(y); r = np.arange(N)
    dc = E["donor_cls"]
    d = pd.DataFrame({"model": name, "tag": tag, "fold": fold, "uid": E["uids"], "label": y,
                      "lesion_area_frac": E["area"], "donor_cls": dc})
    for e in EDITS:
        d[f"p_true_{e}"] = P[e][r, y]
        d[f"p_nt_{e}"] = P[e][:, NT]
        d[f"pred_{e}"] = np.where(np.isnan(P[e][:, 0]), -1, np.nan_to_num(P[e]).argmax(1))
        has = dc >= 0
        d[f"p_donor_{e}"] = np.where(has, P[e][r, np.where(has, dc, 0)], np.nan)
    return d


def summarize(df):
    rows = []
    for (m, t, f), g in df.groupby(["model", "tag", "fold"]):
        les = g[(g.label != NT) & g.p_true_A.notna() & g.p_true_S.notna()]
        nt = g[(g.label == NT) & g.p_donor_B.notna() & g.p_donor_Bs.notna()]
        corr = les[les.pred_orig == les.label]
        rows.append(dict(
            model=m, tag=t, fold=f, n_lesion=len(les), n_nt=len(nt),
            LRS=(les.p_true_S - les.p_true_A).mean(),
            LRS_correct=(corr.p_true_S - corr.p_true_A).mean(),
            drop_A=(les.p_true_orig - les.p_true_A).mean(),
            drop_S=(les.p_true_orig - les.p_true_S).mean(),
            flip_A=(les.pred_A == NT).mean(), flip_S=(les.pred_S == NT).mean(),
            INS=(nt.p_donor_B - nt.p_donor_Bs).mean(),
            tumor_B=(nt.pred_B != NT).mean(), tumor_Bs=(nt.pred_Bs != NT).mean(),
            donor_B=(nt.pred_B == nt.donor_cls).mean(),
            BG_shift=(g.p_true_orig - g.p_true_C).abs().mean(),
            flip_C=(g.pred_C != g.pred_orig).mean()))
    per = pd.DataFrame(rows)
    cols = [c for c in per.columns if c not in ("model", "tag", "fold")]
    s = per.groupby(["model", "tag"])[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    les = df[(df.label != NT) & df.p_true_A.notna() & df.p_true_S.notna()].copy()
    q = les.drop_duplicates("uid").lesion_area_frac.quantile([1 / 3, 2 / 3]).values
    les["size"] = pd.cut(les.lesion_area_frac, [-np.inf, q[0], q[1], np.inf], labels=["small", "medium", "large"])
    les["lrs_i"] = les.p_true_S - les.p_true_A
    bys = (les.groupby(["model", "tag", "size", "fold"], observed=True)
           .agg(LRS=("lrs_i", "mean"), flip_A=("pred_A", lambda v: (v == NT).mean()),
                flip_S=("pred_S", lambda v: (v == NT).mean()))
           .groupby(["model", "tag", "size"], observed=True).agg(["mean", "std"]))
    bys.columns = [f"{a}_{b}" for a, b in bys.columns]
    return per, s, bys


def save_figure(E, k=5, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    les = np.where(E["ok_A"] & E["ok_S"])[0]
    nt = np.where(E["ok_B"])[0]
    pick_l = rng.choice(les, min(k, len(les)), replace=False) if len(les) else []
    pick_n = rng.choice(nt, min(k, len(nt)), replace=False) if len(nt) else []
    fig, ax = plt.subplots(6, max(k, 1), figsize=(2.3 * max(k, 1), 14))
    ax = np.atleast_2d(ax)
    titles = ["original (lesion)", "A: lesion removed", "S: sham", "original (no tumor)", "B: lesion inserted", "Bs: sham insert"]
    for j in range(max(k, 1)):
        for r_, (src, idx) in enumerate([("img_orig", pick_l), ("img_A", pick_l), ("img_S", pick_l),
                                         ("img_orig", pick_n), ("img_B", pick_n), ("img_Bs", pick_n)]):
            a = ax[r_, j]; a.axis("off")
            if j < len(idx):
                a.imshow(E[src][idx[j]], cmap="gray", vmin=0, vmax=255)
                if r_ == 0:
                    for c in cv2.findContours(E["masks"][idx[j]].astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)[0]:
                        a.plot(c[:, 0, 0], c[:, 0, 1], "r-", lw=0.6)
            if j == 0:
                a.set_title(titles[r_], fontsize=8, loc="left")
    fig.tight_layout()
    p = cfg.DIR_FIGS / "fig_counterfactual_examples.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="model:tag")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rebuild-edits", action="store_true")
    ap.add_argument("--max-images", type=int, default=0, help="for testing only")
    a = ap.parse_args()
    device = cfg.get_device()
    df = data_mod.load_folds("P3")
    _, _, te = data_mod.get_split(df, "P3", 0)
    te = te.reset_index(drop=True)
    if a.max_images:
        te = te.groupby("class_name", group_keys=False).head(a.max_images // 4).reset_index(drop=True)

    ep = cfg.DIR_RESULTS / ("cf_edits_P3.npz" if not a.max_images else "cf_edits_P3_test.npz")
    E = build_edits(te, ep) if (a.rebuild_edits or not ep.exists()) else np.load(ep, allow_pickle=True)
    if list(E["uids"]) != list(te["uid"].astype(str)):
        print("  ⚠ saved edits do not match the current test set — rebuilding")
        E = build_edits(te, ep)
    sanity_check(E, te, n=min(24, len(te)))
    save_figure(E)

    runs = [tuple(r.split(":")) for r in a.runs] if a.runs else discover_runs()
    print(f"\n  {len(runs)} models: {runs}")
    out_p = cfg.DIR_RESULTS / "cf_P3_perimage.csv"
    prev = pd.read_csv(out_p) if out_p.exists() and not a.max_images else pd.DataFrame()
    parts = [prev] if len(prev) else []
    for m, t in runs:
        for f in a.folds:
            if len(prev) and ((prev.model == m) & (prev.tag == t) & (prev.fold == f)).any():
                print(f"  ✓ P3_{m}_{t}_f{f} already computed"); continue
            d = run_model(m, t, f, E, device)
            if d is None:
                continue
            parts.append(d)
            les = d[(d.label != NT) & d.p_true_A.notna() & d.p_true_S.notna()]
            print(f"  P3_{m}_{t}_f{f}: LRS={(les.p_true_S - les.p_true_A).mean():+.3f}  "
                  f"flip_A={(les.pred_A == NT).mean():.3f} flip_S={(les.pred_S == NT).mean():.3f}")
            pd.concat(parts, ignore_index=True).to_csv(out_p, index=False)
    if not parts:
        return
    allp = pd.concat(parts, ignore_index=True)
    allp.to_csv(out_p, index=False)
    per, s, bys = summarize(allp)
    s.to_csv(cfg.DIR_RESULTS / "cf_P3_summary.csv")
    bys.to_csv(cfg.DIR_RESULTS / "cf_P3_by_size.csv")
    cfg.banner("Summary (mean ± SD across folds)")
    keys = ["LRS", "flip_A", "flip_S", "INS", "tumor_B", "tumor_Bs", "BG_shift", "flip_C"]
    show = pd.DataFrame({k: s[f"{k}_mean"].round(3).astype(str) + "±" + s[f"{k}_std"].round(3).astype(str)
                         for k in keys})
    print(show.to_string())
    print("\n  LRS by lesion size:")
    print(bys[["LRS_mean", "LRS_std", "flip_A_mean", "flip_S_mean"]].round(3).to_string())
    print(f"\n  → {out_p}\n  → {cfg.DIR_RESULTS / 'cf_P3_summary.csv'}\n  → {cfg.DIR_RESULTS / 'cf_P3_by_size.csv'}")


if __name__ == "__main__":
    main()
