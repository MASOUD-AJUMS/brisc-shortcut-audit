


import argparse, glob, json, os, re, sys
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

cfg = import_module("00_config")
data_mod = import_module("07_data")
models_mod = import_module("08_models")
try:
    data_mod.build_transforms = import_module("07b_transforms_fix").build_transforms
except Exception as e:
    print(f"⚠ 07b_transforms_fix failed to load ({e})")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def discover_runs():
    runs = set()
    for p in glob.glob(str(cfg.DIR_RESULTS / "metrics_P3_*_f*.json")):
        try:
            d = json.load(open(p))
            runs.add((d["model"], d["config_tag"]))
        except Exception:
            pass
    return sorted(runs)


def head_mask(x):
    import cv2
    raw = (x.cpu() * STD + MEAN).mean(1).numpy()                           
    out = []
    for r in raw:
        m = (r > 0.05).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        f = np.zeros_like(m)
        if cnts:
            cv2.drawContours(f, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        out.append(f.astype(bool))
    return torch.from_numpy(np.stack(out))[:, None]


def target_layer(model):
    if hasattr(model, "encoder"):                                    
        return model.encoder, lambda o: o[-1]
    if hasattr(model, "net") and hasattr(model.net, "layer4"):                
        return model.net.layer4, lambda o: o
    if hasattr(model, "net") and hasattr(model.net, "forward_features"):
        return None, None
    return None, None


def seg_eit(seg_feat, mask, eps=1e-6):
    s = F.relu(seg_feat.mean(1, keepdim=True))
    if s.shape[-2:] != mask.shape[-2:]:
        s = F.interpolate(s, mask.shape[-2:], mode="bilinear", align_corners=False)
    s = s.flatten(1)
    s = s - s.min(1, keepdim=True).values
    s = s / (s.max(1, keepdim=True).values + eps)
    m = mask.flatten(1)
    return ((s * m).sum(1) / (s.sum(1) + eps))


def run_one(model_name, tag, fold, df, device, bs, workers, dilate_px):
    run = f"P3_{model_name}_{tag}_f{fold}"
    ck = cfg.DIR_CKPT / f"{run}.pt"
    if not ck.exists():
        print(f"  ⚠ {ck.name} not found — skipped"); return None
    model = models_mod.build_model(model_name, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    _, _, te = data_mod.get_split(df, "P3", fold)
    te = te[te["class_name"] != "no_tumor"]
    dl = data_mod.make_loader(te, False, bs, workers)

    layer, pick = target_layer(model)
    store = {}
    if layer is not None:
        def hook(m, i, o):
            t = pick(o); t.retain_grad(); store["a"] = t
        h = layer.register_forward_hook(hook)
    kern = torch.ones(1, 1, 2 * dilate_px + 1, 2 * dilate_px + 1, device=device)

    rows = []
    for b in dl:
        x = b["image"].to(device); mask = b["mask"].to(device).float()
        y = b["label"].to(device)
        hm = head_mask(b["image"]).to(device).float()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(x, return_feat=True)
            r = {"uid": list(b["uid"]), "label": y.cpu().numpy()}
            if out.get("seg_feat") is not None:
                r["seg_EIT"] = seg_eit(out["seg_feat"].detach().float(), mask).cpu().numpy()
            if out.get("logits_cls") is not None and layer is not None:
                logit = out["logits_cls"].float()
                r["p_true"] = F.softmax(logit.detach(), 1).gather(1, y[:, None])[:, 0].cpu().numpy()
                logit.gather(1, y[:, None]).sum().backward()
                A, G = store["a"].detach().float(), store["a"].grad.float()
                cam = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
                cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)
                md = (F.conv2d(mask, kern, padding=dilate_px) > 0).float()
                tot = cam.flatten(1).sum(1) + 1e-8
                r["cam_EIT"] = ((cam * mask).flatten(1).sum(1) / tot).cpu().numpy()
                r["cam_EIT_d"] = ((cam * md).flatten(1).sum(1) / tot).cpu().numpy()
                area = (mask * hm).flatten(1).sum(1) / hm.flatten(1).sum(1).clamp(min=1)
                r["lesion_area_frac"] = area.cpu().numpy()
                r["enrichment"] = (r["cam_EIT"] / np.maximum(r["lesion_area_frac"], 1e-6))
                r["bg_mass"] = ((cam * (1 - hm)).flatten(1).sum(1) / tot).cpu().numpy()
                am = cam.flatten(1).argmax(1)
                r["pointing"] = md.flatten(1).gather(1, am[:, None])[:, 0].cpu().numpy()
        n = len(r["uid"])
        for i in range(n):
            rows.append({k: (v[i] if hasattr(v, "__len__") and len(v) == n else v) for k, v in r.items()})
    if layer is not None:
        h.remove()
    d = pd.DataFrame(rows)
    d.insert(0, "fold", fold); d.insert(0, "tag", tag); d.insert(0, "model", model_name)
    msg = "  ".join(f"{c}={d[c].mean():.3f}" for c in
                    ["seg_EIT", "cam_EIT", "cam_EIT_d", "enrichment", "pointing", "bg_mass"] if c in d)
    print(f"  {run:32s} n={len(d)}  {msg}")
    del model; torch.cuda.empty_cache()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="model:tag, e.g. mt_base:base")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dilate", type=int, default=7)
    a = ap.parse_args()
    device = cfg.get_device()
    runs = [tuple(r.split(":")) for r in a.runs] if a.runs else discover_runs()
    print(f"  {len(runs)} configurations: {runs}")
    df = data_mod.load_folds("P3")

    out_p = cfg.DIR_RESULTS / "attribution_P3_perimage.csv"
    prev = pd.read_csv(out_p) if out_p.exists() else pd.DataFrame()
    parts = [prev] if len(prev) else []
    for m, t in runs:
        for f in a.folds:
            if len(prev) and ((prev.model == m) & (prev.tag == t) & (prev.fold == f)).any():
                print(f"  ✓ P3_{m}_{t}_f{f} already computed"); continue
            d = run_one(m, t, f, df, device, a.batch_size, a.workers, a.dilate)
            if d is not None:
                parts.append(d)
                pd.concat(parts, ignore_index=True).to_csv(out_p, index=False)                   
    allp = pd.concat(parts, ignore_index=True)
    allp.to_csv(out_p, index=False)

    metrics = [c for c in ["seg_EIT", "cam_EIT", "cam_EIT_d", "enrichment", "pointing", "bg_mass", "p_true"]
               if c in allp.columns]
    per_fold = allp.groupby(["model", "tag", "fold"])[metrics].mean()
    summ = per_fold.groupby(["model", "tag"]).agg(["mean", "std"])
    summ.columns = [f"{a_}_{b_}" for a_, b_ in summ.columns]
    summ.to_csv(cfg.DIR_RESULTS / "attribution_P3_summary.csv")
    cfg.banner("Summary (mean across folds; only lesion-containing P3 test images)")
    show = pd.DataFrame({m_: summ[f"{m_}_mean"].round(3).astype(str) + " ± " + summ[f"{m_}_std"].round(3).astype(str)
                         for m_ in metrics})
    print(show.to_string())
    print(f"\n  → {out_p}\n  → {cfg.DIR_RESULTS / 'attribution_P3_summary.csv'}")


if __name__ == "__main__":
    main()
