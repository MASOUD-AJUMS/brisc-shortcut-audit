








import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from importlib import import_module

cfg = import_module("00_config")
data_mod = import_module("07_data")
models_mod = import_module("08_models")
losses_mod = import_module("09_losses")
metrics_mod = import_module("10_metrics")

try:
    data_mod.build_transforms = import_module("07b_transforms_fix").build_transforms
except Exception as e:
    print(f"⚠ 07b_transforms_fix failed to load ({e}) — using the built-in version")


def warn_if_ephemeral() -> None:
    p = str(cfg.OUT_ROOT.resolve())
    if p.startswith("/content") and "/drive/" not in p:
        print("\n" + "!" * 70)
        print("⚠️  OUT_ROOT is on temporary Colab storage:")
        print(f"    {p}")
        print("    If the session is fully disconnected this path is wiped and resume will not work.")
        print("    Suggestion: run this before starting (or set it in .env):")
        print('    %env BRISC_OUT=/content/drive/MyDrive/BRISC_2025/outputs')
        print("!" * 70 + "\n")


@torch.no_grad()
def evaluate(model, loader, device, thr: float = 0.5, collect_seg: bool = True):
    model.eval()
    P, Y, U = [], [], []
    seg_per_image = []

    for b in tqdm(loader, desc="eval", leave=False):
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x, return_feat=False)

        if out.get("logits_cls") is not None:
            P.append(F.softmax(out["logits_cls"].float(), 1).cpu().numpy())
            Y.append(b["label"].numpy())
            U.extend(b["uid"])

        if collect_seg and out.get("logits_seg") is not None:
            pm = (torch.sigmoid(out["logits_seg"].float()) > thr).cpu().numpy()
            gm = b["mask"].numpy() > 0.5
            for i in range(pm.shape[0]):
                seg_per_image.append(
                    metrics_mod.segmentation_metrics(pm[i, 0], gm[i, 0]))

    res = {}
    if P:
        probs = np.concatenate(P)
        y = np.concatenate(Y)
        res["cls"] = metrics_mod.classification_metrics(y, probs)
        res["_probs"], res["_y"], res["_uids"] = probs, y, np.array(U)
    if seg_per_image:
        res["seg"] = metrics_mod.aggregate_segmentation(seg_per_image)
        res["_seg_per_image"] = seg_per_image
    return res


def train_epoch(model, loader, crit, opt, scaler, sched, device, args, ep: int):
    model.train()
    agg, n = {}, 0

    pbar = tqdm(loader, desc=f"ep{ep:03d}", leave=False)
    for b in pbar:
        x = b["image"].to(device, non_blocking=True)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in b.items() if k != "uid"}

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x, return_feat=True)

            out_abl = None
            if args.w_cf > 0:
                xa = losses_mod.crude_ablate(x, batch["mask"])
                out_abl = model(xa, return_feat=False)

            loss, parts = crit(out, batch, out_abl)

        if not torch.isfinite(loss):
            print("  ⚠ invalid loss, batch skipped")
            continue

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(opt)
        scaler.update()
        try:
            sched.step()
        except Exception:
            pass

        bs = x.size(0)
        n += bs
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
        pbar.set_postfix(loss=f"{float(parts['total']):.3f}")

    return {k: v / max(n, 1) for k, v in agg.items()}


def _rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
         "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _restore_rng(s: dict) -> None:
    try:
        torch.set_rng_state(s["torch"])
        np.random.set_state(s["numpy"])
        random.setstate(s["python"])
        if torch.cuda.is_available() and "cuda" in s:
            torch.cuda.set_rng_state_all(s["cuda"])
    except Exception as e:
        print(f"  ⚠ failed to restore rng state ({e}) — continuing without it")


def save_resume(path, model, crit, opt, sched, scaler, ep, hist, best,
                best_ep, patience, finished, args) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "epoch": ep, "hist": hist, "best": best, "best_ep": best_ep,
        "patience": patience, "finished": finished,
        "model": model.state_dict(), "crit": crit.state_dict(),
        "opt": opt.state_dict(), "sched": sched.state_dict(),
        "scaler": scaler.state_dict(), "rng": _rng_state(),
        "epochs_target": args.epochs,
    }, tmp)
    tmp.replace(path)                                                           


def load_resume(path, device) -> dict | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ⚠ resume checkpoint is corrupted ({e}), starting from scratch")
        return None


def run_fold(args, df, fold: int, device) -> dict:
    tag = f"{args.protocol}_{args.model}_{args.tag}_f{fold}"
    cfg.banner(f"{tag}")

    best_ckpt = cfg.DIR_CKPT / f"{tag}.pt"
    resume_ckpt = cfg.DIR_CKPT / f"resume_{tag}.pt"
    metrics_path = cfg.DIR_RESULTS / f"metrics_{tag}.json"

    if not args.fresh and metrics_path.exists() and best_ckpt.exists():
        print(f"  ✓ already completed, loading from disk: {metrics_path.name}")
        with open(metrics_path) as f:
            return json.load(f)

    cfg.set_seed(args.seed + fold)

    tr, va, te = data_mod.get_split(df, args.protocol, fold)
    print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")

    dl_tr = data_mod.make_loader(tr, True, args.batch_size, args.workers,
                                 args.aug, args.balanced_sampler)
    dl_va = data_mod.make_loader(va, False, args.batch_size, args.workers)
    dl_te = data_mod.make_loader(te, False, args.batch_size, args.workers)

    model = models_mod.build_model(args.model, pretrained=not args.no_pretrained)
    model = model.to(device)
    p_tot, _ = models_mod.count_params(model)
    print(f"  parameters: {p_tot:.1f}M")

    cw = None
    if args.class_weights:
        c = tr["class_name"].value_counts()
        cw = [len(tr) / (cfg.N_CLASSES * c.get(k, 1)) for k in cfg.CLASSES]
        print(f"  class weights: {[round(w,3) for w in cw]}")

    crit = losses_mod.TotalLoss(
        w_con=args.w_con, w_evidence=args.w_evidence, w_cf=args.w_cf,
        w_empty=args.w_empty, w_ds=args.w_ds, use_uw=not args.no_uw,
        class_weights=cw).to(device)

    params = list(model.parameters()) + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(dl_tr), 1),
        pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best, best_ep, patience, start_ep = [], -np.inf, -1, 0, 0

    ck = None if args.fresh else load_resume(resume_ckpt, device)
    if ck is not None:
        if ck.get("epochs_target") != args.epochs:
            print(f"  ⚠ previous epochs ({ck.get('epochs_target')}) differ from the current "
                  f"({args.epochs}) — scheduler restarts from zero, "
                  f"weights are kept")
            load_sched = False
        else:
            load_sched = True

        model.load_state_dict(ck["model"])
        crit.load_state_dict(ck["crit"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if load_sched:
            try:
                sched.load_state_dict(ck["sched"])
            except Exception as e:
                print(f"  ⚠ failed to restore scheduler ({e})")
        _restore_rng(ck["rng"])

        hist = ck["hist"]
        best, best_ep, patience = ck["best"], ck["best_ep"], ck["patience"]
        start_ep = ck["epoch"] + 1

        if ck.get("finished") or start_ep >= args.epochs:
            print(f"  ✓ resume indicates this fold already finished "
                  f"(up to epoch {ck['epoch']}) — going straight to final evaluation")
            start_ep = args.epochs                 
        else:
            print(f"  ↻ resuming from epoch {start_ep} "
                  f"(previous best: {best:.4f} at epoch {best_ep})")
    else:
        print("  starting from scratch")

    t0 = time.time()
    finished = start_ep >= args.epochs

    for ep in range(start_ep, args.epochs):
        trm = train_epoch(model, dl_tr, crit, opt, scaler, sched, device, args, ep)

        vres = evaluate(model, dl_va, device, args.seg_thr)
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in trm.items()})
        if "cls" in vres:
            row.update({f"val_{k}": v for k, v in vres["cls"].items()
                        if isinstance(v, (int, float))})
        if "seg" in vres:
            row.update({f"val_{k}": v for k, v in vres["seg"].items()
                        if isinstance(v, (int, float))})
        hist.append(row)

        sel = []
        if "cls" in vres:
            sel.append(vres["cls"]["balanced_accuracy"])
        if "seg" in vres and not np.isnan(vres["seg"].get("lesion_dice", np.nan)):
            sel.append(vres["seg"]["lesion_dice"])
        score = float(np.mean(sel)) if sel else -np.inf

        msg = f"  ep{ep:03d} loss={trm.get('total',0):.4f}"
        if "cls" in vres:
            msg += f" bal_acc={vres['cls']['balanced_accuracy']:.4f}"
        if "seg" in vres:
            msg += (f" dice={vres['seg'].get('lesion_dice',float('nan')):.4f}"
                    f" empty={vres['seg'].get('empty_mask_accuracy',float('nan')):.3f}")
        if "EIT" in trm:
            msg += f" EIT={trm['EIT']:.3f}"

        if score > best + args.min_delta:
            best, best_ep, patience = score, ep, 0
            torch.save({"model": model.state_dict(), "crit": crit.state_dict(),
                        "epoch": ep, "score": score, "args": vars(args)}, best_ckpt)
            msg += "  ★"
        else:
            patience += 1
        print(msg)

        stop = patience >= args.patience
        if stop:
            print(f"  early stopping at epoch {ep} (best: {best_ep})")

        finished = stop or (ep == args.epochs - 1)
        save_resume(resume_ckpt, model, crit, opt, sched, scaler, ep, hist,
                   best, best_ep, patience, finished, args)

        if stop:
            break

    mins = (time.time() - t0) / 60
    if mins > 0.01:
        print(f"  run time: {mins:.1f} min | best epoch: {best_ep}")

    pd.DataFrame(hist).to_csv(cfg.DIR_RESULTS / f"history_{tag}.csv", index=False)

    if not best_ckpt.exists():
        sys.exit(f"  ✗ no best checkpoint was created for {tag} — check")
    model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=False)["model"])
    out = {"tag": tag, "protocol": args.protocol, "model": args.model,
           "config_tag": args.tag, "fold": fold, "best_epoch": best_ep,
           "train_minutes": mins, "params_M": p_tot,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te)}

    for split, dl in (("val", dl_va), ("test", dl_te)):
        r = evaluate(model, dl, device, args.seg_thr)
        if "cls" in r:
            out.update({f"{split}_{k}": v for k, v in r["cls"].items()})
        if "seg" in r:
            out.update({f"{split}_{k}": v for k, v in r["seg"].items()})
        if "_probs" in r:
            np.savez_compressed(
                cfg.DIR_RESULTS / f"preds_{tag}_{split}.npz",
                probs=r["_probs"], y=r["_y"], uids=r["_uids"])
        if "_seg_per_image" in r:
            pd.DataFrame(r["_seg_per_image"]).to_csv(
                cfg.DIR_RESULTS / f"segper_{tag}_{split}.csv", index=False)

    print(f"\n  test bal_acc = {out.get('test_balanced_accuracy', float('nan')):.4f}")
    print(f"  test lesion_dice = {out.get('test_lesion_dice', float('nan')):.4f}")
    print(f"  test overall_dice = {out.get('test_overall_dice', float('nan')):.4f}")
    print(f"  test empty_mask_acc = "
          f"{out.get('test_empty_mask_accuracy', float('nan')):.4f}")

    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in out.items()}
    with open(metrics_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    return clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mt_ctam", choices=models_mod.MODEL_ZOO)
    ap.add_argument("--protocol", default="P3", choices=["P1", "P2", "P3"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="base", help="configuration label, e.g. B7")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--aug", default="medium",
                    choices=["none", "light", "medium", "strong"])
    ap.add_argument("--seg-thr", type=float, default=0.5)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=cfg.GLOBAL_SEED)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--no-uw", action="store_true", help="disable uncertainty weighting")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any previous resume/result and start from scratch")
    ap.add_argument("--w-con", type=float, default=0.1)
    ap.add_argument("--w-evidence", type=float, default=0.0)
    ap.add_argument("--w-cf", type=float, default=0.0)
    ap.add_argument("--w-empty", type=float, default=0.2)
    ap.add_argument("--w-ds", type=float, default=0.3)
    args = ap.parse_args()

    device = cfg.get_device()
    warn_if_ephemeral()
    cfg.banner("Configuration")
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print(f"  {'device':20s}: {device}")
    if device.type == "cuda":
        print(f"  {'gpu':20s}: {torch.cuda.get_device_name(0)}")

    df = data_mod.load_folds(args.protocol)
    for f in args.folds:
        run_fold(args, df, f, device)

    prefix = f"metrics_{args.protocol}_{args.model}_{args.tag}_f"
    found = sorted(cfg.DIR_RESULTS.glob(f"{prefix}*.json"))
    rows = []
    for p in found:
        with open(p) as fh:
            rows.append(json.load(fh))
    if not rows:
        print("  ⚠ no metrics_*.json found; summary not created")
        return

    res = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    name = f"summary_{args.protocol}_{args.model}_{args.tag}.csv"
    res.to_csv(cfg.DIR_RESULTS / name, index=False)

    cfg.banner(f"Fold summary  (from {len(res)} folds available on disk: "
              f"{sorted(res.fold.tolist())})")
    show = [c for c in ("test_balanced_accuracy", "test_macro_auc", "test_macro_f1",
                        "test_lesion_dice", "test_lesion_iou", "test_lesion_hd95",
                        "test_overall_dice", "test_empty_mask_accuracy", "test_ece")
            if c in res.columns]
    if show:
        t = res[["fold"] + show].copy()
        print(t.round(4).to_string(index=False))
        print("\n  Mean ± standard deviation:")
        for c in show:
            print(f"    {c:32s} {res[c].mean():.4f} ± {res[c].std():.4f}")
    if len(res) < cfg.N_FOLDS:
        print(f"\n  ⚠ only {len(res)}/{cfg.N_FOLDS} folds completed — for the final "
              f"paper table run the rest as well.")
    print(f"\n  → {cfg.DIR_RESULTS/name}")


if __name__ == "__main__":
    main()
