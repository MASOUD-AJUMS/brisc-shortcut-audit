




import argparse
import sys
from pathlib import Path

import numpy as np

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, confusion_matrix)

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 15) -> tuple[float, float]:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray,
                           classes: list[str] = None) -> dict:
    classes = classes or cfg.CLASSES
    K = len(classes)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(K))

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "n": len(y_true),
    }

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(K), zero_division=0)
    for k, c in enumerate(classes):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[f"precision_{c}"] = p[k]
        out[f"sensitivity_{c}"] = r[k]
        out[f"f1_{c}"] = f[k]
        out[f"specificity_{c}"] = tn / max(tn + fp, 1)
        out[f"npv_{c}"] = tn / max(tn + fn, 1)
        out[f"support_{c}"] = int(sup[k])

    out["macro_precision"] = p.mean()
    out["macro_sensitivity"] = r.mean()
    out["macro_f1"] = f.mean()
    out["macro_specificity"] = np.mean([out[f"specificity_{c}"] for c in classes])

    try:
        oh = np.eye(K)[y_true]
        aucs, aps = [], []
        for k, c in enumerate(classes):
            if oh[:, k].sum() in (0, len(y_true)):
                out[f"auc_{c}"] = np.nan
                out[f"ap_{c}"] = np.nan
                continue
            a = roc_auc_score(oh[:, k], probs[:, k])
            ap = average_precision_score(oh[:, k], probs[:, k])
            out[f"auc_{c}"] = a
            out[f"ap_{c}"] = ap
            aucs.append(a); aps.append(ap)
        out["macro_auc"] = np.mean(aucs) if aucs else np.nan
        out["macro_ap"] = np.mean(aps) if aps else np.nan
        out["weighted_auc"] = roc_auc_score(y_true, probs, multi_class="ovr",
                                            average="weighted")
    except Exception:
        out["macro_auc"] = out["macro_ap"] = out["weighted_auc"] = np.nan

    ece, mce = expected_calibration_error(probs, y_true)
    out["ece"] = ece
    out["mce"] = mce
    return out


def _surface(binary: np.ndarray) -> np.ndarray:
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)
    er = binary_erosion(binary, iterations=1, border_value=0)
    return binary & (~er)


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray | None:
    if not HAS_SCIPY:
        return None
    p, g = pred.astype(bool), gt.astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return None
    sp, sg = _surface(p), _surface(g)
    dt_g = distance_transform_edt(~sg)
    dt_p = distance_transform_edt(~sp)
    return np.concatenate([dt_g[sp], dt_p[sg]])


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray,
                         nsd_tau: float = 2.0) -> dict:

    p, g = pred.astype(bool), gt.astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    tn = float((~p & ~g).sum())

    gt_empty = g.sum() == 0
    pr_empty = p.sum() == 0

    out = {
        "gt_empty": gt_empty,
        "pred_empty": pr_empty,
        "empty_correct": float(gt_empty and pr_empty),
        "fp_area": fp / p.size,
        "specificity": tn / max(tn + fp, 1),
    }

    if gt_empty:
        out.update(dice=np.nan, iou=np.nan, sensitivity=np.nan,
                   precision=np.nan, hd95=np.nan, assd=np.nan, nsd=np.nan)
        return out

    out["dice"] = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    out["iou"] = tp / max(tp + fp + fn, 1e-9)
    out["sensitivity"] = tp / max(tp + fn, 1e-9)
    out["precision"] = tp / max(tp + fp, 1e-9) if not pr_empty else 0.0

    d = surface_distances(p, g)
    if d is None:
        out.update(hd95=np.nan, assd=np.nan, nsd=np.nan)
    else:
        out["hd95"] = float(np.percentile(d, 95))
        out["assd"] = float(d.mean())
        out["nsd"] = float((d <= nsd_tau).mean())
    return out


def aggregate_segmentation(per_image: list[dict]) -> dict:
    import pandas as pd
    df = pd.DataFrame(per_image)
    out = {"n_images": len(df), "n_lesion": int((~df.gt_empty).sum()),
           "n_empty": int(df.gt_empty.sum())}

    les = df[~df.gt_empty]
    for m in ("dice", "iou", "sensitivity", "precision", "hd95", "assd", "nsd"):
        if m in les:
            out[f"lesion_{m}"] = float(les[m].mean())
            out[f"lesion_{m}_std"] = float(les[m].std())

    emp = df[df.gt_empty]
    if len(emp):
        out["empty_mask_accuracy"] = float(emp.empty_correct.mean())
        out["empty_fp_area"] = float(emp.fp_area.mean())
    else:
        out["empty_mask_accuracy"] = np.nan
        out["empty_fp_area"] = np.nan

    d = df.dice.copy()
    d[df.gt_empty] = df.loc[df.gt_empty, "empty_correct"]
    out["overall_dice"] = float(d.mean())
    out["overall_specificity"] = float(df.specificity.mean())
    return out


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = cfg.GLOBAL_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    base = fn(*arrays)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(fn(*[a[i] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return base, np.nan, np.nan
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(base), lo, hi


def self_test() -> None:
    cfg.set_seed()
    rng = np.random.default_rng(0)

    cfg.banner("Classification")
    n, K = 400, cfg.N_CLASSES
    y = rng.integers(0, K, n)
    logits = rng.normal(0, 1, (n, K))
    logits[np.arange(n), y] += 2.5                         
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    m = classification_metrics(y, probs)
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc",
              "macro_ap", "mcc", "kappa", "ece"):
        print(f"  {k:20s} = {m[k]:.4f}")

    cfg.banner("Segmentation — slice with lesion")
    gt = np.zeros((128, 128), bool); gt[40:80, 50:95] = True
    pr = np.zeros((128, 128), bool); pr[44:82, 52:92] = True
    s = segmentation_metrics(pr, gt)
    for k, v in s.items():
        print(f"  {k:16s} = {v if isinstance(v,bool) else round(float(v),4)}")

    cfg.banner("Segmentation — healthy slice")
    e = np.zeros((128, 128), bool)
    s1 = segmentation_metrics(e, e)
    s2 = segmentation_metrics(pr, e)
    print(f"  correct empty prediction : empty_correct={s1['empty_correct']}, "
          f"dice={s1['dice']}")
    print(f"  false positive           : empty_correct={s2['empty_correct']}, "
          f"fp_area={s2['fp_area']:.4f}")

    cfg.banner("Aggregation")
    per = [segmentation_metrics(pr, gt) for _ in range(30)]
    per += [segmentation_metrics(e, e) for _ in range(8)]
    per += [segmentation_metrics(pr, e) for _ in range(2)]
    agg = aggregate_segmentation(per)
    for k, v in agg.items():
        print(f"  {k:24s} = {v if isinstance(v,int) else round(float(v),4)}")
    print("\n  Note: lesion_dice and overall_dice are intentionally different.")
    print("  The existing literature reports only lesion_dice — that is the gap.")

    cfg.banner("Bootstrap")
    b, lo, hi = bootstrap_ci(lambda a, p: accuracy_score(a, p.argmax(1)),
                             y, probs, n_boot=300)
    print(f"  accuracy = {b:.4f}  CI95 = [{lo:.4f}, {hi:.4f}]")
    print(f"\n  scipy available: {HAS_SCIPY}  (without it HD95/ASSD/NSD do not work)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        print("This is a module. To test:  python 10_metrics.py --self-test")
