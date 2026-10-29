



import sys
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

cfg = import_module("00_config")


def plot_learning_curves(model_tag: str) -> None:
    files = sorted(cfg.DIR_RESULTS.glob(f"history_*_{model_tag}_f*.csv"))
    if not files:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("train_total", "val_balanced_accuracy"),
              ("val_lesion_dice", "val_empty_mask_accuracy")]

    for f in files:
        fold = f.stem.split("_f")[-1]
        df = pd.read_csv(f)
        if "train_total" in df:
            ax[0, 0].plot(df.epoch, df.train_total, alpha=0.7, label=f"fold{fold}")
        if "val_balanced_accuracy" in df:
            ax[0, 1].plot(df.epoch, df.val_balanced_accuracy, alpha=0.7, label=f"fold{fold}")
        if "val_lesion_dice" in df:
            ax[1, 0].plot(df.epoch, df.val_lesion_dice, alpha=0.7, label=f"fold{fold}")
        if "val_empty_mask_accuracy" in df:
            ax[1, 1].plot(df.epoch, df.val_empty_mask_accuracy, alpha=0.7, label=f"fold{fold}")

    titles = ["Train loss", "Val balanced accuracy",
             "Val lesion Dice", "Val empty-mask accuracy"]
    for a, t in zip(ax.flat, titles):
        a.set_title(t); a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.suptitle(f"Learning curves — {model_tag}")
    plt.tight_layout()
    p = cfg.DIR_FIGS / f"fig5_learning_curves_{model_tag}.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  ✓ {model_tag}: learning curve ({len(files)} folds) → {p.name}")


def subgroup_analysis(model_tag: str) -> None:
    seg_files = sorted(cfg.DIR_RESULTS.glob(f"segper_*_{model_tag}_f*_test.csv"))
    if not seg_files:
        return

    stats = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv")
    folds_df = pd.read_csv(cfg.DIR_GROUPS / "folds.csv")

    all_rows = []
    for f in seg_files:
        fold = int(f.stem.split("_f")[-1].replace("_test", ""))
        seg = pd.read_csv(f)
        te = folds_df[folds_df.P3_role == "test"].reset_index(drop=True)
        if len(seg) != len(te):
            print(f"  ⚠ {f.name}: length {len(seg)} does not match test {len(te)} — skipped")
            continue
        seg = seg.copy()
        seg["uid"] = te["uid"].to_numpy()
        seg["fold"] = fold
        all_rows.append(seg)

    if not all_rows:
        print(f"  ⚠ {model_tag}: no valid segper found for subgroup")
        return

    seg = pd.concat(all_rows, ignore_index=True)
    seg = seg.merge(stats[["uid", "lesion_size_group", "sharpness_group",
                           "snr_group", "plane_name", "class_name"]],
                    on="uid", how="left")

    les = seg[~seg.gt_empty]
    rows = []
    for grp_col in ("lesion_size_group", "sharpness_group", "snr_group",
                    "plane_name", "class_name"):
        for val, sub in les.groupby(grp_col):
            if len(sub) < 5:
                continue
            rows.append({
                "group_type": grp_col, "group_value": val, "n": len(sub),
                "dice_mean": sub.dice.mean(), "dice_std": sub.dice.std(),
                "iou_mean": sub.iou.mean(),
                "hd95_mean": sub.hd95.mean(skipna=True),
                "sensitivity_mean": sub.sensitivity.mean(),
            })
    res = pd.DataFrame(rows)
    out = cfg.DIR_RESULTS / f"subgroup_{model_tag}.csv"
    res.to_csv(out, index=False)
    print(f"  ✓ {model_tag}: subgroup ({len(seg)} samples from {seg.fold.nunique()} folds) "
          f"→ {out.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt = "lesion_size_group"
        sub = res[res.group_type == gt]
        if len(sub):
            order = ["very_small", "small", "medium", "large"]
            sub = sub.set_index("group_value").reindex(
                [o for o in order if o in sub.group_value.values]).reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(sub.group_value, sub.dice_mean, yerr=sub.dice_std, capsize=4,
                  color="steelblue")
            ax.set_ylabel("Dice"); ax.set_title(f"Dice by lesion size — {model_tag}")
            plt.tight_layout()
            p = cfg.DIR_FIGS / f"fig_subgroup_{model_tag}.png"
            plt.savefig(p, dpi=150); plt.close()
    except ImportError:
        pass


def progress_report() -> None:
    cfg.banner("Overall model progress")
    all_metrics = sorted(cfg.DIR_RESULTS.glob("metrics_P3_*.json"))
    import json
    rows = []
    for p in all_metrics:
        with open(p) as f:
            d = json.load(f)
        rows.append({"model": d["model"], "config_tag": d["config_tag"],
                    "fold": d["fold"],
                    "test_bal_acc": d.get("test_balanced_accuracy"),
                    "test_dice": d.get("test_lesion_dice")})
    if not rows:
        print("  No metrics_*.json found yet")
        return
    df = pd.DataFrame(rows)
    g = df.groupby(["model", "config_tag"]).agg(
        folds_done=("fold", lambda s: sorted(s.tolist())),
        n_folds=("fold", "nunique"),
        bal_acc=("test_bal_acc", "mean"),
        dice=("test_dice", "mean"))
    print(g.to_string())
    incomplete = g[g.n_folds < cfg.N_FOLDS]
    if len(incomplete):
        print(f"\n  ⚠ incomplete (fewer than {cfg.N_FOLDS} folds):")
        for idx in incomplete.index:
            print(f"    {idx}: folds {incomplete.loc[idx,'folds_done']}")


def main() -> None:
    progress_report()

    found_models = set()
    for p in cfg.DIR_RESULTS.glob("history_P3_*_f*.csv"):
        name = p.stem.replace("history_P3_", "")
        model_tag = name.rsplit("_f", 1)[0]
        found_models.add(model_tag)

    if not found_models:
        print("\n  No history_*.csv found. Wait until at least one fold finishes.")
        return

    cfg.banner(f"Analysis of {len(found_models)} models: {sorted(found_models)}")
    for mt in sorted(found_models):
        plot_learning_curves(mt)
        subgroup_analysis(mt)

    print("\n  Re-run this script every few hours so it updates "
          "as training progresses.")


if __name__ == "__main__":
    main()
