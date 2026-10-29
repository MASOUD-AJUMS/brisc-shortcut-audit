




import sys
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

from sklearn.model_selection import StratifiedGroupKFold


def report(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    print(f"\n  [{name}]")
    t = pd.crosstab(df[col], df["class_name"], margins=True)
    print(t.to_string())
    return t


def check_no_group_overlap(df: pd.DataFrame, col: str) -> None:
    g = df.groupby("pseudo_patient_id")[col].nunique()
    bad = int((g > 1).sum())
    flag = "✓" if bad == 0 else "✗"
    print(f"    {flag} pseudo-patients shared between parts: {bad}")
    if bad:
        raise AssertionError(f"Leakage in {col}! Must be zero.")


def make_folds(sub: pd.DataFrame, seed: int) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=seed)
    out = np.full(len(sub), -1, dtype=int)
    for k, (_, va) in enumerate(sgkf.split(sub, sub["class_name"],
                                           groups=sub["pseudo_patient_id"])):
        out[va] = k
    return out


def main() -> None:
    cfg.set_seed()
    pp = cfg.DIR_GROUPS / "pseudo_patients.csv"
    if not pp.exists():
        sys.exit("pseudo_patients.csv not found. Run 05_pseudo_patients.py first.")
    df = pd.read_csv(pp)
    stats = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv")[
        ["uid", "img_path", "seg_mask_path", "mask_area_ratio",
         "lesion_size_group", "mask_is_empty"]]
    df = df.merge(stats, on="uid", how="left").reset_index(drop=True)
    print(f"{len(df)} images | {df.pseudo_patient_id.nunique()} pseudo-patients")

    cfg.banner("P1 — official split (untouched)")
    df["P1_role"] = df["official_split"].map({"train": "trainval", "test": "test"})
    report(df, "P1_role", "P1")
    n_bad = df.groupby("pseudo_patient_id")["P1_role"].nunique().gt(1).sum()
    print(f"\n    ⚠ pseudo-patients shared between train/test: {n_bad}  ← this is the leakage")

    tv = df[df.P1_role == "trainval"].copy()
    df.loc[tv.index, "P1_fold"] = make_folds(tv, cfg.GLOBAL_SEED)
    df["P1_fold"] = df["P1_fold"].fillna(-1).astype(int)

    cfg.banner("P2 — cleaned test")
    contaminated = set(
        df[(df.official_split == "test") & (df.pp_spans_official_split)].uid)
    df["P2_role"] = df["P1_role"]
    df.loc[df.uid.isin(contaminated), "P2_role"] = "excluded"
    print(f"    {len(contaminated)} test images removed "
          f"({100*len(contaminated)/1000:.1f}%)")
    report(df, "P2_role", "P2")
    check_no_group_overlap(df[df.P2_role != "excluded"], "P2_role")
    df["P2_fold"] = df["P1_fold"]

    cfg.banner("P3 — full re-split with pseudo-patient grouping")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.GLOBAL_SEED)
    tr_idx, te_idx = next(sgkf.split(df, df["class_name"],
                                     groups=df["pseudo_patient_id"]))
    df["P3_role"] = "trainval"
    df.loc[df.index[te_idx], "P3_role"] = "test"
    report(df, "P3_role", "P3")
    check_no_group_overlap(df, "P3_role")

    tv3 = df[df.P3_role == "trainval"].copy()
    df.loc[tv3.index, "P3_fold"] = make_folds(tv3, cfg.GLOBAL_SEED)
    df["P3_fold"] = df["P3_fold"].fillna(-1).astype(int)

    cfg.banner("Fold validation")
    for proto in ("P1", "P2", "P3"):
        rc, fc = f"{proto}_role", f"{proto}_fold"
        sub = df[df[rc] == "trainval"]
        print(f"\n  [{proto}]  n={len(sub)}")
        print(pd.crosstab(sub[fc], sub["class_name"], margins=True).to_string())
        bad = int(sub.groupby("pseudo_patient_id")[fc].nunique().gt(1).sum())
        print(f"    {'✓' if bad==0 else '✗'} pseudo-patients in multiple folds: {bad}")
        print("    Plane distribution across folds (should be uniform):")
        print(pd.crosstab(sub[fc], sub["plane_name"], normalize="index")
              .round(3).to_string())

    keep = ["uid", "official_split", "class_name", "plane_name",
            "pseudo_patient_id", "pp_size", "pp_spans_official_split",
            "img_path", "seg_mask_path", "mask_area_ratio", "mask_is_empty",
            "lesion_size_group",
            "P1_role", "P1_fold", "P2_role", "P2_fold", "P3_role", "P3_fold"]
    out = cfg.DIR_GROUPS / "folds.csv"
    df[keep].to_csv(out, index=False)

    summ = []
    for proto in ("P1", "P2", "P3"):
        rc = f"{proto}_role"
        summ.append({
            "protocol": proto,
            "n_trainval": int((df[rc] == "trainval").sum()),
            "n_test": int((df[rc] == "test").sum()),
            "n_excluded": int((df[rc] == "excluded").sum()),
            "group_leakage": int(df[df[rc] != "excluded"]
                                 .groupby("pseudo_patient_id")[rc].nunique().gt(1).sum()),
        })
    s = pd.DataFrame(summ)
    s.to_csv(cfg.DIR_GROUPS / "protocol_summary.csv", index=False)

    cfg.banner("Summary")
    print(s.to_string(index=False))
    print(f"\n  → {out}")
    print("\n  Report all three protocols in the paper.")
    print("  The difference of P1 from P2 and P3 = leakage effect. This is the main table of your paper.")


if __name__ == "__main__":
    main()
