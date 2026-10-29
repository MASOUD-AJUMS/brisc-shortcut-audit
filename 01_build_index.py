



import re
import sys
from pathlib import Path

import pandas as pd

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found. Place it next to this file.")

FNAME_RE = re.compile(
    r"^brisc2025_(?P<split>train|test)_(?P<num>\d+)_"
    r"(?P<cls>gl|me|pi|no)_(?P<plane>ax|co|sa)_(?P<mod>t1)$",
    re.IGNORECASE,
)


def parse_stem(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "uid": stem,
        "official_split": g["split"].lower(),
        "seq_num": int(g["num"]),
        "class_code": g["cls"].lower(),
        "class_name": cfg.CODE2CLASS[g["cls"].lower()],
        "plane_code": g["plane"].lower(),
        "plane_name": cfg.CODE2PLANE[g["plane"].lower()],
        "modality": g["mod"].lower(),
    }


def scan_classification() -> pd.DataFrame:
    rows, bad = [], []
    for split, root in (("train", cfg.CLS_TRAIN), ("test", cfg.CLS_TEST)):
        if not root.exists():
            print(f"  ! folder does not exist: {root}")
            continue
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            folder_class = class_dir.name.lower()
            for img in sorted(class_dir.glob("*.jpg")):
                rec = parse_stem(img.stem)
                if rec is None:
                    bad.append(str(img))
                    continue
                rec["folder_class"] = folder_class
                rec["class_mismatch"] = folder_class != rec["class_name"]
                rec["split_mismatch"] = split != rec["official_split"]
                rec["cls_img_path"] = str(img.resolve())
                rows.append(rec)

    if bad:
        print(f"  ! {len(bad)} files with unexpected names (examples: {bad[:3]})")
    return pd.DataFrame(rows)


def scan_segmentation() -> pd.DataFrame:
    rows = []
    pairs = (
        ("train", cfg.SEG_TRAIN_IMG, cfg.SEG_TRAIN_MSK),
        ("test", cfg.SEG_TEST_IMG, cfg.SEG_TEST_MSK),
    )
    for split, img_dir, msk_dir in pairs:
        if not img_dir.exists():
            print(f"  ! folder does not exist: {img_dir}")
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            msk = msk_dir / f"{img.stem}.png"
            rows.append(
                {
                    "uid": img.stem,
                    "seg_img_path": str(img.resolve()),
                    "seg_mask_path": str(msk.resolve()) if msk.exists() else None,
                    "has_mask_file": msk.exists(),
                    "seg_split": split,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cfg.set_seed()
    cfg.banner("1/3  Scanning classification_task")
    df_cls = scan_classification()
    print(f"  {len(df_cls)} images found")

    cfg.banner("2/3  Scanning segmentation_task")
    df_seg = scan_segmentation()
    print(f"  {len(df_seg)} images found")
    print(f"  {df_seg['has_mask_file'].sum()} matching masks present")

    cfg.banner("3/3  Merging")
    if df_cls.empty:
        sys.exit("No classification images found. Check DATA_ROOT.")

    df = df_cls.merge(df_seg, on="uid", how="outer", indicator=True)

    only_cls = (df["_merge"] == "left_only").sum()
    only_seg = (df["_merge"] == "right_only").sum()
    both = (df["_merge"] == "both").sum()
    print(f"  in both tasks : {both}")
    print(f"  classification only : {only_cls}")
    print(f"  segmentation only   : {only_seg}")

    miss = df["class_name"].isna()
    if miss.any():
        recovered = df.loc[miss, "uid"].apply(parse_stem).apply(pd.Series)
        for col in recovered.columns:
            df.loc[miss, col] = recovered[col].values

    df = df.drop(columns=["_merge"])
    df["in_classification"] = df["cls_img_path"].notna()
    df["in_segmentation"] = df["seg_img_path"].notna()

    df["class_idx"] = df["class_name"].map(cfg.CLASS2IDX)
    df["plane_idx"] = df["plane_name"].map(cfg.PLANE2IDX)

    df["img_path"] = df["seg_img_path"].fillna(df["cls_img_path"])

    cfg.banner("Sanity checks")
    n_bad_class = int(df["class_mismatch"].fillna(False).sum())
    n_bad_split = int(df["split_mismatch"].fillna(False).sum())
    print(f"  folder class vs filename mismatch : {n_bad_class}")
    print(f"  folder split vs filename mismatch : {n_bad_split}")
    dup = df["uid"].duplicated().sum()
    print(f"  duplicate uid : {dup}")

    cfg.banner("Class × split distribution")
    tab1 = pd.crosstab(df["class_name"], df["official_split"], margins=True)
    print(tab1.to_string())
    print("\nExpected (per the BRISC paper):")
    print("  glioma 1147/254 | meningioma 1329/306 | "
          "pituitary 1457/300 | no_tumor 1067/140")

    cfg.banner("Plane × class × split distribution")
    tab2 = pd.crosstab(
        [df["official_split"], df["class_name"]], df["plane_name"], margins=True
    )
    print(tab2.to_string())

    cols = [
        "uid", "official_split", "seq_num",
        "class_name", "class_code", "class_idx",
        "plane_name", "plane_code", "plane_idx",
        "modality",
        "img_path", "cls_img_path", "seg_img_path", "seg_mask_path",
        "has_mask_file", "in_classification", "in_segmentation",
        "class_mismatch", "split_mismatch",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["official_split", "class_name", "seq_num"])

    out = cfg.DIR_INDEX / "index.csv"
    df.to_csv(out, index=False)

    tab1.to_csv(cfg.DIR_INDEX / "dist_class_split.csv")
    tab2.to_csv(cfg.DIR_INDEX / "dist_plane_class_split.csv")

    cfg.banner("Done")
    print(f"  {len(df)} rows written → {out}")
    if n_bad_class or n_bad_split or dup:
        print("\n  ⚠ Inconsistency found. Review before continuing.")


if __name__ == "__main__":
    main()
