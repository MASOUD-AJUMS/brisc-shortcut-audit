


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

try:
    import albumentations as A
    import cv2
except ImportError:
    sys.exit("required:  pip install albumentations opencv-python-headless")


def build_transforms(img_size: int = cfg.IMG_SIZE, train: bool = True,
                     strength: str = "medium") -> A.Compose:

    pre = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      value=0, mask_value=0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ]

    if not train:
        return A.Compose(pre + [A.Normalize(mean=cfg.IMAGENET_MEAN,
                                            std=cfg.IMAGENET_STD)])

    if strength == "light":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                     rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
        ]
    elif strength == "strong":
        aug = [
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
            A.Affine(scale=(0.75, 1.3), translate_percent=(-0.12, 0.12),
                     rotate=(-35, 35), shear=(-8, 8), p=0.8),
            A.ElasticTransform(alpha=150, sigma=8, p=0.25),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 140), p=0.4),
            A.GaussNoise(var_limit=(5, 40), p=0.35),
            A.OneOf([A.GaussianBlur(3), A.MedianBlur(3)], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=24, max_width=24,
                            fill_value=0, mask_fill_value=0, p=0.25),
        ]
    else:                    
        aug = [
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.15),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1),
                     rotate=(-30, 30), p=0.7),
            A.ElasticTransform(alpha=120, sigma=6, p=0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 130), p=0.3),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.OneOf([A.GaussianBlur(3), A.MedianBlur(3)], p=0.2),
            A.CoarseDropout(max_holes=6, max_height=20, max_width=20,
                            fill_value=0, mask_fill_value=0, p=0.2),
        ]

    return A.Compose(pre + aug + [A.Normalize(mean=cfg.IMAGENET_MEAN,
                                              std=cfg.IMAGENET_STD)])


class BriscDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform: A.Compose,
                 img_size: int = cfg.IMG_SIZE, return_uid: bool = True):
        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.img_size = img_size
        self.return_uid = return_uid

    def __len__(self) -> int:
        return len(self.df)

    def _load_mask(self, row) -> np.ndarray:
        p = row.get("seg_mask_path")
        if isinstance(p, str) and p and p != "nan" and Path(p).exists():
            m = np.array(Image.open(p).convert("L"))
            return (m > 127).astype(np.uint8)
        return None

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = np.array(Image.open(r["img_path"]).convert("L"))
        img = np.stack([img] * 3, axis=-1)                       

        m = self._load_mask(r)
        if m is None:
            m = np.zeros(img.shape[:2], dtype=np.uint8)

        out = self.tf(image=img, mask=m)
        x = torch.from_numpy(out["image"].transpose(2, 0, 1)).float()
        y = torch.from_numpy(out["mask"]).float().unsqueeze(0)
        y = (y > 0.5).float()

        sample = {
            "image": x,
            "mask": y,
            "label": torch.tensor(int(r["class_idx"]), dtype=torch.long),
            "has_lesion": torch.tensor(float(y.sum() > 0)),
        }
        if self.return_uid:
            sample["uid"] = r["uid"]
        return sample


def load_folds(protocol: str = "P3") -> pd.DataFrame:
    p = cfg.DIR_GROUPS / "folds.csv"
    if not p.exists():
        sys.exit("folds.csv not found. Run 06_make_folds.py first.")
    df = pd.read_csv(p)
    df["class_idx"] = df["class_name"].map(cfg.CLASS2IDX)
    if f"{protocol}_role" not in df.columns:
        sys.exit(f"Protocol {protocol} is not in folds.csv.")
    return df


def get_split(df: pd.DataFrame, protocol: str, fold: int
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rc, fc = f"{protocol}_role", f"{protocol}_fold"
    tv = df[df[rc] == "trainval"]
    tr = tv[tv[fc] != fold]
    va = tv[tv[fc] == fold]
    te = df[df[rc] == "test"]
    return tr, va, te


def make_loader(sub: pd.DataFrame, train: bool, batch_size: int = 16,
                workers: int = 2, aug: str = "medium",
                balanced: bool = False) -> DataLoader:
    tf = build_transforms(train=train, strength=aug)
    ds = BriscDataset(sub, tf)
    sampler = None
    shuffle = train
    if train and balanced:
        w = 1.0 / sub["class_name"].map(sub["class_name"].value_counts())
        sampler = WeightedRandomSampler(torch.DoubleTensor(w.to_numpy()),
                                        num_samples=len(sub), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=workers,
                      pin_memory=True, drop_last=train,
                      persistent_workers=workers > 0)


def self_test() -> None:
    cfg.set_seed()
    df = load_folds("P3")
    cfg.banner("Data sanity checks")
    print(f"  {len(df)} rows | columns: {[c for c in df.columns if '_role' in c]}")

    for proto in ("P1", "P2", "P3"):
        tr, va, te = get_split(df, proto, 0)
        print(f"\n  [{proto} fold 0]  train={len(tr)} val={len(va)} test={len(te)}")
        n_empty_tr = int(tr["mask_is_empty"].sum())
        print(f"    empty masks in train: {n_empty_tr} "
              f"({100*n_empty_tr/max(len(tr),1):.1f}%)")
        print(f"    train class distribution: {dict(tr.class_name.value_counts())}")

    cfg.banner("Single batch test")
    tr, va, te = get_split(df, "P3", 0)
    dl = make_loader(tr.head(64), train=True, batch_size=8, workers=0)
    b = next(iter(dl))
    print(f"  image {tuple(b['image'].shape)}  dtype={b['image'].dtype}")
    print(f"  mask  {tuple(b['mask'].shape)}   unique={b['mask'].unique().tolist()}")
    print(f"  label {b['label'].tolist()}")
    print(f"  has_lesion {b['has_lesion'].tolist()}")
    print(f"  image range: [{b['image'].min():.2f}, {b['image'].max():.2f}]")

    dl2 = make_loader(va.head(64), train=False, batch_size=8, workers=0)
    b2 = next(iter(dl2))
    print(f"\n  val image {tuple(b2['image'].shape)}  "
          f"mask sum={b2['mask'].sum().item():.0f}")

    cfg.banner("Check: do no_tumor images have empty masks?")
    sub = tr[tr.class_name == "no_tumor"].head(16)
    if len(sub):
        dl3 = make_loader(sub, train=False, batch_size=len(sub), workers=0)
        b3 = next(iter(dl3))
        s = b3["mask"].sum(dim=(1, 2, 3))
        print(f"  mask sum over {len(sub)} no_tumor images: "
              f"{s.tolist()}")
        print(f"  {'✓ all zero' if s.sum()==0 else '✗ some non-zero — check'}")
    print("\n  ✓ ready for training")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        print("This file is a module. To test:  python 07_data.py --self-test")
