







import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

import albumentations as A
import cv2

AV = tuple(int(x) for x in A.__version__.split(".")[:2])
IS_V2 = AV >= (2, 0)


def _accepts(t, name: str) -> bool:
    try:
        return name in inspect.signature(t.__init__).parameters
    except (ValueError, TypeError):
        return False


def pad_if_needed(size: int):
    kw = dict(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
    if _accepts(A.PadIfNeeded, "fill"):
        kw.update(fill=0, fill_mask=0)                   
    else:
        kw.update(value=0, mask_value=0)                 
    return A.PadIfNeeded(**kw)


def gauss_noise(low: float, high: float, p: float):
    if _accepts(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(low, high), p=p)               
    v = ((low * 255) ** 2, (high * 255) ** 2)
    return A.GaussNoise(var_limit=v, p=p)                             


def coarse_dropout(n_max: int, hole: int, p: float):
    if _accepts(A.CoarseDropout, "num_holes_range"):                  
        return A.CoarseDropout(
            num_holes_range=(1, n_max),
            hole_height_range=(hole // 3, hole),
            hole_width_range=(hole // 3, hole),
            fill=0, fill_mask=0, p=p)
    return A.CoarseDropout(                                           
        max_holes=n_max, max_height=hole, max_width=hole,
        fill_value=0, mask_fill_value=0, p=p)


def elastic(alpha: float, sigma: float, p: float):
    kw = dict(alpha=alpha, sigma=sigma, p=p)
    if _accepts(A.ElasticTransform, "border_mode"):
        kw["border_mode"] = cv2.BORDER_CONSTANT
    return A.ElasticTransform(**kw)


def build_transforms(img_size: int = cfg.IMG_SIZE, train: bool = True,
                     strength: str = "medium") -> A.Compose:

    pre = [
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        pad_if_needed(img_size),
    ]
    post = [A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)]

    if not train:
        return A.Compose(pre + post)

    if strength == "none":
        aug = []
    elif strength == "light":
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
            elastic(150, 8, 0.25),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.25),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 140), p=0.4),
            gauss_noise(0.02, 0.16, 0.35),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)),
                     A.MedianBlur(blur_limit=3)], p=0.25),
            coarse_dropout(8, 24, 0.25),
        ]
    else:          
        aug = [
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.15),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1),
                     rotate=(-30, 30), p=0.7),
            elastic(120, 6, 0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 130), p=0.3),
            gauss_noise(0.02, 0.12, 0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)),
                     A.MedianBlur(blur_limit=3)], p=0.2),
            coarse_dropout(6, 20, 0.2),
        ]

    return A.Compose(pre + aug + post)


def self_test() -> None:
    cfg.banner(f"albumentations {A.__version__}  (v2={IS_V2})")

    import warnings
    size = cfg.IMG_SIZE
    img = (np.random.rand(300, 380, 3) * 255).astype(np.uint8)
    mask = np.zeros((300, 380), np.uint8)
    mask[120:180, 150:220] = 1

    for s in ("none", "light", "medium", "strong"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tf = build_transforms(size, train=True, strength=s)
            out = tf(image=img, mask=mask)
            bad = [str(x.message) for x in w
                   if "not valid" in str(x.message)]
        status = "✓" if not bad else f"✗ {bad[:1]}"
        print(f"  {s:8s} image={out['image'].shape} "
              f"mask={out['mask'].shape} {status}")

    cfg.banner("Check: is padding done with zeros?")
    tf = build_transforms(size, train=False)
    o = tf(image=img, mask=mask)
    im = o["image"]
    corners = [im[0, 0].mean(), im[0, -1].mean(), im[-1, 0].mean(), im[-1, -1].mean()]
    exp = -cfg.IMAGENET_MEAN[0] / cfg.IMAGENET_STD[0]
    print(f"  mean of corners: {[round(float(c),3) for c in corners]}")
    print(f"  expected for zero padding: ~{exp:.3f}")

    cfg.banner("Critical check: does CoarseDropout erase the mask in sync?")
    full = np.ones((300, 380), np.uint8)
    hits = 0
    for _ in range(60):
        tf = A.Compose([coarse_dropout(8, 60, 1.0)])
        o = tf(image=img, mask=full)
        if o["mask"].sum() < full.sum():
            hits += 1
    print(f"  mask also got a hole in {hits}/60 runs")
    if hits >= 50:
        print("  ✓ in sync")
    else:
        print("  ✗ mask is not in sync — this is a serious bug, check the version")

    cfg.banner("Check: does GaussNoise really add noise?")
    base = A.Compose([A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)])
    a0 = base(image=img)["image"]
    for lo, hi in [(0.02, 0.12), (0.02, 0.16)]:
        tf = A.Compose([gauss_noise(lo, hi, 1.0),
                        A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)])
        a1 = tf(image=img)["image"]
        print(f"  std_range=({lo},{hi})  mean absolute difference = "
              f"{np.abs(a1-a0).mean():.4f}")
    print("\n  ✓ if the difference is non-zero, noise is applied")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        print("This is a module. To test:  python 07b_transforms_fix.py --self-test")
