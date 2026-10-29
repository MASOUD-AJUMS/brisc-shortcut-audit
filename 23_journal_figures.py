import argparse, os, sys, traceback
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
F = import_module("21_make_figures")
OUT = F.cfg.DIR_FIGS / "journal"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 600


def save(fig, n):
    p = OUT / f"Figure{n}"
    kw = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{p}.pdf", **kw)
    tmp = f"{p}_tmp.png"
    fig.savefig(tmp, dpi=DPI, **kw)
    plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    im.save(f"{p}.tif", compression="tiff_lzw", dpi=(DPI, DPI))
    os.remove(tmp)
    w_mm, h_mm = im.width / DPI * 25.4, im.height / DPI * 25.4
    flag = "" if w_mm <= 190 else "   ⚠ wider than 190 mm"
    print(f"  ✓ Figure {n}: {im.width}×{im.height} px, {w_mm:.0f}×{h_mm:.0f} mm at {DPI} dpi{flag}")


def main():
    global DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-model", action="store_true", help="skip Figure 7 (needs checkpoints)")
    a = ap.parse_args()
    DPI = a.dpi
    F.save = save
    todo = a.only or list(F.FIGS)
    if a.no_model and 7 in todo:
        todo.remove(7)
    errs = {}
    for n in todo:
        try:
            F.FIGS[n]()
        except Exception as e:
            errs[n] = traceback.format_exc(limit=3)
            print(f"  ✗ Figure {n}: {type(e).__name__}: {e}")
    for k, v in errs.items():
        print(f"--- Figure {k}\n{v}")
    print(f"\n  Output folder: {OUT}")


if __name__ == "__main__":
    main()
