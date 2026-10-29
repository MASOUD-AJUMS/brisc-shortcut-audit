





import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from importlib import import_module

cfg = import_module("00_config")

try:
    data_mod = import_module("07_data")
    build_transforms = import_module("07b_transforms_fix").build_transforms
except Exception as e:
    sys.exit(f"Dependencies failed to load: {e}")


class InpaintDataset(Dataset):

    def __init__(self, no_tumor_df: pd.DataFrame, lesion_masks: list[str],
                img_size: int = cfg.IMG_SIZE, train: bool = True):
        self.df = no_tumor_df.reset_index(drop=True)
        self.masks = lesion_masks
        self.img_size = img_size
        self.tf = build_transforms(img_size, train=False)                       
        self.train = train
        self.rng = np.random.default_rng(cfg.GLOBAL_SEED)

    def __len__(self):
        return len(self.df)

    def _sample_mask(self, idx: int) -> np.ndarray:
        p = self.masks[idx % len(self.masks)]
        m = np.array(Image.open(p).convert("L"))
        m = (m > 127).astype(np.uint8)
        if self.train:
            if self.rng.random() < 0.5:
                m = np.fliplr(m).copy()
            if self.rng.random() < 0.5:
                m = np.flipud(m).copy()
        return m

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = np.array(Image.open(r["img_path"]).convert("L"))
        img3 = np.stack([img] * 3, axis=-1)

        mi = int(self.rng.integers(0, len(self.masks)))
        mask = self._sample_mask(mi)

        out = self.tf(image=img3, mask=mask)
        x = torch.from_numpy(out["image"].transpose(2, 0, 1)).float()
        m = torch.from_numpy(out["mask"]).float().unsqueeze(0)
        m = (m > 0.5).float()

        x_masked = x * (1 - m)
        model_in = torch.cat([x_masked, m], dim=0)           
        return {"input": model_in, "target": x, "mask": m}


def build_lesion_mask_pool(df: pd.DataFrame, max_n: int = 2000) -> list[str]:
    sub = df[(df["class_name"] != "no_tumor") & df["seg_mask_path"].notna()]
    paths = [p for p in sub["seg_mask_path"].tolist() if Path(p).exists()]
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    if len(paths) > max_n:
        paths = list(rng.choice(paths, max_n, replace=False))
    return paths


class GatedConv(nn.Module):

    def __init__(self, cin, cout, k=3, s=1, p=1, act=True):
        super().__init__()
        self.feat = nn.Conv2d(cin, cout, k, s, p)
        self.gate = nn.Conv2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        f = self.feat(x)
        g = torch.sigmoid(self.gate(x))
        out = self.bn(f * g)
        return F.leaky_relu(out, 0.2) if self.act else out


class InpainterUNet(nn.Module):

    def __init__(self, base=48):
        super().__init__()
        c = base
        self.e1 = GatedConv(4, c)
        self.e2 = GatedConv(c, c * 2, s=2)
        self.e3 = GatedConv(c * 2, c * 4, s=2)
        self.e4 = GatedConv(c * 4, c * 8, s=2)
        self.b = nn.Sequential(
            GatedConv(c * 8, c * 8), GatedConv(c * 8, c * 8))
        self.d4 = GatedConv(c * 8 + c * 8, c * 4)
        self.d3 = GatedConv(c * 4 + c * 4, c * 2)
        self.d2 = GatedConv(c * 2 + c * 2, c)
        self.d1 = nn.Conv2d(c + c, 3, 3, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.b(e4)

        d4 = self.d4(torch.cat([F.interpolate(b, e3.shape[-2:], mode="nearest"), e3], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, e2.shape[-2:], mode="nearest"), e2], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, e1.shape[-2:], mode="nearest"), e1], 1))
        out = self.d1(torch.cat([d2, e1], 1))
        return torch.tanh(out)


class PatchDiscriminator(nn.Module):

    def __init__(self, base=32):
        super().__init__()
        c = base

        def blk(ci, co, s=2, norm=True):
            layers = [nn.Conv2d(ci, co, 4, s, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            blk(4, c, norm=False), blk(c, c * 2), blk(c * 2, c * 4),
            blk(c * 4, c * 8, s=1), nn.Conv2d(c * 8, 1, 4, 1, 1))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1))


def inpaint_losses(pred, target, mask):
    l1_hole = F.l1_loss(pred * mask, target * mask)
    l1_valid = F.l1_loss(pred * (1 - mask), target * (1 - mask))
    tv = (torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
          + torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean())
    return 6.0 * l1_hole + 1.0 * l1_valid + 0.1 * tv, l1_hole.detach(), l1_valid.detach()


@torch.no_grad()
def save_samples(gen, batch, device, path, n=6):
    gen.eval()
    x_in = batch["input"][:n].to(device)
    tgt = batch["target"][:n].to(device)
    mask = batch["mask"][:n].to(device)
    pred = gen(x_in)
    composite = tgt * (1 - mask) + pred * mask

    def denorm(t):
        m = torch.tensor(cfg.IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
        s = torch.tensor(cfg.IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
        return (t * s + m).clamp(0, 1)

    masked_view = tgt * (1 - mask)
    rows = [denorm(tgt), denorm(masked_view), denorm(composite)]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, n, figsize=(2 * n, 6))
        titles = ["original", "masked input", "inpainted"]
        for r in range(3):
            for c in range(n):
                im = rows[r][c].permute(1, 2, 0).cpu().numpy()
                ax[r, c].imshow(im)
                ax[r, c].axis("off")
                if c == 0:
                    ax[r, c].set_ylabel(titles[r])
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
    except ImportError:
        pass
    gen.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--w-adv", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    device = cfg.get_device()
    cfg.set_seed()
    cfg.banner("Inpainter configuration")
    for k, v in vars(args).items():
        print(f"  {k:15s}: {v}")

    df = pd.read_csv(cfg.DIR_GROUPS / "folds.csv")
    no_tumor = df[(df.class_name == "no_tumor") & (df.P3_role == "trainval")]
    print(f"\n  {len(no_tumor)} no_tumor images for inpainter training")
    if len(no_tumor) < 100:
        sys.exit("no_tumor count is too low — check that folds.csv is correct")

    masks = build_lesion_mask_pool(df[df.P3_role == "trainval"])
    print(f"  {len(masks)} mask shapes from tumor-positive images (only the shape is used)")

    n_val = max(1, int(0.1 * len(no_tumor)))
    val_df = no_tumor.iloc[:n_val]
    tr_df = no_tumor.iloc[n_val:]

    ds_tr = InpaintDataset(tr_df, masks, train=True)
    ds_va = InpaintDataset(val_df, masks, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=min(args.batch_size, len(ds_va)),
                       shuffle=False, num_workers=0)

    gen = InpainterUNet().to(device)
    disc = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    resume_path = cfg.DIR_CKPT / "resume_inpainter.pt"
    hist, start_ep = [], 0
    if not args.fresh and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        gen.load_state_dict(ck["gen"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        hist = ck["hist"]; start_ep = ck["epoch"] + 1
        print(f"\n  ↻ resuming from epoch {start_ep}")
    else:
        print("\n  starting from scratch")

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        gen.train(); disc.train()
        agg = {"l1_hole": 0.0, "l1_valid": 0.0, "adv_g": 0.0, "adv_d": 0.0}
        n = 0

        pbar = tqdm(dl_tr, desc=f"ep{ep:03d}", leave=False)
        for b in pbar:
            x_in = b["input"].to(device); tgt = b["target"].to(device)
            mask = b["mask"].to(device)
            bs = x_in.size(0); n += bs

            with torch.no_grad():
                pred = gen(x_in)
                comp = tgt * (1 - mask) + pred * mask
            d_real = disc(tgt, mask)
            d_fake = disc(comp.detach(), mask)
            loss_d = (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            pred = gen(x_in)
            comp = tgt * (1 - mask) + pred * mask
            rec_loss, l1h, l1v = inpaint_losses(pred, tgt, mask)
            d_fake_g = disc(comp, mask)
            adv_g = -d_fake_g.mean()
            loss_g = rec_loss + args.w_adv * adv_g
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            agg["l1_hole"] += float(l1h) * bs
            agg["l1_valid"] += float(l1v) * bs
            agg["adv_g"] += float(adv_g) * bs
            agg["adv_d"] += float(loss_d) * bs
            pbar.set_postfix(l1h=f"{float(l1h):.3f}")

        row = {k: v / max(n, 1) for k, v in agg.items()}
        row["epoch"] = ep
        hist.append(row)
        print(f"  ep{ep:03d}  l1_hole={row['l1_hole']:.4f}  "
              f"l1_valid={row['l1_valid']:.4f}  adv_g={row['adv_g']:.4f}  "
              f"adv_d={row['adv_d']:.4f}")

        if (ep + 1) % args.sample_every == 0 or ep == args.epochs - 1:
            vb = next(iter(dl_va))
            save_samples(gen, vb, device,
                        cfg.DIR_FIGS / f"inpainter_samples_ep{ep:03d}.png")

        torch.save({"gen": gen.state_dict(), "disc": disc.state_dict(),
                   "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                   "epoch": ep, "hist": hist}, resume_path)

    torch.save({"gen": gen.state_dict(), "epoch": args.epochs - 1},
              cfg.DIR_CKPT / "inpainter.pt")
    pd.DataFrame(hist).to_csv(cfg.DIR_RESULTS / "inpainter_history.csv", index=False)

    mins = (time.time() - t0) / 60
    cfg.banner("Done")
    print(f"  time: {mins:.1f} min")
    print(f"  → {cfg.DIR_CKPT/'inpainter.pt'}")
    print(f"  → visual samples in {cfg.DIR_FIGS}")
    print("\n  Next step: look at the inpainter_samples_ep*.png samples.")
    print("  If the inpainted region still has artificial lines or edges,")
    print("  increase epochs or raise w_adv slightly.")
    print("  Then run 13_validate_inpainter.py for the blinded radiological test.")


if __name__ == "__main__":
    main()
