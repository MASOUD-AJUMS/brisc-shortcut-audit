

import os, time, argparse, math, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader

OUT = os.environ.get("BRISC_OUT", "/content/drive/MyDrive/BRISC_2025_Downloaded/outputs")
GRP, CKP, FIG, RES = f"{OUT}/groups", f"{OUT}/checkpoints", f"{OUT}/figures", f"{OUT}/results"
IMG = 224


def load_gray_224(path, interp=cv2.INTER_LINEAR):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    h, w = a.shape
    s = IMG / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    a = cv2.resize(a, (nw, nh), interpolation=interp)
    top, left = (IMG - nh) // 2, (IMG - nw) // 2
    out = np.zeros((IMG, IMG), a.dtype)
    out[top:top + nh, left:left + nw] = a
    return out


def brain_mask(img_u8):
    _, th = cv2.threshold(cv2.GaussianBlur(img_u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        th = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(th)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return cv2.erode(filled, np.ones((15, 15), np.uint8)) > 0


class MaskBank:
    def __init__(self, paths, n_max=2000, seed=0):
        rng = random.Random(seed)
        paths = list(paths); rng.shuffle(paths)
        self.shapes = []
        for p in paths:
            m = load_gray_224(p, cv2.INTER_NEAREST) > 127
            if m.sum() < 20:
                continue
            ys, xs = np.where(m)
            self.shapes.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8))
            if len(self.shapes) >= n_max:
                break

    def sample(self, brain, rng):
        for _ in range(30):
            s = self.shapes[rng.randrange(len(self.shapes))]
            sc = rng.uniform(0.7, 1.4)
            ang = rng.uniform(0, 360)
            h, w = s.shape
            nh, nw = max(4, int(h * sc)), max(4, int(w * sc))
            s2 = cv2.resize(s, (nw, nh), interpolation=cv2.INTER_NEAREST)
            if rng.random() < 0.5:
                s2 = s2[:, ::-1]
            d = int(math.ceil(math.hypot(nh, nw))) + 2
            canvas = np.zeros((d, d), np.uint8)
            canvas[(d - nh) // 2:(d - nh) // 2 + nh, (d - nw) // 2:(d - nw) // 2 + nw] = s2
            M = cv2.getRotationMatrix2D((d / 2, d / 2), ang, 1.0)
            canvas = cv2.warpAffine(canvas, M, (d, d), flags=cv2.INTER_NEAREST)
            ys, xs = np.where(canvas > 0)
            if len(ys) == 0:
                continue
            canvas = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            ch, cw = canvas.shape
            if ch >= IMG - 4 or cw >= IMG - 4:
                continue
            by, bx = np.where(brain)
            if len(by) == 0:
                break
            k = rng.randrange(len(by))
            top, left = int(by[k] - ch // 2), int(bx[k] - cw // 2)
            top, left = min(max(top, 0), IMG - ch), min(max(left, 0), IMG - cw)
            m = np.zeros((IMG, IMG), bool)
            m[top:top + ch, left:left + cw] = canvas > 0
            if (m & brain).sum() >= 0.95 * m.sum():
                k2 = rng.choice([3, 5, 7])
                return cv2.dilate(m.astype(np.uint8), np.ones((k2, k2), np.uint8)).astype(bool) & brain
        m = np.zeros((IMG, IMG), np.uint8)
        by, bx = np.where(brain) if brain.any() else (np.array([IMG // 2]), np.array([IMG // 2]))
        cv2.ellipse(m, (int(bx.mean()), int(by.mean())), (12, 9), rng.uniform(0, 180), 0, 360, 1, -1)
        return m.astype(bool)


class InpaintDS(Dataset):
    def __init__(self, paths, bank, train=True, seed=0):
        self.imgs = [load_gray_224(p) for p in paths]
        self.brains = [brain_mask(a) for a in self.imgs]
        self.bank, self.train, self.seed = bank, train, seed
        if not train:                                                    
            rng = random.Random(seed)
            self.fixed = [bank.sample(b, rng) for b in self.brains]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        a, br = self.imgs[i], self.brains[i]
        if self.train:
            rng = random.Random()
            m = self.bank.sample(br, rng)
            if rng.random() < 0.5:
                a, m = a[:, ::-1].copy(), m[:, ::-1].copy()
            a = a.astype(np.float32)
            if rng.random() < 0.5:
                a = np.clip(a * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8), 0, 255)
        else:
            m = self.fixed[i]
            a = a.astype(np.float32)
        x = torch.from_numpy(a / 127.5 - 1.0)[None]
        return x.float(), torch.from_numpy(m.astype(np.float32))[None]


class GatedConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, d=1, act=True):
        super().__init__()
        p = d * (k - 1) // 2
        self.f = nn.Conv2d(cin, cout, k, s, p, dilation=d)
        self.g = nn.Conv2d(cin, cout, k, s, p, dilation=d)
        self.act = act

    def forward(self, x):
        f = self.f(x)
        return (F.elu(f) if self.act else f) * torch.sigmoid(self.g(x))


class Generator(nn.Module):
    def __init__(self, c=48):
        super().__init__()
        self.e1 = nn.Sequential(GatedConv(2, c, 5), GatedConv(c, c))                           
        self.e2 = nn.Sequential(GatedConv(c, 2 * c, 3, 2), GatedConv(2 * c, 2 * c))           
        self.e3 = nn.Sequential(GatedConv(2 * c, 4 * c, 3, 2), GatedConv(4 * c, 4 * c))      
        self.e4 = nn.Sequential(GatedConv(4 * c, 4 * c, 3, 2))                                
        self.mid = nn.Sequential(*[GatedConv(4 * c, 4 * c, 3, 1, d) for d in (2, 4, 8, 16, 1)])
        self.d4 = nn.Sequential(GatedConv(8 * c, 4 * c), GatedConv(4 * c, 4 * c))
        self.d3 = nn.Sequential(GatedConv(8 * c, 2 * c), GatedConv(2 * c, 2 * c))
        self.d2 = nn.Sequential(GatedConv(4 * c, c), GatedConv(c, c))
        self.d1 = nn.Sequential(GatedConv(2 * c, c // 2), nn.Conv2d(c // 2, 1, 3, 1, 1))

    def forward(self, x, m):
        h1 = self.e1(torch.cat([x * (1 - m), m], 1))
        h2 = self.e2(h1); h3 = self.e3(h2); h4 = self.e4(h3)
        b = self.mid(h4)
        u = self.d4(torch.cat([b, h4], 1))
        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.d3(torch.cat([u, h3], 1))
        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.d2(torch.cat([u, h2], 1))
        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.d1(torch.cat([u, h1], 1))
        return torch.tanh(u)


class PatchD(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        L = []
        cin = 2
        for co in (c, 2 * c, 4 * c, 4 * c, 4 * c):
            L += [spectral_norm(nn.Conv2d(cin, co, 5, 2, 2)), nn.LeakyReLU(0.2, True)]
            cin = co
        L += [spectral_norm(nn.Conv2d(cin, 1, 3, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1))


class VGGFeat(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        v = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList([v[:5], v[5:10], v[10:17]])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        h = ((x + 1) / 2).repeat(1, 3, 1, 1)
        h = (h - self.mu) / self.sd
        out = []
        for s in self.slices:
            h = s(h); out.append(h)
        return out


def gram(f):
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def masked_l1(a, b, m):
    return ((a - b).abs() * m).sum() / m.sum().clamp(min=1.0)


def boundary_tv(img, m):
    ring = (F.max_pool2d(m, 5, 1, 2) - (-F.max_pool2d(-m, 5, 1, 2))).clamp(0, 1)
    dx = (img[..., :, 1:] - img[..., :, :-1]).abs() * ring[..., :, 1:]
    dy = (img[..., 1:, :] - img[..., :-1, :]).abs() * ring[..., 1:, :]
    return (dx.sum() + dy.sum()) / ring.sum().clamp(min=1.0)


_G_CACHE = {}


@torch.no_grad()
def inpaint(img_u8, mask, device=None, ckpt=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ckpt or f"{CKP}/inpainter_v2.pt"
    key = (ckpt, str(device))
    if key not in _G_CACHE:
        G = Generator().to(device).eval()
        G.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["G"])
        _G_CACHE[key] = G
    G = _G_CACHE[key]
    x = torch.from_numpy(img_u8.astype(np.float32) / 127.5 - 1)[None, None].to(device)
    m = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    y = G(x, m)
    comp = x * (1 - m) + y * m
    return ((comp[0, 0].cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def save_samples(G, val_ds, dev, ep, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G.eval()
    idx = np.linspace(0, len(val_ds) - 1, n).astype(int)
    xs, ms = zip(*[val_ds[i] for i in idx])
    x, m = torch.stack(xs).to(dev), torch.stack(ms).to(dev)
    with torch.no_grad():
        comp = x * (1 - m) + G(x, m) * m
    G.train()
    to = lambda t: ((t[:, 0].cpu().numpy() + 1) * 127.5).clip(0, 255)
    X, C, M = to(x), to(comp), m[:, 0].cpu().numpy()
    fig, ax = plt.subplots(4, n, figsize=(2.2 * n, 9))
    for j in range(n):
        hole = X[j].copy(); hole[M[j] > 0] = 0
        cnt = cv2.findContours((M[j] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        for r, (im, t) in enumerate([(X[j], "original"), (hole, "input (hole=0)"),
                                     (C[j], "inpainted"), (C[j], "inpainted + contour")]):
            ax[r, j].imshow(im, cmap="gray", vmin=0, vmax=255)
            if r == 3:
                for c in cnt:
                    ax[r, j].plot(c[:, 0, 0], c[:, 0, 1], "r-", lw=0.6)
            ax[r, j].axis("off")
            if j == 0:
                ax[r, j].set_title(t, fontsize=8, loc="left")
    fig.tight_layout()
    p = f"{FIG}/inpainter_v2_samples_ep{ep:03d}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


@torch.no_grad()
def validate(G, loader, dev):
    G.eval()
    l1h, psnr, seam, n = 0.0, 0.0, 0.0, 0
    for x, m in loader:
        x, m = x.to(dev), m.to(dev)
        comp = x * (1 - m) + G(x, m) * m
        for i in range(len(x)):
            mi = m[i] > 0.5
            if mi.sum() == 0:
                continue
            e = (comp[i] - x[i])[mi]
            l1h += e.abs().mean().item()
            mse = ((e / 2) ** 2).mean().item()
            psnr += 10 * math.log10(1 / max(mse, 1e-10))
            inner = (mi.float() - (-F.max_pool2d(-m[i:i + 1], 5, 1, 2))[0]).clamp(0, 1) > 0.5
            outer = (F.max_pool2d(m[i:i + 1], 5, 1, 2)[0] - m[i]).clamp(0, 1) > 0.5
            if inner.any() and outer.any():
                gap_c = (comp[i][inner].mean() - comp[i][outer].mean()).abs().item()
                gap_x = (x[i][inner].mean() - x[i][outer].mean()).abs().item()
                seam += abs(gap_c - gap_x) * 127.5
            n += 1
    G.train()
    return dict(val_l1_hole=l1h / n, val_psnr_hole=psnr / n, val_seam_gap=seam / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=4e-4)
    ap.add_argument("--w-hole", type=float, default=6.0)
    ap.add_argument("--w-valid", type=float, default=1.0)
    ap.add_argument("--w-perc", type=float, default=0.05)
    ap.add_argument("--w-style", type=float, default=120.0)
    ap.add_argument("--w-tv", type=float, default=0.1)
    ap.add_argument("--w-adv", type=float, default=0.1)
    ap.add_argument("--no-perc", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-images", type=int, default=0, help="for quick testing only")
    a = ap.parse_args()
    for d in (CKP, FIG, RES):
        os.makedirs(d, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    print("=" * 70 + "\ninpainter v2\n" + "=" * 70)
    for k, v in vars(a).items():
        print(f"  {k:14s}: {v}")

    f = pd.read_csv(f"{GRP}/folds.csv")
    nt = f[(f["class_name"] == "no_tumor") & (f["P3_role"] != "test")]
    tum = f[(f["class_name"] != "no_tumor") & (f["P3_role"] != "test")]
    paths = nt["img_path"].tolist()
    if a.max_images:
        paths = paths[:a.max_images]
    rng = random.Random(0); rng.shuffle(paths)
    nv = max(8, len(paths) // 10)
    print(f"\n  no_tumor (excluding P3 test): {len(paths)}  → train {len(paths) - nv} / val {nv}")
    bank = MaskBank(tum["seg_mask_path"].tolist(), n_max=200 if a.max_images else 2000)
    print(f"  {len(bank.shapes)} mask shapes from non-test tumors")
    tr = InpaintDS(paths[nv:], bank, train=True)
    va = InpaintDS(paths[:nv], bank, train=False, seed=123)
    tl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=dev == "cuda", persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.batch_size, shuffle=False, num_workers=0)

    G, D = Generator().to(dev), PatchD().to(dev)
    print(f"  G: {sum(p.numel() for p in G.parameters())/1e6:.2f}M  D: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    vgg = None
    if not a.no_perc and (a.w_perc > 0 or a.w_style > 0):
        try:
            vgg = VGGFeat().to(dev)
        except Exception as e:
            print(f"  ⚠ VGG not available ({e}) — continuing without perceptual/style")
    oG = torch.optim.Adam(G.parameters(), a.lr_g, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), a.lr_d, betas=(0.5, 0.999))

    last, best = f"{CKP}/inpainter_v2_last.pt", f"{CKP}/inpainter_v2.pt"
    start, best_score, hist = 0, float("inf"), []
    if os.path.exists(last) and not a.fresh:
        ck = torch.load(last, map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        oG.load_state_dict(ck["oG"]); oD.load_state_dict(ck["oD"])
        start, best_score, hist = ck["epoch"] + 1, ck["best_score"], ck["hist"]
        print(f"  ↻ resuming from epoch {start}")
    else:
        print("  starting from scratch")

    t0 = time.time()
    for ep in range(start, a.epochs):
        agg = dict(l_hole=0.0, l_valid=0.0, l_perc=0.0, l_style=0.0, l_tv=0.0, adv_g=0.0, adv_d=0.0)
        nb = 0
        for x, m in tl:
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            y = G(x, m)
            comp = x * (1 - m) + y * m
            d_real, d_fake = D(x, m), D(comp.detach(), m)
            lD = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
            l_hole = masked_l1(y, x, m)
            l_valid = masked_l1(y, x, 1 - m)
            l_tv = boundary_tv(comp, m)
            l_adv = -D(comp, m).mean()
            l_perc = l_style = torch.zeros((), device=dev)
            if vgg is not None:
                fx = vgg(x)
                fy, fc = vgg(y), vgg(comp)
                l_perc = sum(F.l1_loss(a_, b_) + F.l1_loss(c_, b_) for a_, c_, b_ in zip(fy, fc, fx))
                l_style = sum(F.l1_loss(gram(a_), gram(b_)) + F.l1_loss(gram(c_), gram(b_))
                              for a_, c_, b_ in zip(fy, fc, fx))
            lG = (a.w_hole * l_hole + a.w_valid * l_valid + a.w_perc * l_perc + a.w_style * l_style
                  + a.w_tv * l_tv + a.w_adv * l_adv)
            oG.zero_grad(set_to_none=True); lG.backward(); oG.step()
            for k, v in zip(agg, (l_hole, l_valid, l_perc, l_style, l_tv, l_adv, lD)):
                agg[k] += float(v.detach())
            nb += 1
        row = {k: v / max(nb, 1) for k, v in agg.items()}
        row.update(validate(G, vl, dev)); row["epoch"] = ep
        hist.append(row)
        score = row["val_l1_hole"] + 0.01 * row["val_seam_gap"]
        star = ""
        if score < best_score:
            best_score = score
            torch.save({"G": G.state_dict(), "epoch": ep, "val": row}, best)
            star = "  ★"
        print(f"  ep{ep:03d}  hole={row['l_hole']:.4f} valid={row['l_valid']:.4f} "
              f"perc={row['l_perc']:.3f} style={row['l_style']:.4f} adv_g={row['adv_g']:.3f} "
              f"adv_d={row['adv_d']:.3f} | val L1h={row['val_l1_hole']:.4f} "
              f"PSNRh={row['val_psnr_hole']:.2f} seam={row['val_seam_gap']:.2f}{star}", flush=True)
        if (ep + 1) % a.sample_every == 0 or ep == a.epochs - 1:
            save_samples(G, va, dev, ep)
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(),
                    "epoch": ep, "best_score": best_score, "hist": hist}, last + ".tmp")
        os.replace(last + ".tmp", last)
        pd.DataFrame(hist).to_csv(f"{RES}/inpainter_v2_history.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  Done — {(time.time() - t0) / 60:.1f} min | best: {best}")
    print(f"  samples: {FIG}/inpainter_v2_samples_ep*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
