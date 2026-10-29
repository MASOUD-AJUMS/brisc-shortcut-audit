




import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")


class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.s = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        p = p.flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        d = (2 * inter + self.s) / (p.sum(1) + t.sum(1) + self.s)
        return 1 - d.mean()


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.a, self.b, self.s = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        return (1 - (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)).mean()


class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def _edge(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits, target):
        return F.l1_loss(self._edge(torch.sigmoid(logits)), self._edge(target))


class SegLoss(nn.Module):

    def __init__(self, w_dice=1.0, w_bce=1.0, w_bnd=0.2, use_tversky=False,
                 pos_weight=None):
        super().__init__()
        self.region = TverskyLoss() if use_tversky else DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight) if pos_weight else None)
        self.bnd = BoundaryLoss()
        self.wd, self.wb, self.wn = w_dice, w_bce, w_bnd

    def forward(self, logits, target):
        return (self.wd * self.region(logits, target)
                + self.wb * self.bce(logits, target)
                + self.wn * self.bnd(logits, target))


class EmptyMaskLoss(nn.Module):

    def forward(self, logits, has_lesion):
        p = torch.sigmoid(logits).flatten(1).mean(1)
        neg = (has_lesion < 0.5).float()
        if neg.sum() == 0:
            return logits.sum() * 0.0
        return (p * neg).sum() / neg.sum()


class ClsLoss(nn.Module):
    def __init__(self, smoothing=0.05, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=smoothing,
            weight=torch.tensor(class_weights) if class_weights is not None else None)

    def forward(self, logits, target):
        return self.ce(logits, target)


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, feat, labels):
        f = F.normalize(feat, dim=1)
        sim = f @ f.t() / self.t
        n = f.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=f.device)
        sim = sim.float().masked_fill(eye, -1e9)
        pos = (labels.view(-1, 1) == labels.view(1, -1)) & (~eye)
        if pos.sum() == 0:
            return f.sum() * 0.0
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_pos = (log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
        valid = pos.sum(1) > 0
        return -mean_pos[valid].mean()


class EvidenceMassLoss(nn.Module):



    def __init__(self, mode: str = "featmap", eps: float = 1e-6):
        super().__init__()
        assert mode in ("grad", "featmap")
        self.mode = mode
        self.eps = eps

    @staticmethod
    def _norm_map(s):
        B = s.size(0)
        s = s.flatten(1)
        s = s - s.min(1, keepdim=True).values
        s = s / (s.max(1, keepdim=True).values + 1e-6)
        return s.view(B, 1, *([int(s.size(1) ** 0.5)] * 2)) if False else s

    def forward(self, saliency, mask, has_lesion):
        if saliency.size(1) > 1:
            saliency = saliency.mean(1, keepdim=True)
        saliency = F.relu(saliency)
        if saliency.shape[-2:] != mask.shape[-2:]:
            saliency = F.interpolate(saliency, mask.shape[-2:],
                                     mode="bilinear", align_corners=False)
        s = saliency.flatten(1)
        s = s - s.min(1, keepdim=True).values
        s = s / (s.max(1, keepdim=True).values + self.eps)
        m = mask.flatten(1)

        inside = (s * m).sum(1)
        total = s.sum(1) + self.eps
        eit = inside / total

        w = (has_lesion > 0.5).float()
        if w.sum() == 0:
            return saliency.sum() * 0.0, torch.tensor(0.0, device=s.device)
        loss = ((1 - eit) * w).sum() / w.sum()
        eit_mean = (eit * w).sum() / w.sum()
        return loss, eit_mean.detach()


class CounterfactualLoss(nn.Module):


    def __init__(self, no_tumor_idx: int = cfg.NO_TUMOR_IDX):
        super().__init__()
        self.idx = no_tumor_idx
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_ablated, has_lesion):
        w = has_lesion > 0.5
        if w.sum() == 0:
            return logits_ablated.sum() * 0.0
        tgt = torch.full((int(w.sum()),), self.idx, dtype=torch.long,
                         device=logits_ablated.device)
        return self.ce(logits_ablated[w], tgt)


@torch.no_grad()
def crude_ablate(x, mask, blur_k: int = 31):

    pad = blur_k // 2
    blurred = F.avg_pool2d(F.pad(x, [pad] * 4, mode="reflect"),
                           blur_k, stride=1)
    m = (mask > 0.5).float()
    if m.shape[-2:] != x.shape[-2:]:
        m = F.interpolate(m, x.shape[-2:], mode="nearest")
    return x * (1 - m) + blurred * m


class UncertaintyWeighting(nn.Module):

    def __init__(self, n_tasks: int = 2):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, l in enumerate(losses):
            total = total + torch.exp(-self.log_var[i]) * l + 0.5 * self.log_var[i]
        return total


class TotalLoss(nn.Module):


    def __init__(self, w_con=0.1, w_evidence=0.0, w_cf=0.0, w_empty=0.2,
                 w_ds=0.3, use_uw=True, evidence_mode="featmap",
                 class_weights=None):
        super().__init__()
        self.seg = SegLoss()
        self.cls = ClsLoss(class_weights=class_weights)
        self.con = SupConLoss()
        self.evi = EvidenceMassLoss(mode=evidence_mode)
        self.cf = CounterfactualLoss()
        self.empty = EmptyMaskLoss()
        self.uw = UncertaintyWeighting(2) if use_uw else None
        self.w = dict(con=w_con, evi=w_evidence, cf=w_cf,
                      empty=w_empty, ds=w_ds)

    def forward(self, out, batch, out_ablated=None):
        parts, dev = {}, batch["label"].device
        z = torch.zeros((), device=dev)

        l_seg = z
        if out.get("logits_seg") is not None:
            l_seg = self.seg(out["logits_seg"], batch["mask"])
            parts["seg"] = l_seg.detach()
            if self.w["ds"] > 0 and out.get("deep_supervision"):
                ds = sum(self.seg(o, batch["mask"]) for o in out["deep_supervision"])
                ds = ds / max(len(out["deep_supervision"]), 1)
                l_seg = l_seg + self.w["ds"] * ds
                parts["ds"] = ds.detach()
            if self.w["empty"] > 0:
                le = self.empty(out["logits_seg"], batch["has_lesion"])
                l_seg = l_seg + self.w["empty"] * le
                parts["empty"] = le.detach()

        l_cls = z
        if out.get("logits_cls") is not None:
            l_cls = self.cls(out["logits_cls"], batch["label"])
            parts["cls"] = l_cls.detach()

        total = (self.uw([l_seg, l_cls]) if self.uw is not None
                 else l_seg + l_cls)

        if self.w["con"] > 0 and out.get("feat") is not None:
            lc = self.con(out["feat"], batch["label"])
            total = total + self.w["con"] * lc
            parts["con"] = lc.detach()

        if self.w["evi"] > 0 and out.get("seg_feat") is not None:
            le, eit = self.evi(out["seg_feat"], batch["mask"], batch["has_lesion"])
            total = total + self.w["evi"] * le
            parts["evidence"] = le.detach()
            parts["EIT"] = eit

        if self.w["cf"] > 0 and out_ablated is not None:
            lcf = self.cf(out_ablated["logits_cls"], batch["has_lesion"])
            total = total + self.w["cf"] * lcf
            parts["cf"] = lcf.detach()

        parts["total"] = total.detach()
        return total, parts


def self_test() -> None:
    cfg.set_seed()
    dev = cfg.get_device()
    B, S = 4, 64
    batch = {
        "mask": (torch.rand(B, 1, S, S) > 0.9).float().to(dev),
        "label": torch.randint(0, cfg.N_CLASSES, (B,)).to(dev),
        "has_lesion": torch.tensor([1., 1., 0., 1.]).to(dev),
    }
    batch["mask"][2] = 0
    out = {
        "logits_seg": torch.randn(B, 1, S, S, requires_grad=True, device=dev),
        "logits_cls": torch.randn(B, cfg.N_CLASSES, requires_grad=True, device=dev),
        "feat": torch.randn(B, 128, requires_grad=True, device=dev),
        "seg_feat": torch.randn(B, 32, S // 2, S // 2, requires_grad=True, device=dev),
        "deep_supervision": [torch.randn(B, 1, S, S, requires_grad=True, device=dev)
                             for _ in range(2)],
    }

    cfg.banner("Loss component tests")
    for n, f in [("Dice", DiceLoss()), ("Tversky", TverskyLoss()),
                 ("Boundary", BoundaryLoss().to(dev)), ("Seg", SegLoss().to(dev))]:
        print(f"  {n:10s} = {f(out['logits_seg'], batch['mask']).item():.4f}")
    print(f"  {'Empty':10s} = "
          f"{EmptyMaskLoss()(out['logits_seg'], batch['has_lesion']).item():.4f}")
    print(f"  {'Cls':10s} = "
          f"{ClsLoss()(out['logits_cls'], batch['label']).item():.4f}")
    print(f"  {'SupCon':10s} = "
          f"{SupConLoss()(out['feat'], batch['label']).item():.4f}")
    le, eit = EvidenceMassLoss()(out["seg_feat"], batch["mask"], batch["has_lesion"])
    print(f"  {'Evidence':10s} = {le.item():.4f}   (EIT={eit.item():.4f})")

    cfg.banner("TotalLoss tests in ablation configurations")
    for name, kw in [("B6 base", dict(w_evidence=0, w_cf=0)),
                     ("B7 +evidence", dict(w_evidence=0.1, w_cf=0)),
                     ("B8 +cf", dict(w_evidence=0.1, w_cf=0.1))]:
        tl = TotalLoss(**kw).to(dev)
        oa = {"logits_cls": torch.randn(B, cfg.N_CLASSES, device=dev)} \
            if kw.get("w_cf", 0) > 0 else None
        t, p = tl(out, batch, oa)
        t.backward(retain_graph=True)
        print(f"  {name:14s} total={t.item():7.4f}  "
              f"components={ {k: round(v.item(),3) for k,v in p.items()} }")

    cfg.banner("crude_ablate test")
    x = torch.randn(B, 3, S, S, device=dev)
    xa = crude_ablate(x, batch["mask"])
    print(f"  input shape {tuple(x.shape)} → output {tuple(xa.shape)}")
    print(f"  change inside the mask: {(x-xa).abs().mean().item():.4f}")
    print("\n  ✓ all components work")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        print("This is a module. To test:  python 09_losses.py --self-test")
