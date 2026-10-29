



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

try:
    import timm
except ImportError:
    sys.exit("required:  pip install timm")


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=None):
        p = k // 2 if p is None else p
        super().__init__(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class ASPP(nn.Module):

    def __init__(self, cin: int, cout: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, 1 if r == 1 else 3, padding=0 if r == 1 else r,
                          dilation=r, bias=False),
                nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
            for r in rates])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(cout * (len(rates) + 1), cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=True), nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.pool(x), (h, w), mode="bilinear", align_corners=False)
        return self.proj(torch.cat(feats + [g], 1))


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.c1 = ConvBNAct(cin + skip, cout)
        self.c2 = ConvBNAct(cout, cout)

    def forward(self, x, s=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if s is not None:
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, s.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, s], 1)
        return self.c2(self.c1(x))


class CrossTaskAttention(nn.Module):

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.n = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.g = nn.Parameter(torch.zeros(1))                     

    def forward(self, q, kv):
        B, C, H, W = q.shape
        qs = q.flatten(2).transpose(1, 2)
        ks = kv.flatten(2).transpose(1, 2)
        o, _ = self.attn(self.n(qs), self.n(ks), self.n(ks), need_weights=False)
        return q + self.g * o.transpose(1, 2).reshape(B, C, H, W)


class ClsOnly(nn.Module):
    def __init__(self, backbone="resnet50", n_classes=cfg.N_CLASSES,
                 pretrained=True, drop=0.2):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=n_classes, drop_rate=drop)

    def forward(self, x):
        return {"logits_cls": self.net(x), "logits_seg": None, "feat": None}


class MultiTaskNet(nn.Module):


    def __init__(self, backbone="tf_efficientnetv2_s.in21k_ft_in1k",
                 n_classes=cfg.N_CLASSES, pretrained=True,
                 use_aspp=True, use_ctam=True, dec_ch=(256, 128, 64, 32),
                 drop=0.2):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         features_only=True)
        chs = self.encoder.feature_info.channels()                  
        self.chs = chs
        self.use_ctam = use_ctam

        deep = chs[-1]
        self.aspp = ASPP(deep, dec_ch[0]) if use_aspp else ConvBNAct(deep, dec_ch[0])

        skips = list(chs[:-1])[::-1]                                
        blocks, cin = [], dec_ch[0]
        for i, cout in enumerate(dec_ch):
            s = skips[i] if i < len(skips) else 0
            blocks.append(DecoderBlock(cin, s, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.seg_head = nn.Conv2d(cin, 1, 1)

        self.ds_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in dec_ch[:-1]])

        if use_ctam:
            self.to_cls = ConvBNAct(dec_ch[0], dec_ch[0], k=1)
            self.ctam_c = CrossTaskAttention(dec_ch[0])
            self.ctam_s = CrossTaskAttention(dec_ch[0])
            self.enc_proj = ConvBNAct(deep, dec_ch[0], k=1)

        cls_in = dec_ch[0] + deep
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, 512), nn.SiLU(inplace=True), nn.Dropout(drop),
            nn.Linear(512, n_classes))
        self.embed_dim = cls_in

    def forward(self, x, return_feat=False):
        H, W = x.shape[-2:]
        feats = self.encoder(x)
        deep = feats[-1]
        b = self.aspp(deep)

        if self.use_ctam:
            e = self.enc_proj(deep)
            b = self.ctam_s(b, e)
            c_ctx = self.ctam_c(self.to_cls(b), b)
        else:
            c_ctx = b

        skips = feats[:-1][::-1]
        d, ds_out = b, []
        for i, blk in enumerate(self.decoder):
            s = skips[i] if i < len(skips) else None
            d = blk(d, s)
            if i < len(self.ds_heads):
                ds_out.append(self.ds_heads[i](d))

        seg = self.seg_head(d)
        if seg.shape[-2:] != (H, W):
            seg = F.interpolate(seg, (H, W), mode="bilinear", align_corners=False)
        ds_out = [F.interpolate(o, (H, W), mode="bilinear", align_corners=False)
                  for o in ds_out]

        pooled = torch.cat([F.adaptive_avg_pool2d(c_ctx, 1).flatten(1),
                            F.adaptive_avg_pool2d(deep, 1).flatten(1)], 1)
        logits = self.cls_head(pooled)

        out = {"logits_cls": logits, "logits_seg": seg, "deep_supervision": ds_out}
        if return_feat:
            out["feat"] = pooled                                    
            out["seg_feat"] = d                               
        return out


class SegOnly(MultiTaskNet):

    def forward(self, x, return_feat=False):
        o = super().forward(x, return_feat)
        o["logits_cls"] = None
        return o


def build_model(name: str, pretrained: bool = True, **kw) -> nn.Module:
    name = name.lower()
    reg = {
        "cls_resnet50": lambda: ClsOnly("resnet50", pretrained=pretrained),
        "cls_effnetv2s": lambda: ClsOnly("tf_efficientnetv2_s.in21k_ft_in1k",
                                         pretrained=pretrained),
        "cls_swint": lambda: ClsOnly("swin_tiny_patch4_window7_224",
                                     pretrained=pretrained),
        "cls_vitb16": lambda: ClsOnly("vit_base_patch16_224", pretrained=pretrained),
        "seg_unet_r50": lambda: SegOnly("resnet50", pretrained=pretrained,
                                        use_ctam=False),
        "seg_unet_effv2s": lambda: SegOnly("tf_efficientnetv2_s.in21k_ft_in1k",
                                           pretrained=pretrained, use_ctam=False),
        "mt_base": lambda: MultiTaskNet("tf_efficientnetv2_s.in21k_ft_in1k",
                                        pretrained=pretrained,
                                        use_aspp=False, use_ctam=False),
        "mt_aspp": lambda: MultiTaskNet("tf_efficientnetv2_s.in21k_ft_in1k",
                                        pretrained=pretrained,
                                        use_aspp=True, use_ctam=False),
        "mt_ctam": lambda: MultiTaskNet("tf_efficientnetv2_s.in21k_ft_in1k",
                                        pretrained=pretrained,
                                        use_aspp=True, use_ctam=True),
        "mt_ctam_r50": lambda: MultiTaskNet("resnet50", pretrained=pretrained,
                                            use_aspp=True, use_ctam=True),
    }
    if name not in reg:
        sys.exit(f"Unknown model: {name}\nAvailable: {list(reg)}")
    m = reg[name]()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


MODEL_ZOO = ["cls_resnet50", "cls_effnetv2s", "cls_swint", "cls_vitb16",
             "seg_unet_r50", "seg_unet_effv2s",
             "mt_base", "mt_aspp", "mt_ctam", "mt_ctam_r50"]


def count_params(m: nn.Module) -> tuple[float, float]:
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot / 1e6, tr / 1e6


def self_test() -> None:
    cfg.set_seed()
    dev = cfg.get_device()
    x = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(dev)
    cfg.banner("Model tests")
    rows = []
    for n in MODEL_ZOO:
        try:
            m = build_model(n, pretrained=False).to(dev).eval()
            with torch.no_grad():
                o = m(x, return_feat=True) if not isinstance(m, ClsOnly) else m(x)
            sc = tuple(o["logits_cls"].shape) if o["logits_cls"] is not None else None
            ss = tuple(o["logits_seg"].shape) if o.get("logits_seg") is not None else None
            p, _ = count_params(m)
            rows.append((n, sc, ss, f"{p:.1f}M", "✓"))
            del m
            torch.cuda.empty_cache() if dev.type == "cuda" else None
        except Exception as e:
            rows.append((n, None, None, "-", f"✗ {str(e)[:40]}"))
    for r in rows:
        print(f"  {r[0]:18s} cls={str(r[1]):14s} seg={str(r[2]):22s} "
              f"{r[3]:>8s} {r[4]}")
    print(f"\n  input: {tuple(x.shape)}  |  device={dev}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        print("This is a module. To test:  python 08_models.py --self-test")
