

import re, sys
from pathlib import Path


def sub_once(text, old, new, name):
    if new in text:
        print(f"  = {name}: already applied"); return text
    if old not in text:
        sys.exit(f"  ✗ {name}: expected text not found — file differs from the known version")
    print(f"  ✓ {name}")
    return text.replace(old, new, 1)


p = Path("09_losses.py"); s = p.read_text(encoding="utf-8")
i0 = s.index("class CounterfactualLoss(nn.Module):")
i1 = s.index("@torch.no_grad()", i0)
new_cls = '''class CounterfactualLoss(nn.Module):
    def __init__(self, no_tumor_idx: int = 3, margin: float = 0.6931):
        super().__init__()
        self.idx = no_tumor_idx
        self.m = margin

    def forward(self, logits_ablated, has_lesion, logits_orig=None, labels=None):
        w = has_lesion > 0.5
        if w.sum() == 0 or logits_orig is None or labels is None:
            return logits_ablated.sum() * 0.0
        y = labels[w][:, None]
        la = F.log_softmax(logits_ablated[w].float(), 1).gather(1, y)[:, 0]
        lo = F.log_softmax(logits_orig[w].float(), 1).gather(1, y)[:, 0].detach()
        return F.relu(la - lo + self.m).mean()

'''
if "F.relu(la - lo + self.m)" in s[i0:i1]:
    print("  = CounterfactualLoss: already applied")
else:
    s = s[:i0] + new_cls + s[i1:]
    print("  ✓ CounterfactualLoss → hinge")
s = sub_once(s,
             'lcf = self.cf(out_ablated["logits_cls"], batch["has_lesion"])',
             'lcf = self.cf(out_ablated["logits_cls"], batch["has_lesion"],\n'
             '                          out["logits_cls"], batch["label"])',
             "TotalLoss: pass original logits and label")
p.write_text(s, encoding="utf-8")

p = Path("11_train.py"); s = p.read_text(encoding="utf-8")
s = sub_once(s, "            if args.w_cf > 0:\n",
             "            if args.w_cf > 0 and ep >= args.cf_warmup:\n", "warmup in train_epoch")
s = sub_once(s, '    ap.add_argument("--w-cf", type=float, default=0.0)\n',
             '    ap.add_argument("--w-cf", type=float, default=0.0)\n'
             '    ap.add_argument("--cf-warmup", type=int, default=5,\n'
             '                    help="cf loss is enabled from this epoch")\n', "--cf-warmup argument")
if 'scaler.load_state_dict(ck["scaler"])' in s and 'if ck.get("scaler")' not in s:
    s = re.sub(r'(\n\s*)scaler\.load_state_dict\(ck\["scaler"\]\)',
               r'\1if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])', s, count=1)
    print("  ✓ scaler fix for resume")
else:
    print("  = scaler fix: already applied")
p.write_text(s, encoding="utf-8")

import importlib, os, torch
sys.path.insert(0, os.getcwd())
L = importlib.import_module("09_losses")
cf = L.CounterfactualLoss()
lo = torch.tensor([[4., 0, 0, 0], [0, 4., 0, 0]]); y = torch.tensor([0, 1]); h = torch.ones(2)
same = cf(lo.clone(), h, lo, y).item()
dropped = cf(torch.tensor([[0., 0, 0, 4.], [0, 0, 0, 4.]]), h, lo, y).item()
print(f"\n  test: no change → loss={same:.3f} (should be ≈0.693) | lesion removed and probability dropped → loss={dropped:.3f} (should be 0)")
assert abs(same - 0.6931) < 1e-3 and dropped == 0.0
print("  ✓ Ready. Try a single fold first:")
print("    python 11_train.py --model mt_ctam --protocol P3 --folds 0 --epochs 40 --batch-size 8 "
      "--w-evidence 0.1 --w-cf 0.1 --tag final2")
