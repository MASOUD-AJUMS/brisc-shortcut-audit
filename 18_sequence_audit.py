

import argparse, json, os, random
import numpy as np
import pandas as pd
from PIL import Image

OUT = os.environ.get("BRISC_OUT", "/content/drive/MyDrive/BRISC_2025_Downloaded/outputs")
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
R = f"{OUT}/results"
LAB, KEY = f"{R}/seq_audit_labels.csv", f"{R}/seq_audit_key.json"
LAB2, KEY2 = f"{R}/seq_audit_labels_reader2.csv", f"{R}/seq_audit_key_reader2.json"
FIGD = f"{OUT}/figures/seq_audit"


def norm(v):
    v = str(v).strip().upper() if isinstance(v, str) else ""
    if not v:
        return ""
    if v.startswith("NON"):                                                                    
        return "other"
    if "CONTRAST" in v or v in ("T1C", "T1CE", "T1+C", "T1-C", "T1 C"):
        return "T1c"
    if v in ("T1", "T1W", "T1 WITHOUT CONTRAST"):
        return "T1"
    if v in ("T2", "T2W"):
        return "T2"
    if "FLAIR" in v:
        return "FLAIR"
    return "other"


def _sheet(paths, codes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(paths); cols = 10; rws = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rws, cols, figsize=(cols * 1.6, rws * 1.75))
    for a, p, c in zip(ax.ravel(), paths, codes):
        a.imshow(Image.open(p).convert("L"), cmap="gray"); a.set_title(c, fontsize=7)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def make(n, seed):
    f = pd.read_csv(f"{OUT}/groups/folds.csv")
    codes = list("ABCD"); rng = random.Random(seed)
    order = CLASSES[:]; rng.shuffle(order); key = dict(zip(codes, order))
    os.makedirs(FIGD, exist_ok=True)
    rows, umap = [], {}
    for code in codes:
        sub = f[f.class_name == key[code]].sample(n, random_state=seed)
        ids = [f"{code}{k + 1:02d}" for k in range(len(sub))]
        _sheet(sub.img_path.tolist(), ids, code, f"{FIGD}/sheet_{code}.png")
        for i, (_, r) in zip(ids, sub.iterrows()):
            rows.append(dict(sheet=code, idx=i, plane=r.plane_name, sequence=""))                                 
            umap[i] = r.uid
    pd.DataFrame(rows).to_csv(LAB, index=False)
    json.dump({"sheet_class": key, "idx_uid": umap}, open(KEY, "w"))
    print(f"  ✓ sheets in {FIGD} — fill in the sequence column in {LAB}")


def _attach_class(d):
    f = pd.read_csv(f"{OUT}/groups/folds.csv")[["uid", "class_name"]]
    if "uid" not in d.columns:
        k = json.load(open(KEY))
        d["uid"] = d.idx.map(k.get("idx_uid", {}))
    return d.merge(f, on="uid", how="left")


def report(d, tag):
    d = d[d.seq != ""]
    print(f"\n  [{tag}] {len(d)} labels")
    print(pd.crosstab(d.class_name, d.seq, margins=True).to_string())
    from scipy.stats import fisher_exact
    nt = d.class_name == "no_tumor"
    for name, m in (("T1c", d.seq == "T1c"), ("T2/FLAIR", d.seq.isin(["T2", "FLAIR"]))):
        t = pd.crosstab(nt, m).reindex(index=[False, True], columns=[False, True], fill_value=0)
        print(f"  no_tumor vs tumors × {name}: Fisher p = {fisher_exact(t.values)[1]:.3g}")
    pct = pd.crosstab(d.class_name, d.seq, normalize="index").round(3)
    print("\n  Percent within each class:\n" + pct.to_string())
    if "plane" in d.columns:
        print("\n  no_tumor by plane:\n" + pd.crosstab(d[nt].plane, d[nt].seq).to_string())
    return pct


def summarize():
    d = _attach_class(pd.read_csv(LAB))
    d["seq"] = d.sequence.map(norm)
    report(d, "reader 1").to_csv(f"{R}/seq_audit_summary.csv")


def second_reader(seed):
    d = _attach_class(pd.read_csv(LAB))
    f = pd.read_csv(f"{OUT}/groups/folds.csv").set_index("uid")
    d = d.sample(frac=1, random_state=seed + 1).reset_index(drop=True)                     
    rng = random.Random(seed + 2)
    codes = [f"{c}{k:02d}" for c in "EFGH" for k in range(1, 61)][:len(d)]
    os.makedirs(FIGD, exist_ok=True)
    for s in "EFGH":
        idx = [i for i, c in enumerate(codes) if c[0] == s]
        if idx:
            _sheet([f.loc[d.uid[i], "img_path"] for i in idx], [codes[i] for i in idx], s, f"{FIGD}/sheet_{s}.png")
    pd.DataFrame(dict(idx=codes, sequence="")).to_csv(LAB2, index=False)
    json.dump(dict(zip(codes, d.uid)), open(KEY2, "w"))
    print(f"  ✓ sheets E..H (mixed, without uid) in {FIGD}")
    print(f"  the second reader should fill in the sequence column in {LAB2} — without seeing the first-round labels")


def agreement():
    from sklearn.metrics import cohen_kappa_score
    d1 = _attach_class(pd.read_csv(LAB)); d1["seq1"] = d1.sequence.map(norm)
    d2 = pd.read_csv(LAB2); k2 = json.load(open(KEY2))
    d2["uid"] = d2.idx.map(k2); d2["seq2"] = d2.sequence.map(norm)
    print(f"  round 1: {int((d1.seq1 != '').sum())}/{len(d1)} labels | round 2: {int((d2.seq2 != '').sum())}/{len(d2)} labels"
          f" | shared uids: {len(set(d1.uid.dropna()) & set(d2.uid.dropna()))}")
    if (d1.seq1 != "").sum() == 0:
        raise SystemExit(f"  ✗ first-round file ({LAB}) on Drive is empty — replace it with the filled version.")
    m = d1.merge(d2[["uid", "seq2"]], on="uid")
    m = m[(m.seq1 != "") & (m.seq2 != "")]
    print(f"  {len(m)} images labeled by both readers")
    print(f"  Cohen's kappa (5 classes) = {cohen_kappa_score(m.seq1, m.seq2):.3f}")
    b1, b2 = m.seq1 == "T1c", m.seq2 == "T1c"
    print(f"  Cohen's kappa (T1c vs non-T1c) = {cohen_kappa_score(b1, b2):.3f}")
    print("\n  Agreement table:\n" + pd.crosstab(m.seq1, m.seq2).to_string())
    m["seq"] = m.seq2
    report(m, "reader 2 (fully blinded)").to_csv(f"{R}/seq_audit_summary_reader2.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--summarize", action="store_true")
    g.add_argument("--second-reader", action="store_true")
    g.add_argument("--agreement", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    elif a.second_reader:
        second_reader(a.seed)
    elif a.agreement:
        agreement()
    else:
        if os.path.exists(LAB) and pd.read_csv(LAB).get("sequence", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").any():
            raise SystemExit("  ✗ seq_audit_labels.csv contains labels — rebuild stopped to avoid overwriting them.")
        make(a.n, a.seed)
