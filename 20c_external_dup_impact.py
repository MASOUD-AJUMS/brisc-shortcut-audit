
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

cfg = import_module("00_config")
data_mod = import_module("07_data")
EXT = cfg.OUT_ROOT / "external"
C = cfg.CLASSES


def metrics(g):
    y, pr = g.label.values.astype(int), g.pred.values.astype(int)
    P = g[[f"p_{c}" for c in C]].values
    r = dict(n=len(g), bal_acc=np.mean([np.mean(pr[y == c] == c) for c in np.unique(y)]),
             acc=np.mean(pr == y))
    try:
        r["macro_auc"] = np.mean([roc_auc_score(y == c, P[:, c]) for c in np.unique(y)])
    except ValueError:
        r["macro_auc"] = np.nan
    if "covered" in g and g.covered.notna().any():
        r["coverage"] = g.covered.astype(float).mean(); r["set_size"] = g.set_size.mean()
    return r


def main():
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    chk = pd.read_csv(cfg.DIR_RESULTS / "external_dup_check.csv")
    d = chk.set_index("uid")
    dup = ((d.dhash_dist <= 10) | (d.ncc >= 0.95))
    idx["dup_near"] = idx.uid.map(dup).fillna(False).astype(bool) | \
        idx.dup_near.astype(str).str.lower().isin(["true", "1"])
    idx.to_csv(EXT / "external_index.csv", index=False)
    me = idx[idx.source == "mendeley"]
    print(f"  Mendeley final duplicates: {int(me.dup_near.sum())} of {len(me)} | "
          f"non-duplicates: {int((~me.dup_near).sum())}")
    print("  Class distribution of non-duplicates:", me[~me.dup_near].label.map(lambda i: C[int(i)]).value_counts().to_dict())

    P = pd.read_csv(EXT / "external_preds.csv")
    P = P[P.pred.notna()].merge(idx[["uid", "source", "label", "dup_near"]], on="uid")
    P = P[P.source == "mendeley"].merge(chk[["uid", "brisc_uid"]], on="uid", how="left")
    folds = data_mod.load_folds("P3")

    rows = []
    for f in sorted(P.fold.unique()):
        tr, va, te = data_mod.get_split(folds, "P3", int(f))
        seen = set(tr.uid) | set(va.uid)
        for (m, t), g in P[P.fold == f].groupby(["model", "tag"]):
            subsets = {"naive_all": g, "dup": g[g.dup_near], "clean": g[~g.dup_near],
                       "dup_seen_in_training": g[g.dup_near & g.brisc_uid.isin(seen)],
                       "dup_unseen(test)": g[g.dup_near & ~g.brisc_uid.isin(seen)]}
            for s, h in subsets.items():
                if len(h) >= 20:
                    rows.append(dict(model=m, tag=t, fold=f, subset=s, **metrics(h)))
    R = pd.DataFrame(rows)
    R.to_csv(cfg.DIR_RESULTS / "external_dup_impact.csv", index=False)
    S = R.groupby(["model", "subset"]).agg(n=("n", "mean"), bal_acc=("bal_acc", "mean"), bal_sd=("bal_acc", "std"),
                                           auc=("macro_auc", "mean"), coverage=("coverage", "mean"))
    pd.set_option("display.width", 200)
    cfg.banner("Mendeley: naive evaluation vs cleaned (mean across folds)")
    print(S.round(3).to_string())
    print(f"\n  → {cfg.DIR_RESULTS / 'external_dup_impact.csv'}")
    print("  Now run:  python 20_external_validation.py --stage report")


if __name__ == "__main__":
    main()
