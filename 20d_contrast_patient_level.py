from importlib import import_module
import numpy as np
import pandas as pd
from scipy import stats

cfg = import_module("00_config")
EXT = cfg.OUT_ROOT / "external"
C = cfg.CLASSES
NT = C.index("no_tumor")


def main():
    idx = pd.read_csv(EXT / "external_index.csv", keep_default_na=False)
    P = pd.read_csv(EXT / "external_preds.csv", low_memory=False)
    P = P[P.pred.notna()].merge(idx[["uid", "pair_id", "source", "sequence", "kind", "case"]], on="uid")
    P = P[P.source == "brats"]
    pc = [f"p_{c}" for c in C]
    e = P.groupby(["model", "tag", "case", "pair_id", "sequence", "kind"])[pc].mean().reset_index()
    e["tumor"] = (e[pc].values.argmax(1) != NT).astype(float)
    rows, rng = [], np.random.default_rng(0)
    for (m, t, kind), g in e.groupby(["model", "tag", "kind"]):
        w = g.pivot_table(index=["case", "pair_id"], columns="sequence", values="tumor").dropna()
        if not {"t1c", "t1"} <= set(w.columns) or not len(w):
            continue
        pt = w.groupby(level="case").mean()
        d = (pt.t1c - pt.t1).values
        cases = pt.index.values
        sl = w.groupby(level="case").size().reindex(cases).values
        boots = []
        for _ in range(5000):
            k = rng.integers(0, len(cases), len(cases))
            boots.append(np.average(pt.t1c.values[k], weights=sl[k]) - np.average(pt.t1.values[k], weights=sl[k]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = stats.wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        rows.append(dict(model=m, tag=t, kind=kind, n_patients=len(cases), n_pairs=len(w),
                         rate_t1c=w.t1c.mean(), rate_t1=w.t1.mean(), diff=w.t1c.mean() - w.t1.mean(),
                         ci95_lo=lo, ci95_hi=hi, patients_t1c_higher=int((d > 0).sum()),
                         patients_t1_higher=int((d < 0).sum()), wilcoxon_p=p))
    R = pd.DataFrame(rows)
    R.to_csv(cfg.DIR_RESULTS / "external_contrast_patient.csv", index=False)
    pd.set_option("display.width", 200)
    cfg.banner("Patient-level contrast (cluster bootstrap + Wilcoxon)")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
