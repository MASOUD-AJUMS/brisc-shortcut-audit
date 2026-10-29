




import sys
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def hex_to_bits(s: pd.Series, nbits: int) -> np.ndarray:
    out = np.zeros((len(s), nbits), dtype=np.uint8)
    for i, h in enumerate(s.to_numpy()):
        v = int(h, 16)
        for b in range(nbits):
            out[i, nbits - 1 - b] = (v >> b) & 1
    return out


def exact_leakage(df: pd.DataFrame) -> pd.DataFrame:
    cfg.banner("Experiment A — exact leakage")
    rows = []
    for col in ("md5_pixels", "md5_resized", "md5_resized_norm"):
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        keys = vc[vc > 1].index
        sub = df[df[col].isin(keys)]
        nsplit = sub.groupby(col)["official_split"].nunique()
        cross_keys = nsplit[nsplit == 2].index
        leaked = sub[sub[col].isin(cross_keys) & (sub.official_split == "test")]
        ncls = sub.groupby(col)["class_name"].nunique()

        rows.append({
            "method": col,
            "dup_groups": len(keys),
            "dup_images": len(sub),
            "cross_split_groups": len(cross_keys),
            "leaked_test_images": len(leaked),
            "leaked_test_pct": 100 * len(leaked) / (df.official_split == "test").sum(),
            "groups_with_label_conflict": int((ncls > 1).sum()),
        })
        print(f"\n  [{col}]")
        print(f"    duplicate groups {len(keys)} | images {len(sub)} | "
              f"test leakage {len(leaked)} ({100*len(leaked)/1000:.1f}%)")
        if (ncls > 1).sum():
            print(f"    ⚠ {int((ncls>1).sum())} groups with conflicting labels! "
                  f"the same image is labeled with two different classes.")
        if len(leaked):
            print("    Test leakage by class:")
            for c, n in leaked.class_name.value_counts().items():
                tot = ((df.official_split == "test") & (df.class_name == c)).sum()
                print(f"      {c:12s} {n:3d}/{tot:3d} = {100*n/tot:5.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.DIR_RESULTS / "leakage_exact.csv", index=False)

    if "md5_resized_norm" in df.columns:
        col = "md5_resized_norm"
        vc = df[col].value_counts()
        sub = df[df[col].isin(vc[vc > 1].index)]
        ns = sub.groupby(col)["official_split"].nunique()
        cross = sub[sub[col].isin(ns[ns == 2].index)]
        cross.sort_values(col)[
            ["uid", "official_split", "class_name", "plane_name", col]
        ].to_csv(cfg.DIR_RESULTS / "leaked_uids_exact.csv", index=False)
        print(f"\n  → List of uids: {cfg.DIR_RESULTS/'leaked_uids_exact.csv'}")
    return res


def near_leakage(df: pd.DataFrame, emb: np.ndarray | None,
                 emb_uids: np.ndarray | None) -> pd.DataFrame:
    cfg.banner("Experiment B — near leakage")
    is_test = (df.official_split == "test").to_numpy()
    rows = []

    if "phash16" in df.columns:
        bits = hex_to_bits(df["phash16"], 256)
        tr_b, te_b = bits[~is_test], bits[is_test]
        best = np.full(te_b.shape[0], 999, dtype=np.int32)
        best_j = np.zeros(te_b.shape[0], dtype=np.int64)
        step = 512
        tr_f = tr_b.astype(np.float32)
        for i in range(0, te_b.shape[0], step):
            chunk = te_b[i:i+step].astype(np.float32)
            d = (chunk.sum(1)[:, None] + tr_f.sum(1)[None, :]
                 - 2 * chunk @ tr_f.T).astype(np.int32)
            best[i:i+step] = d.min(1)
            best_j[i:i+step] = d.argmin(1)
        for thr in (0, 2, 5, 8, 12, 16):
            n = int((best <= thr).sum())
            rows.append({"method": f"phash16_ham<={thr}", "leaked_test": n,
                         "pct": 100 * n / len(best)})
            print(f"    pHash16 distance ≤ {thr:2d} : {n:4d} test images "
                  f"({100*n/len(best):5.1f}%)")
        tr_uid = df.loc[~is_test, "uid"].to_numpy()
        pd.DataFrame({
            "test_uid": df.loc[is_test, "uid"].to_numpy(),
            "test_class": df.loc[is_test, "class_name"].to_numpy(),
            "nearest_train_uid": tr_uid[best_j],
            "nearest_train_class": df.loc[~is_test, "class_name"].to_numpy()[best_j],
            "hamming": best,
        }).to_csv(cfg.DIR_RESULTS / "near_dup_phash.csv", index=False)

    if emb is not None:
        order = pd.Series(range(len(emb_uids)), index=emb_uids)
        sel = df["uid"].map(order).to_numpy()
        E = emb[sel]
        Etr, Ete = E[~is_test], E[is_test]
        sims, js = [], []
        step = 512
        for i in range(0, Ete.shape[0], step):
            S = Ete[i:i+step] @ Etr.T
            sims.append(S.max(1)); js.append(S.argmax(1))
        sims = np.concatenate(sims); js = np.concatenate(js)
        print()
        for thr in (0.999, 0.99, 0.98, 0.95, 0.92, 0.90):
            n = int((sims >= thr).sum())
            rows.append({"method": f"dino_cos>={thr}", "leaked_test": n,
                         "pct": 100 * n / len(sims)})
            print(f"    DINOv2 similarity ≥ {thr:.3f} : {n:4d} test images "
                  f"({100*n/len(sims):5.1f}%)")
        tr_uid = df.loc[~is_test, "uid"].to_numpy()
        tr_cls = df.loc[~is_test, "class_name"].to_numpy()
        pd.DataFrame({
            "test_uid": df.loc[is_test, "uid"].to_numpy(),
            "test_class": df.loc[is_test, "class_name"].to_numpy(),
            "nearest_train_uid": tr_uid[js],
            "nearest_train_class": tr_cls[js],
            "cosine": sims,
        }).to_csv(cfg.DIR_RESULTS / "near_dup_dino.csv", index=False)
    else:
        print("    (embeddings not available — run 03 --stage embed)")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.DIR_RESULTS / "leakage_near.csv", index=False)
    return res


SHORTCUT_SETS = {
    "S1_size": ["img_h", "img_w", "aspect"],
    "S2_intensity": ["img_mean", "img_std", "img_p1", "img_p99", "img_min", "img_max"],
    "S3_brain_texture": ["lap_var", "brain_ratio"],
    "S4_all_non_anatomical": ["img_h", "img_w", "aspect", "img_mean", "img_std",
                            "img_p1", "img_p99", "lap_var", "brain_ratio"],
}


def shortcut_test(df: pd.DataFrame, leaked_uids: set) -> pd.DataFrame:
    cfg.banner("Experiment C — shortcut (classification without any anatomical information)")
    d = df.copy()
    d["aspect"] = d.img_h / d.img_w
    tr = d[d.official_split == "train"]
    te = d[d.official_split == "test"]
    te_clean = te[~te.uid.isin(leaked_uids)]

    models = {
        "DecisionTree_d4": DecisionTreeClassifier(max_depth=4, random_state=0),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, multi_class="multinomial")),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    }

    rows = []
    for sname, feats in SHORTCUT_SETS.items():
        for mname, mk in models.items():
            m = mk.__class__(**mk.get_params()) if hasattr(mk, "get_params") else mk
            m.fit(tr[feats], tr.class_name)
            for tag, s in (("test_full", te), ("test_deleaked", te_clean)):
                p = m.predict(s[feats])
                rows.append({
                    "feature_set": sname, "model": mname, "eval_on": tag,
                    "n": len(s),
                    "accuracy": accuracy_score(s.class_name, p),
                    "balanced_accuracy": balanced_accuracy_score(s.class_name, p),
                    "macro_f1": f1_score(s.class_name, p, average="macro"),
                })

    res = pd.DataFrame(rows)
    res.to_csv(cfg.DIR_RESULTS / "shortcut_baseline.csv", index=False)

    piv = res[res.eval_on == "test_full"].pivot_table(
        index="feature_set", columns="model", values="balanced_accuracy")
    print("\n  Balanced accuracy on the full test set:")
    print(piv.round(4).to_string())

    maj = (te.class_name.value_counts(normalize=True).max())
    print(f"\n  Majority baseline: acc={maj:.4f} | bal_acc=0.2500")

    best = res[(res.eval_on == "test_full")].sort_values("balanced_accuracy").iloc[-1]
    print(f"\n  Best shortcut: {best.feature_set} + {best.model}")
    print(f"    acc={best.accuracy:.4f}  bal_acc={best.balanced_accuracy:.4f}")

    feats = SHORTCUT_SETS["S4_all_non_anatomical"]
    m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    m.fit(tr[feats], tr.class_name)
    p = m.predict(te[feats])
    print("\n  Per-class report (S4 + RandomForest):")
    print(classification_report(te.class_name, p, digits=3))
    print("  Confusion matrix:")
    print(pd.DataFrame(confusion_matrix(te.class_name, p, labels=cfg.CLASSES),
                       index=cfg.CLASSES, columns=cfg.CLASSES).to_string())
    print("\n  Feature importances:")
    for n_, v in sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1]):
        print(f"    {n_:14s} {v:.3f}")
    return res


def make_figure(exact: pd.DataFrame, near: pd.DataFrame, short: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, figure skipped.")
        return

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    if len(exact):
        ax[0].bar(range(len(exact)), exact.leaked_test_pct, color="firebrick")
        ax[0].set_xticks(range(len(exact)))
        ax[0].set_xticklabels(exact.method, rotation=20, ha="right", fontsize=8)
        ax[0].set_ylabel("% of test set leaked")
        ax[0].set_title("(a) Exact duplicate leakage")

    ph = near[near.method.str.startswith("phash")]
    dn = near[near.method.str.startswith("dino")]
    if len(ph):
        ax[1].plot([int(m.split("=")[-1]) for m in ph.method], ph.pct,
                   "o-", label="pHash16", color="darkorange")
        ax[1].set_xlabel("Hamming threshold")
    if len(dn):
        a2 = ax[1].twiny()
        a2.plot([float(m.split("=")[-1]) for m in dn.method], dn.pct,
                "s--", label="DINOv2", color="teal")
        a2.set_xlabel("cosine threshold")
    ax[1].set_ylabel("% of test set")
    ax[1].set_title("(b) Near-duplicate leakage")

    s = short[(short.eval_on == "test_full") & (short.model == "RandomForest")]
    ax[2].bar(range(len(s)), s.balanced_accuracy, color="steelblue")
    ax[2].axhline(0.25, ls="--", c="gray", label="chance")
    ax[2].axhline(0.99, ls=":", c="firebrick", label="published SOTA")
    ax[2].set_xticks(range(len(s)))
    ax[2].set_xticklabels([x.split("_", 1)[0] for x in s.feature_set], fontsize=9)
    ax[2].set_ylabel("Balanced accuracy")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(fontsize=8)
    ax[2].set_title("(c) Shortcut: no anatomy used")

    plt.tight_layout()
    p = cfg.DIR_FIGS / "fig1_leakage_and_shortcut.png"
    plt.savefig(p, dpi=180)
    plt.close()
    print(f"\n→ {p}")


def main() -> None:
    cfg.set_seed()
    st = pd.read_csv(cfg.DIR_INDEX / "image_stats.csv")
    hp = cfg.DIR_EMBED / "hashes.csv"
    if not hp.exists():
        sys.exit("hashes.csv not found. Run 03_fingerprints.py --stage hash first.")
    df = st.merge(pd.read_csv(hp), on="uid", how="left")

    emb = emb_uids = None
    ep = cfg.DIR_EMBED / "dino_embeddings.npy"
    if ep.exists():
        emb = np.load(ep)
        emb_uids = np.load(cfg.DIR_EMBED / "dino_uids.npy", allow_pickle=True)

    exact = exact_leakage(df)
    near = near_leakage(df, emb, emb_uids)

    lk = cfg.DIR_RESULTS / "leaked_uids_exact.csv"
    leaked = set(pd.read_csv(lk).query("official_split=='test'").uid) if lk.exists() else set()
    short = shortcut_test(df, leaked)

    make_figure(exact, near, short)

    cfg.banner("Summary for the paper")
    print(f"  test images with exact duplicate in train : {len(leaked)}")
    if len(near):
        r = near[near.method == "dino_cos>=0.95"]
        if len(r):
            print(f"  test images with DINO similarity ≥ 0.95   : {int(r.leaked_test.iloc[0])}")
    b = short[(short.eval_on == "test_full")].balanced_accuracy.max()
    print(f"  best shortcut accuracy (no anatomy)       : {b:.4f}")
    print("\n  These three numbers are the first paragraph of your Results.")


if __name__ == "__main__":
    main()
