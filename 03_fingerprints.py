




import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from importlib import import_module

cfg = import_module("00_config") if Path("00_config.py").exists() else None
if cfg is None:
    sys.exit("00_config.py not found.")


def md5_of_array(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def hash_one(path: str) -> dict:
    import imagehash

    im = Image.open(path).convert("L")
    a = np.array(im)

    im256 = im.resize((256, 256), Image.BILINEAR)
    a256 = np.array(im256)
    lo, hi = np.percentile(a256, [1, 99])
    a256n = np.clip((a256 - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    return {
        "md5_pixels": md5_of_array(a),
        "md5_resized": md5_of_array(a256),
        "md5_resized_norm": md5_of_array(a256n),
        "phash": str(imagehash.phash(im, hash_size=8)),
        "phash16": str(imagehash.phash(im, hash_size=16)),
        "dhash": str(imagehash.dhash(im, hash_size=8)),
        "ahash": str(imagehash.average_hash(im, hash_size=8)),
        "whash": str(imagehash.whash(im, hash_size=8)),
    }


def stage_hash(df: pd.DataFrame) -> None:
    try:
        import imagehash              
    except ImportError:
        sys.exit("imagehash required:  pip install ImageHash")

    recs = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="hashing"):
        d = hash_one(r["img_path"])
        d["uid"] = r["uid"]
        recs.append(d)

    h = pd.DataFrame(recs)
    out = cfg.DIR_EMBED / "hashes.csv"
    h.to_csv(out, index=False)
    print(f"\n→ {out}")

    m = df[["uid", "official_split", "class_name", "plane_name"]].merge(h, on="uid")

    cfg.banner("Exact pixel-level duplicates (md5_pixels)")
    for col, label in (
        ("md5_pixels", "raw pixels"),
        ("md5_resized", "after resize to 256"),
        ("md5_resized_norm", "after resize + contrast normalization"),
    ):
        vc = m[col].value_counts()
        dup_keys = vc[vc > 1].index
        sub = m[m[col].isin(dup_keys)]
        n_groups = len(dup_keys)
        cross = sub.groupby(col)["official_split"].nunique()
        n_cross = int((cross == 2).sum())
        leaked_test = sub[
            sub[col].isin(cross[cross == 2].index) & (sub.official_split == "test")
        ]
        print(f"\n  [{label}]")
        print(f"    duplicate groups       : {n_groups}")
        print(f"    images involved        : {len(sub)}")
        print(f"    train∩test groups      : {n_cross}")
        print(f"    leaked test images     : {len(leaked_test)}"
              f"  ({100*len(leaked_test)/1000:.1f}% of test)")
        if len(leaked_test):
            print("    by class:")
            for c, n in leaked_test["class_name"].value_counts().items():
                print(f"      {c:12s} {n}")


def stage_embed(df: pd.DataFrame, batch: int, model_name: str) -> None:
    try:
        import torch
        import timm
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        sys.exit("torch and timm required:  pip install torch timm")

    device = cfg.get_device()
    print(f"device={device}  model={model_name}")

    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    data_cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**data_cfg, is_training=False)
    print(f"input={data_cfg['input_size']}  dim={model.num_features}")

    class DS(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            im = Image.open(self.paths[i]).convert("RGB")
            return tf(im)

    paths = df["img_path"].tolist()
    dl = DataLoader(DS(paths), batch_size=batch, num_workers=4, pin_memory=True)

    feats = []
    with torch.no_grad():
        for x in tqdm(dl, desc="embedding"):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                f = model(x)
            feats.append(f.float().cpu().numpy())

    E = np.concatenate(feats, 0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)            

    np.save(cfg.DIR_EMBED / "dino_embeddings.npy", E.astype(np.float32))
    np.save(cfg.DIR_EMBED / "dino_uids.npy", df["uid"].to_numpy())
    print(f"\n→ {cfg.DIR_EMBED/'dino_embeddings.npy'}  shape={E.shape}")

    cfg.banner("Cosine similarity distribution (random sample)")
    rng = np.random.default_rng(cfg.GLOBAL_SEED)
    idx = rng.choice(len(E), size=min(3000, len(E)), replace=False)
    S = E[idx] @ E[idx].T
    iu = np.triu_indices_from(S, k=1)
    v = S[iu]
    qs = [50, 90, 95, 99, 99.5, 99.9, 99.95, 99.99]
    print("  Similarity percentiles of random pairs:")
    for q in qs:
        print(f"    p{q:<6} = {np.percentile(v, q):.4f}")
    print("\n  → Set the pseudo-patient threshold in script 04 around p99.9 to p99.99.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hash", "embed", "both"], default="both")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m",
                    help="lighter alternative: resnet50.a1_in1k")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg.set_seed()
    p = cfg.DIR_INDEX / "image_stats.csv"
    if not p.exists():
        p = cfg.DIR_INDEX / "index.csv"
    df = pd.read_csv(p)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    print(f"{len(df)} images")

    if args.stage in ("hash", "both"):
        cfg.banner("Step 1: hashes")
        stage_hash(df)
    if args.stage in ("embed", "both"):
        cfg.banner("Step 2: embeddings")
        stage_embed(df, args.batch, args.model)


if __name__ == "__main__":
    main()
