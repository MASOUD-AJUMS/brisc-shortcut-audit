

from pathlib import Path
import os
import random

import numpy as np

DATA_ROOT = Path(os.environ.get("BRISC_ROOT", "./brisc2025"))

OUT_ROOT = Path(os.environ.get("BRISC_OUT", "./outputs"))

DIR_INDEX = OUT_ROOT / "index"
DIR_EMBED = OUT_ROOT / "embeddings"
DIR_GROUPS = OUT_ROOT / "groups"
DIR_FIGS = OUT_ROOT / "figures"
DIR_CKPT = OUT_ROOT / "checkpoints"
DIR_RESULTS = OUT_ROOT / "results"

for _d in (DIR_INDEX, DIR_EMBED, DIR_GROUPS, DIR_FIGS, DIR_CKPT, DIR_RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

CLS_TRAIN = DATA_ROOT / "classification_task" / "train"
CLS_TEST = DATA_ROOT / "classification_task" / "test"
SEG_TRAIN_IMG = DATA_ROOT / "segmentation_task" / "train" / "images"
SEG_TRAIN_MSK = DATA_ROOT / "segmentation_task" / "train" / "masks"
SEG_TEST_IMG = DATA_ROOT / "segmentation_task" / "test" / "images"
SEG_TEST_MSK = DATA_ROOT / "segmentation_task" / "test" / "masks"

MANIFEST_CSV = DATA_ROOT / "manifest.csv"
MANIFEST_JSON = DATA_ROOT / "manifest.json"

CODE2CLASS = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "no_tumor",
}
CLASS2CODE = {v: k for k, v in CODE2CLASS.items()}

CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
N_CLASSES = len(CLASSES)

NO_TUMOR_IDX = CLASS2IDX["no_tumor"]

CODE2PLANE = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
PLANE2CODE = {v: k for k, v in CODE2PLANE.items()}
PLANES = ["axial", "coronal", "sagittal"]
PLANE2IDX = {p: i for i, p in enumerate(PLANES)}

IMG_SIZE = 224                               
NATIVE_SIZE = 512                                    
N_FOLDS = 5
SEEDS = [42, 123, 456]                       
GLOBAL_SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PSEUDO_PATIENT_COS_THRESH = 0.92
PSEUDO_PATIENT_PHASH_DIST = 8                                       
PSEUDO_PATIENT_MAX_SIZE = 8                                          

def set_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


if __name__ == "__main__":
    banner("BRISC config")
    print(f"DATA_ROOT : {DATA_ROOT.resolve()}")
    print(f"OUT_ROOT  : {OUT_ROOT.resolve()}")
    print(f"exists : {DATA_ROOT.exists()}")
    for p in (CLS_TRAIN, CLS_TEST, SEG_TRAIN_IMG, SEG_TRAIN_MSK,
              SEG_TEST_IMG, SEG_TEST_MSK):
        print(f"  {'✓' if p.exists() else '✗'} {p}")
    print(f"\nmanifest.csv : {'✓' if MANIFEST_CSV.exists() else '✗'}")
