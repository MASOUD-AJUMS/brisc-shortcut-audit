# Detecting Contrast Rather Than Tumor

Code for the paper **"Detecting Contrast Rather Than Tumor: A Leakage- and Shortcut-Aware Re-evaluation of Deep Learning on a Public Brain Magnetic Resonance Imaging Benchmark"** (submitted to *Academic Radiology*).

Deep learning classifiers trained on public brain tumor MRI benchmarks routinely report accuracies above 99%. This repository contains the full pipeline we used to test whether that performance on **BRISC2025** reflects tumor recognition. It audits the dataset, re-splits it at the (pseudo-)patient level, trains classification, segmentation, and multi-task models, and then probes what those models actually rely on: patient leakage, counterfactual lesion removal and insertion with sham controls, Grad-CAM, calibration and conformal prediction, and external validation on paired contrast-enhanced and noncontrast BraTS 2021 slices.

## Main findings

| Finding | Result |
|---|---|
| Official test images sharing a patient with the training set | 49.8% (10.1% pixel-identical) |
| No-tumor images that are **not** contrast-enhanced T1 (blinded audit) | 83%, vs 2% of tumor images |
| Tumor images still called "tumor" after the tumor is removed | 94%–97% |
| BraTS glioma slices called tumor by MT-Base, T1c → same slice without contrast | 76% → 16% |
| External four-class dataset images that duplicate BRISC2025 | 53.9% |

In short: near-perfect accuracy on BRISC2025 coexists with patient-level leakage and a tumor label confounded with gadolinium enhancement, and the models largely detect contrast rather than tumor.

## Pipeline at a glance

```
 Data audit               Protocols              Models (5-fold CV)          Evaluation
 ──────────               ─────────              ──────────────────          ──────────
 01 index                 06 P1 official split   11 ResNet-50 (cls)          14 effect of leakage (DiD)
 02 image stats           06 P2 purged test      11 U-Net (seg)              15 Grad-CAM attribution
 03 hashes + DINOv2       06 P3 pseudo-patient   11 MT-Base / ASPP / CTAM    16 counterfactual edits
 04 duplicates/shortcut      grouped split       11 + evidence-mass loss     17 calibration + conformal
 05b/05c radiologist-                            12b inpainter (for edits)   19 statistics
     calibrated threshold                                                    20 external validation
 05 pseudo-patients
 18 blinded sequence audit
```

## Repository layout

| Script | Purpose | Main outputs (under `outputs/`) |
|---|---|---|
| `00_config.py` | Paths, class/plane definitions, image size (224), seeds. Every other script imports it. | — |
| `01_build_index.py` | Parses BRISC2025 file names into an image index (class, plane, official split). | `index/index.csv` |
| `02_audit.py` | Per-image statistics: intensity, size, brain area, border features. | `index/image_stats.csv` |
| `03_fingerprints.py` | MD5 of decoded pixels, perceptual hashes, and DINOv2 (ViT-S/14) embeddings. | `embeddings/hashes.csv`, `embeddings/dino_embeddings.npy` |
| `04_leakage_audit.py` | Exact and near-duplicate detection across splits; non-anatomical shortcut probe (random forest on image statistics). | `results/leakage_*.csv`, `results/shortcut_baseline.csv` |
| `05b_review_sheets.py` | Samples image pairs across cosine-similarity bands and builds blinded review sheets for the radiologist. | `groups/review_stratified.csv`, `groups/review_key.csv` |
| `05c_calibrate_threshold.py` | Turns the reviewer's answers into band precision with Wilson intervals and selects the similarity threshold. | `groups/band_precision.csv`, `results/leakage_estimate.csv` |
| `05_pseudo_patients_v2.py` | Hierarchical clustering of DINOv2 embeddings into pseudo-patients; linkage sensitivity and threshold calibration. | `groups/pseudo_patients.csv`, `groups/linkage_sensitivity.csv` |
| `06_make_folds.py` | Builds the three evaluation protocols (P1, P2, P3) and stratified group 5-fold splits. | `groups/folds.csv`, `groups/protocol_summary.csv` |
| `07_data.py`, `07b_transforms_fix.py` | Dataset, sampler, and preprocessing (longest side 224, CLAHE, zero-padding, augmentation). | — |
| `08_models.py` | Model zoo: `cls_resnet50`, `seg_unet_r50`, `mt_base`, `mt_aspp`, `mt_ctam` (EfficientNetV2-S encoder) and others. | — |
| `09_losses.py` | Segmentation, classification, supervised contrastive, evidence-mass, and counterfactual losses; uncertainty weighting. | — |
| `10_metrics.py` | Balanced accuracy, AUC, ECE, Dice, HD95, empty-mask accuracy. | — |
| `11_train.py` | Training and evaluation of one model on one protocol and set of folds. | `checkpoints/`, `results/history_*`, `results/preds_*` |
| `12b_inpainter_v2.py` | Gated-convolution inpainting network used to remove tumors for counterfactual edits. (`12_inpainter.py` is the superseded first version.) | `checkpoints/inpainter_v2.pt` |
| `13_early_analysis.py` | Learning curves and a progress report while training runs. | `results/` |
| `14_leakage_effect.py` | Clean vs contaminated vs duplicated official test images; difference-in-differences against leakage-free P3 models. | `results/leakage_effect_P1_*.csv` |
| `15_attribution_eval.py` | Grad-CAM inside the tumor, enrichment, and attribution outside the head. | `results/attribution_P3_summary.csv` |
| `16_counterfactual.py` | Tumor removal (A), sham removal (S), tumor insertion (B), sham insertion (Bs); lesion reliance score. | `results/cf_P3_summary.csv`, `results/cf_P3_by_size.csv` |
| `17_calibration_conformal.py` | Temperature scaling; LAC, APS, and class-conditional (Mondrian) conformal sets; detection ambiguity. | `results/temperature_P3.json`, `results/conformal_P3_*` |
| `18_sequence_audit.py` | Blinded sequence audit: builds anonymized image sheets, then summarizes the reader's labels. | `results/seq_audit_summary.csv` |
| `19_stats.py` | Paired bootstrap, DeLong, McNemar, and Wilcoxon tests with Holm correction. | `results/stats_P3_*.csv` |
| `20_external_validation.py` | BraTS 2021 slice extraction (paired T1c/T1), external four-class dataset, overlap screening, inference, report. | `external/`, `results/external_*` |
| `20b_check_mendeley_dups.py`, `20c_external_dup_impact.py` | Verification of external duplicates and their effect on accuracy. | `results/external_dup_impact.csv` |
| `20d_contrast_patient_level.py` | Patient-level contrast-removal effect (cluster bootstrap, Wilcoxon). | `results/` |
| `21_make_figures.py` | Main-article figures 1–9 (PNG + PDF). | `figures/paper/` |
| `22_supplementary.py` | Supplementary figures S1–S10 and tables S4–S12. | `figures/supp/` |
| `23_journal_figures.py` | Figures 1–9 as 600-dpi LZW TIFF + vector PDF for journal submission. | `figures/journal/` |
| `patch_cf.py` | Replaces the counterfactual loss in `09_losses.py` with the relative-hinge version used in the pilot. | — |

## Installation

Python 3.10+ and a CUDA GPU are recommended (all experiments were run on a single NVIDIA T4).

```bash
git clone <repository-url>
cd <repository>
pip install torch torchvision timm albumentations opencv-python-headless \
            numpy pandas scipy scikit-learn pillow imagehash nibabel matplotlib tqdm
```

## Data

| Dataset | Use | Source |
|---|---|---|
| BRISC2025 | Development (6,000 T1c slices, 4 classes, expert tumor masks) | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) |
| BraTS 2021 | External: glioma slices, paired T1c / noncontrast T1 | [BraTS 2021 challenge](https://www.synapse.org/brats2021) |
| Multi-class brain tumor MRI dataset | External: four classes (glioma, meningioma, pituitary macroadenoma, healthy) | [Mendeley Data, doi:10.17632/82mtzd8x72.2](https://data.mendeley.com/datasets/82mtzd8x72/2) |

The datasets are not redistributed here. Download them and point the code to them with two environment variables:

```bash
export BRISC_ROOT=/path/to/brisc2025     # contains classification_task/ and segmentation_task/
export BRISC_OUT=/path/to/outputs        # everything the pipeline writes goes here
```

Expected BRISC2025 layout (as distributed on Kaggle):

```
brisc2025/
├── classification_task/{train,test}/<class>/brisc2025_<split>_<n>_<gl|me|pi|no>_<ax|co|sa>_t1.jpg
└── segmentation_task/{train,test}/{images,masks}/
```

## Files provided in this repository

Besides the code, the repository includes the files needed to reproduce the data audit without repeating the manual reviews:

| File | Content |
|---|---|
| `outputs/groups/review_stratified.csv`, `outputs/groups/review_key.csv` | Radiologist's blinded same-examination judgments for 241 image pairs, and the similarity band of each pair |
| `outputs/groups/band_precision.csv` | Precision per similarity band with Wilson intervals (output of `05c`) |
| `outputs/groups/pseudo_patients.csv` | Pseudo-patient ID of every BRISC2025 image |
| `outputs/groups/folds.csv` | Role (train/validation/test) and fold of every image under protocols P1, P2, and P3 |
| `outputs/results/seq_audit_labels.csv`, `outputs/results/seq_audit_key.json` | Blinded sequence labels for 240 images and the key linking them to image IDs |
| `outputs/external_duplicates.csv` | External images identified as duplicates of BRISC2025 images (exact or difference-hash distance ≤ 10) |

Local file paths were removed from these files; `pseudo_patients.csv` and `folds.csv` are provided for reference and are regenerated, with paths, by `05` and `06`. Datasets, model checkpoints, and other intermediate outputs are not included; they are regenerated by the pipeline.

## Reproducing the study

Run the scripts from the repository folder, in order. Each stage only reads what earlier stages wrote to `$BRISC_OUT`.

### 1. Data audit and evaluation protocols

```bash
python 01_build_index.py
python 02_audit.py
python 03_fingerprints.py --stage both
python 04_leakage_audit.py
python 05b_review_sheets.py --per-band 30        # blinded pair sheets for the radiologist
# the reviewer fills in answer_same_study (yes / no / unsure) in groups/review_stratified.csv
python 05c_calibrate_threshold.py --min-precision 1.0
python 05_pseudo_patients_v2.py --threshold 0.98026 --linkage complete
python 06_make_folds.py
```

The pseudo-patient threshold was calibrated by a radiologist who judged 241 image pairs, sampled across nine cosine-similarity bands (pixel-identical pairs, six bands between 0.980 and 0.999, and two control bands), without seeing the similarity scores or the split membership. `05c_calibrate_threshold.py` estimates the precision of each band with Wilson intervals; the threshold (0.98026) is the lowest similarity whose band precision was 1.00. The anonymized review answers are provided in `groups/review_stratified.csv` and `groups/review_key.csv` so that this step can be reproduced without repeating the review.

Blinded sequence audit (60 random images per class):

```bash
python 18_sequence_audit.py              # writes anonymized sheets + an empty label file
# the reader fills in seq_audit_labels.csv (T1c, T1, FLAIR, T2, other)
python 18_sequence_audit.py --summarize
```

### 2. Training

All runs used five folds, batch size 8, up to 40 epochs, and early stopping on the mean of validation balanced accuracy and lesion Dice.

```bash
# Protocol 3 (patient-grouped): main models
for m in cls_resnet50 seg_unet_r50 mt_base mt_aspp mt_ctam; do
  python 11_train.py --model $m --protocol P3 --folds 0 1 2 3 4 --tag base --batch-size 8
done

# Multi-task models with the evidence-mass loss
python 11_train.py --model mt_base --protocol P3 --folds 0 1 2 3 4 --tag ev --w-evidence 0.1 --batch-size 8
python 11_train.py --model mt_ctam --protocol P3 --folds 0 1 2 3 4 --tag ev --w-evidence 0.1 --batch-size 8

# Protocol 1 (official split), used for the effect of leakage
python 11_train.py --model mt_base --protocol P1 --folds 0 1 2 3 4 --tag base --batch-size 8

# Optional single-fold pilot with the counterfactual loss
python patch_cf.py
python 11_train.py --model mt_ctam --protocol P3 --folds 0 --tag cf --w-cf 0.1 --batch-size 8
```

### 3. Analyses

```bash
python 12b_inpainter_v2.py                 # inpainting network for counterfactual edits
python 14_leakage_effect.py
python 15_attribution_eval.py
python 16_counterfactual.py
python 17_calibration_conformal.py
python 19_stats.py
```

### 4. External validation

```bash
python 20_external_validation.py --brats-root /path/to/BraTS2021 --mendeley-dir /path/to/mendeley
python 20b_check_mendeley_dups.py
python 20c_external_dup_impact.py
python 20d_contrast_patient_level.py
```

`20_external_validation.py` can also be run in stages with `--stage prep|dedup|infer|report`. Temperatures and conformal thresholds are transferred unchanged from BRISC2025; nothing is retrained or recalibrated on external data.

### 5. Figures and tables

```bash
python 21_make_figures.py         # Figures 1–9 (PNG + PDF)
python 22_supplementary.py        # Figures S1–S10 and Tables S4–S12
python 23_journal_figures.py      # Figures 1–9 as 600-dpi TIFF for submission
```

Figure 7 and Supplementary Figures S4 and S6 load model checkpoints; use `--no-model` to skip them, or `--only <n>` to regenerate selected figures.

## Mapping to the paper

| Paper item | Produced by |
|---|---|
| Table 1, Figure 3a | `04_leakage_audit.py`, `05_pseudo_patients_v2.py` |
| Figure 3b | `05b_review_sheets.py`, `05c_calibrate_threshold.py` |
| Table 2, Figure 3d | `18_sequence_audit.py` |
| Table 3 | `11_train.py`, `16_counterfactual.py`, `17_calibration_conformal.py` |
| Table 4, Figure 4 | `14_leakage_effect.py` |
| Table 5, Figure 9 | `20_external_validation.py`, `20c`, `20d` |
| Figures 5–6 | `16_counterfactual.py` |
| Figure 7 | `15_attribution_eval.py` (Grad-CAM) |
| Figure 8 | `17_calibration_conformal.py` |
| Supplementary Tables S4–S8 | `19_stats.py`, `17_calibration_conformal.py` |
| Supplementary Tables S9–S11 | `20_external_validation.py`, `20d`, `05_pseudo_patients_v2.py` |

## Notes

- Image size is 224 × 224 throughout (`IMG_SIZE` in `00_config.py`). The same preprocessing is used for training, evaluation, attribution, counterfactual editing, and external data.
- Some scripts fall back to a hard-coded Google Colab output path when `BRISC_OUT` is not set. Always set `BRISC_OUT`.
- Seeds are 42 plus the fold index. cuDNN is set to deterministic mode, but small run-to-run differences on GPU are still possible.

## Citation

If you use this code, please cite the article (citation will be added on publication).
