# 3D Brain Tumor Segmentation — Project Overview

A complete deep learning pipeline for automatically identifying and outlining brain tumor regions in 3D MRI scans, built with PyTorch and MONAI.

---

## 1. What Problem Does This Solve?

Brain tumor segmentation is the task of looking at an MRI scan and drawing a precise boundary around the tumor — down to the individual voxel (a 3D pixel). Doing this manually takes a radiologist 30–60 minutes per patient. An accurate automated system can do it in seconds and can be used to:

- Help radiologists confirm or flag suspicious regions faster
- Track tumor growth across multiple scans over time
- Plan radiation therapy by precisely delineating what needs to be treated

This project tackles that problem using the **BraTS 2021** dataset (Brain Tumor Segmentation Challenge), which is the standard academic benchmark for this task.

---

## 2. The Dataset — BraTS 2021

Each patient case has **four MRI scans** taken with different imaging protocols, each capturing different tissue properties:

| Modality | What It Shows |
|----------|---------------|
| **T1** | Basic anatomical structure |
| **T1ce** | Contrast-enhanced — highlights active tumor (blood vessels light up) |
| **T2** | Shows edema (swelling) clearly |
| **FLAIR** | Suppresses fluid signal; edema stands out sharply |

Each scan is a **3D volume** (~240 × 240 × 155 voxels at 1 mm resolution). Using all four together gives the model richer information than any single scan alone.

### What the Model Predicts

The model assigns one of **4 labels** to every voxel:

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | Background | Healthy brain / skull / air |
| 1 | NCR/NET | Necrotic core — the "dead" center of the tumor |
| 2 | Edema | Swelling around the tumor |
| 3 | Enhancing Tumor | Active, growing tumor tissue |

> **Note:** The original BraTS dataset uses label `4` for enhancing tumor. The code remaps it to `3` so the labels are contiguous (0, 1, 2, 3), which is required by the neural network.

---

## 3. Project Structure

```
medical_imgp/
├── train.py              # Entry point: trains a model
├── evaluate.py           # Entry point: tests a trained model
├── compare.py            # Entry point: benchmarks all 3 loss functions
├── app.py                # Entry point: launches interactive web UI
├── preprocess.py         # Entry point: offline preprocessing pipeline
│
├── configs/
│   ├── default.yaml      # Main config (UNet, Dice+Focal loss)
│   └── segresnet.yaml    # Alternate config for SegResNet
│
└── src/
    ├── data/
    │   ├── preprocessing.py  # Data discovery, normalization, label remapping
    │   └── dataset.py        # MONAI transform pipelines + DataLoaders
    ├── models/
    │   ├── unet3d.py         # Factory for 3 model architectures
    │   └── losses.py         # Factory for 3 loss functions
    ├── training/
    │   └── trainer.py        # Full training loop (AdamW + AMP + TensorBoard)
    ├── evaluation/
    │   ├── metrics.py        # Dice + Hausdorff distance computation
    │   └── error_analysis.py # Per-case CSV breakdown
    └── visualization/
        └── viz.py            # 2D slice grids + 3D Plotly renders
```

Every entry point reads from the same YAML config file. CLI arguments override individual config values, so you can switch models or loss functions without editing any file.

---

## 4. The Neural Network Models

Three architectures are available, all from the **MONAI** medical imaging library:

### 4.1 U-Net (default)
The classic encoder-decoder architecture for segmentation. The encoder progressively shrinks the volume and extracts features; the decoder progressively enlarges back to original resolution. Skip connections copy feature maps from encoder to decoder so fine spatial details are not lost.

- **Channels:** 32 → 64 → 128 → 256 → 320 (5 levels)
- **Strides:** 4× downsampling (2, 2, 2, 2)
- **Residual units:** 2 per block
- **Dropout:** 10%

### 4.2 Attention U-Net
Same as the U-Net but with **attention gates** on each skip connection. The attention mechanism learns to focus on relevant spatial regions and suppress irrelevant ones, which helps when the tumor occupies a small fraction of the volume.

### 4.3 SegResNet
A residual network encoder-decoder without the traditional U-Net skip structure. Uses **residual blocks** (the same idea as ResNet) throughout. The `segresnet.yaml` config pairs this with a larger patch size (128³ instead of 96³) and higher sliding-window overlap (0.75) for sharper boundaries.

All three accept the same **4-channel input** (one channel per MRI modality) and output **4 logit maps** (one per class).

---

## 5. The Data Pipeline

### 5.1 How Cases Are Loaded

The code automatically scans the BraTS directory tree and builds a list of file paths for each case. It can also load cases from a CSV or JSON manifest — useful if you preprocessed the data offline.

The dataset is split reproducibly (fixed random seed):
- **80% training**
- **10% validation** (monitored during training)
- **10% test** (held out; only used for final evaluation)

### 5.2 Two-Stage Transform Pipeline

Processing is split into two stages to avoid wasting compute:

**Stage 1 — Deterministic preprocessing (cached to disk once):**

1. Load all 4 NIfTI files per case
2. Stack them into a single 4-channel volume using `ConcatItemsd`
3. Remap labels (4 → 3)
4. Clip intensities to the 0.5th–99.5th percentile (removes outlier voxels that would skew normalization)
5. Z-score normalize each channel independently over non-zero (brain) voxels
6. Crop to the tight bounding box of the non-zero brain region (removes wasted background)
7. Pad to at least the patch size

These steps are expensive but deterministic, so results are written to `data/cache/` and reused every epoch.

**Stage 2 — Random augmentation (applied live at every load):**

1. **Patch cropping** — randomly extract a 96 × 96 × 96 sub-volume; ensures at least 50% of patches contain tumor (positive:negative ratio = 1:1), 2 patches per image per step
2. **Random flips** along all 3 axes (50% probability)
3. **Random 90° rotations** (50% probability)
4. **Gaussian noise** injection (20% probability, std = 0.05)

Augmentation is applied live (not cached) so the model sees a different random crop/flip/rotation at every step, effectively multiplying the dataset size.

### 5.3 Patch-Based Training

Full 3D brain volumes (~240³ voxels) are too large to fit in GPU memory as-is. Instead, the model trains on small 96 × 96 × 96 patches. The sampling is biased: at least half of all patches are guaranteed to contain at least one tumor voxel, preventing the model from learning to predict "everything is background."

---

## 6. The Loss Functions

Three loss functions are implemented, all excluding the background class (the network only has to get the three tumor classes right):

### 6.1 Dice Loss
Measures overlap between predicted and ground-truth segmentation masks:

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

Ranges from 0 (no overlap) to 1 (perfect overlap). **Dice Loss = 1 − Dice**. It is insensitive to class imbalance because it normalizes by both prediction and ground truth sizes.

### 6.2 Focal Loss
A modification of cross-entropy that down-weights easy (well-classified) voxels and focuses the gradient on hard ones. Controlled by the `gamma` parameter (default 2.0). Useful when most voxels are background (a class imbalance problem in medical images).

### 6.3 Dice + Focal (default, weighted 50/50)
Combines both: Dice handles global shape/overlap; Focal handles voxel-level difficulty. This combination consistently outperforms either alone on BraTS.

---

## 7. The Training Loop

**Optimizer:** AdamW (weight decay = 1e-5)  
**Learning rate:** 1e-4, decayed with cosine annealing down to 1e-6 over 300 epochs  
**Batch size:** 1 case at a time (each case yields 2 patches, so effectively 2 patches per gradient step)  
**Mixed precision:** FP16 via PyTorch AMP (`GradScaler`) — cuts VRAM usage roughly in half and speeds up training on modern GPUs

**Validation** (every 5 epochs):
- Inference uses **sliding window**: a 96³ window slides over the full volume with 50% overlap; predictions from overlapping windows are averaged with a Gaussian weighting (center voxels weighted more heavily than edges)
- Performance is measured as mean **Dice Similarity Coefficient** across the 3 tumor classes

**Checkpointing:**
- `best_model.pth` — saved whenever validation DSC improves
- `last_model.pth` — saved at the end of training
- Both checkpoints embed the full config dict, so evaluation can reconstruct exactly what training settings were used

**Logging:**
- TensorBoard: always on — loss per step, loss per epoch, learning rate, per-class DSC, gradient histograms
- Weights & Biases: optional (`use_wandb: true` in config)

---

## 8. Evaluation Metrics

After training, `evaluate.py` runs on the held-out test split and reports two metrics per tumor class:

| Metric | What It Measures |
|--------|-----------------|
| **DSC** (Dice Similarity Coefficient) | Overlap quality — 1.0 = perfect, 0 = no overlap |
| **HD95** (95th-percentile Hausdorff Distance) | Boundary accuracy in mm — how far off the predicted boundary is at worst (ignoring the most extreme 5%) |

Both metrics exclude background. Results are printed per class and as an average.

**Error analysis** (`--error-analysis` flag) exports a per-case CSV with DSC and HD95 for every test patient, so you can identify which cases the model struggles on.

---

## 9. Offline Preprocessing (`preprocess.py`)

For research use-cases where the raw data needs heavier processing before training, a standalone script applies expensive operations once and saves results to disk:

- **Resampling** — resample volumes to a fixed voxel spacing (e.g., 1 mm isotropic) using SimpleITK
- **N4ITK bias field correction** — removes scanner intensity non-uniformity artifacts
- **Co-registration** — rigidly aligns all modalities to the T1 scan using multi-resolution Mattes mutual information registration
- **Skull stripping** — removes non-brain tissue via HD-BET
- **Intensity clipping + z-score normalization** — same as online preprocessing but saved to disk

After running, a `manifest.csv` is written pointing to the preprocessed files. Set `dataset_format: csv` in the config to use it.

BraTS data is already co-registered and skull-stripped by the dataset organisers, so this script is mainly useful for custom MRI datasets.

---

## 10. The Loss Function Comparison Tool (`compare.py`)

Instead of training three models separately, `compare.py` runs all three loss functions back-to-back on the **same data split and same model architecture**, then prints a ranked leaderboard at the end:

```
FINAL LEADERBOARD (Best Val DSC)
==================================================
  1. unet_dice_focal_lr1e-04       0.8412 <-- WINNER
  2. unet_dice_lr1e-04             0.8301
  3. unet_focal_lr1e-04            0.8154
```

This is the standard ablation study approach: vary one thing at a time (loss function) while keeping everything else fixed.

---

## 11. Interactive Web UI (`app.py`)

A browser-based tool built with **Gradio** that lets you interact with a trained model without writing any code:

**Input:** Upload four NIfTI files (T1, T1ce, T2, FLAIR) or pick a case from the dataset directory.

**What it does:**
1. Loads the selected checkpoint (model is cached between runs)
2. Applies the same preprocessing as training
3. Runs sliding-window inference on the full volume
4. Visualizes results in three ways:
   - **Axial slice viewer** — slider to scroll through slices; FLAIR scan as background with color-coded tumor overlay
   - **Side-by-side comparison** — ground truth vs. prediction, if a segmentation file is provided
   - **Interactive 3D render** — marching-cubes mesh of the tumor displayed in a Plotly 3D plot; each class is a separate mesh with different color and opacity

**Metrics:** Per-class DSC shown in a color-coded table (green ≥ 0.75, yellow ≥ 0.50, red < 0.50) when ground truth is provided.

**Export:** Download the predicted segmentation as a `.nii.gz` NIfTI file preserving the original voxel spacing and affine.

Launch: `python app.py` → open `http://localhost:7860`

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 4-channel input (all modalities stacked) | Each modality reveals different tumor sub-regions; combining them gives the model maximum information |
| Patch-based training (96³) | Full volumes (~240³) exceed GPU VRAM; patch sampling also acts as augmentation |
| Background excluded from loss | Tumor voxels are ~5% of the volume; including background would let the model "cheat" by predicting mostly background |
| Sliding-window inference with Gaussian weighting | Full-volume inference after patch-based training; Gaussian weighting reduces stitching artifacts at patch boundaries |
| Persistent disk cache for preprocessing | First run is slow (transforms computed and saved); all subsequent epochs load the cached result — no redundant I/O |
| Config embedded in checkpoint | Evaluation always reconstructs exact training settings automatically, preventing train/test mismatch bugs |
| Mixed precision (FP16) | ~2× VRAM reduction, ~1.5× speed increase on modern NVIDIA GPUs |
| Cosine annealing LR schedule | Smooth decay without manual step scheduling; avoids sharp LR drops that can destabilize training |

---

## 13. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train with defaults (UNet + Dice+Focal, 300 epochs)
python train.py --config configs/default.yaml

# Quick sanity check (one forward pass, no training)
python train.py --config configs/default.yaml --dry-run

# Try a different architecture/loss
python train.py --config configs/default.yaml --model attention_unet --loss dice

# Evaluate the best checkpoint
python evaluate.py --checkpoint outputs/unet_dice_focal/best_model.pth

# Evaluate + generate visualizations for 5 worst cases
python evaluate.py --checkpoint outputs/unet_dice_focal/best_model.pth \
    --error-analysis --visualize --n-viz 5

# Compare all 3 loss functions in sequence
python compare.py --config configs/default.yaml

# Launch interactive UI
python app.py
```

---

## 14. Technology Stack

| Component | Library / Tool |
|-----------|---------------|
| Deep learning framework | PyTorch |
| Medical imaging toolkit | MONAI |
| NIfTI file I/O | nibabel |
| Advanced preprocessing | SimpleITK |
| 3D visualization | Plotly (marching cubes via scikit-image) |
| 2D visualization | matplotlib |
| Interactive UI | Gradio |
| Experiment tracking | TensorBoard (always) + W&B (optional) |
| Config management | YAML |
