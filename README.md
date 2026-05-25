# 3D Brain Tumor Segmentation

3D brain tumor segmentation on the [BraTS 2021](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) dataset using PyTorch + MONAI. Four MRI modalities (T1, T1ce, T2, FLAIR) are stacked into a 4-channel input; the model predicts 4 classes:

| Label | Class |
|-------|-------|
| 0 | Background |
| 1 | Necrotic core (NCR/NET) |
| 2 | Peritumoral edema |
| 3 | Enhancing tumor (BraTS label 4 remapped to 3) |

---

## Setup

### 1. Create conda environment

```bash
conda create -n imgp python=3.11 -y
conda activate imgp
pip install -r requirements.txt
```

### 2. Get the data

Download **BraTS 2021 Training Data** from [Synapse](https://www.synapse.org/#!Synapse:syn27046444) and place it at:

```
data/raw/BraTS2021_Training_Data/
    BraTS2021_00000/
        BraTS2021_00000_t1.nii.gz
        BraTS2021_00000_t1ce.nii.gz
        BraTS2021_00000_t2.nii.gz
        BraTS2021_00000_flair.nii.gz
        BraTS2021_00000_seg.nii.gz
    BraTS2021_00001/
        ...
```

---

## Training

```bash
conda activate imgp

# Default: UNet + Dice-Focal loss, 300 epochs
python train.py --config configs/default.yaml

# SegResNet (larger, better accuracy, needs more VRAM)
python train.py --config configs/segresnet.yaml

# Override model or loss from CLI
python train.py --config configs/default.yaml --model attention_unet --loss dice

# Resume from a checkpoint
python train.py --config configs/default.yaml --resume outputs/unet_dice_focal/best_model.pth

# Verify setup without full training (one forward pass)
python train.py --config configs/default.yaml --dry-run
```

Checkpoints are saved to `outputs/<run_name>/`:
- `best_model.pth` — best validation Dice
- `last_model.pth` — final epoch

The first run caches preprocessed volumes in `data/cache/` (takes ~10-30 min for 1251 cases). Subsequent runs load instantly from cache.

### Monitor training

```bash
# In a separate terminal
tensorboard --logdir outputs/
# Open http://localhost:6006
```

---

## Evaluation

```bash
# Dice scores on test split
python evaluate.py --checkpoint outputs/unet_dice_focal/best_model.pth

# With per-case CSV + axial slice and 3D visualizations
python evaluate.py --checkpoint outputs/unet_dice_focal/best_model.pth \
    --error-analysis --visualize --n-viz 5

# Evaluate on validation split instead of test
python evaluate.py --checkpoint outputs/unet_dice_focal/best_model.pth --split val
```

---

## Interactive UI

```bash
python app.py
# Open http://localhost:7860
```

Upload four NIfTI files (T1, T1ce, T2, FLAIR) or pick a dataset case directly. Optionally upload the ground truth segmentation to get per-class Dice scores. Exports predictions as `.nii.gz`.

---

## Compare loss functions

Trains three variants (dice / focal / dice_focal) back-to-back and prints a comparison table:

```bash
python compare.py --config configs/default.yaml
```

---

## Configuration

Key options in `configs/default.yaml`:

| Key | Default | Options |
|-----|---------|---------|
| `model` | `unet` | `unet`, `attention_unet`, `segresnet` |
| `loss` | `dice_focal` | `dice`, `focal`, `dice_focal` |
| `patch_size` | `[96, 96, 96]` | `[128, 128, 128]` for segresnet |
| `batch_size` | `1` | increase to `2` if VRAM allows |
| `num_workers` | `4` | increase for faster data loading |
| `max_epochs` | `300` | |
| `sw_overlap` | `0.5` | `0.75` for cleaner boundaries (slower) |
| `use_wandb` | `false` | requires `wandb login` first |
| `cache_dir` | `data/cache` | where preprocessed volumes are stored |

CLI args `--model` and `--loss` override the config file values.

---

## Project structure

```
configs/
    default.yaml        # UNet config (96³ patches)
    segresnet.yaml      # SegResNet config (128³ patches)
src/
    data/
        preprocessing.py    # data discovery, offline preprocessing utilities
        dataset.py          # MONAI transform pipelines, DataLoader factory
    models/
        unet3d.py           # model factory (UNet, AttentionUnet, SegResNet)
        losses.py           # loss factory (Dice, Focal, DiceFocal)
    training/
        trainer.py          # training loop, validation, checkpointing
    evaluation/
        metrics.py          # sliding-window inference + Dice metrics
        error_analysis.py   # per-case CSV export
    visualization/
        viz.py              # axial grid plots + Plotly 3D renders
train.py        # training entry point
evaluate.py     # evaluation entry point
compare.py      # loss comparison entry point
app.py          # Gradio UI entry point
preprocess.py   # offline preprocessing script (bias correction, resampling)
```

---

## Offline preprocessing (optional)

For advanced preprocessing (N4 bias field correction, resampling, skull stripping):

```bash
python preprocess.py --help
```

Outputs a manifest CSV that can be pointed to via `dataset_format: csv` + `manifest: <path>` in the config.
