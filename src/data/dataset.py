"""MONAI transform pipelines and DataLoader factory."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PersistentDataset,
    list_data_collate,
)
from monai.transforms import (
    ConcatItemsd,
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
)

from src.data.preprocessing import MODALITIES


# ---------------------------------------------------------------------------
# Helper functions (must be top-level / picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _remap_brats_label(x: torch.Tensor) -> torch.Tensor:
    """Remap BraTS label 4 → 3 so class indices are contiguous (0,1,2,3)."""
    return torch.where(x == 4, torch.full_like(x, 3), x)


def _remap_msd_label(x: torch.Tensor) -> torch.Tensor:
    """Remap MSD Task01 labels to BraTS convention: 1(edema)↔2(NCR/NET), 3→3."""
    out = x.clone()
    out[x == 1] = 2
    out[x == 2] = 1
    return out


def _clip_percentile(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Clip each channel independently to [lower, upper] percentile of non-zero voxels."""
    out = x.clone()
    for c in range(out.shape[0]):
        ch = out[c]
        nz = ch[ch != 0].float()
        if nz.numel() > 0:
            lo = torch.quantile(nz, lower / 100.0)
            hi = torch.quantile(nz, upper / 100.0)
            out[c] = ch.clamp(lo, hi)
    return out


# ---------------------------------------------------------------------------
# Dataset wrapper: cached base + live random augmentation
# ---------------------------------------------------------------------------

class _AugDataset(torch.utils.data.Dataset):
    """
    Wraps a PersistentDataset / CacheDataset and applies random augmentation
    on-the-fly so the cache stores clean preprocessed volumes, not augmented ones.
    """

    def __init__(self, base: torch.utils.data.Dataset, aug_transform: Compose) -> None:
        self.base = base
        self.aug  = aug_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        # Returns a list of patch dicts when RandCropByPosNegLabeld num_samples > 1
        return self.aug(self.base[idx])


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_cache_transforms(cfg: dict) -> Compose:
    """
    Deterministic preprocessing cached to disk/RAM.
    For BraTS/CSV/JSON: loads 4 separate modality NIfTI files and stacks them.
    For decathlon: image is already a 4-channel NIfTI (no stacking needed).
    """
    fmt          = cfg.get("dataset_format", "brats").lower()
    mod_keys     = cfg.get("modality_keys", MODALITIES)
    patch_size   = cfg["patch_size"]
    pre          = cfg.get("preprocessing", {})
    lower_pct    = pre.get("clip_lower_pct", 0.5)
    upper_pct    = pre.get("clip_upper_pct", 99.5)

    transforms: list = []

    if fmt in ("brats", "csv", "json"):
        all_keys = mod_keys + ["label"]
        transforms += [
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            EnsureTyped(keys=all_keys, dtype=torch.float32),
            # Remap BraTS label 4 → 3 (enhancing tumor) for contiguous class indices
            Lambdad(keys="label", func=_remap_brats_label),
            # Stack 4 modality tensors [1,H,W,D] → [4,H,W,D]
            ConcatItemsd(keys=mod_keys, name="image", dim=0),
            DeleteItemsd(keys=mod_keys),
        ]
    else:  # decathlon — image already 4-channel, channel order [FLAIR,T1,T1ce,T2]
        transforms += [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            # Reorder channels from MSD order [FLAIR,T1,T1ce,T2] → [T1,T1ce,T2,FLAIR]
            Lambdad(keys="image", func=lambda x: x[[1, 2, 3, 0]]),
            # Remap MSD labels to BraTS convention: 1(edema)↔2(NCR), 3→3
            Lambdad(keys="label", func=_remap_msd_label),
        ]

    # Intensity clipping (before z-score so outliers don't skew mean/std)
    if pre.get("clip_intensities", True):
        transforms.append(
            Lambdad(
                keys="image",
                func=partial(_clip_percentile, lower=lower_pct, upper=upper_pct),
            )
        )

    # Z-score normalization per channel over non-zero (brain) voxels
    transforms.append(
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    )

    # Optional resampling (disabled for BraTS which is already 1 mm isotropic)
    if pre.get("resample", False):
        spacing = pre.get("target_spacing", [1.0, 1.0, 1.0])
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            )
        )

    # Crop to tight brain bounding box (removes wasted air/background voxels)
    if pre.get("crop_foreground", True):
        transforms.append(
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            )
        )

    # Pad to minimum patch size so RandCropByPosNegLabeld always has enough volume
    transforms.append(
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant")
    )

    # Store image as float16 to halve cached file size (~40-50 MB vs ~80-100 MB per case),
    # reducing disk read time that causes GPU oscillation in epochs 2+.
    # Label stays float32 (small; needed for loss computation).
    transforms.append(EnsureTyped(keys="image",  dtype=torch.float16, track_meta=False))
    transforms.append(EnsureTyped(keys="label", dtype=torch.float32, track_meta=False))

    return Compose(transforms)


def _build_aug_transforms(cfg: dict) -> Compose:
    """Random augmentation applied live after loading from cache."""
    patch_size = cfg["patch_size"]
    num_samples = cfg.get("num_samples", 4)
    pos_neg     = cfg.get("pos_neg_ratio", 1)
    aug         = cfg.get("augmentation", {})

    return Compose([
        # Restore float32 before aug (cache stores image as float16 to cut I/O)
        EnsureTyped(keys="image", dtype=torch.float32),
        # Biased patch sampling: pos_neg ratio ensures ≥50% patches contain tumor
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos_neg,
            neg=1,
            num_samples=num_samples,
        ),
        RandFlipd(keys=["image", "label"], prob=aug.get("flip_prob", 0.5), spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=aug.get("flip_prob", 0.5), spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=aug.get("flip_prob", 0.5), spatial_axis=2),
        RandRotate90d(
            keys=["image", "label"],
            prob=aug.get("rotate90_prob", 0.5),
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandGaussianNoised(
            keys="image",
            prob=aug.get("gaussian_noise_prob", 0.2),
            std=aug.get("gaussian_noise_std", 0.05),
        ),
        RandScaleIntensityd(
            keys="image",
            factors=aug.get("intensity_scale_factor", 0.1),
            prob=aug.get("intensity_scale_prob", 0.5),
        ),
        RandShiftIntensityd(
            keys="image",
            offsets=aug.get("intensity_shift_offset", 0.1),
            prob=aug.get("intensity_shift_prob", 0.5),
        ),
        # Guarantee plain torch.Tensor before list_data_collate — some MONAI
        # aug transforms return numpy when their probability check doesn't fire,
        # causing a type mismatch in the collate function.
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=False),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_files: list[dict],
    val_files:   list[dict],
    test_files:  list[dict],
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    cache_mode options:
      persistent  Disk cache via PersistentDataset (default; zero RAM pressure).
      ram         In-RAM cache via CacheDataset (fast but needs ~30-50 GB RAM).
                  NOTE: on Windows, RAM cache is NOT shared across workers —
                  use persistent or none when num_workers > 0.
      none        No caching; recomputes transforms every epoch.
    """
    cache_mode      = cfg.get("cache_mode", "persistent").lower()
    cache_dir       = Path(cfg.get("cache_dir", "data/cache"))
    batch_size      = cfg.get("batch_size", 1)
    num_workers     = cfg.get("num_workers", 4)
    pin_mem         = cfg.get("pin_memory", True)
    persist_workers = cfg.get("persistent_workers", False) and num_workers > 0

    cache_tf = _build_cache_transforms(cfg)
    aug_tf   = _build_aug_transforms(cfg)

    if cache_mode == "persistent":
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_base = PersistentDataset(
            train_files, transform=cache_tf, cache_dir=str(cache_dir / "train")
        )
        val_ds  = PersistentDataset(
            val_files,  transform=cache_tf, cache_dir=str(cache_dir / "val")
        )
        test_ds = PersistentDataset(
            test_files, transform=cache_tf, cache_dir=str(cache_dir / "test")
        )
        train_ds = _AugDataset(train_base, aug_tf)

    elif cache_mode == "ram":
        train_base = CacheDataset(train_files, transform=cache_tf, num_workers=num_workers)
        val_ds     = CacheDataset(val_files,   transform=cache_tf, num_workers=num_workers)
        test_ds    = CacheDataset(test_files,  transform=cache_tf, num_workers=num_workers)
        train_ds   = _AugDataset(train_base, aug_tf)

    else:  # none
        all_train_tf = Compose([*cache_tf.transforms, *aug_tf.transforms])
        train_ds = Dataset(train_files, transform=all_train_tf)
        val_ds   = Dataset(val_files,   transform=cache_tf)
        test_ds  = Dataset(test_files,  transform=cache_tf)

    prefetch = cfg.get("prefetch_factor", 4) if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist_workers,
        prefetch_factor=prefetch,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=min(num_workers, 4),
        pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=min(num_workers, 4),
        pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )

    return train_loader, val_loader, test_loader


def warm_cache(train_ds, num_workers: int = 8) -> None:
    """
    Pre-build the PersistentDataset cache before training starts.
    Separates the slow cache-build phase (CPU-bound) from GPU training so the
    GPU is never starved waiting for uncached items during epoch 1.
    """
    if not isinstance(train_ds, _AugDataset) or not isinstance(train_ds.base, PersistentDataset):
        return
    from tqdm import tqdm
    base = train_ds.base
    n = len(base)
    print(f"Pre-building cache for {n} training cases (runs once, skipped if already cached)...")
    loader = DataLoader(
        base,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    for _ in tqdm(loader, total=n, desc="Caching", ncols=80):
        pass
    print("Cache ready.\n")
