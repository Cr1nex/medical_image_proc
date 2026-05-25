"""Data discovery and train/val/test splitting."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def build_data_list_brats(
    data_dir: Path | str,
    modality_keys: list[str] = MODALITIES,
) -> list[dict]:
    """Scan a BraTS-style directory for cases with all modalities + segmentation."""
    data_dir = Path(data_dir)
    cases = []
    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        entry = {mod: str(case_dir / f"{cid}_{mod}.nii.gz") for mod in modality_keys}
        entry["label"] = str(case_dir / f"{cid}_seg.nii.gz")
        if all(Path(v).exists() for v in entry.values()):
            cases.append(entry)
    return cases


# Alias used by scripts/download_brats.py
build_data_list = build_data_list_brats


def build_data_list_csv(
    manifest: str,
    modality_keys: list[str] = MODALITIES,
) -> list[dict]:
    """Load file paths from a CSV manifest. Column names must match modality_keys."""
    cases = []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            entry = {mod: row[mod] for mod in modality_keys if mod in row}
            if "label" in row:
                entry["label"] = row["label"]
            if all(mod in entry for mod in modality_keys):
                cases.append(entry)
    return cases


def build_data_list_json(
    manifest: str,
    modality_keys: list[str] = MODALITIES,
) -> list[dict]:
    """Load file paths from a JSON manifest (list of dicts)."""
    with open(manifest) as f:
        rows = json.load(f)
    cases = []
    for row in rows:
        entry = {mod: row[mod] for mod in modality_keys if mod in row}
        if "label" in row:
            entry["label"] = row["label"]
        if all(mod in entry for mod in modality_keys):
            cases.append(entry)
    return cases


def build_data_list_decathlon(data_dir: Path | str) -> list[dict]:
    """
    Load MSD Task01_BrainTumour cases (pre-stacked 4-channel NIfTI).
    Channel order: T1ce, T1, T2, FLAIR.
    Label convention: 1=edema  2=non-enhancing  3=enhancing  (differs from BraTS).
    """
    task_dir = Path(data_dir) / "Task01_BrainTumour"
    img_dir = task_dir / "imagesTr"
    lbl_dir = task_dir / "labelsTr"
    cases = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        lbl_path = lbl_dir / img_path.name
        if lbl_path.exists():
            cases.append({
                "image": str(img_path),
                "label": str(lbl_path),
                "_format": "decathlon",
            })
    return cases


def build_data_list_auto(cfg: dict) -> list[dict]:
    """Dispatch to the correct loader based on cfg['dataset_format']."""
    fmt  = cfg.get("dataset_format", "brats").lower()
    keys = cfg.get("modality_keys", MODALITIES)

    if fmt == "brats":
        return build_data_list_brats(cfg["data_dir"], keys)
    elif fmt == "csv":
        return build_data_list_csv(cfg["manifest"], keys)
    elif fmt == "json":
        return build_data_list_json(cfg["manifest"], keys)
    elif fmt == "decathlon":
        return build_data_list_decathlon(cfg.get("data_dir", "data/raw/decathlon"))
    else:
        raise ValueError(
            f"Unknown dataset_format '{fmt}'. Choose: brats, csv, json, decathlon"
        )


def split_data(
    data_list: list,
    train_frac: float = 0.8,
    val_frac: float  = 0.1,
    seed: int        = 42,
) -> tuple[list, list, list]:
    """Reproducible random split into train / val / test."""
    items = data_list.copy()
    random.Random(seed).shuffle(items)
    n       = len(items)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        items[:n_train],
        items[n_train : n_train + n_val],
        items[n_train + n_val :],
    )
