"""
Download Medical Segmentation Decathlon Task01_BrainTumour.

484 cases with T1, T1ce, T2, FLAIR + segmentation labels.
No account required — hosted on MONAI's public S3 bucket.

Label convention (differs from BraTS 2021):
  0 = background
  1 = edema
  2 = non-enhancing tumor
  3 = enhancing tumor

Usage
-----
  python scripts/download_decathlon.py --out-dir data/raw/decathlon
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MSD Task01 BrainTumour")
    p.add_argument("--out-dir", default="data/raw/decathlon",
                   help="Root directory for download (default: data/raw/decathlon)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from monai.apps import DecathlonDataset

    print("Downloading MSD Task01_BrainTumour via MONAI (~1.7 GB)...")
    print(f"Output: {out_dir.resolve()}\n")

    # MONAI downloads, extracts, and verifies MD5 automatically.
    # section="training" fetches all 484 labelled training cases.
    DecathlonDataset(
        root_dir=str(out_dir),
        task="Task01_BrainTumour",
        section="training",
        download=True,
        val_frac=0.0,  # splitting is handled by our own pipeline
    )

    task_dir = out_dir / "Task01_BrainTumour"
    images = list((task_dir / "imagesTr").glob("*.nii.gz")) if task_dir.exists() else []
    labels = list((task_dir / "labelsTr").glob("*.nii.gz")) if task_dir.exists() else []
    print(f"\nDownload complete.")
    print(f"  Images : {len(images)}")
    print(f"  Labels : {len(labels)}")
    print(f"\nNOTE: MSD label convention differs from BraTS 2021:")
    print(f"  MSD:   1=edema  2=non-enhancing  3=enhancing")
    print(f"  BraTS: 1=NCR    2=edema          3=enhancing")
    print(f"Set dataset_format: decathlon in your config to use this data.")


if __name__ == "__main__":
    main()
