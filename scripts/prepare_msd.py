"""
Prepare MSD Task01_BrainTumour for combined training with BraTS/UPenn-GBM.

MSD stores each case as a single 4-channel NIfTI with channel order:
  [FLAIR, T1, T1ce, T2]  (from dataset.json "modality" field)

MSD label convention (different from BraTS):
  0 = background
  1 = edema         → remapped to 2  (BraTS label 2)
  2 = non-enhancing → remapped to 1  (BraTS label 1 / NCR)
  3 = enhancing     → stays 3        (BraTS label 3)

This script:
  1. Splits each 4-channel image into 4 separate NIfTIs (matching BraTS modality order)
  2. Writes label NIfTIs with BraTS-convention values
  3. Produces a CSV manifest compatible with dataset_format: csv

Usage
-----
  # After running download_decathlon.py:
  python scripts/prepare_msd.py \\
      --msd-dir   data/raw/decathlon/Task01_BrainTumour \\
      --out-dir   data/preprocessed/msd \\
      --manifest  data/preprocessed/msd/manifest.csv

  # Dry-run (no files written)
  python scripts/prepare_msd.py \\
      --msd-dir data/raw/decathlon/Task01_BrainTumour \\
      --out-dir data/preprocessed/msd \\
      --manifest data/preprocessed/msd/manifest.csv \\
      --dry-run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


# MSD channel index → modality key (from dataset.json "modality" field)
_MSD_CHANNEL_TO_MOD = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}
# Output order (must match modality_keys in config)
_OUTPUT_ORDER = ["t1", "t1ce", "t2", "flair"]
# Source channel index for each output modality
_OUTPUT_CHANNELS = [1, 2, 3, 0]


def _remap_label(arr: np.ndarray) -> np.ndarray:
    """Remap MSD labels to BraTS convention in-place on a copy."""
    out = arr.copy()
    out[arr == 1] = 2  # edema → class 2
    out[arr == 2] = 1  # non-enhancing → class 1
    # 3 (enhancing) stays 3
    return out


def process_case(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    case_id: str,
    dry_run: bool,
) -> dict | None:
    """Split one MSD case into separate modality NIfTIs + remapped label.

    Returns a manifest row dict, or None on failure.
    """
    try:
        img_nib = nib.load(str(img_path))
        lbl_nib = nib.load(str(lbl_path))
    except Exception as e:
        print(f"  ERROR loading {case_id}: {e}")
        return None

    img_data = np.asarray(img_nib.dataobj)  # (H, W, D, C)
    lbl_data = np.asarray(lbl_nib.dataobj)  # (H, W, D)

    if img_data.ndim != 4 or img_data.shape[-1] != 4:
        print(f"  SKIP {case_id}: unexpected image shape {img_data.shape} (expected H×W×D×4)")
        return None

    case_out = out_dir / case_id
    row: dict = {}

    if not dry_run:
        case_out.mkdir(parents=True, exist_ok=True)

    for mod, ch_idx in zip(_OUTPUT_ORDER, _OUTPUT_CHANNELS):
        vol = img_data[..., ch_idx]
        out_path = case_out / f"{mod}.nii.gz"
        row[mod] = str(out_path)
        if not dry_run:
            nib.save(nib.Nifti1Image(vol.astype(np.float32), img_nib.affine), str(out_path))

    remapped = _remap_label(lbl_data.astype(np.uint8))
    lbl_out  = case_out / "seg.nii.gz"
    row["label"] = str(lbl_out)
    if not dry_run:
        nib.save(nib.Nifti1Image(remapped, lbl_nib.affine), str(lbl_out))

    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare MSD Task01 for combined training with BraTS/UPenn-GBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--msd-dir", required=True, metavar="DIR",
                   help="MSD Task01_BrainTumour directory (contains imagesTr/, labelsTr/)")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Output directory for per-case split NIfTIs")
    p.add_argument("--manifest", required=True, metavar="FILE",
                   help="Output CSV manifest path")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing any files")
    args = p.parse_args()

    msd_dir  = Path(args.msd_dir)
    out_dir  = Path(args.out_dir)
    img_dir  = msd_dir / "imagesTr"
    lbl_dir  = msd_dir / "labelsTr"

    if not img_dir.is_dir():
        raise SystemExit(f"ERROR: {img_dir} not found — run download_decathlon.py first")
    if not lbl_dir.is_dir():
        raise SystemExit(f"ERROR: {lbl_dir} not found")

    img_files = sorted(img_dir.glob("*.nii.gz"))
    if not img_files:
        raise SystemExit(f"ERROR: no .nii.gz files found in {img_dir}")

    print(f"Found {len(img_files)} MSD cases")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Manifest   : {args.manifest}\n")

    if args.dry_run:
        print("[DRY RUN — no files will be written]\n")

    rows: list[dict] = []
    errors = 0

    for img_path in img_files:
        case_id  = img_path.stem.replace(".nii", "")  # e.g. "BRATS_001"
        lbl_path = lbl_dir / img_path.name

        if not lbl_path.exists():
            print(f"  SKIP {case_id}: no matching label file")
            errors += 1
            continue

        if args.dry_run:
            print(f"  {case_id}  {img_path.name}")
            continue

        row = process_case(img_path, lbl_path, out_dir, case_id, dry_run=False)
        if row is not None:
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"  Processed {len(rows)}/{len(img_files)} ...")
        else:
            errors += 1

    if args.dry_run:
        print(f"\n{len(img_files)} cases would be processed.")
        return

    print(f"\nDone — {len(rows)} cases prepared, {errors} errors")

    if not rows:
        raise SystemExit("No cases were prepared successfully.")

    out_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _OUTPUT_ORDER + ["label"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written → {out_path}")
    print("\nMerge with BraTS/UPenn-GBM:")
    print("  python scripts/merge_manifests.py \\")
    print("      data/raw/BraTS2021_Training_Data \\")
    print(f"      {out_path} \\")
    print("      --out data/combined_manifest.csv")


if __name__ == "__main__":
    main()
