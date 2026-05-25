"""
Offline preprocessing pipeline for brain MRI segmentation.

Runs the expensive, one-time operations on raw NIfTI files and saves the
results to disk.  Subsequent training/evaluation reads the preprocessed
files, skipping this cost.

Supported input formats
-----------------------
BraTS directory  (--data-dir)
    data_dir/
        CaseXXX/
            CaseXXX_t1.nii.gz  CaseXXX_t1ce.nii.gz  ...  CaseXXX_seg.nii.gz

CSV manifest  (--manifest + --modality-keys)
    t1,t1ce,t2,flair,label
    /abs/path/case1_t1.nii.gz,...

JSON manifest  (--manifest, no --modality-keys needed)
    [{"t1": "...", "label": "..."}, ...]

Preprocessing steps (all optional, off by default except clipping)
-------------------------------------------------------------------
--resample           Resample every volume to --target-spacing (mm)
--bias-correct       N4ITK bias field correction (SimpleITK)
--coregister         Rigidly register all modalities to the first modality key
--skull-strip        Skull strip via HD-BET (requires:  pip install hd-bet)
--no-clip            Disable default percentile intensity clipping

Output
------
Preprocessed NIfTI files are written to --out-dir/<case_id>/.
A CSV manifest (<out-dir>/manifest.csv) is written pointing to the new files,
ready to use with  dataset_format: csv  in a config YAML.

Usage examples
--------------
# BraTS, resample + bias correct
python preprocess.py \\
    --data-dir data/raw/BraTS2021_Training_Data \\
    --out-dir  data/preprocessed \\
    --resample --target-spacing 1.0 1.0 1.0 \\
    --bias-correct

# Custom CSV dataset with 2 modalities, all steps
python preprocess.py \\
    --manifest  data/my_dataset.csv \\
    --modality-keys t1 flair \\
    --out-dir   data/preprocessed \\
    --resample --bias-correct --coregister

# Preview what would happen without writing files
python preprocess.py --data-dir data/raw --out-dir /tmp/test --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.data.preprocessing import (
    MODALITIES,
    bias_field_correction,
    build_data_list,
    build_data_list_from_csv,
    build_data_list_from_json,
    clip_intensities,
    coregister_to_reference,
    load_sitk,
    resample_volume,
    remap_labels,
    save_sitk,
    skull_strip_hdbet,
    zscore_normalize,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline preprocessing pipeline for brain MRI NIfTI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", metavar="DIR",
                     help="BraTS-style input directory")
    src.add_argument("--manifest", metavar="FILE",
                     help="CSV or JSON manifest of input files")

    p.add_argument("--modality-keys", nargs="+", default=None, metavar="KEY",
                   help="Modality column names when using a CSV manifest "
                        "(e.g. --modality-keys t1 flair).  Inferred from JSON.")
    p.add_argument("--label-col", default="label",
                   help="CSV column name for the segmentation file (default: label)")

    # Output
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Root directory for preprocessed output files")

    # Preprocessing steps
    p.add_argument("--resample", action="store_true",
                   help="Resample to --target-spacing")
    p.add_argument("--target-spacing", nargs=3, type=float,
                   default=[1.0, 1.0, 1.0], metavar=("X", "Y", "Z"),
                   help="Target voxel spacing in mm (default: 1.0 1.0 1.0)")
    p.add_argument("--bias-correct", action="store_true",
                   help="N4ITK bias field correction")
    p.add_argument("--bias-iterations", type=int, default=50,
                   help="N4ITK optimizer iterations per level (default: 50)")
    p.add_argument("--bias-levels", type=int, default=4,
                   help="N4ITK fitting levels (default: 4)")
    p.add_argument("--coregister", action="store_true",
                   help="Rigidly register all modalities to the first modality key")
    p.add_argument("--skull-strip", action="store_true",
                   help="Skull strip with HD-BET (requires: pip install hd-bet)")
    p.add_argument("--no-clip", action="store_true",
                   help="Disable percentile intensity clipping (clipping is ON by default)")
    p.add_argument("--clip-lower", type=float, default=0.5,
                   help="Lower clipping percentile (default: 0.5)")
    p.add_argument("--clip-upper", type=float, default=99.5,
                   help="Upper clipping percentile (default: 99.5)")
    p.add_argument("--normalize", action="store_true",
                   help="Also save z-score normalised volumes (adds _norm suffix)")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing any files")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes (default: 1)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(
    entry: dict,
    modality_keys: list[str],
    case_out_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> dict | None:
    """
    Process one case (all modalities + optional label).

    Returns a dict of output paths (same keys as entry), or None on failure.
    """
    case_out_dir.mkdir(parents=True, exist_ok=True)
    out_entry: dict = {}

    # Determine reference path for co-registration (first modality key)
    ref_key = modality_keys[0]

    for key in modality_keys:
        src_path = Path(entry[key])
        dst_path = case_out_dir / src_path.name

        if dry_run:
            print(f"  [dry-run] would write {dst_path}")
            out_entry[key] = str(dst_path)
            continue

        try:
            sitk_img = load_sitk(src_path)

            # ── Co-registration (before resampling) ─────────────────────────
            if args.coregister and key != ref_key:
                ref_path = Path(entry[ref_key])
                coregister_to_reference(src_path, ref_path, dst_path)
                sitk_img = load_sitk(dst_path)

            # ── Bias field correction ────────────────────────────────────────
            if args.bias_correct:
                sitk_img = bias_field_correction(
                    sitk_img,
                    n_fitting_levels=args.bias_levels,
                    n_iterations=args.bias_iterations,
                )

            # ── Resampling ───────────────────────────────────────────────────
            if args.resample:
                import SimpleITK as sitk
                sitk_img = resample_volume(
                    sitk_img,
                    target_spacing=args.target_spacing,
                    interpolator=sitk.sitkLinear,
                )

            # ── Save intermediate NIfTI ──────────────────────────────────────
            if args.bias_correct or args.resample or (args.coregister and key != ref_key):
                save_sitk(sitk_img, dst_path)
            else:
                # No SimpleITK ops on this modality; still copy to out_dir
                import shutil
                shutil.copy2(src_path, dst_path)

            # ── Skull stripping ──────────────────────────────────────────────
            if args.skull_strip:
                stripped_path = case_out_dir / src_path.name.replace(".nii.gz", "_stripped.nii.gz")
                skull_strip_hdbet(dst_path, stripped_path)
                dst_path = stripped_path  # use stripped version going forward

            # ── Numpy-level ops (clipping, optional normalisation) ───────────
            if not args.no_clip or args.normalize:
                vol = nib.load(str(dst_path)).get_fdata(dtype=np.float32)
                affine = nib.load(str(dst_path)).affine

                if not args.no_clip:
                    vol = clip_intensities(vol, args.clip_lower, args.clip_upper)

                if args.normalize:
                    norm_vol = zscore_normalize(vol)
                    norm_path = case_out_dir / src_path.name.replace(".nii.gz", "_norm.nii.gz")
                    nib.save(nib.Nifti1Image(norm_vol, affine), str(norm_path))

                if not args.no_clip:
                    nib.save(nib.Nifti1Image(vol, affine), str(dst_path))

        except Exception as exc:
            print(f"  ERROR processing {key} for {src_path.parent.name}: {exc}")
            return None

        out_entry[key] = str(dst_path)

    # ── Label (copy / resample with nearest-neighbour) ───────────────────────
    if "label" in entry:
        lbl_src = Path(entry["label"])
        lbl_dst = case_out_dir / lbl_src.name

        if dry_run:
            print(f"  [dry-run] would write {lbl_dst}")
            out_entry["label"] = str(lbl_dst)
        else:
            try:
                if args.resample:
                    import SimpleITK as sitk
                    lbl_img = load_sitk(lbl_src)
                    lbl_img = resample_volume(
                        lbl_img,
                        target_spacing=args.target_spacing,
                        interpolator=sitk.sitkNearestNeighbor,
                        default_value=0.0,
                    )
                    save_sitk(lbl_img, lbl_dst)
                else:
                    import shutil
                    shutil.copy2(lbl_src, lbl_dst)
                out_entry["label"] = str(lbl_dst)
            except Exception as exc:
                print(f"  ERROR processing label for {lbl_src.parent.name}: {exc}")
                return None

    return out_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # ── Resolve input data list ──────────────────────────────────────────────
    if args.data_dir:
        print(f"Scanning BraTS directory: {args.data_dir}")
        data_list = build_data_list(args.data_dir)
        modality_keys = MODALITIES
    else:
        manifest_path = Path(args.manifest)
        if manifest_path.suffix == ".json":
            print(f"Loading JSON manifest: {manifest_path}")
            data_list = build_data_list_from_json(manifest_path)
            # Infer modality keys from first entry
            first = data_list[0] if data_list else {}
            modality_keys = [k for k in first if k != "label"]
        else:
            keys = args.modality_keys
            if not keys:
                print(
                    "ERROR: --modality-keys is required when loading a CSV manifest.\n"
                    "Example: --modality-keys t1 t1ce t2 flair",
                    file=sys.stderr,
                )
                sys.exit(1)
            modality_keys = keys
            print(f"Loading CSV manifest: {manifest_path}")
            data_list = build_data_list_from_csv(
                manifest_path, modality_keys, label_col=args.label_col
            )

    if not data_list:
        print("No valid cases found. Exiting.")
        sys.exit(1)

    print(f"Found {len(data_list)} cases  |  modalities: {modality_keys}")

    # ── Print preprocessing plan ─────────────────────────────────────────────
    steps = []
    if args.coregister:
        steps.append(f"co-register all modalities → {modality_keys[0]}")
    if args.bias_correct:
        steps.append(f"N4ITK bias correction  ({args.bias_levels} levels × {args.bias_iterations} iters)")
    if args.resample:
        steps.append(f"resample → {args.target_spacing} mm")
    if args.skull_strip:
        steps.append("skull strip (HD-BET)")
    if not args.no_clip:
        steps.append(f"clip intensities  [{args.clip_lower}th, {args.clip_upper}th pct]")
    if args.normalize:
        steps.append("z-score normalize (saved as *_norm.nii.gz)")

    if steps:
        print("\nPreprocessing steps:")
        for s in steps:
            print(f"  • {s}")
    else:
        print("\nNo preprocessing steps selected (files will be copied only).")

    if args.dry_run:
        print("\n[DRY RUN — no files will be written]\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process cases ────────────────────────────────────────────────────────
    manifest_rows: list[dict] = []
    failed = 0

    # Known modality subfolder names — if the immediate parent of a file has
    # one of these names, the case directory is one level higher.
    _MODALITY_DIRS = {"t1", "t1ce", "t2", "flair", "t2flair", "label", "seg", "dwi", "adc"}

    for entry in tqdm(data_list, desc="Preprocessing", unit="case"):
        # Derive a case ID from the first modality path.
        # When files live inside a per-modality subfolder
        # (e.g. …/UPENN-GBM-00020/t1/file.nii.gz), parent.name is the
        # modality folder name — step up one more level to get the case dir.
        first_path = Path(entry[modality_keys[0]])
        parent = first_path.parent
        if parent.name.lower() in _MODALITY_DIRS:
            case_id = parent.parent.name
        else:
            case_id = parent.name
        if not case_id:
            case_id = first_path.stem.split("_")[0]
        case_out = out_dir / case_id

        result = process_case(entry, modality_keys, case_out, args, dry_run=args.dry_run)
        if result is None:
            failed += 1
        else:
            manifest_rows.append(result)

    # ── Write manifest CSV ───────────────────────────────────────────────────
    if not args.dry_run and manifest_rows:
        manifest_out = out_dir / "manifest.csv"
        fieldnames = modality_keys + (["label"] if "label" in manifest_rows[0] else [])
        with open(manifest_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nManifest written → {manifest_out}")
        print("Use in config:")
        print(f"  dataset_format: csv")
        print(f"  manifest: {manifest_out}")
        print(f"  modality_keys: {modality_keys}")
        print(f"  in_channels: {len(modality_keys)}")

    print(f"\nDone.  Processed: {len(manifest_rows)}  |  Failed: {failed}")


if __name__ == "__main__":
    main()
