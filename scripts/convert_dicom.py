"""
Convert a DICOM directory tree to NIfTI and produce a CSV manifest.

Handles both flat structures and the IDC 4-level layout:
  root/collection/PatientID/StudyUID/Modality_SeriesUID/*.dcm

MR series are converted with dcm2niix.
DICOM SEG objects are converted with highdicom (pip install highdicom).

After conversion, pass the manifest to preprocess.py:
  python preprocess.py \\
      --manifest data/converted/manifest.csv \\
      --modality-keys t1 t1ce t2 flair \\
      --out-dir data/preprocessed \\
      --resample --bias-correct

Requirements
------------
  dcm2niix    conda install -n imgp -c conda-forge dcm2niix
  pydicom     pip install pydicom
  highdicom   pip install highdicom   (for DICOM SEG → NIfTI)

Usage
-----
  python scripts/convert_dicom.py \\
      --dicom-dir data/raw/upenn_gbm \\
      --out-dir   data/converted/upenn_gbm

  # Preview without converting
  python scripts/convert_dicom.py \\
      --dicom-dir data/raw/upenn_gbm \\
      --out-dir   data/converted/upenn_gbm \\
      --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Default keyword → modality map (case-insensitive substring match).
# Order matters — checked top to bottom, first match wins.
# Rules:  t1ce before t1  (post-contrast descriptions contain "t1")
#         flair before t2  (flair descriptions often start with "t2_flair_...")
DEFAULT_MODALITY_MAP: dict[str, list[str]] = {
    "t1ce":  ["t1c", "t1+c", "t1ce", "t1gd", "t1 gd", "post", "contrast",
               "t1_ce", "t1-ce", "gad", "gadolinium", "enhance",
               "stealth-post", "stealth_post"],
    "flair": ["t2_flair_axial: processed_captk",  # UPenn CaPTk exact
               "flair", "fl_ir", "t2-flair", "t2_flair"],
    "t1":    ["t1 axial: processed_captk",         # UPenn CaPTk exact
               "t1w", "t1_w", "t1 w", "mprage", "spgr", "t1"],
    "t2":    ["axial t2 tse: processed_captk",     # UPenn CaPTk exact
               "t2w", "t2_w", "t2 w", "t2"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert DICOM tree to NIfTI with CSV manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dicom-dir", required=True,
                   help="Root DICOM directory")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for NIfTI files and manifest")
    p.add_argument("--modality-map", nargs="+", metavar="LABEL=KEYWORD",
                   help="Extra modality keyword mappings (e.g. t1=MPRAGE)")
    p.add_argument("--modalities", nargs="+", default=["t1", "t1ce", "t2", "flair"],
                   help="Expected modality labels per case (default: t1 t1ce t2 flair)")
    p.add_argument("--require-all", action="store_true",
                   help="Only write cases that have all expected modalities")
    p.add_argument("--dcm2niix-path", default="dcm2niix",
                   help="Path to dcm2niix binary")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and identify series without converting")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DICOM metadata helper
# ---------------------------------------------------------------------------

def _read_series_description(series_dir: Path) -> tuple[str, str]:
    """
    Read (SeriesDescription, Modality) from the first DICOM file in series_dir.
    Returns ("", "") on failure.
    """
    try:
        import pydicom
    except ImportError:
        return "", ""

    for dcm_path in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            desc = str(getattr(ds, "SeriesDescription", "") or
                       getattr(ds, "ProtocolName", "") or "")
            mod  = str(getattr(ds, "Modality", "") or "")
            return desc, mod
        except Exception:
            continue
    return "", ""


def _guess_modality(description: str, modality_map: dict[str, list[str]]) -> str | None:
    desc_lower = description.lower()
    for label, keywords in modality_map.items():
        for kw in keywords:
            if kw.lower() in desc_lower:
                return label
    return None


# ---------------------------------------------------------------------------
# Directory tree traversal — handles both flat and IDC 4-level layouts
# ---------------------------------------------------------------------------

def _collect_patients(dicom_root: Path) -> dict[str, list[Path]]:
    """
    Walk the DICOM tree and group leaf series directories by patient ID.

    Supports two layouts:
      Flat:  root/PatientID/series/*.dcm          → patient = parts[0]
      IDC:   root/collection/PatientID/study/series/*.dcm → patient = parts[1]
    """
    patients: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()

    for dcm in dicom_root.rglob("*.dcm"):
        series_dir = dcm.parent
        if series_dir in seen:
            continue
        seen.add(series_dir)

        relative = series_dir.relative_to(dicom_root)
        parts = relative.parts

        # depth ≥ 4: IDC layout (collection / patient / study / series)
        if len(parts) >= 4:
            patient_id = parts[1]
        elif len(parts) >= 2:
            patient_id = parts[0]
        else:
            patient_id = series_dir.name

        patients[patient_id].append(series_dir)

    return dict(patients)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def check_dcm2niix(path: str) -> None:
    if not shutil.which(path):
        print(
            f"ERROR: dcm2niix not found ('{path}').\n"
            "Install:  conda install -n medimgp -c conda-forge dcm2niix",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Windows long-path junction helper
#
# dcm2niix uses Win32 APIs with MAX_PATH (260). When the source DICOM path
# exceeds this limit (common with IDC UUID-deep layouts), dcm2niix reports
# "Unable to find any DICOM images". We work around this by creating a
# temporary NTFS junction at a short path (C:\dcmtmp) that points to the
# real root, making all sub-paths visible to dcm2niix.
# ---------------------------------------------------------------------------

_JUNCTION_PATH: Path | None = None  # module-level so cleanup can find it


def _setup_windows_junction(dicom_root: Path) -> Path:
    """
    Create a short-path junction for dicom_root on Windows.
    Returns the junction path (or dicom_root unchanged if not needed).
    """
    global _JUNCTION_PATH
    if sys.platform != "win32":
        return dicom_root

    # Resolve to absolute path so the length check works correctly.
    dicom_abs = dicom_root.resolve()

    # Only create the junction if DICOM file paths would exceed MAX_PATH.
    sample_files = list(dicom_abs.rglob("*.dcm"))[:1]
    if not sample_files or len(str(sample_files[0])) <= 260:
        return dicom_abs  # Return absolute path regardless

    junc = Path("C:/dcmtmp")
    _remove_windows_junction(junc)
    result = subprocess.run(
        ["cmd", "/c", f"mklink /J {junc} {dicom_abs}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not junc.exists():
        print(f"WARNING: could not create junction at {junc}: {result.stderr.strip()}")
        return dicom_abs

    _JUNCTION_PATH = junc
    print(f"Created junction {junc} → {dicom_abs} (path-length workaround)")
    return junc


def _remove_windows_junction(junc: Path) -> None:
    if sys.platform == "win32" and (junc.exists() or junc.is_symlink()):
        subprocess.run(["cmd", "/c", f"rmdir /s /q {junc}"], capture_output=True)


def _remap_via_junction(original: Path, original_root: Path, junction: Path) -> Path:
    """Replace original_root prefix with the junction path."""
    if original_root == junction:
        return original
    try:
        rel = original.relative_to(original_root)
        return junction / rel
    except ValueError:
        return original


def convert_mr_series(series_dir: Path, out_dir: Path, dcm2niix: str) -> list[Path]:
    """Convert an MR DICOM series to NIfTI using dcm2niix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [dcm2niix, "-z", "y", "-f", "%p_%s", "-o", str(out_dir), str(series_dir)],
        capture_output=True, text=True,
    )
    return list(out_dir.glob("*.nii.gz"))


def convert_seg_series(series_dir: Path, out_dir: Path, label_name: str = "seg") -> Path | None:
    """
    Convert a DICOM SEG object to NIfTI using highdicom + nibabel.
    Returns the output NIfTI path, or None on failure.
    """
    try:
        import highdicom as hd
        import nibabel as nib
        import numpy as np
    except ImportError:
        return None

    seg_files = sorted(series_dir.glob("*.dcm"))
    if not seg_files:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label_name}.nii.gz"

    try:
        seg = hd.seg.segread(str(seg_files[0]))
        # Get pixel array: shape (frames, rows, cols)
        pixel_array = seg.pixel_array

        # Build a single label volume by taking argmax over segments
        # Each segment is a binary mask for one class
        if pixel_array.ndim == 4:
            # (segments, frames, rows, cols) or (frames, rows, cols, segments)
            # Try to detect shape by looking at number of segment descriptions
            n_segs = len(seg.SegmentSequence)
            if pixel_array.shape[0] == n_segs:
                label_vol = np.zeros(pixel_array.shape[1:], dtype=np.uint8)
                for seg_idx in range(n_segs):
                    seg_num = int(seg.SegmentSequence[seg_idx].SegmentNumber)
                    label_vol[pixel_array[seg_idx] > 0] = seg_num
            else:
                label_vol = np.argmax(pixel_array, axis=-1).astype(np.uint8)
        elif pixel_array.ndim == 3:
            label_vol = pixel_array.astype(np.uint8)
        else:
            return None

        # Try to get affine from the image position / orientation
        try:
            affine = np.eye(4)
            if hasattr(seg, "SharedFunctionalGroupsSequence"):
                pms = seg.SharedFunctionalGroupsSequence[0]
                if hasattr(pms, "PlaneOrientationSequence"):
                    iop = list(map(float, pms.PlaneOrientationSequence[0].ImageOrientationPatient))
                    F = np.array(iop).reshape(2, 3).T
                    n = np.cross(F[:, 0], F[:, 1])
                    affine[:3, :3] = np.column_stack([F, n])
        except Exception:
            pass

        nib.save(nib.Nifti1Image(label_vol, affine), str(out_path))
        return out_path

    except Exception as e:
        print(f"      WARNING: DICOM SEG conversion failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build modality keyword map
    modality_map = {k: list(v) for k, v in DEFAULT_MODALITY_MAP.items()}
    if args.modality_map:
        for item in args.modality_map:
            if "=" not in item:
                print(f"WARNING: skipping malformed --modality-map entry '{item}'")
                continue
            label, keyword = item.split("=", 1)
            modality_map.setdefault(label, []).insert(0, keyword)

    dicom_root_real = Path(args.dicom_dir).resolve()
    out_dir         = Path(args.out_dir)
    expected        = args.modalities

    if not dicom_root_real.exists():
        print(f"ERROR: {dicom_root_real} not found", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        check_dcm2niix(args.dcm2niix_path)

    # On Windows, create a short-path junction if DICOM paths exceed MAX_PATH.
    # dicom_root is either the junction path or the resolved absolute real path.
    dicom_root = _setup_windows_junction(dicom_root_real) if not args.dry_run else dicom_root_real

    try:
        print(f"Scanning: {dicom_root_real}")
        patients = _collect_patients(dicom_root)
        total_series = sum(len(v) for v in patients.values())
        print(f"Found {len(patients)} patients, {total_series} series\n")

        manifest_rows: list[dict] = []
        skipped = 0

        for patient_id, series_dirs in sorted(patients.items()):
            patient_out = out_dir / patient_id
            found: dict[str, str] = {}
            seg_dirs: list[Path] = []

            for series_dir in series_dirs:
                # For desc/modality detection we can still use the real path
                real_series = _remap_via_junction(series_dir, dicom_root, dicom_root_real)
                desc, dicom_mod = _read_series_description(real_series)

                if dicom_mod == "SEG":
                    seg_dirs.append(series_dir)  # keep junction path for dcm2niix
                    continue

                modality = _guess_modality(desc, modality_map)
                if args.dry_run:
                    label = modality or "?"
                    print(f"  {patient_id:25s}  {label:6s}  {desc}")
                    continue

                if modality is None:
                    continue

                if modality in found:
                    continue  # keep first occurrence

                nifti_out = patient_out / modality
                niftis = convert_mr_series(series_dir, nifti_out, args.dcm2niix_path)
                if niftis:
                    found[modality] = str(niftis[0])

            if args.dry_run:
                continue

            # Convert first available SEG series (highdicom uses Python I/O → handles long paths)
            if seg_dirs:
                real_seg = _remap_via_junction(seg_dirs[0], dicom_root, dicom_root_real)
                seg_out  = patient_out / "label"
                seg_path = convert_seg_series(real_seg, seg_out)
                if seg_path:
                    found["label"] = str(seg_path)

            missing = [m for m in expected if m not in found]
            if missing and args.require_all:
                print(f"  SKIPPED {patient_id} (missing: {missing})")
                skipped += 1
                continue

            if missing:
                print(f"  WARNING {patient_id}: missing {missing}")
            else:
                print(f"  OK      {patient_id}")

            row = {m: found.get(m, "") for m in expected}
            if "label" in found:
                row["label"] = found["label"]
            manifest_rows.append(row)

    finally:
        # Always clean up the Windows junction, even on error
        if _JUNCTION_PATH is not None:
            _remove_windows_junction(_JUNCTION_PATH)
            print(f"Removed junction {_JUNCTION_PATH}")

    if args.dry_run:
        print("\n[dry-run] No files converted.")
        return

    if not manifest_rows:
        print("No cases converted.")
        return

    manifest_path = out_dir / "manifest.csv"
    has_label  = any("label" in r for r in manifest_rows)
    fieldnames = expected + (["label"] if has_label else [])
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nConverted: {len(manifest_rows)}  |  Skipped: {skipped}")
    print(f"Manifest  → {manifest_path}")
    print("\nNext: build UPenn manifest:")
    print(f"  conda run -n medimgp python scripts/build_upenn_manifest.py \\")
    print(f"      --converted-dir {out_dir} \\")
    print(f"      --out data/preprocessed/upenn_gbm/manifest.csv")


if __name__ == "__main__":
    main()
