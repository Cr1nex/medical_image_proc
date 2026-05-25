"""
Build a CSV manifest from the UPenn-GBM converted directory structure.

The converter (scripts/convert_dicom.py) produces:
    converted/upenn_gbm/
        UPENN-GBM-XXXXX/
            t1/     *.nii.gz  (may have raw + processed versions)
            t1ce/   *.nii.gz
            t2/     *.nii.gz
            flair/  *.nii.gz
            label/  seg.nii.gz

File selection (per modality folder):
  - If any file's stem ends with 'a' (e.g. t1_axial_4a.nii.gz), prefer those.
  - Among candidates, take the one whose name contains the expected modality
    substring (t1, t1ce, t2, flair), else take the last alphabetically.
  - Cases with any missing modality or missing label are skipped with a warning.

Usage
-----
  python scripts/build_upenn_manifest.py \\
      --converted-dir data/converted/upenn_gbm \\
      --out           data/preprocessed/upenn_gbm/manifest.csv

  # Dry-run: print what would be written without touching disk
  python scripts/build_upenn_manifest.py \\
      --converted-dir data/converted/upenn_gbm \\
      --out           data/preprocessed/upenn_gbm/manifest.csv \\
      --dry-run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODALITY_KEYS = ["t1", "t1ce", "t2", "flair"]
# Substrings to prefer when multiple nii.gz files exist in a modality folder
_MODALITY_HINTS = {
    "t1":   ["t1_axial"],
    "t1ce": ["stealth", "post", "t1ce"],
    "t2":   ["t2_tse", "axial_t2", "t2_axial"],
    "flair": ["flair", "t2_flair"],
}


def pick_file(nii_files: list[Path], modality: str) -> Path | None:
    """Choose the best file from a modality folder."""
    if not nii_files:
        return None

    # Prefer files whose stem ends with 'a' (processed/alternate)
    a_files = [f for f in nii_files if f.stem.endswith("a")]
    candidates = a_files if a_files else nii_files

    # Among candidates, prefer one whose name matches a known hint
    hints = _MODALITY_HINTS.get(modality, [])
    for hint in hints:
        hinted = [f for f in candidates if hint.lower() in f.name.lower()]
        if hinted:
            return sorted(hinted)[-1]

    return sorted(candidates)[-1]


def build_manifest(converted_dir: Path) -> tuple[list[dict], list[str]]:
    """
    Scan converted_dir for UPENN-GBM cases and return (rows, warnings).
    Each row is a dict with keys t1, t1ce, t2, flair, label.
    """
    case_dirs = sorted(d for d in converted_dir.iterdir() if d.is_dir())
    rows: list[dict] = []
    warnings: list[str] = []

    for case_dir in case_dirs:
        entry: dict = {}
        skip = False

        for mod in MODALITY_KEYS:
            mod_dir = case_dir / mod
            if not mod_dir.is_dir():
                warnings.append(f"{case_dir.name}: missing modality folder '{mod}' — skipped")
                skip = True
                break

            nii_files = [f for f in mod_dir.iterdir()
                         if f.suffix == ".gz" and f.stem.endswith(".nii")]
            chosen = pick_file(nii_files, mod)
            if chosen is None:
                warnings.append(f"{case_dir.name}: no .nii.gz in '{mod}/' — skipped")
                skip = True
                break

            if len(nii_files) > 1:
                others = [f.name for f in nii_files if f != chosen]
                warnings.append(
                    f"{case_dir.name}/{mod}: picked '{chosen.name}', "
                    f"ignored {others}"
                )

            entry[mod] = str(chosen)

        if skip:
            continue

        label_path = case_dir / "label" / "seg.nii.gz"
        if not label_path.exists():
            warnings.append(f"{case_dir.name}: missing label/seg.nii.gz — skipped")
            continue

        entry["label"] = str(label_path)
        rows.append(entry)

    return rows, warnings


def main() -> None:
    p = argparse.ArgumentParser(description="Build UPenn-GBM CSV manifest from converted directory")
    p.add_argument("--converted-dir", required=True, metavar="DIR",
                   help="Root of converted cases, e.g. data/converted/upenn_gbm")
    p.add_argument("--out", required=True, metavar="FILE",
                   help="Output manifest CSV path")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without touching disk")
    args = p.parse_args()

    converted_dir = Path(args.converted_dir)
    if not converted_dir.is_dir():
        raise SystemExit(f"ERROR: directory not found: {converted_dir}")

    rows, warnings = build_manifest(converted_dir)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")

    print(f"\nValid cases: {len(rows)}")

    if args.dry_run:
        print("[DRY RUN — no files written]")
        if rows:
            print("First row preview:")
            for k, v in rows[0].items():
                print(f"  {k}: {v}")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = MODALITY_KEYS + ["label"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written → {out_path}")
    print("\nAdd to your config YAML:")
    print("  dataset_format: csv")
    print(f"  manifest: {out_path}")
    print("  modality_keys: [t1, t1ce, t2, flair]")
    print("  in_channels: 4")


if __name__ == "__main__":
    main()
