"""
Merge and validate multiple CSV manifests into a single training manifest.

Reads one or more CSV manifests, validates that all referenced files exist,
deduplicates rows by file path, and writes a combined manifest ready for use
with  dataset_format: csv  in any config YAML.

Usage
-----
  # Merge BraTS 2021 + BraTS 2023 + TCGA
  python scripts/merge_manifests.py \\
      data/raw/BraTS2021_Training_Data \\
      data/preprocessed/BraTS2023/manifest.csv \\
      data/preprocessed/tcga_gbm/manifest.csv \\
      --out data/combined_manifest.csv \\
      --modality-keys t1 t1ce t2 flair

  # Also accept BraTS-format directories directly (no manifest needed)
  # Mix directories and CSV files freely.

  # Validate an existing manifest without merging
  python scripts/merge_manifests.py \\
      data/combined_manifest.csv \\
      --validate-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge and validate multiple dataset manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", metavar="PATH",
                   help="One or more CSV manifest files or BraTS-style directories")
    p.add_argument("--out", default="data/combined_manifest.csv",
                   help="Output CSV path (default: data/combined_manifest.csv)")
    p.add_argument("--modality-keys", nargs="+",
                   default=["t1", "t1ce", "t2", "flair"],
                   help="Expected modality columns (default: t1 t1ce t2 flair)")
    p.add_argument("--label-col", default="label",
                   help="Segmentation column name (default: label)")
    p.add_argument("--require-label", action="store_true",
                   help="Exclude cases without a label (segmentation) file")
    p.add_argument("--require-all-modalities", action="store_true",
                   help="Exclude cases with any missing modality file")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate inputs; do not write output")
    return p.parse_args()


def load_brats_dir(data_dir: Path, modality_keys: list[str]) -> list[dict]:
    """Scan a BraTS-style directory and return path dicts."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.preprocessing import build_data_list
    cases = build_data_list(data_dir)
    # Remap modality keys if the directory uses default BraTS keys
    # (build_data_list always returns t1/t1ce/t2/flair/label)
    return cases


def load_csv(csv_path: Path) -> list[dict]:
    """Load a CSV manifest into a list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})
    return rows


def validate_row(
    row: dict,
    modality_keys: list[str],
    label_col: str,
    require_label: bool,
    require_all: bool,
    row_num: int,
    source: str,
) -> tuple[bool, list[str]]:
    """
    Check a single row for missing / non-existent files.

    Returns (is_valid, list_of_warnings).
    """
    warnings = []
    valid = True

    for key in modality_keys:
        val = row.get(key, "")
        if not val:
            msg = f"  [{source}] row {row_num}: missing column '{key}'"
            if require_all:
                warnings.append(msg + " — EXCLUDED")
                valid = False
            else:
                warnings.append(msg + " — blank")
        elif not Path(val).exists():
            msg = f"  [{source}] row {row_num}: file not found: {val}"
            if require_all:
                warnings.append(msg + " — EXCLUDED")
                valid = False
            else:
                warnings.append(msg + " — WARNING")

    lbl = row.get(label_col, "")
    if require_label and not lbl:
        warnings.append(f"  [{source}] row {row_num}: no label — EXCLUDED")
        valid = False
    elif lbl and not Path(lbl).exists():
        warnings.append(f"  [{source}] row {row_num}: label not found: {lbl} — WARNING")

    return valid, warnings


def main() -> None:
    args = parse_args()
    modality_keys = args.modality_keys
    label_col = args.label_col

    all_rows: list[dict] = []
    seen_keys: set[str] = set()
    total_loaded = 0
    total_dupes  = 0
    total_invalid = 0

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"ERROR: path not found: {p}", file=sys.stderr)
            sys.exit(1)

        if p.is_dir():
            print(f"Scanning BraTS directory: {p}")
            rows = load_brats_dir(p, modality_keys)
            source = str(p)
        else:
            print(f"Loading CSV manifest:     {p}")
            rows = load_csv(p)
            source = p.name

        print(f"  {len(rows)} rows loaded")
        total_loaded += len(rows)

        for i, row in enumerate(rows, 1):
            # Deduplication key: first modality path
            first_mod = row.get(modality_keys[0], "")
            dedup_key = first_mod or str(row)

            if dedup_key in seen_keys:
                total_dupes += 1
                continue
            seen_keys.add(dedup_key)

            valid, warnings = validate_row(
                row, modality_keys, label_col,
                args.require_label, args.require_all_modalities,
                i, source,
            )
            for w in warnings:
                print(w)

            if valid:
                # Ensure row has all expected columns (fill missing with empty string)
                normalised = {k: row.get(k, "") for k in modality_keys}
                if label_col in row:
                    normalised[label_col] = row[label_col]
                all_rows.append(normalised)
            else:
                total_invalid += 1

    print(f"\nSummary")
    print(f"  Total loaded : {total_loaded}")
    print(f"  Duplicates   : {total_dupes}")
    print(f"  Invalid/excl : {total_invalid}")
    print(f"  Final cases  : {len(all_rows)}")

    if args.validate_only:
        print("\n[validate-only] No output written.")
        return

    if not all_rows:
        print("\nERROR: No valid cases after filtering.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_label = any(label_col in r for r in all_rows)
    fieldnames = modality_keys + ([label_col] if has_label else [])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest written → {out_path}")
    print("\nUse in config YAML:")
    print(f"  dataset_format: csv")
    print(f"  manifest: {out_path}")
    print(f"  modality_keys: {modality_keys}")
    print(f"  in_channels: {len(modality_keys)}")


if __name__ == "__main__":
    main()
