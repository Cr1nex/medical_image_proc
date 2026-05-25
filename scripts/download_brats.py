"""
Download BraTS challenge datasets from Synapse.

Setup (one-time)
----------------
1. Create a free account at  https://www.synapse.org
2. Navigate to the BraTS dataset page and accept the data-use agreement
3. Generate a personal access token:
     synapse.org  →  your avatar  →  Account Settings  →  Personal Access Tokens
4. Store the token in the environment (never hardcode it):
     export SYNAPSE_TOKEN="your_token_here"
   Or pass it with --token.

BraTS Synapse IDs
-----------------
Find them by searching "BraTS" on synapse.org and clicking the dataset.
The ID appears in the URL as  syn<number>.

  BraTS 2021 Training:  syn27046444
  BraTS 2023 GLI:       syn51514105
  BraTS 2023 MEN:       syn51514106
  BraTS 2023 MET:       syn51514107
  BraTS 2023 PED:       syn51514108
  BraTS-Africa 2023:    syn51514109

Usage
-----
  # Set token once
  export SYNAPSE_TOKEN="eyJ..."

  # Download BraTS 2021 training data
  python scripts/download_brats.py \\
      --synapse-id syn27046444 \\
      --out-dir data/raw/BraTS2021

  # Download BraTS 2023 GLI (latest, biggest)
  python scripts/download_brats.py \\
      --synapse-id syn51514105 \\
      --out-dir data/raw/BraTS2023_GLI

  # Verify existing download without re-downloading
  python scripts/download_brats.py \\
      --synapse-id syn51514105 \\
      --out-dir data/raw/BraTS2023_GLI \\
      --check-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download BraTS data from Synapse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--synapse-id", required=True,
                   help="Synapse entity ID, e.g. syn27046444")
    p.add_argument("--out-dir", required=True,
                   help="Local directory to download into")
    p.add_argument("--token", default=None,
                   help="Synapse personal access token "
                        "(fallback: $SYNAPSE_TOKEN env var)")
    p.add_argument("--check-only", action="store_true",
                   help="Just count already-downloaded cases; do not download")
    p.add_argument("--manifest", action="store_true",
                   help="After download, auto-generate a BraTS manifest CSV")
    return p.parse_args()


def require_synapseclient():
    try:
        import synapseclient
        return synapseclient
    except ImportError:
        print(
            "synapseclient is not installed.\n"
            "Install it with:\n"
            "  conda run -n imgp pip install synapseclient",
            file=sys.stderr,
        )
        sys.exit(1)


def count_cases(data_dir: Path) -> int:
    """Count subdirectories that look like BraTS cases (contain .nii.gz files)."""
    return sum(
        1 for p in data_dir.iterdir()
        if p.is_dir() and any(p.glob("*.nii.gz"))
    )


def generate_manifest(data_dir: Path, out_csv: Path) -> None:
    """Scan a BraTS-style directory and write a CSV manifest."""
    import csv
    from src.data.preprocessing import build_data_list, MODALITIES

    # Add project root to path so the import works
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    cases = build_data_list(data_dir)
    if not cases:
        print("No valid BraTS cases found for manifest generation.")
        return

    fieldnames = MODALITIES + ["label"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    print(f"Manifest written → {out_csv}  ({len(cases)} cases)")


def main() -> None:
    args = parse_args()
    sc = require_synapseclient()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        n = count_cases(out_dir)
        print(f"Found {n} downloaded cases in {out_dir}")
        return

    # Resolve token
    token = args.token or os.environ.get("SYNAPSE_TOKEN")
    if not token:
        print(
            "ERROR: No Synapse token found.\n"
            "  Pass --token <token>  or  set SYNAPSE_TOKEN in your environment.\n"
            "  Get a token at: synapse.org → Account Settings → Personal Access Tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Logging into Synapse…")
    syn = sc.Synapse()
    syn.login(authToken=token, silent=True)
    print(f"Logged in as: {syn.getUserProfile()['userName']}")

    print(f"\nDownloading  {args.synapse_id}  →  {out_dir}")
    print("This may take a while for large datasets (BraTS 2021 ≈ 14 GB).\n")

    entity = syn.get(
        args.synapse_id,
        downloadLocation=str(out_dir),
        ifcollision="keep.local",  # skip files that already exist
    )
    print(f"\nDownload complete.  Entity type: {type(entity).__name__}")

    n = count_cases(out_dir)
    print(f"Cases found in output directory: {n}")

    if args.manifest:
        manifest_path = out_dir.parent / f"{out_dir.name}_manifest.csv"
        generate_manifest(out_dir, manifest_path)


if __name__ == "__main__":
    main()
