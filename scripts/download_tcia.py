"""
Download brain tumor MRI datasets from the NCI Imaging Data Commons (IDC).

IDC hosts TCIA collections on Google Cloud Storage — no account required.

Recommended collection for brain tumor segmentation (no token needed):
  upenn_gbm   UPenn Glioblastoma, 630 patients, CaPTk pre-processed
              (T1, T1ce, T2, FLAIR already co-registered + skull-stripped)
              Segmentation labels included.

Other collections:
  tcga_gbm    TCGA GBM   (SM/pathology slides, not MRI — skip)
  tcga_lgg    TCGA LGG   (SM/pathology slides, not MRI — skip)
  cptac_gbm   CPTAC GBM  (SM/pathology, not MRI — skip)
  icdc_glioma ICDC Glioma (small MRI set)

Requirements
------------
  conda run -n imgp pip install idc-index pydicom
  conda install -n imgp -c conda-forge dcm2niix

Usage
-----
  # List available brain collections
  python scripts/download_tcia.py --list

  # Download UPenn-GBM (complete 4-modality cases, ~20 GB)
  python scripts/download_tcia.py \\
      --collection upenn_gbm \\
      --out-dir data/raw/upenn_gbm

  # Preview only (no download)
  python scripts/download_tcia.py \\
      --collection upenn_gbm \\
      --out-dir data/raw/upenn_gbm \\
      --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BRAIN_COLLECTIONS = {
    "upenn_gbm":   "UPenn GBM — 630 patients, MR+SEG, CaPTk pre-processed (~20 GB filtered)",
    "icdc_glioma": "ICDC Glioma — small MRI collection",
    "gbm_dsc_mri_dro": "GBM DSC-MRI DRO — synthetic phantom data",
}

# UPenn-GBM: the 4 CaPTk-processed modality series descriptions
UPENN_TARGET_DESCRIPTIONS = [
    "t1 axial: Processed_CaPTk",             # T1 pre-contrast
    "t1 axial stealth-post : Processed_CaPTk",  # T1ce post-contrast
    "Axial T2 tse: Processed_CaPTk",          # T2
    "t2_Flair_axial: Processed_CaPTk",        # FLAIR
]
# Label map: series description → our modality key
UPENN_MODALITY_MAP = {
    "t1 axial: Processed_CaPTk":                "t1",
    "t1 axial stealth-post : Processed_CaPTk":  "t1ce",
    "Axial T2 tse: Processed_CaPTk":            "t2",
    "t2_Flair_axial: Processed_CaPTk":          "flair",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download brain tumor MRI from IDC (no account needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--collection", default="upenn_gbm",
                   help="Collection name (default: upenn_gbm)")
    p.add_argument("--out-dir", required=True,
                   help="Download directory")
    p.add_argument("--list", action="store_true",
                   help="List available collections and exit")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Limit number of cases (useful for a quick test)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded without downloading")
    return p.parse_args()


def require_idc():
    try:
        from idc_index import IDCClient
        return IDCClient
    except ImportError:
        print(
            "idc-index is not installed.\n"
            "  conda run -n imgp pip install idc-index",
            file=sys.stderr,
        )
        sys.exit(1)


def get_upenn_series(client, max_cases=None):
    """
    Return the SeriesInstanceUIDs for complete UPenn-GBM cases:
    4 pre-processed MR modalities + at least 1 segmentation per patient.
    """
    from collections import Counter
    idx = client.index
    upenn = idx[idx["collection_id"] == "upenn_gbm"]

    # MR: only the 4 CaPTk-processed target series
    mr = upenn[upenn["SeriesDescription"].isin(UPENN_TARGET_DESCRIPTIONS)]
    seg = upenn[upenn["Modality"] == "SEG"]

    # Keep only patients with all 4 modalities
    mr_counts = Counter(mr["PatientID"])
    seg_pats = set(seg["PatientID"])
    complete_patients = sorted(
        p for p, cnt in mr_counts.items() if cnt >= 4 and p in seg_pats
    )

    if max_cases:
        complete_patients = complete_patients[:max_cases]

    target_mr  = mr[mr["PatientID"].isin(complete_patients)]
    target_seg = seg[seg["PatientID"].isin(complete_patients)]

    series_uids = (
        list(target_mr["SeriesInstanceUID"]) +
        list(target_seg["SeriesInstanceUID"])
    )
    return series_uids, complete_patients, target_mr, target_seg


def main() -> None:
    args = parse_args()
    IDCClient = require_idc()

    print("Connecting to IDC…")
    client = IDCClient()

    if args.list:
        idx = client.index
        all_cols = idx["collection_id"].unique().tolist()
        brain = [c for c in all_cols if any(
            kw in c.lower() for kw in ["brain","glio","neuro","gbm","lgg","pdgm","tumor"]
        )]
        print("\nBrain-related IDC collections (no token required):")
        for c in sorted(brain):
            desc = BRAIN_COLLECTIONS.get(c, "")
            print(f"  {c:<30}  {desc}")
        return

    out_dir = Path(args.out_dir)

    if args.collection.lower() == "upenn_gbm":
        print("Identifying UPenn-GBM complete cases (4 modalities + segmentation)…")
        series_uids, patients, target_mr, target_seg = get_upenn_series(
            client, args.max_cases
        )
        n_mr  = len(target_mr)
        n_seg = len(target_seg)
        size_gb = (
            target_mr["series_size_MB"].sum() +
            target_seg["series_size_MB"].sum()
        ) / 1024

        print(f"  Complete cases : {len(patients)}")
        print(f"  MR series      : {n_mr}  (T1, T1ce, T2, FLAIR per patient)")
        print(f"  SEG series     : {n_seg}")
        print(f"  Download size  : ~{size_gb:.1f} GB")

        if args.dry_run:
            print("\n[dry-run] No files downloaded.")
            print(f"Remove --dry-run to start the download to: {out_dir}")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading to {out_dir} …")

        client.download_from_selection(
            seriesInstanceUID=series_uids,
            downloadDir=str(out_dir),
            show_progress_bar=True,
        )

        print(f"\nDownload complete.")
        print(f"\nNext — convert DICOM to NIfTI:")
        print(f"  conda install -n imgp -c conda-forge dcm2niix   # if not installed")
        print(f"  conda run -n imgp python scripts/convert_dicom.py \\")
        print(f"      --dicom-dir {out_dir} \\")
        print(f"      --out-dir data/converted/upenn_gbm \\")
        print(f"      --modality-map 't1=t1 axial: Processed_CaPTk' \\")
        print(f"                     't1ce=stealth-post' \\")
        print(f"                     't2=Axial T2 tse' \\")
        print(f"                     'flair=Flair_axial'")
    else:
        # Generic download
        print(f"Downloading collection: {args.collection}")
        if args.dry_run:
            subset = client.index[client.index["collection_id"] == args.collection]
            print(f"  {subset['PatientID'].nunique()} patients, "
                  f"{subset['series_size_MB'].sum()/1024:.1f} GB total")
            print("[dry-run] No files downloaded.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        client.download_from_selection(
            collection_id=args.collection,
            downloadDir=str(out_dir),
            show_progress_bar=True,
        )
        print("Download complete.")


if __name__ == "__main__":
    main()
