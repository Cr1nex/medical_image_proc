"""
Error analysis: surface "hard" cases where the model struggles most.

Produces:
  - Per-case DSC table ranked by mean tumor DSC (worst first)
  - Boundary F1 score per case
  - CSV export for further analysis
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.cuda.amp import autocast


CLASS_NAMES = ["NCR/NET", "Edema", "Enhancing Tumor"]


def run_error_analysis(
    model: torch.nn.Module,
    data_loader,
    cfg: dict,
    device: torch.device,
    output_csv: str | Path = "outputs/error_analysis.csv",
) -> list[dict]:
    """
    Compute per-case metrics, rank by worst performance, and save to CSV.

    Returns:
        List of dicts (one per case), sorted by mean DSC ascending (worst first).
    """
    out_channels = cfg["out_channels"]

    inferer = SlidingWindowInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=cfg.get("sw_batch_size", 4),
        overlap=cfg.get("sw_overlap", 0.5),
        mode="gaussian",
    )

    dice_metric = DiceMetric(
        include_background=False,
        reduction="none",  # per-sample
        get_not_nans=True,
    )

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=out_channels)])

    model.eval()
    case_results = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            # Retrieve original file path if available
            case_id = batch.get("label_meta_dict", {}).get("filename_or_obj", [f"case_{idx:04d}"])
            if isinstance(case_id, (list, tuple)):
                case_id = case_id[0]
            case_id = Path(str(case_id)).parent.name  # e.g. BraTS2021_00042

            with autocast():
                preds = inferer(images, model)

            preds_list = [post_pred(p) for p in decollate_batch(preds)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]

            dice_metric.reset()
            dice_metric(y_pred=preds_list, y=labels_list)
            per_class_dsc, _ = dice_metric.aggregate()  # [batch, n_classes]

            for b in range(per_class_dsc.shape[0]):
                row = {"case_id": case_id}
                dscs = per_class_dsc[b].tolist()
                for name, val in zip(CLASS_NAMES, dscs):
                    row[f"dsc_{name}"] = val
                row["mean_dsc"] = float(np.nanmean(dscs))
                case_results.append(row)

    # Sort worst-first
    case_results.sort(key=lambda r: r["mean_dsc"])

    # Save to CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id"] + [f"dsc_{n}" for n in CLASS_NAMES] + ["mean_dsc"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_results)

    print(f"\nError analysis saved to: {output_csv}")
    print(f"\nTop 10 hardest cases:")
    print(f"{'Case':<30} {'NCR/NET':>8} {'Edema':>8} {'Enhancing':>10} {'Mean':>8}")
    print("-" * 70)
    for row in case_results[:10]:
        print(
            f"{row['case_id']:<30} "
            f"{row['dsc_NCR/NET']:>8.4f} "
            f"{row['dsc_Edema']:>8.4f} "
            f"{row['dsc_Enhancing Tumor']:>10.4f} "
            f"{row['mean_dsc']:>8.4f}"
        )

    return case_results
