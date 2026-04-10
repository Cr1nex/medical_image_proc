"""
Quantitative evaluation metrics for brain tumor segmentation.
  - Dice Similarity Coefficient (DSC) per class
  - 95th-percentile Hausdorff Distance (HD95) per class
"""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
from torch.cuda.amp import autocast
from monai.inferers import SlidingWindowInferer


CLASS_NAMES = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
TUMOR_CLASSES = CLASS_NAMES[1:]  # exclude background


def evaluate(
    model: torch.nn.Module,
    data_loader,
    cfg: dict,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Run inference over data_loader and return per-class DSC and HD95.

    Returns:
        {
            "NCR/NET":         {"dsc": 0.82, "hd95": 4.1},
            "Edema":           {"dsc": 0.88, "hd95": 3.5},
            "Enhancing Tumor": {"dsc": 0.79, "hd95": 5.2},
            "mean":            {"dsc": 0.83, "hd95": 4.3},
        }
    """
    out_channels = cfg["out_channels"]

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=True,
    )
    hd_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean_batch",
        get_not_nans=True,
    )

    inferer = SlidingWindowInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=cfg.get("sw_batch_size", 4),
        overlap=cfg.get("sw_overlap", 0.5),
        mode="gaussian",
    )

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=out_channels)])

    model.eval()
    dice_metric.reset()
    hd_metric.reset()

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                preds = inferer(images, model)

            preds_list = [post_pred(p) for p in decollate_batch(preds)]
            labels_list = [post_label(l) for l in decollate_batch(labels)]

            dice_metric(y_pred=preds_list, y=labels_list)
            hd_metric(y_pred=preds_list, y=labels_list)

    dsc_per_class, _ = dice_metric.aggregate()   # [n_tumor_classes]
    hd_per_class, _ = hd_metric.aggregate()

    results = {}
    for i, name in enumerate(TUMOR_CLASSES):
        results[name] = {
            "dsc": dsc_per_class[i].item(),
            "hd95": hd_per_class[i].item(),
        }

    results["mean"] = {
        "dsc": dsc_per_class.nanmean().item(),
        "hd95": hd_per_class.nanmean().item(),
    }

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print evaluation results."""
    header = f"{'Class':<20} {'DSC':>8} {'HD95 (mm)':>12}"
    print(header)
    print("-" * len(header))
    for name, vals in results.items():
        dsc = vals["dsc"]
        hd = vals["hd95"]
        marker = " *" if name == "mean" else ""
        print(f"{name:<20} {dsc:>8.4f} {hd:>12.2f}{marker}")
