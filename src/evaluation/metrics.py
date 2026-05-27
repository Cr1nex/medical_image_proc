"""
Quantitative evaluation for brain tumor segmentation.

Per-class (NCR/NET, Edema, Enhancing Tumor):
  DSC/F1, Precision, Recall, Specificity, Accuracy, IoU, AUROC, HD95
"""

from __future__ import annotations

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose
from sklearn.metrics import roc_auc_score
from torch.amp import autocast

from src.evaluation.postprocess import remove_small_components

CLASS_NAMES   = ["Background", "NCR/NET", "Edema", "Enhancing Tumor"]
TUMOR_CLASSES = CLASS_NAMES[1:]

_EPS                    = 1e-8
_AUROC_VOXELS_PER_BATCH = 50_000   # subsample to keep memory reasonable


def evaluate(
    model: torch.nn.Module,
    data_loader,
    cfg: dict,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Slide-window inference over data_loader and return a results dict.

    Return shape::
        {
            "NCR/NET":         {"dsc": .., "f1": .., "precision": .., "recall": ..,
                                "specificity": .., "accuracy": .., "iou": ..,
                                "auroc": .., "hd95": ..},
            "Edema":           {...},
            "Enhancing Tumor": {...},
            "mean":            {...},
        }
    """
    out_channels = cfg["out_channels"]
    amp_dtype    = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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

    to_argmax       = AsDiscrete(argmax=True)
    to_onehot       = AsDiscrete(to_onehot=out_channels)
    to_onehot_label = Compose([AsDiscrete(to_onehot=out_channels)])

    n_tumor = len(TUMOR_CLASSES)
    tp = np.zeros(n_tumor)
    fp = np.zeros(n_tumor)
    fn = np.zeros(n_tumor)
    tn = np.zeros(n_tumor)

    auroc_probs  = [[] for _ in range(n_tumor)]
    auroc_labels = [[] for _ in range(n_tumor)]

    model.eval()
    hd_metric.reset()

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", dtype=amp_dtype):
                raw_preds = inferer(images, model)   # [B, C, H, W, D] logits

            probs = torch.softmax(raw_preds.float(), dim=1)  # float32 for stability

            # argmax → post-process → one-hot
            argmax_list = [to_argmax(p) for p in decollate_batch(raw_preds)]  # [1,H,W,D]
            clean_list  = [
                torch.from_numpy(
                    remove_small_components(a.squeeze(0).cpu().numpy().astype(np.int32))
                ).unsqueeze(0).to(device)
                for a in argmax_list
            ]
            preds_list  = [to_onehot(p) for p in clean_list]
            labels_list = [to_onehot_label(l) for l in decollate_batch(labels)]

            hd_metric(y_pred=preds_list, y=labels_list)

            pred_oh  = torch.stack(preds_list,  dim=0).float()   # [B, C, H, W, D]
            label_oh = torch.stack(labels_list, dim=0).float()

            for ci, _ in enumerate(TUMOR_CLASSES):
                c = ci + 1          # channel index (skip background=0)
                p = pred_oh[:,  c]  # [B, H, W, D]
                l = label_oh[:, c]

                tp[ci] += (p *       l ).sum().item()
                fp[ci] += (p * (1 - l)).sum().item()
                fn[ci] += ((1 - p) * l ).sum().item()
                tn[ci] += ((1 - p) * (1 - l)).sum().item()

                prob_flat  = probs[:, c].cpu().numpy().ravel()
                label_flat = l.cpu().numpy().ravel()
                n_vox      = len(prob_flat)
                idx = np.random.choice(n_vox,
                                       min(_AUROC_VOXELS_PER_BATCH, n_vox),
                                       replace=False)
                auroc_probs[ci].append(prob_flat[idx])
                auroc_labels[ci].append(label_flat[idx])

    hd_per_class, _ = hd_metric.aggregate()

    precision   = tp / (tp + fp + _EPS)
    recall      = tp / (tp + fn + _EPS)
    f1          = 2 * tp / (2 * tp + fp + fn + _EPS)   # identical to DSC
    iou         = tp / (tp + fp + fn + _EPS)
    specificity = tn / (tn + fp + _EPS)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + _EPS)

    auroc = []
    for ci in range(n_tumor):
        all_p = np.concatenate(auroc_probs[ci])
        all_l = np.concatenate(auroc_labels[ci])
        try:
            auroc.append(roc_auc_score(all_l, all_p))
        except ValueError:
            auroc.append(float("nan"))
    auroc = np.array(auroc)

    results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TUMOR_CLASSES):
        results[name] = {
            "dsc":         f1[i],
            "f1":          f1[i],
            "precision":   precision[i],
            "recall":      recall[i],
            "specificity": specificity[i],
            "accuracy":    accuracy[i],
            "iou":         iou[i],
            "auroc":       auroc[i],
            "hd95":        hd_per_class[i].item(),
        }

    results["mean"] = {
        "dsc":         float(np.nanmean(f1)),
        "f1":          float(np.nanmean(f1)),
        "precision":   float(np.nanmean(precision)),
        "recall":      float(np.nanmean(recall)),
        "specificity": float(np.nanmean(specificity)),
        "accuracy":    float(np.nanmean(accuracy)),
        "iou":         float(np.nanmean(iou)),
        "auroc":       float(np.nanmean(auroc)),
        "hd95":        float(hd_per_class.nanmean().item()),
    }

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    metrics = ["dsc", "precision", "recall", "f1", "specificity", "accuracy", "iou", "auroc", "hd95"]
    col_w   = 10
    header  = f"{'Class':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("-" * len(header))
    order = TUMOR_CLASSES + ["mean"]
    for name in order:
        if name not in results:
            continue
        vals   = results[name]
        marker = " *" if name == "mean" else ""
        row    = f"{name:<20}"
        for m in metrics:
            v = vals.get(m, float("nan"))
            fmt = f"{v:>{col_w}.2f}" if m == "hd95" else f"{v:>{col_w}.4f}"
            row += fmt
        print(row + marker)
