"""
Per-case error analysis and case-level visualization.

Pass 1 — compute per-case DSC for every test case, export to CSV.
Pass 2 — re-run inference only for the top-N (good) and bottom-N (bad) cases,
          save multi-plane slice grids and 3-D renders under:
              <output_dir>/case_analysis/good/<case_id>/
              <output_dir>/case_analysis/bad/<case_id>/
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.amp import autocast

from src.evaluation.postprocess import remove_small_components
from src.visualization.viz import plot_multiplane_grid, render_3d


CLASS_NAMES = ["NCR/NET", "Edema", "Enhancing Tumor"]
_FLAIR_CH   = 3   # channel index for display (T1=0, T1ce=1, T2=2, FLAIR=3)


def run_error_analysis(
    model: torch.nn.Module,
    data_loader,
    cfg: dict,
    device: torch.device,
    output_csv: str | Path = "outputs/error_analysis.csv",
    n_best:  int = 10,
    n_worst: int = 10,
) -> list[dict]:
    """
    Compute per-case DSC, save CSV, then generate visualizations for the
    n_best and n_worst cases.

    Returns list of per-case result dicts, sorted worst-first.
    """
    out_channels = cfg["out_channels"]
    amp_dtype    = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    inferer = SlidingWindowInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=cfg.get("sw_batch_size", 4),
        overlap=cfg.get("sw_overlap", 0.5),
        mode="gaussian",
    )
    dice_metric  = DiceMetric(include_background=False, reduction="none", get_not_nans=True)
    post_argmax  = AsDiscrete(argmax=True)
    post_onehot  = AsDiscrete(to_onehot=out_channels)
    post_label   = Compose([AsDiscrete(to_onehot=out_channels)])

    # ------------------------------------------------------------------ Pass 1
    model.eval()
    case_results: list[dict] = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            case_id = _extract_case_id(batch, idx)

            with autocast("cuda", dtype=amp_dtype):
                preds = inferer(images, model)

            argmax_list = [post_argmax(p) for p in decollate_batch(preds)]
            clean_list  = [
                torch.from_numpy(
                    remove_small_components(a.squeeze(0).cpu().numpy().astype(np.int32))
                ).unsqueeze(0).to(device)
                for a in argmax_list
            ]
            preds_list  = [post_onehot(p) for p in clean_list]
            labels_list = [post_label(l) for l in decollate_batch(labels)]

            dice_metric.reset()
            dice_metric(y_pred=preds_list, y=labels_list)
            per_class_dsc, _ = dice_metric.aggregate()   # [batch, n_classes]

            for b in range(per_class_dsc.shape[0]):
                dscs = per_class_dsc[b].tolist()
                row  = {"idx": idx, "case_id": case_id}
                for name, val in zip(CLASS_NAMES, dscs):
                    row[f"dsc_{name}"] = val
                row["mean_dsc"] = float(np.nanmean(dscs))
                case_results.append(row)

    case_results.sort(key=lambda r: r["mean_dsc"])

    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id"] + [f"dsc_{n}" for n in CLASS_NAMES] + ["mean_dsc"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "idx"} for r in case_results)

    _print_summary(case_results)

    # ------------------------------------------------------------------ Pass 2
    bad_set  = {r["idx"] for r in case_results[:n_worst]}
    good_set = {r["idx"] for r in case_results[-n_best:]}
    target   = bad_set | good_set

    if not target:
        return case_results

    # Build lookup: idx → {"dsc_*", "mean_dsc"}
    metrics_by_idx = {r["idx"]: r for r in case_results}

    vis_root = output_csv.parent / "case_analysis"
    print(f"\nGenerating visualizations → {vis_root}")

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx not in target:
                continue

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = _extract_case_id(batch, idx)
            split   = "bad" if idx in bad_set else "good"
            case_dir = vis_root / split / (case_id or f"case_{idx:04d}")
            case_dir.mkdir(parents=True, exist_ok=True)

            with autocast("cuda", dtype=amp_dtype):
                preds = inferer(images, model)

            pred_mask = post_argmax(decollate_batch(preds)[0]).squeeze(0).cpu().numpy().astype(np.int32)
            pred_mask = remove_small_components(pred_mask)
            gt_mask   = decollate_batch(labels)[0].squeeze(0).cpu().numpy().astype(np.int32)
            flair     = decollate_batch(images)[0][_FLAIR_CH].cpu().numpy()

            # Build per-class metrics dict for the title
            row = metrics_by_idx[idx]
            metrics = {n: row[f"dsc_{n}"] for n in CLASS_NAMES}
            metrics["Mean"] = row["mean_dsc"]

            plot_multiplane_grid(
                flair, gt_mask, pred_mask,
                metrics=metrics,
                case_id=case_id,
                save_path=case_dir / "slices.png",
            )
            render_3d(pred_mask, save_path=case_dir / "3d_render.html")
            print(f"  [{split:4s}] {case_id}  mean DSC={row['mean_dsc']:.4f}")

    print(f"Done. Case analysis saved to: {vis_root}")
    return case_results


# ---------------------------------------------------------------------------

def _extract_case_id(batch: dict, fallback_idx: int) -> str:
    raw = batch.get("label_meta_dict", {}).get("filename_or_obj", [f"case_{fallback_idx:04d}"])
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    return Path(str(raw)).parent.name


def _print_summary(case_results: list[dict]) -> None:
    print(f"\nError analysis — {len(case_results)} cases  (worst first)")
    print(f"{'Case':<30} {'NCR/NET':>8} {'Edema':>8} {'Enhancing':>10} {'Mean':>8}")
    print("-" * 70)
    for row in case_results[:10]:
        ncr = row["dsc_NCR/NET"]
        edm = row["dsc_Edema"]
        enh = row["dsc_Enhancing Tumor"]
        mn  = row["mean_dsc"]
        print(f"{row['case_id']:<30} {ncr:>8.4f} {edm:>8.4f} {enh:>10.4f} {mn:>8.4f}")
