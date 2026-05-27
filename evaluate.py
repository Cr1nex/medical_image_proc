"""
Entry point for evaluating a trained model on the test split.

Usage:
    python evaluate.py --checkpoint outputs/best_model.pth
    python evaluate.py --checkpoint outputs/best_model.pth --error-analysis
    python evaluate.py --checkpoint outputs/best_model.pth --visualize --n-viz 5
"""

import argparse
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from src.data.preprocessing import build_data_list_auto, split_data
from src.data.dataset import build_dataloaders
from src.models.unet3d import build_model
from src.evaluation.metrics import evaluate, print_results
from src.evaluation.error_analysis import run_error_analysis
from src.visualization.viz import plot_axial_grid, render_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation model")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Config YAML (defaults to cfg saved inside checkpoint)"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Which split to evaluate on"
    )
    parser.add_argument(
        "--error-analysis", action="store_true",
        help="Run per-case error analysis and save CSV"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate slice overlays and 3D renders for the worst N cases"
    )
    parser.add_argument(
        "--n-viz", type=int, default=5,
        help="Number of cases to visualize (used with --visualize)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Config: prefer CLI override, then checkpoint-embedded, then default file
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    elif "cfg" in ckpt:
        cfg = ckpt["cfg"]
        print("Using config from checkpoint.")
    else:
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)
        print("Using default config.")

    # Build model and load weights
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Data
    data_list = build_data_list_auto(cfg)
    train_files, val_files, test_files = split_data(
        data_list,
        train_frac=cfg.get("train_split", 0.8),
        val_frac=cfg.get("val_split", 0.1),
        seed=cfg.get("seed", 42),
    )
    _, val_loader, test_loader = build_dataloaders(
        train_files, val_files, test_files, cfg
    )
    eval_loader = test_loader if args.split == "test" else val_loader
    print(f"Evaluating on {args.split} split ({len(test_files if args.split == 'test' else val_files)} cases)...")

    run_name  = ckpt.get("run_name", Path(args.checkpoint).parent.name)
    run_dir   = Path(cfg.get("output_dir", "outputs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    results = evaluate(model, eval_loader, cfg, device)
    print("\n=== Evaluation Results ===")
    print_results(results)
    print("\nSaving output figures...")
    _save_metrics_figure(results, run_dir)

    # Error analysis
    if args.error_analysis:
        print("\n=== Error Analysis ===")
        output_csv   = run_dir / "error_analysis.csv"
        case_results = run_error_analysis(model, eval_loader, cfg, device, output_csv=output_csv)
        _save_error_table(case_results, run_dir)

    # Visualizations
    if args.visualize:
        print(f"\n=== Generating visualizations for {args.n_viz} cases ===")
        _generate_visualizations(model, eval_loader, cfg, device, args.n_viz)


def _save_metrics_figure(results: dict, output_dir: Path) -> None:
    """Grouped bar chart of all per-class metrics + HD95 sub-plot."""
    classes  = ["NCR/NET", "Edema", "Enhancing Tumor", "mean"]
    rate_metrics = ["dsc", "precision", "recall", "specificity", "iou", "auroc"]
    colors   = ["#e15759", "#4e79a7", "#f28e2b", "#76b7b2"]  # per class

    fig, (ax_rate, ax_hd) = plt.subplots(2, 1, figsize=(13, 9),
                                          gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Evaluation Metrics — Test Split", fontsize=14, fontweight="bold")

    x      = np.arange(len(rate_metrics))
    width  = 0.18
    offset = np.linspace(-(len(classes) - 1) / 2, (len(classes) - 1) / 2, len(classes)) * width

    for ci, (cls, color) in enumerate(zip(classes, colors)):
        if cls not in results:
            continue
        vals = [results[cls].get(m, float("nan")) for m in rate_metrics]
        bars = ax_rate.bar(x + offset[ci], vals, width, label=cls, color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax_rate.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                             f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels([m.upper() for m in rate_metrics], fontsize=10)
    ax_rate.set_ylim(0, 1.12)
    ax_rate.set_ylabel("Score")
    ax_rate.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax_rate.legend(loc="lower right", fontsize=9)
    ax_rate.set_title("Classification Metrics (higher = better)")

    # HD95 — separate scale
    hd_vals   = [results[cls].get("hd95", float("nan")) for cls in classes]
    hd_colors = [c for c in colors]
    ax_hd.bar(classes, hd_vals, color=hd_colors, alpha=0.88)
    for i, (cls, val) in enumerate(zip(classes, hd_vals)):
        if not np.isnan(val):
            ax_hd.text(i, val + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax_hd.set_ylabel("mm")
    ax_hd.set_title("HD95 (lower = better)")

    plt.tight_layout()
    out = output_dir / "metrics_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics figure → {out}")


def _save_error_table(case_results: list[dict], output_dir: Path, n_rows: int = 30) -> None:
    """Color-coded table of per-case DSC, worst cases first."""
    rows    = case_results[:n_rows]
    cols    = ["Case", "NCR/NET", "Edema", "Enhancing", "Mean DSC"]
    data    = []
    for r in rows:
        data.append([
            r["case_id"],
            f"{r['dsc_NCR/NET']:.4f}"        if not np.isnan(r["dsc_NCR/NET"])        else "NaN",
            f"{r['dsc_Edema']:.4f}"          if not np.isnan(r["dsc_Edema"])          else "NaN",
            f"{r['dsc_Enhancing Tumor']:.4f}" if not np.isnan(r["dsc_Enhancing Tumor"]) else "NaN",
            f"{r['mean_dsc']:.4f}",
        ])

    def _dsc_color(val_str: str) -> str:
        try:
            v = float(val_str)
        except ValueError:
            return "#cccccc"
        if v < 0.5:  return "#ff6b6b"
        if v < 0.70: return "#ffa94d"
        if v < 0.85: return "#ffe066"
        return "#8ce99a"

    cell_colors = []
    for row in data:
        row_colors = ["#f0f0f0"]          # case ID column
        for val in row[1:]:
            row_colors.append(_dsc_color(val))
        cell_colors.append(row_colors)

    fig_h = max(4, 0.35 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    fig.suptitle(f"Error Analysis — {len(rows)} Worst Cases (test split)",
                 fontsize=12, fontweight="bold")

    tbl = ax.table(
        cellText=data,
        colLabels=cols,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols))))

    # Bold header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#333333")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "error_table.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Error table    → {out}")


def _generate_visualizations(model, data_loader, cfg, device, n_cases: int) -> None:
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import AsDiscrete, Compose
    from monai.data import decollate_batch
    from torch.amp import autocast
    import nibabel as nib
    import numpy as np

    out_channels = cfg["out_channels"]
    output_dir = Path(cfg.get("output_dir", "outputs")) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    inferer = SlidingWindowInferer(
        roi_size=cfg["patch_size"],
        sw_batch_size=cfg.get("sw_batch_size", 4),
        overlap=cfg.get("sw_overlap", 0.5),
        mode="gaussian",
    )
    post_pred = Compose([AsDiscrete(argmax=True)])

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            if count >= n_cases:
                break

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda"):
                preds = inferer(images, model)

            pred_mask = post_pred(decollate_batch(preds)[0]).squeeze(0).cpu().numpy()
            gt_mask = decollate_batch(labels)[0].squeeze(0).cpu().numpy()

            # Use FLAIR channel (index 3) for background
            flair = decollate_batch(images)[0][3].cpu().numpy()

            case_dir = output_dir / f"case_{count:03d}"
            case_dir.mkdir(exist_ok=True)

            # 2D axial grid
            plot_axial_grid(
                flair, gt_mask, pred_mask,
                n_slices=5,
                save_path=case_dir / "axial_grid.png",
            )

            # 3D render
            render_3d(
                pred_mask,
                save_path=case_dir / "3d_render.html",
            )

            count += 1
            print(f"  Visualized case {count}/{n_cases}")


if __name__ == "__main__":
    main()
