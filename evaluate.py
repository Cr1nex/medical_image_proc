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

    # Metrics
    results = evaluate(model, eval_loader, cfg, device)
    print("\n=== Evaluation Results ===")
    print_results(results)

    # Error analysis
    if args.error_analysis:
        print("\n=== Error Analysis ===")
        output_csv = Path(cfg.get("output_dir", "outputs")) / "error_analysis.csv"
        run_error_analysis(model, eval_loader, cfg, device, output_csv=output_csv)

    # Visualizations
    if args.visualize:
        print(f"\n=== Generating visualizations for {args.n_viz} cases ===")
        _generate_visualizations(model, eval_loader, cfg, device, args.n_viz)


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
