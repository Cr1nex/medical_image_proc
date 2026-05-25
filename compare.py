"""
Comparative experiment runner.

Trains the same model architecture with different loss functions and logs
every run to W&B under the same project so you can overlay them on one chart.

Usage:
    python compare.py --config configs/default.yaml
    python compare.py --config configs/default.yaml --model attention_unet
"""

import argparse
import copy
import random
import yaml
from datetime import datetime

import numpy as np
import torch

from src.data.preprocessing import build_data_list_auto, split_data
from src.data.dataset import build_dataloaders
from src.models.unet3d import build_model, count_parameters
from src.models.losses import build_loss
from src.training.trainer import Trainer


EXPERIMENTS = [
    {"loss": "dice",       "lr": 1e-4},
    {"loss": "focal",      "lr": 1e-4},
    {"loss": "dice_focal", "lr": 1e-4},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None, help="Override model for all runs")
    p.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.model:
        base_cfg["model"] = args.model
    if args.epochs:
        base_cfg["max_epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build data once — shared across all runs
    print("\nScanning dataset...")
    data_list = build_data_list_auto(base_cfg)
    print(f"Found {len(data_list)} cases")

    train_files, val_files, test_files = split_data(
        data_list,
        train_frac=base_cfg.get("train_split", 0.8),
        val_frac=base_cfg.get("val_split", 0.1),
        seed=base_cfg.get("seed", 42),
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        train_files, val_files, test_files, base_cfg
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for exp in EXPERIMENTS:
        cfg = copy.deepcopy(base_cfg)
        cfg.update(exp)

        run_name = f"{cfg['model']}_{cfg['loss']}_lr{cfg['lr']:.0e}_{timestamp}"
        print(f"\n{'#'*60}")
        print(f"  Starting: {run_name}")
        print(f"{'#'*60}")

        set_seed(cfg.get("seed", 42))

        model = build_model(cfg)
        loss_fn = build_loss(cfg)
        print(f"  Params: {count_parameters(model):,}")

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            run_name=run_name,
        )

        best_dsc = trainer.train()
        results[run_name] = best_dsc

    # Final leaderboard
    print(f"\n{'='*50}")
    print("  FINAL LEADERBOARD (Best Val DSC)")
    print(f"{'='*50}")
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, dsc) in enumerate(ranked, 1):
        marker = " <-- WINNER" if rank == 1 else ""
        print(f"  {rank}. {name:<40} {dsc:.4f}{marker}")


if __name__ == "__main__":
    main()
