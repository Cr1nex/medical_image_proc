"""
Entry point for training the 3D brain tumor segmentation model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --dry-run
    python train.py --config configs/default.yaml --model attention_unet --loss dice
"""

import argparse
import random
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import build_data_list_auto, split_data
from src.data.dataset import build_dataloaders, warm_cache
from src.models.unet3d import build_model, count_parameters
from src.models.losses import build_loss
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D brain tumor segmentation model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model", default=None, help="Override model type")
    parser.add_argument("--loss", default=None, help="Override loss function")
    parser.add_argument("--dry-run", action="store_true", help="Run one iteration then exit")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.model:
        cfg["model"] = args.model
    if args.loss:
        cfg["loss"] = args.loss

    set_seed(cfg.get("seed", 42))

    # Enable TF32 for matmul (off by default in PyTorch; ~20-30% speedup on
    # Ampere/Ada/Blackwell with negligible precision difference for this task).
    torch.backends.cuda.matmul.allow_tf32 = True
    # Let cuDNN auto-select fastest kernels for fixed patch shapes.
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nScanning data directory...")
    data_list = build_data_list_auto(cfg)
    if not data_list:
        fmt = cfg.get("dataset_format", "brats")
        src = cfg.get("manifest") if fmt in ("csv", "json") else cfg.get("data_dir")
        raise RuntimeError(
            f"No valid cases found in '{src}' (dataset_format={fmt}). "
            "Check your data_dir / manifest path and modality_keys."
        )
    print(f"Found {len(data_list)} cases")

    train_files, val_files, test_files = split_data(
        data_list,
        train_frac=cfg.get("train_split", 0.8),
        val_frac=cfg.get("val_split", 0.1),
        seed=cfg.get("seed", 42),
    )
    print(f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_files, val_files, test_files, cfg
    )
    if not args.dry_run:
        warm_cache(train_loader.dataset, num_workers=cfg.get("num_workers", 8))

    # ------------------------------------------------------------------
    # Model & Loss
    # ------------------------------------------------------------------
    model = build_model(cfg)
    loss_fn = build_loss(cfg)

    print(f"\nModel:      {cfg['model']}")
    print(f"Loss:       {cfg['loss']}")
    print(f"Parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # ------------------------------------------------------------------
    # Dry run: verify one forward pass
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n--- DRY RUN ---")
        model = model.to(device)
        model.train()
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        print(f"Input:  {tuple(images.shape)}")
        print(f"Output: {tuple(preds.shape)}")
        print(f"Loss:   {loss.item():.4f}")
        print("Dry run passed.")
        return

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['model']}_{cfg['loss']}_{timestamp}"

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        run_name=run_name,
    )
    trainer.train()


if __name__ == "__main__":
    main()
