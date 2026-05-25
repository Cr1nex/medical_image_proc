"""
Loss functions for multi-class brain tumor segmentation.

Supported (selectable via config["loss"]):
  - "dice"       → Soft Dice Loss
  - "focal"      → Focal Loss
  - "dice_focal" → Weighted combination of Dice + Focal (default)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss


def build_loss(cfg: dict) -> nn.Module:
    """
    Return the configured loss function.

    All losses use softmax over the channel (class) dimension and expect
    one-hot encoded targets produced by MONAI's AsDiscrete transform.

    Args:
        cfg: config dict with keys:
            loss               – loss name
            focal_gamma        – gamma for focal loss (default 2.0)
            dice_focal_lambda_dice  – weight for dice term (default 0.5)
            dice_focal_lambda_focal – weight for focal term (default 0.5)
    """
    name = cfg["loss"].lower()
    gamma = cfg.get("focal_gamma", 2.0)

    if name == "dice":
        return DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,  # ignore background class in loss
        )

    elif name == "focal":
        return FocalLoss(
            to_onehot_y=True,
            gamma=gamma,
            include_background=False,
            use_softmax=True,   # MONAI 1.3+ default is False (sigmoid); must be explicit
        )

    elif name == "dice_focal":
        lambda_dice = cfg.get("dice_focal_lambda_dice", 0.5)
        lambda_focal = cfg.get("dice_focal_lambda_focal", 0.5)
        return DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            gamma=gamma,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            include_background=False,
        )

    else:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: dice, focal, dice_focal"
        )
