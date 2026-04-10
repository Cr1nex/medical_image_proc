"""
Model factory for 3D brain tumor segmentation.

Supported architectures (selectable via config["model"]):
  - "unet"           → MONAI 3D U-Net
  - "attention_unet" → MONAI Attention U-Net (attention gates on skip connections)
  - "segresnet"      → MONAI SegResNet (residual blocks encoder-decoder)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, SegResNet, UNet


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate and return the requested model.

    Args:
        cfg: parsed config dict with keys:
            model        – architecture name
            in_channels  – number of input MRI modalities (4 for BraTS)
            out_channels – number of segmentation classes (4)
    """
    name = cfg["model"].lower()
    in_ch = cfg["in_channels"]
    out_ch = cfg["out_channels"]

    if name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
        )

    elif name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            dropout=0.1,
        )

    elif name == "segresnet":
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            init_filters=32,
            dropout_prob=0.1,
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: unet, attention_unet, segresnet"
        )

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = {"model": "unet", "in_channels": 4, "out_channels": 4}
    model = build_model(cfg)
    x = torch.randn(1, 4, 128, 128, 128)
    y = model(x)
    print(f"Model: {cfg['model']}")
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Params: {count_parameters(model):,}")
