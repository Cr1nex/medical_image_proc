"""
Visualization tools for brain tumor segmentation results.

  - 2D axial slice overlays (matplotlib): ground truth vs prediction
  - 3D volume render (plotly): marching cubes mesh of predicted tumor
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes


# Color map for the 4 classes (RGBA, values 0-1)
CLASS_COLORS = {
    0: None,                          # background — transparent
    1: (1.0, 0.0, 0.0, 0.6),         # NCR/NET — red
    2: (0.0, 1.0, 0.0, 0.5),         # Edema — green
    3: (0.0, 0.0, 1.0, 0.7),         # Enhancing Tumor — blue
}
CLASS_LABELS = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "Enhancing Tumor"}


# ---------------------------------------------------------------------------
# 2D Axial Overlay
# ---------------------------------------------------------------------------

def plot_axial_overlay(
    flair: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    slice_idx: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a single axial slice with ground truth and prediction overlays.

    Args:
        flair:      3D FLAIR volume  [H, W, D]
        gt_mask:    3D ground truth segmentation [H, W, D], values 0-3
        pred_mask:  3D predicted segmentation [H, W, D], values 0-3
        slice_idx:  axial slice index; if None, uses the middle slice
        save_path:  if given, save the figure here instead of showing
    """
    if slice_idx is None:
        slice_idx = flair.shape[2] // 2

    flair_sl = flair[:, :, slice_idx]
    gt_sl = gt_mask[:, :, slice_idx]
    pred_sl = pred_mask[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Axial Slice {slice_idx}", fontsize=14)

    # Raw FLAIR
    axes[0].imshow(flair_sl.T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[0].axis("off")

    # Ground truth overlay
    _overlay_mask(axes[1], flair_sl, gt_sl, title="Ground Truth")

    # Prediction overlay
    _overlay_mask(axes[2], flair_sl, pred_sl, title="Prediction")

    # Legend
    patches = [
        mpatches.Patch(color=c[:3], alpha=c[3], label=CLASS_LABELS[k])
        for k, c in CLASS_COLORS.items()
        if c is not None
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _overlay_mask(ax, flair_sl: np.ndarray, mask_sl: np.ndarray, title: str) -> None:
    ax.imshow(flair_sl.T, cmap="gray", origin="lower")
    overlay = np.zeros((*flair_sl.T.shape, 4), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        if color is None:
            continue
        region = (mask_sl.T == cls_id)
        overlay[region] = color
    ax.imshow(overlay, origin="lower")
    ax.set_title(title)
    ax.axis("off")


def plot_axial_grid(
    flair: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    n_slices: int = 5,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a grid of n_slices equally-spaced axial slices side by side.
    Rows: GT | Prediction. Columns: slices.
    """
    depth = flair.shape[2]
    indices = np.linspace(depth * 0.1, depth * 0.9, n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))
    fig.suptitle("GT (top) vs Prediction (bottom)", fontsize=14)

    for col, sl in enumerate(indices):
        flair_sl = flair[:, :, sl]
        _overlay_mask(axes[0, col], flair_sl, gt_mask[:, :, sl], title=f"GT z={sl}")
        _overlay_mask(axes[1, col], flair_sl, pred_mask[:, :, sl], title=f"Pred z={sl}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _best_slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    """Return (x, y, z) slice indices that show the most tumor voxels per plane."""
    tumor = (mask > 0)
    if not tumor.any():
        return mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2
    x_idx = int(np.argmax(tumor.sum(axis=(1, 2))))
    y_idx = int(np.argmax(tumor.sum(axis=(0, 2))))
    z_idx = int(np.argmax(tumor.sum(axis=(0, 1))))
    return x_idx, y_idx, z_idx


def plot_multiplane_grid(
    flair: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict[str, float] | None = None,
    case_id: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    2-row × 3-column grid showing GT and prediction in axial, coronal, and
    sagittal planes through the tumor centre of mass.

    Args:
        flair:      [H, W, D] float array
        gt_mask:    [H, W, D] int label map
        pred_mask:  [H, W, D] int label map
        metrics:    per-class DSC dict, e.g. {"NCR/NET": 0.85, "Edema": 0.90, ...}
        case_id:    displayed in the figure title
        save_path:  save PNG here if given
    """
    x_idx, y_idx, z_idx = _best_slice_indices(pred_mask)

    planes = [
        # (title_suffix, gt_slice,          pred_slice,          flair_slice)
        (f"Axial  z={z_idx}",
         gt_mask[:, :, z_idx],   pred_mask[:, :, z_idx],   flair[:, :, z_idx]),
        (f"Coronal y={y_idx}",
         gt_mask[:, y_idx, :],   pred_mask[:, y_idx, :],   flair[:, y_idx, :]),
        (f"Sagittal x={x_idx}",
         gt_mask[x_idx, :, :],   pred_mask[x_idx, :, :],   flair[x_idx, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (plane_title, gt_sl, pred_sl, flair_sl) in enumerate(planes):
        _overlay_mask(axes[0, col], flair_sl, gt_sl,   title=f"GT — {plane_title}")
        _overlay_mask(axes[1, col], flair_sl, pred_sl, title=f"Pred — {plane_title}")

    # Build suptitle with case id and DSC scores
    title_parts = []
    if case_id:
        title_parts.append(f"Case: {case_id}")
    if metrics:
        score_str = "   ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        title_parts.append(score_str)
    fig.suptitle("\n".join(title_parts), fontsize=12, y=1.01)

    patches = [
        mpatches.Patch(color=c[:3], alpha=c[3], label=CLASS_LABELS[k])
        for k, c in CLASS_COLORS.items()
        if c is not None
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3D Volume Render
# ---------------------------------------------------------------------------

def render_3d(
    pred_mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_path: str | Path | None = None,
) -> go.Figure:
    """
    Generate an interactive 3D mesh render of predicted tumor regions using
    marching cubes and Plotly.

    Args:
        pred_mask: 3D array with class labels 0-3
        spacing:   voxel spacing in mm (x, y, z)
        save_path: if given, save as HTML

    Returns:
        plotly Figure (also shown or saved)
    """
    plotly_colors = {
        1: "rgba(255,  50,  50, 0.6)",  # NCR/NET
        2: "rgba( 50, 200,  50, 0.5)",  # Edema
        3: "rgba( 50,  50, 255, 0.7)",  # Enhancing Tumor
    }

    traces = []
    for cls_id, color in plotly_colors.items():
        binary = (pred_mask == cls_id).astype(np.float32)
        if binary.sum() < 8:
            continue  # skip if region too small for marching cubes

        try:
            verts, faces, _, _ = marching_cubes(binary, level=0.5, spacing=spacing)
        except ValueError:
            continue

        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        traces.append(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color=color,
                opacity=float(color.split(",")[-1].rstrip(")")),
                name=CLASS_LABELS[cls_id],
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Tumor Segmentation",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        print(f"Saved 3D render: {save_path}")
    else:
        fig.show()

    return fig
