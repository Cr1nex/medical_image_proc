"""
Interactive Brain Tumor Segmentation UI

Run: python app.py
Then open: http://localhost:7860
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndi
import torch
import yaml

import gradio as gr
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    AsDiscrete,
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
)
from PIL import Image
from skimage.measure import marching_cubes

from src.models.unet3d import build_model
from src.data.preprocessing import remap_labels
from src.evaluation.postprocess import remove_small_components

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUTS_DIR  = Path("outputs")
DATA_DIR     = Path("data/raw/BraTS2021_Training_Data")
CONFIG_PATH  = Path("configs/default.yaml")
MODALITIES   = ["t1", "t1ce", "t2", "flair"]
NUM_CLASSES  = 4
DEFAULT_PATCH = [96, 96, 96]

CLASS_COLORS = {
    1: (1.00, 0.00, 0.00, 0.90),  # NCR/NET  — bright, high-visibility red
    2: (0.15, 0.75, 0.15, 0.40),  # Edema    — reduced alpha to avoid masking red
    3: (0.15, 0.35, 1.00, 0.50),  # Enhancing — reduced alpha to avoid masking red
}
PLOTLY_COLORS = {
    1: ("255,0,0", 0.90),
    2: ("60,190,60", 0.35),
    3: ("60,100,255", 0.45),
}
PLOTLY_RENDER_ORDER = [2, 3, 1]  # draw NCR last so it remains visible on overlap
CLASS_LABELS = {
    1: "NCR/NET (Necrotic Core)",
    2: "Edema",
    3: "Enhancing Tumor",
}

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def scan_checkpoints() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return ["No checkpoints found"]
    pths = sorted(OUTPUTS_DIR.rglob("*.pth"))
    return [str(p) for p in pths] or ["No checkpoints found"]


def scan_cases() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str) -> tuple[torch.nn.Module, dict, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    else:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg, device


# ---------------------------------------------------------------------------
# Preprocessing (mirrors dataset.py deterministic base transforms)
# ---------------------------------------------------------------------------

def _resolve_modality_order(cfg: dict) -> list[str]:
    """
    Resolve modality order for inference.
    If checkpoint modality_keys is missing or incompatible with this 4-input UI,
    fall back to default BraTS ordering.
    """
    raw = cfg.get("modality_keys", MODALITIES)
    if not isinstance(raw, list):
        return list(MODALITIES)

    ordered: list[str] = []
    for key in raw:
        k = str(key).lower()
        if k in MODALITIES and k not in ordered:
            ordered.append(k)

    if len(ordered) != len(MODALITIES):
        return list(MODALITIES)
    return ordered


def build_inference_transforms(modality_order: list[str], cfg: dict | None = None) -> Compose:
    # Mirror the training base transform pipeline so inference sees the same
    # preprocessing as training.  Omitting clipping (training does it) shifts
    # the input distribution and degrades predictions significantly.
    prep = (cfg or {}).get("preprocessing", {})
    do_clip   = prep.get("clip_intensities", True)
    lower_pct = prep.get("clip_lower_pct",  0.5)
    upper_pct = prep.get("clip_upper_pct", 99.5)

    from src.data.dataset import ClipIntensityByPercentiled

    steps: list = [
        LoadImaged(keys=modality_order, image_only=False),
        EnsureChannelFirstd(keys=modality_order),
        ConcatItemsd(keys=modality_order, name="image", dim=0),
        DeleteItemsd(keys=modality_order),
    ]
    if do_clip:
        steps.append(ClipIntensityByPercentiled(
            keys=["image"], lower_pct=lower_pct, upper_pct=upper_pct,
        ))
    steps += [
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ]
    return Compose(steps)


def load_gt(label_path: str) -> np.ndarray:
    vol = nib.load(label_path).get_fdata(dtype=np.float32)
    return remap_labels(vol.astype(np.int32))


def _voxel_spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Extract voxel spacing (mm) from a 4x4 affine matrix."""
    arr = np.asarray(affine, dtype=np.float64)
    if arr.shape != (4, 4):
        return (1.0, 1.0, 1.0)

    spacing = np.linalg.norm(arr[:3, :3], axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        return (1.0, 1.0, 1.0)
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def _keep_main_tumor_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the dominant connected whole-tumor component for 3D display.
    This removes isolated false-positive islands that can look detached.
    """
    whole_tumor = (mask > 0).astype(np.uint8)
    if whole_tumor.sum() == 0:
        return mask

    labeled, n_labels = ndi.label(whole_tumor)
    if n_labels <= 1:
        return mask

    sizes = ndi.sum(whole_tumor, labeled, index=np.arange(1, n_labels + 1))
    sizes = np.asarray(sizes, dtype=np.float64)
    keep_label = int(np.argmax(sizes)) + 1
    keep_region = (labeled == keep_label)
    return np.where(keep_region, mask, 0).astype(mask.dtype)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    ckpt_path: str,
    t1_path, t1ce_path, t2_path, flair_path,
    gt_path,
    case_id: str,
    state: dict,
) -> tuple[dict, str]:

    if not ckpt_path or ckpt_path == "No checkpoints found":
        return state, "❌ Please select a valid checkpoint."

    # Resolve file paths ------------------------------------------------
    if case_id and case_id != "— select a case —":
        case_dir = DATA_DIR / case_id
        paths = {mod: str(case_dir / f"{case_id}_{mod}.nii.gz") for mod in MODALITIES}
        seg_candidate = case_dir / f"{case_id}_seg.nii.gz"
        gt_resolved = str(seg_candidate) if seg_candidate.exists() else None
    else:
        if any(p is None for p in [t1_path, t1ce_path, t2_path, flair_path]):
            return state, "❌ Please upload all 4 modality files or pick a dataset case."
        paths = {
            "t1":    t1_path if isinstance(t1_path, str) else t1_path.name,
            "t1ce":  t1ce_path if isinstance(t1ce_path, str) else t1ce_path.name,
            "t2":    t2_path if isinstance(t2_path, str) else t2_path.name,
            "flair": flair_path if isinstance(flair_path, str) else flair_path.name,
        }
        gt_resolved = (gt_path if isinstance(gt_path, str) else gt_path.name) if gt_path else None

    # Load model (cache by checkpoint path) ----------------------------
    if state.get("ckpt_path") != ckpt_path:
        try:
            model, cfg, device = load_checkpoint(ckpt_path)
            state["model"]     = model
            state["cfg"]       = cfg
            state["device"]    = device
            state["ckpt_path"] = ckpt_path
        except Exception as e:
            return state, f"❌ Failed to load checkpoint: {e}"
    else:
        model  = state["model"]
        cfg    = state["cfg"]
        device = state["device"]

    modality_order = _resolve_modality_order(cfg)
    ordered_paths = {k: paths[k] for k in modality_order}

    # Preprocess --------------------------------------------------------
    try:
        transforms = build_inference_transforms(modality_order, cfg)
        data = transforms(ordered_paths)
        image_tensor = data["image"]              # [4, H, W, D]
        meta = getattr(image_tensor, "meta", {})
        affine = meta.get("original_affine", np.eye(4))
        spacing = _voxel_spacing_from_affine(affine)
    except Exception as e:
        return state, f"❌ Preprocessing failed: {e}"

    # Inference ---------------------------------------------------------
    patch_size = cfg.get("patch_size", DEFAULT_PATCH)
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=1,         # safe on both CPU and GPU
        overlap=cfg.get("sw_overlap", 0.5),
        mode="gaussian",
    )

    try:
        image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 4, H, W, D]
        with torch.no_grad():
            logits = inferer(image_batch, model)             # [1, 4, H, W, D]
        post = AsDiscrete(argmax=True)
        pred_mask = post(logits[0]).squeeze(0).cpu().numpy().astype(np.int32)
        pred_mask = remove_small_components(pred_mask)
    except Exception as e:
        return state, f"❌ Inference failed: {e}"

    # Ground truth ------------------------------------------------------
    gt_mask = None
    if gt_resolved:
        try:
            gt_mask = load_gt(gt_resolved)
        except Exception:
            pass  # GT is optional

    display_key = "flair" if "flair" in modality_order else modality_order[0]
    display_ch = modality_order.index(display_key)
    display_vol = image_tensor[display_ch].numpy()  # [H, W, D]

    # Update state ------------------------------------------------------
    state.update({
        "pred_mask": pred_mask,
        "gt_mask":   gt_mask,
        "flair_vol": display_vol,
        "display_channel_name": display_key.upper(),
        "affine":    affine,
        "spacing":   spacing,
        "depth":     pred_mask.shape[2],
    })

    device_str = "GPU" if device.type == "cuda" else "CPU"
    gt_str = "with ground truth" if gt_mask is not None else "no ground truth"
    return state, f"✅ Inference complete on {device_str} ({gt_str}) — use the slider to explore slices."


# ---------------------------------------------------------------------------
# Slice renderer
# ---------------------------------------------------------------------------

def _overlay(ax, flair_sl, mask_sl, title):
    ax.imshow(flair_sl.T, cmap="gray", origin="lower")
    rgba = np.zeros((*flair_sl.T.shape, 4), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        rgba[mask_sl.T == cls_id] = color
    ax.imshow(rgba, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_slice(state: dict, slice_idx: int):
    if not state or "pred_mask" not in state:
        return None

    flair_vol = state["flair_vol"]
    pred_mask = state["pred_mask"]
    gt_mask   = state.get("gt_mask")
    channel_name = state.get("display_channel_name", "FLAIR")

    slice_idx = int(np.clip(slice_idx, 0, flair_vol.shape[2] - 1))
    flair_sl  = flair_vol[:, :, slice_idx]
    n_panels  = 3 if gt_mask is not None else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(flair_sl.T, cmap="gray", origin="lower")
    axes[0].set_title(f"{channel_name}  (z={slice_idx})", fontsize=10)
    axes[0].axis("off")

    if gt_mask is not None:
        _overlay(axes[1], flair_sl, gt_mask[:, :, slice_idx], "Ground Truth")
        _overlay(axes[2], flair_sl, pred_mask[:, :, slice_idx], "Prediction")
    else:
        _overlay(axes[1], flair_sl, pred_mask[:, :, slice_idx], "Prediction")

    patches = [
        mpatches.Patch(color=c[:3], alpha=c[3], label=CLASS_LABELS[k])
        for k, c in CLASS_COLORS.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


# ---------------------------------------------------------------------------
# 3D render
# ---------------------------------------------------------------------------

def render_3d(state: dict) -> go.Figure:
    if not state or "pred_mask" not in state:
        return go.Figure()

    pred_mask = state["pred_mask"]
    vis_mask = _keep_main_tumor_component(pred_mask)
    spacing = tuple(state.get("spacing", (1.0, 1.0, 1.0)))
    traces = _build_mesh_traces(
        vis_mask,
        spacing,
        PLOTLY_COLORS,
        plot_order=PLOTLY_RENDER_ORDER,
        fallback_mask=pred_mask,
    )

    # Fallback: if dominant-component view drops too much, render raw mask.
    if not traces and (pred_mask > 0).any():
        traces = _build_mesh_traces(
            pred_mask,
            spacing,
            PLOTLY_COLORS,
            plot_order=PLOTLY_RENDER_ORDER,
        )

    if not traces:
        total_tumor_voxels = int((pred_mask > 0).sum())
        text = "No tumor regions detected" if total_tumor_voxels == 0 else (
            f"Tumor too small/disconnected for stable 3D mesh (voxels={total_tumor_voxels})"
        )
        fig = go.Figure()
        fig.add_annotation(text=text, showarrow=False,
                           font=dict(size=16))
        return fig

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Tumor Segmentation",
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1),
    )
    return fig


def _build_mesh_traces(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    plotly_colors: dict[int, tuple[str, float]],
    plot_order: list[int] | None = None,
    fallback_mask: np.ndarray | None = None,
) -> list[go.Mesh3d]:
    traces: list[go.Mesh3d] = []
    order = plot_order if plot_order is not None else list(plotly_colors.keys())
    for cls_id in order:
        if cls_id not in plotly_colors:
            continue
        rgb, opacity = plotly_colors[cls_id]
        binary = (mask == cls_id).astype(np.uint8)

        # If filtered mask removed this class entirely (common for small NCR),
        # recover this class from the raw prediction mask.
        if binary.sum() == 0 and fallback_mask is not None:
            binary = (fallback_mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        try:
            verts, faces, _, _ = marching_cubes(
                binary.astype(np.float32),
                level=0.5,
                spacing=spacing,
            )
        except ValueError:
            if fallback_mask is not None and not np.array_equal(binary, (fallback_mask == cls_id).astype(np.uint8)):
                binary_fb = (fallback_mask == cls_id).astype(np.uint8)
                if binary_fb.sum() > 0:
                    try:
                        verts, faces, _, _ = marching_cubes(
                            binary_fb.astype(np.float32),
                            level=0.5,
                            spacing=spacing,
                        )
                    except ValueError:
                        continue
                else:
                    continue
            else:
                continue

        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=f"rgb({rgb})", opacity=opacity,
            name=CLASS_LABELS[cls_id], showlegend=True,
            flatshading=True,
        ))
    return traces


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def compute_metrics(state: dict) -> str:
    gt_mask   = state.get("gt_mask")
    pred_mask = state.get("pred_mask")
    if gt_mask is None or pred_mask is None:
        return "*Upload ground truth segmentation to see DSC metrics.*"

    rows = ["| Class | DSC |", "|:------|----:|"]
    dscs = []
    for cls_idx, name in CLASS_LABELS.items():
        gt_b   = (gt_mask == cls_idx).astype(float)
        pred_b = (pred_mask == cls_idx).astype(float)
        denom  = gt_b.sum() + pred_b.sum()
        dsc    = float(2.0 * (gt_b * pred_b).sum() / denom) if denom > 0 else float("nan")
        dscs.append(dsc)
        flag = " 🔴" if dsc < 0.5 else (" 🟡" if dsc < 0.75 else " 🟢")
        rows.append(f"| {name} | {dsc:.4f}{flag} |")

    mean = float(np.nanmean(dscs))
    rows.append(f"| **Mean** | **{mean:.4f}** |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------

def export_nifti(state: dict):
    pred_mask = state.get("pred_mask")
    if pred_mask is None:
        return None
    affine = state.get("affine", np.eye(4))
    tmp = tempfile.NamedTemporaryFile(suffix="_pred_seg.nii.gz", delete=False)
    tmp.close()
    nib.save(nib.Nifti1Image(pred_mask.astype(np.int16), affine), tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

HEADER = """
# 🧠 Brain Tumor Segmentation — Interactive Explorer
Upload MRI scans (T1, T1ce, T2, FLAIR) or pick a dataset case, then click **Run Inference**.
"""

with gr.Blocks(title="Brain Tumor Segmentation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)
    state = gr.State(value={})

    # ── Top bar: checkpoint ────────────────────────────────────────────────
    with gr.Row():
        ckpt_dd = gr.Dropdown(
            label="📦 Checkpoint",
            choices=scan_checkpoints(),
            value=scan_checkpoints()[0],
            scale=4,
        )
        refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

    # ── Input tabs ────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("📂 Upload NIfTI Files"):
            with gr.Row():
                t1_file    = gr.File(label="T1",    file_types=[".gz", ".nii"])
                t1ce_file  = gr.File(label="T1ce",  file_types=[".gz", ".nii"])
                t2_file    = gr.File(label="T2",    file_types=[".gz", ".nii"])
                flair_file = gr.File(label="FLAIR", file_types=[".gz", ".nii"])
            gt_file = gr.File(
                label="Ground Truth Segmentation (optional — enables DSC metrics)",
                file_types=[".gz", ".nii"],
            )

        with gr.Tab("🗂️ Pick Dataset Case"):
            cases = scan_cases()
            case_dd = gr.Dropdown(
                label="Case ID",
                choices=["— select a case —"] + cases,
                value="— select a case —",
            )

    # ── Run button ────────────────────────────────────────────────────────
    with gr.Row():
        run_btn    = gr.Button("▶ Run Inference", variant="primary", scale=3)
        status_txt = gr.Textbox(label="Status", interactive=False, scale=5)

    # ── Results (hidden until inference runs) ────────────────────────────
    with gr.Group(visible=False) as results_group:
        gr.Markdown("## Results")
        with gr.Row():
            with gr.Column(scale=3):
                slice_slider = gr.Slider(
                    label="Axial Slice", minimum=0, maximum=154,
                    step=1, value=77,
                )
                slice_image = gr.Image(label="Slice Overlay", type="numpy")
            with gr.Column(scale=1):
                metrics_md  = gr.Markdown("*Run inference to see metrics.*")
                export_file = gr.File(label="⬇ Download Prediction (.nii.gz)")

        gr.Markdown("## 3D Volume Render")
        plot_3d = gr.Plot(label="Interactive 3D Tumor")

    # ── Event wiring ──────────────────────────────────────────────────────

    refresh_btn.click(
        fn=lambda: gr.Dropdown(choices=scan_checkpoints()),
        outputs=[ckpt_dd],
    )

    def after_inference(s):
        depth = s.get("depth", 155)
        mid   = depth // 2
        return (
            gr.Group(visible=True),
            gr.Slider(maximum=depth - 1, value=mid),
        )

    run_btn.click(
        fn=lambda: gr.Button(interactive=False),
        outputs=[run_btn],
    ).then(
        fn=run_inference,
        inputs=[ckpt_dd, t1_file, t1ce_file, t2_file, flair_file,
                gt_file, case_dd, state],
        outputs=[state, status_txt],
    ).then(
        fn=after_inference,
        inputs=[state],
        outputs=[results_group, slice_slider],
    ).then(
        fn=render_slice,
        inputs=[state, slice_slider],
        outputs=[slice_image],
    ).then(
        fn=compute_metrics,
        inputs=[state],
        outputs=[metrics_md],
    ).then(
        fn=render_3d,
        inputs=[state],
        outputs=[plot_3d],
    ).then(
        fn=export_nifti,
        inputs=[state],
        outputs=[export_file],
    ).then(
        fn=lambda: gr.Button(interactive=True),
        outputs=[run_btn],
    )

    slice_slider.change(
        fn=render_slice,
        inputs=[state, slice_slider],
        outputs=[slice_image],
    )


if __name__ == "__main__":
    demo.queue(max_size=2)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
