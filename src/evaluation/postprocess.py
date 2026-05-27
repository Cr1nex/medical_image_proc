"""
Post-processing for 3D segmentation predictions.

remove_small_components: per-class connected-component filtering.
Removes isolated voxel blobs (spurious false positives) below a minimum
size threshold, keeping all components at or above the threshold so that
multi-lobe edema is preserved.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 50,
    classes: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """
    Remove connected components smaller than *min_size* voxels for each class.

    Args:
        mask:     Integer label map, shape [H, W, D].
        min_size: Components strictly smaller than this are zeroed out.
        classes:  Which label values to clean (background=0 is skipped).

    Returns:
        Cleaned integer label map, same shape and dtype as input.
    """
    out = mask.copy()
    for cls in classes:
        binary = (mask == cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, n = ndi.label(binary)
        if n == 0:
            continue
        sizes = ndi.sum(binary, labeled, index=np.arange(1, n + 1))
        for comp_label, size in enumerate(sizes, start=1):
            if size < min_size:
                out[labeled == comp_label] = 0
    return out
