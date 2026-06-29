# [Week 2] src/data/prompts.py
import numpy as np
from scipy import ndimage


def bbox_from_mask(mask2d, pad=4):
    """Return a SAM 2-format bounding box around the foreground of a 2-D mask.

    Specification:
    - Find all True/nonzero pixels; return None if there are none.
    - Compute the tight axis-aligned bounding box (min/max row and column).
    - Expand every side by pad pixels, clamped to image bounds.
    - Return format: float32 array [x0, y0, x1, y1]  (x = column, y = row).

    Args:
        mask2d (np.ndarray): 2-D bool or uint8 mask, shape (H, W).
        pad    (int):        Extra pixels on every side of the tight box.

    Returns:
        np.ndarray | None: float32 [x0, y0, x1, y1], or None if mask is empty.
    """
<<<<<<< HEAD
    mask2d = np.asarray(mask2d)
    ys, xs = np.nonzero(mask2d)
    if len(xs) == 0 or len(ys) == 0:
        return None

    y0 = max(0, ys.min() - pad)
    y1 = min(mask2d.shape[0] - 1, ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(mask2d.shape[1] - 1, xs.max() + pad)

    return np.array([x0, y0, x1, y1], dtype=np.float32)
=======
    pass
>>>>>>> 1a232303f6b873212e1640fd035fe2c6fa513cdc


def best_start_slice(label_vol, organ_id):
    """Return the axial index of the slice with the largest organ cross-section.

    Specification:
    - Binarise label_vol to the target organ (label_vol == organ_id).
    - Raise ValueError if the organ is absent.
    - For each z-slice compute the foreground pixel count.
    - Return the z index with the maximum count (argmax across the Z axis).
    - Hint: scipy.ndimage can help identify connected components if needed.

    Args:
        label_vol (np.ndarray): Integer label volume, shape (H, W, Z).
        organ_id  (int):        BTCV label integer for the target organ.

    Returns:
        int: Z-axis index of the slice with the most organ pixels.

    Raises:
        ValueError: If organ_id is not present anywhere in label_vol.
    """
<<<<<<< HEAD
    organ_mask = (np.asarray(label_vol) == organ_id)
    if not np.any(organ_mask):
        raise ValueError(f"Organ id {organ_id} is not present in the label volume")

    slice_counts = np.sum(organ_mask, axis=(0, 1))
    return int(np.argmax(slice_counts))
=======
    pass
>>>>>>> 1a232303f6b873212e1640fd035fe2c6fa513cdc
