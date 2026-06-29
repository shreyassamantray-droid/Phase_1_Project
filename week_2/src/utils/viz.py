# [Week 2] src/utils/viz.py
import numpy as np
from pathlib import Path
import imageio.v2 as imageio


def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    """Write an animated GIF overlaying segmentation masks onto the CT volume.

    Specification:
    - Iterate over every axial slice (Z axis) of vol_u8.
    - Convert each greyscale (H, W) slice to RGB by repeating the channel.
    - Where pred3d[:, :, z] is True, apply a RED tint  (blend 50 % with [220,60,60]).
    - Where gt3d[:, :, z] is True (if gt3d is not None), apply a GREEN tint
      (blend 50 % with [60,220,60]).  GT is rendered beneath the prediction.
    - Collect all (H, W, 3) uint8 frames and write them as an animated GIF.
    - Create the parent directory of out_path if it does not exist.
    - Recommended frame duration: 80 ms.

    Args:
        vol_u8   (np.ndarray):      uint8 CT volume, shape (H, W, Z).
        pred3d   (np.ndarray):      bool/uint8 predicted mask, shape (H, W, Z).
        out_path (str | Path):      Destination .gif path.
        gt3d     (np.ndarray|None): Ground-truth mask same shape, or None.
    """
    vol_u8 = np.asarray(vol_u8)
    pred3d = np.asarray(pred3d)
    gt3d = np.asarray(gt3d) if gt3d is not None else None

    if vol_u8.ndim != 3:
        raise ValueError("vol_u8 must be a 3D array of shape (H, W, Z)")

    if pred3d.shape != vol_u8.shape:
        raise ValueError("pred3d must have the same shape as vol_u8")

    if gt3d is not None and gt3d.shape != vol_u8.shape:
        raise ValueError("gt3d must have the same shape as vol_u8")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    red = np.array([220, 60, 60], dtype=np.uint8)
    green = np.array([60, 220, 60], dtype=np.uint8)
    frames = []

    for z in range(vol_u8.shape[2]):
        gray = vol_u8[:, :, z]
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)

        if gt3d is not None:
            gt_mask = np.asarray(gt3d[:, :, z], dtype=bool)
            if np.any(gt_mask):
                rgb[gt_mask] = np.where(rgb[gt_mask] > 127, (rgb[gt_mask] + green) // 2, green)

        pred_mask = np.asarray(pred3d[:, :, z], dtype=bool)
        if np.any(pred_mask):
            rgb[pred_mask] = np.where(rgb[pred_mask] > 127, (rgb[pred_mask] + red) // 2, red)

        frames.append(rgb)

    imageio.mimsave(out_path, frames, duration=0.08)
    return out_path
