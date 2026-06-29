# [Week 2] src/data/volume_to_frames.py
# Dependency: src/data/nifti_io.py (Week 1)
import os
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb


def write_frames_png(nifti_path, out_dir):
    """Convert every axial slice of a CT volume to a numbered PNG file on disk.

    Specification:
    - Load and HU-window the volume using Week 1 utilities.
    - Convert each (H, W) greyscale slice to (H, W, 3) RGB via to_rgb().
    - Save as PNG named 00000.png, 00001.png, … (5-digit zero-padded index = z).
    - Create out_dir if it does not exist.
    - SAM 2's video predictor reads this exact folder structure.

    Args:
        nifti_path (str | Path): Source .nii or .nii.gz file.
        out_dir    (str | Path): Destination folder for PNG frames.

    Returns:
        int: Number of PNGs written (equals the Z depth of the volume).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol, _, _ = load_volume(nifti_path)
    windowed = apply_hu_window(vol, as_uint8=True)

    for z in range(windowed.shape[2]):
        slice_2d = windowed[:, :, z]
        rgb = to_rgb(slice_2d)
        frame_path = out_dir / f"{z:05d}.png"
        Image.fromarray(rgb, mode="RGB").save(frame_path)

    return windowed.shape[2]


def frames_to_arrays(frames_dir):
    """Load all PNG frames from a folder into a single uint8 numpy array.

    Specification:
    - Read every .png in frames_dir in sorted (filename) order.
    - Stack along a new leading axis → shape (Z, H, W, 3).

    Args:
        frames_dir (str | Path): Folder produced by write_frames_png.

    Returns:
        np.ndarray: uint8 array of shape (Z, H, W, 3).
    """
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("*.png"))

    if not frame_paths:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    frames = [np.array(Image.open(path).convert("RGB"), dtype=np.uint8) for path in frame_paths]
    return np.stack(frames, axis=0)
