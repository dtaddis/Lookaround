"""Subject masks, for building a splat of the subject alone.

rembg cuts the subject out of each frame; splat.py then has AnySplat build Gaussians only from
the subject's pixels. Free-space carving (carve.py) cleans up the rest: a Gaussian that lands
on the background in some view is in front of that view's background, which the view sees
straight through.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# rembg model for the subject masks. isnet-general-use is about 180 MB and quick on the CPU.
MASK_MODEL = "isnet-general-use"
# A mask pixel counts as subject above this value (0-255).
MASK_THRESHOLD = 128


def subject_masks(frames: list[Path], out_dir: Path) -> list[Image.Image]:
    """A greyscale subject mask for each frame, saved to out_dir as <frame name>.png.

    Masks already in out_dir are reused, so rebuilding a splat doesn't repeat the work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    session = None
    masks = []
    for frame in frames:
        path = out_dir / frame.name
        if path.exists():
            masks.append(Image.open(path).convert("L"))
            continue
        if session is None:
            from rembg import new_session, remove

            session = new_session(MASK_MODEL)
        mask = remove(Image.open(frame).convert("RGB"), session=session, only_mask=True)
        mask.save(path)
        masks.append(mask)
    return masks


def mask_tensor(mask: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """A [H, W] bool tensor of the subject's pixels at size (width, height)."""
    return torch.from_numpy(np.asarray(mask.resize(size)) >= MASK_THRESHOLD)
