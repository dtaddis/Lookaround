"""Choosing which orbit frames to give AnySplat."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Frames are compared as small greyscale thumbnails, by their mean absolute difference (0-255).
THUMB_SIZE = (160, 90)
# Below this a frame shows nothing its predecessor didn't: H3 often holds still for a few
# frames at the start, and moving frames differ by 2 or more.
STILL = 1.0
# Below this a late frame is back at the photo: the orbit has closed its loop. H3 sometimes
# gets there well before the clip ends and parks on the photo for the rest.
HOME = 3.0
CACHE = "distinct.json"


def _thumb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(THUMB_SIZE), dtype=np.float32)


def distinct_frames(frames_dir: Path) -> list[int]:
    """The orbit frames that show a new view, in order, starting with the photo (frame 0).

    Pauses are left out, and so is everything from the moment the orbit is back at the photo.
    The result is cached in the frames folder.
    """
    cache = frames_dir / CACHE
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    paths = sorted(frames_dir.glob("frame_*.png"))
    thumbs = [_thumb(path) for path in paths]
    differs = lambda a, b: float(np.abs(thumbs[a] - thumbs[b]).mean())  # noqa: E731
    kept = [0]
    for i in range(1, len(thumbs)):
        if differs(i, 0) < HOME and i > len(thumbs) // 2:
            break  # back where it started
        if differs(i, kept[-1]) >= STILL:
            kept.append(i)
    cache.write_text(json.dumps(kept), encoding="utf-8")
    return kept


def pick_indices(distinct: list[int], count: int) -> list[int]:
    """Evenly spaced frames over the whole orbit."""
    if not 2 <= count <= len(distinct):
        raise ValueError(f"Choose between 2 and {len(distinct)} frames for this orbit")
    step = len(distinct) / count
    return [distinct[round(i * step)] for i in range(count)]


def arc_frame_count(distinct: list[int], degrees: float) -> int:
    """How many distinct frames lie within +/-degrees of the photographed viewpoint."""
    period = len(distinct)
    reach = round(degrees / 360 * period)
    return min(period, 2 * reach + 1)


def pick_arc(distinct: list[int], degrees: float, count: int) -> list[int]:
    """Frames spread over +/-degrees around the photographed viewpoint (frame 0).

    The distinct frames make one full circle, so the k-th is about 360 * k / len(distinct)
    degrees round, and negative angles come from the end. The LoRA's speed is not perfectly
    even, so angles are approximate. Frame 0 comes first: AnySplat anchors the world to its
    first view.
    """
    if not 0 < degrees < 180:
        raise ValueError("The arc must be under 180 degrees each side; use the full orbit instead")
    available = arc_frame_count(distinct, degrees)
    if not 2 <= count <= available:
        raise ValueError(f"Choose between 2 and {available} frames for a +/-{degrees:g} degree arc")
    period = len(distinct)
    angles = sorted((-degrees + 2 * degrees * i / (count - 1) for i in range(count)), key=abs)
    angles[0] = 0.0  # the nearest angle to the photo becomes the photo itself
    indices = []
    for angle in angles:
        index = distinct[round(angle / 360 * period) % period]
        if index not in indices:
            indices.append(index)
    return indices


def check_custom(distinct: list[int], indices: list[int]) -> list[int]:
    """A hand-picked selection, in orbit order with the photo first."""
    chosen = sorted(set(indices))
    if 0 not in chosen:
        raise ValueError("Keep the photo (frame 0): the splat is anchored to it")
    if len(chosen) < 2:
        raise ValueError("Choose at least two frames")
    still = [i for i in chosen if i not in distinct]
    if still:
        raise ValueError(f"Frame {still[0]} is a still frame (or not in the orbit); choose another")
    return chosen
