"""Choosing which orbit frames to give AnySplat."""

# The orbit ends where it began, so its last few frames repeat the first ones.
DROP_TAIL = 4


def pick_indices(total: int, count: int) -> list[int]:
    """Evenly spaced frame indices over the whole orbit, skipping the repeated tail."""
    usable = total - DROP_TAIL
    if not 2 <= count <= usable:
        raise ValueError(f"Choose between 2 and {usable} frames for this {total}-frame orbit")
    step = usable / count
    return [round(i * step) for i in range(count)]


def arc_frame_count(total: int, degrees: float) -> int:
    """How many distinct frames lie within +/-degrees of the photographed viewpoint."""
    period = total - 1
    reach = round(degrees / 360 * period)
    return min(period, 2 * reach + 1)


def pick_arc(total: int, degrees: float, count: int) -> list[int]:
    """Frame indices spread over +/-degrees around the photographed viewpoint (frame 0).

    The orbit is one full circle over total - 1 frame steps (its last frame repeats the
    first), so frame i is about 360 * i / (total - 1) degrees round, and negative angles come
    from the end of the clip. The LoRA's speed is not perfectly even, so angles are
    approximate. Frame 0 comes first: AnySplat anchors the world to its first view.
    """
    if not 0 < degrees < 180:
        raise ValueError("The arc must be under 180 degrees each side; use the full orbit instead")
    available = arc_frame_count(total, degrees)
    if not 2 <= count <= available:
        raise ValueError(f"Choose between 2 and {available} frames for a +/-{degrees:g} degree arc")
    period = total - 1
    angles = sorted((-degrees + 2 * degrees * i / (count - 1) for i in range(count)), key=abs)
    angles[0] = 0.0  # the nearest angle to the photo becomes the photo itself
    indices = []
    for angle in angles:
        index = round(angle / 360 * period) % period
        if index not in indices:
            indices.append(index)
    return indices
