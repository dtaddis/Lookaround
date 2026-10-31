"""Free-space carving: drop the Gaussians that some view sees straight through.

AnySplat places every pixel of every view at that view's own depth estimate. Across an H3
orbit those estimates don't quite agree, so each view's Gaussians end up floating in space the
other views can see through: the streaks and layers that fill a raw splat. Each view's depth
map says where its first visible surface is, so anything clearly in front of it, in the view's
line of sight, shouldn't be there. Gaussians hidden behind surfaces are left alone, so a
subject that is occluded in some views (or seen from behind) survives.
"""

import torch

# A Gaussian counts as "seen through" by a view when it is more than this fraction nearer
# than the surface the view sees at that pixel.
TOLERANCE = 0.02
# Drop a Gaussian once this many views see through it.
MIN_VOTES = 1


def free_space(means: torch.Tensor, cameras: list[dict], depth: torch.Tensor,
               tolerance: float = TOLERANCE, min_votes: int = MIN_VOTES) -> torch.Tensor:  # fmt: skip
    """Which Gaussians to keep: a [N] bool tensor on means' device.

    means is [N, 3] in AnySplat's world; cameras are splat.json entries (camera-to-world and
    intrinsics normalised by image size); depth is [views, H, W], each view's depth map.
    A Gaussian no view sees is dropped too.
    """
    means, depth = means.float(), depth.float().to(means.device)
    views, height, width = depth.shape
    seen = torch.zeros(means.shape[0], dtype=torch.int32, device=means.device)
    through = torch.zeros_like(seen)
    for camera, surface in zip(cameras, depth):
        w2c = torch.linalg.inv(torch.tensor(camera["c2w"], dtype=torch.float32, device=means.device))
        local = means @ w2c[:3, :3].T + w2c[:3, 3]
        z = local[:, 2]
        safe = z.clamp(min=1e-4)
        u = camera["fx"] * local[:, 0] / safe + camera["cx"]
        v = camera["fy"] * local[:, 1] / safe + camera["cy"]
        visible = (z > 1e-4) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        column = (u[visible] * width).long().clamp(0, width - 1)
        row = (v[visible] * height).long().clamp(0, height - 1)
        seen[visible] += 1
        through[visible] += (z[visible] < surface[row, column] * (1 - tolerance)).int()
    return (seen > 0) & (through < min_votes)
