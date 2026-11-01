"""Build a Gaussian splat from orbit frames with AnySplat.

Run as: python -m lookaround.splat <frames_dir> <out_dir> --indices 0,2,4,...

The first index should be frame 0, the original photo: AnySplat puts its first view at the
world origin (OpenCV axes: x right, y down, z forward), so the viewer can start exactly where
the photo was taken. Gaussians that some view sees straight through are carved away
(carve.py), and with --subject-masks only the subject's pixels make Gaussians (subject.py).
Writes splat.ply and splat.json (cameras and framing for the viewer), and prints
"PROGRESS <percent> <message>" lines so the web app can show progress.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "AnySplat"))

import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import src.model.encoder.anysplat as anysplat_encoder  # noqa: E402
from src.model.model.anysplat import AnySplat  # noqa: E402
from src.model.ply_export import export_ply  # noqa: E402

from lookaround import carve, subject  # noqa: E402
from lookaround.frames import distinct_frames  # noqa: E402

# Half-width of the central window (in normalised image coordinates) whose Gaussians decide
# how far away the subject is.
CENTRE_WINDOW = 0.2
# AnySplat's own loader centre-crops to 448x448, which throws away a third of a landscape
# frame. Its VGGT backbone takes any size on its 14 px patch grid, so keep the whole frame:
# long side 448, short side rounded to the grid.
LONG_SIDE = 448
PATCH = 14


def load_frame(path: Path, long_side: int = LONG_SIDE) -> torch.Tensor:
    """One frame as a [3, H, W] tensor in [-1, 1], whole and on the patch grid."""
    image = Image.open(path).convert("RGB")
    scale = long_side / max(image.size)
    width, height = (max(PATCH, round(side * scale / PATCH) * PATCH) for side in image.size)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return TF.to_tensor(image) * 2.0 - 1.0


def square_pixels(focal: str) -> None:
    """Make AnySplat's cameras square-pixelled before it unprojects its depth maps.

    Its camera head predicts the horizontal and vertical fields of view separately, and was
    trained on square crops: on portrait frames the two disagree badly (Marilyn's horizontal
    focal length came out 0.68 of the vertical, in pixels), and the splat is built that much too
    wide. focal picks the one focal length both axes get: "height" (from the vertical field of
    view), "width", or "mean" (their geometric mean). On Marilyn all three made the photo view
    match the photo (26 dB, against 10.7 stretched); "height" scored best on held-out frames.
    """
    predict = anysplat_encoder.pose_encoding_to_extri_intri

    def patched(*args, **kwargs):
        extrinsic, intrinsic = predict(*args, **kwargs)
        if intrinsic is not None:
            fx, fy = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
            f = {"height": fy, "width": fx, "mean": (fx * fy).sqrt()}[focal]
            intrinsic = intrinsic.clone()
            intrinsic[..., 0, 0] = f
            intrinsic[..., 1, 1] = f
        return extrinsic, intrinsic

    anysplat_encoder.pose_encoding_to_extri_intri = patched


def keep_only_pixels(model: AnySplat, subject: torch.Tensor) -> None:
    """Make AnySplat build Gaussians only from the pixels marked in subject ([views, H, W]).

    Cameras and depth still come from the whole frames; only the per-pixel Gaussians of the
    background are dropped, before AnySplat merges each voxel's pixels into one Gaussian.
    """
    encoder = model.encoder
    fuse = type(encoder).voxelizaton_with_fusion

    def fuse_subject(img_feat, pts3d, voxel_size, conf=None):
        # The original flattens [views, C, H, W] to one row per pixel; hand it just the
        # subject's pixels as a single 1 x K "image".
        keep = subject.flatten()
        feats = img_feat.permute(1, 0, 2, 3).flatten(1)[:, keep][None, :, None]
        points = pts3d.permute(1, 0, 2, 3).flatten(1)[:, keep][None, :, None]
        return fuse(encoder, feats, points, voxel_size, conf=conf.flatten()[keep])

    encoder.voxelizaton_with_fusion = fuse_subject


def progress(percent: int, message: str) -> None:
    print(f"PROGRESS {percent} {message}", flush=True)


def orbit_centre(c2w: torch.Tensor) -> torch.Tensor | None:
    """The point closest to every camera's viewing axis: what the orbit turns around.

    Returns None when the cameras are too close to parallel to say (e.g. a single view).
    """
    origins, axes = c2w[:, :3, 3].double(), c2w[:, :3, 2].double()
    projectors = torch.eye(3, dtype=torch.float64) - axes[:, :, None] * axes[:, None, :]
    system = projectors.sum(dim=0)
    if torch.linalg.cond(system) > 1e4:
        return None
    return torch.linalg.solve(system, (projectors @ origins[:, :, None]).sum(dim=0))[:, 0]


def describe_cameras(means: torch.Tensor, c2w: torch.Tensor, intrinsics: torch.Tensor) -> dict:
    """Framing for the viewer: the first camera's field of view and the orbit's pivot.

    c2w is [views, 4, 4] camera-to-world; intrinsics is [views, 3, 3], normalised by image size.
    The pivot comes from the camera arc itself; without one it falls back to the depth of the
    Gaussians in the middle of the photo.
    """
    c2w, intrinsics = c2w.cpu(), intrinsics.cpu()
    fy = float(intrinsics[0, 1, 1])
    centre = orbit_centre(c2w)
    w2c = torch.linalg.inv(c2w[0])
    if centre is not None:
        pivot_depth = float((w2c[:3, :3].double() @ centre + w2c[:3, 3].double())[2])
    else:
        local = means.cpu() @ w2c[:3, :3].T + w2c[:3, 3]
        depth = local[:, 2]
        fx, cx, cy = (float(intrinsics[0, 0, 0]), float(intrinsics[0, 0, 2]),
                      float(intrinsics[0, 1, 2]))  # fmt: skip
        safe = depth.clamp(min=1e-3)
        u, v = fx * local[:, 0] / safe + cx, fy * local[:, 1] / safe + cy
        middle = (depth > 1e-3) & ((u - 0.5).abs() < CENTRE_WINDOW) & ((v - 0.5).abs() < CENTRE_WINDOW)
        pivot_depth = float(depth[middle].median() if middle.any() else depth[depth > 1e-3].median())
    return {
        "fov_y_degrees": math.degrees(2 * math.atan(0.5 / fy)),
        "pivot": None if centre is None else centre.tolist(),
        "subject_depth": pivot_depth,
        "cameras": [
            {"c2w": c2w[i].tolist(), "fx": float(k[0, 0]), "fy": float(k[1, 1]),
             "cx": float(k[0, 2]), "cy": float(k[1, 2])}  # fmt: skip
            for i, k in enumerate(intrinsics)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Gaussian splat from orbit frames.")
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--indices", required=True, help="Comma-separated frame numbers")
    parser.add_argument("--subject-masks", type=Path, metavar="DIR",
                        help="Keep only the subject, caching its masks in DIR")  # fmt: skip
    parser.add_argument("--long-side", type=int, default=LONG_SIDE,
                        help="Size AnySplat sees each frame at (a multiple of 14)")  # fmt: skip
    parser.add_argument("--focal", choices=["height", "width", "mean", "anysplat"], default="height",
                        help="Square AnySplat's pixels with this focal length, or keep its own "
                             "cameras (see square_pixels)")  # fmt: skip
    args = parser.parse_args()
    if args.focal != "anysplat":
        square_pixels(args.focal)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.frames_dir.glob("frame_*.png"))
    indices = [int(value) for value in args.indices.split(",")]
    chosen = [frames[i] for i in indices]
    progress(5, f"Selected {len(chosen)} of {len(frames)} orbit frames")

    masks = None
    if args.subject_masks:
        # Mask every distinct frame, not only the chosen ones: refine.py trains on all of them.
        progress(6, "Masking the subject in each frame")
        distinct = distinct_frames(args.frames_dir)
        every_mask = subject.subject_masks([frames[i] for i in distinct], args.subject_masks)
        masks = [every_mask[distinct.index(i)] for i in indices]

    device = torch.device("cuda")
    progress(10, "Loading AnySplat weights")
    model = AnySplat.from_pretrained("lhjiang/anysplat").to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    progress(30, "Preparing images")
    images = torch.stack([load_frame(path, args.long_side) for path in chosen]).unsqueeze(0)
    if masks:
        height, width = images.shape[-2:]
        pixels = [subject.mask_tensor(mask, (width, height)) for mask in masks]
        keep_only_pixels(model, torch.stack(pixels).to(device))
    # Each view's depth map, [1, views, H, W, 1], for free-space carving.
    depth = {}
    model.encoder.depth_head.register_forward_hook(
        lambda module, inputs, output: depth.update(map=output[0])
    )

    progress(40, "Reconstructing the scene")
    with torch.no_grad():
        gaussians, pose = model.inference((images.to(device) + 1) * 0.5)

    means = gaussians.means[0].float()
    meta = describe_cameras(means, pose["extrinsic"][0].float(), pose["intrinsic"][0].float())
    progress(85, "Removing Gaussians the other views see through")
    keep = carve.free_space(means, meta["cameras"], depth["map"][0, ..., 0])
    raw_count = int(means.shape[0])

    progress(90, "Writing splat.ply")
    # export_ply writes opacity as given, but PLY viewers apply a sigmoid to it (the 3DGS
    # convention), which would show every Gaussian at least half opaque. Hand it logits.
    opacities = torch.logit(gaussians.opacities[0][keep].float().clamp(1e-6, 1 - 1e-6))
    export_ply(
        gaussians.means[0][keep],
        gaussians.scales[0][keep],
        gaussians.rotations[0][keep],
        gaussians.harmonics[0][keep],
        opacities,
        args.out_dir / "splat.ply",
    )
    # A refined splat's copy of AnySplat's output is stale now.
    (args.out_dir / "splat_anysplat.ply").unlink(missing_ok=True)
    means = means[keep]
    meta.update(frames=[path.name for path in chosen], gaussians=int(means.shape[0]),
                raw_gaussians=raw_count, subject_only=bool(masks))  # fmt: skip
    (args.out_dir / "splat.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    progress(100, f"Done: {means.shape[0]:,} Gaussians (of {raw_count:,} before carving)")


if __name__ == "__main__":
    main()
