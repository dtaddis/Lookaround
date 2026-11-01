"""Refine an AnySplat splat against every frame of the orbit, at full resolution.

Run as: python -m lookaround.refine <splat_dir> <frames_dir> [--masks DIR] [--steps N]
(in .venv-refine, which has a prebuilt gsplat; see README.md).

AnySplat sees 48 frames at 448 px and predicts Gaussians in one pass, so the splat is soft.
This optimises those Gaussians photometrically, the usual 3DGS way, against all of the orbit's
frames at full size, with gsplat's MCMC strategy: a fixed budget of Gaussians that are moved,
not split and culled, which scored best on held-out orbit frames. AnySplat only gave cameras for the frames it saw; the rest are
interpolated between them, and every camera but the photo's gets a small learned correction
because H3 doesn't turn at a perfectly even speed. Spherical harmonics of degree 1 let a
Gaussian's colour change a little with the view, which absorbs some of H3's drift.

With --masks (the subject masks, one per orbit frame) the background is trained to be empty.
Gaussians left nearly transparent are dropped on saving: MCMC keeps them as spares, and they
can be half the file. If MCMC collapses (from a poor AnySplat start it can fade the whole splat
out), the refine is redone with gsplat's default densification for 4,000 steps. Rewrites splat_dir/splat.ply, keeping AnySplat's version as splat_anysplat.ply, and prints
"PROGRESS <percent> <message>" lines.
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from PIL import Image
from plyfile import PlyData, PlyElement

from lookaround.frames import distinct_frames

SH_DEGREE = 1
# Adam learning rates, as in gsplat's simple_trainer (means scale with the scene).
LEARNING_RATES = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
                  "sh0": 2.5e-3, "shN": 2.5e-3 / 20}  # fmt: skip
# Scored on held-out frames of the portrait run: 3e-5 beat 1e-4, 1e-3 and no correction.
POSE_LEARNING_RATE = 3e-5
SSIM_WEIGHT = 0.2
# Weight of the alpha-matches-mask term when keeping only the subject.
MASK_WEIGHT = 0.5
# MCMC: at most this many times AnySplat's Gaussians, and light pressure towards fewer, smaller
# ones (as in the MCMC paper). Scored on held-out frames of three runs.
MCMC_CAP = 1.5
OPACITY_REG = 0.01
SCALE_REG = 0.01
# Saved Gaussians below this opacity are dropped: they don't change the picture.
MIN_OPACITY = 0.01
# MCMC has collapsed if it kept fewer than this share of AnySplat's Gaussians above MIN_OPACITY
# (good runs keep 70-86%; the collapsed Mona Lisa kept 1.3%), or if its renders of a sample of
# frames score less than this much above the start's. Then the refine is redone by densifying.
MIN_KEPT = 0.25
MIN_GAIN_DB = 0.5
DENSIFY_STEPS = 4000
SH_C0 = 0.28209479177387814


def progress(percent: int, message: str) -> None:
    print(f"PROGRESS {percent} {message}", flush=True)


# --- Cameras ------------------------------------------------------------------------------


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to a unit quaternion (w, x, y, z)."""
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


def orbit_cameras(meta: dict, distinct: list[int]) -> list[dict]:
    """A camera for each distinct orbit frame, interpolating between the ones AnySplat saw.

    Frames are spaced by their place among the distinct frames, so pauses don't count. The
    orbit is a loop back to the photo, so the frames after the last known one are interpolated
    towards frame 0's camera.
    """
    period = len(distinct)
    known = {distinct.index(int(name[6:10])): camera
             for name, camera in zip(meta["frames"], meta["cameras"])}  # fmt: skip
    order = sorted(known)
    cameras = []
    for frame in range(period):
        if frame in known:
            cameras.append(known[frame])
            continue
        before = max(i for i in order if i < frame)
        after = next((i for i in order if i > frame), period)
        t = (frame - before) / (after - before)
        a, b = known[before], known[after % period]
        ma, mb = np.array(a["c2w"]), np.array(b["c2w"])
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_matrix(slerp(matrix_to_quat(ma[:3, :3]), matrix_to_quat(mb[:3, :3]), t))
        c2w[:3, 3] = (1 - t) * ma[:3, 3] + t * mb[:3, 3]
        cameras.append({"c2w": c2w.tolist(),
                        **{k: (1 - t) * a[k] + t * b[k] for k in ("fx", "fy", "cx", "cy")}})  # fmt: skip
    return cameras


def axis_angle_matrix(vector: torch.Tensor) -> torch.Tensor:
    """[V, 3] rotation vectors to [V, 3, 3] rotation matrices (Rodrigues)."""
    angle = vector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = vector / angle
    zero = torch.zeros_like(k[:, 0])
    cross = torch.stack([zero, -k[:, 2], k[:, 1], k[:, 2], zero, -k[:, 0],
                         -k[:, 1], k[:, 0], zero], dim=-1).view(-1, 3, 3)  # fmt: skip
    eye = torch.eye(3, device=vector.device).expand_as(cross)
    sin, cos = angle.sin()[..., None], angle.cos()[..., None]
    return eye + sin * cross + (1 - cos) * (cross @ cross)


# --- Splat in and out ---------------------------------------------------------------------


def load_ply(path: Path, device: str, sh_degree: int = SH_DEGREE) -> dict[str, torch.Tensor]:
    """Trainable parameters from a 3DGS PLY (log scales, logit opacities, wxyz quaternions)."""
    vertex = PlyData.read(str(path))["vertex"].data

    def columns(*names: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([vertex[n] for n in names], axis=1).astype(np.float32))

    sh0 = columns("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    rest = (sh_degree + 1) ** 2 - 1
    return {
        "means": columns("x", "y", "z").to(device),
        "scales": columns("scale_0", "scale_1", "scale_2").to(device),
        "quats": columns("rot_0", "rot_1", "rot_2", "rot_3").to(device),
        "opacities": columns("opacity")[:, 0].to(device),
        "sh0": sh0.to(device),
        "shN": torch.zeros(len(sh0), rest, 3).to(device),
    }


def save_ply(params: dict[str, torch.Tensor], path: Path) -> None:
    means = params["means"].detach().cpu().numpy()
    rest = params["shN"].detach().transpose(1, 2).flatten(1).cpu().numpy()  # channel-major
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])  # fmt: skip
    values = np.concatenate([
        means, np.zeros_like(means), params["sh0"].detach()[:, 0].cpu().numpy(), rest,
        params["opacities"].detach()[:, None].cpu().numpy(),
        params["scales"].detach().cpu().numpy(),
        F.normalize(params["quats"].detach(), dim=-1).cpu().numpy(),
    ], axis=1).astype(np.float32)  # fmt: skip
    elements = np.empty(len(values), dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        elements[name] = values[:, i]
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


# --- Loss ---------------------------------------------------------------------------------


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of two [1, 3, H, W] images (11 px Gaussian window)."""
    coords = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    window = (g[:, None] * g[None, :] / g.sum() ** 2).expand(3, 1, 11, 11)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=5, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2))
            / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).mean()  # fmt: skip


# --- Main ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a splat against the whole orbit.")
    parser.add_argument("splat_dir", type=Path)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--masks", type=Path, help="Subject masks: train the background empty")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--sh-degree", type=int, default=SH_DEGREE)
    parser.add_argument("--pose-lr", type=float, default=POSE_LEARNING_RATE,
                        help="Learning rate of the camera corrections (0 turns them off)")  # fmt: skip
    parser.add_argument("--holdout", type=int, default=0, metavar="N",
                        help="For tuning: train without every Nth frame and score the splat on those")  # fmt: skip
    args = parser.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    progress(1, "Loading the splat and the orbit")
    meta = json.loads((args.splat_dir / "splat.json").read_text(encoding="utf-8"))
    every_frame = sorted(args.frames_dir.glob("frame_*.png"))
    distinct = distinct_frames(args.frames_dir)
    frames = [every_frame[i] for i in distinct]
    cameras = orbit_cameras(meta, distinct)
    width, height = Image.open(frames[0]).size
    images = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))) for f in frames])
    images = images.to(device)  # uint8, [V, H, W, 3]
    masks = None
    if args.masks:
        masks = torch.stack([
            torch.from_numpy(np.asarray(Image.open(args.masks / f.name).convert("L").resize((width, height))))
            for f in frames
        ]).to(device)  # fmt: skip

    c2w = torch.tensor([c["c2w"] for c in cameras], dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w)
    ks = torch.zeros(len(cameras), 3, 3, device=device)
    ks[:, 0, 0] = torch.tensor([c["fx"] for c in cameras]) * width
    ks[:, 1, 1] = torch.tensor([c["fy"] for c in cameras]) * height
    ks[:, 0, 2] = torch.tensor([c["cx"] for c in cameras]) * width
    ks[:, 1, 2] = torch.tensor([c["cy"] for c in cameras]) * height
    ks[:, 2, 2] = 1
    scene_scale = float((c2w[:, :3, 3] - c2w[:, :3, 3].mean(0)).norm(dim=-1).mean())

    # Always start from AnySplat's own splat, so refining twice doesn't refine a refinement.
    original = args.splat_dir / "splat_anysplat.ply"
    if not original.exists():
        shutil.copyfile(args.splat_dir / "splat.ply", original)
    held_out = set(range(args.holdout, len(frames), args.holdout)) if args.holdout else set()
    training = [i for i in range(len(frames)) if i not in held_out]
    # Frames the result is checked against for a collapse (see below).
    check_views = training[:: max(1, len(training) // 16)]

    def render(params, pose, view: int, sh_degree: int, absgrad: bool = False):
        viewmat = w2c[view : view + 1]
        if view != 0:
            # A held-out frame never learns its own correction; borrow its neighbours'.
            delta = pose[view] if view not in held_out else (pose[view - 1] + pose[(view + 1) % len(frames)]) / 2
            correction = torch.eye(4, device=device)
            correction[:3, :3] = axis_angle_matrix(delta[None, :3])[0]
            correction[:3, 3] = delta[3:] * scene_scale
            viewmat = correction @ viewmat
        colours = torch.cat([params["sh0"], params["shN"]], dim=1)
        image, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), params["scales"].exp(),
            params["opacities"].sigmoid(), colours, viewmat, ks[view : view + 1], width, height,
            sh_degree=sh_degree, absgrad=absgrad, packed=False,
        )  # fmt: skip
        return image[0].permute(2, 0, 1)[None], alpha, info

    def target_for(view: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        target = images[view].permute(2, 0, 1)[None].float() / 255
        if masks is None:
            return target, None
        matte = masks[view][None, None].float() / 255
        return target * matte, matte

    def psnr_ssim(params, pose, views, sh_degree: int) -> tuple[float, float]:
        """Mean PSNR and SSIM of the splat's renders of these frames."""
        scores = []
        with torch.no_grad():
            for view in views:
                image = render(params, pose, view, sh_degree)[0].clamp(0, 1)
                target, _ = target_for(view)
                psnr = -10 * torch.log10(((image - target) ** 2).mean())
                scores.append((float(psnr), float(ssim(image, target))))
        return tuple(np.mean(scores, axis=0))

    def score(label: str, params, pose, sh_degree: int) -> None:
        """Print the mean PSNR and SSIM over the held-out frames."""
        psnr, similarity = psnr_ssim(params, pose, sorted(held_out), sh_degree)
        print(f"HELDOUT {label}: {len(held_out)} frames, PSNR {psnr:.2f} dB, SSIM {similarity:.4f}",
              flush=True)  # fmt: skip

    def refine(method: str, steps: int, first: int, last: int):
        """Optimise a fresh copy of AnySplat's splat, with MCMC ("mcmc") or gsplat's default
        densification ("densify", Lookaround's refine before MCMC). Reports progress from first
        to last percent. Returns the parameters and the camera corrections."""
        params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in load_ply(original, device, args.sh_degree).items()}
        )
        optimizers = {
            name: torch.optim.Adam([params[name]], lr=lr * (scene_scale if name == "means" else 1),
                                   eps=1e-15)  # fmt: skip
            for name, lr in LEARNING_RATES.items()
        }
        means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], 0.01 ** (1 / steps))
        # Pose corrections: rotation vector and translation per camera; the photo's stays put.
        pose = torch.nn.Parameter(torch.zeros(len(cameras), 6, device=device))
        pose_optimizer = torch.optim.Adam([pose], lr=args.pose_lr or 1e-12)
        mcmc = method == "mcmc"
        if mcmc:
            strategy = MCMCStrategy(cap_max=int(len(params["means"]) * MCMC_CAP), refine_start_iter=300,
                                    refine_stop_iter=int(steps * 0.6), refine_every=100)  # fmt: skip
            state = strategy.initialize_state()
        else:
            strategy = DefaultStrategy(
                refine_start_iter=300, refine_stop_iter=int(steps * 0.6), refine_every=100,
                reset_every=steps * 10, absgrad=True, grow_grad2d=0.0008, verbose=False,
            )  # fmt: skip
            state = strategy.initialize_state(scene_scale=scene_scale)
        strategy.check_sanity(params, optimizers)

        order: list[int] = []
        for step in range(steps):
            if not order:
                order = random.sample(training, len(training))
            view = order.pop()
            image, alpha, info = render(params, pose, view, min(args.sh_degree, step // 1000), absgrad=not mcmc)
            target, matte = target_for(view)
            loss = (1 - SSIM_WEIGHT) * (image - target).abs().mean() + SSIM_WEIGHT * (1 - ssim(image, target))
            if matte is not None:
                loss = loss + MASK_WEIGHT * (alpha[0, ..., 0] - matte[0, 0]).abs().mean()
            if mcmc:
                loss = (loss + OPACITY_REG * params["opacities"].sigmoid().mean()
                        + SCALE_REG * params["scales"].exp().mean())  # fmt: skip

            strategy.step_pre_backward(params, optimizers, state, step, info)
            loss.backward()
            for optimizer in [*optimizers.values(), pose_optimizer]:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            means_decay.step()
            if mcmc:
                strategy.step_post_backward(params, optimizers, state, step, info,
                                            lr=optimizers["means"].param_groups[0]["lr"])  # fmt: skip
            else:
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

            if step % 100 == 0 or step == steps - 1:
                progress(first + int((last - first) * (step + 1) / steps),
                         f"Refining{'' if mcmc else ' again (densifying)'}: step {step + 1:,} of {steps:,}, "
                         f"{len(params['means']):,} Gaussians, loss {loss.item():.3f}")  # fmt: skip

        if mcmc:
            with torch.no_grad():
                visible = params["opacities"].sigmoid() >= MIN_OPACITY
                for name in params:
                    params[name] = torch.nn.Parameter(params[name][visible])
        return params, pose

    start = load_ply(original, device, args.sh_degree)
    start_count = len(start["means"])
    no_pose = torch.zeros(len(cameras), 6, device=device)
    if held_out:
        score("before refining", start, no_pose, 0)
    start_psnr = psnr_ssim(start, no_pose, check_views, 0)[0]
    del start
    progress(5, f"Refining {start_count:,} Gaussians against {len(training)} frames")
    params, pose = refine("mcmc", args.steps, 5, 95)
    method, steps = "mcmc", args.steps

    # From a poor start (AnySplat sometimes makes a near-black splat, as of a painting H3 turned
    # like a card) MCMC can fade everything out instead of fixing it. The old densification
    # recovers more from there, so if MCMC kept few Gaussians or came out no better than its
    # start, refine again that way.
    psnr = psnr_ssim(params, pose, check_views, args.sh_degree)[0]
    kept = len(params["means"]) / start_count
    if kept < MIN_KEPT or psnr < start_psnr + MIN_GAIN_DB:
        print(f"MCMC kept {kept:.1%} of the Gaussians and scored {psnr:.2f} dB against the start's "
              f"{start_psnr:.2f} dB on the check frames: refining again by densifying", flush=True)  # fmt: skip
        method, steps = "densify", DENSIFY_STEPS
        params, pose = refine(method, steps, 60, 95)
    if held_out:
        score("after refining", params, pose, args.sh_degree)

    progress(96, "Writing splat.ply")
    save_ply(params, args.splat_dir / "splat.ply")  # gsplat's quaternions are wxyz, as in PLY
    meta.update(gaussians=len(params["means"]), refined_steps=steps, refined_frames=len(frames),
                refine_method=method)  # fmt: skip
    (args.splat_dir / "splat.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    progress(100, f"Done: {len(params['means']):,} Gaussians after refining")


if __name__ == "__main__":
    main()
