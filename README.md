# Lookaround

Step into a photo. Lookaround has a video model imagine everything the photo doesn't show by
filming a full 360° orbit around it, turns that orbit into a volumetric Gaussian splat, and
lets you look around the result in a web browser or a VR headset. Everything runs locally:

1. **Orbit.** MiniMax H3 with Pablo Dawson's 360° orbit LoRA films a camera orbit around the
   frozen scene, through a local ComfyUI (`lookaround/orbit.py`).
2. **Splat.** AnySplat reconstructs Gaussians, and a camera for every view, from a selection of
   the orbit's frames (`lookaround/splat.py`), with its cameras corrected to square pixels
   (it guesses the horizontal and vertical fields of view separately, and on portrait frames
   got them badly out of step, building the splat too wide). Gaussians that another view sees straight
   through are then carved away (`lookaround/carve.py`); optionally, only the subject is kept
   (`lookaround/subject.py`).
3. **Refine.** The splat is optimised against every orbit frame at full size, the usual 3DGS
   way (`lookaround/refine.py`, about two minutes). This is what makes it sharp.
4. **View.** A three.js + Spark viewer opens on the photo's own viewpoint (`web/viewer.html`)
   and lets you move freely: drag to turn any way round the subject (round the window's edge
   to roll), right-drag to pan, scroll to move in, WASD/QE to fly, and R to glide back to the
   photo. A WebXR button opens it in a headset, and **3D display** shows it glasses-free on an
   SR display (below).

## Install and run

Lookaround runs on Windows with an NVIDIA GPU (an RTX 4090 takes about 20 minutes per orbit).
You need [Git](https://git-scm.com/download/win); the installer gets everything else.

1. **`install.bat`** sets everything up inside the Lookaround folder (about 45 GB, mostly the
   H3 models). It asks where ComfyUI is: press Enter to have it cloned into `ComfyUI\` with its
   own venv, or give the folder of a ComfyUI you already have (then Lookaround uses it as it is,
   asks which Python runs it, and only adds what's missing). Run it again any time to update.
2. **`run.bat`** starts Lookaround and opens <http://127.0.0.1:8000>. If ComfyUI isn't already
   running, Lookaround starts it in the background (its log is `comfyui.log`) and stops it when
   you close the window.

Drop a photo: the orbit starts at once. When it is done, choose the views and press
**Build splat** (about three minutes with refining), then **Open viewer**. **All runs**
(`/runs.html`) lists every run, newest first, to reopen or view.

**Why carve?** AnySplat places every pixel of every view at that view's own depth guess. Across
an H3 orbit the guesses don't quite agree (and H3 redraws the background as it turns), so each
view's Gaussians float in space the other views look straight through: a raw whole-orbit splat
is a cloud of streaks and layers. Lookaround keeps each view's depth map and drops any Gaussian
that some view sees through, which typically removes half of them and leaves a solid subject.
Gaussians hidden behind something are kept, so a subject that is occluded part of the way round
survives.

**Subject only** cuts the subject out of every frame with rembg (masks are cached in
`runs/<id>/masks/`) and builds Gaussians only from its pixels. It works best when the subject
stays in view all the way round; where H3's camera passes behind something, the mask picks
whatever is in front and a few flecks of it survive.

**Why refine?** AnySplat sees 48 frames at 448 px and predicts the splat in one pass, so it is
soft. Refining optimises the Gaussians (with gsplat) against all ~150 frames at full size,
for 8,000 steps with gsplat's MCMC strategy: rather than splitting and culling Gaussians, it
keeps a fixed budget (1.5 times AnySplat's) and moves weak ones to where they are needed. On
held-out orbit frames this beat the usual densification on every test photo. Gaussians it left
nearly transparent are dropped when the splat is saved. From a poor AnySplat start (the Mona
Lisa, which H3 turned like a flat card, gave a near-black one) MCMC can fade the whole splat
out instead; when it keeps under a quarter of the Gaussians or barely improves on the start,
the refine is redone the usual way, densifying for 4,000 steps, which adds about two minutes.
AnySplat only gives cameras for the frames it saw, so the others are interpolated between them,
and each camera except the photo's learns a small correction, since H3 doesn't turn at a
perfectly even speed. Degree-1 spherical harmonics let colours shift slightly with the view,
which soaks up some of H3's drift. All the frames are filmed from one height, so looking from
far above or below the orbit shows needle-like Gaussians.

**Which views?** The whole orbit (48 frames) by default, for the full volume: on held-out
frames 48 beat 32, and 24 fell apart. Frames where H3's camera has stopped (it often pauses at
the start, and can arrive back at the photo well before the clip ends) are skipped by both
AnySplat and refining (`lookaround/frames.py`). Click frames to pick your own (Views: Custom);
the photo always stays, as the splat is anchored to it. The front arcs
(±30° to ±90° around the photo) are there for a front-only viewer.

## Glasses-free 3D displays

The viewer's **3D display** button shows the splat in true 3D on an SR (Simulated Reality)
display: Acer SpatialLabs monitors and laptops, the Samsung Odyssey 3D, and other eye-tracked
lenticular panels on the LeiaSR / Dimenco runtime. Put the browser window on the SR display
and press it: the page goes fullscreen there and the display's lens switches on.

It works the way [SR Loom](https://github.com/effcol/SR-Loom) does, without Acer's or
Samsung's 3D apps, and needs only the SR Platform runtime that comes with the display.
`lookaround/sr.py`, which the server runs while a viewer is in 3D mode, asks the runtime to
switch the lens on, reads the eye tracker, and has the runtime's weaver work out the lens phase
of every sub-pixel for each eye. The viewer renders a view for each eye, from where that eye
actually is, through the screen as a window (so the image shifts with your head), and weaves
the two itself in WebGL. The lens switches off a few seconds after you leave 3D mode.

If the two eyes' images look swapped (the scene looks inside out), tick **Swap eyes**. The
**Test** pattern (left eye red, right eye blue) checks it: close one eye and that eye's colour
should fill the screen. Made on an Acer Predator SpatialLabs View 27; the Odyssey 3D uses the
same runtime but hasn't been tried. If the runtime isn't where its installer registered it,
set `LOOKAROUND_SR_BIN` to the folder holding `SimulatedRealityCore.dll`.

## What install.bat sets up

- **uv**, if it isn't installed, to make the Python venvs.
- **`.venv`** (Python 3.11, PyTorch 2.5.1 for CUDA 12.4): the web app, AnySplat and the subject
  masks (`requirements.txt`).
- **`.venv-refine`** (Python 3.10, PyTorch 2.4.1): refining. gsplat's CUDA kernels come prebuilt
  for Windows only for Python 3.10, so refining has a venv of its own (`lookaround/refine.py`).
- **AnySplat's code** in `vendor/AnySplat`. Its weights (`lhjiang/anysplat`) download the first
  time you build a splat.
- **ComfyUI** (`lookaround/install.py`): cloned into `ComfyUI\` with its own venv (a current
  PyTorch for CUDA 12.8; AnySplat needs an older one), or an existing ComfyUI. Either way it
  gets the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) nodes and these models, which
  are downloaded only if they aren't already in its `models` folder (or in a folder its
  `extra_model_paths.yaml` names):

  | Folder | File | From |
  | --- | --- | --- |
  | `diffusion_models` | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` | Abiray/MiniMax-H3-Pruned-GGUF |
  | `text_encoders` | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | realrebelai/MiniMax-H3_GGUFs |
  | `vae` | `minimax_h3_video_vae_fp16.safetensors` | Comfy-Org/MiniMax-H3 |
  | `loras` | `minimax_h3_flf2v_lora_v1.safetensors` | pablodawson/MiniMax-H3-360-Orbit-LoRA |

- **`config.json`**: the ComfyUI folder, the Python that runs it, and `comfy_url` (change it if
  your ComfyUI isn't at <http://127.0.0.1:8188>). Before a splat build, Lookaround asks ComfyUI
  to unload its models, which it otherwise keeps in VRAM.

`install.bat --comfy-dir C:\path	o\ComfyUI --yes` installs without asking.

## Each run

`runs/<id>/` holds the photo (`input.*`), `orbit/frame_NNNN.png`, `orbit.mp4`, `masks/`,
`splat/splat.ply` (refined; AnySplat's own is `splat/splat_anysplat.ply`), `splat/splat.json`
(AnySplat's cameras, the photo's field of view and the orbit's pivot, for the viewer),
`state.json` and `job.log`. `runs/` is not part of the repository.

## Licences

- Lookaround source: Apache-2.0 (see `LICENSE`).
- MiniMax H3 weights and the orbit LoRA: the
  [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3), which excludes
  the EU, UK, South Korea and USA without separate authorisation from MiniMax, and reportedly
  forbids training other models on H3 outputs. Read it before publishing anything.
- AnySplat: MIT. Some of its utility code comes from NAVER under CC BY-NC-SA 4.0
  (non-commercial).
- `lookaround/orbit.py` and `lookaround/comfy.py` are adapted from HoloForge (Apache-2.0).

No model weights, photos or generated orbits are distributed with Lookaround.
