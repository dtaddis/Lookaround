# Lookaround

Step into a photo. Lookaround has a video model imagine everything the photo doesn't show by
filming a full 360° orbit around it, turns that orbit into a volumetric Gaussian splat, and
lets you look around the result in a web browser or a VR headset. Everything runs locally:

1. **Orbit.** MiniMax H3 with Pablo Dawson's 360° orbit LoRA films a camera orbit around the
   frozen scene, through a local ComfyUI (`lookaround/orbit.py`).
2. **Splat.** AnySplat reconstructs Gaussians, and a camera for every view, from a selection of
   the orbit's frames (`lookaround/splat.py`). Gaussians that another view sees straight
   through are then carved away (`lookaround/carve.py`); optionally, only the subject is kept
   (`lookaround/subject.py`).
3. **Refine.** The splat is optimised against every orbit frame at full size, the usual 3DGS
   way (`lookaround/refine.py`, about a minute). This is what makes it sharp.
4. **View.** A three.js + Spark viewer opens on the photo's own viewpoint and orbits the subject
   (`web/viewer.html`), with a WebXR button for headsets.

## Run it

Start ComfyUI, then double-click `Lookaround.bat` (or run
`.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000`) and open
<http://127.0.0.1:8000>. Drop a photo: the orbit starts at once and takes about 20 minutes on
an RTX 4090. When it is done, choose the views and press **Build splat** (about two minutes
with refining), then **Open viewer**.

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

**Why refine?** AnySplat sees 32 frames at 448 px and predicts the splat in one pass, so it is
soft. Refining optimises the Gaussians (with gsplat) against all ~150 frames at full size.
AnySplat only gives cameras for the frames it saw, so the others are interpolated between them,
and each camera except the photo's learns a small correction, since H3 doesn't turn at a
perfectly even speed. Degree-1 spherical harmonics let colours shift slightly with the view,
which soaks up some of H3's drift. All the frames are filmed from one height, so looking from
far above or below the orbit shows needle-like Gaussians; the viewer limits the tilt to ±30°.

**Which views?** The whole orbit (32 frames) by default, for the full volume. The front arcs
(±30° to ±90° around the photo) are there for a front-only viewer.

## Set up

- **ComfyUI** with the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) nodes and these
  models:

  | Folder | File | From |
  | --- | --- | --- |
  | `diffusion_models` | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` | Abiray/MiniMax-H3-Pruned-GGUF |
  | `text_encoders` | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | realrebelai/MiniMax-H3_GGUFs |
  | `vae` | `minimax_h3_video_vae_fp16.safetensors` | Comfy-Org/MiniMax-H3 |
  | `loras` | `minimax_h3_flf2v_lora_v1.safetensors` | pablodawson/MiniMax-H3-360-Orbit-LoRA |

- **AnySplat code:** `git clone https://github.com/InternRobotics/AnySplat vendor/AnySplat`.
  Its weights (`lhjiang/anysplat`) download on first use.
- **Python 3.11 venv** (CUDA 12.4 wheels; the CUDA 12.4 toolkit and MSVC are only needed if
  gsplat has to compile):

  ```
  uv venv --python 3.11 .venv
  uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python .venv\Scripts\python.exe xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python .venv\Scripts\python.exe torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  ```

- **Refine venv.** gsplat's CUDA kernels come prebuilt for Windows only for Python 3.10 (building
  them needs MSVC 2022, and CUDA 12.4 can't build with Visual Studio 18), so refining runs in
  a second venv. Without it, the Refine option is greyed out:

  ```
  uv venv --python 3.10 .venv-refine
  uv pip install --python .venv-refine\Scripts\python.exe torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python .venv-refine\Scripts\python.exe "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-win_amd64.whl" "numpy<2" plyfile pillow packaging setuptools
  ```

- **`config.json`** (optional; copy `config.example.json`): `comfy_url`, and
  `gpu_request_flag` / `gpu_go_flag` to share
  the GPU with another pipeline. With the flags set, Lookaround creates the request file before
  each orbit and each splat build, waits for the go file, and deletes the request file
  afterwards. Before a splat build it also asks ComfyUI to unload its models, which it
  otherwise keeps in VRAM.

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
