"""The ComfyUI half of install.bat: choose or clone ComfyUI, add its nodes, fetch the H3 models.

Run by install.bat once Lookaround's own venv exists:
    python -m lookaround.install [--comfy-dir DIR] [--yes]

The default ComfyUI folder is ComfyUI\\ inside Lookaround: it is cloned there with its own venv
(ComfyUI wants a newer PyTorch than AnySplat), and rerunning the install updates it. Any other
folder is an existing ComfyUI that Lookaround uses as it is, with the Python that runs it; the
install only adds what's missing (the GGUF nodes, the models). The choices go in config.json.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from lookaround.config import CONFIG, DEFAULT_COMFY_URL, load_config

ROOT = CONFIG.parent
MANAGED_DIR = ROOT / "ComfyUI"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
# ComfyUI gets a current PyTorch in its own venv (Lookaround's is pinned for AnySplat).
COMFY_PYTHON_VERSION = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# (models folder, file, Hugging Face repo, path in the repo, size in GB)
MODELS = [
    ("diffusion_models", "MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf", "Abiray/MiniMax-H3-Pruned-GGUF",
     "MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf", 14.1),
    ("text_encoders", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", "realrebelai/MiniMax-H3_GGUFs",
     "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14.6),
    ("vae", "minimax_h3_video_vae_fp16.safetensors", "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors", 5.2),
    ("loras", "minimax_h3_flf2v_lora_v1.safetensors", "pablodawson/MiniMax-H3-360-Orbit-LoRA",
     "minimax_h3_flf2v_lora_v1.safetensors", 0.2),
]  # fmt: skip


def run(*command: str | Path, cwd: Path | None = None) -> None:
    print(">", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def ask(question: str, default: str, assume_default: bool) -> str:
    if assume_default:
        return default
    answer = input(f"{question} [{default}]: ").strip().strip('"')
    return answer or default


def yes(question: str, assume_yes: bool) -> bool:
    return assume_yes or input(f"{question} [Y/n]: ").strip().lower() in ("", "y", "yes")


def is_comfyui(folder: Path) -> bool:
    return (folder / "main.py").exists() and (folder / "comfy").is_dir()


def find_python(comfy_dir: Path) -> Path | None:
    """The Python that runs an existing ComfyUI: a venv in or near it, or the portable build's."""
    candidates = [comfy_dir / ".venv", comfy_dir / "venv", comfy_dir.parent / "python_embeded"]
    for parent in list(comfy_dir.parents)[:3]:
        candidates += [parent / ".venv", parent / "venv"]
    for candidate in candidates:
        for python in (candidate / "Scripts" / "python.exe", candidate / "python.exe"):
            if python.exists():
                return python
    return None


def set_up_managed(comfy_dir: Path) -> Path:
    """Clone (or update) ComfyUI inside Lookaround and give it its own venv."""
    if is_comfyui(comfy_dir):
        run("git", "-C", comfy_dir, "pull", "--ff-only")
    else:
        if comfy_dir.exists() and any(comfy_dir.iterdir()):
            sys.exit(f"{comfy_dir} exists but isn't ComfyUI. Move it aside and run install.bat again.")
        run("git", "clone", COMFY_REPO, comfy_dir)
    python = comfy_dir / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        run("uv", "venv", "--python", COMFY_PYTHON_VERSION, comfy_dir / ".venv")
    run("uv", "pip", "install", "--python", python, "torch", "torchvision", "torchaudio",
        "--index-url", TORCH_INDEX)  # fmt: skip
    run("uv", "pip", "install", "--python", python, "-r", comfy_dir / "requirements.txt")
    return python


def install_gguf_nodes(comfy_dir: Path, python: Path, managed: bool, assume_yes: bool) -> None:
    nodes = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
    if nodes.exists():
        if managed:
            run("git", "-C", nodes, "pull", "--ff-only")
        return
    if not managed and not yes(f"ComfyUI-GGUF (needed for the H3 models) isn't in {comfy_dir}. Install it?",
                               assume_yes):  # fmt: skip
        print("Skipped: the orbit won't run until ComfyUI-GGUF is installed.")
        return
    run("git", "clone", GGUF_REPO, nodes)
    run("uv", "pip", "install", "--python", python, "-r", nodes / "requirements.txt")


def model_folders(comfy_dir: Path, folder: str) -> list[Path]:
    """Where ComfyUI looks for a kind of model: its own models folder, then extra_model_paths.yaml."""
    folders = [comfy_dir / "models" / folder]
    extra = comfy_dir / "extra_model_paths.yaml"
    if extra.exists():
        for section in (yaml.safe_load(extra.read_text(encoding="utf-8")) or {}).values():
            if not isinstance(section, dict) or folder not in section:
                continue
            base = Path(section.get("base_path", comfy_dir))
            base = base if base.is_absolute() else comfy_dir / base
            folders += [base / line.strip() for line in str(section[folder]).splitlines() if line.strip()]
    return folders


def download_models(comfy_dir: Path, assume_yes: bool) -> None:
    missing = [m for m in MODELS if not any((f / m[1]).exists() for f in model_folders(comfy_dir, m[0]))]
    if not missing:
        print("The H3 models are all there.")
        return
    total = sum(m[4] for m in missing)
    print("Missing models:\n" + "\n".join(f"  {m[0]}/{m[1]} ({m[4]:.1f} GB)" for m in missing))
    if not yes(f"Download them ({total:.0f} GB) into {comfy_dir / 'models'}?", assume_yes):
        print("Skipped: the orbit won't run until these are in place.")
        return
    for folder, name, repo, path, _ in missing:
        target = comfy_dir / "models" / folder
        staging = target / ".lookaround-download"
        print(f"Downloading {name} from {repo}", flush=True)
        downloaded = Path(hf_hub_download(repo, path, local_dir=staging))
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded, target / name)
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up ComfyUI and the H3 models for Lookaround.")
    parser.add_argument("--comfy-dir", type=Path, help="ComfyUI folder (default: ComfyUI inside Lookaround)")
    parser.add_argument("--comfy-python", type=Path, help="Python that runs an existing ComfyUI")
    parser.add_argument("--yes", action="store_true", help="Accept the defaults without asking")
    args = parser.parse_args()
    config = load_config()

    print("\nComfyUI renders the H3 orbit. Press Enter to install it inside Lookaround, or give the")
    print("folder of a ComfyUI you already have.")
    default = config.get("comfy_dir") or str(MANAGED_DIR)
    comfy_dir = Path(args.comfy_dir or ask("ComfyUI folder", default, args.yes)).resolve()
    managed = comfy_dir == MANAGED_DIR.resolve()
    if managed:
        python = set_up_managed(comfy_dir)
    else:
        if not is_comfyui(comfy_dir) and is_comfyui(comfy_dir / "ComfyUI"):
            comfy_dir = comfy_dir / "ComfyUI"
        if not is_comfyui(comfy_dir):
            sys.exit(f"No ComfyUI in {comfy_dir} (main.py is missing). Run install.bat again.")
        known = config.get("comfy_python") if config.get("comfy_dir") == str(comfy_dir) else None
        found = args.comfy_python or known or find_python(comfy_dir)
        python = Path(ask("Python that runs this ComfyUI", str(found or ""), args.yes and bool(found)))
        if not python.exists():
            sys.exit(f"No Python at {python}. Run install.bat again and give the one that runs ComfyUI.")
    install_gguf_nodes(comfy_dir, python, managed, args.yes)
    download_models(comfy_dir, args.yes)

    config.update(comfy_url=config.get("comfy_url", DEFAULT_COMFY_URL), comfy_dir=str(comfy_dir),
                  comfy_python=str(python))  # fmt: skip
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nComfyUI: {comfy_dir}\nSaved to {CONFIG.name}.")


if __name__ == "__main__":
    main()
