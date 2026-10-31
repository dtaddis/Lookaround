"""Machine-specific settings, read from config.json in the project folder (all optional).

{
  "comfy_url": "http://127.0.0.1:8188"
}
"""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8-sig")) if CONFIG.exists() else {}


def comfy_url() -> str:
    return load_config().get("comfy_url", DEFAULT_COMFY_URL)
