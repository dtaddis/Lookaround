"""Machine-specific settings, read from config.json in the project folder (all optional).

{
  "comfy_url": "http://127.0.0.1:8188",
  "gpu_request_flag": "path/to/request.flag",
  "gpu_go_flag": "path/to/go.flag"
}

With both flag paths set, Lookaround shares the GPU with another pipeline: it creates the
request file before rendering, waits for the other pipeline to write the go file (once it has
paused and freed the GPU), and deletes the request file when done.
"""

import contextlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8-sig")) if CONFIG.exists() else {}


def comfy_url() -> str:
    return load_config().get("comfy_url", DEFAULT_COMFY_URL)


@contextlib.contextmanager
def gpu_claim(
    timeout: float = 600.0, on_wait: Callable[[float], None] | None = None
) -> Iterator[None]:
    """Hold the GPU hand-over for the duration of the block, if one is configured."""
    config = load_config()
    request, go = config.get("gpu_request_flag"), config.get("gpu_go_flag")
    if not request or not go:
        yield
        return
    request, go = Path(request), Path(go)
    request.parent.mkdir(parents=True, exist_ok=True)
    request.touch()
    try:
        started = time.monotonic()
        while not go.exists():
            waited = time.monotonic() - started
            if waited > timeout:
                raise TimeoutError(f"No go signal ({go}) within {timeout:.0f} s")
            if on_wait:
                on_wait(waited)
            time.sleep(2)
        yield
    finally:
        request.unlink(missing_ok=True)
