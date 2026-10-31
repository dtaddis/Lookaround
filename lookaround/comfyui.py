"""Start the ComfyUI that install.bat set up, when Lookaround needs it and it isn't running.

If ComfyUI is already up at comfy_url (started by hand, or by another program), Lookaround just
uses it. Otherwise it starts comfy_dir's main.py with comfy_python, logging to comfyui.log, and
stops it again when Lookaround exits.
"""

import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from lookaround.comfy import ComfyError
from lookaround.config import CONFIG, comfy_url, load_config

LOG = CONFIG.parent / "comfyui.log"


def is_up(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=3).close()
        return True
    except (urllib.error.URLError, OSError):
        return False


class ComfyUI:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        """Start ComfyUI in the background unless it's up already or isn't set up."""
        config = load_config()
        url = comfy_url()
        if self.process and self.process.poll() is None or is_up(url):
            return
        if not config.get("comfy_dir") or not config.get("comfy_python"):
            return
        address = urllib.parse.urlsplit(url)
        log = LOG.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [config["comfy_python"], "main.py", "--listen", address.hostname or "127.0.0.1",
             "--port", str(address.port or 8188)],
            cwd=config["comfy_dir"], stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip

    def wait_until_up(self, timeout: float = 300.0, on_wait: Callable[[float], None] | None = None) -> None:
        """Return once ComfyUI answers, starting it if need be; raise ComfyError if it can't."""
        url = comfy_url()
        self.start()
        started = time.monotonic()
        while not is_up(url):
            if self.process is None:
                raise ComfyError(f"ComfyUI isn't running at {url}, and no ComfyUI folder is set up. "
                                 "Run install.bat, or start ComfyUI yourself.")  # fmt: skip
            if self.process.poll() is not None:
                tail = LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
                raise ComfyError("ComfyUI stopped while starting up:\n" + "\n".join(tail))
            if time.monotonic() - started > timeout:
                raise ComfyError(f"ComfyUI didn't start within {timeout:.0f} s. See {LOG.name}.")
            if on_wait:
                on_wait(time.monotonic() - started)
            time.sleep(2)

    def stop(self) -> None:
        """Stop ComfyUI if Lookaround started it."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
