"""Lookaround web app: drop a photo, watch its H3 orbit, build and view a splat.

Run with: .venv\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000

Each run lives in runs/<id>/: the photo (input.*), orbit/frame_NNNN.png, orbit.mp4,
splat/splat.ply with splat/splat.json, state.json and job.log.
"""

import asyncio
import contextlib
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from lookaround.comfy import ComfyClient, ComfyError
from lookaround.comfyui import ComfyUI
from lookaround.config import comfy_url
from lookaround.frames import arc_frame_count, check_custom, distinct_frames, pick_arc, pick_indices

ROOT = Path(__file__).resolve().parent
# Refining needs gsplat's compiled kernels, which come prebuilt only for Python 3.10, so it
# runs in its own venv (see README.md).
REFINE_PYTHON = ROOT / ".venv-refine" / "Scripts" / "python.exe"
RUNS = ROOT / "runs"
WEB = ROOT / "web"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RUN_ID = re.compile(r"[0-9a-f]{32}")
ORBIT_LINE = re.compile(r"\[\s*([\d.]+)%\]\s*(.*)")
SPLAT_LINE = re.compile(r"PROGRESS (\d+) (.*)")
DEFAULT_ARC = 0  # degrees each side of the photo; 0 means the whole orbit
DEFAULT_COUNT = 48  # scored better on held-out frames than 32 (and 24 falls apart)
# H3's seed for a run's first orbit; "New orbit" rolls a random one.
DEFAULT_SEED = 12345

# H3 and AnySplat would fight over the card, so only one GPU job runs at a time.
gpu_lock = threading.Lock()
RUNS.mkdir(exist_ok=True)
# The ComfyUI install.bat set up, started when it isn't running already.
comfyui = ComfyUI()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    mark_interrupted()
    comfyui.start()  # it takes a while to load, so start it now rather than at the first orbit
    yield
    sr_display.stop(force=True)
    comfyui.stop()


app = FastAPI(title="Lookaround", lifespan=lifespan)


# --- run folders and state -------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id) or not (RUNS / run_id).is_dir():
        raise HTTPException(404, "No such run")
    return RUNS / run_id


def read_state(run: Path) -> dict:
    path = run / "state.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def write_state(run: Path, **updates) -> None:
    state = read_state(run)
    state.update(updates)
    (run / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def mark_interrupted() -> None:
    """Jobs run inside the server, so any run still marked busy at startup was cut off."""
    for run in RUNS.iterdir():
        if read_state(run).get("status") in ("orbiting", "splatting"):
            message = "Interrupted: the server stopped before this finished. Start it again."
            write_state(run, status="error", message=message, error=message)


def orbit_frames(run: Path) -> list[Path]:
    return sorted((run / "orbit").glob("frame_*.png"))


def url_for(run: Path, relative: str) -> str:
    return f"/runs/{run.name}/{relative}"


def select_frames(distinct: list[int], arc: float, count: int, custom: str | None) -> list[int]:
    """Frame numbers for AnySplat: hand-picked, a front arc around the photo, or the whole orbit."""
    if custom is not None:
        try:
            indices = [int(value) for value in custom.split(",") if value.strip()]
        except ValueError:
            raise ValueError("Frames must be comma-separated numbers")
        return check_custom(distinct, indices)
    return pick_arc(distinct, arc, count) if arc else pick_indices(distinct, count)


# --- jobs --------------------------------------------------------------------------------


def run_command(run: Path, command: list[str], on_line) -> int:
    """Run a command, logging every line to job.log and passing it to on_line."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    with (run / "job.log").open("a", encoding="utf-8") as log:
        for line in process.stdout:
            line = line.rstrip()
            log.write(line + "\n")
            log.flush()
            on_line(line)
    return process.wait()


def orbit_job(run: Path) -> None:
    state = read_state(run)
    write_state(run, status="orbiting", progress=0, message="Starting the H3 orbit", error=None)
    comfyui.wait_until_up(on_wait=lambda s: write_state(run, message=f"Starting ComfyUI ({s:.0f} s)"))
    # A new orbit (re-rolled with another seed) makes everything built from the old one stale.
    for stale in ("orbit", "masks", "splat"):
        shutil.rmtree(run / stale, ignore_errors=True)
    (run / "orbit.mp4").unlink(missing_ok=True)

    def on_line(line: str) -> None:
        match = ORBIT_LINE.search(line)
        if match:
            write_state(run, progress=float(match.group(1)), message=match.group(2))

    code = run_command(
        run,
        [
            sys.executable, "-m", "lookaround.orbit", str(run / state["input"]),
            str(run / "orbit"), "--video", str(run / "orbit.mp4"),
            "--seed", str(state.get("seed", DEFAULT_SEED)),
        ],  # fmt: skip
        on_line,
    )
    if code != 0:
        raise RuntimeError(f"The orbit failed (exit code {code}). See job.log in the run folder.")
    write_state(run, status="orbit_ready", progress=100, message="Orbit ready")


def splat_job(run: Path) -> None:
    state = read_state(run)
    refine = state.get("refine") and REFINE_PYTHON.exists()
    write_state(run, status="splatting", progress=0, message="Starting AnySplat", error=None)
    subject_only = ["--subject-masks", str(run / "masks")] if state.get("subject") else []

    def stage(start: int, end: int):
        """Show a stage's own 0-100% progress as start-end% of the whole job."""

        def on_line(line: str) -> None:
            match = SPLAT_LINE.match(line)
            if match:
                percent = start + (end - start) * int(match.group(1)) / 100
                write_state(run, progress=round(percent), message=match.group(2))

        return on_line

    try:
        ComfyClient(comfy_url()).free_memory()  # ComfyUI keeps models in VRAM between jobs
    except ComfyError:
        pass  # not running, so it holds nothing
    command = [
        sys.executable, "-m", "lookaround.splat", str(run / "orbit"), str(run / "splat"),
        "--indices", ",".join(map(str, state["indices"])), *subject_only,
    ]  # fmt: skip
    code = run_command(run, command, stage(0, 30 if refine else 100))
    if code != 0:
        raise RuntimeError(f"AnySplat failed (exit code {code}). See job.log in the run folder.")
    if refine:
        masks = ["--masks", str(run / "masks")] if state.get("subject") else []
        command = [str(REFINE_PYTHON), "-m", "lookaround.refine", str(run / "splat"),
                   str(run / "orbit"), *masks]  # fmt: skip
        code = run_command(run, command, stage(30, 100))
        if code != 0:
            raise RuntimeError(f"Refining failed (exit code {code}). See job.log in the run folder.")
    write_state(run, status="splat_ready", progress=100, message="Splat ready")


def start_job(run: Path, job) -> None:
    """Start a GPU job in the background, or refuse if another one is running."""
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "Another job is using the GPU. Wait for it to finish.")

    def work() -> None:
        try:
            job(run)
        except Exception as error:  # shown on the page rather than lost in a thread
            write_state(run, status="error", message=str(error), error=str(error))
        finally:
            gpu_lock.release()

    threading.Thread(target=work, daemon=True).start()


# --- API ---------------------------------------------------------------------------------


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in EXTENSIONS:
        raise HTTPException(400, "Use a JPG, PNG or WebP photo")
    run_id = uuid.uuid4().hex
    run = RUNS / run_id
    run.mkdir(parents=True)
    input_name = "input" + suffix
    (run / input_name).write_bytes(await file.read())
    write_state(
        run,
        input=input_name,
        original=file.filename,
        status="uploaded",
        progress=0,
        message="Photo received",
    )
    try:
        start_job(run, orbit_job)
    except HTTPException:
        write_state(run, message="The GPU is busy. Start the orbit when it is free.")
    return {"id": run_id}


@app.get("/api/runs")
def list_runs():
    """Every run, newest first, for the All runs page."""
    runs = []
    for run in RUNS.iterdir():
        state = read_state(run)
        if not RUN_ID.fullmatch(run.name) or "input" not in state:
            continue
        meta_path = run / "splat" / "splat.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        runs.append({
            "id": run.name,
            "original": state.get("original"),
            "created": (run / state["input"]).stat().st_mtime,
            "status": state.get("status"),
            "progress": state.get("progress", 0),
            "message": state.get("message", ""),
            "input_url": url_for(run, state["input"]),
            "has_splat": (run / "splat" / "splat.ply").exists(),
            "gaussians": meta.get("gaussians"),
            "refined_steps": meta.get("refined_steps"),
            "subject": bool(state.get("subject")),
        })  # fmt: skip
    return sorted(runs, key=lambda r: r["created"], reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(
    run_id: str,
    count: int = Query(DEFAULT_COUNT, ge=2, le=137),
    arc: float = Query(DEFAULT_ARC, ge=0, lt=180),
    custom: str | None = None,
):
    run = run_dir(run_id)
    state = read_state(run)
    frames = orbit_frames(run) if state.get("status") not in ("uploaded", "orbiting") else []
    distinct = distinct_frames(run / "orbit") if frames else []
    selected, selection_error = [], None
    if frames:
        try:
            selected = select_frames(distinct, arc, count, custom)
        except ValueError as error:
            selection_error = str(error)
    splat = run / "splat" / "splat.ply"
    log = run / "job.log"
    return {
        "id": run_id,
        "status": state.get("status"),
        "progress": state.get("progress", 0),
        "message": state.get("message", ""),
        "error": state.get("error"),
        "input_url": url_for(run, state["input"]),
        "orbit_video_url": url_for(run, "orbit.mp4") if (run / "orbit.mp4").exists() else None,
        "frames": [url_for(run, f.relative_to(run).as_posix()) for f in frames],
        "distinct": distinct,
        "selected": selected,
        "selection_error": selection_error,
        "arc_frames": arc_frame_count(distinct, arc) if distinct and arc else len(distinct),
        # What the last splat was built from, so the page can show it.
        "built": {key: state[key] for key in ("indices", "arc", "count") if key in state},
        "splat_url": url_for(run, "splat/splat.ply") if splat.exists() else None,
        "subject": bool(state.get("subject")),
        "refine": state.get("refine", True),
        "can_refine": REFINE_PYTHON.exists(),
        "log": log.read_text(encoding="utf-8").splitlines()[-12:] if log.exists() else [],
    }


@app.post("/api/runs/{run_id}/orbit")
def make_orbit(run_id: str, reroll: bool = False):
    """Make the run's orbit; with reroll, a new one from another seed (H3 varies a lot by seed)."""
    run = run_dir(run_id)
    if reroll:
        write_state(run, seed=random.randrange(2**31))
    start_job(run, orbit_job)
    return {"ok": True}


@app.post("/api/runs/{run_id}/splat")
def make_splat(
    run_id: str,
    count: int = Query(DEFAULT_COUNT, ge=2, le=137),
    arc: float = Query(DEFAULT_ARC, ge=0, lt=180),
    custom: str | None = None,
    subject: bool = False,
    refine: bool = True,
):
    run = run_dir(run_id)
    frames = orbit_frames(run)
    if read_state(run).get("status") in ("uploaded", "orbiting") or not frames:
        raise HTTPException(409, "Make the orbit before building a splat")
    try:
        indices = select_frames(distinct_frames(run / "orbit"), arc, count, custom)
    except ValueError as error:
        raise HTTPException(400, str(error))
    arc = "custom" if custom is not None else arc
    write_state(run, indices=indices, arc=arc, count=len(indices), subject=subject, refine=refine)
    start_job(run, splat_job)
    return {"ok": True}


# --- SR (glasses-free 3D) displays ---------------------------------------------------------


class SRDisplay:
    """The SR helper (lookaround/sr.py), running while any viewer is in 3D display mode: it
    switches the display's lens on and reports the eyes and the lens for the viewer to weave."""

    LINGER = 5  # seconds to keep the lens on after the last viewer leaves, in case it reloads

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.clients = 0
        self.first: str | None = None  # the helper's first line: the screen and lens, or an error
        self.latest: str | None = None
        self.timer: threading.Timer | None = None
        self.streams: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()  # woken per line

    def attach(self) -> None:
        with self.lock:
            self.clients += 1
            if self.timer:
                self.timer.cancel()
                self.timer = None
            if self.process is None or self.process.poll() is not None:
                self.first = self.latest = None
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "lookaround.sr"],
                    cwd=ROOT,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # the runtime's own log chatter
                    text=True,
                )
                threading.Thread(target=self.read, args=(self.process,), daemon=True).start()

    def read(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            with self.lock:
                if process is not self.process:
                    break
                self.first = self.first or line.strip()
                self.latest = line.strip()
                for loop, event in tuple(self.streams):  # (streams come and go on another thread)
                    loop.call_soon_threadsafe(event.set)

    def detach(self) -> None:
        with self.lock:
            self.clients -= 1
            if self.clients == 0:
                self.timer = threading.Timer(self.LINGER, self.stop)
                self.timer.start()

    def running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def stop(self, force: bool = False) -> None:
        with self.lock:
            if (self.clients and not force) or self.process is None:
                return
            process, self.process = self.process, None
        with contextlib.suppress(OSError):
            process.stdin.write("quit\n")
            process.stdin.close()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()


sr_display = SRDisplay()


@app.get("/api/sr")
async def sr_stream(request: Request):
    """Server-sent events from the SR helper: first the screen and lens (or an error), then the
    eyes and weave as often as the helper reports them."""

    async def events():
        new_line = asyncio.Event()
        stream = (asyncio.get_running_loop(), new_line)
        sr_display.streams.add(stream)
        sr_display.attach()
        try:
            while sr_display.first is None and sr_display.running():
                await asyncio.sleep(0.05)
            if sr_display.first is None:
                yield f"data: {json.dumps({'error': 'The SR helper stopped'})}\n\n"
                return
            yield f"data: {sr_display.first}\n\n"
            if "error" in json.loads(sr_display.first):
                return
            while not await request.is_disconnected():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(new_line.wait(), 1)
                if new_line.is_set():
                    new_line.clear()
                    yield f"data: {sr_display.latest}\n\n"
                elif not sr_display.running():
                    yield f"data: {json.dumps({'error': 'The SR helper stopped'})}\n\n"
                    return
        finally:
            sr_display.streams.discard(stream)
            sr_display.detach()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
