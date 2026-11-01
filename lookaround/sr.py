"""Eye positions and lens figures from a Simulated Reality display, for the viewer to weave.

Run as: python -m lookaround.sr [--latency MICROSECONDS]   (the server starts it; see server.py)

SR displays (Acer SpatialLabs, Samsung Odyssey 3D and other LeiaSR / Dimenco panels) have a
liquid-crystal lens that only the SR Platform runtime, installed with the display, can switch
on, and an eye-tracking camera that only it reads. This talks to that runtime the way SR Loom
does (https://github.com/effcol/SR-Loom), but does no weaving itself: the viewer weaves in
WebGL. It needs no SDK: the runtime's DLLs export typed C++ functions (the types are in their
decorated names), called here with ctypes.

It keeps an SR session open with the lens on, and prints a JSON line about 120 times a second:
    {"eyes": [[x, y, z], [x, y, z]],      the eyes (left, right), mm from the screen's centre,
                                            x right, y up, z towards the viewer
     "weave": [[a, b, r, g, b0], ...],     per eye: the lens phase of a sub-pixel (in lens
                                            periods) is r/g/b0 for its colour at pixel (0, 0),
                                            plus a per pixel across and b per row down; the
                                            eye sees the sub-pixels whose phase is a whole number
     "tracking": bool, "screen": {...}, "lens": {...}}   (screen and lens on the first line)
Closing stdin (or the line "quit") switches the lens off and ends the session.
"""

import argparse
import ctypes as C
import json
import math
import os
import sys
import threading
import winreg
from pathlib import Path


class F2(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]


class F3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class F4(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("w", C.c_float)]


class EyePair(C.Structure):  # SR_eyePair: positions in mm
    _fields_ = [("frame", C.c_uint64), ("time", C.c_uint64), ("left", C.c_double * 3), ("right", C.c_double * 3)]


def platform_bin() -> Path:
    """The SR Platform's DLL folder: LOOKAROUND_SR_BIN, else where the runtime's installer
    registered it."""
    if os.environ.get("LOOKAROUND_SR_BIN"):
        return Path(os.environ["LOOKAROUND_SR_BIN"])
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Dimenco\Simulated Reality") as key:
            return Path(winreg.QueryValueEx(key, "")[0]) / "bin"
    except OSError:
        raise RuntimeError("The SR Platform runtime isn't installed (it comes with the display)") from None


class Runtime:
    def __init__(self) -> None:
        bin_dir = platform_bin()
        os.add_dll_directory(str(bin_dir))
        load = lambda name: C.CDLL(str(bin_dir / name))  # noqa: E731
        core, displays = load("SimulatedRealityCore.dll"), load("SimulatedRealityDisplays.dll")
        faces, directx = load("SimulatedRealityFaceTrackers.dll"), load("SimulatedRealityDirectX.dll")
        weaving = load("DimencoWeaving.dll")

        def fn(lib, name, result, *args):
            f = getattr(lib, name)
            f.restype, f.argtypes = result, list(args)
            return f

        P = C.c_void_p
        self.new_context = fn(core, "newSRContext", P)
        self.initialize = fn(core, "?initialize@SRContext@SR@@QEAAXXZ", None, P)
        self.lens_hint = fn(displays, "?create@SwitchableLensHint@SR@@SAPEAV12@AEAVSRContext@2@@Z", P, P)
        self.lens_on = fn(displays, "lensEnableHint", None, P)
        self.lens_off = fn(displays, "lensDisableHint", None, P)
        self.screen = fn(displays, "?create@Screen@SR@@SAPEAV12@AEAVSRContext@2@@Z", P, P)
        self.screen_width = fn(displays, "?getPhysicalSizeWidth@Screen@SR@@QEAA?BMXZ", C.c_float, P)
        self.screen_height = fn(displays, "?getPhysicalSizeHeight@Screen@SR@@QEAA?BMXZ", C.c_float, P)
        self.screen_columns = fn(displays, "?getResolutionWidth@Screen@SR@@QEAA?BHXZ", C.c_int, P)
        self.screen_rows = fn(displays, "?getResolutionHeight@Screen@SR@@QEAA?BHXZ", C.c_int, P)
        self.eye_tracker = fn(faces, "?create@PredictingEyeTracker@SR@@SAPEAV12@AEAVSRContext@2@@Z", P, P)
        self.predict = fn(faces, "?predict@PredictingEyeTracker@SR@@QEAAX_KAEAUSR_eyePair@@@Z", None,
                          P, C.c_uint64, C.POINTER(EyePair))  # fmt: skip
        self.create_weaver = fn(directx, "?CreateDX11Weaver@SR@@YA?AW4WeaverErrorCode@@PEAVSRContext@1@"
                                         "PEAUID3D11DeviceContext@@PEAUHWND__@@PEAPEAVIDX11Weaver1@1@@Z",
                                C.c_int, P, P, P, C.POINTER(P))  # fmt: skip
        self.set_viewer = fn(weaving, "?SetGlobalParameters@Weaver@Dimenco@@SAXUFLOAT3@@@Z", None, F3)
        self.fill = fn(weaving, "?FillAttributes@Weaver@Dimenco@@SAXAEBUFLOAT2@@AEBUFLOAT4@@AEAU4@2AEAU3@3@Z",
                       None, *([P] * 6))  # fmt: skip
        self.lens = {name.lower(): fn(weaving, f"?Get{name}@Weaver@Dimenco@@SAMXZ", C.c_float)
                     for name in ("Slant", "Px", "N", "DoN")}  # fmt: skip


class Session:
    def __init__(self, latency_us: int) -> None:
        rt = self.rt = Runtime()
        self.latency_us = latency_us
        self.context = rt.new_context()
        if not self.context:
            raise RuntimeError("The SR service did not answer")
        self.hint = rt.lens_hint(self.context)
        self.tracker = rt.eye_tracker(self.context)
        self.screen = rt.screen(self.context)
        rt.initialize(self.context)
        # The weaving library only loads this panel's lens once a weaver exists in the process.
        # This one never weaves: a bare Direct3D 11 device is enough to make it.
        d3d11 = C.WinDLL("d3d11.dll")
        self.device, self.device_context, level = C.c_void_p(), C.c_void_p(), C.c_uint()
        hr = d3d11.D3D11CreateDevice(None, 1, None, 0, None, 0, 7, C.byref(self.device), C.byref(level),
                                     C.byref(self.device_context))  # fmt: skip
        if hr != 0:
            raise RuntimeError(f"Could not make a Direct3D 11 device ({hr & 0xFFFFFFFF:#x})")
        self.weaver = C.c_void_p()
        code = rt.create_weaver(self.context, self.device_context, None, C.byref(self.weaver))
        if code != 0 or not rt.lens["px"]():
            raise RuntimeError(f"The SR runtime has no lens figures for this display (code {code})")
        rt.lens_on(self.hint)
        self.columns, self.rows = rt.screen_columns(self.screen), rt.screen_rows(self.screen)

    def screen_info(self) -> dict:
        width, height = self.rt.screen_width(self.screen), self.rt.screen_height(self.screen)
        return {"width_mm": width * 10, "height_mm": height * 10, "columns": self.columns, "rows": self.rows}

    def eyes(self) -> tuple[list[float], list[float], bool]:
        pair = EyePair()
        self.rt.predict(self.tracker, self.latency_us, C.byref(pair))
        left, right = list(pair.left), list(pair.right)
        # With nobody in view the runtime gives one point for both eyes: spread them about it.
        tracking = math.dist(left, right) > 20
        if not tracking:
            left, right = [left[0] - 31.5, left[1], left[2]], [left[0] + 31.5, left[1], left[2]]
        return left, right, tracking

    def weave(self, eye_mm: list[float]) -> list[float]:
        """For an eye: the lens phase at pixel (0, 0) for each colour, and its step a pixel across
        and a row down (all in lens periods), as the runtime's own weave sees it from there."""
        rt = self.rt
        rt.set_viewer(F3(*(v / 10 for v in eye_mm)))  # (centimetres)
        size = F2(self.columns, self.rows)
        corners = []
        for x, y in ((-1, 1), (1, 1), (1, -1)):  # top-left, top-right, bottom-right
            phase, dxy, at, shift = F4(), F4(), F2(), F2()
            rt.fill(C.addressof(size), C.addressof(F4(x, y, 0, 1)), C.addressof(phase), C.addressof(dxy),
                    C.addressof(at), C.addressof(shift))  # fmt: skip
            corners.append(((phase.x + shift.x, phase.y + shift.x, phase.z + shift.x), (at.x, at.y)))
        (tl, tl_at), (tr, tr_at), (br, br_at) = corners
        across = (tr[1] - tl[1]) / (tr_at[0] - tl_at[0])
        down = (br[1] - tr[1]) / (br_at[1] - tr_at[1])
        origin = [tl[c] - across * tl_at[0] - down * tl_at[1] for c in range(3)]
        whole = math.floor(origin[1])  # only the fraction matters; keep the numbers small
        return [across, down, *(o - whole for o in origin)]

    def close(self) -> None:
        self.rt.lens_off(self.hint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency", type=int, default=40000, help="Predict the eyes this far ahead (µs)")
    parser.add_argument("--rate", type=float, default=120)
    args = parser.parse_args()
    try:
        session = Session(args.latency)
    except Exception as error:  # the runtime missing, no SR display, the service down...
        print(json.dumps({"error": str(error)}), flush=True)
        return
    stop = threading.Event()

    def watch_stdin() -> None:
        for line in sys.stdin:
            if line.strip() == "quit":
                break
        stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()
    C.windll.winmm.timeBeginPeriod(1)  # Windows otherwise sleeps in 15.6 ms steps
    lens = {key: round(get(), 6) for key, get in session.rt.lens.items()}
    first = True
    try:
        while not stop.is_set():
            left, right, tracking = session.eyes()
            message = {"eyes": [[round(v, 2) for v in left], [round(v, 2) for v in right]],
                       "weave": [[round(v, 8) for v in session.weave(e)] for e in (left, right)],
                       "tracking": tracking}  # fmt: skip
            if first:
                message.update(screen=session.screen_info(), lens=lens)
                first = False
            print(json.dumps(message), flush=True)
            stop.wait(1 / args.rate)
    except (BrokenPipeError, OSError):
        pass
    finally:
        session.close()


if __name__ == "__main__":
    main()
