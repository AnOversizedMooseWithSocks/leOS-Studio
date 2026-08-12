"""lestudio -- leStudio: a layer + node based image editor built on the leCore engine.

Photoshop/GIMP-style layered raster editing (brush, eraser, blend modes, opacity, undo/redo)
plus a non-destructive NODE GRAPH whose operators are leCore capabilities: pattern fields,
escape-time fractals, colour transfer, perceptual segmentation, self-calibrating edges,
Landweber sharpening, spectral blur, harmonic inpainting, demodulated upscale, light shafts,
cosine palettes -- all pure NumPy, all deterministic.

    from lestudio import Document, NodeGraph, OPS
    from lestudio.server import serve   # browser UI

Everything the graph computes is cached dependency-keyed (leCore's O(change) re-evaluation
discipline): editing one node's parameter recomputes only that node and its downstream.
"""
from __future__ import annotations

import base64
import hashlib
import time as _time

_MUT_REV = [0]      # bumped by every recorded mutation / undo / redo: signature
                    # memos become stale the instant anything edits
import io
import json
import time

import re
import numpy as np

__all__ = ["Document", "Layer", "NodeGraph", "OPS", "op_catalog", "composite",
           "BLEND_MODES", "save_workspace", "load_workspace", "sdf_to_glsl"]

__version__ = "0.1.0"


# ------------------------------------------------------------------------------------------------
# The engine handle. One UnifiedMind per process is plenty; dim stays small because the studio
# uses the deterministic image/field doors, not the associative memory.
# ------------------------------------------------------------------------------------------------
_MIND = None


_ACCEL = {"gpu": False, "jit": False, "detail": {}}


_FEATURES_CACHE = {}


def have(*names):
    """Capability preflight: which leCore faculties this build actually has.

    leCore 0.2.4 added `mind().features([...]) -> {name: bool}` for exactly this
    (their C14); older builds don't have it, so we fall back to hasattr. Use
    this instead of calling an optional faculty and hoping -- a missing method
    used to surface as a 500 (the wrap_webgl2 incident), which is invisible to
    the person using the app.

    have("texture_image")            -> True/False
    have("ramp", "ramp_texture")     -> True only if BOTH exist
    """
    missing = [n for n in names if n not in _FEATURES_CACHE]
    if missing:
        m = mind()
        probe = getattr(m, "features", None)
        got = None
        if callable(probe):
            try:
                got = probe(list(missing))                 # one call, no guessing
            except Exception:
                got = None
        if not isinstance(got, dict):
            got = {n: hasattr(m, n) for n in missing}      # older leCore
        for n in missing:
            _FEATURES_CACHE[n] = bool(got.get(n, False))
    return all(_FEATURES_CACHE[n] for n in names)


def engine_version():
    """{engine, capabilities_schema, dim, seed} from leCore 0.2.4's version()
    faculty, falling back to the package's own metadata on older builds. We
    report this in /api/state so a client can tell WHICH engine it is talking
    to. We report the engine's own number rather than inferring one, and it is
    for DISPLAY only -- feature availability is decided by have(), never by
    comparing this string."""
    try:
        v = mind().version()
        if isinstance(v, dict):
            return dict(v)
    except Exception:
        pass
    try:
        import lecore
        return {"engine": getattr(lecore, "__version__", "unknown")}
    except Exception:
        return {"engine": "unknown"}


def mind():
    global _MIND
    if _MIND is None:
        import lecore
        _MIND = lecore.UnifiedMind(dim=512, seed=0)
        # opt into every accelerator leCore finds installed ([jit]/[gpu] extras);
        # both fall back cleanly when absent
        try:
            _ACCEL["gpu"] = bool(_MIND.use_gpu(True))
        except Exception:
            _ACCEL["gpu"] = False
        try:
            rep = _MIND.accelerator_report()
            _ACCEL["detail"] = rep if isinstance(rep, dict) else {"report": str(rep)[:2000]}
            # Read the report STRUCTURALLY. The old substring test
            # ("numba" in txt and "installed" in txt) matched the *key* name
            # 'installed' and a `True` from some other row, so it reported JIT
            # as active on a box with no numba at all -- the status bar said
            # "CPU - JIT" while every JIT path was cold.
            _ACCEL["jit"] = False
            _ACCEL["accel"] = {}
            rows = rep if isinstance(rep, (list, tuple)) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name", "")).lower()
                inst = bool(row.get("installed"))
                if nm:
                    _ACCEL["accel"][nm] = inst
                if nm == "numba":
                    _ACCEL["jit"] = inst
        except Exception:
            pass
    return _MIND


def parallel_advice(n_jobs=6, ms_each=1500):
    """Would running slow node evaluations in parallel actually pay?

    Asks leCore's own `should_pool`, which knows the usable core count and
    answers with a REASON. On this machine that reason is "only 1 usable
    core(s); a pool adds overhead and memory but cannot add speed" -- so the
    honest thing is to not spin up workers, and to say why rather than
    silently doing nothing.

    Deliberately advisory-only: the pool is not wired in, because it could not
    be measured here. What ships is the gate and the explanation, so the answer
    is correct on a one-core box and the work is a small step on a real one."""
    if not have("should_pool"):
        return {"worth_it": False, "why": "engine has no pool advisor",
                "cores": None}
    try:
        ok, why = mind().should_pool(n_buckets=int(n_jobs),
                                     est_ms_per_bucket=float(ms_each))
    except Exception as e:
        return {"worth_it": False, "why": str(e), "cores": None}
    cores = None
    if have("cpu_budget"):
        try:
            cores = int(mind().cpu_budget())
        except Exception:
            pass
    return {"worth_it": bool(ok), "why": str(why), "cores": cores}


def image_similarity(a, b):
    """Perceptual similarity in [0, 1] between two images (1 = identical).

    Byte equality answers "did anything change"; this answers "would anyone
    notice". Measured on the engine's own metric: a +0.02 brightness shift
    scores 0.991 (invisible), the same image rolled 4 px scores 0.670 -- the
    structural change ranks far below the tonal one, which is the ordering a
    person would give and the one a pixel diff gets backwards.

    Returns None when the engine is older than 0.2.7, so callers must handle
    its absence rather than silently comparing nothing."""
    if not have("compare_images"):
        return None
    ra, rb = _rgb(a), _rgb(b)
    if ra.shape[:2] != rb.shape[:2]:
        # The metric compares pixel for pixel. Comparing a layer against its
        # own placed source means comparing 300x400 against 900x1200, so bring
        # them to a common size first -- otherwise the caller gets None and
        # cannot tell "the engine cannot do this" from "these are different
        # sizes", which is exactly the confusion this first shipped with.
        rb = _resize(rb, ra.shape[0], ra.shape[1])
    try:
        return float(mind().compare_images(ra, rb))
    except Exception:
        return None


def accel_status():
    """What acceleration is reachable, and -- when it is not -- why.

    Our own probe reports presence/absence. leCore 0.2.7 added `gpu_report()`,
    which answers the same question from the engine's side WITH the reason and
    the install line, plus `should_offload`/`should_pool`, which say whether
    moving work would actually pay on this machine. Capability-gated: on an
    older engine the extra keys are simply absent."""
    mind()
    out = dict(_ACCEL)
    if have("gpu_report"):
        try:
            out["gpu"] = mind().gpu_report()
        except Exception as e:
            out["gpu"] = {"error": str(e)}
    advice = []
    if have("should_pool"):
        try:
            # a realistic heavy pass: several slow nodes at a second or two each
            ok, why = mind().should_pool(n_buckets=6, est_ms_per_bucket=1500)
            advice.append({"kind": "process pool", "worth_it": bool(ok),
                           "why": str(why)})
        except Exception:
            pass
    if have("should_offload"):
        try:
            # a 1080p composite: ~33 MB, a handful of flops per byte
            ok, why = mind().should_offload(n_bytes=33_000_000,
                                            flops_per_byte=4)
            advice.append({"kind": "GPU offload", "worth_it": bool(ok),
                           "why": str(why)})
        except Exception:
            pass
    if advice:
        out["advice"] = advice
    if have("resource_policy"):
        try:
            pol = mind().resource_policy()
            # `bit_exact` is the one that matters to us: a leStudio document is
            # a RECIPE (strokes, node params, seeds) that must render the same
            # way twice. The engine flags which settings would stop that being
            # true -- gpu='on' turns bit_exact False -- so we can warn instead
            # of silently shipping documents that no longer reproduce.
            out["determinism"] = {
                "bit_exact": bool(pol.get("bit_exact", True)),
                "affecting": list(pol.get("numerics_affecting") or []),
                "cores": (pol.get("fields", {}).get("cpu_cores", {})
                          .get("effective")),
            }
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------------------------
# Small image helpers (pure NumPy; used by ops and by the compositor)
# ------------------------------------------------------------------------------------------------

def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _rgb(img):
    """Coerce any field/greyscale/RGBA to (H, W, 3) float in [0, 1]."""
    a = _f32(img)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def _gauss_blur_reflect(img, sigma, pad=8):
    """Gaussian blur with REFLECT boundaries. _gauss_blur is an FFT --
    circular convolution -- so blurring a height field coupled the
    canvas's opposite edges: painting impasto at the right edge subtly
    changed the LEFT edge's lighting, and the window-patch cache could
    never agree with a full recompute. Reflect-pad, blur, crop."""
    a = _f32(img)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    p = int(max(2, pad))
    ap = np.pad(a, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = _gauss_blur(ap, sigma)[p:-p, p:-p]
    return out[..., 0] if single else out


def _gauss_small(img, sigma):
    """Direct separable Gaussian, for the SMALL sigmas the paint code uses.

    `_gauss_blur` convolves in the frequency domain, which is the right call
    for the big postfx kernels it was written for and badly wrong for a
    sigma-2 blur on a stroke window: an FFT pair per channel dominated the
    whole watercolour pass (183 ms of a 578 ms stroke, and 20 transforms for
    one mark). A separable kernel of a dozen taps is the same result for a
    fraction of the work, and it reflects at the border instead of wrapping,
    which the paint code wants anyway.
    """
    if sigma <= 0:
        return _f32(img).copy()
    a = _f32(img)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    r = int(max(1, round(sigma * 3.0)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    pad = ((r, r), (0, 0), (0, 0))
    t = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        if w > 1e-6:
            out += t[i:i + a.shape[0]] * w
    t = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        if w > 1e-6:
            out += t[:, i:i + a.shape[1]] * w
    return out[..., 0] if single else out


def _gauss_blur(img, sigma):
    """Separable Gaussian by FFT per channel -- the spectral blur the postfx algebra fuses."""
    if sigma <= 0:
        return _f32(img).copy()
    a = _f32(img)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ker = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.fft2(a[:, :, c]) * ker))
    return out[:, :, 0] if single else out


_RESIZE_TAB = {}


def _resize(img, h, w):
    """Bilinear resample to (h, w) -- used to conform node inputs to the document size."""
    a = _f32(img)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a[:, :, 0] if single else a
    # the sampling GRID depends only on the shapes, never on the pixels,
    # yet it was rebuilt on every call -- and _resize sits under every
    # media frame render (twice: density and dye), every node-input
    # conform, mask fit, and FX upsample. Profiled at 1000x750 it was
    # the single largest cost in a playback frame. Cache the tables.
    ck = (H, W, h, w)
    tab = _RESIZE_TAB.get(ck)
    if tab is None:
        ys = np.linspace(0, H - 1, h)
        xs = np.linspace(0, W - 1, w)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, H - 1)
        # keep the float64 weights: an existing pin requires this to be
        # BIT-identical to the pre-separable algorithm, and casting the
        # weights to float32 changes the arithmetic
        fy = (ys - y0)[:, None, None]
        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, W - 1)
        fx = (xs - x0)[None, :, None]
        if len(_RESIZE_TAB) > 24:
            _RESIZE_TAB.clear()
        tab = _RESIZE_TAB[ck] = (y0, y1, fy, x0, x1, fx)
    y0, y1, fy, x0, x1, fx = tab
    # bilinear is SEPARABLE: rows first, then columns -- the same sampling
    # grid and the same arithmetic, but two medium gathers instead of four
    # full-image ones. The old form profiled at 0.65 s per call at 1024x768
    # and sat on top of every node-input conform, mask fit, and FX upsample.
    rows = a[y0] * (1 - fy) + a[y1] * fy          # (h, W, C)
    out = rows[:, x0] * (1 - fx) + rows[:, x1] * fx
    out = np.ascontiguousarray(out, np.float32)
    return out[:, :, 0] if single else out


def _resize_window(img, h, w, box):
    """Bilinear resample to (h, w) but COMPUTE ONLY the output window
    box=(y0, y1, x0, x1). Bilinear is separable and every output pixel
    depends only on its own sample position, so slicing the cached
    tables gives results BYTE-IDENTICAL to the full resize -- verified,
    because a seam here would be exactly the kind of artifact this
    round is meant to remove. Used by the media render, where the dye
    occupies a fraction of the canvas for most of a simulation."""
    a = _f32(img)
    single = a.ndim == 2
    if single:
        a = a[:, :, None]
    H, W = a.shape[:2]
    y0b, y1b, x0b, x1b = box
    _resize(a[:1, :1], 1, 1)              # ensure the table cache exists
    ck = (H, W, h, w)
    tab = _RESIZE_TAB.get(ck)
    if tab is None:
        _resize(a, h, w)
        tab = _RESIZE_TAB[ck]
    y0, y1, fy, x0, x1, fx = tab
    ys0, ys1 = y0[y0b:y1b], y1[y0b:y1b]
    fys = fy[y0b:y1b]
    xs0, xs1 = x0[x0b:x1b], x1[x0b:x1b]
    fxs = fx[:, x0b:x1b]
    rows = a[ys0] * (1 - fys) + a[ys1] * fys
    out = rows[:, xs0] * (1 - fxs) + rows[:, xs1] * fxs
    out = np.ascontiguousarray(out, np.float32)
    return out[:, :, 0] if single else out


def _affine(img, sx=1.0, sy=1.0, deg=0.0, dx=0.0, dy=0.0, pivot=None):
    """Scale/rotate/translate an (H, W[, C]) array about `pivot` (x, y) --
    default: the array centre. Inverse-mapped bilinear; outside pixels zero."""
    single = img.ndim == 2
    a = img[..., None] if single else img
    h, w = a.shape[:2]
    if pivot is None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    else:
        cx, cy = float(pivot[0]), float(pivot[1])
    th = np.deg2rad(deg)
    cos, sin = np.cos(th), np.sin(th)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # inverse transform: undo translate, rotate by -th, unscale
    x0 = xs - cx - dx
    y0 = ys - cy - dy
    xr = (x0 * cos + y0 * sin) / max(sx, 1e-6) + cx
    yr = (-x0 * sin + y0 * cos) / max(sy, 1e-6) + cy
    u0 = np.floor(xr).astype(int); v0 = np.floor(yr).astype(int)
    fu = (xr - u0)[..., None]; fv = (yr - v0)[..., None]
    u0c = np.clip(u0, 0, w - 1); u1c = np.clip(u0 + 1, 0, w - 1)
    v0c = np.clip(v0, 0, h - 1); v1c = np.clip(v0 + 1, 0, h - 1)
    out = (a[v0c, u0c] * (1 - fu) * (1 - fv) + a[v0c, u1c] * fu * (1 - fv)
           + a[v1c, u0c] * (1 - fu) * fv + a[v1c, u1c] * fu * fv)
    inside = ((xr >= 0) & (xr <= w - 1) & (yr >= 0) & (yr <= h - 1))[..., None]
    out = np.where(inside, out, 0.0).astype(np.float32)
    return out[..., 0] if single else out


def _maxfilter(a, it):
    """Greyscale dilation by `it` px (iterated 4-neighbourhood max)."""
    out = a.copy()
    for _ in range(int(it)):
        n = out.copy()
        n[1:, :] = np.maximum(n[1:, :], out[:-1, :])
        n[:-1, :] = np.maximum(n[:-1, :], out[1:, :])
        n[:, 1:] = np.maximum(n[:, 1:], out[:, :-1])
        n[:, :-1] = np.maximum(n[:, :-1], out[:, 1:])
        out = n
    return out


def _minfilter(a, it):
    """Greyscale erosion by `it` px."""
    return 1.0 - _maxfilter(1.0 - a, it)


def _rotate_tip(tip, deg):
    """Rotate a square tip by deg (bilinear, zero-padded) -- pure NumPy."""
    if abs(deg) < 0.5:
        return tip
    T = tip.shape[0]
    c = (T - 1) / 2.0
    th = np.deg2rad(deg)
    ys, xs = np.mgrid[0:T, 0:T].astype(np.float32)
    u = (xs - c) * np.cos(th) + (ys - c) * np.sin(th) + c
    v = -(xs - c) * np.sin(th) + (ys - c) * np.cos(th) + c
    u0 = np.clip(np.floor(u).astype(int), 0, T - 1); u1 = np.clip(u0 + 1, 0, T - 1)
    v0 = np.clip(np.floor(v).astype(int), 0, T - 1); v1 = np.clip(v0 + 1, 0, T - 1)
    fu, fv = u - u0, v - v0
    inside = (u >= 0) & (u <= T - 1) & (v >= 0) & (v <= T - 1)
    out = (tip[v0, u0] * (1 - fu) * (1 - fv) + tip[v0, u1] * fu * (1 - fv)
           + tip[v1, u0] * (1 - fu) * fv + tip[v1, u1] * fu * fv)
    return np.where(inside, out, 0.0).astype(np.float32)


def png_bytes(img):
    """Encode an RGB(A) float image [0,1] to PNG. Uses Pillow if present, else a stdlib fallback."""
    a = np.clip(_f32(img), 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, 2)
    u8 = (a * 255 + 0.5).astype(np.uint8)
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(u8).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # minimal stdlib PNG (RGB / RGBA, no filtering)
        import struct, zlib
        h, w = u8.shape[:2]
        ctype = 6 if u8.shape[2] == 4 else 2
        raw = b"".join(b"\x00" + u8[y].tobytes() for y in range(h))
        def chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6))
                + chunk(b"IEND", b""))


def image_dpi(data: bytes):
    """The DPI recorded in an image file, or None. A numpy array cannot carry
    an attribute, so this is asked for separately rather than smuggled."""
    try:
        from PIL import Image
        dpi = Image.open(io.BytesIO(data)).info.get("dpi")
        return float(dpi[0]) if dpi else None
    except Exception:
        return None


def decode_image(data: bytes):
    """Decode PNG/JPEG/... bytes to an RGBA float image (Pillow when available)."""
    from PIL import Image  # [ui]/[images] extra
    raw = Image.open(io.BytesIO(data))
    dpi = raw.info.get("dpi")
    im = raw.convert("RGBA")
    arr = _f32(np.asarray(im)) / 255.0
    return arr


# ------------------------------------------------------------------------------------------------
# Blend modes + compositor
# ------------------------------------------------------------------------------------------------

def _bl_normal(b, t):   return t
def _bl_multiply(b, t): return b * t
def _bl_screen(b, t):   return 1 - (1 - b) * (1 - t)
def _bl_overlay(b, t):  return np.where(b < 0.5, 2 * b * t, 1 - 2 * (1 - b) * (1 - t))
def _bl_add(b, t):      return np.clip(b + t, 0, 1)
def _bl_subtract(b, t): return np.clip(b - t, 0, 1)
def _bl_difference(b, t): return np.abs(b - t)
def _bl_darken(b, t):   return np.minimum(b, t)
def _bl_lighten(b, t):  return np.maximum(b, t)
def _bl_softlight(b, t): return np.clip((1 - 2 * t) * b * b + 2 * t * b, 0, 1)

BLEND_MODES = {
    "normal": _bl_normal, "multiply": _bl_multiply, "screen": _bl_screen, "overlay": _bl_overlay,
    "add": _bl_add, "subtract": _bl_subtract, "difference": _bl_difference,
    "darken": _bl_darken, "lighten": _bl_lighten, "softlight": _bl_softlight,
}


def _box_half(a):
    """Halve an image with a 2x2 box mean.

    _resize() costs ~150 ms on a 1920x1080x4 buffer; this is ~22 ms for the
    same result at exactly half size, because it is four adds and a multiply
    rather than a general resampler. Only used for DISPLAY reductions, where
    the factor is always exactly 2."""
    h = a.shape[0] // 2 * 2
    w = a.shape[1] // 2 * 2
    v = a[:h, :w]
    if a.ndim == 3:
        v = v.reshape(h // 2, 2, w // 2, 2, a.shape[2])
        return (0.25 * (v[:, 0, :, 0] + v[:, 0, :, 1] +
                        v[:, 1, :, 0] + v[:, 1, :, 1])).astype(np.float32)
    v = v.reshape(h // 2, 2, w // 2, 2)
    return (0.25 * (v[:, 0, :, 0] + v[:, 0, :, 1] +
                    v[:, 1, :, 0] + v[:, 1, :, 1])).astype(np.float32)


def composite_display(layers, h, w, masks=None, max_w=None):
    """Composite for the SCREEN: identical maths, optionally at a reduced size.

    Compositing 1920x1080 costs ~800 ms for four layers, and most of those
    pixels are thrown away by a canvas displayed at 60% zoom. Halving (by
    powers of two only, and never below the width actually being displayed)
    cuts the pixel count 4x per step. `max_w=None` keeps full resolution, so
    export and any pixel-exact path is untouched."""
    # halve only while the result still has AT LEAST as many pixels as the
    # screen is showing -- never send the canvas fewer pixels than it displays
    if not max_w or (w >> 1) < max_w:
        # Reduction is POWER-OF-TWO only, and that is deliberate. A 1200 px
        # document in a 984 px pane therefore composites at full size, which
        # looks like an obvious waste -- it is not. MEASURED: reducing every
        # layer to 984 with the bilinear resampler and compositing there took
        # 730 ms against 412 ms for the full-resolution composite. The
        # halving path is cheap because _box_half is a strided mean; an
        # arbitrary ratio needs a real resample of every layer, and five
        # 1200x900x4 resamples cost more than the composite they save.
        # Fidelity was fine (max 0.0001) -- it is purely a speed loss.
        return composite(layers, h, w, masks)
    # Reducing before compositing only commutes for the NORMAL blend. Measured
    # worst-case error of a half-reduction, per blend mode:
    #   normal 0.0000 (opaque) | multiply 0.16 | screen 0.18 | softlight 0.18
    #   overlay 0.32 | darken 0.36 | subtract 0.38 | lighten 0.41
    #   add 0.42 | difference 0.69
    # so anything non-linear takes the exact full path. Masks are fine --
    # reducing the mask alongside its layer measured exactly 0.0000. Normal
    # blend with partial alpha is not bit-exact, but on real artwork (smooth
    # colour, soft-edged alpha) it measured max 0.0004: invisible.
    if any(l.visible and getattr(l, "blend", "normal") != "normal"
           for l in layers):
        return composite(layers, h, w, masks)
    k = 0
    while k < 3 and (w >> (k + 1)) >= max_w:
        k += 1
    if k == 0:
        return composite(layers, h, w, masks)

    class _MaskShim:                   # a stand-in mask holding reduced data
        __slots__ = ("data",)

        def __init__(self, d):
            self.data = d

    class _Shim:                       # a stand-in layer holding reduced pixels
        __slots__ = ("pixels", "visible", "opacity", "blend",
                     "mask", "mask_invert", "clip")

    small_masks = {}
    if masks:
        for mid, m in masks.items():
            d = m.data
            for _ in range(k):
                d = _box_half(d)
            small_masks[mid] = _MaskShim(d)
    out_layers = []
    for l in layers:
        px = _shaded_pixels(l)
        for _ in range(k):
            px = _box_half(px)
        o = _Shim()
        o.pixels, o.visible, o.opacity = px, l.visible, l.opacity
        o.blend, o.mask = l.blend, getattr(l, "mask", None)
        o.mask_invert = getattr(l, "mask_invert", False)
        o.clip = getattr(l, "clip", False)
        out_layers.append(o)
    return composite(out_layers, px.shape[0], px.shape[1], small_masks or None)


# Impasto media. `hold` is how much height a pixel keeps before gravity takes
# the excess (viscosity), `flow` how much of the excess moves per step, `iters`
# how long the paint stays mobile after the brush lifts, `gloss`/`shin` the
# specular look. Oil holds a tall ridge and shines; watery media barely hold
# and run, taking pigment with them.
# `bristle` = how deeply the comb of the brush survives in the paint, `berm`
# = how much paint the bristles shove sideways into rims. Both track how
# STIFF the paint is: acrylic holds a bristle mark best, oil slumps a little,
# watercolour is a fluid and keeps almost nothing.
_MEDIA = {
    # gloss/shin are lower and broader than the pre-slope model needed: once
    # the surface has real slopes, a tight bright highlight reads as wet
    # plastic. Oil is satin, not lacquer.
    "oil":     {"hold": 0.62, "flow": 0.22, "iters": 10, "gloss": 0.34,
                "shin": 17.0, "bristle": 0.54, "berm": 0.85, "pickup": 0.75},
    "acrylic": {"hold": 0.80, "flow": 0.16, "iters": 6,  "gloss": 0.20,
                "shin": 11.0, "bristle": 0.70, "berm": 0.70, "pickup": 0.60},
    # Watercolour is not a thin film on a surface -- it is a fluid IN paper.
    # `absorb` wicks the wash outward through the fibres, `edge_dark` is the
    # dark rim left as water evaporates and carries pigment to the perimeter,
    # `granulate` is pigment settling into the paper's valleys, and `settle`
    # says pigment pools in the LOW places rather than catching on the high
    # ones (see `_deposit`). Curtis et al., SIGGRAPH 97.
    "water":   {"hold": 0.10, "flow": 0.45, "iters": 26, "gloss": 0.05,
                "shin": 5.0,  "bristle": 0.12, "berm": 0.10, "pickup": 0.30,
                "absorb": 0.9, "edge_dark": 0.7, "granulate": 0.42,
                "settle": 1.0},
}

# Height is stored in "paint units"; this converts a unit of height into the
# surface SLOPE the light sees. Without it `np.gradient` on a 1.6-unit ridge
# spread over a 40px radius yields near-flat normals, and every stroke read
# as an airbrushed sticker no matter how much body it had -- the single
# biggest reason the old impasto did not look like paint.
# 4.5, tuned against a dense field of small strokes rather than one fat
# swatch: at 7.0 a radius-8 stroke became a glossy gel tube, because a narrow
# ridge of the same height has far steeper flanks than a wide one.
# How much paint one layer can hold before it is FULL. Beyond this the
# height field was simply clipped, so a heavily worked passage saturated
# after about two loaded passes and every further stroke added nothing --
# the whole mark flattened to a plateau at exactly this value, which is the
# inorganic stepped look. With `Document.auto_stratum` the excess instead
# SPILLS ONTO A NEW LAYER that starts from zero, the way a painter builds a
# heavy passage up in campaigns rather than in one impossible slab.
_HEIGHT_CAP = 4.0

_RELIEF_SLOPE = 4.5

# How much of the canvas weave stands proud in the shading. Without it a
# stroke is an extrusion floating on a perfect plane; paint in the world sits
# IN a surface, and the surrounding tooth is most of what says so.
_CANVAS_RELIEF = 0.30

# How much paint it takes to BURY the weave. Paint is a fluid: it fills the
# cavities of the canvas and levels off, so a thick passage has a smooth,
# globby top surface with no trace of the substrate in it. Only where the
# film is thin does the weave still read -- paint sitting in the valleys and
# scraped bare off the risen threads, which is exactly what dry-brush is.
# Adding the tooth to every pixel's height regardless of thickness printed
# canvas texture onto the top of impasto, which no real painting does.
_PAINT_LEVEL = 0.40

_LIGHT = np.array([-0.45, -0.6, 0.66], np.float32)   # top-left key light
_LIGHT /= np.linalg.norm(_LIGHT)


# PBR materials: painting with STUFF, not just colour. A material stroke lays
# pigment (the brush colour is the albedo), a paint body (same physics as
# media), and two per-pixel surface properties -- roughness and metalness --
# into the layer's material map. The composite lights those properties per
# pixel, so a gold stroke gleams like gold NEXT TO a chalk stroke that stays
# dead matte on the same layer; the old per-layer "last media wins" scalar
# gloss could never say that.
#
# rough: 0 = mirror-tight highlight, 1 = fully matte
# metal: 0 = dielectric (white highlight, full diffuse),
#        1 = metal (highlight TINTED BY THE ALBEDO -- gold's gleam is gold --
#            and diffuse falls away, which is why metals look "dark" away
#            from the light)
# grain/gscale: micro-relief laid with the stroke (hammered/toothy surfaces),
#        a POSITION-STABLE field so replays re-deposit the identical texture
# hold/flow/iters: the paint body, same meaning as _MEDIA
# color: a SUGGESTED albedo for the UI swatch only -- paint() never applies
#        it; the brush colour is always the albedo
# The set is limited to what the shading model can honestly show: no velvet
# (needs retroreflection), no pearl (needs iridescence).
_MATERIALS = {
    "gold":    {"rough": 0.28, "metal": 1.0, "grain": 0.06, "gscale": 2.0,
                "hold": 0.85, "flow": 0.10, "iters": 4,
                "gloss": 0.85, "shin": 42.0, "color": (1.0, 0.78, 0.34)},
    "silver":  {"rough": 0.18, "metal": 1.0, "grain": 0.04, "gscale": 2.0,
                "hold": 0.85, "flow": 0.10, "iters": 4,
                "gloss": 0.9, "shin": 60.0, "color": (0.91, 0.92, 0.94)},
    "copper":  {"rough": 0.32, "metal": 1.0, "grain": 0.08, "gscale": 2.0,
                "hold": 0.85, "flow": 0.10, "iters": 4,
                "gloss": 0.8, "shin": 38.0, "color": (0.95, 0.54, 0.36)},
    "chrome":  {"rough": 0.05, "metal": 1.0, "grain": 0.02, "gscale": 3.0,
                "hold": 0.9, "flow": 0.08, "iters": 3,
                "gloss": 1.0, "shin": 100.0, "color": (0.87, 0.9, 0.93)},
    "brushed_steel": {"rough": 0.45, "metal": 1.0, "grain": 0.18,
                      "gscale": 1.2, "hold": 0.9, "flow": 0.08, "iters": 3,
                      "gloss": 0.6, "shin": 20.0, "color": (0.75, 0.77, 0.8)},
    "lacquer": {"rough": 0.06, "metal": 0.0, "grain": 0.0, "gscale": 2.0,
                "hold": 0.62, "flow": 0.2, "iters": 8,
                "gloss": 0.7, "shin": 90.0, "color": None},
    "plastic": {"rough": 0.3, "metal": 0.0, "grain": 0.0, "gscale": 2.0,
                "hold": 0.8, "flow": 0.12, "iters": 4,
                "gloss": 0.4, "shin": 34.0, "color": None},
    "rubber":  {"rough": 0.85, "metal": 0.0, "grain": 0.05, "gscale": 1.6,
                "hold": 0.85, "flow": 0.1, "iters": 3,
                "gloss": 0.1, "shin": 6.0, "color": None},
    "wax":     {"rough": 0.55, "metal": 0.0, "grain": 0.04, "gscale": 2.4,
                "hold": 0.5, "flow": 0.22, "iters": 8,
                "gloss": 0.22, "shin": 12.0, "color": None},
    "clay":    {"rough": 0.82, "metal": 0.0, "grain": 0.12, "gscale": 1.8,
                "hold": 0.9, "flow": 0.1, "iters": 4,
                "gloss": 0.08, "shin": 5.0, "color": None},
    "chalk":   {"rough": 1.0, "metal": 0.0, "grain": 0.3, "gscale": 1.0,
                "hold": 0.35, "flow": 0.05, "iters": 2,
                "gloss": 0.03, "shin": 3.0, "color": None},
}


def _mat_resample(mm, fn):
    """Resample a material map through `fn` PREMULTIPLIED by coverage.

    Bilinear interpolation of raw [rough, metal, coverage] mixes edge
    properties with the 0/0/0 of uncovered neighbours, dragging roughness
    toward 0 -- a faint CHROME HALO around every resized material stroke.
    Weighting by coverage first is the same hygiene the pixel path applies
    to colour vs alpha."""
    pm = mm.copy()
    pm[..., 0] *= mm[..., 2]
    pm[..., 1] *= mm[..., 2]
    out = fn(pm)
    cov = np.maximum(out[..., 2:3], 1e-6)
    out[..., 0:2] = out[..., 0:2] / cov
    out[..., 2] = np.clip(out[..., 2], 0.0, 1.0)
    return out


_TOOTH_CACHE = {}
_TOOTH_CACHE_HOLD = []


def _canvas_tooth(doc):
    return _tooth_hw(doc.height, doc.width, _paper_of(doc)[0])


# PAPER AND CANVAS STOCKS. The substrate was one fixed field, so every
# surface behaved like the same mid-grain canvas -- and the substrate decides
# a great deal: where thin paint catches (stiff paint on the risen threads),
# where a wash pools (in the dips), how hard dry-brush breaks up, and how
# strongly watercolour granulates. `grain` scales the feature size, `weave`
# how much of the crossed-thread structure shows over the fbm, `depth` the
# overall amplitude, and `drink` how thirsty the surface is (watercolour
# wicks further into rough rag than into sized hot-press).
_PAPERS = {
    "canvas":     {"grain": 3.0, "weave": 0.20, "depth": 1.0,  "drink": 1.0},
    "rough":      {"grain": 6.5, "weave": 0.06, "depth": 1.55, "drink": 1.35},
    "cold_press": {"grain": 4.2, "weave": 0.10, "depth": 1.1,  "drink": 1.15},
    "hot_press":  {"grain": 2.2, "weave": 0.05, "depth": 0.42, "drink": 0.8},
    "smooth":     {"grain": 1.6, "weave": 0.02, "depth": 0.14, "drink": 0.6},
    "linen":      {"grain": 3.4, "weave": 0.42, "depth": 1.2,  "drink": 0.95},
}


def _paper_of(doc):
    name = str(getattr(doc, "paper", "canvas") or "canvas")
    return name if name in _PAPERS else "canvas", _PAPERS.get(name,
                                                              _PAPERS["canvas"])


def _tooth_hw(H, W, paper="canvas"):
    """The SUBSTRATE: woven canvas micro-geometry, 0..1, one field per doc.

    Real paint does not meet a perfect plane -- it meets a weave, and thin
    paint only catches on the peaks. That single fact is what produces
    dry-brush, scumbling and the broken edge of a fast stroke; without a
    substrate every stroke lands with the same dead-even coverage.

    Built from composite noise (fbm) plus a crossed-sinusoid weave term, the
    recipe in Liu et al. 2026 (arXiv:2604.02752) sec.7. Procedural and seeded
    by the document, so it is position-stable: replays and re-cooks land on
    the identical tooth, and it costs nothing to store.
    """
    pk = paper if paper in _PAPERS else "canvas"
    pp = _PAPERS[pk]
    t = _TOOTH_CACHE.get((H, W, pk))  # local rebind: safe against a swap
    if t is not None:
        return t
    # Seeded by a FIXED constant, not the document id: the weave is a
    # property of the canvas stock, so the same stroke must land identically
    # in any document. Keying it to doc.id silently gave every document a
    # different substrate -- a live stroke and its committed twin diverged,
    # and a save/load round trip re-rolled the texture under the paint.
    rng = np.random.default_rng(0x0CA9A5)
    acc = np.zeros((H, W), np.float32)
    amp, sc = 1.0, float(pp["grain"])
    for _ in range(4):                       # fbm
        nh, nw = max(int(H / sc) + 2, 2), max(int(W / sc) + 2, 2)
        n = _gauss_small(rng.random((nh, nw)).astype(np.float32)[..., None],
                         1.0)[..., 0]
        yi = np.clip(np.round(np.linspace(0, nh - 1, H)).astype(int), 0, nh - 1)
        xi = np.clip(np.round(np.linspace(0, nw - 1, W)).astype(int), 0, nw - 1)
        acc += amp * n[yi][:, xi]
        amp *= 0.5
        sc = max(sc * 0.5, 1.0)
    acc = (acc - acc.min()) / max(float(acc.max() - acc.min()), 1e-6)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    # 2.1 rad/px (a ~3px thread), not 0.7 (~9px): at the coarse frequency the
    # crossed sinusoids read as a diagonal corduroy rib rather than canvas,
    # and it dominated the fbm instead of sitting under it.
    # 0.72 rad/px (~8.7px thread), not 2.1 (~3px). At 3px the weave beat
    # against the bristle pitch (~3.3px) and the two produced a low-frequency
    # moire -- a plaid crosshatch over every stroke that looked like a render
    # artifact because it was one. Keeping the substrate well away from the
    # hair pitch removes the beat entirely.
    wf = 0.72 * (3.0 / max(float(pp["grain"]), 0.5))
    weave = 0.5 + 0.5 * np.sin(xx * wf) * np.sin(yy * wf)
    wv = float(pp["weave"])
    t = np.clip((1.0 - wv) * acc + wv * weave, 0.0, 1.0).astype(np.float32)
    # depth flattens the whole field toward its mean rather than clipping it:
    # a hot-press sheet is a shallow tooth, not a truncated rough one
    d = float(pp["depth"])
    t = np.clip(0.5 + (t - 0.5) * d, 0.0, 1.0).astype(np.float32)
    # Rebind rather than mutate: composites are served concurrently, and
    # clear()-then-insert on a shared dict leaves a window where a reader
    # sees neither the old entry nor the new one.
    if len(_TOOTH_CACHE) > 6:
        _TOOTH_CACHE_HOLD.append({})
        globals()["_TOOTH_CACHE"] = _TOOTH_CACHE_HOLD[-1]
    _TOOTH_CACHE[(H, W, pk)] = t
    return t


def _stroke_frame(dense, dwid, radius, x0b, y0b, x1b, y1b):
    """Stroke-LOCAL coordinates for every pixel in the stroke's box.

    Returns (u, s): `u` is the signed distance across the stroke normalised
    by the local brush radius (-1 = one edge, +1 = the other), `s` is how far
    along the stroke the pixel sits, 0..1. Everything that makes a stroke
    look like a brush dragged it needs one or both: bristle furrows run along
    the stroke at fixed `u`, rims sit at |u| ~ 0.8, and the load runs out
    with `s`. The old model had neither coordinate, which is precisely why
    every stroke was a featureless dome.

    Nearest-segment projection, walked at a stride of about one radius so
    the cost tracks brush size rather than point count.
    """
    W, H = x1b - x0b, y1b - y0b
    P = np.asarray(dense, np.float32)
    if len(P) < 2 or W <= 0 or H <= 0:
        return None
    seg = np.hypot(*(P[1:] - P[:-1]).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = max(float(arc[-1]), 1e-6)
    best = np.full((H, W), np.inf, np.float32)
    u = np.zeros((H, W), np.float32)
    sv = np.zeros((H, W), np.float32)
    ov = np.zeros((H, W), np.float32)
    stride = max(1, int(radius * 0.8))
    idx = list(range(0, len(P) - 1, stride))
    if idx[-1] != len(P) - 2:
        idx.append(len(P) - 2)
    for i in idx:
        j = min(i + stride, len(P) - 1)
        ax, ay = float(P[i][0]), float(P[i][1])
        bx, by = float(P[j][0]), float(P[j][1])
        wi = float(dwid[i]) if i < len(dwid) else 1.0
        wj = float(dwid[j]) if j < len(dwid) else wi
        r = radius * max(wi, 0.05)          # window sizing only
        wx0 = max(x0b, int(min(ax, bx) - r - 2))
        wx1 = min(x1b, int(max(ax, bx) + r + 3))
        wy0 = max(y0b, int(min(ay, by) - r - 2))
        wy1 = min(y1b, int(max(ay, by) + r + 3))
        if wx1 <= wx0 or wy1 <= wy0:
            continue
        yy = np.arange(wy0, wy1, dtype=np.float32)[:, None]
        xx = np.arange(wx0, wx1, dtype=np.float32)[None, :]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        traw = (((xx - ax) * dx + (yy - ay) * dy) / L2
                if L2 > 1e-9 else np.zeros((wy1 - wy0, wx1 - wx0), np.float32))
        traw = np.broadcast_to(traw, (wy1 - wy0, wx1 - wx0))
        t = np.clip(traw, 0.0, 1.0)
        # how far past the START or END of the whole path a pixel sits. The
        # round cap comes from distance-to-endpoint being radial; with an
        # axial measure each hair can be cut to its own length instead, which
        # is what makes a brush end chisel-shaped and frayed rather than a
        # capsule.
        seglen = float(np.sqrt(L2)) if L2 > 1e-9 else 0.0
        ovs = np.zeros_like(t)
        if i == idx[0]:
            ovs = np.maximum(ovs, np.maximum(-traw, 0.0) * seglen)
        if j >= len(P) - 1:
            ovs = np.maximum(ovs, np.maximum(traw - 1.0, 0.0) * seglen)
        ox = xx - (ax + t * dx)
        oy = yy - (ay + t * dy)
        dist = np.hypot(ox, oy)
        # normalise by the width AT THIS PIXEL, interpolated along the
        # segment. One width per stride made u's scale piecewise-constant, so
        # a stroke whose width changed showed the stride boundaries as hard
        # steps -- the tiling seam down a pressure ramp.
        rpix = np.maximum(radius * (wi + (wj - wi) * t), 1e-3)
        # signed side of the path: the cross product's sign, so the two
        # flanks of the stroke are distinguishable (a rim on each)
        sgn = np.sign(dx * oy - dy * ox)
        sgn = np.where(sgn == 0, 1.0, sgn)
        win = np.s_[wy0 - y0b:wy1 - y0b, wx0 - x0b:wx1 - x0b]
        take = dist < best[win]
        best[win] = np.where(take, dist, best[win])
        u[win] = np.where(take, sgn * dist / rpix, u[win])
        sv[win] = np.where(
            take, (arc[i] + t * (arc[j] - arc[i])) / total, sv[win])
        ov[win] = np.where(take, ovs / rpix, ov[win])
    return u, sv, total, ov


def _bristle_tracks(u, sv, ov, seed, n, press, spd):
    """Coverage as the UNION OF BRISTLE TRACKS -- the mark generator.

    A brush does not lay a capsule. Each hair drags its own narrow track, and
    the stroke's outline is the union of those tracks: solid through the
    middle where neighbouring hairs overlap, breaking into separate streaks
    and gaps toward the edges, ragged rather than a clean offset curve. Every
    earlier attempt here modulated a disc-shaped mask AFTER the fact -- comb,
    tooth, edge bite, load wobble -- and the capsule kept showing through,
    because decoration cannot change a silhouette. This changes the
    silhouette.

    Evaluated ANALYTICALLY from the stroke-local frame rather than by
    stamping each hair: a hair covers a pixel when |u - offset_b(s)| < its
    half-width, which is a per-pixel test. Built as a small 2-D table over
    (arc bucket, cross-stroke position) and read with one gather, so the cost
    is flat in bristle count -- stamping 26 hairs along 200 path points would
    have been thousands of small array ops per stroke.

    Pressure splays the hairs apart and presses more of them into contact;
    speed lifts them, so a fast pass rides on fewer hairs and breaks up.
    """
    rng = np.random.default_rng(seed)
    n = int(np.clip(n, 6, 64))
    # 24 x 512, not 64 x 1024. Both axes are read with linear interpolation,
    # and the functions sampled are slow: the contact flicker is under one
    # cycle along the whole stroke, and 512 samples across the tuft is still
    # ~5 per pixel on a fat brush. The larger table cost 4x the work for
    # detail that interpolation was reconstructing anyway.
    S, K = 24, 512
    lo, hi = -1.7, 1.7
    axis = np.linspace(lo, hi, K).astype(np.float32)
    sb = np.linspace(0.0, 1.0, S).astype(np.float32)
    pmean = float(np.clip(np.mean(press), 0.05, 2.0))
    smean = float(np.clip(np.mean(spd), 0.0, 2.0))
    # splay: leaning on the brush pushes the hairs apart and flattens the tuft
    splay = float(np.clip(0.72 + 0.42 * pmean, 0.6, 1.5))
    spacing = 2.0 * splay / max(n - 1, 1)
    pos = (np.linspace(-splay, splay, n)
           + rng.uniform(-0.34, 0.34, n).astype(np.float32) * spacing)
    # hairs overlap in the middle (solid) and thin out at the rim
    hw = spacing * (0.52 + 0.40 * rng.random(n).astype(np.float32))
    edge = np.abs(pos) / max(splay, 1e-3)
    # contact: how firmly each hair rides. Outer hairs touch more lightly,
    # pressure presses more of them home, speed lifts them off
    base = np.clip((0.98 - 0.45 * edge ** 2) * (0.55 + 0.5 * pmean)
                   / (1.0 + 0.45 * smean), 0.0, 1.0)
    wob = rng.uniform(0.0, 6.28, n).astype(np.float32)
    # Built for ALL hairs at once. The per-hair Python loop ran one exp() per
    # bristle over the whole table -- about two million exponentials for a
    # single flush, which made this the third-hottest thing in the paint path
    # for a table that is pure setup. Broadcasting gives one exp() call.
    # Each hair wanders a little along the stroke and its contact flickers:
    # that is where interior gaps and dry skips come from.
    off = pos[:, None] + 0.045 * np.sin(wob[:, None] + sb[None, :] * 7.0)
    con = np.clip(base[:, None] * (0.90 + 0.12 * np.sin(
        wob[:, None] * 1.7 + sb[None, :] * 5.0)), 0.0, 1.0)
    d = (np.abs(axis[None, None, :] - off[:, :, None])
         / np.maximum(hw, 1e-3)[:, None, None])
    # a hair is a rounded thread, not a flat-topped ribbon: at 1-d**3 the
    # profiles were plateaus that merged into a few thick bands instead of
    # reading as many fine hairs
    lut = (np.exp(-(d * d) * 2.2) * con[:, :, None]).max(0).astype(np.float32)
    # INTERPOLATE along the arc. Nearest-bucket made the hair pattern jump in
    # S discrete steps down the stroke -- visible as regular vertical seams,
    # a tiling artifact rather than anything physical.
    sf = np.clip(sv, 0.0, 1.0) * (S - 1)
    s0 = np.floor(sf).astype(np.int32)
    s1 = np.minimum(s0 + 1, S - 1)
    w1 = (sf - s0).astype(np.float32)
    uf = np.clip((u - lo) * ((K - 1) / (hi - lo)), 0.0, K - 1)
    u0 = np.floor(uf).astype(np.int32)
    u1 = np.minimum(u0 + 1, K - 1)
    wu = (uf - u0).astype(np.float32)
    def _bi(si_):
        return lut[si_, u0] * (1.0 - wu) + lut[si_, u1] * wu
    field = _bi(s0) * (1.0 - w1) + _bi(s1) * w1
    ui = u0
    # THE END OF THE STROKE. A brush does not stop in a neat semicircle: the
    # hairs are different lengths and the tuft is chisel-shaped, so the mark
    # ends ragged. Each hair is cut at its own reach past the path end.
    reach = (0.02 + 0.55 * rng.random(n).astype(np.float32))
    rl = (np.exp(-((axis[None, :] - pos[:, None])
                   / np.maximum(hw * 1.6, 1e-3)[:, None]) ** 2)
          * reach[:, None]).max(0).astype(np.float32)
    # the end gate is handed back separately: coverage needs it even where
    # the paint film is continuous and not driven by the hairs
    egate = (ov <= rl[ui]).astype(np.float32)
    return field * egate, egate

def _bristle_comb(u, sv, seed, n):
    """The comb the bristles leave: grooves at FIXED lateral positions that
    run the whole length of the stroke. Fixed position is the whole point --
    noise sprinkled per pixel reads as dirt, whereas a bristle holds its lane
    and draws a continuous furrow, which is what the eye recognises as a
    brush. The lanes drift very slightly along the stroke (a real hand and a
    real ferrule are not rigid)."""
    rng = np.random.default_rng(seed)
    n = int(np.clip(n, 5, 26))
    pos = np.linspace(-0.92, 0.92, n).astype(np.float32)
    pos = pos + rng.uniform(-0.5, 0.5, n).astype(np.float32) * (1.8 / n)
    amp = rng.uniform(0.45, 1.0, n).astype(np.float32)
    # width as a fraction of the LANE SPACING, not of the stroke: at 0.62 of
    # spacing the Gaussians overlapped into one smooth bump whenever the lane
    # count was low, so small brushes lost the comb entirely and every short
    # stroke rendered as a flawless extruded tube. 0.32 keeps the furrows
    # separate at any brush size.
    spacing = 1.84 / max(n - 1, 1)
    width = max(spacing * 0.32, 0.02)
    # The comb is a function of the cross-stroke coordinate alone, so build
    # it ONCE as a 1-D profile and gather. Evaluating up to 26 Gaussians over
    # the whole stroke window was the most expensive thing in the deposit
    # (26 exp() passes over every pixel); this is one table build plus one
    # integer gather, and it is exact, not an approximation.
    K = 1024
    axis = np.linspace(-1.45, 1.45, K).astype(np.float32)
    prof = np.zeros(K, np.float32)
    for k in range(n):
        prof += amp[k] * np.exp(-((axis - pos[k]) / width) ** 2)
    m = float(prof.max())
    if m > 1e-6:
        prof /= m
    # the ferrule wanders as a whole -- one shared phase, which is also truer
    # than letting each bristle wobble independently of its neighbours
    uu = u + 0.035 * np.sin(float(rng.uniform(0.0, 6.28)) + sv * 9.0)
    idx = np.clip(((uu + 1.45) * ((K - 1) / 2.9)), 0, K - 1).astype(np.int32)
    return prof[idx]


def _deposit(doc, l, med, mask, opacity, load, dense, dwid, radius,
             x0b, y0b, x1b, y1b, seed, load_field=None, dspd=None):
    """Turn a coverage mask into a PAINT SURFACE.

    The old model was `mask * opacity * load * 1.6` -- height as a rescaled
    copy of alpha. It had a number for thickness but no signature of a tool:
    the same smooth dome for every stroke, even along its whole length,
    identical whatever the brush did or crossed. Four mechanisms, each from
    the painting-simulation literature, are what actually make a mark read as
    paint:

      1. LOAD DEPLETION -- the brush empties as it travels (IMPaSTo, Baxter
         et al. 2004, treats paint as conserved rather than stamped), so a
         stroke thins along its length and a long one runs dry.
      2. BRISTLE COMB -- fixed lanes of furrow and ridge running the length
         of the stroke.
      3. CANVAS TOOTH -- deposit modulated by the substrate, so thin paint
         catches only on the weave peaks and breaks up (Chu & Baxter 2010
         modulate the brush footprint with the canvas tooth for exactly
         this); together with (1) this is what produces dry-brush.
      4. BERM -- bristles shove paint sideways, so the cross-section is not
         a dome but a shallow trough between two raised rims. Redistributed
         with volume conserved, because the paint moves, it does not appear.

    Returns (height, coverage_factor). The coverage factor matters as much as
    the height: a drying brush skipping over the weave lays LESS PIGMENT, not
    just less body. Modulating only the height left a solid-colour stroke
    with an embossed texture -- it read as printed vinyl. Dry-brush is a hole
    in the paint film, so the same mechanisms have to reach the alpha.
    """
    dep = mask * float(opacity) * float(load) * 1.6
    fr = _stroke_frame(dense, dwid, radius, x0b, y0b, x1b, y1b)
    if fr is None:
        return dep, None, None          # a single dab has no along-direction
    u, sv, total, ov = fr
    bristle = float(med.get("bristle", np.clip(med.get("hold", 0.6), 0, 1)))
    berm = float(med.get("berm", np.clip(med.get("hold", 0.6) * 0.9, 0, 1)))

    # PRESSURE and SPEED, looked up per pixel by arc position -- the same
    # trick the reservoir uses, so neither costs anything as the brush grows.
    # Pressure already scales the radius when the mask is stamped; what it
    # adds HERE is everything a wider dab alone does not say.
    nD = len(dense)
    aix = np.clip((sv * (nD - 1)).astype(np.int32), 0, nD - 1)
    press = np.clip(np.asarray(dwid, np.float32)[aix]
                    if len(dwid) >= nD else np.ones_like(sv), 0.05, 2.0)
    if dspd is not None and len(dspd) >= nD:
        # brush-widths per sample; ~0.3 is a considered stroke, >1.5 a flick
        spd = np.clip(np.asarray(dspd, np.float32)[aix], 0.0, 2.0)
    else:
        spd = np.zeros_like(sv)

    # 1. the brush empties. How far it gets is set by how much it carries and
    # how wide it is: `spend` is the fraction of its load used over this
    # stroke, so a short dab barely depletes and a long sweep runs dry.
    if load_field is not None:
        # REAL BRUSH MODE: the load is not a per-stroke setting that resets,
        # it is what is actually left on the brush, tracked across strokes.
        # The reservoir walk already accounts for running dry and recharging,
        # so the synthetic per-stroke depletion below must not also apply.
        dep = mask * float(opacity) * np.maximum(load_field, 0.0) * 1.6
        fade = np.clip(load_field / max(float(load), 1e-6), 0.0, 1.0)
    else:
        spend = float(np.clip(total / max(radius * 34.0 * max(load, 0.12), 1e-6),
                              0.0, 1.0))
        fade = 1.0 - spend * 0.70 * sv
        dep = dep * fade

    # PRESS: leaning on a brush squeezes paint out of it, and it splays the
    # bristles so the comb bites HARDER rather than washing out -- press hard
    # and you want the bristle marks to show, which is why this scales up
    # with pressure rather than smoothing away.
    # SPEED: a fast pass gives the paint less contact time to transfer, so a
    # flick lays a thin broken mark. This is where dry-brush comes from for
    # free -- a thinner deposit trips the canvas-tooth gate below by itself,
    # no separate rule needed.
    dep = dep * (0.5 + 0.5 * press) / (1.0 + 0.55 * spd)

    # 2. the comb: lanes scale with brush width, so a big brush shows more
    # 2. THE HAIRS. Not a comb multiplied over a disc afterward -- the tracks
    # ARE the mark, and they carry both the body and the silhouette.
    n = int(np.clip(radius * 0.6, 5, 30))   # ~3.3px hair pitch: fine, but clear of Nyquist
    track, egate = _bristle_tracks(u, sv, ov, seed, n, press, spd)
    # the end gate must reach the BODY as well as the pigment: cutting only
    # coverage left the old capsule cap standing as a ghost ridge with no
    # paint on it, which the relief lit as a pale blob past the stroke's end
    egate = np.maximum(egate, 0.0)
    # The BODY has to follow the hairs as closely as the pigment does. While
    # height still came from the disc and coverage came from the tracks,
    # there was paint thickness in a ring where there was no paint colour --
    # the relief lit those ghost ridges over bare canvas and haloed every
    # stroke. Height and pigment must be laid by the same hairs.
    # Furrows are GROOVES in a film, not canyons. Letting height swing from
    # 6% to 100% across a hair pitch made the relief violent enough to read
    # as shredded meat rather than dragged paint.
    # HOW DEEP THE HAIRS MARK, which is not a constant:
    #  - thick paint LEVELS. The same fluid behaviour that buries the canvas
    #    weave also flows back into the brush's own furrows, so a heavy
    #    passage is smoother and globbier than a thin one. How much survives
    #    depends on how clay-like the paint is: a stiff medium holds its
    #    ridges (that is impasto), a watery one closes them almost entirely.
    #  - PRESSURE drives the hairs into the film. Pressure already sets the
    #    stroke's diameter; it also sets how much of the brush's texture gets
    #    pressed in, which is why a light touch leaves a smooth mark and
    #    leaning on the brush leaves a raked one.
    levelling = np.exp(-np.maximum(dep, 0.0)
                       / (0.22 + 3.5 * float(med.get("hold", 0.6))))
    press_tex = np.clip(0.30 + 0.80 * press, 0.15, 1.4)
    depth = np.clip(bristle * press_tex * levelling, 0.0, 1.0)
    comb = (1.0 - depth * 0.44 * (1.0 - track)) * egate
    # a real hand and real paint are not uniform: the load fluctuates along
    # the stroke, so the slab is never a constant plateau
    # Gentle and LONG-WAVELENGTH. A random value per arc bucket is applied
    # uniformly across the full width of the stroke -- which is by definition
    # a vertical band. At 96 buckets with light smoothing the bands landed
    # every ~10px on a short stroke and read as a render artifact. Few
    # buckets, heavy smoothing and a small amplitude give the hand-tremor
    # this was meant to be instead of a barcode.
    # Bucket count must follow the stroke's PHYSICAL LENGTH, not be fixed.
    # At a constant K the wobble's wavelength scaled with the stroke, so a
    # short mark got buckets every few pixels and banded -- and the levelling
    # term below amplifies any variation in `dep`, which made it obvious
    # again exactly where the paint is thickest. ~55px per bucket keeps the
    # tremor long-wavelength on a dab and on a sweep alike.
    rngv = np.random.default_rng(seed ^ 0x5EED)
    K = int(np.clip(total / 55.0, 3, 64))
    wob = rngv.random(K).astype(np.float32)
    kb = max(3, min(11, K))
    wob = np.convolve(np.concatenate([wob[-kb:], wob, wob[:kb]]),
                      np.ones(kb, np.float32) / kb, "same")[kb:-kb]
    wob = 1.0 + 0.06 * (wob - wob.mean()) / max(float(wob.std()), 1e-6)
    wf = np.clip(sv, 0.0, 1.0) * (K - 1)
    w0 = np.floor(wf).astype(np.int32)
    wobp = (wob[w0] * (1.0 - (wf - w0))
            + wob[np.minimum(w0 + 1, K - 1)] * (wf - w0)).astype(np.float32)
    dep = dep * comb * wobp

    # 3. the tooth: gate hardest where the paint is thinnest, so heavy loads
    # bury the weave and a starved brush skips across its peaks
    tooth = _canvas_tooth(doc)[y0b:y1b, x0b:x1b]
    thin = np.clip(1.0 - dep / 0.55, 0.0, 1.0)
    # `settle` flips which way the substrate works. Stiff paint dragged by a
    # brush is scraped off the risen threads and catches on their tops; a
    # fluid wash runs off the tops and POOLS IN THE DIPS. Same tooth field,
    # opposite sign, and using the stiff-paint rule for watercolour was
    # putting the wash exactly where the water would have run out of.
    settle = float(med.get("settle", 0.0))
    high = 1.0 - thin * (1.0 - tooth) * 0.92
    low = 1.0 - thin * tooth * 0.92
    grip = high * (1.0 - settle) + low * settle
    dep = dep * grip

    # The film breaks only where the brush is genuinely STARVED. Applying the
    # thinning to coverage flat made even a fully loaded stroke lay a 40%
    # transparent film -- every medium looked like a watercolour wash. The
    # offset holds coverage pinned at 1 until the surviving deposit drops
    # well below a full load, and only then lets the weave show through.
    # (The berm below is pure redistribution of height, so it never reaches
    # coverage: banking paint sideways must not make the stroke see-through.)
    # THE SILHOUETTE. A bristle brush does not lay a capsule. Its outline is
    # the union of individual hair tracks: solid through the middle where the
    # lanes overlap, but breaking into separate streaks and gaps toward the
    # edges, so the boundary is ragged and a few hairs run out past the body.
    # Holding the film's floor constant across the whole width made every
    # stroke a clean rounded bar with a hard rim -- the extruded-plastic look.
    # The floor now falls off toward the rim, so the comb fully controls
    # coverage out there and the lanes separate into hairs.
    # COVERAGE IS THE TRACK FIELD. Where no hair passed there is no paint --
    # a true zero, not a floored minimum -- so the boundary is the union of
    # the hairs and comes out ragged, with gaps and stray streaks, instead of
    # the clean offset curve a disc mask always produces. A well-loaded brush
    # still reads solid because neighbouring hairs overlap through the middle.
    # PAINT IS A FILM THAT THE HAIRS FURROW -- it is not a bundle of threads.
    # Letting the track field drive coverage across the whole width punched
    # canvas between every hair, so a loaded stroke read as combed fibre and
    # the outermost hair became a bright line outlining the mark. A loaded
    # brush lays a CONTINUOUS film; the hairs show as ridges in the HEIGHT,
    # which is what the relief is for. Holes belong in exactly two places:
    # at the rim, where the outer hairs really do separate, and wherever the
    # brush is starved or riding the weave -- which is dry-brush.
    wet = np.clip(fade * grip * wobp, 0.0, 1.0)
    # AN EMPTY BRUSH MUST LAY NOTHING. `film` has a 0.5 floor so that a
    # merely tired brush still puts colour down -- but that floor ignored the
    # reservoir, so painting with a brush at zero charge still laid a 50%
    # film and real-brush mode meant nothing. Fade coverage out over the last
    # sliver of charge instead of cutting it off, or the stroke would end in
    # a hard edge mid-air.
    dry_gate = 1.0
    if load_field is not None:
        dry_gate = np.clip(load_field / max(float(load) * 0.18, 1e-6), 0.0, 1.0)
    rimness = np.clip((np.abs(u) - 0.5) / 0.5, 0.0, 1.0)
    breakup = np.clip(rimness + 1.5 * (1.0 - wet), 0.0, 1.0)
    film = np.clip(0.5 + 0.95 * wet, 0.0, 1.0)
    cover = np.clip(film * (1.0 - breakup * (1.0 - track) * depth),
                    0.0, 1.0) * egate * dry_gate

    # 4. the berm, volume-conserving: take from the middle, pile at the rims
    # a hurried brush has no time to bank paint sideways, a leaning one
    # shoves plenty: the rim is where pressure and speed argue directly
    berm = berm * float(np.clip(np.mean((0.55 + 0.55 * press)
                                        / (1.0 + 0.5 * spd)), 0.0, 1.6))
    if berm > 1e-3:
        au = np.abs(u)
        # broad and low. A narrow tall rim reads as piped icing: two hard
        # parallel lines inside the stroke rather than paint banked up by a
        # bristle. The bank is wide and its crest sits near the edge.
        trough = 1.0 - berm * 0.30 * np.exp(-(u * 1.5) ** 2)
        rim = np.exp(-((au - 0.70) / 0.34) ** 2) * (au < 1.3)
        before = float(dep.sum())
        dep = dep * trough + berm * 0.26 * rim * dep
        after = float(dep.sum())
        if after > 1e-6:
            dep *= before / after       # the paint moved; none was created
    return np.clip(dep, 0.0, None), cover, sv


_BRUSH_LANES = 7          # bands across the tuft that hold colour separately


def _brush_walk(l, dense, color, media_hold, pickup, charge0, spend_rate,
                reload_rate, radius=12.0):
    """Walk the brush along its path, tracking what it CARRIES.

    One 1-D recurrence over the path points returns three tables indexed by
    arc position: the colour on the brush, the charge remaining, and how much
    paint was lifted off the canvas at each step. Everything expensive about
    a physical brush -- mixing, running dry, recharging from thick paint --
    falls out of this single walk, and because the result is looked up per
    pixel by arc length, none of it costs anything as the brush gets bigger.

    Charge falls as paint is laid and rises where the brush is dragged
    through paint that has body. Recharge scales with how EMPTY the brush is:
    a loaded brush mostly deposits, a dry one mostly lifts -- which is what
    makes "go and get more paint from a thick area" work without a mode
    switch. What it lifts, the canvas loses (`taken`), because paint is
    conserved: picking colour up has to leave a scrape behind.
    """
    P = np.asarray(dense, np.float32)
    n = len(P)
    if n < 2:
        return None
    H, W = l.pixels.shape[:2]
    # ACROSS THE WIDTH, not one colour for the whole brush. Loading different
    # parts of the bristles with different colours -- pulling one edge of the
    # brush through blue and the other through white -- is the core of the
    # Bob Ross technique, and a single scalar reservoir cannot express it: it
    # averages the two into one flat mix before the stroke is even laid. Each
    # lane samples the canvas under ITS OWN part of the tuft, so a dip that
    # only touches one side loads only that side.
    tang = np.zeros_like(P)
    tang[1:-1] = P[2:] - P[:-2]
    tang[0] = P[1] - P[0]
    tang[-1] = P[-1] - P[-2]
    tl = np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-6)
    perp = np.stack([-tang[:, 1] / tl, tang[:, 0] / tl], 1)
    # The lanes sample where each part of the TUFT sits, so they must stay
    # inside the mark the brush actually makes. At +/-0.78 radius the outer
    # lanes sampled past the edge of a narrow pile, missed it, and dragged
    # the averaged charge and colour down -- a dip loaded the middle of the
    # brush and reported almost nothing.
    lanes = np.linspace(-0.55, 0.55, _BRUSH_LANES).astype(np.float32)
    lx = P[:, 0:1] + perp[:, 0:1] * lanes[None, :] * radius
    ly = P[:, 1:2] + perp[:, 1:2] * lanes[None, :] * radius
    xi = np.clip(lx.astype(np.int32), 0, W - 1)
    yi = np.clip(ly.astype(np.int32), 0, H - 1)
    under = l.pixels[yi, xi]                       # (n, L, 4)
    ua = under[..., 3]
    hm = getattr(l, "height_map", None)
    raw = hm[yi, xi] if hm is not None else np.zeros((n, _BRUSH_LANES), np.float32)
    body = np.clip(raw / max(media_hold, 0.05), 0.0, 1.0)
    # A RESERVOIR is a pile, not just any wet paint. `body` saturates at 1,
    # so a single normal stroke and a six-stroke mound looked identical to it
    # and the brush merrily recharged off its own last mark -- charge went UP
    # while painting. Only height ABOVE what one loaded stroke leaves counts
    # as somewhere you can dip.
    thick = np.clip(raw - 2.2, 0.0, 6.0) * 0.5      # (n, L)
    step = np.concatenate([[0.0], np.hypot(*(P[1:] - P[:-1]).T)]).astype(np.float32)
    L = _BRUSH_LANES
    cols = np.empty((n, L, 3), np.float32)
    chg = np.empty((n, L), np.float32)
    taken = np.zeros((n, L), np.float32)
    col0 = np.asarray(color, np.float32)
    cur = (col0.copy() if col0.shape == (L, 3)
           else np.repeat(_f32(color).reshape(1, 3), L, 0))
    c = (np.asarray(charge0, np.float32).copy()
         if np.ndim(charge0) else np.full(L, float(charge0), np.float32))
    for i in range(n):
        st = float(step[i])
        # lift: the emptier the brush, the more it takes
        act = thick[i] > 0.01
        if reload_rate > 0.0 and np.any(act):
            # Dragging through a PILE trades paint even when the brush is
            # full. Gating this on "has room" meant a freshly loaded brush
            # dipped into a second colour only got the slow trickle, so red
            # into white stayed red -- and dipping to mix is the whole point
            # of a palette. An emptier brush still takes more, but a full one
            # exchanges: what it gains in colour it gives up in load.
            g = (reload_rate * thick[i] * ua[i] * st
                 * (0.40 + 0.60 * (1.0 - c))) * act
            w = np.minimum(g / np.maximum(c, 0.12), 0.55)[:, None]
            cur = cur * (1.0 - w) + under[i, :, :3] * w
            # A full brush still LIFTS paint as it drags -- it displaces and
            # carries even when it cannot hold more. Capping the take strictly
            # by remaining room meant that once the brush filled (which is
            # fast now) it stopped scraping the mound at all, and paint
            # stopped being conserved half way through a dip.
            taken[i] = g * (0.30 + 0.70 * np.maximum(1.0 - c, 0.0))
            c = np.minimum(c + g, 1.0)
        elif pickup > 1e-3:
            k = np.clip(pickup * ua[i] * body[i] * 0.012 * max(st, 1.0),
                        0.0, 0.05)[:, None]
            cur = cur * (1.0 - k) + under[i, :, :3] * k
        # lay: what leaves the brush
        if spend_rate > 0.0:
            c = np.maximum(c - spend_rate * st, 0.0)
        cols[i] = cur
        chg[i] = c
    return cols, chg, taken


def _wet_mix(l, dense, color, media_hold, pickup, dense_arc_total):
    """Wet-on-wet: the brush PICKS UP what it crosses and drags it forward.

    Real paint mixes on the canvas, not in a colour picker. Drag blue across
    a wet red stroke and the blue goes purple *from the crossing onward* --
    the brush carries the red it lifted, and the further it travels the more
    it has spent. That asymmetry is the whole tell: a stroke that mixes
    symmetrically along its length looks like a gradient, not like paint.

    Modelled as a reservoir walked ALONG the path: at each step the brush
    trades a little of its load for what is under it. One 1-D recurrence over
    the path points -- a few hundred steps -- and the result is looked up per
    pixel by arc length, so the cost does not scale with brush size at all.
    Returns an (N,3) table of brush colour versus arc position.

    `pickup` is 0..1. Stiff media lift more (a loaded bristle drags pigment);
    a watery brush mostly deposits.
    """
    P = np.asarray(dense, np.float32)
    n = len(P)
    if n < 2 or pickup <= 1e-3:
        return None
    H, W = l.pixels.shape[:2]
    xi = np.clip(P[:, 0].astype(np.int32), 0, W - 1)
    yi = np.clip(P[:, 1].astype(np.int32), 0, H - 1)
    under = l.pixels[yi, xi]                       # what the path crosses
    ua = under[:, 3]
    # only WET paint lifts. Height is the paint body: bare canvas and thin
    # stain give nothing back, a fat ridge gives plenty.
    hm = getattr(l, "height_map", None)
    body = (np.clip(hm[yi, xi] / max(media_hold, 0.05), 0.0, 1.0)
            if hm is not None else np.zeros(n, np.float32))
    # PER STEP, and the path is sampled about a pixel apart -- so this is a
    # rate per pixel travelled, not per crossing. At the obvious-looking 0.55
    # a brush crossing one 34px bar compounded (1-0.55)^34 and came out the
    # far side pure red: one touch of wet paint wiped the load entirely.
    # 0.012 leaves roughly two thirds of the original colour after a single
    # crossing, and mud only after several -- which is how it goes.
    take = np.clip(pickup * ua * body * 0.012, 0.0, 0.05).astype(np.float32)
    res = np.empty((n, 3), np.float32)
    cur = _f32(color).reshape(3).copy()
    for i in range(n):
        k = float(take[i])
        if k > 1e-4:
            cur = cur * (1.0 - k) + under[i, :3] * k
        res[i] = cur
    return res


def _resolve_material(material):
    """A material request -> a full parameter dict, or None.

    Accepts a preset name or a dict of overrides (a dict may name a preset
    to start from via "preset"). Unknown names raise -- a silent fallback
    would paint the wrong stuff, and the error names the choices."""
    if not material:
        return None
    if isinstance(material, str):
        m = _MATERIALS.get(material)
        if m is None:
            raise ValueError("unknown material %r -- one of %s"
                             % (material, ", ".join(sorted(_MATERIALS))))
        return dict(m, name=material)
    if isinstance(material, dict):
        base = dict(_MATERIALS.get(str(material.get("preset", "")), {
            "rough": 0.5, "metal": 0.0, "grain": 0.0, "gscale": 2.0,
            "hold": 0.8, "flow": 0.12, "iters": 4,
            "gloss": 0.4, "shin": 20.0, "color": None}))
        for k in ("rough", "metal", "grain", "gscale", "hold", "flow",
                  "iters", "gloss", "shin"):
            if k in material and material[k] is not None:
                base[k] = float(material[k])
        base["rough"] = float(np.clip(base["rough"], 0.0, 1.0))
        base["metal"] = float(np.clip(base["metal"], 0.0, 1.0))
        base["name"] = str(material.get("preset", "custom"))
        return base
    raise ValueError("material must be a preset name or a dict")


def _material_grain(doc, l, gscale):
    """The material's micro-relief field: position-stable noise so a replay
    re-deposits the IDENTICAL texture (seeded by document + layer + scale,
    never by stroke order), centred on zero so grain roughens without
    inflating the mean height."""
    key = round(float(gscale), 2)
    cache = getattr(l, "_mat_grain", None)
    if cache is None:
        cache = l._mat_grain = {}
    g = cache.get(key)
    if g is None or g.shape != (doc.height, doc.width):
        seed = hash((doc.id, l.id, "matgrain", key)) & 0x7fffffff
        rng = np.random.default_rng(seed)
        n = rng.random((doc.height, doc.width)).astype(np.float32)
        g = _gauss_blur(n[..., None], max(key, 0.6))[..., 0]
        lo, hi = float(g.min()), float(g.max())
        g = (g - lo) / max(hi - lo, 1e-6) - 0.5
        cache[key] = g
    return g


def _relief_shade(px, height, gloss=0.3, shin=16.0, material=None,
                  slope=_RELIEF_SLOPE):
    """Light the paint surface: normals from the height field, lambert plus a
    specular term, applied to a COPY of the pixels -- the stored pigment stays
    flat and undamaged, the relief is a view. This is what makes built-up
    paint read as physical: ridges catch the key light on one side and shadow
    on the other, and oil gets its sheen.

    `material` is the layer's (H,W,3) [rough, metal, coverage] window, or
    None. Where coverage is 0 the maths reduces EXACTLY to the scalar
    gloss/shin path, so plain impasto shades byte-for-byte as before; where
    material was painted, the highlight tightens with 1-rough, metals tint
    their gleam with the pigment underneath (gold's shine is gold, chrome's
    is silver) and give up their diffuse -- the two behaviours that make a
    metal read as metal rather than as glossy plastic."""
    # surface tension: real paint rounds itself off, so the normals come
    # from a slightly smoothed surface. Raw mask edges made near-vertical
    # normals, and every stroke boundary inside a blob shaded as a hard dark
    # vein -- the "glitchy" creases in the user's screenshot.
    # 0.6, not the old 1.4: surface tension should round a ridge, not erase
    # the bristle comb and canvas tooth that the deposit model works to lay
    # down. The hard "veins" that 1.4 was hiding came from raw mask edges;
    # the deposit now models a real edge profile (rim, then falloff), so the
    # veins have no reason to exist and a light touch is enough.
    smooth = _gauss_small(height.astype(np.float32), 0.6)
    gy, gx = np.gradient(smooth)
    gy = gy * slope
    gx = gx * slope
    nz = np.ones_like(height, np.float32)
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nzn = -gx * inv, -gy * inv, nz * inv
    lam = np.clip(nx * _LIGHT[0] + ny * _LIGHT[1] + nzn * _LIGHT[2], 0.0, 1.0)
    # Blinn half-vector with the view straight on (0,0,1)
    hx, hy, hz = _LIGHT[0], _LIGHT[1], _LIGHT[2] + 1.0
    hn = 1.0 / np.sqrt(hx * hx + hy * hy + hz * hz)
    sdot = np.clip(nx * hx * hn + ny * hy * hn + nzn * hz * hn, 0.0, 1.0)
    out = px.copy()
    cov = None
    if material is not None:
        cov = np.clip(material[..., 2], 0.0, 1.0)
        if not (cov > 1e-3).any():
            cov = None
    if cov is None:
        spec = sdot ** shin
        lit = 0.72 + 0.42 * lam
        out[..., :3] = np.clip(
            out[..., :3] * lit[..., None]
            + (gloss * spec * (height > 0.02))[..., None], 0.0, 1.0)
        return out
    rough = np.clip(material[..., 0], 0.0, 1.0)
    metal = np.clip(material[..., 1], 0.0, 1.0)
    # highlight WIDTH: rough 0 -> exponent ~114 (chrome pin-dot), rough 1 ->
    # 4 (broad dull sheen); blended toward the layer's scalar exponent by
    # coverage so a material edge fades into plain impasto, never snaps
    # exponent cap 60, not the Blinn-classic ~110: surface tension smooths
    # every ridge, so past ~60 no canvas pixel ever aligns and the highlight
    # exists only in the maths (chrome measured duller than chalk). At 60
    # the pin-dot fires on real ridge flanks and still reads tight.
    shin_px = float(shin) * (1.0 - cov) + (4.0 + (1.0 - rough) ** 2
                                           * 56.0) * cov
    spec = np.power(sdot, shin_px)
    # highlight STRENGTH: dielectrics keep a Fresnel-ish floor (0.06) that
    # grows as they polish; metals are strong but a rough metal scatters
    amt_d = 0.06 + 0.5 * (1.0 - rough) ** 2
    amt_m = 0.9 * (1.0 - 0.55 * rough)
    amt = (float(gloss) * (1.0 - cov)
           + (amt_d * (1.0 - metal) + amt_m * metal) * cov)
    # a SECOND, broad lobe for polished surfaces: with one key light and
    # surface-tension-smoothed ridges, almost no pixel aligns with a ^110
    # half-vector, so chrome shaded as dark slate -- the pin-dot existed
    # in the maths and never on the canvas. Real polished surfaces pick up
    # the whole environment; this wide low lobe is that pickup, scaled by
    # polish and boosted for metals, and it is what makes a chrome flank
    # actually gleam. Matte materials get none ((1-rough)^2 -> 0).
    spec2 = sdot ** 4
    amt2 = cov * (1.0 - rough) ** 2 * (0.1 + 0.35 * metal)
    # metals gleam in their own colour; dielectric highlights stay white
    tint = metal[..., None] * 0.85 * cov[..., None]
    spec_rgb = (1.0 - tint) + out[..., :3] * tint
    # metals surrender diffuse -- kept off a hard zero so an unlit gold
    # stroke reads as dark metal, not a hole in the picture
    lit = 0.72 + 0.42 * lam * (1.0 - 0.72 * metal * cov)
    lit = lit - 0.34 * metal * cov * (1.0 - lam)   # away from light: darker
    vis = np.maximum((height > 0.02).astype(np.float32), cov > 0.05)
    out[..., :3] = np.clip(out[..., :3] * lit[..., None]
                           + spec_rgb * ((amt * spec + amt2 * spec2)
                                         * vis)[..., None],
                           0.0, 1.0)
    return out


def _vol_sample(img, ox, oy):
    """Sample an RGBA canvas at displaced coordinates (screen-space
    refraction), BILINEAR with edge clamping: nearest-neighbour banded
    visibly on smooth ripples -- refraction through calm water showed
    staircases where the bend crossed each integer."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(xx + ox, 0, w - 1.001)
    sy = np.clip(yy + oy, 0, h - 1.001)
    x0 = sx.astype(np.int32)
    y0 = sy.astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


_MEDIA_KINDS = ("inkwater", "smoke", "fire")


def _media_state(doc, l):
    """Per-layer persistent fluid: dye (RGB) + density + velocity on a
    capped grid. The state SURVIVES between strokes, so a second stroke
    stirs the water the first one set moving."""
    # SIMULATION DETAIL is the artist's call, because it is a straight
    # trade of fidelity against frame time. Measured at 1000x750 with
    # the windowed render in place:
    #   coarse  128 cells  ~11 ms/frame   7.8 canvas px per fluid cell
    #   normal  192 cells  ~18 ms/frame   5.2 px per cell   (default)
    #   fine    320 cells  ~45 ms/frame   3.1 px per cell
    # Fine roughly halves the cell size -- visibly finer filaments --
    # for about 2.5x the cost. Nobody is forced to pay it.
    div, cap = {"coarse": (6, 128), "fine": (3, 320)}.get(
        str(getattr(l, "media_res", "normal")), (4, 192))
    gw = min(cap, max(48, doc.width // div))
    gh = max(36, int(round(gw * doc.height / max(doc.width, 1))))
    st = getattr(l, "_media", None)
    if st is None:
        st = {"den": np.zeros((gh, gw), np.float32),
              "dye": np.zeros((gh, gw, 3), np.float32),
              "vx": np.zeros((gh, gw), np.float32),
              "vy": np.zeros((gh, gw), np.float32)}
        l._media = st
    elif st["den"].shape != (gh, gw):
        # RESAMPLE, never wipe: changing the detail (or resizing the
        # document) used to silently zero the medium -- an artist's
        # swirling ink vanished on a settings change. Carry the state
        # across; velocity scales with the new cell size.
        oh, ow = st["den"].shape
        sx, sy = gw / float(ow), gh / float(oh)
        st = {"den": _resize(st["den"][..., None], gh, gw)[..., 0],
              "dye": _resize(st["dye"], gh, gw),
              "vx": _resize(st["vx"][..., None], gh, gw)[..., 0] * sx,
              "vy": _resize(st["vy"][..., None], gh, gw)[..., 0] * sy}
        l._media = st
        l._media_frame_cache = {}      # cached states are the old shape
        l._media_win = None
    return st, gh, gw


def _media_inject(doc, l, x0, y0, x1, y1, strength=1.0):
    """The freshly painted window becomes DYE: whatever alpha the stroke
    left in the region is added to the medium's density, its colour to the
    dye field, and a small velocity kick where paint landed (a brush moving
    through water pushes it)."""
    st, gh, gw = _media_state(doc, l)
    h, w = doc.height, doc.width
    win = l.pixels[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if win.size == 0:
        return
    full_a = np.zeros((h, w), np.float32)
    full_c = np.zeros((h, w, 3), np.float32)
    full_a[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = win[..., 3]
    full_c[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = win[..., :3]
    ga = _resize(full_a[..., None], gh, gw)[..., 0]
    gc = _resize(full_c, gh, gw)
    ga = ga * float(strength)
    st["den"] = np.clip(st["den"] + ga, 0.0, 2.0)
    st["dye"] = st["dye"] + gc * ga[..., None]
    # STABLE seed: Python's hash() of a str is salted PER PROCESS, so
    # this turbulence differed between app launches for the same
    # document and the same actions -- media replays were only
    # reproducible within one run. crc32 is the same everywhere.
    import zlib as _zlib
    # seed by a PER-LAYER INJECTION COUNTER, not the global revision:
    # _MUT_REV differs between otherwise-identical documents (and
    # between replays of the same document), so the same stroke got
    # different turbulence -- an invisible reproducibility leak that
    # also invalidated a batch-vs-singles purity experiment.
    n = int(getattr(l, "_media_inject_n", 0))
    l._media_inject_n = n + 1
    rng = np.random.default_rng(
        (_zlib.crc32(l.id.encode()) ^ (n * 2654435761)) & 0x7fffffff)
    kick = (rng.random((gh, gw)) - 0.5).astype(np.float32) * 2.0
    st["vx"] += kick * ga * 1.5
    st["vy"] += np.abs(kick) * ga * -0.5
    # fresh dye changes all FUTURE evolution: drop cached states past
    # the current frame and baseline the cache here, so a later rewind
    # lands on the world as it was when this stroke went in
    # T6: a stroke does NOT reset the medium's clock. Injection used to
    # write marks=[(now, 0.0)] and wipe the cache, so every stroke threw
    # away the accumulated time and every earlier state -- scrubbing back
    # after a new stroke could not restore what the ink had been doing,
    # and the medium's elapsed time silently jumped to zero. Keep the
    # accumulated fractional steps, invalidate only the FUTURE, and
    # rebaseline the cache at where we actually are.
    # a stroke writes CRISP pixels directly, outside the render's dirty
    # window bookkeeping. Drop the window so the next render clears the
    # whole layer once and re-establishes the invariant 'pixels = the
    # slab render, everything else zero'. Without this, whether a stale
    # painted region survived depended on the scrub PATH -- the same
    # frame reached two ways differed.
    l._media_win = None
    now = float(getattr(doc, "frame", 0.0))
    marks = getattr(l, "_media_marks", None)
    if marks:
        i = len(marks) - 1
        while i > 0 and marks[i][0] > now + 1e-9:
            i -= 1
        f0, s0 = marks[i]
        rate = float(getattr(l, "media_rate", 1.0))
        tmul = float(getattr(doc, "time_scale", 1.0))
        here = max(0.0, s0 + max(0.0, now - f0) * rate * 0.8 * tmul)
    else:
        here = 0.0
    l._media_marks = [(now, here)]
    cache = getattr(l, "_media_frame_cache", None) or {}
    step = int(round(here))
    # the new dye invalidates every FUTURE state; the past is gone too
    # (it was a different world), but this step is the truth right now
    l._media_frame_cache = {}
    _media_cache_put(doc, l, step)


def _layer_fields_grid(doc, l, gh, gw):
    """Sum a layer's FIELD OBJECTS into force components on the media
    grid. Returns (fx, fy) arrays or None. Point fields push away from
    (or toward, negative strength) their centre inside their radius;
    direct pushes uniformly along its angle; vortex swirls about the
    centre. Falloff is a smooth quadratic to the radius edge."""
    fs = [f for f in getattr(doc, "fields", [])
          if f.get("layer") == l.id]
    if not fs:
        return None
    h, w = doc.height, doc.width
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    X = (xx + 0.5) * (w / gw)
    Y = (yy + 0.5) * (h / gh)
    fx = np.zeros((gh, gw), np.float32)
    fy = np.zeros((gh, gw), np.float32)
    for f in fs:
        dx = X - float(f["x"])
        dy = Y - float(f["y"])
        r = np.sqrt(dx * dx + dy * dy)
        R = max(float(f["radius"]), 8.0)
        fall = np.clip(1.0 - (r / R) ** 2, 0.0, 1.0)
        # sign convention, second visit: the original negation was
        # tuned under the PERIODIC solve; with the reflective wall
        # band feeding the global pressure projection, the effective
        # response flipped back. Empirical ground truth (pinned in the
        # fields test): positive strength pushes along the angle.
        # 30 -> 105 for leCore 0.2.9's WALL boundary. This is physics,
        # not tuning: in a SEALED box a uniform body force is balanced
        # by the pressure gradient, so the same dial that moved a blob
        # 120 -> 201 on the periodic solver moved it only 120 -> 132.
        # The field now circulates the medium rather than sliding it
        # off-canvas, which is correct -- but the dial has to be
        # rescaled so a given strength still means what it meant to the
        # artist. Measured: x3.5 restores the old response.
        s = float(f["strength"]) * 105.0
        if f["kind"] == "direct":
            a = np.deg2rad(float(f.get("angle", 0.0)))
            fx += np.cos(a) * s * fall
            fy += np.sin(a) * s * fall
        elif f["kind"] == "vortex":
            nr = np.maximum(r, 1e-3)
            fx += (-dy / nr) * s * fall
            fy += (dx / nr) * s * fall
        else:  # point
            nr = np.maximum(r, 1e-3)
            fx += (dx / nr) * s * fall
            fy += (dy / nr) * s * fall
    return fx, fy


def _apply_point_fields(doc, l, st, gh, gw, steps):
    """Point (radial) fields CANNOT act through the solver force path:
    a radial push is pure divergence and the incompressible pressure
    projection cancels it exactly (watched it do nothing while vortex
    -- pure curl -- and direct -- uniform -- worked fine). So point
    fields displace the density and dye directly: a small
    semi-Lagrangian warp along the radial map per step, outside the
    projection. strength>0 repels, <0 attracts."""
    fs = [f for f in getattr(doc, "fields", [])
          if f.get("layer") == l.id and f.get("kind") == "point"
          and abs(float(f.get("strength", 0.0))) > 1e-6]
    if not fs:
        return
    h, w = doc.height, doc.width
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    X = (xx + 0.5) * (w / gw)
    Y = (yy + 0.5) * (h / gh)
    ox = np.zeros((gh, gw), np.float32)
    oy = np.zeros((gh, gw), np.float32)
    for f in fs:
        dx = X - float(f["x"])
        dy = Y - float(f["y"])
        r = np.sqrt(dx * dx + dy * dy)
        R = max(float(f["radius"]), 8.0)
        fall = np.clip(1.0 - (r / R) ** 2, 0.0, 1.0)
        nr = np.maximum(r, 1e-3)
        # per-iteration displacement, capped below one cell for
        # bilinear stability (1.4-cell hops still bled mass)
        # 0.55 -> 0.85 per iteration (still under one cell, the
        # bilinear-stability limit). leCore 0.2.9's wall boundary and the
        # reflect-padded blur changed how fast the medium spreads on its
        # own, and at 0.55 an ATTRACT field no longer out-pulled that
        # spread -- the blob ended 3.6%% wider than the control instead
        # of tighter.
        k = float(np.clip(f["strength"] * 0.85, -0.9, 0.9))
        ox += (dx / nr) * fall * k * (gw / w)
        oy += (dy / nr) * fall * k * (gh / h)
    # ITERATIVE small steps: one huge displacement made attract sample
    # from beyond the field edge (empty cells) and the dye vanished --
    # advect gently, several times
    iters = max(1, min(int(steps), 12))
    # sample upstream: value at p comes from p - offset
    sx = np.clip(xx - ox, 0, gw - 1.001)
    sy = np.clip(yy - oy, 0, gh - 1.001)
    x0 = sx.astype(np.int32); y0 = sy.astype(np.int32)
    fx_ = sx - x0; fy_ = sy - y0
    x1 = np.minimum(x0 + 1, gw - 1); y1 = np.minimum(y0 + 1, gh - 1)

    def samp(A):
        if A.ndim == 2:
            return (A[y0, x0] * (1 - fx_) * (1 - fy_)
                    + A[y0, x1] * fx_ * (1 - fy_)
                    + A[y1, x0] * (1 - fx_) * fy_
                    + A[y1, x1] * fx_ * fy_)
        return np.stack([samp(A[..., c]) for c in range(A.shape[-1])], -1)

    for _ in range(iters):
        # MASS CONSERVATION PER ITERATION: inward sampling compresses
        # density into few texels and bilinear + clipping quietly loses
        # it (a strong attract once left ZERO alpha; a single post-loop
        # rescale capped at 4x could not recover a 22x loss)
        s0 = float(st["den"].sum())
        st["den"] = samp(st["den"]).astype(np.float32)
        st["dye"] = samp(st["dye"]).astype(np.float32)
        s1 = float(st["den"].sum())
        if s1 > 1e-6 and s0 > 1e-6:
            k2 = min(s0 / s1, 2.0)
            st["den"] = np.clip(st["den"] * k2, 0.0, 2.0)
            st["dye"] = st["dye"] * k2


def _media_slab_step(doc, l, steps):
    """Advance a dynamic media slab: leCore's fluid solve for the density
    and velocity, advect_field carrying the RGB dye along the same flow.
    THICKNESS is the physics dial -- a deep dish disperses further and
    billows harder than a shallow one. Each medium has its own forces:
      inkwater  gentle swirl, dense ink SINKS a little, dye persists
      smoke     buoyant, diffuse, thins as it climbs
      fire      strong buoyancy + turbulence, dissipates fast (burns out)
    Renders the state back into layer.pixels (heat ramp for fire, grey
    ramp for smoke, the ink's own colour for inkwater)."""
    st, gh, gw = _media_state(doc, l)
    kind = getattr(l, "vol_kind", "inkwater")
    T = max(float(getattr(l, "thickness", 8.0)), 1.0)
    m = mind()
    tscale = min(T / 8.0, 3.0)
    curl = _curl_noise(32, 3, hash(l.id) & 0xffff)
    gyy, gxx = np.mgrid[0:gh, 0:gw]
    cgx = np.clip((gxx / max(gw, 1) * 31).astype(int), 0, 31)
    cgy = np.clip((gyy / max(gh, 1) * 31).astype(int), 0, 31)
    # force magnitudes live in the fluid solver's working range (the Fluid
    # node billows convincingly at swirl ~12): the first draft used ~2 and
    # the smoke climbed three pixels in thirty steps
    if kind == "fire":
        buoy, swirl, visc, diss, dyediss = -42.0 * tscale, 20.0, 0.0, 0.975, 0.96
    elif kind == "smoke":
        buoy, swirl, visc, diss, dyediss = -22.0 * tscale, 11.0, 0.004, 0.988, 0.988
    else:
        buoy, swirl, visc, diss, dyediss = 4.5 * tscale, 7.0, 0.008, 0.999, 0.999
    # the slab's own GEOMETRY tips the medium: ink in a tilted dish
    # drifts downhill, smoke in a domed chamber slides off the crown --
    # the base-plane gradient joins the forces, same physics as run_paint
    bp = _layer_base_plane(l, doc.height, doc.width)
    if float(np.ptp(bp)) > 1e-6:
        bgy, bgx = np.gradient(_gauss_blur(bp[..., None], 2.0)[..., 0])
        geo_fx = -_resize(bgx[..., None], gh, gw)[..., 0] * 30.0
        geo_fy = -_resize(bgy[..., None], gh, gw)[..., 0] * 30.0
    else:
        geo_fx = geo_fy = None
    lfld = _strokefx_field(doc, getattr(l, "field", ""),
                           doc.height, doc.width)
    lmode = getattr(l, "field_mode", "attract")
    lstr = float(getattr(l, "field_strength", 120.0)) * 0.25
    if lfld is not None:
        # the field builder works on its own coarse grid: bring every
        # channel to THIS medium's grid before the solve loop
        F0r = _resize(lfld[0][..., None], gh, gw)[..., 0]
        fgxr = _resize(lfld[1][..., None], gh, gw)[..., 0]
        fgyr = _resize(lfld[2][..., None], gh, gw)[..., 0]
        lfld = (F0r, fgxr, fgyr, gw, gh)
    objf = _layer_fields_grid(doc, l, gh, gw)
    _apply_point_fields(doc, l, st, gh, gw, steps)
    # THE CANVAS EDGES ARE WALLS -- now enforced INSIDE the solver.
    # leCore 0.2.9 added boundary="wall" (our P3): a Neumann pressure
    # projection on a mirrored domain, so zero normal flow is a property
    # of the projection rather than a patch applied after it. That
    # replaced ~30 lines here that rewrote the outer 4 cells as a
    # mirrored band every iteration. Measured on the new solver: 100% of
    # the mass is kept when flow is driven into a wall, against 0% for
    # our band and 5% for the old solid-mask workaround.
    for _ in range(max(1, int(steps))):
        fx = curl[0][cgy, cgx] * swirl
        fy = curl[1][cgy, cgx] * swirl * 0.6 + buoy * st["den"]
        if objf is not None:
            # field OBJECTS parented to this layer join the solve
            fx = fx + objf[0]
            fy = fy + objf[1]
        if geo_fx is not None:
            fx = fx + geo_fx * st["den"]
            fy = fy + geo_fy * st["den"]
        if lfld is not None and lstr > 0:
            # the layer's own field herds its MEDIUM: ink drawn to a mask,
            # smoke fenced inside a selection, fire pulled along a stroke
            F0, fgx, fgy, _, _ = lfld
            # signs flipped with the reflective-wall regime: the wall
            # band feeding the global pressure projection reversed the
            # effective force response (see the object-field note)
            if lmode == "repel":
                fx = fx + fgx * lstr
                fy = fy + fgy * lstr
            elif lmode == "flow":
                fx = fx + fgy * lstr
                fy = fy + -fgx * lstr
            elif lmode == "contain":
                # contain goes through the DISPLACEMENT WARP, not the
                # force path: under the reflective-wall regime the
                # pressure projection ate the containment force from
                # either sign (measured 0.22 and 0.01 in-region vs
                # 0.41 free). Like the point fields, a direct
                # semi-Lagrangian nudge of den/dye toward the region
                # is immune to the projection.
                oa = np.clip(0.85 - F0, 0.0, 1.0)
                # 0.8 -> 1.3: with leCore 0.2.9's wall boundary the
                # medium no longer bleeds off the canvas, so a firmer
                # containment nudge stays inside the picture. At 0.8 the
                # in-region share was +0.089 over the free control --
                # right on the pin's 0.08 margin.
                k = min(lstr, 600.0) / 600.0 * 1.3
                # sampling at p-ox moves content ALONG +ox (see the
                # point-field warp): offset points INTO the region
                ox = fgx * oa * k
                oy = fgy * oa * k
                _yy, _xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
                sx_ = np.clip(_xx - ox, 0, gw - 1.001)
                sy_ = np.clip(_yy - oy, 0, gh - 1.001)
                x0_ = sx_.astype(np.int32)
                y0_ = sy_.astype(np.int32)
                fx_ = sx_ - x0_
                fy_ = sy_ - y0_
                x1_ = np.minimum(x0_ + 1, gw - 1)
                y1_ = np.minimum(y0_ + 1, gh - 1)

                def _sampc(A):
                    if A.ndim == 2:
                        return (A[y0_, x0_] * (1 - fx_) * (1 - fy_)
                                + A[y0_, x1_] * fx_ * (1 - fy_)
                                + A[y1_, x0_] * (1 - fx_) * fy_
                                + A[y1_, x1_] * fx_ * fy_)
                    return np.stack([_sampc(A[..., c])
                                     for c in range(A.shape[-1])], -1)
                st["den"] = _sampc(st["den"]).astype(np.float32)
                st["dye"] = _sampc(st["dye"]).astype(np.float32)
            else:
                fx = fx - fgx * lstr
                fy = fy - fgy * lstr
        try:
            vx, vy, den = _fluid_step_walled(m, st, visc, fx, fy)
        except Exception:
            vx = st["vx"] + fx * 0.06
            vy = st["vy"] + fy * 0.06
            den = st["den"]
        st["vx"], st["vy"] = (np.asarray(vx, np.float32),
                              np.asarray(vy, np.float32))
        st["den"] = np.clip(np.asarray(den, np.float32), 0.0, 2.0) * diss
        try:
            # one (H, W, 3) advect instead of three scalar calls (P2),
            # with WALL boundaries so the dye is contained like the
            # density. The mind() facade does not forward `boundary`, so
            # go to the field module directly when it is importable and
            # fall back to the facade otherwise -- without this the
            # velocity respected the walls but the COLOUR still wrapped
            # (measured: 73.8 units of ink reappearing on the far edge).
            st["dye"] = np.asarray(
                _advect_walled(m, st["dye"], st["vx"], st["vy"], 0.06,
                               roi=_media_active_roi(st, st["vx"],
                                                     st["vy"], 0.06)),
                np.float32)
        except Exception:
            pass
        st["dye"] *= dyediss
        if kind == "inkwater":
            # molecular diffusion: ink TENDRILS soften and creep even where
            # the flow is still
            # _gauss_blur is an FFT, i.e. CIRCULAR convolution: with the
            # solver's walls now doing their job, this was the last path
            # that still wrapped, bleeding a faint 73-unit ghost of the
            # ink onto the opposite edge. Reflect-pad by 3 sigma, blur,
            # crop -- a mirrored border is exactly the no-flux condition
            # the wall boundary enforces elsewhere.
            st["den"] = _gauss_blur_reflect(st["den"], 0.55)
            st["dye"] = _gauss_blur_reflect(st["dye"], 0.55)
    _media_render(doc, l, st, kind)


_SHAPE_READOUT = [None]
_SHAPE_KINDS = ("circle", "rectangle", "line")


def _shape_training_set(seed=0, per_class=200):
    """Synthetic strokes to fit the shape readout. We have no corpus of
    labelled human strokes and will not pretend otherwise -- so the
    training set is GENERATED, deterministically, with the variation a
    hand actually produces: partial sweeps, either direction, ellipses
    and oblongs, arbitrary rotation, and jitter from 1% to 6%."""
    rng = np.random.default_rng(seed)

    def norm(p):
        p = np.asarray(p, float)
        p = p - p.mean(0)
        s = np.abs(p).max()
        return p / (s if s > 1e-9 else 1.0)

    X, y = [], []
    for kind in (0, 1, 2):
        for _ in range(per_class):
            n = int(rng.integers(28, 70))
            t = np.linspace(0, 1, n)
            if kind == 0:
                sweep = rng.uniform(0.82, 1.0) * 2 * np.pi * rng.choice([1, -1])
                b = rng.uniform(0.6, 1.0)
                ph = rng.uniform(0, 2 * np.pi)
                p = np.stack([np.cos(t * sweep + ph),
                              b * np.sin(t * sweep + ph)], 1)
            elif kind == 1:
                h = rng.uniform(0.4, 1.0)
                pts = [(-1, -h), (1, -h), (1, h), (-1, h), (-1, -h)]
                seg = np.array_split(t, 4)
                out = []
                for i in range(4):
                    a = np.array(pts[i], float)
                    b2 = np.array(pts[i + 1], float)
                    uu = np.linspace(0, 1, max(len(seg[i]), 2))
                    out.append(a[None] + (b2 - a)[None] * uu[:, None])
                p = np.vstack(out)
            else:
                ang = rng.uniform(0, np.pi)
                p = np.stack([t * np.cos(ang), t * np.sin(ang)], 1)
            X.append(norm(p + rng.normal(0, rng.uniform(0.01, 0.06),
                                         p.shape)))
            y.append(kind)
    return X, y


def _shape_geometry(p):
    """The cheap, independent opinion: is the loop round or cornered?

    CIRCULARITY (4*pi*area / perimeter^2) -- 1.0 for a circle, 0.785
    for a square -- beat the first attempt, radial standard deviation,
    which could not tell a square from an ellipse because both wobble
    about the same amount (measured: square std ~0.12, ellipse up to
    0.14, and rectangles were being abstained on). The split sits at
    0.74: measured medians are 0.84 for hand-drawn circles and 0.62 for
    rectangles, with the 10-90% bands (0.63-0.96 and 0.51-0.72) barely
    touching."""
    if np.linalg.norm(p[0] - p[-1]) >= 0.45:
        return 2
    per = float(np.linalg.norm(
        np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())
    x, y = p[:, 0], p[:, 1]
    area = abs(float(np.dot(x, np.roll(y, -1))
                     - np.dot(y, np.roll(x, -1))) / 2.0)
    q = 4.0 * np.pi * area / max(per * per, 1e-9)
    return 0 if q > 0.74 else 1


def recognise_shape(points):
    """What shape was that stroke? Returns {shape, confident, why} --
    or shape None when we should keep quiet.

    TWO OPINIONS, and we only speak when they agree. leCore 0.2.9's
    HRNN TrajectoryReadout classifies the stroke (measured 94% on
    held-out synthetic strokes, against 72% for geometry alone -- the
    signature's chirality and arrival-time features are doing real
    work). A plain radial-wobble test gives a second, independent
    opinion. Measured on 240 held-out strokes: they agree on 71% of
    strokes and are right 99% of the time when they do.

    So the contract is: agree -> offer, disagree -> say nothing. An
    unasked-for shape replacement that guesses wrong is worse than no
    feature at all, and 29% silence is the price of the other 99%."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[0] < 12:
        return {"shape": None, "confident": False,
                "why": "too few points to judge"}
    p = p - p.mean(0)
    s = float(np.abs(p).max())
    if s < 1e-9:
        return {"shape": None, "confident": False, "why": "no extent"}
    p = p / s
    if _SHAPE_READOUT[0] is None:
        try:
            from holographic.agents_and_reasoning.holographic_hrnn \
                import TrajectoryReadout
            X, y = _shape_training_set()
            tr = TrajectoryReadout(seed=0)
            tr.fit(X, y)
            _SHAPE_READOUT[0] = tr
        except Exception as e:
            _SHAPE_READOUT[0] = False
            return {"shape": None, "confident": False,
                    "why": "shape readout unavailable: %s" % e}
    if _SHAPE_READOUT[0] is False:
        return {"shape": None, "confident": False,
                "why": "shape readout unavailable"}
    try:
        h = int(np.asarray(_SHAPE_READOUT[0].classify([p])).ravel()[0])
    except Exception as e:
        return {"shape": None, "confident": False, "why": str(e)}
    g = _shape_geometry(p)
    if h != g:
        return {"shape": None, "confident": False,
                "why": "the trajectory readout says %s and the geometry "
                       "says %s -- too close to call"
                       % (_SHAPE_KINDS[h], _SHAPE_KINDS[g])}
    return {"shape": _SHAPE_KINDS[h], "confident": True,
            "why": "the trajectory readout and the geometry both say "
                   "%s" % _SHAPE_KINDS[h]}


def cook_until_settled(doc, layer=None, block=8, max_steps=320):
    """Cook a medium until it SETTLES, instead of guessing a step count.

    Cook(+24/+96) makes the artist estimate how long a simulation needs
    to look right, which is a question the simulation can answer itself.
    leCore 0.2.9's HRNN ships regime detection; run it on the medium's
    own change-per-step signal and stop when that signal has entered a
    final, low, stable regime.

    Returns {cooked, steps, settled, why} -- `settled` False with a
    reason when it hit the cap instead, because "we stopped because you
    told us to stop" and "we stopped because it stopped moving" are
    different facts and an artist deserves to know which one happened.
    """
    m = mind()
    out = []
    for l in doc.layers:
        if getattr(l, "vol_kind", "none") not in _MEDIA_KINDS:
            continue
        if layer is not None and l.id != layer:
            continue
        if getattr(l, "_media", None) is None:
            continue
        deltas = []
        done = 0
        settled = False
        why = "reached the step cap without settling"
        prev = np.array(l._media["den"], copy=True)
        while done < max_steps:
            _media_slab_step(doc, l, block)
            done += block
            cur = l._media["den"]
            deltas.append(float(np.abs(cur - prev).mean()))
            prev = np.array(cur, copy=True)
            if len(deltas) < 6:
                continue
            try:
                segs = m.detect_regimes(np.asarray(deltas, np.float64),
                                        min_seg=3)["segments"]
            except Exception:
                break                      # no regime faculty: cap rules
            if len(segs) < 2:
                continue
            first, last = segs[0]["mean"], segs[-1]["mean"]
            # settled = the change per step has entered a final regime an
            # order quieter than the opening one, and has STAYED there
            # for more than one block (a single quiet block is noise)
            if (last < first * 0.15
                    and segs[-1]["length"] >= 2
                    and segs[-1]["stop"] >= len(deltas)):
                settled = True
                why = ("change per step fell to %.0f%% of its opening "
                       "rate and held" % (100.0 * last / max(first, 1e-12)))
                break
        now = float(getattr(doc, "frame", 0.0))
        l._media_marks = [(now, 0.0)]
        l._media_frame_cache = {}
        l._media_win = None
        _media_cache_put(doc, l, 0)
        out.append({"layer": l.id, "steps": done, "settled": settled,
                    "why": why})
    _MUT_REV[0] += 1
    return {"cooked": len(out), "layers": out}


def _cook_state(l):
    """A copy of a medium's slab state, for the cook baseline."""
    st = getattr(l, "_media", None)
    if st is None:
        return None
    return {k: np.array(st[k], copy=True) for k in st
            if isinstance(st.get(k), np.ndarray)}


def cook_media(doc, steps=24, layer=None):
    """Advance (or REWIND) a medium without moving the playhead.

    `steps` may be negative. Cooking used to be one-way -- it advanced
    the medium, rebaselined the marks, and dropped the cache, so there
    was no route back and (measured) undo did not restore it either.
    Now the first cook of a layer stores the UNCOOKED state, and the
    running total is tracked; a negative cook restores that baseline
    and re-advances the remainder, which is exact rather than an
    attempt to run the solver backwards. Cook -8 after +24 gives byte
    for byte what +16 alone would have given.

    Returns {cooked, layers:[{layer, steps, total}]}."""
    n = []
    for l in doc.layers:
        if getattr(l, "vol_kind", "none") not in _MEDIA_KINDS:
            continue
        if layer is not None and l.id != layer:
            continue
        if getattr(l, "_media", None) is None:
            continue          # nothing injected yet: nothing to cook
        if getattr(l, "_cook_base", None) is None:
            l._cook_base = _cook_state(l)
            l._cook_total = 0
        total = int(getattr(l, "_cook_total", 0))
        want = max(0, total + int(steps))
        if want == total:
            n.append({"layer": l.id, "steps": 0, "total": total})
            continue
        if want < total:
            # REWIND: restore the uncooked state and replay the
            # remainder. The solver has no reverse, but it is
            # deterministic, so replaying IS the reverse.
            st, gh, gw = _media_state(doc, l)
            for k, v in (l._cook_base or {}).items():
                if k in st:
                    st[k] = np.array(v, copy=True)
            if want > 0:
                _media_slab_step(doc, l, min(want, 2400))
            else:
                _media_render(doc, l, st)
        else:
            _media_slab_step(doc, l, min(want - total, 2400))
        l._cook_total = want
        now = float(getattr(doc, "frame", 0.0))
        l._media_marks = [(now, 0.0)]     # the cooked state IS the start
        l._media_frame_cache = {}
        l._media_win = None
        _media_cache_put(doc, l, 0)
        n.append({"layer": l.id, "steps": want - total, "total": want})
    _MUT_REV[0] += 1
    return {"cooked": len(n), "layers": n}


def _media_cache_put(doc, l, step):
    """Remember this layer's slab state keyed by ABSOLUTE STEP COUNT
    from the injection baseline. Keying by frames made scrub and jump
    disagree at rounding boundaries; keying by steps makes state a
    pure function of total_steps(t). Grids are coarse (<=96x96): a
    generous cache is a few MB."""
    st = getattr(l, "_media", None)
    if st is None:
        return
    cache = getattr(l, "_media_frame_cache", None)
    if cache is None:
        cache = l._media_frame_cache = {}
    s = int(step)
    cache[s] = {k: np.array(st[k], copy=True)
                for k in ("den", "dye", "vx", "vy") if k in st}
    l._media_cache_at = s
    # size the cache by MEMORY, not by an arbitrary count: a coarse
    # state is 288 KB and a fine one 1.8 MB, so a flat 48 entries meant
    # 14 MB or 86 MB depending on a setting the artist chose for a
    # different reason entirely. ~48 MB buys ~166 coarse / ~74 normal /
    # ~27 fine states -- enough that an ordinary 96-frame range replays
    # almost entirely from cache at normal detail.
    st0 = cache.get(s) or next(iter(cache.values()), None)
    per = sum(a.nbytes for a in st0.values()) if st0 else (1 << 20)
    CAP = int(np.clip((48 << 20) // max(per, 1), 24, 240))
    if len(cache) > CAP:
        # STRIDED retention. The old policy kept the HIGHEST steps, so
        # replaying a range evicted each freshly-computed early step
        # the instant it was stored: every frame re-solved from step 0
        # and replay was O(n^2) -- measured only 2.2x faster than the
        # first pass despite a full cache. Keep step 0, the newest few
        # (scrubbing is usually local), and an evenly spaced spread
        # over everything else, so ANY target is a few steps from a
        # cached state.
        steps = sorted(cache.keys())
        newest = steps[-12:]
        rest = [s for s in steps[:-12] if s != 0]
        room = CAP - len(newest) - 1
        if len(rest) > room and room > 0:
            idx = np.linspace(0, len(rest) - 1, room).round()
            rest = [rest[int(i)] for i in sorted(set(idx.tolist()))]
        keep = set(rest) | set(newest) | {0}
        for k in [k for k in cache.keys() if k not in keep]:
            del cache[k]


def _media_restore_to_step(doc, l, step):
    """Restore the newest cached slab state at or before `step` and
    re-render pixels; l._media_cache_at records where we are (the
    caller advances any remainder)."""
    cache = getattr(l, "_media_frame_cache", None) or {}
    eligible = [s for s in cache.keys() if s <= int(step)]
    if not eligible:
        l._media_cache_at = None
        return False
    s = max(eligible)
    st, gh, gw = _media_state(doc, l)
    for k, v in cache[s].items():
        st[k] = np.array(v, copy=True)
    # bump BEFORE rendering: the render patches the composite cache and
    # marks it current, so any bump AFTER it left the cache one
    # revision behind -- and the very next patch attempt was rejected
    # as stale, which is why playback still paid a full re-composite
    # every frame despite the patching path existing.
    _MUT_REV[0] += 1
    l._media_cache_at = s
    _media_render(doc, l, st)
    return True


def _media_active_roi(st, vx, vy, dt, margin_extra=2):
    """The window the dye can possibly occupy after this step: the cells
    that hold anything now, grown by how far the flow can carry them
    (max|v|*dt) plus a cell for the bilinear tap. Outside it the field is
    zero and stays zero, so advecting only this window is EXACT, not an
    approximation -- leCore 0.2.9's roi= (our P4) makes it expressible."""
    gh, gw = st["den"].shape
    occ = (st["den"] > 1e-4) | (np.abs(st["dye"]).max(-1) > 1e-4)
    if not occ.any():
        return None
    ys, xs = np.where(occ)
    reach = int(np.ceil(max(float(np.abs(vx).max()),
                            float(np.abs(vy).max())) * abs(dt))) \
        + int(margin_extra)
    y0 = max(0, int(ys.min()) - reach); y1 = min(gh, int(ys.max()) + reach + 1)
    x0 = max(0, int(xs.min()) - reach); x1 = min(gw, int(xs.max()) + reach + 1)
    if (y1 - y0) * (x1 - x0) > 0.6 * gh * gw:
        return None                      # most of the grid: no saving
    return (y0, y1, x0, x1)


_MEDIA_WALL_SOLVE = None


def _fluid_step_walled(m, st, visc, fx, fy):
    """fluid_step with WALL boundaries, degrading in STEPS rather than
    all at once. boundary="wall" needs leCore >= 0.2.9; on an older core
    that is a TypeError, and the single catch-all around this call
    dropped straight to a forward-Euler nudge with no advection and no
    projection -- the fluid solver would vanish silently rather than
    merely lose its walls. Try walls, fall back to the periodic solve
    (still a real solve), and only then let the caller's guard take
    over. _MEDIA_WALL_SOLVE records which one we got, so 'why does my
    ink wrap?' has an answer in the state instead of a shrug."""
    global _MEDIA_WALL_SOLVE
    if _MEDIA_WALL_SOLVE is not False:
        try:
            out = m.fluid_step(st["vx"], st["vy"], st["den"], dt=0.06,
                               viscosity=visc, fx=fx, fy=fy,
                               boundary="wall")
            _MEDIA_WALL_SOLVE = True
            return out
        except TypeError:
            _MEDIA_WALL_SOLVE = False      # old core: no boundary arg
    return m.fluid_step(st["vx"], st["vy"], st["den"], dt=0.06,
                        viscosity=visc, fx=fx, fy=fy)


def _advect_walled(m, field, vx, vy, dt, roi=None):
    """Advect with WALL boundaries, multi-channel in one call.

    leCore 0.2.9 grew boundary="wall" and (H, W, C) support on advect,
    but the mind() facade's advect_field still has the old
    (field, vx, vy, dt) signature, so the new arguments are only
    reachable on the module. Try that, and fall back to the facade
    (wrap) if the module moves -- a contained medium is better than a
    crash, and _MEDIA_WALL_OK records which path we got."""
    global _MEDIA_WALL_OK
    try:
        from holographic.misc import holographic_fields as _HF
        out = _HF.advect(field, vx, vy, dt, boundary="wall",
                         roi=roi) if roi is not None else \
            _HF.advect(field, vx, vy, dt, boundary="wall")
        _MEDIA_WALL_OK = True
        return out
    except Exception:
        _MEDIA_WALL_OK = False
        return m.advect_field(field, vx, vy, dt=dt)


_MEDIA_WALL_OK = None


def _media_render(doc, l, st, kind=None):
    """Slab state -> layer.pixels. Split out of the step so a CACHED
    state can be restored and re-rendered without advancing time."""
    if kind is None:
        kind = getattr(l, "vol_kind", "inkwater")
    # ONE upsample, not two: density and dye share the same sampling
    # grid, so stacking them into a single 4-channel gather halves the
    # per-frame allocation and indexing work (this call is the single
    # largest cost in a playback frame at canvas resolution)
    src = np.concatenate([st["den"][..., None], st["dye"]], axis=-1)
    H, W = doc.height, doc.width
    gh, gw = st["den"].shape
    # DIRTY WINDOW: the dye occupies a fraction of the canvas for most
    # of a simulation, so upsample only the region it reaches (plus a
    # 2-cell margin). _resize_window is byte-identical to slicing the
    # full resize, so this is pure speed with no seam risk. Pixels
    # outside are cleared once, using the window written last frame.
    occ = st["den"] > 1e-4
    if occ.any():
        ys, xs = np.where(occ)
        gy0 = max(0, int(ys.min()) - 2); gy1 = min(gh, int(ys.max()) + 3)
        gx0 = max(0, int(xs.min()) - 2); gx1 = min(gw, int(xs.max()) + 3)
        y0p = max(0, int(np.floor(gy0 * H / gh)) - 1)
        y1p = min(H, int(np.ceil(gy1 * H / gh)) + 1)
        x0p = max(0, int(np.floor(gx0 * W / gw)) - 1)
        x1p = min(W, int(np.ceil(gx1 * W / gw)) + 1)
    else:
        y0p = y1p = x0p = x1p = 0
    prev = getattr(l, "_media_win", None)
    if prev != (y0p, y1p, x0p, x1p):
        if prev is not None:
            l.pixels[prev[0]:prev[1], prev[2]:prev[3], :] = 0.0
        else:
            l.pixels[...] = 0.0
        l._media_win = (y0p, y1p, x0p, x1p)
    if y1p <= y0p or x1p <= x0p:
        _MUT_REV[0] += 1
        return
    _up = _resize_window(src, H, W, (y0p, y1p, x0p, x1p))
    den = _up[..., 0]
    dye = _up[..., 1:4]
    d01 = np.clip(den, 0.0, 1.0)
    if kind == "fire":
        # the heat ramp: dense core white -> yellow -> orange -> deep red
        t = np.clip(den / 1.1, 0.0, 1.0)[..., None]
        cold = np.array([0.55, 0.06, 0.02], np.float32)
        mid_ = np.array([1.0, 0.45, 0.08], np.float32)
        hot = np.array([1.0, 0.93, 0.6], np.float32)
        col = np.where(t < 0.5, cold + (mid_ - cold) * (t / 0.5),
                       mid_ + (hot - mid_) * ((t - 0.5) / 0.5))
        a = np.clip(den * 1.6, 0.0, 1.0)
    elif kind == "smoke":
        g = np.clip(0.35 + 0.55 * (1.0 - d01), 0.0, 1.0)[..., None]
        col = np.concatenate([g, g, g * 1.05], axis=-1)
        a = np.clip(den * 1.1, 0.0, 0.92)
    else:
        sd = np.maximum(den, 1e-5)[..., None]
        col = np.clip(dye / sd, 0.0, 1.0)
        a = np.clip(den * 1.6, 0.0, 1.0)
    win = l.pixels[y0p:y1p, x0p:x1p]
    # patch relative to the CACHE's own revision, not the global one.
    # composite_patch's contract is "the cache was current before this
    # edit", and the cache tracks PIXEL state -- but set_frame bumps the
    # revision for its own bookkeeping without touching a pixel, so
    # comparing against the global counter rejected every patch and
    # playback kept paying a full re-composite. Any real pixel mutation
    # (paint, fill, clear) either patches or invalidates the cache
    # itself, so the cache's revision remains the honest reference.
    _cc = getattr(doc, "_ccache", None)
    rev_entry = _cc["rev"] if _cc else _MUT_REV[0]
    win[..., :3] = np.clip(col, 0.0, 1.0).astype(np.float32)
    win[..., 3] = a.astype(np.float32)
    _MUT_REV[0] += 1
    # PATCH the cached frame over just this window. Playback used to
    # invalidate the whole composite every frame and pay a full-canvas
    # re-composite on the next serve -- 106 ms at 1200x900, the single
    # largest cost in a playback round-trip, for a change covering a
    # fraction of a percent of the canvas.
    if prev is not None and prev == (y0p, y1p, x0p, x1p):
        composite_patch(doc, x0p, y0p, x1p, y1p, rev_entry)
    elif prev is not None:
        # the window moved: patch the union so the vacated area is
        # rebuilt too, never leaving a ghost of the old frame
        composite_patch(doc, min(prev[2], x0p), min(prev[0], y0p),
                        max(prev[3], x1p), max(prev[1], y1p), rev_entry)


def _fiber_grain(doc, l=None):
    """The canvas's fibre structure -- PER LAYER: each layer is its own
    sheet with its own tooth, so two absorbent layers in one document bleed
    differently. Bleeding follows the grain, which is what makes soaked
    edges RAGGED the way ink in paper is; a uniform blur would read as
    gaussian softness, not absorption. (Was per-document; the canvas-medium
    round moved it to the layer.)"""
    host = l if l is not None else doc
    g = getattr(host, "_fiber", None)
    if g is None or g.shape != (doc.height, doc.width):
        seed = hash((doc.id, getattr(host, "id", ""))) & 0x7fffffff
        rng = np.random.default_rng(seed)
        n = rng.random((doc.height, doc.width)).astype(np.float32)
        g = _gauss_blur(n[..., None], 1.2)[..., 0]
        g = (g - g.min()) / max(float(g.max() - g.min()), 1e-6)
        host._fiber = g
    return g


def soak_region(doc, lid, x0, y0, x1, y1, amount):
    """Paint SOAKS into the canvas: within the region, pigment diffuses
    along the fibre grain (ragged, directional), the wet spread carries a
    little colour past the original edge, and surface height settles (soaked
    paint sits IN the sheet, not on it). `amount` 0..1 scales everything."""
    if amount <= 0:
        return
    l = doc.layer(lid)
    h, w = doc.height, doc.width
    pad = int(6 + amount * 10)
    x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad); y1 = min(h, int(y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    win = l.pixels[y0:y1, x0:x1]
    grain = _fiber_grain(doc, l)[y0:y1, x0:x1]
    a = win[..., 3]
    pm = win[..., :3] * a[..., None]
    r_small = 1.0 + amount * 1.5
    r_big = 2.5 + amount * 5.0
    wgt = (0.35 + 0.65 * grain)[..., None]        # fibres drink unevenly
    spread_a = (_gauss_blur(a[..., None], r_small)[..., 0] * 0.5
                + _gauss_blur(a[..., None], r_big)[..., 0] * 0.5)
    spread_a = np.clip(spread_a * wgt[..., 0] * 1.25, 0.0, 1.0)
    new_a = np.maximum(a * (1.0 - 0.25 * amount), spread_a * amount
                       + a * (1 - amount))
    spread_pm = (_gauss_blur(pm, r_small) * 0.5 + _gauss_blur(pm, r_big) * 0.5)
    new_pm = pm * (1 - 0.55 * amount) + spread_pm * (0.55 * amount) * wgt
    win[..., 3] = new_a.astype(np.float32)
    sa = np.maximum(new_a, 1e-6)[..., None]
    win[..., :3] = np.clip(new_pm / sa, 0.0, 1.0)
    if l.height_map is not None:
        hm = l.height_map[y0:y1, x0:x1]
        l.height_map[y0:y1, x0:x1] = hm * (1.0 - 0.5 * amount)
    _MUT_REV[0] += 1


def _cast_shadow_radial(S, lx, ly, lz, h, w, steps=26, step_px=8.0):
    """2.5D cast shadows for POSITIONAL lights (spot/point): from every
    pixel, march toward the light's (x, y); where the surface rises
    above the climbing ray, the pixel is shaded. Per-pixel directions,
    gathered with bilinear sampling -- the directional light's uniform
    shift trick cannot bend around a point source."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.float32(lx) - xx
    dy = np.float32(ly) - yy
    dist = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dist, dy / dist
    S35 = S * 0.35
    lz35 = float(lz)                # light height in surface units
    sdw = np.zeros((h, w), np.float32)
    for t in range(1, steps + 1):
        d = step_px * t
        px = np.clip(xx + ux * d, 0, w - 1.001)
        py = np.clip(yy + uy * d, 0, h - 1.001)
        x0 = px.astype(np.int32); y0 = py.astype(np.int32)
        fx = px - x0; fy = py - y0
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        blk = (S35[y0, x0] * (1 - fx) * (1 - fy)
               + S35[y0, x1] * fx * (1 - fy)
               + S35[y1, x0] * (1 - fx) * fy
               + S35[y1, x1] * fx * fy)
        f = np.clip(d / dist, 0.0, 1.0)
        ray_h = S35 * (1 - f) + lz35 * f + 0.4
        # penumbra: farther blockers shade more softly
        sdw = np.maximum(sdw, np.clip((blk - ray_h) / (1.2 + 0.25 * t),
                                      0.0, 1.0))
        past = d >= dist
        if past.all():
            break
        sdw = np.where(past, sdw, sdw)
    return sdw


def _optically_active(l):
    """Does this layer take part in the PHYSICS of the scene?

    Visibility hides PIGMENT; it should not delete a body from the
    world. A sheet of rippled water may be hidden -- you do not want
    its blue wash over the picture -- while its surface still bends
    light, throws caustics on the sand below, and casts a shadow. Set
    `optical=True` to keep a hidden layer in the light simulation
    (Devin: 'layers should be able to have an effect on
    light/refraction/shadows while having the layer still hidden')."""
    return bool(l.visible) or bool(getattr(l, "optical", False))


def _flow_dir(doc, l):
    """Which way is DOWN for this layer's wet paint, and how hard.

    Gravity was hardcoded to screen-down, which is only right for a canvas on
    an easel. A layer standing on one of the room's walls runs down THAT
    wall, and a canvas lying flat on a table has no in-plane gravity at all --
    a puddle there spreads outward and levels instead of running. Baking the
    easel case in meant every surface behaved like an easel.

    Returns (strength, dx, dy). Strength 0 means "level, do not run".
    """
    g = getattr(l, "gravity", None)
    if g is None:
        g = getattr(doc, "gravity", 1.0)
    a = getattr(l, "gravity_angle", None)
    if a is None:
        a = getattr(doc, "gravity_angle", 90.0)
    g = float(np.clip(g, 0.0, 1.0))
    r = np.radians(float(a))
    return g, float(np.cos(r)), float(np.sin(r))   # +y is down the screen

def _on_floor(doc, l):
    """Does this layer lie on the CANVAS, as opposed to standing on one
    of the room's walls?

    A wall layer is not part of the picture's height field. It was
    being counted anyway, so a 6-unit sheet assigned to the back wall
    still embossed the canvas -- the strokes painted on the wall came
    back as raised ridges and cast shadows across the floor, on top of
    the light they were supposed to be projecting. That is the bug in
    Devin's screenshot: the wiggly line should arrive as coloured light
    or as a shadow, never as relief.

    The side currently OPEN FOR PAINTING is the exception: it is lying
    flat on the canvas by definition, so while it is open it belongs to
    the floor like any other layer."""
    if not _optically_active(l):
        return False
    side = getattr(l, "wall", None)
    return side is None or side == getattr(doc, "wall_edit", None)


def _doc_surface(doc):
    """The document's TOP surface z(x, y): the running maximum over every
    visible layer of base plane + effective thickness + relief -- the same
    geometry the volumetric compositor stacks. This is what lights shade
    and what casts their shadows."""
    h, w = doc.height, doc.width
    surf = np.zeros((h, w), np.float32)
    z_top = 0.0
    for l in doc.layers:
        if not _on_floor(doc, l):
            continue
        base = _layer_base_plane(l, h, w) + z_top
        T0 = max(float(getattr(l, "thickness", 0.0)), 0.0)
        fm = _layer_field_scale(doc, l, h, w)
        top = base + T0 * (fm if fm is not None else 1.0)
        if l.height_map is not None:
            top = top + np.asarray(l.height_map, np.float32)
        # only where the layer HAS content does its body shape the
        # surface. The old '| (T0 <= 0)' let an EMPTY zero-thickness
        # layer claim the whole canvas at its base height -- raising a
        # small shelf silently raised the surface everywhere, cancelling
        # its lamp's lift.
        # SMOOTH occupancy: the old binary threshold (alpha > 0.04)
        # raised a full-thickness CLIFF at every soft dab's edge, and
        # the lights rimmed each cliff with a specular halo -- Devin's
        # 'weird glassy loops' were the shadow layer's soft dabs
        # wearing those halos. A translucent wash is a thin film: its
        # body rises with its opacity, so soft edges slope instead of
        # step and there is nothing to rim.
        a = l.pixels[..., 3]
        aa = np.clip((a - 0.04) / 0.56, 0.0, 1.0).astype(np.float32)
        aa = aa * aa * (3.0 - 2.0 * aa)          # smoothstep
        # soften the occupancy ramp a touch further: silhouettes get a
        # shoulder instead of a crease ('the embossing is a little
        # strange around the edges')
        aa = _gauss_blur_reflect(aa, 1.6)
        # RELIEF is the layer's say in how much body it shows the
        # light: 1 = full physical thickness, 0 = optically flat
        # (paint contributes colour but no emboss)
        rlf = float(np.clip(getattr(l, "relief", 1.0), 0.0, 1.0))
        base_f = np.asarray(base, np.float32)
        top_f = np.asarray(top, np.float32)
        cand = base_f + (top_f - base_f) * aa * rlf
        # REPLACE, don't max: the topmost material OWNS the surface.
        # np.maximum let an 8-unit smoke slab's swirled filaments
        # emboss THROUGH the opaque table painted above it -- Devin's
        # glassy loops on the tabletop were the haze's relief poking
        # through a layer that visually covered it completely.
        surf = surf * (1.0 - aa) + cand * aa
        z_top += T0
    return surf


def _doc_emission(doc):
    """HDR emission: any channel above 1.0 EMITS light of that colour
    (a 300%-red pixel glows red), and a layer's `emissive` dial makes its
    whole colour radiate. Returns (h, w, 3) of emitted light, or None."""
    h, w = doc.height, doc.width
    # CHEAP REJECT: this ran a full-canvas HDR scan of every layer on
    # every lit serve -- 27 ms/frame during playback on a document with
    # no emissive layer at all. A max() per layer is orders cheaper than
    # the per-pixel arithmetic it guards.
    live = [l for l in doc.layers if _optically_active(l)]
    if not any(float(getattr(l, "emissive", 0.0)) > 0
               or float(l.pixels[..., :3].max(initial=0.0)) > 1.0
               for l in live):
        return None
    E = None
    for l in doc.layers:
        if not _on_floor(doc, l):
            continue
        a = l.pixels[..., 3:4]
        em = np.maximum(l.pixels[..., :3] - 1.0, 0.0) * a
        k = float(getattr(l, "emissive", 0.0))
        if k > 0:
            ec = getattr(l, "emissive_color", None)
            src = (np.asarray(ec, np.float32)[None, None, :]
                   * np.ones_like(l.pixels[..., :3])
                   if ec else l.pixels[..., :3])
            em = em + src * a * k
        if float(em.max()) > 1e-5:
            E = em if E is None else E + em
    return E


def estimate_perspective(doc, lid=None, image=None):
    """Read the PERSPECTIVE out of a picture: leCore finds the dominant
    vanishing point from the image's line structure; edges pointing at it
    are then suppressed and a second pass looks for a second VP (two-point
    perspective). Returns {"vps": [[x, y], ...], "horizon": [y_left,
    y_right] or None, "confidence": float} and does not change state --
    set_perspective stores it."""
    if image is None:
        l = doc.layer(lid)
        px = l.pixels
        image = (px[..., :3].mean(-1) * px[..., 3]
                 + (1.0 - px[..., 3]))          # content on white
    img = np.asarray(image, np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    mm = mind()
    h, w = img.shape
    # VP by LINE-INTERSECTION VOTING. Feeding the whole image to the
    # estimator returns a compromise between families (measured), and
    # partitioning edges by orientation fails for perspective FANS, whose
    # rays span a wide angle range (that pass put both y's 90px high and
    # starved the confidence). Instead: every strong edge pixel defines a
    # line along its tangent; random pairs intersect; intersections pile
    # up at the true vanishing points. Take the densest pile, retire the
    # lines that support it, and look again for a second.
    vps, confs = [], []
    try:
        e = np.asarray(mm.image_edges(img), np.float32)
        gy, gx = np.gradient(_gauss_blur(img[..., None], 1.0)[..., 0])
        theta = np.arctan2(gx, -gy) % np.pi
        ys, xs = np.nonzero((e > 0.5) & (np.abs(theta - np.pi / 2) > 0.10))
        if ys.size >= 120:
            rng = np.random.default_rng(0)
            take = min(4000, ys.size)
            idx = rng.choice(ys.size, take, replace=False)
            pxx = xs[idx].astype(np.float64)
            pyy = ys[idx].astype(np.float64)
            th = theta[ys[idx], xs[idx]].astype(np.float64)
            dxv, dyv = np.cos(th), np.sin(th)
            active = np.ones(take, bool)
            for _pass in range(2):
                ai = np.nonzero(active)[0]
                if ai.size < 80:
                    break
                pa = rng.choice(ai, 6000)
                pb = rng.choice(ai, 6000)
                ok = pa != pb
                pa, pb = pa[ok], pb[ok]
                # intersect line a with line b
                det = dxv[pa] * dyv[pb] - dyv[pa] * dxv[pb]
                good = np.abs(det) > 0.05
                pa, pb, det = pa[good], pb[good], det[good]
                wx = pxx[pb] - pxx[pa]
                wy = pyy[pb] - pyy[pa]
                t = (wx * dyv[pb] - wy * dxv[pb]) / det
                ix = pxx[pa] + t * dxv[pa]
                iy = pyy[pa] + t * dyv[pa]
                inb = ((ix > -2.0 * w) & (ix < 3.0 * w)
                       & (iy > -2.0 * h) & (iy < 3.0 * h))
                ix, iy = ix[inb], iy[inb]
                if ix.size < 200:
                    break
                cell = 32.0
                cx = np.floor((ix + 2 * w) / cell).astype(np.int64)
                cy = np.floor((iy + 2 * h) / cell).astype(np.int64)
                key = cy * 100000 + cx
                uk, cnt = np.unique(key, return_counts=True)
                best = uk[np.argmax(cnt)]
                nearby = key == best
                vx = float(ix[nearby].mean())
                vy = float(iy[nearby].mean())
                support = float(cnt.max()) / float(ix.size)
                if _pass == 1 and (support < 0.04
                                   or (confs and support * 12.0
                                       < confs[0] * 0.4)):
                    # a second pile must be a real second FAMILY, not the
                    # residue of noisy tangents around the first (a lone
                    # fan produced a phantom second VP on the horizon)
                    break
                vps.append([vx, vy])
                confs.append(min(1.0, support * 12.0))
                # retire supporters: lines passing within 30px of this VP
                relx = vx - pxx
                rely = vy - pyy
                dist = np.abs(relx * dyv - rely * dxv)
                active &= dist > 60.0
    except Exception:
        pass
    if not vps:
        try:
            v1, c1 = mm.vanishing_point(img, return_confidence=True)
        except Exception:
            v1, c1 = mm.vanishing_point(img), 0.5
        vps = [[float(v1[0]), float(v1[1])]]
        confs = [float(c1)]
    if len(vps) == 2 and np.hypot(vps[0][0] - vps[1][0],
                                  vps[0][1] - vps[1][1]) < max(w, h) * 0.2:
        vps = [vps[int(np.argmax(confs))]]
        confs = [max(confs)]
    conf = float(np.mean(confs))
    if len(vps) >= 2:
        (x1, y1), (x2, y2) = vps[0], vps[1]
        if abs(x2 - x1) > 1e-3:
            m = (y2 - y1) / (x2 - x1)
            horizon = [float(y1 - m * x1),
                       float(y1 + m * (doc.width - x1))]
        else:
            horizon = [float(y1), float(y1)]
    else:
        horizon = [float(vps[0][1]), float(vps[0][1])]
    return {"vps": vps, "horizon": horizon, "confidence": conf}


def _ground_shadow(doc, S, lights):
    """The GROUND PLANE as a shadow catcher: an infinite plane at z = 0
    under the whole scene. For each directional light, march from the
    plane toward the light; wherever the document's surface rises above
    the climbing ray, the ground is in shadow. Returns (h, w) occlusion
    in [0, 1]."""
    h, w = doc.height, doc.width
    occ = np.zeros((h, w), np.float32)
    for li in lights:
        if li["kind"] != "directional":
            continue
        az = np.deg2rad(float(li["azimuth"]))
        el = np.deg2rad(np.clip(float(li["elevation"]), 4.0, 89.0))
        Lx, Ly = np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el)
        step = 3.0
        rise = np.tan(el) * step * 0.35
        sdw = np.zeros((h, w), np.float32)
        for t in range(1, 26):
            ox = int(round(Lx * step * t))
            oy = int(round(Ly * step * t))
            if ox == 0 and oy == 0:
                continue
            sy0, sy1 = max(0, -oy), min(h, h - oy)
            dy0, dy1 = max(0, oy), min(h, h + oy)
            sx0, sx1 = max(0, -ox), min(w, w - ox)
            dx0, dx1 = max(0, ox), min(w, w + ox)
            if sy1 <= sy0 or sx1 <= sx0:
                break
            blocker = np.full((h, w), -1e9, np.float32)
            blocker[sy0:sy1, sx0:sx1] = S[dy0:dy1, dx0:dx1]
            ray_h = rise * t + 0.4                      # plane sits at z=0
            sdw = np.maximum(sdw, np.clip(
                (blocker * 0.35 - ray_h) / 1.5, 0.0, 1.0))
        occ = np.maximum(occ, sdw * float(li["intensity"]))
    return np.clip(occ, 0.0, 1.0)


def _light_gel(doc, li):
    """Light SHINES THROUGH layers: every translucent slab (water, glass,
    absorb, fog, inkwater with thickness) is a colour gel. The light that
    reaches the scene below is filtered per channel by Beer-Lambert
    through the slab, and the patch is OFFSET sideways by the light's
    direction times the slab's height -- a stained-glass window throws its
    colours onto the floor, not straight down its own silhouette."""
    if li["kind"] not in ("directional", "point", "spot"):
        return None
    h, w = doc.height, doc.width
    gel = None
    z_top = 0.0
    for l in doc.layers:
        if not _on_floor(doc, l):
            continue
        T = max(float(getattr(l, "thickness", 0.0)), 0.0)
        kind = getattr(l, "vol_kind", "none")
        if T > 0 and kind in ("water", "glass", "absorb", "fog",
                              "inkwater"):
            a = l.pixels[..., 3]
            if float(a.max()) > 0.02:
                dens = float(getattr(l, "vol_density", 0.5))
                path = T * a * dens * 0.10
                trans = np.exp(-(1.0 - np.clip(l.pixels[..., :3], 0, 1))
                               * path[..., None])
                # opacity beyond the tint: dense slabs also dim the light
                trans = trans * (1.0 - 0.25 * np.clip(a * dens, 0, 1)
                                 )[..., None]
                if li["kind"] == "directional":
                    az = np.deg2rad(float(li["azimuth"]))
                    el = np.deg2rad(np.clip(float(li["elevation"]),
                                            10.0, 89.0))
                    hgt = (z_top + T) * 0.35
                    ox = -np.cos(az) / np.tan(el) * hgt
                    oy = np.sin(az) / np.tan(el) * hgt
                else:
                    ox = oy = 0.0
                if abs(ox) > 0.5 or abs(oy) > 0.5:
                    sh = np.ones((h, w, 3), np.float32)
                    ix = int(round(ox))
                    iy = int(round(oy))
                    sy0, sy1 = max(0, iy), min(h, h + iy)
                    dy0, dy1 = max(0, -iy) if iy < 0 else 0, None
                    # simple shifted paste with edge fill
                    src = trans
                    ys0, ys1 = max(0, -iy), min(h, h - iy)
                    xs0, xs1 = max(0, -ix), min(w, w - ix)
                    yd0, yd1 = max(0, iy), min(h, h + iy)
                    xd0, xd1 = max(0, ix), min(w, w + ix)
                    if ys1 > ys0 and xs1 > xs0:
                        sh[yd0:yd1, xd0:xd1] = src[ys0:ys1, xs0:xs1]
                    trans = sh
                gel = trans if gel is None else gel * trans
        z_top += T
    return gel


def _wall_bounce(doc):
    """Light bouncing off the four perpendicular planes.

    A wall is not a backdrop -- it is a surface in the room, and a red
    wall to the left throws red into everything near it. Each assigned
    wall contributes its own average colour, falling off with distance
    from its edge; strength comes from how opaque and how thick the
    wall layer is, so a thin wash tints faintly and a heavy slab
    dominates. Walls act whether or not they are visible, exactly like
    an `optical` layer -- being hidden is what walls DO."""
    walls = doc.wall_layers() if hasattr(doc, "wall_layers") else {}
    if not walls:
        return None
    h, w = doc.height, doc.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w, 3), np.float32)
    any_ = False
    # a side that is OPEN FOR PAINTING is lying flat on the canvas, not
    # standing on its plane: it must not also filter the light, or the
    # artist paints against a copy of their own strokes
    _open = getattr(doc, "wall_edit", None)
    for side, l in walls.items():
        if side == _open:
            continue

        a = l.pixels[..., 3]
        cover = float(a.mean())
        if cover < 1e-3:
            continue                       # an empty wall throws nothing
        wsum = float(a.sum()) + 1e-6
        col = np.array([float((l.pixels[..., c] * a).sum() / wsum)
                        for c in range(3)], np.float32)
        T = max(float(getattr(l, "thickness", 0.0)), 0.0)
        # measured: 0.35+0.05T drove the shade past 1.5 and every
        # channel clipped, so the falloff that makes bounce READ as
        # bounce was invisible. Keep the whole term inside a stop.
        k = cover * float(getattr(l, "opacity", 1.0)) \
            * (0.10 + 0.015 * min(T, 12.0))
        reach = 0.55
        if side == "left":
            f = np.clip(1.0 - xx / (w * reach), 0.0, 1.0)
        elif side == "right":
            f = np.clip(1.0 - (w - 1 - xx) / (w * reach), 0.0, 1.0)
        elif side == "back":
            f = np.clip(1.0 - yy / (h * reach), 0.0, 1.0)
        else:                              # front: nearest the viewer
            f = np.clip(1.0 - (h - 1 - yy) / (h * reach), 0.0, 1.0)
        out += (f ** 2)[..., None] * col[None, None, :] * k
        any_ = True
    return out if any_ else None


def _wall_transmission(doc):
    """Light PASSING THROUGH a wall, carrying the wall's picture with it.

    _wall_bounce answers "what colour does this wall throw?" by
    averaging the whole layer -- which is why a stained-glass window
    and a plain red sheet looked identical, a flat wash with a
    gradient. This answers the other half, the half Devin asked for:
    the wall is a perpendicular PLANE standing on one edge of the
    canvas, and light crossing it is filtered by whatever is painted
    there, so the picture lands on the canvas as coloured light and
    shadow.

    THE MAPPING, which is what makes it perpendicular rather than
    overlaid. Take the left wall: it stands along the canvas's left
    edge, so its own two axes are NOT the canvas's. Its horizontal axis
    runs into the scene (canvas y) and its vertical axis is height
    above the canvas (z). A ray entering at height z travels away from
    the wall before it reaches the floor, so height maps to DISTANCE
    from the wall (canvas x). The wall's image therefore arrives
    transposed and stretched -- which is exactly how a window's pattern
    falls across a floor, and exactly what "overlaying" was not.

    The filter is subtractive: transmitted = 1 - a*(1 - rgb). Opaque
    black blocks (a shadow), opaque red passes red only, unpainted
    passes everything. The pattern loses contrast with distance as the
    light spreads, so it is sharpest against its own wall."""
    walls = doc.wall_layers() if hasattr(doc, "wall_layers") else {}
    if not walls:
        return None
    h, w = doc.height, doc.width
    trans = None
    # the room's height sets how far a wall's vertical axis reaches
    # across the floor. A document of thin washes and one of thick
    # slabs should not need different wall settings to look right, so
    # the stack's own depth normalises it, and wall_scale is the
    # artist's multiplier on top.
    depth = float(doc.stack_height()) if hasattr(doc, "stack_height") else 1.0
    # floor at 0.6, not 0.35: a fresh document's stack is ~1 unit deep,
    # and normalising strictly by that squeezed a wall's whole picture
    # into the first 40px of canvas -- technically a very shallow room,
    # practically a feature that looked broken out of the box.
    auto = float(np.clip(depth / 8.0, 0.6, 3.0))
    scales = getattr(doc, "wall_scale", {}) or {}
    # a side that is OPEN FOR PAINTING is lying flat on the canvas, not
    # standing on its plane: it must not also filter the light, or the
    # artist paints against a copy of their own strokes
    _open = getattr(doc, "wall_edit", None)
    for side, l in walls.items():
        if side == _open:
            continue

        scale = float(scales.get(side, 1.0)) * auto
        a = l.pixels[..., 3]
        if float(a.max(initial=0.0)) < 1e-3:
            continue                       # nothing painted: clear glass
        rgb = l.pixels[..., :3]
        wh, ww = a.shape
        yy, xx = np.mgrid[0:h, 0:w]
        if side in ("left", "right"):
            # depth into the scene -> the wall's own x; distance from
            # the wall -> the wall's own y (height)
            depth = yy
            dist = xx if side == "left" else (w - 1 - xx)
            wx = np.clip((depth * (ww - 1) // max(h - 1, 1)), 0, ww - 1)
            span = max(1.0, w * float(scale))
            wy = np.clip((dist * (wh - 1) / span).astype(np.int32),
                         0, wh - 1)
            near = 1.0 - np.clip(dist / (span * 0.75), 0.0, 1.0)
        else:
            depth = xx
            dist = yy if side == "back" else (h - 1 - yy)
            wx = np.clip((depth * (ww - 1) // max(w - 1, 1)), 0, ww - 1)
            span = max(1.0, h * float(scale))
            wy = np.clip((dist * (wh - 1) / span).astype(np.int32),
                         0, wh - 1)
            near = 1.0 - np.clip(dist / (span * 0.75), 0.0, 1.0)
        av = a[wy, wx][..., None]
        cv = np.clip(rgb[wy, wx], 0.0, 1.0)
        # WHAT THE WALL IS MADE OF decides whether light gets through.
        # A sheet of glass passes its own colour and little else stops
        # it; paper stops light in proportion to how thick it is, which
        # is why a thick sheet is a silhouette and a thin one glows.
        # Beer-Lambert in the thickness either way, with the absorption
        # coefficient set by the material.
        T = max(float(getattr(l, "thickness", 0.0)), 0.0)
        kind = str(getattr(l, "vol_kind", "none") or "none")
        if kind in ("glass", "water"):
            dens = float(np.clip(getattr(l, "vol_density", 0.35), 0.0, 4.0))
            mat = float(np.exp(-0.06 * T * (0.4 + dens)))
        elif kind in ("fog", "smoke", "inkwater", "fire"):
            mat = float(np.exp(-0.18 * T))       # scattering media
        else:
            mat = float(np.exp(-0.35 * T))       # paper / opaque stock
        # unpainted wall passes everything; painted passes its colour,
        # attenuated by the material's own absorption
        filt = (1.0 - av) + av * cv * mat
        # the pattern washes out with distance rather than stopping dead
        soften = np.clip(near, 0.0, 1.0)[..., None] ** 0.7
        t = 1.0 - (1.0 - filt) * soften * float(
            np.clip(getattr(l, "opacity", 1.0), 0.0, 1.0))
        trans = t if trans is None else trans * t
    return trans


def _doc_caustics(doc, lights):
    """Ray-density caustics from rippled water/glass: the refraction bend
    field's inverse Jacobian says where parallel light CONVERGES after
    the surface; the excess density becomes added light on whatever lies
    beneath, tinted by the light and offset by its direction."""
    dirs = [li for li in lights if li["kind"] == "directional"]
    if not dirs:
        return None
    h, w = doc.height, doc.width
    total = None
    z_top = 0.0
    for l in doc.layers:
        if not _on_floor(doc, l):
            z_top += max(float(getattr(l, "thickness", 0.0)), 0.0)
            continue
        T = max(float(getattr(l, "thickness", 0.0)), 0.0)
        kind = getattr(l, "vol_kind", "none")
        hm = getattr(l, "height_map", None)
        if hm is None and T > 0 and kind in ("water", "glass"):
            # THE PAINTED RIPPLE IS THE RIPPLE. height_map only exists
            # when impasto or a displacement op made one, so a water
            # sheet an artist simply PAINTED produced no caustics at
            # all -- the feature looked broken for its most obvious
            # use. Fall back to the layer's own alpha as the surface
            # relief, scaled by thickness.
            hm = _gauss_blur(l.pixels[..., 3:4], 1.0)[..., 0] * T
        if T > 0 and kind in ("water", "glass") and hm is not None \
                and float(np.ptp(np.asarray(hm))) > 0.1:
            a = l.pixels[..., 3]
            ior = max(float(getattr(l, "vol_ior", 1.33)), 1.0)
            gy, gx = np.gradient(
                _gauss_blur(np.asarray(hm, np.float32)[..., None],
                            1.5)[..., 0])
            bend = T * (1.0 - 1.0 / ior) * 60.0
            BX, BY = gx * bend, gy * bend
            dxy, dxx = np.gradient(BX)
            dyy, dyx = np.gradient(BY)
            det = np.clip((1.0 + dxx) * (1.0 + dyy) - dxy * dyx,
                          0.12, 4.0)
            conc = np.clip(1.0 / det - 1.0, 0.0, 2.5) * a
            for li in dirs:
                az = np.deg2rad(float(li["azimuth"]))
                el = np.deg2rad(np.clip(float(li["elevation"]),
                                        10.0, 89.0))
                hgt = (z_top + T) * 0.35
                ix = int(round(-np.cos(az) / np.tan(el) * hgt))
                iy = int(round(np.sin(az) / np.tan(el) * hgt))
                fld = np.zeros((h, w), np.float32)
                ys0, ys1 = max(0, -iy), min(h, h - iy)
                xs0, xs1 = max(0, -ix), min(w, w - ix)
                yd0, yd1 = max(0, iy), min(h, h + iy)
                xd0, xd1 = max(0, ix), min(w, w + ix)
                if ys1 > ys0 and xs1 > xs0:
                    fld[yd0:yd1, xd0:xd1] = conc[ys0:ys1, xs0:xs1]
                col = np.asarray(li["color"], np.float32) \
                    * li["intensity"]
                add = _gauss_blur(fld[..., None], 0.8) * 0.8 \
                    * col[None, None, :]
                total = add if total is None else total + add
        z_top += T
    return total


def _light_base_z(doc, li):
    """A LAYER light rides its host layer: its height is measured from
    the layer's posed base plane at the light's own (x, y), so tilting or
    raising the slab carries its lamps along. Global lights measure from
    the document floor."""
    pl = li.get("layer")
    if not pl:
        return 0.0
    try:
        host = doc.layer(pl)
    except KeyError:
        return 0.0
    h, w = doc.height, doc.width
    base = _layer_base_plane(host, h, w)
    ix = int(np.clip(li["x"], 0, w - 1))
    iy = int(np.clip(li["y"], 0, h - 1))
    z_below = 0.0
    for l in doc.layers:
        if l is host:
            break
        z_below += max(float(getattr(l, "thickness", 0.0)), 0.0)
    return float(base[iy, ix] + z_below) * 0.35


def _underwater(doc, img, lights, ca):
    """Look UP from beneath a refracting sheet. Above the surface you
    see the sheet's own pigment plus what it reflects and refracts;
    below it, the surface is a moving CEILING -- everything beyond is
    displaced by the ripple's own gradient, split into colour by
    dispersion, and the caustic filaments swim toward the eye instead
    of landing on the floor (Devin: 'a view from below the surface
    instead, with the caustics and dispersion and so on')."""
    h, w = doc.height, doc.width
    sheets = [l for l in doc.layers
              if _optically_active(l)
              and max(float(getattr(l, "thickness", 0.0)), 0.0) > 0.5]
    if not sheets:
        return img
    S = np.zeros((h, w), np.float32)
    disp = 0.0
    for l in sheets:
        a = np.clip((l.pixels[..., 3] - 0.04) / 0.56, 0.0, 1.0)
        S = S + a * max(float(getattr(l, "thickness", 0.0)), 0.0)             * float(np.clip(getattr(l, "relief", 1.0), 0.0, 1.0))
        disp = max(disp, float(getattr(l, "dispersion", 0.0)))
    S = _gauss_blur(S[..., None], 1.2)[..., 0]
    gy, gx = np.gradient(S)
    ior = max(float(getattr(sheets[-1], "vol_ior", 1.33)), 1.0)
    # bend scale in PIXELS: (ior-1)*6 put the per-channel dispersion
    # offsets under one pixel, so the split was invisible -- measured
    # and raised until the fringe actually reads
    bend = (ior - 1.0) * 22.0
    out = np.array(img, np.float32, copy=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # per-channel offset = DISPERSION: red bends least, blue most
    for c, k in ((0, 1.0 - 1.1 * disp), (1, 1.0), (2, 1.0 + 1.1 * disp)):
        sx = np.clip(xx + gx * bend * k, 0, w - 1.001)
        sy = np.clip(yy + gy * bend * k, 0, h - 1.001)
        x0 = sx.astype(np.int32); y0 = sy.astype(np.int32)
        fx = sx - x0; fy = sy - y0
        x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
        A = img[..., c]
        out[..., c] = (A[y0, x0] * (1 - fx) * (1 - fy)
                       + A[y0, x1] * fx * (1 - fy)
                       + A[y1, x0] * (1 - fx) * fy
                       + A[y1, x1] * fx * fy)
    if ca is not None:
        # from below the filaments come AT the eye: brighter, and they
        # ride the whole frame rather than only the lit floor
        out[..., :3] = out[..., :3] + np.asarray(ca, np.float32) * 1.6
    return out


def composite_lit(doc, view="flat", vantage="above"):
    """The lit composite: environment lights (view / directional / point,
    any number, summed) shade the document's height-field surface --
    directionals march REAL cast shadows across the canvas -- and emissive
    content adds itself, blooms, and throws its colour onto neighbouring
    surfaces. With no lights and no emission this returns the unlit
    composite BYTE-IDENTICAL."""
    base = (composite_volumetric(doc, view) if view in ("ortho", "persp")
            else composite_cached(doc))
    lights = []
    for li in getattr(doc, "lights", []):
        if not li.get("enabled"):
            continue
        pl = li.get("layer")
        if pl:
            try:
                host = doc.layer(pl)
            except KeyError:
                continue                      # orphaned: host gone
            if not host.visible:
                continue                      # hidden layer, dark lamp
        lights.append(li)
    E = _doc_emission(doc)
    if not lights and E is None:
        return base
    h, w = doc.height, doc.width
    S = _doc_surface(doc)
    Sb = _gauss_blur_reflect(S, 1.4)
    gy, gx = np.gradient(Sb)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny = -gx * nz, -gy * nz
    if lights:
        shade = np.full((h, w, 3), 0.30, np.float32)     # ambient floor
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for li in lights:
            col = np.asarray(li["color"], np.float32) * li["intensity"]
            if li["kind"] == "view":
                diff = nz[..., None]
            elif li["kind"] == "dome":
                # HDRI-style hemisphere for 2.5D: sky colour rains down on
                # up-facing surface, a ground bounce colour rises into
                # whatever tips away -- an environment, not a source
                sky = np.asarray(li["color"], np.float32)
                gnd = np.asarray(li.get("color2", [0.3, 0.28, 0.25]),
                                 np.float32)
                shade = shade + li["intensity"] * (
                    sky[None, None, :] * np.clip(nz, 0, 1)[..., None]
                    + gnd[None, None, :]
                    * np.clip(1.0 - nz, 0, 1)[..., None] * 0.8)
                continue
            elif li["kind"] == "spot":
                dx = li["x"] - xx
                dy = li["y"] - yy
                dz = max(float(li["z"]), 4.0) + _light_base_z(doc, li) \
                    - S * 0.35
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                Lx, Ly, Lz = dx / dist, dy / dist, dz / dist
                ax = li.get("aim_x", w / 2.0) - li["x"]
                ay = li.get("aim_y", h / 2.0) - li["y"]
                azl = -max(float(li["z"]), 4.0)
                an = np.sqrt(ax * ax + ay * ay + azl * azl)
                ca_full = np.cos(np.deg2rad(np.clip(li.get("cone", 30.0),
                                                    3.0, 85.0)))
                # cosine between light->pixel and the aim axis
                cosang = (-Lx * ax / an - Ly * ay / an - Lz * azl / an)
                sft = np.clip(li.get("soft", 0.5), 0.02, 1.0)
                inner = ca_full + (1 - ca_full) * sft * 0.9
                conew = np.clip((cosang - ca_full)
                                / max(inner - ca_full, 1e-4), 0, 1)
                diff = np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
                reach = max(float(li["z"]), 4.0) * 4.0 \
                    * max(float(li.get("scale", 1.0)), 0.05)
                diff = (diff * conew / (1.0 + (dist / reach) ** 2))[..., None]
                if li.get("shadows", True):
                    sdw = _cast_shadow_radial(
                        S, li["x"], li["y"],
                        max(float(li["z"]), 4.0)
                        + _light_base_z(doc, li), h, w)
                    diff = diff * (1.0 - 0.8 * sdw)[..., None]
            elif li["kind"] == "point":
                dx = li["x"] - xx
                dy = li["y"] - yy
                dz = max(float(li["z"]), 4.0) + _light_base_z(doc, li) \
                    - S * 0.35
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                Lx, Ly, Lz = dx / dist, dy / dist, dz / dist
                diff = np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
                reach = max(float(li["z"]), 4.0) * 4.0 \
                    * max(float(li.get("scale", 1.0)), 0.05)
                diff = (diff / (1.0 + (dist / reach) ** 2))[..., None]
                if li.get("shadows", True):
                    sdw = _cast_shadow_radial(
                        S, li["x"], li["y"],
                        max(float(li["z"]), 4.0)
                        + _light_base_z(doc, li), h, w)
                    diff = diff * (1.0 - 0.8 * sdw)[..., None]
            else:                                        # directional
                az = np.deg2rad(float(li["azimuth"]))
                el = np.deg2rad(np.clip(float(li["elevation"]), 4.0, 89.0))
                # azimuth 0 = light FROM the east (+x), 90 = from the north
                Lx = np.cos(az) * np.cos(el)
                Ly = -np.sin(az) * np.cos(el)
                Lz = np.sin(el)
                diff = np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
                # cast shadows: march toward the light; terrain that rises
                # above the climbing ray shades this pixel
                step = 3.0
                rise = np.tan(el) * step * 0.35
                sdw = np.zeros((h, w), np.float32)
                for t in range(1, 21):
                    ox = int(round(Lx * step * t))
                    oy = int(round(Ly * step * t))
                    if ox == 0 and oy == 0:
                        continue
                    sy0, sy1 = max(0, -oy), min(h, h - oy)
                    dy0, dy1 = max(0, oy), min(h, h + oy)
                    sx0, sx1 = max(0, -ox), min(w, w - ox)
                    dx0, dx1 = max(0, ox), min(w, w + ox)
                    if sy1 <= sy0 or sx1 <= sx0:
                        break
                    blocker = np.full((h, w), -1e9, np.float32)
                    blocker[sy0:sy1, sx0:sx1] = S[dy0:dy1, dx0:dx1]
                    ray_h = S * 0.35 + rise * t + 0.4
                    sdw = np.maximum(sdw, np.clip(
                        (blocker * 0.35 - ray_h) / 1.5, 0.0, 1.0))
                diff = (diff * (1.0 - 0.85 * sdw))[..., None]
            gel = _light_gel(doc, li)
            if gel is not None:
                diff = diff * gel
            shade = shade + col[None, None, :] * diff
            if li["kind"] in ("directional", "point", "spot"):
                # specular ping off the relief: light and view half-vector
                if li["kind"] == "directional":
                    az_ = np.deg2rad(float(li["azimuth"]))
                    el_ = np.deg2rad(np.clip(float(li["elevation"]),
                                             4.0, 89.0))
                    hx = np.cos(az_) * np.cos(el_) * 0.5
                    hy = -np.sin(az_) * np.cos(el_) * 0.5
                    hz_ = np.sin(el_) * 0.5 + 0.5
                else:
                    hx, hy, hz_ = Lx * 0.5, Ly * 0.5, Lz * 0.5 + 0.5
                hn = np.sqrt(np.asarray(hx) ** 2 + np.asarray(hy) ** 2
                             + np.asarray(hz_) ** 2) + 1e-6
                ndh = np.clip((nx * hx + ny * hy + nz * hz_) / hn, 0, 1)
                shade = shade + col[None, None, :] \
                    * (ndh ** 28 * 0.55)[..., None] \
                    * (diff[..., 0] > 0.02)[..., None]
    else:
        shade = np.ones((h, w, 3), np.float32)
        ca = _doc_caustics(doc, lights)
        if ca is not None and vantage == "below":
            # handed to _underwater instead: from below they are not
            # floor patterns, they are the light coming at you
            ca_below, ca = ca, None
        if ca is not None:
            # CAUSTICS: rippled water bends parallel light; where the
            # refracted rays CONVERGE the floor brightens into filaments.
            # Ray density is the inverse Jacobian of the bend field --
            # the same conservation law run_paint uses, driving light
            # instead of pigment.
            shade = shade + ca
    gcfg = (getattr(doc, "persp", {}) or {}).get("ground", {})
    if gcfg.get("enabled") and lights:
        # the ground plane catches shadows: an infinite shadow-catcher
        # under the perspective drawing, darkening only where the scene
        # occludes the light -- and only BELOW the horizon when the
        # perspective defines one
        occ = _ground_shadow(doc, S, lights)
        hz = (getattr(doc, "persp", {}) or {}).get("horizon")
        if hz:
            yy2 = np.mgrid[0:h, 0:w][0].astype(np.float32)
            hline = hz[0] + (hz[1] - hz[0]) * np.mgrid[0:h, 0:w][1] / max(w - 1, 1)
            below = np.clip((yy2 - hline) / 14.0, 0.0, 1.0)
            occ = occ * below
        k = float(gcfg.get("opacity", 0.5))
        shade = shade * (1.0 - k * occ[..., None])
    wt = _wall_transmission(doc)
    if wt is not None:
        # light crossing a wall is FILTERED by it: multiply, so an
        # opaque patch is a real shadow and a coloured patch is
        # coloured light, not a wash added on top
        shade = shade * wt
    wb = _wall_bounce(doc)
    if wb is not None:
        # the room's own light: walls tint what stands near them
        shade = shade + wb
    out = base.copy()
    if E is not None:
        # emitted light falls on the NEIGHBOURHOOD: a wide blur of the
        # emission joins the shading, so a glowing ember tints the wall
        # beside it; the emission itself stays self-lit, plus a bloom halo
        cast = _gauss_blur(E, 22.0)
        shade = shade + cast * 1.1
    out[..., :3] = base[..., :3] * np.clip(shade, 0.0, 4.0)
    if E is not None:
        out[..., :3] = out[..., :3] + E + _gauss_blur(E, 5.0) * 0.6
    if vantage == "below":
        out = _underwater(doc, out, lights,
                          locals().get("ca_below"))
    out[..., :3] = np.clip(out[..., :3], 0.0, 1.0)
    return out


def _layer_base_plane(l, h, w):
    """The slab's BASE surface as z(x,y): z_off lifts it, tilt_x/tilt_y tip
    it about the canvas centre (degrees; positive tilt_x raises the far
    edge), `curve` bows it cylindrically across x (+bulge, -sag), and
    `dome` bulges it radially (+dome, -dish). All in canvas depth units.
    Everything downstream -- collision, contact printing, refraction, paint
    running -- reads THIS surface, so a domed stamp touches centre-first
    and curved glass focuses like a lens for free."""
    z0 = float(getattr(l, "z_off", 0.0))
    tx = float(getattr(l, "tilt_x", 0.0))
    ty = float(getattr(l, "tilt_y", 0.0))
    cv = float(getattr(l, "curve", 0.0))
    dm = float(getattr(l, "dome", 0.0))
    if max(abs(tx), abs(ty), abs(cv), abs(dm)) < 1e-6:
        return np.full((h, w), z0, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = (z0 + np.tan(np.deg2rad(tx)) * (h / 2.0 - yy)
         + np.tan(np.deg2rad(ty)) * (xx - w / 2.0))
    if abs(cv) > 1e-6:
        prof = getattr(l, "curve_profile", None)
        axis = getattr(l, "curve_axis", "x")
        if prof:
            # PROFILE-DRIVEN curve: a user ramp of SIGNED depth stops
            # [t, v] along one axis (t 0..1, v -1..1) -- the bend can
            # reverse direction as many times as the ramp does. `curve`
            # is the amplitude in canvas depth units. np.interp holds
            # flat beyond the outermost stops.
            ts = np.asarray([p[0] for p in prof], np.float32)
            vs = np.asarray([p[1] for p in prof], np.float32)
            o = np.argsort(ts)
            ts, vs = ts[o], vs[o]
            t = (xx / max(w - 1, 1)) if axis == "x" else (yy / max(h - 1, 1))
            z = z + cv * np.interp(t, ts, vs).astype(np.float32)
        else:
            # legacy: a single cylindrical arc across x
            nx = (xx - w / 2.0) / (w / 2.0)
            z = z + cv * (1.0 - nx * nx)
    if abs(dm) > 1e-6:
        prof = getattr(l, "dome_profile", None)
        if prof:
            # PROFILE-DRIVEN dome: the SAME kind of ramp read as a
            # RADIUS profile -- t is normalised distance from the
            # centre (1.0 at min(w,h)/2), and the ramp is swept through
            # a full turn to a uniform radial map. `dome` is the
            # amplitude; beyond the last stop the surface holds flat.
            ts = np.asarray([p[0] for p in prof], np.float32)
            vs = np.asarray([p[1] for p in prof], np.float32)
            o = np.argsort(ts)
            ts, vs = ts[o], vs[o]
            r = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) \
                / (min(w, h) / 2.0)
            z = z + dm * np.interp(r, ts, vs).astype(np.float32)
        else:
            nr2 = (((xx - w / 2.0) / (w / 2.0)) ** 2
                   + ((yy - h / 2.0) / (h / 2.0)) ** 2)
            z = z + dm * np.clip(1.0 - nr2, 0.0, 1.0)
    return z.astype(np.float32)


def _layer_field_scale(doc, l, h, w):
    """A FIELD over the layer's VOLUME: the same stroke/mask/selection
    machinery the FX use, here modulating the slab's LOCAL thickness --
    attract swells the medium toward the source, repel hollows it, contain
    keeps thickness inside the region and thins it outside. Returns a
    per-pixel multiplier (1.0 with no field): non-uniform thickness from
    anything already in the document."""
    ref = getattr(l, "field", "")
    if not ref:
        return None
    fld = _strokefx_field(doc, ref, h, w)
    if fld is None:
        return None
    F, _, _, gw, gh = fld
    Ff = _resize(F[..., None], h, w)[..., 0]
    mode = getattr(l, "field_mode", "attract")
    k = float(getattr(l, "field_strength", 120.0)) / 120.0
    if mode == "repel":
        m = 1.0 - Ff * np.clip(k, 0, 1) * 0.95
    elif mode == "contain":
        m = np.clip(Ff * 1.4, 0.0, 1.0) ** max(k, 0.2)
    else:                                            # attract / flow: swell
        m = 1.0 + Ff * k
    return np.clip(m, 0.0, 3.0).astype(np.float32)


def run_paint(doc, lid, steps=10, gx=0.0, gy=0.0, gz=1.0, wet=1.0):
    """WET PAINT RUNS. Gravity is a full vector: gz presses the paint into
    the layer's surface so it flows DOWNHILL along the surface gradient --
    the base geometry (tilt, curve, dome) plus the impasto relief -- while
    gx/gy pull it laterally across the canvas. On a tilted slab the run
    follows the tilt, on a dome it sheets radially off the flanks, in a
    dish it pools at the centre. Semi-Lagrangian advection of the
    premultiplied pigment, with a wetness that dries out over the steps
    (later steps move less). Records undo on the layer."""
    l = doc.layer(lid)
    h, w = doc.height, doc.width
    surf = _layer_base_plane(l, h, w).astype(np.float32)
    if l.height_map is not None:
        surf = surf + np.asarray(l.height_map, np.float32)
    sgy, sgx = np.gradient(_gauss_blur(surf[..., None], 1.2)[..., 0])
    doc.record("Run paint", only=[l.id])
    px = l.pixels
    a = px[..., 3:4]
    pm = px[..., :3] * a
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # ACCUMULATE the displacement in float and sample once: stepping with
    # int-cast sampling truncated asymmetrically -- positive sub-pixel
    # velocities moved a full pixel, negative ones moved nothing, so a
    # tilted sheet refused to run downhill while a dome sheeted fine
    DX = np.zeros((h, w), np.float32)
    DY = np.zeros((h, w), np.float32)
    n_steps = max(1, int(steps))
    for i in range(n_steps):
        wetness = wet * (1.0 - i / n_steps) ** 0.7
        px_at_x = np.clip(xx - DX, 0, w - 1).astype(np.int32)
        px_at_y = np.clip(yy - DY, 0, h - 1).astype(np.int32)
        gx_here = sgx[px_at_y, px_at_x]
        gy_here = sgy[px_at_y, px_at_x]
        DX += (-gx_here * float(gz) * 12.0 + float(gx) * 0.35) * wetness
        DY += (-gy_here * float(gz) * 12.0 + float(gy) * 0.35) * wetness
    sx = np.clip(np.round(xx - DX), 0, w - 1).astype(np.int32)
    sy = np.clip(np.round(yy - DY), 0, h - 1).astype(np.int32)
    # CONSERVE the paint: gather-advection copies the source wherever the
    # flow diverges, so a dome-spread blob kept full opacity while covering
    # four times the area -- mass from nowhere. The Jacobian of the sample
    # map says how much each destination stretched its source; alpha thins
    # by it on diverging flows and builds (capped) where paint pools.
    dxy, dxx = np.gradient(DX)
    dyy, dyx = np.gradient(DY)
    det = np.clip((1.0 - dxx) * (1.0 - dyy) - dxy * dyx, 0.35, 1.6)
    pm = pm[sy, sx] * det[..., None] * 0.96
    a = np.clip(a[sy, sx] * det[..., None], 0.0, 1.0) * 0.96
    l.pixels[..., :3] = np.where(a > 1e-5, pm / np.maximum(a, 1e-5),
                                 l.pixels[..., :3])
    l.pixels[..., 3] = a[..., 0]
    _MUT_REV[0] += 1


def contact_print(doc, top_lid):
    """SCREEN PRINTING between slabs: wherever the top layer's base
    (tilted, lowered) dips below the surface of the stack beneath it, the
    layers are in CONTACT -- and the top layer's pigment transfers onto the
    layer directly below, weighted by penetration depth (deepest contact
    prints hardest, grazing contact prints faint). A tilted stamp meeting
    paper. Records undo on the receiving layer; returns the printed pixel
    count."""
    h, w = doc.height, doc.width
    idx = next(i for i, l in enumerate(doc.layers) if l.id == top_lid)
    if idx == 0:
        raise ValueError("nothing below to print onto")
    top = doc.layers[idx]
    below = doc.layers[idx - 1]
    # the surface beneath: the below layer's base + thickness + relief
    Tb = max(float(getattr(below, "thickness", 0.0)), 0.0)
    fmb = _layer_field_scale(doc, below, h, w)
    # the receiving surface uses the FIELD-EFFECTIVE thickness, so a
    # field-thickened region meets the stamp sooner -- same surface the
    # optics see
    surf = _layer_base_plane(below, h, w) \
        + (Tb * fmb if fmb is not None else Tb)
    if below.height_map is not None:
        surf = surf + np.asarray(below.height_map, np.float32)
    base = _layer_base_plane(top, h, w)
    pen = surf - base                                # >0 where they overlap
    press = np.clip(pen / 4.0, 0.0, 1.0)
    press = press * (top.pixels[..., 3] > 0.02)
    n = int((press > 0.02).sum())
    if n == 0:
        return 0
    doc.record("Contact print", only=[below.id])
    a_t = (top.pixels[..., 3] * press)[..., None]
    dst = below.pixels
    dst[..., :3] = top.pixels[..., :3] * a_t + dst[..., :3] * (1 - a_t)         * dst[..., 3:4] + dst[..., :3] * (1 - dst[..., 3:4]) * 0
    dst[..., :3] = np.where(
        (dst[..., 3:4] + a_t) > 1e-6,
        dst[..., :3], dst[..., :3])
    dst[..., 3] = np.clip(a_t[..., 0] + dst[..., 3] * (1 - a_t[..., 0]),
                          0, 1)
    _MUT_REV[0] += 1
    return n


def composite_volumetric(doc, view="ortho", fov=28.0):
    """The document as a STACK OF SLABS instead of a stack of films.

    Every layer occupies real depth: `thickness` gives it a body, and
    `vol_kind` says what the body is made of --
      water/glass  a clear slab: the view through it REFRACTS (screen-space,
                   normals from the impasto height map or the alpha edge),
                   with Beer-Lambert tinting from the layer's own colour and
                   a slope specular that sells the surface
      absorb       coloured liquid / dark glass: pure Beer-Lambert -- the
                   layer's colour IS the absorption spectrum, path length is
                   thickness x alpha x density
      fog          absorb plus SCATTER: the medium glows milky with its own
                   colour where it is dense (impure glass, murky water)
      puff         a volumetric body (cloud, foam, puffy paint): alpha
                   becomes a soft height dome, lit with high ambient and a
                   rim, silhouette softened by thickness
      none         a classic 2-D film (thickness still positions it in z)

    `view`: "ortho" looks straight down, no parallax. "persp" places each
    slab at its accumulated depth and scales it about the canvas centre --
    slabs high in the stack (near the eye) loom, deep ones recede.

    With every thickness at 0 and view "ortho" this reduces EXACTLY to the
    classic composite -- the compatibility contract the tests pin."""
    h, w = doc.height, doc.width
    out = np.zeros((h, w, 4), np.float32)
    D = (h / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    stack = [l for l in doc.layers if l.visible
             and getattr(l, "wall", None) in
             (None, getattr(doc, "wall_edit", None))]
    depths, z_top = [], 0.0
    surf_below = None            # the running TOP surface of the stack
    clips, fmuls, zmeans = [], [], []
    for l in stack:
        depths.append(z_top)
        base = _layer_base_plane(l, h, w) + z_top
        T0 = max(float(getattr(l, "thickness", 0.0)), 0.0)
        fmul0 = _layer_field_scale(doc, l, h, w)
        fmuls.append(fmul0)          # computed ONCE; the optics reuse it
        Tmap = T0 * (fmul0 if fmul0 is not None else 1.0)
        top_surf = base + Tmap
        if l.height_map is not None:
            top_surf = top_surf + np.asarray(l.height_map, np.float32)
        zmeans.append(float(np.mean(base)))
        if surf_below is None:
            clips.append(None)
            surf_below = np.asarray(top_surf, np.float32)
        else:
            # SOLID layers cannot pass through each other. The clip is a
            # SOFT waterline: visibility fades over a shallow band instead
            # of a binary aliased edge -- a slab easing under a surface
            # dips out of view, it doesn't snap
            depth_below = surf_below - np.asarray(top_surf)
            vis = np.clip(1.0 - depth_below / 1.5, 0.0, 1.0)
            clips.append(vis if float(vis.min()) < 0.999 else None)
            surf_below = np.maximum(surf_below, np.asarray(top_surf))
        z_top += T0
    for l, z0, sub, fmul, zmean in zip(stack, depths, clips, fmuls, zmeans):
        T = max(float(getattr(l, "thickness", 0.0)), 0.0)
        px = l.pixels
        if sub is not None:
            px = px.copy()
            px[..., 3] = px[..., 3] * sub
        if view == "persp" and (z_top > 0 or abs(zmean) > 1e-6):
            # a slab's CONTENT lies at its BASE (paint sits under whatever
            # medium is stacked above it), and deeper content is FARTHER
            # from the eye, so it recedes: s = D/(D+z). zmean folds in
            # z_off and the mean of tilt/curve/dome, so LIFTING a layer
            # brings it toward the eye and it LOOMS (s > 1) -- the first
            # refinement pass found z_off invisible in perspective.
            zdepth = np.clip((z_top - zmean) * 0.9, -D * 0.6, D * 2.0)
            s = D / (D + zdepth) if abs(zdepth) > 1e-6 else 1.0
            if abs(s - 1.0) > 1e-4:
                # a layer is AS LARGE AS ITS DEPTH DEMANDS: receding a slab
                # must not open a transparent border, so instead of pasting
                # a shrunken copy into a void, sample the layer's plane
                # through the inverse map with EDGE CLAMPING -- the border
                # pixels extend outward to meet the frame, exactly as if
                # the layer had the extra length and width to reach it
                yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
                sx = np.clip((xx2 - w / 2.0) / s + w / 2.0, 0, w - 1)
                sy = np.clip((yy2 - h / 2.0) / s + h / 2.0, 0, h - 1)
                px = px[sy.astype(np.int32), sx.astype(np.int32)]
        a = px[..., 3:4] * float(l.opacity)
        rgb = px[..., :3]
        if l.mask is not None:
            mk = doc.mask_by_id(l.mask)
            if mk is not None:
                mm = np.asarray(mk.data, np.float32)[..., None]
                if getattr(l, "mask_invert", False):
                    mm = 1.0 - mm
                a = a * mm
        kind = getattr(l, "vol_kind", "none")
        # dynamic media ride the existing slab optics: ink-in-water IS a
        # water slab whose content is the dye, smoke IS fog, fire is fog
        # plus an emissive add below
        kind_optics = {"inkwater": "water", "smoke": "fog",
                       "fire": "fog"}.get(kind, kind)
        dens = float(getattr(l, "vol_density", 0.5))
        Tpath = T if fmul is None else T * fmul
        if kind_optics in ("water", "glass") and T > 0:
            ior = max(float(getattr(l, "vol_ior", 1.33)), 1.0)
            hm = getattr(l, "height_map", None)
            surf = (np.asarray(hm, np.float32) if hm is not None
                    else _gauss_blur(px[..., 3][..., None], 3.0)[..., 0])
            gy, gx = np.gradient(_gauss_blur(surf[..., None], 1.5)[..., 0])
            bp = _layer_base_plane(l, h, w)
            if float(np.ptp(bp)) > 1e-6:
                # the base geometry -- tilt, curve, dome -- bends the view:
                # its per-pixel gradient joins the local surface slope, so
                # a bowed slab focuses like a lens, not just a shifted pane
                bgy, bgx = np.gradient(bp)
                gy = gy + bgy * 0.35
                gx = gx + bgx * 0.35
            bend = T * (1.0 - 1.0 / ior) * 60.0
            here = a[..., 0] > 0.02
            disp = float(getattr(l, "dispersion", 0.0))
            if disp > 1e-3:
                # DISPERSION: blue bends harder than red through the same
                # glass; sampling each channel at its own offset paints
                # rainbow fringes on every refracted edge
                shifted = np.empty_like(out)
                for ci, sc in ((0, 1.0 - 0.18 * disp), (1, 1.0),
                               (2, 1.0 + 0.22 * disp)):
                    sc_s = _vol_sample(out, gx * bend * sc,
                                       gy * bend * sc)
                    shifted[..., ci] = sc_s[..., ci]
                shifted[..., 3] = _vol_sample(out, gx * bend,
                                              gy * bend)[..., 3]
                out = np.where(here[..., None], shifted, out)
            else:
                out = np.where(here[..., None],
                               _vol_sample(out, gx * bend, gy * bend), out)
            refl = float(getattr(l, "reflect", 0.0))
            if refl > 1e-3:
                ys_c = np.nonzero(a[..., 0].max(axis=1) > 0.02)[0]
                if ys_c.size:
                    y_top = int(ys_c.min())
                    # MIRROR the scene above the waterline into the water,
                    # rippled by the surface slope and fading with depth
                    yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
                    ry = np.clip(2 * y_top - yy2 + gy * bend * 2.0,
                                 0, h - 1)
                    rx = np.clip(xx2 + gx * bend * 1.2, 0, w - 1)
                    mir = out[ry.astype(np.int32), rx.astype(np.int32)]
                    depth = np.clip((yy2 - y_top) / (h * 0.5), 0, 1)
                    fr = (refl * 0.55 * (1.0 - depth * 0.7)
                          * a[..., 0])[..., None]
                    out[..., :3] = out[..., :3] * (1 - fr) \
                        + mir[..., :3] * fr
            path = Tpath * a[..., 0] * dens * 0.12
            trans = np.exp(-(1.0 - rgb) * path[..., None])
            out[..., :3] = out[..., :3] * trans
            spec = np.clip(-gy * 2.2, 0, 1) ** 2 * 0.25 * a[..., 0]
            out[..., :3] = np.clip(out[..., :3] + spec[..., None], 0, 1)
        elif kind_optics in ("absorb", "fog") and T > 0:
            path = Tpath * a[..., 0] * dens * 0.15
            trans = np.exp(-(1.0 - rgb) * path[..., None])
            out[..., :3] = out[..., :3] * trans
            if kind_optics == "fog":
                glow = (1.0 - np.exp(-path * 0.8))[..., None]
                out[..., :3] = out[..., :3] * (1 - glow) + rgb * glow
                out[..., 3:4] = np.clip(out[..., 3:4] + glow * 0.9, 0, 1)
            if kind == "fire":
                # fire EMITS: an additive lift where the medium is dense
                out[..., :3] = np.clip(out[..., :3] + rgb * a * 0.55, 0, 1)
        elif kind_optics == "puff" and T > 0:
            dome = _gauss_blur(a, max(2.0, T * 0.5))[..., 0] * T
            gy, gx = np.gradient(_gauss_blur(dome[..., None], 2.0)[..., 0])
            lam = np.clip(0.62 + 0.5 * (-gy * 1.6 - gx * 0.5), 0.15, 1.25)
            rim = np.clip(1.0 - a[..., 0], 0, 1) ** 2
            lit = np.clip(rgb * (lam * (1 - 0.3 * rim) + 0.42)[..., None],
                          0, 1)
            soft_a = _gauss_blur(a, 1.5 + T * 0.15)
            out = np.concatenate(
                [lit * soft_a + out[..., :3] * (1 - soft_a),
                 np.clip(soft_a + out[..., 3:4] * (1 - soft_a), 0, 1)],
                axis=-1)
        else:
            pm = rgb * a
            out = np.concatenate(
                [pm + out[..., :3] * (1 - a),
                 np.clip(a + out[..., 3:4] * (1 - a), 0, 1)], axis=-1)
    return np.clip(out, 0, 1)


def composite_cached(doc):
    """The document's composite with a WINDOW-PATCH cache: paint() re-blends
    only its own dirty rectangle into the last full composite (see
    composite_patch), so the per-stroke serve cost is the window, not the
    frame -- measured 325 ms per full composite at 1080p x 4 layers. Any
    mutation that does not patch explicitly leaves the cache stale and the
    next call pays one full composite."""
    cc = getattr(doc, "_ccache", None)
    if cc is not None and cc["rev"] == _MUT_REV[0]             and cc["buf"].shape[:2] == (doc.height, doc.width):
        return cc["buf"]
    buf = composite(doc.canvas_layers() if hasattr(doc, "canvas_layers")
                    else doc.layers,
                    doc.height, doc.width, doc.mask_map())
    doc._ccache = {"rev": _MUT_REV[0], "buf": buf}
    return buf


def composite_patch(doc, x0, y0, x1, y1, rev_before):
    """Re-composite one window of the cached frame. Only valid when the cache
    was current before this edit and the edit stayed inside the window."""
    cc = getattr(doc, "_ccache", None)
    if cc is None or cc["rev"] != rev_before             or cc["buf"].shape[:2] != (doc.height, doc.width):
        return
    # RE-BASE periodically: each window re-blend rounds in float32, and
    # ~30 mixed ops accumulated a 2.3/255 drift against a fresh
    # composite -- exactly the faint 'sometimes' tile artifacts. Every
    # 20 patches the cache is dropped and the next serve pays one full
    # composite, so drift can never build past a quantum.
    cc["n"] = cc.get("n", 0) + 1
    if cc["n"] > 20:
        doc._ccache = None
        return
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(doc.width, int(x1)), min(doc.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return

    class _W:                    # window views over the real layers
        __slots__ = ("pixels", "visible", "opacity", "blend",
                     "mask", "mask_invert", "clip")

    class _MW:
        __slots__ = ("data",)

    wl = []
    for l in doc.canvas_layers() if hasattr(doc, "canvas_layers") else doc.layers:
        o = _W()
        o.pixels = _shaded_pixels(l)[y0:y1, x0:x1]
        fill = _layer_bg_fill(l, doc.height, doc.width)
        if fill is not None:
            # merge the window over a CROP of the full-canvas backing --
            # a window-sized texture would misalign with the full frame,
            # and the un-backed shim was exactly how patched strokes
            # dropped their layer's backing (the pile-up artifact class:
            # the cache window disagreed with the true composite)
            fw = fill[y0:y1, x0:x1]
            px = o.pixels
            a = px[..., 3:4]
            merged = px * a + fw * (1.0 - a)
            merged[..., 3:4] = a + fw[..., 3:4] * (1.0 - a)
            o.pixels = merged
        o.visible, o.opacity, o.blend = l.visible, l.opacity, l.blend
        o.mask = getattr(l, "mask", None)
        o.mask_invert = getattr(l, "mask_invert", False)
        o.clip = getattr(l, "clip", False)
        wl.append(o)
    wm = {}
    for mid, m in doc.mask_map().items():
        mw = _MW()
        mw.data = m.data[y0:y1, x0:x1]
        wm[mid] = mw
    cc["buf"][y0:y1, x0:x1] = composite(wl, y1 - y0, x1 - x0, wm or None)
    cc["rev"] = _MUT_REV[0]


def _shaded_pixels(lyr):
    """The layer's pixels with impasto relief applied, cached against the
    global mutation counter so the lighting is paid once per edit, not once
    per composite. paint() PATCHES this cache for its own window (see
    _shade_patch), so a brush stroke re-lights a few thousand pixels rather
    than the whole frame -- measured 268 ms per stroke at 1080p before."""
    hgt = getattr(lyr, "height_map", None)
    mat = getattr(lyr, "material_map", None)
    if mat is not None and not (mat[..., 2] > 1e-3).any():
        mat = None
    if (hgt is None or not (hgt > 0.02).any()) and mat is None:
        return lyr.pixels
    if hgt is None:
        # a flat material wash still gleams: light it over a level surface
        hgt = np.zeros(lyr.pixels.shape[:2], np.float32)
    # A STRATUM IS NOT A SEPARATE SHEET. It is the top of a paint column, and
    # lighting it on its own height alone made it read as a thin slab resting
    # on a plateau -- the visible stepping between layers. Shade the total.
    _bel = getattr(lyr, "height_below", None)
    if _bel is not None and _bel.shape == hgt.shape:
        hgt = hgt + _bel
    # the paint sits ON canvas -- but it FILLS it: the weave shows through a
    # thin film and is buried by a thick one
    hgt = hgt + (_tooth_hw(*hgt.shape, getattr(lyr, "paper", "canvas"))
                 * _CANVAS_RELIEF
                 * np.exp(-np.maximum(hgt, 0.0) / _PAINT_LEVEL))
    if getattr(lyr, "_shade_rev", None) == _MUT_REV[0]:
        return lyr._shaded
    out = _relief_shade(lyr.pixels, hgt, getattr(lyr, "paint_gloss", 0.3),
                        _MEDIA.get(getattr(lyr, "paint_media", ""), {}).get("shin", 16.0),
                        material=mat,
                        slope=_RELIEF_SLOPE * float(np.clip(
                            getattr(lyr, "relief", 1.0), 0.0, 1.0)))
    try:
        lyr._shaded, lyr._shade_rev = out, _MUT_REV[0]
    except AttributeError:
        pass
    return out


def _shade_patch(lyr, x0, y0, x1, y1, rev_before):
    """Re-light ONLY a window of the shading cache. Valid only when the cache
    was current before this edit (rev_before) and the edit stayed inside the
    window -- which paint() knows about itself. The window pads by 8 px: the
    surface-tension blur is sigma 1.4 (3 sigma = 5) plus the gradient's one,
    so lighting inside the pad is influenced by height outside the edit."""
    if getattr(lyr, "height_map", None) is None:
        return
    if getattr(lyr, "_shade_rev", None) != rev_before or not hasattr(lyr, "_shaded"):
        return                                    # stale anyway: full on demand
    H, W = lyr.height_map.shape
    # pad = trim + the shading influence radius (3-sigma blur = 5, plus
    # the gradient's 1). With pad 8 the TRUSTED interior stopped inside
    # the ring the stroke actually re-lit, so a 6 px stale seam of old
    # shading survived around every impasto stroke -- Devin's
    # 'broken artifacts show up sometimes' with paint media active.
    pad = 14
    ex0, ey0 = max(0, x0 - pad), max(0, y0 - pad)
    ex1, ey1 = min(W, x1 + pad), min(H, y1 + pad)
    if ex1 <= ex0 or ey1 <= ey0:
        return
    if ex0 == 0 or ey0 == 0 or ex1 == W or ey1 == H:
        # CANVAS-EDGE strokes cannot be window-patched honestly: the
        # blur is an FFT, so the window's wrap mixes the stroke with
        # itself while the full frame's wrap mixes in the OPPOSITE
        # CANVAS EDGE -- the two disagree in the untrimmed edge zone
        # (measured 2.3/255, healing only on the next full rebuild).
        # Invalidate instead; the next serve recomputes the full shade.
        lyr._shade_rev = None
        return
    mat = getattr(lyr, "material_map", None)
    _hw = lyr.height_map[ey0:ey1, ex0:ex1]
    _bel = getattr(lyr, "height_below", None)
    if _bel is not None and _bel.shape == lyr.height_map.shape:
        _hw = _hw + _bel[ey0:ey1, ex0:ex1]
    win = _relief_shade(lyr.pixels[ey0:ey1, ex0:ex1],
                        _hw + (_tooth_hw(*lyr.height_map.shape,
                                        getattr(lyr, "paper", "canvas"))[ey0:ey1, ex0:ex1]
                               * _CANVAS_RELIEF
                               * np.exp(-np.maximum(_hw, 0.0) / _PAINT_LEVEL)),
                        getattr(lyr, "paint_gloss", 0.3),
                        _MEDIA.get(getattr(lyr, "paint_media", ""), {}).get("shin", 16.0),
                        material=None if mat is None else mat[ey0:ey1, ex0:ex1],
                        slope=_RELIEF_SLOPE * float(np.clip(
                            getattr(lyr, "relief", 1.0), 0.0, 1.0)))
    # the outermost pad ring saw a CROPPED blur neighbourhood; only trust the
    # interior of the window
    trim = 6
    tx0, ty0 = ex0 + (trim if ex0 > 0 else 0), ey0 + (trim if ey0 > 0 else 0)
    tx1, ty1 = ex1 - (trim if ex1 < W else 0), ey1 - (trim if ey1 < H else 0)
    if tx1 <= tx0 or ty1 <= ty0:
        return
    lyr._shaded[ty0:ty1, tx0:tx1] = win[ty0 - ey0:ty1 - ey0, tx0 - ex0:tx1 - ex0]
    lyr._shade_rev = _MUT_REV[0]


def _layer_bg_fill(lyr, h, w):
    """The layer's own backing sheet: a solid colour or a procedural
    paper/canvas/noise texture, composited UNDER the layer's pixels.
    Cached per revision and size."""
    bg = getattr(lyr, "bg", None)
    if not bg:
        return None
    ck = (_MUT_REV[0], h, w)
    if getattr(lyr, "_bg_ck", None) == ck:
        return lyr._bg_fill
    col = np.asarray(bg.get("color", [1, 1, 1, 1]), np.float32)
    fill = np.empty((h, w, 4), np.float32)
    fill[...] = col
    if bg.get("kind") == "texture":
        rng = np.random.default_rng(7)
        sc = max(float(bg.get("scale", 3.0)), 0.5)
        gh, gw = max(int(h / sc / 4), 2), max(int(w / sc / 4), 2)
        g = rng.random((gh, gw), np.float32)
        n = _resize(np.repeat(g[..., None], 3, -1), h, w)[..., 0]
        tex = bg.get("tex", "paper")
        if tex == "canvas":                  # woven: two crossed sines
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            n = 0.5 + 0.25 * np.sin(xx / sc) * np.sin(yy / sc) \
                + 0.35 * (n - 0.5)
        elif tex == "noise":
            pass                             # raw field
        else:                                # paper: soft fibrous mottle
            n = 0.72 + 0.28 * n
        fill[..., :3] *= n[..., None] * 0.35 + 0.72
    try:
        lyr._bg_ck, lyr._bg_fill = ck, fill
    except AttributeError:
        pass
    return fill


def composite(layers, h, w, masks=None):
    """Bottom-up over-composite of visible layers -> (H, W, 4). `masks` maps mask id
    -> Mask; a layer's attached mask (optionally inverted) gates its alpha."""
    out = np.zeros((h, w, 4), np.float32)
    for lyr in layers:
        if not lyr.visible or float(lyr.opacity) <= 0.0:
            continue
        # A layer whose alpha is entirely zero contributes NOTHING, but it was
        # still costing a full blend -- four empty layers at 1920x1080 measured
        # 737 ms of pure no-op. The test is ~3 ms per layer, so it is cached
        # against the global mutation counter: paid once after an edit, free
        # on every composite until the next one.
        if getattr(lyr, "_empty_rev", None) == _MUT_REV[0]:
            if lyr._empty:
                continue
        else:
            empty = (not bool(lyr.pixels[..., 3].any())
                     and not getattr(lyr, "bg", None))
            try:
                lyr._empty, lyr._empty_rev = empty, _MUT_REV[0]
            except AttributeError:
                pass          # __slots__ shim (composite_display): skip caching
            if empty:
                continue
        px = _shaded_pixels(lyr)
        fill = _layer_bg_fill(lyr, px.shape[0], px.shape[1])
        if fill is not None:
            # the layer's pixels sit ON their backing sheet
            a = px[..., 3:4]
            px = px * a + fill * (1.0 - a)
            px[..., 3:4] = a + fill[..., 3:4] * (1.0 - a)
        if px.shape[0] != h or px.shape[1] != w:
            # render_at() evaluates the graph at a target resolution by
            # changing the document size, but layers keep their own pixel
            # arrays -- so the accumulator was target-sized while the sources
            # were canvas-sized and nothing broadcast. Resample per layer.
            px = _resize(px, h, w)
        src_rgb, src_a = px[..., :3], px[..., 3:4] * float(lyr.opacity)
        m = (masks or {}).get(getattr(lyr, "mask", None))
        if m is not None:
            mv = _resize(m.data, h, w)[..., None]
            src_a = src_a * ((1.0 - mv) if lyr.mask_invert else mv)
        if getattr(lyr, "clip", False):
            # CLIPPING MASK (Photoshop's clip-to-below, Procreate's clipping
            # mask): this layer only shows where its base -- the nearest
            # non-clipped layer below it -- has pixels. A run of consecutive
            # clipped layers all clips to the same base.
            base = None
            idx = layers.index(lyr)
            for b in reversed(layers[:idx]):
                if not getattr(b, "clip", False):
                    base = b
                    break
            if base is None or not base.visible:
                continue                        # nothing to clip to: invisible
            bpx = base.pixels
            ba = bpx[..., 3:4]
            if bpx.shape[0] != h or bpx.shape[1] != w:
                ba = _resize(bpx, h, w)[..., 3:4]
            src_a = src_a * ba
        blended = BLEND_MODES.get(lyr.blend, _bl_normal)(out[..., :3], src_rgb)
        # where the backdrop is transparent a separable blend falls back to the source colour
        blended = np.where(out[..., 3:4] > 0, blended, src_rgb)
        out_a = src_a + out[..., 3:4] * (1 - src_a)
        rgb = blended * src_a + out[..., :3] * out[..., 3:4] * (1 - src_a)
        out[..., :3] = np.where(out_a > 0, rgb / np.maximum(out_a, 1e-6), 0)
        out[..., 3:4] = out_a
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------------------------------------
# Layers + Document (undo/redo follows leCore's Scene: snapshot steps, coalescible)
# ------------------------------------------------------------------------------------------------

class Layer:
    _next = 1

    def __init__(self, h, w, name=None, pixels=None):
        self.id = f"L{Layer._next}"; Layer._next += 1
        self.name = name or self.id
        self.visible = True
        self.opacity = 1.0
        self.blend = "normal"
        self.mask = None          # a Mask id, or None
        self.mask_invert = False
        self.height_map = None    # impasto: per-pixel paint thickness, lazy
        self.material_map = None  # PBR paint: (H,W,3) [rough, metal,
                                  # coverage], lazy like height_map
        self.wall = None          # which perpendicular plane this stands on
        self.paint_gloss = 0.3    # specular strength of the last media used
        self.alpha_lock = False   # paint recolors existing pixels only
        self.clip = False         # composite clips to the layer below's alpha
        # --- the physical layer: a slab, not a film -------------------------
        # depth of the slab in canvas units, 1 unit = 0.1 mm. EVERY layer
        # is a physical sheet: the default is paper (0.1 mm) and nothing
        # may be thinner than 0.01 mm -- there is no zero-thickness.
        self.thickness = 1.0
        self.bg = None            # None=transparent, or {"kind":"color"|
                                  # "texture", "color":[...], "tex":..,
                                  # "scale":..}
        self.vol_kind = "none"    # none|water|glass|fog|absorb|puff
        self.vol_ior = 1.33       # refraction index (water 1.33, glass 1.5)
        self.vol_density = 0.5    # how strongly the medium absorbs/scatters
        self.absorbency = 0.0     # canvas fibre: paint soaks and bleeds
        self.emissive = 0.0       # >0: this layer EMITS light
        self.emissive_color = None  # [r,g,b]: emit THIS colour (else pixels)
        self.reflect = 0.0        # water/glass: mirror the scene above
        self.dispersion = 0.0     # per-channel refraction: rainbow fringes
        self.media_rate = 1.0     # media sim speed vs the timeline (keyable)
        self.z_off = 0.0          # lift/lower the whole slab in the stack
        self.tilt_x = 0.0         # degrees about the x-axis: z varies with y
        self.tilt_y = 0.0         # degrees about the y-axis: z varies with x
        self.curve = 0.0          # cylindrical bow: +bulge/-sag across x
        self.dome = 0.0           # radial bulge: +dome/-dish
        self.field = ""           # a stroke/mask/sel driving this VOLUME
        self.field_mode = "attract"
        self.field_strength = 120.0
        if pixels is None:
            pixels = np.zeros((h, w, 4), np.float32)
        self.pixels = _f32(pixels)

    def meta(self):
        return {"id": self.id, "name": self.name, "visible": self.visible,
                "opacity": self.opacity, "blend": self.blend,
                "mask": self.mask, "mask_invert": self.mask_invert,
                # whether the re-render commands mean anything for this layer,
                # so the UI can dim them instead of offering a refusal
                "alpha_lock": bool(getattr(self, "alpha_lock", False)),
                "clip": bool(getattr(self, "clip", False)),
                "thickness": float(getattr(self, "thickness", 0.0)),
                "vol_kind": getattr(self, "vol_kind", "none"),
                "vol_ior": float(getattr(self, "vol_ior", 1.33)),
                "vol_density": float(getattr(self, "vol_density", 0.5)),
                "absorbency": float(getattr(self, "absorbency", 0.0)),
                "emissive": float(getattr(self, "emissive", 0.0)),
                "emissive_color": getattr(self, "emissive_color", None),
                "reflect": float(getattr(self, "reflect", 0.0)),
                "dispersion": float(getattr(self, "dispersion", 0.0)),
                "media_rate": float(getattr(self, "media_rate", 1.0)),
                "bg": json.loads(json.dumps(getattr(self, "bg", None))),
                "z_off": float(getattr(self, "z_off", 0.0)),
                "tilt_x": float(getattr(self, "tilt_x", 0.0)),
                "tilt_y": float(getattr(self, "tilt_y", 0.0)),
                "curve": float(getattr(self, "curve", 0.0)),
                "dome": float(getattr(self, "dome", 0.0)),
                "field": getattr(self, "field", ""),
                "field_mode": getattr(self, "field_mode", "attract"),
                "field_strength": float(getattr(self, "field_strength",
                                                120.0)),
                "placed": getattr(self, "source", None) is not None,
                "place": json.loads(json.dumps(getattr(self, "place",
                                                       None))),
                "locked": bool(getattr(self, "locked", False)),
                "relief": float(getattr(self, "relief", 1.0)),
                # which way is DOWN for this surface: the UI needs to read
                # it back or the control cannot show the layer's state
                "gravity": (None if getattr(self, "gravity", None) is None
                            else float(self.gravity)),
                "gravity_angle": (None if getattr(self, "gravity_angle", None) is None
                                  else float(self.gravity_angle)),
                "optical": bool(getattr(self, "optical", False)),
                "media_res": str(getattr(self, "media_res", "normal")),
                "media_time": str(getattr(self, "media_time", "timeline")),
                "wall": getattr(self, "wall", None),
                "curve_axis": getattr(self, "curve_axis", "x"),
                "curve_profile": json.loads(json.dumps(
                    getattr(self, "curve_profile", None))),
                "dome_profile": json.loads(json.dumps(
                    getattr(self, "dome_profile", None)))}


class Brush:
    """A greyscale stamp tip (T, T) in [0, 1] plus stroke spacing. Standard brushes
    are seeded procedurally; custom ones are written by the Brush out graph node."""
    _next = 1
    TIP = 128

    def __init__(self, name=None, tip=None, spacing=0.25, builtin=False):
        self.id = f"B{Brush._next}"; Brush._next += 1
        self.name = name or self.id
        self.spacing = float(spacing)
        self.builtin = bool(builtin)
        self.follow = False        # rotate the tip to the stroke direction
        self.j_angle = 0.0         # random rotation, degrees
        self.j_size = 0.0          # random scale, fraction
        self.j_scatter = 0.0       # random offset, fraction of size
        if tip is None:
            tip = np.zeros((self.TIP, self.TIP), np.float32)
        self.tip = np.clip(_f32(tip), 0, 1)

    def meta(self):
        return {"id": self.id, "name": self.name, "spacing": self.spacing,
                "builtin": self.builtin, "follow": self.follow,
                "j_angle": self.j_angle, "j_size": self.j_size,
                "j_scatter": self.j_scatter}


class Stamp:
    """A reusable RGBA sticker: captured from a layer (optionally through a
    selection) and placed by click, any number of times, at any scale and
    rotation. Pixels are straight-alpha float32 (h, w, 4)."""
    _next = 1

    def __init__(self, name=None, pixels=None):
        self.id = "ST%d" % Stamp._next
        Stamp._next += 1
        self.name = name or ("Stamp %d" % (Stamp._next - 1))
        self.pixels = np.asarray(pixels, np.float32)


def _standard_brushes():
    """The stock set, all procedural: soft/hard rounds, chalk, calligraphy, spray."""
    T = Brush.TIP
    ys, xs = np.mgrid[0:T, 0:T]
    cx = cy = (T - 1) / 2
    d = np.hypot(xs - cx, ys - cy) / (T / 2)
    rng = np.random.default_rng(7)
    soft = np.clip(1 - d, 0, 1) ** 2
    hard = (d <= 0.92).astype(np.float32)
    hard = _gauss_blur(hard, 1.2)
    chalk = np.clip(1 - d, 0, 1) * np.clip(rng.random((T, T)) * 1.6 - 0.35, 0, 1)
    chalk = np.clip(_gauss_blur(chalk, 0.6) * 1.5, 0, 1)
    ang = np.deg2rad(40.0)
    u = (xs - cx) * np.cos(ang) + (ys - cy) * np.sin(ang)
    v = -(xs - cx) * np.sin(ang) + (ys - cy) * np.cos(ang)
    callig = np.clip(1 - np.hypot(u / (T * 0.46), v / (T * 0.12)), 0, 1) ** 0.7
    spray = np.zeros((T, T), np.float32)
    pts = rng.random((240, 2)) * T
    keep = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < T * 0.46
    for px, py in pts[keep]:
        spray[int(py), int(px)] = 1.0
    spray = np.clip(_gauss_blur(spray, 0.8) * 2.2, 0, 1)
    return [Brush("Soft round", soft, 0.2, True),
            Brush("Hard round", hard, 0.15, True),
            Brush("Chalk", chalk, 0.3, True),
            Brush("Calligraphy", callig, 0.12, True),
            Brush("Spray", spray, 0.5, True)]


class Selection:
    """A stored selection: a greyscale field like a mask, but living in its own
    list, combinable with set operations, and convertible into a Mask."""
    _next = 1

    def __init__(self, h, w, name=None, data=None):
        self.id = f"S{Selection._next}"; Selection._next += 1
        self.name = name or self.id
        self.data = _f32(data) if data is not None else np.zeros((h, w), np.float32)

    def meta(self):
        return {"id": self.id, "name": self.name}


class Spline:
    """An editable path: control points with symmetric tangent handles, cubic
    bezier between neighbours, optionally closed. Strokeable with any brush and
    usable as a movement constraint for freehand painting."""
    _next = 1

    def __init__(self, name=None, points=None, closed=False):
        self.id = f"P{Spline._next}"; Spline._next += 1
        self.name = name or self.id
        # points: [{x, y, hx, hy}]  (hx, hy) = out-handle; in-handle mirrors it
        self.points = [dict(p) for p in (points or [])]
        self.closed = bool(closed)

    def meta(self):
        return {"id": self.id, "name": self.name,
                "points": [dict(p) for p in self.points], "closed": self.closed}

    def flatten(self, per_seg=24):
        """Sample the path into a polyline [(x, y), ...]."""
        pts = self.points
        if len(pts) < 2:
            return [(p["x"], p["y"]) for p in pts]
        segs = list(zip(pts, pts[1:] + ([pts[0]] if self.closed else [])))
        out = []
        for a, b in segs:
            p0 = np.array([a["x"], a["y"]], float)
            p3 = np.array([b["x"], b["y"]], float)
            c1 = p0 + np.array([a.get("hx", 0.0), a.get("hy", 0.0)], float)
            c2 = p3 - np.array([b.get("hx", 0.0), b.get("hy", 0.0)], float)
            for t in np.linspace(0, 1, int(per_seg), endpoint=False):
                q = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1
                     + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3)
                out.append((float(q[0]), float(q[1])))
        last = pts[0] if self.closed else pts[-1]
        out.append((last["x"], last["y"]))
        return out


class Mask:
    """A named greyscale field (H, W) in [0, 1]. Attached to layers it gates their
    alpha; picked as the selection it gates the brush; in the graph it is an image."""
    _next = 1

    def __init__(self, h, w, name=None, data=None):
        self.id = f"M{Mask._next}"; Mask._next += 1
        self.name = name or self.id
        self.data = _f32(data) if data is not None else np.ones((h, w), np.float32)

    def meta(self):
        return {"id": self.id, "name": self.name}


class Document:
    """The single source of truth: an ordered stack of layers with undo history."""

    _next = 1

    def __init__(self, width=768, height=512, name=None, background=(1.0, 1.0, 1.0)):
        self.id = f"D{Document._next}"; Document._next += 1
        self.name = name or f"Untitled {self.id[1:]}"
        self.width, self.height = int(width), int(height)
        self.layers = []
        self.groups = []          # [{id, name, layers: [layer_id, ...]}]
        # Stroke groups bundle a blended passage into ONE editable object.
        # The replay pipeline already makes mixing non-destructive, so this
        # buys organisation, not semantics: you can drag a whole blended sky
        # without rubber-banding a marquee round every contributing mark.
        self.stroke_groups = []   # [{id, name, strokes: [sid, ...]}]
        # REAL BRUSH MODE: the brush is a physical object that holds a finite
        # amount of paint. It empties as you work, it can be recharged by
        # going and getting more from thick paint already on the canvas, and
        # it can be fully reloaded from the palette. See `load_brush`.
        # When a layer fills up, spill the excess onto a fresh one. OPT-IN:
        # it changes how a heavily worked passage builds (and costs a full
        # extra composite pass per stratum), so it is a choice the painter
        # makes rather than something that happens to their document.
        self.auto_stratum = False
        self.brush_charge = 1.0          # 0..1 of capacity
        self.brush_color = (0.0, 0.0, 0.0)
        # A palette is not part of a picture and does not belong to any of its
        # layers. It is its own small surface, and the brush you dip there is
        # the SAME brush you paint with -- see `_brush_host`.
        self._palette_doc = None
        self._brush_host = None
        self._gnext = 1
        self.masks = []
        self.selections = []
        # The UNSAVED working selection. Most selections are momentary, so one
        # scratch slot is reused until the user chooses to keep it.
        self._scratch_sel = None
        # Physical scale. It is what makes "resize the image" and "resize the
        # canvas" different operations, and what lets a brush be 2 mm rather
        # than 8 px. 72 is the screen default; imports adopt the file's value.
        self.dpi = 72.0
        self.splines = []
        # Every brush stroke, kept as a path. The points already travel to the
        # server on every flush, so remembering them costs almost nothing and
        # buys a lot: Stroke FX can attach to any stroke after the fact, and a
        # stroke list is what an animation timeline would replay. Bounded, so a
        # long session cannot grow without limit.
        self.strokes = []
        self.brushes = _standard_brushes()
        self.stamps = []
        self.lights = []          # environment lights: dicts, see add_light
        self.fields = []          # force-field objects: dicts, see add_field
        # THE ROOM. Four planes perpendicular to the canvas, outside it:
        # front, back, left, right. Each slot holds a normal layer id or
        # None, and starts empty -- a document is a flat canvas until an
        # artist decides otherwise. A wall layer is painted like any
        # other (that is the point: no special editor), and is hidden
        # from the canvas except while its slot is being edited.
        self.walls = {"front": None, "back": None,
                      "left": None, "right": None}
        self.wall_edit = None     # which side is open for painting
        # How tall the room is, per side. A wall's own vertical axis is
        # HEIGHT above the canvas, and how far that height reaches
        # across the floor depends on how deep the stack actually is --
        # which varies per document. 1.0 means "the room is as tall as
        # the layer stack is thick"; raise it for a cathedral window,
        # lower it for a slide under glass.
        self.wall_scale = {"front": 1.0, "back": 1.0,
                           "left": 1.0, "right": 1.0}
        self.frame = 0.0          # the global playhead, in frames
        self.fps = 24.0
        self.frame_range = [0.0, 96.0]
        self.tracks = {}          # "kind:id:prop" -> [[t, v], ...] sorted
        self.persp = {"enabled": False, "vps": [], "horizon": None,
                      "snap": False,
                      "ground": {"enabled": False, "grid": False,
                                 "opacity": 0.5}}
        self._lnext = 1
        self._fnext = 1
        self._undo, self._redo = [], []
        bg = Layer(self.height, self.width, "Background")
        if background is None:
            bg.pixels[...] = 0.0                       # transparent canvas
        else:
            r, g, b = (list(background) + [0, 0, 0])[:3]
            bg.pixels[..., 0], bg.pixels[..., 1], bg.pixels[..., 2] = r, g, b
            bg.pixels[..., 3] = 1.0
        self.layers.append(bg)

    # --- undo ------------------------------------------------------------------------------------
    def _snapshot(self, only=None, region=None):
        """`only` = layer ids whose PIXELS this operation can change. Others
        store None and are left as-is on restore.

        A full snapshot copies every layer's pixel buffer -- 132 MB for four
        layers at 1920x1080, which showed up as a ~470 ms stall at the start of
        every brush stroke. A stroke touches exactly one layer, so copying the
        rest is pure waste. Operations that can touch anything (resize, crop,
        merge, restructuring) still pass nothing and get the full copy."""
        keep = None if only is None else set(only)

        def _px(l):
            if keep is not None and l.id not in keep:
                return None                      # untouched: restore leaves it
            if region is not None:
                x0, y0, x1, y1 = region
                return (region, l.pixels[y0:y1, x0:x1].copy())
            return l.pixels.copy()

        def _hg(l):
            # the paint SURFACE is document state exactly like the pigment:
            # undoing an impasto stroke must flatten its ridge too. `False`
            # marks "had no height field" so redo/undo can drop it again.
            hm = getattr(l, "height_map", None)
            if keep is not None and l.id not in keep:
                return None
            if hm is None:
                return False
            if region is not None:
                x0, y0, x1, y1 = region
                return (region, hm[y0:y1, x0:x1].copy())
            return hm.copy()

        def _mat(l):
            # the material map is document state exactly like the height:
            # undoing a gold stroke must take its metalness with it, or the
            # next plain stroke there gleams for no reason. Same window
            # discipline as _hg.
            mm = getattr(l, "material_map", None)
            if keep is not None and l.id not in keep:
                return None
            if mm is None:
                return False
            if region is not None:
                x0, y0, x1, y1 = region
                return (region, mm[y0:y1, x0:x1].copy())
            return mm.copy()
        return {"w": self.width, "h": self.height, "partial": keep is not None,
                "groups": [dict(g, layers=list(g["layers"])) for g in self.groups],
                "stroke_groups": [dict(g, strokes=list(g["strokes"]))
                                  for g in self.stroke_groups],
                "masks": [(m.id, m.name, m.data.copy()) for m in self.masks],
                "selections": [(x.id, x.name, x.data.copy())
                               for x in self.all_selections()],
                "splines": [(p.id, p.name, [dict(q) for q in p.points], p.closed)
                            for p in self.splines],
                "brushes": [(b.id, b.name, b.spacing, b.builtin, b.tip.copy(),
                             b.follow, b.j_angle, b.j_size, b.j_scatter)
                            for b in self.brushes],
                "stamps": [(s.id, s.name, s.pixels.copy())
                           for s in self.stamps],
                "lights": [dict(li) for li in getattr(self, "lights", [])],
                "fields": [dict(f) for f in getattr(self, "fields", [])],
                "walls": dict(getattr(self, "walls", {})),
                "wall_edit": getattr(self, "wall_edit", None),
                "wall_scale": dict(getattr(self, "wall_scale", {}) or {}),
                "persp": json.loads(json.dumps(getattr(self, "persp", {}))),
                "tracks": json.loads(json.dumps(getattr(self, "tracks",
                                                        {}))),
                # rec[11] = the PHYSICAL SHEET: before it existed, every
                # undo rebuilt bare Layer objects and silently reset
                # thickness, volume kind, backing, pose, optics, and
                # placement to defaults across the whole document. `source`
                # rides by REFERENCE (it is never mutated in place), so
                # snapshots stay cheap.
                "layers": [(l.id, l.name, l.visible, l.opacity, l.blend,
                            l.mask, l.mask_invert,
                            _px(l), _hg(l),
                            getattr(l, "paint_gloss", 0.3),
                            getattr(l, "paint_media", None),
                            {"thickness": float(getattr(l, "thickness", 1.0)),
                             "vol_kind": getattr(l, "vol_kind", "none"),
                             "vol_ior": float(getattr(l, "vol_ior", 1.33)),
                             "vol_density": float(getattr(l, "vol_density",
                                                          0.5)),
                             "absorbency": float(getattr(l, "absorbency",
                                                         0.0)),
                             "emissive": float(getattr(l, "emissive", 0.0)),
                             "emissive_color": getattr(l, "emissive_color",
                                                       None),
                             "reflect": float(getattr(l, "reflect", 0.0)),
                             "dispersion": float(getattr(l, "dispersion",
                                                         0.0)),
                             "media_rate": float(getattr(l, "media_rate",
                                                         1.0)),
                             "bg": json.loads(json.dumps(getattr(l, "bg",
                                                                 None))),
                             "z_off": float(getattr(l, "z_off", 0.0)),
                             "tilt_x": float(getattr(l, "tilt_x", 0.0)),
                             "tilt_y": float(getattr(l, "tilt_y", 0.0)),
                             "curve": float(getattr(l, "curve", 0.0)),
                             "dome": float(getattr(l, "dome", 0.0)),
                             "field": getattr(l, "field", ""),
                             "field_mode": getattr(l, "field_mode",
                                                   "attract"),
                             "field_strength": float(getattr(
                                 l, "field_strength", 120.0)),
                             "alpha_lock": bool(getattr(l, "alpha_lock",
                                                        False)),
                             "clip": bool(getattr(l, "clip", False)),
                             "place": json.loads(json.dumps(
                                 getattr(l, "place", None))),
                             "locked": bool(getattr(l, "locked", False)),
                             "relief": float(getattr(l, "relief", 1.0)),
                             # the UNDO snapshot is a separate path from
                             # save/load: without these an undo turned the
                             # palette back into an ordinary layer and broke
                             # its dock, and severed the stratum chain
                             "palette": bool(getattr(l, "palette", False)),
                             "stratum_of": getattr(l, "stratum_of", None),
                             "stratum_next": getattr(l, "stratum_next", None),
                             "stratum_root": getattr(l, "stratum_root", None),
                             # which way is DOWN for this surface's wet
                             # paint -- an easel, a wall, or flat on a table
                             "gravity": (None if getattr(l, "gravity", None) is None
                                         else float(l.gravity)),
                             "gravity_angle": (None if getattr(l, "gravity_angle", None) is None
                                               else float(l.gravity_angle)),
                             "optical": bool(getattr(l, "optical", False)),
                             "media_res": str(getattr(l, "media_res", "normal")),
                             "media_time": str(getattr(l, "media_time", "timeline")),
                             "curve_axis": getattr(l, "curve_axis", "x"),
                             "curve_profile": json.loads(json.dumps(
                                 getattr(l, "curve_profile", None))),
                             "dome_profile": json.loads(json.dumps(
                                 getattr(l, "dome_profile", None))),
                             "source": getattr(l, "source", None)},
                            # rec[12]: the material map, windowed like _hg --
                            # older snapshots simply lack the slot
                            _mat(l))
                           for l in self.layers],
                # Stroke PATHS are document state too. Without them undo put the
                # pixels back but left the paths where the edit moved them, so a
                # nudge or a simulation looked undone until the next replay
                # painted the moved path again. Paths are tiny next to pixels.
                "strokes": [{"id": k["id"], "layer": k["layer"],
                             "points": [list(pt) for pt in k["points"]],
                             "brush": dict(k["brush"]),
                             "rig": ({"bones": list(k["rig"]["bones"]),
                                      "pins": list(k["rig"]["pins"]),
                                      "prev": [list(pt) for pt in k["rig"]["prev"]],
                                      "keys": dict(k["rig"].get("keys", {}))}
                                     if k.get("rig") else None)}
                            for k in self.strokes]}

    def _restore(self, snap):
        self.width, self.height = snap["w"], snap["h"]
        self.groups = [dict(g, layers=list(g["layers"])) for g in snap.get("groups", [])]
        self.stroke_groups = [dict(g, strokes=list(g["strokes"]))
                              for g in snap.get("stroke_groups", [])]
        if "brushes" in snap:
            self.brushes = []
            for rec in snap["brushes"]:
                bid, name, spacing, builtin, tip = rec[:5]
                b = Brush(name, tip.copy(), spacing, builtin)
                b.id = bid
                if len(rec) > 5:
                    b.follow, b.j_angle, b.j_size, b.j_scatter = rec[5:9]
                self.brushes.append(b)
        if "tracks" in snap:
            self.tracks = json.loads(json.dumps(snap["tracks"]))
        if "persp" in snap and snap["persp"]:
            self.persp = json.loads(json.dumps(snap["persp"]))
        if "lights" in snap:
            self.lights = [dict(li) for li in snap["lights"]]
        if "fields" in snap:
            self.fields = [dict(f) for f in snap["fields"]]
        if "walls" in snap:
            self.walls = dict(snap["walls"])
            self.wall_edit = snap.get("wall_edit")
            if snap.get("wall_scale"):
                self.wall_scale = dict(snap["wall_scale"])
        if "stamps" in snap:
            self.stamps = []
            for sid, name, pxs in snap["stamps"]:
                s = Stamp(name, pxs.copy())
                s.id = sid
                self.stamps.append(s)
        self.splines = []
        for pid, name, pts, closed in snap.get("splines", []):
            p = Spline(name, pts, closed)
            p.id = pid
            self.splines.append(p)
        self.selections = []
        for sid, name, data in snap.get("selections", []):
            x = Selection(self.height, self.width, name, data.copy())
            x.id = sid
            self.selections.append(x)
        self.masks = []
        for mid, name, data in snap.get("masks", []):
            m = Mask(self.height, self.width, name, data.copy())
            m.id = mid
            self.masks.append(m)
        if snap.get("strokes") is not None:
            self.strokes = [{"id": k["id"], "layer": k["layer"],
                             "points": [list(pt) for pt in k["points"]],
                             "brush": dict(k["brush"]),
                             **({"rig": {"bones": list(k["rig"]["bones"]),
                                         "pins": list(k["rig"]["pins"]),
                                         "prev": [list(pt) for pt in k["rig"]["prev"]],
                                         "keys": dict(k["rig"].get("keys", {}))}}
                                if k.get("rig") else {})}
                            for k in snap["strokes"]]
        live = {l.id: l.pixels for l in self.layers}
        live_h = {l.id: getattr(l, "height_map", None) for l in self.layers}
        live_m = {l.id: getattr(l, "material_map", None) for l in self.layers}
        self.layers = []
        for rec in snap["layers"]:
            lid, name, vis, op, bl, msk, minv, px = rec[:8]
            hg = rec[8] if len(rec) > 8 else None
            gloss = rec[9] if len(rec) > 9 else 0.3
            pmedia = rec[10] if len(rec) > 10 else None
            if isinstance(px, tuple):          # only the touched rectangle was
                (rx0, ry0, rx1, ry1), sub = px # kept -- paste it back over the
                base = live.get(lid)           # pixels that are on screen now
                if base is None or base.shape[:2] != (self.height, self.width):
                    base = np.zeros((self.height, self.width, 4), np.float32)
                px = base.copy()
                px[ry0:ry1, rx0:rx1] = sub
            elif px is None:                   # untouched by that operation:
                px = live.get(lid)             # keep whatever is on screen now
                if px is None:                 # (layer vanished: start clean)
                    px = np.zeros((self.height, self.width, 4), np.float32)
            l = Layer(self.height, self.width, name, px.copy())
            l.id, l.visible, l.opacity, l.blend = lid, vis, op, bl
            l.mask, l.mask_invert = msk, minv
            if hg is False:
                l.height_map = None            # that state HAD no paint body
            elif isinstance(hg, tuple):
                (rx0, ry0, rx1, ry1), sub = hg
                base = live_h.get(lid)
                if base is None or base.shape != (self.height, self.width):
                    base = np.zeros((self.height, self.width), np.float32)
                l.height_map = base.copy()
                l.height_map[ry0:ry1, rx0:rx1] = sub
            elif hg is not None:
                l.height_map = hg.copy()
            else:
                hm = live_h.get(lid)
                l.height_map = None if hm is None else hm
            mg = rec[12] if len(rec) > 12 else None
            if mg is False:
                l.material_map = None          # that state HAD no material
            elif isinstance(mg, tuple):
                (rx0, ry0, rx1, ry1), sub = mg
                base = live_m.get(lid)
                if base is None or base.shape != (self.height, self.width, 3):
                    base = np.zeros((self.height, self.width, 3), np.float32)
                l.material_map = base.copy()
                l.material_map[ry0:ry1, rx0:rx1] = sub
            elif mg is not None:
                l.material_map = mg.copy()
            else:                              # untouched (or a pre-material
                mm = live_m.get(lid)           # snapshot): keep what is live
                l.material_map = None if mm is None else mm
            l.paint_gloss = gloss
            if pmedia:
                l.paint_media = pmedia
            xa = rec[11] if len(rec) > 11 else None
            if xa:
                for k, v in xa.items():
                    if k == "source":
                        if v is not None:
                            l.source = v
                    elif v is not None or k in ("bg", "place",
                                                "emissive_color",
                                                "curve_profile",
                                                "dome_profile"):
                        setattr(l, k, v)
            self.layers.append(l)

    def record(self, label="Edit", only=None, region=None):
        """`region` = (x0, y0, x1, y1) the operation cannot paint outside.

        A brush stroke covers a few hundred pixels but used to snapshot the
        whole 33 MB layer. Twenty-four of those is ~1 GB of retained undo
        history, and the resulting memory pressure -- not the copying, which
        measures 6 ms -- is what made the start of every stroke cost ~340 ms."""
        _MUT_REV[0] += 1
        self._sim_run = None              # any recorded edit ends a sim run
        self._undo.append((label, self._snapshot(only, region)))
        # Trim on MEMORY, not just on count. Twenty-four full snapshots of a
        # 4-layer 1920x1080 document is ~3 GB -- measured -- which is fatal on
        # a modest machine and was the likeliest cause of "it crashed while I
        # was messing about". A count alone cannot bound this because one entry
        # can be 130 MB or 30 KB depending on the document.
        while len(self._undo) > 24 or (
                len(self._undo) > 1 and self._undo_bytes() > self.UNDO_BUDGET):
            self._undo.pop(0)
        # Floor of one: a single snapshot can exceed the whole budget on a big
        # document, and keeping ONE undo is worth more than honouring the cap
        # exactly. Reported so the UI can warn instead of pretending.
        self.undo_over_budget = self._undo_bytes() > self.UNDO_BUDGET
        self._redo.clear()

    UNDO_BUDGET = 512 * 1024 * 1024      # bytes of pixel data kept for undo

    @staticmethod
    def _snap_bytes(snap):
        n = 0
        for t in snap.get("layers", ()):
            px = t[7]
            if px is None:
                continue
            n += (px[1].nbytes if isinstance(px, tuple) else px.nbytes)
        for m in snap.get("masks", ()):
            n += m[2].nbytes
        for x in snap.get("selections", ()):
            n += x[2].nbytes
        return n

    def _undo_bytes(self):
        """Size of the retained history. Cached per entry -- recomputing it by
        walking every snapshot on every record turned the trim into an O(n^2)
        stall (measured 54 s in a test that records 40 times)."""
        if not hasattr(self, "_undo_sizes"):
            self._undo_sizes = {}
        total = 0
        live = set()
        for _lbl, sn in self._undo:
            k = id(sn)
            live.add(k)
            if k not in self._undo_sizes:
                self._undo_sizes[k] = self._snap_bytes(sn)
            total += self._undo_sizes[k]
        for k in [k for k in self._undo_sizes if k not in live]:
            del self._undo_sizes[k]
        return total

    def undo_stats(self):
        """What the history actually costs, so the UI can be honest about it."""
        n = self._undo_bytes()
        return {"entries": len(self._undo), "bytes": n,
                "budget": self.UNDO_BUDGET,
                "over_budget": n > self.UNDO_BUDGET}

    def _bump(self):
        _MUT_REV[0] += 1

    def undo(self):
        self._sim_run = None
        _MUT_REV[0] += 1
        if not self._undo:
            return False
        label, snap = self._undo.pop()
        self._redo.append((label, self._snapshot()))
        self._restore(snap)
        return True

    def redo(self):
        self._sim_run = None
        _MUT_REV[0] += 1
        if not self._redo:
            return False
        label, snap = self._redo.pop()
        self._undo.append((label, self._snapshot()))
        self._restore(snap)
        return True

    # --- layer ops -------------------------------------------------------------------------------
    def layer(self, lid):
        for l in self.layers:
            if l.id == lid:
                return l
        raise KeyError(lid)

    PLACED_BUDGET = 256 * 1024 * 1024      # bytes of native source pixels kept

    def add_layer(self, name=None, pixels=None, record=True, placed=False,
                  below=None):
        """below=<layer id> inserts the new layer UNDER that one --
        found the hard way while dogfooding: a shadow painted after the
        apple landed ON TOP of it, because new layers only ever stacked
        highest.

        USE THE RETURN VALUE. With below=, the new layer is NOT last,
        so the old idiom layers[-1].id grabs the WRONG layer -- one
        script wiped its own leaves that way (edited + erased the last
        layer believing it was the fresh one)."""
        _MUT_REV[0] += 1
        if record:
            self.record("Add layer", only=[])
        src = None
        if pixels is not None:
            if pixels.shape[-1] == 3:
                pixels = np.concatenate([pixels, np.ones_like(pixels[..., :1])], -1)
            if placed and (pixels.shape[0] != self.height
                           or pixels.shape[1] != self.width):
                # PLACED: keep the file's own pixels beside the rendered layer.
                # Fitting an import to the canvas is lossy and permanent --
                # measured 42% of stripe contrast gone on a 1200x900 photo
                # squeezed into 400x300 and resized back. With the source kept,
                # the layer can be re-rendered from it at any later size.
                src = pixels
            pixels = _resize(pixels, self.height, self.width)
        l = Layer(self.height, self.width, name, pixels)
        if src is not None:
            l.source = src
            self._trim_placed()
        if below is not None:
            idx = next((i for i, x in enumerate(self.layers)
                        if x.id == below), None)
            if idx is None:
                self.layers.append(l)
            else:
                self.layers.insert(idx, l)
        else:
            self.layers.append(l)
        return l

    def _trim_placed(self):
        """Bounded, like every other retained copy in the document."""
        held = [l for l in self.layers if getattr(l, "source", None) is not None]
        total = sum(l.source.nbytes for l in held)
        while held and total > self.PLACED_BUDGET:
            victim = held.pop(0)          # oldest placement loses its source
            total -= victim.source.nbytes
            victim.source = None

    def flip_layer(self, lid, axis="x", record=True):
        """Mirror this layer along one axis (x = left/right, y =
        up/down) -- call twice or once per axis for both. Destructive
        and undoable; the impasto height map and a placed layer's
        retained source flip too, so replays and re-placements stay
        consistent with what is on screen."""
        self._locked_guard(lid)
        l = self.layer(lid)
        if record:
            self.record("Flip layer", only=[lid])
        ax = 1 if axis == "x" else 0
        l.pixels = np.flip(l.pixels, axis=ax).copy()
        hm = getattr(l, "height_map", None)
        if hm is not None:
            l.height_map = np.flip(hm, axis=ax).copy()
        mm = getattr(l, "material_map", None)
        if mm is not None:
            l.material_map = np.flip(mm, axis=ax).copy()
        src = getattr(l, "source", None)
        if src is not None:
            l.source = np.flip(src, axis=ax).copy()
        _MUT_REV[0] += 1
        return True

    def place_source(self, lid, x=None, y=None, scale=None, rot=None,
                     record=True):
        """Re-rasterise a PLACED layer from its original pixels with a
        transform: centre (x, y) in document coordinates -- anywhere,
        including outside the canvas -- uniform scale (1.0 = the source's
        own pixels), and rotation in degrees. Nothing is ever cropped
        away: the source stays whole, the canvas just shows what falls
        inside it, and moving or shrinking the placement later brings
        hidden regions back."""
        self._locked_guard(lid)
        l = self.layer(lid)
        src = getattr(l, "source", None)
        if src is None:
            return False
        pl = dict(getattr(l, "place", None)
                  or {"x": self.width / 2.0, "y": self.height / 2.0,
                      "scale": 1.0, "rot": 0.0})
        if x is not None:
            pl["x"] = float(x)
        if y is not None:
            pl["y"] = float(y)
        if scale is not None:
            pl["scale"] = max(float(scale), 0.02)
        if rot is not None:
            pl["rot"] = float(rot)
        if record:
            self.record("Place image", only=[lid])
        l.place = pl
        sh, sw = src.shape[:2]
        ys, xs = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        # inverse map: doc pixel -> source pixel
        dx = xs - pl["x"]
        dy = ys - pl["y"]
        th = np.deg2rad(-pl["rot"])
        rx = dx * np.cos(th) - dy * np.sin(th)
        ry = dx * np.sin(th) + dy * np.cos(th)
        sx = rx / pl["scale"] + sw / 2.0
        sy = ry / pl["scale"] + sh / 2.0
        inside = (sx >= 0) & (sx <= sw - 1) & (sy >= 0) & (sy <= sh - 1)
        x0 = np.clip(np.floor(sx), 0, sw - 2).astype(np.int32)
        y0 = np.clip(np.floor(sy), 0, sh - 2).astype(np.int32)
        fx = np.clip(sx - x0, 0, 1)[..., None]
        fy = np.clip(sy - y0, 0, 1)[..., None]
        s00 = src[y0, x0]
        s01 = src[y0, x0 + 1]
        s10 = src[y0 + 1, x0]
        s11 = src[y0 + 1, x0 + 1]
        out = (s00 * (1 - fx) * (1 - fy) + s01 * fx * (1 - fy)
               + s10 * (1 - fx) * fy + s11 * fx * fy).astype(np.float32)
        out[~inside] = 0.0
        l.pixels = out
        _MUT_REV[0] += 1
        return True

    def replace_from_source(self, lid):
        """Re-render a placed layer from its ORIGINAL pixels at the current
        size. After a resize this recovers detail that fitting to the old
        canvas had thrown away; there is nothing to recover if the layer was
        not placed, so it reports False rather than pretending."""
        l = self.layer(lid)
        src = getattr(l, "source", None)
        if src is None:
            return False
        self.record("Re-render image", only=[lid])
        l.pixels = _resize(src, self.height, self.width)
        _MUT_REV[0] += 1
        return True

    def remove_layer(self, lid):
        self.record("Remove layer")
        self.layers = [l for l in self.layers if l.id != lid]
        for g in self.groups:
            g["layers"] = [x for x in g["layers"] if x != lid]

    # --- groups -----------------------------------------------------------------
        if getattr(self, "lights", None):
            self.lights = [li for li in self.lights
                           if li.get("layer") != lid]
        if getattr(self, "fields", None):
            self.fields = [f for f in self.fields
                           if f.get("layer") != lid]
        for s, cur in getattr(self, "walls", {}).items():
            if cur == lid:
                self.walls[s] = None
                if self.wall_edit == s:
                    self.wall_edit = None
    def group(self, gid):
        for g in self.groups:
            if g["id"] == gid:
                return g
        raise KeyError(gid)

    def add_group(self, name=None, layers=()):
        self.record("Add group", only=[])
        g = {"id": f"G{self._gnext}", "name": name or f"Group {self._gnext}",
             "layers": [x for x in layers if any(l.id == x for l in self.layers)]}
        self._gnext += 1
        self.groups.append(g)
        return g

    def remove_group(self, gid):
        self.record("Remove group", only=[])
        self.groups = [g for g in self.groups if g["id"] != gid]

    def edit_group(self, gid, name=None, layers=None):
        _MUT_REV[0] += 1
        g = self.group(gid)
        if name is not None:
            g["name"] = name
        if layers is not None:
            g["layers"] = [x for x in layers if any(l.id == x for l in self.layers)]
        return g

    def group_composite(self, gid):
        g = self.group(gid)
        members = [l for l in self.layers if l.id in set(g["layers"])]
        return composite(members, self.height, self.width, self.mask_map())

    def move_layer(self, lid, to_index):
        self.record("Reorder layers", only=[])
        l = self.layer(lid)
        self.layers.remove(l)
        self.layers.insert(max(0, min(len(self.layers), int(to_index))), l)

    def merge_layers(self, ids):
        """Bake the given layers (in stack order, with blend/opacity/masks applied)
        into one layer at the lowest member's position; remove the rest."""
        members = [l for l in self.layers if l.id in set(ids)]
        if len(members) < 2:
            return members[0] if members else None
        self.record("Merge layers")
        comp = composite(members, self.height, self.width, self.mask_map())
        lowest = min(self.layers.index(l) for l in members)
        merged = Layer(self.height, self.width, members[0].name + " merged", comp)
        self.layers = [l for l in self.layers if l.id not in set(ids)]
        self.layers.insert(lowest, merged)
        return merged

    def merge_layer_down(self, lid):
        i = self.layers.index(self.layer(lid))
        if i == 0:
            return None
        return self.merge_layers([self.layers[i - 1].id, lid])

    def merge_visible_layers(self):
        return self.merge_layers([l.id for l in self.layers if l.visible])

    def edit_layer(self, lid, **props):
        _MUT_REV[0] += 1
        l = self.layer(lid)
        for k in ("name", "visible", "opacity", "blend", "mask_invert",
                  "alpha_lock", "clip", "thickness", "vol_kind", "vol_ior",
                  "vol_density", "absorbency", "emissive",
                  "emissive_color", "reflect", "dispersion",
                  "media_rate", "z_off",
                  "tilt_x", "tilt_y",
                  "curve", "dome", "field", "field_mode", "field_strength",
                  "curve_axis", "locked", "relief", "optical",
                  # which way is DOWN for this surface's wet paint
                  "gravity", "gravity_angle",
                  "media_res", "media_time"):
            if k in props and props[k] is not None:
                v = props[k]
                if k == "gravity":
                    # zero is MEANINGFUL here (flat on a table) rather
                    # than absent, so it must not be gated away
                    v = float(np.clip(float(v), 0.0, 1.0))
                elif k == "gravity_angle":
                    v = float(v) % 360.0
                elif k == "thickness":
                    v = max(float(v), 0.1)       # >= 0.01 mm, always
                elif k in ("tilt_x", "tilt_y"):
                    # past +/-90 the slab faces away and reads as a
                    # mirrored image -- use Flip for that instead
                    v = float(np.clip(float(v), -90.0, 90.0))
                setattr(l, k, v)
        for pk in ("curve_profile", "dome_profile"):
            # None = untouched (the route sends None for absent keys);
            # an explicit EMPTY list clears back to the legacy arc
            if pk in props and props[pk] is not None:
                pr = props[pk]
                if pr:
                    pr = [[float(np.clip(t, 0.0, 1.0)),
                           float(np.clip(v, -1.0, 1.0))]
                          for t, v in pr][:16]   # bounded, like everything
                    setattr(l, pk, pr)
                else:
                    setattr(l, pk, None)
        if "bg" in props:                        # None clears -> transparent
            b = props["bg"]
            if b:
                b = {"kind": str(b.get("kind", "color")),
                     "color": [float(c) for c in b.get("color",
                                                       [1, 1, 1, 1])][:4],
                     "tex": str(b.get("tex", "paper")),
                     "scale": float(b.get("scale", 3.0))}
            l.bg = b or None
        if "mask" in props:                      # None / "" detaches
            l.mask = props["mask"] or None

    # --- brushes ----------------------------------------------------------------
    def brush_by_id(self, bid):
        for b in self.brushes:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def add_brush(self, name=None, tip=None, spacing=0.25):
        self.record("Add brush", only=[])
        b = Brush(name, tip, spacing)
        self.brushes.append(b)
        return b

    def remove_brush(self, bid):
        b = self.brush_by_id(bid)
        if b.builtin:
            return
        self.record("Remove brush", only=[])
        self.brushes = [x for x in self.brushes if x.id != bid]

    # --- stamps / stickers ------------------------------------------------
    def stamp_by_id(self, sid):
        for s in self.stamps:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def make_stamp_from(self, lid, sel=None, name=None):
        """Capture a sticker from a layer: the alpha bounding box of its
        content, cut through the given selection if one is supplied (so a
        lasso becomes the sticker's outline)."""
        l = self.layer(lid)
        px = l.pixels
        a = px[..., 3]
        if sel is not None:
            m = np.clip(np.asarray(self.selection_to_mask(sel).data,
                                   np.float32), 0, 1)
            a = a * m
        ys, xs = np.nonzero(a > 0.02)
        if not ys.size:
            raise ValueError("nothing to capture there")
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cut = px[y0:y1, x0:x1].copy()
        if sel is not None:
            cut[..., 3] = a[y0:y1, x0:x1]
        self.record("Make stamp", only=[])
        s = Stamp(name, cut)
        self.stamps.append(s)
        _MUT_REV[0] += 1
        return s

    def remove_stamp(self, sid):
        self.stamp_by_id(sid)
        self.record("Remove stamp", only=[])
        self.stamps = [x for x in self.stamps if x.id != sid]
        _MUT_REV[0] += 1

    # --- environment lights ----------------------------------------------
    def add_light(self, kind="directional", color=(1.0, 1.0, 1.0),
                  intensity=1.0, azimuth=315.0, elevation=45.0,
                  x=None, y=None, z=60.0, aim_x=None, aim_y=None,
                  cone=30.0, soft=0.5, color2=(0.25, 0.22, 0.18),
                  layer=None, scale=1.0, shadows=True):
        """A light in the environment. Kinds: "view" (aligned with the
        viewer -- frontal, like a camera lamp), "directional" (casts
        ACROSS the canvas from a compass azimuth at an elevation, with
        real height-field shadows), "point" (hangs above the canvas at
        (x, y, z) with distance falloff). Colour channels may exceed 1
        for over-driven light. Multiple lights sum."""
        li = {"id": "LI%d" % self._lnext,
              "kind": kind, "color": [float(c) for c in color],
              "intensity": float(intensity),
              "azimuth": float(azimuth), "elevation": float(elevation),
              "x": float(x if x is not None else self.width / 2.0),
              "y": float(y if y is not None else self.height / 2.0),
              "z": float(z),
              "aim_x": float(aim_x if aim_x is not None
                             else self.width / 2.0),
              "aim_y": float(aim_y if aim_y is not None
                             else self.height / 2.0),
              "cone": float(cone), "soft": float(soft),
              "shadows": bool(shadows),
              "color2": [float(c) for c in color2],
              "layer": layer,           # a LAYER light lives inside one
              "scale": float(scale),    # physical size: broadens the pool
              "enabled": True}
        self._lnext += 1
        self.record("Add light", only=[])
        self.lights.append(li)
        _MUT_REV[0] += 1
        return li


    WALL_SIDES = ("front", "back", "left", "right")

    def assign_wall(self, side, lid):
        """Put an ordinary layer on one of the four perpendicular
        planes. The layer keeps everything it had -- strokes, thickness,
        volume, fields, lights -- it simply now stands off the canvas on
        that side. Assigning does not copy or convert anything, so the
        artist can pull it back to the canvas with clear_wall and lose
        nothing."""
        if side not in self.WALL_SIDES:
            raise ValueError("side must be one of %s" % (self.WALL_SIDES,))
        l = self.layer(lid)                    # raises if unknown
        self.record("Assign wall")
        for s, cur in self.walls.items():      # a layer stands on one wall
            if cur == lid and s != side:
                self.walls[s] = None
        self.walls[side] = lid
        l.wall = side
        _MUT_REV[0] += 1
        return dict(self.walls)

    def clear_wall(self, side):
        """Take the layer off that plane and give it back to the canvas."""
        if side not in self.WALL_SIDES:
            raise ValueError("side must be one of %s" % (self.WALL_SIDES,))
        lid = self.walls.get(side)
        self.record("Clear wall")
        self.walls[side] = None
        if lid:
            try:
                self.layer(lid).wall = None
            except KeyError:
                pass
        if self.wall_edit == side:
            self.wall_edit = None
        _MUT_REV[0] += 1
        return dict(self.walls)

    def edit_wall(self, side):
        """Open a wall for painting: while a side is being edited its
        layer shows on the canvas like any other, so every existing tool
        works on it unchanged. Pass None to close and let it stand back
        up on its plane."""
        if side is not None and side not in self.WALL_SIDES:
            raise ValueError("side must be one of %s" % (self.WALL_SIDES,))
        self.wall_edit = side
        _MUT_REV[0] += 1
        return side

    def stack_height(self):
        """The document's total depth: every optically active layer's
        thickness plus what its relief adds. This is the room's height
        in the same units the walls are scaled against, so a document
        of thin washes and one of thick slabs do not need the same
        wall settings to look right."""
        h = 0.0
        for l in self.layers:
            if not _optically_active(l):
                continue
            if getattr(l, "wall", None):
                continue                 # a wall is not part of the floor
            h += max(float(getattr(l, "thickness", 0.0)), 0.0) \
                * float(np.clip(getattr(l, "relief", 1.0), 0.0, 1.0))
        return float(h)

    def set_wall_scale(self, side, scale):
        """Vertical scale for one side of the room."""
        if side not in self.WALL_SIDES:
            raise ValueError("side must be one of %s" % (self.WALL_SIDES,))
        s = float(scale)
        if not (0.05 <= s <= 20.0):
            raise ValueError("scale must be between 0.05 and 20")
        self.record("Wall scale", only=[])
        if not hasattr(self, "wall_scale") or not self.wall_scale:
            self.wall_scale = {k: 1.0 for k in self.WALL_SIDES}
        self.wall_scale[side] = s
        _MUT_REV[0] += 1
        return dict(self.wall_scale)

    def wall_layers(self):
        """{side: layer} for the assigned planes, skipping empty slots."""
        out = {}
        for s in self.WALL_SIDES:
            lid = self.walls.get(s)
            if not lid:
                continue
            try:
                out[s] = self.layer(lid)
            except KeyError:
                pass
        return out

    def add_field(self, kind="point", layer=None, x=None, y=None,
                  radius=120.0, strength=1.0, angle=0.0):
        """A FORCE FIELD as a first-class object, parented to a layer
        like a light child: it rides the layer's visibility and dies
        with it. Kinds: "point" (radial push/pull -- strength sign
        chooses attract vs repel), "direct" (uniform push along angle
        degrees), "vortex" (swirl about the centre). Fields shape the
        layer's LIVING MEDIA each timeline step; several sum."""
        self.record("Add field", only=[])
        f = {"id": "F%d" % self._fnext,
             "kind": kind, "layer": layer,
             "x": float(x if x is not None else self.width / 2.0),
             "y": float(y if y is not None else self.height / 2.0),
             "radius": max(8.0, float(radius)),
             "strength": float(strength), "angle": float(angle)}
        self._fnext += 1
        self.fields.append(f)
        _MUT_REV[0] += 1
        return f

    def field_by_id(self, fid):
        for f in self.fields:
            if f["id"] == fid:
                return f
        return None

    def edit_field(self, fid, **kw):
        f = self.field_by_id(fid)
        if f is None:
            raise KeyError(fid)
        self.record("Edit field", only=[])
        for k, v in kw.items():
            if v is None or k not in ("kind", "layer", "x", "y",
                                      "radius", "strength", "angle"):
                continue
            f[k] = v if k in ("kind", "layer") else float(v)
        f["radius"] = max(8.0, float(f["radius"]))
        _MUT_REV[0] += 1
        return f

    def delete_field(self, fid):
        self.record("Remove field", only=[])
        n = len(self.fields)
        self.fields = [f for f in self.fields if f["id"] != fid]
        _MUT_REV[0] += 1
        return len(self.fields) < n

    def edit_light(self, lid, **kw):
        for li in self.lights:
            if li["id"] == lid:
                self.record("Edit light", only=[])
                for k, v in kw.items():
                    if k in ("kind",):
                        li[k] = str(v)
                    elif k == "color":
                        li[k] = [float(c) for c in v]
                    elif k in ("enabled", "shadows"):
                        li[k] = bool(v)
                    elif k == "color2":
                        li[k] = [float(c) for c in v]
                    elif k == "layer":
                        li[k] = v
                    elif k in ("aim_x", "aim_y"):
                        # a light always TARGETS a point inside the
                        # document bounds -- it can orbit anywhere, but
                        # it never shines off into nothing
                        lim = self.width if k == "aim_x" else self.height
                        li[k] = float(np.clip(float(v), 0.0, lim))
                    elif k in ("intensity", "azimuth", "elevation",
                               "x", "y", "z", "cone", "soft", "scale"):
                        li[k] = float(v)
                _MUT_REV[0] += 1
                return li
        raise KeyError(lid)

    def remove_light(self, lid):
        if not any(li["id"] == lid for li in self.lights):
            raise KeyError(lid)
        self.record("Remove light", only=[])
        self.lights = [li for li in self.lights if li["id"] != lid]
        _MUT_REV[0] += 1

    def light_preset(self, name):
        """One-click lighting RIGS for the environment: "sun" (warm key
        + blue sky dome), "studio" (big soft key, dome fill, cool rim),
        "three_point" (key spot, fill point, rim directional), "dome"
        (sky/ground hemisphere only -- the HDRI analog for 2.5D). Replaces
        the current lights, undoably."""
        self.record("Light preset", only=[])
        w, h = self.width, self.height
        self.lights = []
        def _mk(**kw):
            kw.setdefault("kind", "directional")
            li = self.add_light(**kw)
            return li
        if name == "sun":
            _mk(kind="directional", color=(1.0, 0.92, 0.75), intensity=1.15,
                azimuth=305, elevation=38)
            _mk(kind="dome", color=(0.45, 0.55, 0.8), intensity=0.5,
                color2=(0.30, 0.26, 0.22))
        elif name == "studio":
            _mk(kind="directional", color=(1.0, 0.98, 0.95), intensity=1.0,
                azimuth=320, elevation=55)
            _mk(kind="dome", color=(0.55, 0.55, 0.6), intensity=0.45,
                color2=(0.35, 0.33, 0.3))
            _mk(kind="directional", color=(0.7, 0.8, 1.0), intensity=0.55,
                azimuth=130, elevation=25)
        elif name == "three_point":
            _mk(kind="spot", color=(1.0, 0.97, 0.9), intensity=1.5,
                x=w * 0.22, y=h * 0.18, z=max(w, h) * 0.5,
                aim_x=w * 0.5, aim_y=h * 0.55, cone=38, soft=0.6)
            _mk(kind="point", color=(0.75, 0.8, 0.95), intensity=0.6,
                x=w * 0.85, y=h * 0.6, z=max(w, h) * 0.35)
            _mk(kind="directional", color=(0.9, 0.85, 1.0), intensity=0.5,
                azimuth=105, elevation=20)
        elif name == "dome":
            _mk(kind="dome", color=(0.65, 0.72, 0.9), intensity=1.0,
                color2=(0.4, 0.35, 0.3))
        else:
            raise KeyError(name)
        return self.lights

    # --- the timeline: keyframed properties + a global playhead --------
    ANIMATABLE = {"layer": ("opacity", "z_off", "tilt_x", "tilt_y",
                            "thickness", "emissive", "reflect",
                            "dispersion", "media_rate"),
                  "light": ("intensity", "azimuth", "elevation",
                            "x", "y", "z", "cone"),
                  "node": ()}       # any numeric param, checked live

    def _track_target(self, kind, tid):
        if kind == "node":
            g = getattr(self, "graph_ref", None)
            if g is None or tid not in g.nodes:
                raise KeyError(tid)
            return g.nodes[tid]
        if kind == "layer":
            return self.layer(tid)
        if kind == "light":
            for li in self.lights:
                if li["id"] == tid:
                    return li
            raise KeyError(tid)
        raise KeyError(kind)

    def _prop_get(self, kind, tid, prop):
        tgt = self._track_target(kind, tid)
        if kind == "node":
            return float((tgt.get("params") or {}).get(prop, 0.0))
        if kind == "light":
            return float(tgt[prop])
        if prop == "media_rate":
            return float(getattr(tgt, "media_rate", 1.0))
        return float(getattr(tgt, prop, 0.0))

    def _prop_set(self, kind, tid, prop, v):
        tgt = self._track_target(kind, tid)
        if kind == "node":
            tgt.setdefault("params", {})[prop] = float(v)
        elif kind == "light":
            tgt[prop] = float(v)
        else:
            setattr(tgt, prop, float(v))

    def set_key(self, kind, tid, prop, t=None, v=None):
        """Set a KEYFRAME: pin this property to a value at a frame (both
        default to right now / the live value). Re-keying an existing
        frame moves its value. Undoable."""
        if kind == "node":
            n = self._track_target(kind, tid)     # raises if absent
            od = OPS.get(n.get("type"), {})
            kinds = {pp["name"]: pp.get("kind", "float")
                     for pp in od.get("params", [])}
            if kinds.get(prop) not in ("float", "int"):
                raise KeyError(prop)
        elif prop not in self.ANIMATABLE.get(kind, ()):
            raise KeyError(prop)
        t = float(self.frame if t is None else t)
        v = float(self._prop_get(kind, tid, prop) if v is None else v)
        self.record("Set key", only=[])
        key = "%s:%s:%s" % (kind, tid, prop)
        ks = [k for k in self.tracks.get(key, []) if abs(k[0] - t) > 1e-6]
        ks.append([t, v])
        ks.sort(key=lambda k: k[0])
        self.tracks[key] = ks
        _MUT_REV[0] += 1
        return ks

    def del_key(self, kind, tid, prop, t=None):
        """Delete the keyframe at a frame (default: the playhead).
        Deleting the last key removes the track and the property stays at
        its live value. Undoable."""
        t = float(self.frame if t is None else t)
        key = "%s:%s:%s" % (kind, tid, prop)
        ks = [k for k in self.tracks.get(key, [])
              if abs(k[0] - t) > 1e-6]
        self.record("Delete key", only=[])
        if ks:
            self.tracks[key] = ks
        else:
            self.tracks.pop(key, None)
        _MUT_REV[0] += 1

    def track_eval(self, key, t):
        """A track's value at time t: linear between keys, held flat
        before the first and after the last."""
        ks = self.tracks.get(key)
        if not ks:
            return None
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for i in range(1, len(ks)):
            if t <= ks[i][0]:
                a, b = ks[i - 1], ks[i]
                f = (t - a[0]) / max(b[0] - a[0], 1e-6)
                return a[1] * (1 - f) + b[1] * f
        return ks[-1][1]

    def set_frame(self, t, record=False):
        """Move the PLAYHEAD: every keyframed property takes its
        interpolated value, and living media (ink/smoke/fire) advance by
        the elapsed frames times their (keyable) media_rate -- so a
        medium whose rate is keyed 0 until frame 30 simply waits, then
        starts. Media are forward-only: scrubbing backward re-poses the
        keyed properties exactly but cannot un-simulate fluid."""
        t = float(np.clip(t, self.frame_range[0], self.frame_range[1]))
        dt = t - float(getattr(self, "frame", 0.0))
        self.frame = t
        # bump FIRST so anything below that patches the composite
        # cache leaves it marked current (see the note at the end)
        _MUT_REV[0] += 1
        for key in list(self.tracks.keys()):
            kind, tid, prop = key.split(":", 2)
            try:
                v = self.track_eval(key, t)
                if v is not None:
                    self._prop_set(kind, tid, prop, v)
            except KeyError:
                continue                      # target deleted; track idles
        if dt != 0:
            for l in self.layers:
                if getattr(l, "vol_kind", "none") in _MEDIA_KINDS:
                    if str(getattr(l, "media_time", "timeline")) == "live":
                        # LIVE media are timeline-INDEPENDENT: they cook
                        # on their own clock and are never restored to a
                        # past state. Some sources genuinely cannot be
                        # rewound -- a live video feed is the honest
                        # example -- and pretending otherwise would be a
                        # lie the rest of the timeline machinery has to
                        # keep. Scrubbing simply does not touch them.
                        continue
                    rate = float(getattr(l, "media_rate", 1.0))
                    # THICKNESS scales time: a deep dish holds more
                    # fluid, so the same elapsed frames move it further
                    # -- this used to live in the per-stroke burst
                    # (8 + thickness); with the playhead as the only
                    # clock, the dial rides the frame advance instead.
                    tmul = float(np.clip(
                        getattr(l, "thickness", 8.0) / 8.0, 0.25, 3.0))
                    # ONE TIME MODEL, both directions: media time is
                    # FRACTIONAL STEPS integrated along the playhead's
                    # visited path, recorded as (frame, fsteps) marks.
                    # The target for any t extends from the last mark
                    # at or before t under the CURRENT rate, so rate
                    # edits (including media_rate 0 = freeze in place)
                    # start a new segment instead of rewriting
                    # history; fractional nudges accumulate without
                    # loss (the old per-dt rounding lost 85% of the
                    # medium's time under a slow drag); and rewind
                    # restores the recorded past byte-identically.
                    marks = getattr(l, "_media_marks", None)
                    if marks is None:
                        continue            # nothing injected yet
                    i = len(marks) - 1
                    while i > 0 and marks[i][0] > t + 1e-9:
                        i -= 1
                    f0, s0 = marks[i]
                    target_f = max(0.0, s0 + max(0.0, t - f0)
                                   * rate * 0.8 * tmul)
                    tgt = int(round(target_f))
                    cur = getattr(l, "_media_cache_at", None)
                    if cur is not None and tgt == cur:
                        # time did not move for this layer: touch
                        # NOTHING (a restore re-renders pixels from
                        # the slab, which is not byte-equal to a
                        # freshly painted stamp)
                        del marks[i + 1:]
                        if abs(marks[i][0] - t) < 1e-9:
                            marks[i] = (t, target_f)
                        else:
                            marks.append((t, target_f))
                        continue
                    _media_restore_to_step(self, l, tgt)
                    at = getattr(l, "_media_cache_at", None)
                    if at is None:
                        continue
                    if tgt > at:
                        _media_slab_step(self, l, min(tgt - at, 64))
                        _media_cache_put(self, l, tgt)
                    # record the mark (replace same-frame, drop future)
                    del marks[i + 1:]
                    if abs(marks[i][0] - t) < 1e-9:
                        marks[i] = (t, target_f)
                    else:
                        marks.append((t, target_f))
                    if len(marks) > 600:
                        del marks[1:len(marks) - 500]
        # NOTE the bump is at the TOP of this method, not here. Bumping
        # after the media renders left the composite cache exactly one
        # revision stale at serve time, so every playback frame paid a
        # full-canvas re-composite even though the media path had just
        # patched the changed window correctly.
        return t

    def place_stamp(self, lid, sid, x, y, scale=1.0, rotation=0.0,
                    opacity=1.0, record=True):
        """Press the sticker onto a layer, centred at (x, y): scaled,
        rotated (degrees), straight-alpha OVER. Alpha-locked layers take
        the colour but keep their transparency, like the brush."""
        l = self.layer(lid)
        s = self.stamp_by_id(sid)
        sh, sw = s.pixels.shape[:2]
        sc = max(float(scale), 0.02)
        rad = np.deg2rad(float(rotation))
        ca, sa = np.cos(rad), np.sin(rad)
        # destination bbox of the transformed sticker
        hw, hh = sw * sc / 2.0, sh * sc / 2.0
        ext = np.abs([hw * ca, hh * sa]).sum(), np.abs([hw * sa, hh * ca]).sum()
        x0 = int(max(0, np.floor(x - ext[0]))); x1 = int(min(self.width, np.ceil(x + ext[0]) + 1))
        y0 = int(max(0, np.floor(y - ext[1]))); y1 = int(min(self.height, np.ceil(y + ext[1]) + 1))
        if x1 <= x0 or y1 <= y0:
            return
        if record:
            self.record("Stamp", only=[lid])
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - x, yy - y
        u_ = (dx * ca + dy * sa) / sc + sw / 2.0     # inverse rotate+scale
        v_ = (-dx * sa + dy * ca) / sc + sh / 2.0
        inside = (u_ >= 0) & (u_ <= sw - 1) & (v_ >= 0) & (v_ <= sh - 1)
        u_c = np.clip(u_, 0, sw - 1.001); v_c = np.clip(v_, 0, sh - 1.001)
        iu, iv = u_c.astype(np.int32), v_c.astype(np.int32)
        fu = (u_c - iu)[..., None]; fv = (v_c - iv)[..., None]
        iu1 = np.minimum(iu + 1, sw - 1); iv1 = np.minimum(iv + 1, sh - 1)
        sp = s.pixels
        samp = (sp[iv, iu] * (1 - fu) * (1 - fv) + sp[iv, iu1] * fu * (1 - fv)
                + sp[iv1, iu] * (1 - fu) * fv + sp[iv1, iu1] * fu * fv)
        a_s = (samp[..., 3:4] * float(np.clip(opacity, 0, 1))
               * inside[..., None])
        win = l.pixels[y0:y1, x0:x1]
        a_b = win[..., 3:4]
        if getattr(l, "alpha_lock", False):
            eff = a_s * a_b
            out_a = a_b
            num = samp[..., :3] * eff + win[..., :3] * a_b * (1 - eff)
            win[..., :3] = np.where(out_a > 1e-6,
                                    num / np.maximum(out_a, 1e-6),
                                    win[..., :3])
        else:
            out_a = np.clip(a_s + a_b * (1 - a_s), 0, 1)
            num = samp[..., :3] * a_s + win[..., :3] * a_b * (1 - a_s)
            win[..., :3] = np.where(out_a > 1e-6,
                                    num / np.maximum(out_a, 1e-6),
                                    win[..., :3])
            win[..., 3] = out_a[..., 0]
        _MUT_REV[0] += 1

    def edit_brush(self, bid, name=None, spacing=None, follow=None,
                   j_angle=None, j_size=None, j_scatter=None):
        _MUT_REV[0] += 1
        b = self.brush_by_id(bid)
        if name is not None:
            b.name = name
        if spacing is not None:
            b.spacing = float(spacing)
        if follow is not None:
            b.follow = bool(follow)
        for k, v in (("j_angle", j_angle), ("j_size", j_size), ("j_scatter", j_scatter)):
            if v is not None:
                setattr(b, k, float(v))
        return b

    # --- selections -------------------------------------------------------------
    def keep_selection(self, sid, name=None):
        """Promote the working selection into the saved list."""
        sel = self.selection_by_id(sid)
        if sel in self.selections:
            if name:
                sel.name = name
            return sel
        sel.name = name or sel.name or "Selection %d" % (len(self.selections) + 1)
        self.selections.append(sel)
        if getattr(self, "_scratch_sel", None) is sel:
            self._scratch_sel = None      # it is saved now; start a fresh scratch
        _MUT_REV[0] += 1
        return sel

    def selection_by_id(self, sid):
        for x in self.selections:
            if x.id == sid:
                return x
        # the unsaved WORKING selection is a first-class selection in every way
        # except that it is not in the saved list -- painting, cropping and
        # everything else must still find it by id
        sc = getattr(self, "_scratch_sel", None)
        if sc is not None and sc.id == sid:
            return sc
        raise KeyError(sid)

    def all_selections(self):
        """Saved selections plus the working one, which the UI lists apart."""
        sc = getattr(self, "_scratch_sel", None)
        return list(self.selections) + ([sc] if sc is not None else [])

    def gate_by_id(self, sid):
        """A mask OR a selection, by id -- whatever gates the brush/composite."""
        try:
            return self.mask_by_id(sid)
        except KeyError:
            return self.selection_by_id(sid)

    def _tool_field(self, tool, prm):
        """Build the raw (H, W) selection field for one tool invocation."""
        h, w = self.height, self.width
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        if tool == "rect":
            x0, y0 = min(prm["x0"], prm["x1"]), min(prm["y0"], prm["y1"])
            x1, y1 = max(prm["x0"], prm["x1"]), max(prm["y0"], prm["y1"])
            return ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).astype(np.float32)
        if tool == "ellipse":
            cx = (prm["x0"] + prm["x1"]) / 2.0; cy = (prm["y0"] + prm["y1"]) / 2.0
            rx = max(abs(prm["x1"] - prm["x0"]) / 2.0, 1e-3)
            ry = max(abs(prm["y1"] - prm["y0"]) / 2.0, 1e-3)
            return ((((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1).astype(np.float32)
        if tool == "alpha":
            # the layer's own coverage as a selection -- what Ctrl+clicking a
            # layer thumbnail does in every other editor
            return np.clip(np.asarray(self.layer(prm["layer"]).pixels[..., 3],
                                      np.float32), 0, 1)
        comp = self.composite()
        rgb = comp[..., :3] * comp[..., 3:4] + 1.0 * (1 - comp[..., 3:4])
        px, py = int(prm["x"]), int(prm["y"])
        px = np.clip(px, 0, w - 1); py = np.clip(py, 0, h - 1)
        if tool == "color":
            tol = float(prm.get("tolerance", 0.15))
            d = np.sqrt(((rgb - rgb[py, px]) ** 2).sum(-1))
            return (d <= tol * np.sqrt(3)).astype(np.float32)
        if tool == "brightness":
            tol = float(prm.get("tolerance", 0.15))
            lum = rgb.mean(-1)
            return (np.abs(lum - lum[py, px]) <= tol).astype(np.float32)
        if tool == "object":
            # leCore >= 0.2.2 bounds its own working size and returns full-res masks
            segs = _segment_compat(np.clip(rgb, 0, 1), k=int(prm.get("k", 6)))
            spx, spy = px, py
            for sg in segs:
                msk = np.asarray(sg["mask"] if isinstance(sg, dict) and "mask" in sg else sg)
                if msk.dtype != bool:
                    msk = msk > 0.5
                if msk.shape != (h, w):                       # defensive conform
                    msk = _resize(msk.astype(np.float32), h, w) > 0.5
                if msk[min(spy, h - 1), min(spx, w - 1)]:
                    return msk.astype(np.float32)
            return np.zeros((h, w), np.float32)
        raise ValueError(f"unknown selection tool {tool!r}")

    def select(self, tool, prm, mode="new", target=None, name=None, feather=0.0):
        """Run a selection tool and compose it into a stored Selection.
        mode: new | add | subtract | intersect. Returns the Selection."""
        self.record("Select", only=[])
        field = self._tool_field(tool, prm)
        if feather and feather > 0:
            field = np.clip(_gauss_blur(field, float(feather)), 0, 1)
        sel = None
        if target:
            try:
                sel = self.selection_by_id(target)
            except KeyError:
                sel = None
        if sel is None or mode == "new":
            if sel is None:
                sel = Selection(self.height, self.width, name)
                # Remember the SHAPE for geometric tools. A rect or ellipse is
                # geometry, not pixels: keeping it means a resize can
                # re-rasterise it exactly instead of resampling a mask and
                # gaining a soft edge (measured 6348 partially-covered pixels
                # on a 3x upscale, against zero for a native selection).
                if tool in ("rect", "ellipse") and not feather:
                    sel.shape = {"tool": tool, "params": dict(prm),
                                 "w": self.width, "h": self.height}
                if name:
                    self.selections.append(sel)   # named = the user asked to keep it
                else:
                    # Otherwise this is a WORKING selection. Most selections are
                    # momentary -- drag a marquee, paint inside it, move on --
                    # and appending every one filled the saved list with junk
                    # nobody named or wanted. One scratch slot is reused until
                    # the user chooses to keep it.
                    self._scratch_sel = sel
            sel.data = field
            return sel
        if mode == "add":
            sel.data = np.maximum(sel.data, field)
        elif mode == "subtract":
            sel.data = np.clip(sel.data - field, 0, 1)
        elif mode == "intersect":
            sel.data = np.minimum(sel.data, field)
        return sel

    def remove_selection(self, sid):
        self.record("Remove selection", only=[])
        self.selections = [x for x in self.selections if x.id != sid]

    def edit_selection(self, sid, name=None):
        _MUT_REV[0] += 1
        x = self.selection_by_id(sid)
        if name is not None:
            x.name = name
        return x

    def modify_selection(self, sid, op, amount):
        """expand | contract | feather the stored selection in place."""
        self.record("Modify selection", only=[])
        x = self.selection_by_id(sid)
        amount = float(amount)
        if op in ("expand", "grow"):
            x.data = _maxfilter(x.data, int(round(amount)))
        elif op in ("contract", "shrink"):
            x.data = _minfilter(x.data, int(round(amount)))
        elif op == "feather":
            x.data = np.clip(_gauss_blur(x.data, amount), 0, 1)
        else:
            raise ValueError(f"unknown selection op {op!r}")
        return x

    def move_selection(self, sid, index):
        self.record("Reorder selections", only=[])
        x = self.selection_by_id(sid)
        self.selections.remove(x)
        self.selections.insert(max(0, min(len(self.selections), int(index))), x)

    def merge_selections(self, ids=None):
        """Union the given selections (or ALL) into the first; remove the rest."""
        items = ([self.selection_by_id(i) for i in ids] if ids
                 else list(self.selections))
        if len(items) < 2:
            return items[0] if items else None
        self.record("Merge selections", only=[])
        base = items[0]
        for x in items[1:]:
            base.data = np.maximum(base.data, x.data)
            self.selections.remove(x)
        return base

    def selection_to_mask(self, sid, name=None):
        """Freeze a selection into a document Mask.

        A mask made from a SHAPED selection inherits that shape, so a resize
        re-rasterises it exactly rather than resampling -- the same reasoning as
        vector selections, and masks matter more because they gate a layer for
        the life of the document."""
        x = self.selection_by_id(sid)
        m = self.add_mask(name or x.name + " mask", x.data.copy())
        shp = getattr(x, "shape", None)
        if shp:
            m.shape = dict(shp)
        return m

    # --- transforms / document geometry -----------------------------------------
    def transform(self, kind, oid, sx=1.0, sy=1.0, deg=0.0, dx=0.0, dy=0.0,
                  layer=None):
        """Scale / rotate / translate a layer, mask, or selection in place.
        Rotation and scale pivot about the CONTENT's bounding-box centre (with
        the selection auto-shrink when `layer` is given) -- exactly the box the
        transform tool draws, so what you previewed is what you get."""
        if kind == "strokes":
            # route to the path-space transform: strokes are vectors, so they
            # re-render sharp instead of being resampled like pixels
            b = self.content_bbox("strokes", oid)
            return self.transform_strokes(
                [s for s in str(oid).split(",") if s], sx=sx, sy=sy, deg=deg,
                dx=dx, dy=dy, cx=(b[0] + b[2]) / 2.0, cy=(b[1] + b[3]) / 2.0)
        self.record("Transform " + kind)
        b = self.content_bbox(kind, oid, layer=layer)
        pivot = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        if kind == "layer":
            l = self.layer(oid)
            l.pixels = _affine(l.pixels, sx, sy, deg, dx, dy, pivot=pivot)
            if getattr(l, "height_map", None) is not None:
                # ridges travel with their pigment, or the relief light shades
                # the moved image with the OLD topography -- stale-height
                # glitches were visible the moment transform met impasto
                l.height_map = _affine(l.height_map, sx, sy, deg, dx, dy,
                                       pivot=pivot)
            if getattr(l, "material_map", None) is not None:
                # the stuff travels with its pigment for the same reason
                l.material_map = _mat_resample(
                    l.material_map,
                    lambda a: _affine(a, sx, sy, deg, dx, dy, pivot=pivot))
        elif kind == "mask":
            m = self.mask_by_id(oid)
            m.data = _affine(m.data, sx, sy, deg, dx, dy, pivot=pivot)
        elif kind == "selection":
            x = self.selection_by_id(oid)
            x.data = _affine(x.data, sx, sy, deg, dx, dy, pivot=pivot)
        else:
            raise ValueError(f"unknown transform target {kind!r}")

    def resize(self, width, height, mode="resample"):
        """Change the document size. resample: everything scales with the canvas.
        canvas: content keeps its pixel size, centred (crop or pad)."""
        self.record("Resize document")
        width, height = int(width), int(height)
        ow, oh = self.width, self.height
        if mode == "resample":
            fx, fy = width / ow, height / oh
            for l in self.layers:
                l.pixels = _resize(l.pixels, height, width)
                if getattr(l, "height_map", None) is not None:
                    l.height_map = _resize(l.height_map[..., None],
                                           height, width)[..., 0]
                if getattr(l, "material_map", None) is not None:
                    l.material_map = _mat_resample(
                        l.material_map, lambda a: _resize(a, height, width))
            for m in self.masks:
                shp = getattr(m, "shape", None)
                if shp and shp.get("w") and shp.get("h"):
                    kx, ky = width / float(shp["w"]), height / float(shp["h"])
                    pr = dict(shp["params"])
                    for a in ("x0", "x1"):
                        if a in pr:
                            pr[a] = float(pr[a]) * kx
                    for a in ("y0", "y1"):
                        if a in pr:
                            pr[a] = float(pr[a]) * ky
                    self.width, self.height = width, height
                    try:
                        m.data = self._tool_field(shp["tool"], pr)
                        m.shape = {"tool": shp["tool"], "params": pr,
                                   "w": width, "h": height}
                        continue
                    except Exception:
                        pass
                    finally:
                        self.width, self.height = ow, oh
                m.data = _resize(m.data, height, width)
            for x in self.all_selections():
                shp = getattr(x, "shape", None)
                if shp and shp.get("w") and shp.get("h"):
                    # exact: scale the GEOMETRY and redraw at the new size
                    kx, ky = width / float(shp["w"]), height / float(shp["h"])
                    pr = dict(shp["params"])
                    for a in ("x0", "x1"):
                        if a in pr:
                            pr[a] = float(pr[a]) * kx
                    for a in ("y0", "y1"):
                        if a in pr:
                            pr[a] = float(pr[a]) * ky
                    self.width, self.height = width, height
                    try:
                        x.data = self._tool_field(shp["tool"], pr)
                        x.shape = {"tool": shp["tool"], "params": pr,
                                   "w": width, "h": height}
                        continue
                    except Exception:
                        pass
                    finally:
                        self.width, self.height = ow, oh
                x.data = _resize(x.data, height, width)
            for p in self.splines:
                for q in p.points:
                    q["x"] *= fx; q["y"] *= fy
                    q["hx"] = q.get("hx", 0) * fx; q["hy"] = q.get("hy", 0) * fy
            # Recorded STROKE paths are document geometry too. Splines were
            # already scaled here but strokes were not, so after a resize every
            # stroke pointed at where it used to be: nudge and the rig acted on
            # stale coordinates, and replay_is_faithful CRASHED comparing a
            # resampled layer against an old-size base.
            fs = (fx + fy) * 0.5           # brush radius has one scale, not two
            for k in self.strokes:
                for q in k["points"]:
                    q[0] *= fx; q[1] *= fy
                    if len(q) > 2:
                        pass               # width is a multiplier: scale-free
                b = k.get("brush") or {}
                if "radius" in b:
                    b["radius"] = float(b["radius"]) * fs
                rg = k.get("rig")
                if rg:
                    rg["bones"] = [bl * fs for bl in rg["bones"]]
                    rg["prev"] = [[q[0] * fx, q[1] * fy] for q in rg["prev"]]
                    for t, pose in list((rg.get("keys") or {}).items()):
                        rg["keys"][t] = [[q[0] * fx, q[1] * fy] for q in pose]
            # replay bases are pixel data: resample or they no longer match
            for _lid, base in list(getattr(self, "_replay_base", {}).items()):
                self._replay_base[_lid] = _resize(base, height, width)
        else:                                     # canvas: centre crop / pad
            oy, ox = (height - oh) // 2, (width - ow) // 2
            def fit(a, fill=0.0):
                out = np.full((height, width) + a.shape[2:], fill, np.float32)
                sy0, dy0 = max(0, -oy), max(0, oy)
                sx0, dx0 = max(0, -ox), max(0, ox)
                hh = min(oh - sy0, height - dy0); ww = min(ow - sx0, width - dx0)
                if hh > 0 and ww > 0:
                    out[dy0:dy0 + hh, dx0:dx0 + ww] = a[sy0:sy0 + hh, sx0:sx0 + ww]
                return out
            for l in self.layers:
                l.pixels = fit(l.pixels)
                if getattr(l, "height_map", None) is not None:
                    l.height_map = fit(l.height_map)
                if getattr(l, "material_map", None) is not None:
                    l.material_map = fit(l.material_map)
            for m in self.masks:
                m.data = fit(m.data)
                m.shape = None  # content was re-framed, not rescaled
            for x in self.all_selections():
                x.data = fit(x.data)
                x.shape = None
            for p in self.splines:
                for q in p.points:
                    q["x"] += ox; q["y"] += oy
        self.width, self.height = width, height

    def crop(self, x0, y0, x1, y1):
        """Crop the document to a pixel rectangle (all layers/masks/selections)."""
        x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self.width, x1); y1 = min(self.height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("crop region too small")
        self.record("Crop")
        for l in self.layers:
            l.pixels = l.pixels[y0:y1, x0:x1].copy()
            if getattr(l, "height_map", None) is not None:
                l.height_map = l.height_map[y0:y1, x0:x1].copy()
            if getattr(l, "material_map", None) is not None:
                l.material_map = l.material_map[y0:y1, x0:x1].copy()
        for m in self.masks:
            m.data = m.data[y0:y1, x0:x1].copy()
            m.shape = None      # the stored geometry described the OLD frame
        for x in self.all_selections():
            x.data = x.data[y0:y1, x0:x1].copy()
            x.shape = None
        for p in self.splines:
            for q in p.points:
                q["x"] -= x0; q["y"] -= y0
        self.width, self.height = x1 - x0, y1 - y0

    ORIENT_OPS = ("rot90", "rot270", "rot180", "fliph", "flipv")

    def reorient(self, op):
        """Rotate or flip the whole document (Image > Rotate / Flip).

        Lossless: these are pure array reorderings, not resampling, so a
        90 + 90 + 90 + 90 round trip is bit-identical. Everything the document
        owns has to move together -- layers, masks, selections and spline
        control points -- or a rotate would silently desynchronise a mask from
        its layer."""
        if op not in self.ORIENT_OPS:
            raise ValueError("unknown orientation %r" % (op,))
        self.record("Rotate" if op.startswith("rot") else "Flip")
        w, h = self.width, self.height

        def move(a):
            if op == "rot90":                       # clockwise
                return np.rot90(a, k=-1, axes=(0, 1)).copy()
            if op == "rot270":
                return np.rot90(a, k=1, axes=(0, 1)).copy()
            if op == "rot180":
                return np.rot90(a, k=2, axes=(0, 1)).copy()
            if op == "fliph":
                return np.flip(a, axis=1).copy()
            return np.flip(a, axis=0).copy()        # flipv

        for l in self.layers:
            l.pixels = move(l.pixels)
        for m in self.masks:
            m.data = move(m.data)
            m.shape = None      # rotate/flip: the stored params no longer apply
        for x in self.all_selections():
            x.data = move(x.data)
            x.shape = None
        for p in self.splines:
            for q in p.points:
                px, py = q["x"], q["y"]
                if op == "rot90":
                    q["x"], q["y"] = (h - 1) - py, px
                elif op == "rot270":
                    q["x"], q["y"] = py, (w - 1) - px
                elif op == "rot180":
                    q["x"], q["y"] = (w - 1) - px, (h - 1) - py
                elif op == "fliph":
                    q["x"] = (w - 1) - px
                else:
                    q["y"] = (h - 1) - py
                for hk, hv in (("hx", "hy"),):       # bezier handles rotate too
                    if hk in q or hv in q:
                        dx, dy = q.get("hx", 0.0), q.get("hy", 0.0)
                        if op == "rot90":
                            q["hx"], q["hy"] = -dy, dx
                        elif op == "rot270":
                            q["hx"], q["hy"] = dy, -dx
                        elif op == "rot180":
                            q["hx"], q["hy"] = -dx, -dy
                        elif op == "fliph":
                            q["hx"] = -dx
                        else:
                            q["hy"] = -dy
        if op in ("rot90", "rot270"):
            self.width, self.height = h, w

    def fill_layer(self, lid, content, record=True, respect_alpha=False):
        """Wash the WHOLE layer with generated content in one call --
        {"kind": "solid", "color": [r,g,b,(a)]},
        {"kind": "gradient", "from": [...], "to": [...], "angle": deg}, or
        {"kind": "radial", "inner": [...], "outer": [...],
         "cx"?, "cy"?, "radius"?}.
        Dogfooding friction: a background vignette took ~40 overlapping
        soft stamps because there was no field fill. respect_alpha=True
        recolours only where the layer already has pixels (a one-call
        glaze). Locked layers refuse; undoable."""
        self._locked_guard(lid)
        l = self.layer(lid)
        if record:
            self.record("Fill layer", only=[lid])
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        kind = content.get("kind", "solid")

        def col(c, default=(0, 0, 0, 1)):
            c = list(c if c is not None else default)
            if len(c) == 3:
                c = c + [1.0]
            return np.asarray(c, np.float32)

        if kind == "solid":
            field = np.broadcast_to(col(content.get("color")),
                                    (h, w, 4)).copy()
        elif kind == "gradient":
            a = np.deg2rad(float(content.get("angle", 0.0)))
            t = ((xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a))
            t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
            c0, c1 = col(content.get("from")), col(content.get("to"))
            field = c0[None, None] * (1 - t[..., None])                 + c1[None, None] * t[..., None]
        elif kind == "radial":
            cx = float(content.get("cx", w / 2.0))
            cy = float(content.get("cy", h / 2.0))
            rad = float(content.get("radius", max(w, h) / 2.0))
            t = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad,
                        0.0, 1.0)
            c0, c1 = col(content.get("inner")), col(content.get("outer"))
            field = c0[None, None] * (1 - t[..., None])                 + c1[None, None] * t[..., None]
        else:
            raise ValueError("unknown fill kind %r" % kind)
        field = field.astype(np.float32)
        if respect_alpha:
            keep = l.pixels[..., 3:4]
            l.pixels = np.concatenate(
                [field[..., :3] * field[..., 3:4]
                 + l.pixels[..., :3] * (1 - field[..., 3:4]),
                 keep], -1).astype(np.float32)
        else:
            # normal source-over onto the existing pixels
            fa = field[..., 3:4]
            l.pixels = np.concatenate(
                [field[..., :3] * fa + l.pixels[..., :3]
                 * l.pixels[..., 3:4] * (1 - fa),
                 fa + l.pixels[..., 3:4] * (1 - fa)], -1)
            nz = l.pixels[..., 3:4] > 1e-6
            l.pixels[..., :3] = np.where(
                nz, l.pixels[..., :3] / np.maximum(l.pixels[..., 3:4],
                                                   1e-6),
                l.pixels[..., :3])
            l.pixels = l.pixels.astype(np.float32)
        _MUT_REV[0] += 1
        return True

    def flood_fill(self, lid, x, y, content, tolerance=0.12, contiguous=True,
                   selection=None):
        """Paint-bucket: fill the region of layer `lid` around (x, y) with
        `content` (an (H, W, 3) float image sampled per-pixel -- solid colours,
        gradients, patterns, and node outputs are all just content images).
        The region is pixels within `tolerance` (RGBA distance) of the seed;
        contiguous=True limits it to the connected component under the seed.
        Filled pixels become opaque. Pass a selection id in `selection` to
        confine the fill to that selection (Photoshop/GIMP behaviour: the
        bucket never spills past the marquee). Returns the filled count."""
        self._locked_guard(lid)
        self.record("Fill", only=[lid])
        l = self.layer(lid)
        h, w = l.pixels.shape[:2]
        x = int(np.clip(x, 0, w - 1)); y = int(np.clip(y, 0, h - 1))
        px = l.pixels.astype(np.float32)
        seed = px[y, x]
        dist = np.sqrt(((px - seed) ** 2).sum(-1))
        close = dist <= float(tolerance) * 2.0       # RGBA space: diagonal is 2
        if contiguous:
            import cv2
            m = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(close.astype(np.uint8), m, (x, y), 2,
                          loDiff=0, upDiff=0, flags=8)
            region = m[1:-1, 1:-1].astype(bool)
        else:
            region = close
        if selection:
            try:
                region = region & (self.selection_by_id(selection).data > 0.5)
            except KeyError:
                pass                             # stale id: fill unconfined
        content = np.asarray(content, np.float32)
        if content.shape[:2] != (h, w):
            content = _resize(content, h, w)
        l.pixels[region, :3] = np.clip(content[region, :3], 0, 1)
        l.pixels[region, 3] = 1.0
        return int(region.sum())

    def content_bbox(self, kind, oid, thresh=0.02, layer=None):
        """Tight bbox (x0, y0, x1, y1 inclusive) of a layer's opaque content, a
        mask's coverage, or a selection's coverage -- the transform tool's box.
        For selections, pass `layer` (a layer id) to AUTO-SHRINK the box to the
        non-transparent pixels inside the marquee, the way Photoshop and GIMP
        do -- so a rotate pivots about the drawing, not the loose rectangle.
        (leCore 0.2.3: tighten_selection; numpy fallback on older cores.)"""
        if kind == "strokes":
            # oid is a comma-joined stroke id list; the box is over the paths
            # plus each stroke's brush radius, so the handles hug the ink
            ids = [s for s in str(oid).split(",") if s]
            xs0 = xs1 = ys0 = ys1 = None
            for sid in ids:
                k = self.stroke_by_id(sid)
                r = float(k["brush"].get("radius", 8.0))
                for p2 in k["points"]:
                    xs0 = p2[0] - r if xs0 is None else min(xs0, p2[0] - r)
                    xs1 = p2[0] + r if xs1 is None else max(xs1, p2[0] + r)
                    ys0 = p2[1] - r if ys0 is None else min(ys0, p2[1] - r)
                    ys1 = p2[1] + r if ys1 is None else max(ys1, p2[1] + r)
            if xs0 is None:
                raise ValueError("no strokes given")
            return [max(0, int(xs0)), max(0, int(ys0)),
                    min(self.width - 1, int(xs1)), min(self.height - 1, int(ys1))]
        if kind == "layer":
            field = self.layer(oid).pixels[..., 3]
        elif kind == "mask":
            field = self.mask_by_id(oid).data
        else:
            field = self.selection_by_id(oid).data
            if layer:
                try:
                    alpha = self.layer(layer).pixels[..., 3] * (field > thresh)
                    m = mind()
                    if hasattr(m, "tighten_selection"):
                        t = m.tighten_selection(alpha, threshold=float(thresh))
                        if not t["empty"]:
                            r0, c0, r1, c1 = t["bbox"]
                            return [int(c0), int(r0), int(c1), int(r1)]
                    else:                              # older core: plain numpy
                        ys, xs = np.where(alpha > thresh)
                        if len(ys):
                            return [int(xs.min()), int(ys.min()),
                                    int(xs.max()), int(ys.max())]
                except KeyError:
                    pass                               # bad layer hint: loose box
        ys, xs = np.where(field > thresh)
        if not len(ys):
            return [0, 0, self.width - 1, self.height - 1]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def selection_bbox(self, sid, thresh=0.3):
        d = self.selection_by_id(sid).data
        ys, xs = np.where(d > thresh)
        if not len(ys):
            raise ValueError("selection is empty")
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # --- splines ----------------------------------------------------------------
    def spline_by_id(self, pid):
        for p in self.splines:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def add_spline(self, name=None, points=None, closed=False):
        self.record("Add spline", only=[])
        p = Spline(name, points, closed)
        self.splines.append(p)
        return p

    def remove_spline(self, pid):
        self.record("Remove spline", only=[])
        self.splines = [p for p in self.splines if p.id != pid]

    def edit_spline(self, pid, name=None, points=None, closed=None, record=False):
        _MUT_REV[0] += 1
        if record:
            self.record("Edit spline", only=[])
        p = self.spline_by_id(pid)
        if name is not None:
            p.name = name
        if points is not None:
            p.points = [dict(q) for q in points]
        if closed is not None:
            p.closed = bool(closed)
        return p

    def add_text(self, lid, text, x=0, y=0, size=48, color=(1, 1, 1),
                 font=None, spline=None, letter_spacing=0.0, shadow=None):
        """Rasterise text onto layer `lid` (recorded, undoable).
        Plain mode: draw at (x, y) = the text's top-left.
        Path mode: pass a spline id in `spline` and each glyph is placed along
        the curve, rotated to its tangent -- classic text-on-a-path; (x, y) is
        ignored and letter_spacing (px) spreads the glyphs.
        `font` is a font NAME from list_fonts() (default: DejaVu Sans).
        `shadow` = {dx, dy, blur, opacity, color} adds a drop shadow beneath.
        Returns the number of glyphs drawn."""
        from PIL import Image as PImage, ImageDraw, ImageFont
        self.record("Text", only=[lid])
        l = self.layer(lid)
        h, w = l.pixels.shape[:2]
        path = _font_path(font)
        fnt = ImageFont.truetype(path, int(size))

        canvas = PImage.new("L", (w, h), 0)                 # text coverage
        if spline:
            sp = self.spline_by_id(spline)
            poly = np.array(sp.flatten(48), np.float32)
            if len(poly) < 2:
                raise ValueError("that spline has fewer than 2 points")
            seg = np.diff(poly, axis=0)
            seglen = np.hypot(seg[:, 0], seg[:, 1])
            cum = np.concatenate([[0], np.cumsum(seglen)])
            total = float(cum[-1])
            def at(dist):                                   # point + tangent at arc length
                dist = np.clip(dist, 0, total - 1e-6)
                i = int(np.searchsorted(cum, dist, side="right") - 1)
                i = min(max(i, 0), len(seg) - 1)
                t = (dist - cum[i]) / max(seglen[i], 1e-6)
                p = poly[i] + seg[i] * t
                ang = float(np.degrees(np.arctan2(seg[i][1], seg[i][0])))
                return p, ang
            d = 0.0
            drawn = 0
            for ch in text:
                cw = fnt.getlength(ch)
                if cw <= 0:
                    d += float(letter_spacing); continue
                if d + cw / 2 > total:
                    break                                    # ran off the path's end
                p, ang = at(d + cw / 2)
                pad = int(size)
                tile = PImage.new("L", (int(cw) + 2 * pad, int(size * 1.6) + 2 * pad), 0)
                td = ImageDraw.Draw(tile)
                td.text((pad, pad), ch, fill=255, font=fnt)
                tile = tile.rotate(-ang, expand=True,
                                   center=(pad + cw / 2, pad + size * 0.6))
                px, py = int(p[0] - tile.width / 2), int(p[1] - tile.height / 2)
                canvas.paste(tile, (px, py), tile)
                d += cw + float(letter_spacing)
                drawn += 1
        else:
            dr = ImageDraw.Draw(canvas)
            dr.text((int(x), int(y)), text, fill=255, font=fnt)
            drawn = len(text)

        cov = np.asarray(canvas, np.float32) / 255.0
        def _over(rgb, alpha):                              # straight-alpha over
            a = alpha[..., None]
            l.pixels[..., :3] = rgb[None, None, :] * a +                 l.pixels[..., :3] * (1 - a)
            l.pixels[..., 3] = np.clip(alpha + l.pixels[..., 3] * (1 - alpha), 0, 1)
        if shadow:
            sh = dict(shadow)
            scov = np.asarray(
                PImage.fromarray((cov * 255).astype("uint8")), np.float32) / 255.0
            dx, dy = int(sh.get("dx", 4)), int(sh.get("dy", 4))
            scov = np.roll(np.roll(scov, dy, axis=0), dx, axis=1)
            if dy > 0: scov[:dy] = 0
            elif dy < 0: scov[dy:] = 0
            if dx > 0: scov[:, :dx] = 0
            elif dx < 0: scov[:, dx:] = 0
            blur = float(sh.get("blur", 4))
            if blur > 0:
                scov = _gauss_blur(scov, blur)
            _over(np.asarray(sh.get("color", (0, 0, 0)), np.float32)[:3],
                  np.clip(scov * float(sh.get("opacity", 0.6)), 0, 1))
        _over(np.asarray(color, np.float32)[:3], cov)
        return drawn

    def duplicate_layer(self, lid):
        """Insert an independent copy of layer `lid` directly above it
        (recorded, undoable). Copies pixels and every property, including the
        mask BINDING (the mask itself is shared, as in Photoshop). Returns the
        new layer."""
        self.record("Duplicate layer", only=[])
        src = self.layer(lid)
        i = self.layers.index(src)
        cp = Layer(self.height, self.width, src.name + " copy",
                   src.pixels.copy())
        cp.visible, cp.opacity, cp.blend = src.visible, src.opacity, src.blend
        cp.mask, cp.mask_invert = src.mask, src.mask_invert
        self.layers.insert(i + 1, cp)
        return cp

    def duplicate_mask(self, mid):
        """Insert an independent copy of mask `mid` (recorded, undoable).
        Layers bound to the original stay bound to the original. Returns the
        new mask."""
        self.record("Duplicate mask", only=[])
        src = self.mask_by_id(mid)
        cp = Mask(self.height, self.width, src.name + " copy", src.data.copy())
        self.masks.append(cp)
        return cp

    def copy_region(self, lid, selection=None):
        """Clipboard COPY: the layer's pixels -- confined to `selection` when
        one is given (outside pixels become transparent), cropped to the
        content bbox. Returns {'pixels': (h,w,4), 'x', 'y'} or None when the
        region is empty. Not recorded (copying isn't an edit)."""
        l = self.layer(lid)
        px = l.pixels.copy()
        if selection:
            try:
                sel = self.selection_by_id(selection).data
                px[..., 3] = px[..., 3] * (sel > 0.5)
            except KeyError:
                pass
        ys, xs = np.where(px[..., 3] > 1e-3)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return {"pixels": px[y0:y1, x0:x1].copy(), "x": int(x0), "y": int(y0)}

    def cut_region(self, lid, selection=None):
        """Clipboard CUT: copy_region, then clear the copied pixels to
        transparent on the source layer (recorded, undoable)."""
        clip = self.copy_region(lid, selection)
        if clip is None:
            return None
        self.record("Cut")
        l = self.layer(lid)
        if selection:
            try:
                sel = self.selection_by_id(selection).data
                l.pixels[..., 3] = l.pixels[..., 3] * (sel <= 0.5)
            except KeyError:
                l.pixels[..., 3] = 0
        else:
            l.pixels[..., 3] = 0
        return clip

    def paste(self, clip, x=None, y=None, name="Pasted"):
        """Clipboard PASTE: a NEW layer above the stack holding the clip's
        pixels at (x, y) -- defaults to where they were copied from, clamped
        into the canvas (recorded, undoable). Returns the new layer."""
        self.record("Paste")
        ph, pw = clip["pixels"].shape[:2]
        x = int(clip["x"] if x is None else x)
        y = int(clip["y"] if y is None else y)
        x = max(min(x, self.width - 1), 1 - pw)
        y = max(min(y, self.height - 1), 1 - ph)
        l = Layer(self.height, self.width, name)
        sy, sx = max(0, y), max(0, x)
        oy, ox = sy - y, sx - x
        ey, ex = min(self.height, y + ph), min(self.width, x + pw)
        l.pixels[sy:ey, sx:ex] = clip["pixels"][oy:oy + (ey - sy),
                                                ox:ox + (ex - sx)]
        self.layers.append(l)
        return l

    def stroke_spline(self, lid, pid, **paint_kwargs):
        """Use the spline as the guide for a brush stroke on a layer."""
        path = self.spline_by_id(pid).flatten()
        if len(path) < 1:
            return
        self.paint(lid, path, **paint_kwargs)

    # --- masks ------------------------------------------------------------------
    def mask_by_id(self, mid):
        for m in self.masks:
            if m.id == mid:
                return m
        raise KeyError(mid)

    def mask_map(self):
        return {m.id: m for m in self.masks}

    def add_mask(self, name=None, data=None):
        self.record("Add mask")
        m = Mask(self.height, self.width, name, data)
        self.masks.append(m)
        return m

    def remove_mask(self, mid):
        self.record("Remove mask")
        self.masks = [m for m in self.masks if m.id != mid]
        for l in self.layers:
            if l.mask == mid:
                l.mask = None

    def edit_mask(self, mid, name=None):
        _MUT_REV[0] += 1
        m = self.mask_by_id(mid)
        if name is not None:
            m.name = name
        return m

    def move_mask(self, mid, index):
        self.record("Reorder masks")
        m = self.mask_by_id(mid)
        self.masks.remove(m)
        self.masks.insert(max(0, min(len(self.masks), int(index))), m)

    def merge_masks(self, ids=None):
        """Union the given masks (or ALL) into the first; layers pointing at the
        removed masks are re-pointed at the merged one."""
        items = [self.mask_by_id(i) for i in ids] if ids else list(self.masks)
        if len(items) < 2:
            return items[0] if items else None
        self.record("Merge masks")
        base = items[0]
        gone = set()
        for m in items[1:]:
            base.data = np.maximum(base.data, m.data)
            self.masks.remove(m)
            gone.add(m.id)
        for l in self.layers:
            if l.mask in gone:
                l.mask = base.id
        return base

    MAX_STROKES = 512

    def paint_live(self, lid, points, first, **kw):
        """Live mode: repaint the WHOLE in-progress stroke each flush.

        The old protocol painted each 140 ms chunk as its own paint() call.
        paint() resolves its points as ONE coverage mask with the opacity
        applied once at the end -- so a stroke split into N chunks composited
        alpha-over at every chunk boundary and came out darker there than the
        same points painted in one call. replay_layer() replays each stroke AS
        one call, so a live-painted layer never matched its own replay:
        replay_is_faithful() said False and nudge refused with "content that
        was not painted as strokes" -- on a layer that was nothing BUT
        strokes. The chunking was the unfaithful part, not the content.

        So: on the first flush, snapshot the layer and paint normally. On
        every later flush the client sends the FULL accumulated point list;
        restore the snapshot, drop the previous partial stroke record, and
        repaint the whole stroke as one call. The final pixels are exactly
        what a single paint() would have produced, byte for byte, so live and
        non-live strokes are indistinguishable afterwards -- to replay, to
        nudge, and to undo.

        Undo: recorded once, on the first flush, with a FULL-layer snapshot.
        The old per-chunk protocol recorded only the first chunk's bounding
        box, so undoing a live stroke restored that box and left the rest of
        the stroke on the canvas."""
        kw.pop("record", None)
        kw.pop("stroke_new", None)
        live = getattr(self, "_live_stroke", None)
        if first or not live or live.get("lid") != lid:
            self.record("Brush (live)", only=[lid])       # full snapshot: the
            hm = self.layer(lid).height_map
            mm = getattr(self.layer(lid), "material_map", None)
            self._live_stroke = {                         # stroke can grow
                "lid": lid,                               # anywhere from here
                "before": self.layer(lid).pixels.copy(),
                # the paint SURFACE restores with the pigment, or every live
                # flush would re-deposit the whole stroke's height on top of
                # the last flush's -- media strokes thickened with flush count
                "before_h": None if hm is None else hm.copy(),
                # the material map has the same flush-count failure mode:
                # without a restore, coverage saturates and grain doubles
                "before_m": None if mm is None else mm.copy(),
                # the brush itself rewinds too: a live flush repaints the
                # whole stroke, and without this the reservoir was spent once
                # per flush instead of once per stroke
                "before_charge": np.asarray(
                    getattr(self, "brush_charges",
                            np.full(_BRUSH_LANES, self.brush_charge,
                                    np.float32)), np.float32).copy(),
                "before_lanes": (None if getattr(self, "brush_lanes", None)
                                 is None else np.asarray(
                                     self.brush_lanes, np.float32).copy()),
                "sid": None,
            }
        else:
            self.layer(lid).pixels[:] = live["before"]
            if live.get("before_h") is not None:
                self.layer(lid).height_map[:] = live["before_h"]
            elif self.layer(lid).height_map is not None:
                self.layer(lid).height_map[:] = 0.0
            if live.get("before_m") is not None:
                self.layer(lid).material_map[:] = live["before_m"]
            elif getattr(self.layer(lid), "material_map", None) is not None:
                self.layer(lid).material_map[:] = 0.0
            if live.get("before_charge") is not None:
                self.brush_charges = live["before_charge"].copy()
                self.brush_charge = float(self.brush_charges.mean())
            if live.get("before_lanes") is not None:
                self.brush_lanes = live["before_lanes"].copy()
                self.brush_color = tuple(
                    float(v) for v in self.brush_lanes.mean(0))
            if live.get("sid"):
                try:                       # gone already (undo mid-stroke): fine
                    self.strokes.remove(self.stroke_by_id(live["sid"]))
                except KeyError:
                    pass
        self.paint(lid, points, record=False, stroke_new=True, **kw)
        self._live_stroke["sid"] = self.strokes[-1]["id"] if self.strokes else None
        self._mark_replay_ok(lid)
        return self._live_stroke["sid"]

    def record_stroke(self, lid, points, brush, new):
        """Append a painted path. `new` starts a stroke; otherwise the points
        extend the one in progress -- the client flushes a stroke in segments,
        and `record=True` marks the first flush, so that flag doubles as the
        stroke boundary without inventing any new protocol."""
        pts = [([float(p[0]), float(p[1]), float(p[2])] if len(p) > 2
                else [float(p[0]), float(p[1])]) for p in points]
        if not pts:
            return None
        if getattr(self, "_replaying", False):
            return None                      # a replay must not re-record
        self._capture_replay_base(lid)
        if not new and self.strokes and self.strokes[-1]["layer"] == lid:
            self.strokes[-1]["points"].extend(pts)
            return self.strokes[-1]["id"]
        self._stroke_n = getattr(self, "_stroke_n", 0) + 1
        sid = "K%d" % self._stroke_n
        self.strokes.append({"id": sid, "layer": lid, "points": pts,
                             "brush": dict(brush or {})})
        if len(self.strokes) > self.MAX_STROKES:
            del self.strokes[:len(self.strokes) - self.MAX_STROKES]
        return sid

    REPLAY_BASE_BUDGET = 192 * 1024 * 1024   # bytes of replay bases retained

    def _capture_replay_base(self, lid):
        """Remember what a layer looked like BEFORE its first recorded stroke.
        Replay = this base, plus every stroke re-applied in order.

        Bounded. One full-size copy per painted layer sounds small until you
        paint on a dozen layers: measured 398 MB of hidden copies at 1920x1080,
        roughly DOUBLING the document's memory with something the user cannot
        see. Oldest bases are dropped first; losing one only means nudge
        declines to move that layer's strokes (replay_is_faithful returns
        False), which is already a handled, honest outcome."""
        if not hasattr(self, "_replay_base"):
            self._replay_base = {}
        if lid in self._replay_base:
            return
        px = self.layer(lid).pixels
        self._replay_base[lid] = px.copy()
        budget = self.REPLAY_BASE_BUDGET
        while len(self._replay_base) > 1 and \
                sum(v.nbytes for v in self._replay_base.values()) > budget:
            oldest = next(iter(self._replay_base))
            if oldest == lid:                    # never evict the one just taken
                break
            del self._replay_base[oldest]

    def replay_region(self, lid, x0, y0, x1, y1):
        """Rebuild only a rectangle of a layer from its strokes.

        A nudge changes a small neighbourhood, but a full replay repaints every
        stroke over the whole canvas -- 6 s for 120 strokes at 1080p, which is
        not a tool, it is a wait. Only strokes whose bounding box touches the
        region can affect it, and each is clipped to that region, so the cost
        follows the edit rather than the document."""
        if any(k["layer"] == lid and (k["brush"].get("media")
                                      or k["brush"].get("material")
                                      or k["brush"].get("blend")
                                      or k["brush"].get("knife"))
               for k in self.strokes):
            # a media stroke's gravity flow runs BELOW its own bbox and reads
            # accumulated height from earlier strokes; a rectangle replay
            # cannot reproduce that. Material strokes share the same body
            # physics AND accumulate per-pixel stuff. Decline, and the caller
            # falls back to the full-layer rebuild, which can.
            return False
        base = getattr(self, "_replay_base", {}).get(lid)
        if base is None:
            return False
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.width, int(x1)); y1 = min(self.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            return False
        L = self.layer(lid)
        keep = L.pixels
        win = base[y0:y1, x0:x1].copy()
        full = np.zeros_like(keep)
        full[y0:y1, x0:x1] = win
        L.pixels = full
        self._replaying = True
        try:
            for k in self.strokes:
                if k["layer"] != lid:
                    continue
                pts = k["points"]
                if not pts:
                    continue
                r = float(k["brush"].get("radius", 8.0)) + 2.0
                bx0 = min(p[0] for p in pts) - r; bx1 = max(p[0] for p in pts) + r
                by0 = min(p[1] for p in pts) - r; by1 = max(p[1] for p in pts) + r
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue                       # cannot touch this region
                b = k["brush"]
                self.paint(lid, [tuple(p) for p in pts],
                           color=tuple(b.get("color", (0, 0, 0))),
                           radius=float(b.get("radius", 8.0)),
                           opacity=float(b.get("opacity", 1.0)),
                           erase=bool(b.get("erase")),
                           hardness=float(b.get("hardness", 0.7)),
                           record=False)
            keep[y0:y1, x0:x1] = L.pixels[y0:y1, x0:x1]
        finally:
            self._replaying = False
            L.pixels = keep
        return True

    def replay_layer(self, lid, into=None):
        """Rebuild a layer from its base plus its recorded strokes.

        Returns the rebuilt pixels, or None if there is no base to replay from.
        Strokes are re-applied with _replaying set, so replaying does not
        re-record them (which would double the list every time)."""
        base = getattr(self, "_replay_base", {}).get(lid)
        if base is None:
            return None
        keep = self.layer(lid).pixels
        keep_h = self.layer(lid).height_map
        keep_m = getattr(self.layer(lid), "material_map", None)
        self.layer(lid).pixels = base.copy() if into is None else into
        if any(k["layer"] == lid and (k["brush"].get("media")
                                      or k["brush"].get("material"))
               for k in self.strokes):
            self.layer(lid).height_map = np.zeros_like(
                self.layer(lid).pixels[..., 0])
        if any(k["layer"] == lid and k["brush"].get("material")
               for k in self.strokes):
            # material strokes rebuild their stuff from zero exactly like
            # the paint body -- replaying over the live map would double it
            self.layer(lid).material_map = np.zeros(
                self.layer(lid).pixels.shape[:2] + (3,), np.float32)
        # the stratum this layer spills into is DERIVED from these strokes, so
        # it has to be cleared before a rebuild or each replay piles onto the
        # last one
        _nx = getattr(self.layer(lid), "stratum_next", None)
        for _ in range(12):                       # the whole chain, not just one
            if not _nx:
                break
            try:
                _u = self.layer(_nx)
            except KeyError:
                break
            _u.pixels[...] = 0.0
            if _u.height_map is not None:
                _u.height_map[...] = 0.0
            _nx = getattr(_u, "stratum_next", None)
        if not hasattr(self, "_replay_height"):
            self._replay_height = {}
        self._replay_height.pop(lid, None)
        if not hasattr(self, "_replay_material"):
            self._replay_material = {}
        self._replay_material.pop(lid, None)
        self._replaying = True
        try:
            for k in self.strokes:
                if k["layer"] != lid:
                    continue
                b = k["brush"]
                if b.get("knife"):
                    # the knife shapes existing paint, so like a blend it is a
                    # mark in its own right and replays IN ORDER
                    self.knife(lid, [tuple(p) for p in k["points"]],
                               mode=str(b.get("knife", "smooth")),
                               radius=float(b.get("radius", 26.0)),
                               strength=float(b.get("strength", 0.7)),
                               record=False, stroke_new=False)
                    continue
                if b.get("blend"):
                    # a blend is a mark like any other and replays IN ORDER --
                    # that ordering is what makes this non-destructive: paint,
                    # paint, blend rebuilds exactly, and editing any of the
                    # three re-derives the result
                    self.blend_stroke(lid, [tuple(p) for p in k["points"]],
                                      radius=float(b.get("radius", 18.0)),
                                      strength=float(b.get("strength", 0.6)),
                                      brush=b.get("brush"),
                                      record=False, stroke_new=False)
                    continue
                self.paint(lid, [tuple(p) for p in k["points"]],
                           color=tuple(b.get("color", (0, 0, 0))),
                           radius=float(b.get("radius", 8.0)),
                           opacity=float(b.get("opacity", 1.0)),
                           erase=bool(b.get("erase")),
                           hardness=float(b.get("hardness", 0.7)),
                           record=False,
                           media=b.get("media"),
                           material=b.get("material"),
                           mix=float(b.get("mix", 0.0)),
                           real_brush=bool(b.get("real_brush", False)),
                           charge0=b.get("charge0"),
                           lanes0=b.get("lanes0"),
                           load=float(b.get("load", 0.6)),
                           alpha_lock=bool(b.get("alpha_lock", False)))
            out = self.layer(lid).pixels
            # the paint surface rebuilds with the pigment: strokes with media
            # re-deposit and re-flow into the fresh field set up above
            self._replay_height[lid] = self.layer(lid).height_map
            self._replay_material[lid] = getattr(
                self.layer(lid), "material_map", None)
        finally:
            self._replaying = False
            self.layer(lid).pixels = keep
            self.layer(lid).height_map = keep_h
            self.layer(lid).material_map = keep_m
        return out

    def _layer_fingerprint(self, lid):
        """A cheap signature of a layer's pixels. Strided so it costs ~1 ms on
        a 1080p layer while still moving whenever the image does."""
        px = self.layer(lid).pixels
        s = px[::16, ::16]
        return (float(s.sum()), float(np.abs(s).max()), px.shape)

    def _replay_ok_cached(self, lid):
        """Was this layer verified replayable, and is it still the same image?

        Replaying is expensive (~6 s for 120 strokes at 1080p) and nudge did it
        three times per drag. Caching the verdict is worth a lot -- but the
        mutation counter ALONE is not safe evidence: writing to
        `layer.pixels` directly never bumps it, so a cache keyed only on the
        counter would keep saying "faithful" after foreign content appeared,
        and nudge would happily overwrite it. So the fingerprint is checked
        too: cheap, and it moves whenever the pixels do."""
        seen = getattr(self, "_replay_ok", {})
        rec = seen.get(lid)
        if not rec:
            return False
        rev, fp = rec
        return rev == _MUT_REV[0] and fp == self._layer_fingerprint(lid)

    def _stratum_for(self, lid):
        """The layer that catches paint once `lid` is full -- find or create.

        Linked both ways so a replay reuses the same stratum instead of
        breeding a new one per rebuild, and so the base can clear its
        stratum before replaying.
        """
        l = self.layer(lid)
        nxt = getattr(l, "stratum_next", None)
        if nxt:
            try:
                return self.layer(nxt)
            except KeyError:
                pass
        # name from the ROOT of the chain, not from the layer we spilled off,
        # or a deep build reads "p ~2 ~2 ~2 ~2 ~2 ~2"
        root = getattr(l, "stratum_root", None)
        base_name = root or re.sub(r" ~\d+$", "", getattr(l, "name", "paint"))
        n = 2
        while any(x.name == "%s ~%d" % (base_name, n) for x in self.layers):
            n += 1
        idx = [x.id for x in self.layers].index(lid)
        above = (self.layers[idx + 1].id if idx + 1 < len(self.layers)
                 else None)
        new = (self.add_layer("%s ~%d" % (base_name, n), record=False,
                              below=above) if above
               else self.add_layer("%s ~%d" % (base_name, n), record=False))
        new = new if hasattr(new, "id") else self.layers[-1]   # USE THE RETURN VALUE
        # it is the same paint, one stratum higher
        new.paper = getattr(l, "paper", "canvas")
        new.paint_media = getattr(l, "paint_media", None)
        new.paint_gloss = getattr(l, "paint_gloss", 0.3)
        new.relief = float(getattr(l, "relief", 1.0))
        new.thickness = float(getattr(l, "thickness", 1.0))
        new.z_off = float(getattr(l, "z_off", 0.0)) + _HEIGHT_CAP
        new.gravity = getattr(l, "gravity", None)
        new.gravity_angle = getattr(l, "gravity_angle", None)
        # A stratum is the SAME MARK continued, so it must inherit how the
        # layer relates to the picture. A clipped glaze that spilled produced
        # unclipped strata, and the glaze escaped its base -- the exact
        # workflow the clip is there to support. Same for the blend mode and
        # opacity: the overflow of a multiply glaze is still multiply.
        new.clip = bool(getattr(l, "clip", False))
        new.blend = getattr(l, "blend", "normal")
        new.opacity = float(getattr(l, "opacity", 1.0))
        new.alpha_lock = bool(getattr(l, "alpha_lock", False))
        new.stratum_of = lid
        new.stratum_root = base_name
        l.stratum_next = new.id
        return new

    def _mark_replay_ok(self, lid):
        if not hasattr(self, "_replay_ok"):
            self._replay_ok = {}
        self._replay_ok[lid] = (_MUT_REV[0], self._layer_fingerprint(lid))

    def revector_layer(self, lid):
        """Re-render a layer's strokes at the CURRENT resolution.

        After a resize, a layer's pixels are an UPSCALE of the old ones, but the
        stroke paths are exact -- so repainting them produces a genuinely
        sharper result than the resample did (measured: peak difference 0.29
        against a blurred upscale). This is where "resolution independent"
        actually lives: the strokes are the master, the pixels are a render of
        them at whatever size the document happens to be.

        Only valid when the layer's content really is its strokes; returns
        False otherwise rather than discarding whatever else is there."""
        base = getattr(self, "_replay_base", {}).get(lid)
        if base is None:
            return False
        h, w = self.height, self.width
        if base.shape[0] != h or base.shape[1] != w:
            self._replay_base[lid] = base = _resize(base, h, w)
        rebuilt = self.replay_layer(lid)
        if rebuilt is None:
            return False
        self.record("Re-render strokes", only=[lid])
        self.layer(lid).pixels = rebuilt
        if getattr(self, '_replay_height', {}).get(lid) is not None:
            self.layer(lid).height_map = self._replay_height[lid]
        if getattr(self, '_replay_material', {}).get(lid) is not None:
            self.layer(lid).material_map = self._replay_material[lid]
        _MUT_REV[0] += 1
        return True

    def _stroke_edit_guard(self, lids):
        """Refuse a stroke edit that would rebuild away non-stroke content.

        Every stroke edit (move a point, taper, simulate, pose, transform)
        works by rewriting the paths and REPLACING the layer's pixels with a
        replay of base + strokes. If the layer also holds content that is not
        a stroke -- a flood fill, an imported image, a node bake -- the replay
        does not contain it, so the replacement silently deletes it. That
        exact loss shipped: dragging one point of a stroke reverted a fill on
        the same layer to the background colour with no error and no toast.
        nudge_strokes had this gate from day one; nothing else did."""
        for lid in lids:
            if not self.replay_is_faithful(lid):
                name = next((l.name for l in self.layers if l.id == lid), lid)
                raise ValueError(
                    "layer %r has content that was not painted as strokes "
                    "(a fill, an imported image, or a bake) -- editing its "
                    "strokes would discard that content" % name)

    def replay_is_faithful(self, lid, tol=2e-3):
        """Does replaying the recorded strokes reproduce what is on the layer?

        This is the safety gate for nudging. Rather than tracking every
        possible way a layer might have been touched (fills, bakes, imported
        images, node output) and hoping the list is complete, just CHECK: if a
        faithful replay does not match the current pixels, something else
        contributed and moving strokes would silently destroy it."""
        if self._replay_ok_cached(lid):
            return True
        rep = self.replay_layer(lid)
        if rep is None:
            return False
        ok = float(np.abs(rep - self.layer(lid).pixels).max()) <= tol
        if ok:
            self._mark_replay_ok(lid)
        return ok

    def nudge_strokes(self, lid, path, radius=40.0, strength=1.0, record=True):
        """Push recorded stroke POINTS around instead of smearing pixels.

        A smudge drags pixels, so repeated use turns crisp sketch lines into
        swirls. This moves the underlying paths and re-renders them, so lines
        stay as clean as when they were drawn -- only their shape changes.

        Refuses (returns 0) unless a faithful replay is possible, so it can
        never eat content it did not draw."""
        self._locked_guard(lid)
        if len(path) < 2 or not self.replay_is_faithful(lid):
            return 0
        if record:
            self.record("Nudge", only=[lid])
        moved = 0
        added = 0
        r2 = float(radius) ** 2
        step = max(3.0, float(radius) * 0.16)
        for i in range(1, len(path)):
            ox, oy = float(path[i - 1][0]), float(path[i - 1][1])
            dx = float(path[i][0]) - ox
            dy = float(path[i][1]) - oy
            if dx == 0.0 and dy == 0.0:
                continue
            # give the falloff something to act on, right here and nowhere else
            added += self._insert_points_near(lid, ox, oy, float(radius), step)
            for k in self.strokes:
                if k["layer"] != lid:
                    continue
                for q in k["points"]:
                    d2 = (q[0] - ox) ** 2 + (q[1] - oy) ** 2
                    if d2 < r2:
                        w = (1.0 - (d2 / r2) ** 0.5) ** 2      # smooth falloff
                        q[0] += dx * w * float(strength)
                        q[1] += dy * w * float(strength)
                        moved += 1
        if moved:
            for k in self.strokes:
                rig = k.get("rig")
                if (k["layer"] == lid and rig
                        and len(rig.get("bones", [])) == len(k["points"]) - 1):
                    # the nudge pushed joints; the rig pulls the bone lengths
                    # back with pins held, so a rigged stroke bends and swings
                    # under the brush instead of stretching like putty
                    self._relax_lengths(k["points"], rig["bones"],
                                        set(rig.get("pins", [])),
                                        iterations=16)
                    self._sync_rig_after_edit(k)
            # only the neighbourhood the drag passed through can have changed,
            # plus the brush radius and the falloff radius on either side
            pad = float(radius) + 24.0
            rx0 = min(p[0] for p in path) - pad
            rx1 = max(p[0] for p in path) + pad
            ry0 = min(p[1] for p in path) - pad
            ry1 = max(p[1] for p in path) + pad
            if not self.replay_region(lid, rx0, ry0, rx1, ry1):
                rebuilt = self.replay_layer(lid)
                if rebuilt is not None:
                    self.layer(lid).pixels = rebuilt
                    if getattr(self, '_replay_height', {}).get(lid) is not None:
                        self.layer(lid).height_map = self._replay_height[lid]
                    if getattr(self, '_replay_material', {}).get(lid) is not None:
                        self.layer(lid).material_map = self._replay_material[lid]
            _MUT_REV[0] += 1
            self._mark_replay_ok(lid)      # the layer IS the replay right now
        self.last_nudge_added = added
        return moved

    def _insert_points_near(self, lid, cx, cy, radius, step):
        """Add stroke points ONLY where a nudge is about to act, and only when
        the existing samples are too sparse to carry the deformation.

        Blanket-densifying every stroke worked but was wasteful and wrong in a
        subtle way: it rewrote the whole path on every nudge, so point counts
        grew in regions the user never touched and the original sampling was
        lost. Inserting locally keeps distant geometry byte-for-byte identical,
        and converges -- nudging the same place twice adds nothing the second
        time, because the spacing is already fine enough."""
        added = 0
        for k in self.strokes:
            if k["layer"] != lid or len(k["points"]) < 2:
                continue
            pts = k["points"]
            out = [pts[0]]
            seg_ins = []           # inserts per source segment, for the rig
            for a, b in zip(pts, pts[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                seg = (dx * dx + dy * dy) ** 0.5
                # Solve for the PORTION of this segment inside the influence
                # circle. Testing the segment as a whole was too coarse: one
                # long segment that merely passes near the cursor would get
                # subdivided end to end, resampling geometry nowhere near the
                # nudge.
                L = dx * dx + dy * dy
                if L <= 1e-9 or seg <= step * 1.5:
                    out.append(b)
                    seg_ins.append([])
                    continue
                fx, fy = a[0] - cx, a[1] - cy
                bq = 2.0 * (fx * dx + fy * dy)
                cq = fx * fx + fy * fy - radius * radius
                disc = bq * bq - 4.0 * L * cq
                if disc <= 0.0:
                    out.append(b)              # never enters the circle
                    seg_ins.append([])
                    continue
                rt = disc ** 0.5
                t0 = max(0.0, (-bq - rt) / (2.0 * L))
                t1 = min(1.0, (-bq + rt) / (2.0 * L))
                if t1 <= t0:
                    out.append(b)
                    seg_ins.append([])
                    continue
                span = (t1 - t0) * seg
                n = int(span / max(step, 1e-3))
                ts = []
                for j in range(1, n + 1):
                    t = t0 + (t1 - t0) * (j / float(n + 1))
                    out.append([a[0] + dx * t, a[1] + dy * t])
                    ts.append(t)
                    added += 1
                out.append(b)
                seg_ins.append(ts)
            if len(out) <= 6000:
                rig = k.get("rig")
                if rig and len(rig.get("bones", [])) == len(pts) - 1:
                    # a rigged stroke's bones, pins and verlet history are
                    # per-index arrays: splitting a segment must split its
                    # bone's REST LENGTH at the same parameter and remap the
                    # pins, or the next simulate reads past the end. The first
                    # fix simply refused to densify rigs -- which silently
                    # turned nudge into a no-op on any sparse rigged stroke.
                    bones2, prev2 = [], [list(rig["prev"][0])]
                    idx_map = {0: 0}
                    cursor = 0
                    for si, ts in enumerate(seg_ins):
                        rest = rig["bones"][si]
                        pa, pb = rig["prev"][si], rig["prev"][si + 1]
                        last_t = 0.0
                        for t in ts:
                            bones2.append(rest * (t - last_t))
                            prev2.append([pa[0] + (pb[0] - pa[0]) * t,
                                          pa[1] + (pb[1] - pa[1]) * t])
                            last_t = t
                            cursor += 1
                        bones2.append(rest * (1.0 - last_t))
                        prev2.append(list(pb))
                        cursor += 1
                        idx_map[si + 1] = cursor
                    rig["bones"] = bones2
                    rig["prev"] = prev2
                    rig["pins"] = sorted(idx_map[p2] for p2 in rig["pins"]
                                         if p2 in idx_map)
                k["points"] = out
        return added

    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        L = vx * vx + vy * vy
        t = 0.0 if L <= 1e-9 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L))
        dx, dy = ax + vx * t - px, ay + vy * t - py
        return (dx * dx + dy * dy) ** 0.5

    def strokes_at(self, x, y, layer=None, slack=3.0):
        """Stroke ids whose path passes under (x, y), nearest first.

        Hit distance is the stroke's own brush radius plus a little slack, so
        clicking a fat stroke works where it looks like it should, and several
        overlapping strokes all report rather than only the top one."""
        hits = []
        for k in self.strokes:
            if layer is not None and k["layer"] != layer:
                continue
            pts = k["points"]
            if len(pts) < 1:
                continue
            r = float(k["brush"].get("radius", 8.0)) + float(slack)
            best = min((self._seg_dist(x, y, a[0], a[1], b[0], b[1])
                        for a, b in zip(pts, pts[1:])),
                       default=((pts[0][0] - x) ** 2 + (pts[0][1] - y) ** 2) ** 0.5)
            if best <= r:
                hits.append((best, k["id"]))
        hits.sort()
        return [sid for _d, sid in hits]

    def resolve_stroke_selection(self, anchors, forward=0, back=0, ignore=(),
                                 layer=None):
        """Grow a set of anchor strokes along the ORDER THEY WERE PAINTED.

        forward/back count in strokes, not pixels -- "the three strokes I drew
        after this one" is the useful unit when refining a sketch. Restricted
        to one layer when `layer` is given, and `ignore` always wins so a
        stroke can be dropped without re-picking the whole set."""
        order = [k["id"] for k in self.strokes
                 if layer is None or k["layer"] == layer]
        pos = {sid: i for i, sid in enumerate(order)}
        ignore = set(ignore or ())
        out = []
        for a in anchors or ():
            if a not in pos:
                continue
            i = pos[a]
            lo = max(0, i - int(back))
            hi = min(len(order), i + int(forward) + 1)
            out.extend(order[lo:hi])
        seen, keep = set(), []
        for sid in out:                       # de-dupe, keep paint order
            if sid in seen or sid in ignore:
                continue
            seen.add(sid)
            keep.append(sid)
        keep.sort(key=lambda s: pos[s])
        return keep

    BRUSH_KEYS = ("radius", "opacity", "hardness", "erase", "color")

    def brush_compatible(self, a, b, tol=1e-4):
        """Do two strokes carry the same brush? Only compatible strokes may be
        joined -- merging strokes with different radius or colour would have to
        throw one of them away, and silently losing a setting is worse than
        refusing."""
        ka = a["brush"] if isinstance(a, dict) else self.stroke_by_id(a)["brush"]
        kb = b["brush"] if isinstance(b, dict) else self.stroke_by_id(b)["brush"]
        for key in self.BRUSH_KEYS:
            va, vb = ka.get(key), kb.get(key)
            if key == "color":
                va = [float(c) for c in (va or (0, 0, 0))]
                vb = [float(c) for c in (vb or (0, 0, 0))]
                if any(abs(x - y) > tol for x, y in zip(va, vb)):
                    return False
            elif key == "erase":
                if bool(va) != bool(vb):
                    return False
            elif abs(float(va or 0) - float(vb or 0)) > tol:
                return False
        return True

    def points_at(self, x, y, radius=12.0, layer=None):
        """Point-level selection: [(stroke_id, index), ...] within `radius`,
        nearest first, SPANNING strokes.

        The points are the real data -- a stroke is just a shape over them that
        carries shared brush settings -- so a selection is naturally a set of
        points, not a set of strokes."""
        hits = []
        for k in self.strokes:
            if layer is not None and k["layer"] != layer:
                continue
            for i, q in enumerate(k["points"]):
                d = ((q[0] - x) ** 2 + (q[1] - y) ** 2) ** 0.5
                if d <= radius:
                    hits.append((d, k["id"], i))
        hits.sort()
        return [(sid, i) for _d, sid, i in hits]

    def move_points(self, sel, dx, dy, record=True, falloff=0.0, strength=1.0):
        """Move an explicit set of (stroke_id, index) points.

        `falloff` > 0 makes the edit SOFT: neighbouring points along the same
        stroke follow with a smoothstep weight that fades to zero over that
        many pixels of ARC LENGTH. Distance runs along the thread, not through
        space, so a hairpin's far side does not get dragged just for being
        near. Without it, dense freehand strokes (a point every couple of
        pixels) sheared: the grabbed point jumped and its untouched neighbours
        stayed, creating a kink -- the exact complaint soft editing answers.
        `strength` scales the whole edit; selected points move dx*strength."""
        by = {}
        for sid, i in sel or ():
            by.setdefault(sid, set()).add(int(i))
        if not by:
            return 0
        lids = {self.stroke_by_id(s)["layer"] for s in by}
        self._stroke_edit_guard(lids)
        if record:
            self.record("Move points", only=list(lids))
        fall = max(0.0, float(falloff))
        stg = float(strength)
        n = 0
        for sid, idxs in by.items():
            k = self.stroke_by_id(sid)
            pts = k["points"]
            valid = sorted(i for i in idxs if 0 <= i < len(pts))
            if not valid:
                continue
            if fall <= 0.0:
                for i in valid:
                    pts[i][0] += float(dx) * stg
                    pts[i][1] += float(dy) * stg
                    n += 1
                continue
            # cumulative arc length, then each point's distance ALONG the
            # stroke to the nearest selected point
            arc = [0.0]
            for a, b in zip(pts, pts[1:]):
                arc.append(arc[-1] + ((b[0] - a[0]) ** 2
                                      + (b[1] - a[1]) ** 2) ** 0.5)
            sel_arcs = [arc[i] for i in valid]
            for i, p in enumerate(pts):
                d = min(abs(arc[i] - a2) for a2 in sel_arcs)
                if d >= fall:
                    continue
                t = 1.0 - d / fall
                w = t * t * (3.0 - 2.0 * t) * stg     # smoothstep falloff
                p[0] += float(dx) * w
                p[1] += float(dy) * w
                n += 1
            self._sync_rig_after_edit(k)
        for lid in {self.stroke_by_id(s)["layer"] for s in by}:
            rebuilt = self.replay_layer(lid)
            if rebuilt is not None:
                self.layer(lid).pixels = rebuilt
                if getattr(self, '_replay_height', {}).get(lid) is not None:
                    self.layer(lid).height_map = self._replay_height[lid]
                if getattr(self, '_replay_material', {}).get(lid) is not None:
                    self.layer(lid).material_map = self._replay_material[lid]
        _MUT_REV[0] += 1
        for lid in {self.stroke_by_id(s)["layer"] for s in by}:
            self._mark_replay_ok(lid)
        return n

    def palette_layer(self, create=True):
        """The dedicated palette layer -- find or create.

        Squeezing paint onto whatever layer happened to be active put the
        mounds INTO the painting, where they had to be erased afterwards. A
        palette is a separate surface you work beside the picture, so it gets
        its own layer, marked and reusable. It sits on top so it is visible
        and dippable, and it can be hidden or deleted like any other layer.
        """
        for l in self.layers:
            if getattr(l, "palette", False):
                return l
        if not create:
            return None
        lay = self.add_layer("Palette", record=False)
        lay = lay if hasattr(lay, "id") else self.layers[-1]
        lay.palette = True
        return lay

    def palette_png(self, pad=14):
        """The palette on its own, cropped to the paint, for the dock."""
        l = self.palette_layer(create=False)
        if l is None:
            return None
        a = l.pixels[..., 3]
        ys, xs = np.nonzero(a > 0.02)
        if not len(xs):
            return None
        x0, x1 = max(0, int(xs.min()) - pad), min(self.width, int(xs.max()) + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(self.height, int(ys.max()) + pad)
        # Widen the crop toward a strip-like shape. The dock draws this into a
        # wide, short canvas, and a canvas cannot letterbox -- it stretches --
        # so a crop of the wrong proportions rendered round mounds as ovals
        # (measured up to 33% vertical stretch). Giving the image the shape it
        # will be shown in is the only way to keep the paint looking like
        # paint, since the client cannot fix it without distorting something.
        want = 4.6
        w, h = x1 - x0, y1 - y0
        if w < h * want:
            grow = int((h * want - w) / 2)
            x0, x1 = max(0, x0 - grow), min(self.width, x1 + grow)
            if x1 - x0 < (y1 - y0) * want:      # ran out of room sideways
                shrink = int(((y1 - y0) - (x1 - x0) / want) / 2)
                y0, y1 = y0 + shrink, y1 - shrink
        sub = _shaded_pixels(l)[y0:y1, x0:x1]
        out = np.zeros(sub.shape, np.float32)
        out[..., :3] = 0.16          # the dock's own dark ground
        out[..., 3] = 1.0
        al = sub[..., 3:4]
        out[..., :3] = sub[..., :3] * al + out[..., :3] * (1.0 - al)
        return out, (x0, y0, x1, y1)

    def lay_palette(self, lid=None, colors=(), x=None, y=None, size=None,
                    media="oil", record=True):
        """Squeeze mounds of thick paint out onto the canvas -- a PALETTE.

        Deliberately not a new mechanism. A palette is just paint: piles thick
        enough that a brush dragged through them picks colour up and reloads,
        which is exactly what `real_brush` and `mix` already do with any thick
        passage. So the mounds are laid with the ordinary brush at a heavy
        load, and everything downstream -- dipping, gathering two colours on
        one brush, scraping a mound thinner as you take from it -- falls out
        of the physics that is already there. Dip, drag through a second
        colour, and paint: the Bob Ross loop.

        Returns the mound centres so a caller can aim at them.
        """
        if lid is None:
            lid = self.palette_layer().id
        # A palette must never spill into strata. The mounds are laid heavily
        # on purpose (they have to be a pile you can dip into), so with
        # `auto_stratum` on they overflowed and bred "Palette ~2 ~3 ~4..."
        # layers -- junk in the layer list and baffling to look at.
        _spill, self.auto_stratum = getattr(self, "auto_stratum", False), False
        h, w = self.height, self.width
        n = max(len(colors), 1)
        r = float(size) if size else max(min(h, w) * 0.055, 10.0)
        cx = float(x) if x is not None else r * 1.6
        cy = float(y) if y is not None else r * 1.6
        spots = []
        for i, c in enumerate(colors):
            px = cx + i * r * 2.7
            # several short crossing passes build a genuine mound rather than
            # one flat dab -- height has to clear the "this is a pile you can
            # dip into" threshold the reload model uses
            for k in range(5):
                a = k * 0.62
                pts = [(px + np.cos(a) * t * r * 0.5,
                        cy + np.sin(a) * t * r * 0.5)
                       for t in np.linspace(-1.0, 1.0, 9)]
                self.paint(lid, pts, color=tuple(c), radius=r * 0.55,
                           media=media, load=1.5, record=record and k == 0)
            spots.append((px, cy))
        self.auto_stratum = _spill
        return spots

    def set_paper(self, name):
        """Choose the stock: canvas, rough, cold_press, hot_press, smooth,
        linen. The substrate decides where thin paint catches, where a wash
        pools, how hard dry-brush breaks up and how much watercolour
        granulates -- one fixed field made every surface the same mid-grain
        canvas."""
        if name not in _PAPERS:
            raise ValueError("unknown paper %r -- one of %s"
                             % (name, ", ".join(sorted(_PAPERS))))
        self.paper = name
        for l in self.layers:
            l.paper = name          # shading reads it off the layer
        _MUT_REV[0] += 1
        return name

    def _res(self):
        """Whose brush is this? A palette surface borrows its owner's, so a
        dip on the palette loads the brush you then paint the picture with --
        the alternative is two reservoirs that silently disagree."""
        return getattr(self, "_brush_host", None) or self

    def palette_doc(self, create=True):
        """The palette surface: a small document of its own.

        Not a layer of the picture. It never spills into strata (its own
        `auto_stratum` stays off and it is deep enough for any mixing), it is
        never exported, and it cannot be nudged out of place by editing the
        painting. Mixing on it is ordinary painting, so blend, knife and
        wet-on-wet all work there exactly as they do on the canvas.
        """
        host = self._res()
        pd = getattr(host, "_palette_doc", None)
        if pd is not None or not create:
            return pd
        pd = Document(560, 150, background=(0.13, 0.13, 0.15))
        pd.name = "palette"
        pd.auto_stratum = False
        pd.paper = getattr(host, "paper", "canvas")
        pd._brush_host = host          # dipping loads the OWNER's brush
        host._palette_doc = pd
        return pd

    def load_brush(self, color=None, amount=1.0):
        """Dip in the palette: fill the brush right back up.

        The escape hatch from realism -- a real brush runs out, and sometimes
        you just want to keep painting. Recharging from thick paint on the
        canvas is the physical route; this is the palette."""
        h = self._res()
        h.brush_charge = float(np.clip(amount, 0.0, 1.0))
        h.brush_charges = np.full(_BRUSH_LANES, h.brush_charge, np.float32)
        if color is not None:
            c = _f32(color)
            if c.reshape(-1).size == 3:
                h.brush_color = tuple(float(v) for v in c.reshape(3))
                h.brush_lanes = np.repeat(
                    np.asarray(h.brush_color, np.float32).reshape(1, 3),
                    _BRUSH_LANES, 0)
            else:
                # a colour PER LANE: load one edge of the tuft differently
                # from the other without having to go and dip for it
                h.brush_lanes = _resize(c.reshape(1, -1, 3), 1,
                                        _BRUSH_LANES)[0].astype(np.float32)
                h.brush_color = tuple(
                    float(v) for v in h.brush_lanes.mean(0))
        return {"charge": h.brush_charge, "color": list(h.brush_color),
                "lanes": [[float(v) for v in row]
                          for row in getattr(h, "brush_lanes", [])]}

    def brush_state(self):
        """What is on the brush right now, for the UI's charge meter."""
        h = self._res()
        return {"charge": float(h.brush_charge),
                "color": [float(v) for v in h.brush_color],
                "lanes": [[float(v) for v in row]
                          for row in getattr(h, "brush_lanes", [])]}

    def group_strokes(self, ids, name=None, record=True):
        """Bundle strokes (and/or existing groups) into one editable object."""
        sids = self._expand_strokes(ids)
        if not sids:
            return None
        if record:
            self.record("Group strokes")
        # a stroke belongs to at most one group, or "move the group" would be
        # ambiguous the moment two groups overlapped
        for g in self.stroke_groups:
            g["strokes"] = [s for s in g["strokes"] if s not in sids]
        self.stroke_groups = [g for g in self.stroke_groups if g["strokes"]]
        gid = "G%d" % (len(self.stroke_groups) + 1)
        while any(g["id"] == gid for g in self.stroke_groups):
            gid += "x"
        self.stroke_groups.append(
            {"id": gid, "name": name or "Group", "strokes": list(sids)})
        return gid

    def ungroup_strokes(self, gid, record=True):
        """Dissolve a group. The strokes themselves are untouched -- grouping
        never rewrote them, it only said they belong together."""
        g = next((x for x in self.stroke_groups if x["id"] == gid), None)
        if g is None:
            raise KeyError(gid)
        if record:
            self.record("Ungroup strokes")
        self.stroke_groups.remove(g)
        return list(g["strokes"])

    def stroke_group_of(self, sid):
        """Which group a stroke belongs to, or None -- so a click on any
        member can select the whole passage."""
        for g in self.stroke_groups:
            if sid in g["strokes"]:
                return g["id"]
        return None

    def _expand_strokes(self, ids):
        """Resolve a mixed list of stroke ids and group ids to stroke ids.

        Every stroke editor runs through this, so passing a group id anywhere
        a stroke id is accepted moves the whole passage as one unit. Order is
        preserved and duplicates dropped: a group and one of its own members
        in the same selection must not transform that member twice.
        """
        out, seen = [], set()
        live = {k["id"] for k in self.strokes}
        for i in (ids or ()):
            members = next((g["strokes"] for g in self.stroke_groups
                            if g["id"] == i), None)
            for sid in (members if members is not None else [i]):
                if sid not in seen and sid in live:
                    seen.add(sid)
                    out.append(sid)
        return out

    def transform_strokes(self, ids, sx=1.0, sy=1.0, deg=0.0, dx=0.0, dy=0.0,
                          cx=None, cy=None, record=True):
        """Affine-map whole strokes: scale/rotate about (cx, cy) -- the
        selection's centre when not given -- then translate. The paths are the
        master, so the result re-renders as crisply as it was painted; scaling
        also scales the brush radius so ink weight stays proportional."""
        import math
        ks = [self.stroke_by_id(s) for s in self._expand_strokes(ids)]
        if not ks:
            return 0
        lids = {k["layer"] for k in ks}
        for _l in lids:
            self._locked_guard(_l)
        self._stroke_edit_guard(lids)
        if cx is None or cy is None:
            xs = [p[0] for k in ks for p in k["points"]]
            ys = [p[1] for k in ks for p in k["points"]]
            cx = (min(xs) + max(xs)) / 2.0 if cx is None else cx
            cy = (min(ys) + max(ys)) / 2.0 if cy is None else cy
        if record:
            self.record("Transform strokes", only=list(lids))
        r = math.radians(float(deg))
        co, si = math.cos(r), math.sin(r)
        n = 0
        for k in ks:
            for p in k["points"]:
                px = (p[0] - cx) * float(sx)
                py = (p[1] - cy) * float(sy)
                p[0] = cx + px * co - py * si + float(dx)
                p[1] = cy + px * si + py * co + float(dy)
                n += 1
            # uniform-ish scale carries into the brush radius; a mirrored or
            # squashed stroke keeps its painted weight rather than inverting
            f = (abs(float(sx)) + abs(float(sy))) / 2.0
            if abs(f - 1.0) > 1e-6:
                k["brush"]["radius"] = max(0.5, float(k["brush"].get("radius", 8.0)) * f)
        for lid in lids:
            self._rebuild_after_stroke_edit(lid)
        return n

    def duplicate_strokes(self, ids, dx=14.0, dy=14.0, layer=None, record=True):
        """Copy strokes (optionally onto another layer, offset so the copy is
        visible). The copies are full strokes -- selectable, nudgeable,
        transformable -- not a pixel stamp.

        The target layer is rebuilt from replay when that is faithful;
        otherwise the copies are painted ON TOP, which adds ink without
        touching what is already there (and simply leaves that layer
        non-replayable, exactly as it already was)."""
        ks = [self.stroke_by_id(s) for s in (ids or ())]
        if not ks:
            return []
        tgt = layer or ks[0]["layer"]
        self.layer(tgt)                                  # raises on a bad id
        if record:
            self.record("Duplicate strokes", only=[tgt])
        if not any(k["layer"] == tgt for k in self.strokes):
            self._capture_replay_base(tgt)       # fresh target: base = as-is
        faithful = self.replay_is_faithful(tgt)
        out = []
        for k in ks:
            self._stroke_n = getattr(self, "_stroke_n", len(self.strokes)) + 1
            nk = {"id": "K%d" % self._stroke_n, "layer": tgt,
                  "points": [[p[0] + float(dx), p[1] + float(dy)] + list(p[2:])
                             for p in k["points"]],
                  "brush": dict(k["brush"])}
            self.strokes.append(nk)
            out.append(nk["id"])
        if len(self.strokes) > self.MAX_STROKES:
            del self.strokes[:len(self.strokes) - self.MAX_STROKES]
        if faithful:
            self._rebuild_after_stroke_edit(tgt)
        else:
            for sid in out:
                k = self.stroke_by_id(sid)
                b = k["brush"]
                self.paint(tgt, [tuple(p[:2]) for p in k["points"]],
                           color=tuple(b.get("color", (0, 0, 0))),
                           radius=float(b.get("radius", 8.0)),
                           opacity=float(b.get("opacity", 1.0)),
                           erase=bool(b.get("erase")),
                           hardness=float(b.get("hardness", 0.7)),
                           record=False, stroke_new=False)
            _MUT_REV[0] += 1
        return out

    def strokes_hit(self, lid, points, radius=12.0):
        """Strokes on a layer whose PATH passes within reach of any of the
        given eraser points (reach = eraser radius + the stroke's own
        width). Returns ids in paint order."""
        ep = np.asarray([(p[0], p[1]) for p in points], np.float32)
        out = []
        for k in self.strokes:
            if k["layer"] != lid or not k["points"]:
                continue
            raw = np.asarray([(p[0], p[1]) for p in k["points"]],
                             np.float32)
            # a stroke stores WAYPOINTS; the ink lives on the segments
            # between them, so densify before measuring (a point-to-point
            # test missed a 160px straight line entirely)
            if len(raw) > 1:
                segs = [raw[:1]]
                for i in range(1, len(raw)):
                    seg = raw[i] - raw[i - 1]
                    n = max(2, int(np.hypot(*seg) / 4.0) + 1)
                    tt = np.linspace(0, 1, n)[1:, None]
                    segs.append(raw[i - 1][None, :] + seg[None, :] * tt)
                sp = np.concatenate(segs, 0)
            else:
                sp = raw
            reach = float(radius) + float(k["brush"].get("radius", 6.0))
            d2 = ((sp[:, None, 0] - ep[None, :, 0]) ** 2
                  + (sp[:, None, 1] - ep[None, :, 1]) ** 2)
            if float(d2.min()) <= reach * reach:
                out.append(k["id"])
        return out

    def erase_strokes(self, lid, points, radius=12.0, topmost=False):
        """The stroke ERASER: everything whose path the eraser touches is
        removed whole (its ink comes out from under later strokes via the
        faithful replay). topmost=True peels only the LAST-painted touched
        stroke -- one layer of paint at a time, like lifting the newest
        coat."""
        ids = self.strokes_hit(lid, points, radius)
        if not ids:
            return 0
        if topmost:
            ids = ids[-1:]
        return self.delete_strokes(ids)

    def erase_restore(self, lid, points, radius=12.0, record=True):
        """PAINT UNDO as a brush: the area under the eraser returns to the
        layer's replay BASE (its pre-stroke state) instead of turning
        transparent -- a local undo, feathered at the rim. Layers with no
        recorded base restore toward empty, which equals classic erase."""
        l = self.layer(lid)
        h, w = self.height, self.width
        if record:
            self.record("Paint undo", only=[lid])
        base = getattr(self, "_replay_base", {}).get(lid)
        if base is None:
            base = np.zeros((h, w, 4), np.float32)
        pts = np.asarray([(p[0], p[1]) for p in points], np.float32)
        if not len(pts):
            return
        r = float(max(radius, 2.0))
        x0 = int(max(0, pts[:, 0].min() - r * 1.6))
        y0 = int(max(0, pts[:, 1].min() - r * 1.6))
        x1 = int(min(w, pts[:, 0].max() + r * 1.6))
        y1 = int(min(h, pts[:, 1].max() + r * 1.6))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        for p in pts:
            dd = np.sqrt((xx - p[0]) ** 2 + (yy - p[1]) ** 2)
            m = np.maximum(m, np.clip(1.15 - dd / r, 0.0, 1.0))
        m = np.clip(m, 0, 1)[..., None]
        win = l.pixels[y0:y1, x0:x1]
        win[...] = base[y0:y1, x0:x1] * m + win * (1 - m)
        _MUT_REV[0] += 1

    def erase_depth(self, lid, points, radius=12.0, strength=1.0,
                    record=True):
        """The DEPTH eraser carves the paint BODY before it cuts colour:
        impasto relief under the eraser is scraped down first, and only
        once the height is gone does the alpha start to lift -- a palette
        knife, not a rubber."""
        l = self.layer(lid)
        h, w = self.height, self.width
        if record:
            self.record("Erase depth", only=[lid])
        pts = np.asarray([(p[0], p[1]) for p in points], np.float32)
        if not len(pts):
            return
        r = float(max(radius, 2.0))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        m = np.zeros((h, w), np.float32)
        for p in pts:
            dd = np.sqrt((xx - p[0]) ** 2 + (yy - p[1]) ** 2)
            m = np.maximum(m, np.clip(1.15 - dd / r, 0.0, 1.0))
        carve = m * float(strength) * 3.0
        if l.height_map is None:
            hm = np.zeros((h, w), np.float32)
        else:
            hm = np.asarray(l.height_map, np.float32)
        taken = np.minimum(hm, carve)
        hm = hm - taken
        overflow = np.clip((carve - taken) / max(float(strength) * 3.0,
                                                 1e-4), 0, 1)
        l.height_map = hm
        l.pixels[..., 3] = l.pixels[..., 3] * (1.0 - overflow * 0.65)
        if getattr(l, "material_map", None) is not None:
            # the palette knife takes the STUFF with the body it scrapes:
            # material coverage erodes with the same overflow that lifts the
            # pigment, so a scraped-clean patch stops gleaming
            l.material_map[..., 2] *= (1.0 - overflow * 0.65)
        _MUT_REV[0] += 1

    def delete_strokes(self, ids, record=True):
        """Remove strokes and their ink. Needs a faithful replay: the only way
        to take a stroke's ink OUT from under later overlapping strokes is to
        rebuild the layer without it."""
        ks = [self.stroke_by_id(s) for s in (ids or ())]
        if not ks:
            return 0
        lids = {k["layer"] for k in ks}
        self._stroke_edit_guard(lids)
        if record:
            self.record("Delete strokes", only=list(lids))
        for k in ks:
            self.strokes.remove(k)
        for lid in lids:
            self._rebuild_after_stroke_edit(lid)
        return len(ks)

    def strokes_to_layer(self, ids, layer, record=True):
        """Move strokes to another layer: their ink leaves the source (which
        therefore must replay faithfully) and lands on the target."""
        ks = [self.stroke_by_id(s) for s in (ids or ())]
        if not ks:
            return 0
        self.layer(layer)                                # raises on a bad id
        srcs = {k["layer"] for k in ks} - {layer}
        self._stroke_edit_guard(srcs)
        if record:
            self.record("Strokes to layer", only=list(srcs | {layer}))
        if not any(k["layer"] == layer for k in self.strokes):
            self._capture_replay_base(layer)     # fresh target: base = as-is
        tgt_faithful = self.replay_is_faithful(layer)
        for k in ks:
            k["layer"] = layer
        for lid in srcs:
            self._rebuild_after_stroke_edit(lid)
        if tgt_faithful:
            self._rebuild_after_stroke_edit(layer)
        else:
            for k in ks:
                b = k["brush"]
                self.paint(layer, [tuple(p[:2]) for p in k["points"]],
                           color=tuple(b.get("color", (0, 0, 0))),
                           radius=float(b.get("radius", 8.0)),
                           opacity=float(b.get("opacity", 1.0)),
                           erase=bool(b.get("erase")),
                           hardness=float(b.get("hardness", 0.7)),
                           record=False, stroke_new=False)
            _MUT_REV[0] += 1
        return len(ks)

    def smooth_strokes(self, ids, amount=0.5, iterations=2, record=True):
        """Laplacian-relax stroke points: each interior point eases toward the
        midpoint of its neighbours. This is the "clean up my tangents" gesture
        -- shaky freehand corners round off while endpoints stay pinned."""
        ks = [self.stroke_by_id(s) for s in (ids or ())]
        if not ks:
            return 0
        lids = {k["layer"] for k in ks}
        self._stroke_edit_guard(lids)
        if record:
            self.record("Smooth strokes", only=list(lids))
        a = max(0.0, min(1.0, float(amount)))
        n = 0
        for k in ks:
            pts = k["points"]
            for _ in range(max(1, int(iterations))):
                if len(pts) < 3:
                    break
                prev = [list(p) for p in pts]
                for i in range(1, len(pts) - 1):
                    mx = (prev[i - 1][0] + prev[i + 1][0]) / 2.0
                    my = (prev[i - 1][1] + prev[i + 1][1]) / 2.0
                    pts[i][0] += (mx - pts[i][0]) * a
                    pts[i][1] += (my - pts[i][1]) * a
            n += len(pts)
        for lid in lids:
            self._rebuild_after_stroke_edit(lid)
        return n

    @staticmethod
    def _relax_lengths(pts, rest, pins, iterations=24, stiffness=1.0):
        """Pull segment lengths back toward `rest` with `pins` held fixed --
        the same constraint step the physics uses, minus gravity and time."""
        pins = set(pins)
        for it in range(max(1, int(iterations))):
            # alternate sweep direction: a one-way Gauss-Seidel pass piles the
            # residual against whichever end it sweeps FROM (measured: 33px on
            # the pinned segment vs 27 elsewhere on an over-stretched rope);
            # ping-ponging spreads it evenly
            order = (range(len(rest)) if it % 2 == 0
                     else range(len(rest) - 1, -1, -1))
            for b in order:
                a, c = pts[b], pts[b + 1]
                ddx = c[0] - a[0]
                ddy = c[1] - a[1]
                dist = (ddx * ddx + ddy * ddy) ** 0.5
                if dist < 1e-9:
                    continue
                corr = (dist - rest[b]) / dist * 0.5 * float(stiffness)
                ax, ay = ddx * corr, ddy * corr
                if b not in pins:
                    a[0] += ax
                    a[1] += ay
                if (b + 1) not in pins:
                    c[0] -= ax
                    c[1] -= ay

    def _sync_rig_after_edit(self, k):
        """A direct edit repositions joints; the verlet history must follow or
        the next sim step snaps everything back to where it used to be."""
        rig = k.get("rig")
        if rig:
            rig["prev"] = [list(p) for p in k["points"]]

    def pull_stroke(self, sid, index, tx, ty, iterations=120, record=True):
        """Grab one joint and PULL: the joint goes to (tx, ty) and the rest of
        the stroke follows under its segment-length constraints, like a thread
        dragged across a table. Rigged strokes use their rig -- rest lengths
        and pins are honoured, so pinned joints hold and the slack drapes
        between them. Unrigged strokes get rest lengths from their current
        geometry with no pins, which is exactly the loose-thread case: pull an
        end and the whole stroke slides after it."""
        k = self.stroke_by_id(sid)
        self._stroke_edit_guard([k["layer"]])
        pts = k["points"]
        i = int(index)
        if not (0 <= i < len(pts)):
            raise ValueError("no such joint")
        if record:
            self.record("Pull stroke", only=[k["layer"]])
        rig = k.get("rig")
        if rig and len(rig.get("bones", [])) == len(pts) - 1:
            rest = list(rig["bones"])
            pins = set(int(p) for p in rig.get("pins", []))
        else:
            rest = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
                    for a, b in zip(pts, pts[1:])]
            pins = set()
        pts[i][0] = float(tx)
        pts[i][1] = float(ty)
        pins_now = pins | {i}          # the grab is a pin AT the target
        self._relax_lengths(pts, rest, pins_now, iterations=iterations)
        self._sync_rig_after_edit(k)
        self._rebuild_after_stroke_edit(k["layer"])
        return len(pts)

    def split_stroke(self, sid, index):
        """Cut a stroke in two at a point index. The point is shared by both
        halves so the ink stays continuous where it was cut."""
        k = self.stroke_by_id(sid)
        i = int(index)
        if i < 1 or i > len(k["points"]) - 2:
            raise ValueError("split index must leave at least 2 points a side")
        self.record("Split stroke", only=[k["layer"]])   # pixels unchanged, but
        pos = self.strokes.index(k)                      # the records must undo
        self._stroke_n = getattr(self, "_stroke_n", len(self.strokes)) + 1
        tail = {"id": "K%d" % self._stroke_n, "layer": k["layer"],
                "points": [list(p) for p in k["points"][i:]],
                "brush": dict(k["brush"])}
        k["points"] = [list(p) for p in k["points"][:i + 1]]
        self.strokes.insert(pos + 1, tail)       # keep paint order meaningful
        _MUT_REV[0] += 1
        return [k["id"], tail["id"]]

    def join_strokes(self, ids, gap=1e9):
        """Merge strokes into one, in paint order.

        Refuses unless every stroke shares the same brush and the same layer --
        a join that had to discard one stroke's colour or radius would be
        losing work silently. `gap` optionally requires the ends to be close."""
        ids = list(ids or ())
        if len(ids) < 2:
            raise ValueError("need at least two strokes to join")
        ks = [self.stroke_by_id(s) for s in ids]
        ks.sort(key=lambda k: self.strokes.index(k))
        base = ks[0]
        for k in ks[1:]:
            if k["layer"] != base["layer"]:
                raise ValueError("strokes are on different layers")
            if not self.brush_compatible(base, k):
                raise ValueError("strokes use different brush settings")
        for a, b in zip(ks, ks[1:]):
            ax, ay = a["points"][-1]
            bx, by = b["points"][0]
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > gap:
                raise ValueError("stroke ends are too far apart to join")
        self.record("Join strokes", only=[base["layer"]])
        pts = [list(p) for p in base["points"]]
        for k in ks[1:]:
            nxt = k["points"]
            if nxt and pts and pts[-1] == list(nxt[0]):
                nxt = nxt[1:]                    # don't duplicate a shared point
            pts.extend(list(p) for p in nxt)
        base["points"] = pts
        for k in ks[1:]:
            self.strokes.remove(k)
        _MUT_REV[0] += 1
        return base["id"]

    # ---- strokes as armatures: joints (points) linked by bones (segments) ----

    def rig_stroke(self, sid, pins=None):
        """Turn a stroke into an armature: its points become JOINTS and the
        segments between them become BONES with a rest length.

        Rest lengths are what separate this from a particle sim -- enforcing
        them means the stroke swings and drapes instead of stretching apart,
        which is what makes it read as a chain of bones rather than dust.
        `pins` are joint indices that stay put (default: the first joint, so a
        stroke hangs from where it started)."""
        k = self.stroke_by_id(sid)
        pts = k["points"]
        if len(pts) < 2:
            raise ValueError("a stroke needs at least two joints to rig")
        bones = [((pts[i + 1][0] - pts[i][0]) ** 2 +
                  (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
                 for i in range(len(pts) - 1)]
        k["rig"] = {"bones": bones,
                    "pins": sorted(set(int(p) for p in (pins if pins is not None
                                                        else [0]))),
                    "prev": [list(p) for p in pts],       # verlet history
                    "keys": k.get("rig", {}).get("keys", {})}
        return {"joints": len(pts), "bones": len(bones), "pins": k["rig"]["pins"]}

    def simulate_stroke(self, sid, steps=1, gravity=(0.0, 60.0), wind=0.0,
                        damping=0.02, stiffness=1.0, iterations=None, seed=0,
                        record=False, wind_detail=12):
        """Advance a rigged stroke: Verlet integration, then relax the bone
        lengths back toward rest. Pinned joints never move."""
        k = self.stroke_by_id(sid)
        self._stroke_edit_guard([k["layer"]])
        rig = k.get("rig")
        if not rig:
            raise ValueError("stroke is not rigged")
        if record:
            # Coalesce a continuous run: holding "Drop" is ONE action to the
            # user, but each press was taking its own history slot -- six
            # presses ate a quarter of the 24-entry stack and pushed the edit
            # you actually wanted back off the end.
            tag = ("sim", k["layer"], sid)
            if getattr(self, "_sim_run", None) != tag:
                self.record("Simulate stroke", only=[k["layer"]])
                self._sim_run = tag       # set AFTER record(), which clears it
        pts = k["points"]
        prev = rig["prev"]
        pins = set(rig["pins"])
        bones = rig["bones"]
        gx, gy = float(gravity[0]), float(gravity[1])
        # leCore's CurlWind is a divergence-free (volume-preserving) turbulent
        # field with a per-point force -- the strand ripples instead of
        # ballooning, and every joint samples the field at its OWN position, so
        # the tip lags the root exactly as Maya's Paint Effects describes. Fall
        # back to the sampled 2-D curl field on builds without it.
        cwind = None
        curl = None
        if wind:
            if have("hair_wind"):
                try:
                    # MEASURED build cost of the wind field: res 8 = 0.6 s,
                    # 12 = 1.9 s, 16 = 7.5 s, 24 = 24 s, while the per-point
                    # force varies about the same from res 12 up. 12 is the
                    # knee -- a 4x saving on the first use for no visible loss.
                    # Exposed as a dial rather than hidden, since a still frame
                    # can afford more detail than an interactive drag.
                    cwind = _hair_wind(
                        float(wind), max(4, min(32, int(wind_detail))),
                        ((0, max(self.width, 1)),
                         (0, max(self.height, 1)), (-1, 1)),
                        2, int(seed))
                except Exception:
                    cwind = None
            if cwind is None:
                curl = _curl_noise(32, 3, int(seed))
        dt2 = 0.0016
        # A Gauss-Seidel pass propagates a correction one bone along the chain,
        # so a short fixed iteration count leaves a long stroke visibly
        # stretched -- the correction never reaches the far end. Scale with the
        # chain, and alternate sweep direction so neither end is favoured.
        if iterations is None:
            iterations = max(8, min(60, len(bones) * 2))
        shortest = min(bones) if bones else 1.0
        max_step = max(1.0, shortest * 0.5)      # no joint may outrun a bone
        class _Strand(object):                    # what CurlWind.force expects
            __slots__ = ("points",)

        for _ in range(max(1, int(steps))):
            wf = None
            if cwind is not None:
                st = _Strand()
                st.points = np.array([[p[0], p[1], 0.0] for p in pts], float)
                try:
                    wf = np.asarray(cwind.force(st), float)
                except Exception:
                    wf = None
            for i, p in enumerate(pts):                       # integrate
                if i in pins:
                    prev[i] = list(p)
                    continue
                fx, fy = gx, gy
                if wf is not None:
                    fx += float(wf[i][0])
                    fy += float(wf[i][1])
                elif curl is not None:
                    ix = int(min(31, max(0, p[0] / max(self.width, 1) * 31)))
                    iy = int(min(31, max(0, p[1] / max(self.height, 1) * 31)))
                    fx += float(curl[0][iy][ix]) * wind
                    fy += float(curl[1][iy][ix]) * wind
                vx = (p[0] - prev[i][0]) * (1.0 - damping)
                vy = (p[1] - prev[i][1]) * (1.0 - damping)
                sx = vx + fx * dt2
                sy = vy + fy * dt2
                mag = (sx * sx + sy * sy) ** 0.5     # keep the solver stable
                if mag > max_step:
                    sx *= max_step / mag
                    sy *= max_step / mag
                prev[i] = list(p)
                p[0] += sx
                p[1] += sy
            for it in range(max(1, int(iterations))):        # keep bone lengths
                order = range(len(bones)) if (it % 2 == 0) else \
                    range(len(bones) - 1, -1, -1)
                for b in order:
                    rest = bones[b]
                    a, c = pts[b], pts[b + 1]
                    dx, dy = c[0] - a[0], c[1] - a[1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < 1e-9:
                        continue
                    corr = (d - rest) / d * 0.5 * float(stiffness)
                    ax, ay = dx * corr, dy * corr
                    if b not in pins:
                        a[0] += ax; a[1] += ay
                    if (b + 1) not in pins:
                        c[0] -= ax; c[1] -= ay
        rebuilt = self.replay_layer(k["layer"])
        if rebuilt is not None:
            self.layer(k["layer"]).pixels = rebuilt
        _MUT_REV[0] += 1
        self._mark_replay_ok(k["layer"])
        return len(pts)

    def key_stroke(self, sid, t):
        """Store the current joint positions as a keyframe at time `t`."""
        k = self.stroke_by_id(sid)
        rig = k.setdefault("rig", {}).setdefault("keys", {})
        rig[str(float(t))] = [list(p) for p in k["points"]]
        return sorted(float(x) for x in rig)

    def apply_stroke_keys(self, sid, t, interp="linear"):
        """Pose a stroke at time `t`, interpolating between its keyframes and
        clamping outside the keyed range.

        `interp` uses leCore's Timeline when available: 'smooth' (ease in-out),
        'ease_in', 'ease_out' or 'step' beside the default 'linear'. Straight
        linear motion is the giveaway of a machine-made animation; an ease is
        what makes a pose change read as movement rather than a slide. Falls
        back to linear on an older engine, so nothing breaks -- it just stays
        linear."""
        k = self.stroke_by_id(sid)
        self._stroke_edit_guard([k["layer"]])
        keys = (k.get("rig") or {}).get("keys") or {}
        if not keys:
            raise ValueError("stroke has no keyframes")
        times = sorted(float(x) for x in keys)
        t = float(t)
        if t <= times[0]:
            pose = keys[str(times[0])]
        elif t >= times[-1]:
            pose = keys[str(times[-1])]
        else:
            hi = next(x for x in times if x >= t)
            lo = max(x for x in times if x <= t)
            a, b = keys[str(lo)], keys[str(hi)]
            u = 0.0 if hi == lo else (t - lo) / (hi - lo)
            if interp != "linear" and have("timeline"):
                try:
                    tl = mind().timeline()
                    tl.key("u", 0.0, 0.0, interp=interp)
                    tl.key("u", 1.0, 1.0, interp=interp)
                    u = float(tl.sample("u", float(u)))
                except Exception:
                    pass                      # older engine or bad easing name
            pose = [[p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u]
                    for p, q in zip(a, b)]
        k["points"] = [list(p) for p in pose]
        if k.get("rig"):
            k["rig"]["prev"] = [list(p) for p in pose]
        rebuilt = self.replay_layer(k["layer"])
        if rebuilt is not None:
            self.layer(k["layer"]).pixels = rebuilt
        _MUT_REV[0] += 1
        self._mark_replay_ok(k["layer"])
        return times

    def set_point_width(self, sid, index, w, spread=0):
        self._stroke_edit_guard([self.stroke_by_id(sid)["layer"]])
        """Set the width multiplier at a joint, optionally tapering outward.

        Width is the third component of a point, and the renderer already
        scales every dab by it -- recording was the only place it was being
        dropped. `spread` blends back toward 1.0 over that many neighbours,
        which is what makes a change read as a swell rather than a step."""
        k = self.stroke_by_id(sid)
        pts = k["points"]
        i = int(index)
        if not 0 <= i < len(pts):
            raise IndexError("no such point")
        w = max(0.0, float(w))
        rng = int(max(0, spread))
        for j in range(max(0, i - rng), min(len(pts), i + rng + 1)):
            t = 1.0 if rng == 0 else 1.0 - abs(j - i) / float(rng + 1)
            val = 1.0 + (w - 1.0) * t
            p = pts[j]
            if len(p) > 2:
                p[2] = val
            else:
                p.append(val)
        self._rebuild_after_stroke_edit(k["layer"])
        return [(p[2] if len(p) > 2 else 1.0) for p in pts]

    def taper_stroke(self, sid, tip=0.15, root=1.0):
        """Taper root to tip -- what makes a rigged stroke read as hair or a
        vine rather than a length of wire."""
        k = self.stroke_by_id(sid)
        self._stroke_edit_guard([k["layer"]])
        pts = k["points"]
        n = max(1, len(pts) - 1)
        for i, p in enumerate(pts):
            val = float(root) + (float(tip) - float(root)) * (i / n)
            if len(p) > 2:
                p[2] = val
            else:
                p.append(val)
        self._rebuild_after_stroke_edit(k["layer"])
        return len(pts)

    def _rebuild_after_stroke_edit(self, lid):
        rebuilt = self.replay_layer(lid)
        if rebuilt is not None:
            self.layer(lid).pixels = rebuilt
            if getattr(self, '_replay_height', {}).get(lid) is not None:
                self.layer(lid).height_map = self._replay_height[lid]
            if getattr(self, '_replay_material', {}).get(lid) is not None:
                self.layer(lid).material_map = self._replay_material[lid]
        _MUT_REV[0] += 1
        self._mark_replay_ok(lid)

    def render_strokes_rgba(self, ids):
        """Just these strokes on transparency -- the transform tool's drag
        preview. Painted with the strokes' own brushes onto a blank layer, so
        the preview weighs exactly what the ink does."""
        blank = np.zeros((self.height, self.width, 4), np.float32)
        lid = "__stroke_preview__"

        class _Tmp:
            id = lid
            pixels = blank
        self.layers.append(_Tmp())
        try:
            self._replaying = True                 # never re-record these
            try:
                for sid in ids:
                    k = self.stroke_by_id(sid)
                    b = k["brush"]
                    self.paint(lid, [tuple(p[:2]) for p in k["points"]],
                               color=tuple(b.get("color", (0, 0, 0))),
                               radius=float(b.get("radius", 8.0)),
                               opacity=float(b.get("opacity", 1.0)),
                               erase=bool(b.get("erase")),
                               hardness=float(b.get("hardness", 0.7)),
                               record=False)
            finally:
                self._replaying = False
        finally:
            self.layers.pop()
        return blank

    def stroke_meta(self, sid):
        k = self.stroke_by_id(sid)
        xs = [p[0] for p in k["points"]] or [0.0]
        ys = [p[1] for p in k["points"]] or [0.0]
        b = k["brush"]
        return {"id": k["id"], "layer": k["layer"], "n": len(k["points"]),
                "radius": float(b.get("radius", 8.0)),
                "erase": bool(b.get("erase")),
                "rigged": bool(k.get("rig")),
                "pins": [int(i) for i in (k.get("rig") or {}).get("pins", [])],
                "color": [float(c) for c in b.get("color", (0, 0, 0))],
                "bbox": [min(xs), min(ys), max(xs), max(ys)]}

    def stroke_by_id(self, sid):
        for k in self.strokes:
            if k["id"] == sid:
                return k
        raise KeyError(sid)

    def clear(self, lid, selection=None, sel_invert=False, record=True):
        """Erase the selection's contents on a layer (Delete in every editor).

        Alpha only: the colour underneath is left alone, so undo restores it
        exactly. With no selection the whole layer is cleared."""
        # Announce the change even when NOT recording undo. record() bumps the
        # mutation counter, and caches key on it -- so an unrecorded edit was
        # invisible to them. Mid-stroke flushes (and the FINAL flush of every
        # stroke) pass record=False, which left the node graph showing a stroke
        # missing its last segment until some other edit happened to bump it.
        _MUT_REV[0] += 1
        if record:
            self.record("Clear", only=[lid])
        l = self.layer(lid)
        h, w = self.height, self.width
        if selection:
            sv = _resize(self.gate_by_id(selection).data, h, w)
            if sel_invert:
                sv = 1.0 - sv
            l.pixels[..., 3:4] = l.pixels[..., 3:4] * (1.0 - sv[..., None])
        else:
            l.pixels[..., 3:4] = 0.0

    # --- painting --------------------------------------------------------------------------------
    def _locked_guard(self, lid):
        """A LOCKED layer refuses every edit -- pixels, strokes, pose,
        placement -- not just transparency (that is alpha_lock's job,
        and conflating the two was exactly the confusion reported:
        'locking a layer doesn't prevent edits like it should')."""
        l = self.layer(lid)
        if getattr(l, "locked", False):
            raise ValueError("layer %r is locked -- unlock it to edit"
                             % l.name)

    def paint(self, lid, points, color=(0, 0, 0), radius=8.0, opacity=1.0,
              erase=False, hardness=0.7, record=True,
              selection=None, sel_invert=False, brush=None, target_mask=None,
              stroke_new=None, media=None, load=0.6, alpha_lock=None,
              taper=0.0, material=None, mix=0.0, real_brush=False,
              charge0=None, lanes0=None):
        self._locked_guard(lid)
        self.layer(lid).paper = _paper_of(self)[0]   # stock this sits on
        matdef = _resolve_material(material)   # raises on an unknown name
        # FREEZE the reservoir HERE, at the top, before any branch. The brush
        # carries a colour and a charge per band across its width; recording a
        # single scalar meant a replay restarted every lane equal and diverged
        # from the original, quietly changing the picture on rebuild. It has
        # to happen before the deposit (which updates the live reservoir) and
        # outside the media branch (a real-brush stroke with no medium never
        # enters that branch but still writes a record).
        _rb_c0, _rb_lanes = None, None
        _host = self._res()          # a palette surface uses its owner's brush
        if real_brush:
            _c = charge0 if charge0 is not None else getattr(
                _host, "brush_charges", None)
            if _c is None:
                _c = float(_host.brush_charge)
            _rb_c0 = [float(v) for v in
                      np.atleast_1d(np.asarray(_c, np.float32)).reshape(-1)]
            if len(_rb_c0) != _BRUSH_LANES:
                _rb_c0 = [_rb_c0[0]] * _BRUSH_LANES
            _l = lanes0 if lanes0 is not None else getattr(
                _host, "brush_lanes", None)
            if _l is None or np.asarray(_l, np.float32).size < 3:
                _l = np.repeat(_f32(color).reshape(1, 3), _BRUSH_LANES, 0)
            _l = np.asarray(_l, np.float32).reshape(-1, 3)
            if len(_l) != _BRUSH_LANES:
                _l = np.repeat(_l[:1], _BRUSH_LANES, 0)
            _rb_lanes = [[float(v) for v in row] for row in _l]
        rev_entry = _MUT_REV[0]      # cache-patch validity: pre-edit revision
        if taper and len(points) > 2 and not any(
                len(p) > 2 for p in points):
            # LIVE TAPER: materialise as per-point pressure at entry --
            # width ramps 0->1 over the first `taper` fraction of the
            # stroke's arc length and back down over the last. Because
            # the pressures are written into the points themselves, the
            # recorded stroke replays with its taper for free, and the
            # post-hoc joint tools keep working on top.
            import numpy as _np
            arr = _np.asarray([[p[0], p[1]] for p in points], _np.float32)
            seg = _np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(1))
            t = _np.concatenate([[0.0], _np.cumsum(seg)])
            total = float(t[-1]) or 1.0
            t = t / total
            T = float(np.clip(taper, 0.02, 0.5))
            f = _np.minimum(1.0, _np.minimum(t / T, (1.0 - t) / T))
            f = _np.clip(f, 0.06, 1.0)      # a hair, never a zero stamp
            points = [[float(p[0]), float(p[1]), float(f[i])]
                      for i, p in enumerate(points)]
        faith_entry = bool(record) and self._replay_ok_cached(lid)
        if alpha_lock is None:
            # resolve the layer's lock NOW: the stroke record is written
            # before the pixel work, and the record must carry the effective
            # lock or a replay lays unlocked paint (found by the faithfulness
            # gate: replay recolored pixels the lock had frozen)
            alpha_lock = bool(getattr(self.layer(lid), "alpha_lock", False))
        """Stamp a stroke (list of (x, y)) with a soft round brush.

        target_mask paints the stroke into a MASK instead of the layer's
        pixels -- the standard non-destructive workflow: white reveals, black
        hides, and the eraser hides. The coverage maths is identical; only the
        write differs, so a mask stroke feels exactly like a pixel stroke."""
        # Announce the change even when NOT recording undo. record() bumps the
        # mutation counter, and caches key on it -- so an unrecorded edit was
        # invisible to them. Mid-stroke flushes (and the FINAL flush of every
        # stroke) pass record=False, which left the node graph showing a stroke
        # missing its last segment until some other edit happened to bump it.
        _MUT_REV[0] += 1
        # ORDER MATTERS: the undo snapshot must be taken BEFORE the stroke is
        # recorded. Snapshots carry the stroke list, so a snapshot taken after
        # record_stroke() already contains the new stroke -- undoing then
        # restored the pixels but left a GHOST stroke record behind. From that
        # point the layer never matched its own replay again: nudge refused
        # with "content that was not painted as strokes", and a rebuild-based
        # edit would have re-applied the undone ink. One undo after painting
        # was enough to trigger it.
        if record:
            # a stroke touches ONE layer (or a mask, whose data is snapshotted
            # in full either way) -- no need to copy every other layer's pixels
            pts = [(float(p[0]), float(p[1])) for p in points] or [(0.0, 0.0)]
            pad = float(radius) + 3.0
            rx0 = max(0, int(min(p[0] for p in pts) - pad))
            ry0 = max(0, int(min(p[1] for p in pts) - pad))
            rx1 = min(self.width, int(max(p[0] for p in pts) + pad) + 1)
            ry1 = min(self.height, int(max(p[1] for p in pts) + pad) + 1)
            reg = (rx0, ry0, rx1, ry1) if (rx1 > rx0 and ry1 > ry0) else None
            # The premultiply-hygiene fill below writes the brush colour into
            # EVERY fully transparent pixel of the layer, so on such a layer a
            # stroke is not confined to its own rectangle and a region-limited
            # snapshot would not restore it. Only claim the region when the
            # layer is already fully opaque -- then the stroke really does stay
            # inside its box. (Found by the undo test, not by reading the code.)
            if reg is not None and bool((self.layer(lid).pixels[..., 3] <= 0).any()):
                reg = None
            self.record("Mask paint" if target_mask
                        else ("Erase" if erase else "Brush"),
                        only=[lid], region=None if target_mask else reg)
        if target_mask is None:
            self.record_stroke(lid, points, {
                "radius": float(radius), "opacity": float(opacity),
                "hardness": float(hardness), "erase": bool(erase),
                "color": [float(c) for c in np.asarray(color).reshape(-1)[:3]],
                # media rides in the stroke record so a replay lays the same
                # paint with the same body -- old records stay byte-identical
                **({"media": str(media), "load": float(load)} if media else {}),
                # the material rides AS GIVEN (name or dict) so a replay lays
                # the same stuff -- resolved values would fossilise one
                # build's preset table into every old stroke
                **({"material": material, "load": float(load)}
                   if matdef else {}),
                **({"mix": float(mix)} if mix else {}),
                # the charge the brush ACTUALLY had, frozen: a replay must
                # not re-derive it from a reservoir that has moved on
                **({"real_brush": True, "charge0": _rb_c0,
                    "lanes0": _rb_lanes} if real_brush else {}),
                **({"alpha_lock": True} if alpha_lock else {}),
            }, new=bool(record) if stroke_new is None else bool(stroke_new))
        l = self.layer(lid)
        h, w = self.height, self.width
        # Densify the polyline so fast strokes stay continuous. A point may
        # carry a third component: a WIDTH FACTOR, so a stroke can taper. It is
        # interpolated along with position, which is what makes a rigged stroke
        # read as hair or a vine rather than uniform wire.
        pts = []
        widths = []
        for p in (points or []):
            pts.append((float(p[0]), float(p[1])))
            widths.append(float(p[2]) if len(p) > 2 else 1.0)
        # SPEED, derived from the RAW point spacing. The client samples the
        # pointer at a roughly fixed rate, so the gap between consecutive raw
        # points is how fast the hand was moving -- no timestamps to plumb
        # through, no protocol change, and because the recorded points keep
        # their spacing a replay re-derives exactly the same speed. Measured
        # in brush-widths per sample so a flick means the same thing to a
        # small brush as to a big one.
        raw_sp = [0.0] * len(pts)
        # ...but ONLY for a path that was actually sampled by a hand. Spacing
        # means speed because the client polls the pointer at a fixed rate;
        # it means nothing for a path that arrived some other way. A
        # two-point straight line from the API has one enormous gap and was
        # being read as a maximum-speed flick, so programmatic strokes came
        # out three times too thin. Below a real gesture's worth of samples,
        # speed is UNKNOWN, and unknown must mean neutral rather than fast.
        if len(pts) >= 6:
            for i in range(1, len(pts)):
                raw_sp[i] = np.hypot(pts[i][0] - pts[i - 1][0],
                                     pts[i][1] - pts[i - 1][1]) / max(radius, 1.0)
            raw_sp[0] = raw_sp[1]
        dense = []
        dwid = []
        dspd = []
        for i, p in enumerate(pts):
            if i:
                q = pts[i - 1]
                d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                # spacing must follow the LOCAL (pressure-scaled) dab, not
                # the base radius: a light-pressure dab is a fraction of the
                # size but was still stepped a full base-radius apart, so the
                # thin end of a pressure ramp came out as a string of
                # separated beads instead of a stroke
                wmin = max(min(widths[i - 1], widths[i]), 0.05)
                n = int(d / max(radius * wmin * 0.35, 1)) + 1
                for t in np.linspace(0, 1, n + 1)[1:]:
                    dense.append((q[0] + (p[0] - q[0]) * t, q[1] + (p[1] - q[1]) * t))
                    dwid.append(widths[i - 1] + (widths[i] - widths[i - 1]) * float(t))
                    dspd.append(raw_sp[i - 1] + (raw_sp[i] - raw_sp[i - 1]) * float(t))
            else:
                dense.append(p)
                dwid.append(widths[0])
                dspd.append(raw_sp[0])
        mask = np.zeros((h, w), np.float32)
        r = float(radius)
        tip = None
        bobj = None
        if brush is not None:
            try:
                bobj = b = self.brush_by_id(brush)
                side = max(int(2 * r), 2)
                tip = _resize(b.tip, side, side)
                # re-space the stroke by the brush's own spacing
                step = max(b.spacing * 2 * r * max(min(widths), 0.05), 1.0)
                sp, spw, sps, acc = [dense[0]], [dwid[0]], [dspd[0]], 0.0
                for i in range(1, len(dense)):
                    acc += np.hypot(dense[i][0] - dense[i-1][0], dense[i][1] - dense[i-1][1])
                    if acc >= step:
                        sp.append(dense[i]); spw.append(dwid[i])
                        sps.append(dspd[i]); acc = 0.0
                dense, dwid, dspd = sp, spw, sps
            except KeyError:
                tip = None
        # deterministic per-stroke rng: same stroke -> same jitter
        rng = np.random.default_rng(
            int(hashlib.md5(np.asarray(pts, np.float32).tobytes()).hexdigest()[:8], 16))
        for di, (px, py) in enumerate(dense):
            if tip is not None:
                st = tip
                if bobj is not None:
                    ang = 0.0
                    if bobj.follow and di > 0:
                        q = dense[di - 1]
                        ang = np.degrees(np.arctan2(py - q[1], px - q[0]))
                    if bobj.j_angle > 0:
                        ang += rng.uniform(-bobj.j_angle, bobj.j_angle)
                    if abs(ang) > 0.5:
                        st = _rotate_tip(tip, ang)
                    if bobj.j_size > 0:
                        sc = 1.0 + rng.uniform(-bobj.j_size, bobj.j_size)
                        ns = max(int(st.shape[0] * sc), 2)
                        st = _resize(st, ns, ns)
                    if bobj.j_scatter > 0:
                        px = px + rng.uniform(-1, 1) * bobj.j_scatter * st.shape[0]
                        py = py + rng.uniform(-1, 1) * bobj.j_scatter * st.shape[0]
                side = st.shape[0]
                tx0, ty0 = int(round(px - side / 2)), int(round(py - side / 2))
                sx0, sy0 = max(0, -tx0), max(0, -ty0)
                dx0, dy0 = max(0, tx0), max(0, ty0)
                dx1, dy1 = min(w, tx0 + side), min(h, ty0 + side)
                if dx1 > dx0 and dy1 > dy0:
                    np.maximum(mask[dy0:dy1, dx0:dx1],
                               st[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)],
                               out=mask[dy0:dy1, dx0:dx1])
                continue
            rr = r * (dwid[di] if di < len(dwid) else 1.0)
            if rr <= 0.05:
                continue                       # a zero-width point lays nothing
            x0, x1 = max(0, int(px - rr - 2)), min(w, int(px + rr + 3))
            y0, y1 = max(0, int(py - rr - 2)), min(h, int(py + rr + 3))
            if x0 >= x1 or y0 >= y1:
                continue
            # Distances for THIS dab's window only. This used to index into
            # two full-canvas mgrid arrays built per flush (15 MB at 1080p,
            # ~11 ms of pure allocation) even though every dab touches a few
            # hundred pixels. Cost now scales with brush size, not canvas size.
            yy = np.arange(y0, y1, dtype=np.float32)[:, None] - py
            xx = np.arange(x0, x1, dtype=np.float32)[None, :] - px
            d = np.hypot(xx, yy)
            core = rr * float(hardness)
            fall = np.clip(1.0 - (d - core) / max(rr - core, 1e-3), 0, 1)
            np.maximum(mask[y0:y1, x0:x1], np.where(d <= core, 1.0, fall), out=mask[y0:y1, x0:x1])
        if selection:
            try:
                sv = _resize(self.gate_by_id(selection).data, h, w)
                mask = mask * ((1.0 - sv) if sel_invert else sv)
            except KeyError:
                pass
        # Composite only the rectangle the stroke actually touched. The alpha
        # maths below builds several full-canvas 3-channel temporaries; at
        # 1920x1080 that ran ~25 MB of arithmetic per flush to change a few
        # hundred pixels. Two boolean reductions find the box far more cheaply.
        # NOTE: this is NOT the whole story -- see the transparent-pixel fill
        # after the composite, which genuinely does span the canvas.
        rows = np.any(mask > 0, axis=1)
        if not rows.any():
            return                                   # nothing landed on canvas
        cols = np.any(mask > 0, axis=0)
        y0b, y1b = int(np.argmax(rows)), h - int(np.argmax(rows[::-1]))
        x0b, x1b = int(np.argmax(cols)), w - int(np.argmax(cols[::-1]))
        mask = mask[y0b:y1b, x0b:x1b]
        if target_mask is not None:            # paint the mask field, not pixels
            m = self.mask_by_id(target_mask)
            cov = mask * float(opacity)
            # brush colour's luminance is the value written, so picking white
            # reveals and black hides, exactly like painting a Photoshop mask
            val = 0.0 if erase else float(
                np.dot(_f32(color).reshape(3), (0.2126, 0.7152, 0.0722)))
            if getattr(m, "shape", None):
                m.shape = None       # hand-painted: the shape no longer describes it
            win = m.data[y0b:y1b, x0b:x1b]
            m.data[y0b:y1b, x0b:x1b] = np.clip(win * (1.0 - cov) + val * cov,
                                               0.0, 1.0).astype(np.float32)
            composite_patch(self, x0b, y0b, x1b, y1b, rev_entry)
            return
        # The paint surface is decided BEFORE the pigment is laid: where the
        # brush skips the weave or runs dry it lays less colour AND less
        # body, so the deposit has to be known here rather than after the
        # composite. `_dep_cache` hands the height to the media block below
        # without computing the stroke frame twice.
        _dep_cache = None
        _wetcol = None
        if not erase and (matdef is not None or media in _MEDIA):
            _med0 = matdef if matdef is not None else _MEDIA[media]
            _walk = _lf = None
            if real_brush:
                # the load is whatever is LEFT on the brush. `charge0` is
                # frozen into the record so a replay -- or nudging stroke
                # three of forty -- cannot retroactively change how much
                # paint stroke forty had. What happens WITHIN the stroke is
                # recomputed from canvas state and stays deterministic.
                c0 = np.asarray(_rb_c0, np.float32)
                bc = np.asarray(_rb_lanes, np.float32)
                _walk = _brush_walk(
                    l, dense, bc, float(_med0.get("hold", 0.6)),
                    mix * float(_med0.get("pickup", 0.6)), c0,
                    radius=float(radius),
                    spend_rate=float(radius) / (20.0 * 1400.0),
                    # tuned so a few passes through a mound actually LOADS
                    # the brush with that colour: at 0.0045 a full dip
                    # exchanged ~23% and red dipped in white stayed red
                    reload_rate=0.022 * float(_med0.get("pickup", 0.6)))
            _fr = _stroke_frame(dense, dwid, float(radius),
                                x0b, y0b, x1b, y1b) if _walk else None
            if _walk is not None and _fr is not None:
                _cols, _chg, _taken = _walk
                _ix = np.clip((_fr[1] * (len(_chg) - 1)).astype(np.int32),
                              0, len(_chg) - 1)
                # look the brush up by BOTH arc position and cross-stroke
                # lane: which part of the tuft is over this pixel decides
                # what colour lands there, so a brush loaded blue on one
                # edge and white on the other paints a variegated band in
                # ONE stroke instead of a pre-averaged flat mix
                _ln = np.clip(((_fr[0] * 0.5 + 0.5) * (_BRUSH_LANES - 1)
                               ).astype(np.int32), 0, _BRUSH_LANES - 1)
                _lf = _chg[_ix, _ln] * float(load)
                _wetcol = _cols[_ix, _ln]
            _dep_cache, _cover, _sv = _deposit(
                self, l, _med0, mask, opacity, load, dense, dwid,
                float(radius), x0b, y0b, x1b, y1b,
                seed=int(hashlib.md5(
                    np.asarray(pts, np.float32).tobytes()).hexdigest()[:8], 16),
                load_field=_lf, dspd=dspd)
            if _cover is not None:
                mask = mask * _cover
            if _walk is not None and _sv is not None:
                _cols, _chg, _taken = _walk
                if float(_taken.max()) > 1e-6 and l.height_map is not None:
                    # paint is CONSERVED: what the brush lifted, the canvas
                    # lost. Recharging from a thick passage has to leave a
                    # scrape, or the brush is a paint printer, not a brush.
                    ti = np.clip((_sv * (len(_taken) - 1)).astype(np.int32),
                                 0, len(_taken) - 1)
                    # scaled with the reload rate: the brush now fills from a
                    # mound roughly five times faster, and if the scrape does
                    # not keep pace the stroke deposits more than it lifts and
                    # the mound never goes down -- paint stops being conserved
                    # exactly where dipping happens
                    _ln2 = np.clip(((_sv * 0 + (fr0 := _fr[0]) * 0.5 + 0.5)
                                    * (_BRUSH_LANES - 1)).astype(np.int32),
                                   0, _BRUSH_LANES - 1)
                    scrape = np.clip(_taken[ti, _ln2] * 90.0 * mask, 0.0, 0.95)
                    l.height_map[y0b:y1b, x0b:x1b] *= (1.0 - scrape)
                # Painting ALWAYS changes what is on the brush. This used to
                # be gated on `record`, which conflated "should this stroke be
                # undoable" with "did the brush pick anything up" -- so a dab
                # passed with record=False (the normal way to lay texture
                # without a stroke record per dab) silently did not load the
                # brush at all. Found by painting: dipping into a palette in a
                # loop left the brush exactly the colour it started.
                if charge0 is None and not getattr(self, "_replaying", False):
                    _host.brush_lanes = _cols[-1].copy()
                    _host.brush_charges = _chg[-1].copy()
                    _host.brush_charge = float(_chg[-1].mean())
                    _host.brush_color = tuple(
                        float(v) for v in _cols[-1].mean(0))
            elif _sv is not None and mix > 1e-3:
                tbl = _wet_mix(l, dense, color, float(_med0.get("hold", 0.6)),
                               mix * float(_med0.get("pickup", 0.6)), None)
                if tbl is not None:
                    # per-pixel brush colour by arc position: the stroke is
                    # one colour at its start and whatever it has gathered by
                    # its end
                    ix = np.clip((_sv * (len(tbl) - 1)).astype(np.int32),
                                 0, len(tbl) - 1)
                    _wetcol = tbl[ix]
        px_win = l.pixels[y0b:y1b, x0b:x1b]
        a = (mask * float(opacity))[..., None]
        if alpha_lock:
            # Procreate/Photoshop alpha lock: transparency is FROZEN. The
            # brush recolors what is already there (coverage scaled by the
            # existing alpha) and the eraser cannot cut -- both leave the
            # alpha channel untouched.
            if not erase:
                col = (_wetcol if _wetcol is not None
                       else _f32(color).reshape(1, 1, 3))
                aw = a * px_win[..., 3:4]
                px_win[..., :3] = col * aw + px_win[..., :3] * (1 - aw)
        elif erase:
            px_win[..., 3:4] = px_win[..., 3:4] * (1 - a)
        else:
            col = (_wetcol if _wetcol is not None
                   else _f32(color).reshape(1, 1, 3))
            old_a = px_win[..., 3:4]
            new_a = a + old_a * (1 - a)
            px_win[..., :3] = np.where(new_a > 0,
                                       (col * a + px_win[..., :3] * old_a * (1 - a))
                                       / np.maximum(new_a, 1e-6), col)
            px_win[..., 3:4] = new_a
            # Fully transparent pixels carry the brush colour rather than
            # black. This looks pointless (they are invisible) but it is
            # premultiply hygiene: a later Blur or Unpremult mixes RGB across
            # the alpha edge, and black bleeding in would darken every soft
            # edge. The old full-canvas composite did this implicitly via the
            # np.where above; windowing the composite dropped it outside the
            # box, which measurably changed blurred edges. One boolean pass is
            # far cheaper than the full float pipeline it replaced.
            clear = l.pixels[..., 3] <= 0
            if clear.any():
                l.pixels[..., :3][clear] = _f32(color).reshape(3)
        flow_bottom = y1b
        if erase and l.height_map is not None:
            # erasing removes the paint BODY too, whatever media is selected
            # right now -- leaving invisible ridges under later strokes was
            # wrong physically and looked haunted under the relief light
            l.height_map[y0b:y1b, x0b:x1b] *= (1.0 - mask * float(opacity))
        if erase:
            # THE STRATA ARE THE SAME PAINT. A passage that built past a
            # layer's ceiling lives on several layers, so erasing only the
            # base left the paint sitting on the strata above it -- you wiped
            # a mark and it was still there. The eraser goes through the
            # whole column, as it must, since the column is one body.
            _up = getattr(l, "stratum_next", None)
            _keep = 1.0 - mask * float(opacity)
            for _ in range(12):
                if not _up:
                    break
                try:
                    _u = self.layer(_up)
                except KeyError:
                    break
                _uw = _u.pixels[y0b:y1b, x0b:x1b]
                _uw[..., 3:4] *= _keep[..., None]
                if getattr(_u, "height_map", None) is not None:
                    _u.height_map[y0b:y1b, x0b:x1b] *= _keep
                if getattr(_u, "height_below", None) is not None:
                    _u.height_below[y0b:y1b, x0b:x1b] *= _keep
                if getattr(_u, "material_map", None) is not None:
                    _u.material_map[y0b:y1b, x0b:x1b, 2] *= _keep
                _up = getattr(_u, "stratum_next", None)
        if erase and getattr(l, "material_map", None) is not None:
            # the eraser takes the STUFF with the paint: leaving invisible
            # gold coverage under a cleared area would make the next plain
            # stroke there gleam for no visible reason
            l.material_map[y0b:y1b, x0b:x1b, 2] *= (1.0 - mask * float(opacity))
        if not erase and (matdef is not None or media in _MEDIA):
            med = matdef if matdef is not None else _MEDIA[media]
            if l.height_map is None:
                l.height_map = np.zeros((h, w), np.float32)
            if matdef is None:
                # the SCALAR look belongs to plain media ("last media wins");
                # a material's look lives per-pixel in its map, so a gold
                # stroke must not re-tune how the layer's existing oil shades
                l.paint_gloss = med["gloss"]
                l.paint_media = media
            hw = l.height_map[y0b:y1b, x0b:x1b]
            # the brush LAYS paint: a real surface, not a rescaled alpha.
            # Height accumulates across strokes -- that is the build-up.
            dep = _dep_cache
            hw += dep
            # SPILL: once this layer is full, the excess starts a new stratum
            # instead of being clipped away. Clipping is what made a worked
            # passage saturate after ~2 loaded passes and flatten to a
            # plateau; a painter builds heavy impasto in campaigns, and each
            # stratum begins again from zero.
            # Spilling MUST happen during replay too. Suppressing it kept the
            # overflow on the base instead of moving it up, so a rebuilt
            # layer did not match what was painted -- strata were the only
            # thing in the engine that broke replay determinism. It is safe
            # because `replay_layer` clears the whole chain first and
            # `_stratum_for` reuses the existing link rather than breeding a
            # new layer per rebuild.
            if (getattr(self, "auto_stratum", False)
                    and not getattr(l, "palette", False)):
                over = hw - _HEIGHT_CAP
                np.clip(over, 0.0, None, out=over)
                if float(over.max()) > 1e-3:
                    np.clip(hw, 0.0, _HEIGHT_CAP, out=hw)
                    ucol = (_wetcol if _wetcol is not None
                            else _f32(color).reshape(1, 1, 3))
                    src, cur = lid, over
                    # CASCADE. Spilling once only defers the ceiling: the new
                    # stratum filled and flattened in its turn (measured at
                    # 28.9 units on a layer whose cap is 4). Keep spilling
                    # upward until the excess is gone, so a passage can be
                    # worked as heavily as the painter likes.
                    for _ in range(12):
                        up = self._stratum_for(src)
                        if up.height_map is None:
                            up.height_map = np.zeros((h, w), np.float32)
                        uh = up.height_map[y0b:y1b, x0b:x1b]
                        uh += cur
                        frac = np.clip(cur / np.maximum(dep, 1e-6), 0.0, 1.0)
                        ua = (mask * float(opacity) * frac)[..., None]
                        uw = up.pixels[y0b:y1b, x0b:x1b]
                        na = np.clip(ua + uw[..., 3:4] * (1.0 - ua), 0.0, 1.0)
                        uw[..., :3] = np.where(
                            na > 1e-6,
                            (ucol * ua + uw[..., :3] * uw[..., 3:4] * (1.0 - ua))
                            / np.maximum(na, 1e-6), uw[..., :3])
                        uw[..., 3:4] = na
                        # what this stratum SITS ON, so it can be lit as the
                        # top of one continuous column rather than a slab
                        if getattr(up, "height_below", None) is None or \
                                up.height_below.shape != (h, w):
                            up.height_below = np.zeros((h, w), np.float32)
                        _sl = self.layer(src)
                        _under = _sl.height_map[y0b:y1b, x0b:x1b]
                        _sb = getattr(_sl, "height_below", None)
                        up.height_below[y0b:y1b, x0b:x1b] = (
                            _under + (_sb[y0b:y1b, x0b:x1b] if _sb is not None
                                      and _sb.shape == (h, w) else 0.0))
                        nxt = uh - _HEIGHT_CAP
                        np.clip(nxt, 0.0, None, out=nxt)
                        if float(nxt.max()) <= 1e-3:
                            break
                        np.clip(uh, 0.0, _HEIGHT_CAP, out=uh)
                        src, cur = up.id, nxt.copy()
            if matdef is not None:
                # the stroke also lays the STUFF: rough/metal blend into the
                # material map with the same unpremultiplied alpha-over as
                # the pigment, so a gold stroke crossing chalk transitions
                # exactly the way their colours do
                if l.material_map is None:
                    l.material_map = np.zeros((h, w, 3), np.float32)
                ga = float(matdef.get("grain", 0.0))
                if ga > 0.0:
                    # micro-relief rides the deposit -- position-stable, so a
                    # replay lays the identical tooth
                    g = _material_grain(self, l, matdef.get("gscale", 2.0))
                    hw += g[y0b:y1b, x0b:x1b] * ga * dep
                    np.clip(hw, 0.0, None, out=hw)
                mm = l.material_map[y0b:y1b, x0b:x1b]
                a = np.clip(mask * float(opacity), 0.0, 1.0)
                old_c = mm[..., 2]
                new_c = a + old_c * (1.0 - a)
                for ch, val in ((0, float(matdef["rough"])),
                                (1, float(matdef["metal"]))):
                    mm[..., ch] = np.where(
                        new_c > 0,
                        (val * a + mm[..., ch] * old_c * (1.0 - a))
                        / np.maximum(new_c, 1e-6), val)
                mm[..., 2] = new_c
            self._paint_flow(l, x0b, y0b, x1b, y1b, med, dep)
            self._watercolour(l, x0b, y0b, x1b, y1b, med, dep)
            flow_bottom = min(h, y1b + int(med["iters"]) + 2)
        # realtime feedback: re-light and re-blend ONLY this stroke's window.
        # Validity is judged against the revision captured on entry, so the
        # record/announce bumps inside this very call don't invalidate it.
        if not getattr(self, "_replaying", False):
            _shade_patch(l, x0b, y0b, x1b, flow_bottom, rev_entry)
            # the composite (and the client's dirty window) must cover
            # the RE-LIT ring around the stroke, not just the pigment
            # bbox -- relief shading reaches ~6 px past the mask
            pd = 8
            composite_patch(self, x0b - pd, y0b - pd, x1b + pd,
                            flow_bottom + pd, rev_entry)
            self._last_paint_rect = (max(0, int(x0b) - pd),
                                     max(0, int(y0b) - pd),
                                     min(w, int(x1b) + pd),
                                     min(h, int(flow_bottom) + pd))
        # REPLAY paints must not touch the caches: replay_layer is used as a
        # read-only PROBE by the faithfulness guard, which swaps in base
        # pixels, replays, and restores -- patching mid-replay stamped the
        # composite cache rev-current with ghost strokes over the real
        # frame. That was the recurring "images piled on top of each other"
        # artifact: any guarded stroke edit (eraser on a layer with a fill,
        # a nudge probe) poisoned the cache without ever failing.
        if getattr(l, "vol_kind", "none") in _MEDIA_KINDS:
            # a DYNAMIC medium: the stroke is an injection of dye, and the
            # slab immediately runs a burst of solve steps scaled by its
            # thickness -- release the stroke, watch the medium take it.
            # BRUSH LOAD is how much liquid the brush carries: a loaded
            # brush dumps more ink and disturbs the water more; a dry one
            # barely tints it.
            _media_inject(self, l, x0b, y0b, x1b, flow_bottom,
                          strength=min(float(load) / 0.6, 2.0))
            # THE DOCUMENT TIMELINE RULES TIME. Painting used to run its
            # own burst of solver steps (8 + thickness) -- "it just
            # animates some random increment and I have zero control".
            # Now a stroke only INJECTS dye and disturbs the velocity
            # field; the medium advances exclusively when the playhead
            # moves (set_frame / play, scaled by the layer's keyable
            # media_rate). No settle either -- even two global solver
            # steps drifted distant ink 0.45. Fresh dye sits exactly as
            # painted until the playhead moves; that IS the control.
        _absorb = float(getattr(l, "absorbency", 0.0))
        if _absorb > 0.0:
            # a wet brush soaks deeper than a dry one: load scales the
            # bleed (default load 0.6 == exactly the layer's absorbency,
            # so existing behaviour is unchanged)
            _absorb = min(_absorb * float(load) / 0.6, 1.0)
        if _absorb > 0.0:
            # canvas fibre drinks the stroke: bleed within (and a little
            # past) the stroke's own window, along the document's grain
            soak_region(self, lid, x0b, y0b, x1b, flow_bottom, _absorb)
        if faith_entry:
            # a RECORDED stroke keeps a faithful layer faithful by
            # construction -- stamp the replay-ok cache forward so the next
            # nudge skips its ~0.4 s full-replay safety check (profiled: the
            # check was 392 of nudge's 494 ms)
            self._mark_replay_ok(lid)
        # hand back the stroke's id, as `knife` and `blend_stroke` do. The
        # server was reaching into `strokes[-1]` for it, and a caller who
        # wanted to group, move or delete what it had just painted had no way
        # to name it.
        return (self.strokes[-1]["id"]
                if (record and self.strokes) else None)

    def _watercolour(self, l, x0, y0, x1, y1, med, dep):
        """Wet media in absorbent paper: wicking, edge darkening, granulation.

        Curtis et al. (SIGGRAPH 97) name the effects that make watercolour
        read as watercolour, and leStudio had none of them -- its "water" was
        just oil with a low hold and more gravity, which is a runny film on a
        surface rather than a fluid inside paper. Three of theirs are worth
        the cost here:

        WICKING: the paper drinks the wash sideways along its fibres, so the
        mark is softer and larger than the brush that made it, with a feathery
        boundary no stroke mask would give.

        EDGE DARKENING: as the water evaporates it carries pigment to the
        perimeter and leaves it there. That dark rim around a drying wash is
        the single most recognisable watercolour signature, and the reason a
        flat wash never reads as flat.

        GRANULATION: pigment is heavier than water and settles into the
        paper's valleys, strongest where the paper is wettest -- the grainy
        texture that emphasises the weave. Note this is the OPPOSITE of how
        stiff paint meets the tooth: dragged paint catches on the risen
        threads, a wash pools in the dips between them.
        """
        absorb = float(med.get("absorb", 0.0))
        if absorb <= 1e-3:
            return
        # rough rag drinks; sized hot-press holds the wash on top
        absorb = float(np.clip(absorb * _paper_of(self)[1]["drink"],
                               0.0, 1.0))
        H, W = self.height, self.width
        pad = int(np.clip(4.0 + absorb * 7.0, 4, 22))
        ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
        ax1, ay1 = min(W, x1 + pad), min(H, y1 + pad)
        if ax1 <= ax0 + 2 or ay1 <= ay0 + 2:
            return
        px = l.pixels[ay0:ay1, ax0:ax1]
        wet = np.zeros(px.shape[:2], np.float32)
        wet[y0 - ay0:y0 - ay0 + dep.shape[0],
            x0 - ax0:x0 - ax0 + dep.shape[1]] = dep
        wet = np.clip(wet * 2.2, 0.0, 1.0)
        # where the paper is damp ENOUGH to move pigment -- the wash spreads
        # into this, which is why the mark ends up bigger than the brush
        damp = np.clip(_gauss_small(wet[..., None], 1.5 + absorb * 3.0)[..., 0]
                       * 2.4, 0.0, 1.0)
        # --- wicking, premultiplied so colour and coverage travel together
        pm = px.copy()
        pm[..., :3] *= pm[..., 3:4]
        sp = _gauss_small(pm, 1.0 + absorb * 2.6)
        w = (damp * absorb * 0.85)[..., None]
        pm = pm * (1.0 - w) + sp * w
        a = np.clip(pm[..., 3:4], 0.0, 1.0)
        rgb = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), px[..., :3])
        # --- edge darkening: pigment stranded at the perimeter of the wash
        ed = float(med.get("edge_dark", 0.0))
        if ed > 1e-3:
            gy, gx = np.gradient(damp)
            rim = np.hypot(gx, gy)
            # only where pigment actually is, or the rim lands on bare paper
            # and haloes the mark instead of darkening it
            rim = rim * (a[..., 0] > 0.10)
            rs = float(rim.sum())
            if rs > 1e-6:
                # MOVE pigment, do not amplify it. The water carries pigment
                # to the perimeter as it evaporates and strands it there, so
                # the rim ends up darker than the middle -- which scaling a
                # thin rim can never achieve against a thick centre. Taking
                # from the interior and depositing on the rim conserves the
                # pigment and produces the real effect.
                take = a[..., 0] * ed * 0.45 * damp
                moved = float(take.sum())
                a = np.clip(a - take[..., None]
                            + (rim / rs * moved)[..., None], 0.0, 1.0)
        # --- granulation: the pigment is heavier than the water
        gr = float(med.get("granulate", 0.0))
        if gr > 1e-3:
            tooth = _canvas_tooth(self)[ay0:ay1, ax0:ax1]
            a = np.clip(a * (1.0 + gr * (0.5 - tooth) * 1.6 * damp)[..., None],
                        0.0, 1.0)
        px[..., :3] = np.clip(rgb, 0.0, 1.0)
        px[..., 3:4] = a
        _MUT_REV[0] += 1

    def _paint_flow(self, l, x0, y0, x1, y1, med, dep):
        """Gravity on WET paint only. `dep` is what this stroke just laid
        down: that is the mobile paint. Total height above the medium's
        `hold` is excess, but the amount that can move is capped by the wet
        fraction -- old strokes are dry and stay put. Without the cap, a new
        stroke's flow window re-flowed every old ridge it covered, and the
        ridge collapsed INSIDE the window while staying tall outside: a hard
        rectangular seam exactly at the window edge (measured 3.28 height
        cliff, 0.09 luminance step -- the user's "glitchy" screenshot).
        Excess moves one pixel down per step, carrying its share of pigment,
        so heavy loads sag and watery media run. Deterministic, so replay
        rebuilds the same drips."""
        H, W = l.height_map.shape
        iters = int(med["iters"])
        gstr, gdx, gdy = _flow_dir(self, l)
        # the window has to open in the direction the paint will actually
        # travel, not just downward
        pad = iters + 2
        y0e = max(0, y0 - (pad if gdy < 0 or gstr < 0.05 else 0))
        y1e = min(H, y1 + (pad if gdy > 0 or gstr < 0.05 else 0))
        x0e = max(0, x0 - (pad if gdx < 0 or gstr < 0.05 else 0))
        x1e = min(W, x1 + (pad if gdx > 0 or gstr < 0.05 else 0))
        hg = l.height_map[y0e:y1e, x0e:x1e]
        px = l.pixels[y0e:y1e, x0e:x1e]
        wet = np.zeros_like(hg)
        wet[y0 - y0e:y0 - y0e + dep.shape[0],
            x0 - x0e:x0 - x0e + dep.shape[1]] += dep
        # work premultiplied: moving paint moves colour AND coverage together
        pm = px.copy()
        pm[..., :3] *= pm[..., 3:4]
        hold, fl = float(med["hold"]), float(med["flow"])
        touched = np.zeros(hg.shape, bool)
        # Reused scratch: the loop runs up to 26 times for watercolour and
        # every iteration was allocating four full-window arrays, one of them
        # 4-channel. Allocation, not arithmetic, was most of the cost.
        excess = np.empty_like(hg)
        frac = np.empty_like(hg)
        moved = np.empty_like(pm)
        for _ in range(iters):
            np.subtract(hg, hold, out=excess)
            np.clip(excess, 0.0, None, out=excess)
            np.minimum(excess, wet, out=excess)
            np.multiply(excess, fl, out=excess)
            if excess.max() <= 1e-4:
                break
            np.maximum(hg, 1e-6, out=frac)
            np.divide(excess, frac, out=frac)
            np.clip(frac, 0.0, 0.6, out=frac)
            np.multiply(pm, frac[..., None], out=moved)
            hg -= excess
            wet -= excess
            pm -= moved
            m = excess > 1e-6
            touched |= m

            def _push(dst_h, dst_w, dst_p, ex, mv, sy, sx, k):
                # move a share `k` of the excess one pixel along (sy, sx)
                if k <= 1e-6:
                    return
                e = ex if k == 1.0 else ex * k
                v = mv if k == 1.0 else mv * k
                ys_d = slice(1, None) if sy > 0 else (slice(None, -1)
                                                      if sy < 0 else slice(None))
                ys_s = slice(None, -1) if sy > 0 else (slice(1, None)
                                                       if sy < 0 else slice(None))
                xs_d = slice(1, None) if sx > 0 else (slice(None, -1)
                                                      if sx < 0 else slice(None))
                xs_s = slice(None, -1) if sx > 0 else (slice(1, None)
                                                       if sx < 0 else slice(None))
                dst_h[ys_d, xs_d] += e[ys_s, xs_s]
                dst_w[ys_d, xs_d] += e[ys_s, xs_s]
                dst_p[ys_d, xs_d] += v[ys_s, xs_s]
                touched[ys_d, xs_d] |= m[ys_s, xs_s]

            if gstr < 0.05:
                # FLAT: no in-plane gravity. A puddle levels -- it spreads to
                # every neighbour equally rather than running one way.
                for sy, sx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    _push(hg, wet, pm, excess, moved, sy, sx, 0.25)
            else:
                # split the step between the two axes so the paint drifts
                # along the real direction instead of snapping to 8 compass
                # points, which would show as staircased drips
                ax, ay = abs(gdx), abs(gdy)
                tot = max(ax + ay, 1e-6)
                _push(hg, wet, pm, excess, moved,
                      int(np.sign(gdy)), 0, ay / tot)
                _push(hg, wet, pm, excess, moved,
                      0, int(np.sign(gdx)), ax / tot)
        a = pm[..., 3:4]
        # write back ONLY where paint actually moved: the unpremultiply
        # round-trip is float arithmetic, and rewriting untouched pixels
        # would drift them
        newa = np.clip(a, 0.0, 1.0)
        newrgb = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6),
                          px[..., :3])
        t3 = touched[..., None]
        px[..., 3:4] = np.where(t3, newa, px[..., 3:4])
        px[..., :3] = np.where(t3, newrgb, px[..., :3])
        np.clip(hg, 0.0, _HEIGHT_CAP, out=hg)

    def _tip_for(self, brush, r, hardness=0.7):
        """The stamp footprint for any brush choice, as (side, side) alpha."""
        side = max(int(2 * r), 4)
        if brush is not None:
            try:
                return np.clip(_resize(self.brush_by_id(brush).tip, side, side), 0, 1)
            except KeyError:
                pass
        ys, xs = np.mgrid[0:side, 0:side]
        c = (side - 1) / 2
        d = np.hypot(xs - c, ys - c)
        core = r * float(hardness)
        return np.clip(np.where(d <= core, 1.0,
                                1.0 - (d - core) / max(r - core, 1e-3)), 0, 1).astype(np.float32)

    def _dense_points(self, points, step):
        pts = [tuple(map(float, p)) for p in points] or []
        dense = []
        for i, p in enumerate(pts):
            if i:
                q = pts[i - 1]
                d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                n = int(d / max(step, 1)) + 1
                for t in np.linspace(0, 1, n + 1)[1:]:
                    dense.append((q[0] + (p[0] - q[0]) * t, q[1] + (p[1] - q[1]) * t))
            else:
                dense.append(p)
        return dense

    def _stratum_chain(self, lid):
        """The whole paint column from this layer upward, bottom first."""
        out, seen = [], set()
        try:
            cur = self.layer(lid)
        except KeyError:
            return out
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            out.append(cur)
            nxt = getattr(cur, "stratum_next", None)
            try:
                cur = self.layer(nxt) if nxt else None
            except KeyError:
                cur = None
        return out

    def _column(self, chain):
        """Total paint depth over the chain, as one field."""
        tot = None
        for l in chain:
            if l.height_map is None:
                continue
            tot = l.height_map.copy() if tot is None else tot + l.height_map
        return tot

    def _refill_column(self, chain, total):
        """Pour `total` back down the chain, filling each layer to the cap
        before starting the next. This is what removes the STEPPING: the
        column is re-levelled as one body of paint rather than as a stack of
        independent sheets, and any layer that ends up empty simply holds
        nothing rather than leaving a shelf."""
        rest = np.clip(total, 0.0, None)
        below = np.zeros_like(rest)
        for i, l in enumerate(chain):
            cap = _HEIGHT_CAP if i < len(chain) - 1 else np.inf
            take = np.minimum(rest, cap)
            if l.height_map is None:
                l.height_map = np.zeros_like(rest)
            l.height_map[...] = take
            if i:
                l.height_below = below.copy()
            below = below + take
            rest = rest - take
        _MUT_REV[0] += 1

    def knife(self, lid, points, mode="smooth", radius=26.0, strength=0.7,
              record=True, stroke_new=True):
        """The PALETTE KNIFE: shape the paint itself rather than add more.

        Paint on this canvas is a real depth that can span several strata, and
        nothing could push it around. A knife works the COLUMN -- the total
        across the whole stratum chain -- and the result is poured back down
        through the layers, so the body stays one continuous mass instead of a
        stack of sheets.

        Modes:
          smooth  -- level the surface toward its local average. This is also
                     the cure for stepping between strata, because the column
                     is re-levelled as one body.
          push    -- shove the paint along the stroke, banking it up ahead the
                     way a knife ploughs a ridge. Volume is conserved.
          scrape  -- take the tops off and leave the hollows, the flat-bladed
                     pull that reveals the colour underneath.
          spread  -- drag paint outward into a thin even film.
        """
        self._locked_guard(lid)
        chain = self._stratum_chain(lid)
        if not chain:
            raise KeyError(lid)
        total = self._column(chain)
        if total is None:
            return None
        if record:
            self.record("Knife (%s)" % mode, only=[l.id for l in chain])
        h, w = total.shape
        tip = self._tip_for(None, float(radius))
        side = tip.shape[0]
        st = float(np.clip(strength, 0.0, 1.0))
        pts = list(self._dense_points(points, max(radius * 0.3, 1)))
        prev = None
        for (px, py) in pts:
            x0, y0 = int(round(px - side / 2)), int(round(py - side / 2))
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + side), min(h, y0 + side)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            t = tip[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0] * st
            reg = total[dy0:dy1, dx0:dx1]
            if mode == "smooth":
                avg = _gauss_small(reg, max(radius * 0.45, 1.0))
                reg[...] = reg * (1 - t) + avg * t
            elif mode == "scrape":
                # take the tops off: everything above the local floor goes
                floor = _gauss_small(reg, max(radius * 0.7, 1.0)) * 0.72
                reg[...] = np.where(reg > floor, reg * (1 - t * 0.8)
                                    + floor * (t * 0.8), reg)
            elif mode == "spread":
                avg = _gauss_small(reg, max(radius * 1.1, 1.0))
                reg[...] = reg * (1 - t * 0.9) + avg * (t * 0.9)
            else:                                   # push
                if prev is not None:
                    vx, vy = px - prev[0], py - prev[1]
                    n = np.hypot(vx, vy)
                    if n > 1e-6:
                        sx = int(round(vx / n * max(radius * 0.22, 1)))
                        sy = int(round(vy / n * max(radius * 0.22, 1)))
                        moved = reg * t
                        reg -= moved
                        # bank it up AHEAD of the blade, volume conserved
                        a0, a1 = max(0, dy0 + sy), min(h, dy1 + sy)
                        b0, b1 = max(0, dx0 + sx), min(w, dx1 + sx)
                        mh, mw = a1 - a0, b1 - b0
                        if mh > 0 and mw > 0:
                            total[a0:a1, b0:b1] += moved[:mh, :mw]
            prev = (px, py)
        np.clip(total, 0.0, None, out=total)
        self._refill_column(chain, total)
        if record or stroke_new:
            self.record_stroke(lid, [(float(p[0]), float(p[1])) for p in points],
                               {"knife": str(mode), "radius": float(radius),
                                "strength": float(strength)}, stroke_new)
        self._mark_replay_ok(lid)
        return self.strokes[-1]["id"] if self.strokes else None

    def blend_stroke(self, lid, points, radius=18.0, strength=0.6,
                     brush=None, record=True, stroke_new=True):
        """The BLENDER: a clean brush that carries no pigment and works the
        paint already on the canvas -- Bob Ross's second brush, the one that
        turns two bands of colour into a sky.

        Unlike `smudge`, this is a RECORDED, REPLAYABLE stroke. That is the
        whole point. `smudge` mutates pixels and is explicitly not recorded,
        so any layer you smudged lost stroke editing entirely -- and a
        blending gesture is exactly what you most want to be able to nudge,
        because blending is where the picture actually gets made. A blend
        stroke sits in the stroke list like any other mark; the layer rebuilds
        by replaying paint, paint, blend, IN ORDER, so moving one of the
        colours underneath re-blends the result automatically, and moving the
        blend gesture itself moves where the softening happened.

        Physically it does two things a smear does not: it is gated by the
        paint BODY (thin or dry paint barely moves, thick wet paint moves a
        lot -- you cannot blend what is not there), and it knocks the ridges
        down as it goes, because dragging a soft brush through wet impasto
        flattens the peaks.
        """
        self._locked_guard(lid)
        _MUT_REV[0] += 1
        if record:
            self.record("Blend", only=[lid])
        l = self.layer(lid)
        h, w = self.height, self.width
        hm = getattr(l, "height_map", None)
        hold = 0.6
        tip = self._tip_for(brush, float(radius))
        side = tip.shape[0]
        carry = None
        st = float(np.clip(strength, 0.0, 1.0))
        for (px, py) in self._dense_points(points, max(radius * 0.25, 1)):
            x0 = int(round(px - side / 2)); y0 = int(round(py - side / 2))
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + side), min(h, y0 + side)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            sx0, sy0 = dx0 - x0, dy0 - y0
            t = tip[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
            reg = l.pixels[dy0:dy1, dx0:dx1]
            # WETNESS gate: only paint with body blends. Blending bare canvas
            # or a dry stain must do nothing, or the blender is just a smear
            # tool and the physicality is gone.
            if hm is not None:
                body = np.clip(hm[dy0:dy1, dx0:dx1] / hold, 0.0, 1.0)
            else:
                body = np.ones(t.shape, np.float32)
            a = (t * st * body)[..., None]
            # soften toward the local average, so two colours meeting become a
            # gradient rather than a seam
            avg = _gauss_small(reg, max(radius * 0.5, 1.0))
            reg[...] = reg * (1 - a * 0.55) + avg * (a * 0.55)
            # and drag: the brush carries what it just passed over
            if carry is not None and carry.shape == reg.shape:
                reg[...] = reg * (1 - a * 0.45) + carry * (a * 0.45)
            carry = reg.copy()
            if hm is not None:
                hreg = hm[dy0:dy1, dx0:dx1]
                havg = _gauss_small(hreg[..., None], max(radius * 0.5, 1.0))[..., 0]
                hm[dy0:dy1, dx0:dx1] = (hreg * (1 - a[..., 0] * 0.6)
                                        + havg * (a[..., 0] * 0.6)) * (
                                        1.0 - 0.10 * a[..., 0])
        if record or stroke_new:
            self.record_stroke(lid, [(float(p[0]), float(p[1])) for p in points],
                               {"blend": True, "radius": float(radius),
                                "strength": float(strength),
                                **({"brush": brush} if brush else {})},
                               stroke_new)
        self._mark_replay_ok(lid)
        return self.strokes[-1]["id"] if self.strokes else None

    def smudge(self, lid, points, radius=12.0, strength=0.6, brush=None, record=True):
        """Drag colour along the stroke: the tip picks paint up and lays it back down."""
        self._locked_guard(lid)
        _MUT_REV[0] += 1
        if record:
            self.record("Smudge")
        l = self.layer(lid)
        h, w = self.height, self.width
        tip = self._tip_for(brush, float(radius))
        side = tip.shape[0]
        carry = None
        carry_h = None
        hm = getattr(l, "height_map", None)
        for (px, py) in self._dense_points(points, max(radius * 0.3, 1)):
            x0 = int(round(px - side / 2)); y0 = int(round(py - side / 2))
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + side), min(h, y0 + side)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            sx0, sy0 = dx0 - x0, dy0 - y0
            a = (tip[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
                 * float(strength))[..., None]
            region = l.pixels[dy0:dy1, dx0:dx1]
            if carry is not None and carry.shape == region.shape:
                l.pixels[dy0:dy1, dx0:dx1] = region * (1 - a) + carry * a
            carry = l.pixels[dy0:dy1, dx0:dx1].copy()
            if hm is not None:
                # a smudge drags the paint BODY with the colour -- leaving the
                # ridge behind shaded the smeared pigment with topography that
                # was no longer under it
                hreg = hm[dy0:dy1, dx0:dx1]
                if carry_h is not None and carry_h.shape == hreg.shape:
                    hm[dy0:dy1, dx0:dx1] = hreg * (1 - a[..., 0]) + carry_h * a[..., 0]
                carry_h = hm[dy0:dy1, dx0:dx1].copy()

    def heal(self, lid, points, radius=14.0, record=True):
        """The HEAL brush: paint over a blemish and it repairs from the
        surroundings. The stroked region is treated as UNKNOWN and
        reconstructed by leCore's inpaint from everything around it --
        content-aware fill along a brush path. Alpha heals with the colour
        (a hole in a transparent sticker refills; an opaque photo stays
        opaque), and the seam is feathered so the repair sits flush.
        Partial undo on the layer, like every brush."""
        l = self.layer(lid)
        h, w = self.height, self.width
        if record:
            self.record("Heal", only=[lid])
        pts = np.asarray([(p[0], p[1]) for p in points], np.float32)
        if len(pts) == 0:
            return
        r = float(max(radius, 2.0))
        x0 = int(max(0, pts[:, 0].min() - r * 3.5))
        y0 = int(max(0, pts[:, 1].min() - r * 3.5))
        x1 = int(min(w, pts[:, 0].max() + r * 3.5))
        y1 = int(min(h, pts[:, 1].max() + r * 3.5))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return
        wh, ww = y1 - y0, x1 - x0
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        m = np.zeros((wh, ww), np.float32)
        for p in pts:
            d2 = (xx - p[0]) ** 2 + (yy - p[1]) ** 2
            m = np.maximum(m, np.clip(1.2 - np.sqrt(d2) / r, 0.0, 1.0))
        hole = m > 0.25
        if not hole.any() or hole.all():
            return
        win = l.pixels[y0:y1, x0:x1]
        a = win[..., 3:4]
        pm = np.concatenate([win[..., :3] * a, a], axis=-1)
        # a brush must answer in brush time: the solver's cost scales with
        # window AREA (a 20-point stroke cost ~3s, which the first E2E
        # mis-read as a hang). Large windows solve at reduced resolution
        # and the fill upsamples back -- repairs are low-frequency
        # continuations, so the downsampled solve reads the same.
        LIMIT = 12000
        area = wh * ww
        if area > LIMIT:
            sc = (LIMIT / float(area)) ** 0.5
            sh2, sw2 = max(24, int(wh * sc)), max(24, int(ww * sc))
            pm_s = _resize(pm, sh2, sw2)
            hole_s = _resize(hole.astype(np.float32)[..., None],
                             sh2, sw2)[..., 0] > 0.35
            try:
                fill_s = np.asarray(mind().inpaint(pm_s, ~hole_s), np.float32)
            except Exception:
                fill_s = pm_s.copy()
                for _ in range(40):
                    fill_s = np.where(hole_s[..., None],
                                      _gauss_blur(fill_s, 2.0), pm_s)
            filled = _resize(fill_s, wh, ww)
        else:
            try:
                filled = np.asarray(mind().inpaint(pm, ~hole), np.float32)
            except Exception:
                # diffusion fallback: iterate blur-fill from the boundary
                filled = pm.copy()
                for _ in range(40):
                    blur = _gauss_blur(filled, 2.0)
                    filled = np.where(hole[..., None], blur, pm)
        feather = _gauss_blur(m[..., None], max(1.5, r * 0.12))[..., 0]
        f = np.clip(feather, 0.0, 1.0)[..., None]
        out = filled * f + pm * (1 - f)
        na = np.clip(out[..., 3:4], 0.0, 1.0)
        win[..., :3] = np.where(na > 1e-5, out[..., :3] / np.maximum(na, 1e-5),
                                win[..., :3])
        win[..., 3] = na[..., 0]
        _MUT_REV[0] += 1

    def clone(self, lid, points, source, radius=12.0, opacity=1.0, brush=None,
              record=True, origin=None):
        """Clone-stamp: paint pixels sampled from `source` (the alt-clicked point),
        keeping the source->destination offset constant along the stroke. `origin`
        anchors the offset when a stroke arrives in chunks (defaults to points[0])."""
        _MUT_REV[0] += 1
        if record:
            self.record("Clone")
        l = self.layer(lid)
        h, w = self.height, self.width
        comp = self.composite()                     # sample what the eye sees
        tip = self._tip_for(brush, float(radius))
        side = tip.shape[0]
        if isinstance(source, dict) and "offset" in source:
            offx, offy = float(source["offset"][0]), float(source["offset"][1])
        else:
            org = origin if origin is not None else points[0]
            offx = float(org[0]) - float(source[0])
            offy = float(org[1]) - float(source[1])
        for (px, py) in self._dense_points(points, max(radius * 0.3, 1)):
            x0 = int(round(px - side / 2)); y0 = int(round(py - side / 2))
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + side), min(h, y0 + side)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            sx = np.clip(np.arange(dx0, dx1) - int(round(offx)), 0, w - 1)
            sy = np.clip(np.arange(dy0, dy1) - int(round(offy)), 0, h - 1)
            src = comp[np.ix_(sy, sx)]
            a = (tip[dy0 - y0:dy0 - y0 + (dy1 - dy0), dx0 - x0:dx0 - x0 + (dx1 - dx0)]
                 * float(opacity))[..., None]
            dst = l.pixels[dy0:dy1, dx0:dx1]
            new_a = a * src[..., 3:4] + dst[..., 3:4] * (1 - a * src[..., 3:4])
            rgb = (src[..., :3] * a * src[..., 3:4]
                   + dst[..., :3] * dst[..., 3:4] * (1 - a * src[..., 3:4]))
            dst[..., :3] = np.where(new_a > 0, rgb / np.maximum(new_a, 1e-6), dst[..., :3])
            dst[..., 3:4] = new_a

    def paint_image(self, lid, points, image, radius=12.0, opacity=1.0,
                    hardness=0.7, brush=None, record=True,
                    selection=None, sel_invert=False):
        """Paint with an IMAGE as the pigment: the stroke's coverage works
        exactly like the brush, but each covered pixel takes its colour from
        `image` at the same canvas position -- so dragging reveals the wired
        node's output where the stroke passes, like cloning from a picture
        that only exists in the graph.

        Not recorded as a stroke: the pigment is whatever the graph outputs at
        paint time, which a brush dict cannot reproduce, so a replay would
        repaint it wrongly. Same honest deal as smudge and clone -- nudge and
        the stroke editors simply decline the layer."""
        _MUT_REV[0] += 1
        if record:
            self.record("Node paint", only=[lid])
        l = self.layer(lid)
        h, w = self.height, self.width
        img = np.asarray(image, np.float32)
        if img.shape[:2] != (h, w):
            img = _resize(img, h, w)
        if img.shape[-1] == 3:
            img = np.concatenate([img, np.ones_like(img[..., :1])], -1)
        gate = None
        if selection:
            s = self.selection_by_id(selection)
            gate = (1.0 - s.data) if sel_invert else s.data
        tip = self._tip_for(brush, float(radius))
        side = tip.shape[0]
        hard = max(0.0, min(1.0, float(hardness)))
        if hard < 1.0 and brush is None:
            # the round tip honours hardness the way paint() does: linear
            # falloff from radius*hardness out to radius
            yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
            r = np.hypot(xx - (side - 1) / 2, yy - (side - 1) / 2)
            inner = float(radius) * hard
            tip = np.clip((float(radius) - r)
                          / max(float(radius) - inner, 1e-6), 0.0, 1.0)
        for (px, py) in self._dense_points(points, max(radius * 0.3, 1)):
            x0 = int(round(px - side / 2)); y0 = int(round(py - side / 2))
            dx0, dy0 = max(0, x0), max(0, y0)
            dx1, dy1 = min(w, x0 + side), min(h, y0 + side)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            src = img[dy0:dy1, dx0:dx1]
            a = (tip[dy0 - y0:dy0 - y0 + (dy1 - dy0),
                     dx0 - x0:dx0 - x0 + (dx1 - dx0)] * float(opacity))
            if gate is not None:
                a = a * gate[dy0:dy1, dx0:dx1]
            a = a[..., None]
            dst = l.pixels[dy0:dy1, dx0:dx1]
            sa = a * src[..., 3:4]
            new_a = sa + dst[..., 3:4] * (1 - sa)
            rgb = src[..., :3] * sa + dst[..., :3] * dst[..., 3:4] * (1 - sa)
            dst[..., :3] = np.where(new_a > 0, rgb / np.maximum(new_a, 1e-6),
                                    dst[..., :3])
            dst[..., 3:4] = new_a

    def canvas_layers(self):
        """The layers that belong to the PICTURE. A layer standing on a
        wall is not canvas content -- it is hidden until its side is
        opened for painting, at which point it lies flat and every
        ordinary tool works on it unchanged."""
        we = getattr(self, "wall_edit", None)
        return [l for l in self.layers
                if getattr(l, "wall", None) in (None, we)
                # nor is the PALETTE: it is a surface you work beside the
                # picture, drawn in its own dock, so it belongs in neither the
                # canvas view nor an export
                and not getattr(l, "palette", False)]

    def composite(self):
        return composite(self.canvas_layers(), self.height, self.width,
                         self.mask_map())


# ------------------------------------------------------------------------------------------------
# Node operators -- each wraps a leCore capability (or a small NumPy primitive where the engine's
# door is field-shaped). Every op declares its sockets + params so the UI's property panel is
# generated, not hand-written (leCore's `describe` discipline).
# ------------------------------------------------------------------------------------------------

OPS = {}


def op(name, category, inputs=(), params=(), doc="", outputs=("out",), rgba=False,
       alpha="keep", requires=()):
    """requires: leCore faculties this node needs. op_catalog() marks the node
    available/unavailable via have(), so the UI can dim it in the add menu
    BEFORE the person places it and hits the error."""
    def deco(fn):
        OPS[name] = {"fn": fn, "category": category, "inputs": list(inputs),
                     "params": [dict(p) for p in params], "doc": doc,
                     "outputs": list(outputs), "rgba": bool(rgba),
                     "alpha": alpha, "requires": list(requires)}
        return fn
    return deco


def P(name, kind="float", default=0.5, lo=0.0, hi=1.0, choices=None,
      hint=None, when=None):
    """hint: one plain sentence shown as the dial's tooltip.
    when: {choice_param: [values]} -- the dial only bites for those values of
    that choice; the UI dims it (with a tooltip saying so) otherwise."""
    d = {"name": name, "kind": kind, "default": default}
    if kind == "float" or kind == "int":
        d["lo"], d["hi"] = lo, hi
    if choices:
        d["choices"] = choices
    if hint:
        d["hint"] = hint
    if when:
        d["when"] = when
    return d


@op("time", "Generate", inputs=(),
    params=(P("mode", "choice", "normalized",
              choices=["normalized", "frame", "seconds", "pulse"],
              hint="normalized: playhead position 0..1 across the frame "
                   "range; frame/seconds: raw clock scaled by speed; "
                   "pulse: a 0..1..0 triangle each second"),
            P("speed", "float", 1.0, 0.0, 8.0,
              hint="multiplies the clock"),
            P("offset", "float", 0.0, -4.0, 4.0,
              hint="added after scaling")),
    doc="The TIMELINE as a node: a uniform image whose value is the "
        "global playhead, for wiring time into any other node's image "
        "inputs (mix factors, displacement amounts, masks). Everything "
        "it drives scrubs and plays with the timeline -- no clock of "
        "its own.")
def _op_time(size, ins, params):
    h, w = size
    # the evaluator ALWAYS injects _frame (including a real 0.0); a bare
    # fn call without injection is a test harness, which gets a nonzero
    # probe frame so mode/speed visibly bite in the dead-param audit
    f = float(params.get("_frame", 30.0))
    fps = max(float(params.get("_fps", 24.0)), 1e-3)
    lo, hi = params.get("_range", [0.0, 96.0])
    mode = params.get("mode", "normalized")
    if mode == "frame":
        v = f
    elif mode == "seconds":
        v = f / fps
    elif mode == "pulse":
        s = f / fps * max(float(params.get("speed", 1.0)), 1e-6)
        v = 1.0 - abs((s % 1.0) * 2.0 - 1.0)
    else:
        v = (f - lo) / max(hi - lo, 1e-6)
    if mode != "pulse":
        v = v * float(params.get("speed", 1.0))
    v += float(params.get("offset", 0.0))
    out = np.empty((h, w, 4), np.float32)
    out[..., :3] = np.clip(v, 0.0, 1.0)
    out[..., 3] = 1.0
    return out


def _grid_pts(h, w, dims=3):
    ys, xs = np.mgrid[0:h, 0:w]
    cols = [xs.ravel() / max(w - 1, 1), ys.ravel() / max(h - 1, 1)]
    while len(cols) < dims:
        cols.append(np.zeros(h * w))
    return np.stack(cols, 1)


# ---- generators --------------------------------------------------------------------------------

@op("Solid", "Generate", params=[P("r"), P("g"), P("b")],
    doc="One flat colour filling the canvas -- the starting point for backgrounds, a tint source for Blend/Merge, or a colour to cut shapes from with a matte.")
def _solid(ctx, ins, p):
    h, w = ctx
    return np.full((h, w, 3), [p["r"], p["g"], p["b"]], np.float32)


@op("Gradient", "Generate",
    params=[P("angle", "float", 0.0, 0.0, 360.0),
            P("mirror", "bool", False), P("center", "float", 0.5, 0.0, 1.0)],
    doc="A smooth ramp across the canvas -- the bread and butter for sky "
        "fades, lighting falloff, and as a matte that blends two images "
        "gradually (wire it into Mask mix or Merge's matte). Orientation: "
        "angle 0 ramps left->right (dark left); angle 90 ramps top->bottom "
        "(dark top, bright bottom). Turn on mirror for a symmetric V that is "
        "dark at `center` and brightens both ways -- perfect for a spotlight "
        "down the middle or distance-from-a-line masks.")
def _gradient(ctx, ins, p):
    h, w = ctx
    ys, xs = np.mgrid[0:h, 0:w]
    a = np.deg2rad(p["angle"])
    t = (xs / max(w - 1, 1)) * np.cos(a) + (ys / max(h - 1, 1)) * np.sin(a)
    t = (t - t.min()) / max(np.ptp(t), 1e-9)
    if p.get("mirror"):
        t = np.abs(t - float(p["center"])) / max(float(p["center"]),
                                                 1 - float(p["center"]), 1e-6)
    return _rgb(t)


@op("Radial gradient", "Generate",
    params=[P("cx", "float", 0.5, 0.0, 1.0), P("cy", "float", 0.5, 0.0, 1.0),
            P("radius", "float", 0.5, 0.05, 1.5), P("falloff", "float", 1.0, 0.2, 4.0),
            P("aspect", "float", 1.0, 0.2, 5.0), P("invert", "bool", False)],
    doc="A circular ramp radiating from a point -- bright at the centre "
        "(cx, cy), fading to dark at `radius`. The right tool for a sun or "
        "moon disc, a glow, a spotlight, or a round vignette-style matte "
        "(true circles, unlike stacking two straight gradients). falloff "
        "shapes the fade (1 = linear, higher = tighter core); aspect squashes "
        "it into an ellipse; invert flips it to dark-centre.")
def _radial(ctx, ins, p):
    h, w = ctx
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs / max(w - 1, 1) - float(p["cx"]))
    dy = (ys / max(h - 1, 1) - float(p["cy"])) * float(p["aspect"])
    d = np.hypot(dx, dy) / max(float(p["radius"]), 1e-6)
    t = np.clip(1.0 - d, 0, 1) ** float(p["falloff"])
    if p.get("invert"):
        t = 1.0 - t
    return _rgb(t.astype(np.float32))


@op("Band", "Filter", inputs=["image"],
    params=[P("lo", "float", 0.3, 0.0, 1.0), P("hi", "float", 0.6, 0.0, 1.0),
            P("smooth", "float", 0.05, 0.0, 0.5), P("invert", "bool", False)],
    doc="Makes a mask that is bright only where the input's brightness sits "
        "between lo and hi -- an isolate-a-slice tool. Feed it a Gradient to "
        "carve a horizontal band (a treeline strip, a horizon zone), or any "
        "image to select just its midtones. `smooth` softens the edges; "
        "invert selects everything OUTSIDE the range. Replaces the old "
        "Threshold+Invert+multiply chains.")
def _band(ctx, ins, p):
    v = _lum(_rgb(ins["image"]))
    lo, hi = float(p["lo"]), float(p["hi"])
    if hi < lo:
        lo, hi = hi, lo
    sm = max(float(p["smooth"]), 1e-6)
    up = np.clip((v - lo) / sm, 0, 1)                # rising edge at lo
    dn = np.clip((hi - v) / sm, 0, 1)               # falling edge at hi
    m = up * dn
    if p.get("invert"):
        m = 1.0 - m
    return _rgb(m.astype(np.float32))


@op("Pattern", "Generate",
    params=[P("kind", "choice", "fbm", choices=["checker", "stripes", "gradient", "dots", "noise", "fbm"]),
            P("scale", "float", 6.0, 0.5, 40.0), P("seed", "int", 0, 0, 99)],
    doc="A quick deterministic field: checker, stripes, gradient, dots, noise "
        "or fbm. All six also appear in **Procedural texture**, which adds "
        "marble, wood, brick, voronoi, musgrave, wave, magic and white and "
        "gives per-texture controls -- prefer that node for new work. This one "
        "stays because existing documents wire it, and its output is "
        "unchanged.")
def _pattern(ctx, ins, p):
    h, w = ctx
    pat = mind().pattern_field(p["kind"], seed=int(p["seed"]))   # uniform since 0.2.2
    pts = _grid_pts(h, w, 3) * float(p["scale"])
    return _rgb(np.asarray(pat(pts)).reshape(h, w))


@op("Fractal", "Generate",
    params=[P("cx", "float", -0.5, -2.0, 2.0), P("cy", "float", 0.0, -2.0, 2.0),
            P("span", "float", 3.0, 0.001, 4.0), P("power", "float", 2.0, 2.0, 8.0),
            P("julia", "bool", 0), P("jre", "float", -0.8, -1.5, 1.5), P("jim", "float", 0.156, -1.5, 1.5),
            P("iters", "int", 100, 10, 400)],
    doc="The Mandelbrot / Julia fractal as an endlessly detailed grayscale field -- psychedelic backdrops, organic-looking masks, displacement fuel. Zoom and re-seed for infinite variation, then colour it with Gradient map.")
def _fractal(ctx, ins, p):
    h, w = ctx
    jc = (p["jre"], p["jim"]) if p.get("julia") else None
    f = np.asarray(mind().escape_time(width=w, height=h, center=(p["cx"], p["cy"]),
                                      span=p["span"], max_iter=int(p["iters"]),
                                      power=p["power"], julia_c=jc))
    f = (f - f.min()) / max(np.ptp(f), 1e-9)
    return _rgb(f)


# ---- colour ------------------------------------------------------------------------------------

@op("Palette map", "Color", inputs=["image"],
    params=[P("palette", "choice", "cosine", choices=["cosine", "random", "blackbody"]),
            P("phase", "float", 0.0, 0.0, 1.0), P("freq", "float", 1.0, 0.1, 4.0),
            P("seed", "int", 0, 0, 99)],
    doc="Map luminance through a leCore palette: cosine_palette (iq), random_palette "
        "(seeded k-ramp), or a blackbody_color temperature ramp. For a simple two-colour tonal grade, Gradient map is the familiar one.")
def _palette(ctx, ins, p):
    img = _rgb(ins["image"])
    t = np.clip(img.mean(-1) * p["freq"] + p["phase"], 0, None)
    if p["palette"] == "cosine":
        pal = np.asarray(mind().cosine_palette(t.ravel())).reshape(*t.shape, 3)
    elif p["palette"] == "random":
        a, b, c, d = [np.asarray(v, np.float32) for v in
                      mind().random_palette(seed=int(p["seed"]))]
        pal = a + b * np.cos(2 * np.pi * (c * t[..., None] + d))
    else:  # blackbody: map t in [0,1] to 1500K..9000K
        temps = 1500.0 + np.clip(t % 1.0, 0, 1) * 7500.0
        uniq, inv = np.unique(np.round(temps / 50) * 50, return_inverse=True)
        lut = _f32([mind().blackbody_color(float(T)) for T in uniq])
        pal = lut[inv].reshape(*t.shape, 3)
    return np.clip(pal, 0, 1)


@op("Color transfer", "Color", inputs=["image", "reference"],
    params=[P("strength", "float", 1.0, 0.0, 1.0),
            P("mode", "choice", "covariance", choices=["covariance", "meanstd"])],
    doc="Makes the image adopt the colour mood of the reference input (the 'match "
        "this film still' trick). covariance matches the full colour distribution; "
        "meanstd is the lighter classic.")
def _ctransfer(ctx, ins, p):
    return _rgb(mind().color_transfer(_rgb(ins["image"]), _rgb(ins["reference"]),
                                      mode=p["mode"], strength=p["strength"]))


@op("Levels", "Color", inputs=["image"],
    params=[P("black", "float", 0.0, 0.0, 1.0), P("white", "float", 1.0, 0.0, 1.0),
            P("gamma", "float", 1.0, 0.2, 3.0)],
    doc="Black/white points + gamma. For per-range shaping (shadows/mids/highlights separately), see Curves.")
def _levels(ctx, ins, p):
    img = _rgb(ins["image"])
    lo, hi = p["black"], max(p["white"], p["black"] + 1e-3)
    return np.clip(((img - lo) / (hi - lo)) ** (1.0 / p["gamma"]), 0, 1)


@op("Hue / Saturation", "Color", inputs=["image"],
    params=[P("hue", "float", 0.0, -180.0, 180.0), P("sat", "float", 1.0, 0.0, 2.0),
            P("light", "float", 0.0, -0.5, 0.5)],
    doc="The classic colour dial (Photoshop: Hue/Saturation): spin every colour around the wheel, drain or boost intensity, lighten or darken -- turn a red car blue, mute a busy background, or tint a whole shot.")
def _huesat(ctx, ins, p):
    img = _rgb(ins["image"])
    mx, mn = img.max(-1), img.min(-1)
    v = mx; s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    c = mx - mn
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    hh = np.zeros_like(v)
    m = c > 1e-9
    hh[m & (mx == r)] = ((g - b) / np.maximum(c, 1e-9))[m & (mx == r)] % 6
    hh[m & (mx == g)] = ((b - r) / np.maximum(c, 1e-9) + 2)[m & (mx == g)]
    hh[m & (mx == b)] = ((r - g) / np.maximum(c, 1e-9) + 4)[m & (mx == b)]
    hh = (hh * 60 + p["hue"]) % 360 / 60
    s = np.clip(s * p["sat"], 0, 1); v = np.clip(v + p["light"], 0, 1)
    c = v * s; x = c * (1 - np.abs(hh % 2 - 1)); mmm = v - c
    out = np.zeros_like(img)
    for i, (rr, gg, bb) in enumerate([(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]):
        sel = (hh >= i) & (hh < i + 1)
        out[..., 0][sel] = (rr if np.isscalar(rr) else rr[sel])
        out[..., 1][sel] = (gg if np.isscalar(gg) else gg[sel])
        out[..., 2][sel] = (bb if np.isscalar(bb) else bb[sel])
    return np.clip(out + mmm[..., None], 0, 1)


@op("Invert", "Color", inputs=["image"], doc="Flips the image to its photographic negative -- brights go dark, colours swap to their opposites. Handy on mattes too: invert a selection's matte to affect everything EXCEPT the subject.")
def _invert(ctx, ins, p):
    return 1.0 - _rgb(ins["image"])


@op("Posterize", "Color", inputs=["image"],
    params=[P("colors", "int", 6, 2, 24), P("seed", "int", 0, 0, 99)],
    doc="leCore image_colours: quantise to the image's own k-means palette.")
def _posterize(ctx, ins, p):
    img = _rgb(ins["image"])
    palette, _ = mind().image_colours(img, k=int(p["colors"]), seed=int(p["seed"]),
                                      as_float=True)          # leCore >= 0.2.2
    pal = _f32(palette)
    flat = img.reshape(-1, 3)
    idx = np.argmin(((flat[:, None, :] - pal[None]) ** 2).sum(-1), 1)
    return pal[idx].reshape(img.shape)


# ---- filters -----------------------------------------------------------------------------------

@op("Spectral chain", "Filter", inputs=["image"],
    params=[P("blur", "float", 3.0, 0.0, 24.0),
            P("shift_x", "float", 0.0, -40.0, 40.0), P("shift_y", "float", 0.0, -40.0, 40.0),
            P("unsharp", "float", 0.0, 0.0, 1.5), P("unsharp_r", "float", 8.0, 1.0, 40.0),
            P("gain", "float", 1.0, 0.0, 2.0)],
    doc="leCore shader_pipeline: blur -> sub-pixel shift -> unsharp -> gain compiled "
        "into ONE spectral transfer before any pixel is touched -- one FFT round-trip "
        "no matter how many stages, and the shift is EXACT at fractional pixels "
        "(a phase ramp has no resampling filter). The compiled transfer is cached "
        "per size+settings, so repeated frames (video!) pay only the apply.", alpha="process")
def _spectral(ctx, ins, p, _cache={}):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    key = (h, w, round(p["blur"], 3), round(p["shift_x"], 3), round(p["shift_y"], 3),
           round(p["unsharp"], 3), round(p["unsharp_r"], 3), round(p["gain"], 3))
    pipe = _cache.get(key)
    if pipe is None:
        def gk(sig):
            r = max(int(sig * 3), 1)
            x = np.arange(-r, r + 1)
            k = np.exp(-x ** 2 / (2 * max(sig, 1e-3) ** 2))
            k = np.outer(k, k)
            return k / k.sum()
        pipe = mind().shader_pipeline((h, w))
        if p["blur"] > 1e-3:
            pipe = pipe.blur(gk(p["blur"]))
        if abs(p["shift_x"]) > 1e-6 or abs(p["shift_y"]) > 1e-6:
            pipe = pipe.translate((float(p["shift_y"]), float(p["shift_x"])))
        if p["unsharp"] > 1e-3:
            pipe = pipe.unsharp(gk(p["unsharp_r"]), float(p["unsharp"]))
        if abs(p["gain"] - 1) > 1e-6:
            pipe = pipe.gain(float(p["gain"]))
        _cache.clear() if len(_cache) > 8 else None
        _cache[key] = pipe
    out = np.asarray(pipe.apply(img))                    # channel-batched since 0.2.2
    return np.clip(out, 0, 1)


ASSETS = {}                 # id -> {"name", "path", "data": bytes, "ext"}
_ASSET_MESH = {}            # id -> cached Mesh


def register_asset(name, data, ext, aid=None):
    """Register an uploaded 3-D model (.obj / .glb / .gltf bytes). Returns the
    asset id. The mesh itself is imported lazily on first render."""
    import re as _re
    aid = aid or ("A%d" % (max([int(_re.sub(r'\D', '', k) or 0)
                                for k in ASSETS] + [0]) + 1))
    ASSETS[aid] = {"name": name, "data": bytes(data), "ext": ext.lower()}
    _ASSET_MESH.pop(aid, None)
    _mut()
    return aid


def asset_mesh(aid):
    """The imported Mesh for an asset id (cached). Raises KeyError for an
    unknown id."""
    if aid in _ASSET_MESH:
        return _ASSET_MESH[aid]
    a = ASSETS[aid]
    import tempfile, os
    d = tempfile.mkdtemp()
    p = os.path.join(d, "asset" + a["ext"])
    with open(p, "wb") as f:
        f.write(a["data"])
    la = mind().import_asset(p)
    mesh = la.mesh() if callable(getattr(la, "mesh", None)) else         getattr(la, "mesh", la)
    _ASSET_MESH[aid] = mesh
    return mesh


def _mut():
    _MUT_REV[0] += 1


# ---- Shadertoy support: the browser's GPU renders, the graph orchestrates ----
SHADER_GEN = [0]        # bumped when a frame/error arrives: joins the node sig
SHADER_FRAMES = {}      # key -> (H, W, 4) float32 rendered by the client
SHADER_ERRORS = {}      # key -> GLSL compile/link error text from the client
SHADER_PENDING = {}     # key -> spec the client should render


_ST_DEFAULT = """// Real Shadertoy code runs here, unchanged: mainImage, iTime,
// iResolution, iMouse, iChannel0/1 all work. Paste any shader from
// shadertoy.com that uses those. `time` below is the node's time dial --
// wire a Value node into it to animate.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5*cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    fragColor = vec4(col, 1.0);
}"""


def _st_key(src, t, mx, my, w, h, chsig=""):
    import hashlib
    return hashlib.md5(("%s|%.4f|%.4f|%.4f|%dx%d|%s" %
                        (src, t, mx, my, w, h, chsig)).encode()).hexdigest()


def _st_placeholder(h, w, msg):
    """A calm 'rendering in your browser' card (PIL text over dark)."""
    img = np.full((h, w, 3), [0.055, 0.06, 0.08], np.float32)
    try:
        from PIL import Image, ImageDraw, ImageFont
        cv = Image.fromarray((img * 255).astype("uint8"))
        d = ImageDraw.Draw(cv)
        try:
            fnt = ImageFont.truetype(_font_path(None), max(12, h // 16))
        except Exception:
            fnt = ImageFont.load_default()
        tw = d.textlength(msg, font=fnt)
        d.text(((w - tw) / 2, h / 2 - h // 20), msg, fill=(150, 160, 190),
               font=fnt)
        img = np.asarray(cv, np.float32) / 255.0
    except Exception:
        pass
    return img


_SAMPLE_OBJ = (b"v -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\nv 0 1.7 0\n"
               b"v -0.55 0 -0.55\nv 0.55 0 -0.55\nv 0.55 0 0.55\nv -0.55 0 0.55\n"
               b"f 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\nf 4 3 2 1\n"
               b"f 6 7 8 9\n")


def _ensure_sample_asset():
    if "sample" not in ASSETS:
        ASSETS["sample"] = {"name": "Sample pyramid (built-in)",
                            "data": _SAMPLE_OBJ, "ext": ".obj"}


_FONTS = None


def list_fonts():
    """{name: path} for every TrueType/OpenType font we can find, scanned once.
    Searches the usual system directories AND matplotlib's bundled fonts (which
    ship with the package, so text always works even on a headless box with no
    system fonts installed). Names come from the filename stem."""
    global _FONTS
    if _FONTS is None:
        import glob, os
        _FONTS = {}
        pats = ["/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf",
                "/usr/share/fonts/**/*.ttc",
                "/usr/local/share/fonts/**/*.ttf",
                "/Library/Fonts/**/*.ttf", "/Library/Fonts/**/*.ttc",
                "/System/Library/Fonts/**/*.ttf",
                "C:/Windows/Fonts/*.ttf", "C:/Windows/Fonts/*.ttc",
                os.path.expanduser("~/.fonts/**/*.ttf"),
                os.path.expanduser("~/.local/share/fonts/**/*.ttf")]
        # matplotlib bundles DejaVu + others and is a hard dependency, so this
        # guarantees at least a dozen usable fonts on any platform
        try:
            import matplotlib
            mpl = os.path.join(os.path.dirname(matplotlib.__file__),
                               "mpl-data", "fonts", "ttf")
            pats.append(mpl + "/*.ttf")
        except Exception:
            pass
        for pat in pats:
            for p in glob.glob(pat, recursive=True):
                _FONTS.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return dict(_FONTS)


def _font_path(name):
    fonts = list_fonts()
    if name and name in fonts:
        return fonts[name]
    for pref in ("DejaVuSans", "Arial", "LiberationSans-Regular", "FreeSans",
                 "Carlito-Regular", "DejaVuSans-Bold"):
        if pref in fonts:
            return fonts[pref]
    if fonts:
        return next(iter(fonts.values()))
    # last-ditch: PIL's built-in bitmap font path (always present)
    from PIL import ImageFont
    fallback = getattr(ImageFont, "load_default", None)
    if fallback is not None:
        f = ImageFont.load_default()
        p = getattr(f, "path", None)
        if p:
            return p
    raise RuntimeError("no usable fonts found (install any .ttf, or matplotlib)")


def _segment_compat(img, k, seed=0, max_dim=128):
    """mind().segment_image with a fallback for cores predating max_dim
    (published 0.2.2 wheels lack it; 0.2.3 has it): downsample ourselves,
    segment, and upsample the masks back to full resolution."""
    m = mind()
    try:
        return m.segment_image(img, k=k, seed=seed, max_dim=max_dim)
    except TypeError:
        h, w = img.shape[:2]
        sc = max(h, w) / float(max_dim)
        if sc <= 1.0:
            return m.segment_image(img, k=k, seed=seed)
        sh, sw = max(int(round(h / sc)), 8), max(int(round(w / sc)), 8)
        segs = m.segment_image(_resize(img, sh, sw), k=k, seed=seed)
        def up(sg):                       # segments may be dicts with a "mask"
            if isinstance(sg, dict):
                sg = dict(sg)
                sg["mask"] = np.clip(_resize(
                    np.asarray(sg["mask"], np.float32), h, w), 0, 1) > 0.5
                return sg
            return np.clip(_resize(np.asarray(sg, np.float32), h, w), 0, 1)
        return [up(sg) for sg in segs]


def _to_rgba(v, h, w):
    """Anything on a wire becomes (h, w, 4): scalars broadcast to constant grey,
    RGB gains opaque alpha, RGBA passes through resized."""
    if isinstance(v, (int, float)):
        g = np.full((h, w), float(v), np.float32)
        return np.stack([g, g, g, np.ones_like(g)], -1)
    a = np.asarray(v, np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
    if a.shape[:2] != (h, w):
        a = _resize(a, h, w)
    return a


def _conform_rgba(out, h, w, in_alpha):
    """Op results become RGBA. RGB ops pass the first input's alpha through
    untouched (filters don't invent transparency); RGBA-aware ops own theirs."""
    a = np.asarray(out, np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], -1)
    if a.shape[2] == 3:
        al = in_alpha if in_alpha is not None else np.ones(a.shape[:2], np.float32)
        if al.shape != a.shape[:2]:
            al = _resize(al[..., None], a.shape[0], a.shape[1])[..., 0]
        a = np.concatenate([np.clip(a, 0, 1), al[..., None]], -1)
    if a.shape[:2] != (h, w):
        a = _resize(a, h, w)
    return a


def _lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


@op("Brightness / Contrast", "Adjust", inputs=["image"],
    params=[P("brightness", "float", 0.0, -1.0, 1.0),
            P("contrast", "float", 0.0, -1.0, 1.0)],
    doc="The everyday tonal tweak (Photoshop/GIMP: Brightness-Contrast). "
        "Brightness shifts everything; contrast pivots around mid-grey.")
def _brightcon(ctx, ins, p):
    img = _rgb(ins["image"])
    c = float(np.tan((p["contrast"] * 0.999 + 1) * np.pi / 4))
    return np.clip((img - 0.5) * c + 0.5 + p["brightness"], 0, 1)


@op("Curves", "Adjust", inputs=["image"],
    params=[P("channel", "choice", "rgb", choices=["rgb", "r", "g", "b"]),
            P("shadows", "float", 0.0, -0.5, 0.5),
            P("midtones", "float", 0.0, -0.5, 0.5),
            P("highlights", "float", 0.0, -0.5, 0.5)],
    doc="A three-point tone curve (Photoshop: Curves): lift or crush shadows, "
        "midtones, and highlights independently, per channel or on all of RGB. "
        "The curve is smooth and monotone -- no banding, no clipping surprises.")
def _curves(ctx, ins, p):
    img = _rgb(ins["image"]).copy()
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ys = np.clip(np.array([0.0, 0.25 + p["shadows"], 0.5 + p["midtones"],
                           0.75 + p["highlights"], 1.0]), 0, 1)
    ys = np.maximum.accumulate(ys)                    # keep it monotone
    lut_x = np.linspace(0, 1, 256)
    lut = np.interp(lut_x, xs, ys)
    k = np.ones(9) / 9.0                              # soften the knees
    lut = np.convolve(np.pad(lut, 4, mode="edge"), k, mode="valid")
    chans = {"rgb": [0, 1, 2], "r": [0], "g": [1], "b": [2]}[p["channel"]]
    for c in chans:
        img[..., c] = np.interp(img[..., c], lut_x, lut)
    return np.clip(img, 0, 1)


@op("Vibrance", "Adjust", inputs=["image"],
    params=[P("vibrance", "float", 0.4, -1.0, 1.0),
            P("saturation", "float", 0.0, -1.0, 1.0)],
    doc="Smart saturation (Photoshop: Vibrance): boosts muted colours much "
        "more than already-vivid ones, so skies pop without skin going neon. "
        "The saturation slider is the ordinary uniform version, for contrast.")
def _vibrance(ctx, ins, p):
    img = _rgb(ins["image"])
    mx = img.max(-1, keepdims=True); mn = img.min(-1, keepdims=True)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    lum = _lum(img)[..., None]
    amt = p["vibrance"] * (1.0 - sat) ** 2 + p["saturation"]
    return np.clip(lum + (img - lum) * (1.0 + amt), 0, 1)


@op("Color balance", "Adjust", inputs=["image"],
    params=[P("range", "choice", "midtones", choices=["shadows", "midtones", "highlights"]),
            P("cyan_red", "float", 0.0, -1.0, 1.0),
            P("magenta_green", "float", 0.0, -1.0, 1.0),
            P("yellow_blue", "float", 0.0, -1.0, 1.0)],
    doc="Photoshop's Color Balance: push shadows, midtones, or highlights "
        "along the cyan-red, magenta-green, and yellow-blue axes. Stack three "
        "nodes (one per range) for the full classic panel.")
def _colorbalance(ctx, ins, p):
    img = _rgb(ins["image"])
    l = _lum(img)
    wt = {"shadows": (1 - l) ** 2, "midtones": 4 * l * (1 - l),
          "highlights": l ** 2}[p["range"]][..., None]
    shift = np.array([p["cyan_red"], p["magenta_green"], p["yellow_blue"]],
                     np.float32) * 0.35
    return np.clip(img + wt * shift[None, None], 0, 1)


@op("Black & white", "Adjust", inputs=["image"],
    params=[P("red", "float", 0.4, -1.0, 2.0), P("green", "float", 0.6, -1.0, 2.0),
            P("blue", "float", 0.2, -1.0, 2.0),
            P("tint", "float", 0.0, 0.0, 1.0), P("tint_hue", "float", 0.1, 0.0, 1.0)],
    doc="Channel-weighted mono conversion (Photoshop: Black & White): choose "
        "how much each colour contributes to the grey -- darken a sky by "
        "pulling blue down, brighten foliage by pushing green up. Optional "
        "duotone tint (sepia at the defaults).")
def _blackwhite(ctx, ins, p):
    img = _rgb(ins["image"])
    w = np.array([p["red"], p["green"], p["blue"]], np.float32)
    tot = w.sum()
    if abs(tot) > 1e-6:
        w = w / tot
    g = np.clip((img * w[None, None]).sum(-1), 0, 1)
    if p["tint"] > 1e-3:
        h = p["tint_hue"] * 2 * np.pi
        tint = 0.5 + 0.5 * np.cos(h - np.array([0.0, 2.094, 4.188]))
        col = g[..., None] * (1 - p["tint"] * 0.5) + \
            g[..., None] * tint[None, None] * p["tint"] * 0.5
        return np.clip(col, 0, 1)
    return np.stack([g, g, g], -1)


@op("Photo filter", "Adjust", inputs=["image"],
    params=[P("filter", "choice", "warming", choices=["warming", "cooling", "sepia", "magenta", "green"]),
            P("density", "float", 0.3, 0.0, 1.0),
            P("preserve_lum", "bool", 1)],
    doc="A colour gel over the lens (Photoshop: Photo Filter): warming 85, "
        "cooling 80, and friends. Preserve-luminosity keeps exposure steady "
        "while the cast shifts.")
def _photofilter(ctx, ins, p):
    img = _rgb(ins["image"])
    col = {"warming": (0.925, 0.541, 0.0), "cooling": (0.0, 0.42, 1.0),
           "sepia": (0.7, 0.5, 0.25), "magenta": (0.9, 0.2, 0.8),
           "green": (0.2, 0.8, 0.25)}[p["filter"]]
    d = p["density"]
    out = img * (1 - d) + img * np.asarray(col, np.float32)[None, None] * d * 2
    out = np.clip(out, 0, 1)
    if int(p["preserve_lum"]):
        l0, l1 = _lum(img), _lum(out)
        out = np.clip(out * ((l0 + 1e-4) / (l1 + 1e-4))[..., None], 0, 1)
    return out


@op("Channel mixer", "Adjust", inputs=["image"],
    params=[P("channel", "choice", "red", choices=["red", "green", "blue"]),
            P("from_red", "float", 1.0, -2.0, 2.0),
            P("from_green", "float", 0.0, -2.0, 2.0),
            P("from_blue", "float", 0.0, -2.0, 2.0)],
    doc="Rebuild one output channel from a mix of the input channels "
        "(Photoshop: Channel Mixer). Chain one node per channel for a full "
        "matrix -- infrared looks, channel swaps, custom mono.")
def _channelmixer(ctx, ins, p):
    img = _rgb(ins["image"]).copy()
    mix = img[..., 0] * p["from_red"] + img[..., 1] * p["from_green"] \
        + img[..., 2] * p["from_blue"]
    img[..., {"red": 0, "green": 1, "blue": 2}[p["channel"]]] = np.clip(mix, 0, 1)
    return img


@op("Threshold", "Adjust", inputs=["image"],
    params=[P("level", "float", 0.5, 0.0, 1.0), P("smooth", "float", 0.0, 0.0, 0.2)],
    doc="Hard black-and-white cut at a luminance level (Photoshop/GIMP: "
        "Threshold). A touch of smooth turns the cliff into a short ramp -- "
        "great masks, posters, and screen-print looks.")
def _threshold(ctx, ins, p):
    l = _lum(_rgb(ins["image"]))
    if p["smooth"] > 1e-4:
        g = np.clip((l - p["level"]) / p["smooth"] + 0.5, 0, 1)
    else:
        g = (l >= p["level"]).astype(np.float32)
    return np.stack([g, g, g], -1)


_GMAP_PRESETS = {                       # shadow rgb -> highlight rgb
    "custom":       None,
    "teal-gold":    ((0.05, 0.05, 0.25), (1.0, 0.85, 0.55)),
    "meadow green": ((0.10, 0.26, 0.08), (0.60, 0.82, 0.30)),
    "sky blue":     ((0.20, 0.36, 0.62), (0.86, 0.93, 0.98)),
    "golden hour":  ((0.22, 0.10, 0.15), (1.0, 0.80, 0.42)),
    "autumn":       ((0.18, 0.06, 0.03), (0.95, 0.62, 0.18)),
    "dusk":         ((0.08, 0.06, 0.20), (0.92, 0.52, 0.55)),
    "moonlight":    ((0.02, 0.04, 0.10), (0.62, 0.72, 0.90)),
    "ember":        ((0.06, 0.01, 0.02), (1.0, 0.42, 0.12)),
}


@op("Gradient map", "Adjust", inputs=["image"],
    params=[P("preset", "choice", "custom", choices=list(_GMAP_PRESETS)),
            P("shadow_r", "float", 0.05, 0.0, 1.0), P("shadow_g", "float", 0.05, 0.0, 1.0),
            P("shadow_b", "float", 0.25, 0.0, 1.0),
            P("highlight_r", "float", 1.0, 0.0, 1.0), P("highlight_g", "float", 0.85, 0.0, 1.0),
            P("highlight_b", "float", 0.55, 0.0, 1.0),
            P("reverse", "bool", 0)],
    doc="Map tones to a two-colour gradient (Photoshop: Gradient Map): "
        "shadows take the first colour, highlights the second, midtones "
        "blend between. Pick a `preset` (meadow green, sky blue, golden hour, "
        "autumn, dusk...) to fill the six colour sliders instantly, then nudge "
        "them -- the picker just seeds values, it doesn't lock them. Leave it "
        "on 'custom' to dial your own. For wilder multi-stop palettes, see "
        "Palette map.")
def _gradientmap(ctx, ins, p):
    t = _lum(_rgb(ins["image"]))[..., None]
    if int(p["reverse"]):
        t = 1 - t
    preset = _GMAP_PRESETS.get(p.get("preset", "custom"))
    if preset is not None:
        (sr, sg, sb), (hr, hg, hb) = preset
        a = np.array([sr, sg, sb], np.float32)
        b = np.array([hr, hg, hb], np.float32)
    else:
        a = np.array([p["shadow_r"], p["shadow_g"], p["shadow_b"]], np.float32)
        b = np.array([p["highlight_r"], p["highlight_g"], p["highlight_b"]], np.float32)
    return np.clip(a[None, None] * (1 - t) + b[None, None] * t, 0, 1)


@op("Pixelize", "Filter", inputs=["image"],
    params=[P("size", "int", 12, 2, 64)],
    doc="Mosaic blocks (GIMP: Pixelize; Photoshop: Mosaic) -- the classic "
        "censor / retro effect. Each block becomes its average colour.", alpha="process")
def _pixelize(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    n = int(p["size"])
    sh, sw = max(h // n, 1), max(w // n, 1)
    small = _resize(img, sh, sw)
    ys = (np.arange(h) * sh // h).clip(0, sh - 1)
    xs = (np.arange(w) * sw // w).clip(0, sw - 1)
    return small[ys[:, None], xs[None, :]]


@op("Motion blur", "Filter", inputs=["image"],
    params=[P("length", "float", 18.0, 1.0, 80.0),
            P("angle", "float", 0.0, 0.0, 180.0)],
    doc="Linear motion streaks at an angle (GIMP/Photoshop: Motion Blur) -- "
        "speed, drama, or a cheap long exposure.", alpha="process")
def _motionblur(ctx, ins, p):
    import cv2
    img = _rgb(ins["image"])
    L = max(int(p["length"]), 1)
    k = np.zeros((L, L), np.float32)
    cv2.line(k, (0, L // 2), (L - 1, L // 2), 1.0, 1)
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), -float(p["angle"]), 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= max(k.sum(), 1e-6)
    return np.clip(cv2.filter2D(img, -1, k), 0, 1)


@op("Emboss", "Filter", inputs=["image"],
    params=[P("angle", "float", 45.0, 0.0, 360.0),
            P("depth", "float", 1.0, 0.1, 4.0),
            P("keep_color", "bool", 0)],
    doc="Carve the image into lit relief (GIMP/Photoshop: Emboss): edges "
        "become ridges lit from the chosen angle. keep_color embosses on top "
        "of the original colours instead of grey.")
def _emboss(ctx, ins, p):
    import cv2
    img = _rgb(ins["image"])
    a = np.deg2rad(p["angle"])
    dx, dy = np.cos(a), np.sin(a)
    k = np.array([[-dx - dy, -dy, 0], [-dx, 0, dx], [0, dy, dx + dy]],
                 np.float32) * float(p["depth"])
    rel = cv2.filter2D(_lum(img), -1, k)
    if int(p["keep_color"]):
        return np.clip(img + rel[..., None], 0, 1)
    return np.clip(np.stack([rel, rel, rel], -1) + 0.5, 0, 1)


@op("Blur", "Filter", inputs=["image"],
    params=[P("sigma", "float", 3.0, 0.0, 30.0)],
    doc="Spectral Gaussian blur (a diagonal operator -- the postfx algebra's atom).", alpha="process")
def _blur(ctx, ins, p):
    return np.clip(_gauss_blur(_rgb(ins["image"]), p["sigma"]), 0, 1)


@op("Sharpen", "Filter", inputs=["image"],
    params=[P("sigma", "float", 2.0, 0.3, 10.0), P("amount", "float", 0.8, 0.0, 3.0)],
    doc="The everyday sharpener (Photoshop: Unsharp Mask): boosts contrast along edges so the image reads crisper. For rescuing a genuinely blurry photo, Deconvolve digs deeper; this one is for the final snap.", alpha="process")
def _sharpen(ctx, ins, p):
    img = _rgb(ins["image"])
    return np.clip(img + p["amount"] * (img - _gauss_blur(img, p["sigma"])), 0, 1)


@op("Denoise", "Filter", inputs=["image"],
    params=[P("sigma", "float", 1.5, 0.2, 8.0), P("edge_keep", "float", 0.6, 0.0, 1.0)],
    doc="Edge-preserving smooth: blur held back where leCore's edge map fires.", alpha="process")
def _denoise(ctx, ins, p):
    img = _rgb(ins["image"])
    e = _f32(mind().image_edges(img))
    e = _gauss_blur(e, 1.0)
    e = e / max(e.max(), 1e-9)
    keep = np.clip(e * p["edge_keep"] * 3.0, 0, 1)[..., None]
    return np.clip(img * keep + _gauss_blur(img, p["sigma"]) * (1 - keep), 0, 1)


@op("Edges", "Filter", inputs=["image"],
    params=[P("quantile", "float", 0.85, 0.5, 0.99)],
    doc="Traces the outlines in the image -- a white-on-black drawing of every edge (leCore's self-calibrating detector). Use it for sketch looks, as a matte for sharpening only along edges, or to drive Displace for a hand-drawn wobble.")
def _edges(ctx, ins, p):
    e = _f32(mind().image_edges(_rgb(ins["image"]), quantile=p["quantile"]))
    return _rgb(e / max(e.max(), 1e-9))


@op("Fill out", "Output", inputs=["image"],
    doc="Marks this image as a FILL SOURCE: the flood-fill tool's 'node' mode "
        "lists every Fill out node, and filling stamps the region with this "
        "image sampled per-pixel at canvas size. Passes its input through.")
def _fill_out(ctx, ins, p):
    return _rgb(ins["image"])


@op("Media in", "Input", inputs=[],
    params=[P("source", "text", ""), P("fps", "float", 10.0, 0.0, 30.0),
            P("play", "bool", 1), P("pos", "float", 0.0, 0.0, 1.0)],
    doc="External media: an image file path, a video file, a network stream URL "
        "(MJPEG/RTSP/HTTP), or test:clock for a built-in animated test signal. "
        "fps is how often the frame refreshes while a live session runs.")
def _media_in(ctx, ins, p):
    h, w = ctx
    return np.zeros((h, w, 3), np.float32)     # resolved by the server media hook


@op("Segment", "Filter", inputs=["image"],
    params=[P("k", "int", 5, 2, 8), P("seed", "int", 0, 0, 99),
            P("detail", "int", 128, 64, 384,
              hint="Region-finding resolution: lower = much faster, masks stay full size")],
    outputs=["out"] + [f"seg{i}" for i in range(1, 9)],
    doc="Splits the picture into perceptual regions by colour: `out` is the "
        "image painted with each region's mean colour, and seg1..segK are the "
        "individual region masks (largest first), each on its own socket -- "
        "wire one into Mask mix, Mask out, or Inpaint to work on just that "
        "object. `detail` is the speed/quality dial: the sweep runs at that "
        "resolution and the masks come back full size, so lower is much "
        "faster. Busy or noisy images cost far more than flat ones -- raise "
        "detail only when you need finer region edges.")
def _segment(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    segs = _segment_compat(img, k=int(p["k"]), seed=int(p["seed"]),
                           max_dim=int(p.get("detail", 128)))
    out = np.zeros_like(img)
    masks = []
    for s in segs:
        msk = np.asarray(s["mask"] if isinstance(s, dict) and "mask" in s else s)
        if msk.dtype != bool:
            msk = msk > 0.5
        if msk.shape != (h, w):                      # defensive conform (masks are
            msk = _resize(msk.astype(np.float32), h, w) > 0.5     # full-res in 0.2.2)
        out[msk] = img[msk].mean(0) if msk.any() else 0
        masks.append(msk.astype(np.float32))
    masks.sort(key=lambda m: -m.sum())
    res = {"out": out}
    for i in range(8):
        res[f"seg{i+1}"] = _rgb(masks[i]) if i < len(masks) else np.zeros((h, w, 3), np.float32)
    return res


@op("Inpaint", "Filter", inputs=["image", "mask"],
    params=[P("invert", "bool", 0)],
    doc="leCore inpaint: ERASES the image where the mask is BRIGHT and re-grows those "
        "pixels harmonically from the surrounding image (remove objects, fill holes). "
        "With no mask wired there is nothing to fill, so the image passes through. "
        "Set invert if your mask marks the region to KEEP instead.")
def _inpaint(ctx, ins, p):
    img = _rgb(ins["image"])
    hole = _rgb(ins["mask"]).mean(-1) > 0.5
    if int(p.get("invert", 0)):
        hole = ~hole
    known = ~hole
    out = np.asarray(mind().inpaint(img.astype(float), known))   # (H,W,3) since 0.2.2
    return np.clip(out, 0, 1)


@op("Upscale 2x", "Filter", inputs=["image"],
    params=[P("sharpness", "float", 0.4, 0.0, 1.0)],
    doc="Doubles the pixel detail with leCore's smart upscaler -- edges stay crisp instead of going soft like a plain zoom. Chain two for 4x. (The graph then fits the result back to your canvas, so use it before an Export at higher resolution.)", alpha="process")
def _upscale(ctx, ins, p):
    return _rgb(mind().upscale(_rgb(ins["image"]), 2.0, sharpness=p["sharpness"]))


@op("Displace", "Filter", inputs=["image", "map"],
    params=[P("amount", "float", 12.0, 0.0, 80.0)],
    doc="Shifts each pixel by the brightness of a second image (GIMP/Photoshop: "
        "Displace): bright areas of the map push pixels further. Wire clouds or "
        "noise into the map for heat-haze, glass, or flag-ripple looks.", alpha="process")
def _displace(ctx, ins, p):
    img = _rgb(ins["image"]); mp = _rgb(ins["map"]).mean(-1)
    h, w = img.shape[:2]
    gy, gx = np.gradient(mp)
    ys, xs = np.mgrid[0:h, 0:w]
    yy = np.clip(ys + gy * p["amount"] * h, 0, h - 1).astype(int)
    xx = np.clip(xs + gx * p["amount"] * w, 0, w - 1).astype(int)
    return img[yy, xx]


# ---- combine / fx ------------------------------------------------------------------------------

@op("Premult", "Comp", inputs=["image"], rgba=True,
    doc="Prepares a transparent element for maths (Nuke: Premult): multiplies the colour by its own transparency so glass, glow and smoke ADD correctly over a background. Rule of thumb: Premult before Blend add/screen, Unpremult before colour-correcting, Merge handles it for you.")
def _premult(ctx, ins, p):
    v = ins["image"]
    return np.concatenate([v[..., :3] * v[..., 3:4], v[..., 3:4]], -1)


@op("Unpremult", "Comp", inputs=["image"], rgba=True,
    doc="Divide RGB by alpha (Nuke: Unpremult). Do colour corrections on "
        "unpremultiplied images to avoid dark fringes on soft edges, then "
        "Premult again.")
def _unpremult(ctx, ins, p):
    v = ins["image"]
    a = np.maximum(v[..., 3:4], 1e-6)
    return np.concatenate([np.clip(v[..., :3] / a, 0, 1), v[..., 3:4]], -1)


@op("Set alpha", "Comp", inputs=["image", "alpha"], rgba=True,
    doc="Replace the stream's alpha with the luminance of the alpha input "
        "(Nuke: Copy into rgba.alpha) -- attach any matte to any image.")
def _setalpha(ctx, ins, p):
    v = ins["image"]
    aw = ins.get("alpha")
    a = _lum(_rgb(aw)) if aw is not None else np.ones(v.shape[:2], np.float32)
    return np.concatenate([v[..., :3], a[..., None]], -1)


@op("Fluid", "FX", inputs=["image", "solid"],
    params=[P("steps", "int", 40, 1, 200,
              hint="Simulation steps — scrub it to watch the ink move"),
            P("buoyancy", "float", 30.0, -120.0, 120.0,
              hint="Dense areas rise (positive) or sink like ink in water"),
            P("wind", "float", 0.0, -80.0, 80.0, hint="Steady sideways push"),
            P("swirl", "float", 12.0, 0.0, 60.0,
              hint="Curl-noise stirring — turbulence without hand-placed forces"),
            P("viscosity", "float", 0.0, 0.0, 0.05),
            P("dissipate", "float", 0.0, 0.0, 0.05,
              hint="Dye fades a little each step — smoke thins, ink lingers"),
            P("seed", "int", 0, 0, 9999),
            P("r", "float", 0.9, 0.0, 1.0), P("g", "float", 0.9, 0.0, 1.0),
            P("b", "float", 1.0, 0.0, 1.0)],
    doc="A REAL fluid solve (leCore's grid solver): the input's luminance is "
        "dye density, buoyancy lifts or sinks it, curl-noise stirs it, and an "
        "optional `solid` matte is an obstacle the flow moves around. Scrub "
        "`steps` to animate smoke rising off a stroke or ink blooming in "
        "water. Deterministic per seed, so it renders the same twice.",
    rgba=True, alpha="own", outputs=["out", "density"])
def _fluid(ctx, ins, p):
    h, w = ctx
    # the solve runs on a capped grid: 20 steps at 96x128 measured 0.05 s,
    # and smoke does not need canvas-resolution advection to read as smoke
    gw = min(192, w)
    gh = max(2, int(round(h * gw / max(w, 1))))
    src = ins.get("image")
    if src is None:
        den = np.zeros((gh, gw), np.float32)
    else:
        den = _resize(_rgb(np.asarray(src, np.float32)), gh, gw)[..., 0]
    solid = ins.get("solid")
    sol = None
    if solid is not None:
        sol = (_resize(_rgb(np.asarray(solid, np.float32)), gh, gw)[..., 0]
               > 0.5)
    vx = np.zeros((gh, gw), np.float32)
    vy = np.zeros((gh, gw), np.float32)
    fy = np.full((gh, gw), -float(p["buoyancy"]), np.float32) * den
    fx = np.full((gh, gw), float(p["wind"]), np.float32)
    sw = float(p["swirl"])
    if sw > 0:
        cx, cy = _curl_noise(32, 3, int(p["seed"]))
        fx = fx + _resize(np.asarray(cx, np.float32)[..., None],
                          gh, gw)[..., 0] * sw
        fy = fy + _resize(np.asarray(cy, np.float32)[..., None],
                          gh, gw)[..., 0] * sw
    diss = 1.0 - float(p["dissipate"])
    mm = mind()
    for _ in range(int(p["steps"])):
        vx, vy, den = mm.fluid_step(vx, vy, den, dt=0.06,
                                    viscosity=float(p["viscosity"]),
                                    fx=fx, fy=fy * (0.3 + 0.7 * den),
                                    solid=sol)
        if diss < 1.0:
            den = den * diss
    den = np.clip(np.asarray(den, np.float32), 0.0, 1.5)
    dfull = _resize(den[..., None], h, w)[..., 0]
    a = np.clip(dfull, 0.0, 1.0)
    col = np.array([p["r"], p["g"], p["b"]], np.float32)
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = col * a[..., None]              # premultiplied
    out[..., 3] = a
    return {"out": out, "density": _rgb(np.clip(dfull, 0, 1))}


@op("Light direction", "Values", inputs=["image"],
    params=[P("power", "float", 2.0, 0.5, 8.0,
              hint="How strongly highlights dominate the estimate")],
    doc="Estimates where the light in an image comes from (leCore's "
        "gradient-statistics estimator) and puts the answer ON WIRES: x and "
        "y of the direction (0..1, canvas sense) and the angle as a 0..1 "
        "turn. Wire them into any parameter -- or read the numbers to set "
        "the impasto key light to match a photo's lighting.",
    outputs=["out", "x", "y", "angle"])
def _lightdir(ctx, ins, p):
    h, w = ctx
    img = _rgb(np.asarray(ins.get("image"), np.float32))
    try:
        d = np.asarray(mind().estimate_light_direction(
            img, power=float(p["power"])), np.float32).reshape(-1)[:2]
    except Exception:
        d = np.array([0.0, -1.0], np.float32)
    n = float(np.hypot(d[0], d[1])) or 1.0
    dx, dy = float(d[0]) / n, float(d[1]) / n
    ang = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
    # the visual: a sphere lit from the estimated direction, so the answer
    # is judged by eye as well as by number
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, r = w / 2.0, h / 2.0, min(h, w) * 0.4
    u2 = (xx - cx) / r
    v2 = (yy - cy) / r
    m2 = u2 * u2 + v2 * v2
    nz = np.sqrt(np.clip(1 - m2, 0, 1))
    lam = np.clip(u2 * dx + v2 * dy + nz * 0.5, 0, 1) * (m2 <= 1)
    return {"out": _rgb(lam.astype(np.float32)),
            "x": float((dx + 1) / 2), "y": float((dy + 1) / 2),
            "angle": float(ang)}


@op("Paint relief 3D", "FX", inputs=[],
    params=[P("layer", "layerref", "",
              hint="Which layer's paint BODY to view — impasto strokes give "
                   "it height"),
            P("tilt", "float", 55.0, 5.0, 85.0,
              hint="Camera tilt from face-on — raking angles show the ridges"),
            P("turn", "float", 0.0, -180.0, 180.0),
            P("relief", "float", 6.0, 0.5, 30.0,
              hint="Height exaggeration"),
            P("detail", "int", 140, 40, 260,
              hint="Mesh grid width — more is finer and slower")],
    doc="Your impasto AS SCULPTURE: the layer's height field becomes a real "
        "mesh (leCore depth_to_mesh) painted with the layer's colours and "
        "rendered lit from a camera you tilt and turn. A raking tilt makes "
        "every ridge and drip cast honest shading -- the fastest way to "
        "judge how the paint is building up.",
    rgba=True, alpha="own")
def _relief3d(ctx, ins, p):
    h, w = ctx
    doc = _CTX_DOC[0] if _CTX_DOC else None
    z = np.zeros((h, w, 4), np.float32)
    if doc is None:
        return z
    try:
        l = doc.layer(str(p.get("layer", "")))
    except Exception:
        return z
    hm = getattr(l, "height_map", None)
    if hm is None or not (hm > 0.02).any():
        return z
    m = mind()
    gw = int(p["detail"])
    gh = max(2, int(round(hm.shape[0] * gw / max(hm.shape[1], 1))))
    hh = _resize(hm[..., None], gh, gw)[..., 0]
    col = _resize(l.pixels, gh, gw)
    r = m.depth_to_mesh((hh * float(p["relief"]) / 10.0 + 1.0)
                        .astype(np.float32), discontinuity=10.0)
    obj = r[0] if isinstance(r, (tuple, list)) else r
    V = np.asarray(obj.vertices, np.float32).copy()
    F = np.asarray(obj.faces, np.int64)
    if not len(V):
        return z
    # vertex colours from the layer's pigment; transparent areas read as canvas
    xi = np.clip((V[:, 0] * (gw - 1)).astype(int) if V[:, 0].max() <= 1.5
                 else V[:, 0].astype(int), 0, gw - 1)
    yi = np.clip((V[:, 1] * (gh - 1)).astype(int) if V[:, 1].max() <= 1.5
                 else V[:, 1].astype(int), 0, gh - 1)
    a2 = col[yi, xi, 3:4]
    vc = col[yi, xi, :3] * a2 + 0.92 * (1 - a2)
    mesh = {"vertices": V.tolist(), "faces": F.tolist()}
    cam = dict(m.fit_camera(mesh, direction=(
        float(np.sin(np.deg2rad(p["turn"])) * np.sin(np.deg2rad(p["tilt"]))),
        float(np.cos(np.deg2rad(p["tilt"]))),
        float(np.cos(np.deg2rad(p["turn"])) * np.sin(np.deg2rad(p["tilt"]))))))
    rw = min(w, 512)
    rh = max(1, int(round(h * rw / max(w, 1))))
    img = np.asarray(m.render_mesh(mesh, cam, width=rw, height=rh,
                                   background=(0, 0, 0), ambient=0.35,
                                   smooth=True, vertex_colors=vc.tolist(),
                                   dtype=np.float32))
    if (rw, rh) != (w, h):
        img = _resize(img, h, w)
    alpha = (img.max(axis=-1) > 0.01).astype(np.float32)
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = img * alpha[..., None]
    out[..., 3] = alpha
    return out


@op("Value", "Values", inputs=[],
    params=[P("value", "float", 0.5, 0.0, 1.0),
            P("scale", "float", 1.0, -10.0, 10.0)],
    doc="A number on a wire (Blender: Value node). Drag its slider and every parameter you've wired it into moves together -- drive Blur size and Glow strength from one knob, or scrub it to animate a Smoke or Branching growth node.")
def _value(ctx, ins, p):
    return float(p["value"]) * float(p["scale"])


@op("Sample image", "Values", inputs=["image"],
    outputs=["out", "value", "r", "g", "b"],
    params=[P("u", "float", 0.5, 0.0, 1.0),
            P("v", "float", 0.5, 0.0, 1.0),
            P("mode", "choice", "bilinear", choices=["bilinear", "nearest"]),
            P("wrap", "choice", "clamp", choices=["clamp", "repeat"])],
    doc="An eyedropper on a wire: reads the picture at (u, v) -- 0,0 is the "
        "top-left, 1,1 the bottom-right -- and puts what it finds on NUMBER "
        "sockets. `value` is the brightness there and r/g/b the channels; wire "
        "any of them into a parameter pin to drive Blur size, Glow strength, "
        "or a Value chain from a painted map (leCore sample_image, the "
        "texture-as-numerical-input primitive). Wire Value nodes into u and v "
        "to move the probe from the graph. `out` is a swatch of the sampled "
        "colour, so it also feeds image inputs directly.")
def _sampleimage(ctx, ins, p):
    h, w = ctx
    if not have("sample_image"):
        raise ValueError("this leCore build has no sample_image faculty -- "
                         "update leos-core to use the Sample image node")
    img = _rgb(ins["image"]).astype(float)
    uv = np.array([[float(p["u"]), float(p["v"])]], float)
    c = np.asarray(mind().sample_image(img, uv, mode=p.get("mode", "bilinear"),
                                       wrap=p.get("wrap", "clamp")))[0]
    # number sockets report what is actually there (the sampler's round-trip
    # contract is exact, so don't clip the numbers); only the display swatch
    # is clamped to the visible range
    r, g, b = (float(x) for x in c[:3])
    sw = np.empty((h, w, 3), np.float32)
    sw[...] = (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))
    return {"out": sw, "value": (r + g + b) / 3.0, "r": r, "g": g, "b": b}


@op("Values to texture", "Values", inputs=[],
    params=[P("v1", "float", 0.1, 0.0, 1.0), P("v2", "float", 0.4, 0.0, 1.0),
            P("v3", "float", 0.7, 0.0, 1.0), P("v4", "float", 1.0, 0.0, 1.0),
            P("count", "int", 4, 1, 4),
            P("smooth", "bool", True),
            P("vertical", "bool", False)],
    doc="Numbers become a texture: the values v1..v4 (use `count` for fewer) "
        "are laid out as a strip and stretched across the canvas -- smooth "
        "gives a gradient between them, off gives hard bands. Wire Value "
        "nodes into the v pins and the texture follows the numbers live: an "
        "adjustable gradient, a threshold ramp for Gradient map, a stepped "
        "grey wedge. Built on leCore values_to_texture + sample_image, whose "
        "round trip is exact -- with smooth off, a Sample image node probing "
        "a band centre reads back precisely the number you fed in.")
def _valuestotexture(ctx, ins, p):
    h, w = ctx
    if not have("values_to_texture", "sample_image"):
        raise ValueError("this leCore build has no values_to_texture / "
                         "sample_image faculties -- update leos-core to use "
                         "the Values to texture node")
    n = int(np.clip(p.get("count", 4), 1, 4))
    vals = np.array([float(p[f"v{i + 1}"]) for i in range(n)], float)
    tex = mind().values_to_texture(vals)          # (1, n) strip
    mode = "bilinear" if p.get("smooth", True) else "nearest"
    m = h if p.get("vertical") else w
    # stretch the strip with the SAME sampler (half-texel-centre convention),
    # so band centres land exactly on the fed-in values
    line = np.stack([(np.arange(m) + 0.5) / m, np.full(m, 0.5)], -1)
    row = np.asarray(mind().sample_image(tex, line, mode=mode, wrap="clamp"),
                     float).reshape(-1)
    if p.get("vertical"):
        out = np.repeat(row[:, None], w, axis=1)
    else:
        out = np.repeat(row[None, :], h, axis=0)
    return _rgb(np.clip(out, 0, 1).astype(np.float32))


@op("Color value", "Values", inputs=[],
    outputs=["out", "r", "g", "b"],
    params=[P("r", "float", 1.0, 0.0, 1.0), P("g", "float", 0.5, 0.0, 1.0),
            P("b", "float", 0.2, 0.0, 1.0)],
    doc="A colour on wires (Blender: RGB node): out is a solid image of the "
        "colour; r, g, b are its components as numbers -- wire them into "
        "colour-component parameters (Gradient map's shadow_r etc.) or use "
        "out anywhere an image goes.")
def _colorvalue(ctx, ins, p):
    h, w = ctx
    img = np.zeros((h, w, 3), np.float32)
    img[:] = [p["r"], p["g"], p["b"]]
    return {"out": img, "r": float(p["r"]), "g": float(p["g"]), "b": float(p["b"])}


@op("Channel split", "Comp", inputs=["image", "alpha"], rgba=True,
    outputs=["out", "r", "g", "b", "a"],
    doc="Break an image into channels (Nuke/Shake: Shuffle): r, g, b come out "
        "as grey images, a passes the optional alpha input through (wire a "
        "Layer node's alpha socket; opaque white when unwired). out is the "
        "untouched image. Recombine with Channel combine.")
def _chansplit(ctx, ins, p):
    rgba = ins["image"]
    img = rgba[..., :3]
    grey = lambda c: np.stack([c, c, c], -1)
    aw = ins.get("alpha")
    a = _lum(_rgb(aw)) if aw is not None else rgba[..., 3]    # the stream's own alpha
    return {"out": rgba, "r": grey(img[..., 0]), "g": grey(img[..., 1]),
            "b": grey(img[..., 2]), "a": grey(a)}


@op("Channel combine", "Comp", inputs=["r", "g", "b", "alpha"],
    doc="Rebuild an RGB image from three grey inputs (Nuke: Shuffle/Copy the "
        "other way): each input's luminance becomes one channel. Swap wires "
        "for channel-swap looks; feed processed mattes back into colour.")
def _chancombine(ctx, ins, p):
    rgb = np.stack([_lum(_rgb(ins[k])) for k in ("r", "g", "b")], -1)
    aw = ins.get("alpha")
    if aw is not None:
        return np.concatenate([rgb, _lum(_rgb(aw))[..., None]], -1)
    return rgb


@op("Merge", "Comp", inputs=["a", "b", "matte"], rgba=True,
    params=[P("operation", "choice", "over",
              choices=["over", "under", "plus", "screen", "multiply", "difference"]),
            P("mix", "float", 1.0, 0.0, 1.0)],
    doc="The compositor's Merge (Nuke): A over B through a matte. Wire a Layer "
        "node's alpha socket (or any keyer's matte) into matte -- over lays A "
        "onto B where the matte is white; under is the reverse; plus/screen/"
        "multiply/difference are the classic light and shadow ops. mix fades "
        "the whole operation. Unwired matte = fully opaque A. Blend is the simpler opacity-mix cousin for two opaque images.")
def _merge(ctx, ins, p):
    A, B = ins["a"], ins["b"]
    a, aa = A[..., :3], A[..., 3]
    b, ba = B[..., :3], B[..., 3]
    m = ins.get("matte")
    ma = _lum(_rgb(m)) if m is not None else aa       # matte wire overrides A's alpha
    M = ma[..., None]
    op_ = p["operation"]
    if op_ == "over":                                  # true alpha compositing
        out = a * M + b * (1 - M)
        oa = ma + ba * (1 - ma)
    elif op_ == "under":
        out = b * ba[..., None] + a * (1 - ba[..., None])
        oa = ba + ma * (1 - ba)
    elif op_ == "plus":
        out = b + a * M; oa = np.maximum(ma, ba)
    elif op_ == "screen":
        out = 1 - (1 - b) * (1 - a * M); oa = np.maximum(ma, ba)
    elif op_ == "multiply":
        out = b * (a * M + (1 - M)); oa = ba
    else:
        out = np.abs(b - a * M); oa = np.maximum(ma, ba)
    rgb = np.clip(b + (out - b) * p["mix"], 0, 1)
    al = np.clip(ba + (oa - ba) * p["mix"], 0, 1)
    return np.concatenate([rgb, al[..., None]], -1)


@op("Chroma key", "Comp", inputs=["image"],
    outputs=["out", "matte"],
    params=[P("screen", "choice", "green", choices=["green", "blue"]),
            P("tolerance", "float", 0.35, 0.05, 1.0),
            P("softness", "float", 0.1, 0.0, 0.5),
            P("despill", "float", 0.7, 0.0, 1.0)],
    doc="Green/blue-screen keyer (Nuke: ChromaKeyer / Keylight; Shake: "
        "Primatte): the matte output is white where the subject is, black on "
        "the screen; out is the image with screen spill pulled out of edges. "
        "Wire out + matte into Merge to comp over a new background.")
def _chromakey(ctx, ins, p):
    img = _rgb(ins["image"])
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    key = (g - np.maximum(r, b)) if p["screen"] == "green" else (b - np.maximum(r, g))
    lo = p["tolerance"] * 0.5 - p["softness"] * 0.5
    hi = p["tolerance"] * 0.5 + p["softness"] * 0.5 + 1e-6
    matte = 1.0 - np.clip((key - lo) / (hi - lo), 0, 1)
    out = img.copy()
    if p["despill"] > 1e-3:
        if p["screen"] == "green":
            lim = (r + b) / 2
            out[..., 1] = g - np.maximum(g - lim, 0) * p["despill"]
        else:
            lim = (r + g) / 2
            out[..., 2] = b - np.maximum(b - lim, 0) * p["despill"]
    mg = np.stack([matte, matte, matte], -1)
    out4 = np.concatenate([np.clip(out, 0, 1), matte[..., None]], -1)
    return {"out": out4, "matte": mg}


@op("Luma key", "Comp", inputs=["image"],
    outputs=["out", "matte"],
    params=[P("low", "float", 0.5, 0.0, 1.0), P("high", "float", 1.0, 0.0, 1.0),
            P("softness", "float", 0.1, 0.0, 0.5), P("invert", "bool", 0)],
    doc="Alpha from brightness (Nuke: Keyer luminance; Shake: LumaKey): pixels "
        "between low and high go white in the matte, with soft shoulders. "
        "Classic for pulling glows, skies, and self-illuminated elements.")
def _lumakey(ctx, ins, p):
    img = _rgb(ins["image"])
    l = _lum(img)
    sft = max(p["softness"], 1e-4)
    m = np.clip((l - (p["low"] - sft)) / sft, 0, 1) * \
        np.clip(((p["high"] + sft) - l) / sft, 0, 1)
    if int(p["invert"]):
        m = 1 - m
    return {"out": np.concatenate([img, m[..., None]], -1),
            "matte": np.stack([m, m, m], -1)}


@op("Grade", "Comp", inputs=["image"],
    params=[P("blackpoint", "float", 0.0, -0.5, 0.5),
            P("whitepoint", "float", 1.0, 0.5, 1.5),
            P("lift", "float", 0.0, -0.5, 0.5), P("gain", "float", 1.0, 0.0, 2.0),
            P("gamma", "float", 1.0, 0.2, 3.0),
            P("multiply", "float", 1.0, 0.0, 2.0),
            P("offset", "float", 0.0, -0.5, 0.5)],
    doc="THE compositor's colour node (Nuke: Grade), applying the classic "
        "chain: normalise blackpoint->whitepoint, remap to lift->gain, gamma, "
        "then multiply and offset. Every VFX shot you've seen went through "
        "hundreds of these.")
def _grade(ctx, ins, p):
    img = _rgb(ins["image"])
    x = (img - p["blackpoint"]) / max(p["whitepoint"] - p["blackpoint"], 1e-6)
    x = x * (p["gain"] - p["lift"]) + p["lift"]
    x = np.power(np.clip(x, 0, None), 1.0 / max(p["gamma"], 1e-6))
    return np.clip(x * p["multiply"] + p["offset"], 0, 1)


@op("Transform", "Comp", inputs=["image"],
    params=[P("translate_x", "float", 0.0, -1000.0, 1000.0),
            P("translate_y", "float", 0.0, -1000.0, 1000.0),
            P("rotate", "float", 0.0, -180.0, 180.0),
            P("scale", "float", 1.0, 0.05, 4.0)],
    doc="Move, rotate, and scale IN the graph (Nuke: Transform) -- reposition "
        "an element non-destructively before a Merge, animate a bug into the "
        "corner, punch in on a video feed. This is the NODE (non-destructive); "
        "the toolbar's \u2934 Transform tool edits document layers "
        "destructively.", alpha="process")
def _transform_node(ctx, ins, p):
    img = _rgb(ins["image"])
    return np.clip(_affine(img, sx=p["scale"], sy=p["scale"], deg=p["rotate"],
                           dx=p["translate_x"], dy=p["translate_y"]), 0, 1)


@op("Dilate / Erode", "Comp", inputs=["image"],
    params=[P("size", "int", 4, -32, 32),
            P("shape", "choice", "round", choices=["round", "square"])],
    doc="Grow (positive) or choke (negative) the bright areas of a matte or glow (Nuke: Dilate/Erode). Use it to fatten a thin keyed edge before Merge, spread a glow mask, or shrink a selection matte that grabbed too much.", alpha="process")
def _dilate_erode(ctx, ins, p):
    import cv2
    img = _rgb(ins["image"])
    n = int(abs(p["size"]))
    if n == 0:
        return img
    shape = cv2.MORPH_ELLIPSE if p["shape"] == "round" else cv2.MORPH_RECT
    k = cv2.getStructuringElement(shape, (2 * n + 1, 2 * n + 1))
    fn = cv2.dilate if p["size"] > 0 else cv2.erode
    return np.clip(fn(img, k), 0, 1)


@op("Layer style", "Comp", inputs=["image"],
    params=[P("style", "choice", "shadow", choices=["shadow", "glow"]),
            P("dx", "float", 8.0, -60.0, 60.0, when={"style": ["shadow"]},
              hint="Shadow offset — light from the opposite side"),
            P("dy", "float", 8.0, -60.0, 60.0, when={"style": ["shadow"]}),
            P("blur", "float", 10.0, 0.0, 60.0),
            P("opacity", "float", 0.6, 0.0, 1.0),
            P("r", "float", 0.0, 0.0, 1.0), P("g", "float", 0.0, 0.0, 1.0),
            P("b", "float", 0.0, 0.0, 1.0)],
    doc="The STYLE pixels alone -- a drop shadow (offset + blurred + tinted "
        "silhouette) or an outer glow (halo beyond the edges), built from the "
        "input's alpha and nothing else. Commit it to a layer BELOW the "
        "source (the one-click buttons in the Layers panel do exactly that) "
        "and the style stays live: repaint the source and the shadow follows "
        "on the next graph push.", rgba=True, alpha="own")
def _layer_style(ctx, ins, p):
    h, w = ctx
    src = np.asarray(ins["image"], np.float32)
    a = src[..., 3] if src.ndim == 3 and src.shape[-1] == 4         else _rgb(src).max(axis=-1)
    col = np.array([p["r"], p["g"], p["b"]], np.float32)
    if p["style"] == "shadow":
        dx, dy = int(round(p["dx"])), int(round(p["dy"]))
        off = np.zeros_like(a)
        sy0, sy1 = max(0, dy), min(h, h + dy)
        sx0, sx1 = max(0, dx), min(w, w + dx)
        off[sy0:sy1, sx0:sx1] = a[max(0, -dy):h - max(0, dy),
                                  max(0, -dx):w - max(0, dx)]
        soft = off
        if p["blur"] > 0:
            soft = _gauss_blur(off[..., None], p["blur"])[..., 0]
        sa = np.clip(soft * float(p["opacity"]), 0.0, 1.0)
    else:
        halo = _gauss_blur(a[..., None], max(float(p["blur"]), 1.0))[..., 0]
        # OUTER glow: the halo beyond the silhouette, nothing inside it
        sa = np.clip((halo - a) * 2.2 * float(p["opacity"]), 0.0, 1.0)             * (1.0 - a)
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = col[None, None] * sa[..., None]
    out[..., 3] = sa
    return out


@op("Glow", "Comp", inputs=["image"],
    params=[P("threshold", "float", 0.6, 0.0, 1.0),
            P("radius", "float", 12.0, 1.0, 60.0),
            P("intensity", "float", 1.0, 0.0, 3.0)],
    doc="Bright areas bloom outward (Nuke: Glow; Shake: Glow): everything "
        "above the threshold is blurred and screened back over the image. "
        "Neon, magic, lightsabers, hot practicals.", alpha="process")
def _glow(ctx, ins, p):
    img = _rgb(ins["image"])
    bright = np.maximum(img - p["threshold"], 0) / max(1 - p["threshold"], 1e-6)
    halo = np.clip(_gauss_blur(bright, p["radius"]) * p["intensity"], 0, 1)
    return 1 - (1 - img) * (1 - halo)


@op("Switch", "Comp", inputs=["a", "b"],
    params=[P("which", "bool", 0)],
    doc="Pass through input a or input b (Nuke: Switch) -- flip between two "
        "whole branches for before/after checks or alternate looks without "
        "rewiring anything.")
def _switch(ctx, ins, p):
    return _rgb(ins["b"] if int(p["which"]) else ins["a"])


@op("Blend", "Combine", inputs=["a", "b"],
    params=[P("mode", "choice", "normal", choices=list(BLEND_MODES)),
            P("mix", "float", 0.5, 0.0, 1.0)],
    doc="Blend two inputs by mode, mixed by `mix`. For alpha-aware compositing (over a transparent element), use Merge instead.")
def _blendop(ctx, ins, p):
    a, b = _rgb(ins["a"]), _rgb(ins["b"])
    return np.clip(a + (BLEND_MODES[p["mode"]](a, b) - a) * p["mix"], 0, 1)


@op("Morph", "Combine", inputs=["a", "b"],
    params=[P("t", "float", 0.5, 0.0, 1.0),
            P("method", "choice", "blend", choices=["blend", "phase", "dct"])],
    doc="Cross-dissolve WITH shape warping between two images (think face-morph "
        "GIFs): t=0 is input a, t=1 is input b, and midpoints warp features toward "
        "each other instead of just fading. phase warps by spectral phase; dct "
        "blends in leCore's DCT-coefficient domain (morph_scene) -- structure "
        "dissolves before texture, a distinctly different in-between. Wire two "
        "images and slide t.")
def _morph(ctx, ins, p):
    a, b = _rgb(ins["a"]), _rgb(ins["b"])
    if p["method"] == "phase":
        out = np.empty_like(a)
        for c in range(3):
            out[..., c] = np.real(np.asarray(mind().phase_morph(a[..., c].astype(float),
                                                        b[..., c].astype(float), float(p["t"]))))
        return np.clip(out, 0, 1)
    if p["method"] == "dct":
        h, w = a.shape[:2]
        n = min(256, max(h, w))                # morph_scene needs SQUARE (backlog D)
        sa, sb = _resize(a, n, n), _resize(b, n, n)
        frames = mind().morph_scene(sa.astype(float), sb.astype(float), steps=11)
        f = np.asarray(frames[int(round(float(p["t"]) * (len(frames) - 1)))],
                       np.float32)
        return np.clip(_resize(f, h, w), 0, 1)
    frames = mind().blend_images(a, b, steps=21)
    return _rgb(frames[int(round(float(p["t"]) * (len(frames) - 1)))])


@op("Mask mix", "Combine", inputs=["a", "b", "mask"],
    doc="Combine two images through a mask: shows a where the mask is dark and b where it is bright, with smooth transitions in between -- the classic 'sky replacement' wiring (bottom shot, new sky, and a gradient or keyed matte deciding which shows where).")
def _maskmix(ctx, ins, p):
    m = _rgb(ins["mask"]).mean(-1, keepdims=True)
    return _rgb(ins["a"]) * (1 - m) + _rgb(ins["b"]) * m


@op("Light shafts", "FX", inputs=["image"],
    params=[P("x", "float", 0.5, 0.0, 1.0), P("y", "float", 0.2, 0.0, 1.0),
            P("threshold", "float", 0.7, 0.0, 1.0), P("weight", "float", 0.5, 0.0, 2.0),
            P("warmth", "float", 0.0, 0.0, 1.0), P("length", "float", 0.6, 0.1, 1.0)],
    doc="Volumetric god-rays streaming FROM a bright point (the sun, a gap in "
        "the trees) ACROSS your image. It finds the bright areas, smears them "
        "radially outward from (x, y), and screens the light back OVER your "
        "untouched picture -- so colours stay put and the frame only gains "
        "light. Put the sun's position at x,y; threshold picks how bright a "
        "spot has to be to cast rays; warmth adds optional golden tint (0 = "
        "colour-neutral). The classic forest-clearing shaft look.")
def _shafts(ctx, ins, p):
    img = _rgb(ins["image"])
    H, W = img.shape[:2]
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    thr = float(p["threshold"])
    bright = (np.clip((lum - thr) / (1.0 - thr + 1e-6), 0, 1)[..., None] * img
              ).astype(np.float32)
    if bright.max() < 1e-4:
        return img                                   # nothing bright: no rays
    lx, ly = float(p["x"]) * W, float(p["y"]) * H
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    from scipy.ndimage import map_coordinates
    samples = 56
    reach = 0.85 * float(p["length"])                # how far the rays travel
    decay = 0.96
    acc = np.zeros_like(img)
    norm = 0.0
    for i in range(samples):
        f = 1.0 - (i / samples) * reach              # march toward the light
        sx = lx + (xs - lx) * f
        sy = ly + (ys - ly) * f
        wgt = decay ** i
        for ch in range(3):
            acc[..., ch] += map_coordinates(bright[..., ch], [sy, sx],
                                            order=1, mode="constant") * wgt
        norm += wgt
    acc /= max(norm, 1e-6)
    acc *= float(p["weight"])                      # weight scales the rays
    if p.get("warmth", 0.0) > 0:                      # optional golden tint
        warm = np.array([1.0, 0.93, 0.74], np.float32)
        k = float(p["warmth"])
        acc = acc * ((1 - k) + k * warm)
    return 1.0 - (1.0 - img) * (1.0 - np.clip(acc, 0, 1))   # screen over input


@op("Grain", "FX", inputs=["image"],
    params=[P("amount", "float", 0.08, 0.0, 0.5), P("size", "float", 1.0, 0.5, 6.0),
            P("colour", "bool", False), P("seed", "int", 0, 0, 999)],
    doc="Adds film grain -- the fine noise that stops flat digital areas "
        "looking sterile and glues a composite together. amount sets "
        "strength, size makes the grains coarser, colour switches between "
        "monochrome grain (classic film) and RGB speckle. A touch (0.04-0.10) "
        "at the very end of a grade is the norm.", alpha="process")
def _grain(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(p["seed"]))
    sz = max(float(p["size"]), 0.5)
    gh, gw = max(1, int(h / sz)), max(1, int(w / sz))
    if p.get("colour"):
        g = rng.standard_normal((gh, gw, 3)).astype(np.float32)
    else:
        g = rng.standard_normal((gh, gw, 1)).astype(np.float32).repeat(3, -1)
    if (gh, gw) != (h, w):
        g = _resize(g, h, w)
    # grain modulates more in midtones, less in deep shadow/highlight (film-like)
    lum = _lum(img)[..., None]
    mask = (4.0 * lum * (1.0 - lum)).clip(0, 1)
    return np.clip(img + g * float(p["amount"]) * mask, 0, 1)


@op("Chromatic aberration", "FX", inputs=["image"],
    params=[P("shift", "float", 3.0, 0.0, 20.0), P("radial", "bool", True)],
    doc="Splits the red and blue channels apart the way a real lens fringes "
        "colour toward the edges of the frame -- a subtle shift sells "
        "'photographed through glass' and takes the CG edge off a render. "
        "radial pushes the fringing outward from the centre (true lens look); "
        "off does a uniform sideways split.", alpha="process")
def _chroma(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    shift = float(p["shift"])
    if shift < 1e-3:
        return img
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if p.get("radial", True):
        dx = (xs / w - 0.5); dy = (ys / h - 0.5)
        r = np.hypot(dx, dy) + 1e-6
        ox, oy = dx / r * shift, dy / r * shift
    else:
        ox = np.full((h, w), shift, np.float32); oy = np.zeros((h, w), np.float32)
    from scipy.ndimage import map_coordinates
    out = img.copy()
    for ch, s in ((0, 1.0), (2, -1.0)):              # R out, B in
        out[..., ch] = map_coordinates(img[..., ch],
                                       [ys + oy * s, xs + ox * s],
                                       order=1, mode="nearest")
    return np.clip(out, 0, 1)


@op("Vignette", "FX", inputs=["image"],
    params=[P("amount", "float", 0.5, 0.0, 1.0), P("softness", "float", 0.6, 0.1, 1.5)],
    doc="Darkens gently toward the corners, drawing the eye to the middle -- the finishing touch on portraits and product shots. Also available inside Post FX as one stage of the grading chain.")
def _vignette(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.hypot((xs / w - 0.5) * 2, (ys / h - 0.5) * 2)
    v = 1 - np.clip((d - (1 - p["softness"])) / p["softness"], 0, 1) * p["amount"]
    return img * v[..., None]



_PROCTEX_NAMES = ["marble", "wood", "brick", "voronoi", "musgrave", "wave",
                  "magic", "checker", "stripes", "dots", "noise", "fbm",
                  "white", "gradient"]
# which optional dials each texture actually accepts (sending an unknown one
# raises), measured against the engine rather than guessed
_PROCTEX_ARGS = {
    "voronoi":  ("scale", "seed", "kind"),
    "musgrave": ("scale", "seed", "octaves"),
    "fbm":      ("scale", "seed", "octaves"),
    "wave":     ("scale", "seed", "distortion"),
    "marble":   ("scale", "seed", "distortion"),
    "wood":     ("scale", "seed", "distortion"),
    "brick":    ("scale", "seed"),
    "noise":    ("scale", "seed"),
    "magic":    ("scale", "seed"),
    "stripes":  ("scale",),
    "checker":  ("scale",),
    "dots":     ("scale",),
    "gradient": ("scale",),
    "white":    ("scale", "seed"),
}


@op("Reroute", "Utility", inputs=["image"], params=[],
    doc="A tidy corner for a wire. It passes the image through untouched -- "
        "drop one onto a link (or drag it onto a wire) to bend a long "
        "connection around a busy part of the graph instead of letting it cut "
        "across everything. Costs nothing to evaluate.")
def _reroute(ctx, ins, p):
    img = ins.get("image")
    if img is None:
        h, w = ctx
        return np.zeros((h, w, 3), np.float32)
    return img


@op("Procedural texture", "Generate", requires=["texture_image"],
    params=[P("name", "choice", "marble", choices=_PROCTEX_NAMES),
            P("scale", "float", 4.0, 0.5, 40.0,
              hint="How big the features are -- higher = smaller, busier"),
            P("octaves", "int", 4, 1, 8,
              when={"name": ["fbm", "musgrave"]},
              hint="Layers of detail stacked on top of each other"),
            P("distortion", "float", -1.0, -1.0, 4.0,
              when={"name": ["wave", "marble", "wood"]},
              hint="-1 keeps the preset's own character; raise to warp harder"),
            P("kind", "choice", "f2f1",
              choices=["f1", "f2", "f2f1", "cell", "smooth"],
              when={"name": ["voronoi"]},
              hint="Voronoi flavour: cell edges, centres, or flat cells"),
            P("seed", "int", 0, 0, 999)],
    doc="The standard texture menu every 3D app has, as a flat image: marble, "
        "wood, brick, voronoi cells, musgrave ridges, waves, magic, checker, "
        "stripes, dots and plain noise. Pick a `name`, set `scale` for how big "
        "the features are, and use it as a base coat, a bump pattern, or a "
        "mask (wire it into Merge's matte or Mask mix). Leave `distortion` at "
        "-1 to keep each texture's own character (marble's heavy veining, "
        "wood's gentle grain); raise it to warp them yourself. `octaves` only "
        "bites on fbm/musgrave, `kind` on voronoi.")
def _proctex(ctx, ins, p):
    h, w = ctx
    if not have("texture_image"):
        raise ValueError("this leCore build has no texture_image faculty -- "
                         "update leos-core to use the Procedural texture node")
    name = p.get("name", "marble")
    kw = {"scale": float(p["scale"]), "seed": int(p["seed"]),
          "octaves": int(p["octaves"])}
    d = float(p["distortion"])
    if d >= 0:                             # -1 = auto: keep the preset's own
        kw["distortion"] = d               # character (marble veins, wood grain)
    if name == "voronoi":
        kw["kind"] = p.get("kind", "f2f1")
    # Each texture takes a different subset (magic has no seed, checker no
    # octaves...). Rather than hard-code a table that rots when leCore changes,
    # drop whatever the engine rejects and retry -- self-correcting on any build.
    for _ in range(len(kw) + 1):
        try:
            probe = mind().texture_image(name, size=8, **kw)
            del probe
            break
        except TypeError as e:
            bad = re.findall(r"unexpected keyword argument '([^']+)'", str(e))
            if not bad or bad[0] not in kw:
                kw = {"scale": float(p["scale"])}      # last-ditch minimal call
                break
            kw.pop(bad[0])
    side = int(max(32, min(max(h, w), 1024)))
    # Sample a region CENTRED on the origin. The solid textures are 3-D and the
    # ring-shaped ones (wood) measure radius from the axis line, so on the
    # default 0..1 corner region sqrt(x^2+z^2) collapses to x and wood comes out
    # byte-identical to marble. Straddling the origin makes rings actually ring.
    region = ((-1.0, -1.0), (1.0, 1.0))
    img = np.asarray(mind().texture_image(name, size=side, region=region, **kw),
                     np.float32)
    return _resize(_rgb(np.clip(img, 0, 1)), h, w)


def _apply_ramp(v, stops, cols, interp):
    if have("ramp"):
        try:
            f = mind().ramp(stops, cols, interp=interp)
            out = np.asarray(f(v.reshape(-1)), np.float32).reshape(v.shape + (3,))
            return np.clip(out, 0, 1)
        except Exception:
            pass
    out = np.zeros(v.shape + (3,), np.float32)
    for i in range(3):
        lo, hi = stops[i], stops[i + 1]
        a = np.array(cols[i], np.float32)
        b = np.array(cols[i + 1], np.float32)
        seg = (v >= lo) & (v <= hi + (1e-6 if i == 2 else 0))
        t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
        if interp == "constant":
            t = np.zeros_like(t)
        out[seg] = (a[None] * (1 - t) + b[None] * t)[seg]
    out[v < stops[0]] = cols[0]
    out[v > stops[-1]] = cols[-1]
    return np.clip(out, 0, 1)


_RAMP_LOOKS = {
    "dusk":     (0.32, 0.68, [(0.05, 0.06, 0.22), (0.45, 0.16, 0.40),
                              (0.98, 0.55, 0.25), (1.00, 0.94, 0.80)]),
    "ember":    (0.30, 0.70, [(0.02, 0.01, 0.03), (0.45, 0.08, 0.05),
                              (0.95, 0.45, 0.10), (1.00, 0.90, 0.55)]),
    "ocean":    (0.35, 0.72, [(0.02, 0.07, 0.15), (0.05, 0.30, 0.45),
                              (0.20, 0.65, 0.70), (0.85, 0.98, 0.95)]),
    "forest":   (0.33, 0.70, [(0.04, 0.08, 0.04), (0.13, 0.30, 0.12),
                              (0.45, 0.62, 0.25), (0.93, 0.95, 0.75)]),
    "toon pop": (0.34, 0.66, [(0.15, 0.10, 0.35), (0.90, 0.20, 0.45),
                              (1.00, 0.75, 0.10), (1.00, 1.00, 0.95)]),
    "mono ink": (0.35, 0.70, [(0.05, 0.05, 0.06), (0.30, 0.30, 0.33),
                              (0.65, 0.65, 0.68), (0.97, 0.97, 0.98)]),
}


@op("Color ramp", "Color", inputs=["image"],
    params=[P("look", "choice", "custom",
              choices=["custom", "dusk", "ember", "ocean", "forest",
                       "toon pop", "mono ink"],
              hint="Ready-made palettes; pick custom to place the four stops yourself"),
            P("shadows_r", "float", 0.03, 0.0, 1.0, when={"look": ["custom"]}),
            P("shadows_g", "float", 0.05, 0.0, 1.0, when={"look": ["custom"]}),
            P("shadows_b", "float", 0.18, 0.0, 1.0, when={"look": ["custom"]}),
            P("low_mid_r", "float", 0.55, 0.0, 1.0, when={"look": ["custom"]}),
            P("low_mid_g", "float", 0.22, 0.0, 1.0, when={"look": ["custom"]}),
            P("low_mid_b", "float", 0.35, 0.0, 1.0, when={"look": ["custom"]}),
            P("high_mid_r", "float", 0.95, 0.0, 1.0, when={"look": ["custom"]}),
            P("high_mid_g", "float", 0.62, 0.0, 1.0, when={"look": ["custom"]}),
            P("high_mid_b", "float", 0.30, 0.0, 1.0, when={"look": ["custom"]}),
            P("highlights_r", "float", 1.0, 0.0, 1.0, when={"look": ["custom"]}),
            P("highlights_g", "float", 0.96, 0.0, 1.0, when={"look": ["custom"]}),
            P("highlights_b", "float", 0.85, 0.0, 1.0, when={"look": ["custom"]}),
            P("low_pos", "float", 0.35, 0.0, 1.0,
              hint="Where the low_mid colour sits along the ramp (0=black end)", when={"look": ["custom"]}),
            P("high_pos", "float", 0.7, 0.0, 1.0,
              hint="Where the high_mid colour sits along the ramp (1=white end)", when={"look": ["custom"]}),
            P("smooth", "bool", True,
              hint="Off = hard colour steps (toon / poster banding)")],
    doc="A four-stop colour ramp (the ColorRamp / gradient-map node): the "
        "input's brightness picks a colour along the ramp, so shadows take "
        "the first colour and highlights the last with two stops you place in "
        "between. Far richer than a two-colour Gradient map -- this is how you "
        "get sunsets, thermal looks, and toon banding (turn `smooth` off for "
        "hard steps). Slide p1/p2 to squeeze where the middle colours land.")
def _colorramp(ctx, ins, p):
    v = _lum(_rgb(ins["image"]))
    interp = "linear" if p.get("smooth", True) else "constant"
    look = p.get("look", "custom")
    if look in _RAMP_LOOKS:
        lp, hp, lc = _RAMP_LOOKS[look]
        return _apply_ramp(v, [0.0, lp, hp, 1.0], [list(x) for x in lc], interp)
    stops = [0.0, float(p["low_pos"]), float(p["high_pos"]), 1.0]
    cols = [[p["shadows_r"], p["shadows_g"], p["shadows_b"]],
            [p["low_mid_r"], p["low_mid_g"], p["low_mid_b"]],
            [p["high_mid_r"], p["high_mid_g"], p["high_mid_b"]],
            [p["highlights_r"], p["highlights_g"], p["highlights_b"]]]
    order = np.argsort(stops)
    stops = [stops[k] for k in order]
    cols = [cols[k] for k in order]
    return _apply_ramp(v, stops, cols, interp)


@op("Refract", "FX", inputs=["image", "mask"], requires=["mask_refraction"],
    params=[P("strength", "float", 12.0, 0.0, 60.0,
              hint="How hard the lens bends -- pixels of displacement at the edge"),
            P("ior", "float", 1.33, 1.0, 2.5,
              hint="The material: 1.33 water, 1.5 glass, higher = denser"),
            P("chromatic", "float", 0.0, 0.0, 1.0,
              hint="Rainbow fringing at the edges, like real glass"),
            P("ripple", "float", 0.0, 0.0, 1.0),
            P("seed", "int", 0, 0, 999)],
    doc="Bends the picture through a shape as if it were glass or water: wire "
        "a mask and the bright region becomes a lens sitting on your image. "
        "The distortion is strongest right at the mask's edge and fades to "
        "nothing in the middle, which is exactly how a real water droplet or "
        "a glass blob reads. `ior` is the material (1.33 water, 1.5 glass), "
        "`chromatic` adds rainbow fringing, `ripple` adds a water shimmer.",
    alpha="process")
def _refract(ctx, ins, p):
    img = _rgb(ins["image"])
    if ins.get("mask") is None or np.asarray(ins["mask"]).size <= 1:
        return img                                      # no lens: passthrough
    if not have("mask_refraction"):
        raise ValueError("this leCore build has no mask_refraction faculty -- "
                         "update leos-core to use the Refract node")
    mask = _lum(_rgb(ins["mask"]))
    kw = {}
    if float(p["ripple"]) > 0:
        kw["ripple"] = (float(p["ripple"]) * 6.0, 4.0)
    out = mind().mask_refraction(img.astype(float), mask.astype(float),
                                 strength=float(p["strength"]),
                                 ior=float(p["ior"]),
                                 chromatic=float(p["chromatic"]),
                                 seed=int(p["seed"]), **kw)
    return np.clip(np.asarray(out, np.float32), 0, 1)


_SKY_MEMO = {}


def _sky_fn(hour, kind, cover, sun, seed):
    """sky_model construction is a pure function of its arguments (3 ms), but
    the SAMPLING is the cost. Memo the model so dragging `hour` does not
    rebuild it; bounded like every other memo in this file."""
    key = (round(float(hour), 3), str(kind), round(float(cover), 3),
           round(float(sun), 2), int(seed))
    hit = _SKY_MEMO.get(key)
    if hit is None:
        clouds = ((str(kind), float(cover)),) if cover > 0.01 else ()
        hit = mind().sky_model(hour=float(hour), clouds=clouds,
                               sun_intensity=float(sun), stars_seed=int(seed))
        if len(_SKY_MEMO) > 8:
            _SKY_MEMO.clear()
        _SKY_MEMO[key] = hit
    return hit


@op("Sky", "Generate",
    params=[P("hour", "float", 12.0, 0.0, 24.0,
              hint="Time of day. The sun arcs, the colour follows it: warm at "
                   "dawn and dusk, blue at noon, stars after dark."),
            P("high_cloud", "choice", "cirrus",
              choices=["none", "cirrus", "altostratus", "nimbostratus"],
              hint="HIGH cloud layers. For low puffy clouds use the Clouds node."),
            P("cover", "float", 0.35, 0.0, 1.0, when={"high_cloud": ["cirrus", "altostratus", "nimbostratus"]}),
            P("exposure", "float", 1.0, 0.1, 4.0,
              hint="Brightness of the result. (leCore's own sun_intensity has "
                   "no effect on the sampled sky -- measured identical output "
                   "from 0 to 40 -- so this scales the render instead of "
                   "shipping a dial that does nothing.)"),
            P("fov", "float", 1.2, 0.3, 3.0, hint="How much sky the frame sees"),
            P("tilt", "float", 0.7, 0.0, 2.0,
              hint="Camera pitch: lower looks toward the horizon, higher at the zenith"),
            P("seed", "int", 0, 0, 999, hint="Star placement")],
    doc="A physically-shaped sky: sun arc, time of day, high cloud layers and "
        "deterministic stars. Drag `hour` from 6 to 22 and watch dawn become "
        "noon, dusk, then a starfield. Pairs with the Clouds node, which does "
        "the low puffy volumetric layer this one deliberately leaves out.",
    requires=("sky_model",))
def _sky(ctx, ins, p):
    h, w = ctx
    kind = p.get("high_cloud", "cirrus")
    cover = 0.0 if kind == "none" else float(p["cover"])
    sky = _sky_fn(p["hour"], kind if kind != "none" else "cirrus", cover,
                  14.0, int(p["seed"]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(w, 1)) * 2.0 - 1.0
    v = 1.0 - (yy / max(h, 1)) * 2.0
    d = np.stack([u * float(p["fov"]), v * float(p["tilt"]),
                  np.ones_like(u)], -1)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    img = np.asarray(sky(d.reshape(-1, 3)), np.float32).reshape(h, w, 3)
    return np.clip(img * float(p["exposure"]), 0.0, 1.0)


@op("Clouds", "Generate", requires=["cloud_scene"], inputs=["shape"],
    params=[P("preset", "choice", "cumulus",
              choices=["cumulus", "wispy", "storm", "sunset"]),
            P("quality", "choice", "fast", choices=["fast", "balanced"],
              hint="fast ~6s, balanced much longer -- progress bar + Esc cancels"),
            P("erode", "float", 0.3, 0.0, 1.0,
              hint="Eats away the cloud edges -- higher = wispier, more broken"),
            P("seed", "int", 0, 0, 999)],
    doc="A real volumetric cloud, raymarched with a lit sky behind it -- "
        "fluffy cumulus, thin wispy, dark storm, or a warm sunset bank. This "
        "is a proper 3-D cloud rather than a noise trick, so it costs real "
        "time: 'fast' takes several seconds and 'balanced' considerably "
        "longer (the render shows a progress bar and can be cancelled). Great "
        "as a sky plate to composite behind a subject. Wire a picture into "
        "`shape` and the cloud takes THAT form instead: paint a blob, get "
        "that blob back as a lit volumetric cloud -- and the shaped path "
        "renders in under a second. `erode` eats away the faint edges.")
def _clouds(ctx, ins, p):
    h, w = ctx
    shape = ins.get("shape")
    if shape is not None and float(np.asarray(shape).max()) > 1e-4:
        # PAINTED cloud: the wired picture's brightness IS the density, wrapped
        # as a field (leCore image_field) and windowed the way cloud_scene
        # treats its texture path -- soft radial falloff, thin along the view
        # axis, eroded -- so it reads as a blob in the sky, not an extruded
        # slab. Direct field evaluation skips the grid bake, which is why the
        # painted path renders FASTER than the presets (~0.5s vs ~7s).
        if not have("image_field", "make_cloud", "camera"):
            raise ValueError("this leCore build lacks image_field/make_cloud "
                             "-- update leos-core to shape clouds from images")
        m = mind()
        lum = _lum(_rgb(shape)).astype(float)
        fld = m.image_field(lum, scale=1.0, wrap="clamp")
        erode = float(p["erode"]) * 0.4
        def density(P):
            P = np.atleast_2d(np.asarray(P, float))
            uv = np.stack([(P[:, 0] + 1.0) * 0.5,
                           1.0 - (P[:, 1] + 1.0) * 0.5], -1)
            tex = np.asarray(fld(np.concatenate(
                [uv, np.zeros((len(uv), 1))], 1)))
            r = np.linalg.norm(P, axis=1)
            fall = np.clip(1.2 - r, 0, 1)
            zfall = np.exp(-(P[:, 2] / 0.4) ** 2)
            return np.maximum(tex * fall * zfall * 3.0 - erode, 0.0)
        # make_cloud takes a real Camera (no dict coercion on this faculty --
        # noted in the backlog's SD); front-on so the painting maps straight
        # onto the sky
        cam = m.camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0),
                       up=(0, 1, 0), fov_deg=45.0)
        steps = 64 if p.get("quality", "fast") == "fast" else 100
        ph, pw = int(min(h, 288)), int(min(w, 384))
        img = m.make_cloud(field=density, camera=cam, width=pw, height=ph,
                           steps=steps, seed=int(p["seed"]))
        return _resize(_rgb(np.clip(np.asarray(img, np.float32), 0, 1)), h, w)
    if not have("cloud_scene"):
        raise ValueError("this leCore build has no cloud_scene faculty -- "
                         "update leos-core to use the Clouds node")
    img = mind().cloud_scene(preset=p.get("preset", "cumulus"),
                             quality=p.get("quality", "fast"),
                             erode=float(p["erode"]), seed=int(p["seed"]))
    return _resize(_rgb(np.clip(np.asarray(img, np.float32), 0, 1)), h, w)


@op("Water", "Generate", requires=["render_water"],
    params=[P("preset", "choice", "ocean",
              choices=["calm", "ocean", "storm"]),
            P("level", "float", 0.72, 0.0, 1.0,
              hint="Where the horizon sits in frame (0 = top)"),
            P("time", "float", 0.0, 0.0, 60.0,
              hint="Wave phase -- wire a Value node in here to animate"),
            P("ripple", "float", 0.35, 0.0, 1.0),
            P("seed", "int", 0, 0, 999)],
    doc="A rendered water surface -- rolling ocean swell, calm "
        "glassy water, or a storm chop -- with real wave shapes rather than a noise "
        "bump. `level` sets the horizon height in frame, `time` animates the "
        "waves (wire a Value node in to make it move), and `ripple` adds fine "
        "surface detail. Use it as a backplate or composite something onto it.")
def _water(ctx, ins, p):
    h, w = ctx
    if not have("render_water"):
        raise ValueError("this leCore build has no render_water faculty -- "
                         "update leos-core to use the Water node")
    ph, pw = int(min(h, 320)), int(min(w, 426))
    preset = {"pond": "calm", "pool": "calm"}.get(   # legacy .lews names
        p.get("preset", "ocean"), p.get("preset", "ocean"))
    img = mind().render_water(preset=preset,
                              level=float(p["level"]), t=float(p["time"]),
                              ripple=float(p["ripple"]), seed=int(p["seed"]),
                              quality="fast", width=pw, height=ph)
    return _resize(_rgb(np.clip(np.asarray(img, np.float32), 0, 1)), h, w)


def _stroke_path(doc, sid, n=160):
    """Sample a spline OR a recorded brush stroke into an (n,2) polyline.

    Both are just paths, so Stroke FX does not care which it was handed -- that
    is what lets an effect attach to a stroke painted freehand rather than only
    to one drawn with the pen tool."""
    pts = None
    for k in getattr(doc, "strokes", []):
        if k["id"] == sid:
            pts = np.asarray(k["points"], np.float32)
            break
    if pts is None:
        sp = None
        for p in getattr(doc, "splines", []):
            if p.id == sid:
                sp = p
                break
        if sp is None or len(sp.points) < 2:
            return None
        pts = np.array([[float(q["x"]), float(q["y"])] for q in sp.points], np.float32)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 1e-6:
        return None
    u = np.linspace(0.0, t[-1], int(n))
    return np.stack([np.interp(u, t, pts[:, 0]), np.interp(u, t, pts[:, 1])], -1)


@op("Stroke FX", "FX",
    params=[P("spline", "splineref", ""),
            P("mode", "choice", "particles", choices=["particles", "tubes"],
              hint="particles: spray, drips, embers. tubes: branches GROW from "
                   "the stroke -- Paint-Effects style 2.5-D, with a depth "
                   "output socket."),
            P("count", "int", 800, 20, 6000, when={"mode": ["particles"]},
              hint="How many particles the stroke emits"),
            P("life", "int", 26, 4, 120, when={"mode": ["particles"]},
              hint="Integration steps — longer trails, more cost"),
            P("gravity", "float", 40.0, -160.0, 240.0,
              hint="Positive drips downward; negative floats upward like embers"),
            P("wind", "float", 0.0, 0.0, 140.0,
              hint="Curl-noise turbulence — a swirling field, not a flat push"),
            P("spread", "float", 5.0, 0.0, 60.0, when={"mode": ["particles"]},
              hint="How far particles start from the stroke path"),
            P("heat", "float", 0.0, 0.0, 1.0, when={"mode": ["particles"]},
              hint="Dense cores glow toward white-hot: embers, sparks, "
                   "molten drips. 0 keeps a single flat colour"),
            P("particle", "choice", "dot",
              choices=["dot", "streak", "flake", "bubble", "spark"],
              when={"mode": ["particles"]},
              hint="What each particle IS: dot (splat), streak (motion-"
                   "smeared), flake (snow), bubble (ring), spark (star)"),
            P("stagger", "float", 0.0, 0.0, 1.0, when={"mode": ["particles"]},
              hint="Birth spread: particles start at different moments, so "
                   "trails have different lengths and heads don't line up "
                   "in a hem"),
            P("life_var", "float", 0.0, 0.0, 1.0,
              when={"mode": ["particles"]},
              hint="Lifetime spread: some die young, some outlive the rest"),
            P("size_var", "float", 0.3, 0.0, 1.0,
              when={"mode": ["particles"]},
              hint="Per-particle size spread (mixed snowflakes, mixed "
                   "bubbles)"),
            P("depth", "float", 0.0, 0.0, 1.0, when={"mode": ["particles"]},
              hint="Z scatter: particles spread toward/away from you and the "
                   "lens does the rest -- near ones large and bright, far ones "
                   "small and dim"),
            P("rise", "float", 0.0, -1.0, 1.0, when={"mode": ["particles"]},
              hint="Z drift: positive floats particles toward the camera as "
                   "they live, negative sinks them away"),
            P("swirl", "float", 0.0, -300.0, 300.0,
              when={"mode": ["particles"]},
              hint="A vortex around the attractor point: positive spins "
                   "counter-clockwise. Works with or without attraction"),
            P("floor", "float", 1.0, 0.05, 1.0, when={"mode": ["particles"]},
              hint="A ledge at this canvas height: particles land on it"),
            P("bounce", "float", 0.0, 0.0, 0.9, when={"mode": ["particles"]},
              hint="What landing does: 0 pools and sticks, higher splashes"),
            P("field", "str", "",
              hint="Existing content as a FORCE: a stroke id (S3), mask:<id>, "
                   "or sel -- that object then pulls, pushes, guides, or "
                   "fences this effect"),
            P("field_mode", "choice", "attract",
              choices=["attract", "repel", "flow", "contain"],
              hint="attract pulls toward it, repel pushes away, flow carries "
                   "along it, contain fences inside it"),
            P("field_strength", "float", 120.0, 0.0, 500.0),
            P("tubes", "int", 26, 2, 120, when={"mode": ["tubes"]},
              hint="How many tubes sprout along the stroke"),
            P("length", "float", 60.0, 8.0, 240.0, when={"mode": ["tubes"]},
              hint="How far a tube grows from its root"),
            P("branches", "int", 2, 0, 4, when={"mode": ["tubes"]},
              hint="Child tubes each tube may sprout — 0 for grass, more for shrubs"),
            P("droop", "float", 30.0, -120.0, 120.0, when={"mode": ["tubes"]},
              hint="Gravity on the growth: positive bends tips down like vines, "
                   "negative reaches upward like grass"),
            P("rise", "float", 0.0, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="Growth OUT of the canvas, toward you: 0 stays in the "
                   "plane, 1 grows ONLY in Z — tips loom closer and larger "
                   "(the camera widens with rise so the perspective reads)"),
            P("trunk", "bool", 0, when={"mode": ["tubes"]},
              hint="A connected centre vine: the drawn stroke itself becomes "
                   "a thicker tube every branch grows from"),
            P("up", "float", 0.0, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="Grow toward world-up instead of off the stroke's side: "
                   "grass points at the sky no matter which way the stroke "
                   "was drawn"),
            P("clump", "float", 0.0, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="Tufting: roots gather into clusters along the stroke -- "
                   "grass grows in tussocks, not a picket fence"),
            P("leaves", "float", 0.0, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="Foliage: this fraction of branch tips grows a leaf"),
            P("leaf_size", "float", 14.0, 4.0, 48.0, when={"mode": ["tubes"]}),
            P("leaf_r", "float", 0.30, 0.0, 1.0, when={"mode": ["tubes"]}),
            P("leaf_g", "float", 0.62, 0.0, 1.0, when={"mode": ["tubes"]}),
            P("leaf_b", "float", 0.22, 0.0, 1.0, when={"mode": ["tubes"]}),
            P("grow", "float", 1.0, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="Growth animation: 0 is bare stroke, 1 fully grown — "
                   "scrub it and the branches sprout"),
            P("depth3d", "float", 0.6, 0.0, 1.0, when={"mode": ["tubes"]},
              hint="How strongly depth shades and thins the far tubes — the "
                   "2.5-D dial. The depth itself is on the depth output socket."),
            P("size", "float", 1.6, 0.4, 8.0),
            P("density", "float", 1.0, 0.05, 2.0,
              hint="Overall opacity of the spray. Lower for a faint mist, "
                   "higher to build up solid colour."),
            P("attract_x", "float", 0.5, 0.0, 1.0,
              when={"attract": ["on"]}, hint="Attractor position across the canvas"),
            P("attract_y", "float", 0.5, 0.0, 1.0, when={"attract": ["on"]}),
            P("attract_strength", "float", 60.0, -200.0, 200.0,
              when={"attract": ["on"]},
              hint="Positive pulls particles in, negative pushes them away"),
            P("attract", "choice", "off", choices=["off", "on"]),
            P("r", "float", 1.0, 0.0, 1.0), P("g", "float", 0.55, 0.0, 1.0),
            P("b", "float", 0.15, 0.0, 1.0),
            P("seed", "int", 0, 0, 999)],
    doc="Paint Effects in the graph: a stroke that keeps growing after you "
        "draw it. Point it at a spline (draw one with the pen tool, P) and it "
        "emits particles along that path, then carries them under gravity, "
        "curl-noise wind and an optional attractor. Drips, sparks, spray and "
        "smoke trails all come from the same dials. The spline stays "
        "editable -- drag a point and the effect re-renders, because this "
        "node reads the stroke live rather than baking it. In TUBES mode "
        "branches grow from the stroke instead (grass, vines, shrubs) with a "
        "per-tube depth: far tubes render first, thinner and darker, and the "
        "depth itself comes out of the depth socket for fog or masking -- "
        "Paint Effects' 2.5-D, no meshes.",
    rgba=True, alpha="process", outputs=["out", "depth"])
def _strokefx(ctx, ins, p):
    h, w = ctx
    doc = _CTX_DOC[0] if _CTX_DOC else None
    # Accept SEVERAL paths, comma separated: a stroke selection is a set, so
    # one effect should cover the whole set rather than needing a node each.
    refs = [r.strip() for r in str(p.get("spline", "")).split(",") if r.strip()]
    paths = ([q for q in (_stroke_path(doc, r) for r in refs) if q is not None]
             if doc is not None else [])
    if not paths:
        z = np.zeros((h, w, 4), np.float32)         # nothing wired: contribute nothing
        return {"out": z, "depth": np.zeros((h, w, 3), np.float32)}
    if p.get("mode", "particles") == "tubes":
        return _strokefx_tubes(ctx, paths, p)
    path = np.concatenate(paths, axis=0) if len(paths) > 1 else paths[0]
    rng = np.random.default_rng(int(p["seed"]))     # deterministic per seed
    n = int(p["count"])
    idx = rng.integers(0, len(path), n)
    pos = path[idx] + rng.normal(0.0, max(float(p["spread"]), 1e-3), (n, 2))
    vel = rng.normal(0.0, 2.0, (n, 2)).astype(np.float32)
    # every particle its own mass: equal weights rendered drips as a uniform
    # veil -- lognormal weights give heavy rivulets threading through mist
    wgt = rng.lognormal(0.0, 0.55, n).astype(np.float32)
    wgt /= max(float(wgt.mean()), 1e-6)
    steps0 = int(p["life"])
    stag = float(p.get("stagger", 0.0))
    lvar = float(p.get("life_var", 0.0))
    birth = (rng.random(n) * stag * steps0 * 0.75).astype(np.float32)
    plife = steps0 * (1.0 - lvar * rng.random(n)).astype(np.float32)
    svar = float(p.get("size_var", 0.3))
    psize = (1.0 + svar * rng.normal(0.0, 0.6, n)).clip(0.35, 2.5)         .astype(np.float32)
    depth = float(p.get("depth", 0.0))
    risep = float(p.get("rise", 0.0))
    zrange = max(w, h) * 0.10
    pz = (rng.uniform(-1.0, 1.0, n) * depth * zrange).astype(np.float32)
    zstep = np.float32(risep * zrange * 1.6 / max(1, int(p["life"])))
    use3d = depth > 0.0 or abs(risep) > 1e-6
    if use3d:
        fovp = _strokefx_fov({"rise": abs(risep)})
        Dp = (h / 2.0) / np.tan(np.deg2rad(fovp / 2.0))
    grav = float(p["gravity"])
    windx = float(p["wind"])
    steps = int(p["life"])
    cx, cy = float(p["attract_x"]) * w, float(p["attract_y"]) * h
    attract_on = p.get("attract", "off") == "on"
    astr = float(p["attract_strength"])
    swirl = float(p.get("swirl", 0.0))
    floor_y = float(p.get("floor", 1.0)) * h
    bounce = float(p.get("bounce", 0.0))
    fld = _strokefx_field(doc, p.get("field", ""), h, w)
    fmode = str(p.get("field_mode", "attract"))
    fstr = float(p.get("field_strength", 120.0))
    curl = _curl_noise(32, 3, int(p["seed"])) if windx > 0 else None
    # TRAIL and HEAD are different things: the trail is where a particle has
    # BEEN (faint), the head is where it IS (bright). One combined field
    # rendered drips as uniform streaks with no droplet ends and embers as
    # smoke instead of sparks. Heads also FLICKER when heat is on.
    acc = np.zeros((h, w), np.float32)
    head = np.zeros((h, w), np.float32)
    heat0 = float(p.get("heat", 0.0))
    for _si in range(max(1, steps)):
        alive = (birth <= _si) & (_si < birth + plife)
        f = np.zeros_like(pos)
        f[:, 1] += grav
        if curl is not None:                        # swirling wind, not a flat push
            gx = np.clip((pos[:, 0] / max(w, 1) * 31).astype(int), 0, 31)
            gy = np.clip((pos[:, 1] / max(h, 1) * 31).astype(int), 0, 31)
            f[:, 0] += curl[0][gy, gx] * windx
            f[:, 1] += curl[1][gy, gx] * windx
        if attract_on and have("attractor_force"):
            try:
                f = f + np.asarray(mind().attractor_force(
                    pos, center=(cx, cy), strength=astr, softening=8.0), np.float32)
            except Exception:
                pass
        if fld is not None and fstr > 0.0:
            F0, fgx, fgy, fgw, fgh = fld
            ix = np.clip((pos[:, 0] / max(w, 1) * fgw).astype(int), 0, fgw - 1)
            iy = np.clip((pos[:, 1] / max(h, 1) * fgh).astype(int), 0, fgh - 1)
            sx, sy = fgx[iy, ix], fgy[iy, ix]
            if fmode == "attract":
                f[:, 0] += sx * fstr
                f[:, 1] += sy * fstr
            elif fmode == "repel":
                f[:, 0] -= sx * fstr
                f[:, 1] -= sy * fstr
            elif fmode == "flow":
                # along the contours: the gradient turned a quarter turn --
                # particles ride beside the guide stroke like a current
                f[:, 0] += -sy * fstr
                f[:, 1] += sx * fstr
            else:                                    # contain
                # uphill force that only exists OUTSIDE the region: inside
                # (F near 1) the fence disappears
                out_amt = np.clip(0.85 - F0[iy, ix], 0.0, 1.0) * 2.2
                f[:, 0] += sx * fstr * out_amt
                f[:, 1] += sy * fstr * out_amt
        if swirl != 0.0:
            # a VORTEX about the attractor point: force along the tangent,
            # fading with distance -- spirals, tornados, stirred embers.
            # Independent of attraction, so pure rotation works too.
            rx = pos[:, 0] - cx
            ry = pos[:, 1] - cy
            rr = np.sqrt(rx * rx + ry * ry) + 8.0
            f[:, 0] += swirl * (-ry / rr) * (60.0 / rr).clip(0.0, 1.0)
            f[:, 1] += swirl * (rx / rr) * (60.0 / rr).clip(0.0, 1.0)
        try:
            out = mind().advance_particles(pos, vel, force=f, dt=0.05, damping=0.04)
            np_pos, np_vel = out if isinstance(out, tuple) else (out, vel)
            np_pos = np.asarray(np_pos, np.float32)
            np_vel = np.asarray(np_vel, np.float32)
        except Exception:
            np_vel = vel + f * 0.05
            np_pos = pos + np_vel * 0.05
        # the DEAD stay where they died: their frozen position is the head
        pos = np.where(alive[:, None], np_pos, pos)
        vel = np.where(alive[:, None], np_vel, vel)
        if floor_y < h - 0.5:
            hit = alive & (pos[:, 1] > floor_y)
            if hit.any():
                pos[hit, 1] = floor_y
                if bounce > 0.0:
                    vel[hit, 1] = -np.abs(vel[hit, 1]) * bounce   # splash
                else:
                    vel[hit, 1] = 0.0                             # pool
                    vel[hit, 0] *= 0.6                            # friction
        if use3d:
            pz = np.where(alive, pz + zstep, pz)
        keep = alive & ((pos[:, 0] >= 0) & (pos[:, 0] < w) &
                        (pos[:, 1] >= 0) & (pos[:, 1] < h))
        if keep.any():
            dep = wgt[keep]
            if heat0 > 0.0:
                dep = dep * (0.55 + 0.9 * rng.random(int(keep.sum()))
                             .astype(np.float32))     # sparks flicker
            dpos = pos[keep]
            if use3d:
                # the shared lens: near particles land bigger and brighter,
                # far ones shrink and dim -- same projection as the tubes
                zc = np.minimum(pz[keep], Dp * 0.9)
                sc = Dp / (Dp - zc)
                dpos = np.stack([w / 2.0 + (dpos[:, 0] - w / 2.0) * sc,
                                 h / 2.0 + (dpos[:, 1] - h / 2.0) * sc], 1)
                dep = dep * (sc ** 1.5).astype(np.float32)
                inb = ((dpos[:, 0] >= 0) & (dpos[:, 0] < w) &
                       (dpos[:, 1] >= 0) & (dpos[:, 1] < h))
                dpos = dpos[inb]
                dep = dep[inb]
                if not len(dep):
                    continue
            try:
                acc += np.asarray(mind().scatter_to_field(
                    (h, w), dpos, dep), np.float32)
            except Exception:
                yy = dpos[:, 1].astype(int); xx = dpos[:, 0].astype(int)
                np.add.at(acc, (yy, xx), dep)
    shape = str(p.get("particle", "dot"))
    hw = wgt * float(max(1, steps)) * 0.55
    if use3d:
        # heads ride the same lens as trails: projected position, proximity
        # scaling for both brightness and stamp radius
        zc_all = np.minimum(pz, Dp * 0.9)
        sc_all = (Dp / (Dp - zc_all)).astype(np.float32)
        hpos = np.stack([w / 2.0 + (pos[:, 0] - w / 2.0) * sc_all,
                         h / 2.0 + (pos[:, 1] - h / 2.0) * sc_all], 1)
        hw = hw * sc_all ** 1.5
    else:
        sc_all = np.ones(len(pos), np.float32)
        hpos = pos
    keep = ((hpos[:, 0] >= 0) & (hpos[:, 0] < w) &
            (hpos[:, 1] >= 0) & (hpos[:, 1] < h))
    if keep.any() and shape == "dot":
        try:
            head += np.asarray(mind().scatter_to_field(
                (h, w), hpos[keep], hw[keep]), np.float32)
        except Exception:
            yy = hpos[keep, 1].astype(int); xx = hpos[keep, 0].astype(int)
            np.add.at(head, (yy, xx), hw[keep])
    elif keep.any():
        # SHAPED particles: each head is drawn as what it IS -- a motion
        # streak, a snow flake, a bubble ring, a four-point spark. Stamped
        # crisp (no blur pass) so the shape survives to the pixels.
        base_r = max(1.2, float(p["size"]))
        idxs = np.nonzero(keep)[0]
        for i2 in idxs:
            px, py = float(hpos[i2, 0]), float(hpos[i2, 1])
            r = base_r * float(psize[i2]) * float(sc_all[i2])
            wv = float(hw[i2])
            x0 = int(max(0, px - r * 2 - 2)); x1 = int(min(w, px + r * 2 + 3))
            y0 = int(max(0, py - r * 2 - 2)); y1 = int(min(h, py + r * 2 + 3))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = xx - px, yy - py
            if shape == "streak":
                v = vel[i2]
                L2 = max(float(np.hypot(v[0], v[1])), 1e-3)
                ux2, uy2 = v[0] / L2, v[1] / L2
                lu = dx * ux2 + dy * uy2
                lv = -dx * uy2 + dy * ux2
                m = np.clip((0.7 * r - np.abs(lv)) * 1.4, 0, 1)                     * np.clip((r * 2.2 - np.abs(lu + r)) / (r * 2.2), 0, 1)
            elif shape == "flake":
                a6 = np.arctan2(dy, dx) * 3.0 + float(psize[i2]) * 7.0
                arm = 0.55 + 0.45 * np.cos(a6) ** 2
                m = np.clip((r * arm - np.hypot(dx, dy)) * 1.3, 0, 1)
            elif shape == "bubble":
                d0 = np.hypot(dx, dy)
                m = np.clip((0.9 - np.abs(d0 - r)) * 1.3, 0, 1)
                m += np.clip((0.6 - np.hypot(dx + r * 0.35, dy + r * 0.35))
                             * 1.2, 0, 1)          # a specular glint
            else:                                   # spark: 4-point star
                d0 = np.hypot(dx, dy) + 1e-3
                star = (np.abs(dx * dy) / (d0 ** 1.5 + 0.4))
                m = np.clip((r * 1.6 * (1.0 - star) - d0) * 0.9, 0, 1)
            head[y0:y1, x0:x1] += m * wv * 0.14
    del keep
    if float(p["size"]) > 1.0:
        acc = _gauss_blur(acc[..., None], float(p["size"]) * 0.6)[..., 0]
        if str(p.get("particle", "dot")) == "dot":
            head = _gauss_blur(head[..., None], float(p["size"]) * 0.75)[..., 0]
    # Normalising by the MAX made the effect read as faint dust: one dense
    # pixel (where many particles happened to pile up) set the scale and
    # crushed everything else to a few percent alpha -- measured mean alpha of
    # 0.02 against a peak of 1.0. Scale by a high PERCENTILE instead, so the
    # bulk of the spray lands at a visible opacity and only genuine hot spots
    # clip. `density` then controls the overall strength honestly.
    nz = acc[acc > 0]
    scale = float(np.percentile(nz, 92.0)) if nz.size else 1.0
    hz = head[head > 0]
    hscale = float(np.percentile(hz, 96.0)) if hz.size else 1.0
    # trails faint, heads bright -- droplets end in drops, sparks are points
    a = (np.clip(acc / max(scale, 1e-6), 0.0, 1.0) * 0.55
         + np.clip(head / max(hscale, 1e-6), 0.0, 1.0))
    a = np.clip(a * float(p["density"]), 0.0, 1.0)
    col = np.array([p["r"], p["g"], p["b"]], np.float32).reshape(1, 1, 3)
    heat = float(p.get("heat", 0.0))
    if heat > 0.0:
        # dense cores run toward white-hot while sparse edges keep the base
        # colour -- the flat single orange read as stickers, not embers
        hotc = np.array([1.0, 0.93, 0.62], np.float32).reshape(1, 1, 3)
        mix = (a ** 0.75)[..., None] * heat
        col = col * (1 - mix) + hotc * mix
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = col * a[..., None]                # PREMULTIPLIED: composites
    out[..., 3] = a
    # particles are a flat spray: uniform mid-depth on the socket, so wiring
    # depth still yields something sensible rather than a black card
    return {"out": out, "depth": _rgb(np.full((h, w), 0.5, np.float32) * (a > 0))}


_FIELD_MEMO = {}


def _strokefx_field(doc, ref, h, w):
    """A FORCE FIELD from existing content: a stroke's path, a mask's
    grayscale, or a selection's region, turned into a scalar field F plus a
    unit-gradient on a coarse grid. Anything already in the document can
    then pull, push, guide, or contain the FX -- the Maya control-curve
    idea generalised to every paint object.

    ref: "S<id>" / "stroke:<id>" (path), "mask:<id>", "sel" / "sel:<id>".
    Memoised per (doc, ref, revision): masks repaint, strokes nudge, and
    the field must follow."""
    if not ref:
        return None
    # id(doc) alone is unsafe: after GC a NEW document can reuse the
    # address while the global revision happens to match, and a stale
    # grid built for the old document's size comes back -- an
    # intermittent wrong-size render that reproduced only under a full
    # chunk's allocation churn. The doc id string + dimensions pin it.
    key = (id(doc), getattr(doc, "id", ""), doc.width, doc.height,
           str(ref), _MUT_REV[0])
    if key in _FIELD_MEMO:
        return _FIELD_MEMO[key]
    gw = max(24, min(96, w // 8))
    gh = max(18, min(96, int(round(gw * h / max(w, 1)))))
    F = None
    r = str(ref).strip()
    try:
        if r.startswith("mask:"):
            mk = doc.mask_by_id(r[5:].strip())
            if mk is not None:
                F = _resize(np.asarray(mk.data, np.float32), gh, gw)
        elif r.startswith("sel"):
            sid = r[4:].strip() if r.startswith("sel:") else ""
            target = None
            # selections live in all_selections(); the .selections attribute
            # is a DIFFERENT (stroke-selection) list -- found the hard way
            for s in (doc.all_selections() or []):
                if not sid or getattr(s, "id", "") == sid:
                    target = s
                    break
            if target is not None:
                sm = doc.selection_to_mask(target.id)
                sm = np.asarray(getattr(sm, "data", sm), np.float32)
                F = _resize(sm, gh, gw)
        else:
            if r.startswith("stroke:"):
                r = r[7:].strip()
            path = _stroke_path(doc, r)
            if path is not None and len(path) >= 2:
                pts = np.asarray(path, np.float32)[:, :2]
                if len(pts) > 220:
                    pts = pts[:: len(pts) // 220 + 1]
                ys = (np.arange(gh, dtype=np.float32) + 0.5) * h / gh
                xs = (np.arange(gw, dtype=np.float32) + 0.5) * w / gw
                gx2, gy2 = np.meshgrid(xs, ys)
                d2 = np.full((gh, gw), 1e18, np.float32)
                for q in pts:
                    d2 = np.minimum(d2, (gx2 - q[0]) ** 2 + (gy2 - q[1]) ** 2)
                sig = 0.14 * max(w, h)
                F = np.exp(-np.sqrt(d2) / sig).astype(np.float32)
    except Exception:
        F = None
    if F is None:
        _FIELD_MEMO[key] = None
        return None
    # gradient from a SOFTENED copy: a hard region (selection, crisp mask)
    # has slope only in the single boundary cell -- zero force everywhere
    # else, so "contain" couldn't herd distant particles home. The soft
    # field gives the far field a slope; F itself stays sharp for the
    # inside/outside test.
    # TWO scales: the small blur keeps local structure (a painted blob
    # still pulls sharply), the large one gives the far field a slope (a
    # hard selection can herd particles from across the canvas). One big
    # blur alone flattened mask attraction from +61 px to +13.
    f32 = F.astype(np.float32)[..., None]
    near = _gauss_blur(f32, 1.6)[..., 0]
    # each scale normalised SEPARATELY, far deferring to near: blending the
    # raw gradients let the big blur's edge artifacts flip mid-range unit
    # directions and mask attraction collapsed +61 px -> +14 (A/B/C
    # measured: raw 61, near-only 61, naive blend 14). Near wins wherever
    # it exists; far only fills regions near genuinely cannot reach, which
    # is exactly what "contain" needs to herd distant particles.
    def _unit_gate(field2):
        gy2, gx2 = np.gradient(field2.astype(np.float32))
        m2 = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-8
        g2 = np.clip(m2 * gw * 6.0, 0.0, 1.0)
        return gx2 / m2 * g2, gy2 / m2 * g2, g2
    nx, ny, ngate = _unit_gate(near)
    # the far field points at the CENTROID: blur-based far gradients kept
    # inheriting boundary-reflection artifacts and flipping mid-range
    # directions (mask attraction collapsed 61 -> 8-14 px whichever way
    # they were blended); a mass-centroid direction cannot point wrong,
    # and long reach is exactly what contain and distant attraction need
    tot = float(F.sum()) + 1e-6
    cyx = float((F * np.arange(gh)[:, None]).sum()) / tot
    cxx = float((F * np.arange(gw)[None, :]).sum()) / tot
    yy2, xx2 = np.mgrid[0:gh, 0:gw].astype(np.float32)
    fdx, fdy = cxx - xx2, cyx - yy2
    fm = np.sqrt(fdx * fdx + fdy * fdy) + 1e-6
    gxg = (nx + (fdx / fm) * 0.8 * (1.0 - ngate)).astype(np.float32)
    gyg = (ny + (fdy / fm) * 0.8 * (1.0 - ngate)).astype(np.float32)
    out = (F.astype(np.float32), gxg, gyg, gw, gh)
    if len(_FIELD_MEMO) > 8:
        _FIELD_MEMO.clear()
    _FIELD_MEMO[key] = out
    return out



def _strokefx_fov(p):
    """One camera for every tube consumer. Rise needs a wider lens: at the
    resting 14 deg, a tube growing straight at the camera is an end-on dot
    with ~3%% scale change -- invisible. Widening with rise makes near tips
    genuinely LOOM, and mesh, fallback, and leaf stamping must all agree on
    the same projection or leaves drift off their tips."""
    return 14.0 + 40.0 * float(p.get("rise", 0.0))


def _strokefx_project(pt3, h, w, fov):
    """Perspective-project one skeleton point (canvas x, y, z-toward-camera)
    the same way the mesh camera does: eye on +z at the distance that frames
    the canvas plane 1:1. Returns (x, y, scale)."""
    D = (h / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    z = min(float(pt3[2]), D * 0.9)
    s = D / (D - z)
    cx, cy = w / 2.0, h / 2.0
    return (cx + (float(pt3[0]) - cx) * s, cy + (float(pt3[1]) - cy) * s, s)


def _strokefx_grow(ctx, paths, p, rng):
    """Grow the branch SKELETONS: 3-D polylines (canvas x, canvas y, depth z
    in canvas units) with a radius and a normalised depth each. Shared by the
    mesh renderer and the flat fallback so both draw the same plant."""
    h, w = ctx
    n_tubes = int(p.get("tubes", 26))
    length = float(p.get("length", 60.0))
    branches = int(p.get("branches", 2))
    droop = float(p.get("droop", 30.0))
    windx = float(p.get("wind", 0.0))
    curl = _curl_noise(32, 3, int(p["seed"])) if windx > 0 else None
    base_w = 1.2 + float(p.get("size", 1.6))
    rise = float(p.get("rise", 0.0))
    # rise splits each growth step between the canvas plane and Z: the arc
    # length stays the tube's length, so rise=1 grows the SAME amount, all
    # of it toward the camera, from a frozen root point
    kxy = float(np.sqrt(max(0.0, 1.0 - rise * rise)))
    zrange = max(w, h) * 0.10          # how deep the thicket is, canvas units
    out = []

    fld = _strokefx_field(_CTX_DOC[0] if _CTX_DOC else None,
                          p.get("field", ""), h, w)
    fmode = str(p.get("field_mode", "attract"))
    fsteer = float(p.get("field_strength", 120.0)) * 0.0022

    def grow(x, y, ang, ln, wd, z01, gen, side=1.0):
        # A stem is an ARC, not a random walk: one persistent curvature per
        # stem (plus gentle noise an order of magnitude smaller than before)
        # is what separates a plant from pick-up sticks. Children leave the
        # parent FORWARD at 30-50 degrees, alternating sides -- phyllotaxis
        # -- rather than the old perpendicular coin-flips that read as
        # scattered twigs in the user's review.
        steps = max(6, min(10, int(ln / 6.0)))
        z = z01 * zrange
        pts = [(x, y, z)]
        a2 = ang
        k0 = rng.normal(0.0, 0.055)            # this stem's own curve
        child_side = side
        for s in range(steps):
            t = s / max(steps - 1, 1)
            a2 += k0 + rng.normal(0.0, 0.03)
            a2 += (droop / 900.0) * t * np.cos(a2)
            if fld is not None and fsteer > 0.0:
                # growth STEERS by the field: vines reach toward an attract
                # stroke, part around a repel mask, run along a flow guide
                F0, fgx, fgy, fgw, fgh = fld
                ix = int(np.clip(x / max(w, 1) * fgw, 0, fgw - 1))
                iy = int(np.clip(y / max(h, 1) * fgh, 0, fgh - 1))
                gx0, gy0 = float(fgx[iy, ix]), float(fgy[iy, ix])
                if abs(gx0) + abs(gy0) > 1e-4:
                    if fmode == "repel":
                        tgt = np.arctan2(-gy0, -gx0)
                    elif fmode == "flow":
                        t1 = np.arctan2(gx0, -gy0)
                        t2 = np.arctan2(-gx0, gy0)
                        d1 = (t1 - a2 + np.pi) % (2 * np.pi) - np.pi
                        d2c = (t2 - a2 + np.pi) % (2 * np.pi) - np.pi
                        tgt = t1 if abs(d1) < abs(d2c) else t2
                    else:                            # attract / contain
                        tgt = np.arctan2(gy0, gx0)
                    dw = (tgt - a2 + np.pi) % (2 * np.pi) - np.pi
                    a2 += float(np.clip(dw * fsteer, -0.22, 0.22))
            step = ln / steps
            x += np.cos(a2) * step * kxy
            y += np.sin(a2) * step * kxy
            z += step * rise
            if curl is not None:
                gx = int(np.clip(x / max(w, 1) * 31, 0, 31))
                gy = int(np.clip(y / max(h, 1) * 31, 0, 31))
                x += curl[0][gy, gx] * windx * 0.02 * t
                y += curl[1][gy, gx] * windx * 0.02 * t
            pts.append((x, y, z + rng.normal(0.0, 0.4)))
            if gen > 0 and 0.25 < t < 0.85 and s % 2 == 0 \
                    and rng.random() < 0.55:
                child_side = -child_side
                # the child leaves at the parent's LOCAL width, not its
                # base width -- a base-width child bulged past the tapered
                # parent near its tip and every joint read as a knuckle
                local_w = wd * (1.0 - 0.78 * t)
                grow(x, y, a2 + child_side * rng.uniform(0.5, 0.9),
                     ln * (0.62 - 0.25 * t), max(0.55, local_w * 0.75),
                     min(1.0, max(0.0, z01 + rng.normal(0.0, 0.08))),
                     gen - 1, child_side)
        out.append((np.asarray(pts, np.float32), wd, z01))

    for path in paths:
        if len(path) < 2:
            continue
        if p.get("trunk"):
            # the CONNECTED option: the drawn stroke itself is a thicker
            # tube, and every branch already roots exactly on it -- so one
            # trunk turns scattered twigs into a single plant. It TAPERS:
            # the blunt chopped end read as a cut pipe, not a vine.
            tp = np.asarray(path, np.float32)
            step = max(1, len(tp) // 26)
            tp = tp[::step]
            tpts = np.zeros((len(tp), 3), np.float32)
            tpts[:, :2] = tp[:, :2]
            out.append((tpts, base_w * 2.2, 0.5))
        # roots spread by fractional position WITH jitter -- integer linspace
        # duplicated indices on short strokes, planting several tubes at the
        # SAME point: the starburst clumps in the review. Sides alternate
        # (phyllotaxis) and shoots shorten toward the stroke's end.
        fr = (np.arange(n_tubes) + 0.5) / n_tubes             + rng.uniform(-0.35, 0.35, n_tubes) / n_tubes
        clump = float(p.get("clump", 0.0))
        if clump > 0.0:
            # tufting: pull each root toward the nearest of a few cluster
            # centres -- full clump is discrete tussocks, half is loose ones
            nc = max(2, n_tubes // 5)
            centers = np.sort(rng.uniform(0.03, 0.97, nc))
            nearest = centers[np.argmin(
                np.abs(fr[:, None] - centers[None, :]), axis=1)]
            fr = fr * (1.0 - clump) + nearest * clump \
                + rng.normal(0.0, 0.006, n_tubes)
        side = 1.0
        for f in np.clip(fr, 0.0, 0.999):
            # a CONTINUOUS position along the path -- snapping to integer
            # indices put several roots on the same point whenever tubes
            # rivals the point count (pigeonhole), which the root-distance
            # check caught after the first "fix" only jittered the fraction
            fi = f * (len(path) - 1)
            i0 = int(fi)
            tfrac = fi - i0
            a = np.asarray(path[i0], np.float32)
            b = np.asarray(path[min(i0 + 1, len(path) - 1)], np.float32)
            rx = float(a[0] + (b[0] - a[0]) * tfrac)
            ry = float(a[1] + (b[1] - a[1]) * tfrac)
            tang = np.arctan2(b[1] - a[1], b[0] - a[0])
            side = -side
            ang = tang + side * rng.uniform(0.55, 0.95)
            upb = float(p.get("up", 0.0))
            if upb > 0.0:
                # gravity defines "up", not the stroke: blend the launch
                # angle toward straight up, keeping a fan of lean. The
                # sideways bird-foot tufts came from tangent-relative angles
                # on a horizontal stroke.
                target = -np.pi / 2 + side * rng.uniform(0.06, 0.5)
                dwrap = (target - ang + np.pi) % (2 * np.pi) - np.pi
                ang = ang + dwrap * upb
            tipward = 1.0 - 0.45 * f               # shorter near the tip
            grow(rx, ry, ang,
                 length * tipward * (0.75 + 0.5 * rng.random()),
                 base_w * (0.7 + 0.6 * rng.random()),
                 float(rng.random()), branches, side)
    grow = float(p.get("grow", 1.0))
    if grow < 1.0:
        # growth animation: every tube keeps its ROOT and loses its tip --
        # truncating the polyline reads as sprouting when scrubbed, and
        # children (which start mid-parent) vanish first, as growth should
        out = [(pts[:max(2, int(round(len(pts) * grow)))], wd, z)
               for pts, wd, z in out if grow > 0.02]
    # hard cap: growth is recursive and the renderer cost is tris x pixels
    return out[:140]


def _strokefx_tubes(ctx, paths, p):
    """Paint-Effects tube growth, 2.5-D. The skeletons become REAL geometry:
    every branch is swept into a tube mesh (leCore sweep_tube), all tubes
    merge into one mesh, and leCore's mesh renderer lights it -- so what used
    to be flat stamped polylines now has roundness, shading and true
    perspective (near tubes render slightly larger, a small-fov camera aimed
    square at the canvas). A second render pass with depth painted into
    vertex colours yields the matte AND the per-pixel depth for the `depth`
    socket in one go. Deterministic per seed; falls back to the flat stamper
    on builds without a mesh renderer."""
    rng = np.random.default_rng(int(p["seed"]))
    skel = _strokefx_grow(ctx, paths, p, rng)
    if have("sweep_tube", "render_mesh"):
        try:
            return _strokefx_render_mesh(ctx, skel, p)
        except Exception:
            pass                       # any mesh-path surprise: flat fallback
    return _strokefx_render_flat(ctx, skel, p)


def _strokefx_render_mesh(ctx, skel, p):
    h, w = ctx
    m = mind()
    d3 = float(p.get("depth3d", 0.6))
    verts, faces, vz, vcols = [], [], [], []
    base_col = np.array([float(p["r"]), float(p["g"]),
                         float(p["b"])], np.float32)
    off = 0
    for ti, (pts, wd, z01) in enumerate(skel):
        # render y is UP; canvas y is DOWN -- flip while building so the
        # output needs no post-flip and text stays with the maths
        p3 = pts.copy()
        p3[:, 1] = h - p3[:, 1]
        # sweep_tube takes ONE radius, so a single sweep gave constant-width
        # worms (measured 6x the flat renderer's coverage). Sweep in three
        # chunks with stepped radii instead: root -> mid -> tip taper, chunks
        # sharing an endpoint so the joins stay closed.
        n = len(p3)
        # every stem its own slight colour, and WOOD at the base: the root
        # chunks blend toward brown, stronger on thick primaries -- uniform
        # green plastic was part of the "not natural" read
        trng = np.random.default_rng((int(p.get("seed", 0)) * 7919
                                      + ti * 271) & 0x7fffffff)
        tube_col = base_col * float(trng.uniform(0.86, 1.10))
        woodiness = float(np.clip((wd - 1.0) / 2.6, 0.0, 1.0))
        wood = np.array([0.33, 0.25, 0.16], np.float32)
        cuts = [0, max(1, n // 4), max(2, n // 2), max(3, (3 * n) // 4), n - 1]
        for c in range(4):
            seg = p3[cuts[c]:cuts[c + 1] + 1]
            if len(seg) < 2:
                continue
            r = float(max(0.5, wd * (1.0, 0.72, 0.45, 0.22)[c]))
            try:
                V, F = m.sweep_tube(seg, radius=r)
            except Exception:
                continue
            V = np.asarray(V, np.float32)
            F = np.asarray(F, np.int64)
            verts.append(V)
            faces.append(F + off)
            off += len(V)
            vz.append(np.full(len(V), z01, np.float32))
            wb = (0.55, 0.28, 0.0, 0.0)[c] * woodiness
            cc = tube_col * (1 - wb) + wood * wb
            vcols.append(np.tile(np.clip(cc, 0, 1), (len(V), 1)))
    if not off:
        z = np.zeros((h, w, 4), np.float32)
        return {"out": z, "depth": _rgb(np.zeros((h, w), np.float32))}
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    Z = np.concatenate(vz)
    mesh = {"vertices": V.tolist(), "faces": F.tolist()}
    # camera looking straight at the canvas plane, framed so canvas units are
    # pixels; a small fov keeps parallax subtle (2.5-D, not a fly-through)
    fov = _strokefx_fov(p)
    D = (h / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    cam = {"eye": [w / 2.0, h / 2.0, float(D)],
           "target": [w / 2.0, h / 2.0, 0.0], "up": [0, 1, 0],
           "fov_deg": fov}
    rw = min(w, 512)
    rh = max(1, int(round(h * rw / max(w, 1))))
    VC = np.concatenate(vcols)
    col = np.asarray(m.render_mesh(mesh, cam, width=rw, height=rh,
                                   base_color=(1.0, 1.0, 1.0),
                                   vertex_colors=VC.tolist(),
                                   background=(0.0, 0.0, 0.0),
                                   ambient=0.35, smooth=True, two_sided=True,
                                   dtype=np.float32))
    # one extra pass carries BOTH the matte and the depth: vertex colours are
    # the tube depths lifted to [0.1, 1], flat-lit, on black -- anything > 0
    # is coverage, and the value decodes back to depth
    vc = np.stack([(1.0 - Z) * 0.9 + 0.1] * 3, axis=1)     # near = bright
    dm = np.asarray(m.render_mesh(mesh, cam, width=rw, height=rh,
                                  base_color=(1, 1, 1),
                                  background=(0.0, 0.0, 0.0), ambient=1.0,
                                  vertex_colors=vc.tolist(),
                                  dtype=np.float32)).max(axis=-1)
    if (rw, rh) != (w, h):
        col = _resize(col, h, w)
        dm = _resize(dm[..., None], h, w)[..., 0]
    alpha = np.clip(dm / 0.1, 0.0, 1.0)
    alpha = np.clip(alpha * float(p.get("density", 1.0)), 0.0, 1.0)
    # depth3d keeps its meaning as the shading dial: scale how much depth
    # darkens, on top of the renderer's own lighting
    depth01 = np.clip((dm - 0.1) / 0.9, 0.0, 1.0)
    shade = 1.0 - d3 * (1.0 - depth01) * 0.45
    out = np.zeros((h, w, 4), np.float32)
    out[..., 3] = alpha
    out[..., :3] = np.clip(col * shade[..., None], 0, 1) * alpha[..., None]
    _strokefx_leaves(out, skel, p, h, w)
    return {"out": out, "depth": _rgb(depth01 * (alpha > 0.01))}


def _stamp_leaf(out, pt3, ang, size, lc, h, w, fov, curl=0.0):
    """One leaf, anchored by its PETIOLE: a short stalk runs from the stem
    to the blade, the blade's axis bends by `curl` (real leaves sweep, they
    don't lie on rulers), and the teardrop keeps its sine profile, root->tip
    light, and darker midrib. Projected with the shared lens; premultiplied-
    over blending."""
    tx, ty, ts = _strokefx_project(pt3, h, w, fov)
    ll = float(size) * ts
    if ll < 1.5:
        return
    pl = ll * 0.22                             # petiole length
    ca, sa = np.cos(ang), np.sin(ang)
    pad = ll + pl + abs(curl) * ll + 2
    x0 = int(max(0, np.floor(tx - pad)))
    y0 = int(max(0, np.floor(ty - pad)))
    x1 = int(min(w, np.ceil(tx + pad)))
    y1 = int(min(h, np.ceil(ty + pad)))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rx, ry = xx - tx, yy - ty
    uu = rx * ca + ry * sa                    # along the stalk+leaf
    vv = -rx * sa + ry * ca                   # across it
    ub = uu - pl                               # blade coordinate, 0..ll
    t = np.clip(ub / max(ll, 1e-3), 0.0, 1.0)
    # the axis sweeps: the centreline moves sideways with t^1.6
    vc2 = vv - curl * ll * (t ** 1.6)
    half = np.sin(np.pi * t) ** 0.8 * ll * 0.34 + 0.6 * (1.0 - t)
    blade = np.clip((half - np.abs(vc2)) * 0.9, 0.0, 1.0)         * ((ub >= -0.3) & (ub <= ll)).astype(np.float32)
    stalk = np.clip((0.75 - np.abs(vv)) * 1.2, 0.0, 1.0)         * ((uu >= -0.5) & (ub <= 0.3)).astype(np.float32)
    a = np.maximum(blade, stalk)
    if float(a.max()) <= 0.0:
        return
    lit = 0.72 + 0.5 * t                       # brighter toward the tip
    rib = 1.0 - 0.35 * np.clip(1.0 - np.abs(vc2) / (0.08 * ll + 0.4), 0, 1)
    rgb = lc[None, None] * (lit * rib)[..., None]
    stalk_col = lc[None, None] * 0.55
    rgb = np.where(stalk[..., None] > blade[..., None], stalk_col
                   * np.ones_like(rgb), rgb)
    win = out[y0:y1, x0:x1]
    # premultiplied "over": the tube image stores premultiplied colour, and
    # straight-leaf-rgb x alpha IS the premultiplied contribution
    win[..., :3] = rgb * a[..., None] + win[..., :3] * (1 - a[..., None])
    win[..., 3] = np.clip(a + win[..., 3] * (1 - a), 0, 1)


def _strokefx_leaves(out, skel, p, h, w):
    """Foliage on branch tips: a procedural teardrop leaf -- sine width
    profile, a light gradient root->tip, a darker midrib -- stamped at the
    end of each chosen tube, oriented along the tip's own last segment with
    a little deterministic jitter. Far tips stamp first so near leaves
    overlap them the way depth says they should. leCore has no leaf-sprite
    faculty (texture_leaf is a texture-DSL constructor), so the shapes are
    ours; the probe that established that is in BACKLOG.md."""
    density = float(p.get("leaves", 0.0))
    if density <= 0.0 or not skel:
        return
    L = float(p.get("leaf_size", 14.0))
    lc = np.array([p.get("leaf_r", 0.3), p.get("leaf_g", 0.62),
                   p.get("leaf_b", 0.22)], np.float32)
    seed = int(p.get("seed", 0))
    fov = _strokefx_fov(p)
    order = sorted(range(len(skel)), key=lambda i: -skel[i][2])  # far first
    for i in order:
        pts, wd, z01 = skel[i]
        rng = np.random.default_rng((seed * 9176 + i * 131) & 0x7fffffff)
        if len(pts) < 3:
            continue
        # gate LEAF SITES, not whole tubes: the old per-tube coin flip left
        # a skipped tube completely bare, and those naked logs were what
        # read as "cut sticks" in the review. Now every stem can carry
        # leaves; density scales how many sites fire.
        do_tip = rng.random() < min(1.0, density + 0.3)
        # leaves grow ALONG the stem too, on little petiole angles that
        # alternate sides -- tip-only leaves left the stems bare, one of the
        # "sloppy" reads in the review. Stem leaves are smaller than the tip
        # leaf and skip the lowest third (old wood).
        # depth shade + per-leaf colour life: every leaf gets its own value
        # and a green-channel wobble -- one flat green read as plastic
        d3 = float(p.get("depth3d", 0.6))
        zshade = 1.0 - d3 * z01 * 0.45

        def leaf_col():
            j = np.array([rng.uniform(0.86, 1.10),
                          rng.uniform(0.90, 1.12),
                          rng.uniform(0.86, 1.10)], np.float32)
            return np.clip(lc * j * zshade, 0.0, 1.0)

        pside = 1.0 if rng.random() < 0.5 else -1.0
        for j2 in range(len(pts) // 3 + 1, len(pts) - 2, 2):
            if rng.random() > density * 0.6:
                continue
            pside = -pside
            dj = pts[min(j2 + 1, len(pts) - 1)][:2] - pts[max(j2 - 1, 0)][:2]
            if float(np.hypot(dj[0], dj[1])) < 1e-3:
                dj = np.array([1.0, 0.0], np.float32)
            aj = float(np.arctan2(dj[1], dj[0]))                 + pside * float(rng.uniform(0.7, 1.1))
            _stamp_leaf(out, pts[j2], aj, L * float(rng.uniform(0.45, 0.7))
                        * (0.6 + 0.8 * (1.0 - z01)), leaf_col(), h, w, fov,
                        curl=float(rng.uniform(-0.25, 0.25)))
        d = pts[-1][:2] - pts[-3][:2]
        if float(np.hypot(d[0], d[1])) < 1e-3:
            d = np.array([1.0, 0.0], np.float32)   # a pure-Z tip: any facing
        base_ang = float(np.arctan2(d[1], d[0]))
        if not do_tip:
            continue
        # a TERMINAL FAN: shoots end in 2-3 leaves around the tip direction,
        # the way new growth actually clusters, not one lone flag
        nfan = 2 + (rng.random() < 0.5)
        for kf in range(nfan):
            off = (kf - (nfan - 1) / 2.0) * float(rng.uniform(0.45, 0.65))
            _stamp_leaf(out, pts[-1], base_ang + off,
                        L * float(rng.uniform(0.62, 1.05))
                        * (0.6 + 0.8 * (1.0 - z01)), leaf_col(), h, w, fov,
                        curl=float(rng.uniform(-0.3, 0.3)))


def _strokefx_render_flat(ctx, skel, p):
    """The pre-mesh renderer, kept as the fallback for leCore builds without
    render_mesh: the SAME skeletons, stamped flat with painter's-algorithm
    depth ordering, tip taper, and depth-based darkening/thinning."""
    h, w = ctx
    d3 = float(p.get("depth3d", 0.6))
    col = np.array([p["r"], p["g"], p["b"]], np.float32)
    acc = np.zeros((h, w, 4), np.float32)
    zbuf = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    fov = _strokefx_fov(p)
    for pts3, wd, z in sorted(skel, key=lambda t: -t[2]):        # far first
        shade = 1.0 - d3 * z * 0.75
        thin = 1.0 - d3 * z * 0.45
        c = col * shade
        n = len(pts3)
        for i2 in range(n):
            t = i2 / max(n - 1, 1)
            px, py, ps = _strokefx_project(pts3[i2], h, w, fov)
            r = max(0.4, wd * (1.0 - 0.85 * t) * thin * ps)     # root -> tip
            x0 = max(0, int(px - r - 1))
            x1 = min(w, int(px + r + 2))
            y0 = max(0, int(py - r - 1))
            y1 = min(h, int(py + r + 2))
            if x1 <= x0 or y1 <= y0:
                continue
            d2 = ((xx[y0:y1, x0:x1] - px) ** 2
                  + (yy[y0:y1, x0:x1] - py) ** 2)
            mk = np.clip(1.0 - np.sqrt(d2) / r, 0.0, 1.0)
            win = acc[y0:y1, x0:x1]
            win[..., :3] = c * mk[..., None] + win[..., :3] * (1 - mk[..., None])
            win[..., 3] = np.maximum(win[..., 3], mk)
            zw = zbuf[y0:y1, x0:x1]
            zbuf[y0:y1, x0:x1] = np.where(mk > 0.2, 1.0 - z, zw)
    out = np.zeros((h, w, 4), np.float32)
    out[..., 3] = np.clip(acc[..., 3] * float(p.get("density", 1.0)), 0, 1)
    out[..., :3] = acc[..., :3] * out[..., 3:4]
    _strokefx_leaves(out, skel, p, h, w)
    return {"out": out, "depth": _rgb(zbuf)}


@op("Scatter", "Generate", inputs=["mask", "palette"],
    outputs=["out", "matte"],
    params=[P("density", "float", 0.5, 0.02, 1.0), P("size", "float", 6.0, 1.0, 40.0),
            P("size_jitter", "float", 0.5, 0.0, 1.0),
            P("r", "float", 1.0, 0.0, 1.0), P("g", "float", 0.4, 0.0, 1.0),
            P("b", "float", 0.6, 0.0, 1.0), P("seed", "int", 3, 0, 999)],
    doc="Scatters dots -- flowers in a meadow, stars, confetti, snow -- evenly "
        "across the frame using blue-noise (no clumps, no grid). Wire a mask "
        "to confine them to a region (dots only land where the mask is bright); "
        "wire a palette image to colour each dot by sampling it, or use the "
        "r/g/b colour. size_jitter varies the dot sizes so they don't look "
        "stamped. Outputs the dots plus a matte for compositing with Merge.")
def _scatter(ctx, ins, p):
    h, w = ctx
    rng = np.random.default_rng(int(p["seed"]))
    # blue-noise points in the unit square; radius from desired density
    dens = float(p["density"])
    radius = float(np.clip(0.9 / np.sqrt(max(dens, 0.02) * 900.0), 0.008, 0.2))
    try:
        pts = np.asarray(mind().blue_noise_sample(radius, [(0, 0), (1, 1)],
                                                  k=30, seed=int(p["seed"])))
    except Exception:
        pts = rng.random((int(dens * 400) + 20, 2))
    if pts.size == 0:
        return {"out": np.zeros((h, w, 3), np.float32),
                "matte": np.zeros((h, w), np.float32)}
    mask = None
    if ins.get("mask") is not None and np.asarray(ins["mask"]).size > 1:
        mask = _lum(_rgb(ins["mask"]))
    pal = None
    if ins.get("palette") is not None and np.asarray(ins["palette"]).size > 1:
        pal = _rgb(ins["palette"])
    out = np.zeros((h, w, 3), np.float32)
    matte = np.zeros((h, w), np.float32)
    base_col = np.array([p["r"], p["g"], p["b"]], np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for i, (fx, fy) in enumerate(pts):
        px, py = int(fx * (w - 1)), int(fy * (h - 1))
        if mask is not None and mask[py, px] < 0.5:
            continue                                 # confined to the mask
        jitter = 1.0 + (rng.random() - 0.5) * 2 * float(p["size_jitter"])
        rad = max(1.0, float(p["size"]) * jitter)
        d2 = (xs - px) ** 2 + (ys - py) ** 2
        dot = np.clip(1.0 - d2 / (rad * rad), 0, 1)
        if dot.max() < 1e-6:
            continue
        col = pal[py, px] if pal is not None else base_col
        out += dot[..., None] * col[None, None, :]
        matte = np.maximum(matte, dot)
    return {"out": np.clip(out, 0, 1), "matte": matte}


@op("Warped noise", "Generate",
    params=[P("scale", "float", 2.0, 0.3, 10.0), P("octaves", "int", 4, 1, 8),
            P("warp", "float", 0.4, 0.0, 1.5), P("seed", "int", 0, 0, 99)],
    doc="leCore warped_noise: fbm with domain warping -- the marble/flow-field look.")
def _warpednoise(ctx, ins, p):
    h, w = ctx
    f = mind().warped_noise(scale=p["scale"], octaves=int(p["octaves"]),
                            warp=p["warp"], seed=int(p["seed"]))
    v = np.asarray(f(_grid_pts(h, w, 3))).reshape(h, w)
    return _rgb((v - v.min()) / max(np.ptp(v), 1e-9))


@op("SDF render", "Generate",
    params=[P("preset", "choice", "dsl", choices=["dsl", "mandelbulb", "mandelbox"]),
            P("power", "float", 8.0, 2.0, 16.0),
            P("dsl", "text", "(smooth_union 0.3 (sphere 0.8) (translate 0.9 0 0 (box 0.45 0.45 0.45)))"),
            P("orbit", "float", 35.0, 0.0, 360.0), P("height", "float", 1.6, -3.0, 4.0),
            P("dist", "float", 3.2, 1.2, 10.0),
            P("r", "float", 0.85, 0.0, 1.0), P("g", "float", 0.5, 0.0, 1.0), P("b", "float", 0.35, 0.0, 1.0),
            P("reflect", "float", 0.25, 0.0, 1.0)],
    doc="leCore render_sdf: raymarch an SDF with soft shadows, AO and reflection. "
        "preset=dsl parses the (kind p0 ...) expression -- compose with union / "
        "smooth_union / subtract / twist / repeat / rounded, the full "
        "holographic_sdf algebra. mandelbulb (power = polar exponent) and "
        "mandelbox render leCore's classic fractal distance estimators; orbit / "
        "height / dist frame them like any other shape.")
def _sdfrender(ctx, ins, p):
    from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
    h, w = ctx
    a = np.deg2rad(p["orbit"]); d = float(p["dist"])
    # a plain dict: leCore coerces it at the faculty boundary (their C2), so we
    # avoid reaching into holographic.rendering for the Camera class. Older
    # builds fall back to m.camera(), then to the class import as a last resort.
    cam = {"eye": (d * np.cos(a), float(p["height"]), d * np.sin(a)),
           "target": (0, 0, 0)}
    if p.get("preset") == "mandelbulb":
        tree = mind().mandelbulb(power=float(p["power"]))
    elif p.get("preset") == "mandelbox":
        tree = mind().fold_fractal(iterations=max(4, int(p["power"])))
    else:
        tree = parse_dsl(p["dsl"])
    # preview at capped res, conform to canvas afterwards (the coarse-first discipline)
    ph, pw = min(h, 288), min(w, 384)
    try:
        img = np.asarray(mind().render_sdf(tree, cam, width=pw, height=ph,
                                           base_color=(p["r"], p["g"], p["b"]),
                                           reflect=p["reflect"]))
    except (TypeError, AttributeError):              # older leCore: needs a real Camera
        try:
            cam_obj = mind().camera(**cam) if have("camera") else None
        except Exception:
            cam_obj = None
        if cam_obj is None:
            from holographic.rendering.holographic_render import Camera
            cam_obj = Camera(**cam)
        img = np.asarray(mind().render_sdf(tree, cam_obj, width=pw, height=ph,
                                           base_color=(p["r"], p["g"], p["b"]),
                                           reflect=p["reflect"]))
    return _rgb(img)


@op("Texture synth", "Filter", inputs=["exemplar"],
    params=[P("psize", "int", 24, 8, 48), P("overlap", "int", 6, 2, 16), P("seed", "int", 0, 0, 99)],
    doc="Grows MORE texture that looks like the input (Photoshop: Content-Aware-"
        "style synthesis): give it a small patch -- grass, fabric, noise -- and it "
        "paints a whole canvas of it. Slow at large sizes; great for backdrops.")
def _texsynth(ctx, ins, p):
    h, w = ctx
    ex = _rgb(ins["exemplar"])
    # keep the exemplar modest so quilting stays interactive
    eh, ew = min(ex.shape[0], 160), min(ex.shape[1], 160)
    ex = ex[:eh, :ew]
    ps = int(min(p["psize"], eh - 1, ew - 1))
    ov = int(min(p["overlap"], ps - 1))
    return _rgb(mind().synthesize_texture(ex, min(h, 320), min(w, 320),
                                          psize=ps, overlap=ov, seam="mincut",
                                          seed=int(p["seed"])))


@op("Align", "Combine", inputs=["moving", "reference"],
    doc="Auto-registers input b onto input a (like Photoshop's Auto-Align Layers): "
        "estimates the shift between the two and moves b to match. Use before Blend "
        "when combining hand-held shots.")
def _align(ctx, ins, p):
    a = _rgb(ins["moving"]); b = _rgb(ins["reference"])
    out = np.empty_like(a)
    for c in range(3):
        out[..., c] = np.asarray(mind().reproject(a[..., c].astype(float),
                                                  b[..., c].astype(float)))
    return np.clip(out, 0, 1)


@op("Depth fog", "FX", inputs=["image"],
    params=[P("depth", "choice", "fused", choices=["fused", "shading", "haze", "sharpness", "ground"]),
            P("density", "float", 0.6, 0.0, 3.0),
            P("fog_r", "float", 0.55, 0.0, 1.0),
            P("fog_g", "float", 0.65, 0.0, 1.0),
            P("fog_b", "float", 0.82, 0.0, 1.0),
            P("detail", "int", 192, 96, 512,
              hint="Depth-estimate resolution -- below ~192 the depth goes wrong, not just soft")],
    doc="leCore monocular depth (shading / fused haze+defocus / haze / sharpness / "
        "ground) + depth_fog: fade the estimated depth into atmosphere by "
        "Beer-Lambert. ground uses linear perspective (the detected vanishing "
        "point) -- the right choice for roads, rails, hallways and other "
        "one-point-perspective shots where texture cues mislead.")
def _depthfog(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    # Estimate depth on a CAPPED copy and upsample: a depth map is a smooth,
    # low-frequency field, so this is nearly free in quality but much cheaper.
    # MEASURED at 256x384: cap 192 = 4.8x faster at 0.944 correlation with the
    # full-res estimate; going to 128 would be 27x but correlation collapses to
    # 0.30 -- a fast wrong answer. 192 is the honest floor, hence the minimum.
    cap = max(int(p.get("detail", 192)), 96)
    sc = max(h, w) / float(cap)
    src = img if sc <= 1.0 else _resize(img, max(int(h / sc), 8),
                                        max(int(w / sc), 8))
    est = {"shading": lambda: mind().depth_from_image(src),
           "fused": lambda: mind().auto_fuse_depth(src),
           "haze": lambda: mind().haze_depth(src),
           "sharpness": lambda: mind().sharpness_depth(src),
           "ground": lambda: mind().ground_plane_depth(src.astype(float))}[
               p.get("depth", "fused")]
    depth = np.asarray(est()).astype(float)
    depth = (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    if depth.shape[:2] != (h, w):
        depth = _resize(depth.astype(np.float32), h, w).astype(float)
    return _rgb(mind().depth_fog(img, depth, density=p["density"],
                                 fog_color=(p["fog_r"], p["fog_g"], p["fog_b"])))



def _postfx_steps(p):
    steps = []
    if abs(p["exposure"]) > 1e-3: steps.append(("exposure", {"ev": p["exposure"]}))
    if abs(p["contrast"] - 1) > 1e-3 or abs(p["saturation"] - 1) > 1e-3 or abs(p["temperature"]) > 1e-3:
        steps.append(("color_grade", {"contrast": p["contrast"], "saturation": p["saturation"],
                                      "temperature": p["temperature"]}))
    if p["bloom"] > 1e-3: steps.append(("bloom", {"intensity": p["bloom"]}))
    if p["glare"] > 1e-3: steps.append(("glare", {"intensity": p["glare"]}))
    if p["flare"] > 1e-3: steps.append(("lens_flare", {"intensity": p["flare"]}))
    if p["chroma"] > 1e-5: steps.append(("chromatic_aberration", {"strength": p["chroma"]}))
    if p["tonemap"] != "none": steps.append((p["tonemap"], {}))
    if p["grain"] > 1e-3: steps.append(("film_grain", {"amount": p["grain"]}))
    if p["vignette"] > 1e-3: steps.append(("vignette", {"strength": p["vignette"]}))
    return steps


@op("Post FX", "FX", inputs=["image"],
    params=[P("exposure", "float", 0.0, -2.0, 2.0), P("contrast", "float", 1.0, 0.5, 1.8),
            P("saturation", "float", 1.0, 0.0, 2.0), P("temperature", "float", 0.0, -1.0, 1.0),
            P("bloom", "float", 0.0, 0.0, 1.5), P("glare", "float", 0.0, 0.0, 1.0),
            P("flare", "float", 0.0, 0.0, 1.0), P("chroma", "float", 0.0, 0.0, 0.02),
            P("grain", "float", 0.0, 0.0, 0.15), P("vignette", "float", 0.0, 0.0, 1.0),
            P("tonemap", "choice", "none", choices=["none", "reinhard", "aces"])],
    doc="leCore postfx_chain: the fusable post-processing algebra -- exposure, colour "
        "grade, bloom, glare streaks, lens flare, chromatic aberration, film grain, "
        "vignette, Reinhard/ACES tonemap -- compiled into one chain and applied once.")
def _postfx(ctx, ins, p):
    steps = _postfx_steps(p)
    if not steps:
        return _rgb(ins["image"])
    chain = mind().postfx_chain(*steps)
    return _rgb(chain.apply(_rgb(ins["image"]).astype(float)))


_CTX_DOC = []      # [Document] the graph is evaluating
_WIND_MEMO = {}


def _hair_wind(strength, res, bounds, octaves, seed):
    """CurlWind construction is a pure function of its arguments and measured
    6.4 s -- essentially all of a simulation's runtime, paid again on every
    call. Memoised on the arguments (bounded); nothing mutable is involved, so
    it cannot go stale. Same case as _curl_noise."""
    key = (round(float(strength), 4), int(res), tuple(map(tuple, bounds)),
           int(octaves), int(seed))
    hit = _WIND_MEMO.get(key)
    if hit is None:
        hit = mind().hair_wind(strength=float(strength), res=int(res),
                               bounds=bounds, octaves=int(octaves),
                               seed=int(seed))
        if len(_WIND_MEMO) > 8:
            _WIND_MEMO.clear()
        _WIND_MEMO[key] = hit
    return hit


_CURL_MEMO = {}


def _curl_noise(res, octaves, seed):
    """curl_noise is a pure function of (res, octaves, seed) -- MEASURED
    deterministic -- and costs ~2.4 s, which was essentially all of Flow warp's
    runtime. It does not depend on the image at all, so every tweak of `amount`
    or `steps` paid that 2.4 s again. Memoised on its arguments: nothing
    mutable is involved, so it cannot go stale."""
    key = (int(res), int(octaves), int(seed))
    hit = _CURL_MEMO.get(key)
    if hit is None:
        hit = np.asarray(mind().curl_noise(res=int(res), octaves=int(octaves),
                                           seed=int(seed)))
        if len(_CURL_MEMO) > 12:              # bounded: a few fields, not a leak
            _CURL_MEMO.clear()
        _CURL_MEMO[key] = hit
    return hit


@op("Flow warp", "Filter", inputs=["image"],
    params=[P("amount", "float", 8.0, 0.0, 40.0), P("octaves", "int", 4, 1, 6),
            P("steps", "int", 3, 1, 8), P("seed", "int", 0, 0, 99)],
    doc="leCore curl_noise: advect the image along a divergence-free curl-noise flow "
        "field (semi-Lagrangian steps) -- smoke-like smearing that never piles up.", alpha="process")
def _flowwarp(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    v = _curl_noise(64, int(p["octaves"]), int(p["seed"]))
    vy = _resize(v[0], h, w); vx = _resize(v[1], h, w)
    sc = float(p["amount"]) / max(np.abs(v).max(), 1e-9) / max(int(p["steps"]), 1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img
    for _ in range(int(p["steps"])):
        yy = np.clip(ys - vy * sc, 0, h - 1).astype(int)
        xx = np.clip(xs - vx * sc, 0, w - 1).astype(int)
        out = out[yy, xx]
    return out


@op("Seamless clone", "Combine", inputs=["base", "insert", "mask"],
    params=[P("mix_gradients", "bool", 1)],
    doc="Poisson image editing on leCore solve_poisson_periodic: paste `insert` into "
        "`base` where the mask is bright by solving for the image whose gradients match "
        "-- seams vanish because only gradients, not colours, are transplanted.")
def _seamless(ctx, ins, p):
    a = _rgb(ins["base"]); b = _rgb(ins["insert"])
    msk = (_rgb(ins["mask"]).mean(-1) > 0.5).astype(np.float32)
    if msk.max() == 0:
        return a
    out = np.empty_like(a)
    for c in range(3):
        A, B = a[..., c].astype(float), b[..., c].astype(float)
        gyA, gxA = np.gradient(A); gyB, gxB = np.gradient(B)
        if p.get("mix_gradients"):
            useB = (gxB ** 2 + gyB ** 2) > (gxA ** 2 + gyA ** 2)
            gx = np.where(msk > 0.5, np.where(useB, gxB, gxA), gxA)
            gy = np.where(msk > 0.5, np.where(useB, gyB, gyA), gyA)
        else:
            gx = np.where(msk > 0.5, gxB, gxA); gy = np.where(msk > 0.5, gyB, gyA)
        div = np.gradient(gy, axis=0) + np.gradient(gx, axis=1)
        sol = np.asarray(mind().solve_poisson_periodic(div))
        sol = sol - sol[msk < 0.5].mean() + A[msk < 0.5].mean() if (msk < 0.5).any() else sol
        out[..., c] = np.where(msk > 0.5, sol, A)
    return np.clip(out, 0, 1)


@op("Annotate", "Filter", inputs=["image"],
    params=[P("lines", "int", 5, 0, 12), P("corners", "int", 12, 0, 40)],
    doc="leCore image_lines (Hough) + image_corners (Harris-style): overlay the "
        "structure the engine sees -- detected lines and interest points.")
def _annotate(ctx, ins, p):
    img = _rgb(ins["image"]).copy()
    h, w = img.shape[:2]
    if int(p["lines"]) > 0:
        for (theta, rho, _s) in np.asarray(mind().image_lines(img, top=int(p["lines"]))):
            ct, st = np.cos(theta), np.sin(theta)
            for t in np.linspace(-max(h, w), max(h, w), 4 * max(h, w)):
                x = int(rho * ct - t * st); y = int(rho * st + t * ct)
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = [0.35, 0.9, 0.85]
    if int(p["corners"]) > 0:
        for (cy, cx) in np.asarray(mind().image_corners(img, n=int(p["corners"]))).astype(int):
            y0, y1 = max(0, cy - 2), min(h, cy + 3); x0, x1 = max(0, cx - 2), min(w, cx + 3)
            img[y0:y1, cx: cx + 1] = [1.0, 0.45, 0.85]
            img[cy: cy + 1, x0:x1] = [1.0, 0.45, 0.85]
    return img


@op("ASCII art", "FX", inputs=["image"],
    params=[P("columns", "int", 96, 24, 200),
            P("font", "font", "DejaVuSansMono"),
            P("color", "choice", "source",
              choices=["source", "mono", "background"]),
            P("bg_r", "float", 0.05, 0.0, 1.0), P("bg_g", "float", 0.05, 0.0, 1.0),
            P("bg_b", "float", 0.07, 0.0, 1.0),
            P("invert", "bool", 0)],
    doc="The image as a luminance-ramp character grid, rendered back to pixels. "
        "color=source paints each glyph with the colour it samples from the "
        "image (full colour ASCII); mono is classic grey-on-dark; background "
        "fills each cell with the sampled colour behind a dark glyph. Each "
        "glyph is placed by the font's true advance width, so proportional "
        "fonts stay aligned (a monospace font gives the tightest grid).")
def _asciiart(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = ctx
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(_font_path(p.get("font")), 14)
        except Exception:
            font = ImageFont.load_default()
        ramp = " .:-=+*#%@"
        if p.get("invert"):
            ramp = ramp[::-1]
        # cell size from the font's true metrics: width = the max ADVANCE over
        # the ramp (so proportional fonts still tile), height from the ascent
        widths = [max(font.getlength(ch), 1) for ch in ramp if ch != " "]
        cw = max(int(round(max(widths))), 1)
        asc, desc = font.getmetrics()
        chh = max(asc + desc, 1)
        cols = int(p["columns"])
        rows = max(1, int(round((h / w) * cols * (cw / chh))))
        small = _resize(img, rows, cols)              # keep colour per cell
        lum = small.mean(-1)
        idx = np.clip((lum * (len(ramp) - 1)).round().astype(int),
                      0, len(ramp) - 1)
        bg = (int(p["bg_r"] * 255), int(p["bg_g"] * 255), int(p["bg_b"] * 255))
        canvas = Image.new("RGB", (cols * cw, rows * chh), bg)
        d = ImageDraw.Draw(canvas)
        mode = p.get("color", "source")
        for i in range(rows):
            for j in range(cols):
                ch = ramp[int(idx[i, j])]
                cell = small[i, j]
                col = tuple(int(np.clip(c, 0, 1) * 255) for c in cell[:3])
                if mode == "background":
                    d.rectangle([j * cw, i * chh, (j + 1) * cw, (i + 1) * chh],
                                fill=col)
                    glyph = bg
                elif mode == "mono":
                    glyph = (220, 224, 236)
                else:                                  # source: glyph takes the colour
                    glyph = col
                if ch != " " and mode != "background":
                    adv = font.getlength(ch)
                    d.text((j * cw + (cw - adv) / 2, i * chh), ch,
                           fill=glyph, font=font)
                elif ch != " ":                        # background mode: dark glyph on colour
                    adv = font.getlength(ch)
                    d.text((j * cw + (cw - adv) / 2, i * chh), ch,
                           fill=bg, font=font)
        a = _f32(np.asarray(canvas)) / 255.0
        ah, aw = a.shape[:2]
        scale = min(h / ah, w / aw)
        nh, nw = max(1, int(ah * scale)), max(1, int(aw * scale))
        fit = _resize(a, nh, nw)
        out = np.full((h, w, 3),
                      [p["bg_r"], p["bg_g"], p["bg_b"]], np.float32)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = fit
        return out
    except Exception:
        return img  # Pillow missing: pass through


@op("Smart smooth", "Filter", inputs=["image", "guide"],
    params=[P("radius", "int", 8, 2, 32), P("eps", "float", 0.02, 0.0005, 0.3)],
    doc="leCore's guided filter (He/Sun/Tang): smooths WHERE the guide image is "
        "smooth, holds detail where it has edges -- the pro version of "
        "edge-preserving blur (skin, sky, denoise without mush). The guide input "
        "is optional; unwired, the image guides itself. Smaller eps = stricter "
        "edges. Denoise is the lighter edge-gated cousin.", alpha="process")
def _smartsmooth(ctx, ins, p):
    img = _rgb(ins["image"])
    g = ins.get("guide")
    guide = _rgb(g).mean(-1) if g is not None and np.asarray(g).size > 1 else img.mean(-1)
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = np.asarray(mind().guided_filter(
            guide.astype(float), img[..., c].astype(float),
            radius=int(p["radius"]), eps=float(p["eps"])))
    return np.clip(out, 0, 1)


@op("Deconvolve", "Filter", inputs=["image"],
    params=[P("sigma", "float", 2.0, 0.4, 8.0), P("iters", "int", 25, 3, 120),
            P("strength", "float", 1.0, 0.0, 1.0)],
    doc="TRUE deblurring by iterative (Van Cittert) deconvolution -- leCore's "
        "sharpen_loop algorithm applied in 2-D: recovers detail a gaussian-ish "
        "blur destroyed, instead of just edging contrast like Sharpen's unsharp "
        "mask. sigma should roughly match the blur being undone; more iters digs "
        "deeper (and slower).", alpha="process")
def _deconvolve(ctx, ins, p):
    # leCore's sharpen_image is 1-D in 0.2.3 (see APP_BACKLOG D); this is the
    # same residual-fitting loop run against our 2-D gaussian.
    img = _rgb(ins["image"])
    sigma, iters = float(p["sigma"]), int(p["iters"])
    out = img.copy()
    lam = 0.9
    for _ in range(iters):
        out = out + lam * (img - _gauss_blur(out, sigma))
        out = np.clip(out, -0.25, 1.25)
    out = np.clip(out, 0, 1)
    return np.clip(img + float(p["strength"]) * (out - img), 0, 1)


@op("Seamless noise", "Generate",
    params=[P("beta", "float", 2.0, 0.5, 4.0), P("seed", "int", 7, 0, 9999),
            P("contrast", "float", 1.0, 0.2, 3.0)],
    doc="leCore spectral_field: 1/f^beta fractal noise synthesised in the Fourier "
        "domain, so it TILES PERFECTLY (verified by seam_continuity in our tests). "
        "The texture staple Warped noise can't guarantee: beta ~1 = fine grain, "
        "~3 = soft clouds. Feed Gradient map or Displace.")
def _seamlessnoise(ctx, ins, p):
    h, w = ctx
    f = np.asarray(mind().spectral_field((h, w), beta=float(p["beta"]),
                                         seed=int(p["seed"])), np.float32)
    f = (f - f.mean()) * (0.22 * float(p["contrast"]) / max(f.std(), 1e-6)) + 0.5
    return np.clip(f, 0, 1)


@op("Erosion", "Filter", inputs=["image"],
    params=[P("droplets", "int", 4000, 200, 30000), P("seed", "int", 1, 0, 9999),
            P("strength", "float", 1.0, 0.0, 1.0)],
    doc="leCore hydraulic erosion: rain droplets carve drainage channels into the "
        "image's luminance as a heightfield -- weathered stone, aged paint, "
        "instant canyon texture from any gradient. strength blends the carved "
        "height back over the original.")
def _erosion(ctx, ins, p):
    img = _rgb(ins["image"])
    hgt = img.mean(-1).astype(float)
    er = np.asarray(mind().terrain_erode(hgt, droplets=int(p["droplets"]),
                                         seed=int(p["seed"])), np.float32)
    delta = (er - hgt)[..., None]
    return np.clip(img + float(p["strength"]) * delta, 0, 1)


@op("Branching growth", "Generate",
    params=[P("kind", "choice", "lightning", choices=["lightning", "frost"]),
            P("steps", "int", 220, 20, 1200), P("seed", "int", 1, 0, 9999),
            P("glow", "float", 0.5, 0.0, 1.0)],
    doc="leCore dielectric-breakdown growth: the SAME physics grows a lightning "
        "bolt (sparse, jagged) or a frost dendrite (dense, feathery). "
        "Deterministic per seed; wire a Value into steps to animate the growth. "
        "glow mixes in the potential field around the branches -- composite with "
        "Glow + Merge for storm shots.")
def _branchgrow(ctx, ins, p):
    h, w = ctx
    n = max(48, min(256, min(h, w)))
    g = (mind().grow_lightning if p["kind"] == "lightning"
         else mind().grow_ice)((n, n), steps=int(p["steps"]), seed=int(p["seed"]))
    body = np.asarray(g.cluster, np.float32)
    phi = np.asarray(g.phi, np.float32)
    phi = (phi - phi.min()) / max(phi.max() - phi.min(), 1e-6)
    f = np.clip(body + float(p["glow"]) * phi * (1 - body), 0, 1)
    return _resize(f, h, w)


@op("Reaction diffusion", "Generate",
    params=[P("steps", "int", 60, 5, 400), P("seed", "int", 0, 0, 9999),
            P("scale", "int", 96, 32, 256)],
    doc="leCore's HyperCA reaction-diffusion: Turing patterns -- coral, "
        "fingerprints, animal skin -- grown from a seeded field. steps is growth "
        "time; scale is the simulation grid (small = chunky cells). Already "
        "colourful; Black & white + Threshold gives crisp masks.")
def _reactdiff(ctx, ins, p):
    h, w = ctx
    rd = mind().reaction_diffusion(size=int(p["scale"]), steps=int(p["steps"]),
                                   seed=int(p["seed"]))
    img = np.asarray(rd.image(), np.float32)
    return _resize(np.clip(img, 0, 1), h, w)


@op("Smoke", "Generate",
    params=[P("preset", "choice", "rising",
              choices=["rising", "plume", "swirl", "opposing", "shear", "buoyant"]),
            P("steps", "int", 40, 2, 300), P("seed", "int", 0, 0, 9999),
            P("scale", "int", 96, 32, 192)],
    doc="leCore's FFT smoke solver, six named presets: density rendered as a "
        "soft plume. Wire a Value into steps and scrub it (or drive it from "
        "Media playback) to animate the simulation deterministically.")
def _smokegen(ctx, ins, p):
    h, w = ctx
    name = p["preset"] if p["preset"] in set(mind().smoke_preset_names()) else "rising"
    r = mind().smoke_preset(name, nx=int(p["scale"]), ny=int(p["scale"]),
                            steps=int(p["steps"]), seed=int(p["seed"]))
    d = np.asarray(r["density"], np.float32)
    d = d / max(d.max(), 1e-6)
    return _resize(np.clip(d, 0, 1), h, w)


@op("Perceptual diff", "Comp", inputs=["a", "b"], outputs=["out", "score"],
    doc="leCore compare_images: a multi-scale perceptual A/B check. The image "
        "output is the amplified difference heat map (black = identical); the "
        "score socket is similarity in 0..1 (1 = identical) -- wire it into a "
        "Value-driven parameter or just read it on the node.")
def _pdiff(ctx, ins, p):
    a = _rgb(ins["a"]); b = _rgb(ins["b"])
    score = float(mind().compare_images(a, b))
    d = np.abs(a - b).mean(-1, keepdims=True)
    heat = np.clip(d / max(d.max(), 1e-6), 0, 1)
    out = np.concatenate([heat, heat * 0.35, 1 - heat], -1) * (d > 1e-5)
    return {"out": np.clip(out, 0, 1), "score": score}


@op("Distance field", "Comp", inputs=["matte"],
    params=[P("invert", "bool", 0), P("soften", "float", 1.0, 0.05, 4.0)],
    doc="How far every pixel is from the nearest bright area of the matte, as a smooth ramp. The engine behind neon outlines, contour lines and soft auras: Threshold it at rising levels for concentric rings around a logo, or feed it to Gradient map for a glow that follows a shape.")
def _distfield(ctx, ins, p):
    mtt = _rgb(ins["matte"]).mean(-1)
    seeds = mtt > 0.5
    if not seeds.any():
        return np.zeros((*mtt.shape, 3), np.float32)
    d = np.asarray(mind().distance_transform(seeds), np.float32)
    d = d / max(d.max(), 1e-6)
    d = d ** (1.0 / float(p["soften"]))
    if p.get("invert"):
        d = 1 - d
    return np.repeat(np.clip(d, 0, 1)[..., None], 3, -1)


@op("Splatify", "FX", inputs=["image"],
    params=[P("splats", "int", 220, 16, 900),
            P("mode", "choice", "colour", choices=["colour", "luminance"])],
    doc="leCore splat_field: rebuild the image as a superposition of Gaussian "
        "splats fitted by matching pursuit -- the painterly 'few soft strokes' "
        "look, and the same primitives 3D Gaussian Splatting uses (export the "
        "composite as .ply from the Export menu). colour fits each channel; "
        "luminance fits one set and tints it from the source.")
def _splatify(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = ctx
    k = int(p["splats"])
    src = img if img.shape[:2] == (h, w) else _resize(img, h, w)
    sh, sw = min(h, 160), min(w, 214)               # fit at working res, upsample
    small = _resize(src, sh, sw)
    if p.get("mode") == "luminance":
        _, ren = mind().splat_field(small.mean(-1).astype(float), k=k)
        ren = np.asarray(ren, np.float32)
        lum = np.clip(small.mean(-1), 1e-4, 1)
        out = np.clip(small * (ren / lum)[..., None], 0, 1)
    else:
        chans = []
        for c in range(3):
            _, ren = mind().splat_field(small[..., c].astype(float), k=k)
            chans.append(np.asarray(ren, np.float32))
        out = np.clip(np.stack(chans, -1), 0, 1)
    return _resize(out, h, w)


@op("Perspective grid", "FX", inputs=["image"],
    params=[P("lines", "int", 12, 4, 32), P("opacity", "float", 0.5, 0.05, 1.0),
            P("grid_r", "float", 0.2, 0.0, 1.0),
            P("grid_g", "float", 0.9, 0.0, 1.0),
            P("grid_b", "float", 1.0, 0.0, 1.0)],
    doc="leCore vanishing_point: finds the image's dominant perspective "
        "convergence from its strong oblique lines and overlays a radiating "
        "guide grid + horizon through it -- the drawing-tutor / architecture "
        "check. Composites over the image at the chosen opacity.")
def _perspgrid(ctx, ins, p):
    img = _rgb(ins["image"])
    h, w = img.shape[:2]
    try:
        vp = mind().vanishing_point(img.astype(float))
        vx, vy = float(vp[0]), float(vp[1])
    except Exception:
        vx, vy = w / 2, h / 2
    vx = float(np.clip(vx, -w, 2 * w)); vy = float(np.clip(vy, -h, 2 * h))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    overlay = np.zeros((h, w), np.float32)
    n = int(p["lines"])
    ang = np.arctan2(ys - vy, xs - vx)
    spokes = np.abs(((ang / np.pi * n) + 0.5) % 1.0 - 0.5)
    r = np.hypot(xs - vx, ys - vy)
    overlay = np.maximum(overlay, np.clip(1.0 - spokes * r * 0.35, 0, 1))
    overlay = np.maximum(overlay, np.clip(1.0 - np.abs(ys - vy) / 1.2, 0, 1))  # horizon
    col = np.asarray([p["grid_r"], p["grid_g"], p["grid_b"]], np.float32)
    a = (overlay * float(p["opacity"]))[..., None]
    return np.clip(img * (1 - a) + col[None, None, :] * a, 0, 1)


@op("3D model", "Generate",
    params=[P("asset", "asset", "sample"),
            P("orbit", "float", 35.0, 0.0, 360.0),
            P("height", "float", 0.7, -2.0, 3.0),
            P("zoom", "float", 1.0, 0.4, 3.0),
            P("r", "float", 0.8, 0.0, 1.0), P("g", "float", 0.78, 0.0, 1.0),
            P("b", "float", 0.75, 0.0, 1.0),
            P("ambient", "float", 0.35, 0.0, 1.0)],
    doc="Render YOUR OWN 3-D model in the graph: upload a .obj or .glb with "
        "the asset picker and orbit around it with the dials -- product "
        "mock-ups, reference turnarounds, a hero object to composite into a "
        "scene (its background is dark; key or Luma-key it out, or Merge over "
        "anything). The camera auto-frames the model; zoom pulls in and out. "
        "SDF render is the sibling for procedural shapes and fractals.")
def _model3d(ctx, ins, p):
    h, w = ctx
    _ensure_sample_asset()
    aid = (p.get("asset") or "").strip()
    if not aid or aid not in ASSETS:
        raise ValueError("no 3-D model loaded -- pick one in the node's asset "
                         "picker (upload a .obj or .glb)")
    mesh = asset_mesh(aid)
    m = mind()
    a = np.deg2rad(float(p["orbit"]))
    cam = m.fit_camera(mesh, direction=(float(np.cos(a)),
                                        float(p["height"]) + 0.35,
                                        float(np.sin(a))))
    cam = dict(cam)
    cam.pop("aspect", None)
    tgt = np.asarray(cam["target"], float)
    eye = tgt + (np.asarray(cam["eye"], float) - tgt) / max(float(p["zoom"]), 0.05)
    cam["eye"] = eye.tolist()
    ph, pw = min(h, 320), min(w, 426)
    # leCore 0.2.4 coerces a plain dict at the faculty boundary (their C2), so
    # the deep `holographic.rendering...Camera` class import is no longer
    # needed -- that import was exactly the kind of version-fragile reach that
    # broke the shader node. Older builds still get a real Camera via m.camera.
    cam_arg = cam
    if not have("render_mesh"):
        raise ValueError("this leCore build cannot render meshes")
    try:
        img = np.asarray(m.render_mesh(mesh, cam_arg, width=pw, height=ph,
                                       base_color=(p["r"], p["g"], p["b"]),
                                       ambient=float(p["ambient"]),
                                       dtype=np.float32))
    except TypeError:
        # older leCore: no dtype= and/or no dict coercion
        try:
            cam_obj = m.camera(**cam) if have("camera") else None
        except Exception:
            cam_obj = None
        if cam_obj is None:
            from holographic.rendering.holographic_render import Camera
            cam_obj = Camera(**cam)
        img = np.asarray(m.render_mesh(mesh, cam_obj, width=pw, height=ph,
                                       base_color=(p["r"], p["g"], p["b"]),
                                       ambient=float(p["ambient"])))
    return _resize(_rgb(img), h, w)


@op("Shadertoy", "Generate", inputs=["channel0", "channel1"],
    outputs=["out", "value"],
    params=[P("source", "shader", _ST_DEFAULT),
            P("time", "float", 0.0, 0.0, 120.0),
            P("mouse_x", "float", 0.5, 0.0, 1.0),
            P("mouse_y", "float", 0.5, 0.0, 1.0)],
    doc="Run REAL Shadertoy GLSL on your graphics card, in the graph. Paste "
        "any shadertoy.com shader that uses mainImage / iTime / iResolution / "
        "iMouse / iChannel0-1 -- it runs unchanged. The optional inputs feed "
        "iChannel0 and iChannel1, so a shader can distort or colour another "
        "node's image. The `value` socket is the frame's brightness as a "
        "number: wire it into any parameter (Blur size, Glow strength...) and "
        "your shader becomes an animation curve -- wire a Value node into "
        "`time` and scrub. GLSL errors appear right on the node.")
def _shadertoy(ctx, ins, p):
    h, w = ctx
    ph, pw = min(h, 512), min(w, 512)
    src = p.get("source") or _ST_DEFAULT
    chsig, chans = [], []
    for sock in ("channel0", "channel1"):
        im = ins.get(sock)
        if im is not None and np.asarray(im).size > 1:
            arr = _rgb(im)
            chans.append(arr)
            chsig.append("%x" % (hash(arr.tobytes()) & 0xffffffff))
        else:
            chans.append(None)
            chsig.append("-")
    key = _st_key(src, float(p["time"]), float(p["mouse_x"]),
                  float(p["mouse_y"]), pw, ph, ",".join(chsig))
    if key in SHADER_ERRORS:
        raise ValueError("GLSL: " + SHADER_ERRORS[key])
    if key in SHADER_FRAMES:
        SHADER_PENDING.pop(key, None)
        f = SHADER_FRAMES[key]
        out = _resize(f[..., :3], h, w)
        return {"out": out, "value": float(out.mean())}
    SHADER_PENDING[key] = {
        "key": key, "source": src, "time": float(p["time"]),
        "mouse_x": float(p["mouse_x"]), "mouse_y": float(p["mouse_y"]),
        "width": pw, "height": ph,
        "channels": [c for c in chans],
    }
    ph_img = _st_placeholder(h, w, "shader renders in your browser\u2026")
    return {"out": ph_img, "value": float(ph_img.mean())}


@op("Layer", "Input", params=[P("layer", "layerref", "")],
    outputs=["out", "alpha"],
    doc="Read a document layer (by id) into the graph. The alpha socket carries "
        "the layer's transparency as a grey image -- wire it into Merge or "
        "Channel split for real compositing.")
def _layer_in(ctx, ins, p, doc=None):
    raise RuntimeError("resolved by the graph")  # special-cased in NodeGraph.evaluate


@op("Layer group", "Input", params=[P("group", "text", "")],
    outputs=["out", "alpha"],
    doc="Read a document layer GROUP (made in the Layers panel): its member layers "
        "are composited in stack order with their blend modes and opacities, and the "
        "combined image feeds the graph. The alpha socket carries the group's "
        "combined transparency.")
def _layergroup(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Mask", "Input", params=[P("mask", "maskref", ""), P("invert", "bool", 0)],
    doc="Read a document mask as a greyscale image (optionally inverted). Wire it "
        "into Mask mix, Inpaint, Seamless clone -- anywhere a mask socket lives.")
def _maskin(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Mask out", "Output", inputs=["image"], params=[P("mask", "maskref", "")],
    doc="Write the wired image's luminance into a document mask whenever the graph "
        "changes. Drive it from a Layer or Layer group (via Edges, Segment, "
        "Threshold-ish Levels...) to turn image content into a selection.")
def _maskout(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Paint out", "Output", inputs=["image"],
    params=[P("label", "text", "Paint source")],
    doc="Make the wired image PAINTABLE: pick the Node paint tool on the canvas "
        "and brush strokes reveal this node's output where they pass -- a clone "
        "brush whose source photo is the graph. The tool stays disabled until an "
        "image is wired in here.")
def _paintout(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Brush out", "Output", inputs=["image"], params=[P("brush", "text", "")],
    doc="Write the wired image's luminance into a CUSTOM brush tip (downsampled to "
        "the tip resolution, centre-weighted). Compose tips from Pattern, Fractal, "
        "Warped noise, Edges... then paint with them on the canvas.")
def _brushout(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Layer out", "Output", inputs=["image"], params=[P("layer", "layerref", "")],
    doc="Write whatever is wired in HERE into a document layer, every time the graph "
        "changes. Stack several Layer outs -- the document composites them with each "
        "layer's blend mode and opacity, and the Output node shows the combined image.")
def _layerout(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


@op("Group", "Combine", inputs=["in0", "in1", "in2", "in3"],
    params=[P("label", "text", "Group")],
    doc="A reusable bundle: several nodes wrapped into one tidy block with its "
        "own inputs and promoted knobs. Build a look once (say noise -> colour "
        "-> mask), group it, then drop copies wherever you need it and tweak "
        "the exposed dials -- change the group and every copy... stays "
        "independent (each is its own instance). Great for repeated motifs "
        "like 'a patch of flowers' or 'a colour-graded layer'. Select nodes "
        "and choose Group to make one; the subgraph rides inside the .lews.")
def _group_stub(ctx, ins, p):
    # never actually called: the engine intercepts "Group" in _eval_all and runs
    # the subgraph. This stub exists so OPS registration / schema stay uniform.
    h, w = ctx
    return np.zeros((h, w, 3), np.float32)


@op("Output", "Output", inputs=["image"],
    doc="The final image: the composite of ALL layers (including ones driven by "
        "Layer out nodes). Wire an image in to override with a single node instead.")
def _output(ctx, ins, p):
    raise RuntimeError("resolved by the graph")


def op_catalog():
    """The UI's node menu: every operator's sockets, params, category, and doc."""
    cat = {}
    for name, meta in OPS.items():
        d = {k: v for k, v in meta.items() if k != "fn"}
        req = d.get("requires") or []
        d["available"] = (not req) or have(*req)     # ask the engine, per build
        cat[name] = d
    return cat


# ------------------------------------------------------------------------------------------------
# NodeGraph -- dependency-keyed evaluation with an O(change) cache, leCore-style.
# ------------------------------------------------------------------------------------------------

def _group_ids(n, doc):
    gid = (n.get("params") or {}).get("group", "")
    try:
        return list(doc.group(gid)["layers"])
    except KeyError:
        return []


def _src_ref(v):
    """Split an input value into (node_id, out_socket). Accepts the client's
    dot form ("NID.depth"), a bare id ("NID" -> out), or a [id, socket] pair
    -- the pair form used to be silently stringified into a bogus node id
    ("['LD', 'x']"), which the signature walk then KeyError'd on."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return str(v[0]), str(v[1])
    v = str(v)
    if "." in v:
        a, b = v.split(".", 1)
        return a, b
    return v, "out"


def _node_doc(graph, n):
    """The document a Layer / Layer group / Mask node reads from: its own by
    default, or another workspace document via the node's `doc` param."""
    did = (n.get("params") or {}).get("doc", "")
    if did and did != graph.doc.id and graph.resolver:
        d = graph.resolver(did)
        if d is not None:
            return d
    return graph.doc


class NodeGraph:
    def __init__(self, document: Document):
        self.doc = document
        try:
            document.graph_ref = self   # the timeline drives node params
        except Exception:
            pass
        self.nodes = {}          # id -> {id, type, params, inputs {socket: "nid[.sock]"}, x, y}
        self._cache = {}         # id -> (signature, {socket: image})
        self._counter = 0
        self.progress_cb = None  # optional fn(done, total)
        self.cancel_event = None # optional threading.Event
        self.timings = {}        # nid -> seconds of the last actual compute
        self._pass_comp = None   # document composite, memoised per evaluation
        self._pass_depth = 0
        self.resolver = None     # optional fn(doc_id) -> Document, for cross-doc reads
        self.media = None        # optional fn(node_id, params, want) -> frame | seq
        self._sigmemo = {}       # nid -> (time, sig, rev): a ~150ms TTL memo, so one
                                 # interactive tick hashes each layer only once
        self._laststruct = None  # structural fingerprint guarding the memo

    # --- structure -------------------------------------------------------------------------------
    def set_graph(self, nodes):
        # Editing the GRAPH is a mutation too. The signature memo keys on this
        # counter, so without the bump a freshly posted graph could be scored
        # with the previous graph's signatures and serve cached pixels for a
        # node whose wiring or flags just changed.
        _MUT_REV[0] += 1
        self.nodes = {n["id"]: n for n in nodes}

    def patch_node(self, nid, params=None, inputs=None, pos=None):
        """Update one node in place: merge param and/or input deltas without
        re-sending the whole graph. Returns the updated node dict. Raises
        KeyError for an unknown id. Setting an input value to null removes
        that wire."""
        n = self.nodes[nid]
        _MUT_REV[0] += 1                      # same reason as set_graph
        if params:
            n.setdefault("params", {}).update(params)
        if inputs is not None:
            wires = n.setdefault("inputs", {})
            for k, v in inputs.items():
                if v is None:
                    wires.pop(k, None)
                else:
                    wires[k] = v
        if pos is not None:
            n["x"], n["y"] = pos
        return n

    def to_list(self):
        return list(self.nodes.values())

    def ensure_default(self):
        """Seed the graph on first use with the Output node: it shows the composite
        of all layers until Layer / Layer group inputs and Layer outs are built."""
        if not self.nodes:
            self.nodes = {
                "output0": {"id": "output0", "type": "Output", "params": {},
                            "inputs": {}, "x": 320, "y": 80},
            }
        return self.nodes

    def input_layer_ids(self):
        """Every layer the graph READS: Layer nodes + Layer group members."""
        ids = set()
        for n in self.nodes.values():
            foreign = (n.get("params") or {}).get("doc", "")
            if foreign and foreign != self.doc.id:
                continue                      # cross-doc reads never conflict with local writes
            if n["type"] == "Layer":
                lid = (n.get("params") or {}).get("layer", "")
                if lid:
                    ids.add(lid)
            elif n["type"] == "Layer group":
                ids.update(_group_ids(n, self.doc))
        return ids

    def input_mask_ids(self):
        """Every mask the graph READS via a Mask node."""
        return {(n.get("params") or {}).get("mask", "")
                for n in self.nodes.values() if n["type"] == "Mask"} - {""}

    def commit_layer_outputs(self):
        """Evaluate every Layer out node against the CURRENT document, then write all
        targets in one step (snapshot semantics: reads happen before any write, so
        chains that read layers stay well-defined even when they feed other layers)."""
        blocked = self.input_layer_ids()
        blocked_masks = self.input_mask_ids()
        writes, mask_writes, conflicts = [], [], []
        brush_writes = []
        for n in self.nodes.values():
            if n["type"] == "Brush out":
                tgt = (n.get("params") or {}).get("brush", "")
                src = (n.get("inputs") or {}).get("image")
                if not tgt or src is None:
                    continue
                try:
                    b = self.doc.brush_by_id(tgt)
                except KeyError:
                    continue
                if b.builtin:
                    continue                     # standard brushes stay standard
                img = self.evaluate(n["id"]).mean(-1)
                side = min(img.shape)            # centre square crop, then tip-size
                y0 = (img.shape[0] - side) // 2; x0 = (img.shape[1] - side) // 2
                brush_writes.append((tgt, _resize(img[y0:y0+side, x0:x0+side],
                                                  b.TIP, b.TIP)))
                continue
            if n["type"] == "Mask out":
                tgt = (n.get("params") or {}).get("mask", "")
                src = (n.get("inputs") or {}).get("image")
                if not tgt or src is None:
                    continue
                if tgt in blocked_masks:
                    conflicts.append(tgt)
                    continue
                try:
                    self.doc.mask_by_id(tgt)
                except KeyError:
                    continue
                mask_writes.append((tgt, self.evaluate(n["id"]).mean(-1)))
                continue
            if n["type"] != "Layer out":
                continue
            tgt = (n.get("params") or {}).get("layer", "")
            src = (n.get("inputs") or {}).get("image")
            if not tgt or src is None:
                continue
            if tgt in blocked:
                conflicts.append(tgt)      # a Layer out may not drive a layer the graph reads
                continue
            try:
                self.doc.layer(tgt)
            except KeyError:
                continue
            writes.append((tgt, self.evaluate(n["id"])))
        for tgt, img in writes:
            l = self.doc.layer(tgt)
            img = np.asarray(img, np.float32)
            l.pixels[..., :3] = img[..., :3]
            l.pixels[..., 3] = img[..., 3] if img.shape[-1] == 4 else 1.0
        for tgt, mv in mask_writes:
            self.doc.mask_by_id(tgt).data = np.clip(_lum(_rgb(_f32(mv))), 0, 1)
        for tgt, tv in brush_writes:
            self.doc.brush_by_id(tgt).tip = np.clip(_lum(_rgb(_f32(tv))), 0, 1)
        self.last_conflicts = conflicts
        return len(writes) + len(mask_writes) + len(brush_writes)

    def upstream_ids(self, nid, acc=None):
        acc = acc if acc is not None else set()
        if nid in acc or nid not in self.nodes:
            return acc
        acc.add(nid)
        for src in (self.nodes[nid].get("inputs") or {}).values():
            self.upstream_ids(_src_ref(src)[0], acc)
        return acc

    def output_node(self):
        for n in self.nodes.values():
            if n["type"] == "Output":
                return n["id"]
        return None

    # --- evaluation ------------------------------------------------------------------------------
    def _struct_key(self):
        """Cheap fingerprint of the graph's SHAPE -- types, params, wiring. No
        pixel hashing; changing any of these invalidates the signature memo."""
        return hash(tuple(sorted(
            (nid, n["type"], repr(sorted((n.get("params") or {}).items())),
             repr(sorted((n.get("inputs") or {}).items())))
            for nid, n in self.nodes.items())))

    def _pixhash(self, arr):
        """Content hash for change detection. Strided (every 2nd row/col): 1/4 the
        bytes for the same practical sensitivity -- any brush touch, transform, or
        video frame moves many contiguous pixels."""
        return hashlib.md5(np.ascontiguousarray(arr[::2, ::2]).tobytes()).hexdigest()

    def _sig(self, nid, seen=None):
        if seen is None:                       # top-level: memo, valid for 150ms AND
            struct = self._struct_key()        # only while neither edits nor the
            if struct != self._laststruct:     # graph's own structure moved
                self._sigmemo.clear()
                self._laststruct = struct
            hit = self._sigmemo.get(nid)
            now = _time.monotonic()
            if hit and now - hit[0] < 0.15 and hit[2] == _MUT_REV[0]:
                return hit[1]
            sig = self._sig(nid, set())
            self._sigmemo[nid] = (now, sig, _MUT_REV[0])
            if len(self._sigmemo) > 256:
                self._sigmemo.clear()
            return sig
        return self._sig_inner(nid, seen)

    def _sig_inner(self, nid, seen=None):
        seen = seen or set()
        if nid in seen:
            raise ValueError("cycle at " + nid)
        n = self.nodes[nid]
        parts = [n["type"], json.dumps(n.get("params", {}), sort_keys=True),
                 "muted" if n.get("mute") else ""]
        if n["type"] in ("Layer", "Layer group", "Mask"):
            parts.append("doc:" + str((n.get("params") or {}).get("doc", "")))
        if n["type"] == "time":
            dd = _node_doc(self, n)
            parts.append("clock:%.4f" % float(getattr(dd, "frame", 0.0)))
        if n["type"] == "Shadertoy":
            parts.append("stgen:%d" % SHADER_GEN[0])   # frames arrive out-of-band
        if n["type"] == "Media in":
            p = n.get("params") or {}
            seq = self.media(nid, p, "seq") if self.media else 0
            parts.append(f"media:{p.get('source','')}:{seq}")
        if n["type"] == "Layer out":
            parts.append("target:" + str((n.get("params") or {}).get("layer", "")))
        elif n["type"] == "Mask out":
            parts.append("mtarget:" + str((n.get("params") or {}).get("mask", "")))
        elif n["type"] == "Brush out":
            parts.append("btarget:" + str((n.get("params") or {}).get("brush", "")))
        elif n["type"] == "Mask":
            mid = (n.get("params") or {}).get("mask", "")
            try:
                m = _node_doc(self, n).mask_by_id(mid)
                parts.append(mid + self._pixhash(m.data[:, :, None])
                             + str((n.get("params") or {}).get("invert", 0)))
            except KeyError:
                parts.append(mid + ":gone")
        elif n["type"] == "Layer":
            l = _node_doc(self, n).layer(n.get("params", {}).get("layer", ""))
            parts.append(self._pixhash(l.pixels))
        elif n["type"] == "Layer group":
            gdoc = _node_doc(self, n)
            for lid in _group_ids(n, gdoc):
                try:
                    l = gdoc.layer(lid)
                    parts.append(lid + self._pixhash(l.pixels)
                                 + f"{l.blend}{l.opacity}{l.visible}{l.mask}{l.mask_invert}")
                    if l.mask:
                        try:
                            parts.append(self._pixhash(gdoc.mask_by_id(l.mask).data[:, :, None]))
                        except KeyError:
                            pass
                except KeyError:
                    parts.append(lid + ":gone")
        elif n["type"] == "Output" and not (n.get("inputs") or {}).get("image"):
            comp = self._doc_comp()
            parts.append(hashlib.md5(comp.tobytes()).hexdigest())
        for sock, src in (n.get("inputs") or {}).items():
            sid, ssock = _src_ref(src)
            parts.append(sock + ":" + ssock + ":" + self._sig_inner(sid, seen | {nid}))
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def render_at(self, nid, width, height, sock="out"):
        """Evaluate a node at a target resolution instead of the document size.
        Procedural nodes (noise, gradients, fractals) synthesize genuinely more
        detail; pixel sources upscale. Restores everything afterward."""
        d = self.doc
        ow, oh = d.width, d.height
        saved_cache, saved_memo = self._cache, self._sigmemo
        self._cache, self._sigmemo = {}, {}
        d.width, d.height = int(width), int(height)
        try:
            return self.evaluate(nid, sock)
        finally:
            d.width, d.height = ow, oh
            self._cache, self._sigmemo = saved_cache, saved_memo

    def _eval_group(self, n, h, w):
        """Evaluate a Group node: an encapsulated subgraph, spliced into this
        graph under a private id prefix and evaluated with the outer wires fed
        in. The group's params carry `subgraph` (internal node dicts), `output`
        (internal node id returned), `imports` ({external_socket: internal
        node id} so an outer wire feeds an inner node), and `promote`
        ({group_param: [internal_id, internal_param]}) for exposed knobs.
        Groups may nest (prefixes stack)."""
        gp = n.get("params") or {}
        sub = gp.get("subgraph") or []
        if not sub:
            return np.zeros((h, w, 3), np.float32)
        pre = "__grp_%s__" % n["id"]
        internal_ids = {m["id"] for m in sub}

        def remap(ref):
            sid, ssock = _src_ref(ref)
            return (pre + sid + ("." + ssock if ssock != "out" else "")) \
                if sid in internal_ids else ref

        # splice copies with prefixed ids + remapped internal wires
        spliced = {}
        for m in sub:
            mm = json.loads(json.dumps(m))
            mm["id"] = pre + mm["id"]
            wires = {}
            for k, vv in (mm.get("inputs") or {}).items():
                wires[k] = remap(vv)
            mm["inputs"] = wires
            spliced[mm["id"]] = mm
        # promoted params: push group values down into internal nodes
        for pname, target in (gp.get("promote") or {}).items():
            tid, tparam = target
            pid = pre + tid
            if pid in spliced and pname in gp:
                spliced[pid].setdefault("params", {})[tparam] = gp[pname]
        # imports: wire each internal importer to the group's external source
        imports = gp.get("imports") or {}
        for ext_sock, inner_id in imports.items():
            src = (n.get("inputs") or {}).get(ext_sock)
            pid = pre + inner_id
            if src is not None and pid in spliced:
                # the internal node's first input socket receives the outer feed
                itype = spliced[pid]["type"]
                isock = (OPS[itype]["inputs"][0]
                         if itype in OPS and OPS[itype]["inputs"] else "image")
                spliced[pid].setdefault("inputs", {})[isock] = src
        out_id = gp.get("output") or (sub[-1]["id"] if sub else None)
        if out_id is None:
            return np.zeros((h, w, 3), np.float32)
        # temporarily merge, evaluate, restore
        saved = self.nodes
        merged = dict(saved)
        merged.update(spliced)
        self.nodes = merged
        try:
            res = self.evaluate(pre + out_id)
        finally:
            self.nodes = saved
            for k in spliced:                        # don't leak cache entries
                self._cache.pop(k, None)
                self._sigmemo.pop(k, None)
        return res

    def _doc_comp(self):
        """The document composite, computed at most ONCE per evaluation pass.

        A bare Output node needs it twice -- `_sig()` composites the document
        and hashes it to build the cache key, then `_eval_all()` composites it
        again to produce the pixels: two full composites (~630 ms each for four
        layers at 1920x1080) for one render.

        Scoping the memo to a single pass is what makes it safe. Evaluation
        only READS the document, so nothing can mutate between the two uses,
        and the memo is dropped the instant the pass ends. That needs no
        invalidation threaded through the ~20 mutation sites, and cannot serve
        a stale canvas -- the failure mode that kept a global composite cache
        off the table."""
        if self._pass_comp is None:
            self._pass_comp = self.doc.composite()
        return self._pass_comp

    def evaluate(self, nid, sock="out"):
        _CTX_DOC[:] = [self.doc]      # doc-aware ops (Stroke FX) read the live doc
        """Evaluate one node's output socket (memoised). Returns (H, W, 3) float."""
        self._pass_depth += 1          # nested evaluate() stays one pass
        try:
            outs = self._eval_all(nid)
        finally:
            self._pass_depth -= 1
            if self._pass_depth == 0:
                self._pass_comp = None     # never outlives the pass
        return outs.get(sock, outs["out"])

    def _eval_all(self, nid):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("cancelled")
        n = self.nodes[nid]
        sig = self._sig(nid)
        hit = self._cache.get(nid)
        if hit and hit[0] == sig:
            return hit[1]
        _t0 = _time.time()
        h, w = self.doc.height, self.doc.width
        in_alpha = None
        if n.get("mute"):
            # Bypass (Blender's M / Nuke's D): keep the node and its settings
            # wired in place but pass the input straight through, so a step can
            # be A/B'd without unwiring the graph. With nothing wired in -- or
            # on a generator -- it yields empty, which is what "contributes
            # nothing" should look like.
            for sock in (OPS.get(n.get("type"), {}).get("inputs") or []):
                src = (n.get("inputs") or {}).get(sock)
                if src:
                    sid, ssock = _src_ref(src)
                    up = self._eval_all(sid)
                    res = {"out": np.asarray(up.get(ssock, up.get("out")))}
                    self.timings[nid] = _time.time() - _t0
                    self._cache[nid] = (sig, res)
                    return res
            res = {"out": np.zeros((h, w, 3), np.float32)}
            self.timings[nid] = _time.time() - _t0
            self._cache[nid] = (sig, res)
            return res
        if n["type"] == "Layer":
            l = _node_doc(self, n).layer(n.get("params", {}).get("layer", ""))
            px = l.pixels
            if px.shape[0] != h or px.shape[1] != w:
                # render_at() evaluates the graph at a target resolution by
                # changing the document size, but a layer still holds pixels at
                # its own size -- so a Layer node handed canvas-sized data into
                # target-sized maths and everything downstream failed to
                # broadcast. Resample here: procedural nodes gain real detail,
                # pixel sources scale, which is exactly what render_at promises.
                px = _resize(px, h, w)
            a = px[..., 3]
            out = {"out": px.copy(),                       # straight (unpremultiplied) RGBA
                   "alpha": np.stack([a, a, a], -1)}
        elif n["type"] == "Layer group":
            gdoc = _node_doc(self, n)
            ids = set(_group_ids(n, gdoc))
            members = [l for l in gdoc.layers if l.id in ids]
            c = composite(members, gdoc.height, gdoc.width, gdoc.mask_map())
            if c.shape[0] != h or c.shape[1] != w:
                c = _resize(c, h, w)          # same reason as the Layer branch
            a = c[..., 3]
            out = {"out": c.copy(), "alpha": np.stack([a, a, a], -1)}
        elif n["type"] == "Mask":
            mid = (n.get("params") or {}).get("mask", "")
            try:
                mv = _resize(_node_doc(self, n).mask_by_id(mid).data, h, w)
            except KeyError:
                mv = np.ones((h, w), np.float32)
            if (n.get("params") or {}).get("invert"):
                mv = 1.0 - mv
            out = _rgb(mv)
        elif n["type"] in ("Mask out", "Brush out", "Paint out"):
            src = (n.get("inputs") or {}).get("image")
            out = (self.evaluate(*_src_ref(src)) if src is not None
                   else np.zeros((h, w, 3), np.float32))
        elif n["type"] == "Layer out":
            src = (n.get("inputs") or {}).get("image")
            out = (self.evaluate(*_src_ref(src)) if src is not None
                   else np.zeros((h, w, 3), np.float32))
        elif n["type"] == "Media in":
            if self.media is not None:
                out = self.media(nid, n.get("params") or {}, "frame")
                if out is None:
                    out = np.zeros((h, w, 3), np.float32)
            else:
                out = np.zeros((h, w, 3), np.float32)
        elif n["type"] == "Output":
            src = (n.get("inputs") or {}).get("image")
            if src is not None:
                out = self.evaluate(*_src_ref(src))
            else:
                c = self._doc_comp()
                out = c.copy()                             # straight RGBA
        elif n["type"] == "Group":
            out = self._eval_group(n, h, w)
        else:
            meta = OPS[n["type"]]
            ins = {}                               # first image input's alpha rides along
            for sock in meta["inputs"]:
                src = (n.get("inputs") or {}).get(sock)
                if src is None:
                    ins[sock] = None if sock in ("alpha", "matte") \
                        else np.zeros((h, w, 4 if meta.get("rgba") else 3), np.float32)
                    if not meta.get("rgba") and ins[sock] is not None:
                        ins[sock] = ins[sock][..., :3]
                else:
                    sid, ssock = _src_ref(src)
                    v = self.evaluate(sid, ssock)
                    v = _to_rgba(v, h, w)          # values broadcast to constant images
                    if in_alpha is None:
                        in_alpha = v[..., 3]
                    ins[sock] = v if meta.get("rgba") else v[..., :3]
            params = {p["name"]: p["default"] for p in meta["params"]}
            params.update(n.get("params") or {})
            if n["type"] == "time":
                dd = _node_doc(self, n)
                params["_frame"] = float(getattr(dd, "frame", 0.0))
                params["_fps"] = float(getattr(dd, "fps", 24.0))
                params["_range"] = list(getattr(dd, "frame_range",
                                                [0.0, 96.0]))
            # VALUE WIRES: inputs keyed "param:<name>" drive parameters from the
            # graph -- a Value/Color node's number, or any image's mean luminance
            for key, src in (n.get("inputs") or {}).items():
                if not key.startswith("param:"):
                    continue
                v = self.evaluate(*_src_ref(src))
                if isinstance(v, (int, float)):
                    params[key[6:]] = float(v)
                else:
                    a = _rgb(np.asarray(v, np.float32))
                    params[key[6:]] = float(_lum(a).mean())
            out = meta["fn"]((h, w), ins, params)
            # filters that MOVE or SPREAD pixels must move the alpha with them
            # (a blurred element needs blurred transparency, not a crisp crop)
            if (meta.get("alpha") == "process" and in_alpha is not None
                    and not isinstance(out, dict)
                    and in_alpha.min() < 0.999):
                first = meta["inputs"][0]
                a3 = np.stack([in_alpha] * 3, -1)
                ins_a = dict(ins)
                ins_a[first] = a3 if not meta.get("rgba") else _to_rgba(a3, h, w)
                try:
                    a_out = _lum(_rgb(np.asarray(
                        meta["fn"]((h, w), ins_a, params), np.float32)))
                    out = np.concatenate(
                        [_rgb(np.asarray(out, np.float32)),
                         np.clip(_resize(a_out[..., None], h, w), 0, 1)], -1)
                except Exception:
                    pass                            # fall back to passthrough
        if isinstance(out, (int, float)):          # value nodes: numbers flow as-is
            outs = {"out": float(out)}
        elif isinstance(out, dict) and all(
                isinstance(v, (int, float)) for v in out.values()):
            outs = {k: float(v) for k, v in out.items()}
        elif isinstance(out, dict):
            outs = {k: (float(v) if isinstance(v, (int, float))
                        else _conform_rgba(v, h, w, in_alpha)) for k, v in out.items()}
        else:
            outs = {"out": _conform_rgba(out, h, w, in_alpha)}
        self.timings[nid] = _time.time() - _t0
        self._cache[nid] = (sig, outs)
        if self.progress_cb:
            self.progress_cb(nid)
        return outs

    def apply_to_layer(self, nid, name=None, layer_id=None):
        """Bake a node's output into the document: a new layer, or overwrite an
        existing one when layer_id is given (node output *assigned as* that layer)."""
        img = np.asarray(self.evaluate(nid), np.float32)
        if layer_id:
            self.doc.record("Assign node to layer")
            l = self.doc.layer(layer_id)
        else:
            l = self.doc.add_layer(name or f"{self.nodes[nid]['type']} bake")
        l.pixels[..., :3] = img[..., :3]
        l.pixels[..., 3] = img[..., 3] if img.shape[-1] == 4 else 1.0   # bakes keep alpha
        return l


# ------------------------------------------------------------------------------------------------
# Workspace persistence: every document (layers, masks, selections, groups, brushes)
# plus every node graph, in one .lews file (a zip of a JSON manifest + npz arrays).
# ------------------------------------------------------------------------------------------------

def vectorize_svg(img, levels=6, simplify=1.2, max_dim=768):
    """Trace an image into a layered SVG poster (the vectorize door).

    Posterises luminance into `levels` bands, traces each band's boundary with
    marching squares (scikit-image find_contours), simplifies the polygons,
    and stacks them dark-to-light as filled SVG paths -- each band coloured
    with the MEAN colour of its own pixels, so the poster keeps the image's
    palette. Verified against a rasterised round-trip at ~0.96 correlation
    with the posterised source. `simplify` is the polygon tolerance in pixels
    (higher = fewer points, chunkier shapes); resolution is capped at max_dim
    because contour cost scales with pixel count."""
    try:
        from skimage import measure
    except ImportError:
        raise ValueError("SVG export needs scikit-image -- "
                         "pip install scikit-image")
    a = np.asarray(img, np.float32)
    a = _rgb(a)
    h, w = a.shape[:2]
    sc = max(h, w) / float(max_dim)
    if sc > 1.0:
        a = _resize(a, max(int(h / sc), 8), max(int(w / sc), 8))
        h, w = a.shape[:2]
    v = a.mean(-1)
    levels = max(2, min(int(levels), 12))
    qs = np.quantile(v, np.linspace(0, 1, levels + 1))[1:-1]
    bands = np.digitize(v, qs)

    def hexc(c):
        return "#%02x%02x%02x" % tuple(int(round(255 * x))
                                       for x in np.clip(c, 0, 1))
    sel0 = bands == 0
    base = a[sel0].mean(0) if sel0.any() else a.mean((0, 1))
    out = ['<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 %d %d">' % (w, h),
           '<rect width="%d" height="%d" fill="%s"/>' % (w, h, hexc(base))]
    npaths = 0
    for i, t in enumerate(qs):
        mask = (v >= t).astype(float)
        sel = bands == (i + 1)
        col = a[sel].mean(0) if sel.any() else a.mean((0, 1))
        dparts = []
        for cnt in measure.find_contours(mask, 0.5):
            cnt = measure.approximate_polygon(cnt, tolerance=float(simplify))
            if len(cnt) < 3:
                continue
            dparts.append("M " + " L ".join("%.1f,%.1f" % (x, y)
                                            for y, x in cnt) + " Z")
        if dparts:
            out.append('<path d="%s" fill="%s" fill-rule="evenodd"/>'
                       % (" ".join(dparts), hexc(col)))
            npaths += 1
    out.append("</svg>")
    return "\n".join(out)


def shader_from_image(img):
    """Match an image with a PROCEDURAL SHADER (leCore fit_shape).

    Given a picture, leCore fits a procedural fBm to the image's statistical
    signature and emits a GLSL snippet. That snippet is helper functions plus a
    comment -- not runnable -- so we compose it into a complete Shadertoy
    source the Shadertoy node can run and the artist can then tweak.

    Returns {source, quality, baseline, ratio, note, params}. `note` is
    leCore's OWN wording and is deliberately passed through unedited: this is a
    same-family match on roughness and detail, NOT a pixel match or parameter
    recovery. Ratios sit near 1.2x in practice -- useful as a starting point,
    not a tracing of the image."""
    if not have("fit_shape"):
        raise ValueError("this leCore build has no fit_shape faculty -- "
                         "update leos-core to match shaders to images")
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        a = _lum(_rgb(a))
    r = mind().fit_shape(np.asarray(a, float))
    if not isinstance(r, dict):
        raise ValueError("fit_shape returned no fit for this image")
    code = r.get("glsl") or r.get("shadertoy") or ""
    p = r.get("params") or {}
    bw = float(p.get("base_bandwidth", 2.0))
    oct_ = int(p.get("octaves", 5))
    lac = float(p.get("lacunarity", 2.0))
    gain = float(p.get("gain", 0.5))
    if "shadertoy" in r and "mainImage" in code:
        source = code                              # already complete
    else:
        source = (code.rstrip() + "\n\n"
                  "// Matched to your image's roughness + detail. Tweak freely:\n"
                  "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n"
                  "    vec2 uv = fragCoord / iResolution.xy;\n"
                  "    float v = fbm(uv * %.3f + iTime * 0.02, %d, %.3f, %.3f);\n"
                  "    fragColor = vec4(vec3(v), 1.0);\n"
                  "}\n" % (bw, oct_, lac, gain))
    q = float(r.get("quality", 0.0)); b = float(r.get("baseline", 0.0))
    return {"source": source, "quality": q, "baseline": b,
            "ratio": (q / b) if b > 1e-9 else None,
            "note": r.get("note", ""), "kind": r.get("kind", ""), "params": p}


def sdf_to_glsl(dsl):
    """leCore's demoscene door: an SDF DSL expression compiled to a complete
    Shadertoy-style GLSL raymarcher (map + normals + march + light)."""
    from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
    return mind().to_shadertoy(parse_dsl(dsl))


def _doc_section(d, g):
    """(meta, arrays) for one document -- the shared vocabulary of both the
    core container path and the legacy zip path."""
    arrays = {}
    dm = {"id": d.id, "name": d.name, "width": d.width, "height": d.height,
          "dpi": float(getattr(d, "dpi", 72.0)),
          # the studio the picture was painted in: losing these on save meant
          # a reopened painting sat on a different substrate and stopped
          # building past the layer ceiling
          "paper": str(getattr(d, "paper", "canvas")),
          "auto_stratum": bool(getattr(d, "auto_stratum", False)),
          "brush_charge": float(getattr(d, "brush_charge", 1.0)),
          "brush_color": [float(v) for v in getattr(d, "brush_color", (0.0, 0.0, 0.0))],
          "brush_lanes": ([[float(v) for v in row]
                           for row in np.asarray(d.brush_lanes, np.float32)]
                          if getattr(d, "brush_lanes", None) is not None else None),
          "brush_charges": ([float(v) for v in np.asarray(d.brush_charges, np.float32)]
                            if getattr(d, "brush_charges", None) is not None else None),
          # the palette is a surface of its own, so it saves as one. Its
          # pixel arrays go under a "pal_" prefix because BOTH documents
          # number their layers from L1 and the keys would collide.
          "palette_doc": None,
          "groups": [dict(g, layers=list(g["layers"])) for g in d.groups],
          "splines": [p.meta() for p in d.splines],
          # points are stored as a uniform (x, y) list with widths alongside:
          # mixing 2- and 3-long rows makes a ragged array the container
          # cannot serialise, and that silently broke reloading
          "strokes": [{"id": k["id"], "layer": k["layer"],
                       "brush": dict(k["brush"]),
                       "points": [[float(p[0]), float(p[1])] for p in k["points"]],
                       "widths": ([float(p[2]) if len(p) > 2 else 1.0
                                   for p in k["points"]]
                                  if any(len(p) > 2 for p in k["points"]) else []),
                       # rig + keyframes belong to the stroke, or a saved
                       # animation silently vanishes when the file reopens
                       "rig": ({"bones": [float(b) for b in k["rig"]["bones"]],
                                "pins": [int(i) for i in k["rig"]["pins"]],
                                "prev": [[float(v) for v in pp]
                                         for pp in k["rig"]["prev"]],
                                "keys": {str(t): [[float(v) for v in pp]
                                                  for pp in pose]
                                         for t, pose in (k["rig"].get("keys") or {}).items()}}
                               if k.get("rig") else None)}
                      for k in getattr(d, "strokes", [])],
          "layers": [], "masks": [], "selections": [], "brushes": [],
          "stamps": [],
          "graph": g.to_list() if g is not None else []}
    for l in d.layers:
        # the FULL physical sheet, not just identity: the loader has read
        # these keys all along, but the saver never wrote them -- so
        # thickness, volume kind, pose, optics, and backgrounds silently
        # reverted to defaults on every reopen until now.
        dm["layers"].append({"id": l.id, "name": l.name, "visible": l.visible,
                             "opacity": l.opacity, "blend": l.blend,
                             "mask": l.mask, "mask_invert": l.mask_invert,
                             "alpha_lock": bool(getattr(l, "alpha_lock", False)),
                             "clip": bool(getattr(l, "clip", False)),
                             # a palette is not part of the picture, and the
                             # stratum chain is what makes a deep paint body
                             # shade as ONE column instead of stepped slabs
                             "palette": bool(getattr(l, "palette", False)),
                             "stratum_of": getattr(l, "stratum_of", None),
                             "stratum_next": getattr(l, "stratum_next", None),
                             "stratum_root": getattr(l, "stratum_root", None),
                             "thickness": float(getattr(l, "thickness", 1.0)),
                             "vol_kind": getattr(l, "vol_kind", "none"),
                             "vol_ior": float(getattr(l, "vol_ior", 1.33)),
                             "vol_density": float(getattr(l, "vol_density", 0.5)),
                             "absorbency": float(getattr(l, "absorbency", 0.0)),
                             "emissive": float(getattr(l, "emissive", 0.0)),
                             "emissive_color": getattr(l, "emissive_color", None),
                             "reflect": float(getattr(l, "reflect", 0.0)),
                             "dispersion": float(getattr(l, "dispersion", 0.0)),
                             "media_rate": float(getattr(l, "media_rate", 1.0)),
                             "bg": json.loads(json.dumps(getattr(l, "bg", None))),
                             "z_off": float(getattr(l, "z_off", 0.0)),
                             "tilt_x": float(getattr(l, "tilt_x", 0.0)),
                             "tilt_y": float(getattr(l, "tilt_y", 0.0)),
                             "curve": float(getattr(l, "curve", 0.0)),
                             "dome": float(getattr(l, "dome", 0.0)),
                             "field": getattr(l, "field", ""),
                             "field_mode": getattr(l, "field_mode", "attract"),
                             "field_strength": float(getattr(l, "field_strength", 120.0)),
                             "place": json.loads(json.dumps(
                                 getattr(l, "place", None))),
                             "locked": bool(getattr(l, "locked", False)),
                             "relief": float(getattr(l, "relief", 1.0)),
                             # the UNDO snapshot is a separate path from
                             # save/load: without these an undo turned the
                             # palette back into an ordinary layer and broke
                             # its dock, and severed the stratum chain
                             "palette": bool(getattr(l, "palette", False)),
                             "stratum_of": getattr(l, "stratum_of", None),
                             "stratum_next": getattr(l, "stratum_next", None),
                             "stratum_root": getattr(l, "stratum_root", None),
                             # which way is DOWN for this surface's wet
                             # paint -- an easel, a wall, or flat on a table
                             "gravity": (None if getattr(l, "gravity", None) is None
                                         else float(l.gravity)),
                             "gravity_angle": (None if getattr(l, "gravity_angle", None) is None
                                               else float(l.gravity_angle)),
                             "optical": bool(getattr(l, "optical", False)),
                             "media_res": str(getattr(l, "media_res", "normal")),
                             "media_time": str(getattr(l, "media_time", "timeline")),
                             "curve_axis": getattr(l, "curve_axis", "x"),
                             "curve_profile": json.loads(json.dumps(
                                 getattr(l, "curve_profile", None))),
                             "dome_profile": json.loads(json.dumps(
                                 getattr(l, "dome_profile", None)))})
        arrays[f"layer_{l.id}"] = l.pixels
        base = getattr(d, "_replay_base", {}).get(l.id)
        if base is not None:
            # Without the base, a reopened document cannot replay its strokes,
            # so every stroke edit -- nudge, width, transform, delete -- dies
            # after ANY save/load. Before the faithfulness guard existed this
            # failed worse: the edit changed the record, could not rebuild the
            # pixels, and the two silently disagreed forever after.
            arrays[f"replaybase_{l.id}"] = base
            dm["layers"][-1]["has_replay_base"] = True
        if getattr(l, "height_map", None) is not None and l.height_map.any():
            # the paint's BODY: without it a reopened impasto piece comes back
            # flat and every later stroke flows over ridges that are not there
            arrays[f"height_{l.id}"] = l.height_map
            dm["layers"][-1]["has_height"] = True
            hb = getattr(l, "height_below", None)
            if hb is not None and hb.any():
                # what this stratum sits on -- without it the column shades as
                # a thin sheet on a plateau again
                arrays[f"below_{l.id}"] = hb
                dm["layers"][-1]["has_below"] = True
            dm["layers"][-1]["paint_gloss"] = float(getattr(l, "paint_gloss", 0.3))
            dm["layers"][-1]["paint_media"] = getattr(l, "paint_media", None)
        mm = getattr(l, "material_map", None)
        if mm is not None and (mm[..., 2] > 1e-3).any():
            # the paint's STUFF: without it a reopened piece keeps its gold
            # ridges but they shade as plain paint -- the material was the
            # point of the strokes
            arrays[f"material_{l.id}"] = mm
            dm["layers"][-1]["has_material"] = True
        src = getattr(l, "source", None)
        if src is not None:
            # the file's own pixels, so a reopened document can still recover
            # detail a resize would otherwise have lost for good
            dm["layers"][-1]["placed"] = True
            arrays[f"src_{l.id}"] = src
    for m in d.masks:
        # `shape` is what lets a resize re-rasterise exactly instead of
        # resampling; without persisting it a reopened document silently
        # reverts to soft-edged resizes.
        dm["masks"].append({"id": m.id, "name": m.name,
                            "shape": getattr(m, "shape", None)})
        arrays[f"mask_{m.id}"] = m.data
    for x in d.selections:
        dm["selections"].append({"id": x.id, "name": x.name,
                                 "shape": getattr(x, "shape", None)})
        arrays[f"sel_{x.id}"] = x.data
    dm["lights"] = [dict(li) for li in getattr(d, "lights", [])]
    dm["fields"] = [dict(f) for f in getattr(d, "fields", [])]
    dm["walls"] = dict(getattr(d, "walls", {}))
    dm["wall_scale"] = dict(getattr(d, "wall_scale", {}) or {})
    dm["persp"] = json.loads(json.dumps(getattr(d, "persp", {})))
    dm["timeline"] = {"frame": float(getattr(d, "frame", 0.0)),
                      "fps": float(getattr(d, "fps", 24.0)),
                      "range": list(getattr(d, "frame_range", [0.0, 96.0])),
                      "tracks": json.loads(json.dumps(getattr(d, "tracks",
                                                              {})))}
    for s in getattr(d, "stamps", []):
        dm["stamps"].append({"id": s.id, "name": s.name})
        arrays[f"stamp_{s.id}"] = s.pixels
    for b in d.brushes:
        dm["brushes"].append({"id": b.id, "name": b.name, "spacing": b.spacing,
                              "builtin": b.builtin, "follow": b.follow,
                              "j_angle": b.j_angle, "j_size": b.j_size,
                              "j_scatter": b.j_scatter})
        arrays[f"brush_{b.id}"] = b.tip
    # the palette surface saves WITH the picture, under a prefix because both
    # documents number their layers from L1 and the array keys would collide
    _pd = getattr(d, "_palette_doc", None)
    if _pd is not None:
        pdm, parr = _doc_section(_pd, None)
        dm["palette_doc"] = pdm
        for k, v in parr.items():
            arrays["pal_" + k] = v
    return dm, arrays


def save_workspace(docs, graphs, active_id, extras=None):
    """One .lews file via leCore's sectioned container (>= 0.2.2). Every
    document (with its graph) is a section of kind "lestudio.document"; extras
    are foreign sections carried verbatim -- the container never interprets
    kinds it does not know, which is what makes the file forward-compatible."""
    from holographic.io_and_interop.holographic_container import save_container
    sections = []
    for did, d in docs.items():
        dm, arrays = _doc_section(d, graphs.get(did))
        sections.append({"kind": "lestudio.document", "id": did,
                         "meta": dm, "arrays": arrays})
    sections += list(extras or [])
    return save_container(sections,
                          meta={"app": "lestudio", "active": active_id})


def _doc_from_section(dm, arrays):
    import re as _re
    def bump(cls, ident):
        m = _re.search(r"(\d+)$", str(ident))
        if m:
            cls._next = max(cls._next, int(m.group(1)) + 1)
    if True:
        d = Document(dm["width"], dm["height"], dm["name"])
        d.id = dm["id"]; bump(Document, d.id)
        d.layers, d.masks, d.selections, d.brushes = [], [], [], []
        d.lights = [dict(li) for li in dm.get("lights", [])]
        d.fields = [dict(f) for f in dm.get("fields", [])]
        d.walls = dict(dm.get("walls", {"front": None, "back": None,
                                        "left": None, "right": None}))
        if dm.get("wall_scale"):
            d.wall_scale = dict(dm["wall_scale"])
        # (the layers do not exist yet -- the slots are linked to their
        # layers after the layer loop below)
        d._fnext = 1 + max([0] + [int(f["id"][1:]) for f in d.fields
                                  if str(f.get("id", "")).startswith("F")
                                  and str(f["id"])[1:].isdigit()])
        if dm.get("persp"):
            d.persp = json.loads(json.dumps(dm["persp"]))
        tl = dm.get("timeline")
        if tl:
            d.frame = float(tl.get("frame", 0.0))
            d.fps = float(tl.get("fps", 24.0))
            d.frame_range = list(tl.get("range", [0.0, 96.0]))
            d.tracks = json.loads(json.dumps(tl.get("tracks", {})))
        for li in d.lights:
            m2 = _re.search(r"(\d+)$", li["id"])
            if m2:
                d._lnext = max(d._lnext, int(m2.group(1)) + 1)
        d.stamps = []
        for sm in dm.get("stamps", []):
            s = Stamp(sm["name"], arrays[f"stamp_{sm['id']}"])
            s.id = sm["id"]; bump(Stamp, s.id)
            d.stamps.append(s)
        d.groups = [dict(g, layers=list(g["layers"])) for g in dm.get("groups", [])]
        for g in d.groups:
            bump_n = _re.search(r"(\d+)$", g["id"])
            if bump_n:
                d._gnext = max(d._gnext, int(bump_n.group(1)) + 1)
        for lm in dm["layers"]:
            l = Layer(d.height, d.width, lm["name"], arrays[f"layer_{lm['id']}"])
            l.id = lm["id"]; bump(Layer, l.id)
            if lm.get("placed") and f"src_{l.id}" in arrays:
                l.source = arrays[f"src_{l.id}"]
            l.visible, l.opacity, l.blend = lm["visible"], lm["opacity"], lm["blend"]
            l.mask, l.mask_invert = lm.get("mask"), lm.get("mask_invert", False)
            l.alpha_lock = bool(lm.get("alpha_lock", False))
            l.clip = bool(lm.get("clip", False))
            l.thickness = float(lm.get("thickness", 0.0))
            l.vol_kind = lm.get("vol_kind", "none")
            l.vol_ior = float(lm.get("vol_ior", 1.33))
            l.vol_density = float(lm.get("vol_density", 0.5))
            l.absorbency = float(lm.get("absorbency", 0.0))
            l.emissive = float(lm.get("emissive", 0.0))
            l.emissive_color = lm.get("emissive_color")
            l.reflect = float(lm.get("reflect", 0.0))
            l.dispersion = float(lm.get("dispersion", 0.0))
            l.media_rate = float(lm.get("media_rate", 1.0))
            l.bg = json.loads(json.dumps(lm.get("bg"))) if lm.get("bg") \
                else None
            l.place = json.loads(json.dumps(lm.get("place"))) \
                if lm.get("place") else None
            l.locked = bool(lm.get("locked", False))
            l.relief = float(lm.get("relief", 1.0))
            l.palette = bool(lm.get("palette", False))
            l.stratum_of = lm.get("stratum_of")
            l.stratum_next = lm.get("stratum_next")
            l.stratum_root = lm.get("stratum_root")
            l.gravity = (None if lm.get("gravity") is None
                         else float(lm["gravity"]))
            l.gravity_angle = (None if lm.get("gravity_angle") is None
                               else float(lm["gravity_angle"]))
            l.optical = bool(lm.get("optical", False))
            l.media_res = str(lm.get("media_res", "normal"))
            l.media_time = str(lm.get("media_time", "timeline"))
            l.curve_axis = lm.get("curve_axis", "x")
            l.curve_profile = json.loads(json.dumps(
                lm.get("curve_profile"))) if lm.get("curve_profile") else None
            l.dome_profile = json.loads(json.dumps(
                lm.get("dome_profile"))) if lm.get("dome_profile") else None
            l.thickness = max(float(lm.get("thickness", 1.0)), 0.1)
            l.z_off = float(lm.get("z_off", 0.0))
            l.tilt_x = float(lm.get("tilt_x", 0.0))
            l.tilt_y = float(lm.get("tilt_y", 0.0))
            l.curve = float(lm.get("curve", 0.0))
            l.dome = float(lm.get("dome", 0.0))
            l.field = lm.get("field", "")
            l.field_mode = lm.get("field_mode", "attract")
            l.field_strength = float(lm.get("field_strength", 120.0))
            d.layers.append(l)
            if lm.get("has_replay_base") and f"replaybase_{l.id}" in arrays:
                if not hasattr(d, "_replay_base"):
                    d._replay_base = {}
                d._replay_base[l.id] = np.asarray(
                    arrays[f"replaybase_{l.id}"], np.float32)
            if lm.get("has_below") and f"below_{l.id}" in arrays:
                l.height_below = np.asarray(arrays[f"below_{l.id}"], np.float32)
            if lm.get("has_height") and f"height_{l.id}" in arrays:
                l.height_map = np.asarray(arrays[f"height_{l.id}"], np.float32)
                l.paint_gloss = float(lm.get("paint_gloss", 0.3))
                if lm.get("paint_media"):
                    l.paint_media = lm["paint_media"]
            if lm.get("has_material") and f"material_{l.id}" in arrays:
                l.material_map = np.asarray(arrays[f"material_{l.id}"],
                                            np.float32)
        for mm in dm["masks"]:
            m = Mask(d.height, d.width, mm["name"], arrays[f"mask_{mm['id']}"])
            m.id = mm["id"]; bump(Mask, m.id)
            if mm.get("shape"):
                m.shape = dict(mm["shape"])
            d.masks.append(m)
        d.splines = []
        for pm in dm.get("splines", []):
            p = Spline(pm["name"], pm["points"], pm["closed"])
            p.id = pm["id"]; bump(Spline, p.id)
            d.splines.append(p)
        d.dpi = float(dm.get("dpi", 72.0))     # physical scale rides along too
        # the studio comes back with the picture
        d.paper = str(dm.get("paper", "canvas"))
        d.auto_stratum = bool(dm.get("auto_stratum", False))
        d.brush_charge = float(dm.get("brush_charge", 1.0))
        bc = dm.get("brush_color")
        if bc:
            d.brush_color = tuple(float(v) for v in bc)
        bl = dm.get("brush_lanes")
        if bl:
            d.brush_lanes = np.asarray(bl, np.float32)
        bch = dm.get("brush_charges")
        if bch:
            d.brush_charges = np.asarray(bch, np.float32)
        pdm = dm.get("palette_doc")
        if pdm:
            sub = {k[4:]: v for k, v in arrays.items() if k.startswith("pal_")}
            pd, _pg = _doc_from_section(pdm, sub)
            pd.name = "palette"
            pd.auto_stratum = False
            pd._brush_host = d           # it borrows THIS picture's brush
            d._palette_doc = pd
        # remembered brush strokes ride along with the document
        d.strokes = []
        for k in dm.get("strokes", []):
            ws = list(k.get("widths") or [])
            pts = []
            for i, p in enumerate(k["points"]):
                row = [float(p[0]), float(p[1])]
                if i < len(ws):
                    row.append(float(ws[i]))
                pts.append(row)
            d.strokes.append({"id": k["id"], "layer": k["layer"],
                              "points": pts,
                              "brush": dict(k.get("brush") or {}),
                              **({"rig": {
                                  "bones": [float(b) for b in k["rig"]["bones"]],
                                  "pins": [int(i) for i in k["rig"]["pins"]],
                                  "prev": [[float(v) for v in pp]
                                           for pp in k["rig"]["prev"]],
                                  "keys": {str(t): [[float(v) for v in pp]
                                                    for pp in pose]
                                           for t, pose in (k["rig"].get("keys") or {}).items()}}}
                                 if k.get("rig") else {})})
        d._stroke_n = len(d.strokes)
        for sm in dm["selections"]:
            x = Selection(d.height, d.width, sm["name"], arrays[f"sel_{sm['id']}"])
            x.id = sm["id"]; bump(Selection, x.id)
            if sm.get("shape"):
                x.shape = dict(sm["shape"])
            d.selections.append(x)
        for bm in dm["brushes"]:
            b = Brush(bm["name"], arrays[f"brush_{bm['id']}"], bm["spacing"], bm["builtin"])
            b.id = bm["id"]; bump(Brush, b.id)
            b.follow, b.j_angle = bm.get("follow", False), bm.get("j_angle", 0.0)
            b.j_size, b.j_scatter = bm.get("j_size", 0.0), bm.get("j_scatter", 0.0)
            d.brushes.append(b)
        g = NodeGraph(d)
        g.set_graph(dm.get("graph", []))
        # link wall slots to their layers. Must run AFTER the layers
        # exist -- placed earlier (chasing anchors that turned out to
        # precede the layer loop) every lookup missed and each slot
        # quietly emptied itself, which read exactly like "walls do not
        # persist".
        for _s in list(getattr(d, "walls", {})):
            _lid = d.walls.get(_s)
            if not _lid:
                continue
            try:
                d.layer(_lid).wall = _s
            except KeyError:
                d.walls[_s] = None           # the layer is gone
        return d, g


def load_workspace(data):
    """bytes -> (docs, graphs, active_id, extras). Reads the leCore 0.2.2 core
    container first; legacy app-local zips still open. Foreign sections come
    back verbatim in both paths."""
    try:
        from holographic.io_and_interop.holographic_container import load_container
        loaded = load_container(data)
        sections, meta = loaded["sections"], loaded.get("meta") or {}
        assert meta.get("app") == "lestudio"
        docs, graphs, extras = {}, {}, []
        for sec in sections:
            if sec.get("kind") == "lestudio.document":
                d, g = _doc_from_section(sec["meta"], sec["arrays"])
                docs[d.id], graphs[d.id] = d, g
            else:
                extras.append(sec)
        assert docs
        active = meta.get("active")
        return docs, graphs, active if active in docs else next(iter(docs)), extras
    except Exception:
        pass
    return _load_workspace_legacy(data)


def _load_workspace_legacy(data):
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(data))
    manifest = json.loads(z.read("manifest.json"))
    docs, graphs = {}, {}
    for dm in manifest["docs"]:
        arrays = np.load(io.BytesIO(z.read(f"{dm['id']}.npz")))
        d, g = _doc_from_section(dm, arrays)
        docs[d.id], graphs[d.id] = d, g
    extras = []
    for sm in manifest.get("sections", []):
        sec = {"kind": sm.get("kind", "unknown"), "id": sm.get("id", ""),
               "meta": sm.get("meta", {}), "arrays": {}}
        fname = f"section_{sec['kind']}_{sec['id']}.npz"
        if fname in z.namelist():
            arrs = np.load(io.BytesIO(z.read(fname)))
            sec["arrays"] = {k: arrs[k] for k in arrs.files}
        extras.append(sec)
    return docs, graphs, manifest.get("active") or next(iter(docs), None), extras
