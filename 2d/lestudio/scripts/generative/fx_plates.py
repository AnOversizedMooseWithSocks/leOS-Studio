"""Raster FX plates: painted fire cone, moon disc, gnarled branches.
Built in numpy/PIL where shape control is exact, pasted as layers."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

W, H = 1280, 800
rng = np.random.RandomState(33)


def save_rgba(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------------- FIRE ---------------------------------------------------
# A cone of flame from the maw (855, 212) sweeping down-left toward
# (505, 330): distance field around a curved spine, noise-warped edges,
# core/mid/edge color ramp -- Norem fire, not gaussian smear.
def flame():
    SS = 2
    w2, h2 = W * SS, H * SS
    yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
    # spine: quadratic from maw to target with sag
    p0 = np.array([858, 208]) * SS
    p1 = np.array([700, 285]) * SS      # control (sag down)
    p2 = np.array([515, 318]) * SS
    ts = np.linspace(0, 1, 60)
    spine = np.array([(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
                      for t in ts])
    # distance to spine + width profile (narrow at maw, bellies, tapers)
    dmin = np.full((h2, w2), 1e9, np.float32)
    tpar = np.zeros((h2, w2), np.float32)
    for i, (sx, sy) in enumerate(spine):
        d = (xx - sx) ** 2 + (yy - sy) ** 2
        m = d < dmin
        dmin[m] = d[m]
        tpar[m] = ts[i]
    dmin = np.sqrt(dmin) / SS
    width = 14 + 90 * np.maximum(np.sin(np.clip(tpar, 0, 1) * math.pi), 0.0) ** 0.9 * (1 - 0.25 * tpar)
    # noise warp: licking tongues at the boundary
    n1 = ndi.gaussian_filter(rng.random((h2, w2)).astype(np.float32), 18)
    n2 = ndi.gaussian_filter(rng.random((h2, w2)).astype(np.float32), 6)
    n = (n1 - 0.5) * 3.2 + (n2 - 0.5) * 1.8
    field = dmin / np.maximum(width * (1.0 + n * 0.9), 1e-3)
    # tongue streaks: elongate along the flow direction using anisotropic blur
    body = np.clip(1.0 - field, 0, 1)
    body = ndi.gaussian_filter(body, (1.5, 9))
    body = np.clip(body * 1.25, 0, 1) ** 1.2
    # cut past the end
    body *= np.clip(1.4 - tpar * 1.32, 0, 1) ** 0.6
    # color ramp
    core = np.array([1.0, 0.97, 0.72])
    mid = np.array([1.0, 0.62, 0.14])
    edge = np.array([0.72, 0.22, 0.04])
    t = body[..., None]
    col = edge + (mid - edge) * np.clip(t * 1.6 - 0.15, 0, 1)
    col = col + (core - mid) * np.clip(t * 2.2 - 1.15, 0, 1)
    alpha = np.clip(body * 1.8, 0, 1) ** 1.1
    # inner maw hotspot
    hot = np.exp(-(((xx - p0[0]) / (46 * SS)) ** 2
                   + ((yy - p0[1]) / (30 * SS)) ** 2))
    col = col * (1 - hot[..., None]) + np.array([1.0, 0.99, 0.85])[
        None, None] * hot[..., None]
    alpha = np.maximum(alpha, hot * 0.95)
    out = np.concatenate([np.clip(col, 0, 1), alpha[..., None]], -1)
    out = np.asarray(Image.fromarray(
        (out * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
        np.float32) / 255.0
    # embers: sparks drifting off the flame
    im = Image.fromarray((out * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for i in range(90):
        t = rng.uniform(0.15, 1.0)
        sx = (1 - t) ** 2 * 858 + 2 * (1 - t) * t * 700 + t ** 2 * 515
        sy = (1 - t) ** 2 * 208 + 2 * (1 - t) * t * 285 + t ** 2 * 318
        sx += rng.uniform(-30, 30)
        sy += rng.uniform(-60, 20) - 40 * t
        r = rng.uniform(0.7, 2.4)
        heat = rng.uniform(0.3, 1.0)
        c = (255, int(150 + 90 * heat), int(40 + 60 * heat),
             int(120 + 120 * heat))
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=c)
    im.save("/root/work/r33/fire_plate.png")
    print("fire plate done")


# ---------------- MOON ---------------------------------------------------
def moon():
    mx, my, mr = 0.62 * W, 0.14 * H, 0.14 * W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - mx) ** 2 + ((yy - my) * 1.08) ** 2)
    disc = np.clip(1.0 - (d - mr) / (mr * 0.08), 0, 1)
    halo = np.exp(-np.clip(d - mr, 0, None) / (mr * 0.85))
    # mottled surface
    n = ndi.gaussian_filter(rng.random((H, W)).astype(np.float32), 9)
    surf = 1.0 - (n - 0.5) * 0.35
    core = np.array([0.99, 0.78, 0.34])
    hot = np.array([1.0, 0.90, 0.60])
    halo_c = np.array([0.55, 0.35, 0.12])
    col = halo_c[None, None] * np.ones((H, W, 1))
    col = col * (1 - disc[..., None]) + (
        core[None, None] * surf[..., None]
        + (hot - core)[None, None] * np.clip(1 - d / (mr * 0.5), 0, 1)[..., None]) * disc[..., None]
    alpha = np.clip(disc * 0.95 + halo * 0.55, 0, 1)
    save_rgba(np.concatenate([np.clip(col, 0, 1), alpha[..., None]], -1),
              "/root/work/r33/moon_plate.png")
    print("moon plate done")


# ---------------- BRANCHES -----------------------------------------------
def branches():
    SS = 2
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (8, 10, 8, 255)

    def grow(x, y, ang, ln, wd, depth):
        if depth <= 0 or ln < 6 or wd < 0.5:
            return
        # gnarled: each segment kinks
        steps = max(2, int(ln / 14))
        cx, cy = x, y
        ca = ang
        for s in range(steps):
            ca += rng.uniform(-0.55, 0.55)
            nx2 = cx + math.cos(ca) * ln / steps
            ny2 = cy + math.sin(ca) * ln / steps
            d.line([cx * SS, cy * SS, nx2 * SS, ny2 * SS], fill=ink,
                   width=max(1, int(wd * SS)))
            # twig chance
            if rng.random() < 0.5:
                ta = ca + rng.uniform(0.5, 1.3) * (1 if rng.random() < 0.5
                                                   else -1)
                grow(nx2, ny2, ta, ln * rng.uniform(0.3, 0.5),
                     wd * 0.45, depth - 1)
            cx, cy = nx2, ny2
        # continue main limb split
        grow(cx, cy, ca + rng.uniform(-0.4, 0.4),
             ln * rng.uniform(0.6, 0.8), wd * 0.68, depth - 1)
        if rng.random() < 0.7:
            grow(cx, cy, ca + rng.uniform(0.6, 1.2) * (1 if rng.random()
                                                       < 0.5 else -1),
                 ln * rng.uniform(0.45, 0.65), wd * 0.55, depth - 1)

    # limbs clawing down from the top edge, x 0.3-0.85
    for fx, ang, ln, wd in ((0.32, 1.35, 210, 13), (0.45, 1.15, 260, 15),
                            (0.58, 1.45, 240, 14), (0.70, 1.25, 220, 12),
                            (0.82, 1.5, 180, 11)):
        grow(fx * W, -12, ang + rng.uniform(-0.2, 0.2), ln, wd, 5)
    # two limbs crossing the moon disc (from the canopy left of it)
    grow(0.47 * W, 0.02 * H, 0.55, 300, 10, 5)
    grow(0.52 * W, -6, 0.9, 260, 9, 5)
    im = im.filter(ImageFilter.GaussianBlur(0.5 * SS))
    im = im.resize((W, H), Image.LANCZOS)
    im.save("/root/work/r33/branch_plate.png")
    print("branch plate done")


flame()
