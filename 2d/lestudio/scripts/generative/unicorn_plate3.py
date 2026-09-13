"""Unicorn plate v3 -- approved silhouette (skel_test v4) + full shading,
features, mane, tail. Rearing levade facing right."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

SS = 4
PW, PH = 560, 640


def bez(pts, n=40):
    pts = np.asarray(pts, float)
    t = np.linspace(0, 1, n)[:, None]
    if len(pts) == 3:
        return ((1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1]
                + t ** 2 * pts[2])
    return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1]
            + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])


def chain(d, pts, widths):
    pts = np.asarray(pts, float) * SS
    widths = np.asarray(widths, float) * SS
    for i in range(len(pts) - 1):
        for t in np.linspace(0, 1, 16):
            p = pts[i] * (1 - t) + pts[i + 1] * t
            w = widths[i] * (1 - t) + widths[i + 1] * t
            d.ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=255)


def build_mask():
    m = Image.new("L", (PW * SS, PH * SS), 0)
    d = ImageDraw.Draw(m)
    spine = bez([(222, 418), (268, 352), (338, 292)], 24)
    chain(d, spine, np.linspace(58, 44, len(spine)))
    chain(d, [(214, 426), (224, 418)], [60, 58])
    chain(d, [(342, 294), (352, 282)], [42, 38])
    neck = bez([(344, 284), (362, 240), (384, 196)], 16)
    chain(d, neck, np.linspace(40, 20, len(neck)))
    head = [(384, 166), (400, 160), (412, 166),
            (452, 206), (460, 215), (458, 226), (448, 232), (434, 234),
            (412, 226), (394, 222), (380, 208), (374, 190)]
    d.polygon([(x * SS, y * SS) for x, y in head], fill=255)
    d.polygon([(402 * SS, 162 * SS), (400 * SS, 132 * SS),
               (388 * SS, 160 * SS)], fill=255)
    d.polygon([(388 * SS, 164 * SS), (378 * SS, 138 * SS),
               (372 * SS, 164 * SS)], fill=255)
    chain(d, [(346, 308), (366, 352), (388, 382), (380, 408), (370, 422)],
          [20, 14, 9, 6.5, 7])
    chain(d, [(330, 318), (344, 362), (360, 394), (350, 414)],
          [18, 12.5, 8, 6.5])
    chain(d, [(234, 434), (276, 486), (264, 544), (270, 592), (274, 610)],
          [32, 20, 11, 7.5, 8.5])
    chain(d, [(208, 440), (230, 494), (214, 550), (221, 598), (217, 614)],
          [28, 17, 10, 7, 8])
    a = np.asarray(m, float) / 255.0
    a = ndi.gaussian_filter(a, 2.0 * SS)
    return np.clip((a - 0.45) * 8, 0, 1)


def build():
    W2, H2 = PW * SS, PH * SS
    mask = build_mask()
    sdf = ndi.distance_transform_edt(mask > 0.5)
    sdfs = ndi.gaussian_filter(sdf, 9.0)
    gy, gx = np.gradient(sdfs)
    nz = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    nx, ny = gx / nz, gy / nz
    lam = np.clip(nx * 0.55 + ny * 0.83, -1, 1)
    lit = np.clip(0.5 + 0.5 * lam, 0, 1) ** 1.15
    depth = np.clip(sdfs / 55.0, 0, 1)
    base_lo = np.array([0.40, 0.48, 0.66])
    base_hi = np.array([0.90, 0.92, 0.99])
    col = base_lo[None, None] + (base_hi - base_lo)[None, None] \
        * (0.3 * depth + 0.7 * lit)[..., None]
    edge = np.clip(1 - sdf / (6.5 * SS), 0, 1)
    er = np.clip(-nx, 0, 1) ** 2 * edge
    col = col * (1 - er[..., None] * 0.8) + np.array([0.95, 0.55, 0.30])[
        None, None] * (er * 0.8)[..., None]
    mr = np.clip(nx * 0.6 - ny * 0.75, 0, 1) ** 2 * edge
    col = col * (1 - mr[..., None] * 0.75) + np.array([0.98, 0.99, 1.0])[
        None, None] * (mr * 0.75)[..., None]
    rgba = np.concatenate([col, mask[..., None]], -1)
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))

    dr = ImageDraw.Draw(im)

    def line(pts, w, fill):
        dr.line([tuple(q) for q in np.asarray(pts) * SS], fill=fill,
                width=max(1, int(w * SS)), joint="curve")

    # eye: almond high on the skull, glint
    ex, ey = 414, 184
    dr.ellipse([(ex - 5) * SS, (ey - 3.5) * SS, (ex + 5) * SS,
                (ey + 3.5) * SS], fill=(22, 24, 38, 255))
    dr.ellipse([(ex + 1) * SS, (ey - 2.5) * SS, (ex + 3.5) * SS,
                ey * SS], fill=(240, 244, 255, 255))
    # nostril + mouth
    dr.ellipse([449 * SS, 214 * SS, 455 * SS, 220 * SS],
               fill=(40, 40, 60, 220))
    line(bez([(456, 224), (446, 229), (436, 230)], 10), 1.1,
         (36, 40, 60, 200))
    # cheek round
    line(bez([(414, 196), (406, 210), (394, 216)], 12), 1.1,
         (70, 76, 110, 140))
    # muscle contours
    line(bez([(300, 300), (320, 332), (330, 366)], 14), 1.6,
         (96, 106, 148, 85))
    line(bez([(250, 418), (272, 450), (280, 478)], 12), 1.6,
         (96, 106, 148, 85))
    line(bez([(244, 330), (256, 368), (268, 396)], 12), 1.6,
         (216, 224, 250, 95))
    line(bez([(348, 236), (356, 266), (352, 296)], 12), 1.8,
         (80, 88, 130, 75))
    # belly core shadow
    line(bez([(282, 452), (306, 428), (326, 396)], 14), 3.0,
         (86, 96, 138, 60))
    # HORN from the forehead, up-right
    hx0, hy0, hx1, hy1 = 408, 158, 448, 112
    for i in range(10):
        t0, t1 = i / 10.0, (i + 1) / 10.0
        w = 5.0 * (1 - t0) + 0.7
        c = (250, 250, 255, 255) if i % 2 == 0 else (214, 206, 240, 255)
        dr.line([(hx0 + (hx1 - hx0) * t0) * SS,
                 (hy0 + (hy1 - hy0) * t0) * SS,
                 (hx0 + (hx1 - hx0) * t1) * SS,
                 (hy0 + (hy1 - hy0) * t1) * SS], fill=c, width=int(w * SS))
    for i in range(5):
        t = 0.12 + i * 0.18
        x, y = hx0 + (hx1 - hx0) * t, hy0 + (hy1 - hy0) * t
        w = 5.0 * (1 - t) + 0.7
        dr.line([(x - w * 0.7) * SS, (y + w * 0.55) * SS,
                 (x + w * 0.7) * SS, (y - w * 0.55) * SS],
                fill=(150, 150, 190, 210), width=SS)
    for ddx, ddy, ln in ((1, 0, 7), (0, 1, 7), (0.7, 0.7, 4.4),
                         (0.7, -0.7, 4.4)):
        dr.line([(hx1 - ddx * ln) * SS, (hy1 - ddy * ln) * SS,
                 (hx1 + ddx * ln) * SS, (hy1 + ddy * ln) * SS],
                fill=(255, 255, 255, 255), width=SS)
    # hooves
    for hx, hy in ((274, 610), (217, 614), (370, 422), (350, 414)):
        dr.ellipse([(hx - 8) * SS, (hy - 5) * SS, (hx + 8) * SS,
                    (hy + 6) * SS], fill=(118, 126, 165, 235))

    arr = np.asarray(im, np.float64) / 255.0

    # MANE off the crest, streaming down-left
    rng = np.random.RandomState(5)
    mane_im = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    md = ImageDraw.Draw(mane_im)
    crest = bez([(382, 190), (356, 236), (342, 286)], 20)
    for i, (cx, cy) in enumerate(crest):
        if i % 2:
            continue
        for k in range(3):
            ln = rng.uniform(55, 115)
            pts = bez([(cx, cy),
                       (cx - ln * 0.5, cy + rng.uniform(-6, 20)),
                       (cx - ln * 0.55 - rng.uniform(0, 30),
                        cy + rng.uniform(28, 64))], 16)
            t = rng.uniform(0.25, 0.95)
            c = tuple(int(v * 255) for v in (
                0.70 + 0.22 * t, 0.64 + 0.22 * t, 0.88 + 0.10 * t)) + (205,)
            for j in range(len(pts) - 1):
                w = max(1, int((3.4 * (1 - j / len(pts)) + 0.4) * SS))
                md.line([tuple(pts[j] * SS), tuple(pts[j + 1] * SS)],
                        fill=c, width=w)
    for k in range(4):
        pts = bez([(396, 158), (380 - k * 5, 172), (368 - k * 7, 192)], 12)
        for j in range(len(pts) - 1):
            md.line([tuple(pts[j] * SS), tuple(pts[j + 1] * SS)],
                    fill=(205, 200, 238, 200), width=max(1, int(2.0 * SS)))
    mane = np.asarray(mane_im.filter(
        ImageFilter.GaussianBlur(0.6 * SS)), np.float64) / 255.0

    tail_im = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    td = ImageDraw.Draw(tail_im)
    for k in range(14):
        x0, y0 = 196 + rng.uniform(-6, 6), 428 + rng.uniform(-8, 8)
        pts = bez([(x0, y0),
                   (x0 - rng.uniform(26, 50), y0 + rng.uniform(60, 95)),
                   (x0 - rng.uniform(16, 56), y0 + rng.uniform(130, 180))],
                  18)
        t = rng.uniform(0.3, 0.95)
        c = tuple(int(v * 255) for v in (
            0.58 + 0.26 * t, 0.56 + 0.26 * t, 0.82 + 0.13 * t)) + (195,)
        for j in range(len(pts) - 1):
            w = max(1, int(3.0 * (1 - 0.5 * j / len(pts)) * SS))
            td.line([tuple(pts[j] * SS), tuple(pts[j + 1] * SS)],
                    fill=c, width=w)
    tail = np.asarray(tail_im.filter(
        ImageFilter.GaussianBlur(0.6 * SS)), np.float64) / 255.0

    for layer in (tail, mane):
        la = layer[..., 3:4]
        arr = arr * (1 - la) + layer * la
        arr[..., 3] = np.maximum(arr[..., 3], layer[..., 3])

    out = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    out = out.resize((PW, PH), Image.LANCZOS)
    out.save("/root/work/r33/unicorn_plate.png")
    print("v3 plate saved")


build()
