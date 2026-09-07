"""Wing plate + ruin-tower plate, per Vex's amendments.
Wing: raked from shoulder (0.82,0.30) exiting frame top near (0.65,0),
membrane translucent amber (0.55,0.30,0.10) where the moon backlights,
bones near-black, moon's carved left limb untouched.
Tower: flat mist-tone silhouette (0.20,0.26,0.23) at (0.50,0.45)."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

W, H = 1280, 800
SS = 2
rng = np.random.RandomState(21)


def bezq(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    P = np.asarray([p0, p1, p2], float)
    return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]


# ---------------- WING ---------------------------------------------------
def wing():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    root = (1055, 268)
    wrist = (902, 95)
    tips = [(760, -12), (846, -30), (958, -8), (1078, 58)]
    # membrane silhouette: root -> along f1 -> scalloped across tips -> back
    mem = [root]
    mem += [tuple(q) for q in bezq(root, (940, 170), tips[0], 14)]
    for i in range(len(tips) - 1):
        t1, t2 = tips[i], tips[i + 1]
        sag = ((t1[0] + t2[0]) / 2 + 8, (t1[1] + t2[1]) / 2 + 74)
        mem += [t1, sag, t2]
    mem += [tuple(q) for q in bezq(tips[-1], (1120, 130), root, 12)]
    d.polygon([(x * SS, y * SS) for x, y in mem], fill=(120, 66, 24, 210))
    arr = np.asarray(im, np.float64) / 255.0

    # backlight gradient: brighter translucent amber toward the moon glow
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(float) / SS
    glow = np.exp(-(((xx - 810) / 250) ** 2 + ((yy - 80) / 150) ** 2))
    lit = np.array([0.50, 0.27, 0.10])
    dkm = np.array([0.10, 0.06, 0.05])
    g = np.clip(glow * 1.25, 0, 1)[..., None]
    arr[..., :3] = dkm[None, None] + (lit - dkm)[None, None] * g
    # mottled membrane (skin patches like the Vallejo ref)
    n1 = ndi.gaussian_filter(rng.random((H * SS, W * SS)), 26)
    arr[..., :3] *= (0.82 + 0.5 * n1[..., None])
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # finger bones: near-black tapering lines root/wrist->tips + humerus
    def bone(a, b, w0, w1):
        pts = bezq(a, ((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2), b, 20)
        for j in range(len(pts) - 1):
            wd = w0 + (w1 - w0) * j / len(pts)
            d.line([tuple(pts[j] * SS), tuple(pts[j + 1] * SS)],
                   fill=(14, 10, 8, 255), width=max(1, int(wd * SS)))
    bone(root, wrist, 12, 8)
    for tp, w0 in zip(tips, (8, 7, 6.5, 6)):
        bone(wrist, tp, w0, 2.5)
    # membrane veins from wrist fanning
    for tp in tips:
        for k in range(3):
            t0 = rng.uniform(0.3, 0.6)
            bx = wrist[0] + (tp[0] - wrist[0]) * t0
            by = wrist[1] + (tp[1] - wrist[1]) * t0
            d.line([bx * SS, by * SS, (bx + rng.uniform(-20, 30)) * SS,
                    (by + rng.uniform(26, 60)) * SS],
                   fill=(30, 16, 10, 130), width=SS)
    # thumb claw at the wrist
    d.polygon([(wrist[0] * SS, (wrist[1] - 4) * SS),
               ((wrist[0] - 26) * SS, (wrist[1] - 26) * SS),
               ((wrist[0] + 6) * SS, (wrist[1] + 6) * SS)],
              fill=(16, 12, 10, 255))
    # amber rim on the leading edge (root->wrist->f1) facing the moon
    lead = list(bezq(root, (945, 190), (902, 95), 16)) + \
        list(bezq((902, 95), (825, 30), (760, -12), 12))
    for j in range(len(lead) - 1):
        d.line([tuple(np.asarray(lead[j]) * SS),
                tuple(np.asarray(lead[j + 1]) * SS)],
               fill=(250, 190, 90, 150), width=int(2.2 * SS))
    out = im.filter(ImageFilter.GaussianBlur(0.7 * SS)).resize(
        (W, H), Image.LANCZOS)
    out.save("/root/work/r33/wing_plate.png")
    print("wing plate done")


# ---------------- TOWER --------------------------------------------------
def tower():
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = (51, 66, 59, 150)          # (0.20,0.26,0.23) mist tone, translucent
    cx = 0.505 * W
    top = 0.30 * H
    base = 0.68 * H
    w0 = 0.028 * W
    # tapering shaft with a slight lean
    pts = []
    for t in np.linspace(0, 1, 12):
        y = top + (base - top) * t
        wshaft = w0 * (0.72 + 0.55 * t)
        lean = 8 * (1 - t)
        pts.append((cx - wshaft + lean, y))
    for t in np.linspace(1, 0, 12):
        y = top + (base - top) * t
        wshaft = w0 * (0.72 + 0.55 * t)
        lean = 8 * (1 - t)
        pts.append((cx + wshaft + lean, y))
    d.polygon([(x * SS, y * SS) for x, y in pts], fill=col)
    # broken crown: jagged merlons
    for k in range(5):
        x = cx - w0 * 0.75 + k * w0 * 0.38 + 8
        h2 = rng.uniform(8, 26)
        d.polygon([(x * SS, top * SS), ((x + 8) * SS, (top - h2) * SS),
                   ((x + 15) * SS, top * SS)], fill=col)
    # fallen arch attached at the base, leaning right
    arch = bezq((cx + w0 * 1.2, base), (cx + 0.075 * W, 0.47 * H),
                (cx + 0.135 * W, 0.60 * H), 24)
    for j in range(len(arch) - 1):
        d.line([tuple(arch[j] * SS), tuple(arch[j + 1] * SS)], fill=col,
               width=int(11 * SS))
    # one dim window slit
    d.rectangle([(cx - 6) * SS, (0.40 * H) * SS, (cx + 2) * SS,
                 (0.435 * H) * SS], fill=(28, 36, 32, 150))
    im = im.filter(ImageFilter.GaussianBlur(2.0 * SS))
    # fade the base into the fog: alpha ramp
    a = np.asarray(im, np.float64)
    yy = np.mgrid[0:H * SS, 0:W * SS][0] / SS
    fade = np.clip((0.66 * H - yy) / (0.10 * H) + 1.0, 0, 1)
    topfade = np.clip((yy - 0.24 * H) / (0.06 * H), 0, 1)
    a[..., 3] *= fade * topfade
    im = Image.fromarray(a.astype(np.uint8)).resize((W, H), Image.LANCZOS)
    im.save("/root/work/r33/tower_plate.png")
    print("tower plate done")


wing()
