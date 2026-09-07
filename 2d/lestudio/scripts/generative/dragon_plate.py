"""Dragon plate -- capsule/polygon construction, facing left, jaw open,
wing sail up-right. Firelit from its own mouth (point-light shading),
faint cool moon from upper-left. RGBA, 760x640."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

SS = 3
PW, PH = 760, 640


def bez(pts, n=40):
    pts = np.asarray(pts, float)
    t = np.linspace(0, 1, n)[:, None]
    if len(pts) == 3:
        return ((1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1]
                + t ** 2 * pts[2])
    return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1]
            + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])


def chain(d, pts, widths, fill=255):
    pts = np.asarray(pts, float) * SS
    widths = np.asarray(widths, float) * SS
    for i in range(len(pts) - 1):
        for t in np.linspace(0, 1, 16):
            p = pts[i] * (1 - t) + pts[i + 1] * t
            w = widths[i] * (1 - t) + widths[i + 1] * t
            d.ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=fill)


def P(pts):
    return [(x * SS, y * SS) for x, y in pts]


def build_mask():
    m = Image.new("L", (PW * SS, PH * SS), 0)
    d = ImageDraw.Draw(m)

    # NECK: S curve from behind the skull to the shoulder
    neck = bez([(205, 190), (300, 200), (350, 290), (425, 355)], 30)
    chain(d, neck, np.linspace(34, 62, len(neck)))
    # BODY: shoulder -> haunch (off toward lower right)
    body = bez([(430, 360), (540, 400), (640, 470)], 20)
    chain(d, body, np.linspace(80, 96, len(body)))
    chain(d, [(650, 480), (700, 520)], [95, 90])
    # TAIL: off right edge
    chain(d, [(700, 520), (760, 560)], [60, 40])

    # SKULL: horned wedge, jaw OPEN, facing left
    upper = [(206, 150), (170, 138), (120, 148), (76, 168), (60, 180),
             (70, 190), (110, 196), (160, 198), (205, 194)]
    d.polygon(P(upper), fill=255)
    lower = [(196, 200), (150, 206), (98, 216), (72, 228), (84, 238),
             (130, 236), (176, 224), (204, 214)]
    d.polygon(P(lower), fill=255)
    # cheek mass joining jaw to neck
    chain(d, [(200, 185), (215, 190)], [34, 36])
    # BROW HORNS swept back
    d.polygon(P([(198, 152), (258, 108), (272, 116), (216, 168)]), fill=255)
    d.polygon(P([(212, 160), (262, 132), (272, 140), (224, 176)]), fill=255)
    # jaw spikes
    d.polygon(P([(196, 222), (216, 248), (188, 234)]), fill=255)
    d.polygon(P([(176, 230), (188, 252), (164, 238)]), fill=255)

    # WING: humerus + fingers + membranes
    chain(d, [(450, 330), (520, 220), (565, 165)], [26, 18, 13])
    fingers = [((565, 165), (668, 34), 9),
               ((565, 165), (716, 96), 8),
               ((565, 165), (744, 196), 7),
               ((565, 165), (740, 300), 6)]
    for a, b, w in fingers:
        chain(d, [a, b], [w, max(2.5, w - 4)])
    # membranes: polygons between adjacent finger tips, scalloped
    tiplist = [(668, 34), (716, 96), (744, 196), (740, 300)]
    root = (565, 165)
    anchor = (470, 350)
    for i in range(len(tiplist) - 1):
        t1, t2 = tiplist[i], tiplist[i + 1]
        mid = ((t1[0] + t2[0]) / 2 - 18, (t1[1] + t2[1]) / 2 - 10)
        d.polygon(P([root, t1, mid, t2]), fill=255)
    # membrane from last finger to the body
    d.polygon(P([root, (740, 300), (640, 420), anchor]), fill=255)
    # membrane between first finger and neck-side
    d.polygon(P([root, (668, 34), (600, 120), (520, 220)]), fill=255)

    # FORELEG: down into the fog with talons
    chain(d, [(430, 400), (398, 470), (410, 540), (396, 586)],
          [30, 18, 13, 10])
    for dx in (-14, 2, 16):
        d.polygon(P([(396 + dx - 6, 580), (396 + dx, 620),
                     (396 + dx + 8, 582)]), fill=255)

    # DORSAL SPIKES along neck crest
    ncrest = bez([(210, 158), (300, 168), (352, 260), (430, 330)], 16)
    for i in range(1, len(ncrest) - 1, 2):
        x, y = ncrest[i]
        dxn = ncrest[i + 1] - ncrest[i - 1]
        nrm = np.array([dxn[1], -dxn[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-6)
        h = 26 - i
        tip = (x + nrm[0] * h, y + nrm[1] * h)
        d.polygon(P([(x - 9, y), tip, (x + 9, y)]), fill=255)

    a = np.asarray(m, float) / 255.0
    a = ndi.gaussian_filter(a, 1.2 * SS)
    return np.clip((a - 0.45) * 8, 0, 1)


def build():
    mask = build_mask()
    sdf = ndi.distance_transform_edt(mask > 0.5)
    sdfs = ndi.gaussian_filter(sdf, 7.0)
    gy, gx = np.gradient(sdfs)
    nz = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    nx, ny = gx / nz, gy / nz          # inward normal

    H2, W2 = mask.shape
    yy, xx = np.mgrid[0:H2, 0:W2].astype(float)
    # FIRE point just left of the open jaw
    fx, fy = 55 * SS, 215 * SS
    dxp, dyp = fx - xx, fy - yy
    dist = np.sqrt(dxp ** 2 + dyp ** 2) + 1e-6
    fdir_x, fdir_y = dxp / dist, dyp / dist
    fire_lam = np.clip(-(nx * fdir_x + ny * fdir_y), 0, 1)
    # outward normal dotted toward fire  (outward = -inward)
    fire = fire_lam * np.clip(1.0 - dist / (620 * SS / 3.0), 0, 1) ** 1.6

    moon_lam = np.clip(nx * 0.55 + ny * 0.8, 0, 1)

    dk_lo = np.array([0.055, 0.045, 0.05])
    dk_hi = np.array([0.16, 0.11, 0.09])
    depth = np.clip(sdfs / 40.0, 0, 1)
    col = dk_lo[None, None] + (dk_hi - dk_lo)[None, None] * depth[..., None]
    # fire light: warm gradient by fire term
    ember = np.array([0.92, 0.42, 0.14])
    blood = np.array([0.45, 0.12, 0.08])
    f = np.clip(fire * 1.35, 0, 1)
    fcol = blood[None, None] + (ember - blood)[None, None] \
        * np.clip(f * 1.4 - 0.25, 0, 1)[..., None]
    col = col * (1 - f[..., None] * 0.85) + fcol * (f * 0.85)[..., None]
    # moon kiss, faint, top-left facing edges
    edge = np.clip(1 - sdf / (7.0 * SS), 0, 1)
    mk = moon_lam ** 2 * edge * 0.5
    col = col * (1 - mk[..., None]) + np.array([0.55, 0.66, 0.85])[
        None, None] * mk[..., None]

    rgba = np.concatenate([col, mask[..., None]], -1)
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)

    # EYE: ember slit under the brow
    exq, eyq = 178, 168
    dr.ellipse([(exq - 8) * SS, (eyq - 5) * SS, (exq + 8) * SS,
                (eyq + 5) * SS], fill=(235, 140, 40, 255))
    dr.ellipse([(exq - 2) * SS, (eyq - 4) * SS, (exq + 2) * SS,
                (eyq + 4) * SS], fill=(30, 12, 10, 255))
    # TEETH: upper and lower triangles in the open mouth
    for i in range(5):
        x = 96 + i * 22
        h = 9 if i % 2 == 0 else 6
        dr.polygon(P([(x, 196), (x + 3, 196 + h), (x + 6, 196)]),
                   fill=(186, 166, 146, 230))
    # mouth interior: a thin burning slit
    dr.polygon(P([(92, 200), (188, 198), (196, 206), (100, 210)]),
               fill=(120, 44, 18, 200))
    dr.polygon(P([(100, 201), (180, 200), (186, 204), (106, 206)]),
               fill=(220, 110, 40, 200))
    # nostril flare
    dr.ellipse([76 * SS, 172 * SS, 86 * SS, 180 * SS], fill=(240, 130, 40, 200))
    # membrane veins (dark branching on the sails)
    rng = np.random.RandomState(4)
    root = (565, 165)
    for tip in ((668, 34), (716, 96), (744, 196), (740, 300)):
        for k in range(3):
            t0 = rng.uniform(0.25, 0.5)
            bx = root[0] + (tip[0] - root[0]) * t0
            by = root[1] + (tip[1] - root[1]) * t0
            ex2 = bx + rng.uniform(-30, 50)
            ey2 = by + rng.uniform(20, 60)
            dr.line([bx * SS, by * SS, ex2 * SS, ey2 * SS],
                    fill=(20, 12, 12, 120), width=SS)
    # organic hide texture: high-pass noise bands, no doodles
    arr2 = np.asarray(im, np.float64) / 255.0
    rng2 = np.random.RandomState(8)
    noise = rng2.random(mask.shape)
    lo = ndi.gaussian_filter(noise, 6.0)
    hi = ndi.gaussian_filter(noise, 1.6) - lo
    tex = 1.0 + hi * 0.9
    body_zone = (mask > 0.5) & (sdf > 4)
    arr2[..., :3] *= np.where(body_zone[..., None], tex[..., None], 1.0)
    arr2[..., :3] = np.clip(arr2[..., :3], 0, 1)
    im = Image.fromarray((arr2 * 255).astype(np.uint8))

    out = Image.fromarray(np.asarray(im))
    out = out.resize((PW, PH), Image.LANCZOS)
    out.save("/root/work/r33/dragon_plate.png")
    print("dragon plate saved")


build()
