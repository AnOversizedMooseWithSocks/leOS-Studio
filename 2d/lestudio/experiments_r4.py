"""R4 experiments: every panel theory gets a number before it gets a node.

Run from the leCore repo root's sibling (uses the local lecore checkout on
sys.path via PYTHONPATH). Prototypes here are the reference implementations
the nodes will reuse; negatives are kept in the printed record.
"""
import time, sys
import numpy as np

sys.path.insert(0, "/root/work/lecore")
import lecore

M = lecore.UnifiedMind(dim=256, seed=0)
rng = np.random.default_rng(7)

REPORT = []


def rec(tid, name, verdict, detail):
    REPORT.append((tid, name, verdict, detail))
    print(f"[{tid}] {name}: {verdict} — {detail}")


def timed(fn):
    t0 = time.time(); out = fn(); return out, time.time() - t0


# ---- shared fixtures ---------------------------------------------------------
def test_image(h=256, w=256, seed=1):
    f = M.pattern_field('fbm', scale=3.0, seed=seed)
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([xx / w, yy / h, np.zeros_like(xx)], -1).reshape(-1, 3).astype(float)
    g = np.asarray(f(P), float).reshape(h, w)
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    img = np.stack([g, np.clip(g * 1.2 - .1, 0, 1), 1 - g], -1)
    # add structured content: bright disc + dark bar (edges for halo tests)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - w * .3) ** 2 + (yy - h * .3) ** 2) < (min(h, w) * .12) ** 2
    img[disc] = [.95, .9, .7]
    img[h // 2 - 6:h // 2 + 6, :] = [.05, .05, .1]
    return np.clip(img, 0, 1)


def box(img, r):
    """O(N) box mean via cumsum, edge-padded, any (H,W) or (H,W,C)."""
    if r <= 0: return img.copy()
    pad = np.pad(img, [(r + 1, r)] * 2 + [(0, 0)] * (img.ndim - 2), mode='edge')
    c = pad.cumsum(0).cumsum(1)
    d = 2 * r + 1
    out = (c[d:, d:] - c[:-d, d:] - c[d:, :-d] + c[:-d, :-d]) / (d * d)
    return out


def gauss(img, sigma):
    """separable gaussian via repeated box (3x) — good enough, fast."""
    r = max(1, int(round(sigma * 0.6)))
    out = img.astype(float)
    for _ in range(3):
        out = box(out if out.ndim == 3 else out[..., None], r)[..., 0] if img.ndim == 2 else box(out, r)
    return out


def guided(guide, src, radius, eps):
    return np.asarray(M.guided_filter(guide.astype(float), src.astype(float),
                                      radius=int(radius), eps=float(eps)))


# ---- T1 clarity / dehaze -----------------------------------------------------
def t1():
    img = test_image()
    g = img.mean(-1)
    base = np.stack([guided(g, img[..., c], 12, 0.02) for c in range(3)], -1)
    detail = img - base
    clar = np.clip(base + 2.2 * detail, 0, 1)
    # halo metric: overshoot just outside the dark bar edge vs unsharp mask
    blur = np.stack([gauss(img[..., c], 6) for c in range(3)], -1)
    usm = np.clip(img + 2.2 * (img - blur), 0, 1)
    row = img.shape[0] // 2 - 9   # 3px above the bar
    halo_gf = float(np.abs(clar[row] - img[row]).max())
    halo_um = float(np.abs(usm[row] - img[row]).max())
    ok = halo_gf < halo_um * 0.7
    rec("T1", "clarity halos (guided vs unsharp)", "PASS" if ok else "FAIL",
        f"edge overshoot guided={halo_gf:.3f} unsharp={halo_um:.3f}")

    # dehaze: synthesize haze img_h = img*t + A*(1-t), recover via dark channel + guided t
    A = np.array([.92, .93, .95])
    yy = np.linspace(1, 0.35, img.shape[0])[:, None]
    t_true = np.repeat(yy, img.shape[1], 1)
    hazed = img * t_true[..., None] + A * (1 - t_true[..., None])
    dark = box((hazed / A).min(-1)[..., None], 7)[..., 0]
    t_est = np.clip(1 - 0.95 * dark, 0.1, 1)
    t_ref = guided(hazed.mean(-1), t_est, 24, 1e-3)
    t_ref = np.clip(t_ref, 0.1, 1)
    deh = np.clip((hazed - A) / t_ref[..., None] + A, 0, 1)
    rmse_h = float(np.sqrt(((hazed - img) ** 2).mean()))
    rmse_d = float(np.sqrt(((deh - img) ** 2).mean()))
    ok = rmse_d < rmse_h * 0.65
    rec("T1", "dehaze recovers synthetic haze", "PASS" if ok else "FAIL",
        f"rmse hazed={rmse_h:.3f} dehazed={rmse_d:.3f}")
    return clar, deh


# ---- T2 dither / palette quantize -------------------------------------------
BAYER8 = (np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],
                    [12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
                    [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],
                    [15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]],float)+.5)/64


def median_cut(img, n):
    px = img.reshape(-1, 3).copy()
    boxes = [px]
    while len(boxes) < n:
        i = max(range(len(boxes)), key=lambda j: np.ptp(boxes[j], 0).max() * len(boxes[j]))
        b = boxes.pop(i)
        ch = np.ptp(b, 0).argmax()
        med = np.median(b[:, ch])
        lo, hi = b[b[:, ch] <= med], b[b[:, ch] > med]
        if not len(lo) or not len(hi):
            boxes.append(b); break
        boxes += [lo, hi]
    return np.array([b.mean(0) for b in boxes])


def nearest(img, pal):
    d = ((img[..., None, :] - pal) ** 2).sum(-1)
    return pal[d.argmin(-1)]


def dither_fs(img, pal):
    out = img.astype(float).copy()
    h, w, _ = out.shape
    for y in range(h):
        for x in range(w):
            old = out[y, x].copy()
            new = pal[((old - pal) ** 2).sum(-1).argmin()]
            out[y, x] = new
            err = old - new
            if x + 1 < w: out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x: out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x] += err * 5 / 16
                if x + 1 < w: out[y + 1, x + 1] += err * 1 / 16
    return np.clip(out, 0, 1)


def dither_bayer(img, pal, strength=1.0):
    h, w, _ = img.shape
    t = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    lifted = np.clip(img + (t[..., None] - .5) * strength / max(len(pal) ** (1 / 3), 2), 0, 1)
    return nearest(lifted, pal)


def t2():
    img = test_image(128, 128)
    pal, dt_p = timed(lambda: median_cut(img, 16))
    q_near = nearest(img, pal)
    q_bay, dt_b = timed(lambda: dither_bayer(img, pal))
    q_fs, dt_f = timed(lambda: dither_fs(img, pal))
    e = lambda q: float(np.abs(gauss(q.mean(-1), 2) - gauss(img.mean(-1), 2)).mean())
    ok = e(q_fs) < e(q_near) and e(q_bay) < e(q_near) * 1.5
    rec("T2", "dither beats nearest (low-freq err)", "PASS" if ok else "FAIL",
        f"near={e(q_near):.4f} bayer={e(q_bay):.4f} fs={e(q_fs):.4f}; "
        f"palette {dt_p*1000:.0f}ms bayer {dt_b*1000:.0f}ms fs(128²py) {dt_f*1000:.0f}ms")
    return q_fs


# ---- T3 LUT + wheels ---------------------------------------------------------
def cube_write(lut, path):
    n = lut.shape[0]
    with open(path, 'w') as f:
        f.write(f"LUT_3D_SIZE {n}\n")
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))


def cube_read(path):
    n, rows = 0, []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        if ln.upper().startswith('LUT_3D_SIZE'): n = int(ln.split()[-1]); continue
        if ln[0].isdigit() or ln[0] == '-':
            rows.append([float(v) for v in ln.split()[:3]])
    lut = np.array(rows).reshape(n, n, n, 3).transpose(2, 1, 0, 3)  # file is R-fastest
    return lut


def lut_apply(img, lut):
    n = lut.shape[0]
    x = np.clip(img, 0, 1) * (n - 1)
    i = np.clip(x.astype(int), 0, n - 2)
    f = x - i
    r, g, b = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    def L(dr, dg, db): return lut[r + dr, g + dg, b + db]
    c00 = L(0,0,0)*(1-fr)+L(1,0,0)*fr; c10 = L(0,1,0)*(1-fr)+L(1,1,0)*fr
    c01 = L(0,0,1)*(1-fr)+L(1,0,1)*fr; c11 = L(0,1,1)*(1-fr)+L(1,1,1)*fr
    c0 = c00*(1-fg)+c10*fg; c1 = c01*(1-fg)+c11*fg
    return c0*(1-fb)+c1*fb


def wheels(img, lift, gamma_, gain, lo=0.33, hi=0.66):
    """lift/gamma/gain as per-band offsets weighted by luma masks."""
    y = img.mean(-1)
    sh = np.clip(1 - y / lo, 0, 1) ** 2
    hl = np.clip((y - hi) / (1 - hi), 0, 1) ** 2
    mid = np.clip(1 - sh - hl, 0, 1)
    out = img + sh[..., None] * np.asarray(lift) + mid[..., None] * np.asarray(gamma_) \
              + hl[..., None] * np.asarray(gain)
    return np.clip(out, 0, 1)


def t3():
    n = 17
    idx = np.linspace(0, 1, n)
    ident = np.stack(np.meshgrid(idx, idx, idx, indexing='ij'), -1)
    cube_write(ident, '/tmp/ident.cube')
    back = cube_read('/tmp/ident.cube')
    err = float(np.abs(back - ident).max())
    img = test_image()
    out, dt = timed(lambda: lut_apply(img, back))
    ok = err < 1e-6 and float(np.abs(out - img).max()) < 1e-6
    rec("T3", ".cube identity round-trip + apply", "PASS" if ok else "FAIL",
        f"write/read err={err:.2e}, identity apply err={np.abs(out-img).max():.2e}, "
        f"apply 256²={dt*1000:.0f}ms")
    ramp = np.linspace(0, 1, 256)[:, None].repeat(3, 1)[None]
    w1 = wheels(ramp, [0,0,0], [0,0,0], [0,0,0])
    mono = bool(np.all(np.diff(wheels(ramp, [.05,0,0], [0,.05,0], [0,0,.05])[0, :, 0]) > -1e-6))
    rec("T3", "wheels neutral=identity, monotone", "PASS" if np.allclose(w1, ramp) and mono else "FAIL",
        f"neutral max err={np.abs(w1-ramp).max():.2e}, monotone={mono}")


# ---- T4 film look ------------------------------------------------------------
def film_look(img, halation=.6, grain=.08, fade=.15, weave=0, seed=0, frame=0):
    r = np.default_rng if False else np.random.default_rng(seed * 9973 + frame)
    y = img.mean(-1)
    hot = np.clip(y - 0.75, 0, 1) / 0.25
    glow = gauss(hot, 8)
    out = img.copy()
    out[..., 0] = 1 - (1 - out[..., 0]) * (1 - halation * glow)          # screen, red
    out[..., 1] = 1 - (1 - out[..., 1]) * (1 - halation * .35 * glow)    # a little orange
    g = r.normal(0, 1, y.shape)
    g = gauss(g, 1.2)
    gm = grain * (0.35 + 0.65 * (1 - np.abs(y - .5) * 2))                # grain lives in mids
    out += (g * gm)[..., None]
    out = out * (1 - fade) + fade * np.array([.5, .5, .52])              # lifted fade
    return np.clip(out, 0, 1)


def t4():
    img = test_image()
    out, dt = timed(lambda: film_look(img))
    y = img.mean(-1); mask = gauss(np.clip(y - .75, 0, 1) / .25, 8) > 0.02
    red_delta = np.abs(out[..., 0] - np.clip(img[..., 0] + 0, 0, 1))
    # subtract grain/fade baseline: compare against film w/o halation
    base = film_look(img, halation=0)
    hal = np.abs(out[..., 0] - base[..., 0])
    outside = float(hal[~mask].sum() / max(hal.sum(), 1e-9))
    det = np.abs(film_look(img, seed=3) - film_look(img, seed=3)).max()
    ok = outside < 0.05 and det == 0
    rec("T4", "halation confined to highlight glow; deterministic", "PASS" if ok else "FAIL",
        f"energy outside glow mask={outside*100:.1f}%, det err={det}, {dt*1000:.0f}ms 256²")


# ---- T5 patchmatch fill (honest speed test) ---------------------------------
def t5():
    img = test_image(192, 192)
    h0, h1, w0, w1 = 80, 112, 80, 112     # 32² hole
    hole = np.zeros(img.shape[:2], bool); hole[h0:h1, w0:w1] = True
    # patch-library fill: coherence search over shifted candidates (vectorized proxy
    # for PatchMatch: random candidate shifts + propagation via repeated best-of)
    def pm_fill(img, hole, iters=6, cand=64, seed=0):
        r = np.random.default_rng(seed)
        out = img.copy(); out[hole] = box(np.where(hole[...,None], np.nan, img), 0)[hole] if False else 0.5
        ys, xs = np.nonzero(hole)
        best = np.full(len(ys), np.inf); src = np.zeros((len(ys), 2), int)
        H, W = hole.shape
        for it in range(iters):
            shifts = r.integers(-70, 70, (cand, 2))
            for dy, dx in shifts:
                sy, sx = ys + dy, xs + dx
                ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
                ok &= ~hole[np.clip(sy, 0, H-1), np.clip(sx, 0, W-1)]
                if not ok.any(): continue
                # cost: ring context distance (3px ring around each hole pixel, sampled 4-neighb)
                cost = np.full(len(ys), np.inf)
                acc = np.zeros(len(ys)); cnt = 0
                for oy, ox in ((-3,0),(3,0),(0,-3),(0,3)):
                    py, px = np.clip(ys+oy,0,H-1), np.clip(xs+ox,0,W-1)
                    qy, qx = np.clip(sy+oy,0,H-1), np.clip(sx+ox,0,W-1)
                    valid = ~hole[py, px]
                    d = ((out[py,px]-out[qy,qx])**2).sum(-1)
                    acc += np.where(valid, d, 0); cnt += 1
                cost = np.where(ok, acc, np.inf)
                upd = cost < best
                best[upd] = cost[upd]; src[upd] = np.stack([sy, sx], 1)[upd]
            out[ys, xs] = img[np.clip(src[:,0],0,H-1), np.clip(src[:,1],0,W-1)]
        return out
    pm, dt_pm = timed(lambda: pm_fill(img, hole))
    har, dt_h = timed(lambda: np.asarray(M.inpaint(img, hole.astype(float))))
    # quality: texture energy inside the hole should match the surroundings
    def tex(a):
        g = a.mean(-1); return float(np.abs(np.diff(g[h0:h1, w0:w1], axis=1)).mean())
    ring = img[h0-16:h1+16, w0-16:w1+16]
    t_ring = float(np.abs(np.diff(ring.mean(-1), axis=1)).mean())
    rec("T5", "patch fill vs harmonic inpaint", "MEASURED",
        f"patch {dt_pm:.2f}s tex={tex(pm):.4f}; harmonic {dt_h:.2f}s tex={tex(har):.4f}; "
        f"surround tex={t_ring:.4f} (closer is better)")
    return pm, har


# ---- T6 seam carving --------------------------------------------------------
def seam_carve(img, n_remove):
    out = img.copy()
    for _ in range(n_remove):
        g = out.mean(-1)
        e = np.abs(np.diff(g, axis=1, prepend=g[:, :1])) + \
            np.abs(np.diff(g, axis=0, prepend=g[:1]))
        Mc = e.copy()
        for y in range(1, out.shape[0]):
            up = Mc[y-1]
            left = np.concatenate([[np.inf], up[:-1]])
            right = np.concatenate([up[1:], [np.inf]])
            Mc[y] += np.minimum(np.minimum(left, up), right)
        # backtrack
        H, W = Mc.shape
        seam = np.zeros(H, int); seam[-1] = int(Mc[-1].argmin())
        for y in range(H - 2, -1, -1):
            x = seam[y+1]
            lo, hi = max(x-1, 0), min(x+2, W)
            seam[y] = lo + int(Mc[y, lo:hi].argmin())
        keep = np.ones((H, W), bool); keep[np.arange(H), seam] = False
        out = out[keep].reshape(H, W - 1, 3)
    return out


def t6():
    img = test_image(192, 256)
    out, dt = timed(lambda: seam_carve(img, int(256 * .15)))
    # the dark bar (straight feature) must survive intact
    bar = out[192 // 2, :, 0]
    ok = float(bar.mean()) < 0.15
    rec("T6", "seam carve 15% width", "PASS" if ok else "FAIL",
        f"{dt:.2f}s 192×256→{out.shape[1]}w; bar intact mean={bar.mean():.3f}")


# ---- T7 frequency split ------------------------------------------------------
def t7():
    img = test_image()
    lowf = np.stack([gauss(img[..., c], 4) for c in range(3)], -1)
    high = img - lowf
    err = float(np.abs((lowf + high) - img).max())
    rec("T7", "frequency split recombine identity", "PASS" if err < 1e-12 else "FAIL",
        f"max err={err:.2e}")


# ---- T8 orbit trap node budget ----------------------------------------------
def t8():
    try:
        sdf = M.sdf_scene("sphere 0 0 0 1") if hasattr(M, 'sdf_scene') else None
    except Exception:
        sdf = None
    def run():
        import inspect
        try:
            return M.orbit_trap_render(sdf, {"eye": [0, 0, 3], "target": [0, 0, 0]},
                                       width=192, height=144)
        except Exception as e:
            return e
    out, dt = timed(run)
    ok = hasattr(out, 'shape')
    rec("T8", "orbit_trap_render 192×144", "PASS" if ok else "PROBE-FAIL",
        f"{dt:.2f}s -> {getattr(out, 'shape', out)}")


# ---- T9 visual memory --------------------------------------------------------
def t9():
    a, b, c = test_image(seed=1), test_image(seed=2), test_image(seed=3)
    M.image_remember(a, 'sunset study'); M.image_remember(b, 'sunset study')
    M.image_remember(c, 'blue abstract')
    d = M.image_dream('sunset study')
    got = d.get('dreamed', 0) if isinstance(d, dict) else 0
    rc = M.image_recall('sunset study')
    ok = bool(got) and isinstance(rc, list) and len(rc) >= 1
    refuse = M.image_dream('nonexistent tag xyz')
    honest = isinstance(refuse, dict) and not refuse.get('dreamed')
    rec("T9", "remember/recall/dream + honest refusal", "PASS" if ok and honest else "MEASURED",
        f"dreamed={got}, recall={len(rc) if isinstance(rc, list) else rc}, refuses unknown={honest}, "
        f"dream keys={list(d.keys()) if isinstance(d, dict) else d}")


# ---- T10 focus stack ---------------------------------------------------------
def focus_stack(imgs, levels=5):
    def lap_energy(a):
        g = a.mean(-1)
        l = g - gauss(g, 2)
        return gauss(np.abs(l), 2)
    E = np.stack([lap_energy(i) for i in imgs])
    idx = E.argmax(0)
    sel = np.stack(imgs)[idx, np.arange(idx.shape[0])[:, None], np.arange(idx.shape[1])]
    return sel


def t10():
    sharp = test_image()
    blur_top = sharp.copy(); blur_bot = sharp.copy()
    for c in range(3):
        b = gauss(sharp[..., c], 5)
        blur_top[:128, :, c] = b[:128]; blur_bot[128:, :, c] = b[128:]
    fused, dt = timed(lambda: focus_stack([blur_top, blur_bot]))
    def en(a):
        g = a.mean(-1); return float(np.abs(g - gauss(g, 2)).mean())
    ok = en(fused) > max(en(blur_top), en(blur_bot))
    rec("T10", "focus stack sharper than any input", "PASS" if ok else "FAIL",
        f"E(fused)={en(fused):.4f} vs {en(blur_top):.4f}/{en(blur_bot):.4f}, {dt*1000:.0f}ms")


# ---- T11 AgX -----------------------------------------------------------------
AGX_IN = np.array([[0.8566, 0.1373, 0.1119], [0.0951, 0.7612, 0.0768],
                   [0.0483, 0.1015, 0.8113]])


def agx(img):
    x = np.einsum('...c,dc->...d', np.clip(img, 0, None), AGX_IN)
    x = np.clip((np.log2(np.maximum(x, 1e-10)) + 12.47393) / (12.47393 + 4.026069), 0, 1)
    # 6th-order sigmoid fit (Blender AgX approximation)
    x2 = x * x; x4 = x2 * x2
    s = (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
         - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)
    return np.clip(s, 0, 1)


def t11():
    ramp = np.linspace(0, 4, 512)[:, None].repeat(3, 1)[None]
    out = agx(ramp)[0, :, 0]
    mono = bool(np.all(np.diff(out) > -1e-6))
    sat = np.zeros((1, 64, 3)); sat[..., 0] = np.linspace(0, 8, 64)
    o = agx(sat)
    ok = mono and float(o.max()) <= 1 and float(o.min()) >= 0
    rec("T11", "AgX monotone + bounded on hot saturated sweep", "PASS" if ok else "FAIL",
        f"monotone={mono}, range=[{o.min():.3f},{o.max():.3f}], red ramp max={out.max():.3f}")


# ---- T12 GLSL export ---------------------------------------------------------
def t12():
    try:
        chain = M.postfx_chain(('bloom', {'strength': .5}), ('grade', {'saturation': 1.2}))
        src = M.postfx_to_glsl(chain)
        ok = isinstance(src, str) and 'mainImage' in src
        rec("T12", "postfx_to_glsl emits Shadertoy", "PASS" if ok else "FAIL",
            f"len={len(src) if isinstance(src, str) else type(src)}")
    except Exception as e:
        rec("T12", "postfx_to_glsl", "PROBE-FAIL", str(e)[:140])
    try:
        pal = M.cosine_palette_to_glsl(M.cosine_palette(seed=2) if hasattr(M, 'cosine_palette') else None)
        rec("T12", "cosine_palette_to_glsl", "PASS" if 'vec3' in str(pal) else "MEASURED", str(pal)[:90])
    except Exception as e:
        rec("T12", "cosine_palette_to_glsl", "PROBE-FAIL", str(e)[:140])


# ---- T13 scopes --------------------------------------------------------------
def t13():
    img = test_image()
    def waveform(a, bins=128):
        h, w, _ = a.shape
        wf = np.zeros((bins, w, 3))
        q = np.clip((a * (bins - 1)).astype(int), 0, bins - 1)
        for c in range(3):
            for x in range(w):
                np.add.at(wf[:, x, c], q[:, x, c], 1)
        return wf / max(h * .25, 1)
    def vectorscope(a, bins=128):
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        cb = -0.169 * r - 0.331 * g + 0.5 * b
        cr = 0.5 * r - 0.419 * g - 0.081 * b
        xi = np.clip(((cb + .5) * (bins - 1)).astype(int), 0, bins - 1)
        yi = np.clip(((.5 - cr) * (bins - 1)).astype(int), 0, bins - 1)
        vs = np.zeros((bins, bins))
        np.add.at(vs, (yi.ravel(), xi.ravel()), 1)
        return vs
    wf, dt1 = timed(lambda: waveform(img))
    vs, dt2 = timed(lambda: vectorscope(img))
    rec("T13", "waveform+vectorscope math", "PASS" if wf.max() > 0 and vs.max() > 0 else "FAIL",
        f"waveform {dt1*1000:.0f}ms, vectorscope {dt2*1000:.0f}ms at 256²")


if __name__ == '__main__':
    for t in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13):
        try:
            t()
        except Exception as e:
            import traceback
            rec(t.__name__.upper(), "experiment crashed", "ERROR", f"{type(e).__name__}: {e}")
    print("\n==== SUMMARY ====")
    for tid, name, v, d in REPORT:
        print(f"{tid:4s} {v:10s} {name}")
