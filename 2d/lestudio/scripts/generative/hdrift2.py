"""HDRIFT v2 -- high-capacity colour-splat drift with tangible structure.

The R27 dream was 48 isotropic grayscale-ish splats at 64x96 -- bokeh by
construction. v2: K anisotropic COLOUR splats per image (matching
pursuit: place at the residual peak, shape from the residual's local
second moments, joint per-channel lstsq refit), importance-ordered so
splat slot j means the same thing across the collection, whitened, drift
in that space, decode with anisotropic rendering + Reinhard. Elongated
splats draw walls, stairs and trails as STROKES, not dots.

Deterministic in (collection, K, seed) end to end.
"""
import numpy as np


# ---------------------------------------------------------------- fitting
def _gauss_field(shape, cy, cx, S_inv, cut=4.0):
    """Anisotropic gaussian over the full frame (vectorised)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    y = yy - cy
    x = xx - cx
    q = (S_inv[0, 0] * y * y + 2 * S_inv[0, 1] * x * y
         + S_inv[1, 1] * x * x)
    g = np.exp(-0.5 * np.clip(q, 0, 2 * cut * cut))
    g[q > cut * cut] = 0.0
    return g


def _pursuit(res, K, patch, sig_max, jitter=None, blur=2):
    """Matching pursuit on the residual `res` (modified in place).
    Returns per-splat rows (cy, cx, aR, aG, aB, l11, l21, l22) in FIT
    ORDER (residual-energy order = importance order)."""
    H, W = res.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    feats, fields = [], []
    for k in range(K):
        mag = np.abs(res).sum(-1)
        if blur > 1:                      # box blur via cumsum: find MASSES
            c = np.cumsum(np.cumsum(mag, 0), 1)
            cpad = np.pad(c, ((blur, 0), (blur, 0)))
            mag = (cpad[blur:, blur:] - cpad[:-blur, blur:]
                   - cpad[blur:, :-blur] + cpad[:-blur, :-blur])
        idx = int(mag.argmax())
        cy, cx = idx // W, idx % W
        if jitter is not None:
            cy = int(np.clip(cy + jitter.randint(-1, 2), 0, H - 1))
            cx = int(np.clip(cx + jitter.randint(-1, 2), 0, W - 1))
        y0, y1 = max(0, cy - patch), min(H, cy + patch + 1)
        x0, x1 = max(0, cx - patch), min(W, cx + patch + 1)
        pw = np.abs(res[y0:y1, x0:x1]).sum(-1) + 1e-8
        py = yy[y0:y1, x0:x1] - cy
        px = xx[y0:y1, x0:x1] - cx
        wsum = pw.sum()
        Syy = float((pw * py * py).sum() / wsum) + 0.35
        Sxx = float((pw * px * px).sum() / wsum) + 0.35
        Sxy = float((pw * py * px).sum() / wsum)
        S = np.asarray([[Syy, Sxy], [Sxy, Sxx]])
        w_eig, V = np.linalg.eigh(S)
        # asymmetric clamp: the MAJOR axis may run long (strokes: stairs,
        # trails, wall edges) while the minor axis stays tight -- round
        # blobs cannot draw architecture. Boost the major axis past the
        # patch horizon; the joint refit re-balances amplitudes.
        w_eig = np.clip(w_eig, 0.35, sig_max * sig_max)
        if w_eig[1] > 4 * w_eig[0]:
            w_eig[1] = min(w_eig[1] * 2.6, (sig_max * 2.2) ** 2)
        S = (V * w_eig) @ V.T
        S_inv = np.linalg.inv(S)
        g = _gauss_field((H, W), cy, cx, S_inv)
        gg = float((g * g).sum()) + 1e-8
        amp = (res * g[..., None]).sum((0, 1)) / gg
        res -= g[..., None] * amp[None, None, :]
        L = np.linalg.cholesky(S)
        feats.append([cy, cx, amp[0], amp[1], amp[2],
                      L[0, 0], L[1, 0], L[1, 1]])
        fields.append(g)
    return feats, fields


def fit_color_splats(img, K=128, jitter=None):
    """Coarse-to-fine anisotropic colour-splat fit.

    Three stages so the budget covers every scale: big soft masses (sky,
    walls, water) fitted on a 4x downsample with huge sigmas, mid
    structure (stairs, trail segments, glow) at 2x, detail (windows,
    figures, sparks) at full res. All features live in full-res
    coordinates; stage order + in-stage fit order = importance order,
    stable across a collection. Ends with a joint per-channel lstsq
    refit of every amplitude against the original image."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    k1 = max(8, K // 8)              # coarse
    k2 = max(16, (3 * K) // 8)       # mid
    k3 = K - k1 - k2                 # fine
    F_all, fields = [], []
    res = img.copy()
    for (kk, ds, patch, sig_max, blur) in (
            (k1, 4, 40, 90.0, 6),
            (k2, 2, 16, 26.0, 3),
            (k3, 1, 8, 9.0, 2)):
        if ds > 1:
            small = res[::ds, ::ds].copy()
            feats, _ = _pursuit(small, kk, patch, sig_max / ds,
                                jitter=jitter, blur=blur)
            for row in feats:
                row[0] *= ds
                row[1] *= ds
                row[5] *= ds
                row[6] *= ds
                row[7] *= ds
        else:
            feats, _ = _pursuit(res.copy(), kk, patch, sig_max,
                                jitter=jitter, blur=blur)
        # subtract this stage's splats from the FULL-res residual and
        # keep the full-res fields for the final joint refit
        for row in feats:
            cy, cx = row[0], row[1]
            L = np.asarray([[row[5], 0.0], [row[6], row[7]]])
            S_inv = np.linalg.inv(L @ L.T)
            g = _gauss_field((H, W), cy, cx, S_inv)
            gg = float((g * g).sum()) + 1e-8
            amp = (res * g[..., None]).sum((0, 1)) / gg
            res -= g[..., None] * amp[None, None, :]
            row[2], row[3], row[4] = amp
            fields.append(g)
            F_all.append(row)
    # joint per-channel refit against the ORIGINAL image
    A = np.stack([f.ravel() for f in fields], 1)           # (N, K)
    AtA = A.T @ A
    # RELATIVE ridge, and a strong one: the weak-ridge joint solve builds
    # +/-200 cancellation pairs that render fine together but explode the
    # moment drift decorrelates them (the R21 lesson, relearned with
    # colour). 0.02*mean-diag keeps every amplitude in ~[-1, 1].
    lam = 0.02 * float(np.trace(AtA)) / max(len(fields), 1)
    AtA = AtA + lam * np.eye(len(fields), dtype=np.float32)
    coef = np.linalg.solve(AtA, A.T @ img.reshape(-1, 3))  # (K, 3)
    F = np.asarray(F_all, np.float64)
    F[:, 2:5] = coef
    # canonical order WITHIN each stage: coarse spatial cells, then x.
    # The collection shares one composition family, so slot j lands on
    # the same region and scale in every image -- that correspondence is
    # what lets the drift interpolate structure instead of scrambling it.
    out = []
    off = 0
    for kk in (k1, k2, k3):
        blk = F[off:off + kk]
        order = np.lexsort((blk[:, 1], np.round(blk[:, 1] / 33),
                            np.round(blk[:, 0] / 46)))
        out.append(blk[order])
        off += kk
    return np.concatenate(out, 0), fields


def render_splats(F, shape):
    """Decode a (K, 8) feature block to RGB with Reinhard tone-mapping."""
    H, W = shape
    out = np.zeros((H, W, 3), np.float32)
    for row in F:
        cy, cx, aR, aG, aB, l11, l21, l22 = row
        l11 = max(float(l11), 0.45)
        l22 = max(float(l22), 0.45)
        L = np.asarray([[l11, 0.0], [l21, l22]])
        S = L @ L.T
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        g = _gauss_field((H, W), float(cy), float(cx), S_inv)
        out += g[..., None] * np.asarray([aR, aG, aB],
                                         np.float32)[None, None, :]
    out = np.maximum(out, 0.0)
    return out / (1.0 + 0.15 * out)          # gentle Reinhard


def refit_psnr(img, F):
    rec = render_splats(F, img.shape[:2])
    err = float(((np.asarray(img) - rec) ** 2).mean())
    return 10 * np.log10(1.0 / max(err, 1e-9))


# ---------------------------------------------------------------- drifting
def encode_collection(frames, K=128, seed=0):
    """All frames -> (n, K*8) whitened feature matrix + codec state."""
    rng = np.random.RandomState(seed)
    rows = []
    for i, f in enumerate(frames):
        F, _ = fit_color_splats(f, K=K,
                                jitter=np.random.RandomState(seed + i))
        rows.append(F.ravel())
    raw = np.stack(rows)
    mu, sd = raw.mean(0), raw.std(0) + 1e-6
    lo, hi = raw.min(0), raw.max(0)
    return {"raw": raw, "mu": mu, "sd": sd, "lo": lo, "hi": hi, "K": K,
            "shape": frames[0].shape[:2]}


def dream(codec, n=6, seed=1, dim=1536, steps=60, expand=0.06):
    """Drift in the whitened splat space; decode n frames.

    `expand` relaxes the on-manifold clamp a little beyond the training
    min/max so samples can leave the convex hull without blowing up."""
    import sys
    sys.path.insert(0, "/root/work/lecore_main")
    from holographic.sampling_and_signal.holographic_hdrift import (
        build_drift_model, drift_sample)
    raw, mu, sd = codec["raw"], codec["mu"], codec["sd"]
    lo, hi = codec["lo"], codec["hi"]
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    lo2, hi2 = lo - expand * span, hi + expand * span
    model = build_drift_model((raw - mu) / sd, dim=dim, seed=seed)
    X = drift_sample(model, n=n, seed=seed, steps=steps)
    outs = []
    for x in X:
        p = np.clip(np.asarray(x) * sd + mu, lo2, hi2)
        outs.append(render_splats(p.reshape(-1, 8), codec["shape"]))
    return outs
