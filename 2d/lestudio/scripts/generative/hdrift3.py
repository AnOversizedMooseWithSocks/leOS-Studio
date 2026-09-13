"""HDRIFT v3: semantic-slot splat codes + PCA-latent drift.

Each element layer gets its own fixed slot block, fitted separately
with a scale prior matched to what that element IS (sky = 6 huge
splats, windows = 56 pinpoints, trail = 28 elongated strokes...).
Slot j means the same thing in every frame, so the PCA of the codes
captures real scene variation and the drift's samples decode into
tangible pictures. Deterministic in (collection, seed).
"""
import numpy as np

from hdrift2 import _pursuit, _gauss_field
from refgen2 import ELEMENTS

# per-element budgets and fit priors: (slots, downsample, patch, sig_max)
BUDGET = {
    "sky":     (6, 4, 60, 140.0, 8),
    "walls":   (30, 2, 24, 60.0, 4),
    "windows": (84, 1, 4, 3.2, 1),
    "stairs":  (64, 1, 10, 30.0, 2),
    "trail":   (40, 1, 9, 26.0, 2),
    "glow":    (12, 2, 20, 30.0, 3),
    "figure":  (10, 1, 6, 7.0, 1),
    "water":   (22, 2, 18, 40.0, 3),
}
K_TOTAL = sum(v[0] for v in BUDGET.values())


def encode_frame(layers, jitter=None):
    """dict of element layers -> (K_TOTAL, 8) semantic splat code."""
    blocks = []
    for e in ELEMENTS:
        kk, ds, patch, sig_max, blur = BUDGET[e]
        img = np.asarray(layers[e], np.float32)
        if ds > 1:
            small = img[::ds, ::ds].copy()
            feats, _ = _pursuit(small, kk, patch, sig_max / ds,
                                jitter=jitter, blur=blur)
            for row in feats:
                row[0] *= ds
                row[1] *= ds
                row[5] *= ds
                row[6] *= ds
                row[7] *= ds
        else:
            feats, _ = _pursuit(img.copy(), kk, patch, sig_max,
                                jitter=jitter, blur=blur)
        blk = np.asarray(feats, np.float64)
        # canonical in-block order: reading order on a coarse grid
        order = np.lexsort((blk[:, 1], np.round(blk[:, 1] / 33),
                            np.round(blk[:, 0] / 46)))
        blocks.append(blk[order])
    return np.concatenate(blocks, 0)


def render_code(code, shape, tone=0.12, scale=1.0):
    """Decode; `scale` renders the same splats at scale x resolution."""
    H, W = int(shape[0] * scale), int(shape[1] * scale)
    out = np.zeros((H, W, 3), np.float32)
    for row in code:
        cy, cx, aR, aG, aB, l11, l21, l22 = row
        cy, cx = cy * scale, cx * scale
        l11, l21, l22 = l11 * scale, l21 * scale, l22 * scale
        l11 = max(float(l11), 0.4)
        l22 = max(float(l22), 0.4)
        L = np.asarray([[l11, 0.0], [l21, l22]])
        try:
            S_inv = np.linalg.inv(L @ L.T)
        except np.linalg.LinAlgError:
            continue
        g = _gauss_field((H, W), float(cy), float(cx), S_inv)
        out += g[..., None] * np.asarray([aR, aG, aB],
                                         np.float32)[None, None, :]
    out = np.maximum(out, 0.0)
    return out / (1.0 + tone * out)


def encode_collection(frame_layers, seed=0):
    rows = []
    for i, layers in enumerate(frame_layers):
        code = encode_frame(layers, jitter=np.random.RandomState(seed + i))
        rows.append(code.ravel())
    raw = np.stack(rows)
    mu, sd = raw.mean(0), raw.std(0) + 1e-6
    return {"raw": raw, "mu": mu, "sd": sd,
            "lo": raw.min(0), "hi": raw.max(0),
            "shape": (368, 264), "K": K_TOTAL}


def dream(codec, n=12, seed=7, m=14, dim=512, steps=60, latitude=0.15):
    """PCA-latent drift: model the collection's coefficient cloud, drift
    there, decode through the basis. `latitude` lets samples step a
    little beyond the training coefficient range."""
    import sys
    sys.path.insert(0, "/root/work/lecore_main")
    from holographic.sampling_and_signal.holographic_hdrift import (
        build_drift_model, drift_sample)
    raw, mu, sd = codec["raw"], codec["mu"], codec["sd"]
    Z = (raw - mu) / sd
    zm = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - zm, full_matrices=False)
    C = U[:, :m] * S[:m]
    model = build_drift_model(C, dim=dim, seed=seed)
    X = drift_sample(model, n=n, seed=seed, steps=steps)
    lo_c, hi_c = C.min(0), C.max(0)
    span = hi_c - lo_c
    outs = []
    for c in X:
        c = np.clip(c, lo_c - latitude * span, hi_c + latitude * span)
        p = (zm + c @ Vt[:m]) * sd + mu
        p = np.clip(p, codec["lo"], codec["hi"])
        outs.append(render_code(p.reshape(-1, 8), codec["shape"]))
    return outs
