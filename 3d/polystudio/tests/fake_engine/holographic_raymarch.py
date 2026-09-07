"""Preview raymarcher stand-in: correct shapes and signatures, trivial shading."""
import numpy as np

def _eval(scene, P):
    if hasattr(scene, "eval"): return np.asarray(scene.eval(P), float)
    return np.asarray(scene(P), float)

def sphere_trace(scene, origins, dirs, max_steps=64, max_dist=60.0, eps=1e-3, **kw):
    """sphere_trace(scene, O, D) -> (hit, t, P). Real sphere tracing, just unshaded."""
    O = np.atleast_2d(np.asarray(origins, float)); D = np.atleast_2d(np.asarray(dirs, float))
    n = len(D)
    t = np.zeros(n); hit = np.zeros(n, bool); alive = np.ones(n, bool)
    for _ in range(int(max_steps)):
        if not alive.any(): break
        P = O[alive] + D[alive] * t[alive][:, None]
        d = _eval(scene, P)
        t[alive] += np.maximum(d, eps * 0.5)
        newly = d < eps
        idx = np.flatnonzero(alive)
        hit[idx[newly]] = True
        alive[idx[newly]] = False
        alive[idx[t[idx] > max_dist]] = False
    P = O + D * t[:, None]
    return hit, t, P

def render_sdf(scene, cam=None, width=64, height=48, **kw):
    W, H = int(width), int(height)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx/max(W-1,1), yy/max(H-1,1), np.full((H,W), 0.6)], axis=-1)
    return np.clip(img, 0, 1)


def sdf_curvature(scene, P, eps=1e-3, **kw):
    """Mean curvature by finite differences of the distance field (the real one's signature)."""
    P = np.atleast_2d(np.asarray(P, float))
    lap = np.zeros(len(P))
    d0 = _eval(scene, P)
    for ax in range(3):
        o = np.zeros(3); o[ax] = eps
        lap += _eval(scene, P + o) + _eval(scene, P - o) - 2.0 * d0
    return lap / (eps * eps)
