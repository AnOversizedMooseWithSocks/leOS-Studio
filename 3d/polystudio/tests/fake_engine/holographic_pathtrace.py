"""Stand-in tracer. NOT a renderer -- a deterministic gradient with the real call signature, so the
route's streaming, progress callback and cancellation can be exercised."""
import time
import numpy as np

def path_trace(scene, cam, width=64, height=48, spp=8, max_bounce=3, material=None, sky=None,
               seed=0, antialias=True, on_progress=None, progress_every=1, **kw):
    W, H = int(width), int(height)
    acc = np.zeros((H, W, 3), float)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([xx / max(W - 1, 1), yy / max(H - 1, 1), np.full((H, W), 0.5)], axis=-1)
    for s in range(1, int(spp) + 1):
        time.sleep(0.004)                       # a sample is not free; makes cancellation observable
        acc += base * (1.0 + 0.01 * s)
        if on_progress is not None and (s % max(1, int(progress_every)) == 0 or s == spp):
            on_progress(acc / s, s, int(spp))   # may raise to cancel, exactly like the real one
    return acc / max(int(spp), 1)
