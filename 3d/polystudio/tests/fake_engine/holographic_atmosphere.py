import numpy as np
def depth_fog(hdr, depth, density=0.075, fog_color=(0.58,0.66,0.80), start=1.2):
    hdr = np.asarray(hdr, float); d = np.asarray(depth, float)
    t = np.clip(1.0 - np.exp(-np.maximum(d - start, 0.0) * density), 0, 1)[..., None]
    return hdr * (1 - t) + np.asarray(fog_color, float) * t
