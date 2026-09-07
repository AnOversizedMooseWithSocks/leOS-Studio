import numpy as np
def clamp_fireflies(hdr, percentile=99.5):
    hdr = np.asarray(hdr, float)
    hi = np.percentile(hdr, percentile) if hdr.size else 1.0
    return np.minimum(hdr, max(hi, 1e-6))
