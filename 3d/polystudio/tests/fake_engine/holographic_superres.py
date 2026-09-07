import numpy as np
def easu_upscale(rgb, scale=2.0):
    """Nearest-neighbour stand-in with the real signature and output shape."""
    rgb = np.asarray(rgb, float)
    H, W = rgb.shape[:2]
    nh, nw = max(1, int(round(H*scale))), max(1, int(round(W*scale)))
    yi = np.clip((np.arange(nh)/scale).astype(int), 0, H-1)
    xi = np.clip((np.arange(nw)/scale).astype(int), 0, W-1)
    return rgb[yi][:, xi]
