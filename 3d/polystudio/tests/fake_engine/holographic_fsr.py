import numpy as np
def lanczos_upscale(img, out_hw, **kw):
    """lanczos_upscale(img, (H, W)) -- nearest-neighbour stand-in, correct output shape."""
    img = np.asarray(img, float); H, W = img.shape[:2]
    nh, nw = (int(out_hw[0]), int(out_hw[1])) if hasattr(out_hw, "__len__") else (int(out_hw), int(out_hw))
    nh, nw = max(1, nh), max(1, nw)
    yi = np.clip((np.arange(nh) * H) // nh, 0, H - 1)
    xi = np.clip((np.arange(nw) * W) // nw, 0, W - 1)
    return img[yi][:, xi]
def fsr_upscale(img, out_hw=None, scale=None, **kw):
    if out_hw is None:
        H, W = np.asarray(img).shape[:2]
        out_hw = (int(H * (scale or 2)), int(W * (scale or 2)))
    return lanczos_upscale(img, out_hw)
