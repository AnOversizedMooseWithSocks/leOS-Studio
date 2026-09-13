import numpy as np
def denoise(img, sigma=1.0):
    img = np.asarray(img, float)
    if sigma <= 0: return img
    k = np.array([1.0, 2.0, 1.0]); k = k / k.sum()
    out = img.copy()
    for ax in (0, 1):
        pad = np.pad(out, [(1,1) if a == ax else (0,0) for a in range(out.ndim)], mode="edge")
        sl = [slice(None)] * out.ndim
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl[ax] = slice(i, i + out.shape[ax])
            acc += w * pad[tuple(sl)]
        out = acc
    return out
def sharpen(img, amount=0.5):
    img = np.asarray(img, float)
    return np.clip(img + float(amount) * (img - denoise(img, 1.0)), 0, 1)
