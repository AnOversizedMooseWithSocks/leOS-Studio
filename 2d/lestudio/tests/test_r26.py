"""tests/test_r26.py -- the anisotropic colour-splat codec (dream v2).

Devin: 'the HDRIFT result should be high quality/complexity, not just
abstract dots depicting nothing tangible.' The pins that keep it so:
the staged coarse-to-fine fit actually reconstructs structure (a floor
on PSNR for a structured synthetic scene), the ridge refit keeps
amplitudes drift-safe (the +/-200 cancellation pairs of the weak-ridge
solve are the R21 failure relearned in colour), and the code is
deterministic so seeds can be passed around instead of pixels."""
import numpy as np


def _scene():
    """A structured synthetic: gradient sky, dark wall, bright diagonal
    stroke, dots -- the shapes the fitter must not blur into mush."""
    H, W = 184, 132
    img = np.zeros((H, W, 3), np.float32)
    t = (np.mgrid[0:H, 0:W][0] / H)[..., None]
    img += np.asarray([0.5, 0.6, 0.8]) * (1 - t) + \
        np.asarray([0.05, 0.05, 0.12]) * t
    img[:, :30] = [0.08, 0.07, 0.14]                      # wall
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.abs((yy - 40) * 0.6 - (xx - 20))               # diagonal stroke
    img[d < 3] = [0.95, 0.2, 0.15]
    rng = np.random.RandomState(5)
    for _ in range(40):                                   # window dots
        y, x = rng.randint(10, H - 2), rng.randint(2, W - 2)
        img[y:y + 2, x:x + 2] = [0.9, 0.7, 0.3]
    return np.clip(img, 0, 1)


def test_r26_staged_fit_reconstructs_structure():
    from lestudio.hdrift_aniso import fit_color_splats, refit_psnr
    img = _scene()
    F, _ = fit_color_splats(img, K=128)
    p = refit_psnr(img, F)
    assert p > 15.0, "staged fit lost the structure (psnr %.2f)" % p


def test_r26_amplitudes_stay_drift_safe():
    """The weak-ridge joint refit built +/-200 cancellation pairs that
    exploded the moment the drift decorrelated them. The relative ridge
    must keep every amplitude in a sane band."""
    from lestudio.hdrift_aniso import fit_color_splats
    F, _ = fit_color_splats(_scene(), K=128)
    amps = F[:, 2:5]
    assert np.abs(amps).max() < 4.0, \
        "cancellation-coupled amplitudes are back (max %.1f)" % \
        np.abs(amps).max()


def test_r26_code_is_deterministic():
    from lestudio.hdrift_aniso import fit_color_splats
    img = _scene()
    F1, _ = fit_color_splats(img, K=96)
    F2, _ = fit_color_splats(img, K=96)
    assert np.allclose(F1, F2), "same image must give the same code"


def test_r26_render_scales_with_resolution():
    """Splat codes are resolution-free: decoding at 2x must be the same
    picture, larger -- pinned by comparing downsampled means."""
    from lestudio.hdrift_aniso import fit_color_splats, render_splats
    img = _scene()
    F, _ = fit_color_splats(img, K=96)
    a = render_splats(F, img.shape[:2])
    F2 = F.copy()
    F2[:, 0] *= 2
    F2[:, 1] *= 2
    F2[:, 5:8] *= 2
    b = render_splats(F2, (img.shape[0] * 2, img.shape[1] * 2))
    bs = b[::2, ::2]
    assert abs(float(a.mean()) - float(bs.mean())) < 0.02
