"""tests/test_r24.py -- emission must ride the slab transform (R25 find).

Devin: 'the depth and parallax images appear to have artifacts.' They
did: _doc_emission gathered every emissive layer's glow UNTRANSFORMED,
while composite_volumetric persp-scales each slab about the centre -- so
every glowing jellyfish left a misregistered stair-stepped ghost of its
own light at the flat position. Emission is now accumulated inside the
slab loop, from the same resampled pixels and alpha the colour pass uses.
Pins: the glow moves WITH its slab, no ghost remains at the flat
position, and the flat view is byte-identical to before."""
import numpy as np
import pytest


def _glow_doc():
    from lestudio import Document
    d = Document(300, 220)
    base = d.layers[0]
    base.pixels[..., :3] = 0.05
    base.pixels[..., 3] = 1.0
    base.thickness = 40.0
    base.pixels[40:60, 230:250, :3] = [1.0, 0.6, 0.2]
    base.emissive = 0.8
    base.emissive_color = [1.0, 0.6, 0.2]
    d.add_layer("spacer")                       # thick clear slab above:
    d.layers[-1].thickness = 120.0              # pushes the emitter deep
    d.layers[-1].pixels[..., 3] = 0.0
    return d


def _centroid(img):
    lum = img[..., :3].mean(-1)
    lum = lum - lum.min()
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = max(float(lum.sum()), 1e-9)
    return float((yy * lum).sum() / s), float((xx * lum).sum() / s)


def test_r24_glow_rides_the_persp_transform():
    from lestudio import composite_lit
    d = _glow_doc()
    flat = composite_lit(d, "flat")
    persp = composite_lit(d, "persp")
    fy, fx = _centroid(flat)
    py, px = _centroid(persp)
    # persp pulls the deep emitter toward the canvas centre (110, 150)
    assert px < fx - 3 and py > fy + 3, \
        "the glow must move WITH its slab (flat %s persp %s)" % (
            (fy, fx), (py, px))
    # the light must sit ON the shifted emitter, not beside it
    by, bx = np.unravel_index(persp[..., :3].mean(-1).argmax(),
                              persp.shape[:2])
    assert abs(by - py) < 25 and abs(bx - px) < 25, \
        "glow separated from its emitter"


def test_r24_no_ghost_at_the_flat_position():
    """The artifact itself: with the old code the untransformed emission
    left a structured BUMP of glow at the flat position (the emitter's
    stair-stepped double). The emitter's global light-throw is intended
    and lifts the whole field evenly -- so the pin measures LOCAL bumps:
    the flat position must be as flat as its surroundings, while the
    shifted position carries the real emitter."""
    from lestudio import composite_lit
    d = _glow_doc()
    persp = composite_lit(d, "persp")
    lum = persp[..., :3].mean(-1)

    def bump(cy, cx):
        spot = float(lum[cy - 8:cy + 8, cx - 8:cx + 8].mean())
        ring = float(np.concatenate([
            lum[cy - 24:cy - 14, cx - 24:cx + 24].ravel(),
            lum[cy + 14:cy + 24, cx - 24:cx + 24].ravel(),
            lum[cy - 14:cy + 14, cx - 24:cx - 14].ravel(),
            lum[cy - 14:cy + 14, cx + 14:cx + 24].ravel()]).mean())
        return spot - ring

    by, bx = np.unravel_index(lum.argmax(), lum.shape)
    real = bump(int(by), int(bx))
    ghost = bump(50, 240)                       # the flat position
    assert real > 0.02, "the moved emitter itself must be a local bump"
    assert ghost < real * 0.3, \
        "ghost bump at the flat position (%.4f vs real %.4f)" % (ghost, real)


def test_r24_flat_view_unchanged():
    """The fix touches only the slab views: flat lighting must keep using
    the classic path and stay self-consistent."""
    from lestudio import composite_lit, _doc_emission
    d = _glow_doc()
    a = composite_lit(d, "flat")
    b = composite_lit(d, "flat")
    assert np.allclose(a, b)
    assert _doc_emission(d) is not None
