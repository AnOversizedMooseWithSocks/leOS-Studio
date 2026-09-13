"""tests/test_r27.py -- clearing a layer clears its living medium (R33).

Painting the dragon's breath, a smoke layer was cleared and repainted
three times -- and every media step brought the CLEARED plumes back.
Document.clear zeroed pixel alpha only; the medium's density and dye
live in l._media on the sim grid, and the next _media_slab_step
re-rendered them into the freshly cleared pixels. A whole-layer clear
now drops the sim state; a selection clear gates density and dye by the
selection resampled onto the sim grid.
"""
import numpy as np


def _smoke_doc():
    from lestudio import Document
    d = Document(128, 96)
    l = d.add_layer("plume")
    d.edit_layer(l.id, vol_kind="smoke", thickness=6.0)
    d.paint(l.id, [[40, 60], [44, 62]], color=(0.9, 0.9, 0.9),
            radius=12.0, opacity=0.9, hardness=0.4)
    return d, l


def test_r27_clear_kills_the_medium():
    from lestudio import _media_slab_step
    d, l = _smoke_doc()
    _media_slab_step(d, l, 2)
    assert float(l.pixels[..., 3].max()) > 0.05, "smoke must render"
    d.clear(l.id)
    assert getattr(l, "_media", None) is None, \
        "whole-layer clear must drop the sim state"
    _media_slab_step(d, l, 2)
    # a fresh (empty) medium renders nothing back
    assert float(l.pixels[..., 3].max()) < 0.02, \
        "cleared smoke must STAY cleared through a media step"


def test_r27_selection_clear_gates_the_density():
    from lestudio import _media_slab_step
    d, l = _smoke_doc()
    _media_slab_step(d, l, 1)
    den0 = float(l._media["den"].sum())
    assert den0 > 0
    # a selection covering everything: same contract as a full clear
    from lestudio import Selection
    sel = Selection(d.height, d.width,
                    data=np.ones((d.height, d.width), np.float32))
    d.selections.append(sel)
    d.clear(l.id, selection=sel.id)
    assert float(l._media["den"].sum()) < den0 * 0.01, \
        "a full-cover selection clear must empty the density too"


def test_r27_plain_layers_clear_as_before():
    from lestudio import Document
    d = Document(64, 48)
    l = d.add_layer("paint")
    d.paint(l.id, [[10, 10], [30, 30]], color=(1, 0, 0), radius=6.0)
    d.clear(l.id)
    assert float(l.pixels[..., 3].max()) == 0.0


def test_r27_water_glazes_stain_but_do_not_pile():
    """R33 user report: 'an emboss effect with no color change'. Thirty
    low-opacity water glazes were building a full impasto crust; a wash
    stains, it should leave ~a tenth of a stiff paint's body."""
    from lestudio import Document
    d = Document(160, 120)
    lw = d.add_layer("washes")
    lo = d.add_layer("oils")
    pts = [[30, 60], [130, 60]]
    for i in range(30):
        d.paint(lw.id, pts, color=(0.2, 0.3, 0.3), radius=16.0,
                opacity=0.2, media="water", load=0.5)
        d.paint(lo.id, pts, color=(0.2, 0.3, 0.3), radius=16.0,
                opacity=0.2, media="oil", load=0.5)
    hwat = float(lw.height_map.max()) if lw.height_map is not None else 0.0
    hoil = float(lo.height_map.max()) if lo.height_map is not None else 0.0
    assert hoil > 0.3, "oil glazes must still build body (%.3f)" % hoil
    assert hwat < hoil * 0.25, \
        "water body (%.3f) must stay well under oil body (%.3f)" % (hwat,
                                                                    hoil)
    # and the pigment still lands: the wash is visible
    assert float(lw.pixels[..., 3].max()) > 0.3
