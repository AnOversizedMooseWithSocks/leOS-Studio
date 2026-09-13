"""tests/test_r50.py -- the Textile tool: thread depth + live preview.

R50 (user commission): a dedicated textile tool -- thread properties,
stitch pattern, DEPTH of stitching, an intuitive UI with a preview.
Engine side: `depth` on the textile generators lays oil-bodied strokes
(threads get real paint height, shaded by the layer's relief light);
`/api/textile/preview.png` renders a swatch with the REAL generators on
a scratch document, never touching the workspace or the journal.
"""
import numpy as np


def _doc():
    from lestudio import Document
    d = Document(200, 150)
    d.layers[0].pixels[..., :3] = 0.95
    d.layers[0].pixels[..., 3] = 1.0
    return d


def test_r50_depth_raises_the_thread():
    d = _doc()
    L = d.add_layer("t").id
    d.hatch_fill(L, 100, 75, radius=55, mode="stitch", depth=0.8, seed=5)
    hm = d.layer(L).height_map
    assert hm is not None and float(np.abs(hm).max()) > 1e-4, \
        "depth>0 must deposit real paint body, not just pigment"
    assert d.replay_is_faithful(L), \
        "bodied threads must still replay from the journal"
    # and depth=0 stays flat
    d2 = _doc()
    L2 = d2.add_layer("t").id
    d2.hatch_fill(L2, 100, 75, radius=55, mode="stitch", depth=0.0, seed=5)
    hm2 = d2.layer(L2).height_map
    assert hm2 is None or float(np.abs(hm2).max()) < 1e-6, \
        "depth=0 must not grow a height field"


def test_r50_depth_is_deterministic_and_differs_from_flat():
    a, b = _doc(), _doc()
    La, Lb = a.add_layer("t").id, b.add_layer("t").id
    a.hatch_fill(La, 100, 75, radius=50, mode="weave", depth=0.6, seed=9)
    b.hatch_fill(Lb, 100, 75, radius=50, mode="weave", depth=0.6, seed=9)
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels)
    flat = _doc()
    Lf = flat.add_layer("t").id
    flat.hatch_fill(Lf, 100, 75, radius=50, mode="weave", depth=0.0, seed=9)
    assert not np.array_equal(a.layer(La).pixels, flat.layer(Lf).pixels), \
        "the body must change the deposit"


def test_r50_textile_preview_is_journal_free():
    from lestudio.server import app, WS
    c = app.test_client()
    doc = WS.docs[WS.active]
    n0 = len(doc.strokes)
    u0 = len(doc._undo)
    r = c.get("/api/textile/preview.png?mode=weave&weave=basket&angle=0"
              "&spacing=8&thickness=1.5&depth=0.5&color=9e2640&seed=3")
    assert r.status_code == 200 and r.data[1:4] == b"PNG" \
        and r.mimetype == "image/png"
    assert len(r.data) > 2000, "swatch suspiciously empty"
    assert len(doc.strokes) == n0 and len(doc._undo) == u0, \
        "the preview must never touch the workspace document"
    # different settings render different swatches
    r2 = c.get("/api/textile/preview.png?mode=cross&angle=0&spacing=8"
               "&thickness=1.5&depth=0&color=9e2640&seed=3")
    assert r2.status_code == 200 and r2.data != r.data


def test_r50_hatchfill_endpoint_accepts_depth_and_weave():
    from lestudio.server import app, WS
    c = app.test_client()
    doc = WS.docs[WS.active]
    lid = doc.layers[0].id
    r = c.post("/api/hatchfill", json={
        "layer": lid, "x": 200, "y": 150, "radius": 60, "mode": "weave",
        "weave": "satin", "depth": 0.7, "spacing": 8, "seed": 4})
    assert r.status_code == 200 and r.json["ok"] and r.json["strokes"] > 5
    hm = doc.layer(lid).height_map
    assert hm is not None and float(np.abs(hm).max()) > 1e-4


def test_r50_generated_fill_is_one_undo_entry():
    # the paint_batch law applied to generators: a fill is ONE record --
    # per-stroke record() was ~70% of a big fill's wall time, and ctrl+Z
    # peeling off one thread of a woven patch was never right
    d = _doc()
    L = d.add_layer("t").id
    u0 = len(d._undo)
    n = d.hatch_fill(L, 100, 75, radius=55, mode="weave", seed=5)
    assert n > 10
    assert len(d._undo) == u0 + 1, \
        "a %d-stroke fill must cost exactly one undo entry" % n
    before = d.layer(L).pixels.copy()
    assert d.undo()
    import numpy as _np
    assert float(d.layer(L).pixels[..., 3].max()) == 0.0, \
        "one undo must take the whole cloth back"
    assert d.redo()
    assert _np.array_equal(d.layer(L).pixels, before), \
        "redo must re-lay every thread"
