"""tests/test_r18.py -- the splat audit round.

Devin: 'leCore can represent a 3d scene as 3d splats and generate an
entire 2d image as splats... we are under utilizing HDRIFT.' Audited the
suite (aniso_fit/densify_fit are full n-D 3DGS-grade primitives; the
HDRIFT image adapter used 4 grayscale numbers per splat), and shipped:
splatify (a painting as K colour splats, with the compact code returned --
generate instead of store), standard 3DGS .ply export, and dreams that
drift in colour-splat space, deterministic in seed."""
import numpy as np
import pytest


def _need_lecore():
    try:
        from holographic.rendering.holographic_splat import splat_fit  # noqa
    except Exception:
        pytest.skip("leCore not on the path")


def test_r18_splatify_lands_layer_and_returns_the_code():
    """The abstraction dial -- and the code is the deliverable: ~7 floats
    per splat that regenerate the layer deterministically anywhere."""
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "sp", "width": 200, "height": 140,
                             "background": [0.9, 0.9, 0.95]})
    lid = srv.DOC.layers[0].id
    for i in range(3):
        c.post("/api/paint", json={"layer": lid,
                                   "points": [[10, 20 + 34 * i],
                                              [190, 26 + 34 * i]],
                                   "color": [0.2 * i, 0.5, 0.8 - 0.2 * i],
                                   "radius": 10, "record": True})
    n0 = len(srv.DOC.layers)
    r = c.post("/api/splatify", json={"k": 48}).get_json()
    assert r.get("ok") and r["k"] == 48
    assert len(srv.DOC.layers) == n0 + 1
    code = r["code"]
    assert len(code["splats"]) == 48 and len(code["colors"]) == 48
    assert len(code["splats"][0]) == 3 and len(code["colors"][0]) == 3
    # the code regenerates the layer: render it and compare to the layer
    from holographic.rendering.holographic_splat import _gaussian
    sh, sw = code["shape"]
    out = np.zeros((sh, sw, 3))
    for (cy, cx, sg), col in zip(code["splats"], code["colors"]):
        out += _gaussian((sh, sw), cy, cx, sg)[..., None] * \
            np.asarray(col)[None, None, :]
    out = np.clip(out, 0, 1)
    from PIL import Image
    up = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize(
        (srv.DOC.width, srv.DOC.height), Image.LANCZOS), np.float32) / 255.0
    lay = srv.DOC.layers[-1].pixels[..., :3]
    assert float(np.abs(up - lay).mean()) < 0.02, \
        "the compact code must regenerate the layer"


def test_r18_splats_export_standard_ply_and_json():
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "pl", "width": 160, "height": 120,
                             "background": [0.8, 0.85, 0.9]})
    lid = srv.DOC.layers[0].id
    c.post("/api/paint", json={"layer": lid, "points": [[10, 60], [150, 66]],
                               "color": [0.7, 0.3, 0.2], "radius": 12,
                               "record": True})
    r = c.get("/api/splats/export.ply?k=32")
    assert r.status_code == 200 and r.data.startswith(b"ply"), \
        "must be a standard 3DGS .ply any splat viewer opens"
    j = c.get("/api/splats/export.ply?k=16&fmt=json")
    assert j.status_code == 200
    import json as _json
    parsed = _json.loads(j.data)
    assert len(parsed["splats"]) == 16
    assert {"position", "scale", "rotation", "color"} <= set(parsed["splats"][0])


def test_r18_dreams_are_deterministic_in_seed():
    """Devin: 'our system is deterministic, so anything we can generate
    instead of store should help' -- the same views + seed must produce
    byte-identical dreams, so a seed IS the artifact."""
    _need_lecore()
    pytest.importorskip("flask")
    import os
    import lestudio.server as srv
    paths = ["/root/work/unplugged_v3.png", "/root/work/golden_hour_lake.png",
             "/root/work/little_orchestra.png", "/root/work/north_lake.png"]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("gallery images not present")
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "dd", "width": 160, "height": 120})
    a = c.post("/api/dream", json={"n": 2, "seed": 9,
                                   "paths": paths}).get_json()
    b = c.post("/api/dream", json={"n": 2, "seed": 9,
                                   "paths": paths}).get_json()
    assert a.get("ok") and b.get("ok")
    assert a["seed"] == 9
    assert a["thumbs"] == b["thumbs"], "same seed must mean same dreams"
    # and dreams carry colour now (not the old grayscale blobs colorized):
    # decode one thumb and check the channels actually differ
    import base64
    import io
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(
        base64.b64decode(a["thumbs"][0]))).convert("RGB"), np.float32)
    assert float(np.abs(im[..., 0] - im[..., 2]).mean()) > 1.0, \
        "dreams should carry real colour structure"


def test_r18_ui_has_splatify():
    import os
    import lestudio.server as srv
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    for needle in ("lcSplGo", "lcSplK", "/api/splatify",
                   "/api/splats/export.ply"):
        assert needle in ui, needle
