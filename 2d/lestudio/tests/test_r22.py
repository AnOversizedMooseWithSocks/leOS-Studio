"""tests/test_r22.py -- depth-aware splats (R22).

A layered painting is not flat: each pane sits at a real depth
(accumulated thickness + z_off). ?scope=layers exports one splat code per
visible painted layer, LIFTED to its pane's z -- a glass painting leaves
as a true 3D splat sculpture whose parallax survives in any 3DGS viewer."""
import json

import numpy as np
import pytest


def _need_lecore():
    try:
        from holographic.rendering.holographic_splat import splat_fit  # noqa
    except Exception:
        pytest.skip("leCore not on the path")


def test_r22_layer_scope_exports_true_depth():
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gl3", "width": 200, "height": 140,
                             "background": [0.9, 0.9, 0.95]})
    ids = [srv.DOC.layers[0].id]
    for nm in ("mid", "front"):
        ids.append(c.post("/api/layer", json={"action": "add",
                                              "name": nm}).get_json()["id"])
    for i, lid in enumerate(ids):
        c.post("/api/layer", json={"action": "edit", "id": lid,
                                   "thickness": 2.0})
        c.post("/api/paint", json={"layer": lid,
                                   "points": [[20, 30 + 40 * i],
                                              [180, 36 + 40 * i]],
                                   "color": [0.2 + 0.3 * i, 0.5,
                                             0.8 - 0.3 * i],
                                   "radius": 10, "record": True})
    r = c.get("/api/splats/export.ply?scope=layers&k=90&fmt=json")
    assert r.status_code == 200
    j = json.loads(r.data)
    zs = sorted(set(round(s["position"][2], 1) for s in j["splats"]))
    assert len(zs) == 3, "three panes must land on three z planes: %s" % zs
    assert zs[0] == 0.0 and zs[1] > 0 and zs[2] > zs[1]
    # and the binary .ply is a real 3DGS file
    r2 = c.get("/api/splats/export.ply?scope=layers&k=90")
    assert r2.status_code == 200 and r2.data.startswith(b"ply")
    # an empty document refuses honestly
    c.post("/api/new", json={"name": "empty", "width": 100, "height": 80})
    for l in srv.DOC.layers:
        l.pixels[..., 3] = 0.0
    assert c.get("/api/splats/export.ply?scope=layers").status_code == 400


def test_r22_flat_scope_still_works():
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "fl", "width": 160, "height": 120,
                             "background": [0.8, 0.85, 0.9]})
    lid = srv.DOC.layers[0].id
    c.post("/api/paint", json={"layer": lid, "points": [[10, 60], [150, 66]],
                               "color": [0.7, 0.3, 0.2], "radius": 12,
                               "record": True})
    r = c.get("/api/splats/export.ply?k=32")
    assert r.status_code == 200 and r.data.startswith(b"ply")
