"""tests/test_r23.py -- the silent-graph trap (R24 finding, pinned).

Found grading 'Abyssal Lanterns': node types are capitalised ('Grade',
'Glow', 'Media in'), a lowercase graph evaluates to an error, and
/api/graph/output.png silently fell back to the raw composite -- three
rounds of 'graded' finals were never graded, with no error anywhere.
Pins: the POST names unknown types, a broken explicit graph 409s instead
of impersonating success, real types actually transform pixels, and the
untouched default graph keeps its composite convenience."""
import numpy as np
import pytest


def test_r23_unknown_node_types_are_named_at_post():
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gt", "width": 120, "height": 90})
    r = c.post("/api/graph", json={"force": True, "nodes": [
        {"id": "in", "type": "media", "params": {"path": "/dev/null"}},
        {"id": "gr", "type": "grade", "params": {}, "inputs": {"in": "in"}},
        {"id": "out", "type": "output", "inputs": {"in": "gr"}}]}).get_json()
    assert r.get("ok")
    assert "warning" in r and "unknown" in r, r
    assert set(r["unknown"]) == {"media", "grade", "output"}
    assert "capitalised" in r["warning"]


def test_r23_broken_graph_refuses_instead_of_impersonating():
    """The composite fallback was the trap: a failing explicit graph must
    409 with the reason, never serve pixels that pretend the grade ran."""
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gb", "width": 120, "height": 90})
    c.post("/api/graph", json={"force": True, "nodes": [
        {"id": "in", "type": "media", "params": {"path": "/dev/null"}},
        {"id": "out", "type": "output", "inputs": {"in": "in"}}]})
    r = c.get("/api/graph/output.png")
    assert r.status_code == 409, "must refuse, not impersonate the composite"
    assert "did not evaluate" in (r.get_json() or {}).get("error", "")


def test_r23_real_types_actually_grade():
    pytest.importorskip("flask")
    pytest.importorskip("PIL")
    import io
    import os
    import tempfile
    from PIL import Image
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gr", "width": 120, "height": 90,
                             "background": [0.5, 0.5, 0.5]})
    lid = srv.DOC.layers[0].id
    c.post("/api/paint", json={"layer": lid, "points": [[10, 40], [110, 50]],
                               "color": [0.9, 0.6, 0.2], "radius": 12,
                               "record": True})
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        src_path = f.name
    raw = c.get("/api/export.png").data
    open(src_path, "wb").write(raw)
    try:
        c.post("/api/graph", json={"force": True, "nodes": [
            {"id": "in", "type": "Media in",
             "params": {"source": src_path, "play": 0}},
            {"id": "gl", "type": "Glow",
             "params": {"threshold": 0.2, "radius": 30, "intensity": 2.0},
             "inputs": {"image": "in"}},
            {"id": "out", "type": "Output", "inputs": {"image": "gl"}}]})
        r = c.get("/api/graph/output.png")
        assert r.status_code == 200
        a = np.asarray(Image.open(io.BytesIO(r.data)).convert("RGB"), float)
        b = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), float)
        if a.shape != b.shape:
            b = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")
                           .resize((a.shape[1], a.shape[0])), float)
        assert float(np.abs(a - b).mean()) > 2.0, \
            "a real Glow at intensity 2 must visibly change the pixels"
    finally:
        os.unlink(src_path)


def test_r23_default_graph_keeps_the_composite_convenience():
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gd", "width": 100, "height": 80,
                             "background": [1, 1, 1]})
    srv.GRAPH.set_graph([])            # untouched default
    r = c.get("/api/graph/output.png")
    assert r.status_code == 200 and r.data[:4] == b"\x89PNG"


def test_r23_glsl_export_accepts_real_type_names():
    try:
        from holographic.rendering.holographic_postfx import chain_to_glsl  # noqa
    except Exception:
        pytest.skip("leCore not on the path")
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gx", "width": 100, "height": 80})
    c.post("/api/graph", json={"force": True, "nodes": [
        {"id": "in", "type": "Media in", "params": {"source": "/dev/null",
                                                    "play": 0}},
        {"id": "gr", "type": "Grade",
         "params": {"blackpoint": 0.03, "whitepoint": 0.97, "gamma": 0.95},
         "inputs": {"image": "in"}},
        {"id": "vg", "type": "Vignette", "params": {"amount": 0.3},
         "inputs": {"image": "gr"}},
        {"id": "out", "type": "Output", "inputs": {"image": "vg"}}]})
    r = c.get("/api/graph/export.glsl")
    assert r.status_code == 200
    assert "lestudio_grade" in r.data.decode()
