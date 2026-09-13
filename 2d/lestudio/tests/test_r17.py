"""tests/test_r17.py -- the leCore bridge (R17).

Devin: 'see what you can do with the leCore tech we have to build with' --
HDRIFT (holographic drift generation), colour transfer, the semantic
memory, and the GLSL emitters, wired into leStudio for humans (the
✨ leCore dialog) and agents (four endpoints). Everything degrades
honestly when leCore is absent (503, not a crash)."""
import numpy as np
import pytest


def _need_lecore():
    """Skip (via the runner's pytest shim) when leCore is not importable."""
    try:
        from holographic.materials_and_texture.holographic_colortransfer \
            import color_transfer                              # noqa: F401
    except Exception:
        pytest.skip("leCore not on the path")


def test_r17_style_match_lands_as_a_new_layer():
    """The mood knob: grade toward a reference, never touch the original."""
    _need_lecore()
    pytest.importorskip("flask")
    import base64
    import io
    from PIL import Image
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "sm", "width": 160, "height": 120,
                             "background": [0.9, 0.9, 0.95]})
    lid = srv.DOC.layers[0].id
    c.post("/api/paint", json={"layer": lid, "points": [[10, 40], [150, 60]],
                               "color": [0.2, 0.4, 0.8], "radius": 12,
                               "record": True})
    n0 = len(srv.DOC.layers)
    before = srv.DOC.layer(lid).pixels.copy()
    ref = Image.new("RGB", (40, 30), (250, 140, 30))     # hot orange mood
    buf = io.BytesIO()
    ref.save(buf, "PNG")
    r = c.post("/api/style/match", json={
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
        "strength": 1.0}).get_json()
    assert r.get("ok"), r
    assert len(srv.DOC.layers) == n0 + 1, "match must land as a NEW layer"
    assert np.allclose(srv.DOC.layer(lid).pixels, before), \
        "the original layer must be untouched"
    top = srv.DOC.layers[-1].pixels
    assert float(top[..., 0].mean()) > float(top[..., 2].mean()), \
        "graded toward the orange reference, red should now lead blue"
    # refusal without a reference
    assert c.post("/api/style/match", json={}).status_code == 400


def test_r17_dream_seeds_and_refusal():
    """Dreams from real distinct views; an honest 400 (leCore's degenerate-
    data refusal, relayed with guidance) when there is nothing to learn."""
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "dr", "width": 160, "height": 120,
                             "background": [1, 1, 1]})
    paths = ["/root/work/unplugged_v3.png", "/root/work/golden_hour_lake.png",
             "/root/work/little_orchestra.png", "/root/work/north_lake.png"]
    import os
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("gallery images not present")
    r = c.post("/api/dream", json={"n": 2, "seed": 3,
                                   "paths": paths}).get_json()
    assert r.get("ok") and r["count"] == 2, r
    n0 = len(srv.DOC.layers)
    p = c.post("/api/dream/place", json={"i": 0}).get_json()
    assert p.get("ok") and len(srv.DOC.layers) == n0 + 1
    assert srv.DOC.layers[-1].pixels.shape[:2] == (120, 160)
    # a blank doc has nothing to dream from -- 400 with guidance, not a 500
    c.post("/api/new", json={"name": "blank", "width": 100, "height": 80})
    rr = c.post("/api/dream", json={"n": 2})
    assert rr.status_code == 400
    assert "paint more" in (rr.get_json().get("error") or "")


def test_r17_sage_answers_from_the_taught_lore():
    """The /api/advise endpoint shares the SAME memory store the painting
    sessions teach -- ask an exact taught question, get the taught answer."""
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    r = c.post("/api/advise", json={
        "q": "How can swarm painters coordinate?"})
    if r.status_code == 503:
        pytest.skip("sage memory not present on this machine")
    j = r.get_json()
    assert j.get("ok")
    assert "blackboard" in (j.get("answer") or ""), j
    # teach-through-the-studio: the agents' blackboard, productised
    t = c.post("/api/advise", json={"teach": {
        "q": "r17 pin: what colour is the test wall?",
        "a": "sage green"}}).get_json()
    assert t.get("ok")
    j2 = c.post("/api/advise", json={
        "q": "r17 pin: what colour is the test wall?"}).get_json()
    assert (j2.get("answer") or "") == "sage green"
    assert c.post("/api/advise", json={}).status_code == 400


def test_r17_grade_chain_exports_as_glsl():
    _need_lecore()
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "gl", "width": 100, "height": 80})
    c.post("/api/graph", json={"force": True, "nodes": [
        {"id": "in", "type": "media", "params": {"path": "/dev/null",
                                                 "play": 0}},
        {"id": "gr", "type": "grade",
         "params": {"blackpoint": 0.03, "whitepoint": 0.98, "gamma": 0.96},
         "inputs": {"in": "in"}},
        {"id": "gl", "type": "glow",
         "params": {"threshold": 0.7}, "inputs": {"in": "gr"}},
        {"id": "vg", "type": "vignette",
         "params": {"amount": 0.3, "radius": 0.8}, "inputs": {"in": "gl"}},
        {"id": "out", "type": "output", "inputs": {"in": "vg"}}]})
    r = c.get("/api/graph/export.glsl")
    assert r.status_code == 200
    txt = r.data.decode()
    assert "lestudio_grade" in txt and "vec3" in txt
    assert "APPROXIMATE" in txt, "the export must say it is approximate"
    assert "skipped" in txt and "glow" in txt, \
        "multi-pass nodes must be listed as skipped, not silently dropped"


def test_r17_bridge_absent_is_a_503_not_a_crash():
    """With leCore unavailable the endpoints answer 503 with a reason."""
    pytest.importorskip("flask")
    import lestudio.server as srv
    saved = dict(srv._LECORE)
    try:
        srv._LECORE.clear()
        srv._LECORE.update(tried=True, err="unit test says no")
        c = srv.app.test_client()
        for ep, body in (("/api/style/match", {"path": "/tmp/x.png"}),
                         ("/api/dream", {"n": 2})):
            r = c.post(ep, json=body)
            assert r.status_code == 503, ep
            assert "unit test says no" in r.get_json()["error"]
    finally:
        srv._LECORE.clear()
        srv._LECORE.update(saved)


def test_r17_ui_has_the_lecore_dialog():
    import os
    import lestudio.server as srv
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    for needle in ('id="lcBtn"', "/api/style/match", "/api/dream",
                   "/api/dream/place", "/api/advise",
                   "/api/graph/export.glsl", "lcSageQ", "lcDreamStrip"):
        assert needle in ui, needle
