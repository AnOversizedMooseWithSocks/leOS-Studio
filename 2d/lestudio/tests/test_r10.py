"""tests/test_r10.py -- pins for the R10 abstract session's findings:
the replay GIF export job (progress + download UX), and the stroke-frame
arc-stride fix (big-brush curved strokes framed by giant chords whose
rectangular clip windows stamped axis-aligned bites into the deposit)."""
import time

import numpy as np
import pytest


def _spiral(cx, cy, r, n=44, turns=2.4):
    import math
    return [[cx + r * (1 - 0.82 * i / (n - 1)) * math.cos(i / (n - 1) * turns * 6.283),
             cy + r * (1 - 0.82 * i / (n - 1)) * math.sin(i / (n - 1) * turns * 6.283),
             1.05] for i in range(n)]


def test_r10_stroke_frame_strides_by_arc_not_points():
    """paint() densifies at ~radius*0.35 px between points, so striding the
    frame walk by POINT COUNT built chords hundreds of pixels long on big
    brushes: the chords cut corners, and points ON the true curve measured
    as far off-axis. Pin: for a curved path sampled the way paint() samples
    it, every point of the path itself must sit near the stroke's own axis
    (|u| small)."""
    import math
    from lestudio import _stroke_frame

    n = 44
    dense = [(250 + 90 * math.cos(i / (n - 1) * 5.5),
              200 + 90 * math.sin(i / (n - 1) * 5.5)) for i in range(n)]
    dwid = [1.0] * n
    radius = 33
    x0b, y0b, x1b, y1b = 100, 40, 400, 360
    fr = _stroke_frame(dense, dwid, radius, x0b, y0b, x1b, y1b)
    assert fr is not None
    u = fr[0]
    worst = 0.0
    for (px, py) in dense:
        ix, iy = int(px) - x0b, int(py) - y0b
        if 0 <= ix < u.shape[1] and 0 <= iy < u.shape[0]:
            worst = max(worst, abs(float(u[iy, ix])))
    # pre-fix the giant chords put true-curve points at |u| > 1 (outside
    # the stroke entirely); with arc-length chords they hug the axis
    assert worst < 0.55, worst


def test_r10_heavy_curved_stroke_has_no_axis_aligned_cliffs():
    """The visible symptom: razor-straight vertical/horizontal height
    cliffs inside a self-overlapping heavy oil stroke. Count long straight
    runs of big one-pixel height jumps -- brush ridges are curved, the bug's
    window edges were not."""
    from lestudio import Document
    d = Document(400, 500)
    lid = d.layers[0].id
    d.paint(lid, _spiral(150, 200, 70), color=(0.13, 0.42, 0.75), radius=33,
            opacity=1.0, hardness=0.7, media="oil", load=1.1)
    hm = d.layer(lid).height_map
    jump_v = np.abs(np.diff(hm, axis=1)) > 0.5      # vertical edges
    jump_h = np.abs(np.diff(hm, axis=0)) > 0.5      # horizontal edges
    def longest_run(cols):
        best = 0
        for c in range(cols.shape[1]):
            col = cols[:, c]
            run = mx = 0
            for v in col:
                run = run + 1 if v else 0
                mx = max(mx, run)
            best = max(best, mx)
        return best
    assert longest_run(jump_v) < 40, "straight vertical cliff in deposit"
    assert longest_run(jump_h.T) < 40, "straight horizontal cliff in deposit"


def test_r10_replay_render_job_end_to_end():
    """The GIF export UX: start -> poll progress -> download. The job holds
    the document while rebuilding and reports stroke-level progress, and
    the result endpoint serves a well-formed GIF."""
    pytest.importorskip("flask")
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "tl", "width": 200, "height": 150,
                             "background": [1, 1, 1]})
    lid = srv.DOC.layers[0].id
    for k in range(4):
        c.post("/api/paint", json={"layer": lid,
                                   "points": [[10, 20 + k * 30], [180, 25 + k * 30]],
                                   "color": [k * 0.2, 0.3, 0.6], "radius": 6,
                                   "record": True})
    r = c.post("/api/replay/render", json={"frames": 4, "w": 100, "fps": 6,
                                           "hold": 2})
    assert r.status_code == 200
    for _ in range(200):
        s = c.get("/api/replay/render/status").get_json()
        if s["state"] != "running":
            break
        time.sleep(0.05)
    assert s["state"] == "done", s
    assert s["total"] >= 4 and s["done"] == s["total"]
    g = c.get("/api/replay/render/result.gif")
    assert g.status_code == 200 and g.mimetype == "image/gif"
    im = Image.open(io.BytesIO(g.data))
    assert im.n_frames >= 4 and im.size[0] == 100
    # a second start while idle is allowed; while running it would 409
    r2 = c.post("/api/replay/render", json={"frames": 2, "w": 80})
    assert r2.status_code == 200
    for _ in range(200):
        s = c.get("/api/replay/render/status").get_json()
        if s["state"] != "running":
            break
        time.sleep(0.05)
    assert s["state"] == "done"


def test_r10_replay_render_refuses_empty_and_missing():
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "empty", "width": 100, "height": 80})
    assert c.post("/api/replay/render", json={}).status_code == 400
    srv._REPLAY_JOB.clear()
    srv._REPLAY_JOB["state"] = "idle"
    assert c.get("/api/replay/render/result.gif").status_code == 404


def test_r10_ui_replay_dialog_exports_gif():
    """The button opens a dialog with options, progress, and an explicit
    Save GIF download -- not a right-click scavenger hunt."""
    import os
    import lestudio.server as srv
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    for needle in ('id="replayBtn"', "/api/replay/render", "rpGo", "rpSave",
                   "download=", "/api/replay/render/status"):
        assert needle in ui, needle


def test_r11_paint_warns_when_stroke_lands_under_opaque_layer():
    """Found swarm-painting the portrait: a fix painted on a layer BELOW
    the flaw succeeds silently and the picture does not change. The paint
    response now warns when a visible, opaque, normal-blend layer above
    covers the stroked area — and stays quiet when the cover is
    translucent or absent."""
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "occ", "width": 200, "height": 160,
                             "background": [1, 1, 1]})
    lo = srv.DOC.layers[0].id
    hi = c.post("/api/layer", json={"action": "add", "name": "cover"}
                ).get_json()["id"]
    # opaque cover over the left half
    for yy in range(10, 150, 10):
        c.post("/api/paint", json={"layer": hi, "points": [[5, yy], [95, yy]],
                                   "color": [0.2, 0.5, 0.3], "radius": 8,
                                   "opacity": 1.0, "hardness": 0.9,
                                   "record": True})
    r = c.post("/api/paint", json={"layer": lo, "points": [[20, 60], [80, 70]],
                                   "color": [1, 0, 0], "radius": 5,
                                   "record": True}).get_json()
    assert "UNDER" in (r.get("warning") or ""), r
    assert "cover" in r["warning"]
    # painting where the cover is absent stays quiet
    r2 = c.post("/api/paint", json={"layer": lo,
                                    "points": [[130, 60], [180, 70]],
                                    "color": [1, 0, 0], "radius": 5,
                                    "record": True}).get_json()
    assert not r2.get("warning"), r2
    # a translucent cover shows the stroke through: no warning
    c.post("/api/layer", json={"action": "edit", "id": hi, "opacity": 0.5})
    r3 = c.post("/api/paint", json={"layer": lo, "points": [[20, 90], [80, 95]],
                                    "color": [0, 0, 1], "radius": 5,
                                    "record": True}).get_json()
    assert not r3.get("warning"), r3


def test_r12_paint_fills_the_canvas_tooth():
    """User report: 'the painting ends up looking like it's made out of
    canvas material.' Two causes, both pinned here. (1) The dry-brush gate
    read the STATIC weave, so every thin pass was re-stamped by the same
    tooth at the same phase -- now the gate scales with canvas EXPOSURE:
    a thin stroke over a built-up film lands smooth, while the same stroke
    on bare canvas still dry-brushes. (2) The weave was modelled as deep
    as a whole stroke and buried too slowly (_CANVAS_RELIEF/_PAINT_LEVEL)."""
    import lestudio
    from lestudio import Document
    # the constants: primed canvas, one honest coat buries the weave
    assert lestudio._CANVAS_RELIEF <= 0.20
    assert lestudio._PAINT_LEVEL <= 0.25

    # THE RENDER (what the user saw): a uniformly painted passage must go
    # smooth once one honest coat of height is on it, while a thin wash
    # still shows the weave.
    from lestudio import _shaded_pixels

    def shaded_ripple(h):
        d = Document(200, 140)
        d.add_layer("p")
        l = d.layer(d.layers[-1].id)
        l.pixels[..., :3] = [0.3, 0.4, 0.6]
        l.pixels[..., 3] = 1.0
        l.height_map = np.full((140, 200), h, np.float32)
        out = _shaded_pixels(l)
        return float(out[20:120, 20:180, :3].mean(-1).std())

    wash = shaded_ripple(0.05)
    coat = shaded_ripple(0.6)
    assert wash > 0.005, wash                     # bare-ish canvas shows tooth
    assert coat < wash * 0.15 and coat < 0.002, (wash, coat)

    # THE GATE: the same starved stroke breaks less over a filled film
    # than on bare canvas (deterministic under the fixed seed)
    def starved_ripple(prefill):
        d = Document(320, 120)
        d.add_layer("p")
        lid = d.layers[-1].id
        if prefill:
            d.layer(lid).height_map = np.full((120, 320), 1.0, np.float32)
        d.paint(lid, [(15, 52), (160, 54), (300, 52)],
                color=(0.9, 0.1, 0.1), radius=12, opacity=0.95,
                hardness=0.3, media="oil", load=0.3)
        a = d.layer(lid).pixels[46:60, 40:280, 3]
        return float(np.std(a))

    bare = starved_ripple(False)
    filled = starved_ripple(True)
    assert bare > 0.10, bare                      # dry-brush survives
    assert filled < bare, (bare, filled)          # the film fills the tooth
