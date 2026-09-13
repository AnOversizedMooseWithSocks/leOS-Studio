"""tests/test_r9.py -- pins for painting PLAYBACK (the stroke timelapse)
and the pressure-sensitivity work that proved it out.

Every stroke has always been recorded (that is what stroke editing rides
on); R9 adds Document.timelapse_frames() -- the document rebuilt from each
painted layer's replay base with every recorded stroke re-applied in global
paint order -- and /api/replay/* to serve it as an animated GIF, with a
[?] Replay button in the timebar."""
import numpy as np
import pytest


def _tiny_doc():
    from lestudio import Document
    d = Document(160, 120)
    lid = d.layers[0].id
    d.add_layer("top")
    l2 = d.layers[-1].id
    d.paint(lid, [(10, 10), (150, 20)], color=(1, 0, 0), radius=8)
    d.paint(l2, [(20, 80, 1.6), (150, 90, 0.15)], color=(0, 0, 1), radius=10)
    d.paint(lid, [(10, 110), (150, 100)], color=(0, 1, 0), radius=8)
    return d, lid, l2


def test_r9_timelapse_progresses_and_ends_honest():
    """Frames change monotonically toward the truth: every step differs,
    the final frame IS the current composite, and the document is left
    exactly as it was."""
    d, lid, l2 = _tiny_doc()
    before = d.composite().copy()
    fr = list(d.timelapse_frames(frames=4))
    assert len(fr) >= 3
    for a, b in zip(fr, fr[1:]):
        assert float(np.abs(a - b).sum()) > 0
    assert np.allclose(fr[-1], before)
    assert np.array_equal(d.composite(), before)


def test_r9_timelapse_is_chronological_not_layerwise():
    """The stroke list interleaves layers in paint order, and playback
    follows it: the blue stroke (painted second, on the TOP layer) must
    appear in the middle frame while the green stroke (painted third, on
    the BOTTOM layer) is still absent."""
    d, lid, l2 = _tiny_doc()
    fr = list(d.timelapse_frames(frames=3))
    mid = fr[len(fr) // 2]
    blue = (mid[..., 2] > 0.5) & (mid[..., 0] < 0.4) & (mid[..., 3] > 0.1)
    green = (mid[..., 1] > 0.5) & (mid[..., 0] < 0.4) & (mid[..., 2] < 0.4) \
        & (mid[..., 3] > 0.1)
    assert blue.any() and not green.any()


def test_r9_pressure_tapers_programmatic_strokes():
    """A point may carry (x, y, pressure); the API accepts it and the
    stroke's rendered width follows the ramp -- wide at 1.6, hairline at
    0.15. This is the same channel a stylus feeds."""
    d, lid, l2 = _tiny_doc()
    a = d.composite()
    blue = (a[..., 2] > 0.5) & (a[..., 0] < 0.4) & (a[..., 3] > 0.1)
    cols = np.where(blue.any(0))[0]
    assert len(cols) > 40
    left = blue[:, cols[:15]].sum(0).mean()
    right = blue[:, cols[-15:]].sum(0).mean()
    assert left > right * 2.5, (left, right)


def test_r9_timelapse_survives_deleted_layer():
    """Strokes whose layer is gone are skipped, not crashed on."""
    d, lid, l2 = _tiny_doc()
    d.remove_layer(l2)
    fr = list(d.timelapse_frames(frames=3))
    assert np.allclose(fr[-1], d.composite())


def test_r9_replay_endpoints_serve_playback():
    pytest.importorskip("flask")
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    import lestudio.server as srv
    c = srv.app.test_client()
    with srv._DOC_LOCK:
        lid = srv.DOC.layers[0].id
        srv.DOC.paint(lid, [(5, 5), (100, 40)], color=(1, 0, 0), radius=6)
        srv.DOC.paint(lid, [(5, 60), (100, 80)], color=(0, 0, 1), radius=6)
    info = c.get("/api/replay/info").get_json()
    assert info["ok"] and info["strokes"] >= 2
    assert any(l["from_base"] for l in info["layers"])
    r = c.get("/api/replay/timelapse.gif?frames=3&w=80&fps=5&hold=1")
    assert r.status_code == 200 and r.mimetype == "image/gif"
    im = Image.open(io.BytesIO(r.data))
    assert im.n_frames >= 3
    assert im.size[0] == 80


def test_r9_ui_has_replay_button():
    import lestudio.server as srv
    import os
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    assert 'id="replayBtn"' in ui and "/api/replay/info" in ui
    # R10 moved the UI from the one-shot GET to the render job (progress +
    # explicit download); the GET stays as a scripting door
    assert "/api/replay/render" in ui


def test_r9_stroke_budget_fits_a_painting_session():
    """512 fit a sketch, not a painting -- the Golden Hour Lake session
    recorded ~4.6k strokes. The cap must hold a real session or playback
    silently starts mid-painting."""
    from lestudio import Document
    assert Document.MAX_STROKES >= 4096
