"""tests/test_r16.py -- pins for the R16 performance round.

Devin: 'a lot of memory being consumed by the painting process... we should
be storing deltas.' Three causes found and fixed, each pinned here:
(1) the premultiply-hygiene fill swept the WHOLE layer on every stroke,
    which forced every brush stroke's undo snapshot to copy the full layer
    (~15 MB x 24 entries x N layers -- the OOM);
(2) every snapshot deep-copied every stroke path, invisible to the undo
    budget (O(n) per stroke: painting got slower as the picture grew);
(3) height/material/media planes, brush tips and stamps were not counted
    by the budget, so the trim believed half the truth.
Plus /api/paint_batch: many strokes, one round trip, one undo entry."""
import numpy as np
import pytest

from lestudio import Document


def test_r16_hygiene_fill_is_windowed_after_first_stroke():
    """First stroke on a layer may sweep it (that IS the hygiene); later
    strokes must not touch transparent pixels far from their own box."""
    d = Document(300, 200)
    d.add_layer("p")
    l = d.layer(d.layers[-1].id)
    d.paint(l.id, [(10, 10), (60, 12)], color=(1, 0, 0), radius=5)
    assert getattr(l, "_hyg_filled", False)
    far = l.pixels[150, 250, :3].copy()          # transparent, far away
    d.paint(l.id, [(10, 100), (60, 102)], color=(0, 0, 1), radius=5)
    assert np.allclose(l.pixels[150, 250, :3], far), \
        "second stroke swept the whole layer again"


def test_r16_paint_snapshots_are_region_sized():
    """The undo entry for a paint stroke on an already-hygienic layer holds
    a windowed patch, not the full layer -- this is the memory fix.
    R33: on a replay-clean layer stroke entries store NO pixels at all
    (path-delta + rerender); the windowed snapshot is now the FALLBACK
    for replay-dirty layers, so this pins that fallback."""
    d = Document(400, 300)
    d.add_layer("p")
    lid = d.layers[-1].id
    d.layer(lid)._replay_ok = False        # force the pixel-snapshot path
    d.paint(lid, [(20, 20), (90, 24)], color=(0.2, 0.5, 0.3), radius=6)
    d.paint(lid, [(30, 60), (120, 66)], color=(0.8, 0.2, 0.1), radius=6)
    snap = d._undo[-1][1]
    t = [t for t in snap["layers"] if t[0] == lid][0]
    assert isinstance(t[7], tuple), "pixels: full copy where a region would do"
    (x0, y0, x1, y1), sub = t[7]
    assert (x1 - x0) < 200 and (y1 - y0) < 100
    # and undo through the window is EXACT
    d.undo()
    d2 = Document(400, 300)
    d2.add_layer("p")
    d2.paint(d2.layers[-1].id, [(20, 20), (90, 24)],
             color=(0.2, 0.5, 0.3), radius=6)
    assert np.allclose(d.layer(lid).pixels,
                       d2.layer(d2.layers[-1].id).pixels)


def test_r16_stroke_snapshots_share_storage():
    """Consecutive paint snapshots reference the SAME stroke-copy objects
    (the shadow) instead of deep-copying every path per entry."""
    d = Document(200, 150)
    d.add_layer("s")
    lid = d.layers[-1].id
    for k in range(4):
        d.paint(lid, [(5, 5 + 20 * k), (60, 6 + 20 * k)],
                color=(0, 0, 0), radius=4)
    s_prev = d._undo[-2][1]["strokes"]
    s_last = d._undo[-1][1]["strokes"]
    assert s_prev and s_last[0] is s_prev[0], "stroke snapshots not shared"
    # the shadow is a COPY of the live strokes, never the live objects
    assert s_last[-1] is not d.strokes[len(s_last) - 1]


def test_r16_stroke_edits_still_undo_exactly():
    """Sharing must never corrupt history: edit a stroke in place, undo,
    and the geometry must come back."""
    d = Document(200, 150)
    d.add_layer("s")
    lid = d.layers[-1].id
    d.paint(lid, [(5, 45), (50, 46)], color=(0, 0, 0), radius=4)
    sid = d.strokes[-1]["id"]
    x_before = float(d.strokes[-1]["points"][0][0])
    d.transform_strokes([sid], dx=25)
    assert float(d.strokes[-1]["points"][0][0]) != x_before
    d.undo()
    assert abs(float(d.strokes[-1]["points"][0][0]) - x_before) < 1e-6
    # and the record=False continuation path keeps the shadow honest
    d.paint(lid, [(5, 80), (50, 81)], color=(0, 0, 0), radius=4)
    n0 = len(d.strokes[-1]["points"])
    d.paint(lid, [(60, 82), (90, 83)], color=(0, 0, 0), radius=4,
            record=False)
    assert len(d.strokes[-1]["points"]) > n0
    snap_pts = d._stroke_shadow[-1]["points"]
    assert len(snap_pts) == len(d.strokes[-1]["points"])


def test_r16_undo_budget_counts_planes_and_tips():
    """_snap_bytes must see the height plane of a media stroke -- before
    R16 it counted pixels only and the trim believed half the truth."""
    d = Document(300, 200)
    d.add_layer("p")
    lid = d.layers[-1].id
    d.layer(lid)._replay_ok = False        # R33: pin the fallback path
    d.paint(lid, [(20, 50), (200, 60)], color=(0.5, 0.4, 0.3), radius=10,
            media="oil", load=0.8)
    d.paint(lid, [(20, 100), (200, 110)], color=(0.5, 0.4, 0.3), radius=10,
            media="oil", load=0.8)
    snap = d._undo[-1][1]
    t = [t for t in snap["layers"] if t[0] == lid][0]
    assert t[8] not in (None, False), "height plane missing from snapshot"
    n = d._snap_bytes(snap)
    px = t[7][1].nbytes if isinstance(t[7], tuple) else t[7].nbytes
    assert n > px, "budget still blind to the non-pixel planes"


def test_r16_painting_memory_stays_bounded():
    """The user-visible symptom: RSS climbing by hundreds of MB while
    painting. Pin the retained undo for a 200-stroke session on a layered
    document to single-digit MB."""
    d = Document(400, 300)
    for nm in ("a", "b"):
        d.add_layer(nm)
    lids = [l.id for l in d.layers]
    rng = np.random.default_rng(3)
    for i in range(200):
        x0, y0 = rng.uniform(20, 350), rng.uniform(20, 260)
        d.paint(lids[i % len(lids)], [(x0, y0), (x0 + 40, y0 + 5)],
                color=(0.5, 0.4, 0.3), radius=7, media="oil", load=0.7)
    n = d._undo_bytes()
    assert n < 12 * 1024 * 1024, "retained undo %.1f MB" % (n / 1e6)


def test_r16_paint_batch_end_to_end():
    """One request, many strokes: applied in order, ONE undo entry that
    reverts the whole batch, and every stroke still in the replay log."""
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "b", "width": 240, "height": 180,
                             "background": [1, 1, 1]})
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}
                 ).get_json()["id"]
    n_undo = len(srv.DOC._undo)
    strokes = [{"layer": lid, "points": [[10, 20 + 18 * i], [220, 22 + 18 * i]],
                "color": [0.1 * i, 0.4, 0.5], "radius": 5} for i in range(5)]
    strokes.append({"layer": lid, "points": [[20, 120], [200, 126]],
                    "mode": "blend", "radius": 12, "opacity": 0.5})
    r = c.post("/api/paint_batch", json={"strokes": strokes}).get_json()
    assert r["ok"] and r["count"] == 6
    assert all(s for s in r["sids"]), r["sids"]
    assert len(srv.DOC._undo) == n_undo + 1, "batch must be ONE undo entry"
    assert srv.DOC._undo[-1][0].startswith("Brush x")
    info = c.get("/api/replay/info").get_json()
    assert info["strokes"] >= 6, "batch strokes missing from the replay log"
    painted = srv.DOC.layer(lid).pixels.copy()
    c.post("/api/undo", json={})
    assert float(srv.DOC.layer(lid).pixels[..., 3].sum()) < 1e-3
    c.post("/api/redo", json={})
    assert np.allclose(srv.DOC.layer(lid).pixels, painted)


def test_r16_paint_batch_refusals():
    pytest.importorskip("flask")
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "b2", "width": 100, "height": 80})
    lid = srv.DOC.layers[0].id
    assert c.post("/api/paint_batch", json={"strokes": []}).status_code == 400
    assert c.post("/api/paint_batch", json={}).status_code == 400
    r = c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[1, 1], [9, 9]], "mode": "smudge"}]})
    assert r.status_code == 400
    r = c.post("/api/paint_batch", json={"strokes": [
        {"layer": "nope", "points": [[1, 1], [9, 9]]}]})
    assert r.status_code == 400
