"""tests/test_r62.py -- the swarm fast path lost journal-first saving.

Found by asking why a .lews for one 1600x1000 painting was 150 MB when the
picture displays as a 1.3 MB PNG.

Two bugs, compounding. `/api/paint_batch` called `DOC.record(...)` without
`journaled=True` (the R16 batch-record law says journaled), and -- the real
one -- `Document.paint` demoted a layer out of replay-faithfulness on
`record=False` ALONE. But `record=False` means "do not open your own undo
entry", not "this paint is unrecorded": the batch path pairs it with
`stroke_new=True`, which DOES append a replay record. So every layer ever
painted through the agent/swarm fast path was permanently non-replayable,
its pixels had to be written into every save, and the journal-first .lews
(DETERMINISM_BACKLOG P3.1) silently never applied to the one workflow that
makes the biggest files. Measured after the fix at that scale: 97.3 MB ->
1.1 MB, 88x.

There was also a decoy: a block in paint_batch that looks like it repairs
this by calling `_mark_replay_ok`, which writes the DOCUMENT's `_replay_ok`
dict (the "is the verdict still current" cache) while the demotion cleared
the LAYER's `_replay_ok` attribute (the verdict). Two things, one name.
"""
import io
import json
import zipfile


def test_r62_record_false_alone_demotes_but_a_recorded_stroke_does_not():
    """The exact distinction the guard was missing."""
    from lestudio import Document
    d = Document(120, 90)
    a, b, c = (d.add_layer(n).id for n in ("a", "b", "c"))
    d.paint(a, [[5, 5], [100, 80]], color=(1, 0, 0), radius=4, record=True)
    d.paint(b, [[5, 5], [100, 80]], color=(1, 0, 0), radius=4,
            record=False, stroke_new=True)
    d.paint(c, [[5, 5], [100, 80]], color=(1, 0, 0), radius=4, record=False)
    assert getattr(d.layer(a), "_replay_ok", True) is not False, "recorded stroke"
    assert getattr(d.layer(b), "_replay_ok", True) is not False, \
        "record=False + stroke_new=True IS in the replay log and must stay faithful"
    assert getattr(d.layer(c), "_replay_ok", True) is False, \
        "a genuinely unrecorded dab must still demote the layer (R33)"


def test_r62_stroke_new_really_does_record():
    """The premise of the fix, checked rather than assumed: the batch path's
    flag combination appends a replay record and a bare record=False does
    not."""
    from lestudio import Document
    d = Document(120, 90)
    lid = d.add_layer("t").id
    n0 = len(d.strokes)
    d.paint(lid, [[5, 5], [100, 80]], color=(1, 0, 0), radius=4,
            record=False, stroke_new=True)
    n1 = len(d.strokes)
    d.paint(lid, [[5, 50], [100, 60]], color=(0, 1, 0), radius=4, record=False)
    n2 = len(d.strokes)
    assert n1 == n0 + 1, "record=False + stroke_new=True must record"
    assert n2 == n1, "record=False alone must not record"


def test_r62_batch_painted_layers_stay_replay_faithful():
    """Through the real route, which is what agents and swarms use."""
    from lestudio.server import app, WS
    c = app.test_client()
    assert c.post("/api/new", json={"width": 300, "height": 200}).get_json()["ok"]
    lid = c.post("/api/layer", json={"action": "add", "name": "batch"}).get_json()["id"]
    r = c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[10, 10 + i * 8, 1], [280, 20 + i * 8, 1]],
         "color": [1, 0.4, 0.2], "radius": 5} for i in range(8)]})
    assert r.get_json()["ok"]
    assert getattr(WS.doc.layer(lid), "_replay_ok", True) is not False, \
        "/api/paint_batch demoted the layer it painted"
    # and the R16 law: one undo entry for the whole batch, journalled
    import inspect
    from lestudio import server as SV
    src = inspect.getsource(SV.paint_batch) + inspect.getsource(SV._paint_batch_locked)
    assert "journaled=True" in src, "the batch record must declare itself journalled"


def test_r62_a_batch_painted_document_saves_journal_first():
    """The payoff: a light save stores no pixels for batch-painted layers,
    and reopening it reproduces the picture exactly."""
    import hashlib
    from lestudio.server import app
    c = app.test_client()
    assert c.post("/api/new", json={"name": "r62", "width": 320, "height": 240}).get_json()["ok"]
    lid = c.post("/api/layer", json={"action": "add", "name": "batch"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[8, 8 + i * 6, 1], [300, 16 + i * 6, 1]],
         "color": [0.2, 0.6, 1.0], "radius": 4} for i in range(20)]})
    before = c.get("/api/composite.png").data

    light = c.get("/api/workspace.lews?light=1").data
    z = zipfile.ZipFile(io.BytesIO(light))
    man = json.loads(z.read("manifest.json"))
    sec = [s for s in man["sections"] if s["kind"] == "lestudio.document"
           and s["meta"].get("name") == "r62"][0]
    lay = {l["id"]: l for l in sec["meta"]["layers"]}
    assert lay[lid].get("pixels_cached") is False, \
        "a batch-painted layer must save journal-only under ?light=1"
    assert ("layer_%s" % lid) not in sec["arrays"], \
        "no pixel array should be written for a replay-faithful layer"

    c.post("/api/workspace/open", data={"file": (io.BytesIO(light), "w.lews")},
           content_type="multipart/form-data")
    after = c.get("/api/composite.png").data
    assert hashlib.sha256(before).hexdigest() == hashlib.sha256(after).hexdigest(), \
        "a journal-only save must reopen to the identical picture"


def test_r62_the_saving_is_large_at_a_real_canvas_size():
    """Worth a pin because the whole point is the size: at painting scale a
    journal-first save is orders of magnitude smaller, and a regression
    would be invisible except in file size."""
    from lestudio.server import app
    c = app.test_client()
    assert c.post("/api/new", json={"name": "r62big", "width": 900, "height": 600}).get_json()["ok"]
    lid = c.post("/api/layer", json={"action": "add", "name": "b"}).get_json()["id"]
    for chunk in range(3):
        c.post("/api/paint_batch", json={"strokes": [
            {"layer": lid, "points": [[(i * 7) % 900, (i * 13) % 600, 1],
                                      [(i * 11) % 900, (i * 17) % 600, 1]],
             "color": [0.4, 0.5, 0.9], "radius": 5, "opacity": 0.5}
            for i in range(chunk * 200, chunk * 200 + 200)]})
    # R67: the plain save IS the journal-first save now. ?pixels=1 is the
    # opt-out for a caller that wants the rasters in the file regardless.
    full = len(c.get("/api/workspace.lews?pixels=1").data)
    light = len(c.get("/api/workspace.lews?light=1").data)
    auto = len(c.get("/api/workspace.lews").data)
    assert light * 4 < full, \
        "journal-first save is not saving anything: light=%d full=%d" % (light, full)
    assert auto * 4 < full, \
        "the DEFAULT save must be journal-first: auto=%d full=%d" % (auto, full)
