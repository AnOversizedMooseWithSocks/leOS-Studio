"""tests/test_r35.py -- P1.5 (text + stamps as ops), G5 (gated media
stepping), P0.5 (concurrency discipline pinned by a stress test).

Text is {string, font name, pos, size, color, spacing, shadow} with any
spline path FROZEN into the record as points. A stamp presses pixels
frozen into the asset store, so editing or deleting the stamp later
cannot change what history already pressed. A gated media step advances
the fluid inside the gate and puts the outside state back, and the gate
rides the record. Duplicate/merge need no ops BY DESIGN: their layers
become replayable through body-carrying base capture at first stroke.

P0.5: one writer per document -- concurrent painters on separate layers
plus readers (composite, state, autosave) through the server must leave
every painted layer replay-faithful with a complete journal.
"""
import numpy as np


def test_r35_text_is_a_journaled_op():
    from lestudio import Document
    d = Document(200, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    n = d.add_text(lid, "Hi", x=20, y=20, size=24, color=(1, 0.5, 0))
    assert n == 2
    k = d.strokes[-1]
    assert k["brush"].get("op") == "text" and k["brush"]["text"] == "Hi"
    assert d.replay_is_faithful(lid), "text must journal, not dirty"
    before = d.layer(lid).pixels.copy()
    assert d.undo()
    assert float(d.layer(lid).pixels[..., 3].max()) < 1e-6
    assert d.redo()
    assert np.array_equal(d.layer(lid).pixels, before)


def test_r35_text_on_a_path_freezes_the_spline():
    from lestudio import Document
    d = Document(240, 160)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    p = d.add_spline("arc", [{"x": 20, "y": 100, "hx": 60, "hy": 40},
                             {"x": 200, "y": 100, "hx": 140, "hy": 40}])
    d.add_text(lid, "curve", spline=p.id, size=20, color=(0, 0, 0))
    assert d.strokes[-1]["brush"].get("spline_pts"), \
        "the path must be frozen into the record"
    # deleting the spline must not hurt replay
    d.remove_spline(p.id)
    d.layer(lid)._replay_ok = True
    assert d.replay_is_faithful(lid), \
        "text-on-a-path replays from its frozen points"


def test_r35_stamp_freezes_its_pixels():
    from lestudio import Document, Stamp
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    sp = np.zeros((30, 30, 4), np.float32)
    sp[5:25, 5:25] = (0, 0.8, 0.2, 1)
    st = Stamp("dot", sp)
    st.id = d._mint_id("ST")
    d.stamps.append(st)
    d.place_stamp(lid, st.id, 80, 60, scale=1.5, rotation=30)
    k = d.strokes[-1]
    assert k["brush"].get("op") == "stamp" and k["brush"].get("asset")
    assert d.replay_is_faithful(lid)
    # mutate AND delete the live stamp: history must not notice
    st.pixels[...] = 0.0
    d.stamps.clear()
    d.layer(lid)._replay_ok = True
    assert d.replay_is_faithful(lid), \
        "the pressed pixels are frozen -- the stamp's fate is irrelevant"
    before = d.layer(lid).pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(d.layer(lid).pixels, before)


def test_r35_gated_media_step_confines_the_advance():
    from lestudio import Document, Selection
    d = Document(128, 96)
    l = d.add_layer("plume")
    d.edit_layer(l.id, vol_kind="smoke", thickness=6.0)
    d.paint(l.id, [[30, 60], [34, 62]], color=(0.9, 0.9, 0.9), radius=10.0)
    d.paint(l.id, [[90, 60], [94, 62]], color=(0.9, 0.9, 0.9), radius=10.0)
    sel = Selection(96, 128, data=np.zeros((96, 128), np.float32))
    sel.data[:, :64] = 1.0
    d.selections.append(sel)
    st0 = {f: l._media[f].copy() for f in ("den", "dye")}
    d.media_step(l.id, 4, selection=sel.id)
    st = l._media
    gh, gw = st["den"].shape
    # outside the gate the density field is EXACTLY the pre-step state
    assert np.array_equal(st["den"][:, gw * 5 // 8:],
                          st0["den"][:, gw * 5 // 8:]), \
        "outside the gate the medium must hang still"
    assert float(np.abs(st["den"][:, :gw // 2]
                        - st0["den"][:, :gw // 2]).max()) > 1e-4, \
        "inside the gate the medium must actually advance"
    k = d.strokes[-1]
    assert k["brush"].get("op") == "media_step" and k["brush"].get("sel_asset")
    assert d.replay_is_faithful(l.id)
    px = l.pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(l.pixels, px), "gated step must undo/redo exactly"


def test_r35_duplicate_needs_no_op_by_design():
    """A duplicated layer becomes replayable the moment it is painted:
    base capture takes it as-is, body included (P2.2)."""
    from lestudio import Document
    d = Document(120, 90)
    d.layers[0].pixels[...] = 0.0
    a = d.layers[0].id
    d.paint(a, [[10, 20], [100, 20]], color=(1, 0, 0), radius=5.0,
            media="oil", load=0.7)
    cp = d.duplicate_layer(a)
    d.paint(cp.id, [[10, 60], [100, 60]], color=(0, 0, 1), radius=5.0)
    assert d.replay_is_faithful(cp.id), \
        "a painted duplicate must replay from its as-is base"


def test_r35_concurrent_painters_leave_a_faithful_document():
    """P0.5: one writer per document. Four threads paint on four separate
    layers while two reader threads hammer composite/state/autosave.
    Afterwards: every stroke that got a 200 is in the journal, every
    painted layer replays faithfully, and undo still walks."""
    import threading
    from lestudio.server import app, DOC
    c0 = app.test_client()
    lids = []
    for i in range(4):
        c0.post("/api/layer", json={"action": "add", "name": "t%d" % i})
    st = c0.get("/api/state").json
    lids = [l["id"] for l in st["layers"] if l["name"].startswith("t")]
    ok_counts = [0] * 4
    errs = []

    def painter(ix):
        c = app.test_client()
        for j in range(6):
            y = 10 + ix * 18 + (j % 3)
            r = c.post("/api/paint", json={
                "layer": lids[ix], "points": [[5 + j * 15, y],
                                              [15 + j * 15, y + 4]],
                "color": [ix * 0.25, 0.5, 1 - ix * 0.2], "radius": 3,
                "record": True})
            if r.status_code == 200 and not (r.json or {}).get("error"):
                ok_counts[ix] += 1
            else:
                errs.append((ix, j, r.status_code))

    def reader():
        c = app.test_client()
        for _ in range(8):
            c.get("/api/composite.png?maxw=200")
            c.get("/api/state")
            c.get("/api/autosave")

    ts = [threading.Thread(target=painter, args=(i,)) for i in range(4)]
    ts += [threading.Thread(target=reader) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, "concurrent paints failed: %s" % errs[:5]
    for ix, lid in enumerate(lids):
        got = sum(1 for k in DOC._iter_strokes(lid))
        assert got == ok_counts[ix], \
            "layer %s journal has %d strokes for %d acknowledged paints" % (
                lid, got, ok_counts[ix])
        assert DOC.replay_is_faithful(lid), \
            "concurrent painting must leave %s replay-faithful" % lid
    assert c0.post("/api/undo").json.get("ok"), \
        "undo must still walk after the storm"
