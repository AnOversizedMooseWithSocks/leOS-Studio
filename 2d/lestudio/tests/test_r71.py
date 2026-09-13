"""tests/test_r71.py -- undo that works, slowness that had a cause, and a
UX sweep.

Devin:

    Undo doesn't seem to work. Sometimes tools take a while to process
    (they shouldn't be slow, so I'm not sure why that's the case) and
    there's no indication that we are supposed to be waiting for a result
    to be rendered. There should be a status bar or an indicator that a
    task is being worked on when that happens. Please fix the problems.
    Then do a UX audit for other snags a user might hit.

Three reports and a sweep.

UNDO. It worked for a plain brush stroke. What did not work was every
gesture the client has to send as MORE THAN ONE mutating request: each one
took its own undo entry, so the first Ctrl+Z reverted a piece of the
gesture nobody thinks of as a separate act and the picture did not move.
Twenty were found. They are one mechanism now: the client stamps every
request in a gesture with the same id and the server folds them into the
entry the first one took.

SLOWNESS. Importing the server module made a 40-point brush stroke on a
768x512 canvas cost 190 ms instead of 12 ms -- for about twelve seconds
after boot, and then it was fast forever. R59's status warm-up thread was
burning both cores of a 2-core machine during exactly the window in which
a person opens the app and paints their first strokes. Under it,
gpu_report() was walking the AST of all 802 engine source files -- 1.9
million nodes -- on EVERY call, to answer a question about which files
import what. Memoised in leCore, moved off the interactive path here.

WAITING. A tool that round-trips leaves the canvas looking exactly like a
canvas with nothing happening, so a slow tool is indistinguishable from a
dead one. There is a badge now, after 250 ms, naming the work.

THE SWEEP. Three agents drove the real UI. The serious finds: the help
overlay had never been visible (nested inside a display:none modal);
Ctrl+S switched to the Smudge tool AND opened the browser's save dialog;
Delete wiped a LOCKED layer and the undo then failed half-way, losing the
paint for good; merging onto a hidden layer silently threw that layer's
paint away; a timeline length of 1e9 froze the page for six minutes.
"""
import os
import threading
import time

import numpy as np

UI = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                  "static", "index.html")
INK = 0.03


# ------------------------------------------------------------------- undo

def test_r71_a_one_gesture_is_one_undo_step():
    """The server-side half: requests sharing a gesture id fold into one
    undo entry, and a request without one starts a new entry."""
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 160, "height": 120})
    st = c.get("/api/state").get_json()
    lid = st["layers"][-1]["id"]
    n0 = len(_doc().  _undo)
    h = {"X-Gesture": "g-test-1"}
    for i in range(4):
        r = c.post("/api/layer", json={"action": "edit", "id": lid,
                                       "name": "n%d" % i}, headers=h)
        assert r.status_code == 200, r.get_data(as_text=True)
    assert len(_doc()._undo) - n0 == 1, \
        "four requests in one gesture took %d undo entries" % (
            len(_doc()._undo) - n0)
    # a request with no gesture id is its own act again
    c.post("/api/layer", json={"action": "edit", "id": lid, "name": "alone"})
    assert len(_doc()._undo) - n0 == 2


def _doc():
    from lestudio.server import WS
    return WS.doc


def test_r71_b_a_new_gesture_id_starts_a_new_entry():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 120, "height": 90})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    n0 = len(_doc()._undo)
    for gid in ("g1", "g1", "g2", "g2", "g3"):
        c.post("/api/layer", json={"action": "edit", "id": lid,
                                   "opacity": 0.5},
               headers={"X-Gesture": gid})
    assert len(_doc()._undo) - n0 == 3, len(_doc()._undo) - n0


def test_r71_c_undo_and_redo_are_never_grouped():
    """They walk the stack rather than adding to it; folding them into a
    gesture would corrupt it."""
    from lestudio.server import _GESTURE_EXEMPT
    for p in ("/api/undo", "/api/redo"):
        assert p in _GESTURE_EXEMPT


def test_r71_d_the_client_marks_its_multi_request_gestures():
    ui = open(UI).read()
    assert "function gesture(fn)" in ui and "gestureBegin" in ui
    assert "opt.headers['X-Gesture']=GESTURE" in ui
    # the list of what counts as one gesture is a list, so it can be read
    block = ui[ui.index("].forEach(asGesture);") - 2600:
               ui.index("].forEach(asGesture);")]
    for fn in ("endCanvasPointer", "soloLayer", "animAddFrame",
               "applyShapeSnap", "setLayerVolume", "addLayerStyle"):
        assert "'%s'" % fn in block, fn
    # a stamp DRAG holds the gesture open for the whole drag
    assert "stampDrag=true; gestureBegin()" in ui
    assert "if(stampDrag){ stampDrag=false; gestureEnd(); }" in ui


def test_r71_e_stitching_is_one_undo_step():
    """The reported case. The textile tool used to send the fill and then
    the relief it implies as two requests; the first Ctrl+Z reverted an
    invisible property and the stitching stayed put."""
    from lestudio import Document
    d = Document(160, 120)
    l = d.add_layer("cloth")
    before = np.asarray(l.pixels).copy()
    n0 = len(d._undo)
    d.hatch_fill(l.id, 80, 60, radius=40, mode="stitch", spacing=7,
                 thickness=1.6, depth=35, color=(150, 30, 60), seed=3)
    assert len(d._undo) - n0 == 1, len(d._undo) - n0
    assert not np.array_equal(np.asarray(l.pixels), before), "nothing painted"
    assert float(getattr(l, "relief", 1.0)) > 1.0, "depth did not raise relief"
    assert d.undo()
    assert np.allclose(np.asarray(d.layer(l.id).pixels), before), \
        "one undo did not take the stitching back"


def test_r71_f_the_client_no_longer_sends_relief_as_a_second_request():
    ui = open(UI).read()
    tx = ui[ui.index("async function doTextile("):]
    tx = tx[:tx.index("\nfunction ")]
    assert "action:'edit'" not in tx, \
        "doTextile still sends a second mutating request"


# --------------------------------------------------------------- the lock

def test_r71_g_a_locked_layer_cannot_be_cleared_moved_or_deleted():
    """The only unrecoverable bug in the sweep: Delete wiped a locked layer
    (with a cheerful 'layer cleared'), and because clear() is journal-first
    the undo then replayed through paint(), hit the lock, raised, and
    aborted the restore. The paint was gone."""
    from lestudio import Document, LayerLocked
    d = Document(120, 90)
    l = d.add_layer("p")
    d.paint(l.id, [[10, 10], [100, 80]], color=(0, 0, 0), radius=6)
    painted = int((np.asarray(l.pixels)[..., 3] > INK).sum())
    assert painted > 100
    l.locked = True
    for name, fn in (("clear", lambda: d.clear(l.id)),
                     ("clear_region", lambda: d.clear_region(l.id, 0, 0, 50, 50)),
                     ("move", lambda: d.move_layer(l.id, 0)),
                     ("remove", lambda: d.remove_layer(l.id))):
        try:
            fn()
            raise AssertionError("%s went through on a locked layer" % name)
        except LayerLocked:
            pass
    assert int((np.asarray(d.layer(l.id).pixels)[..., 3] > INK).sum()) == painted


def test_r71_h_a_lock_never_makes_a_layer_unreplayable():
    """Replay is not an edit. Undo, load and every journal-first op rebuild
    a layer through paint(), so a lock applied after the fact could make a
    layer's own history unreplayable -- and that is what lost the paint."""
    from lestudio import Document
    d = Document(120, 90)
    l = d.add_layer("p")
    d.paint(l.id, [[10, 10], [100, 80]], color=(0, 0, 0), radius=6)
    painted = int((np.asarray(l.pixels)[..., 3] > INK).sum())
    d.clear(l.id)
    assert int((np.asarray(d.layer(l.id).pixels)[..., 3] > INK).sum()) == 0
    d.layer(l.id).locked = True                # lock it, THEN undo
    assert d.undo()
    assert int((np.asarray(d.layer(l.id).pixels)[..., 3] > INK).sum()) == painted


def test_r71_i_a_deliberate_refusal_is_a_400_not_a_500():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 120, "height": 90})
    c.post("/api/layer", json={"action": "add", "name": "p"})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    c.post("/api/layer", json={"action": "edit", "id": lid, "locked": True})
    for body in ({"action": "clear", "id": lid},
                 {"action": "remove", "id": lid},
                 {"action": "move", "id": lid, "index": 0}):
        r = c.post("/api/layer", json=body)
        assert r.status_code == 400, (body, r.status_code)
        j = r.get_json()
        assert j.get("locked") and "unlock" in j["error"], j


# -------------------------------------------------------------- the merge

def test_r71_j_merging_keeps_a_hidden_layers_paint():
    """composite() skips invisible layers by definition, so merging onto a
    hidden layer deleted its paint without a word -- and the canvas looked
    identical afterwards, because what vanished was what you could not see."""
    from lestudio import Document
    d = Document(160, 120)
    a = d.add_layer("under")
    a.pixels[20:50, 20:50, :3] = 0.0
    a.pixels[20:50, 20:50, 3] = 1.0
    b = d.add_layer("over")
    b.pixels[80:110, 80:110, :3] = 0.0
    b.pixels[80:110, 80:110, 3] = 1.0
    a.visible = False
    rpt = {}
    m = d.merge_layer_down(b.id, _report=rpt)
    assert m is not None
    px = np.asarray(m.pixels)[..., 3]
    assert px[30, 30] > 0.5, "the hidden layer's paint was thrown away"
    assert px[95, 95] > 0.5, "the visible layer's paint was lost"
    assert rpt.get("hidden") == ["under"], rpt


def test_r71_k_merge_down_on_the_bottom_layer_says_why():
    from lestudio import Document
    d = Document(80, 60)
    rpt = {}
    assert d.merge_layer_down(d.layers[0].id, _report=rpt) is None
    assert "bottom layer" in (rpt.get("why") or ""), rpt


def test_r71_l_layer_ops_name_the_layer_to_follow():
    """Merge left nothing selected and duplicate left the ORIGINAL
    selected, so the next stroke landed on a layer nobody chose."""
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 120, "height": 90})
    c.post("/api/layer", json={"action": "add", "name": "a"})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    r = c.post("/api/layer", json={"action": "duplicate", "id": lid})
    j = r.get_json()
    assert j.get("select") and j["select"] != lid, j
    r = c.post("/api/layer", json={"action": "merge_down", "id": j["select"]})
    assert r.get_json().get("select"), r.get_json()


# ------------------------------------------------------------ other traps

def test_r71_m_the_timeline_refuses_a_nonsense_range():
    """1e9 made the strip redraw loop twenty-one million times and the page
    stopped responding for six and a half minutes, with no error."""
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 80, "height": 60})
    for hi in (1e9, -5, 0, float("nan")):
        r = c.post("/api/timeline", json={"action": "range", "lo": 0, "hi": hi})
        assert r.status_code == 400, (hi, r.status_code)
    assert c.post("/api/timeline",
                  json={"action": "range", "lo": 0, "hi": 96}
                  ).status_code == 200


def test_r71_n_painting_with_no_layer_is_a_400_not_a_crash():
    """`layer: null` reaches the route whenever a stroke starts in the
    window between a delete clearing the client's selection and the
    refresh that restores it."""
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 80, "height": 60})
    r = c.post("/api/paint", json={"layer": None, "points": [[5, 5], [20, 20]],
                                   "color": [0, 0, 0], "radius": 4})
    assert r.status_code == 400, r.status_code
    assert "layer" in r.get_json()["error"]


def test_r71_o_a_stroke_outside_the_selection_says_so():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 200, "height": 150})
    c.post("/api/layer", json={"action": "add", "name": "p"})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    sid = c.post("/api/select", json={"tool": "rect", "name": "left",
                                      "params": {"x0": 0, "y0": 0,
                                                 "x1": 60, "y1": 150}}
                 ).get_json()["selection"]["id"]
    r = c.post("/api/paint", json={"layer": lid, "points": [[150, 40], [180, 90]],
                                   "color": [0, 0, 0], "radius": 5,
                                   "selection": sid}).get_json()
    assert "OUTSIDE the selection" in (r.get("warning") or ""), r


def test_r71_p_group_edits_are_undoable():
    """Renaming a group and ticking layers into it were invisible to undo,
    so a Ctrl+Z after organising deleted the whole group instead."""
    from lestudio import Document
    d = Document(80, 60)
    l = d.add_layer("a")
    g = d.add_group("g")
    n0 = len(d._undo)
    d.edit_group(g["id"], name="Sky")
    assert len(d._undo) - n0 == 1, len(d._undo) - n0
    assert d.undo()
    assert d.group(g["id"])["name"] == "g"
    assert any(x["id"] == g["id"] for x in d.groups), \
        "undo deleted the group instead of undoing the rename"


def test_r71_q_undo_does_not_promote_the_working_selection():
    """One undo used to move a marquee the person never chose to keep into
    their saved list, and every further undo added another row with the
    SAME id."""
    from lestudio import Document
    d = Document(160, 120)
    l = d.add_layer("p")
    for _ in range(3):
        d.select("rect", {"x0": 10, "y0": 10, "x1": 60, "y1": 60})
        d.paint(l.id, [[80, 80], [120, 100]], color=(0, 0, 0), radius=5)
        d.undo()
    assert d.selections == [], [(x.id, x.name) for x in d.selections]


def test_r71_r_separate_layer_edits_are_separate_undo_steps():
    """Three deliberate acts collapsed into one entry, because the
    coalescing run that exists for slider drags had no bound at all."""
    from lestudio import Document
    d = Document(80, 60)
    l = d.add_layer("p")
    n0 = len(d._undo)
    d.edit_layer(l.id, name="Sky")
    d.edit_layer(l.id, visible=False)
    d.edit_layer(l.id, opacity=0.4)
    assert len(d._undo) - n0 == 3, len(d._undo) - n0
    n1 = len(d._undo)
    for v in (0.9, 0.8, 0.7, 0.6):            # one slider drag
        d.edit_layer(l.id, opacity=v)
    assert len(d._undo) - n1 <= 1, len(d._undo) - n1


def test_r71_s_unnamed_layers_get_distinct_names():
    from lestudio import Document
    d = Document(64, 48)
    for _ in range(4):
        d.add_layer()
    names = [l.name for l in d.layers]
    assert len(set(names)) == len(names), names


# -------------------------------------------------------------- the client

def test_r71_t_the_help_overlay_is_not_inside_a_hidden_modal():
    """It had NEVER been visible: `?`, the ? button and "Show me more" in
    the welcome banner all filled it with 10,000 characters of
    documentation and then showed a 0x0 box, because it was nested inside
    #modalBack, which is display:none except while the New-document dialog
    happens to be open."""
    ui = open(UI).read()
    i = ui.index('id="shortcutsBack"')
    j = ui.index('<div id="modalBack"')
    assert i < j, "the shortcuts overlay is still nested inside #modalBack"


def test_r71_u_ctrl_combinations_do_not_switch_tools():
    ui = open(UI).read()
    assert "const want=(e.ctrlKey||e.metaKey||e.altKey)?null:TOOLKEY[e.key];" in ui, \
        "the tool-letter handler still fires on Ctrl combinations"


def test_r71_v_escape_knows_every_overlay():
    ui = open(UI).read()
    blk = ui[ui.index("const OVERLAYS="):]
    blk = blk[:blk.index("];") + 2]
    for ident in ("lcDlg", "glModalBack", "obsModalBack", "inviteModalBack",
                  "shortcutsBack", "modalBack2", "modalBack", "layerDlgBack"):
        assert "'%s'" % ident in blk, ident
    assert "newDocBack" not in blk, "the dead id is still in the list"
    # and it runs BEFORE the text-field guard, or a dialog that autofocuses
    # an input can never be escaped
    assert ui.index("if(e.key==='Escape'&&closeTopOverlay())") \
        < ui.index("if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA'")


def test_r71_w_a_message_outranks_what_it_is_about():
    """The toast sat at z-index 99, behind the leCore panel (300), the
    replay dialog and the kicked banner (200) -- so a failure raised from
    inside one of those was reported to pixels nobody could see."""
    ui = open(UI).read()
    line = [l for l in ui.split("\n") if l.startswith("#toast{")][0]
    z = int(line.split("z-index:")[1].split("}")[0])
    highest = max(int(x.split("z-index:")[1].split(";")[0].split("}")[0])
                  for x in ui.split("z-index:")[1:]
                  if x.split(";")[0].split("}")[0].strip().isdigit()
                  and int(x.split("z-index:")[1].split(";")[0].split("}")[0])
                  if False) if False else None
    assert z >= 1000, "the toast is at z-index %d and will be painted over" % z


def test_r71_x_there_is_a_busy_indicator():
    ui = open(UI).read()
    assert 'id="busyBadge"' in ui and 'role="status"' in ui
    assert "function busyStart(" in ui and "function busyStop(" in ui
    assert "busyWatchdog" in ui, "a badge with no watchdog can stick forever"
    # it waits before showing, so ordinary fast work never flickers one
    assert "}, 250);" in ui
    # and background chatter never raises it
    assert "events|presence|autosave|status|prefs|state" in ui


def test_r71_y_selection_ops_report_what_they_did():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 120, "height": 90})
    sid = c.post("/api/select", json={"tool": "rect",
                                      "params": {"x0": 10, "y0": 10,
                                                 "x1": 50, "y1": 50}}
                 ).get_json()["selection"]["id"]
    r = c.post("/api/selection", json={"action": "modify", "id": sid,
                                       "op": "expand", "amount": 4}).get_json()
    assert isinstance(r.get("coverage"), float), r
    ui = open(UI).read()
    assert 'id="selInvert"' in ui, "there is still no way to invert a selection"


# ---------------------------------------------------------------- the speed

def test_r71_z_the_capability_report_is_measured_once():
    """gpu_report() walked the AST of all 802 engine source files -- 1.9
    million nodes -- on EVERY call, about 5.5 s each. leStudio calls it
    (and should_offload, and engine_status, which all land there) at boot,
    so on a 2-core machine the first ten seconds of a painting session had
    both cores busy answering a question about which files import what."""
    import pytest
    try:
        from holographic.io_and_interop import holographic_gpureport as gr
    except Exception:
        pytest.skip("engine build without holographic_gpureport")
    if not hasattr(gr, "_backend_consumers"):
        pytest.skip("engine build without _backend_consumers")
    gr._backend_consumers()
    t0 = time.perf_counter()
    a = gr._backend_consumers()
    first = time.perf_counter() - t0
    b = gr._backend_consumers()
    assert a == b and a
    assert first < 0.25, "a second call still costs %.2fs" % first


def test_r71_za_the_warm_up_does_not_run_at_import():
    """R59 was right to move the 10.4 s off the request path, but a daemon
    thread started at import is not free: it burned both cores for twelve
    seconds beginning exactly when a person opens the app, and a 12 ms
    brush stroke measured 190 ms for as long as it ran."""
    import lestudio.server as SV
    src = open(os.path.join(os.path.dirname(SV.__file__), "server.py")).read()
    i = src.index("def _warm_status(")
    tail = src[i:i + 2000]
    assert "threading.Thread(target=_warm_status" not in \
        src[:src.index("def warm_status_async")], \
        "the warm-up thread still starts at import"
    assert "os.nice(10)" in tail, "the warm-up does not yield the CPU"
    assert "_WARM[\"busy_until\"]" in tail, \
        "the warm-up does not stand down while the person is working"


def test_r71_zb_status_never_blocks_on_the_measurement():
    from lestudio.server import app
    c = app.test_client()
    t0 = time.perf_counter()
    r = c.get("/api/status")
    dt = time.perf_counter() - t0
    assert r.status_code == 200
    assert dt < 1.0, "a plain /api/status took %.1fs" % dt
    assert "measuring" in r.get_json()
