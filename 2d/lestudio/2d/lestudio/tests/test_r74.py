"""tests/test_r74.py -- R74 step 01: the lasso.

The brush panel has promised a lasso since R70 ("selections are made with the
marquee, wand and lasso tools") and there was no lasso: the only ways to
select a shape were a rectangle, an ellipse, and three tools that read pixels.
That is the first thing a person reaches for in any editor.

A freehand lasso and a clicked polygon lasso are the SAME selection once the
client has the points -- a closed outline -- so both land on `select("poly",
{points})`, which rasterises through `_poly_gate`: the identical function the
scribble / hatch / textile / scatter / creature generators already use for an
inline region. Nothing in this app rasterises a shape twice.

The gesture half (drag, corners, Backspace, Enter, Esc, the modifier keys)
is driven against the DOM shim in tests/test_lasso_ui.js.
"""
import numpy as np


def _mk(w=200, h=150):
    from lestudio import Document
    return Document(w, h)


def _ring(cx, cy, rx, ry, n=48):
    return [[cx + rx * np.cos(t), cy + ry * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, n)]


def test_r74_lasso_selects_its_interior():
    d = _mk()
    s = d.select("poly", {"points": _ring(100, 75, 60, 45)})
    g = d._resolve_gate(s.id)
    assert float(g[75, 100]) == 1.0, "the middle of the lasso must be selected"
    assert float(g[5, 5]) == 0.0, "the corner outside it must not be"
    assert 0.2 < float(g.mean()) < 0.4, \
        "an ellipse of this size covers about a quarter of the canvas"


def test_r74_lasso_gates_a_stroke_like_any_selection():
    """The point of a selection: paint stops at its edge."""
    d = _mk()
    L = d.add_layer("p").id
    s = d.select("poly", {"points": _ring(100, 75, 50, 40)})
    d.paint(L, [[10, 75], [190, 75]], color=(0, 0, 0), radius=6,
            selection=s.id)
    a = d.layer(L).pixels[..., 3]
    assert float(a[75, 100]) > 0.5, "inside the lasso the stroke landed"
    assert float(a[75, 20]) == 0.0, "outside it the stroke was cut"


def test_r74_lasso_composes_with_the_other_modes():
    d = _mk()
    s = d.select("poly", {"points": _ring(100, 75, 40, 30)})
    assert float(d._resolve_gate(s.id)[10, 10]) == 0.0
    s2 = d.select("rect", {"x0": 0, "y0": 0, "x1": 40, "y1": 40},
                  mode="add", target=s.id)
    g = d._resolve_gate(s2.id)
    assert float(g[10, 10]) == 1.0 and float(g[75, 100]) == 1.0, \
        "add must keep the lasso and gain the rectangle"
    s3 = d.select("poly", {"points": _ring(100, 75, 40, 30)},
                  mode="subtract", target=s2.id)
    g3 = d._resolve_gate(s3.id)
    assert float(g3[10, 10]) == 1.0 and float(g3[75, 100]) == 0.0, \
        "subtract must take the lasso back out"


def test_r74_lasso_feathers_like_the_rest():
    # separate documents: a second select() on the same document reuses the
    # slot, and comparing a selection against its own replacement proves
    # nothing (it cost a debugging minute -- hence the note).
    a, b = _mk(), _mk()
    hard = a.select("poly", {"points": _ring(100, 75, 50, 40)})
    soft = b.select("poly", {"points": _ring(100, 75, 50, 40)}, feather=10)
    gh, gs = a._resolve_gate(hard.id), b._resolve_gate(soft.id)
    assert float(gh[75, 100]) == 1.0 and float(gs[75, 100]) > 0.99, \
        "both effectively solid in the middle (a wide feather bleeds a hair)"
    band = slice(44, 60)                      # across the left edge at y=75
    assert float(np.abs(np.diff(gh[75, band])).max()) > 0.9, \
        "the hard lasso steps from 0 to 1 in one pixel"
    assert float(np.abs(np.diff(gs[75, band])).max()) < 0.2, \
        "the feathered one ramps instead"


def test_r74_lasso_refuses_a_shape_that_is_not_one():
    import pytest
    d = _mk()
    with pytest.raises(ValueError):
        d.select("poly", {"points": [[10, 10], [20, 20]]})
    with pytest.raises(ValueError):
        d.select("poly", {"points": []})
    # a NaN corner is dropped, not propagated into the mask
    s = d.select("poly", {"points": [[10, 10], [float("nan"), 20],
                                     [80, 20], [80, 80], [10, 80]]})
    g = d._resolve_gate(s.id)
    assert np.isfinite(g).all(), "a NaN must never reach the selection mask"
    assert float(g[50, 50]) == 1.0


def test_r74_lasso_route_and_its_hygiene():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 200, "height": 150})
    r = c.post("/api/select", json={"tool": "poly",
                                    "params": {"points": _ring(100, 75, 60, 45, 24)}})
    assert r.status_code == 200, r.get_data(as_text=True)
    cov = r.get_json()["selection"]["coverage"]
    assert 0.2 < cov < 0.4, cov
    for junk in (5, "nope", None, [[1, 2]], [["a", "b"], [1, 2], [3, 4]],
                 [[1, 2], [3, 4]]):
        rr = c.post("/api/select", json={"tool": "poly",
                                         "params": {"points": junk}})
        assert rr.status_code == 400, (junk, rr.status_code)
    # a long path is accepted (a real freehand drag is hundreds of points)
    many = [[100 + 60 * np.cos(t), 75 + 45 * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, 900)]
    assert c.post("/api/select", json={"tool": "poly",
                                       "params": {"points": many}}).status_code == 200


def test_r74_lasso_is_in_the_client_and_l_is_the_lasso():
    import os
    ui = open(os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                           "static", "index.html")).read()
    assert 'id="tLasso"' in ui and 'id="tPolyLasso"' in ui
    assert "lasso:'tLasso'" in ui and "polylasso:'tPolyLasso'" in ui
    assert "l:'lasso'" in ui, "L is the lasso in every other editor"
    # the creature moved off L; it now has its own key and its own visible
    # button (see test_r74_the_creature_brush_has_a_button_you_can_see)
    assert "creature:'Creature — A" in ui, "the creature says which key it has"
    assert "Creature (A)" in ui, "its tooltip must agree"
    assert "function selModFromEvent" in ui and "const SHIFTTOOLKEY=" in ui
    # the promise the brush panel has been making since R70 is now true
    assert "marquee, wand and lasso tools" in ui


# ----------------------------------------------------------- step 02: gradient
# Every editor has a gradient tool; this app had a Gradient NODE and no way to
# drag one onto a layer. It is built the pwarp way -- a PIXEL-FREE journal
# record and ONE applier shared by the tool and by replay -- so what is pinned
# here is that the record, not an image, is what survives.

def test_r74_gradient_ramps_between_its_two_ends():
    d = _mk(200, 120)
    L = d.add_layer("g").id
    d.gradient(L, 20, 60, 180, 60, kind="linear",
               color=(1, 0, 0), color2=(0, 0, 1))
    px = d.layer(L).pixels
    assert np.allclose(px[60, 20, :3], (1, 0, 0), atol=0.02), "start colour"
    assert np.allclose(px[60, 180, :3], (0, 0, 1), atol=0.02), "end colour"
    assert np.allclose(px[60, 100, :3], (0.5, 0, 0.5), atol=0.05), "half way"
    assert float(px[60, 5, 0]) > 0.95, "before the start it clamps to the start"
    assert float(px[60, 195, 2]) > 0.95, "past the end it clamps to the end"


def test_r74_gradient_replays_from_a_pixel_free_record():
    d = _mk(200, 120)
    L = d.add_layer("g").id
    d.gradient(L, 10, 10, 190, 110, kind="radial", color=(0.9, 0.2, 0.1))
    rec = d.strokes[-1]["brush"]
    assert rec["op"] == "gradient" and rec["kind"] == "radial"
    assert "stops" in rec and not any(
        k in rec for k in ("pixels", "image", "asset")), \
        "the record must carry geometry, not an image"
    assert d.replay_is_faithful(L), "a gradient must replay from the journal"


def test_r74_gradient_survives_a_lews_round_trip_and_undoes_as_one():
    from lestudio import save_workspace, load_workspace
    d = _mk(160, 100)
    L = d.add_layer("g").id
    d.gradient(L, 0, 0, 160, 100, kind="diamond", color=(0.2, 0.6, 0.3))
    before = np.asarray(d.layer(L).pixels).copy()
    docs, _, act, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    after = np.asarray(docs[act].layer(L).pixels)
    assert np.abs(after - before).max() < 1e-5, "the gradient did not round trip"
    d.undo()
    assert float(d.layer(L).pixels[..., 3].max()) == 0.0, \
        "one drag is one undo entry"


def test_r74_gradient_honours_a_feathered_selection():
    d = _mk(200, 120)
    L = d.add_layer("g").id
    s = d.select("poly", {"points": _ring(100, 60, 60, 40)}, feather=6)
    d.gradient(L, 20, 60, 180, 60, color=(1, 0, 0), selection=s.id)
    a = d.layer(L).pixels[..., 3]
    assert float(a[60, 100]) > 0.9, "inside the selection the gradient landed"
    assert float(a[5, 5]) == 0.0, "outside it nothing was touched"
    assert d.replay_is_faithful(L), "the frozen gate must replay"


def test_r74_gradient_to_transparent_fades_the_alpha():
    d = _mk(200, 60)
    L = d.add_layer("g").id
    d.gradient(L, 0, 30, 200, 30, color=(0.1, 0.2, 0.9), to_transparent=True)
    a = d.layer(L).pixels[..., 3]
    assert float(a[30, 5]) > 0.9 and float(a[30, 195]) < 0.1, \
        "to-transparent must ramp the alpha, not the colour"


def test_r74_gradient_kinds_differ_and_all_replay():
    seen = {}
    for kind in ("linear", "radial", "angle", "reflected", "diamond"):
        d = _mk(120, 120)
        L = d.add_layer("g").id
        d.gradient(L, 60, 60, 110, 60, kind=kind)
        assert d.replay_is_faithful(L), kind
        seen[kind] = np.asarray(d.layer(L).pixels[..., 0]).copy()
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(seen[a] - seen[b]).max() > 0.05, \
                "%s and %s produced the same picture" % (a, b)


def test_r74_gradient_dither_is_deterministic():
    out = []
    for _ in range(2):
        d = _mk(120, 60)
        L = d.add_layer("g").id
        d.gradient(L, 0, 30, 120, 30, dither=0.5, seed=9)
        out.append(np.asarray(d.layer(L).pixels).copy())
    assert np.array_equal(out[0], out[1]), \
        "a seeded dither must reproduce, or replay diverges"
    plain = _mk(120, 60)
    P = plain.add_layer("g").id
    plain.gradient(P, 0, 30, 120, 30, dither=0.0)
    assert not np.array_equal(out[0], np.asarray(plain.layer(P).pixels)), \
        "dither must actually do something"


def test_r74_gradient_stops_are_sorted_and_cleaned():
    d = _mk(120, 40)
    L = d.add_layer("g").id
    # deliberately out of order, with junk that must be dropped
    d.gradient(L, 0, 20, 120, 20, stops=[
        {"pos": 1.0, "color": [0, 0, 1]},
        {"pos": "nope"},
        {"pos": 0.0, "color": [1, 0, 0]},
        {"pos": 0.5, "color": [0, 1, 0]}])
    px = d.layer(L).pixels
    assert np.allclose(px[20, 2, :3], (1, 0, 0), atol=0.05)
    assert np.allclose(px[20, 60, :3], (0, 1, 0), atol=0.06)
    assert np.allclose(px[20, 117, :3], (0, 0, 1), atol=0.05)


def test_r74_gradient_route_and_its_hygiene():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 200, "height": 120})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    assert c.post("/api/gradient", json={
        "layer": lid, "x0": 0, "y0": 0, "x1": 200, "y1": 0,
        "color": [1, 0, 0], "color2": [0, 0, 1]}).status_code == 200
    for junk in ({}, {"x0": float("nan"), "y0": 0, "x1": 1, "y1": 1},
                 {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "kind": "bogus"},
                 {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "stops": 5}):
        r = c.post("/api/gradient", json={"layer": lid, **junk})
        assert r.status_code == 400, (junk, r.status_code)


def test_r74_gradient_is_in_the_client():
    import os
    ui = open(os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                           "static", "index.html")).read()
    assert 'id="tGradient"' in ui and "gradient:'tGradient'" in ui
    assert 'id="gradHud"' in ui and "gradient:'gradHud'" in ui
    assert "g:(tool==='gradient'?'fill':'gradient')" in ui, "Shift+G"
    assert "async function doGradient(g)" in ui and "'/api/gradient'" in ui
    for el in ("grKind", "grFrom", "grTo", "grSwap", "grAlpha", "grOp",
               "grDither", "grToLbl"):
        assert ('id="%s"' % el) in ui, el
    # the HUD must be docked with the others or it floats over the canvas
    assert "'gradHud'" in ui.split("dock.appendChild(el)")[0][-400:]


def test_r74_gradient_angle_seam_sits_behind_the_drag():
    """The sweep's one discontinuity belongs OPPOSITE the drag. The first
    version put it on the drag direction itself, which laid a hard edge
    straight across the middle of the shape you had just dragged through."""
    d = _mk(161, 161)
    L = d.add_layer("g").id
    d.gradient(L, 80, 80, 150, 80, kind="angle",
               color=(1, 0, 0), color2=(0, 0, 1))
    # measure across the seam, not along it: the centre row lies ON both the
    # drag axis and its opposite, where a sweep is flat either side.
    col_left = d.layer(L).pixels[:, 20, 0]     # opposite the drag
    col_right = d.layer(L).pixels[:, 140, 0]   # along the drag
    jump_left = float(np.abs(np.diff(col_left[60:100])).max())
    jump_right = float(np.abs(np.diff(col_right[60:100])).max())
    assert jump_right < 0.15, \
        "the drag direction must be smooth, not a seam (%.2f)" % jump_right
    assert jump_left > 0.5, \
        "the seam must be on the far side (%.2f)" % jump_left


# --------------------------------------- step 03: stroke / fill a selection
# The shape tools, without a shape tool: once there are marquee, ellipse and
# lasso selections, "paint the outline with the current brush" IS the
# rectangle / ellipse / polygon tool -- and it comes out as ORDINARY PAINT.

def test_r74_stroking_a_rect_selection_draws_a_rectangle():
    d = _mk(200, 150)
    L = d.add_layer("s").id
    s = d.select("rect", {"x0": 40, "y0": 30, "x1": 160, "y1": 120})
    assert d.stroke_selection(L, s.id, color=(0.9, 0.1, 0.1), radius=4) == 1
    a = d.layer(L).pixels[..., 3]
    assert float(a[30, 100]) > 0.9, "the top edge is drawn"
    assert float(a[120, 100]) > 0.9 and float(a[75, 40]) > 0.9, "and the rest"
    assert float(a[75, 100]) == 0.0, "the middle stays empty -- it is an outline"
    assert float(a[5, 5]) == 0.0, "nothing outside"


def test_r74_a_stroked_outline_is_ordinary_editable_paint():
    """The whole point: not a special object. One journaled stroke per ring,
    so nudge, restyle and replay all work on it."""
    d = _mk(200, 150)
    L = d.add_layer("s").id
    n0 = len(d.strokes)
    s = d.select("ellipse", {"x0": 40, "y0": 30, "x1": 160, "y1": 120})
    d.stroke_selection(L, s.id, color=(0, 0, 0), radius=3)
    assert len(d.strokes) - n0 == 1
    assert "op" not in d.strokes[-1]["brush"], \
        "an outline is plain paint, not a special op"
    assert d.replay_is_faithful(L)


def test_r74_stroke_selection_carries_the_brushs_media():
    d = _mk(160, 120)
    d.set_paper("rough")
    L = d.add_layer("s").id
    s = d.select("rect", {"x0": 30, "y0": 25, "x1": 130, "y1": 95})
    d.stroke_selection(L, s.id, color=(0.2, 0.3, 0.8), radius=5,
                       media="water", load=0.9)
    assert d.layer(L).pixels[..., 3].max() > 0.05, "a watercolour outline landed"
    assert d.replay_is_faithful(L), "and it replays"


def test_r74_stroke_selection_traces_a_lasso_and_several_rings():
    d = _mk(220, 160)
    L = d.add_layer("s").id
    s = d.select("poly", {"points": _ring(70, 80, 40, 40)})
    s = d.select("poly", {"points": _ring(160, 80, 35, 35)},
                 mode="add", target=s.id)
    assert d.stroke_selection(L, s.id, radius=3) == 2, \
        "two disjoint islands must give two rings"
    a = d.layer(L).pixels[..., 3]
    assert float(a[80, 70]) == 0.0 and float(a[80, 160]) == 0.0, "both hollow"
    assert d.replay_is_faithful(L)


def test_r74_stroke_selection_says_nothing_to_do_rather_than_nothing():
    d = _mk(120, 90)
    L = d.add_layer("s").id
    assert d.stroke_selection(L, None) == 0, "no selection, no rings"
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 120, "height": 90})
    lid = c.get("/api/state").get_json()["layers"][-1]["id"]
    j = c.post("/api/stroke_selection", json={"layer": lid}).get_json()
    assert j["rings"] == 0 and "warning" in j, j


def test_r74_fill_selection_fills_the_gate_and_replays():
    d = _mk(200, 150)
    L = d.add_layer("f").id
    s = d.select("ellipse", {"x0": 40, "y0": 30, "x1": 160, "y1": 120})
    n = d.fill_selection(L, s.id, color=(0.1, 0.3, 0.8))
    assert n > 5000
    px = d.layer(L).pixels
    assert np.allclose(px[75, 100, :3], (0.1, 0.3, 0.8), atol=0.02)
    assert float(px[5, 5, 3]) == 0.0, "the fill stopped at the selection"
    assert d.replay_is_faithful(L), "it journals pixel-free, so it replays"
    d.undo()
    assert float(d.layer(L).pixels[..., 3].max()) == 0.0


def test_r74_fill_with_no_selection_fills_the_layer():
    d = _mk(80, 60)
    L = d.add_layer("f").id
    d.fill_selection(L, None, color=(1, 1, 0))
    a = d.layer(L).pixels[..., 3]
    assert float(a.min()) > 0.99, \
        "Edit > Fill with nothing selected fills the layer, as everywhere else"


def test_r74_selection_outline_needs_no_optional_dependency():
    """It traces with cv2, which the flood fill already needs. scikit-image
    is optional here and missing on plenty of installs, so the shape tool
    must not be the thing that drags it in."""
    import ast
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "lestudio", "__init__.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "selection_outline")
    imported = {a.name.split(".")[0] for n in ast.walk(fn)
                if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(fn)
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert "skimage" not in imported, imported
    assert "cv2" in imported, imported


def test_r74_shape_tools_are_in_the_client():
    import os
    ui = open(os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                           "static", "index.html")).read()
    assert 'id="selStroke"' in ui and 'id="selFill"' in ui and 'id="selStrokeIn"' in ui
    assert "'/api/stroke_selection'" in ui and "'/api/fill_selection'" in ui
    # the Shift straight line, and the preview call that must NOT be a phantom
    assert "R74: SHIFT DRAWS A STRAIGHT LINE" in ui
    i = ui.index("R74: SHIFT DRAWS A STRAIGHT LINE")
    branch = ui[i:i + 1200]
    assert "drawLocalStroke()" in branch and "drawStrokePreview" not in ui, \
        "the preview call must name a function that exists"
    assert "stroke.length=1; stroke.push(" in branch, \
        "a straight line is two points, not an accumulating path"


# ------------------------------------------------- every advertised key works
# The creature brush had NO reachable UI: its button was the fourth in a
# collapsed group whose visible face is Scribble, and its tooltip promised
# Shift+L -- which the polygon lasso answered first, so the creature was two
# presses away behind a label that said one. Scribble, Hatch and Textile were
# advertising (G), (D) and (W) too: (G) is the bucket, (W) is the wand, (D)
# was bound to nothing. A toolbar that shows a key which does not work is
# worse than one that shows none, so this walks EVERY tooltip and proves the
# key it names reaches that tool.

def _ui():
    import os
    return open(os.path.join(os.path.dirname(__file__), "..", "src",
                             "lestudio", "static", "index.html")).read()


def _keymaps(ui):
    import re
    def table(name):
        blk = ui[ui.index("const %s=" % name):]
        blk = blk[:blk.index("};") + 2]
        return dict(re.findall(r"(\w+):'([a-z]+)'", blk))
    plain = table("TOOLKEY")
    # SHIFTTOOLKEY entries may be conditional; collect every tool each can reach
    blk = ui[ui.index("const SHIFTTOOLKEY="):]
    blk = blk[:blk.index("};") + 2]
    shift = {}
    for letter, body in re.findall(r"(\w+):\(?([^,}]+)", blk):
        shift[letter] = set(re.findall(r"'([a-z]+)'", body))
    return plain, shift


def test_r74_every_tool_key_a_tooltip_advertises_actually_works():
    import re
    ui = _ui()
    plain, shift = _keymaps(ui)
    btn2tool = dict(re.findall(r"(\w+):'(t[A-Z]\w+)'", ui[ui.index("const TOOLBTN="):
                                                          ui.index("const TOOLBTN=") + 900]))
    btn2tool = {v: k for k, v in btn2tool.items()}
    problems = []
    for bid, title in re.findall(r'id="(t[A-Z]\w+)"[^>]*title="([^"]*)"', ui):
        tool = btn2tool.get(bid)
        if not tool:
            continue
        m = re.match(r"[^(]*\(([^)]+)\)", title)
        if not m:
            continue                      # advertises no key: nothing to check
        key = m.group(1).strip()
        if key.lower().startswith("shift+"):
            letter = key[6:7].lower()
            if tool not in shift.get(letter, set()):
                problems.append("%s says %s but Shift+%s cannot reach it"
                                % (bid, key, letter))
        elif len(key) == 1:
            if plain.get(key.lower()) != tool:
                problems.append("%s says (%s) but %s maps to %r"
                                % (bid, key, key, plain.get(key.lower())))
    assert not problems, "; ".join(problems)


def test_r74_the_creature_brush_has_a_button_you_can_see():
    """Not hidden behind another tool's face: `.tgw>button:not(.cur)` is
    display:none, so a button without `cur` is invisible until its group is
    opened -- which is how the creature came to have no UI at all."""
    import re
    ui = _ui()
    i = ui.index('id="tCreature"')
    tag = ui[ui.rindex("<button", 0, i):i]
    assert 'class="cur"' in tag, \
        "the creature button is hidden inside a collapsed group"
    grp = ui.rindex('<div class="tgrp', 0, i)
    assert 'data-grp="creature"' in ui[grp:grp + 120], \
        "it should stand on its own, not behind the scribble face"


def test_r74_one_press_reaches_every_family_member():
    ui = _ui()
    plain, shift = _keymaps(ui)
    assert plain.get("a") == "creature", "A is the creature brush"
    assert plain.get("l") == "lasso" and shift.get("l") == {"polylasso"}, \
        "L is the lasso, Shift+L the polygon lasso -- one press each"
    assert plain.get("d") == "scribble", "D opens the generator family"
    assert {"hatch", "textile", "scribble"} <= shift.get("d", set()), \
        "Shift+D cycles the rest of the generators"
    # no tool may be reachable ONLY by cycling through two other tools
    assert "creature" not in shift.get("l", set()), \
        "the creature must not be buried at the end of the lasso cycle again"


def test_r74_the_status_line_tips_name_the_same_keys_as_the_tooltips():
    """Two places tell a person a tool's key -- the button's tooltip and the
    status-line tip. They disagreed for three generators (the tip said G, D,
    W; those are the bucket, nothing, and the wand), so both are checked."""
    import re
    ui = _ui()
    plain, shift = _keymaps(ui)
    bad = []
    for tool, tip in re.findall(r"\n  (\w+):'([A-Z][^']*?) ·", ui):
        m = re.match(r"[^—]*— ([A-Za-z+]+)", tip)
        if not m:
            continue
        key = m.group(1)
        if key.lower().startswith("shift+"):
            if tool not in shift.get(key[6:7].lower(), set()):
                bad.append("%s tip says %s" % (tool, key))
        elif len(key) == 1 and plain.get(key.lower()) != tool:
            bad.append("%s tip says (%s) but that is %r"
                       % (tool, key, plain.get(key.lower())))
    assert not bad, "; ".join(bad)
