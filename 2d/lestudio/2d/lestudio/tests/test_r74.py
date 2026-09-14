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
    assert "creature:'Creature — Shift+L" in ui, \
        "the creature brush moved off L and must say so"
    assert "Creature (Shift+L)" in ui, "its tooltip must agree"
    assert "function selModFromEvent" in ui and "const SHIFTTOOLKEY=" in ui
    # the promise the brush panel has been making since R70 is now true
    assert "marquee, wand and lasso tools" in ui
