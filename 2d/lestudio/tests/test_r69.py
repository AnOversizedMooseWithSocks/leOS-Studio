"""tests/test_r69.py -- "None of them allowed me to perform strokes."

Devin, on the scribble / hatch-shading / textile tools:

    None of them allowed me to perform strokes, and only allowed me to
    click to perform a stamp/fill sort of action. The textile brush was
    VERY slow and unresponsive [...] Scribble brush didn't respect brush
    tool properties, and I don't think the hatch brush did either. The
    textile brush definitely ignored the brush tool settings.

Three separate defects, one per complaint:

1. THEY COULD NOT STROKE. The generators only ever took a centre point and
   a radius, and the client fired them on pointerdown and returned -- so a
   drag painted one disc at the press point and nothing after it. That is
   the behaviour of a stamp tool, which is what they felt like. The engine
   now takes a `path` and builds its gate by sweeping the brush nib along
   it (`_path_gate`); the client captures the drag and posts the swept
   path on release.

2. THEY IGNORED THE BRUSH PANEL. They sit in the brush row and carry the
   brush cursor, but read neither size, colour, opacity, hardness nor the
   selection gate. They all read the brush panel now, with each tool's own
   radius slider kept as a fallback.

3. TEXTILE WAS UNUSABLY SLOW. Every one of its hundreds of tiny thread
   strokes re-ran the relief shading pass over the whole layer, because
   the shade memo is keyed on a revision that each paint bumps. The fill
   now shades ONCE at the end (`shading_deferred`), which is the same
   picture for a fraction of the work: stitch 12.85s -> 2.93s.
"""
import io
import os
import socket
import threading
import time

import numpy as np

UI = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                  "static", "index.html")


def _doc(w=320, h=240):
    from lestudio import Document
    d = Document(w, h)
    d.add_layer("gen")
    return d


def _alpha(layer):
    """Layer pixels are float32 RGBA in 0..1."""
    return np.asarray(layer.pixels)[..., 3]


INK = 0.03          # "there is ink here" -- alpha is float 0..1 here,
                    # while the RGB channels stay 0..255. Every generator
                    # is seeded in these tests: they spend their randomness
                    # at generation time, so an unseeded run is a different
                    # (equally valid) picture and nothing can be compared.


# ---------------------------------------------------------------- 1. strokes

def test_r69_a_scribble_follows_a_dragged_path():
    """A long diagonal drag must put ink along the whole path, not in a
    disc at one end of it. Pre-fix the seeding drew from a disc around the
    centre point and broke out the moment a strand left that disc, so a
    path produced a blob at the midpoint."""
    d = _doc()
    l = d.layers[-1]
    path = [[30, 30], [90, 70], [150, 110], [220, 160], [280, 200]]
    d.scribble(l.id, 0, 0, radius=9, path=path, color=(10, 10, 10),
               density=1.6, seed=11)
    a = _alpha(l)
    ys, xs = np.where(a > INK)
    assert len(xs) > 300, "no ink at all"
    # Ink over the WHOLE path, measured in thirds along it -- the pre-fix
    # blob sat at the midpoint, which is exactly what a third-by-third
    # test catches and a bounding-box test does not.
    for lo, hi, where in ((30, 113, "first third"), (113, 196, "middle"),
                          (196, 280, "last third")):
        n = ((xs >= lo) & (xs < hi)).sum()
        assert n > 40, "%s of the drag got %d px of ink" % (where, n)


def test_r69_b_ink_stays_inside_the_swept_band():
    """The path gate is the nib swept along the path. Ink outside that band
    would mean the gate is not actually steering the generator."""
    d = _doc()
    l = d.layers[-1]
    path = [[40, 120], [140, 120], [240, 120]]
    r = 12
    d.hatch_fill(l.id, 0, 0, radius=r, path=path, color=(10, 10, 10))
    a = _alpha(l)
    ys, xs = np.where(a > INK)
    assert len(xs) > 200
    # distance from the horizontal segment y=120, x in [40,240]
    dx = np.clip(xs, 40, 240)
    dist = np.hypot(xs - dx, ys - 120)
    inside = (dist <= r * 1.9).mean()
    assert inside > 0.95, "only %.1f%% of the ink is in the band" % (inside * 100)


def test_r69_c_a_path_paints_a_different_region_than_a_click():
    """The regression that would put us back where we started: a path being
    quietly dropped and the tool falling back to the click behaviour."""
    d = _doc()
    l1, l2 = d.add_layer("a"), d.add_layer("b")
    mid = [160, 120]
    d.hatch_fill(l1.id, mid[0], mid[1], radius=14, color=(10, 10, 10),
                 seed=3)
    d.hatch_fill(l2.id, 0, 0, radius=14, path=[[20, 20], [300, 220]],
                 color=(10, 10, 10), seed=3)
    a1, a2 = _alpha(l1) > INK, _alpha(l2) > INK
    assert a1.sum() > 50 and a2.sum() > 50
    # the click's disc and the long diagonal band can overlap a little, but
    # they cannot be the same region
    iou = (a1 & a2).sum() / float((a1 | a2).sum())
    assert iou < 0.35, "path and click painted the same region (iou %.2f)" % iou


def test_r69_d_a_one_point_path_still_behaves_like_a_click():
    """A click is a path of one point. It must keep working -- people who
    liked the stamp behaviour did not lose it."""
    d = _doc()
    l = d.layers[-1]
    d.scribble(l.id, 160, 120, radius=30, path=[[160, 120]],
               color=(10, 10, 10))
    assert (_alpha(l) > INK).sum() > 100


# ------------------------------------------------------- 2. brush properties

def test_r69_e_generators_honour_colour_and_opacity():
    """'Scribble brush didn't respect brush tool properties.'"""
    d = _doc()
    red, faint = d.add_layer("red"), d.add_layer("faint")
    path = [[40, 60], [260, 60]]
    d.scribble(red.id, 0, 0, radius=10, path=path, color=(220, 20, 20),
               opacity=1.0, density=1.4, seed=5)
    d.scribble(faint.id, 0, 0, radius=10, path=path, color=(220, 20, 20),
               opacity=0.2, density=1.4, seed=5)
    rp = np.asarray(red.pixels)
    hit = rp[..., 3] > 0.4
    assert hit.sum() > 100
    assert rp[hit][:, 0].mean() > 150 and rp[hit][:, 1].mean() < 90, \
        "colour ignored: mean rgb %s" % rp[hit][:, :3].mean(0)
    assert _alpha(faint).mean() < _alpha(red).mean() * 0.6, "opacity ignored"


def test_r69_f_generators_honour_a_selection_gate():
    """The brush panel's selection gate has to clip a generator the same
    way it clips a brush, or 'limited to selection' is a lie for half the
    tool row."""
    d = _doc()
    l = d.layers[-1]
    sel = d.select("rect", {"x0": 0, "y0": 0, "x1": 150, "y1": 240},
                   name="left half")
    d.hatch_fill(l.id, 0, 0, radius=14, path=[[20, 120], [300, 120]],
                 color=(10, 10, 10), selection=sel.id)
    a = _alpha(l)
    assert a[:, :150].sum() > 0, "nothing painted inside the selection"
    assert a[:, 165:].max() < INK, "ink escaped the selection"


def test_r69_g_the_client_reads_the_brush_panel_for_every_generator():
    ui = open(UI).read()
    assert "function genCommon()" in ui and "function genRadius(" in ui
    for fn in ("doScribble", "doHatch", "doTextile"):
        i = ui.index("async function %s(" % fn)
        body = ui[i:i + 1400]
        assert "...g" in body or "...genCommon()" in body, fn
        assert "genRadius(" in body, fn
    # the old hidden-panel tolerance read is gone
    assert "$('fiTol')" not in ui.split("async function doTextile")[1][:1200]


def test_r69_h_the_client_captures_a_drag_for_the_generators():
    """The client half of complaint 1: pointerdown must open a path instead
    of firing a one-shot request and returning."""
    ui = open(UI).read()
    assert "const GENTOOLS=" in ui
    assert "let genStroke=null" in ui
    assert "if(GENTOOLS[tool]){" in ui
    # and the release posts it
    post = ui[ui.index("  if(genStroke){\n    const g=genStroke"):][:900]
    for fn in ("doScribble", "doHatch", "doTextile"):
        assert fn in post, fn
    assert "path:path||undefined" in ui
    # the one-shot handlers are gone
    assert "if(tool==='scribble'){ const [x,y]=canvasXY(e); doScribble(x,y); return; }" not in ui


def test_r69_i_the_textile_panel_owns_its_own_tolerance():
    """It used to read `fiTol`, which lives in the FILL panel -- hidden
    whenever textile is the active tool."""
    ui = open(UI).read()
    assert 'id="txtTol"' in ui
    assert 'id="txtColor"' not in ui, "dead thread-colour input still present"
    assert "function txtRegionRows()" in ui


# -------------------------------------------------------------- 3. the speed

def test_r69_j_a_textile_fill_shades_once_not_once_per_thread():
    """The performance defect, pinned by counting the work rather than the
    clock: a fill of N thread strokes must run the relief shading pass a
    handful of times, not N times."""
    from lestudio import Document
    d = Document(320, 240)
    l = d.add_layer("cloth")
    l.height_map = np.zeros((240, 320), dtype=np.float32)
    import lestudio
    calls = []
    real = lestudio._shaded_pixels
    lestudio._shaded_pixels = lambda *a, **k: (calls.append(1),
                                               real(*a, **k))[1]
    try:
        n = d.hatch_fill(l.id, 160, 120, radius=45, mode="stitch",
                         spacing=7, thickness=1.6, depth=35,
                         color=(150, 30, 60), seed=9)
    finally:
        lestudio._shaded_pixels = real
    n = n if isinstance(n, int) else (n or {}).get("strokes", 0)
    assert n > 40, "fill did not produce a meaningful number of strokes: %r" % n
    assert len(calls) <= 6, \
        "shaded %d times for %d strokes -- the per-stroke pass is back" % (
            len(calls), n)


def test_r69_k_deferring_the_shading_does_not_change_the_picture():
    """The determinism law: this is a speed fix, so the pixels must be
    identical to shading after every stroke."""
    from lestudio import Document

    def build(defer):
        d = Document(160, 120)
        l = d.add_layer("cloth")
        l.height_map = np.zeros((120, 160), dtype=np.float32)
        if defer:
            d.hatch_fill(l.id, 80, 60, radius=30, mode="cross", spacing=8,
                         thickness=1.4, depth=30, color=(150, 30, 60),
                         seed=7)
        else:
            import contextlib
            cm = d.shading_deferred
            d.shading_deferred = lambda: contextlib.nullcontext()
            try:
                d.hatch_fill(l.id, 80, 60, radius=30, mode="cross", spacing=8,
                             thickness=1.4, depth=30, color=(150, 30, 60),
                             seed=7)
            finally:
                d.shading_deferred = cm
        return np.asarray(d.layers[-1].pixels).copy()

    assert np.array_equal(build(True), build(False)), \
        "deferred shading changed the pixels"


# ------------------------------------------------------------ 4. the round trip

def test_r69_l_the_route_takes_a_path_and_shrugs_off_junk():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 200, "height": 160})
    st = c.get("/api/state").get_json()
    lid = st["layers"][-1]["id"]
    ok = c.post("/api/scribble", json={
        "layer": lid, "x": 100, "y": 80, "radius": 10,
        "path": [[20, 20], [90, 70], [170, 130]], "color": [10, 10, 10]})
    assert ok.status_code == 200, ok.get_data(as_text=True)
    for junk in (5, "nope", [[1]], [["a", "b"]], [[float("nan"), 1]],
                 [[1, 2]] * 9000):
        r = c.post("/api/hatchfill", json={
            "layer": lid, "x": 100, "y": 80, "radius": 10, "path": junk,
            "color": [10, 10, 10]})
        assert r.status_code == 200, (junk if not isinstance(junk, list)
                                      or len(junk) < 5 else "long", r.status_code)


def test_r69_m_a_dragged_generator_stroke_survives_a_round_trip():
    """It is journalled like any other paint, so the document must rebuild
    to the same pixels."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(200, 160)
    l = d.add_layer("gen")
    d.hatch_fill(l.id, 0, 0, radius=12, path=[[20, 40], [180, 120]],
                 color=(10, 10, 10))
    before = np.asarray(d.layers[-1].pixels).copy()
    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    after = np.asarray(docs[active].layers[-1].pixels)
    assert np.abs(after - before).max() < 1e-5, \
        "the dragged generator stroke did not survive a .lews round trip"


# ------------------------------------------------------------------ 5. e2e

def test_r69_n_a_real_mouse_drag_paints_a_band_not_a_dab():
    """The whole complaint, end to end, through a real browser: press,
    drag, release with the scribble tool and the ink must span the drag."""
    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    from lestudio.server import app, WS
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    threading.Thread(target=lambda: app.run(port=port, use_reloader=False),
                     daemon=True).start()
    time.sleep(1.0)
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        try:
            pg = b.new_page(viewport={"width": 1366, "height": 800})
            pg.goto("http://127.0.0.1:%d" % port)
            for _ in range(200):
                if pg.evaluate("typeof state!=='undefined'&&!!state"):
                    break
                pg.wait_for_timeout(50)
            pg.wait_for_timeout(600)
            pg.evaluate("""()=>{window.__posts=[];
              const of=window.fetch;
              window.fetch=function(u,o){ if((''+u).indexOf('/api/scribble')>=0)
                window.__posts.push(JSON.parse(o.body)); return of.apply(this,arguments); };}""")
            pg.evaluate("()=>{ setTool('scribble'); $('bSize').value=10; "
                        "$('bSize').dispatchEvent(new Event('input',{bubbles:true})); }")
            box = pg.eval_on_selector("#view", "e=>{const r=e.getBoundingClientRect();"
                                      "return {x:r.x,y:r.y,w:r.width,h:r.height}}")
            x0, y0 = box["x"] + box["w"] * 0.25, box["y"] + box["h"] * 0.3
            x1, y1 = box["x"] + box["w"] * 0.7, box["y"] + box["h"] * 0.65
            pg.mouse.move(x0, y0)
            pg.mouse.down()
            for i in range(1, 25):
                pg.mouse.move(x0 + (x1 - x0) * i / 24, y0 + (y1 - y0) * i / 24)
                pg.wait_for_timeout(10)
            pg.mouse.up()
            pg.wait_for_timeout(3000)
            posts = pg.evaluate("window.__posts")
            assert len(posts) == 1, "expected one request for one drag: %d" % len(posts)
            body = posts[0]
            assert body.get("path") and len(body["path"]) >= 8, \
                "the drag did not reach the server as a path: %r" % (
                    body.get("path") and len(body["path"]))
            assert body.get("radius") == 10, "brush size did not reach the server: %r" % body.get("radius")
            assert "opacity" in body and "color" in body
            lid = body["layer"]
            lay = [x for x in WS.doc.layers if x.id == lid][0]
            a = np.asarray(lay.pixels)[..., 3] > INK
            ys, xs = np.where(a)
            assert len(xs) > 200, "no ink"
            span = max(xs.max() - xs.min(), ys.max() - ys.min())
            assert span > 120, "ink spans only %d px -- that is a dab" % span
        finally:
            b.close()
