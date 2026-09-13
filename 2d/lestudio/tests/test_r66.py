"""tests/test_r66.py -- the paint physics the kit could not reach.

Devin: "When I said it lacked realism, I meant that it looked like a
cartoon... Also be aware that leStudio has a material library and a variety
of media that can be used."

He was right twice. Every source on why a painting reads as a cartoon names
the same faults, and the R64/R65 still life had all of them -- but the one
that mattered here is the last on the list: brushwork "uniformly flat, as
though applied with a roller". That is LITERALLY what a scanline fill of
flat colour is, and flat colour was the only thing tools/painter.py could
express. leStudio ships a physical paint model (oil holds a ridge and takes
a satin specular off its own slopes; acrylic holds a bristle comb; water
wicks into the paper and dries with a dark rim), an eleven-entry PBR
material library, brush PICKUP, and ten blend modes -- and the kit every
scripted painter and the agent uses reached none of it.
"""


def _kit():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "painter_kit", os.path.join(os.path.dirname(__file__), "..", "tools",
                                    "painter.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Offline:
    """A Painter that collects strokes instead of sending them."""

    def __new__(cls, kit):
        import random
        p = object.__new__(kit.Painter)
        p.W, p.H = 200, 140
        p.batch, p.refusals, p.skipped_passes = [], [], []
        p._pass = p._pass_run = None
        p._force = False
        p._paint = {}
        p.rnd = random.Random(3)
        p.lids = {}
        p.dry = False
        p.counts, p.sent = {}, 0
        p.flush = lambda: None
        return p


def test_r66_the_kit_can_reach_the_paint_physics():
    """`media`, `material`, `load`, `mix` and `real_brush` all reach the
    engine from the kit, nest inside a `medium()` block, and an inner block
    overrides one field without discarding the rest."""
    kit = _kit()
    p = _Offline(kit)
    p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4)
    assert "media" not in p.batch[-1], "plain by default"

    with p.medium("oil", load=0.6, real_brush=True):
        p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4)
        d = p.batch[-1]
        assert d["media"] == "oil" and d["load"] == 0.6 and d["real_brush"] is True
        with p.medium(mix=0.5):
            p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4)
            d = p.batch[-1]
            assert d["mix"] == 0.5, "an inner block adds"
            assert d["media"] == "oil" and d["load"] == 0.6, "...without discarding"
        p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4)
        assert "mix" not in p.batch[-1], "leaving the inner block drops only its own"
        assert p.batch[-1]["media"] == "oil"
    p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4)
    assert "media" not in p.batch[-1], "the block restores what it found"

    # a per-stroke override, and a typo that would silently do nothing
    p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4, material="clay")
    assert p.batch[-1]["material"] == "clay"
    try:
        p.st("L1", [[1, 1], [9, 9]], [1, 0, 0], 4, medium="oil")
        assert False, "a misspelt paint setting must not pass silently"
    except TypeError as e:
        assert "unknown paint setting" in str(e)

    # an eraser carries no paint: erasing THROUGH a medium made no sense and
    # the engine would have been asked to deposit pigment while removing it
    with p.medium("oil", load=0.8, mix=0.5, material="clay", real_brush=True):
        p.st("L1", [[1, 1], [9, 9]], [0, 0, 0], 4, erase=True)
    d = p.batch[-1]
    assert d["erase"] is True
    assert not any(k in d for k in ("media", "material", "mix", "real_brush"))

    # fill() and line() carry it too -- a mass is where it matters most
    p.batch = []
    with p.medium("oil", load=0.5):
        p.fill("L1", [[0, 0], [40, 0], [40, 40], [0, 40]], lambda x, y: [1, 1, 1],
               radius=5.0)
        p.line("L1", (0, 0), (30, 30), [1, 1, 1], 4, mix=0.3)
    assert all(s.get("media") == "oil" for s in p.batch), "fill and line too"
    assert p.batch[-1]["mix"] == 0.3


def test_r66_a_real_brush_runs_out_and_the_kit_dips():
    """leStudio models a real reservoir: with `real_brush` on, the brush
    holds a finite charge and, once spent, every further stroke deposits
    NOTHING -- silently, at HTTP 200. Measured on a live server: charge
    0.725, 0.450, 0.175, 0.000, and then seven more strokes that changed
    not one pixel. Correct physics; a trap for a script, which has no hand
    to feel the brush go dry with. The kit dips before each real-brush
    stroke, which is what a painter does without thinking about it."""
    kit = _kit()
    p = _Offline(kit)
    dips = []
    sent = []
    p.dip = lambda color=None, amount=1.0: dips.append(color)
    p.post = lambda path, body: sent.append((path, body)) or {}
    del p.flush                                   # use the real one

    with p.medium("oil", load=0.6, real_brush=True):
        for i in range(4):
            p.st("L1", [[i, 1], [i + 9, 9]], [0.5, 0.2, 0.1], 4)
    kit.Painter.flush(p)
    # ...and it dips IN BAND, on the batch item, so a real-brush pass is
    # still one request. Dipping out of band meant two round trips per
    # mark, and the R66 accents pass took twelve minutes for no other
    # reason. The physics is unchanged: the engine reloads the brush
    # immediately before each of these strokes, in its own colour.
    assert len(sent) == 1, ("a real-brush pass still batches", len(sent))
    assert len(dips) == 0, "no out-of-band /api/brush_load round trips"
    strokes = sent[0][1]["strokes"]
    assert len(strokes) == 4
    assert all(s["dip"] is True for s in strokes), "one dip per mark"
    assert all(s["real_brush"] is True for s in strokes)

    # and an ordinary stroke is NOT dipped -- only real_brush spends a
    # reservoir, and dipping for a mass would reload the brush 26,000 times
    sent[:] = []
    p.batch = []
    with p.medium("oil", load=0.6):
        p.st("L1", [[1, 1], [9, 9]], [0.5, 0.2, 0.1], 4)
    kit.Painter.flush(p)
    assert "dip" not in sent[0][1]["strokes"][0]
    assert all(len(b["strokes"]) == 1 for _, b in sent), \
        "real-brush strokes go one at a time -- they each spend the reservoir"

    # ordinary strokes still batch: a MASS must not pay the per-stroke cost
    p2 = _Offline(kit)
    sent2 = []
    p2.dip = lambda color=None, amount=1.0: dips.append("nope")
    p2.post = lambda path, body: sent2.append(body) or {}
    del p2.flush
    with p2.medium("oil", load=0.6):
        for i in range(30):
            p2.st("L1", [[i, 1], [i + 9, 9]], [0.5, 0.2, 0.1], 4)
    kit.Painter.flush(p2)
    assert len(sent2) == 1 and len(sent2[0]["strokes"]) == 30, \
        "a plain medium does not deplete, so it batches"


def test_r66_the_engine_really_does_all_of_this():
    """The physics is not decoration. Against the live engine: oil, acrylic
    and water each paint, `mix` drags what is already on the canvas into
    the stroke (broken colour -- the fix for 'one flat local tint per
    object'), and every advertised material resolves."""
    import numpy as np
    from lestudio import Document, _MATERIALS, _MEDIA

    assert set(_MEDIA) >= {"oil", "acrylic", "water"}
    assert len(_MATERIALS) >= 10 and "clay" in _MATERIALS and "copper" in _MATERIALS

    d = Document(240, 160)
    pts = [[30 + i * 6, 80] for i in range(30)]
    for media in ("oil", "acrylic", "water"):
        lid = d.add_layer(media).id
        d.paint(lid, pts, color=(0.8, 0.3, 0.2), radius=12, opacity=1.0,
                media=media, load=0.6)
        assert (d.layer(lid).pixels[..., 3] > 0.1).sum() > 500, media

    # every material name in the library is paintable
    for name in _MATERIALS:
        lid = d.add_layer("m_" + name).id
        d.paint(lid, pts, color=(0.7, 0.5, 0.3), radius=10, opacity=1.0,
                media="oil", load=0.5, material=name)
        assert (d.layer(lid).pixels[..., 3] > 0.1).sum() > 300, name

    # PICKUP: a stroke laid across wet red, with mix on, must not stay pure
    # blue -- that is broken colour, and it is what stops every object being
    # one flat tint
    base = d.add_layer("wet").id
    d.paint(base, [[20, 80], [220, 80]], color=(0.9, 0.15, 0.05), radius=22,
            opacity=1.0, media="oil", load=1.0)
    d.paint(base, [[20, 80], [220, 80]], color=(0.05, 0.2, 0.9), radius=10,
            opacity=1.0, media="oil", load=0.6, mix=0.9)
    px = d.layer(base).pixels
    band = px[74:86, 60:180]
    lit = band[band[..., 3] > 0.5]
    assert len(lit) > 100
    # pure blue would have r << g < b everywhere; pickup lifts the red
    assert float(lit[:, 0].mean()) > 0.12, \
        ("pickup must drag the underlying red into the blue",
         float(lit[:, 0].mean()))


def test_r66_the_layer_stack_is_a_painters_stack():
    """The blend modes a grisaille workflow needs are all there, and a
    layer really carries one. Value lives in an opaque monochrome
    underpainting; colour is GLAZED over it on a multiply layer so the
    value beneath shows through and mixes optically; light is added on a
    screen layer. That optical stacking is where luminosity comes from, and
    it is the thing painting flat colour directly cannot produce."""
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": 120, "height": 90})
    modes = c.get("/api/state").json["blend_modes"]
    for needed in ("multiply", "screen", "overlay", "add", "softlight"):
        assert needed in modes, needed

    lid = c.post("/api/layer", json={"action": "add", "name": "Glaze"}).json["id"]
    assert c.post("/api/layer", json={"action": "edit", "id": lid,
                                      "blend": "multiply"}).status_code == 200
    meta = {l["id"]: l for l in c.get("/api/state").json["layers"]}[lid]
    assert meta["blend"] == "multiply"

    # and multiply really darkens what is under it rather than replacing it
    import numpy as np
    d = WS.doc
    bot = d.layers[0].id
    d.paint(bot, [[10, 45], [110, 45]], color=(0.9, 0.9, 0.9), radius=30,
            opacity=1.0)
    d.paint(lid, [[10, 45], [110, 45]], color=(0.5, 0.2, 0.2), radius=30,
            opacity=1.0)
    out = np.asarray(d.composite())
    px = out[45, 60]
    assert px[0] < 0.75 and px[1] < 0.45, ("multiply must darken", px[:3])


def test_r66_a_mass_that_runs_off_the_canvas_is_painted_off_the_canvas():
    """A wall, a ground, a counter -- any mass whose polygon extends past
    the canvas -- must be painted past the canvas too.

    fill() used to rasterise the polygon into a canvas-sized mask, which
    CLIPPED it, and then walked that. So the canvas edge looked exactly
    like the shape's edge: the first scanline sat on row 0 and every row
    near it got dab coverage from one side only. Measured on the R66 room
    grisaille, row 0 came back at alpha 0.52 and row 1 at 0.75 against
    0.95 in the middle of the picture -- the dark ground grinning through
    as a ragged frame round all four sides, which is precisely the class
    of artifact Devin has already had to point at twice.

    The mask is now padded by a radius, so an over-extended polygon really
    does get scanlines at negative y and strokes that start left of zero.
    A shape that genuinely ends inside the canvas is untouched: padding
    adds no scanlines where the polygon has no area."""
    kit = _kit()
    p = _Offline(kit)

    # a ground deliberately drawn past every edge, exactly as the room is
    p.batch = []
    p.fill("L1", [[-50, -50], [p.W + 50, -50], [p.W + 50, p.H + 50], [-50, p.H + 50]],
           lambda x, y: [1, 1, 1], radius=12.0, step=6.0, seg=12.0)
    ys = sorted({s["points"][0][1] for s in p.batch})
    xs = [pt[0] for s in p.batch for pt in s["points"]]
    assert ys[0] < 0, ("scanlines must start above the canvas", ys[0])
    assert ys[-1] > p.H, ("...and finish below it", ys[-1])
    assert min(xs) < 0 and max(xs) > p.W, ("and reach past both sides",
                                           min(xs), max(xs))

    # the same coverage argument, counted, and it must be EXACT: the
    # boundary scanline lands on a different phase of the pitch from the
    # interior, so "within one" still left rows 0-3 measurably darker than
    # row 10 right across the picture. Every canvas row gets the same
    # number of passes over it as the middle of the picture does.
    def reach(row, r=12.0):
        return sum(1 for y in ys if abs(y - row) <= r)
    mid = min(reach(r) for r in range(40, p.H - 40))
    for row in (0, 1, 2, 3, p.H - 4, p.H - 3, p.H - 2, p.H - 1):
        assert reach(row) >= mid, ("row %d is under-covered: %d vs %d"
                                   % (row, reach(row), mid))

    # ...and a shape that stops inside the canvas is NOT smeared outward
    p.batch = []
    p.fill("L1", [[60, 50], [120, 50], [120, 100], [60, 100]],
           lambda x, y: [1, 1, 1], radius=6.0, step=3.0, seg=6.0)
    ys = sorted({s["points"][0][1] for s in p.batch})
    assert ys[0] >= 50 - 3.0 * 0.35 - 1 and ys[-1] <= 100 + 3.0 * 0.35 + 1, (
        "a closed shape keeps its own bounds, give or take the jitter",
        ys[0], ys[-1])

    # ...and the scanlines are NOT on a ruler. A regular pitch beats against
    # the gradient it is laying: measured, every lemon came back with a
    # period-5 ripple of +/-0.018 down it, which on a smooth sphere reads
    # as contour banding. The same fill laid soft measured +/-0.001, so it
    # is the pitch, not the coverage.
    p.batch = []
    p.fill("L1", [[60, 50], [120, 50], [120, 100], [60, 100]],
           lambda x, y: [1, 1, 1], radius=6.0, step=3.0, seg=6.0)
    def scanrows(batch):            # the RUNS, not the contour's 1 px dabs
        return sorted({s["points"][0][1] for s in batch
                       if s["points"][-1][0] - s["points"][0][0] > 1.0})
    gaps = sorted({round(b - a, 3) for a, b in zip(scanrows(p.batch),
                                                   scanrows(p.batch)[1:])})
    assert len(gaps) > 4, ("the scanline pitch must not be constant", gaps)
    p.batch = []
    p.fill("L1", [[60, 50], [120, 50], [120, 100], [60, 100]],
           lambda x, y: [1, 1, 1], radius=6.0, step=3.0, seg=6.0, jitter=0.0)
    rows = scanrows(p.batch)
    assert all(abs((b - a) - 3.0) < 1e-6 for a, b in zip(rows, rows[1:])), (
        "...but jitter=0 must still give an exact pitch when asked for", rows)

    # the colour function is still asked in CANVAS coordinates, or every
    # gradient in the picture would shift by a radius
    seen = []
    p.batch = []
    p.fill("L1", [[-20, -20], [p.W + 20, -20], [p.W + 20, 40], [-20, 40]],
           lambda x, y: (seen.append((x, y)), [1, 1, 1])[1],
           radius=10.0, step=5.0, seg=10.0)
    assert min(y for _, y in seen) < 0, "asked above the canvas"
    assert max(y for _, y in seen) <= 41, ("and not past the polygon",
                                           max(y for _, y in seen))


def test_r66_a_setting_that_is_dropped_is_worse_than_one_that_is_refused():
    """Two silent no-ops, both found by painting a whole picture wrong.

    `{"action": "settings", "paper": "smooth"}` answered ok:true and
    changed nothing -- the stock lived only on /api/paper -- so the R66
    still life was painted on heavy canvas weave while the script and its
    log both said smooth. Nothing anywhere said no. The stock is a
    document setting and now lives where document settings live, and the
    route refuses a key it does not understand rather than accepting it
    and forgetting it.

    The same pass also needed a dip INSIDE a batch. `real_brush` models a
    finite charge -- three long strokes and the brush is dry, after which
    every stroke deposits nothing at HTTP 200 -- and the only refill was a
    separate POST, so a real-brush pass could not be batched at all."""
    import numpy as np
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": 140, "height": 100})

    # the stock, set the obvious way, and this time it takes
    assert c.get("/api/state").json["paper"] == "canvas"
    r = c.post("/api/doc", json={"action": "settings", "paper": "smooth"})
    assert r.status_code == 200, r.json
    assert c.get("/api/state").json["paper"] == "smooth", "it must STICK"

    # a stock that does not exist is a 400, not a shrug
    r = c.post("/api/doc", json={"action": "settings", "paper": "velvet"})
    assert r.status_code == 400
    assert c.get("/api/state").json["paper"] == "smooth", "and changes nothing"

    # a key this route does not understand is refused, and names itself
    r = c.post("/api/doc", json={"action": "settings", "papper": "smooth"})
    assert r.status_code == 400, "a dropped setting must not answer ok"
    assert "papper" in r.json["error"] and "paper" in r.json["error"]
    # ...and the settings it DOES take still work together
    assert c.post("/api/doc", json={"action": "settings", "width": 150,
                                    "height": 110, "dpi": 144,
                                    "paper": "linen"}).status_code == 200
    st = c.get("/api/state").json
    assert (st["width"], st["height"], st["paper"]) == (150, 110, "linen")

    # ---- the in-band dip
    lid = WS.doc.layers[0].id
    mark = {"layer": lid, "points": [[10, 50], [140, 50]],
            "color": [0.9, 0.1, 0.1], "radius": 9, "opacity": 1.0,
            "hardness": 0.5, "media": "oil", "load": 0.6, "real_brush": True}
    # eight loaded marks in ONE batch, each dipping first
    r = c.post("/api/paint_batch",
               json={"strokes": [dict(mark, dip=True) for _ in range(8)]})
    assert r.status_code == 200, r.json
    after = np.asarray(WS.doc.layers[0].pixels)[:, :, 3].copy()
    assert after.max() > 0.2, "eight dipped marks must leave paint"

    # and without the dip the same eight run the brush dry: strictly less
    # paint lands. (This is the engine's physics, not a bug -- the point is
    # that a batch can now answer it.)
    for L in WS.doc.layers:
        L.pixels[:] = 0
    c.post("/api/brush_load", json={"color": [0.9, 0.1, 0.1], "amount": 1.0})
    c.post("/api/paint_batch", json={"strokes": [dict(mark) for _ in range(8)]})
    dry = np.asarray(WS.doc.layers[0].pixels)[:, :, 3]
    assert dry.sum() < after.sum(), ("a dry brush must lay less paint than a "
                                     "dipped one", float(dry.sum()),
                                     float(after.sum()))

    # a malformed dip is a 400 with a sentence, not a traceback
    r = c.post("/api/paint_batch", json={"strokes": [dict(mark, dip="lots")]})
    assert r.status_code == 400 and "dip" in r.json["error"]


def test_r66_a_painting_pass_can_be_rehearsed_before_it_is_painted():
    """`Painter(..., dry=True)` builds every stroke and sends none.

    A stage of this picture is minutes of engine time, and fill() asks a
    colour function for points OUTSIDE the shape and past the edges of the
    canvas -- so a hand-written value function with a bare `**` in it
    raises on a negative coordinate and takes the whole pass with it. That
    cost forty minutes to discover once. A rehearsal answers the same
    question in seconds, and the only difference from the real thing is
    that nothing is sent."""
    kit = _kit()
    p = kit.Painter("u_t", canvas=(200, 140), dry=True)
    assert (p.W, p.H) == (200, 140)
    p.add_layer("Grisaille")

    def value(x, y):
        assert isinstance(x, float) and isinstance(y, float)
        return [0.5, 0.5, 0.5]

    p.fill("Grisaille", [[-40, -40], [240, -40], [240, 180], [-40, 180]],
           value, radius=10.0, step=5.0, seg=10.0)
    p.line("Grisaille", (0, 0), (60, 60), [1, 1, 1], 4)
    p.flush()
    assert p.sent == 0, ("a rehearsal sends NOTHING", p.sent)
    assert p.counts.get("Grisaille", 0) > 100, p.counts
    assert p.batch == []

    # it reads too, so a pass that judges what is under it rehearses
    under = p.below("Grisaille")
    assert under is not None and under.shape == (140, 200, 4)
    assert all(abs(v - 0.45) < 1e-6 for v in p.sample(under, 100, 70))

    # and a colour function that cannot take a negative coordinate fails
    # HERE, with the sentence that says what to use instead
    def broken(x, y):
        return kit.lerp([0, 0, 0], [1, 1, 1], (x / 200.0) ** 0.7)

    try:
        p.fill("Grisaille", [[-40, 10], [240, 10], [240, 60], [-40, 60]],
               broken, radius=10.0, step=5.0, seg=10.0)
        assert False, "a complex t must not reach the engine"
    except TypeError as e:
        assert "pw(" in str(e) and "complex" in str(e), str(e)
