"""tests/test_r73.py -- the creature brush.

Devin: "give a brush freedom to sort of do its own thing for a bit" --
Substance's paint-rolling balls, the people who trace a ladybird's walk.
Up to ten leCore CreatureMinds explore the layer as a MAP for a set time
and each paints its walk. Same generator contract as scribble: the
exploration is spent at generation time and every walk lands in the
journal as ONE ordinary stroke, so replay/undo/nudge need nothing new.

What is pinned here is INTENT, not a picture: the strokes are plain
journaled paint, a seed reproduces, the rules actually change where a
creature goes (a light-seeker ends up brighter than a dark-seeker), a
drag spawns them along the path, a selection contains them, junk on the
route is refused not crashed, and the client reads the brush panel.
"""
import os
import numpy as np

UI = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                  "static", "index.html")


def _mk(w=240, h=180):
    from lestudio import Document
    return Document(w, h)


def test_r73_a_walks_are_journaled_strokes_and_replay():
    d = _mk()
    L = d.add_layer("cre").id
    n0 = len(d.strokes)
    r = d.creature_paint(L, 120, 90, seconds=1.0, creatures=4, seed=3)
    assert r["strokes"] == 4 and len(d.strokes) - n0 == 4, \
        "one ordinary journaled stroke per creature"
    assert all("op" not in d.strokes[i]["brush"]
               for i in range(n0, len(d.strokes)))
    assert r["steps"] == 60, "time is steps at 60/s, not the wall clock"
    assert d.replay_is_faithful(L), "a creature walk must replay from the journal"


def test_r73_b_same_seed_same_walk():
    a, b = _mk(), _mk()
    La, Lb = a.add_layer("c").id, b.add_layer("c").id
    for dd, l in ((a, La), (b, Lb)):
        dd.creature_paint(l, 120, 90, seconds=1.5, creatures=3, seed=11,
                          random_rules=True)
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels), \
        "same seed (including a random rule draw) must reproduce exactly"


def test_r73_c_the_light_rule_steers():
    """A gradient map; a light-seeker must finish brighter than a
    dark-seeker released at the same spot with the same seed."""
    from lestudio import Document

    def run(light):
        d = Document(300, 120)
        base = d.layers[0]
        g = np.linspace(0, 1, 300, dtype=np.float32)[None, :]
        base.pixels[..., :3] = np.repeat(g, 120, 0)[..., None]
        base.pixels[..., 3] = 1.0
        L = d.add_layer("c").id
        d.creature_paint(L, 150, 60, seconds=2.0, creatures=1, seed=5,
                         field="none", sample="composite",
                         rules={"light": light, "self": 0, "wander": 0.05})
        a = d.layer(L).pixels[..., 3]
        ys, xs = np.nonzero(a > 0.2)
        return float(xs.mean())

    assert run(1.0) > run(-1.0) + 30, "light +1 must walk brighter than light -1"


def test_r73_d_self_avoid_spreads_self_seek_tangles():
    def spread(selfw):
        d = _mk(400, 300)
        L = d.add_layer("c").id
        d.creature_paint(L, 200, 150, seconds=2.0, creatures=1, seed=2,
                         field="none", rules={"self": selfw, "wander": 0.1})
        a = d.layer(L).pixels[..., 3]
        ys, xs = np.nonzero(a > 0.2)
        return int(np.ptp(xs) + np.ptp(ys))      # how far the walk ranges
    assert spread(-1.0) > spread(1.0), \
        "avoiding its own trail must range further than retracing it"


def test_r73_e_a_drag_lines_the_creatures_up_along_the_path():
    d = _mk(300, 100)
    L = d.add_layer("c").id
    d.creature_paint(L, 0, 0, seconds=0.3, creatures=5, seed=1,
                     field="none", rules={"wander": 0.0},
                     path=[[20, 50], [280, 50]])
    a = d.layer(L).pixels[..., 3]
    cols = (a > 0.2).any(0)
    assert cols[10:40].any() and cols[130:170].any() and cols[260:295].any(), \
        "five creatures on a horizontal drag must start spaced along it"


def test_r73_f_a_selection_contains_the_creatures():
    d = _mk()
    L = d.add_layer("c").id
    sel = d.select("rect", {"x0": 60, "y0": 40, "x1": 180, "y1": 140})
    r = d.creature_paint(L, 120, 90, seconds=2.0, creatures=6, seed=4,
                         selection=sel.id, rules={"self": -1, "wander": 0.3})
    assert r["strokes"] >= 1
    a = d.layer(L).pixels[..., 3]
    assert float(a[:30, :].max()) == 0.0 and float(a[:, 200:].max()) == 0.0, \
        "creatures must not leave the selection"


def test_r73_g_caps_and_rule_hygiene():
    import pytest
    d = _mk()
    L = d.add_layer("c").id
    r = d.creature_paint(L, 120, 90, seconds=0.3, creatures=99, seed=1)
    assert r["strokes"] <= 10, "ten creatures at most"
    r = d.creature_paint(L, 120, 90, seconds=0.3, seed=1,
                         rules={"lines": 7.0, "target": [2, -1, 0.5]})
    assert r["rules"]["lines"] == 1.0 and r["rules"]["target"] == [1.0, 0.0, 0.5]
    with pytest.raises(ValueError):
        d.creature_paint(L, 120, 90, seconds=0.3, seed=1,
                         rules={"light": float("nan")})
    rr = d.creature_paint(L, 120, 90, seconds=0.3, seed=8, random_rules=True)
    assert set(("lines", "self", "others", "light", "color", "field",
                "wander", "target")) <= set(rr["rules"]), \
        "a random draw is echoed back in full"


def test_r73_j_solid_paint_is_a_wall():
    """A painted ring; creatures released inside with solid=True must
    stay inside, and with solid=False some must get out."""
    from lestudio import Document

    def escaped(solid):
        d = Document(300, 300)
        L = d.add_layer("c").id
        ring = [[150 + 90 * np.cos(t), 150 + 90 * np.sin(t)]
                for t in np.linspace(0, 2 * np.pi, 90)]
        d.paint(L, ring, color=(0, 0, 0), radius=6)
        d.creature_paint(L, 150, 150, seconds=3.0, creatures=6, seed=6,
                         color=(1, 0, 0), field="none",
                         rules={"self": -1, "wander": 0.3, "solid": solid})
        px = d.layer(L).pixels
        red = (px[..., 0] > 0.5) & (px[..., 3] > 0.2)
        ys, xs = np.nonzero(red)
        return int((np.hypot(xs - 150, ys - 150) > 100).sum())

    assert escaped(True) == 0, "solid: nobody crosses the ring"
    assert escaped(False) > 0, "not solid: paint is only a preference"


def test_r73_k_a_glow_pulls_from_a_distance():
    """A bright pool 120px from the release point on a dark ground, no
    field, no wander: a light-seeker must reach it."""
    from lestudio import Document
    d = Document(400, 200)
    d.layers[0].pixels[..., :3] = 0.1
    d.layers[0].pixels[..., 3] = 1.0
    G = d.add_layer("goal").id
    d.paint(G, [[320, 100], [321, 100]], color=(1, 1, 0.8), radius=30,
            hardness=0.3)
    L = d.add_layer("c").id
    d.creature_paint(L, 200, 100, seconds=3.0, creatures=1, seed=9,
                     heading=90, sample="composite", field="none",
                     rules={"light": 1.0, "self": 0, "wander": 0.0})
    a = d.layer(L).pixels[..., 3]
    ys, xs = np.nonzero(a > 0.2)
    assert np.hypot(xs - 320, ys - 100).min() < 20, \
        "the creature never found the light"


def test_r73_h_the_route_round_trips_and_refuses_junk():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 200, "height": 160})
    st = c.get("/api/state").get_json()
    lid = st["layers"][-1]["id"]
    ok = c.post("/api/creature", json={
        "layer": lid, "x": 100, "y": 80, "seconds": 0.5, "creatures": 3,
        "random_rules": True, "path": [[20, 20], [170, 130]],
        "color": [0.8, 0.1, 0.1]}).get_json()
    assert ok.get("ok") and ok["strokes"] == 3 and "rules" in ok, ok
    for junk in ({"rules": 5}, {"seconds": "x"}, {"rules": {"light": "nan"}}):
        r = c.post("/api/creature", json={"layer": lid, "x": 1, "y": 1, **junk})
        assert r.status_code == 400, junk
    assert any(x["path"] == "/api/creature"
               for x in c.get("/api/schema").get_json()["routes"])


def test_r73_i_the_client_reads_the_brush_panel_and_registers_the_tool():
    ui = open(UI).read()
    i = ui.index("async function doCreature(")
    body = ui[i:i + 1600]
    assert "...c" in body and "genCommon()" in body and "'/api/creature'" in body
    # every function the tool calls exists (node --check passes on phantoms)
    for fn in ("creatureRules", "creatureShowRules", "creatureRandomise",
               "creaturePreset", "creaturePresetMark", "creatureReadouts",
               "creatureAgain", "creatureLastLine", "rgb2hex", "hex2rgb"):
        assert ("function %s(" % fn) in ui, fn
    assert "creature:1" in ui[ui.index("const GENTOOLS="):][:120]
    assert "creature:'tCreature'" in ui and "creature:'creatureHud'" in ui
    assert "l:'creature'" in ui, "hotkey L"
    for el in ("crSecs", "crCount", "crThick", "crLines", "crSelf",
               "crOthers", "crLight", "crColor", "crTarget", "crField",
               "crFieldW", "crWander", "crRandom", "crRandomNow", "crSolid",
               "crHead", "crPresets", "crAgain", "crRetry", "crKeepSeed",
               "crLast"):
        assert ('id="%s"' % el) in ui, el
    assert "'creatureHud'" in ui[ui.index("].forEach(id=>{ const el=$(id); if(el){ el.style.display='none'; dock.appendChild(el); } });") - 200:
                                 ui.index("].forEach(id=>{ const el=$(id); if(el){ el.style.display='none'; dock.appendChild(el); } });")], \
        "the HUD must be docked at boot like the other generators"


def test_r73_l_a_polygon_is_a_fill_region_and_walks_carry_media():
    """The rose window: an inline polygon confines AND spreads the release,
    and a walk takes the brush's medium like any stroke -- still replaying."""
    d = _mk(300, 300)
    d.set_paper("rough")
    L = d.add_layer("c").id
    poly = [[150, 30], [270, 150], [150, 270], [30, 150]]
    r = d.creature_paint(L, 150, 150, poly=poly, seconds=2.0, creatures=6,
                         seed=5, media="water", load=0.9, radius=4,
                         color=(0.2, 0.3, 0.8), rules={"self": -1, "wander": 0.2})
    assert r["strokes"] == 6
    a = d.layer(L).pixels[..., 3]
    assert float(a[:20, :].max()) == 0.0 and float(a[:, :20].max()) == 0.0, \
        "a walk left the polygon"
    firsts = [np.asarray(st["points"][0][:2]) for st in d.strokes[-6:]]
    assert max(np.hypot(*(p - firsts[0])) for p in firsts) > 30, \
        "six creatures in a region must be released spread through it"
    assert d.replay_is_faithful(L), "a watercolour walk must replay"


def test_r73_m_a_vortex_spirals_out_instead_of_knotting():
    d = _mk(300, 300)
    L = d.add_layer("c").id
    d.creature_paint(L, 150, 150, seconds=3.0, creatures=1, seed=2, radius=2,
                     field="vortex", rules={"field": 1.0, "self": 0, "wander": 0.0})
    pts = np.asarray([p[:2] for p in d.strokes[-1]["points"]], np.float32)
    rr = np.hypot(pts[:, 0] - 150, pts[:, 1] - 150)
    assert rr[-20:].mean() > rr[:20].mean() + 25, \
        "a vortex must carry the creature outward, not spin it on the spot"
