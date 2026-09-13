"""tests/test_r65.py -- what repainting the R64 still life needed.

Devin's report was two things: "this painting lacks realism", and "there is
an artifact behind the bowl of lemons that looks like a rectangle". The
rectangle turned out to be the tail of an R64 finding: a painter could not
see what was UNDER her own layer, so she sampled the whole composite to
repair a halo and painted a flat wall-coloured patch onto an object layer.
Everything here is either the tool that makes that mistake unnecessary, or
a trap the repaint walked into and should not be able to walk into again.
"""
import time


def _c():
    from lestudio.server import app
    return app.test_client()


def _fresh(c, w=200, h=140):
    from lestudio.server import WS
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": w, "height": h})
    d = WS.doc
    del d.layers[1:]
    d.layers[0].owner = ""
    d.layers[0].shared = []
    d.strokes.clear()
    d._undo.clear()
    return d


def _img(c, path):
    import io as _io
    import numpy as np
    from PIL import Image
    r = c.get(path)
    assert r.status_code == 200, (path, r.status_code)
    return np.asarray(Image.open(_io.BytesIO(r.data)).convert("RGBA")
                      ).astype(float) / 255.0


def test_r65_a_painter_can_see_what_is_under_and_over_a_layer():
    """THE fix for the rectangle. A painter repairing her own layer had only
    /api/composite.png to sample -- which contains her layer AND everything
    stacked above it -- so "match the background here" painted the whole
    stack into one layer as a flat patch. below.png is the surface you are
    actually painting on; above.png is what will be drawn over you."""
    import numpy as np
    c = _c()
    d = _fresh(c)
    bot = d.layers[0].id
    mid = c.post("/api/layer", json={"action": "add", "name": "mid"}).json["id"]
    top = c.post("/api/layer", json={"action": "add", "name": "top"}).json["id"]
    for lid, col in ((bot, [1, 0, 0]), (mid, [0, 1, 0]), (top, [0, 0, 1])):
        assert c.post("/api/paint", json={
            "layer": lid, "points": [[20, 70], [180, 70]], "color": col,
            "radius": 30, "opacity": 1, "hardness": 0.9}).status_code == 200

    below = _img(c, "/api/layer/%s/below.png" % mid)
    above = _img(c, "/api/layer/%s/above.png" % mid)
    px_b, px_a = below[70, 100], above[70, 100]
    assert px_b[0] > 0.5 and px_b[1] < 0.3, ("below mid must be the RED layer", px_b)
    assert px_a[2] > 0.5 and px_a[1] < 0.3, ("above mid must be the BLUE layer", px_a)
    # neither view contains the layer itself
    assert px_b[1] < 0.3 and px_a[1] < 0.3, "a layer must not appear in its own context"

    # the bottom layer has nothing under it, and the top nothing over it
    assert _img(c, "/api/layer/%s/below.png" % bot)[..., 3].max() < 0.02
    assert _img(c, "/api/layer/%s/above.png" % top)[..., 3].max() < 0.02


def test_r65_a_mistake_can_be_taken_out_instead_of_painted_over():
    """Nobody erases when erasing first needs a selection object minted, so
    everybody paints over -- and painting over with a sampled colour is
    exactly how the flat rectangle got onto the bowl layer. A rectangle
    clear, journal-first (it records the same frozen gate asset a selection
    clear does), so the layer stays replay-faithful and the replay needs no
    second code path."""
    import numpy as np
    c = _c()
    d = _fresh(c)
    lid = d.layers[0].id
    c.post("/api/paint", json={"layer": lid, "points": [[10, 70], [190, 70]],
                               "color": [1, 1, 1], "radius": 40, "opacity": 1,
                               "hardness": 0.9})
    before = _img(c, "/api/layer/%s.png" % lid)
    assert before[70, 100, 3] > 0.5

    n0 = len(d.strokes)
    r = c.post("/api/layer", json={"action": "clear", "id": lid,
                                   "region": [60, 40, 140, 100]})
    assert r.status_code == 200, r.json
    after = _img(c, "/api/layer/%s.png" % lid)
    assert after[70, 100, 3] < 0.02, "inside the region must be empty"
    assert after[70, 20, 3] > 0.5, "outside it must be untouched"
    # journal-first: the clear is a RECORD, not a silent pixel poke
    assert len(d.strokes) > n0
    assert d.strokes[-1]["brush"].get("op") == "clear"
    assert d.strokes[-1]["brush"].get("sel_asset"), \
        "the rectangle must be frozen as a gate asset, like a selection clear"

    # a nonsense region is refused rather than silently doing nothing
    assert c.post("/api/layer", json={"action": "clear", "id": lid,
                                      "region": [10, 10]}).status_code == 400


def test_r65_a_named_pass_is_applied_once_and_survives_being_large():
    """Low-opacity build-up passes have no way to tell they already ran, and
    a script re-run by accident doubled every one of them -- that is what
    washed out the window in R64 and cost two fix rounds to find. Passes are
    named and applied once.

    The first version of this broke on its own first real use: a pass
    bigger than one batch arrives as several calls, the first registered
    the name and the server refused the REST OF THE SAME PASS. `pass_run`
    is the client's id for one instance, so continuing chunks go through
    and only a genuinely new run is refused."""
    c = _c()
    d = _fresh(c)
    lid = d.layers[0].id

    def batch(run, force=False, n=2):
        body = {"pass": "underpaint", "pass_run": run,
                "strokes": [{"layer": lid, "points": [[10 + i, 20], [50 + i, 20]],
                             "color": [0.4, 0.4, 0.4], "radius": 6,
                             "opacity": 0.2} for i in range(n)]}
        if force:
            body["force"] = True
        return c.post("/api/paint_batch", json=body)

    assert batch("runA").status_code == 200
    assert batch("runA").status_code == 200, "the same pass instance continues"
    r = batch("runB")
    assert r.status_code == 409 and r.json["pass_already_applied"] is True
    assert batch("runB", force=True).status_code == 200, "force is the way back in"
    # an unnamed batch is never gated
    assert c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[5, 5], [9, 9]], "color": [0, 0, 0],
         "radius": 3, "opacity": 1}]}).status_code == 200


def test_r65_the_crash_net_does_not_live_in_a_browser_tab():
    """R64 left this open and R65 paid for it: the autosave loop was a
    setInterval in the PAGE, so a session driven entirely by scripts and
    agents never autosaved, and a server restart lost a whole painting. The
    server runs the same timer itself now."""
    import lestudio.server as SV
    c = _c()
    d = _fresh(c)
    assert hasattr(SV, "_autosave_tick")
    c.post("/api/paint", json={"layer": d.layers[0].id,
                               "points": [[5, 5], [40, 40]],
                               "color": [1, 1, 1], "radius": 4, "opacity": 1})
    from lestudio import _MUT_REV
    SV._AUTOSAVE_LAST_REV[0] = -1
    with SV.app.test_request_context("/api/autosave", method="POST"):
        r = SV.autosave_write()
    body = r[0] if isinstance(r, tuple) else r
    assert body.get_json()["ok"] is True, body.get_json()
    # and it is wired into the server's own startup, not a client's
    src = open(SV.__file__).read()
    assert "_autosave_tick()" in src.split("def serve(")[1]


def test_r65_an_agents_assist_can_be_taken_back_one_at_a_time():
    """The layer list lets you hide or delete the agent's whole layer. That
    is the wrong grain: a painter wants "that one, no" -- the contact
    shadow it put under the wrong thing -- while keeping the rest. Each
    assist carries the note the agent wrote and the ids of exactly the
    strokes it painted."""
    c = _c()
    d = _fresh(c)
    A = {"X-User": "agent:r65"}
    lid = c.post("/api/layer", json={"action": "add", "name": "agent · x"},
                 headers=A).json["id"]
    r = c.post("/api/paint_batch", json={
        "note": "grounded 2 masses on Bowl",
        "strokes": [{"layer": lid, "points": [[10, 60], [80, 60]],
                     "color": [0, 0, 0], "radius": 8, "opacity": 0.5}]},
        headers=A)
    assert r.status_code == 200
    sids = r.json["sids"]

    feed = c.get("/api/agent/assists").json["assists"]
    assert feed and feed[0]["note"] == "grounded 2 masses on Bowl"
    assert feed[0]["sids"] == sids and feed[0]["by_name"] == "r65"
    assert feed[0]["layer_names"] == ["agent · x"]

    painted = _img(c, "/api/layer/%s.png" % lid)[..., 3].max()
    assert painted > 0.2
    # anyone may take it back: it is the agent's layer, and an agent never
    # locks a human out
    u = c.post("/api/agent/assists", json={"action": "undo", "id": feed[0]["id"]},
               headers={"X-User": "u_r65human"})
    assert u.status_code == 200 and u.json["deleted"] >= 1, u.json
    assert _img(c, "/api/layer/%s.png" % lid)[..., 3].max() < 0.05
    assert c.get("/api/agent/assists").json["assists"][0]["undone"] is True
    # and a second undo is a no-op, not an error
    assert c.post("/api/agent/assists",
                  json={"action": "undo", "id": feed[0]["id"]}).json["already"] is True


def test_r65_not_right_now_is_a_switch_not_a_disconnection():
    """Sometimes you do not want help. The only ways to say so were to kill
    the agent or write a brief telling it to do nothing -- both of which
    lose the brief you actually wrote. A document-level pause: the agent
    keeps watching and reports it, and may_act is false while it holds."""
    c = _c()
    d = _fresh(c)
    lid = d.layers[0].id
    c.post("/api/agent/brief", json={"brief": "Keep the shadows warm."})
    assert c.post("/api/paint", json={"layer": lid, "points": [[5, 5], [40, 40]],
                                      "color": [1, 1, 1], "radius": 4,
                                      "opacity": 1},
                  headers={"X-User": "u_r65p"}).status_code == 200
    time.sleep(0.05)
    A = {"X-User": "agent:r65p"}
    t = c.get("/api/agent/tick?since=0&pause_ms=1", headers=A).json
    assert t["may_act"] is True and t["paused"] is False

    assert c.post("/api/agent/brief", json={"paused": True}).json["paused"] is True
    t = c.get("/api/agent/tick?since=0&pause_ms=1", headers=A).json
    assert t["paused"] is True and t["may_act"] is False
    # the brief itself is untouched -- that is the whole point
    g = c.get("/api/agent/brief").json
    assert g["brief"] == "Keep the shadows warm." and g["paused"] is True

    c.post("/api/agent/brief", json={"paused": False})
    assert c.get("/api/agent/tick?since=0&pause_ms=1", headers=A).json["may_act"] is True


def test_r65_the_painter_kit_enforces_the_laws_it_documents():
    """tools/painter.py is the module every scripted painter and the agent
    import instead of re-writing a brush and re-hitting the same traps.
    Each of these was paid for with a ruined pass in R64 or R65."""
    import importlib.util
    import os
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "painter_kit", os.path.join(os.path.dirname(__file__), "..", "tools",
                                    "painter.py"))
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)

    # the COVERAGE law is enforced, not just described: a step at or above
    # the radius leaves screen gaps and the mass comes out striped
    class _P(kit.Painter):
        def __init__(self):
            self.W, self.H = 200, 140
            self.batch = []
            self.refusals = []
            self.skipped_passes = []
            self._pass = self._pass_run = None
            self._force = False
            import random
            self.rnd = random.Random(1)
            self.lids = {}
            self._paint = {}          # R66: the medium in force
            self.dry = False
            self.counts, self.sent = {}, 0
    p = _P()
    try:
        p.fill("L1", [[10, 10], [90, 10], [90, 90], [10, 90]],
               lambda x, y: [1, 0, 0], step=9.0, radius=9.0)
        assert False, "a step >= radius must be refused"
    except ValueError as e:
        assert "striped" in str(e)

    # a fractional power of a negative base must not go COMPLEX -- it took
    # down a whole window pass, and it is the second time in this project
    assert isinstance(kit.pw(-0.03, 1.35), float)
    assert kit.pw(4.0, 0.5) == 2.0
    assert kit.pw(-4.0, 0.5) == -2.0

    # a dab wider than the feature it paints inflates the silhouette: a
    # 10 px highlight with a radius-4 brush came out 18 px and rectangular
    p.batch = []
    p.flush = lambda: None            # collect, never send
    p.fill("L1", [[50, 50], [60, 50], [60, 60], [50, 60]],
           lambda x, y: [1, 1, 1], radius=9.0)
    assert p.batch and max(s["radius"] for s in p.batch) <= 10 * 0.34 + 0.01

    # and `seg` must sample a small mass more than a couple of times across,
    # or a sphere comes out as flat blocks with a step down the middle
    p.batch = []
    p.flush = lambda: None            # collect, never send
    p.fill("L1", [[0, 0], [80, 0], [80, 40], [0, 40]],
           lambda x, y: [x / 80.0, 0, 0], radius=6.0)
    xs = sorted({round(s["color"][0], 3) for s in p.batch})
    assert len(xs) >= 10, ("a gradient must survive segmentation", len(xs))
