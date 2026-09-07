"""tests/test_r53.py -- the perspective scatter generator + atomic gates.

R53 (user commission): generate vegetation/rocks/water/custom elements
ACROSS an area, not one stroke at a time, with perspective control --
looking down yields fewer, uniform elements; looking across a field
yields many small far ones and few large near ones. Regions arrive as
polygons carried inline in the call (atomic -- the fix for the swarm
selection race seen in R51/R52).
"""
import numpy as np


def _doc(w=800, h=500):
    from lestudio import Document
    d = Document(w, h)
    d.layers[0].pixels[..., :3] = 0.94
    d.layers[0].pixels[..., 3] = 1.0
    return d


POLY = [[40, 120], [760, 120], [760, 480], [40, 480]]


def _run(persp, seed=7, element="grass"):
    d = _doc()
    L = d.add_layer("v").id
    n = d.scatter_fill(L, element=element, poly=POLY, horizon=60,
                       perspective=persp, density=1.0, size=1.0, seed=seed)
    return d, L, n


def test_r53_perspective_controls_count_and_size():
    _, _, n0 = _run(0.0)
    d5, _, n5 = _run(0.5)
    d9, _, n9 = _run(0.9)
    assert n0 < n5 < n9, \
        "looking across a field must show MORE elements than looking " \
        "down (%d / %d / %d)" % (n0, n5, n9)
    # at high perspective: far strokes outnumber near ones and are thinner
    ks = d9.strokes
    far = [k for k in ks if k["points"][0][1] < 240]
    near = [k for k in ks if k["points"][0][1] > 360]
    assert len(far) > 2 * len(near), "far rows must crowd (projection)"
    rf = np.mean([k["brush"]["radius"] for k in far])
    rn = np.mean([k["brush"]["radius"] for k in near])
    assert rn > rf * 1.3, \
        "near elements must be fatter than far ones (%.2f vs %.2f)" % (
            rn, rf)
    # top-down: sizes uniform
    d0, _, _ = _run(0.0)
    ks0 = d0.strokes
    f0 = np.mean([k["brush"]["radius"] for k in ks0
                  if k["points"][0][1] < 240])
    n0_ = np.mean([k["brush"]["radius"] for k in ks0
                   if k["points"][0][1] > 360])
    assert abs(f0 - n0_) < 0.25, "top-down scatter must be uniform"


def test_r53_poly_gate_is_atomic_and_contains():
    d = _doc()
    L = d.add_layer("v").id
    n = d.scatter_fill(L, element="grass", seed=3, perspective=0.7,
                       poly=[[300, 200], [600, 200], [600, 420],
                             [300, 420]])
    assert n > 20
    a = d.layer(L).pixels[..., 3]
    assert float(a[:, :260].max()) == 0.0 and float(a[:, 660:].max()) == 0.0, \
        "scatter must stay inside the inline polygon"
    assert d.replay_is_faithful(L), "scattered elements must replay"
    # poly gates on the other generators too (the race fix is family-wide)
    d2 = _doc()
    L2 = d2.add_layer("v").id
    m = d2.hatch_fill(L2, mode="line", poly=[[100, 100], [400, 100],
                                             [400, 300], [100, 300]],
                      seed=2)
    assert m > 0
    a2 = d2.layer(L2).pixels[..., 3]
    assert float(a2[:, 460:].max()) == 0.0, \
        "hatch poly gate must contain the shading"


def test_r53_every_element_kind_runs_and_differs():
    outs = {}
    for el in ("grass", "flowers", "rocks", "pebbles", "reeds", "ripples"):
        d = _doc(400, 300)
        L = d.add_layer("x").id
        n = d.scatter_fill(L, element=el, seed=3, perspective=0.8,
                           poly=[[20, 80], [380, 80], [380, 280],
                                 [20, 280]])
        assert n > 10, "%s produced too little" % el
        outs[el] = d.layer(L).pixels.copy()
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.array_equal(outs[names[i]], outs[names[j]]), \
                "%s and %s render identically" % (names[i], names[j])


def test_r53_custom_element_spec():
    star = [dict(pts=[[0.5, 1.0], [0.5, 0.4]], radius=0.05),
            dict(pts=[[0.3, 0.55], [0.7, 0.55]], radius=0.04,
                 color=[0.8, 0.2, 0.2], opacity=0.9)]
    a = _doc(400, 300)
    b = _doc(400, 300)
    La, Lb = a.add_layer("x").id, b.add_layer("x").id
    na = a.scatter_fill(La, custom=star, element="custom", seed=11,
                        poly=[[30, 60], [370, 60], [370, 280], [30, 280]])
    nb = b.scatter_fill(Lb, custom=star, element="custom", seed=11,
                        poly=[[30, 60], [370, 60], [370, 280], [30, 280]])
    assert na == nb > 10
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels), \
        "custom scatter must be seed-deterministic"
    # the red crossbar colour must actually appear
    px = a.layer(La).pixels
    reds = (px[..., 0] > 0.5) & (px[..., 1] < 0.4) & (px[..., 3] > 0.3)
    assert int(reds.sum()) > 20, "custom per-stroke colour ignored"


def test_r53_scatter_is_one_undo_and_fill_source():
    d = _doc()
    d.layers[0].pixels[150:400, 100:700, :3] = 0.7    # a blob to bucket
    L = d.add_layer("v").id
    u0 = len(d._undo)
    n = d.fill_generated(L, 400, 250, style="scatter", tolerance=0.05,
                         sample="composite", seed=4, element="grass",
                         perspective=0.7)
    assert n > 20
    assert len(d._undo) == u0 + 1, "a scatter fill is ONE undo entry"
    a = d.layer(L).pixels[..., 3]
    assert float(a[:80, :].max()) == 0.0, \
        "scatter fill must stay in the bucket region (tops of far blades " \
        "may rise slightly, but not 70px above it)"
    assert d.undo()
    assert float(d.layer(L).pixels[..., 3].max()) == 0.0


def test_r54_sage_answers_paraphrases():
    # R54 audit: 723 taught pairs, and every REWORDED question came back
    # empty -- the ladder only served near-verbatim hits. /api/advise now
    # falls back to the mind's rare-token-weighted taught-log search.
    import pytest
    from lestudio.server import app, _sage
    if _sage() is None:
        pytest.skip("leCore mind unavailable")
    c = app.test_client()
    r = c.post("/api/advise", json={"teach": {
        "q": "How should a zebra unicycle be painted for the festival?",
        "a": "Stripe the wheel first, then the frame, in alternating "
             "matte charcoal and warm ivory."}})
    assert r.status_code == 200 and r.json.get("taught")
    r2 = c.post("/api/advise", json={
        "q": "what colors go on the unicycle zebra for festival painting"})
    assert r2.status_code == 200
    assert "charcoal" in (r2.json.get("answer") or ""), \
        "a paraphrase of a taught lesson must recall it (got %r)" % (
            r2.json,)
    assert r2.json.get("matched"), "the served hit must name its source"
    # hygiene: scrub the test rows (and any empty-answer ask-echoes) from
    # the DURABLE store -- this test runs against the real partition
    m = _sage()
    lad = m.zoo["ladder"]
    lad.taught_log[:] = [t for t in lad.taught_log
                         if "zebra unicycle" not in str(t[0]).lower()
                         and str(t[1]).strip()]
    try:
        m.learning_save(__import__("lestudio.server", fromlist=["_LECORE"])
                        ._LECORE.get("mind_part") or "lecore_memory")
    except Exception:
        pass
