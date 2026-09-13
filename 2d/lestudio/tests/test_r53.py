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


def test_r55_agent_surface_and_engine_fixes():
    """R55: leCore 0.2.21 adoption. The engine's standard agent doors are
    mounted beside ours, /api/status carries engine_status, and the two
    app-side workarounds the engine fixed upstream are retired (Deconvolve
    uses the now-2D sharpen_image; the DCT morph keeps its aspect)."""
    import pytest
    from lestudio.server import app
    from lestudio import mind
    if not hasattr(mind(), "agent_surface"):
        pytest.skip("engine predates agent_surface")
    c = app.test_client()
    man = c.get("/api/agent/tools")
    assert man.status_code == 200
    tools = {t["name"] for t in man.get_json()["tools"]}
    assert "scatter" in tools and "hatchfill" in tools and "paint" in tools
    assert man.get_json()["identity"]["X-User"]
    eng = c.get("/api/engine")
    assert eng.status_code == 200 and eng.get_json().get("engine")
    # R71: /api/status answers with whatever has been measured rather than
    # paying several seconds to finish it -- nothing a person is waiting on
    # may block on the capability report. ?wait=1 asks for the full answer,
    # which is what a test about the engine panel wants.
    st = c.get("/api/status?wait=1").get_json()
    assert (st.get("engine") or {}).get("engine"), \
        "/api/status must carry the engine panel on a 0.2.21+ build"
    assert c.get("/api/status").get_json().get("measuring") in (True, False)
    # the retired morph workaround: non-square frames keep their aspect
    from lestudio import OPS
    a = np.zeros((40, 64, 3), np.float32); a[10:30, 20:44] = 1.0
    b = np.zeros((40, 64, 3), np.float32); b[5:35, 10:54, 2] = 1.0
    out = OPS["Morph"]["fn"]((40, 64), {"a": a, "b": b},
                             {"method": "dct", "t": 0.5})
    assert out.shape == (40, 64, 3)


def test_r56_ids_are_container_scoped_and_survive_processes():
    """R56 (lews_mint law): no process-global counters. A fresh document
    mints the same ids wherever it is created; a saved document reloaded
    in a FRESH interpreter mints the NEXT id, not a colliding one."""
    import subprocess
    import sys
    import tempfile
    import os
    from lestudio import Document, NodeGraph, save_workspace
    d = Document(120, 90)
    ids = [d.add_layer("a").id, d.add_layer("b").id]
    assert ids == ["L2", "L3"], ids                 # L1 is the background
    blob = save_workspace({d.id: d}, {d.id: NodeGraph(d)}, d.id,
                          cache_pixels=False)
    fp = tempfile.mktemp(suffix=".lews")
    open(fp, "wb").write(blob)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lestudio import load_workspace\n"
        "docs, graphs, active, extras = load_workspace(open(%r,'rb').read())\n"
        "d = docs[active]\n"
        "assert [l.id for l in d.layers] == ['L1', 'L2', 'L3'], d.layers\n"
        "assert d.add_layer('c').id == 'L4'\n"
        "print('FRESH-PROCESS-OK')\n"
    ) % (os.path.abspath("src"), fp)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=dict(os.environ))
    assert "FRESH-PROCESS-OK" in r.stdout, r.stderr[-800:]
    # two standalone documents never collide (the old global counter's one
    # virtue, kept without the global)
    assert Document(64, 48).id != Document(64, 48).id


def test_r56_struct_key_is_process_stable():
    """R56: the graph shape fingerprint uses crc32 over stable bytes, so
    it is the same number in every interpreter (hash() is salted)."""
    import subprocess
    import sys
    import os
    from lestudio import Document, NodeGraph
    d = Document(64, 48)
    g = NodeGraph(d)
    g.ensure_default()
    here = g._struct_key()
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lestudio import Document, NodeGraph\n"
        "d = Document(64, 48); g = NodeGraph(d); g.ensure_default()\n"
        "print('KEY=%%d' %% g._struct_key())\n"
    ) % (os.path.abspath("src"),)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=dict(os.environ))
    assert ("KEY=%d" % here) in r.stdout, (here, r.stdout, r.stderr[-400:])


def test_r56_per_user_memory_door():
    """R56 (APP_FOUNDATION §6): /api/memory gives each X-User a
    physically separate app_substrate partition."""
    import pytest
    from lestudio.server import app
    from lestudio import mind
    if not hasattr(mind(), "app_substrate"):
        pytest.skip("engine predates app_substrate")
    c = app.test_client()
    h = {"X-User": "test-ana"}
    r = c.post("/api/memory", headers=h, json={
        "action": "remember", "q": "which brush for skin",
        "a": "soft round at 12 percent flow"})
    assert r.status_code == 200 and r.get_json()["ok"], r.get_json()
    r2 = c.post("/api/memory", headers=h, json={
        "action": "recall", "q": "which brush for skin"})
    got = r2.get_json()["result"]
    assert "soft round" in str(got), got
    # another user cannot see it (physical isolation)
    r3 = c.post("/api/memory", headers={"X-User": "test-bob"}, json={
        "action": "recall", "q": "which brush for skin"})
    assert "soft round" not in str(r3.get_json().get("result")), r3.get_json()
    # no identity, no memory
    assert c.post("/api/memory", json={"action": "habits"}).status_code == 400


def test_r56_presence_mirrors_into_the_lews_workspace():
    """R56 (APP_FOUNDATION §2c): an activity ping lands in the shared
    .lews workspace via lews_touch, so other apps see this painter."""
    import os
    import pytest
    from lestudio.server import app
    from lestudio import mind
    m = mind()
    if not hasattr(m, "lews_touch"):
        pytest.skip("engine predates the lews live session")
    c = app.test_client()
    r = c.post("/api/presence/activity", headers={"X-User": "mirror-test"},
               json={"tool": "brush", "layer": "L2"})
    assert r.status_code == 200
    from lestudio.server import _WS_ROOT as root      # R57: one root for all
    who = {p["who"]: p for p in m.lews_presence(root)}
    assert "mirror-test" in who, sorted(who)
    assert who["mirror-test"]["app"] == "lestudio"


# ---------------------------------------------------------------- R57 ----
# Full lews-Workspace adoption: the shared live directory IS the document
# backing -- boot restores from it, autosave publishes into it through the
# engine's locked/journalled put, doc ids come from the engine's persisted
# mint counter, and the SSE feed forwards other apps' workspace writes.

def _lews_or_skip():
    import pytest
    from lestudio.server import _lews_ws
    ws = _lews_ws()
    if ws is None:
        pytest.skip("engine predates the lews Workspace")
    return ws


def test_r57_autosave_publishes_documents_through_the_engine():
    """Autosave lands every document as a 'lestudio.document' section via
    Workspace.put (rev advances, journal records it) -- and an UNCHANGED
    second autosave writes nothing, so idle timers don't spam every other
    app's change feed with no-op container rewrites."""
    from lestudio.server import app, WS
    ws = _lews_or_skip()
    c = app.test_client()
    r = c.post("/api/paint", json={"layer": WS.doc.layers[0].id,
                                   "points": [[5, 5, 1], [60, 40, 1]],
                                   "color": [0, 0, 1], "radius": 6})
    assert r.get_json().get("ok")
    j1 = c.post("/api/autosave").get_json()
    assert j1["ok"] and j1.get("lews_rev"), j1
    ids = {s["id"] for s in ws.sections("lestudio.document", upgrade=False)}
    assert set(WS.docs) <= ids, (sorted(WS.docs), sorted(ids))
    ops = [e["op"] for e in ws.changes_since(0) if e.get("app") == "lestudio"]
    assert "put" in ops
    j2 = c.post("/api/autosave").get_json()
    assert j2["ok"] and not j2.get("lews_rev"), \
        "an unchanged autosave must not rewrite the shared container: %r" % j2


def test_r57_doc_ids_come_from_the_engine_mint():
    """/api/new ids are issued by the engine's persisted per-prefix counter
    (the lews_mint law): distinct across calls AND across Workspace handles,
    with the counter visible in the directory's ids.json."""
    from lestudio.server import app, WS, _WS_ROOT
    ws = _lews_or_skip()
    c = app.test_client()
    a = c.post("/api/new", json={"width": 64, "height": 48}).get_json()
    b = c.post("/api/new", json={"width": 64, "height": 48}).get_json()
    assert a["ok"] and b["ok"] and a["doc"]["id"] != b["doc"]["id"]
    assert int(ws.id_counters().get("D", 0)) >= 2
    # another app minting on the same directory can never collide with us
    from holographic.io_and_interop.holographic_lews import Workspace
    other = Workspace(_WS_ROOT, app="other")
    assert other.mint("D") not in WS.docs


def test_r57_boot_restores_documents_from_the_live_workspace():
    """A FRESH process pointed at a live directory that holds documents
    rebuilds them at import -- restarting the server must not orphan the
    shared truth. Painted pixels survive the round trip."""
    import os
    import subprocess
    import sys
    import tempfile
    _lews_or_skip()
    root = tempfile.mkdtemp(prefix="lestudio_r57_boot_")
    env = dict(os.environ, LESTUDIO_WS=root)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(repo, "src")] + sys.path)
    seed = (
        "from lestudio.server import app, WS\n"
        "c = app.test_client()\n"
        "r = c.post('/api/paint', json={'layer': WS.doc.layers[0].id,"
        " 'points': [[10, 10, 1], [90, 90, 1]], 'color': [1, 0, 0],"
        " 'radius': 8})\n"
        "assert r.get_json().get('ok')\n"
        "assert c.post('/api/autosave').get_json().get('lews_rev')\n"
        "print(sorted(WS.docs))\n")
    check = (
        "from lestudio.server import WS\n"
        "assert list(WS.docs) == ['D1'], list(WS.docs)\n"
        "comp = WS.doc.composite()\n"
        "assert (comp[..., 0] > 0.9).sum() > 50, 'painted stroke lost'\n"
        "print('restored', sorted(WS.docs))\n")
    for script in (seed, check):
        p = subprocess.run([sys.executable, "-c", script], env=env,
                           capture_output=True, text=True, timeout=240)
        assert p.returncode == 0, p.stderr[-2000:]
    assert "restored ['D1']" in p.stdout


def test_r57_events_forward_foreign_workspace_writes():
    """The SSE feed forwards OTHER apps' journal entries (put/note, with app
    and section id) so the UI can say who changed the shared workspace --
    and never echoes leStudio's own writes back as foreign."""
    import json
    import threading
    import time
    from lestudio.server import app, _WS_ROOT
    _lews_or_skip()
    from holographic.io_and_interop.holographic_lews import Workspace

    def later():
        time.sleep(0.8)
        Workspace(_WS_ROOT, app="polystudio").put(
            {"kind": "poly.scene", "id": "r57-s1", "meta": {}, "arrays": {}})
    threading.Thread(target=later, daemon=True).start()
    c = app.test_client()
    r = c.get("/api/events?client=r57c&user=r57u")
    out, t0 = [], time.time()
    for chunk in r.response:
        out.append(chunk.decode())
        if '"workspace"' in out[-1] or time.time() - t0 > 10:
            break
    lines = [l for l in "".join(out).splitlines()
             if l.startswith("data:") and '"workspace"' in l]
    assert lines, "no foreign workspace event within 10s"
    ent = json.loads(lines[-1][5:])["workspace"]
    assert any(e["app"] == "polystudio" and e["id"] == "r57-s1" for e in ent)
    assert all(e["app"] != "lestudio" for e in ent), \
        "own writes must never come back as foreign (%r)" % ent


def test_r57_opening_a_file_imports_it_into_the_live_dir():
    """/api/workspace/open replaces the LIVE directory through the engine's
    Workspace.from_file: sections carried verbatim and one journal 'import'
    line, so a late-joining app can tell 'opened a file' from 'edited'."""
    import io
    from lestudio.server import app
    ws = _lews_or_skip()
    c = app.test_client()
    data = c.get("/api/workspace.lews").data
    n_before = len([e for e in ws.changes_since(0) if e["op"] == "import"])
    r = c.post("/api/workspace/open",
               data={"file": (io.BytesIO(data), "w.lews")},
               content_type="multipart/form-data")
    assert r.get_json().get("ok")
    from lestudio.server import _lews_ws
    ws2 = _lews_ws()                       # from_file returns a NEW handle
    imports = [e for e in ws2.changes_since(0) if e["op"] == "import"]
    assert len(imports) > n_before, "no journal line recorded the import"
    assert ws2.sections("lestudio.document", upgrade=False), \
        "the opened file's documents must land in the live directory"
