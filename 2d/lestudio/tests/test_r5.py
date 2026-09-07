"""R5 sweep pins -- Wave 1 (model/server correctness).

Each test pins one numbered finding from SWEEP_R5_BACKLOG.md. Kept apart
from test_studio.py for the same reason test_r4.py is: the sweep's fixtures
stay next to its backlog doc.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from lestudio import Document, NodeGraph  # noqa: E402


# ---------------------------------------------------------------------------
# A. layer model
# ---------------------------------------------------------------------------

def test_r5_edit_layer_validates_every_numeric_key():
    """#1: a string opacity used to be raw-setattr'd, return ok, then 500
    every composite AND save the poison into the .lews. Every numeric key is
    now coerced+clipped; garbage raises ValueError with the key named."""
    d = Document(64, 48)
    lid = d.layers[0].id
    for key, bad in (("opacity", "cheese"), ("vol_ior", "glassy"),
                     ("vol_density", []), ("absorbency", "damp"),
                     ("emissive", "bright"), ("reflect", "mirror"),
                     ("dispersion", "rainbow"), ("media_rate", "fast"),
                     ("z_off", "up"), ("relief", "high"),
                     ("paint_gloss", "shiny"), ("opacity", float("nan")),
                     ("emissive", float("inf"))):
        before = getattr(d.layers[0], key, None)
        try:
            d.edit_layer(lid, **{key: bad})
            assert False, "%s=%r was accepted" % (key, bad)
        except ValueError as e:
            assert key in str(e), "the error must name the key: %s" % e
        assert getattr(d.layers[0], key, None) == before, \
            "a refused edit must not half-apply"
    # emissive_color: three floats or refusal
    for bad in ("gold", 3, [1, "x", 0], [0.5]):
        try:
            d.edit_layer(lid, emissive_color=bad)
            assert False, "emissive_color=%r was accepted" % (bad,)
        except ValueError as e:
            assert "emissive_color" in str(e)
    d.edit_layer(lid, emissive_color=(2, 0.5, "0.25"))
    assert d.layers[0].emissive_color == [2.0, 0.5, 0.25]
    # good values coerce and clip
    d.edit_layer(lid, opacity="0.5")
    assert abs(d.layers[0].opacity - 0.5) < 1e-9
    d.edit_layer(lid, opacity=7.0)
    assert d.layers[0].opacity == 1.0, "opacity clips to [0, 1]"
    d.edit_layer(lid, relief=-3)
    assert d.layers[0].relief == 0.0


def test_r5_edit_layer_vol_kind_named_options():
    """#2: vol_kind rejects unknown kinds and names the valid ones (the
    paint(material=...) courtesy)."""
    d = Document(64, 48)
    lid = d.layers[0].id
    try:
        d.edit_layer(lid, vol_kind="lava")
        assert False, "unknown vol_kind accepted"
    except ValueError as e:
        for k in ("water", "glass", "inkwater", "smoke", "fire", "absorb"):
            assert k in str(e), "the refusal must list the options: %s" % e
    d.edit_layer(lid, vol_kind="inkwater")
    assert d.layers[0].vol_kind == "inkwater"


def test_r5_edit_layer_and_field_undo_coalesced():
    """#3: edit_layer is undoable, and a slider drag (many consecutive edits
    of one layer) is ONE history slot. Same for field/light gizmo drags,
    which used to evict the whole 24-slot history."""
    d = Document(64, 48)
    lid = d.layers[0].id
    n0 = len(d._undo)
    for i in range(10):
        d.edit_layer(lid, opacity=i / 10.0)
    assert len(d._undo) - n0 == 1, "a drag is one undo step"
    d.undo()
    assert d.layers[0].opacity == 1.0, "undo restores the pre-drag value"
    # an undo ends the run: the next edit snapshots again
    d.edit_layer(lid, opacity=0.25)
    assert len(d._undo) - n0 == 1
    # a DIFFERENT layer breaks the run
    l2 = d.add_layer("two").id
    a = len(d._undo)
    d.edit_layer(l2, opacity=0.5)
    assert len(d._undo) - a == 1
    # unrelated recorded edits break it too
    d.paint(lid, [(5, 5), (20, 20)], color=(1, 0, 0), radius=3)
    b = len(d._undo)
    d.edit_layer(lid, opacity=0.9)
    assert len(d._undo) - b == 1

    # fields: a drag streams x/y edits -- one slot
    f = d.add_field(kind="vortex", layer=lid, x=10, y=10, strength=1.0)
    c0 = len(d._undo)
    for i in range(8):
        d.edit_field(f["id"], x=10 + i, y=10 + i)
    assert len(d._undo) - c0 == 1, "a field gizmo drag is one undo step"
    d.undo()
    assert d.fields[0]["x"] == 10.0

    # lights share record(): same coalescing
    li = d.add_light(kind="point", x=5, y=5)
    e0 = len(d._undo)
    for i in range(8):
        d.edit_light(li["id"], x=5 + i)
    assert len(d._undo) - e0 == 1, "a light gizmo drag is one undo step"


def test_r5_duplicate_layer_copies_physical_properties():
    """#8: the docstring promises 'every property'; the copy used to drop the
    whole physical layer (thickness, volume, optics, impasto height, PBR
    paint). Arrays must be COPIES, not shared buffers."""
    d = Document(64, 48)
    lid = d.add_layer("slab").id
    d.edit_layer(lid, thickness=8.0, vol_kind="glass", vol_ior=1.5,
                 vol_density=1.2, absorbency=0.4, emissive=2.0,
                 emissive_color=[1, 0.5, 0.25], reflect=0.6, dispersion=0.3,
                 media_rate=2.0, z_off=3.0, tilt_x=10.0, tilt_y=-5.0,
                 curve=0.5, dome=-0.25, gravity=0.7, gravity_angle=45.0,
                 optical=True, media_res="fine", media_time="always",
                 relief=0.8, locked=False, alpha_lock=True, clip=True,
                 paint_gloss=0.9)
    src = d.layer(lid)
    src.height_map = np.random.default_rng(0).random((48, 64)).astype("f4")
    src.material_map = np.zeros((48, 64, 3), np.float32) + 0.5
    cp = d.duplicate_layer(lid)
    for k in ("thickness", "vol_kind", "vol_ior", "vol_density",
              "absorbency", "emissive", "reflect", "dispersion",
              "media_rate", "z_off", "tilt_x", "tilt_y", "curve", "dome",
              "gravity", "gravity_angle", "optical", "media_res",
              "media_time", "relief", "locked", "alpha_lock", "clip",
              "paint_gloss"):
        assert getattr(cp, k) == getattr(src, k), \
            "duplicate dropped %s (%r vs %r)" % (k, getattr(cp, k, None),
                                                 getattr(src, k, None))
    assert cp.emissive_color == src.emissive_color \
        and cp.emissive_color is not src.emissive_color
    assert np.array_equal(cp.height_map, src.height_map) \
        and cp.height_map is not src.height_map
    cp.height_map[0, 0] = 99.0
    assert src.height_map[0, 0] != 99.0, "height_map must be a deep copy"
    assert np.array_equal(cp.material_map, src.material_map) \
        and cp.material_map is not src.material_map


def test_r5_layer_defaults_live_in_init():
    """#5 (P2): the scattered getattr defaults are canonical in
    Layer.__init__, so a fresh layer answers without getattr fallbacks."""
    d = Document(32, 24)
    l = d.layers[0]
    assert l.locked is False and l.relief == 1.0 and l.gravity is None
    assert l.gravity_angle is None and l.optical is False
    assert l.media_res == "normal" and l.media_time == "timeline"
    assert l.curve_axis == "x" and l.place is None and l.source is None
    assert l.paint_media is None and abs(l.paint_gloss - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# B. HTTP guards
# ---------------------------------------------------------------------------

def test_r5_layer_and_mask_routes_refuse_unknown_ids():
    """#10: unknown-id / missing-index paths answer 400 with the id named,
    instead of a 500 or a silent no-op that still burned an undo snapshot."""
    from lestudio.server import app, WS
    c = app.test_client()
    lid = WS.doc.layers[0].id
    for act in ("duplicate", "merge_down", "clear", "delete", "flip",
                "move", "edit"):
        r = c.post("/api/layer", json={"action": act, "id": "L999"})
        assert r.status_code == 400 and "L999" in r.json["error"], \
            "%s of unknown id: %s %s" % (act, r.status_code, r.json)
        r = c.post("/api/layer", json={"action": act})
        assert r.status_code == 400, "%s with no id" % act
    # delete of an unknown id must NOT have burned an undo snapshot
    n0 = len(WS.doc._undo)
    c.post("/api/layer", json={"action": "delete", "id": "L999"})
    assert len(WS.doc._undo) == n0, "refused delete recorded an undo entry"
    r = c.post("/api/layer", json={"action": "move", "id": lid})
    assert r.status_code == 400 and "index" in r.json["error"]
    # masks
    for act in ("duplicate", "remove", "edit", "move"):
        r = c.post("/api/mask", json={"action": act, "id": "M999"})
        assert r.status_code == 400 and "M999" in r.json["error"], \
            "mask %s: %s %s" % (act, r.status_code, r.json)
    m = c.post("/api/mask", json={"action": "add", "name": "m"}).json["mask"]
    r = c.post("/api/mask", json={"action": "move", "id": m["id"]})
    assert r.status_code == 400 and "index" in r.json["error"]
    # groups: add with ONLY unknown ids is a stale-client mistake
    r = c.post("/api/group", json={"action": "add", "layers": ["L998",
                                                               "L999"]})
    assert r.status_code == 400
    r = c.post("/api/group", json={"action": "edit", "id": "G999"})
    assert r.status_code == 400
    # a valid mixed add still works (stale ids are dropped by the engine)
    r = c.post("/api/group", json={"action": "add", "layers": [lid]})
    assert r.status_code == 200 and r.json["ok"]


def test_r5_layer_edit_route_400s_on_garbage():
    """#1 (route half): thickness:'thick' used to 500; the model's helpful
    ValueError now surfaces as a 400."""
    from lestudio.server import app, WS
    c = app.test_client()
    lid = WS.doc.layers[0].id
    for payload in ({"thickness": "thick"}, {"opacity": "solid"},
                    {"emissive_color": "gold"}, {"vol_kind": "lava"},
                    {"vol_ior": [1, 2]}):
        r = c.post("/api/layer", json=dict(payload, action="edit", id=lid))
        assert r.status_code == 400, (payload, r.status_code, r.json)
        assert r.json["error"], payload
    # and the layer survived unpoisoned: composite still works
    assert c.get("/api/state").status_code == 200
    r = c.post("/api/layer", json={"action": "edit", "id": lid,
                                   "opacity": 0.5, "thickness": 2})
    assert r.status_code == 200 and WS.doc.layers[0].opacity == 0.5


def test_r5_doc_route_guards():
    """#32 + consistency: settings with a stale id must NOT resize the
    active doc (404 instead); activate/rename/close of unknown ids answer
    plainly; closing the last doc reports instead of pretending."""
    from lestudio.server import app, WS
    c = app.test_client()
    w0, h0 = WS.doc.width, WS.doc.height
    r = c.post("/api/doc", json={"action": "settings", "id": "DSTALE",
                                 "width": 100, "height": 100})
    assert r.status_code == 404 and "DSTALE" in r.json["error"]
    assert (WS.doc.width, WS.doc.height) == (w0, h0), \
        "a stale settings id resized the ACTIVE doc"
    # no id -> the caller's active doc (unchanged behaviour)
    r = c.post("/api/doc", json={"action": "settings", "width": 128,
                                 "height": 96})
    assert r.status_code == 200 and WS.doc.width == 128
    for act in ("activate", "rename", "close"):
        r = c.post("/api/doc", json={"action": act, "id": "DSTALE"})
        assert r.status_code == 404, (act, r.status_code)
        r = c.post("/api/doc", json={"action": act})
        assert r.status_code == 400, "%s with no id" % act
    # closing the last document: WS.close returns False -- surfaced now
    r = c.post("/api/doc", json={"action": "close", "id": WS.active,
                                 "force": True})
    assert r.status_code == 200 and r.json["ok"] is False \
        and "last document" in r.json["error"]
    assert len(WS.docs) == 1


def test_r5_dirty_clears_on_save():
    """#51: dirty used to be bool(_undo) -- true forever after the first
    edit. It is now 'the undo history moved since the last save'."""
    from lestudio.server import app, WS
    c = app.test_client()
    did = WS.active

    def dirty():
        docs = c.get("/api/state").json["docs"]
        return next(x["dirty"] for x in docs if x["id"] == did)

    assert dirty() is False
    lid = WS.doc.layers[0].id
    assert c.post("/api/layer", json={"action": "edit", "id": lid,
                                      "opacity": 0.7}).status_code == 200
    assert dirty() is True
    r = c.post("/api/autosave")
    assert r.status_code == 200 and r.json.get("ok"), r.json
    assert dirty() is False, "a save must clear the dirty flag"
    # and close no longer demands confirmation for saved work
    c.post("/api/new", json={"name": "other", "width": 64, "height": 48})
    r = c.post("/api/doc", json={"action": "close", "id": did})
    assert r.status_code == 200 and r.json["ok"], r.json
    try:
        os.unlink(os.path.expanduser("~/.lestudio_autosave.lews"))
    except OSError:
        pass


def test_r5_close_prunes_viewing_entries():
    """#53: SYNC['viewing'] entries pointing at a closed doc are pruned."""
    from lestudio.server import app, WS, SYNC
    c = app.test_client()
    c.post("/api/new", json={"name": "b", "width": 64, "height": 48},
           headers={"X-User": "u_bob"})
    did = WS.active
    assert SYNC["viewing"].get("u_bob") == did
    assert c.post("/api/doc", json={"action": "close", "id": did,
                                    "force": True},
                  headers={"X-User": "u_bob"}).status_code == 200
    assert "u_bob" not in SYNC["viewing"], \
        "closing a doc must prune viewing entries pointing at it"


# ---------------------------------------------------------------------------
# C. multi-user core
# ---------------------------------------------------------------------------

def test_r5_graph_revision_conflict_409_with_rebase_payload():
    """#6: whole-graph POST was last-write-wins -- a concurrent editor's
    added node silently vanished. With base_rev, a stale POST 409s and
    hands back {grev, nodes} to rebase onto. force and no-base_rev bypass."""
    from lestudio.server import app
    c = app.test_client()
    st = c.get("/api/state").json
    grev0, nodes = st["grev"], st["graph"]
    # editor A posts a node at the rev it loaded
    a_nodes = nodes + [{"id": "NA", "type": "Value",
                        "params": {"value": 0.5}, "inputs": {},
                        "x": 0, "y": 0}]
    r = c.post("/api/graph", json={"nodes": a_nodes, "base_rev": grev0})
    assert r.status_code == 200 and r.json["ok"]
    grev1 = r.json["grev"]
    assert grev1 != grev0
    # editor B still holds grev0: their post must NOT erase NA
    b_nodes = nodes + [{"id": "NB", "type": "Value",
                        "params": {"value": 0.9}, "inputs": {},
                        "x": 0, "y": 40}]
    r = c.post("/api/graph", json={"nodes": b_nodes, "base_rev": grev0})
    assert r.status_code == 409, (r.status_code, r.json)
    assert r.json["grev"] == grev1
    assert any(n["id"] == "NA" for n in r.json["nodes"]), \
        "the 409 must carry the CURRENT nodes so B can rebase"
    # B rebases (their one local op onto the live nodes) and re-posts
    r = c.post("/api/graph", json={"nodes": r.json["nodes"] + [b_nodes[-1]],
                                   "base_rev": grev1})
    assert r.status_code == 200
    ids = {n["id"] for n in c.get("/api/state").json["graph"]}
    assert {"NA", "NB"} <= ids, "both editors' nodes survive"
    # force bypasses; so does omitting base_rev (older clients/agents)
    assert c.post("/api/graph", json={"nodes": a_nodes, "base_rev": 0,
                                      "force": True}).status_code == 200
    assert c.post("/api/graph", json={"nodes": a_nodes}).status_code == 200
    # a node patch moves grev too
    g0 = c.get("/api/state").json["grev"]
    r = c.patch("/api/graph/node/NA", json={"params": {"value": 0.1}})
    assert r.status_code == 200 and r.json["grev"] != g0


def test_r5_per_client_active_doc_isolation():
    """#7: the active document was ONE GLOBAL -- one client switching docs
    redirected everyone's edits. Each user's activate now moves only their
    own view; edits land on THEIR doc."""
    from lestudio.server import app, WS
    a = app.test_client()
    b = app.test_client()
    A = {"X-User": "u_alice"}
    B = {"X-User": "u_bob"}
    d1 = WS.active
    r = b.post("/api/new", json={"name": "bobs", "width": 64, "height": 48},
               headers=B)
    d2 = r.json["doc"]["id"]
    assert a.post("/api/doc", json={"action": "activate", "id": d1},
                  headers=A).status_code == 200
    # each caller sees THEIR OWN active doc
    assert a.get("/api/state", headers=A).json["active_doc"] == d1
    assert b.get("/api/state", headers=B).json["active_doc"] == d2
    # and each caller's edits land on THEIR doc
    la = WS.docs[d1].layers[0].id
    lb = WS.docs[d2].layers[0].id
    assert a.post("/api/layer", json={"action": "edit", "id": la,
                                      "opacity": 0.25},
                  headers=A).status_code == 200
    assert b.post("/api/layer", json={"action": "edit", "id": lb,
                                      "opacity": 0.75},
                  headers=B).status_code == 200
    assert WS.docs[d1].layers[0].opacity == 0.25
    assert WS.docs[d2].layers[0].opacity == 0.75
    # a client with NO viewing entry follows the global default (old
    # single-user behaviour): the last activate/new set it to d2... make it
    # explicit via a fresh activate
    assert b.post("/api/doc", json={"action": "activate", "id": d2},
                  headers=B).status_code == 200
    anon = app.test_client()
    assert anon.get("/api/state").json["active_doc"] == d2
    b.post("/api/doc", json={"action": "close", "id": d2, "force": True},
           headers=B)


def test_r5_undo_attribution_409_and_force():
    """#29: A's Ctrl+Z used to silently revert B's stroke. The last change's
    author now rides on the undo entry; a different user gets a 409 naming
    them, force proceeds. Single-user (same or no id) is unchanged."""
    from lestudio.server import app, WS
    a = app.test_client()
    b = app.test_client()
    A = {"X-User": "u_alice", "X-Client": "tab_a"}
    B = {"X-User": "u_bob", "X-Client": "tab_b"}
    a.post("/api/editors/name", json={"name": "Alice"}, headers=A)
    lid = WS.doc.layers[0].id
    # Alice edits; Alice can undo her own change freely
    assert a.post("/api/layer", json={"action": "edit", "id": lid,
                                      "opacity": 0.3},
                  headers=A).status_code == 200
    assert a.post("/api/undo", json={}, headers=A).json["ok"]
    # Alice edits again; Bob's undo is challenged
    a.post("/api/layer", json={"action": "edit", "id": lid, "opacity": 0.4},
           headers=A)
    r = b.post("/api/undo", json={}, headers=B)
    assert r.status_code == 409, (r.status_code, r.json)
    assert "Alice" in r.json["error"] and "force" in r.json["error"]
    assert WS.doc.layers[0].opacity == 0.4, "the 409 must not have undone"
    # force proceeds
    r = b.post("/api/undo", json={"force": True}, headers=B)
    assert r.status_code == 200 and r.json["ok"]
    # redo is symmetric: the entry to redo is Alice's
    r = b.post("/api/redo", json={}, headers=B)
    assert r.status_code == 409
    assert b.post("/api/redo", json={"force": True},
                  headers=B).json["ok"]
    # entries recorded OUTSIDE a request (engine use) are unowned: no 409
    WS.doc._last_author = ""
    WS.doc.record("Engine edit", only=[])
    assert b.post("/api/undo", json={}, headers=B).status_code == 200


def test_r5_presence_keys_unified_on_user_id():
    """#30: names/viewing were keyed by tab id in some routes and user id in
    others -- people saw their own ghost chip and peers' docs read wrongly.
    Everything now keys by user id; /api/presence/name aliases
    /api/editors/name; `me` compares the USER id."""
    from lestudio.server import app, SYNC
    c = app.test_client()
    H = {"X-User": "u_carol", "X-Client": "tab_1"}
    assert c.post("/api/presence/name", json={"name": "Carol"},
                  headers=H).status_code == 200
    assert SYNC["names"].get("u_carol") == "Carol", \
        "presence/name must write the USER-keyed map"
    assert "tab_1" not in SYNC["names"], "no tab-id ghost entry"
    # a second tab of the same person: still ONE peer row, marked me
    H2 = {"X-User": "u_carol", "X-Client": "tab_2"}
    peers = c.get("/api/state", headers=H2).json["peers"]
    rows = [p for p in peers if p["name"] == "Carol"]
    assert len(rows) == 1 and rows[0]["me"] is True \
        and rows[0]["id"] == "u_carol", peers
    # editors/name writes the same map
    c.post("/api/editors/name", json={"name": "Caz"}, headers=H2)
    assert SYNC["names"]["u_carol"] == "Caz"
    # viewing keys by user id after an activate
    from lestudio.server import WS
    c.post("/api/doc", json={"action": "activate", "id": WS.active},
           headers=H)
    assert SYNC["viewing"].get("u_carol") == WS.active
    peers = c.get("/api/state", headers=H2).json["peers"]
    assert next(p for p in peers if p["id"] == "u_carol")["doc"] == WS.active


def test_r5_presence_activity():
    """#31: a lightweight activity ping (tool/layer) stored per user, echoed
    in /api/editors and /api/state peers, and excluded from the rev bump so
    pings do not make every client refresh."""
    from lestudio.server import app, SYNC
    c = app.test_client()
    H = {"X-User": "u_dave", "X-Client": "tab_d"}
    rev0 = SYNC["rev"]
    r = c.post("/api/presence/activity", json={"tool": "brush",
                                               "layer": "L1"}, headers=H)
    assert r.status_code == 200 and r.json["ok"]
    assert SYNC["rev"] == rev0, \
        "activity pings must not bump the sync rev (they would make every " \
        "client run its foreign-edit refresh)"
    act = SYNC["activity"]["u_dave"]
    assert act["tool"] == "brush" and act["layer"] == "L1"
    c.post("/api/presence/name", json={"name": "Dave"}, headers=H)
    peers = c.get("/api/state", headers=H).json["peers"]
    me = next(p for p in peers if p["id"] == "u_dave")
    assert me["activity"]["tool"] == "brush"
    # no id -> a plain 400, not a crash
    assert app.test_client().post("/api/presence/activity",
                                  json={"tool": "x"}).status_code == 400


# ---------------------------------------------------------------------------
# D. physics
# ---------------------------------------------------------------------------

def test_r5_vortex_conserves_mass_and_rotates():
    """#3: at the shared x105 force scale, a default-strength vortex DESTROYED
    the medium: 0.0% of the density left after 25 steps (control kept 93%).
    Vortex now has its own x30 scale plus a CFL-style clamp on the curl
    force (|f|*dt under ~0.25 cell). Pinned: mass conserved AND the blob
    genuinely rotates about the field centre."""
    def run(strength):
        d = Document(320, 240)
        lid = d.add_layer("x").id
        d.layer(lid).pixels[...] = 0.0
        d.edit_layer(lid, thickness=10.0, vol_kind="inkwater")
        # OFF-CENTRE blob: rotation of a centred symmetric blob is invisible
        d.paint(lid, [(200.0, 120.0)], color=(0.7, 0.1, 0.1), radius=10)
        if strength:
            d.add_field(kind="vortex", layer=lid, x=160, y=120, radius=200,
                        strength=strength)
        d.set_frame(25.0)
        den = d.layer(lid)._media["den"]
        gh, gw = den.shape
        yy, xx = np.mgrid[0:gh, 0:gw]
        m = max(float(den.sum()), 1e-9)
        cx, cy = float((xx * den).sum() / m), float((yy * den).sum() / m)
        # field centre on the media grid: (160, 120) px -> (gw/2, gh/2)
        ang = float(np.degrees(np.arctan2(cy - gh / 2.0, cx - gw / 2.0)))
        rad = float(np.hypot(cx - gw / 2.0, cy - gh / 2.0))
        return den, ang, rad

    ctl, ang0, rad0 = run(0)
    den, ang1, rad1 = run(1.0)
    kept = float(den.sum()) / max(float(ctl.sum()), 1e-9)
    assert kept >= 0.80, \
        "vortex destroyed the medium: %.1f%% of control kept" % (100 * kept)
    # measured at the pinning run: 101% kept, 25.5 degrees of rotation
    turn = abs(ang1 - ang0)
    assert turn > 8.0, "the blob must measurably rotate (%.1f deg)" % turn
    # it ORBITS rather than being flung off or sucked in
    assert 0.4 * rad0 <= rad1 <= 1.8 * rad0, (rad0, rad1)
    # and the density fields genuinely differ from the control
    assert float(np.abs(den - ctl).mean()) > 1e-4
    # negative strength spins the other way
    _, ang2, _ = run(-1.0)
    assert (ang1 - ang0) * (ang2 - ang0) < 0, \
        "opposite strengths must rotate opposite ways (%.1f vs %.1f)" \
        % (ang1, ang2)


# ---------------------------------------------------------------------------
# E. node fixes
# ---------------------------------------------------------------------------

def test_r5_distance_field_unwired_matte():
    """#4: 'Distance field' with nothing wired crashed in _rgb(None); an
    unwired matte now yields transparent zeros."""
    d = Document(96, 64)
    g = NodeGraph(d)
    g.set_graph([{"id": "DF1", "type": "Distance field", "params": {},
                  "inputs": {}, "x": 0, "y": 0}])
    out = np.asarray(g.evaluate("DF1"))
    # the graph pipeline may broadcast to RGBA; the op's own contribution
    # is zero everywhere (the old behaviour was a hard crash in _rgb)
    assert out.shape[:2] == (64, 96) \
        and float(np.abs(out[..., :3]).max()) == 0.0


def test_r5_strokefx_param_kinds_and_rise_rename():
    """#26 + #27: the Stroke FX `field` param declared kind 'str' (rendered
    as a garbage slider) -> 'text'; the TUBES `rise` duplicated the
    particles `rise` (two dials over one value) -> renamed `rise_t`, with
    the fn falling back to `rise` so old workspaces keep growing the same."""
    from lestudio import op_catalog
    cat = op_catalog()
    fx = cat["Stroke FX"]
    names = [p["name"] for p in fx["params"]]
    assert len(names) == len(set(names)), \
        "duplicate Stroke FX param names: %s" % sorted(
            n for n in set(names) if names.count(n) > 1)
    assert "rise" in names and "rise_t" in names
    fld = next(p for p in fx["params"] if p["name"] == "field")
    assert fld["kind"] == "text", fld
    assert not any(p["kind"] == "str" for p in fx["params"])
    # nothing anywhere declares the bogus 'str' kind
    for oname, o in cat.items():
        for p in o["params"]:
            assert p["kind"] != "str", (oname, p["name"])
    # legacy fallback: a saved workspace's tubes "rise" still projects wide
    from lestudio import _strokefx_fov
    assert _strokefx_fov({"rise": 1.0}) == _strokefx_fov({"rise_t": 1.0})
    assert _strokefx_fov({"rise_t": 1.0}) > _strokefx_fov({})


def test_r5_gone_layer_degrades_and_close_warns_referrers():
    """#35: a Layer node whose layer (possibly in another, closed doc) is
    gone used to KeyError raw; it now signs ':gone' like Mask and evaluates
    to transparent zeros. And closing a doc other graphs read 409s with the
    referrers named, unless forced."""
    d = Document(80, 60)
    g = NodeGraph(d)
    g.set_graph([{"id": "LN", "type": "Layer",
                  "params": {"layer": "L9999"}, "inputs": {}, "x": 0,
                  "y": 0}])
    sig = g._sig_inner("LN")            # must not raise
    assert isinstance(sig, str) and sig
    out = np.asarray(g.evaluate("LN"))
    assert out.shape[:2] == (60, 80) and float(np.abs(out).max()) == 0.0

    # the server half: close warns when other docs' graphs read this one
    from lestudio.server import app, WS
    c = app.test_client()
    d1 = WS.active
    d2 = c.post("/api/new", json={"name": "source", "width": 64,
                                  "height": 48}).json["doc"]["id"]
    lid2 = WS.docs[d2].layers[0].id
    # doc1's graph reads doc2 through a cross-doc Layer node
    g1 = WS.graphs[d1]
    g1.set_graph(list(g1.ensure_default().values())
                 + [{"id": "XD", "type": "Layer",
                     "params": {"layer": lid2, "doc": d2}, "inputs": {},
                     "x": 0, "y": 0}])
    r = c.post("/api/doc", json={"action": "close", "id": d2})
    assert r.status_code == 409 and r.json.get("needs_confirm"), r.json
    assert WS.docs[d1].name in r.json["referrers"], r.json
    assert d2 in WS.docs, "the 409 must not have closed it"
    r = c.post("/api/doc", json={"action": "close", "id": d2,
                                 "force": True})
    assert r.status_code == 200 and d2 not in WS.docs
    # the referring graph now degrades instead of crashing
    sig = g1._sig_inner("XD")
    assert sig and float(np.abs(np.asarray(g1.evaluate("XD"))).max()) == 0.0


def test_r5_media_sources_keyed_per_doc():
    """#33: media sources were keyed by node id only, so doc A's N1 and doc
    B's N1 fought over one video slot (every graph counts ids from N1).
    Keyed by (doc, node) now; a doc's sources are purged when it closes."""
    from lestudio.server import app, WS, MEDIA
    c = app.test_client()
    d1 = WS.active
    d2 = c.post("/api/new", json={"name": "two", "width": 64,
                                  "height": 48}).json["doc"]["id"]
    g1, g2 = WS.graphs[d1], WS.graphs[d2]
    # the SAME node id in both graphs, different sources
    s1 = g1.media("N1", {"source": "test:clock", "fps": 10}, "seq")
    s2 = g2.media("N1", {"source": "test:bars", "fps": 10}, "seq")
    assert (d1, "N1") in MEDIA.sources and (d2, "N1") in MEDIA.sources, \
        sorted(MEDIA.sources)
    assert MEDIA.sources[(d1, "N1")].source == "test:clock"
    assert MEDIA.sources[(d2, "N1")].source == "test:bars", \
        "doc B's N1 clobbered doc A's source"
    del s1, s2
    # statuses are node-keyed PER DOC for the UI
    assert "N1" in MEDIA.statuses(d1) and "N1" in MEDIA.statuses(d2)
    # closing a doc reaps its sources
    assert c.post("/api/doc", json={"action": "close", "id": d2,
                                    "force": True}).status_code == 200
    assert (d2, "N1") not in MEDIA.sources
    assert (d1, "N1") in MEDIA.sources
    MEDIA.purge_doc(d1)


def test_r5_cross_doc_bake_recommits_on_activate():
    """#34: a Layer-out bake fed from ANOTHER doc went stale while that doc
    was edited. Activating the baking doc re-runs its commit path when its
    graph reads a foreign doc and writes a bake."""
    from lestudio.server import app, WS
    c = app.test_client()
    d1 = WS.active                      # the SOURCE doc
    r = c.post("/api/new", json={"name": "baker", "width": 768,
                                 "height": 512})
    d2 = r.json["doc"]["id"]            # the BAKING doc (same size: 1:1 px)
    src_lid = WS.docs[d1].layers[0].id
    tgt = WS.docs[d2].add_layer("baked")
    g2 = WS.graphs[d2]
    g2.set_graph(list(g2.ensure_default().values()) + [
        {"id": "IN", "type": "Layer",
         "params": {"layer": src_lid, "doc": d1}, "inputs": {},
         "x": 0, "y": 0},
        {"id": "OUT", "type": "Layer out", "params": {"layer": tgt.id},
         "inputs": {"image": "IN"}, "x": 100, "y": 0}])
    g2.commit_layer_outputs()
    v0 = float(WS.docs[d2].layer(tgt.id).pixels[10, 10, 0])
    # edit the SOURCE doc (fill its layer green) while d2 is inactive --
    # through the real edit path, so the mutation counter and signature
    # caches move exactly as they would for a user's edit
    WS.docs[d1].fill_layer(src_lid, {"kind": "solid",
                                     "color": [0.0, 1.0, 0.0, 1.0]})
    # activating the baking doc refreshes the stale bake
    assert c.post("/api/doc", json={"action": "activate", "id": d2},
                  headers={"X-User": "u_baker"}).status_code == 200
    px = WS.docs[d2].layer(tgt.id).pixels
    assert float(px[10, 10, 1]) > 0.9 and float(px[10, 10, 0]) < 0.1, \
        "the bake stayed stale after activate (was %.2f)" % v0
    c.post("/api/doc", json={"action": "close", "id": d2, "force": True},
           headers={"X-User": "u_baker"})


# ---------------------------------------------------------------------------
# Wave 2: 2.5D nodes, gbuffer sockets, perf keys, hints
# ---------------------------------------------------------------------------

def _grad_scene(h=48, w=64):
    """A synthetic scene with unambiguous depth structure: bright/sharp on the
    left fading dark/hazy to the right."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([1.0 - xs / w, 0.8 - 0.6 * xs / w,
                    0.6 + 0.3 * np.sin(ys / 3) * (1 - xs / w)], -1)
    return np.clip(img, 0, 1).astype(np.float32)


def test_r5_depth_node_grey_and_polarity():
    """#14: the Depth node outputs the estimator's field as a grey image in
    [0, 1]; the `near` choice flips polarity exactly."""
    from lestudio import OPS
    img = _grad_scene()
    p = {q["name"]: q["default"] for q in OPS["Depth"]["params"]}
    out = np.asarray(OPS["Depth"]["fn"]((48, 64), {"image": img}, p))
    assert out.shape == (48, 64, 3)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert np.allclose(out[..., 0], out[..., 1]), "grey means grey"
    assert float(out.std()) > 1e-3, "a real scene must not yield a flat field"
    flipped = np.asarray(OPS["Depth"]["fn"](
        (48, 64), {"image": img}, dict(p, near="dark")))
    assert np.allclose(out + flipped, 1.0, atol=1e-5), \
        "near=bright and near=dark are exact complements"


def test_r5_depth_fog_wired_depth_skips_estimation():
    """#14: Depth fog's optional `depth` input replaces the estimator -- so a
    Depth node's one estimate can feed fog + relight + parallax. Pinned by
    counting estimator calls."""
    import lestudio as L
    from lestudio import OPS
    img = _grad_scene()
    calls = [0]
    orig = L._estimate_depth

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    L._estimate_depth = counting
    try:
        p = {q["name"]: q["default"] for q in OPS["Depth fog"]["params"]}
        wired = np.zeros((48, 64, 3), np.float32)
        wired[:, :32] = 1.0                      # left half near, right far
        OPS["Depth fog"]["fn"]((48, 64), {"image": img, "depth": wired}, p)
        assert calls[0] == 0, "wired depth must skip estimation"
        OPS["Depth fog"]["fn"]((48, 64), {"image": img, "depth": None}, p)
        assert calls[0] == 1, "unwired depth estimates as before"
    finally:
        L._estimate_depth = orig
    # and the wired field actually steers the fog: the far (dark-wire) half
    # carries more fog colour shift than the near half
    fogged = np.asarray(OPS["Depth fog"]["fn"](
        (48, 64), {"image": img.copy() * 0 + 0.1, "depth": wired},
        dict(p, density=2.0)))
    assert float(fogged[:, 40:].mean()) > float(fogged[:, :24].mean()) + 0.05, \
        "far side (dark on the wire) must catch more fog"


def test_r5_relight_opposed_lights_differ_and_lit_side_brighter():
    """#15 (audited probe: std .061 vs .039 across opposed lights): moving the
    light produces a measurably different image, and the side the light sits
    on comes out brighter. Deterministic."""
    from lestudio import OPS
    img = np.full((48, 64, 3), 0.6, np.float32)
    ys, xs = np.mgrid[0:48, 0:64].astype(np.float32)
    depth = np.stack([np.clip(1.0 - xs / 64, 0, 1)] * 3, -1)  # left = near hill
    p = {q["name"]: q["default"] for q in OPS["Relight"]["params"]}
    left = np.asarray(OPS["Relight"]["fn"](
        (48, 64), {"image": img, "depth": depth},
        dict(p, light_x=0.05, light_y=0.5)))
    right = np.asarray(OPS["Relight"]["fn"](
        (48, 64), {"image": img, "depth": depth},
        dict(p, light_x=0.95, light_y=0.5)))
    assert float(np.abs(left - right).max()) > 0.05, \
        "opposed lights must visibly differ"
    assert float(left[:, :20].mean()) > float(left[:, 44:].mean()), \
        "the side nearest the light is brighter (light at left)"
    assert float(right[:, 44:].mean()) > float(right[:, :20].mean()), \
        "the side nearest the light is brighter (light at right)"
    again = np.asarray(OPS["Relight"]["fn"](
        (48, 64), {"image": img, "depth": depth},
        dict(p, light_x=0.05, light_y=0.5)))
    assert np.array_equal(left, again), "pure NumPy relighting is deterministic"


def test_r5_parallax_foreground_moves_more():
    """#16: pixels shift along (dx, dy) by (depth - pivot) * strength, so the
    bright(near) plane slides while the far plane stays put with pivot=0."""
    from lestudio import OPS
    h, w = 48, 64
    img = np.zeros((h, w, 3), np.float32)
    img[:, 20:24] = [1, 0, 0]                    # a red bar in the near half
    img4 = np.concatenate([img, np.ones((h, w, 1), np.float32)], -1)
    depth = np.zeros((h, w, 3), np.float32)
    depth[:24] = 1.0                             # top rows near, bottom rows far
    p = {q["name"]: q["default"] for q in OPS["Parallax"]["params"]}
    p.update(strength=0.1, pivot=0.0, dx=1.0, dy=0.0)
    out = np.asarray(OPS["Parallax"]["fn"](
        (h, w), {"image": img4, "depth": depth}, p))
    # near rows: bar moved right by ~strength*w = 6.4 px; far rows: unmoved
    assert float(out[10, 27:30, 0].mean()) > 0.5, "foreground shifted"
    assert float(out[10, 20:22, 0].mean()) < 0.5, "foreground left its origin"
    assert float(out[40, 21:23, 0].mean()) > 0.5, "background stayed put"
    # edge clamp: no wrap-around garbage, alpha carried
    assert out.shape[-1] == 4 and np.isfinite(out).all()


def test_r5_sdf_render_gbuffer_sockets():
    """#17: SDF render grows depth+normal sockets from render_gbuffer --
    shaped right, non-flat, near = bright with background 0; on an engine
    without the faculty they are honestly black, not a crash."""
    import lestudio as L
    from lestudio import OPS, mind
    p = {q["name"]: q["default"] for q in OPS["SDF render"]["params"]}
    p["dsl"] = "(sphere 0.8)"
    outs = OPS["SDF render"]["fn"]((72, 96), {}, p)
    assert set(outs) == {"out", "depth", "normal"}
    dg = np.asarray(outs["depth"]); ng = np.asarray(outs["normal"])
    assert dg.shape[:2] == ng.shape[:2] and ng.shape[-1] == 3
    assert float(dg.max()) > 0.3, "the sphere must register in depth"
    assert float(dg.min()) == 0.0, "background is 0 (documented)"
    assert float(ng.std()) > 0.05, "normals vary across a sphere"
    # centre of frame is the sphere's nearest point: brighter than its rim
    ch, cw = dg.shape[0] // 2, dg.shape[1] // 2
    on = dg[..., 0] > 0
    assert dg[ch, cw, 0] >= float(dg[..., 0][on].mean()), "near = bright"
    # honest degradation: engine without render_gbuffer -> black aux, live out
    real = mind()

    class _NoG:
        def __getattr__(self, k):
            if k == "render_gbuffer":
                raise AttributeError(k)
            return getattr(real, k)
    saved = L.mind
    L.mind = lambda: _NoG()
    try:
        outs2 = OPS["SDF render"]["fn"]((36, 48), {}, p)
        assert float(np.asarray(outs2["depth"]).max()) == 0.0
        assert float(np.asarray(outs2["normal"]).max()) == 0.0
        assert float(np.asarray(outs2["out"]).max()) > 0.0, \
            "the beauty pass must survive the missing faculty"
    finally:
        L.mind = saved


def test_r5_selection_node_reads_stored_and_active():
    """#28: the Selection node mirrors the Mask node for selections -- a
    stored id reads that selection, blank reads the active one, nothing
    selected is transparent zeros, and marquee edits move the signature."""
    d = Document(64, 48)
    g = NodeGraph(d)
    g.set_graph([{"id": "S", "type": "Selection", "params": {},
                  "inputs": {}, "x": 0, "y": 0}])
    # no selection at all: honest empty
    out = np.asarray(g.evaluate("S"))
    assert float(out.max()) == 0.0 and out.shape[-1] == 4
    # a working (scratch) marquee is the active selection
    sel = d.select("rect", {"x0": 8, "y0": 8, "x1": 24, "y1": 24})
    s_active = g._sig("S")
    out = np.asarray(g.evaluate("S"))
    assert float(out[16, 16, 0]) > 0.9 and float(out[40, 40, 0]) < 0.05
    # a stored id reads THAT selection even after the scratch changes
    kept = d.keep_selection(sel.id, name="subject")
    g.patch_node("S", params={"selection": kept.id})
    out = np.asarray(g.evaluate("S"))
    assert float(out[16, 16, 0]) > 0.9
    # editing the selection changes the signature (no stale mattes)
    s1 = g._sig("S")
    d.modify_selection(kept.id, "expand", 6)
    assert g._sig("S") != s1
    assert np.asarray(g.evaluate("S"))[16, 28, 0] > 0.5, "expanded matte serves"
    # a stale id degrades to transparent zeros, like the doc promises
    g.patch_node("S", params={"selection": "SEL_gone"})
    assert float(np.asarray(g.evaluate("S")).max()) == 0.0
    assert s_active != g._sig("S")


def test_r5_keyers_preserve_source_alpha():
    """#46: Luma/Chroma key are rgba-aware -- out's alpha is source alpha x
    matte, so keying a semi-transparent element keeps its transparency, and
    opaque inputs keep the old matte-in-alpha contract exactly (x1.0)."""
    from lestudio import OPS
    h, w = 32, 40
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = [0.1, 0.85, 0.12]             # green screen
    img[10:20, 10:30, :3] = [0.8, 0.5, 0.4]      # subject
    img[..., 3] = 0.5                            # semi-transparent SOURCE
    for name in ("Chroma key", "Luma key"):
        assert OPS[name].get("rgba"), "%s must see the source alpha" % name
    pc = {q["name"]: q["default"] for q in OPS["Chroma key"]["params"]}
    out = OPS["Chroma key"]["fn"]((h, w), {"image": img.copy()}, pc)
    m = np.asarray(out["matte"])[..., 0]
    a = np.asarray(out["out"])[..., 3]
    assert np.allclose(a, m * 0.5, atol=1e-5), \
        "chroma out alpha = source alpha * matte"
    pl = {q["name"]: q["default"] for q in OPS["Luma key"]["params"]}
    out = OPS["Luma key"]["fn"]((h, w), {"image": img.copy()}, pl)
    m = np.asarray(out["matte"])[..., 0]
    a = np.asarray(out["out"])[..., 3]
    assert np.allclose(a, m * 0.5, atol=1e-5), \
        "luma out alpha = source alpha * matte"
    # opaque input: identity with the old behaviour (matte IS the alpha)
    img[..., 3] = 1.0
    out = OPS["Chroma key"]["fn"]((h, w), {"image": img.copy()}, pc)
    assert np.allclose(np.asarray(out["out"])[..., 3],
                       np.asarray(out["matte"])[..., 0], atol=1e-5)


def test_r5_splatify_doc_dropped_stale_ply_claim():
    """#18: the .ply export left in an earlier round; the doc stops
    advertising it."""
    from lestudio import OPS
    assert ".ply" not in OPS["Splatify"]["doc"]
    assert "Export menu" not in OPS["Splatify"]["doc"]


def test_r5_graph_run_honors_width():
    """#5: /api/graph/run with `w` renders through render_at at that width
    (measured 3.4x vs full res) and serves the PNG from the job result;
    without `w` the full-res evaluate+commit path is unchanged."""
    import io as _io
    import time as _time
    from PIL import Image
    from lestudio.server import app, WS
    c = app.test_client()
    r = c.post("/api/graph/run", json={"w": 96})
    assert r.status_code == 200
    jid = r.json["job"]
    for _ in range(200):
        st = c.get("/api/job/%s" % jid).json
        if st["done"]:
            break
        _time.sleep(0.02)
    assert st["done"] and not st["error"], st
    pr = c.get("/api/job/%s/result" % jid)
    assert pr.status_code == 200
    im = Image.open(_io.BytesIO(pr.data))
    assert im.size[0] == 96, "the render honoured w (got %s)" % (im.size,)
    assert im.size[1] == round(96 * WS.doc.height / WS.doc.width)
    # garbage w is refused, absent w still runs the full-res job
    assert c.post("/api/graph/run", json={"w": "big"}).status_code == 400
    assert c.post("/api/graph/run", json={"w": 4}).status_code == 400
    r = c.post("/api/graph/run", json={})
    assert r.status_code == 200
    # preview + output.png accept w too (evaluation-capped, not encode-capped)
    st = c.get("/api/state").json
    onode = next(n["id"] for n in st["graph"] if n["type"] == "Output")
    pr = c.get("/api/graph/preview/%s.png?w=80" % onode)
    assert pr.status_code == 200
    assert Image.open(_io.BytesIO(pr.data)).size[0] == 80
    pr = c.get("/api/graph/output.png?w=80")
    assert Image.open(_io.BytesIO(pr.data)).size[0] == 80


def test_r5_sigs_endpoint_no_longer_composites():
    """#37: /api/graph/sigs used to composite + hash the document per poll
    (~232 ms on a bare-Output graph). The Output signature is now built from
    _MUT_REV + doc id + frame; pinned by counting composite() calls."""
    from lestudio.server import app, WS
    c = app.test_client()
    doc = WS.doc
    calls = [0]
    orig = doc.composite

    def counting(*a, **k):
        calls[0] += 1
        return orig(*a, **k)
    doc.composite = counting
    try:
        s1 = c.get("/api/graph/sigs").json
        s2 = c.get("/api/graph/sigs").json
        assert calls[0] == 0, \
            "sigs composited the document %d time(s)" % calls[0]
    finally:
        del doc.composite
    assert s1["__output"] == s2["__output"], "idle polls are stable"
    # ... and the signature still MOVES when the document does
    doc.paint(doc.layers[0].id, [(10, 10), (30, 30)], radius=8,
              color=(1, 0, 0))
    s3 = c.get("/api/graph/sigs").json
    assert s3["__output"] != s1["__output"], "an edit must move the signature"


def test_r5_shade_cache_is_per_layer():
    """#36: the impasto shade cache keyed on the GLOBAL mutation counter, so
    any edit anywhere re-lit every impasto layer (measured 1716 ms -> 832 ms
    for a composite after an unrelated opacity edit, 1080p x 3 layers).
    Pinned by call count: editing layer A's opacity re-shades NOTHING, and
    painting on B re-shades only B."""
    import lestudio as L
    from lestudio import composite_cached
    d = Document(256, 192)
    rng = np.random.default_rng(0)
    la = d.add_layer("A", record=False)
    lb = d.add_layer("B", record=False)
    for l in (la, lb):
        l.pixels[...] = rng.random((192, 256, 4), np.float32)
        l.height_map = rng.random((192, 256)).astype(np.float32) * 2.0
    composite_cached(d)                      # warm both shades
    shaded = []
    orig = L._relief_shade

    def counting(px, *a, **k):
        shaded.append(px.shape)
        return orig(px, *a, **k)
    L._relief_shade = counting
    try:
        d.edit_layer(la.id, opacity=0.5)     # unrelated to any height field
        composite_cached(d)
        assert len(shaded) == 0, \
            "an opacity edit re-lit %d layer(s)" % len(shaded)
        # a real content edit re-shades ONLY the touched layer
        d.fill_layer(lb.id, {"kind": "solid", "color": [0.0, 1.0, 0.0, 1.0]})
        composite_cached(d)
        assert len(shaded) == 1, \
            "filling B re-lit %d layers (want 1: only B)" % len(shaded)
    finally:
        L._relief_shade = orig


def test_r5_missing_upstream_and_unknown_type_read_like_english():
    """#59: a dangling wire or a node type from a newer build surfaces as a
    sentence naming the problem, not a KeyError chip."""
    d = Document(64, 48)
    g = NodeGraph(d)
    g.set_graph([
        {"id": "B", "type": "Blur", "params": {},
         "inputs": {"image": "GONE_1"}, "x": 0, "y": 0}])
    try:
        g.evaluate("B")
        assert False, "dangling wire evaluated"
    except ValueError as e:
        msg = str(e)
        assert "GONE_1" in msg and "deleted node" in msg, msg
    except KeyError:
        assert False, "still a raw KeyError"
    g.set_graph([{"id": "X", "type": "Hologram phaser", "params": {},
                  "inputs": {}, "x": 0, "y": 0}])
    try:
        g.evaluate("X")
        assert False, "unknown type evaluated"
    except ValueError as e:
        assert "Hologram phaser" in str(e) and "newer build" in str(e), e
    except KeyError:
        assert False, "still a raw KeyError"
    # real op errors keep their type/traceback (here: a cycle stays ValueError
    # with its own message, and a healthy graph still runs)
    g.set_graph([{"id": "S", "type": "Solid",
                  "params": {"r": 1, "g": 0, "b": 0}, "inputs": {},
                  "x": 0, "y": 0}])
    assert float(np.asarray(g.evaluate("S"))[..., 0].mean()) > 0.9


def test_r5_hints_on_the_audited_worst_offenders():
    """#38: the ~90 blind dials from the audit carry one-sentence hints in
    op_catalog (the UI renders them as tooltips); #57: Fluid's default swirl
    no longer buries buoyancy."""
    from lestudio import op_catalog
    cat = op_catalog()

    def hints(node):
        return {q["name"]: q.get("hint") for q in cat[node]["params"]}
    for node, names in (
            ("Fractal", ["cx", "cy", "span", "power", "julia", "jre", "jim",
                         "iters"]),
            ("Orbit trap", ["dsl", "trap_kind", "trap_x", "trap_y", "trap_z",
                            "trap_scale", "orbit", "height", "dist"]),
            ("SDF render", ["dsl", "power", "orbit", "height", "dist",
                            "reflect"]),
            ("Sample image", ["u", "v"]),
            ("Band", ["lo", "hi"]),
            ("Smart smooth", ["eps"]),
            ("Segment", ["k"]),
            ("Values to texture", ["v1", "v2", "v3", "v4"]),
            ("Color wheels", ["shadows_end", "highlights_start"]),
            ("Morph", ["t"]),
            ("Deconvolve", ["iters", "sigma"]),
            ("Levels", ["black", "white", "gamma"]),
            ("Curves", ["shadows", "midtones", "highlights"]),
            ("Channel mixer", ["from_red", "from_green", "from_blue"]),
            ("Fluid", ["buoyancy", "swirl", "viscosity"])):
        hs = hints(node)
        for pname in names:
            h = hs.get(pname)
            assert h and len(h) > 12, "%s.%s has no hint" % (node, pname)
            if len(pname) > 2:                    # 1-letter dials false-positive
                assert not ("%s" % h).lower().startswith(pname.lower()), \
                    "%s.%s hint just restates the name" % (node, pname)
    # the Julia seed hint carries the audited phrasing idea, not a restated name
    fr = hints("Fractal")
    assert "seed" in fr["jre"], fr["jre"]
    # Fluid #57: swirl default down where buoyancy can breathe, hint says why
    fl = {q["name"]: q for q in cat["Fluid"]["params"]}
    assert fl["swirl"]["default"] == 8.0
    assert "buoyancy" in fl["swirl"]["hint"]


# ---------------------------------------------------------------------------
# Wave 3: the per-stroke media model
# ---------------------------------------------------------------------------

def _full_reshade(d):
    """Force the next composite to re-light everything, the way the audit's
    probe did: drop the per-layer shade cache and the composite patch cache
    that was hiding the retroactive re-shade until save/reload."""
    for l in d.layers:
        l._shade_rev = None
    d._ccache = None


def test_r5_media_is_per_stroke_not_per_layer():
    """#2 (P0): painting water stroke B must NEVER re-shade oil stroke A.
    Before media_map the layer scalars were 'last media wins' and a full
    reshade (or save/reload) silently re-rendered A matte -- the audit
    measured 74/255 over 1532 px."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(200, 120)
    lid = d.layers[0].id
    pts_a = [(30 + i, 60) for i in range(0, 40, 4)]     # left
    pts_b = [(150 + i, 60) for i in range(0, 40, 4)]    # right, disjoint
    d.paint(lid, pts_a, color=(1, 0, 0), radius=8, media="oil")
    _full_reshade(d)
    before = np.asarray(d.composite()).copy()
    d.paint(lid, pts_b, color=(0, 0, 1), radius=8, media="water")
    _full_reshade(d)
    after = np.asarray(d.composite())
    reg_a = (slice(40, 80), slice(10, 90))              # A plus its skirt
    assert np.array_equal(after[reg_a], before[reg_a]), \
        "water stroke B retroactively re-shaded oil stroke A: max delta %r" \
        % float(np.abs(after[reg_a] - before[reg_a]).max())
    # ...and B genuinely landed as water somewhere
    assert not np.array_equal(after, before)

    # pin 2: the same guarantee survives save + reload
    data = save_workspace({d.id: d}, {}, d.id)
    docs, _, _, *_x = load_workspace(data)
    d2 = docs[d.id] if isinstance(docs, dict) else docs[0]
    re = np.asarray(d2.composite())
    assert np.array_equal(re[reg_a], after[reg_a]), \
        "save/reload re-shaded stroke A: max delta %r" \
        % float(np.abs(re[reg_a] - after[reg_a]).max())


def test_r5_old_format_doc_renders_on_the_scalar_fallback():
    """#2 back-compat pin: a .lews with NO media_map section (every file
    written before wave 3) must render byte-identically to the scalar
    'last media wins' path -- old pictures may not change appearance."""
    from lestudio import Document, _doc_section, _doc_from_section
    d = Document(160, 100)
    lid = d.layers[0].id
    d.paint(lid, [(30 + i, 50) for i in range(0, 40, 4)],
            color=(0, 0.5, 0), radius=8, media="oil")
    dm, arrays = _doc_section(d, None)
    assert any(k.startswith("mediamap_") for k in arrays), \
        "a media stroke must serialise its map"
    # forge the OLD format: strip the map section from the payload
    for k in [k for k in arrays if k.startswith("mediamap_")]:
        del arrays[k]
    for lm in dm["layers"]:
        lm.pop("has_media_map", None)
    old, _g = _doc_from_section(dm, arrays)
    assert all(getattr(l, "media_map", None) is None for l in old.layers)
    old_px = np.asarray(old.composite())
    # the reference scalar-fallback render: same doc, map dropped by hand
    ref, _g2 = _doc_from_section(*_doc_section(d, None))
    for l in ref.layers:
        l.media_map = None
    assert np.array_equal(old_px, np.asarray(ref.composite())), \
        "an old-format doc must hit the scalar path byte-for-byte"


def test_r5_restyle_strokes_edits_the_record_and_rerenders():
    """#12: the medium (and colour/radius/opacity/load/mix) of a PAST stroke
    is editable: restyle rewrites k['brush'] and replays the layer."""
    from lestudio import Document
    d = Document(200, 120)
    lid = d.layers[0].id
    sid = d.paint(lid, [(30 + i, 60) for i in range(0, 40, 4)],
                  color=(1, 0, 0), radius=8, media="oil")
    d.paint(lid, [(150 + i, 60) for i in range(0, 40, 4)],
            color=(0, 0, 1), radius=8, media="water")
    _full_reshade(d)
    before = np.asarray(d.composite()).copy()
    r = d.restyle_strokes([sid], media="water", color=(0, 1, 0))
    assert r["restyled"] == [sid] and r["layers"] == [lid]
    k = d.stroke_by_id(sid)
    assert k["brush"]["media"] == "water"
    assert k["brush"]["color"] == [0.0, 1.0, 0.0]
    _full_reshade(d)
    after = np.asarray(d.composite())
    reg_a = (slice(40, 80), slice(10, 90))
    assert float(np.abs(after[reg_a] - before[reg_a]).max()) > 0.1, \
        "restyling A to green water must visibly re-render it"
    # one undo entry puts everything back -- record, pixels, media map
    assert d._undo[-1][0] == "Restyle strokes"
    d.undo()
    assert d.stroke_by_id(sid)["brush"]["media"] == "oil"
    _full_reshade(d)
    assert np.array_equal(np.asarray(d.composite()), before), \
        "undo must restore the pre-restyle render exactly"
    # validation: unknown keys, bad values, unknown media all refuse cleanly
    for kw in ({"paper": "vellum"}, {"media": "gouache"},
               {"color": [1, 2]}, {"radius": float("nan")}):
        try:
            d.restyle_strokes([sid], **kw)
            assert False, "%r was accepted" % (kw,)
        except ValueError:
            pass
    try:
        d.restyle_strokes([sid])
        assert False, "no-op restyle was accepted"
    except ValueError as e:
        assert "no valid updates" in str(e)
    # media and material are ONE choice: setting a material clears the medium
    d.restyle_strokes([sid], material="gold")
    b = d.stroke_by_id(sid)["brush"]
    assert b.get("material") == "gold" and "media" not in b
    d.restyle_strokes([sid], media="oil")
    b = d.stroke_by_id(sid)["brush"]
    assert b.get("media") == "oil" and "material" not in b


def test_r5_restyle_refuses_unfaithful_layers():
    """#12: restyle rides the same faithfulness gate as every stroke edit --
    a layer holding non-stroke content refuses rather than eating it."""
    from lestudio import Document
    d = Document(120, 80)
    lid = d.layers[0].id
    sid = d.paint(lid, [(20, 40), (60, 40)], color=(1, 0, 0), radius=6,
                  media="oil")
    d.layers[0].pixels[10:20, 80:110, :] = 0.7   # foreign content: a "fill"
    try:
        d.restyle_strokes([sid], media="water")
        assert False, "restyle rebuilt away non-stroke content"
    except ValueError as e:
        assert "not painted as strokes" in str(e)
    assert d.stroke_by_id(sid)["brush"]["media"] == "oil", \
        "a refused restyle must not half-apply"


def test_r5_restyle_endpoint_and_stroke_meta():
    """#12 server half: POST /api/strokes/restyle with friendly 400s, and
    stroke_meta now exposes what the panel edits (media, material, opacity,
    hardness, load, tip)."""
    import lestudio.server as SV
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    SV.DOC.resize(200, 120)
    del SV.DOC.layers[1:]
    SV.DOC.strokes = []
    lid = SV.DOC.layers[0].id
    SV.DOC.layer(lid).pixels[:] = 0.0
    SV.DOC.layer(lid).source = None
    SV.DOC._replay_base = {}
    sid = SV.DOC.paint(lid, [(30 + i, 60) for i in range(0, 40, 4)],
                       color=(1, 0, 0), radius=8, media="oil")
    m = SV.DOC.stroke_meta(sid)
    assert m["media"] == "oil" and m["material"] is None
    assert abs(m["opacity"] - 1.0) < 1e-9 and abs(m["hardness"] - 0.7) < 1e-9
    assert abs(m["load"] - 0.6) < 1e-9 and m["tip"] is None
    r = c.post("/api/strokes/restyle", json={"ids": [sid], "media": "water",
                                             "opacity": 0.5})
    assert r.status_code == 200 and r.json["restyled"] == [sid], r.json
    m = SV.DOC.stroke_meta(sid)
    assert m["media"] == "water" and abs(m["opacity"] - 0.5) < 1e-9
    # unknown ids are LISTED, not a 500 on the first KeyError
    r = c.post("/api/strokes/restyle", json={"ids": ["K999", sid, "K998"],
                                             "media": "oil"})
    assert r.status_code == 400
    assert "K999" in r.json["error"] and "K998" in r.json["error"]
    # no valid updates is a friendly 400 too
    r = c.post("/api/strokes/restyle", json={"ids": [sid]})
    assert r.status_code == 400 and "no valid updates" in r.json["error"]
    # the UI reaches it (the endpoint sweep will hold this forever)
    ui = open(os.path.join(os.path.dirname(__file__), "..", "src",
                           "lestudio", "static", "index.html")).read()
    assert "/api/strokes/restyle" in ui


def test_r5_custom_tip_is_stamped_and_replays_faithfully():
    """#13: the custom brush TIP id rides in the stroke record, so a replay
    re-stamps the same tip and the whole stroke-edit suite works on tip
    layers instead of refusing them as unfaithful."""
    from lestudio import Document
    d = Document(160, 100)
    lid = d.layers[0].id
    tip = np.zeros((15, 15), np.float32)
    tip[2:13, 6:9] = 1.0                       # a slit tip: nothing like round
    b = d.add_brush("slit", tip)
    sid = d.paint(lid, [(30 + i, 50 + (i % 8)) for i in range(0, 80, 5)],
                  color=(0.1, 0.2, 0.8), radius=9, brush=b.id)
    assert d.stroke_by_id(sid)["brush"]["tip"] == b.id
    assert d.replay_is_faithful(lid), \
        "a tip stroke must replay byte-close (this gated ALL stroke edits)"
    # nudge -- the flagship guarded edit -- now accepts the layer
    moved = d.nudge_strokes(lid, [(40, 54), (44, 58)], radius=24)
    assert moved > 0
    # join refuses across DIFFERENT tips rather than re-stamping half wrong
    b2 = d.add_brush("dot", np.ones((7, 7), np.float32))
    s2 = d.paint(lid, [(90, 50), (110, 50)], color=(0.1, 0.2, 0.8),
                 radius=9, brush=b2.id)
    try:
        d.join_strokes([sid, s2], gap=1e9)
        assert False, "joined strokes with different tips"
    except ValueError as e:
        assert "brush" in str(e)
    # a DELETED tip falls back to the round brush instead of crashing
    d.brushes = [x for x in d.brushes if x.id != b.id]
    assert d.replay_layer(lid) is not None


def test_r5_media_map_rides_undo_resize_duplicate():
    """media_map follows the material_map pattern everywhere state moves:
    undo snapshots, canvas resize (premultiplied resample), duplicate."""
    from lestudio import Document
    d = Document(120, 80)
    lid = d.layers[0].id
    d.paint(lid, [(30, 40), (60, 40)], color=(1, 0, 0), radius=7,
            media="oil")
    l = d.layers[0]
    assert l.media_map is not None and l.media_map.shape == (80, 120, 3)
    cov = l.media_map[..., 2].copy()
    assert (cov > 0.5).any()
    # duplicate: an independent buffer, not a shared view
    cp = d.duplicate_layer(lid)
    assert cp.media_map is not None
    cp.media_map[..., 2] = 0.0
    assert (l.media_map[..., 2] > 0.5).any(), "duplicate must deep-copy"
    # resize: the map resamples with the canvas
    d.resize(240, 160)
    assert l.media_map.shape == (160, 240, 3)
    assert (l.media_map[..., 2] > 0.5).any()
    # undo (of the resize) restores the old map
    d.undo()
    assert l.media_map.shape == (80, 120, 3) or \
        d.layers[0].media_map.shape == (80, 120, 3)
    # erase takes the medium's claim with the paint
    d.paint(lid, [(30, 40), (60, 40)], color=(0, 0, 0), radius=9,
            erase=True, opacity=1.0)
    assert float(d.layer(lid).media_map[38:42, 40:50, 2].max()) < 0.2


# ---------------------------------------------------------------------------
# W4. frontend -- collab rebase, physics UI, node editor wiring, labels
# ---------------------------------------------------------------------------

def _ui():
    return open(os.path.join(os.path.dirname(__file__), "..", "src",
                             "lestudio", "static", "index.html")).read()


def test_r5_op_catalog_exposes_pos_grouping():
    """#21: positional params were blind sliders. P(...) now carries a `pos`
    key grouping each x/y pair (declared x first), op_catalog passes it
    through, and the UI renders one crosshair row per group that sets both
    params from a click on the node's preview."""
    from lestudio import op_catalog
    cat = op_catalog()

    def pos_of(op, name):
        return {q["name"]: q.get("pos") for q in cat[op]["params"]}.get(name)

    assert pos_of("Stroke FX", "attract_x") == "attract"
    assert pos_of("Stroke FX", "attract_y") == "attract"
    assert pos_of("Light shafts", "x") == "sun" == pos_of("Light shafts", "y")
    assert pos_of("Radial gradient", "cx") == "centre"
    assert pos_of("Radial gradient", "cy") == "centre"
    # ungrouped params carry no key
    assert pos_of("Radial gradient", "radius") is None
    ui = _ui()
    assert "p.pos&&!posDone.has(p.pos)" in ui, "the UI renders the pos row"
    assert "click the node's preview" in ui.replace("\\'", "'")


def test_r5_media_vectors_endpoint():
    """#20: the sim was a black box. /api/media/vectors?layer= returns the
    velocity field block-averaged to a <=24x18 grid; 404 (never state
    creation) when the layer has no living medium."""
    import lestudio.server as SV
    from lestudio.server import app
    c = app.test_client()
    SV.DOC.resize(240, 160)
    lid = SV.DOC.layers[0].id
    # a plain layer has no medium: 404, and the probe must not create one
    r = c.get("/api/media/vectors?layer=%s" % lid)
    assert r.status_code == 404
    assert getattr(SV.DOC.layer(lid), "_media", None) is None
    assert c.get("/api/media/vectors?layer=L999").status_code == 404
    # make it ink, stir it, and read the arrows
    assert c.post("/api/layer", json={"action": "edit", "id": lid,
                                      "vol_kind": "inkwater",
                                      "thickness": 10}).status_code == 200
    c.post("/api/paint", json={"layer": lid, "points": [[40, 80], [200, 80]],
                               "color": [0.1, 0.1, 0.9], "radius": 12,
                               "opacity": 1})
    assert c.post("/api/media/step",
                  json={"layer": lid, "steps": 6}).status_code == 200
    j = c.get("/api/media/vectors?layer=%s" % lid).json
    assert j["ok"] and j["gw"] <= 24 and j["gh"] <= 18
    assert len(j["vx"]) == j["gh"] and len(j["vx"][0]) == j["gw"]
    assert len(j["vy"]) == j["gh"] and len(j["vy"][0]) == j["gw"]
    ui = _ui()
    assert 'id="lFlowVis"' in ui and "/api/media/vectors" in ui
    assert "function drawFlowOverlay(" in ui


def test_r5_ui_collab_wiring():
    """#6/#29/#31 client side: graph POSTs carry base_rev and rebase on 409;
    undo/redo surface the author 409 with a force path; tool/layer changes
    ping /api/presence/activity and the roster renders it."""
    ui = _ui()
    # 6: base_rev + rebase (their nodes as base, our delta upserted)
    assert "body.base_rev=GREV" in ui
    assert "function rebaseGraphOnto(" in ui
    assert "merged a teammate's changes" in ui
    assert "force:true" in ui                       # the explicit fallback
    # 29: undo/redo 409 -> message + confirm + force retry
    assert "async function undoRedo(" in ui
    assert "r.error&&r.author" in ui
    # 31: activity pings, debounced, and rendered in the roster
    assert "/api/presence/activity" in ui
    assert "function pingActivity(" in ui and "},2000);" in ui
    assert "e.activity&&e.activity.tool" in ui      # "Bob · painting on Sky"


def test_r5_ui_physics_and_fields():
    """#19/#56/#22 client side: field rows list + always-on canvas markers
    with a modifier-click hit-test; non-media field add warns; gizmo/chip
    edits are merge-throttled instead of one request per pointer tick."""
    ui = _ui()
    assert 'id="fieldRows"' in ui and "function drawFieldRows(" in ui
    assert "function drawFieldMarkers(" in ui
    assert "drawFieldMarkers();" in ui              # wired into repaintOverlay
    assert "(e.ctrlKey||e.metaKey)&&FIELDS.length" in ui   # marker hit-test
    assert "this layer has no living medium" in ui  # 56
    # 22: both push helpers merge + throttle
    assert "_fldQ=Object.assign(_fldQ||{},patch)" in ui
    assert "_ltQ=Object.assign(_ltQ||{},patch)" in ui


def test_r5_ui_node_editor_wiring():
    """#23/#24/#25/#45/#44/#42/#5 client side."""
    ui = _ui()
    # 23: the audited number-socket map replaced "everything but out"
    assert "const NUM_SOCKS=" in ui
    for op in ("'Sample image'", "'Color value'", "'Light direction'",
               "'Shadertoy'", "'Perceptual diff'"):
        assert op in ui.split("const NUM_SOCKS=")[1][:400], op
    assert "return sock==='out' ? 'image' : 'value';" not in ui
    # 24: splineref pickers list paths, and the sel-vs-s refresh bug is gone
    assert "const cur=sel.value;" not in ui
    assert "kind==='layerref'||kind==='maskref'||kind==='splineref'" in ui
    # 25 + 45: delete prunes dot-form wires, for the whole selected set
    seg = ui.split("$('delNode').onclick")[1][:900]
    assert "selNodes" in seg and "String(x.inputs[s]).split('.')[0]" in seg
    # 44: colour-trio rows render their per-channel wire pins
    assert "the row expands to per-channel sliders once wired" in ui
    # 42: when-gated rows hidden, not just dimmed
    assert "row.style.display='none'" in ui.split("if(p.when){")[1][:700]
    # 5: interactive output runs at display width, full-res elsewhere
    assert "function outputDisplayWidth(" in ui
    assert "useW?{w:useW}:{}" in ui.replace('JSON.stringify(useW?{w:useW}:{})',
                                           'useW?{w:useW}:{}')


def test_r5_ui_discoverability_and_labels():
    """#39/#40/#41/#43/#11/#47/#48/#49/#50/#55 -- the labeling sweep."""
    ui = _ui()
    assert 'id="helpBtn"' in ui and 'onclick="toggleShortcuts()"' in ui  # 39
    # 40: the four prompt() flows are miniForm modals now
    assert "function miniForm(" in ui
    for frag in ("miniForm('Animate '+p.name",
                 "miniForm('Export frame sequence'",
                 "miniForm('Export SVG poster'",
                 "to layer',"):
        assert frag in ui, frag
    assert "prompt('First frame'" not in ui
    assert "prompt('SVG poster" not in ui
    assert "prompt('Animate " not in ui
    assert "to which layer?" not in ui
    # 41: one-shot verbs live in Actions, persistent styles stay in Style
    assert 'id="lActions"' in ui
    la = ui.split('id="lActions"')[1][:1600]
    for verb in ("vol_tilt", "vol_print", "vol_run", "vol_stir"):
        assert verb in la, verb
    st = ui.split('id="lStyle"')[1].split('id="lActions"')[0]
    # R6 (WETMEDIA_ANIM_REDESIGN.md): Style is Effects-only now -- the slab /
    # living-media presets moved wholly into the Type select (they used to be
    # duplicated under different names)
    assert "vol_tilt" not in st and "vol_water" not in st and "shadow" in st
    # 43: the bake bar says what it does
    assert '<button id="bakeNode" title=' in ui and ">Bake</button>" in ui
    # 11: absorb has a UI option and soak reads back as soak, not empty
    assert 'value="absorb"' in ui and "vol_kind:'absorb'" in ui
    assert "(l.absorbency||0)>0?'soak':''" in ui
    # 47/48/49/50
    assert "◈ Dispersion" in ui
    assert "$('vantage').style.display=e.target.value==='flat'?'none':''" in ui
    assert 'id="vantage" style="display:none"' in ui
    assert ">◺</button>" in ui                       # palette knife glyph
    assert "Pre-roll: advance the simulation before frame 0" in ui
    # 55: invite modal stops overclaiming access control
    assert "no access control beyond the link" in ui
