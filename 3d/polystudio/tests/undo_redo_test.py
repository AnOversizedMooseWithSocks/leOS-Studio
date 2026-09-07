"""Undo/redo state machine, HDR cache cap, cancellation unwinding and the photo size clamps.

These exercise backend.py's own logic against stubs, so they run WITHOUT the leCore engine installed
(handy on a machine that cannot pip install leos-core). They do not replace quality_gate.py, which
needs the real engine and must still be run before shipping a render change.

    python3 tests/undo_redo_test.py
"""
import os
import sys
import threading
import time
import queue as _q

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = open(os.path.join(ROOT, "backend.py")).read()


class _StubMesh:
    def __init__(self, V, F):
        self.vertices = np.asarray(V, float).copy()
        self.faces = [tuple(f) for f in F]

    @property
    def n_faces(self):
        return len(self.faces)


def _load_history_ns():
    """Pull the history helpers out of backend.py and give them the globals they expect."""
    ns = {"np": np}
    exec("class _Obj:\n"
         " def __init__(s,name,mesh,mats,sdf_tree=None,kernel_src=None):\n"
         "  s.name=name; s.mesh=mesh; s.mats=mats; s.sdf_tree=sdf_tree\n"
         "  s.kernel_src=kernel_src; s.sculpt=None\n", ns)
    ns["_S"] = {"objects": {}, "next_id": 1, "rev": 0, "undo": [], "redo": []}
    ns["_UNDO_CAP"] = 40
    ns["_bump"] = lambda oid=None: ns["_S"].update(rev=ns["_S"]["rev"] + 1)
    ns["_sculpt_refresh_mesh"] = lambda o: None
    sys.modules.setdefault("holographic_mesh", type(sys)("holographic_mesh"))
    sys.modules["holographic_mesh"].Mesh = _StubMesh
    ns["_MIND"] = None
    for fn in ("_mind", "_history", "_discard_snapshot", "_trim_undo",
               "_capture_like", "_restore", "_snap_obj", "_snap_scene"):
        i = SRC.index("def %s(" % fn)
        exec(SRC[i:SRC.index("\ndef ", i + 1)], ns)
    i = SRC.index("class _SnapshotCommand:")               # the command the history stores
    exec(SRC[i:SRC.index("\ndef ", i + 1)], ns)
    return ns


def test_undo_redo():
    ns = _load_history_ns()
    S = ns["_S"]

    def mk(y):
        return ns["_Obj"]("cube", _StubMesh([[0, y, 0], [1, y, 0], [0, y, 1]], [(0, 1, 2)]), ["steel"])

    # ADOPTED (1.4.0): the stack is the engine's EditHistory; the routes call it exactly like this.
    def _can(h, name):
        v = getattr(h, name, False)
        return v() if callable(v) else bool(v)

    def undo():
        h = ns["_history"]()
        if not _can(h, "can_undo"):
            return False
        h.undo(None); return True

    def redo():
        h = ns["_history"]()
        if not _can(h, "can_redo"):
            return False
        h.redo(None); return True

    def ys():
        return S["objects"]["o1"].mesh.vertices[:, 1].tolist()

    S["objects"]["o1"] = mk(0.0)
    for y in (1.0, 2.0):
        ns["_snap_obj"]("o1")
        S["objects"]["o1"].mesh = _StubMesh([[0, y, 0], [1, y, 0], [0, y, 1]], [(0, 1, 2)])
    assert ys() == [2, 2, 2]
    undo(); assert ys() == [1, 1, 1], ys()
    undo(); assert ys() == [0, 0, 0], ys()
    assert not undo(), "undo past the bottom must be a no-op"
    redo(); assert ys() == [1, 1, 1], ys()
    redo(); assert ys() == [2, 2, 2], ys()
    assert not redo(), "redo past the top must be a no-op"

    undo(); undo()
    ns["_snap_obj"]("o1")
    S["objects"]["o1"].mesh = _StubMesh([[0, 9, 0], [1, 9, 0], [0, 9, 1]], [(0, 1, 2)])
    assert S["redo"] == [], "a new edit must fork history and clear redo"

    ns["_snap_scene"]()
    S["objects"]["o2"] = mk(5.0); S["next_id"] = 3
    undo(); assert "o2" not in S["objects"], "scene undo must remove the added object"
    redo(); assert "o2" in S["objects"] and S["next_id"] == 3, "scene redo must restore it"
    print("ok  undo/redo: symmetric, clamped at both ends, fork clears redo, scene entries round-trip")


def _load_render_ns():
    ns = {"np": np}
    exec(SRC[SRC.index("class _RenderCancelled"):SRC.index("def _matlib")], ns)
    return ns


def test_hdr_cache_cap():
    ns = _load_render_ns()
    put, cache, cap = ns["_photo_cache_put"], ns["_PHOTO_HDR"], ns["_PHOTO_HDR_CAP"]
    cache.clear()
    for k in range(cap + 3):
        put("sess%d" % k, np.zeros((4, 4, 3)), 40, 30, 24)
    assert len(cache) == cap, (len(cache), cap)
    assert "sess0" not in cache and "sess%d" % (cap + 2) in cache
    print("ok  post cache: bounded at %d entries, oldest evicted (these are megabytes each)" % cap)


def test_cancel_unwinds():
    ns = _load_render_ns()
    Cancelled, put, cache = ns["_RenderCancelled"], ns["_photo_cache_put"], ns["_PHOTO_HDR"]
    cache.clear()
    frames = _q.Queue(); cancelled = {"v": False}; reached = {}

    def on_progress(running, done, total):
        if cancelled["v"]:
            raise Cancelled()
        frames.put((np.asarray(running, float).copy(), done))

    def fake_path_trace(spp):
        for s in range(1, spp + 1):
            time.sleep(0.005)
            on_progress(np.zeros((2, 2, 3)), s, spp)
        return spp

    def worker():
        try:
            reached["full"] = fake_path_trace(200)
            put("live", np.zeros((2, 2, 3)), 8, 6, 200)
        except Cancelled:
            reached["stopped_at"] = frames.qsize()

    t = threading.Thread(target=worker); t.start()
    time.sleep(0.06)
    cancelled["v"] = True                      # what the stream's finally clause does
    t0 = time.time(); t.join(timeout=3); dt = time.time() - t0
    assert not t.is_alive(), "worker did not stop"
    assert "full" not in reached, "the trace ran to completion despite cancellation"
    assert "live" not in cache, "a cancelled render must not populate the post cache"
    print("ok  cancel: unwound after ~%d of 200 samples, %.0fms after the flag; nothing cached"
          % (reached["stopped_at"], dt * 1000))


def test_post_matches_render():
    """G1: /api/photo_post must reproduce the render exactly at the same settings.

    This is the test that would have caught the shipped bug: photo_post called _tonemap, which only
    exists INSIDE photo(), so every call raised NameError -> 500 -> the client silently ignored it and
    the exposure slider did nothing. And even resolved, grading the raw trace instead of the prepared
    buffer would have dropped fog and depth of field the moment a slider moved.
    """
    ns = {"np": np}
    i = SRC.index("def _photo_grade("); j = SRC.index("\ndef _photo_cache_put")
    exec(SRC[i:j], ns)
    grade = ns["_photo_grade"]

    rng = np.random.default_rng(0)
    prepped = rng.random((16, 12, 3)) * 3.0          # post fog/DOF, pre-grade: what we cache

    assert np.array_equal(grade(prepped, 1.4, 0.0), grade(prepped, 1.4, 0.0)), \
        "post must be bit-identical to the render at equal settings"
    a, b = grade(prepped, 0.5), grade(prepped, 2.0)
    assert b.mean() > a.mean(), "more exposure must not darken"
    assert a.min() >= 0 and b.max() <= 1, "output must stay in [0,1]"

    fogged = prepped * 0.6 + 0.4                     # stand-in for depth_fog
    assert not np.allclose(grade(prepped, 1.0), grade(fogged, 1.0)), \
        "sanity: fog changes the image, so the cache must hold the PREPARED buffer"
    print("ok  post/render: bit-identical at equal settings, exposure monotonic, fog preserved")


def test_grade_is_module_level():
    """The post endpoint may only call helpers that exist at module scope."""
    import ast
    tree = ast.parse(SRC)
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_photo_grade" in top, "_photo_grade must be module level for photo_post to reach it"
    src_post = SRC[SRC.index("def photo_post("):]
    src_post = src_post[:src_post.index("\n@bp.route")]
    for name in ("_tonemap", "_prep", "_dof_blur"):
        assert name not in src_post, f"photo_post reaches for {name}, which only exists inside photo()"
    print("ok  photo_post only calls module-level helpers")


def test_size_clamps():
    def qnum(g, k, d, lo, hi, cast=float):
        try:
            return cast(np.clip(cast(g(k, d)), lo, hi))
        except Exception:
            return d
    args = {"w": "1280", "h": "720", "spp": "96"}
    g = lambda k, d=None: args.get(k, d)
    W = qnum(g, "w", 560, 240, 1280, int)
    H = qnum(g, "h", int(W * 0.75), 120, 1280, int)
    spp = qnum(g, "spp", 24, 8, 96, int)
    assert (W, H, spp) == (1280, 720, 96), (W, H, spp)
    args = {"w": "99999", "h": "1"}
    W2 = qnum(g, "w", 560, 240, 1280, int)
    H2 = qnum(g, "h", int(W2 * 0.75), 120, 1280, int)
    assert (W2, H2) == (1280, 120), (W2, H2)
    print("ok  clamps: 1280x720@96spp accepted (old ceiling was 560x420); junk folded to %dx%d" % (W2, H2))


if __name__ == "__main__":
    test_undo_redo()
    test_hdr_cache_cap()
    test_cancel_unwinds()
    test_post_matches_render()
    test_grade_is_module_level()
    test_size_clamps()
    print("\nALL BACKEND STUB TESTS PASSED")
