"""tests/test_r37.py -- the backlog's endgame.

P1.4 (doc-wide geometry): crop, resize and reorient change the
coordinate frame every recorded path lives in, and resampling does not
commute with painting -- so they start a NEW JOURNAL ERA instead of
pretending. The pre-op state lives in the op's own undo snapshot;
replayability resumes at the next stroke from the post-op pixels.

P2.1: a layer that is empty at base capture stores the SENTINEL "empty"
-- no pixels in memory or in the .lews; truth is zeros + journal.

P3.1: save_workspace(cache_pixels=False) writes a JOURNAL-FIRST file --
replay-faithful layers carry no pixel arrays and are rebuilt on open.
Baked pixels are cache; the journal is the document.

P3.2: one history system, pinned -- a session of journaled ops leaves
every undo entry pixel-free; snapshots survive only as the structural
shim they were designed to become.

P3.4: golden-journal CI -- a committed corpus exercising every op type
renders to a pinned crc; any op _replay_apply dispatches must appear in
the corpus, so a new op type cannot ship without a golden journal.

P0.4 (close): playback/timelapse functions join the no-wall-clock AST
pin -- frames are indexed, never timed.
"""
import json
import os
import zlib
import numpy as np

GOLD = os.path.join(os.path.dirname(__file__), "golden")


def test_r37_crop_carries_the_journal_exactly():
    """P1.4 (doc-wide geometry): the journal is document geometry. A crop
    TRANSLATES every recorded path and crops the bases/bodies with the
    pixels, so a stroke that lived inside the kept region replays
    bit-faithfully in the new frame -- no era reset, no dirty flag."""
    from lestudio import Document
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    d.paint(lid, [[40, 40], [120, 90]], color=(1, 0, 0), radius=5.0)
    d.crop(20, 20, 150, 110)
    assert len(d.strokes) == 1, "the crop must carry the journal"
    p0 = d.strokes[0]["points"][0]
    assert abs(p0[0] - 20) < 1e-6 and abs(p0[1] - 20) < 1e-6, \
        "paths translate with the crop"
    assert d.replay_is_faithful(lid), \
        "an interior stroke must replay faithfully in the cropped frame"
    d.paint(lid, [[10, 10], [100, 80]], color=(0, 0, 1), radius=5.0)
    assert d.replay_is_faithful(lid)
    assert d.undo() and d.undo(), "undo walks back across the crop"
    assert (d.width, d.height) == (160, 120)


def test_r37_reorient_carries_the_journal():
    """Reorients are pure array reorderings: paths, bases and bodies all
    turn together, and spooled journal segments turn too."""
    from lestudio import Document
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    d.MAX_STROKES = 4                      # force part of the journal to spool
    for i in range(9):
        d.paint(lid, [[20 + i * 12, 30], [20 + i * 12, 90]],
                color=(0, 0, 0), radius=3.0)
    d.reorient("rot90")
    assert (d.width, d.height) == (120, 160)
    assert len(list(d._iter_strokes(lid))) == 9, \
        "spooled segments must turn with the live list"
    assert d.replay_is_faithful(lid), \
        "a lossless reorient must keep the layer replay-faithful"


def test_r37_empty_base_is_a_sentinel_everywhere():
    from lestudio import Document, save_workspace, load_workspace
    d = Document(160, 120)
    l = d.add_layer("fresh")
    d.paint(l.id, [[20, 30], [140, 60]], color=(1, 0, 0), radius=6.0)
    assert d._replay_base[l.id] == "empty", \
        "an empty layer's base must cost nothing"
    assert d.replay_is_faithful(l.id)
    blob = save_workspace({d.id: d}, {}, d.id)
    docs, graphs, active, extras = load_workspace(blob)
    d2 = docs[active]
    assert d2._replay_base.get(l.id) == "empty", \
        "the sentinel must ride the .lews as a flag, not an array"
    assert d2.replay_is_faithful(l.id)
    assert np.allclose(d2.layer(l.id).pixels, d.layer(l.id).pixels,
                       atol=1e-6)


def test_r37_journal_first_lews_round_trips():
    from lestudio import Document, save_workspace, load_workspace
    d = Document(320, 240)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    l2 = d.add_layer("oil")
    for i in range(20):
        d.paint(lid, [[10 + i * 15, 20], [10 + i * 15, 220]],
                color=(1, 0, 0), radius=4.0)
    d.paint(l2.id, [[20, 60], [300, 120]], color=(0.7, 0.5, 0.2),
            radius=12.0, media="oil", load=0.8)
    full = save_workspace({d.id: d}, {}, d.id)
    light = save_workspace({d.id: d}, {}, d.id, cache_pixels=False)
    assert len(light) < len(full) * 0.8, \
        "the journal-first file must be substantially smaller"
    docs, graphs, active, extras = load_workspace(light)
    d2 = docs[active]
    for L in (lid, l2.id):
        a, b = d.layer(L).pixels, d2.layer(L).pixels
        cov = np.maximum(a[..., 3:4], b[..., 3:4])
        assert float(max(np.abs((a[..., :3] - b[..., :3]) * cov).max(),
                         np.abs(a[..., 3] - b[..., 3]).max())) <= 2e-3, \
            "%s must rebuild picture-exactly from base + journal" % L
        assert d2.replay_is_faithful(L)


def test_r37_one_history_system_journaled_ops_snapshot_nothing():
    """P3.2: for the whole journaled-op family, undo entries carry NO
    pixels -- the snapshot machinery survives only for structural ops."""
    from lestudio import Document, Selection, Stamp
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    sel = Selection(120, 160, data=np.zeros((120, 160), np.float32))
    sel.data[:, :80] = 1.0
    d.selections.append(sel)
    n0 = len(d._undo)
    d.paint(lid, [[10, 20], [140, 30]], color=(1, 0, 0), radius=6.0)
    d.paint(lid, [[10, 50], [140, 55]], color=(0, 0, 1), radius=6.0,
            selection=sel.id)
    d.smudge(lid, [[20, 40], [120, 44]], radius=10.0, strength=0.6)
    d.heal(lid, [[50, 25], [80, 25]], radius=8.0)
    d.fill_layer(lid, {"kind": "solid", "color": [0, 0.4, 0.2, 0.3]},
                 respect_alpha=True)
    d.clear(lid, selection=sel.id)
    d.flip_layer(lid, axis="x")
    d.transform("layer", lid, deg=15)
    d.add_text(lid, "op", x=10, y=90, size=16, color=(1, 1, 1))
    sp = np.zeros((20, 20, 4), np.float32)
    sp[4:16, 4:16] = (0.9, 0.1, 0.5, 1)
    st = Stamp("g", sp)
    st.id = d._mint_id("ST")
    d.stamps.append(st)
    d.place_stamp(lid, st.id, 100, 90)
    d.media_step  # (media covered in golden; this doc has no vol layer)
    for label, snap, _a in d._undo[n0:]:
        px_bytes = sum((r[7][1].nbytes if isinstance(r[7], tuple)
                        else r[7].nbytes)
                       for r in snap.get("layers", ()) if r[7] is not None)
        assert px_bytes == 0, \
            "journaled op %r still snapshots %d pixel bytes" % (
                label, px_bytes)


def test_r37_golden_journal_renders_to_its_pinned_crc():
    """P3.4: the committed corpus is a journal-first .lews; opening it
    REPLAYS every layer, and the composite must match the crc recorded
    when the corpus was authored (same-machine bit-exactness law)."""
    from lestudio import load_workspace
    man = json.load(open(os.path.join(GOLD, "manifest.json")))
    for fname, meta in man.items():
        blob = open(os.path.join(GOLD, fname), "rb").read()
        docs, graphs, active, extras = load_workspace(blob)
        d = docs[active]
        comp = d.composite()
        crc = zlib.crc32(np.ascontiguousarray(comp).tobytes())
        assert crc == meta["composite_crc"], \
            "%s rendered crc %d, pinned %d -- an op changed behaviour " \
            "without regenerating the corpus" % (fname, crc,
                                                 meta["composite_crc"])


def test_r37_every_dispatched_op_has_a_golden_journal():
    """Any new op type must ship with a golden journal: the ops the
    corpus exercises must cover everything _replay_apply dispatches."""
    import ast
    import inspect
    import lestudio
    man = json.load(open(os.path.join(GOLD, "manifest.json")))
    covered = set()
    for meta in man.values():
        covered |= set(meta["ops"])
    src = inspect.getsource(lestudio.Document._replay_apply)
    tree = ast.parse("class _D:\n" + src.replace("\n    ", "\n "))
    dispatched = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "op"
                and isinstance(node.comparators[0], ast.Constant)):
            dispatched.add(node.comparators[0].value)
    dispatched |= {"smudge", "heal", "knife", "blend", "paint"}
    missing = dispatched - covered
    assert not missing, \
        "op(s) with no golden journal -- extend tests/golden: %s" % missing


def test_r37_playback_is_frame_indexed_never_timed():
    """P0.4 closed: the playback/timelapse family joins the wall-clock
    AST pin -- a replay frame is an index into the journal, never a
    function of when it renders."""
    import ast
    import inspect
    import lestudio
    tree = ast.parse(inspect.getsource(lestudio))
    playback = {"timelapse_frames", "history_frames", "_media_render",
                "media_step", "_media_slab_step"}
    bad = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Attribute(self, node):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in ("time", "datetime") \
                    and node.attr in ("time", "perf_counter", "monotonic",
                                      "now", "utcnow", "today"):
                fn = next((f for f in reversed(self.stack)
                           if f in playback), None)
                if fn:
                    bad.append((fn, node.lineno))
            self.generic_visit(node)

    V().visit(tree)
    assert not bad, "wall-clock in the playback path: %s" % bad
