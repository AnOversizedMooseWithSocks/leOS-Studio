"""tests/test_r36.py -- P1.8 (node fills + graph bakes journal as baked
assets), P3.3 (checkpoint ring), P1.11 (unrecorded paints pinned to an
allowlist).

A node-source fill bakes its content into the asset store at fill time
and journals {op: flood, asset} -- the honest bake, and the last fill
dirty-site gone. A graph bake onto an existing layer journals
{op: bake, asset} with the RESOLVED gate frozen as its own asset.

The checkpoint ring caches the rebuilt state at 48-stroke boundaries
during replay; undo truncations keep journal prefixes intact, so warm
undos replay only the tail (~14 ms vs 180 ms measured at 160 strokes on
640x400). In-place journal edits bump an epoch that invalidates stale
checkpoints; correctness never depends on a checkpoint existing.
"""
import numpy as np


def test_r36_node_fill_journals_as_a_baked_asset():
    from lestudio import Document
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    d.paint(lid, [[20, 40], [140, 40]], color=(1, 0, 0), radius=8.0)
    # a pixel content with NO parametric spec -- the node-fill shape
    content = np.random.RandomState(4).rand(120, 160, 4).astype(np.float32)
    d.flood_fill(lid, 5, 5, content, tolerance=0.05, spec=None)
    k = d.strokes[-1]
    assert k["brush"].get("op") == "flood" and k["brush"].get("asset"), \
        "a node fill must bake its content and journal the asset"
    assert d.replay_is_faithful(lid), \
        "the baked fill replays exactly (this was the last fill dirty-site)"
    before = d.layer(lid).pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(d.layer(lid).pixels, before)


def test_r36_graph_bake_journals_with_frozen_gate():
    from lestudio import Document, NodeGraph, Selection
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    d.paint(lid, [[20, 60], [140, 60]], color=(0, 0, 1), radius=10.0)
    sel = Selection(120, 160, data=np.zeros((120, 160), np.float32))
    sel.data[:, :80] = 1.0
    d.selections.append(sel)
    g = NodeGraph(d)
    g.set_graph([{"id": "n1", "type": "Solid",
                  "params": {"color": [1, 0.5, 0]}, "inputs": {}}])
    before_right = d.layer(lid).pixels[:, 90:150].copy()
    g.apply_to_layer("n1", layer_id=lid, selection=sel.id, sel_feather=3.0)
    k = d.strokes[-1]
    assert k["brush"].get("op") == "bake" and k["brush"].get("asset")
    assert k["brush"].get("gate_asset"), \
        "the RESOLVED gate (feather included) must freeze as an asset"
    assert np.allclose(d.layer(lid).pixels[:, 100:150],
                       before_right[:, 10:], atol=2e-3), \
        "past the feather tail the bake must not land"
    assert d.replay_is_faithful(lid), \
        "a gated graph bake is a journaled op now, not a dirty stamp"
    px = d.layer(lid).pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(d.layer(lid).pixels, px)


def test_r36_checkpoint_ring_speeds_undo_and_stays_exact():
    import time
    from lestudio import Document
    d = Document(320, 240)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    for i in range(120):
        d.paint(lid, [[5 + (i * 5) % 300, 10 + (i * 9) % 220],
                      [15 + (i * 5) % 300, 25 + (i * 9) % 220]],
                color=(1, 0, 0), radius=3.0)
    px = d.layer(lid).pixels.copy()
    d.undo()                                    # cold: captures a checkpoint
    assert d._replay_ckpt.get(lid), "the first replay must leave a checkpoint"
    # R68: a LADDER of positions, not one slot -- painting lays them too,
    # so by here there are several and every one sits on the interval
    ck_is = sorted(d._replay_ckpt[lid])
    assert ck_is, "the ladder must not be empty"
    assert all(i % d.CKPT_EVERY == 0 and i > 0 for i in ck_is), ck_is
    assert d._ckpt_reach(lid) <= d.CKPT_EVERY, \
        "the head must stay within one interval of a checkpoint"
    t0 = time.time()
    for _ in range(8):
        assert d.undo()
    warm = time.time() - t0
    for _ in range(9):
        assert d.redo()
    assert np.array_equal(d.layer(lid).pixels, px), \
        "checkpointed undo/redo must stay bit-exact"
    assert warm < 2.0, "warm undos must ride the checkpoint (%.2fs)" % warm


def test_r36_checkpoints_are_invalidated_by_surgery():
    from lestudio import Document
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    for i in range(60):
        d.paint(lid, [[5 + i * 2, 10], [5 + i * 2, 110]],
                color=(0, 0, 0), radius=2.0)
    d.undo()                                    # capture a checkpoint
    ladder = d._replay_ckpt[lid]
    assert ladder, "the ladder must not be empty"
    epoch0 = next(iter(ladder.values()))["epoch"]
    sid = d.strokes[2]["id"]                    # edit INSIDE the prefix
    d.smooth_strokes([sid], amount=0.6, iterations=2)
    assert d._journal_epoch > epoch0, \
        "an in-place edit must bump the journal epoch"
    # R68: the ladder may have gained fresh rungs since (a rerender lays
    # them), but not one rung from the OLD epoch may ever be chosen --
    # every rung is stamped, and selection checks the stamp
    best = d._ckpt_best(lid, [k for k in d._iter_strokes()
                              if k["layer"] == lid])
    assert best is None or best["epoch"] == d._journal_epoch, \
        "a rung from before the surgery must never be selected"
    assert d.replay_is_faithful(lid), \
        "the stale checkpoint must be ignored, never replayed from"
    assert d.undo(), "surgery undo walks through the invalidation"
    assert d.replay_is_faithful(lid)


def test_r36_unrecorded_paints_are_pinned_to_an_allowlist():
    """P1.11: paint(record=False) lays pixels no replay can regenerate,
    so the set of callers is CLOSED -- replay machinery, the live-stroke
    protocol, the honest unfaithful fallbacks, and preview rendering.
    Anything new must journal instead."""
    import ast
    import inspect
    import lestudio
    allowed = {"paint_live", "replay_region", "_replay_apply",
               "duplicate_strokes", "strokes_to_layer",
               "render_strokes_rgba"}
    tree = ast.parse(inspect.getsource(lestudio))
    bad = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "paint":
                kws = {kw.arg for kw in node.keywords}
                for kw in node.keywords:
                    if (kw.arg == "record"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False):
                        # R50: record=False WITH stroke_new is the
                        # paint_batch law -- one undo record for the
                        # batch, every stroke still its own journal
                        # entry. That call journals; it is not an
                        # unrecorded paint.
                        if "stroke_new" in kws:
                            continue
                        fn = self.stack[-1] if self.stack else "?"
                        if fn not in allowed:
                            bad.append((fn, node.lineno))
            self.generic_visit(node)

    V().visit(tree)
    assert not bad, \
        "new unrecorded paint caller(s) -- journal instead: %s" % bad
