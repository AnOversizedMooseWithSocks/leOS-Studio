"""tests/test_r33.py -- Phase G: the gate (R35, Devin's request).

"Make a selection using the lasso tool or whatever, then use that mask as
a boundary for an effect or tool. Paint fill, particle generation,
smudging/nudging, blur, etc." The standard model everywhere else: an
active selection gates every destructive edit; generation writes only
inside it; no selection means the whole canvas.

Determinism contract: a gated op journals its FROZEN gate as a
content-addressed asset, so replay uses the mask the op ran under, not
whatever the live selection became.
"""
import numpy as np


def _sel_doc(w=160, h=120, left=True):
    from lestudio import Document, Selection
    d = Document(w, h)
    d.layers[0].pixels[...] = 0.0
    sel = Selection(h, w, data=np.zeros((h, w), np.float32))
    if left:
        sel.data[:, :w // 2] = 1.0
    d.selections.append(sel)
    return d, d.layers[0].id, sel


def test_r33_smudge_respects_and_journals_the_gate():
    d, lid, sel = _sel_doc()
    d.paint(lid, [[20, 40], [140, 40]], color=(1, 0, 0), radius=8.0)
    before_right = d.layer(lid).pixels[:, 90:150].copy()
    d.smudge(lid, [[20, 52], [140, 52]], radius=14.0, strength=0.9,
             selection=sel.id)
    assert np.array_equal(d.layer(lid).pixels[:, 90:150], before_right), \
        "outside the gate the smudge must not move a single pixel"
    k = d.strokes[-1]
    assert k["brush"].get("smudge") and k["brush"].get("sel_asset"), \
        "the gated smudge must journal its frozen gate"
    assert d.replay_is_faithful(lid)


def test_r33_heal_writes_only_inside_the_gate():
    d, lid, sel = _sel_doc()
    d.paint(lid, [[20, 40], [140, 40]], color=(0.2, 0.6, 0.9), radius=10.0)
    before_right = d.layer(lid).pixels[:, 90:150].copy()
    d.heal(lid, [[60, 40], [110, 40]], radius=10.0, selection=sel.id)
    assert np.array_equal(d.layer(lid).pixels[:, 90:150], before_right), \
        "heal may read context outside the gate but must write inside only"
    assert d.replay_is_faithful(lid)


def test_r33_gated_clear_and_fill_journal_with_frozen_gate():
    d, lid, sel = _sel_doc()
    d.paint(lid, [[20, 40], [140, 40]], color=(1, 0, 0), radius=8.0)
    d.clear(lid, selection=sel.id)
    assert float(d.layer(lid).pixels[35:45, 20:70, 3].max()) < 1e-5
    assert float(d.layer(lid).pixels[35:45, 90:140, 3].max()) > 0.5
    k = d.strokes[-1]
    assert k["brush"].get("op") == "clear" and k["brush"].get("sel_asset"), \
        "a selection clear is a journaled op now, not a snapshot"
    content = np.zeros((120, 160, 4), np.float32)
    content[...] = (0, 1, 0, 1)
    d.flood_fill(lid, 5, 5, content, tolerance=0.05, selection=sel.id,
                 spec={"type": "color", "color": [0, 1, 0]})
    k = d.strokes[-1]
    assert k["brush"].get("op") == "flood" and k["brush"].get("sel_asset"), \
        "a gated parametric fill journals too (it used to snapshot)"
    assert d.replay_is_faithful(lid)
    # the live selection dying must not corrupt replay: the gate is frozen
    d.selections.clear()
    d.layer(lid)._replay_ok = True
    assert d.replay_is_faithful(lid)


def test_r33_gated_ops_walk_undo_exactly():
    d, lid, sel = _sel_doc()
    d.paint(lid, [[20, 40], [140, 40]], color=(1, 0, 0), radius=8.0)
    d.smudge(lid, [[20, 52], [140, 52]], radius=12.0, strength=0.8,
             selection=sel.id)
    d.clear(lid, selection=sel.id)
    content = np.zeros((120, 160, 4), np.float32)
    content[...] = (0, 1, 0, 1)
    d.flood_fill(lid, 5, 5, content, tolerance=0.05, selection=sel.id,
                 spec={"type": "color", "color": [0, 1, 0]})
    before = d.layer(lid).pixels.copy()
    for _ in range(4):
        assert d.undo()
    for _ in range(4):
        assert d.redo()
    after = d.layer(lid).pixels
    cov = np.maximum(before[..., 3:4], after[..., 3:4])
    diff = max(float(np.abs((before[..., :3] - after[..., :3]) * cov).max()),
               float(np.abs(before[..., 3] - after[..., 3]).max()))
    assert diff <= 2e-3, \
        "a gated edit chain must undo/redo picture-exactly (%.5f)" % diff


def test_r33_nudge_holds_points_outside_the_gate():
    d, lid, sel = _sel_doc()
    d.paint(lid, [[20, 60], [140, 60]], color=(0, 0, 0), radius=4.0)
    pts_before = [list(p) for p in d.strokes[-1]["points"]]
    moved = d.nudge_strokes(lid, [[80, 60], [80, 90]], radius=60.0,
                            strength=1.0, selection=sel.id)
    assert moved > 0, "inside the gate the nudge must still act"
    pts_after = d.strokes[-1]["points"]
    for a, b in zip(pts_after, pts_before):
        if a[0] > 90:
            assert abs(a[1] - b[1]) < 0.5, \
                "a path point outside the boundary must stay put"
    assert d.replay_is_faithful(lid)


def test_r33_selection_modifiers_cover_the_standard_set():
    """G6: feather/grow/shrink existed; smooth, border and invert join
    them -- the full refinement set every other editor ships."""
    from lestudio import Document, Selection
    d = Document(100, 80)
    sel = Selection(80, 100, data=np.zeros((80, 100), np.float32))
    sel.data[20:60, 20:60] = 1.0
    d.selections.append(sel)
    area0 = float(sel.data.sum())
    d.modify_selection(sel.id, "grow", 4)
    assert float(sel.data.sum()) > area0
    d.modify_selection(sel.id, "shrink", 4)
    assert abs(float(sel.data.sum()) - area0) < area0 * 0.2
    d.modify_selection(sel.id, "border", 3)
    b = sel.data
    assert float(b[35:45, 35:45].max()) < 0.5, \
        "border must hollow out the interior"
    d.modify_selection(sel.id, "invert", 0)
    assert float(sel.data[40, 40]) > 0.5, "invert must flip coverage"
    sel.data[:] = 0.0
    sel.data[30:50, 30:50] = 1.0
    d.modify_selection(sel.id, "smooth", 2)
    assert set(np.unique(sel.data)) <= {0.0, 1.0}, \
        "smooth re-thresholds to a hard, de-jiggled boundary"


def test_r33_generation_focuses_into_the_gate():
    """G3 at the engine level: a generated output alpha-multiplied by the
    gate lands only inside the boundary, and the baked asset is BORN
    gated -- replay needs no knowledge of the selection at all."""
    d, lid, sel = _sel_doc()
    out = np.random.RandomState(9).rand(120, 160, 4).astype(np.float32)
    out[..., 3] = 1.0
    g = d._resolve_gate(sel.id)
    gated = out.copy()
    gated[..., 3] = gated[..., 3] * g
    l = d.add_layer("Dream", pixels=gated, asset=True)
    assert float(d.layer(l.id).pixels[:, 90:150, 3].max()) < 1e-5, \
        "generation must not land outside the boundary"
    assert float(d.layer(l.id).pixels[:, 10:70, 3].min()) > 0.9
    assert d.replay_is_faithful(l.id), \
        "the gated bake replays from its asset with no selection knowledge"
