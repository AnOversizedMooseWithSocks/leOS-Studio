"""tests/test_r32.py -- determinism backlog, third sweep.

P1.10: stroke surgery (nudge, smooth, transform, delete, restyle, move
points, to-layer...) edits the PATH SET -- the journal's native material.
The snapshot's stroke shadow already holds the pre-edit paths, so these
undo entries are now pixel-free: restore paths + re-render.

P1.12 (strokes half): a stroke painted under a selection records its gate
as a content-addressed asset. Such strokes used to replay UNMASKED -- a
silent dirty site; now the layer stays faithful even after the live
selection is edited or deleted.

P1.9: generated outputs (dream, match, splats) bake to assets and journal
{op: import, asset} -- the honest rule for anything a seed cannot cheaply
regenerate at replay time.
"""
import numpy as np


def _same_picture(a, b, tol=2e-3):
    """Equality as the faithfulness gate defines it: RGB weighted by
    coverage (replay lays premultiply hygiene under zero alpha in a
    different but equally invisible pattern -- the R16 rule)."""
    cov = np.maximum(a[..., 3:4], b[..., 3:4])
    return float(max(np.abs((a[..., :3] - b[..., :3]) * cov).max(),
                     np.abs(a[..., 3] - b[..., 3]).max())) <= tol


def _flat_doc(w=160, h=120):
    from lestudio import Document
    d = Document(w, h)
    d.layers[0].pixels[...] = 0.0
    return d


def test_r32_selection_strokes_replay_their_gate():
    from lestudio import Selection
    d = _flat_doc()
    lid = d.layers[0].id
    sel = Selection(120, 160, data=np.zeros((120, 160), np.float32))
    sel.data[:, :80] = 1.0
    d.selections.append(sel)
    d.paint(lid, [[20, 40], [140, 40]], color=(1, 0, 0), radius=8.0,
            selection=sel.id)
    k = d.strokes[-1]
    assert k["brush"].get("sel_asset"), \
        "the gate must ride in the stroke record as an asset"
    assert d.replay_is_faithful(lid), \
        "a selection stroke must replay masked (this was a silent dirty)"
    rep = d.replay_layer(lid)
    assert float(rep[35:45, 90:150, 3].max()) < 1e-6, \
        "replay must respect the recorded gate, not paint through it"
    # the LIVE selection moving on must not corrupt the journal
    sel.data[:, :] = 1.0
    d.selections.clear()
    d.layer(lid)._replay_ok = True          # force a fresh pixel compare
    assert d.replay_is_faithful(lid), \
        "the frozen gate asset keeps replay exact after the selection dies"


def test_r32_same_selection_shares_one_asset():
    from lestudio import Selection
    d = _flat_doc()
    lid = d.layers[0].id
    sel = Selection(120, 160, data=np.zeros((120, 160), np.float32))
    sel.data[30:90, 30:130] = 1.0
    d.selections.append(sel)
    for y in (30, 50, 70):
        d.paint(lid, [[20, y], [140, y]], color=(0, 1, 0), radius=6.0,
                selection=sel.id)
    keys = {k["brush"]["sel_asset"] for k in d.strokes}
    assert len(keys) == 1, "one selection, many strokes, ONE stored gate"
    assert len(d._assets) == 1


def test_r32_stroke_surgery_undo_is_pixel_free():
    d = _flat_doc()
    lid = d.layers[0].id
    zig = [[10.0 + i * 15, 60.0 + (14 if i % 2 else -14)] for i in range(8)]
    d.paint(lid, zig, color=(0, 0, 1), radius=5.0)
    sid = d.strokes[-1]["id"]
    before = d.layer(lid).pixels.copy()
    d.smooth_strokes([sid], amount=0.7, iterations=3)
    ent = d._undo[-1]
    assert ent[0] == "Smooth strokes" and ent[1].get("rerender") == [lid]
    assert all(r[7] is None for r in ent[1].get("layers", [])), \
        "a smooth on a faithful layer must snapshot NO pixels"
    assert d.undo()
    assert np.array_equal(d.layer(lid).pixels, before), \
        "path-delta surgery undo must restore bit-exactly"
    # delete-strokes rides the same machinery
    d.paint(lid, [[20, 100], [140, 100]], color=(1, 0, 1), radius=4.0)
    s2 = d.strokes[-1]["id"]
    mid = d.layer(lid).pixels.copy()
    d.delete_strokes([s2])
    ent = d._undo[-1]
    assert ent[0] == "Delete strokes" and ent[1].get("rerender") == [lid]
    assert all(r[7] is None for r in ent[1].get("layers", []))
    assert d.undo()
    assert _same_picture(d.layer(lid).pixels, mid), \
        "undoing a stroke delete must resurrect the ink exactly"


def test_r32_surgery_on_a_dirty_layer_still_snapshots():
    """The pixel-free path is only for layers replay can rebuild; a dirty
    layer's surgery keeps its honest windowed snapshot."""
    d = _flat_doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 30], [140, 30]], color=(0, 0, 1), radius=5.0)
    sid = d.strokes[-1]["id"]
    d.layer(lid)._replay_ok = False           # something unreplayable landed
    d.record("Smooth strokes", only=[lid])
    ent = d._undo[-1]
    assert not ent[1].get("rerender"), \
        "a dirty layer must NOT get a rerender-tag undo entry"


def test_r32_generated_layers_journal_as_assets():
    d = _flat_doc()
    out = np.random.RandomState(5).rand(120, 160, 4).astype(np.float32)
    l = d.add_layer("Dream", pixels=out, asset=True)
    k = [k for k in d.strokes if k["layer"] == l.id]
    assert len(k) == 1 and k[0]["brush"].get("op") == "import"
    assert d.replay_is_faithful(l.id), \
        "a baked generative layer must be replayable from its asset"
    before = d.layer(l.id).pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(d.layer(l.id).pixels, before)
