"""tests/test_r29.py -- stroke undo is a PATH DELTA, not pixels (R33).

Devin's design: the painting is a series of paths/splines with
properties and a deterministic render, so storing images for stroke
undo is waste. A stroke's undo entry now stores only the pre-stroke
path list (shared via the shadow) and a rerender tag; undo truncates
the paths and re-renders the layer from its replay base. Pixel
snapshots remain only where replay cannot regenerate the state.
"""
import numpy as np


def _doc(w=320, h=200):
    from lestudio import Document
    d = Document(w, h)
    d.layers[0].pixels[...] = 0.0        # transparent sheet, not white
    return d


def test_r29_stroke_entries_carry_no_pixels():
    d = _doc()
    lid = d.layers[0].id
    for i in range(6):
        d.paint(lid, [[20 + i * 40, 50], [50 + i * 40, 80]],
                color=(1, 0, 0), radius=7.0,
                media="oil" if i % 2 else None, load=0.8)
    for ent in d._undo[-6:]:
        snap = ent[1]
        assert snap.get("rerender") == [lid]
        assert all(rec[7] is None for rec in snap["layers"]), \
            "a stroke's undo entry must store no pixel data"
        assert all(rec[8] in (None, False) for rec in snap["layers"]), \
            "nor any height data"
    assert d._undo_bytes() < 2_000_000, \
        "six strokes of history should cost well under 2 MB (%d)" \
        % d._undo_bytes()


def test_r29_path_delta_undo_redo_is_bit_exact():
    d = _doc()
    lid = d.layers[0].id
    states = [d.layer(lid).pixels.copy()]
    heights = [None]
    seq = [dict(points=[[20, 50], [120, 60]], color=(1, 0, 0), radius=8.0),
           dict(points=[[40, 100], [200, 110]], color=(0, 0.5, 1),
                radius=10.0, media="oil", load=0.8),
           dict(points=[[60, 150], [240, 150]], color=(0.2, 0.9, 0.2),
                radius=6.0, media="water", load=0.5),
           dict(points=[[10, 30], [300, 40]], color=(1, 1, 0), radius=9.0,
                media="oil", load=0.7)]
    for s in seq:
        d.paint(lid, **s)
        states.append(d.layer(lid).pixels.copy())
        hm = d.layer(lid).height_map
        heights.append(None if hm is None else hm.copy())

    def check(i):
        assert np.array_equal(d.layer(lid).pixels, states[i]), \
            "pixels must be BIT-exact at state %d" % i
        hm = d.layer(lid).height_map
        assert (hm is None) == (heights[i] is None)
        if hm is not None:
            assert np.array_equal(hm, heights[i]), \
                "paint body must be bit-exact at state %d" % i
    for i in range(3, -1, -1):
        assert d.undo()
        check(i)
    for i in range(1, 5):
        assert d.redo()
        check(i)


def test_r29_unreplayable_ops_fall_back_to_pixel_snapshots():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 50], [120, 60]], color=(1, 0, 0), radius=8.0)
    assert d._undo[-1][1].get("rerender") == [lid]
    # clone samples the COMPOSITE (other layers' state at that moment):
    # replay of this layer alone cannot regenerate it, so it must dirty
    d.clone(lid, [[40, 90], [80, 95]], source=(20, 50), radius=6.0)
    assert not getattr(d.layer(lid), "_replay_ok", True)
    # ...so the NEXT stroke stores real pixels again
    before = d.layer(lid).pixels.copy()
    d.paint(lid, [[30, 120], [200, 130]], color=(0, 1, 0), radius=8.0)
    snap = d._undo[-1][1]
    assert not snap.get("rerender")
    assert any(rec[7] is not None for rec in snap["layers"]), \
        "after an unreplayable op, stroke undo must carry pixels"
    assert d.undo()
    assert np.array_equal(d.layer(lid).pixels, before)


def test_r29_unrecorded_dab_disables_path_delta():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 50], [120, 60]], color=(1, 0, 0), radius=8.0)
    d.paint(lid, [[40, 90], [60, 95]], color=(0, 0, 1), radius=5.0,
            record=False)          # texture dab: replay cannot see it
    assert not getattr(d.layer(lid), "_replay_ok", True)
    d.paint(lid, [[30, 120], [200, 130]], color=(0, 1, 0), radius=8.0)
    assert not d._undo[-1][1].get("rerender")


def test_r29_base_eviction_never_orphans_entries():
    """Evicting a replay base pinned by a live path-delta entry would turn
    its undo into a silent no-op; pinned bases must survive the budget."""
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 50], [120, 60]], color=(1, 0, 0), radius=8.0)
    assert d._undo[-1][1].get("rerender") == [lid]
    d.REPLAY_BASE_BUDGET = 1          # force the budget under any base
    l2 = d.add_layer("second")
    d.paint(l2.id, [[20, 20], [60, 20]], color=(0, 1, 0), radius=5.0)
    assert lid in d._replay_base, \
        "a base pinned by a live path-delta entry must not be evicted"
    state = d.layer(lid).pixels.copy()
    # walk back over the second layer's stroke and its Add layer, then
    # the pinned entry itself must still restore exactly
    while d._undo:
        d.undo()
    assert float(d.layer(lid).pixels[..., 3].max()) < 1e-6, \
        "the pinned path-delta entry must still undo to blank"


def test_r29_history_replay_still_reaches_the_beginning():
    d = _doc()
    lid = d.layers[0].id
    for i in range(5):
        d.paint(lid, [[8 + i * 40, 8], [8 + i * 40, 180]],
                color=(1, 1, 1), radius=5.0)
    frames = list(d.history_frames(frames=32))
    assert float(frames[-1][..., 3].max()) < 1e-6, \
        "history must reach the blank beginning through path-delta entries"
    seq = frames[::-1]
    areas = [float((f[..., 3] > 0.1).sum()) for f in seq]
    assert all(b >= a - 1 for a, b in zip(areas, areas[1:]))
