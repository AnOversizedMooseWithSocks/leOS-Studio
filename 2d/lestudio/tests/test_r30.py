"""tests/test_r30.py -- DETERMINISM_BACKLOG lands: canvas actions journal
as ops (paths/points/bboxes + properties), determinism bugs fixed.

Phase 0: launch-stable texture seeds (no PYTHONHASHSEED pin needed),
per-document id minting. Phase 1: smudge/heal journal as stroke kinds,
fills/clear/flip journal as parametric ops -- pixel-free undo entries,
bit-exact undo/redo through deterministic re-render. Phase 2: replay
bases carry the paint BODY, so pre-existing impasto no longer forces
snapshots and a knife needs no special case.
"""
import numpy as np


def _doc(w=320, h=200):
    from lestudio import Document
    d = Document(w, h)
    d.layers[0].pixels[...] = 0.0
    return d


def test_r30_ids_are_per_document():
    from lestudio import Document
    d1, d2 = Document(64, 48), Document(64, 48)
    assert d1.layers[0].id == d2.layers[0].id == "L1"
    assert d1.add_layer("a").id == d2.add_layer("a").id == "L2"
    m1 = d1.add_mask("m")
    m2 = d2.add_mask("m")
    assert m1.id == m2.id, "mask ids must not depend on other documents"


def test_r30_texture_seeds_survive_hash_salt():
    """P0.1: grain/fiber seeds must come from crc32, never salted hash().
    Source-pinned because the salt cannot change within one process."""
    import os
    eng = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "lestudio", "__init__.py")).read()
    assert 'hash((doc.id, l.id, "matgrain", key))' not in eng
    assert 'hash((doc.id, getattr(host, "id", "")))' not in eng
    assert "matgrain" in eng and "crc32" in eng


def test_r30_ops_journal_pixel_free_and_undo_exactly():
    """fill_layer, whole-layer clear, flip and a bucket fill journal as
    ops: no pixels in their undo entries, and undo/redo walks every state
    bit-exactly through re-render."""
    d = _doc()
    lid = d.layers[0].id
    states = [d.layer(lid).pixels.copy()]

    def snap_ok():
        s = d._undo[-1][1]
        assert s.get("rerender") == [lid], "op must journal"
        assert all(rec[7] is None for rec in s["layers"]), \
            "op undo entry must store no pixels"

    d.paint(lid, [[20, 50], [120, 60]], color=(1, 0, 0), radius=8.0)
    states.append(d.layer(lid).pixels.copy())
    d.fill_layer(lid, {"kind": "gradient", "from": [0, 0, 0.2],
                       "to": [0.1, 0.3, 0.5], "angle": 30},
                 respect_alpha=True)
    snap_ok()
    states.append(d.layer(lid).pixels.copy())
    d.flood_fill(lid, 5, 5, d._resolve_fill_spec({"type": "color",
                                                  "color": [0.6, 0.2, 0.1]}),
                 tolerance=0.2, contiguous=False,
                 spec={"type": "color", "color": [0.6, 0.2, 0.1]})
    snap_ok()
    states.append(d.layer(lid).pixels.copy())
    d.flip_layer(lid, axis="x")
    snap_ok()
    states.append(d.layer(lid).pixels.copy())
    d.clear(lid)
    snap_ok()
    states.append(d.layer(lid).pixels.copy())
    for i in range(len(states) - 2, -1, -1):
        assert d.undo()
        assert np.array_equal(d.layer(lid).pixels, states[i]), \
            "undo to state %d must be bit-exact" % i
    for i in range(1, len(states)):
        assert d.redo()
        assert np.array_equal(d.layer(lid).pixels, states[i]), \
            "redo to state %d must be bit-exact" % i


def test_r30_smudge_and_heal_journal_as_strokes():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 50], [200, 60]], color=(1, 0, 0), radius=10.0)
    s1 = d.layer(lid).pixels.copy()
    d.smudge(lid, [[30, 55], [180, 58]], radius=12.0, strength=0.7)
    assert d._undo[-1][1].get("rerender") == [lid], "smudge must journal"
    assert d.strokes[-1]["brush"].get("smudge")
    s2 = d.layer(lid).pixels.copy()
    assert not np.array_equal(s1, s2), "smudge must actually smear"
    d.heal(lid, [[100, 55]], radius=10.0)
    assert d._undo[-1][1].get("rerender") == [lid], "heal must journal"
    s3 = d.layer(lid).pixels.copy()
    assert d.undo() and np.array_equal(d.layer(lid).pixels, s2), \
        "heal undo must re-render exactly"
    assert d.undo() and np.array_equal(d.layer(lid).pixels, s1), \
        "smudge undo must re-render exactly"
    assert d.redo() and np.array_equal(d.layer(lid).pixels, s2)
    assert d.redo() and np.array_equal(d.layer(lid).pixels, s3)


def test_r30_pre_existing_body_no_longer_dirties():
    """P2.2: the replay base carries height/material/media, so painting
    onto pre-existing impasto keeps path-delta undo -- and restores the
    old body exactly."""
    d = _doc()
    lid = d.layers[0].id
    d.layer(lid).height_map = np.zeros((200, 320), np.float32)
    d.layer(lid).height_map[50:80, 40:200] = 1.5      # pre-existing ridge
    h0 = d.layer(lid).height_map.copy()
    d.paint(lid, [[20, 120], [200, 130]], color=(0.4, 0.3, 0.2),
            radius=9.0, media="oil", load=0.8)
    assert d._undo[-1][1].get("rerender") == [lid], \
        "pre-existing body must not force a pixel snapshot any more"
    assert d.undo()
    assert np.array_equal(d.layer(lid).height_map, h0), \
        "the pre-existing ridge must restore exactly from the base body"


def test_r30_knife_needs_no_media_special_case():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[20, 60], [220, 70]], color=(0.5, 0.4, 0.3), radius=12.0,
            media="oil", load=0.9)
    d.knife(lid, [[30, 60], [200, 68]], mode="push", radius=16.0,
            strength=0.8)
    assert d._undo[-1][1].get("rerender") == [lid], "knife must journal"
    h_after = (d.layer(lid).height_map.copy()
               if d.layer(lid).height_map is not None else None)
    p_after = d.layer(lid).pixels.copy()
    assert d.undo() and d.redo()
    assert np.array_equal(d.layer(lid).pixels, p_after)
    if h_after is not None:
        assert np.array_equal(d.layer(lid).height_map, h_after)
