"""tests/test_r28.py -- replay IS the undo history (R33 user request).

The stroke replay rebuilt only brush strokes, so a painting built from
pasted plates (figures, moon, branches) started the timelapse fully
formed at frame one. The fix per the user's design: the replay is a run
through the undo history back to the beginning, captured newest-first
and played in reverse so it progresses forward in time. Evicted undo
entries spool to disk so the history actually reaches the beginning
instead of stopping two dozen edits back.
"""
import numpy as np


def _doc():
    from lestudio import Document
    d = Document(96, 64)
    d.layers[0].pixels[...] = 0.0
    return d


def test_r28_history_replay_carries_a_paste():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[10, 30], [50, 30]], color=(1, 0, 0), radius=6.0)
    # a paste-like op: a new layer arriving with content, recorded but
    # NOT a stroke -- the stroke replay is blind to it
    arr = np.zeros((64, 96, 4), np.float32)
    arr[10:30, 60:90] = (0.0, 1.0, 0.0, 1.0)
    l2 = d.add_layer("pasted", pixels=arr)
    d.paint(lid, [[10, 50], [50, 50]], color=(0, 0, 1), radius=6.0)

    frames = list(d.history_frames(frames=64))
    assert len(frames) >= 3
    newest, oldest = frames[0], frames[-1]
    # newest-first: the first frame is the finished state (paste present)
    assert float(newest[15:25, 70:80, 1].mean()) > 0.4, \
        "the finished frame must show the pasted green"
    # the beginning must NOT contain the pasted content
    assert float(oldest[15:25, 70:80, 3].max()) < 0.05, \
        "walking to the beginning must remove the paste"
    # somewhere in between the paste appears while the blue stroke is absent
    mid_has_paste = any(float(f[15:25, 70:80, 1].mean()) > 0.4
                        and float(f[47:53, 20:40, 2].mean()) < 0.2
                        for f in frames[1:-1])
    assert mid_has_paste or len(frames) == 3, \
        "the paste must appear as a step in the history, not at frame one"


def test_r28_document_is_untouched_after_the_walk():
    d = _doc()
    lid = d.layers[0].id
    d.paint(lid, [[10, 30], [50, 30]], color=(1, 0, 0), radius=6.0)
    d.paint(lid, [[10, 50], [50, 50]], color=(0, 0, 1), radius=6.0)
    before = d.composite().copy()
    undo_len = len(d._undo)
    for _ in d.history_frames(frames=16):
        pass
    assert np.allclose(d.composite(), before, atol=1e-5), \
        "the walk must put the document back exactly"
    assert len(d._undo) == undo_len, "undo stack must be untouched"
    assert d.undo(), "interactive undo must still work after a replay"


def test_r28_spool_reaches_past_the_undo_cap():
    d = _doc()
    d.UNDO_KEEP = 3                     # tiny cap: force eviction
    lid = d.layers[0].id
    for i in range(9):
        d.paint(lid, [[5 + i * 9, 10], [5 + i * 9, 50]],
                color=(1, 1, 1), radius=3.0)
    assert len(d._undo) == 3
    assert d.history_len() >= 9, \
        "evicted entries must be spooled, not lost (%d)" % d.history_len()
    frames = list(d.history_frames(frames=64))
    oldest = frames[-1]
    assert float(oldest[..., 3].max()) < 0.05, \
        "with the spool the history must reach the blank beginning"


def test_r28_render_reversal_progresses_forward():
    """The server reverses the captured frames; pinned at the model level:
    reversing history_frames yields monotonically non-decreasing painted
    area for an append-only painting session."""
    d = _doc()
    lid = d.layers[0].id
    for i in range(5):
        d.paint(lid, [[8 + i * 16, 8], [8 + i * 16, 56]],
                color=(1, 1, 1), radius=4.0)
    seq = list(d.history_frames(frames=32))[::-1]     # forward in time
    areas = [float((f[..., 3] > 0.1).sum()) for f in seq]
    assert all(b >= a - 1 for a, b in zip(areas, areas[1:])), \
        "played forward, paint must accumulate: %s" % areas


def test_r28_undo_does_not_recopy_untouched_data():
    """R33 user diagnosis: undo was bloated because every record() copied
    every mask, selection, brush tip and stamp afresh -- a small stroke's
    entry measured ~21 MB of which the painted window was 0.13 MB.
    Snapshots now intern those arrays: unchanged content is SHARED by
    reference across entries, so N strokes retain one mask copy, not N."""
    from lestudio import Document, Mask
    d = Document(320, 200)
    lid = d.layers[0].id
    d.masks.append(Mask(200, 320, "m0",
                        np.random.rand(200, 320).astype(np.float32)))
    mask_nb = d.masks[0].data.nbytes
    for i in range(10):
        d.paint(lid, [[10 + i * 28, 40], [30 + i * 28, 60]],
                color=(1, 0, 0), radius=5.0)
    a = d._undo[-1][1]["masks"][0][2]
    b = d._undo[-2][1]["masks"][0][2]
    assert a is b, "consecutive snapshots must SHARE the unchanged mask"
    # the budget counts the shared copy once: ten entries must cost far
    # less than ten mask copies
    assert d._undo_bytes() < mask_nb * 3 + d.layers[0].pixels.nbytes * 2, \
        "undo bytes still duplicating untouched data: %d" % d._undo_bytes()


def test_r28_interned_undo_is_still_exact():
    """Sharing must never trade correctness: an in-place mask edit earns a
    fresh copy, and undo restores each stage exactly."""
    from lestudio import Document, Mask
    d = Document(160, 120)
    lid = d.layers[0].id
    d.masks.append(Mask(120, 160, "m0",
                        np.random.rand(120, 160).astype(np.float32)))
    m0 = d.masks[0].data.copy()
    p0 = d.layer(lid).pixels.copy()
    d.paint(lid, [[20, 30], [80, 30]], color=(0, 1, 0), radius=6.0)
    snap_before_edit = d._undo[-1][1]["masks"][0][2]
    d.record("Mask paint", only=[])
    d.masks[0].data[:40, :40] = 0.25
    d.paint(lid, [[20, 80], [80, 80]], color=(0, 0, 1), radius=6.0)
    snap_after_edit = d._undo[-1][1]["masks"][0][2]
    assert snap_before_edit is not snap_after_edit, \
        "an edited mask must get a fresh interned copy"
    assert float(np.abs(snap_after_edit[:40, :40] - 0.25).max()) < 1e-6
    d.undo()          # blue stroke
    d.undo()          # mask edit
    d.undo()          # green stroke
    assert np.array_equal(d.masks[0].data, m0), "mask must restore exactly"
    assert np.array_equal(d.layer(lid).pixels, p0), \
        "pixels must restore exactly through interned snapshots"


def test_r28_deleted_mask_does_not_pin_its_intern():
    from lestudio import Document, Mask
    d = Document(160, 120)
    lid = d.layers[0].id
    d.masks.append(Mask(120, 160, "m0",
                        np.random.rand(120, 160).astype(np.float32)))
    d.paint(lid, [[20, 30], [80, 30]], color=(1, 0, 0), radius=6.0)
    assert ("mask", d.masks[0].id) in d._snap_intern
    d.masks.clear()
    d.paint(lid, [[20, 60], [80, 60]], color=(1, 0, 0), radius=6.0)
    assert all(k[0] != "mask" for k in d._snap_intern), \
        "pruning must drop interns for deleted objects"


def test_r28_undo_does_not_push_full_snapshots_for_redo():
    """Eleven undos OOM-killed the live studio: each pushed a FULL
    document snapshot (~300 MB on a many-layer canvas) onto _redo, which
    has no budget. The counter-snapshot now carries the same (only,
    region) scope as the entry it reverses."""
    from lestudio import Document
    d = Document(640, 400)
    for i in range(5):
        d.add_layer("l%d" % i)
    lid = d.layers[0].id
    full_doc = sum(l.pixels.nbytes for l in d.layers)
    for i in range(6):
        d.paint(lid, [[20 + i * 30, 40], [40 + i * 30, 60]],
                color=(1, 0, 0), radius=5.0)
    px_before = d.layer(lid).pixels.copy()
    for _ in range(6):
        assert d.undo()
    redo_px = 0
    for _, sn, _a in d._redo:
        for rec in sn.get("layers", ()):
            px = rec[7]
            if px is None:
                continue
            redo_px += (px[1].nbytes if isinstance(px, tuple) else px.nbytes)
    assert redo_px < full_doc, \
        "redo retained %.1f MB of pixels for six small strokes" % (
            redo_px / 1e6)
    for _ in range(6):
        assert d.redo()
    assert np.array_equal(d.layer(lid).pixels, px_before), \
        "scoped redo must restore exactly"
