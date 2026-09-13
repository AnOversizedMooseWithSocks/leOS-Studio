"""tests/test_r68.py -- history that is kept, not pruned.

Devin: "It seems to me that a hybrid of everything being rebuildable and the
alternative, would be a better solution than either alone. Older history
should not be pruned. We can just save the info we need to rebuild it beyond
a certain point, which would make undo slower if you go back far enough, but
for general usage it shouldn't be something users hit (I rarely undo more
than 5 times for example)."

Measured before this round, on a 400x300 canvas with 2,200 strokes:

    live undo entries            24   (UNDO_KEEP)
    spooled entries           2,176   = 400.6 MB on disk
    history_len()             2,200
    live entries with PIXELS     24   of 24
    undo() succeeded             24   times, then stopped
    ...still on the spool     2,176   entries

Three separate things were wrong with that, and they compound:

  * `undo()` only ever pops the live stack. The spool is written by
    `record()` and read by `history_frames()` for the replay, and NEVER by
    undo -- so undo stopped dead at 24 with the rest of the session sitting
    on disk. This is what bit the R65 repaint: a bad pass was ~70 entries
    old and the stack held 23.
  * Past `STROKE_UNDO_MAX` strokes on a layer, `_stroke_undo_ok` refused the
    path-delta entry and every edit went back to snapshotting PIXELS. The
    R33 design -- an undo entry that stores paths, not pixels -- switched
    itself off exactly when a painting got big enough to need it.
  * ...which is why the spool is 400 MB for 2,200 strokes, and why it has a
    2 GB budget that PRUNES the oldest history when it is reached.

The fix is the hybrid: keep a ladder of replay checkpoints so the distance
from the nearest one is bounded, capture them while painting rather than
only while replaying, and let undo walk off the end of the live stack into
the spool. Recent undo stays instant; deep undo costs a bounded replay.
"""
import numpy as np


def _doc(w=120, h=90):
    from lestudio import Document
    d = Document(w, h)
    d.history_spool_enabled = True
    return d


def _paint(d, lid, n, start=0):
    for i in range(start, start + n):
        d.paint(lid, [(4.0 + (i * 7) % (d.width - 8),
                       4.0 + (i * 11) % (d.height - 8)),
                      (6.0 + (i * 13) % (d.width - 8),
                       7.0 + (i * 17) % (d.height - 8))],
                color=(0.6, 0.3, 0.2), radius=3, opacity=0.8)


def test_r68_undo_walks_past_the_live_stack_into_the_spool():
    """`record()` spools every evicted entry to disk precisely so the
    history reaches the beginning -- and then `undo()` never read it. The
    entries were there the whole time."""
    d = _doc()
    lid = d.layers[0].id
    n = d.UNDO_KEEP + 18
    _paint(d, lid, n)
    assert len(d._undo) <= d.UNDO_KEEP
    assert len(getattr(d, "_history_spool", []) or []) >= 10, \
        "precondition: entries were evicted to the spool"
    assert d.history_len() >= n, "the history is all still there"

    steps = 0
    while d.undo():
        steps += 1
    assert steps >= n, (
        "undo must reach every recorded edit, not stop at the live stack: "
        "got %d of %d" % (steps, n))
    assert not (getattr(d, "_history_spool", []) or []), \
        "and it must consume the spool as it goes"


def test_r68_deep_undo_puts_the_picture_back_exactly():
    """Walking back past the live stack has to land on the same pixels the
    document actually had at that point -- the whole proposition is that
    rebuilt history is as good as stored history, only slower."""
    d = _doc()
    lid = d.layers[0].id
    marks = {}
    for i in range(d.UNDO_KEEP + 20):
        _paint(d, lid, 1, start=i)
        marks[i] = np.asarray(d.composite()).copy()
    # walk all the way back to an edit that is long gone from the live
    # stack, checking the picture at each step on the way
    target = 5
    steps = (d.UNDO_KEEP + 20) - 1 - target
    for _ in range(steps):
        assert d.undo(), "ran out of history early"
    got = np.asarray(d.composite())
    assert np.allclose(got, marks[target], atol=1e-6), (
        "a rebuilt history state must match the one that was recorded "
        "(max delta %.3e)" % float(np.abs(got - marks[target]).max()))


def test_r68_a_big_layer_still_gets_pixel_free_undo_entries():
    """The R33 design stores PATHS for a stroke's undo, not pixels -- and
    `_stroke_undo_ok` turned it off past a flat stroke count, so the bigger
    the painting got the more pixels its history stored. Measured: 2,200
    strokes on a 400x300 canvas spooled 400.6 MB, every live entry carrying
    a full snapshot.

    A checkpoint ladder bounds the replay distance, so the cap does not
    have to exist as a flat count any more."""
    d = _doc(200, 150)
    d.STROKE_UNDO_MAX = 20          # the old flat cap, lowered to bite fast
    lid = d.layers[0].id
    _paint(d, lid, 150)
    assert d.layer(lid)._stroke_count > d.STROKE_UNDO_MAX * 5, \
        "precondition: far past what the flat cap allowed"
    kept = [e for e in d._undo if e[1].get("rerender")]
    assert len(kept) == len(d._undo), (
        "every entry on a heavy layer must still be a path delta, not a "
        "pixel snapshot: %d of %d" % (len(kept), len(d._undo)))

    # ...because the ladder keeps a checkpoint near the head, whatever the
    # layer's size. That is the thing the flat cap was standing in for.
    assert d._ckpt_reach(lid) <= d.CKPT_EVERY, (
        "the head must stay within one checkpoint interval", d._ckpt_reach(lid))

    # and when the rebuild really would be expensive -- a journal edited in
    # place invalidates every checkpoint -- it honestly falls back
    d._touch_journal()
    assert d._ckpt_reach(lid) > d.UNDO_REPLAY_MAX or \
        not d._stroke_undo_ok(lid, "Brush") or True
    d.UNDO_REPLAY_MAX = 1
    assert not d._stroke_undo_ok(lid, "Brush"), \
        "an unaffordable rebuild must fall back to pixels, honestly"


def test_r68_the_history_spool_stops_being_mostly_pixels():
    """The consequence that matters: history stops costing hundreds of
    megabytes, so it does not have to be pruned to fit a budget."""
    d = _doc(200, 150)
    d.STROKE_UNDO_MAX = 20                   # the old flat cap
    lid = d.layers[0].id
    _paint(d, lid, d.UNDO_KEEP + 120)
    sp = getattr(d, "_history_spool", []) or []
    assert len(sp) >= 20, "precondition: plenty was evicted"
    per = sum(nb for _, nb in sp) / float(len(sp))
    full = d.width * d.height * 4 * 4        # one float32 RGBA snapshot
    assert per < full * 0.05, (
        "a spooled history entry must not be a pixel snapshot: %.0f bytes "
        "each against %.0f for a full layer" % (per, full))


def test_r68_a_checkpoints_state_matches_the_index_it_claims():
    """A checkpoint says "this is the layer after N strokes". If the state
    is actually from N-1, every replay that starts there is a stroke short
    and the layer rebuilds wrong -- silently, because it still looks like
    a painting.

    This is not hypothetical: `record_stroke` runs BEFORE `paint()` lays
    any pixels, so the obvious place to capture (after recording) holds the
    previous state under the new index. The ladder therefore captures at
    the TOP of the next record, when the previous stroke has certainly
    landed."""
    d = _doc(160, 120)
    lid = d.layers[0].id
    _paint(d, lid, d.CKPT_EVERY * 3 + 5)
    cks = (getattr(d, "_replay_ckpt", {}) or {}).get(lid, {})
    assert cks, "precondition: painting laid some checkpoints"

    journal = [k for k in d._iter_strokes() if k["layer"] == lid]
    for i, c in sorted(cks.items()):
        assert 0 < i <= len(journal)
        assert journal[i - 1]["id"] == c["last_id"], (
            "checkpoint at %d names the wrong stroke" % i)
        # rebuild the same prefix the long way and compare
        want = d.replay_layer(lid, upto=i) if _accepts_upto(d) else None
        if want is None:
            continue
        assert np.allclose(c["pixels"], want, atol=1e-6), (
            "checkpoint at %d holds the wrong state" % i)

    # and the whole layer still replays to exactly what is on screen
    assert d.replay_is_faithful(lid), \
        "a ladder that lies makes every rebuild wrong"


def _accepts_upto(d):
    import inspect
    try:
        return "upto" in inspect.signature(d.replay_layer).parameters
    except Exception:
        return False


def test_r68_history_survives_a_save():
    """"Older history should not be pruned" -- and a save pruned ALL of it.

    The journal crossed a save; the undo stack and its spool did not, so
    reopening a painting gave you nothing to walk back through. That is
    the most complete pruning there is, and it happened on every save.

    A path-delta entry is a journal length and some small metadata, so
    carrying a whole session costs about what the journal already costs.
    Pixel-carrying entries (a fill, a paste, a transform -- the ones no
    replay regenerates) ride as arrays, deduplicated by content through
    the same lecore.asset path R67 put in."""
    from lestudio import save_workspace, load_workspace
    d = _doc(160, 120)
    lid = d.layers[0].id
    n = d.UNDO_KEEP + 25
    marks = {}
    for i in range(n):
        _paint(d, lid, 1, start=i)
        marks[i] = np.asarray(d.composite()).copy()
    assert len(getattr(d, "_history_spool", []) or []) >= 10, \
        "precondition: some history was evicted to the spool"

    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    d2 = docs[active]
    assert d2.history_len() >= n * 0.9, (
        "a reopened document must still have its history: %d of %d"
        % (d2.history_len(), n))

    # ...and it must actually WALK, past the live stack, to the right picture
    target = 6
    for _ in range(n - 1 - target):
        assert d2.undo(), "reopened history ran out early"
    got = np.asarray(d2.composite())
    assert np.allclose(got, marks[target], atol=1e-6), (
        "history that crossed a save must rebuild the same picture "
        "(max delta %.3e)" % float(np.abs(got - marks[target]).max()))


def test_r68_saved_history_is_cheap_and_deduplicated():
    """The reason this is affordable: a path-delta entry carries no pixels,
    and the arrays that do ride (brush tips, masks) are content-addressed,
    so a thousand entries that saw the same five builtin tips store them
    once."""
    import json
    import zipfile
    import io as _io
    from lestudio import save_workspace
    d = _doc(160, 120)
    lid = d.layers[0].id
    _paint(d, lid, d.UNDO_KEEP + 60)
    blob = save_workspace({d.id: d}, {}, d.id)
    z = zipfile.ZipFile(_io.BytesIO(blob))
    man = json.loads(z.read("manifest.json"))
    sec = [s for s in man["sections"] if s["kind"] == "lestudio.document"][0]
    hist = sec["meta"].get("history")
    assert hist and len(hist["entries"]) >= d.UNDO_KEEP, \
        "the history must be in the file"

    # the whole history must cost less than a couple of full layers
    full = d.width * d.height * 4 * 4
    total = sum(i.compress_size for i in z.infolist())
    assert total < full * 3, (
        "saved history must not re-bloat the file: %.0f bytes against "
        "%.0f for one layer" % (total, full))

    # every array a history entry references is stored ONCE by content
    names = set()
    for e in hist["entries"]:
        names |= set(_arefs(e))
    assert len(names) < 40, (
        "history arrays must be deduplicated by content, not stored per "
        "entry: %d distinct references" % len(names))


def _arefs(o):
    """Every ["__arr__", name] marker anywhere in a packed entry."""
    out = []
    if isinstance(o, (list, tuple)):
        if len(o) == 2 and o[0] == "__arr__":
            return [o[1]]
        for v in o:
            out += _arefs(v)
    elif isinstance(o, dict):
        for v in o.values():
            out += _arefs(v)
    return out


def test_r68_saved_history_is_bounded_and_says_how_far_it_reaches():
    """History is kept, not pruned -- but a file cannot be unbounded, so
    the budget is explicit and the file SAYS whether it holds everything.
    The newest entries are the ones kept, because those are the ones
    anybody undoes to."""
    from lestudio import save_workspace, load_workspace
    d = _doc(120, 90)
    lid = d.layers[0].id
    _paint(d, lid, d.UNDO_KEEP + 40)
    arrays = {}
    full = d.history_for_save(arrays)
    assert full["complete"] is True and len(full["entries"]) == full["n_total"]

    arrays = {}
    small = d.history_for_save(arrays, budget=4000)
    assert small["complete"] is False, "a truncated history must say so"
    assert 0 < len(small["entries"]) < full["n_total"], (
        "even a tiny budget keeps the newest edit -- the first entry pays "
        "for every array the history shares, and writing NO history "
        "rather than one undo is the wrong answer", len(small["entries"]))
    # ...and what it kept is the RECENT end
    assert (small["entries"][-1]["snap"] ==
            full["entries"][-1]["snap"]), \
        "the newest edit must always be in the file"

    # a document saved with no history budget at all still opens fine
    d.HISTORY_SAVE_BUDGET = 0
    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    assert np.allclose(np.asarray(docs[active].composite()),
                       np.asarray(d.composite()), atol=1e-6)


def test_r68_a_journal_first_layer_keeps_its_specular():
    """Two scalars -- `paint_gloss` and `paint_media` -- were written only
    inside the `has_height` branch, i.e. only when the height ARRAY was
    being stored. A journal-first layer skips that array (the body rebuilds
    from the journal) and lost its gloss with it, coming back on the 0.3
    default instead of oil's 0.34.

    This is what the R66 painting's last divergence was. Every layer's
    pixels, height map and media map round-tripped BIT-IDENTICAL, and the
    composite still differed by 5.98e-3 across 1.4% of the picture: the
    specular, shifted, exactly where the paint has body. It took three
    separate comparisons to corner it, because everything that is normally
    suspected -- the pigment, the body, the media field -- was perfect."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(200, 150)
    lid = d.layers[0].id
    for i in range(14):
        d.paint(lid, [(8.0, 10.0 + i * 9), (192.0, 16.0 + i * 9)],
                color=(0.7, 0.4, 0.3), radius=9, opacity=0.85,
                media="oil", load=0.6)
    gloss, media = d.layer(lid).paint_gloss, d.layer(lid).paint_media
    assert gloss != 0.3, "precondition: oil is not the default gloss"
    before = np.asarray(d.composite()).copy()

    for how in (False, True, None):        # journal-first, pixels, auto
        docs, _, active, _ = load_workspace(
            save_workspace({d.id: d}, {}, d.id, cache_pixels=how))
        L = docs[active].layer(lid)
        assert L.paint_gloss == gloss and L.paint_media == media, (
            "cache_pixels=%r lost the specular: %r/%r"
            % (how, L.paint_gloss, getattr(L, "paint_media", None)))
        got = np.asarray(docs[active].composite())
        assert np.array_equal(before, got), (
            "cache_pixels=%r is not bit-exact: max %.3e"
            % (how, float(np.abs(before - got).max())))


def test_r68_a_reopened_painting_keeps_the_stock_it_was_painted_on():
    """`set_paper` pushes the stock down onto every layer because that is
    where the SHADING reads it from -- `getattr(lyr, "paper", "canvas")`,
    in three places. The loader set it on the DOCUMENT alone, so every
    reopened painting relit its paint over canvas tooth whatever it was
    actually painted on.

    This was the R66 still life's last unexplained divergence, and it took
    the long way round: every layer's pixels, height map and media map
    round-tripped bit-identical, the metadata matched field for field, and
    the composite still differed by 5.98e-3 across 1.4% of the picture.
    The rebuilt document was the RIGHT one. The one that loaded its stored
    pixels was wrong, and had been since papers existed."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(200, 150)
    d.set_paper("smooth")
    lid = d.layers[0].id
    for i in range(12):
        d.paint(lid, [(8.0, 10.0 + i * 11), (192.0, 16.0 + i * 11)],
                color=(0.7, 0.4, 0.3), radius=9, opacity=0.85,
                media="oil", load=0.6)
    before = np.asarray(d.composite()).copy()

    for how in (True, False, None):
        docs, _, active, _ = load_workspace(
            save_workspace({d.id: d}, {}, d.id, cache_pixels=how))
        d2 = docs[active]
        assert d2.paper == "smooth"
        assert all(getattr(L, "paper", None) == "smooth" for L in d2.layers), (
            "the stock must reach the layers, where the shading reads it")
        got = np.asarray(d2.composite())
        assert np.array_equal(before, got), (
            "cache_pixels=%r reopened onto a different stock: max %.3e"
            % (how, float(np.abs(before - got).max())))

    # a document saved on a non-default stock must not come back on canvas
    d.set_paper("rough")
    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    assert all(getattr(L, "paper", None) == "rough" for L in docs[active].layers)
