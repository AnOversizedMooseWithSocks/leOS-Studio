"""tests/test_r67.py -- the .lews storage audit.

Devin: "I want you to audit the .lews and make sure that we are not storing
information inefficiently or incorrectly. leCore functionality should be
fully adopted."

Measured on the R66 painting (880x600, six layers, 51,622 strokes) the file
came to 160.3 MB, and almost none of it had to be there:

  * 74% was SEVEN ABANDONED DRAFTS. Nothing ever closes a document, so eight
    runs of a script left eight full paintings resident, and the writer saved
    every one of them.
  * Of the live document, 32.7 MB was rasters the journal already rebuilds.
    Every layer in the file says `replay_ok: true` and every one of them
    shipped its pixels anyway.
  * 137 arrays were only 64 distinct by content: 39 MB of byte-identical
    duplicates, because leStudio nests arrays inside each document section
    instead of using leCore's content-addressed `lecore.asset` sections.
  * The journal is 2.6 MB deflated -- the one part that was NOT the problem.

Journal-first, live document only: 2.8 MB. The same picture, 57x smaller.

But the size was the smaller half. `replay_ok: true` is written from an
OPTIMISTIC flag that three separate paths can leave standing when it is not
true, and a journal-first file has no pixels to fall back on when it is
wrong. Those three are pinned first, because they are what makes the size
fix safe to turn on.
"""
import io
import numpy as np


def _client(w=120, h=90):
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": w, "height": h})
    return c


def test_r67_a_lost_journal_segment_must_not_still_claim_replay_ok():
    """A journal segment that cannot be read off disk is skipped, and the
    document sets `_journal_lost`. That flag is READ NOWHERE in the tree.

    The comment on `_iter_strokes` says the faithfulness gate's pixel
    comparison catches the resulting mismatch -- and it does, on the nudge
    and warp paths, which call `replay_is_faithful()`. The SAVE path does
    not: it writes `bool(getattr(l, "_replay_ok", True))`, an optimistic
    default that only specific damage events clear. So a save can write a
    journal with a hole in it and stamp the layer replay_ok anyway. With
    pixels in the file that is merely wasteful; journal-first it is a
    layer that comes back wrong, quietly."""
    from lestudio import save_workspace, load_workspace
    from lestudio.server import WS
    c = _client()
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[10, 10 + i * 6], [110, 14 + i * 6]],
         "color": [0.8, 0.2, 0.2], "radius": 6, "opacity": 1.0}
        for i in range(10)]})
    d = WS.doc
    assert getattr(d.layer(lid), "_replay_ok", True), \
        "precondition: a batch-painted layer starts out replay-faithful"

    # a segment of the journal is gone -- a truncated spool, a full disk,
    # a file removed under us. The document notices and carries on.
    d._journal_segments = ["/nonexistent/lestudio_journal_gone/seg0.z"]
    list(d._iter_strokes())
    assert getattr(d, "_journal_lost", False), "the read must notice"

    blob = save_workspace({d.id: d}, {}, d.id)
    docs, _, active, _ = load_workspace(blob)
    lm = {l.id: l for l in docs[active].layers}
    assert not getattr(lm[lid], "_replay_ok", True), (
        "a layer whose journal has a HOLE in it must not be saved as "
        "replay_ok -- journal-first there are no pixels to fall back on")


def test_r67_the_replay_bases_fill_decision_must_survive_a_save():
    """`_replay_base_hyg` records whether the base was captured after a
    layer-wide fill, and replay consumes it to repeat the SAME decision the
    original first stroke made. It is captured, it is read twice -- and it
    is never serialized. After a save/load it is an empty dict, so replay
    runs with `_hyg_filled = False` even for a base that had the fill, and
    the saved `replay_ok: true` outlived the precondition it depended on."""
    from lestudio import save_workspace, load_workspace
    from lestudio.server import WS
    c = _client()
    d = WS.doc
    lid = d.layers[0].id
    d.paint(lid, [[5, 5], [115, 85]], color=(0.2, 0.5, 0.9), radius=9,
            opacity=1.0)
    if not hasattr(d, "_replay_base_hyg"):
        d._replay_base_hyg = {}
    d._replay_base_hyg[lid] = True

    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    assert getattr(docs[active], "_replay_base_hyg", {}).get(lid) is True, (
        "the base's fill decision must cross the save, or replay repeats a "
        "DIFFERENT decision than the one the picture was painted with")


def test_r67_a_stroke_painted_under_a_selection_keeps_its_gate():
    """`_asset_refs` -- which decides what gets written into the file --
    collects `k["brush"]["asset"]` and nothing else. A stroke painted under
    a selection references its gate through a different key, `sel_asset`,
    so the gate mask is never saved. On reload replay finds `_sm is None`,
    drops the stroke and clears `_replay_ok` -- honest at replay time, but
    the file already said `replay_ok: true`, and journal-first that layer
    has no pixels left to fall back on."""
    from lestudio import save_workspace, load_workspace
    from lestudio.server import WS
    c = _client()
    d = WS.doc
    lid = d.layers[0].id
    sid = d.select("rect", {"x0": 20, "y0": 20, "x1": 90, "y1": 60}).id
    sel = d._sel_record(sid, False)
    assert sel and sel.get("sel_asset"), "expected a recorded selection gate"
    d.paint(lid, [[5, 40], [115, 40]], color=(0.1, 0.7, 0.3), radius=10,
            opacity=1.0, selection=sid)

    key = sel["sel_asset"]
    assert key in d._asset_refs(), (
        "a selection gate is referenced by the journal and must be saved "
        "with it -- _asset_refs only looks at brush['asset']")
    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    assert docs[active]._asset_get(key) is not None, (
        "the gate mask must come back, or every stroke painted under a "
        "selection is dropped on replay")


def test_r67_the_default_save_leaves_out_what_the_journal_rebuilds():
    """The measurement that started this: on the R66 painting every layer
    said `replay_ok: true` and every one of them shipped its pixels anyway,
    because journal-first was an opt-in (`?light=1`) that nothing opted into
    -- not the save route, not autosave, not the live mirror.

    It is the default now. `?pixels=1` is the opt-out."""
    c = _client(600, 400)
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    for chunk in range(3):
        c.post("/api/paint_batch", json={"strokes": [
            {"layer": lid, "points": [[(i * 7) % 600, (i * 13) % 400, 1],
                                      [(i * 11) % 600, (i * 17) % 400, 1]],
             "color": [0.4, 0.5, 0.9], "radius": 5, "opacity": 0.6}
            for i in range(chunk * 150, chunk * 150 + 150)]})

    before = c.get("/api/composite.png").data
    fat = c.get("/api/workspace.lews?pixels=1").data
    auto = c.get("/api/workspace.lews").data
    assert len(auto) * 3 < len(fat), (
        "the DEFAULT save must leave the rebuildable rasters out: "
        "auto=%d pixels=%d" % (len(auto), len(fat)))

    import hashlib
    c.post("/api/workspace/open", data={"file": (io.BytesIO(auto), "w.lews")},
           content_type="multipart/form-data")
    after = c.get("/api/composite.png").data
    assert hashlib.sha256(before).hexdigest() == hashlib.sha256(after).hexdigest(), \
        "and it must still reopen to the identical picture"


def test_r67_a_rebuild_that_does_not_match_is_caught_on_open():
    """Leaving pixels out of the file is only safe if a WRONG rebuild is
    loud. The writer leaves a signature of the pixels it did not store --
    three numbers -- and the loader checks its own replay against it.

    Without this, a layer that comes back blank is discovered by looking at
    the picture, possibly weeks later. That is the whole risk of a
    journal-first format and it costs almost nothing to close."""
    from lestudio import save_workspace, load_workspace
    from lestudio.server import WS
    c = _client(200, 150)
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[10, 10 + i * 9], [190, 16 + i * 9]],
         "color": [0.9, 0.3, 0.1], "radius": 7, "opacity": 1.0}
        for i in range(14)]})
    d = WS.doc
    assert d._replay_provable(lid), "precondition: this layer is provable"

    blob = save_workspace({d.id: d}, {}, d.id)
    import json
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(blob))
    man = json.loads(z.read("manifest.json"))
    sec = [s for s in man["sections"] if s["id"] == d.id][0]
    lm = {L["id"]: L for L in sec["meta"]["layers"]}[lid]
    assert lm.get("pixels_cached") is False, "no pixels for a provable layer"
    assert lm.get("pixels_fp"), "but a signature of them, so the open can check"

    # now make the journal lie: the file claims pixels the strokes cannot
    # rebuild. (A real one of these is a dropped asset, a changed brush
    # default, an engine change -- all of which look exactly like this.)
    sec["meta"]["strokes"] = sec["meta"]["strokes"][:2]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for it in z.infolist():
            w.writestr(it.filename,
                       json.dumps(man).encode() if it.filename == "manifest.json"
                       else z.read(it.filename))
    docs, _, active, _ = load_workspace(out.getvalue())
    d2 = docs[active]
    assert lid in getattr(d2, "_replay_mismatch", []), \
        "a rebuild that does not match its signature must be REPORTED"
    assert not getattr(d2.layer(lid), "_replay_ok", True), \
        "...and the layer must stop claiming it replays"


def test_r67_a_document_nobody_painted_in_is_not_saved():
    """Nothing in leStudio closes a document. `/api/new` switches the active
    one and leaves the old one resident -- no cap, no LRU, no staleness --
    and the writer saved every resident document unconditionally. Eight runs
    of a painting script therefore shipped eight full paintings: measured on
    the R66 file, 74% of 160 MB was seven abandoned drafts.

    Nothing is closed behind the person's back here. The file just stops
    carrying documents nobody put a mark in, and /api/state says how many
    are resident so the app can offer to close the rest."""
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": 100, "height": 80})
    painted = WS.active
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[5, 5], [95, 75]], "color": [1, 0, 0],
         "radius": 6, "opacity": 1.0}]})
    # ...and now three `New`s nobody ever painted in, exactly as a re-run
    # of a script leaves behind
    for _ in range(3):
        c.post("/api/new", json={"width": 100, "height": 80})
    blank = WS.active
    c.post("/api/doc", json={"action": "activate", "id": painted})

    st = c.get("/api/state").get_json()
    assert st["docs_resident"] == 4, st["docs_resident"]
    assert blank in st["docs_blank"] and painted not in st["docs_blank"], \
        "the app must be able to see which ones are blanks"

    import json
    import zipfile
    blob = c.get("/api/workspace.lews").data
    man = json.loads(zipfile.ZipFile(io.BytesIO(blob)).read("manifest.json"))
    saved = [s["id"] for s in man["sections"]
             if s["kind"] == "lestudio.document"]
    assert painted in saved, "the document with work in it is always saved"
    assert blank not in saved, "a document nobody painted in is not"

    # the strictest reading of "untouched": ONE mark makes it work
    c.post("/api/doc", json={"action": "activate", "id": blank})
    bl = c.post("/api/layer", json={"action": "add", "name": "q"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": bl, "points": [[9, 9], [11, 11]], "color": [0, 0, 1],
         "radius": 3, "opacity": 1.0}]})
    c.post("/api/doc", json={"action": "activate", "id": painted})
    man = json.loads(zipfile.ZipFile(
        io.BytesIO(c.get("/api/workspace.lews").data)).read("manifest.json"))
    assert blank in [s["id"] for s in man["sections"]
                     if s["kind"] == "lestudio.document"], \
        "one mark is somebody's work and must be kept"


def test_r67_an_array_used_twice_is_stored_once():
    """leCore has carried content-addressed assets since 0.2.21: `asset_key`
    is sha256 over shape, dtype and bytes, and a `lecore.asset` section's id
    IS its content. leStudio adopted the live-collaboration half of that
    module and none of the persistence half -- it nested every array inside
    its own document section, so the same brush tip, the same pasted image,
    the same base in two documents was written once per document. Measured
    on the R66 file: 137 arrays, 64 distinct by content, 39 MB of
    byte-identical duplicates."""
    import json
    import zipfile
    from lestudio import save_workspace, load_workspace
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": 80, "height": 60})

    # two documents that genuinely share content: the same five builtin
    # brush tips, and the same painted layer
    made = []
    for n in range(2):
        if n:
            c.post("/api/new", json={"width": 80, "height": 60})
        did = WS.active
        made.append(did)
        lid = c.post("/api/layer", json={"action": "add",
                                         "name": "p"}).get_json()["id"]
        c.post("/api/paint_batch", json={"strokes": [
            {"layer": lid, "points": [[4, 4 + i * 7], [76, 8 + i * 7]],
             "color": [0.3, 0.6, 0.2], "radius": 5, "opacity": 1.0}
            for i in range(7)]})

    blob = save_workspace(WS.docs, WS.graphs, made[0])
    man = json.loads(zipfile.ZipFile(io.BytesIO(blob)).read("manifest.json"))
    kinds = [s["kind"] for s in man["sections"]]
    assert "lecore.asset" in kinds, (
        "shared arrays must be hoisted to leCore asset sections, not "
        "written once per document: %s" % sorted(set(kinds)))
    for s in man["sections"]:
        if s["kind"] == "lecore.asset":
            assert s["id"].startswith("asset:") and s["meta"]["sha256"], \
                "an asset's id IS its content address"
    docsec = [s for s in man["sections"] if s["kind"] == "lestudio.document"]
    assert any(s["meta"].get("array_refs") for s in docsec), \
        "and the documents must reference them"

    # every section says which schema it carries, so the NEXT change has
    # somewhere to hang a migration instead of sniffing at field presence
    assert all(s["meta"].get("schema") for s in docsec), \
        "document sections must be stamped with their schema version"

    # and it all still round-trips
    docs, _, active, _ = load_workspace(blob)
    assert set(docs) == set(made)
    for did in made:
        assert len(docs[did].strokes) == 7


def test_r67_a_file_written_before_the_schema_existed_still_opens():
    """The migration path is registered, not theoretical: a v1 section (no
    `schema`, no `array_refs`, pixels for every layer) is what every .lews
    written before this round looks like, and it must open unchanged."""
    import json
    import zipfile
    from lestudio import load_workspace, save_workspace
    from lestudio.server import app, WS
    c = app.test_client()
    for k in list(WS.docs)[1:]:
        WS.close(k)
    c.post("/api/doc", json={"action": "settings", "width": 90, "height": 70})
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[5, 5 + i * 8], [85, 9 + i * 8]],
         "color": [0.8, 0.4, 0.1], "radius": 6, "opacity": 1.0}
        for i in range(6)]})
    d = WS.doc
    before = np.asarray(d.composite()).copy()

    # the pre-R67 file: pixels for everything, no schema stamp, no refs
    blob = save_workspace({d.id: d}, {}, d.id, cache_pixels=True)
    z = zipfile.ZipFile(io.BytesIO(blob))
    man = json.loads(z.read("manifest.json"))
    for sec in man["sections"]:
        sec["meta"].pop("schema", None)
        sec["meta"].pop("array_refs", None)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as w:
        for it in z.infolist():
            w.writestr(it.filename,
                       json.dumps(man).encode() if it.filename == "manifest.json"
                       else z.read(it.filename))

    docs, _, active, _ = load_workspace(out.getvalue())
    after = np.asarray(docs[active].composite())
    assert np.allclose(before, after, atol=1e-6), \
        "a file written before the schema existed must open to the same picture"


def test_r67_a_locked_layer_still_rebuilds():
    """A lock guards the PERSON against editing a layer by accident. A
    replay is not an edit -- it is how a journal-first layer gets its pixels
    back at all -- and the rebuild went straight through `paint()` into the
    lock guard and died as "layer 'Background' is locked".

    Journal-first is what made this reachable: with pixels in the file the
    rebuild never ran, so the guard was never in its way."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(120, 90)
    lid = d.layers[0].id
    d.paint(lid, [(20.0, 20.0), (100.0, 70.0)], color=(1, 0, 0), radius=8)
    before = np.asarray(d.composite()).copy()
    d.edit_layer(lid, locked=True)

    docs, _, active, _ = load_workspace(save_workspace({d.id: d}, {}, d.id))
    d2 = docs[active]
    assert d2.layer(lid).locked is True, "the lock itself must survive"
    assert np.allclose(before, np.asarray(d2.composite()), atol=1e-6), \
        "a locked layer must come back with its paint on it"


def test_r67_a_heavy_layer_keeps_its_pixels():
    """Journal-first trades bytes for TIME, and the time is paid on the
    person's next Open.

    Measured on the R66 painting -- 880x600, six layers, 51,622 oil strokes
    -- the file went from 160.3 MB to 8.5 MB and the open went from 13 s to
    977 s: nineteen times smaller, seventy-five times slower. That is a good
    trade for an archive and a bad one for a document somebody is working
    in, so the saving is gated on a budget rather than taken every time.
    `?light=1` is how you ask for the small file anyway."""
    from lestudio import Document, save_workspace, REPLAY_BUDGET
    import json
    import zipfile

    def layer_arrays(blob, lid):
        man = json.loads(zipfile.ZipFile(io.BytesIO(blob)).read("manifest.json"))
        sec = [s for s in man["sections"]
               if s["kind"] == "lestudio.document"][0]
        return sec["arrays"], {L["id"]: L for L in sec["meta"]["layers"]}[lid]

    d = Document(120, 90)
    lid = d.layers[0].id
    light_n = 30
    for i in range(light_n):
        d.paint(lid, [(5.0, 5.0 + i * 2.5), (115.0, 8.0 + i * 2.5)],
                color=(0.7, 0.2, 0.5), radius=4, opacity=0.8)
    arrays, lm = layer_arrays(save_workspace({d.id: d}, {}, d.id), lid)
    assert lm.get("pixels_cached") is False, \
        "a light layer is cheap to rebuild and ships journal-first"

    # ...and the same layer once it is past the budget
    d._replay_first_set = lambda budget=None: set()
    arrays, lm = layer_arrays(save_workspace({d.id: d}, {}, d.id), lid)
    assert lm.get("pixels_cached") is not False, \
        "a layer too heavy to rebuild quickly must keep its pixels"
    assert ("layer_%s" % lid) in arrays

    # the budget is a real, documented number -- not an accident
    assert REPLAY_BUDGET > 0

    # and it is spent on the CHEAPEST layers first, so a heavy layer never
    # crowds out a light one: what the person waits for on Open is bounded
    # either way, which a per-layer budget could not promise
    d2 = Document(120, 90)
    light = d2.layers[0].id
    heavy = d2.add_layer("heavy").id
    for i in range(4):
        d2.paint(light, [(5.0, 5.0 + i * 3), (115.0, 8.0 + i * 3)],
                 color=(0.2, 0.8, 0.4), radius=4)
    for i in range(40):
        d2.paint(heavy, [(5.0, 6.0 + i * 2), (115.0, 9.0 + i * 2)],
                 color=(0.9, 0.2, 0.2), radius=4)
    chosen = d2._replay_first_set(budget=10)
    assert light in chosen and heavy not in chosen, (
        "the budget must buy the cheap layer, not be eaten by the "
        "expensive one", chosen)


def test_r67_the_rebuild_budget_is_a_dial_on_the_save_route():
    """How much rebuild time a file may cost on Open is a judgement about
    the document, not a constant. `?budget=0` is the smallest file the
    journal can produce; a large budget keeps the Open instant."""
    c = _client(300, 200)
    lid = c.post("/api/layer", json={"action": "add", "name": "p"}).get_json()["id"]
    c.post("/api/paint_batch", json={"strokes": [
        {"layer": lid, "points": [[(i * 7) % 300, (i * 13) % 200, 1],
                                  [(i * 11) % 300, (i * 17) % 200, 1]],
         "color": [0.5, 0.3, 0.8], "radius": 5, "opacity": 0.7}
        for i in range(400)]})
    small = len(c.get("/api/workspace.lews?budget=0").data)
    tight = len(c.get("/api/workspace.lews?budget=1").data)
    assert small < tight, (
        "budget=0 must take every rebuildable layer: %d vs %d"
        % (small, tight))
    # nonsense is ignored rather than obeyed
    assert len(c.get("/api/workspace.lews?budget=-5").data) == \
        len(c.get("/api/workspace.lews").data)
