"""tests/test_r31.py -- determinism backlog, second sweep (R33/R34).

P1.3: imported pixels are the one thing replay cannot generate. They now
live ONCE in a content-addressed asset store; paste, import and placement
journal pixel-free {op, asset} records, so pasted/placed layers replay
from an empty base and stay stroke-editable, and the .lews carries each
asset exactly once.

P2.4: the journal is never trimmed. MAX_STROKES head-cuts spool to disk
and _iter_strokes serves the complete record, so replay stays exact at
any session length instead of silently dirtying old layers.

P1.7: media_step is a journaled op; the fluid's sim state (density, dye,
velocity, injection counter) is DERIVED from the journal and rebuilds
bit-exactly through undo/redo.

P0.4: wall-clock never steers the render path -- pinned by AST scan.
"""
import numpy as np


def _clip(w=50, h=40, seed=7):
    rng = np.random.RandomState(seed)
    return {"pixels": rng.rand(h, w, 4).astype(np.float32),
            "x": 20, "y": 30}


def test_r31_paste_is_a_journaled_asset_op():
    from lestudio import Document
    d = Document(160, 120)
    l1 = d.paste(_clip())
    l2 = d.paste(_clip(), x=60, y=10)
    # content addressing: the same clip pasted twice stores ONE asset
    assert len(d._assets) == 1, "identical pastes must share one asset"
    # the paste is a pixel-free journal record, not a snapshot
    ks = [k for k in d.strokes if k["layer"] in (l1.id, l2.id)]
    assert len(ks) == 2 and all(k["brush"].get("op") == "paste"
                                and k["brush"].get("asset") for k in ks)
    assert d.replay_is_faithful(l1.id) and d.replay_is_faithful(l2.id), \
        "a pasted layer must be stroke-editable from birth"
    before = d.layer(l2.id).pixels.copy()
    assert d.undo() and len(d.layers) == 2, "undo must remove the paste"
    assert d.redo() and len(d.layers) == 3
    assert np.array_equal(d.layer(l2.id).pixels, before), \
        "redo must rebuild the paste from base + journal, bit-exact"


def test_r31_import_and_place_journal_parametrically():
    from lestudio import Document
    d = Document(160, 120)
    arr = np.random.RandomState(3).rand(200, 260, 4).astype(np.float32)
    li = d.add_layer("plate", pixels=arr, placed=True)
    assert d.replay_is_faithful(li.id), "an import must journal, not dirty"
    fitted = d.layer(li.id).pixels.copy()
    d.place_source(li.id, x=40, y=60, scale=0.5, rot=15)
    assert d.replay_is_faithful(li.id), \
        "a placement is parametric given the asset -- it must journal"
    placed = d.layer(li.id).pixels.copy()
    assert d.undo(), "place must be undoable"
    assert np.array_equal(d.layer(li.id).pixels, fitted), \
        "undoing the placement must restore the fitted raster exactly"
    assert d.redo()
    assert np.array_equal(d.layer(li.id).pixels, placed), \
        "redoing the placement must re-rasterise exactly"


def test_r31_assets_ride_the_lews_once():
    from lestudio import Document, save_workspace, load_workspace
    d = Document(160, 120)
    l1 = d.paste(_clip())
    l2 = d.paste(_clip(), x=60, y=10)          # same content, same asset
    arr = np.random.RandomState(3).rand(200, 260, 4).astype(np.float32)
    li = d.add_layer("plate", pixels=arr, placed=True)
    d.place_source(li.id, x=40, y=60, scale=0.5, rot=15)
    blob = save_workspace({d.id: d}, {}, d.id)
    docs, graphs, active, extras = load_workspace(blob)
    d2 = docs[active]
    assert len(d2._assets) == 2, "the file must carry each asset once"
    for L in (l1.id, l2.id, li.id):
        assert d2.replay_is_faithful(L), \
            "%s must reopen replayable (asset + journal restored)" % L
        assert np.allclose(d2.layer(L).pixels, d.layer(L).pixels,
                           atol=1e-6)


def test_r31_journal_never_trims():
    from lestudio import Document
    d = Document(96, 64)
    d.layers[0].pixels[...] = 0.0
    d.MAX_STROKES = 16
    lid = d.layers[0].id
    for i in range(40):
        d.paint(lid, [[3 + i * 2, 8], [3 + i * 2, 56]],
                color=(1, 1, 1), radius=2.0)
    assert len(d.strokes) == 16, "the live list stays capped"
    assert d._journal_spooled.get(lid) == 24, \
        "head-cut strokes must spool, not vanish"
    assert len(list(d._iter_strokes(lid))) == 40, \
        "_iter_strokes must serve the COMPLETE journal"
    assert d.replay_is_faithful(lid), \
        "replay must stay exact past the cap (this used to dirty the layer)"
    rep = d.replay_layer(lid)
    assert np.allclose(rep, d.layer(lid).pixels, atol=1e-6)
    # and the timelapse reaches all the way back
    frames = list(d.timelapse_frames(frames=8))
    first_area = float((frames[0][..., 3] > 0.1).sum())
    last_area = float((frames[-1][..., 3] > 0.1).sum())
    assert first_area < last_area * 0.5, \
        "playback must start near the beginning, not mid-painting"


def test_r31_media_step_is_a_journaled_op():
    from lestudio import Document
    d = Document(128, 96)
    l = d.add_layer("plume")
    d.edit_layer(l.id, vol_kind="smoke", thickness=6.0)
    d.paint(l.id, [[40, 60], [44, 62]], color=(0.9, 0.9, 0.9),
            radius=12.0, opacity=0.9, hardness=0.4)
    d.media_step(l.id, 3)
    d.paint(l.id, [[70, 60], [74, 64]], color=(0.7, 0.7, 0.9), radius=10.0)
    d.media_step(l.id, 2)
    assert d.replay_is_faithful(l.id), \
        "the sim state is derived from the journal -- steps must not dirty"
    px = l.pixels.copy()
    den = l._media["den"].copy()
    assert d.undo() and d.undo() and d.redo() and d.redo()
    assert np.array_equal(l.pixels, px), \
        "media undo/redo must be bit-exact through the journal"
    assert l._media is not None and np.array_equal(l._media["den"], den), \
        "the SIM STATE must rebuild too, or the next step continues a lie"


def test_r31_wall_clock_stays_out_of_the_render_path():
    """DETERMINISM_BACKLOG P0.4: time.time() in the engine is timings-only.
    Pin it: no wall-clock or datetime read inside any function on the
    render/replay path, so a replay can never depend on when it runs."""
    import ast
    import inspect
    import lestudio
    src = inspect.getsource(lestudio)
    tree = ast.parse(src)
    render_path = {
        "composite", "paint", "smudge", "heal", "knife", "blend_stroke",
        "flood_fill", "fill_layer", "clear", "flip_layer", "paste",
        "place_source", "media_step", "replay_layer", "replay_region",
        "_replay_apply", "_blit_asset", "_resolve_fill_spec",
        "_media_slab_step", "_media_inject", "_media_state",
        "_stroke_rerender", "_capture_replay_base", "record_stroke",
    }
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
                           if f in render_path), None)
                if fn:
                    bad.append((fn, node.lineno))
            self.generic_visit(node)

    V().visit(tree)
    assert not bad, "wall-clock read inside the render path: %s" % bad


def test_r31_evicted_asset_degrades_honestly():
    from lestudio import Document
    d = Document(120, 90)
    l = d.paste(_clip(seed=11))
    key = next(iter(d._assets))
    del d._assets[key]                       # simulate budget eviction
    assert not d.replay_is_faithful(l.id), \
        "a lost asset must mark the layer dirty, never replay it wrong"


def test_r31_reloaded_document_never_reissues_ids():
    """Found live: the constructor's background layer primed the per-doc
    id sequence at 1 before the loader replaced the layer list, so the
    first paste into a reopened workspace minted a SECOND "L2" -- which
    resolved to the original layer, inherited its strokes, and made undo
    and place act on the wrong object."""
    from lestudio import Document, save_workspace, load_workspace
    d = Document(96, 64)
    for i in range(4):
        d.add_layer("l%d" % i)
    ids = {l.id for l in d.layers}
    blob = save_workspace({d.id: d}, {}, d.id)
    docs, graphs, active, extras = load_workspace(blob)
    d2 = docs[active]
    clip = {"pixels": np.random.rand(20, 20, 4).astype(np.float32),
            "x": 5, "y": 5}
    nl = d2.paste(clip)
    assert nl.id not in ids, \
        "a reopened document reissued id %r" % nl.id
    assert len([l for l in d2.layers if l.id == nl.id]) == 1
