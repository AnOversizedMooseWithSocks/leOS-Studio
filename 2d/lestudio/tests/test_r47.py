"""tests/test_r47.py -- generator brushes + perspective warp.

R47 adds three drawing tools grown out of the R46 architectural round:

* scribble    -- curl-noise scribble brush. It is a STROKE GENERATOR:
                 randomness is spent at generation time and each strand
                 lands in the journal as an ordinary paint stroke, so
                 replay/undo/nudge need zero new machinery.
* hatch_fill  -- uniform line/hatch shading brush (modes line | hatch |
                 both, where 'both' reads the composite and crosses only
                 the darks). Same generator contract.
* warp_perspective -- 4-point perspective warp of a region. This one IS
                 a new op: pixel-free {op:'pwarp'} journal record (bbox
                 + quad + frozen gate asset), one shared applier for the
                 tool and for replay.

All three respect selections/masks WITH feathering: a soft gate thins
and fades the generators instead of shearing them, and pwarp carries
only the gate's share of each pixel.
"""
import numpy as np


def _mk(w=200, h=150):
    from lestudio import Document
    d = Document(w, h)
    return d


def test_r47_scribble_is_journaled_strokes_and_replays():
    d = _mk()
    L = d.add_layer("scr").id
    n0 = len(d.strokes)
    made = d.scribble(L, 100, 75, radius=50, curl=0.7, thickness=1.5,
                      color=(0.1, 0.1, 0.1), seed=7)
    assert made >= 2 and len(d.strokes) - n0 == made, \
        "every strand must be one ordinary journaled stroke"
    assert all("op" not in d.strokes[i]["brush"]
               for i in range(n0, len(d.strokes))), \
        "scribble strands are plain paint strokes, not special ops"
    assert d.replay_is_faithful(L), \
        "a generated scribble must replay from the journal alone"


def test_r47_scribble_determinism_by_seed():
    a, b = _mk(), _mk()
    La, Lb = a.add_layer("s").id, b.add_layer("s").id
    a.scribble(La, 90, 70, radius=45, curl=0.6, seed=11)
    b.scribble(Lb, 90, 70, radius=45, curl=0.6, seed=11)
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels), \
        "same seed, same scribble -- generation must be deterministic"


def test_r47_feathered_selection_shapes_and_fades_scribble():
    d = _mk()
    L = d.add_layer("scr").id
    sel = d.select("ellipse", {"x0": 40, "y0": 30, "x1": 160, "y1": 120},
                   feather=12)
    made = d.scribble(L, 100, 75, radius=70, curl=0.5, seed=9,
                      selection=sel.id)
    assert made > 0
    a = d.layer(L).pixels[..., 3]
    assert float(a[0:12, 0:12].max()) == 0.0, \
        "strands must not escape the gate (corner is far outside)"
    assert d.replay_is_faithful(L)


def test_r47_hatch_modes_line_hatch_both():
    d = _mk()
    L = d.layers[0].id
    # gradient composite so 'both' has darks and lights to read
    g = np.linspace(0, 1, d.width, dtype=np.float32)[None, :]
    d.layers[0].pixels[..., :3] = np.repeat(g, d.height, 0)[..., None]
    d.layers[0].pixels[..., 3] = 1.0
    Lh = d.add_layer("hatch").id
    n_line = d.hatch_fill(Lh, 100, 75, radius=60, mode="line", seed=3)
    d.undo()
    n_hatch = d.hatch_fill(Lh, 100, 75, radius=60, mode="hatch", seed=3)
    d.undo()
    n_both = d.hatch_fill(Lh, 100, 75, radius=60, mode="both", seed=3)
    assert n_line > 0 and n_hatch > n_line, \
        "'hatch' adds a cross pass everywhere, so it must out-stroke 'line'"
    assert n_line < n_both <= n_hatch, \
        "'both' crosses only the darks: between 'line' and 'hatch'"
    assert d.replay_is_faithful(Lh)


def test_r47_hatch_in_feathered_selection_area():
    d = _mk()
    L = d.add_layer("h").id
    sel = d.select("ellipse", {"x0": 40, "y0": 30, "x1": 160, "y1": 120},
                   feather=12)
    made = d.hatch_fill(L, area="selection", mode="hatch", spacing=6,
                        selection=sel.id, seed=4)
    assert made > 0
    a = d.layer(L).pixels[..., 3]
    assert float(a[0:12, 0:12].max()) == 0.0, \
        "hatching must stay inside the gate"
    assert d.replay_is_faithful(L)


def test_r47_pwarp_is_a_pixel_free_journaled_op():
    d = _mk()
    L = d.add_layer("box").id
    d.paint(L, [[60, 50, 1], [110, 50, 1], [110, 90, 1], [60, 90, 1],
                [60, 50, 1]], color=(0.1, 0.1, 0.1), radius=3)
    assert d.warp_perspective(L, [60, 30, 140, 45, 130, 110, 55, 100])
    ent = d._undo[-1]
    assert ent[1].get("rerender") == [L], \
        "pwarp undo must carry the rerender tag like xform does"
    assert all(r[7] is None for r in ent[1].get("layers", [])), \
        "pwarp must journal, not snapshot the document"
    b = d.strokes[-1]["brush"]
    assert b.get("op") == "pwarp" and len(b.get("quad", [])) == 8, \
        "the journal record is the parametric warp, no pixels"
    assert d.replay_is_faithful(L), \
        "replaying the pwarp op must reproduce the warped layer"


def test_r47_pwarp_moves_content_to_quad_and_survives_undo_redo():
    d = _mk()
    L = d.add_layer("box").id
    d.paint(L, [[60, 50, 1], [110, 50, 1], [110, 90, 1], [60, 90, 1],
                [60, 50, 1]], color=(0.1, 0.1, 0.1), radius=3)
    quad = [60, 30, 140, 45, 130, 110, 55, 100]
    d.warp_perspective(L, quad)
    a = d.layer(L).pixels[..., 3]
    ys, xs = np.where(a > 0.1)
    qx = quad[0::2]
    qy = quad[1::2]
    assert xs.min() >= min(qx) - 6 and xs.max() <= max(qx) + 6
    assert ys.min() >= min(qy) - 6 and ys.max() <= max(qy) + 6, \
        "warped content must land inside the destination quad"
    before = d.layer(L).pixels.copy()
    assert d.undo() and d.redo()
    after = d.layer(L).pixels
    cov = np.maximum(before[..., 3:4], after[..., 3:4])
    assert float(max(np.abs((before[..., :3] - after[..., :3]) * cov).max(),
                     np.abs(before[..., 3] - after[..., 3]).max())) <= 2e-3
