"""tests/test_r49.py -- textile generators + parametric pattern styles.

R49 (user commission): patterns beyond curl noise and line/hatch --
woven cloth, cross-stitch, satin-stitch embroidery as STROKE GENERATORS
(hatch_fill modes weave/cross/stitch, drawing on the drawdown notation
of hand weaving and the satin-stitch/direction-field literature), and
hexagon / voronoi-crackle / fbm-contour / weave / cross styles on the
Shade node. Generator contract unchanged: threads, Xs and stitches are
ordinary journaled paint strokes.
"""
import numpy as np


def _doc():
    from lestudio import Document
    d = Document(220, 170)
    d.layers[0].pixels[..., :3] = 0.95
    d.layers[0].pixels[..., 3] = 1.0
    return d


def test_r49_textile_modes_emit_journaled_strokes_and_replay():
    d = _doc()
    L = d.add_layer("t").id
    counts = {}
    for mode in ("weave", "cross", "stitch"):
        counts[mode] = d.hatch_fill(L, 110, 85, radius=55, mode=mode,
                                    spacing=7, seed=5)
        assert counts[mode] > 10, "%s made too few strokes" % mode
    assert d.replay_is_faithful(L), \
        "textile strokes must replay from the journal alone"
    assert all("op" not in k["brush"] for k in d.strokes), \
        "threads/Xs/stitches are plain paint strokes, not ops"


def test_r49_weave_interlacements_differ():
    outs = {}
    for wv in ("plain", "twill", "satin", "basket"):
        d = _doc()
        L = d.add_layer("t").id
        d.hatch_fill(L, 110, 85, radius=55, mode="weave", weave=wv,
                     spacing=7, angle=0, seed=5)
        outs[wv] = d.layer(L).pixels.copy()
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.array_equal(outs[names[i]], outs[names[j]]), \
                "%s and %s wove the same cloth" % (names[i], names[j])


def test_r49_textile_seed_determinism_and_gate():
    from lestudio import Document
    a, b = _doc(), _doc()
    La, Lb = a.add_layer("t").id, b.add_layer("t").id
    a.hatch_fill(La, 110, 85, radius=60, mode="stitch", seed=11)
    b.hatch_fill(Lb, 110, 85, radius=60, mode="stitch", seed=11)
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels)
    # gated: a feathered ellipse shapes the cloth
    d = _doc()
    L = d.add_layer("t").id
    sel = d.select("ellipse", {"x0": 50, "y0": 40, "x1": 180, "y1": 140},
                   feather=10)
    n = d.hatch_fill(L, None, None, mode="weave", area="selection",
                     selection=sel.id, spacing=7, seed=3)
    assert n > 0
    al = d.layer(L).pixels[..., 3]
    assert float(al[0:15, 0:15].max()) == 0.0, \
        "threads must not escape the gate"


def test_r49_fill_generated_accepts_textile_styles():
    d = _doc()
    d.layers[0].pixels[40:130, 30:190, :3] = 0.7      # a blob to bucket
    L = d.add_layer("t").id
    n = d.fill_generated(L, 100, 80, style="weave", tolerance=0.05,
                         sample="composite", seed=4, spacing=8)
    assert n > 0
    a = d.layer(L).pixels[..., 3]
    assert float(a[0:20, 0:15].max()) == 0.0, \
        "woven fill must stay inside the bucket region"
    assert d.replay_is_faithful(L)


def test_r49_shade_node_pattern_styles():
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    g = np.linspace(0, 1, 200, dtype=np.float32)[None, :].repeat(140, 0)
    img = np.repeat((1 - g)[..., None], 4, -1).astype(np.float32)
    img[..., 3] = 1.0
    outs = {}
    for st in ("weave", "cross", "hex", "crackle", "fbm"):
        out = meta["fn"]((140, 200), {"image": img}, dict(prm, style=st))
        assert out.shape == (140, 200, 4) and out.dtype == np.float32, st
        dark = float(out[:, 160:, 3].mean())   # img = 1-g: right is dark
        light = float(out[:, :50, 3].mean())
        assert dark > light * 2.5, \
            "%s must ink the darks more (%.3f vs %.3f)" % (st, dark, light)
        out2 = meta["fn"]((140, 200), {"image": img}, dict(prm, style=st))
        assert np.array_equal(out, out2), "%s not deterministic" % st
        outs[st] = out[..., 3]
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert float(np.abs(outs[names[i]]
                                - outs[names[j]]).max()) > 0.1, \
                "%s and %s render identically" % (names[i], names[j])


def test_r49_shade_pattern_styles_respect_alpha():
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    img = np.zeros((100, 140, 4), np.float32)
    img[30:70, 40:100, :3] = 0.15
    img[30:70, 40:100, 3] = 1.0
    for st in ("weave", "cross", "hex", "crackle", "fbm"):
        out = meta["fn"]((100, 140), {"image": img}, dict(prm, style=st))
        assert float(out[:15, :, 3].max()) == 0.0, \
            "%s inked transparent input" % st
        assert float(out[30:70, 40:100, 3].mean()) > 0.03, \
            "%s left the dark patch blank" % st
