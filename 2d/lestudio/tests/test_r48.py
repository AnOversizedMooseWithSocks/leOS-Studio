"""tests/test_r48.py -- generated fills + the Shade node.

R48 folds the R47 generators into the two places shading actually
happens:

* fill_generated -- the paint bucket fills its region with scribble or
  line/hatch strokes instead of flat content. Same region semantics as
  flood_fill (one shared _flood_region), soft rim, sample='composite'
  for shading onto a clean layer above lineart. Generator contract
  holds: every emitted stroke is an ordinary journaled paint stroke.
* Shade node -- luminance -> automatic line/hatch/stipple ink, the
  value-aware convention of the hatch brush as a deterministic node.
"""
import numpy as np


def _two_blobs():
    from lestudio import Document
    d = Document(200, 150)
    d.layers[0].pixels[..., :3] = 1.0
    d.layers[0].pixels[..., 3] = 1.0
    d.layers[0].pixels[30:80, 20:90, :3] = 0.75      # blob A
    d.layers[0].pixels[90:140, 100:180, :3] = 0.75   # blob B
    return d


def test_r48_fill_generated_stays_in_the_bucket_region():
    d = _two_blobs()
    L = d.add_layer("shade").id
    n = d.fill_generated(L, 50, 55, style="hatch", tolerance=0.05,
                         seed=5, sample="composite")
    assert n > 0
    a = d.layer(L).pixels[..., 3]
    assert float(a[35:75, 25:85].max()) > 0.5, "blob A must be shaded"
    assert float(a[95:135, 105:175].max()) == 0.0, \
        "the bucket must not spill into the other blob"
    assert d.replay_is_faithful(L), \
        "generated fills are journaled paint strokes -- they must replay"


def test_r48_fill_generated_scribble_and_seed_determinism():
    a, b = _two_blobs(), _two_blobs()
    La, Lb = a.add_layer("s").id, b.add_layer("s").id
    na = a.fill_generated(La, 120, 115, style="scribble", tolerance=0.05,
                          seed=9, sample="composite", curl=0.6)
    nb = b.fill_generated(Lb, 120, 115, style="scribble", tolerance=0.05,
                          seed=9, sample="composite", curl=0.6)
    assert na == nb > 0
    assert np.array_equal(a.layer(La).pixels, b.layer(Lb).pixels), \
        "same seed, same generated fill"


def test_r48_fill_generated_samples_layer_vs_composite():
    d = _two_blobs()
    L = d.add_layer("shade").id
    # the shade layer is empty: flooding IT covers everything, so both
    # blobs get ink -- that is what sample='layer' means on a new layer
    n = d.fill_generated(L, 50, 55, style="line", tolerance=0.05,
                         seed=3, sample="layer")
    a = d.layer(L).pixels[..., 3]
    assert n > 0 and float(a[95:135, 105:175].max()) > 0.0, \
        "sample='layer' on an empty layer floods the whole canvas"


def test_r48_flood_fill_still_shares_the_region():
    # the refactor moved region maths into _flood_region: pin that the
    # plain bucket still fills exactly its blob
    d = _two_blobs()
    lid = d.layers[0].id
    filled = d.flood_fill(lid, 50, 55, np.zeros((150, 200, 3), np.float32),
                          tolerance=0.05)
    px = d.layer(lid).pixels
    assert filled > 0
    assert float(px[35:75, 25:85, :3].max()) == 0.0, "blob A filled black"
    assert float(px[95:135, 105:175, :3].min()) > 0.5, "blob B untouched"


def test_r48_shade_node_reads_luminance():
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    g = np.linspace(0, 1, 200, dtype=np.float32)[None, :].repeat(150, 0)
    img = np.repeat(g[..., None], 4, -1).astype(np.float32)
    img[..., 3] = 1.0
    out = meta["fn"]((150, 200), {"image": img}, prm)
    assert out.shape == (150, 200, 4) and out.dtype == np.float32
    dark = float(out[:, :40, 3].mean())
    light = float(out[:, 160:, 3].mean())
    assert dark > light * 3, \
        "ink must concentrate where the input is dark (%.3f vs %.3f)" % (
            dark, light)
    out2 = meta["fn"]((150, 200), {"image": img}, prm)
    assert np.array_equal(out, out2), "a node render must be deterministic"
    inv = meta["fn"]((150, 200), {"image": img}, dict(prm, invert=True))
    assert float(inv[:, 160:, 3].mean()) > float(inv[:, :40, 3].mean()), \
        "invert shades the lights instead"


def test_r48_shade_node_styles_differ():
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    img = np.full((120, 160, 4), 0.25, np.float32)   # uniformly dark
    img[..., 3] = 1.0
    outs = {st: meta["fn"]((120, 160), {"image": img},
                           dict(prm, style=st)) for st in
            ("line", "hatch", "noise")}
    assert float(outs["hatch"][..., 3].mean()) > \
        float(outs["line"][..., 3].mean()), \
        "hatch adds cross passes in the darks, so it lays more ink"
    for st, o in outs.items():
        assert float(o[..., 3].mean()) > 0.02, "%s made no ink" % st


def test_r48_shade_node_ignores_transparency():
    # outside the drawing there is no ink: transparent pixels must not
    # read as "black, therefore dark, therefore fully hatched" (the
    # first build did exactly that and hatched the whole canvas)
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    img = np.zeros((100, 140, 4), np.float32)
    img[30:70, 40:100, :3] = 0.2                     # a dark patch...
    img[30:70, 40:100, 3] = 1.0                      # ...only IT is opaque
    out = meta["fn"]((100, 140), {"image": img}, prm)
    assert float(out[30:70, 40:100, 3].mean()) > 0.05, \
        "the opaque dark patch must be inked"
    assert float(out[:20, :, 3].max()) == 0.0, \
        "transparent input must produce no ink"


def test_r48_shade_node_every_param_moves_the_ink():
    # the RGB-only audit in test_studio skips Shade (ink lives in alpha);
    # this is the alpha-channel version of that audit for it
    from lestudio import OPS
    meta = OPS["Shade"]
    prm = {p["name"]: p["default"] for p in meta["params"]}
    g = np.linspace(0, 1, 160, dtype=np.float32)[None, :].repeat(120, 0)
    img = np.repeat(g[..., None], 4, -1).astype(np.float32)
    img[..., 3] = 1.0
    base = meta["fn"]((120, 160), {"image": img}, prm)[..., 3]
    sweeps = {"style": "noise", "angle": 90.0, "spacing": 14.0,
              "thickness": 3.0, "wobble": 2.5, "gamma": 2.2,
              "invert": True, "seed": 77}
    for k, v in sweeps.items():
        out = meta["fn"]((120, 160), {"image": img}, dict(prm, **{k: v}))
        assert float(np.abs(out[..., 3] - base).max()) > 1e-4, \
            "param %r has no effect on the ink" % k
