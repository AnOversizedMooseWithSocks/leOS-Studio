"""tests/test_r20.py -- splat quality + the morph (R20).

Round 2 of the improvement sweep. Measured first: splat_clone_split
WITHOUT gradient re-optimisation REGRESSES a matching-pursuit fit
(19.6 -> 12.7 dB after refit -- recorded as a negative, not shipped);
the real quality gap was COLOUR-BLIND placement, fixed adaptively.
And the morph: two splat codes flowing into each other, deterministic,
every frame the same closed form the browser parity-renders."""
import io

import numpy as np
import pytest


def _need_lecore():
    try:
        from holographic.rendering.holographic_splat import splat_fit  # noqa
    except Exception:
        pytest.skip("leCore not on the path")


def test_r20_splat_code_is_colour_aware():
    """A field where two regions differ ONLY in colour (equal luminance)
    must still be resolved -- luminance-only placement cannot see it."""
    _need_lecore()
    import lestudio.server as srv
    H, W = 60, 90
    img = np.zeros((H, W, 3))
    img[:, :45] = [0.7, 0.2, 0.1]        # red half
    img[:, 45:] = [0.1, 0.2, 0.7]        # blue half, same luminance
    splats, A = srv._splat_code(img, 40)
    rec = srv._splat_render_color(splats, A, (H, W))
    err = float(np.abs(rec - img).mean())
    assert err < 0.08, "colour-aware placement must resolve equal-luma " \
                       "colour structure (mean err %.3f)" % err
    # red stays red, blue stays blue
    assert float(rec[30, 15, 0]) > float(rec[30, 15, 2])
    assert float(rec[30, 75, 2]) > float(rec[30, 75, 0])


def test_r20_morph_is_a_deterministic_gif():
    _need_lecore()
    pytest.importorskip("flask")
    pytest.importorskip("PIL")
    import os
    from PIL import Image
    import lestudio.server as srv
    if not os.path.exists("/root/work/golden_hour_lake.png"):
        pytest.skip("gallery image not present")
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "mo", "width": 200, "height": 140,
                             "background": [0.85, 0.88, 0.92]})
    lid = srv.DOC.layers[0].id
    for i in range(3):
        c.post("/api/paint", json={"layer": lid,
                                   "points": [[12, 22 + 34 * i],
                                              [188, 30 + 34 * i]],
                                   "color": [0.2 + 0.2 * i, 0.5,
                                             0.8 - 0.2 * i],
                                   "radius": 11, "record": True})
    body = {"path": "/root/work/golden_hour_lake.png", "k": 60,
            "frames": 10, "fps": 12}
    r1 = c.post("/api/splats/morph", json=body)
    assert r1.status_code == 200 and r1.mimetype == "image/gif"
    im = Image.open(io.BytesIO(r1.data))
    assert im.n_frames == 10 + 8, "boomerang loop: forth and back"
    # deterministic: same request, same bytes
    r2 = c.post("/api/splats/morph", json=body)
    assert r1.data == r2.data, "the morph must be deterministic"
    # first frame is the painting's code render, last-of-forth the target's:
    # they must differ substantially (it actually morphs)
    im.seek(0)
    f0 = np.asarray(im.convert("RGB"), float)
    im.seek(9)
    f9 = np.asarray(im.convert("RGB"), float)
    assert float(np.abs(f0 - f9).mean()) > 8.0, "no visible morph"
    # refusal without a reference
    assert c.post("/api/splats/morph", json={}).status_code == 400


def test_r20_ui_has_the_morph():
    import os
    import lestudio.server as srv
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    for needle in ("lcSplMorph", "/api/splats/morph"):
        assert needle in ui, needle
