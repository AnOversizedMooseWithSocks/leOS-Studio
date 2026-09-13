"""tests/test_r4.py -- sweep pins for the R4 release nodes (LECORE_SWEEP_R4.md).

Each adopted node gets: it runs, its headline param bites, and the specific
property its experiment verified (LUT identity, freq-split exactness, wheels
neutrality, halation confinement, dream honesty...) stays true. Same bar as
the engine: deterministic, measured, no vibes.
"""
import os
import tempfile

import numpy as np
import pytest

from lestudio import OPS

H, W = 48, 64


def _img(seed=0):
    return np.random.default_rng(seed).random((H, W, 3)).astype(np.float32)


def _run(name, ins=None, **params):
    meta = OPS[name]
    base = {p["name"]: p["default"] for p in meta["params"]}
    base.update(params)
    fill = dict(ins or {})
    for s in meta["inputs"]:
        if s not in fill:
            im = _img()
            if meta.get("rgba"):
                im = np.concatenate([im, np.ones((H, W, 1), np.float32)], -1)
            fill[s] = im
    return meta["fn"]((H, W), fill, base)


def test_r4_nodes_registered_and_gated():
    for name, req in [("Clarity", ["guided_filter"]), ("Dehaze", ["guided_filter"]),
                      ("Orbit trap", ["orbit_trap_render"]),
                      ("Remember", ["image_remember"]), ("Dream", ["image_dream"])]:
        assert name in OPS and OPS[name]["requires"] == req
    for name in ("Color wheels", "LUT", "Dither", "Film look", "Frequency split",
                 "Frequency merge", "Content-aware scale", "Focus stack", "Scope"):
        assert name in OPS


def test_clarity_bites_and_negative_flattens():
    img = _img()
    boosted = _run("Clarity", {"image": img}, clarity=1.5)
    flattened = _run("Clarity", {"image": img}, clarity=-0.8)
    assert not np.allclose(boosted, img)
    # negative clarity reduces local variance, positive raises it
    v = lambda a: float(np.var(a - a.mean((0, 1))))
    assert v(flattened) < v(img) < v(boosted)


def test_dehaze_removes_synthetic_veil():
    rng = np.random.default_rng(3)
    clean = rng.random((H, W, 3)).astype(np.float32) * 0.6
    A = np.array([0.92, 0.93, 0.95], np.float32)
    t = np.repeat(np.linspace(1, 0.4, H)[:, None], W, 1).astype(np.float32)
    hazed = clean * t[..., None] + A * (1 - t[..., None])
    out = _run("Dehaze", {"image": hazed}, strength=1.0)
    assert np.sqrt(((out - clean) ** 2).mean()) < np.sqrt(((hazed - clean) ** 2).mean())


def test_wheels_neutral_is_exact_identity():
    img = _img()
    out = _run("Color wheels", {"image": img})
    assert np.abs(out - img).max() < 1e-6
    warm = _run("Color wheels", {"image": img}, gain_r=0.2)
    bright = img.mean(-1) > 0.66
    if bright.any():
        assert (warm[..., 0][bright] >= img[..., 0][bright] - 1e-6).all()
        assert warm[..., 0][bright].mean() > img[..., 0][bright].mean()


def test_lut_identity_cube_roundtrip_exact():
    n = 9
    idx = np.linspace(0, 1, n)
    lat = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1)
    fd, path = tempfile.mkstemp(suffix=".cube")
    with os.fdopen(fd, "w") as f:
        f.write("LUT_3D_SIZE %d\n" % n)
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("%.8f %.8f %.8f\n" % tuple(lat[r, g, b]))
    try:
        img = _img()
        out = _run("LUT", {"image": img}, file=path)
        assert np.abs(out - img).max() < 1e-6
        # missing file passes through instead of erroring
        out2 = _run("LUT", {"image": img}, file="/nonexistent/nope.cube")
        assert np.allclose(out2, img)
    finally:
        os.unlink(path)


def test_lut_intensity_dial_bites_with_a_real_cube():
    # an inverting LUT; intensity 0.5 must land halfway (the dead-param
    # audit exempts LUT because the dial is inert without a file -- this is
    # the promised behaviour verification WITH one)
    n = 5
    idx = np.linspace(0, 1, n)
    lat = 1.0 - np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1)
    fd, path = tempfile.mkstemp(suffix=".cube")
    with os.fdopen(fd, "w") as f:
        f.write("LUT_3D_SIZE %d\n" % n)
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("%.8f %.8f %.8f\n" % tuple(lat[r, g, b]))
    try:
        img = _img()
        full = _run("LUT", {"image": img}, file=path, intensity=1.0)
        half = _run("LUT", {"image": img}, file=path, intensity=0.5)
        assert np.abs(full - (1 - img)).max() < 1e-5
        assert np.abs(half - (img + (1 - 2 * img) * 0.5)).max() < 1e-5
    finally:
        os.unlink(path)


def test_dither_hits_palette_and_bayer_beats_nearest():
    img = _img(5)
    out = _run("Dither", {"image": img}, palette="game boy", method="nearest")
    pal = np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]]) / 255.0
    d = np.abs(out.reshape(-1, 1, 3) - pal).sum(-1).min(1)
    assert d.max() < 1e-6                       # every pixel IS a palette colour
    # ordered dither preserves smooth ramps better than nearest (T2's number)
    ramp = np.repeat(np.linspace(0, 1, W)[None, :, None], H, 0).repeat(3, -1).astype(np.float32)
    qn = _run("Dither", {"image": ramp}, palette="1-bit", method="nearest")
    qb = _run("Dither", {"image": ramp}, palette="1-bit", method="bayer")
    err = lambda q: np.abs(q.mean(-1).mean(0) - ramp.mean(-1).mean(0)).mean()
    assert err(qb) < err(qn)


def test_film_look_deterministic_and_halation_gated():
    img = _img(7)
    a = _run("Film look", {"image": img}, film="custom", halation=0.8, grain=0.1, seed=4)
    b = _run("Film look", {"image": img}, film="custom", halation=0.8, grain=0.1, seed=4)
    assert np.array_equal(a, b)                 # deterministic per (seed, frame)
    dark = np.full((H, W, 3), 0.2, np.float32)  # no highlights -> no halation
    out = _run("Film look", {"image": dark}, film="custom", halation=1.0,
               grain=0.0, fade=0.0, weave=0.0)
    assert np.abs(out - dark).max() < 1e-4


def test_frequency_split_merge_roundtrip_exact():
    img = _img(9)
    parts = _run("Frequency split", {"image": img})
    back = _run("Frequency merge", {"low": parts["out"], "high": parts["high"]})
    assert np.abs(back - img).max() < 1e-5


def test_content_aware_scale_shrinks_content_keeps_canvas():
    img = _img(11)
    out = _run("Content-aware scale", amount=20.0)
    assert out.shape == (H, W, 4)
    a = out[..., 3]
    assert (a[:, 0] < 0.5).all() and (a[:, -1] < 0.5).all()   # transparent margins
    assert (a[:, W // 2] > 0.5).all()                          # content in the middle


def test_focus_stack_beats_both_inputs():
    from lestudio import _r4_gauss
    sharp = _img(13)
    top, bot = sharp.copy(), sharp.copy()
    top[:H // 2] = _r4_gauss(sharp, 4)[:H // 2]
    bot[H // 2:] = _r4_gauss(sharp, 4)[H // 2:]
    fused = _run("Focus stack", {"a": top, "b": bot, "c": None, "d": None})
    e = lambda a: float(np.abs(a.mean(-1) - _r4_gauss(a.mean(-1), 2)).mean())
    assert e(fused) > max(e(top), e(bot))


def test_orbit_trap_renders_varied_colours():
    out = _run("Orbit trap")
    assert out.shape == (H, W, 3) and np.isfinite(out).all()
    px = out.reshape(-1, 3)
    assert px.std(0).max() > 0.02               # varied, not a flat fill


def test_scope_modes_all_render():
    img = _img(17)
    for mode in ("waveform", "parade", "vectorscope", "histogram"):
        out = _run("Scope", {"image": img}, mode=mode)
        assert out.shape == (H, W, 3)
        assert out.max() > 0.05                 # a visible trace


def test_remember_passthrough_and_dream_honest_refusal():
    img = _img(19)
    out = _run("Remember", {"image": img}, label="r4 pin test tag")
    assert np.allclose(out, np.clip(img, 0, 1), atol=1e-6)   # inline passthrough
    # a query nothing matches renders the refusal card, not garbage
    card = _run("Dream", query="zz unmatched query zz")
    assert card.shape == (H, W, 3) and np.isfinite(card).all()
    assert card.max() <= 0.5                     # the dark told-you card


def test_postfx_agx_choice_runs_and_bounds():
    img = _img(23) * 2.0                        # HDR-ish input
    out = _run("Post FX", {"image": np.clip(img, 0, 1)}, tonemap="agx")
    assert out.min() >= 0 and out.max() <= 1


def test_lut_export_endpoint_bakes_wheels():
    flask = pytest.importorskip("flask")
    from lestudio import server as srv
    client = srv.app.test_client()
    srv.GRAPH.ensure_default()
    srv.GRAPH.set_graph(list(srv.GRAPH.ensure_default().values()) + [
        {"id": "SRC", "type": "Solid", "params": {}, "inputs": {}},
        {"id": "WH", "type": "Color wheels",
         "params": {"gain_r": 0.2}, "inputs": {"image": "SRC"}},
    ])
    r = client.post("/api/export/lut", json={"node": "WH", "size": 17})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_data(as_text=True)
    assert "LUT_3D_SIZE 17" in body
    assert "WARNING" not in body                # wheels are pointwise
    assert len([l for l in body.splitlines()
                if l and l[0].isdigit() or l.startswith("0")]) >= 17 ** 3


def test_glsl_export_endpoint_compiles_pointwise():
    flask = pytest.importorskip("flask")
    from lestudio import server as srv
    client = srv.app.test_client()
    srv.GRAPH.ensure_default()
    srv.GRAPH.set_graph(list(srv.GRAPH.ensure_default().values()) + [
        {"id": "PFX", "type": "Post FX",
         "params": {"exposure": 0.5, "saturation": 1.2, "vignette": 0.3,
                    "tonemap": "aces", "bloom": 0.4}, "inputs": {}},
    ])
    r = client.post("/api/export/glsl", json={"node": "PFX"})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] and "mainImage" in j["glsl"]
    assert "skipped" in j["glsl"]               # bloom reported, not hidden
