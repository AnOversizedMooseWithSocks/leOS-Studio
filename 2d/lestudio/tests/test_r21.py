"""tests/test_r21.py -- taste-steered dreams (R21).

Round 3 of the improvement sweep: star a dream and future batches lean
toward the looks you starred -- the semantic-compass pattern (record what
worked, bias candidates toward it), implemented as a scale-preserving
centroid blend in the stable raw feature space (compass.steer's unit
renormalisation would crush mixed-unit vectors -- measured, recorded).
Taste persists with the leCore partition and steering never breaks
determinism: same views + seed + stars = same dreams."""
import json
import os

import numpy as np
import pytest


def _need_lecore():
    try:
        from holographic.rendering.holographic_splat import splat_fit  # noqa
    except Exception:
        pytest.skip("leCore not on the path")


GALLERY = ["/root/work/unplugged_v3.png", "/root/work/golden_hour_lake.png",
           "/root/work/little_orchestra.png", "/root/work/north_lake.png"]


def _iso_taste(srv, tmp="/tmp/lestudio_taste_pin"):
    os.makedirs(tmp, exist_ok=True)
    srv._LECORE["mind_part"] = tmp
    p = srv._taste_path()
    if os.path.exists(p):
        os.remove(p)
    return p


def test_r21_starred_dreams_steer_future_batches():
    _need_lecore()
    pytest.importorskip("flask")
    if not all(os.path.exists(p) for p in GALLERY):
        pytest.skip("gallery images not present")
    import lestudio.server as srv
    taste_file = _iso_taste(srv)
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "tt", "width": 160, "height": 120})
    base = c.post("/api/dream", json={"n": 2, "seed": 4,
                                      "paths": GALLERY}).get_json()
    assert base.get("ok")
    # starring requires a prior batch; the vector persists to disk
    f = c.post("/api/dream/fave", json={"i": 0}).get_json()
    assert f.get("ok") and f["stars"] == 1
    assert os.path.exists(taste_file)
    assert len(json.load(open(taste_file))) == 1
    # same request now leans toward the star -- and stays deterministic
    s1 = c.post("/api/dream", json={"n": 2, "seed": 4,
                                    "paths": GALLERY}).get_json()
    s2 = c.post("/api/dream", json={"n": 2, "seed": 4,
                                    "paths": GALLERY}).get_json()
    assert s1["thumbs"] != base["thumbs"], "taste must change the dreams"
    assert s1["thumbs"] == s2["thumbs"], "steering must stay deterministic"
    # steer:false reproduces the unsteered batch exactly
    off = c.post("/api/dream", json={"n": 2, "seed": 4, "paths": GALLERY,
                                     "steer": False}).get_json()
    assert off["thumbs"] == base["thumbs"]
    # starring without a batch in memory is a 400, not a crash
    del srv._DREAM_PTS[:]
    assert c.post("/api/dream/fave", json={"i": 0}).status_code == 400


def test_r21_taste_skips_mismatched_dimensions():
    """A taste vector recorded at another k must be ignored, silently and
    safely -- never a crash, never a corrupt blend."""
    _need_lecore()
    pytest.importorskip("flask")
    if not all(os.path.exists(p) for p in GALLERY):
        pytest.skip("gallery images not present")
    import lestudio.server as srv
    _iso_taste(srv)
    srv._taste_save([[0.5] * 42])            # wrong dimension on purpose
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "tm", "width": 160, "height": 120})
    a = c.post("/api/dream", json={"n": 2, "seed": 4,
                                   "paths": GALLERY}).get_json()
    b = c.post("/api/dream", json={"n": 2, "seed": 4, "paths": GALLERY,
                                   "steer": False}).get_json()
    assert a.get("ok")
    assert a["thumbs"] == b["thumbs"], \
        "mismatched taste must be a no-op, not a corrupt blend"


def test_r21_ui_has_the_star():
    import os as _os
    import lestudio.server as srv
    ui = open(_os.path.join(_os.path.dirname(srv.__file__), "static",
                            "index.html")).read()
    for needle in ("/api/dream/fave", "Taste remembered"):
        assert needle in ui, needle
