"""tests/test_r8.py -- pins for the leCore 0.2.20 (main-branch) adoption
(LECORE_UPDATE_R8.md). The update is a clean superset (2253 -> 2323 public
faculties, 0 removed); these pin what leStudio adopted from it."""
import numpy as np
import pytest


def test_r8_engine_superset_and_used_faculties_present():
    """Every faculty leStudio calls exists on the new engine, and the R8
    adoptees are present (skip-tolerant: on an older engine the app degrades
    per have(), and so does this pin)."""
    from lestudio import mind
    m = mind()
    for name in ("guided_filter", "orbit_trap_render", "image_dream",
                 "postfx_chain", "fluid_step", "curl_noise"):
        assert hasattr(m, name), name
    if not hasattr(m, "smoke_animation"):
        pytest.skip("engine predates 0.2.20 -- R8 adoptees gated off")
    assert hasattr(m, "particle_animation")
    assert hasattr(m, "material_data")


def test_r8_sim_media_sources_render_and_loop():
    pytest.importorskip("flask")
    import lestudio.server as srv
    from lestudio import mind
    if not hasattr(mind(), "smoke_animation"):
        pytest.skip("engine predates 0.2.20")
    s = srv._MediaSource("sim:smoke", fps=10)
    f1, _ = s.get()
    assert f1 is not None and f1.ndim == 3 and f1.shape[-1] == 3
    assert "engine simulation" in s.status and "48" in s.status
    # frames vary across the loop (an animation, not a still)
    frames = {np.asarray(fr).tobytes() for fr in s._sim_frames[:8]}
    assert len(frames) >= 6
    p = srv._MediaSource("sim:particles", fps=10)
    fp, _ = p.get()
    assert fp is not None and fp.shape[-1] == 3
    s.close(); p.close()


def test_r8_sim_source_is_deterministic():
    pytest.importorskip("flask")
    import lestudio.server as srv
    from lestudio import mind
    if not hasattr(mind(), "smoke_animation"):
        pytest.skip("engine predates 0.2.20")
    a = srv._MediaSource("sim:smoke", fps=10); a.get()
    b = srv._MediaSource("sim:smoke", fps=10); b.get()
    assert len(a._sim_frames) == len(b._sim_frames)
    assert all(np.array_equal(x, y)
               for x, y in zip(a._sim_frames, b._sim_frames))
    a.close(); b.close()


def test_r8_substances_endpoint_serves_real_optics():
    pytest.importorskip("flask")
    import lestudio.server as srv
    from lestudio import mind
    c = srv.app.test_client()
    r = c.get("/api/materials/substances")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"]
    if not hasattr(mind(), "material_data"):
        assert j["substances"] == []
        return
    subs = {x["name"]: x for x in j["substances"]}
    assert "water" in subs and abs(subs["water"]["refractive"] - 1.333) < 0.01
    assert all("refractive" in x and x["refractive"] >= 1.0
               for x in j["substances"])
    # the UI wires the picker to the IOR control
    ui = open(srv.os.path.join(srv.os.path.dirname(srv.__file__), "static",
                               "index.html")).read()
    assert 'id="lSubstance"' in ui and "materials/substances" in ui


def test_r8_media_in_doc_mentions_sim_sources():
    from lestudio import OPS
    assert "sim:smoke" in OPS["Media in"]["doc"]
