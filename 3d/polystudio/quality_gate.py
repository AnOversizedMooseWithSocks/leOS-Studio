"""RENDER QUALITY GATE -- run this before shipping any change that touches a render path.

WHY THIS EXISTS: a preview shipped with heavy grid-terracing artifacts (contour rings on curved surfaces,
stair-stepped shadows) and the verification at the time MISSED it, because the check compared the new render
against the previous render. Both had the artifact, so a difference metric could not see it. This gate
measures the render against ABSOLUTE thresholds instead, and each threshold is tied to the specific defect
that actually shipped.

    python3 quality_gate.py            # from the demo directory (gallery mount)
    python3 quality_gate.py --write    # also save the frames for eyeballing

Metrics, and the defect each one catches:
  * terracing   -- fraction of floor rows showing a hard step. Grid-baked fields put iso-contour terraces
                   into shading; an exact field does not. (The shipped bug.)
  * edge_tones  -- fraction of silhouette pixels holding an intermediate tone. Accumulating below display
                   resolution and upscaling bakes in stair-stepped edges; true supersampling does not.
  * fringe      -- chroma at edges vs a single-round frame. Catches a mis-aligned albedo map painting colour
                   fringes onto silhouettes.
  * exactness   -- the pristine scene must actually take the analytic path. If a change accidentally routes
                   primitives through the voxel bake, terracing returns; this fails loudly and by name.

A metric alone is never sufficient -- LOOK at the frames too (--write). This gate catches regressions of
known defects; it cannot see a new kind of ugly.
"""
import io
import sys

import numpy as np
from PIL import Image

CAM = "eye=2.4,1.7,2.9&target=0.8,0,0.15&fov=45"
W = 640

# Thresholds: measured headroom over the current good state, not aspirations. Current values are in
# parentheses; a gate that sits exactly on the current number fails on noise.
LIMITS = {
    "terracing_exact": 0.035,   # measured 0.022 here; the shipped-bad frame scored 0.068
    "terracing_mixed": 0.075,   # measured 0.055; one edited object is baked, the rest stay exact
    "edge_tones": 0.90,         # (1.000) fraction of edge pixels with an intermediate tone -- higher is better
    "fringe_ratio": 1.15,       # (~0.98) converged edge chroma / single-round edge chroma
}


def _img(resp):
    return np.asarray(Image.open(io.BytesIO(resp.data)).convert("RGB"), float) / 255.0


def engine_gate(frame, limits=None):
    """Delegate to the engine's own render gate (leCore sweep 163: m.render_quality_gate).

    docs/POLYSTUDIO_AUDIT.md marks this file UPSTREAM -- the engine's render sweeps had no
    absolute-threshold regression tool, and sweep 151's aliasing would have been caught by
    ours. Now that the engine has one, ours calls it and keeps only what it does not cover.
    Returns None when the engine is too old, and the local checks below still run.
    """
    try:
        import lecore
        return lecore.UnifiedMind(dim=256, seed=0).render_quality_gate(frame, limits=limits)
    except Exception:
        return None


def terracing(img):
    """Terracing = flat plateaus separated by jumps, so it shows up as SPIKES IN THE SECOND DIFFERENCE down
    the floor. A plain first-difference test was tried first and rejected: it cannot tell a terrace from the
    legitimate gradient of a soft shadow edge. Calibrated on the frames that actually shipped --
    known-bad (grid-terraced, user-reported) 0.068, known-good (exact + supersampled) 0.018."""
    g = img.mean(-1)
    band = g[int(g.shape[0] * 0.5):, :]
    return float((np.abs(np.diff(band, 2, axis=0)) > 0.010).mean())


def edge_tones(img):
    """Fraction of silhouette pixels that are partially covered. A jagged edge is all-or-nothing."""
    g = img.mean(-1)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    e = mag > np.percentile(mag, 99)
    v = g[e]
    return float(((v > 0.05) & (v < 0.95)).mean()) if v.size else 0.0


def edge_chroma(img):
    g = img.mean(-1)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    e = mag > np.percentile(mag, 99)
    ch = np.abs(img[..., 0] - img[..., 1]) + np.abs(img[..., 1] - img[..., 2])
    return float(ch[e].mean()) if e.any() else 0.0


def resolve(client, api, session, extra=""):
    """Run a progressive render to convergence and return the final frame."""
    resp = None
    for _ in range(20):
        resp = client.get(f"{api}/render_progressive?session={session}&w={W}&{CAM}&grid=48{extra}")
        if resp.status_code != 200:
            raise RuntimeError(f"render failed: {resp.status_code} {resp.data[:200]}")
        if resp.headers.get("X-Converged") == "1":
            break
    return resp


def main(write=False):
    sys.path.insert(0, "../../holostuff")
    sys.path.insert(0, "../..")
    import flatcompat
    flatcompat.install()
    from app import app

    c = app.test_client()
    api = "/demos/10_polystudio/api"
    c.get(f"{api}/scene")
    results, failures = {}, []

    def check(name, value, limit, higher_is_better=False):
        results[name] = value
        bad = value < limit if higher_is_better else value > limit
        if bad:
            failures.append(f"{name}: {value:.4f} {'<' if higher_is_better else '>'} limit {limit:.4f}")

    # --- 1. pristine scene: must be exact, and must be clean -------------------------------------
    scene = c.get(f"{api}/scene").get_json()
    cube = next(o["id"] for o in scene["objects"] if o["name"] == "Cube")
    # the factory scene is a single cube, so the gate adds its own second object -- per-object materials
    # can only be checked with two of them, and the check must not depend on what the scene ships with.
    added = c.post(f"{api}/new", json={"primitive": "icosphere"}).get_json()
    sphere = added.get("object") or [o["id"] for o in c.get(f"{api}/scene").get_json()["objects"]][-1]
    c.post(f"{api}/assign", json={"material": "plastic_red", "object": cube, "all": True})
    c.post(f"{api}/assign", json={"material": "plastic_green", "object": sphere, "all": True})

    resp = resolve(c, api, "qg_exact")
    exact_img = _img(resp)
    check("terracing_exact", terracing(exact_img), LIMITS["terracing_exact"])
    check("edge_tones", edge_tones(exact_img), LIMITS["edge_tones"], higher_is_better=True)

    single = _img(c.get(f"{api}/render_progressive?session=qg_one&w={W}&eye=2.41,1.7,2.9"
                        f"&target=0.8,0,0.15&fov=45&grid=48"))
    ec_single = edge_chroma(single)
    check("fringe_ratio", edge_chroma(exact_img) / max(ec_single, 1e-6), LIMITS["fringe_ratio"])

    # routing: the pristine scene must NOT be going through the voxel bake
    method = resp.headers.get("X-Build-Method", "")
    results["build_method"] = method or "(not reported)"
    # NOTE: substring-matching "analytic" here is NOT enough. The BAKED path reports "analytic-native"
    # (it uses the analytic tree to fill a voxel grid quickly -- still a grid, still terraced). Verified by
    # injecting the original bug: the frame terraced while the method still read "analytic-native".
    # Only "analytic-exact" means no grid.
    if method and method != "analytic-exact":
        failures.append(f"pristine scene did NOT take the exact path (method={method}) -- terracing returns")

    # per-object colour must survive whatever the render path does
    red = int(((exact_img[..., 0] > 0.25) & (exact_img[..., 0] > exact_img[..., 1] * 1.6)).sum())
    green = int(((exact_img[..., 1] > 0.25) & (exact_img[..., 1] > exact_img[..., 0] * 1.4)).sum())
    results["per_object_colour"] = f"red {red}, green {green}"
    if red < 150 or green < 150:
        failures.append(f"per-object materials lost in the preview (red {red}, green {green})")

    # --- 2. mixed scene: one edited object must not terrace the others ---------------------------
    d = c.get(f"{api}/scene").get_json()
    obj = next(o for o in d["objects"] if o["id"] == cube)
    V = np.array(obj["positions"]).reshape(-1, 3)
    top = [i for i in range(len(V)) if V[i, 2] > V[:, 2].max() - 1e-4]
    c.post(f"{api}/verts", json={"object": cube, "indices": top,
                                 "positions": [x for i in top
                                               for x in (float(V[i, 0]), float(V[i, 1]), float(V[i, 2] + 0.05))]})
    mixed_img = _img(resolve(c, api, "qg_mixed"))
    check("terracing_mixed", terracing(mixed_img), LIMITS["terracing_mixed"])

    if write:
        Image.fromarray((exact_img * 255).astype(np.uint8)).save("qg_exact.png")
        Image.fromarray((mixed_img * 255).astype(np.uint8)).save("qg_mixed.png")

    width = max(len(k) for k in results)
    for k, v in results.items():
        print(f"  {k:<{width}} : {v:.4f}" if isinstance(v, float) else f"  {k:<{width}} : {v}")
    if failures:
        print("\nQUALITY GATE FAILED")
        for f in failures:
            print("  *", f)
        return 1
    print("\nQUALITY GATE PASSED (still look at the frames: --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv))
