"""The engine Scene-document bridge, against the REAL engine (skips if no engine).

  * /api/scene/describe: words -> leStudio3d objects, exact SDF trees kept, colour words -> the same
    materials /api/semantic would choose; unknown phrases pass through;
  * /api/scene/doc: the engine's reading of THIS scene, handle -> object id;
  * /api/scene_preview: a PNG in well under a photo's cost; /api/camera/fit frames every vertex;
  * /api/scene/animate: keyframes by object id -> a GIF;
  * the new Mesh ops (orient / manifold_cleanup / cvt_remesh / retopo) leave a non-empty mesh and report;
  * /api/env engine sources (studio rig, parametric sky) light the renderer.
Run:  PYTHONHASHSEED=0 python3 tests/scene_doc_test.py
"""
import base64, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
try:
    import lecore  # noqa
except Exception:
    print("skip: no leos-core installed"); sys.exit(0)
os.environ.pop("LESTUDIO3D_WORKSPACE", None)
sys.path.insert(0, ROOT)
import backend
from flask import Flask
app = Flask("t"); app.register_blueprint(backend.bp); app.config["PROPAGATE_EXCEPTIONS"] = True
backend.mount_foundation(app, verbose=False)
c = app.test_client()
PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name); print(("ok  " if cond else "FAIL ") + name + ((" -- " + str(detail)) if detail and not cond else ""))

r = c.post("/api/scene/describe", json={"text": "a red cube and a green sphere", "res": 36}).get_json()
check("describe makes two objects", len(r.get("made", [])) == 2, r)
mats = {m["name"]: m["material"] for m in r.get("made", [])}
check("colour words map like /api/semantic (red->ruby, green->emerald)", mats.get("red box") == "ruby" and mats.get("green sphere") == "emerald", mats)
oid = r["made"][0]["object"]
check("described object exposes no half-contract tree (to_dsl-less SDF classes are dropped)",
      backend._S["objects"][oid].sdf_tree is None or hasattr(backend._S["objects"][oid].sdf_tree, "to_dsl"))
sem = c.post("/api/semantic", json={"command": "make the sphere bigger"})
check("/api/semantic still runs after describe (a described object used to 500 it)", sem.status_code < 500, sem.status_code)
bad = c.post("/api/scene/describe", json={"text": "a flibbertigibbet"})
check("nonsense is refused with the engine's unknown list", bad.status_code == 400 and bad.get_json().get("unknown") is not None, bad.get_json())
info = c.get("/api/scene/doc").get_json()
check("scene/doc counts every object", info["n_objects"] == len(backend._S["objects"]), info["n_objects"])
check("handles map to object ids", set(info["handles"].values()) == set(backend._S["objects"]), info["handles"])
t = time.time(); p = c.get("/api/scene_preview?w=96&h=72")
check("preview is a PNG", p.status_code == 200 and p.data[:8] == b"\x89PNG\r\n\x1a\n", p.status_code)
check("preview is fast (< 5 s at 96x72)", time.time() - t < 5.0, round(time.time() - t, 1))
fit = c.get("/api/camera/fit?w=320&h=200").get_json()
check("fit_camera returns eye/target/fov", {"eye", "target", "fov_deg"} <= set(fit), fit)
a = c.post("/api/scene/animate", json={"keys": {oid: {"position": [[0, [-1, 0, 0]], [1, [1, 0, 0]]]}}, "n_frames": 3, "w": 48, "h": 36}).get_json()
check("animate returns a GIF", a.get("frames") == 3 and str(a.get("gif", "")).startswith("data:image/gif;base64,"), a.get("error"))
check("animate refuses an unknown object", c.post("/api/scene/animate", json={"keys": {"nope": {"position": [[0, [0, 0, 0]]]}}}).status_code == 400)
for op, extra in (("orient", {}), ("manifold_cleanup", {}), ("cvt_remesh", {"sites": 120}), ("retopo", {"density": 0.8})):
    resp = c.post("/api/op", json={"op": op, "object": oid, **extra}); j = resp.get_json()
    check(f"op {op} succeeds with a face report", resp.status_code == 200 and j.get("faces", {}).get("after", 0) > 0, j.get("error"))
for body, src in (({"studio": "classic"}, "studio"), ({"sky": {"hour": 19.0, "clouds": [["cirrus", 0.5]]}}, "sky")):
    e = c.post("/api/env", json=body).get_json()
    check(f"env source {src} lights the dome", e.get("source") == src and e.get("max_radiance", 0) > 0 and len(str(e.get("preview", ""))) > 100, e.get("error"))
man = c.get("/api/agent/tools").get_json(); names = {t["name"] for t in man["tools"]}
check("agent manifest exposes the new doors", {"scene/describe", "scene/doc", "scene_preview", "scene/animate", "camera/fit", "presets", "memory"} <= names, sorted(n for n in names if "scene" in n))
inv = c.post("/api/agent/invoke", json={"tool": "scene_preview", "args": {"w": 32, "h": 24}}).get_json()
check("invoke returns the preview as a data URL (how an agent sees its work)", str(inv.get("result", "")).startswith("data:image/png;base64,") or str((inv.get("result") or {}).get("image", "")).startswith("data:image/png"), str(inv)[:120])
print(f"\n{len(PASS)} passed, {len(FAIL)} failed"); sys.exit(1 if FAIL else 0)
