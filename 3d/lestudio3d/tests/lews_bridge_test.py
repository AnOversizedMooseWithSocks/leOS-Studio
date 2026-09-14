"""The leStudio bridge, against the REAL engine and a live .lews directory (skips if no engine).

What it proves (1.6.0):
  * an R67-style leStudio document -- layer pixels hoisted into `lecore.asset` (meta.array_refs) and a
    journal-first layer with no pixels -- composites instead of being skipped, and says which layers are
    approximate;
  * the live workspace publishes CANONICAL kinds (lecore.mesh / material / scene / image) beside the private
    blob, unchanged sections are not re-put, and a second app's put shows up in /api/workspace?since=;
  * object ids come from the engine's per-prefix counter and survive save -> load;
  * {"from":"workspace"} import pulls textures AND lecore.mesh sections another app put.
Run:  PYTHONHASHSEED=0 python3 tests/lews_bridge_test.py
"""
import json, os, shutil, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
try:
    import lecore  # noqa
except Exception:
    print("skip: no leos-core installed"); sys.exit(0)
ROOT_WS = tempfile.mkdtemp(prefix="ps_lews_")
os.environ["LESTUDIO3D_WORKSPACE"] = ROOT_WS
sys.path.insert(0, ROOT)
import backend
from flask import Flask
app = Flask("t"); app.register_blueprint(backend.bp); app.config["PROPAGATE_EXCEPTIONS"] = True
backend.mount_foundation(app, verbose=False)
c = app.test_client()
PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name); print(("ok  " if cond else "FAIL ") + name + ((" -- " + str(detail)) if detail and not cond else ""))

ids = [o["id"] for o in c.get("/api/scene").get_json()["objects"]]
check("ids are engine-minted (prefix O)", all(i.startswith("O") for i in ids), ids)
sv = c.get("/api/scene/save").get_json()
check("save carries ids", [o["id"] for o in sv["objects"]] == ids)
check("save puts into the workspace", isinstance(sv.get("workspace_rev"), int), sv.get("workspace_error"))
ch = c.get("/api/workspace?since=0").get_json()["changes"]
kinds = {e.get("kind") for e in ch}
check("canonical kinds published", {"lecore.mesh", "lecore.material", "lecore.scene", "lestudio3d.scene"} <= kinds, sorted(kinds))
rev1 = sv["workspace_rev"]; sv2 = c.get("/api/scene/save").get_json()
check("unchanged canonical sections are not re-put", sv2["workspace_rev"] == rev1 + 1, (rev1, sv2["workspace_rev"]))  # only the private blob rewrites

# a second app writes: leStudio-shaped document (R67 hoisting + journal-only layer) and a lecore.mesh
m = backend._mind(); painter = m.lews_open(ROOT_WS, app="lestudio")
px = np.zeros((16, 16, 4), np.float32); px[:, :8] = [1, 0, 0, 1]; px[:, 8:] = [0, 0, 1, 1]
painter.put({"kind": "lecore.asset", "id": "asset:t1", "meta": {}, "arrays": {"data": px}})
painter.put({"kind": "lestudio.document", "id": "D1",
             "meta": {"name": "skin", "width": 16, "height": 16, "array_refs": {"layer_L1": "asset:t1"},
                      "layers": [{"id": "L1", "name": "base", "visible": True, "opacity": 1.0, "blend": "normal"},
                                 {"id": "L2", "name": "strokes", "visible": True, "opacity": 1.0, "blend": "normal",
                                  "pixels_cached": False}]},
             "arrays": {"replaybase_L2": np.zeros((16, 16, 4), np.float32)}})
painter.put(m.lews_mesh_section([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2], [1, 3, 2]], sid="agent:mesh:1", name="agent quad"))
docs = backend._lews_documents({"sections": backend._ws_sections(backend._workspace())})
check("R67 hoisted layer resolves (2 layers, not 0)", docs and docs[0]["layers"] == 2, docs)
check("journal-only layer is named, not hidden", docs and docs[0]["journal_only"] == ["L2"], docs)
check("composite matches the painter's pixels", docs and float(docs[0]["rgb"][0, 0, 0]) > 0.9 and float(docs[0]["rgb"][0, 15, 2]) > 0.9)
r = c.post("/api/workspace/import?object=" + ids[0], data=json.dumps({"from": "workspace"}), content_type="application/json").get_json()
check("live import applies the texture", r.get("workspace", {}).get("applied_to") == ids[0], r)
check("live import creates an object from lecore.mesh", len(r.get("workspace", {}).get("imported_meshes", [])) == 1, r)
sc = c.get("/api/scene").get_json()
check("imported quad is in the scene", any(o["name"] == "agent quad" for o in sc["objects"]))
# feed excludes our own echo
seen = backend._workspace().since(0, exclude="lestudio3d")
check("since(exclude=lestudio3d) shows only the painter", seen and all(e.get("app") != "lestudio3d" for e in seen if "app" in e), seen[:2])
# save -> load keeps ids
before = [o["id"] for o in sc["objects"]]
c.get("/api/scene/save"); c.post("/api/scene/load", json={"from": "workspace"})
after = [o["id"] for o in c.get("/api/scene").get_json()["objects"]]
check("ids survive save -> load", before == after, (before, after))
# our removed object disappears from the canonical set
c.post("/api/op", json={"op": "delete_object", "object": after[-1]}); c.get("/api/scene/save")
mesh_ids = {s["id"] for s in backend._workspace().sections("lecore.mesh")}
check("deleted object's lecore.mesh is deleted too", f"lestudio3d:mesh:{after[-1]}" not in mesh_ids, sorted(mesh_ids))
# presets are visible to the painter
c.post("/api/presets", json={"name": "soft", "target": "lestudio3d.render", "params": {"spp": 32}})
check("preset is a lecore.preset section the painter can list", any(s["meta"]["name"] == "soft" for s in painter.sections("lecore.preset")))

# ---- 1.7.0: the rest of the .lews contract --------------------------------------------------------------
c.get("/api/render?w=32&h=24&eye=1,2,3&target=0,0,0"); c.post("/api/env", json={"studio": "soft"})
c.post("/api/new", json={"kind": "cube"})                   # an analytic object (loaded scenes carry meshes only)
c.get("/api/scene/save"); kinds = {s["kind"] for s in backend._workspace().sections()}
check("lecore.sdf published for analytic objects", "lecore.sdf" in kinds, sorted(kinds))
check("lecore.camera published from the last render", "lecore.camera" in kinds)
env_key = backend._workspace().get("lestudio3d:env")["meta"]["env_asset"]
check("environment is a content-addressed asset, referenced (GC keeps it)", env_key not in c.post("/api/workspace/gc", json={"dry_run": True}).get_json()["dropped"])
o0 = [o["id"] for o in c.get("/api/scene").get_json()["objects"]][0]
c.post("/api/op", json={"op": "subdivide", "object": o0}); c.get("/api/scene/save")
check("edits publish as a lecore.journal", any(s["kind"] == "lecore.journal" for s in backend._workspace().sections()))
painter.put(m.lews_sdf_section(m.shape("torus").to_dsl(), sid="agent:sdf:1", name="ring"))
painter.put(m.lews_camera_section((3, 3, 3), (0, 0, 0), fov_deg=35, sid="agent:cam"))
r = c.post("/api/workspace/import", data=json.dumps({"from": "workspace"}), content_type="application/json").get_json()
ring = [o for o in c.get("/api/scene").get_json()["objects"] if o["name"] == "ring"]
check("another app's lecore.sdf imports as an object with its tree kept", ring and backend._S["objects"][ring[0]["id"]].sdf_tree is not None, r.get("error"))
check("another app's lecore.camera is adopted", r["workspace"].get("camera", {}).get("fov_deg") == 35.0, r["workspace"].get("camera"))
blob = c.get("/api/workspace/export").data
check("export is the live workspace file (Workspace.export_bytes)", blob[:2] == b"PK" and len(blob) > 10000, len(blob))
r = c.post("/api/workspace/import", data=blob)
check("a .lews file imports through Workspace.from_file (the painter's sections come back in)", r.status_code == 200 and len(r.get_json()["workspace"]["imported_meshes"]) >= 1, r.status_code)
check("our own lecore.image textures are not re-imported as foreign", all(not t["id"].startswith("lestudio3d:") for t in r.get_json()["workspace"]["textures"]))
check("lews_note records a non-section change", isinstance(c.post("/api/workspace/note", json={"kind": "selection", "data": {"object": o0}}, headers={"X-User": "moose"}).get_json().get("rev"), int))
check("lews_wait long-poll returns changes", len(c.get("/api/workspace/wait?rev=0&t=0.2").get_json()["changes"]) > 0)
inv = c.post("/api/invite", json={}).get_json()
check("invite link minted", bool(inv.get("link")) and bool(inv.get("code")), inv)
check("join from code admits a guest", c.post("/api/join", json={"link_or_code": inv["code"]}, headers={"X-User": "alice"}).get_json().get("principal") == "alice")
st = c.get("/api/workspace?since=0").get_json()
check("workspace status reports presence, kinds and assets", "presence" in st and "kinds" in st and st.get("assets") == 1, {k: st.get(k) for k in ("kinds", "assets")})
# ---- the whole engine beside the app -------------------------------------------------------------------
t = c.get("/engine/tools").get_json()
check("every engine faculty is mounted at /engine/tools", t["ok"] and len(t["tools"]) > 2000, len(t.get("tools", [])))
ref = c.get("/api/scene/doc").get_json().get("ref")
check("scene document carries a service handle", str(ref).startswith("ref:Scene:"), ref)
r = c.post("/engine/invoke", json={"name": "scene_info", "args": {"scene": ref}}).get_json()
check("any faculty runs on our scene via /engine/invoke", r["ok"] and r["result"]["n_objects"] == len(backend._S["objects"]), r)
r = c.post("/engine/capabilities/search", json={"query": "cvt remesh", "k": 3}).get_json()
check("capability search answers at /engine", r["ok"] and r["matches"], str(r)[:100])
shutil.rmtree(ROOT_WS, ignore_errors=True)
print(f"\n{len(PASS)} passed, {len(FAIL)} failed"); sys.exit(1 if FAIL else 0)
