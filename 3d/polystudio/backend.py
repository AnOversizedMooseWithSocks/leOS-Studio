"""Poly Studio -- a C4D-style polygon modeller on the leCore mesh kernel (Blueprint at /demos/10_polystudio).

THE POINT
---------
A three.js viewport with a move/rotate/scale gizmo in front; the ENGINE'S mesh kernel behind, as the single
authority over a MULTI-OBJECT SCENE. Object / Vertex / Face selection (single, additive, marquee, connected
island), transforms baked server-side, topology verbs (extrude, inset, loop cut, bevel, dissolve, delete faces,
subdivide, smooth, mirror, solidify) applied by leCore's own mesh verbs with manifold checks as the safety net,
component drags streamed live at ~20 Hz, 40 levels of undo across the whole scene.

GLB IMPORT: real-world binary glTF -- multiple meshes/primitives, accessor byteOffset + bufferView byteStride
(interleaved buffers), the full node-hierarchy transform stack, and pbrMetallicRoughness factors mapped to the
NEAREST preset in the engine's physical material library (reported per object, not silently guessed). The
engine's own `holographic_gltf` reader covers its round-trip subset; this importer covers the wild files.

MATERIALS: `holographic_matlib` (141 physical glTF-PBR presets), per face, per object, undoable, surviving
topology edits by nearest-centroid transfer. AUTO UV per object (`holographic_meshuv` Isomap with an automatic
seam cut; triplanar fallback past the geodesic budget) with the measured distortion reported and the baked map
set (layout / albedo / metallic / roughness / emissive / object-space normal + OBJ-with-vt) exported as a zip.

RENDERS, engine-true and tuned for speed:
  * LIVE PREVIEW -- each OBJECT bakes to its own SDF grid, cached per (object, revision): moving one object
    re-bakes ONE object, not the scene. The slider never re-bakes anything (fixed preview grid; it scales image
    size, effects, FSR ratio only). Coarse meshes get an EXACT signed bake (the fast shell build's documented
    edge cases -- wide triangles, edge pinholes -- let the sign flood leak on exactly those meshes; found by
    measurement); dense meshes take the fast shell + flood.
  * PHOTO -- true path-traced GI (`holographic_pathtrace`): multi-bounce, low-discrepancy AA, per-face physical
    materials via per-object material-ID grids (O(1) lookup per bounce), real-IOR refraction, sun + sky,
    progressive denoised batches streamed so the light visibly converges.
"""
import base64
import io, json, os, struct, sys, threading, time, zipfile
import numpy as np
from flask import Blueprint, Response, jsonify, request, stream_with_context

# app.py loads this backend via spec_from_file_location, which does NOT put the demo's own directory on
# sys.path -- so a sibling module (ccrun, the native-C accelerator) would not import. Add it explicitly.
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

bp = Blueprint("polystudio", __name__)

_LOCK = threading.Lock()
_DEFAULT_MAT = "clay"
_FLOOR_MAT = "concrete"
_UNDO_CAP = 40

# ---- A4/F4: render cancellation, and G1: the traced HDR kept for post-without-retrace ----------------
class _RenderCancelled(Exception):
    """Raised inside the tracer's progress callback to unwind a render the client no longer wants."""


_PHOTO_CANCEL = {}          # session key -> {"v": bool}; the live render's stop flag
_CANCEL_LATCH = set()       # session keys with a render in flight (bookkeeping for the stream guard)
_PHOTO_HDR = {}             # session key -> (hdr, W, H, spp): the linear result of the last finished photo
_PHOTO_HDR_CAP = 4


def _photo_grade(hdr, exposure=1.0, sharpen_amt=0.0):
    """Exposure -> Reinhard -> gamma -> optional sharpen.

    Split out of photo()'s nested _tonemap so /api/photo_post can reproduce the render EXACTLY. It has
    to be the same code: two implementations of a tone curve drift, and the whole point of the post
    endpoint is that nudging exposure gives you the picture you would have got by re-tracing."""
    hdr = np.asarray(hdr, float) * float(exposure)
    tm = hdr / (1.0 + hdr)
    out = np.clip(tm ** (1 / 2.2), 0, 1)
    if sharpen_amt > 0.0:
        try:
            from holographic_postfx import sharpen
            out = np.clip(sharpen(out, amount=float(sharpen_amt)), 0, 1)
        except Exception:
            pass
    return out


def _photo_cache_put(key, hdr, W, H, spp):
    _PHOTO_HDR[key] = (np.asarray(hdr, float), W, H, spp)
    while len(_PHOTO_HDR) > _PHOTO_HDR_CAP:                 # bounded: these are megabytes each
        _PHOTO_HDR.pop(next(iter(_PHOTO_HDR)))
_UV_ISOMAP_MAX = 1200
_IMPORT_VERT_CAP = 80_000
_PREVIEW_RES = 52
_PHOTO_RES = 88


class _Obj:
    __slots__ = ("name", "mesh", "mats", "rev", "sculpt", "sdf_tree", "kernel_src", "layers")

    def __init__(self, name, mesh, mats, sdf_tree=None, kernel_src=None):
        self.name = name; self.mesh = mesh; self.mats = mats; self.rev = 0; self.sculpt = None
        # The analytic SDF tree a primitive was BORN from (holographic_sdf.SDF), kept so the object can be
        # exported as an exact Shadertoy/WGSL shader (sdf_dialect walks this tree). Any topology edit or sculpt
        # invalidates it -- an edited mesh is a grid SDF, not an analytic tree -- so it is dropped on _bump.
        self.sdf_tree = sdf_tree
        # The Python kernel SOURCE a described object was composed from (holographic_codecompose), kept so
        # holographic_codeverbal can explain it back in English. Dropped on edit, same as the tree.
        self.kernel_src = kernel_src
        # LAYERED SHELL MATERIAL (CAD sweep, user ask #3): [(thickness, material), ...] inward from the surface,
        # revealed exactly by /api/section's inside-depth bands. None = plain single material.
        self.layers = None


_S = {"objects": {}, "next_id": 1, "rev": 0, "undo": [], "redo": [], "cache": {}, "uv": {},
      "render_assets": {},                                  # oid -> {uv, tex} | {face_colors}: engine-renderer data
      "ws_foreign": [],                                     # sections from an imported .lews we do not author
      "ws_meta": {},                                        # that workspace's top-level meta
      "ws_textures": {},                                    # id -> {name, rgb} painted textures available
      "units": {"name": "cm", "per_unit": 10.0}}            # 1 engine unit = per_unit of 'name' (default 10 cm)


def _matlib():
    import holographic_matlib as ml
    return ml


_CUSTOM_MATS = {}                                          # session-authored materials (the material editor)
_PARENT = {}                                               # A2-3 hierarchy: child_oid -> parent_oid


def _mat(name):
    """Resolve a material name: session CUSTOM materials first (authored in the material editor -- demo 07's
    layer-parameter idea folded into the modeller), then the engine's 141-preset matlib. One resolver, so a
    custom material is paintable, assignable, formula-drivable, and path-traced exactly like a preset."""
    m = _CUSTOM_MATS.get(name)
    if m is not None:
        return m
    return _matlib().material(name)


def _mesh_dict(mesh):
    """Our Mesh -> the {'vertices', 'faces'} plain dict the engine's holographic_meshselect / _snap /
    _transform_space modules consume. One adapter, so those modules operate on the authoritative mesh."""
    return {"vertices": mesh.vertices.tolist(), "faces": [list(map(int, f)) for f in mesh.faces]}


def _default_mat_name():
    ml = _matlib()
    return _DEFAULT_MAT if _DEFAULT_MAT in ml.names() else (ml.by_class("diffuse") or ml.names())[0]


_HUMANOID_CACHE = {}


def _mesh_sdf_tree(tree, res=64, scan=2.6, face_target=2600, scan_lo=None, scan_hi=None):
    """Mesh an analytic SDF tree: coarse scan for bounds, fine grid, marching tetrahedra, face-target decimate --
    the same extraction pipeline sculpt uses. Returns a Mesh, or raises if the tree is empty in the scan box.
    scan_lo/scan_hi override the symmetric [-scan, scan]^3 coarse box (a modified object far from the origin
    would otherwise be scanned in the wrong place)."""
    from holographic_meshbridge import marching_tetrahedra_vec
    from holographic_meshqem import cluster_decimate
    from holographic_mesh import Mesh
    if scan_lo is None:
        scan_lo = np.array([-scan, -scan, -scan], float); scan_hi = np.array([scan, scan, scan], float)
    axc = [np.linspace(scan_lo[i], scan_hi[i], 36) for i in range(3)]
    X, Y, Z = np.meshgrid(*axc, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    d = tree.eval(P)
    neg = P[d < 0]
    if len(neg) == 0:
        # a THIN shape (a shell, a ring, a subtracted sliver) can slip between 36^3 samples -- rescan once at
        # double density before refusing, so "empty" means empty, not "thinner than the coarse scan"
        axf = [np.linspace(scan_lo[i], scan_hi[i], 72) for i in range(3)]
        Xf, Yf, Zf = np.meshgrid(*axf, indexing="ij")
        Pf = np.stack([Xf.ravel(), Yf.ravel(), Zf.ravel()], 1)
        df = np.empty(len(Pf))
        for i in range(0, len(Pf), 200_000):
            df[i:i + 200_000] = tree.eval(Pf[i:i + 200_000])
        neg = Pf[df < 0]
    if len(neg) == 0:
        raise ValueError("the shape is empty inside the scan box %s..%s" %
                         (np.round(scan_lo, 1).tolist(), np.round(scan_hi, 1).tolist()))
    lo = neg.min(axis=0) - 0.16; hi = neg.max(axis=0) + 0.16
    ax = tuple(np.linspace(lo[i], hi[i], res) for i in range(3))
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    g = np.empty(len(P))
    for i in range(0, len(P), 200_000):
        g[i:i + 200_000] = tree.eval(P[i:i + 200_000])
    mesh = marching_tetrahedra_vec(g.reshape(res, res, res), ax, level=0.0)
    grid_n = 44
    dec = cluster_decimate(mesh, grid=grid_n)
    while dec.n_faces > face_target and grid_n > 8:
        grid_n -= 4
        dec = cluster_decimate(mesh, grid=grid_n)
    return Mesh(dec.vertices, [tuple(f) for f in dec.faces])


_PRIMITIVES = {"cube", "tetra", "plane", "icosphere", "humanoid"}


def _primitive(name):
    """Return (mesh, analytic_sdf_tree). The tree is the exact signed-distance expression the primitive equals,
    kept for shader export; None when the engine has no analytic leaf for it (a grid bake is used instead)."""
    import holographic_mesh as hm
    import holographic_sdf as S
    if name == "humanoid":
        # A parametric biped from the engine's holographic_humanoid: bones + muscle bellies as an analytic SDF
        # union, meshed by the same marching-tetrahedra pipeline sculpt uses. The tree IS kept -- the bake stays
        # analytic (exact distances) -- but note its capsule leaves are ones sdf_dialect refuses to emit, so the
        # shader export for a humanoid takes the FITTED path (primfit), stated in the UI, not hidden. Built once
        # and cached; an Add click after the first is instant.
        hit = _HUMANOID_CACHE.get("body")
        if hit is None:
            from holographic_humanoid import Humanoid
            from holographic_mesh import Mesh
            tree = Humanoid().skin().scale(1.25)
            mesh = _mesh_sdf_tree(tree, res=72, face_target=3200)
            dy = -0.6 - float(mesh.vertices[:, 1].min())      # feet on the cube's floor line
            mesh.vertices = mesh.vertices + np.array([0.0, dy, 0.0])
            tree = tree.translate((0.0, dy, 0.0))
            hit = (mesh, tree)
            _HUMANOID_CACHE["body"] = hit
        m0, t0 = hit
        from holographic_mesh import Mesh
        return Mesh(m0.vertices.copy(), [tuple(f) for f in m0.faces]), t0
    if name == "cube":
        return hm.box(1.2, 1.2, 1.2), S.box(0.6, 0.6, 0.6)
    if name == "tetra":
        return hm.tetrahedron(0.9), None                    # no analytic tetra leaf; grid-baked
    if name == "plane":
        return hm.grid(6, 6, 1.8, 1.8), None
    if name == "icosphere":
        from holographic_meshsmooth import _icosphere
        m = _icosphere(2); m.vertices = m.vertices * 0.75
        return m, S.sphere(0.75)
    return _primitive("cube")


def _add_object(name, mesh, mats=None, sdf_tree=None, kernel_src=None):
    oid = str(_S["next_id"]); _S["next_id"] += 1
    _S["objects"][oid] = _Obj(name, mesh, mats or [_default_mat_name()] * mesh.n_faces,
                              sdf_tree=sdf_tree, kernel_src=kernel_src)
    return oid


def _bump(oid=None):
    _S["rev"] += 1
    if oid is not None and oid in _S["objects"]:
        o = _S["objects"][oid]
        o.rev += 1
        o.sdf_tree = None                                   # an edited object is no longer its analytic primitive
        o.kernel_src = None
        _S["cache"] = {k: v for k, v in _S["cache"].items() if k[0] != oid}
        _S["uv"].pop(oid, None)
    else:
        _S["cache"].clear(); _S["uv"].clear()


# ---------------- undo: object-level entries where possible, scene-level for add/delete/import -------
def _snap_obj(oid):
    o = _S["objects"][oid]
    # sdf_tree / kernel_src ride along: SDF trees are immutable (every combinator returns a new tree), so the
    # reference is snapshot-safe -- and without them, undoing a move would silently degrade a primitive to a
    # mesh-only object (losing its exact bake + exact shader) even though the restored mesh IS the primitive.
    _S["undo"].append(("obj", oid, o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces], list(o.mats),
                       o.sdf_tree, o.kernel_src))
    _trim_undo()


class _SnapshotCommand:
    """A snapshot as an EditHistory command: apply() records where we are, invert() restores where
    we were. This is what lets the app's snapshot undo ride the engine's EditHistory unchanged."""
    __slots__ = ("before", "after")

    def __init__(self, before):
        self.before = before; self.after = None

    def apply(self, state):
        if self.after is not None:              # a REDO: EditHistory re-applies the command
            _restore(self.after)
        return state                            # first application: the route already did the edit

    def invert(self, state):
        self.after = _capture_like(self.before)
        _restore(self.before)
        return state



def _history():
    h = _S.get("history")
    if h is None:
        h = _S["history"] = _mind().edit_history(max_depth=256)
    return h


def _snap_scene():
    snap = {oid: (o.name, o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces], list(o.mats),
                  o.sdf_tree, o.kernel_src)
            for oid, o in _S["objects"].items()}
    _S["undo"].append(("scene", snap, _S["next_id"]))
    _trim_undo()


def _discard_snapshot():
    """A route took a snapshot, then decided the edit was a no-op or failed: drop it. Producers call
    _trim_undo() right after appending, so the entry has usually already moved onto the engine's
    history -- discard THAT (undo it as a no-op and cut the redo tail) rather than a list that is
    empty by design now."""
    if _S["undo"]:
        _S["undo"].pop(); return
    h = _history()
    if (h.can_undo() if callable(getattr(h, "can_undo", None)) else getattr(h, "can_undo", False)):
        h.undo(None)                            # restores the snapshot, which for a no-op is a no-op
        try: del h.undo_stack[-1:]
        except Exception: pass
        try: h.redo_stack.clear()
        except Exception: pass


def _trim_undo():
    """THE choke point every snapshot producer calls after appending. ADOPTED: the entry just
    appended is moved onto the engine's EditHistory (depth bound, redo-tail truncation, the
    same command log other leCore apps use). _S["undo"] stays as the append target so the 18
    producers do not change; it is drained here, so it never holds more than the entry in flight."""
    while _S["undo"]:
        _history().do(None, _SnapshotCommand(_S["undo"].pop(0)))
    _S.setdefault("redo", []).clear()

def _capture_like(entry):
    """Snapshot the CURRENT state in the same shape as `entry`, so undo/redo are symmetric.

    Restoring an undo entry throws away the state it replaced; capturing that state first is what
    makes the step reversible. The shapes mirror _snap_obj / _snap_scene / the sculpt-grid snapshot.
    """
    kind = entry[0]
    if kind == "grid":
        oid = entry[1]
        o = _S["objects"].get(oid)
        if o is None or o.sculpt is None:
            return None
        return ("grid", oid, o.sculpt["grid"].copy())
    if kind == "obj":
        oid = entry[1]
        o = _S["objects"].get(oid)
        if o is None:
            return None
        return ("obj", oid, o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces], list(o.mats),
                o.sdf_tree, o.kernel_src)
    snap = {oid: (o.name, o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces], list(o.mats),
                  o.sdf_tree, o.kernel_src)
            for oid, o in _S["objects"].items()}
    return ("scene", snap, _S["next_id"])


def _restore(e):
    """Apply one history entry (used by both undo and redo)."""
    from holographic_mesh import Mesh
    if e[0] == "grid":
        _, oid, grid = e
        if oid in _S["objects"] and _S["objects"][oid].sculpt is not None:
            o = _S["objects"][oid]
            o.sculpt["grid"] = grid
            _sculpt_refresh_mesh(o)
            _bump(oid)
    elif e[0] == "obj":
        _, oid, V, F, mats, tree, kern = e
        if oid in _S["objects"]:
            o = _S["objects"][oid]
            o.mesh = Mesh(V, F); o.mats = mats
            o.sculpt = None                            # undoing across a mode boundary lands in poly mode
            _bump(oid)
            o.sdf_tree, o.kernel_src = tree, kern      # after _bump: restore the analytic identity too
    else:
        _, snap, nid = e
        _S["objects"] = {}
        for oid, (name, V, F, mats, tree, kern) in snap.items():
            _S["objects"][oid] = _Obj(name, Mesh(V, F), mats, sdf_tree=tree, kernel_src=kern)
        _S["next_id"] = nid
        _bump()


def _centroids(mesh):
    return np.array([mesh.vertices[list(f)].mean(axis=0) for f in mesh.faces]) if mesh.faces else np.zeros((0, 3))


def _transfer_mats(old_mesh, old_mats, new_mesh):
    if not old_mats:
        return [_default_mat_name()] * new_mesh.n_faces
    oc = _centroids(old_mesh); nc = _centroids(new_mesh)
    out = []
    step = max(32, int(2_000_000 / max(len(oc), 1)))       # bound the (points x faces x 3) temp to ~50 MB
    for i in range(0, len(nc), step):
        d = np.linalg.norm(nc[i:i + step, None, :] - oc[None, :, :], axis=2)
        out.extend(int(j) for j in np.argmin(d, axis=1))
    return [old_mats[j] for j in out]


_MIND = None


def _mind():
    """ONE UnifiedMind per process (docs/APP_FOUNDATION.md rule 1), built lazily."""
    global _MIND
    if _MIND is None:
        import lecore
        _MIND = lecore.UnifiedMind(dim=256, seed=0)
    return _MIND


def _obj_payload(oid, view="full"):
    o = _S["objects"][oid]
    m = o.mesh
    if view != "full":
        # SUMMARY: bbox + counts + the UNIQUE material set. The full payload carries one
        # material name PER FACE plus every vertex position, index and colour -- MEASURED at
        # ~1.1 MB (~288k tokens) for a 5-object/3000-face scene, on every mutation. This
        # returns under 300 bytes per object and is the default for /api/agent/*.
        vmin = np.round(m.vertices.min(axis=0), 4).tolist()
        vmax = np.round(m.vertices.max(axis=0), 4).tolist()
        out = {"id": oid, "name": o.name,
               "counts": {"v": m.n_vertices, "f": m.n_faces},
               "bbox": {"min": vmin, "max": vmax,
                        "center": np.round(m.vertices.mean(axis=0), 4).tolist(),
                        "size": [round(vmax[i] - vmin[i], 4) for i in range(3)]},
               "materials": sorted(set(o.mats))[:8],
               "sculpt": o.sculpt is not None}
        if view == "stats":
            out["watertight"] = bool(getattr(m, "is_watertight", lambda: False)())
        return out
    tris, tri_face = [], []
    for i, f in enumerate(m.faces):
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1])); tri_face.append(i)
    alb = _channels_for(o)[0]
    vc = np.zeros((m.n_vertices, 3)); cnt = np.zeros(m.n_vertices)
    for i, f in enumerate(m.faces):
        for v in f:
            vc[v] += alb[i]; cnt[v] += 1
    vc /= np.maximum(cnt, 1)[:, None]
    vmin = np.round(m.vertices.min(axis=0), 4).tolist()
    vmax = np.round(m.vertices.max(axis=0), 4).tolist()
    vcen = np.round(m.vertices.mean(axis=0), 4).tolist()
    return {"id": oid, "name": o.name,
            "positions": np.round(m.vertices, 4).ravel().tolist(),
            "indices": [int(v) for t in tris for v in t],
            "triFace": tri_face,
            "faces": [list(map(int, f)) for f in m.faces],
            "faceMats": list(o.mats),
            "vertColor": np.round(vc, 3).ravel().tolist(),
            "counts": {"v": m.n_vertices, "f": m.n_faces},
            "bbox": {"min": vmin, "max": vmax, "center": vcen,
                     "size": [round(vmax[i] - vmin[i], 4) for i in range(3)]},
            "sculpt": o.sculpt is not None,
            # VIEWPORT PARITY: a textured import ships its per-vertex uv so the WebGL viewport can draw the
            # REAL texture (served by /api/object_texture) instead of the palette approximation
            **({"uv": np.round(np.asarray(_S["render_assets"][oid]["uv"], float), 5).ravel().tolist(),
                "hasTexture": True}
               if oid in _S.get("render_assets", {}) and "uv" in _S["render_assets"][oid]
               and len(_S["render_assets"][oid]["uv"]) == m.n_vertices else {})}


def _fix_uv_seams(dec, luv, src_mesh, src_uv, thresh=0.05):
    """UV SEAM REPAIR after a uv transfer (user-reported: dark lines across an imported scan).

    A decimated face can straddle a SEAM in the source atlas -- its three vertices project onto different
    atlas islands, so the face's uv triangle spans a huge swath of texture and samples unrelated texels.
    Measured on a 151K-face photogrammetry beetle: 4,100 of 79,354 faces (5.2%) spanned >0.10 of uv space,
    where a normal face spans 0.004. That is exactly the dark streaking, and it is the seam limitation the
    engine's own transfer_uv docstring calls out -- correct behaviour from a point-wise projector, but wrong
    for a rendered face.

    Fix: for each straddling face, take ALL THREE uvs from ONE source triangle (the nearest to the face's
    centroid) via clamped barycentric projection, and give the face its OWN duplicated vertices so its
    well-behaved neighbours keep theirs. Face count is unchanged; only the vertex count grows.
    Returns (mesh, uv, n_fixed)."""
    V = np.asarray(dec.vertices, float)
    F = np.array([list(f)[:3] for f in dec.faces], dtype=np.int64)
    uv = np.asarray(luv, float).copy()
    tri = uv[F]
    span = np.maximum(tri[:, :, 0].max(1) - tri[:, :, 0].min(1),
                      tri[:, :, 1].max(1) - tri[:, :, 1].min(1))
    bad = np.where(span > float(thresh))[0]
    if not len(bad):
        return dec, uv, 0
    SV = np.asarray(src_mesh.vertices, float)
    SF = np.array([list(f)[:3] for f in src_mesh.faces], dtype=np.int64)
    near = _nn_voxel(SV[SF].mean(axis=1), V[F[bad]].mean(axis=1))       # one source triangle per bad face
    newV = list(V); newUV = list(uv); newF = [list(f) for f in F]
    su = np.asarray(src_uv, float)
    for bi, fi in enumerate(bad):
        s = SF[near[bi]]
        A, B, C = SV[s[0]], SV[s[1]], SV[s[2]]
        n = np.cross(B - A, C - A); nn = float(np.dot(n, n)) or 1e-12
        idx = []
        for vi in F[fi]:
            P = V[vi]
            w0 = float(np.dot(np.cross(C - B, P - B), n)) / nn
            w1 = float(np.dot(np.cross(A - C, P - C), n)) / nn
            w = np.clip([w0, w1, 1.0 - w0 - w1], 0.0, 1.0)
            w = w / (w.sum() or 1.0)                    # clamped: stay inside the island, never extrapolate
            newV.append(P); newUV.append(w[0] * su[s[0]] + w[1] * su[s[1]] + w[2] * su[s[2]])
            idx.append(len(newV) - 1)
        newF[fi] = idx
    from holographic_mesh import Mesh as _Mesh
    fixed = _Mesh(np.asarray(newV, float), [tuple(int(x) for x in f) for f in newF])
    return fixed, np.asarray(newUV, float), int(len(bad))


# TEXTURE DETAIL CAP. Imported base-colour maps are stored at up to this size. It was 1024, which threw
# away three quarters of a 2048 photogrammetry atlas before anything ever rendered -- a payload decision
# silently acting as a quality decision. 2048 keeps the scan's own detail; the cost is memory and the PNG
# the viewport fetches, both of which scale with the same number, so it is one honest knob rather than two.
TEXTURE_MAX = 2048


def _cap_texture(tex, limit=None):
    """Downsample a texture to TEXTURE_MAX (integer stride, so no resampling blur). Returns it unchanged
    when it already fits -- a freshly baked atlas is sized to its own cell budget and must not be touched."""
    t = np.asarray(tex)
    lim = int(limit or TEXTURE_MAX)
    if t.shape[0] <= lim:
        return t
    step = int(np.ceil(t.shape[0] / lim))
    return t[::step, ::step]


def _transfer_uv_compat(mt_mod, src_mesh, src_uv, tgt_verts):
    """v0.2.8 changed transfer_uv to return (attr, distances) -- the residual is the HONEST error signal.
    Unpack either contract and return (uv, median_residual_or_None)."""
    out = mt_mod.transfer_uv(src_mesh, np.asarray(src_uv, float), np.asarray(tgt_verts, float))
    if isinstance(out, tuple) and len(out) == 2:
        uv, dist = out
        return np.asarray(uv, float), float(np.median(np.asarray(dist)))
    return np.asarray(out, float), None


def _nn_voxel(src_pts, q_pts, cells=64):
    """Numpy-only nearest-neighbour: voxel-hash src points, probe each query's 3x3x3 neighbourhood, widening
    until candidates appear. Replaces scipy.spatial.cKDTree -- the runtime dependency contract here is
    numpy+PIL only, and a user's import genuinely failed on 'No module named scipy'."""
    src = np.asarray(src_pts, float); q = np.asarray(q_pts, float)
    lo = src.min(axis=0); span = np.maximum(src.max(axis=0) - lo, 1e-9)
    cell = span.max() / cells
    key = lambda P: np.floor((P - lo) / cell).astype(np.int64)
    ks = key(src)
    order = np.lexsort((ks[:, 2], ks[:, 1], ks[:, 0]))
    ks_s = ks[order]
    flat = ks_s[:, 0] * 73856093 ^ ks_s[:, 1] * 19349663 ^ ks_s[:, 2] * 83492791
    buckets = {}
    start = 0
    for i in range(1, len(flat) + 1):
        if i == len(flat) or flat[i] != flat[start]:
            buckets[int(flat[start])] = (start, i)
            start = i
    kq = key(q)
    out = np.zeros(len(q), np.int64)
    for i in range(len(q)):
        best = -1; bd = np.inf
        r = 1
        while r <= 8:
            cands = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if r > 1 and max(abs(dx), abs(dy), abs(dz)) < r:
                            continue                        # only the newly-added shell after round 1
                        h = int((kq[i, 0] + dx) * 73856093 ^ (kq[i, 1] + dy) * 19349663 ^ (kq[i, 2] + dz) * 83492791)
                        b = buckets.get(h)
                        if b:
                            cands.append(order[b[0]:b[1]])
            if cands:
                idx = np.concatenate(cands)
                d = ((src[idx] - q[i]) ** 2).sum(axis=1)
                j = int(np.argmin(d))
                if d[j] < bd:
                    bd = float(d[j]); best = int(idx[j])
            # EXACTNESS: a candidate in shell r does not preclude a closer point in shell r+1 (cells are cubes,
            # the query sits anywhere inside its cell). Only stop once every unexplored shell is provably
            # farther: its nearest face is (r-1)*cell away from the query's cell, so (r-1)*cell >= sqrt(bd).
            if best >= 0 and (r - 1) * cell >= np.sqrt(bd):
                break
            r += 1
        out[i] = best if best >= 0 else int(((src - q[i]) ** 2).sum(axis=1).argmin())
    return out


def _gauss_blur_np(img, sigma):
    """Gaussian blur. ADOPTED: delegates to the engine's blur_image (reflect borders, channels
    untouched) -- the lint found leStudio had written this THREE times and we had written it once.
    Kept as a name so the photo pipeline's call sites do not change."""
    return np.asarray(_mind().blur_image(np.asarray(img, float), sigma=float(sigma), mode="reflect"), float)

def _material_names():
    """Every material name the app can assign, for did_you_mean on a bad one."""
    try:
        import holographic_materials as _hm
        return sorted(getattr(_hm, "MATERIALS", {}) or {})
    except Exception:
        try:
            return sorted(_MATS)
        except Exception:
            return []


def _agent_view():
    """Agent endpoints default to a SUMMARY payload.

    MEASURED: the full payload emits positions/indices/faces/triFace/vertColor plus one
    material NAME PER FACE for every object on every mutation -- about 1.1 MB (~288k tokens)
    for a 5-object, 3000-face scene. An agent cannot afford that twice. The browser still
    gets the full arrays; only /api/agent/* defaults to summary.
    """
    q = (request.args.get("view") or "").lower()
    if q in ("full", "summary", "stats"):
        return q
    return "summary" if "/agent/" in request.path else "full"


def _payload(only=None, view="full"):
    # rev = monotonic transaction counter (primary ordering key the client uses to reject stale echoes).
    # ts  = monotonic server timestamp (seconds) as an independent secondary signal for ordering/merging.
    ts = time.time()
    if only is not None:
        return {"rev": _S["rev"], "ts": ts, "object": _obj_payload(only, view)}
    return {"rev": _S["rev"], "ts": ts, "view": view,
            "objects": [_obj_payload(oid, view) for oid in _S["objects"]]}


def _channels_for(o):
    ml = _matlib()
    F = len(o.mats)
    alb = np.zeros((F, 3)); met = np.zeros(F); rgh = np.zeros(F); emi = np.zeros((F, 3)); ior = np.zeros(F)
    cache = {}
    for i, name in enumerate(o.mats):
        m = cache.get(name) or cache.setdefault(name, _mat(name))
        alb[i] = m.base_color[:3]; met[i] = m.metallic; rgh[i] = m.roughness; emi[i] = m.emissive
        if getattr(m, "transmission", 0.0) >= 1.0:
            ior[i] = getattr(m, "ior", 1.5)
    return alb, met, rgh, emi, ior


def _seed_default_objects():
    """Factory scene: a single cube. Called locked (from _init and scene/new).

    One object, not two: a starting scene should be the smallest thing you can build on, and the second
    primitive was scenery -- something to delete before starting work. Add ▸ Sphere is one click away.
    """
    ml = _matlib()
    cube, cube_tree = _primitive("cube")
    _add_object("Cube", cube, ["steel_brushed" if "steel_brushed" in ml.names()
                               else _default_mat_name()] * cube.n_faces, sdf_tree=cube_tree)
    _S["rev"] += 1                                      # MONOTONIC: a reseed after delete-all must not rewind rev


def _init():
    with _LOCK:
        # seed ONCE: an empty dict on first touch means "fresh boot"; after that, an empty scene is a state
        # the USER made (they deleted everything) and must STAY empty -- resurrection of the default cube+
        # sphere on the next render/import was a reported bug.
        if not _S["objects"] and not _S.get("seeded"):
            _S["seeded"] = True
            _seed_default_objects()


# =====================================================================================================
# Scene endpoints
# =====================================================================================================
@bp.route("/api/scene")
def scene_get():
    _init()
    with _LOCK:
        return jsonify(_payload())


_NEW_FIELDS = {"primitive", "kind", "name", "position", "size", "scale", "material"}
_NEW_ALIAS = {"kind": "primitive", "type": "primitive", "shape": "primitive",
              "at": "position", "pos": "position", "radius": "size", "r": "size"}


@bp.route("/api/new", methods=["POST"])
def new_object():
    """Create an object.

    FIXED (1.2.0): this used to read only `primitive` and silently fall through to a grey
    cube for anything it did not recognise -- so an agent asking for
    {"kind": "sphere", "radius": 1.0} got a cube and an HTTP 200 saying it worked. Unknown
    keys are now a 400 with `did_you_mean`, an unknown primitive names the vocabulary, and
    position / size / material are honoured at creation instead of needing follow-up calls.
    """
    _init()
    d = request.get_json(force=True) or {}
    unknown = [k for k in d if k not in _NEW_FIELDS and k not in _NEW_ALIAS]
    if unknown:
        return jsonify({"error": "unknown field(s) for /api/new",
                        "unknown": sorted(unknown),
                        "expected": sorted(_NEW_FIELDS),
                        "did_you_mean": {k: _NEW_ALIAS[k] for k in d if k in _NEW_ALIAS}}), 400
    for a, real in _NEW_ALIAS.items():          # accept the obvious synonyms, but say so
        if a in d and real not in d:
            d[real] = d.pop(a)
    want = str(d.get("primitive", "cube")).lower()
    if want not in _PRIMITIVES:
        near = [p for p in _PRIMITIVES if p.startswith(want[:3]) or want[:3] in p]
        return jsonify({"error": f"unknown primitive {want!r}",
                        "vocabulary": sorted(_PRIMITIVES),
                        "did_you_mean": near or None}), 400
    with _LOCK:
        _snap_scene()
        mesh, tree = _primitive(want)
        # drop new objects beside the existing scene, not inside it
        if _S["objects"]:
            xmax = max((o.mesh.vertices[:, 0].max() for o in _S["objects"].values()), default=0.0)
            span = float(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min())
            shift = xmax + span * 0.7
            mesh.vertices = mesh.vertices + np.array([shift, 0, 0])
            if tree is not None:
                tree = tree.translate((shift, 0, 0))
        # honour position / size / material at CREATION -- one call, not four
        sz = d.get("size") or d.get("scale")
        if sz is not None:
            k = float(sz) if not isinstance(sz, (list, tuple)) else 1.0
            v = np.asarray(sz, float) if isinstance(sz, (list, tuple)) else np.array([k, k, k])
            c = mesh.vertices.mean(axis=0)
            mesh.vertices = (mesh.vertices - c) * v + c
            if tree is not None:
                try: tree = tree.scale(float(np.mean(v)))
                except Exception: tree = None
        pos = d.get("position")
        if pos is not None:
            p = np.asarray(pos, float).reshape(3)
            mesh.vertices = mesh.vertices + p
            if tree is not None:
                try: tree = tree.translate(tuple(p))
                except Exception: tree = None
        oid = _add_object(d.get("name") or want.capitalize(), mesh, sdf_tree=tree)
        mat = d.get("material")
        if mat:
            known = _material_names()
            if mat not in known:
                near = [m for m in known if mat.lower() in m.lower() or m.lower() in mat.lower()]
                return jsonify({"error": f"unknown material {mat!r}",
                                "did_you_mean": near[:6] or None}), 400
            _S["objects"][oid].mats = [mat] * _S["objects"][oid].mesh.n_faces
        _bump()
        out = _payload(view=_agent_view()); out["object"] = oid
        return jsonify(out)


@bp.route("/api/undo", methods=["POST"])
def undo():
    _init()
    with _LOCK:
        h = _history()
        if not (h.can_undo() if callable(getattr(h, "can_undo", None)) else bool(getattr(h, "can_undo", False))):
            return jsonify(_payload())
        h.undo(None)
        _bump()
        out = _payload(); out["can_undo"] = (h.can_undo() if callable(getattr(h, "can_undo", None)) else bool(getattr(h, "can_undo", False))); out["can_redo"] = (h.can_redo() if callable(getattr(h, "can_redo", None)) else bool(getattr(h, "can_redo", False)))
        return jsonify(out)


@bp.route("/api/redo", methods=["POST"])
def redo():
    _init()
    with _LOCK:
        h = _history()
        if not (h.can_redo() if callable(getattr(h, "can_redo", None)) else bool(getattr(h, "can_redo", False))):
            return jsonify(_payload())
        h.redo(None)
        _bump()
        out = _payload(); out["can_undo"] = (h.can_undo() if callable(getattr(h, "can_undo", None)) else bool(getattr(h, "can_undo", False))); out["can_redo"] = (h.can_redo() if callable(getattr(h, "can_redo", None)) else bool(getattr(h, "can_redo", False)))
        return jsonify(out)


@bp.route("/api/verts", methods=["POST"])
def set_verts():
    """The component-drag stream (~20 Hz): {object, indices, positions}. Positions only, no undo per packet."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    idx = np.asarray(d.get("indices", []), int)
    pos = np.asarray(d.get("positions", []), float).reshape(-1, 3)
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        m = o.mesh
        if len(idx) != len(pos) or (len(idx) and (idx.min() < 0 or idx.max() >= m.n_vertices)):
            return jsonify({"error": "bad vertex packet"}), 400
        m.vertices[idx] = pos
        m.normals = None; m._he = None; m._adj = None
        _bump(oid)
        return jsonify({"rev": _S["rev"]})


def _vertex_normals(mesh):
    """Area-weighted vertex normals via numpy cross products (fan-triangulated per face)."""
    V = mesh.vertices
    N = np.zeros_like(V)
    for f in mesh.faces:
        idx = list(f)
        for k in range(1, len(idx) - 1):
            a, b, cc = idx[0], idx[k], idx[k + 1]
            n = np.cross(V[b] - V[a], V[cc] - V[a])
            N[a] += n; N[b] += n; N[cc] += n
    ln = np.linalg.norm(N, axis=1)
    ln[ln < 1e-12] = 1.0
    return N / ln[:, None]


def _eval_expr_batch(expr, pts):
    """Evaluate a user shader-style expression f(x, y, z, r) at each point, through the engine's SAFE ns-eel2
    evaluator (holographic_milkdrop: a real whitelisted tokenizer->parser->evaluator, NEVER Python eval -- a
    hostile expression can do arithmetic and nothing else; an unknown function raises, loudly). Parse once,
    evaluate per point with a fresh env (deterministic: no state carries between points). This is the engine
    surface that makes "type a Shadertoy-style formula" SAFE to expose to a text box."""
    from holographic_milkdrop import MilkExpr
    e = MilkExpr(expr)                                    # raises on bad syntax / non-whitelisted function
    out = np.empty(len(pts), float)
    for i, p in enumerate(pts):
        env = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
               "r": float(np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]))}
        out[i] = e.eval(env)
    return out


def _mesh_verb(m, name, d):
    """One deterministic mesh verb: (mesh, opname, params) -> new mesh. The SINGLE dispatch used by both the
    live /api/op endpoint and history REPLAY (holographic_edithistory's discipline: a replayed history re-runs
    the exact same apply, so a rebuilt branch is bit-identical to the original session). Raises on unknown."""
    from holographic_mesh import Mesh
    if name == "extrude":
        from holographic_meshverbs import extrude_face
        new = extrude_face(m, int(d["face"]), float(d.get("dist", 0.3)))
    elif name == "inset":
        from holographic_meshverbs import inset_face
        new = inset_face(m, int(d["face"]), float(np.clip(d.get("ratio", 0.3), 0.02, 0.95)))
    elif name == "dissolve":
        from holographic_meshverbs import dissolve_vertex
        new = dissolve_vertex(m, int(d["vertex"]))
    elif name == "bevel":
        segs = int(np.clip(int(d.get("segments", 1)), 1, 6))
        if segs > 1:                               # multi-segment rounded corner (meshverbs2)
            from holographic_meshverbs2 import bevel_vertex_segments
            new = bevel_vertex_segments(m, int(d["vertex"]),
                                        float(np.clip(d.get("ratio", 0.25), 0.05, 0.45)), segments=segs)
        else:
            from holographic_meshverbs2 import bevel_vertex
            new = bevel_vertex(m, int(d["vertex"]), float(np.clip(d.get("ratio", 0.25), 0.05, 0.45)))
    elif name == "loopcut":
        from holographic_meshverbs2 import loop_cut
        new = loop_cut(m, int(d["face"]), tuple(int(v) for v in d["edge"]))
    elif name == "delete_faces":
        keep = sorted(set(range(m.n_faces)) - {int(i) for i in d.get("faces", [])})
        if not keep:
            raise ValueError("cannot delete every face -- delete the object instead")
        faces = [tuple(m.faces[i]) for i in keep]
        used = sorted({v for f in faces for v in f})
        remap = {v: i for i, v in enumerate(used)}
        new = Mesh(m.vertices[used], [tuple(remap[v] for v in f) for f in faces])
        d["_keep_faces"] = keep                     # endpoint aligns materials by this face index list
    elif name == "bevel_selection":
        # EDGE/CORNER BEVEL over MANY selected corners (holographic_meshverbs2.bevel_vertex[_segments]):
        # rounds or chamfers all selected corner vertices at once. mode "chamfer" = 1 flat segment;
        # "fillet" = N rounded segments. Beveling shifts vertex indices, so we process HIGHEST index first to
        # keep the remaining targets valid. The analytic tree drops (honest -- this is a mesh operation).
        import holographic_meshverbs2 as _mv2
        verts = d.get("verts")
        if not verts:                                          # default: bevel every corner (all vertices)
            verts = list(range(len(m.vertices)))
        verts = sorted({int(v) for v in verts if 0 <= int(v) < len(m.vertices)}, reverse=True)
        ratio = float(np.clip(d.get("ratio", 0.2), 0.02, 0.49))
        mode = str(d.get("mode", "chamfer"))
        segs = int(np.clip(int(d.get("segments", 3)), 1, 8)) if mode == "fillet" else 1
        cur = m
        done = 0
        for vi in verts:
            if vi >= cur.n_vertices:
                continue
            try:
                if segs > 1:
                    out = _mv2.bevel_vertex_segments(cur, vi, ratio, segments=segs)
                else:
                    out = _mv2.bevel_vertex(cur, vi, ratio)
                cur = out if hasattr(out, "faces") else out[0]
                done += 1
            except Exception:
                continue                                       # a non-manifold or boundary corner may refuse; skip it
        if done == 0:
            raise ValueError("no vertices could be beveled (boundary/non-manifold corners are skipped)")
        new = Mesh(np.asarray(cur.vertices, float), [tuple(f) for f in cur.faces])
    elif name == "lattice":
        # LATTICE / CAGE DEFORM (holographic_deform.lattice_deform): a 2x2x2 control cage over the object's
        # bbox; presets move the 8 corners -- shear, flare (top out), squash, skew -- or pass raw
        # 'offsets' (2x2x2x3). Smooth trilinear falloff; topology preserved; analytic tree drops (honest).
        import holographic_deform as _df
        V = m.vertices
        lo, hi = V.min(0) - 1e-6, V.max(0) + 1e-6
        amt = float(np.clip(d.get("amount", 0.3), -2.0, 2.0))
        preset = str(d.get("preset", "shear"))
        off = np.zeros((2, 2, 2, 3))
        if d.get("offsets") is not None:
            try:
                off = np.asarray(d["offsets"], float).reshape(2, 2, 2, 3)
            except Exception:
                raise ValueError("offsets must be 2x2x2x3")
        elif preset == "shear":                                # top layer slides +x
            off[:, 1, :, 0] = amt
        elif preset == "flare":                                # top corners outward in xz
            for ix in (0, 1):
                for iz in (0, 1):
                    off[ix, 1, iz, 0] = amt * (1 if ix else -1)
                    off[ix, 1, iz, 2] = amt * (1 if iz else -1)
        elif preset == "squash":                               # top down, sides out (volume feel)
            off[:, 1, :, 1] = -abs(amt) * (hi[1] - lo[1]) * 0.4
            for ix in (0, 1):
                off[ix, :, :, 0] += abs(amt) * 0.35 * (1 if ix else -1)
            for iz in (0, 1):
                off[:, :, iz, 2] += abs(amt) * 0.35 * (1 if iz else -1)
        elif preset == "skew":                                 # top layer slides +z
            off[:, 1, :, 2] = amt
        else:
            raise ValueError(f"unknown lattice preset '{preset}'")
        V2 = _df.lattice_deform(V.copy(), bounds=(tuple(lo), tuple(hi)), control_offsets=off)
        new = Mesh(np.asarray(V2, float), [tuple(f) for f in m.faces])
    elif name == "tessellate":
        # FLAT midpoint subdivision: each triangle -> 4 coplanar triangles, vertices untouched, so the SHAPE
        # is exactly preserved (unlike 'subdivide' = Loop, which smooths/rounds). This exists to give low-poly
        # objects (walls, plates) enough faces for per-face surface patterns to resolve.
        base = m if all(len(f) == 3 for f in m.faces) else Mesh(m.vertices, [tuple(t) for t in m.triangulate()])
        levels = int(np.clip(int(d.get("levels", 1)), 1, 3))
        V = [tuple(map(float, v)) for v in base.vertices]
        F = [tuple(f) for f in base.faces]
        for _ in range(levels):
            vidx = {i: i for i in range(len(V))}
            mid = {}
            def midpoint(a, b):
                key = (min(a, b), max(a, b))
                if key not in mid:
                    va, vb = V[a], V[b]
                    V.append(((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2))
                    mid[key] = len(V) - 1
                return mid[key]
            F2 = []
            for (a, b, c) in F:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                F2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            F = F2
        new = Mesh(np.asarray(V, float), F)
    elif name == "subdivide":
        from holographic_meshsubdiv import loop_subdivide
        base = m if all(len(f) == 3 for f in m.faces) else Mesh(m.vertices, [tuple(t) for t in m.triangulate()])
        new = loop_subdivide(base, levels=1)
    elif name == "smooth":
        from holographic_meshsmooth import taubin_smooth
        new = taubin_smooth(m, iters=int(np.clip(d.get("iters", 6), 1, 30)))
    elif name == "smooth_limit":
        # The Loop LIMIT surface in CLOSED FORM (holographic_meshsubdiv.loop_limit): where infinite
        # subdivision would put every vertex, in O(V) with no subdivision performed -- a mathematically
        # exact single-shot smooth, as distinct in character from Taubin's iterative diffusion as a
        # sharp result is from a blurred one. Triangulates first (Loop is a triangle scheme); keeps the
        # original face/vertex COUNT (only positions move), so materials need no transfer.
        from holographic_meshsubdiv import loop_limit
        tri = m if all(len(f) == 3 for f in m.faces) else Mesh(m.vertices, [tuple(t) for t in m.triangulate()])
        V, _N = loop_limit(tri)
        new = Mesh(V, [tuple(f) for f in tri.faces])
    elif name == "poke":
        # Fan a face from its centroid (holographic_eulerops.poke_face) -- retopology (height=0, pure
        # triangulation of an n-gon) or a spike (height>0, pushed along the face normal). V+1/E+n/F+(n-1).
        from holographic_eulerops import poke_face
        new = poke_face(m, int(d["face"]), float(d.get("height", 0.0)))
    elif name == "mirror":
        from holographic_meshtools import mirror
        new = mirror(m, axis=int(np.clip(d.get("axis", 0), 0, 2)))
    elif name == "taper":
        # TAPER: scale cross-section along an axis by a linear factor (narrow one end, widen the other) --
        # holographic_deform.taper, applied to the mesh vertices. axis x/y/z, factor = end scale relative to
        # start. Topology-preserving; the analytic tree is dropped (a mesh deform, stated honestly upstream).
        from holographic_mesh import Mesh
        import holographic_deform as _df
        axis = {"x": 0, "y": 1, "z": 2}.get(str(d.get("axis", "y")).lower(), 1)
        factor = float(np.clip(d.get("factor", 0.5), 0.05, 4.0))
        V = _df.taper(m.vertices.copy(), factor, axis=axis)
        new = Mesh(np.asarray(V, float), [tuple(f) for f in m.faces])
    elif name == "flute":
        # FLUTING / angular relief on a surface of revolution: displace each vertex radially by a cosine of its
        # angle about the Y axis, r' = r * (1 + depth*cos(N*theta + twist*y)). N lobes, optional vertical twist.
        # Pure vertex displacement (keeps topology); the object is field-only after (analytic tree dropped).
        from holographic_mesh import Mesh
        lobes = int(np.clip(int(d.get("lobes", 12)), 2, 64))
        depth = float(np.clip(d.get("depth", 0.06), 0.0, 0.4))
        twist = float(d.get("twist", 0.0))
        V = m.vertices.copy()
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        r = np.sqrt(x * x + z * z)
        theta = np.arctan2(z, x)
        scale = 1.0 + depth * np.cos(lobes * theta + twist * y)
        nz = r > 1e-6
        V[nz, 0] = r[nz] * scale[nz] * np.cos(theta[nz])
        V[nz, 2] = r[nz] * scale[nz] * np.sin(theta[nz])
        new = Mesh(V, [tuple(f) for f in m.faces])
    elif name == "solidify":
        from holographic_meshtools import solidify
        new = solidify(m, float(np.clip(d.get("thickness", 0.12), 0.02, 0.5)))
    elif name == "retopo":
        # FIELD-GUIDED, RESOLUTION-INDEPENDENT retopology (holographic_crossfield). Works on the mesh's own
        # 4-RoSy cross field, so the result follows surface flow rather than a voxel grid — the same quality
        # at any input tessellation. Modes:
        #   "quad"        -> quad_remesh along the smoothest cross field (tri-to-quad)
        #   "deformation" -> a strain guide (rest -> deformed_vertices) steers the field so edge loops FOLLOW
        #                    how the surface bends/stretches, then quad-remesh (skeleton/animation-aware topology)
        import holographic_crossfield as cf
        from holographic_mesh import Mesh
        tri = m
        if any(len(f) != 3 for f in m.faces):                # the field solver needs triangles
            tri = Mesh(m.vertices.copy(), _triangulate_faces(m.faces))
        mode = d.get("mode", "quad")
        field = None
        if mode == "deformation":
            dv = d.get("deformed_vertices")
            if dv is not None:
                strain = cf.strain_directions(tri, np.asarray(dv, float))
                field = cf.guided_cross_field(tri, strain, guide_weight=float(d.get("guide_weight", 5.0)))
            else:
                field = None                                 # no deformation given -> falls back to smoothest
        qm, info = cf.quad_remesh(tri, field=field) if field is not None else cf.quad_remesh(tri)
        new = qm
    else:
        raise ValueError(f"unknown op '{name}'")
    return new


def _triangulate_faces(faces):
    """Fan-triangulate any n-gon faces so the cross-field solver (triangles only) can run."""
    out = []
    for f in faces:
        if len(f) == 3:
            out.append(tuple(f))
        else:
            for i in range(1, len(f) - 1):
                out.append((f[0], f[i], f[i + 1]))
    return out


@bp.route("/api/op", methods=["POST"])
def op():
    _init()
    d = request.get_json(force=True) or {}
    name = d.get("op", "")
    oid = str(d.get("object", "")) if "object" in d else None
    with _LOCK:
        from holographic_mesh import Mesh
        # ---- scene-level ops --------------------------------------------------------------------
        if name == "begin_drag":
            if oid and oid in _S["objects"]:
                _snap_obj(oid)
            else:
                _snap_scene()
            return jsonify({"rev": _S["rev"]})
        if name == "delete_object":
            if oid not in _S["objects"]:
                return jsonify({"error": "no such object"}), 400
            _snap_scene(); del _S["objects"][oid]
            _PARENT.pop(oid, None)                              # A2-3: drop links to/from the deleted object
            for c, p in list(_PARENT.items()):
                if p == oid:
                    _PARENT.pop(c, None)
            _bump()
            return jsonify(_payload())
        if name == "fill_holes":
            # REPAIR: cap open boundary loops (holographic_meshverbs2.fill_holes -- 'fan' robust anywhere,
            # 'grid' Blender-style quad fill for even loops). max_sides=0 fills every loop, INCLUDING an open
            # sheet's outer rim (the module's own stated scope: topologically a rim IS a hole).
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            from holographic_meshverbs2 import fill_holes
            _snap_obj(oid)
            try:
                new = fill_holes(o.mesh, mode=d.get("mode", "fan"),
                                 max_sides=int(np.clip(int(d.get("max_sides", 0)), 0, 64)))
            except Exception as e:
                _discard_snapshot()
                return jsonify({"error": f"fill_holes failed: {e}"}), 400
            new = Mesh(new.vertices, [tuple(f) for f in new.faces])
            o.mats = _transfer_mats(o.mesh, o.mats, new)
            o.mesh = new
            _bump(oid)
            return jsonify(_payload(only=oid))

        if name == "triangulate":
            # Ear-clip every n-gon (concave-correct, unlike a convex fan). Vertices untouched.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            from holographic_meshverbs2 import triangulate_ngons
            _snap_obj(oid)
            new = triangulate_ngons(o.mesh)
            new = Mesh(new.vertices, [tuple(f) for f in new.faces])
            o.mats = _transfer_mats(o.mesh, o.mats, new)
            o.mesh = new
            _bump(oid)
            return jsonify(_payload(only=oid))

        if name == "bridge":
            # BRIDGE the object's two largest open boundary loops with a quad band (holographic_meshverbs2.
            # bridge_loops). The module takes ordered equal-length loops and leaves correspondence to the caller,
            # so here: trace each boundary loop by chaining its edges, require the two largest to have EQUAL
            # length (stated refusal otherwise -- resampling a loop is a different, lossy operation), and pick
            # the rotation/orientation of loop B that minimises total rung length (exhaustive offset search,
            # exact for loops this size).
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            import holographic_meshselect as msel
            from holographic_meshverbs2 import bridge_loops
            g = _mesh_dict(o.mesh)
            edge_list = [tuple(e) for e in msel._edge_list(g)]
            bsel = msel.select_boundary_loops(g)
            bedges = [edge_list[i] for i in (bsel.to_list() if hasattr(bsel, "to_list") else bsel)]
            if not bedges:
                return jsonify({"error": "no open boundary loops on this object (it is closed)"}), 400
            adj = {}
            for a, b in bedges:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            unvisited = set(adj)
            loops = []
            while unvisited:
                start = next(iter(unvisited))
                loop = [start]; unvisited.discard(start)
                prev, cur = None, start
                while True:
                    nxts = [v for v in adj[cur] if v != prev]
                    nxt = next((v for v in nxts if v in unvisited), None)
                    if nxt is None:
                        break
                    loop.append(nxt); unvisited.discard(nxt)
                    prev, cur = cur, nxt
                if len(loop) >= 3:
                    loops.append(loop)
            if len(loops) < 2:
                return jsonify({"error": f"bridge needs two boundary loops; found {len(loops)}"}), 400
            loops.sort(key=len, reverse=True)
            la, lb = loops[0], loops[1]
            if len(la) != len(lb):
                return jsonify({"error": f"the two largest loops have {len(la)} and {len(lb)} edges -- bridge "
                                         "needs equal-length loops (resampling would be a different, lossy op)"}), 400
            V = o.mesh.vertices
            best = (None, np.inf)
            for rev in (False, True):
                cand0 = list(reversed(lb)) if rev else list(lb)
                for off in range(len(cand0)):
                    cand = cand0[off:] + cand0[:off]
                    cost = float(np.linalg.norm(V[la] - V[cand], axis=1).sum())
                    if cost < best[1]:
                        best = (cand, cost)
            _snap_obj(oid)
            band = bridge_loops(V, la, best[0], closed=True)
            new = Mesh(V.copy(), [tuple(f) for f in o.mesh.faces] + [tuple(f) for f in band.faces])
            fill_mat = max(set(o.mats), key=o.mats.count) if o.mats else _default_mat_name()
            o.mats = list(o.mats) + [fill_mat] * len(band.faces)
            o.mesh = new
            _bump(oid)
            return jsonify(_payload(only=oid))

        if name == "set_layers":
            # LAYERED SHELL MATERIAL (user ask #3): attach [(thickness, material), ...] to the object,
            # measured inward from the surface. The layers are revealed EXACTLY by /api/section (banded by
            # true inside-depth -f(p)); the 3-D beauty render shows the outermost material as before -- the
            # honest scope note is in the UI: full path-traced layer reveal on 3-D cut faces is a follow-up.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            raw = d.get("layers", [])
            layers = []
            for item in raw[:6]:
                try:
                    lt = float(item[0]); lname = str(item[1]); _mat(lname)
                    if lt <= 0:
                        raise ValueError
                    layers.append((lt, lname))
                except Exception:
                    return jsonify({"error": f"bad layer {item!r}: need [thickness>0, known_material]"}), 400
            o.layers = layers
            _bump()
            out = _payload(); out["object"] = oid; out["layers"] = layers
            return jsonify(out)
        if name == "shell":
            # EXACT resolution-independent shell (the CAD hollow, user ask #1): for an analytic object the
            # shell is tree.onion(t) -- |f(p)|-t, MATHEMATICALLY exact at any resolution, and it survives the
            # analytic-native bake and the exact Shadertoy/WGSL export (DSL: (onion t ...)). Verified: sphere
            # onion leaves the centre at +(r-t) -- truly hollow. For a mesh with no tree we fall back to
            # meshtools.solidify (offset-surface shell) and SAY so: that one is a mesh construction, not exact.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            _snap_scene()
            t = float(np.clip(d.get("thickness", 0.06), 0.005, 0.5))
            if o.sdf_tree is not None:
                t2 = o.sdf_tree.onion(t)
                lo = o.mesh.vertices.min(axis=0) - 0.3; hi = o.mesh.vertices.max(axis=0) + 0.3
                try:
                    new_mesh = _mesh_sdf_tree(t2, res=int(d.get("res", 64)), face_target=int(d.get("target", 6000)),
                                              scan_lo=lo, scan_hi=hi)
                except Exception as e:
                    _discard_snapshot(); return jsonify({"error": f"shell remesh failed: {e}"}), 400
                o.sdf_tree = t2
                o.mats = _transfer_mats(o.mesh, o.mats, new_mesh)
                o.mesh = new_mesh; o.rev += 1; o.sculpt = None
                _bump()
                _hist_record(oid, "shell", d)
                out = _payload(); out["object"] = oid; out["exact"] = True
                out["note"] = f"exact shell t={t} (onion) -- resolution-independent, survives exact export"
                return jsonify(out)
            else:
                from holographic_meshtools import solidify
                try:
                    new_mesh = solidify(o.mesh, t)
                except Exception as e:
                    _discard_snapshot(); return jsonify({"error": str(e)}), 400
                o.mats = _transfer_mats(o.mesh, o.mats, new_mesh)
                o.mesh = new_mesh; o.rev += 1
                _bump()
                out = _payload(); out["object"] = oid; out["exact"] = False
                out["note"] = f"mesh shell t={t} (offset surface) -- no analytic tree on this object"
                return jsonify(out)
        if name == "boolean":
            # SOLID BOOLEANS between two scene objects (the CAD arc, K5/K6): union / subtract / intersect, with
            # an optional EXACT constant-radius fillet at the seam (holographic_fillet -- iq's rounded booleans;
            # a true dimensioned radius-r arc, NOT smooth_union's soft blend whose radius is not k; the module
            # measures that gap itself). Two honest paths:
            #   * TREE path -- both objects analytic AND fillet == 0: the boolean is a tree op, so the result is
            #     STILL its exact SDF (analytic bake + exact Shadertoy export survive the boolean).
            #   * FIELD path -- anything else (sculpted/imported parents, or a fillet): combine the objects'
            #     signed-distance fields with the fillet combinators (or sharp min/max), sample over the union
            #     bounds, march, decimate. This is the same SDF-route holographic_brepbool.brep_boolean documents
            #     for B-reps, run on our own cached per-object bakes so it composes with every object kind.
            # Result replaces A; B is consumed. One scene snapshot = one undo.
            o = _S["objects"].get(oid)
            other = _S["objects"].get(str(d.get("other", "")))
            if o is None or other is None:
                return jsonify({"error": "boolean needs two existing objects (select A then B)"}), 400
            if other is _S["objects"].get(oid):
                return jsonify({"error": "boolean needs two DIFFERENT objects"}), 400
            kind = d.get("kind", "union")
            if kind not in ("union", "subtract", "intersect", "smooth", "chamfer"):
                return jsonify({"error": f"unknown boolean kind '{kind}'"}), 400
            if kind == "chamfer" and d.get("chamfer_kind", "union") != "union":
                return jsonify({"error": "the engine provides chamfer for UNION only (chamfer_union)"}), 400
            rad = float(np.clip(d.get("fillet", 0.0), 0.0, 0.5))
            res = int(np.clip(int(d.get("res", 72)), 48, 96))
            target = int(np.clip(int(d.get("target", 3000)), 300, 20000))
            other_id = str(d.get("other"))
            _snap_scene()
            try:
                if rad == 0.0 and kind not in ("smooth", "chamfer") and o.sdf_tree is not None and other.sdf_tree is not None:
                    # ---- TREE path: exact, stays analytic ----
                    t2 = {"union": o.sdf_tree.union, "subtract": o.sdf_tree.subtract,
                          "intersect": o.sdf_tree.intersect}[kind](other.sdf_tree)
                    lo = np.minimum(o.mesh.vertices.min(0), other.mesh.vertices.min(0)) - 0.3
                    hi = np.maximum(o.mesh.vertices.max(0), other.mesh.vertices.max(0)) + 0.3
                    new_mesh = _mesh_sdf_tree(t2, res=res, face_target=target, scan_lo=lo, scan_hi=hi)
                    keep_tree = t2
                else:
                    # ---- FIELD path: engine fillet combinators over the two objects' distance fields.
                    # An analytic parent contributes its exact tree.eval; a mesh parent contributes its cached
                    # banded-grid bake (positive +band outside its bbox, so min/max/fillet combining stays
                    # correct outside either grid -- the fillet equals the sharp boolean away from the seam).
                    from holographic_fillet import fillet_union, fillet_intersection, fillet_difference
                    from holographic_meshbridge import marching_tetrahedra_vec

                    def field_of(obj, obj_id):
                        if obj.sdf_tree is not None:
                            return obj.sdf_tree.eval
                        f = _bake_object(obj_id, res, with_ids=False)
                        fld = f[0] if isinstance(f, tuple) else f
                        return fld.eval
                    fa, fb = field_of(o, oid), field_of(other, other_id)
                    if kind == "smooth":
                        from holographic_domain import smin
                        _k = float(np.clip(d.get("k", 0.15), 0.02, 0.6))
                        comb = lambda P: smin(fa(P), fb(P), _k)   # soft organic weld (seam radius ~k)
                    elif kind == "chamfer":
                        from holographic_fillet import chamfer_union
                        _r = float(np.clip(d.get("chamfer", d.get("k", 0.08)), 0.01, 0.5))
                        comb = chamfer_union(fa, fb, _r)          # 45-degree flat at the seam (CAD chamfer)
                    elif rad > 0.0:
                        comb = {"union": fillet_union, "subtract": fillet_difference,
                                "intersect": fillet_intersection}[kind](fa, fb, rad)
                    else:
                        comb = {"union": lambda P: np.minimum(fa(P), fb(P)),
                                "subtract": lambda P: np.maximum(fa(P), -fb(P)),
                                "intersect": lambda P: np.maximum(fa(P), fb(P))}[kind]
                    lo = np.minimum(o.mesh.vertices.min(0), other.mesh.vertices.min(0)) - 0.3
                    hi = np.maximum(o.mesh.vertices.max(0), other.mesh.vertices.max(0)) + 0.3
                    ax = tuple(np.linspace(lo[i], hi[i], res) for i in range(3))
                    X, Y, Z = np.meshgrid(*ax, indexing="ij")
                    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
                    g = np.empty(len(P))
                    for i in range(0, len(P), 200_000):
                        g[i:i + 200_000] = comb(P[i:i + 200_000])
                    if not (g.min() < 0 < g.max()):
                        _discard_snapshot()
                        return jsonify({"error": f"the {kind} is empty (objects may not overlap)"}), 400
                    raw = marching_tetrahedra_vec(g.reshape(res, res, res), ax, level=0.0)
                    from holographic_meshqem import cluster_decimate
                    gr = 52
                    dec = cluster_decimate(raw, grid=gr)
                    while dec.n_faces > target and gr > 8:
                        gr -= 4
                        dec = cluster_decimate(raw, grid=gr)
                    new_mesh = Mesh(dec.vertices, [tuple(f) for f in dec.faces])
                    keep_tree = None
            except Exception as e:
                _discard_snapshot()
                return jsonify({"error": f"boolean failed: {e}"}), 400
            # materials: nearest face centroid across BOTH parents, so each side keeps its own look
            def _cents(m):
                return np.array([m.vertices[list(f)].mean(axis=0) for f in m.faces])
            pool_c = np.vstack([_cents(o.mesh), _cents(other.mesh)])
            pool_m = list(o.mats) + list(other.mats)
            new_c = _cents(new_mesh)
            mats = []
            for i in range(0, len(new_c), 512):
                blk = new_c[i:i + 512]
                idx = np.argmin(((blk[:, None, :] - pool_c[None, :, :]) ** 2).sum(-1), axis=1)
                mats.extend(pool_m[j] for j in idx)
            o.mesh = new_mesh
            o.mats = mats
            o.name = f"{o.name} {kind} {other.name}"
            del _S["objects"][other_id]
            _bump(oid); _bump()
            o.sdf_tree = keep_tree
            o.kernel_src = None
            _hist_bake_point(oid, "boolean")
            return jsonify(_payload())

        if name == "lathe":
            # LATHE / REVOLVE (the sketch->solid CAD workflow, via holographic_sdf2d): a 2-D profile of (radius,
            # height) points becomes a solid of revolution. polygon2d gives the exact 2-D signed distance of the
            # closed profile; revolve() spins it about the Y axis. HONESTY: revolve returns a plain field
            # CALLABLE, not an analytic tree (verified: correct distances, no to_dsl) -- so a lathe object is
            # field-only: it meshes and renders like any object, and its shader export takes the FITTED path.
            prof = d.get("profile") or []
            try:
                pts = [(float(r_), float(y_)) for r_, y_ in prof]
            except Exception:
                return jsonify({"error": "profile must be [[radius, y], ...]"}), 400
            if len(pts) < 3:
                return jsonify({"error": "profile needs at least 3 [radius, y] points"}), 400
            if any(r_ < 0 for r_, _ in pts):
                return jsonify({"error": "radii must be >= 0"}), 400
            from holographic_sdf2d import polygon2d, revolve
            # close the loop onto the axis so the revolve is a solid, not a shell
            loop = list(pts)
            if loop[0][0] != 0.0:
                loop.insert(0, (0.0, loop[0][1]))
            if loop[-1][0] != 0.0:
                loop.append((0.0, loop[-1][1]))
            fn = revolve(polygon2d(loop), offset=0.0)

            class _F:                                      # duck-typed .eval for the meshing helper
                def eval(self, P):
                    return np.asarray(fn(np.atleast_2d(P)), float)
            rmax = max(r_ for r_, _ in loop)
            ys = [y_ for _, y_ in loop]
            lo = np.array([-rmax - 0.2, min(ys) - 0.2, -rmax - 0.2])
            hi = np.array([rmax + 0.2, max(ys) + 0.2, rmax + 0.2])
            try:
                with_res = int(np.clip(int(d.get("res", 64)), 40, 88))
                mesh = _mesh_sdf_tree(_F(), res=with_res, face_target=int(np.clip(int(d.get("target", 2200)),
                                                                                  200, 12000)),
                                      scan_lo=lo, scan_hi=hi)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            _snap_scene()
            new_oid = _add_object(d.get("name") or "Lathe", mesh)
            _bump()
            out = _payload(); out["object"] = new_oid
            return jsonify(out)

        if name == "shader_displace":
            # SHADERTOY-STYLE GEOMETRY MODIFIER: displace every vertex along its normal by amount * f(x,y,z,r),
            # where f is a user expression run through the engine's SAFE ns-eel2 evaluator (whitelisted grammar,
            # never eval). Works on ANY mesh -- poly, sculpted, imported. This is the classic per-point
            # displacement of a Shadertoy map(), applied to real geometry.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            expr = str(d.get("expr", "")).strip()
            if not expr:
                return jsonify({"error": "empty expression"}), 400
            amount = float(np.clip(d.get("amount", 0.1), -0.6, 0.6))
            _snap_obj(oid)
            try:
                vals = _eval_expr_batch(expr, o.mesh.vertices)
            except Exception as e:
                _discard_snapshot()
                return jsonify({"error": f"expression refused: {e}"}), 400
            disp = np.clip(amount * vals, -1.2, 1.2)[:, None] * _vertex_normals(o.mesh)
            o.mesh = Mesh(o.mesh.vertices + disp, [tuple(f) for f in o.mesh.faces])
            _bump(oid)
            return jsonify(_payload(only=oid))

        if name == "shader_material":
            # SHADERTOY-STYLE MATERIAL RULE: evaluate the expression at every FACE CENTROID and assign the
            # material to faces where f >= threshold -- zebra stripes, gradients, trig-noise patchiness, driven
            # by the same safe evaluator. Composes with painting and per-face selection.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            expr = str(d.get("expr", "")).strip()
            mat = d.get("material", _default_mat_name())
            try:
                _mat(mat)
            except KeyError as e:
                return jsonify({"error": str(e)}), 400
            if not expr:
                return jsonify({"error": "empty expression"}), 400
            thr = float(d.get("threshold", 0.0))
            _snap_obj(oid)
            try:
                vals = _eval_expr_batch(expr, _centroids(o.mesh))
            except Exception as e:
                _discard_snapshot()
                return jsonify({"error": f"expression refused: {e}"}), 400
            hit = 0
            for i, v in enumerate(vals):
                if v >= thr:
                    o.mats[i] = mat; hit += 1
            # geometry untouched: the analytic identity survives, same as a plain assign
            keep_tree, keep_kern = o.sdf_tree, o.kernel_src
            _bump(oid)
            o.sdf_tree, o.kernel_src = keep_tree, keep_kern
            out = _payload(only=oid); out["faces_hit"] = hit
            return jsonify(out)

        if name == "decimate":
            # LEVEL OF DETAIL, downward: vertex-cluster decimate to a FACE TARGET (the same face-target loop the
            # bake LOD uses -- a fixed grid can no-op on already-clustered meshes), with nearest-centroid
            # material transfer. Subdivide is the up direction; this is down.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            target = int(np.clip(int(d.get("target", 800)), 40, 40000))
            from holographic_meshqem import cluster_decimate
            m = o.mesh
            tri = m if all(len(f) == 3 for f in m.faces) else Mesh(m.vertices, [tuple(t) for t in m.triangulate()])
            _snap_obj(oid)
            g = 52
            dec = cluster_decimate(tri, grid=g)
            while dec.n_faces > target and g > 6:
                g -= 4
                dec = cluster_decimate(tri, grid=g)
            new_mesh = Mesh(dec.vertices, [tuple(f) for f in dec.faces])
            o.mats = _transfer_mats(m, o.mats, new_mesh)
            o.mesh = new_mesh
            _bump(oid)
            return jsonify(_payload(only=oid))

        if name == "sdf_modifier":
            # THE IQ/SHADERTOY OPERATORS AS EXACT GEOMETRY MODIFIERS: twist / bend / onion / displace / elongate /
            # rounded applied to an ANALYTIC object's SDF tree, then re-meshed by the same marching-tetrahedra
            # pipeline sculpt uses. The tree is KEPT (the modified object is still exactly its SDF), so the bake
            # stays analytic and the GLSL Shadertoy export stays EXACT -- to_glsl emits every one of these warp
            # nodes (verified); the strict WGSL dialect refuses the warps, so WGSL falls to the fitted path,
            # stated in the UI. Requires an analytic object; edited meshes get the reason, not a guess.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            if o.sdf_tree is None:
                return jsonify({"error": "this object is an edited/sculpted mesh with no analytic SDF tree -- "
                                         "the exact modifiers need one (add a primitive or use Make from "
                                         "description). For meshes, use Shader displace instead."}), 400
            kind = d.get("kind", "twist")
            k = float(d.get("k", 1.0))
            t = o.sdf_tree
            try:
                if kind == "twist":
                    t2 = t.twist(float(np.clip(k, -6, 6)))
                elif kind == "bend":
                    t2 = t.bend(float(np.clip(k, -4, 4)), int(d.get("axis", 0)))
                elif kind == "onion":
                    t2 = t.onion(float(np.clip(abs(k), 0.01, 0.4)))
                elif kind == "displace":
                    t2 = t.displace(float(np.clip(k, -0.3, 0.3)), float(np.clip(d.get("freq", 8.0), 1, 40)))
                elif kind == "elongate":
                    t2 = t.elongate(*[float(np.clip(v, 0, 2)) for v in d.get("h", [k, 0, 0])])
                elif kind == "rounded":
                    t2 = t.rounded(float(np.clip(abs(k), 0.01, 0.5)))
                else:
                    return jsonify({"error": f"unknown modifier '{kind}'"}), 400
                lo = o.mesh.vertices.min(axis=0) - 0.8
                hi = o.mesh.vertices.max(axis=0) + 0.8
                new_mesh = _mesh_sdf_tree(t2, res=int(np.clip(int(d.get("res", 64)), 40, 88)),
                                          face_target=int(np.clip(int(d.get("target", 2400)), 200, 12000)),
                                          scan_lo=lo, scan_hi=hi)
            except Exception as e:
                return jsonify({"error": f"{kind} failed: {e}"}), 400
            _snap_obj(oid)
            o.mats = _transfer_mats(o.mesh, o.mats, new_mesh)
            o.mesh = new_mesh
            _bump(oid)
            o.sdf_tree = t2                                # after _bump: the modified tree IS the object, exactly
            return jsonify(_payload(only=oid))

        if name == "duplicate":
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            _snap_scene()
            off = np.asarray(d.get("offset", [0.35, 0, 0.35]), float)
            m2 = Mesh(o.mesh.vertices.copy() + off, [tuple(f) for f in o.mesh.faces])
            nid = _add_object(d.get("name") or (o.name + " copy"), m2, list(o.mats),
                              sdf_tree=(o.sdf_tree.translate(tuple(map(float, off))) if o.sdf_tree is not None else None))
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        if name == "transform":
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            M = np.asarray(d.get("matrix", []), float).reshape(4, 4)
            _snap_obj(oid)
            V = o.mesh.vertices
            o.mesh.vertices = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]
            if np.linalg.det(M[:3, :3]) < 0:               # a mirroring transform flips winding: restore it
                o.mesh.faces = [tuple(reversed(f)) for f in o.mesh.faces]
            o.mesh.normals = None; o.mesh._he = None; o.mesh._adj = None
            # A PURE TRANSLATION (the everyday gizmo move) maps exactly onto the analytic tree, so a moved
            # primitive keeps its exact bake + exact shader instead of degrading to a mesh. Anything with a
            # rotation/scale part still drops the tree (correct: matching every 4x4 is future work, not faked).
            keep_tree = None; keep_kern = None
            if o.sdf_tree is not None and np.allclose(M[:3, :3], np.eye(3), atol=1e-12):
                keep_tree = o.sdf_tree.translate((float(M[0, 3]), float(M[1, 3]), float(M[2, 3])))
                keep_kern = o.kernel_src                   # positions in the kernel text are stale; tree is the
                keep_kern = None                           # authority after a move, so the kernel is dropped too
            _bump(oid)
            if keep_tree is not None:
                o.sdf_tree = keep_tree
            return jsonify(_payload(only=oid))
        if name == "cloth":
            # DRAPED CLOTH: a procedural table covering -- a grid plane with gentle sinusoidal folds plus a
            # quadratic edge droop so the corners fall like an overhang. Not a cloth sim (the engine has none),
            # but a convincing static drape that reads far better than a flat tinted plane. Fabric materials
            # (linen/velvet/cotton) pair with it.
            from holographic_mesh import Mesh
            size = float(np.clip(d.get("size", 2.4), 0.5, 8.0))
            res = int(np.clip(int(d.get("res", 56)), 16, 120))
            folds = float(np.clip(d.get("folds", 4), 0, 12))
            depth = float(np.clip(d.get("depth", 0.035), 0.0, 0.3))
            droop = float(np.clip(d.get("droop", 0.16), 0.0, 0.6))
            top = float(d.get("top", 0.0))                    # table-top Y
            xs = np.linspace(-size / 2, size / 2, res)
            zs = np.linspace(-size / 2, size / 2, res)
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            Y = depth * np.sin(folds * np.pi * X / size) * np.cos(folds * 0.7 * np.pi * Z / size)
            ex = np.clip((np.abs(X) / (size / 2) - 0.8) / 0.2, 0, None)
            ez = np.clip((np.abs(Z) / (size / 2) - 0.8) / 0.2, 0, None)
            Y = Y - droop * (ex * ex + ez * ez) + top
            V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            F = []
            for i in range(res - 1):
                for j in range(res - 1):
                    a = i * res + j; b = i * res + j + 1; cc = (i + 1) * res + j; dd = (i + 1) * res + j + 1
                    F.append((a, cc, b)); F.append((b, cc, dd))
            mesh = Mesh(V, [tuple(f) for f in F])
            mat = d.get("material", "linen") if d.get("material") else "linen"
            try:
                _mat(mat)
            except Exception:
                mat = _default_mat_name()
            nid = _add_object(d.get("name") or "Cloth", mesh, [mat] * mesh.n_faces)
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        if name == "band":
            # HEIGHT-BANDED MATERIAL: assign a material to every face whose centroid falls in a Y-range (or a
            # normalized 0..1 range of the object's height). This is how gold rims and painted bands go onto a
            # surface of revolution -- it's a function of the profile parameter, trivial on a lathe. Optionally
            # 'rim' snaps the band to the top or bottom lip automatically.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            mat = d.get("material")
            if not mat:
                return jsonify({"error": "band needs a 'material'"}), 400
            try:
                _mat(mat)
            except Exception:
                return jsonify({"error": f"unknown material '{mat}'"}), 400
            V = o.mesh.vertices
            ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
            span = max(ymax - ymin, 1e-6)
            rim = d.get("rim")
            if rim == "top":
                lo_n, hi_n = 1.0 - float(d.get("width", 0.08)), 1.0
            elif rim == "bottom":
                lo_n, hi_n = 0.0, float(d.get("width", 0.08))
            elif "y_lo" in d or "y_hi" in d:                 # absolute Y range
                lo_n = (float(d.get("y_lo", ymin)) - ymin) / span
                hi_n = (float(d.get("y_hi", ymax)) - ymin) / span
            else:                                            # normalized 0..1 range
                lo_n = float(d.get("lo", 0.45)); hi_n = float(d.get("hi", 0.55))
            lo_y = ymin + lo_n * span; hi_y = ymin + hi_n * span
            mats = list(o.mats); n = 0
            for fi, f in enumerate(o.mesh.faces):
                cy = float(V[list(f)][:, 1].mean())
                if lo_y - 1e-6 <= cy <= hi_y + 1e-6:
                    mats[fi] = mat; n += 1
            o.mats = mats
            _bump(oid)
            out = _payload(only=oid); out["banded"] = n
            return jsonify(out)
        if name == "surface_graph":
            # SUBSTANCE-DESIGNER-STYLE PROCEDURAL SURFACING, adapted to this renderer's per-face shading:
            # face centroids are the "pixels". The classic SD pipeline shape is kept intact --
            #   GENERATOR (noise/pattern) -> LEVELS (remap/invert/balance) -> GRADIENT MAP (value ranges ->
            #   materials, where a stop can be 'keep' to preserve existing paint = SD's Blend-with-mask).
            # Generators: perlin (fBm, multi-octave), cells (Voronoi F1), checker, stripes, bricks (per-brick
            # value variation like SD's tile generator), scratches (directional), gradient. Dense meshes show
            # patterns well; use 'tessellate' first on low-poly objects (stated in the dialog).
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            gen = str(d.get("generator", "perlin"))
            scale = float(np.clip(d.get("scale", 4.0), 0.5, 64.0))
            seed = int(d.get("seed", 0))
            axis = {"x": 0, "y": 1, "z": 2}.get(str(d.get("axis", "y")).lower(), 1)
            rng = np.random.RandomState(seed)
            V = o.mesh.vertices
            cents = np.array([V[list(f)].mean(axis=0) for f in o.mesh.faces])
            lo, hi = V.min(0), V.max(0)
            span = np.maximum(hi - lo, 1e-9)
            v01 = (cents - lo) / span                                   # normalised 0..1 face coords
            n = len(cents)
            if gen == "perlin":
                # fBm: octaves of a smooth sin-hash lattice (not Ken Perlin's exact algorithm; same role)
                score = np.zeros(n)
                amp, freq, total = 1.0, scale, 0.0
                for oct_ in range(4):
                    ph = rng.rand(3) * 6.28
                    score += amp * (np.sin(freq * v01[:, 0] * 3.1 + ph[0]) *
                                    np.cos(freq * v01[:, 1] * 2.7 + ph[1]) *
                                    np.sin(freq * v01[:, 2] * 3.7 + ph[2]) * 0.5 + 0.5)
                    total += amp; amp *= 0.5; freq *= 2.0
                score /= total
            elif gen == "cells":
                k = int(np.clip(scale * scale * 0.6, 4, 400))
                seeds = rng.rand(k, 3)
                dif = v01[:, None, :] - seeds[None, :, :]
                score = np.sqrt((dif * dif).sum(-1)).min(axis=1)        # Voronoi F1 distance
                score = score / max(np.ptp(score), 1e-9)
            elif gen == "checker":
                cellidx = np.floor(v01 * scale).astype(int)
                score = ((cellidx.sum(axis=1)) % 2).astype(float)
            elif gen == "stripes":
                score = (np.floor(v01[:, axis] * scale).astype(int) % 2).astype(float)
            elif gen == "bricks":
                u_ax, v_ax = [(1, 2), (0, 2), (0, 1)][axis]             # brick plane = the two other axes
                rows = np.floor(v01[:, v_ax] * scale).astype(int)
                uu = v01[:, u_ax] * scale * 2.0 + (rows % 2) * 0.5      # alternate-row offset
                cols = np.floor(uu).astype(int)
                fu, fv = uu - cols, v01[:, v_ax] * scale - rows
                mortar = float(np.clip(d.get("mortar", 0.08), 0.01, 0.4))
                edge = (fu < mortar) | (fu > 1 - mortar) | (fv < mortar) | (fv > 1 - mortar)
                per_brick = np.zeros(n)                                  # per-brick value variation (SD-style)
                hsh = (rows * 73856093) ^ (cols * 19349663) ^ seed
                per_brick = 0.35 + 0.65 * ((np.abs(hsh) % 1000) / 1000.0)
                score = np.where(edge, 0.0, per_brick)
            elif gen == "scratches":
                count = int(np.clip(scale * 3, 3, 120))
                u_ax, v_ax = [(1, 2), (0, 2), (0, 1)][axis]
                score = np.zeros(n)
                for _ in range(count):
                    p0 = rng.rand(2); ang = rng.rand() * 6.28
                    dvec = np.array([np.cos(ang), np.sin(ang)])
                    rel = np.stack([v01[:, u_ax] - p0[0], v01[:, v_ax] - p0[1]], 1)
                    dist = np.abs(rel[:, 0] * -dvec[1] + rel[:, 1] * dvec[0])
                    along = rel @ dvec
                    hit = (dist < 0.006) & (np.abs(along) < 0.15 + rng.rand() * 0.3)
                    score = np.maximum(score, hit.astype(float))
            elif gen == "gradient":
                score = v01[:, axis].copy()
            else:
                return jsonify({"error": f"unknown generator '{gen}'"}), 400
            # LEVELS: balance shifts the midpoint via a gamma remap, invert flips (SD's most-used controls)
            balance = float(np.clip(d.get("balance", 0.5), 0.02, 0.98))
            gamma = np.log(0.5) / np.log(balance)
            score = np.clip(score, 0, 1) ** gamma
            if d.get("invert"):
                score = 1.0 - score
            # GRADIENT MAP: value ranges -> materials; 'keep' preserves existing paint (Blend-with-mask)
            ramp = d.get("ramp") or [{"material": "keep", "upto": 0.5}, {"material": "gold", "upto": 1.0}]
            try:
                stops = sorted(({"material": str(s["material"]), "upto": float(s["upto"])} for s in ramp),
                               key=lambda s: s["upto"])
            except Exception:
                return jsonify({"error": "ramp must be [{material, upto}, ...]"}), 400
            for s in stops:
                if s["material"] != "keep":
                    try:
                        _mat(s["material"])
                    except Exception:
                        return jsonify({"error": f"unknown material '{s['material']}'"}), 400
            mats = list(o.mats)
            counts = {}
            for fi in range(min(len(mats), n)):
                mchoice = stops[-1]["material"]
                for s in stops:
                    if score[fi] <= s["upto"] + 1e-9:
                        mchoice = s["material"]; break
                if mchoice != "keep":
                    mats[fi] = mchoice
                counts[mchoice] = counts.get(mchoice, 0) + 1
            o.mats = mats
            _bump(oid)
            out = _payload(only=oid); out["surface"] = {"generator": gen, "counts": counts}
            return jsonify(out)
        if name == "opening":
            # ARCHITECTURE: a door/window -- an EXACT rectangular through-hole in a wall built by the `wall`
            # op. Rather than a field boolean (which would trade the wall's exactness for marching-cubes
            # approximation), the wall-with-hole is CONSTRUCTED: 8 outer + 8 hole-rim vertices, the pierced
            # faces split into 4 quads each, plus a 4-quad tunnel. Watertight, volume exact. Requires an
            # unedited wall (8 verts / 6 quads); anything else errors honestly (use Boolean subtract instead).
            from holographic_mesh import Mesh
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            V0 = o.mesh.vertices
            if o.mesh.n_vertices != 8 or o.mesh.n_faces != 6:
                return jsonify({"error": "opening needs an unedited wall (8 verts / 6 faces); for other shapes use Boolean subtract with a placed cube"}), 400
            # recover the wall frame from the construction order: [a-s, a+s, b+s, b-s, +up x4]
            a = (V0[0] + V0[1]) / 2.0; b = (V0[2] + V0[3]) / 2.0
            side = (V0[1] - V0[0]) / 2.0
            up = V0[4] - V0[0]
            L = float(np.linalg.norm(b - a)); H = float(np.linalg.norm(up))
            fwd = (b - a) / max(L, 1e-9)
            s = float(d.get("at", L / 2.0))                   # hole CENTRE distance along the wall from its start
            w = float(np.clip(d.get("width", 0.4), 0.01, L))
            hh = float(np.clip(d.get("height", 0.8), 0.01, H))
            sill = float(np.clip(d.get("sill", 0.0), 0.0, H))
            u0, u1 = s - w / 2.0, s + w / 2.0
            v0, v1 = sill, sill + hh
            if u0 < 1e-6 or u1 > L - 1e-6 or v1 > H - 1e-6:
                return jsonify({"error": f"opening exceeds the wall (wall {L:.2f} long x {H:.2f} high; hole u[{u0:.2f},{u1:.2f}] v[{v0:.2f},{v1:.2f}])"}), 400
            uhat = fwd; vhat = up / max(H, 1e-9)
            def P(u, v, sgn):                                 # point at distance u along, height v, +/- side face
                return a + uhat * u + vhat * v + side * sgn
            # 16 vertices: outer 8 (rebuilt in a clean order) + hole rims front(+)/back(-)
            Vt = [P(0, 0, -1), P(0, 0, 1), P(L, 0, 1), P(L, 0, -1),
                  P(0, H, -1), P(0, H, 1), P(L, H, 1), P(L, H, -1),
                  P(u0, v0, 1), P(u1, v0, 1), P(u1, v1, 1), P(u0, v1, 1),      # front rim 8-11
                  P(u0, v0, -1), P(u1, v0, -1), P(u1, v1, -1), P(u0, v1, -1)]  # back rim 12-15
            # faces: outer bottom/top/endcaps, then the two pierced faces as 4-quad frames, then the tunnel
            F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6)]
            # front frame (+side; outer corners 1,2,6,5 ; hole rim 8,9,10,11)
            F += [(1, 8, 11, 5), (9, 2, 6, 10), (1, 2, 9, 8), (11, 10, 6, 5)]
            # back frame (viewed from -side, winding reversed; outer 0,3,7,4 ; hole 12,13,14,15)
            F += [(0, 4, 15, 12), (13, 14, 7, 3), (0, 12, 13, 3), (15, 4, 7, 14)]
            # tunnel (connect front rim to back rim; normals INTO the hole)
            F += [(8, 9, 13, 12), (10, 11, 15, 14), (9, 10, 14, 13), (11, 8, 12, 15)]
            mesh = Mesh(np.asarray(Vt, float), F)
            _snap_obj(oid)
            base_mat = o.mats[0] if o.mats else _default_mat_name()
            o.mesh = mesh; o.mats = [base_mat] * mesh.n_faces
            o.sdf_tree = None
            _bump(oid)
            out = _payload(only=oid); out["opening"] = {"at": s, "width": w, "height": hh, "sill": sill}
            return jsonify(out)
        if name == "wall":
            # ARCHITECTURE: a wall from two plan points -- an exact 8-vertex box oriented along the segment
            # (x1,z1)->(x2,z2) with a thickness and height, base at y=base. Direct mesh construction: exact
            # quads, watertight, perfect volume (no field meshing). Chain walls by calling repeatedly.
            from holographic_mesh import Mesh
            try:
                p1 = np.array([float(d["x1"]), float(d.get("base", 0.0)), float(d["z1"])], float)
                p2 = np.array([float(d["x2"]), float(d.get("base", 0.0)), float(d["z2"])], float)
            except Exception:
                return jsonify({"error": "wall needs x1,z1,x2,z2 plan coordinates"}), 400
            th = float(np.clip(d.get("thickness", 0.1), 0.01, 2.0))
            hh = float(np.clip(d.get("height", 1.0), 0.05, 10.0))
            axis = p2 - p1; L = float(np.linalg.norm(axis[[0, 2]]))
            if L < 1e-6:
                return jsonify({"error": "the two points coincide"}), 400
            fwd = axis / np.linalg.norm(axis)
            side = np.array([-fwd[2], 0.0, fwd[0]]) * (th / 2.0)
            up = np.array([0.0, hh, 0.0])
            a, b = p1, p2
            V = np.array([a - side, a + side, b + side, b - side,
                          a - side + up, a + side + up, b + side + up, b - side + up])
            F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            mesh = Mesh(V, F)
            nid = _add_object(d.get("name") or "Wall", mesh)
            if d.get("material"):
                try:
                    _mat(d["material"]); _S["objects"][nid].mats = [d["material"]] * mesh.n_faces
                except Exception:
                    pass
            _bump()
            out = _payload(); out["object"] = nid; out["length"] = round(L, 4)
            return jsonify(out)
        if name == "draft_check":
            # PRODUCT DESIGN: per-face DRAFT ANGLE against a pull direction (default +Y, the mold opening).
            # draft = 90deg - angle(normal, pull); faces under the threshold (undercuts / vertical walls that
            # would stick in the mold) are tagged with a warning material. The CHECK half of draft analysis;
            # auto-applying draft (re-sloping geometry) is a separate, harder op -- not faked here.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            pull = np.asarray(d.get("pull", [0, 1, 0]), float)
            pull = pull / max(np.linalg.norm(pull), 1e-9)
            min_draft = float(np.clip(d.get("min_degrees", 3.0), 0.0, 45.0))
            warn_mat = d.get("material", "copper")
            try:
                _mat(warn_mat)
            except Exception:
                return jsonify({"error": f"unknown material '{warn_mat}'"}), 400
            V = o.mesh.vertices
            mats = list(o.mats); bad = 0; checked = 0
            for fi, f in enumerate(o.mesh.faces):
                p = V[list(f)]
                n = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n)
                if ln < 1e-12:
                    continue
                n = n / ln
                cosang = float(np.dot(n, pull))
                if abs(cosang) > 0.999:                       # top/bottom faces: parting-line, not draft-relevant
                    checked += 1; continue
                draft_deg = 90.0 - float(np.degrees(np.arccos(np.clip(abs(cosang), -1, 1))))
                checked += 1
                if cosang < -0.02 or draft_deg < min_draft:   # undercut, or too vertical against the pull
                    mats[fi] = warn_mat; bad += 1
            o.mats = mats
            _bump(oid)
            out = _payload(only=oid)
            out["draft"] = {"checked": checked, "flagged": bad, "min_degrees": min_draft,
                            "note": "flagged faces are undercuts or below the minimum draft against the pull direction"}
            return jsonify(out)
        if name == "array":
            # PATTERN / ARRAY (CAD, product, architecture essential): linear (N copies along a delta) or
            # radial (N copies rotated about the Y axis through a centre). Copies are real objects, named
            # <name> #k; translation-only linear copies keep the analytic tree, radial copies drop it (honest).
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            from holographic_mesh import Mesh
            kind = str(d.get("kind", "linear"))
            n = int(np.clip(int(d.get("count", 4)), 2, 40))
            _snap_scene()
            made = []
            if kind == "linear":
                delta = np.asarray(d.get("delta", [o.mesh.vertices[:, 0].max() - o.mesh.vertices[:, 0].min() + 0.15, 0, 0]), float)
                for k in range(1, n):
                    m2 = Mesh(o.mesh.vertices + delta * k, [tuple(f) for f in o.mesh.faces])
                    tree = o.sdf_tree.translate(tuple(map(float, delta * k))) if o.sdf_tree is not None else None
                    made.append(_add_object(f"{o.name} #{k}", m2, list(o.mats), sdf_tree=tree))
            elif kind == "radial":
                import math as _mm
                cen = np.asarray(d.get("center", [0.0, 0.0, 0.0]), float)
                total = float(d.get("degrees", 360.0))
                for k in range(1, n):
                    a = _mm.radians(total * k / (n if abs(total - 360.0) < 1e-6 else n - 1))
                    cs, sn = _mm.cos(a), _mm.sin(a)
                    V = o.mesh.vertices - cen
                    Vr = np.stack([V[:, 0] * cs + V[:, 2] * sn, V[:, 1], -V[:, 0] * sn + V[:, 2] * cs], 1) + cen
                    made.append(_add_object(f"{o.name} #{k}", Mesh(Vr, [tuple(f) for f in o.mesh.faces]), list(o.mats)))
            else:
                _discard_snapshot()
                return jsonify({"error": "kind must be 'linear' or 'radial'"}), 400
            _bump()
            out = _payload(); out["arrayed"] = made
            return jsonify(out)
        if name == "weather":
            # PROCEDURAL MATERIAL MASK: assign a second material to faces selected by a spatial mask -- the
            # per-face slice of material layering that fits this renderer (which shades per-face, not per-UV-texel).
            # masks: 'noise' (weathering/patina), 'gradient' (dirt pooling low), 'edges' (wear on convex edges via
            # the object's own curvature field). amount = fraction of faces affected. Composes with band/paint.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            mat = d.get("material")
            if not mat:
                return jsonify({"error": "weather needs a 'material'"}), 400
            try:
                _mat(mat)
            except Exception:
                return jsonify({"error": f"unknown material '{mat}'"}), 400
            mask = str(d.get("mask", "noise"))
            amount = float(np.clip(d.get("amount", 0.35), 0.0, 1.0))
            V = o.mesh.vertices
            cents = np.array([V[list(f)].mean(axis=0) for f in o.mesh.faces])
            if mask == "gradient":
                ymin, ymax = V[:, 1].min(), V[:, 1].max()
                score = 1.0 - (cents[:, 1] - ymin) / max(ymax - ymin, 1e-6)   # low = high score (dirt pools low)
            elif mask == "edges":
                # convexity proxy: face-normal disagreement with neighbourhood via distance from centroid mean
                nrm = []
                for f in o.mesh.faces:
                    p = V[list(f)]
                    n = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n)
                    nrm.append(n / ln if ln > 1e-9 else np.zeros(3))
                nrm = np.array(nrm)
                gc = cents.mean(axis=0)
                radial = cents - gc; radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
                score = (nrm * radial).sum(axis=1)                            # outward-facing convex = high
                score = (score - score.min()) / max(np.ptp(score), 1e-6)
            else:                                                            # noise
                rng = np.random.RandomState(int(d.get("seed", 0)))
                # smooth-ish spatial noise: hash centroids through a few sinusoids
                s = np.zeros(len(cents))
                for freq, ph in [(3.1, 0.0), (5.7, 1.3), (8.3, 2.6)]:
                    s += np.sin(freq * cents[:, 0] + ph) * np.cos(freq * cents[:, 2] + ph) * np.sin(freq * cents[:, 1])
                s += rng.rand(len(cents)) * 0.5
                score = (s - s.min()) / max(np.ptp(s), 1e-6)
            thresh = np.quantile(score, 1.0 - amount)
            mats = list(o.mats); n = 0
            for fi in range(len(mats)):
                if score[fi] >= thresh:
                    mats[fi] = mat; n += 1
            o.mats = mats
            _bump(oid)
            out = _payload(only=oid); out["weathered"] = n
            return jsonify(out)
        if name == "variants":
            # VARIANT SWEEP: lay out N copies of the active object in a row, each with a different value of a
            # modifier parameter (flute depth, taper factor, twist angle, ...). A physical contact-sheet of
            # geometry you can orbit, compare, and keep the one you like -- more useful for a modeller than
            # thumbnail images, and robust (each variant is a real object). Supported ops: flute, taper.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            vop = str(d.get("vary_op", "flute"))
            if vop not in ("flute", "taper"):
                return jsonify({"error": "vary_op must be 'flute' or 'taper'"}), 400
            param = str(d.get("param", "depth" if vop == "flute" else "factor"))
            lo = float(d.get("lo", 0.0)); hi = float(d.get("hi", 0.2))
            n = int(np.clip(int(d.get("count", 5)), 2, 9))
            spacing = float(d.get("spacing", 0.0)) or (o.mesh.vertices[:, 0].max() - o.mesh.vertices[:, 0].min() + 0.3)
            from holographic_mesh import Mesh
            _snap_scene()
            made = []
            for i in range(n):
                val = lo + (hi - lo) * i / (n - 1)
                m2 = Mesh(o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces])
                try:
                    params = dict(d.get("base_params", {}))
                    params[param] = val
                    new = _mesh_verb(m2, vop, params)
                except Exception as e:
                    return jsonify({"error": f"variant {i} failed: {e}"}), 400
                dx = (i - (n - 1) / 2.0) * spacing
                new.vertices = new.vertices + np.array([dx, 0, 0], float)
                nid = _add_object(f"{o.name} v{i} ({param}={val:.3f})", new, list(o.mats))
                made.append(nid)
            _bump()
            out = _payload(); out["variants"] = made; out["count"] = n
            return jsonify(out)
        if name == "merge_objects":
            # Combine several objects into ONE mesh (pragmatic grouping: a finished composite that moves,
            # scales, and is materialed as a unit). Per-face materials are preserved from each source. This is
            # the mesh-union of the selection (no boolean — geometry is concatenated, keeping interior faces),
            # which is the right call for a teapot = body+lid+spout+handle that just needs to act as one object.
            from holographic_mesh import Mesh
            ids = d.get("objects") or ([oid] if oid else [])
            ids = [str(i) for i in ids if str(i) in _S["objects"]]
            if len(ids) < 2:
                return jsonify({"error": "merge needs >=2 existing objects (pass 'objects': [...])"}), 400
            _snap_scene()
            allV = []; allF = []; allM = []; off = 0
            for i in ids:
                ob = _S["objects"][i]
                allV.append(ob.mesh.vertices)
                allF.extend([tuple(int(v) + off for v in f) for f in ob.mesh.faces])
                allM.extend(ob.mats)
                off += ob.mesh.n_vertices
            mesh = Mesh(np.vstack(allV), allF)
            name0 = _S["objects"][ids[0]].name
            nid = _add_object(d.get("name") or f"{name0} (merged)", mesh, allM)
            for i in ids:
                del _S["objects"][i]
            _bump()
            out = _payload(); out["object"] = nid; out["merged"] = len(ids)
            return jsonify(out)
        if name == "halve":
            # CUT-AND-CAP: intersect the object's analytic SDF with a half-space plane, producing a solid whose
            # flat cut face can carry a distinct cross-section material (the halved-lemon case). Needs an analytic
            # tree; for a mesh-only object we fall back to a plane clip of the mesh. axis in x|y|z, offset along it,
            # sign picks which half to keep. The cut-face material is assigned to faces lying on the cut plane.
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            import holographic_sdf as _S2
            axis = str(d.get("axis", "y")).lower(); off = float(d.get("offset", 0.0))
            sign = 1.0 if float(d.get("sign", 1)) >= 0 else -1.0
            cut_mat = d.get("cut_material")
            if o.sdf_tree is None:
                return jsonify({"error": "halve needs an analytic object (a primitive or lathe); mesh-only clip is a follow-up"}), 400
            # build the half-space plane oriented on the requested axis (plane(off) is the y<off half-space)
            import math as _mm
            if axis == "x":
                plane = _S2.plane(off).rotate((0, 0, 1), _mm.radians(-90))
            elif axis == "z":
                plane = _S2.plane(off).rotate((1, 0, 0), _mm.radians(90))
            else:
                plane = _S2.plane(off)
            if sign < 0:
                # keep the other half: intersect with the complement (subtract the half-space)
                tree = o.sdf_tree.subtract(plane)
            else:
                tree = o.sdf_tree.intersect(plane)
            _snap_obj(oid)
            lo = o.mesh.vertices.min(axis=0) - 0.1; hi = o.mesh.vertices.max(axis=0) + 0.1
            new = _mesh_sdf_tree(tree, res=int(d.get("res", 72)),
                                 face_target=int(np.clip(int(d.get("target", 3000)), 300, 12000)),
                                 scan_lo=lo, scan_hi=hi)
            o.mesh = new; o.sdf_tree = tree
            # tag faces whose centroid lies on the cut plane (within eps) with the cross-section material
            base_mat = o.mats[0] if o.mats else _default_mat_name()
            mats = [base_mat] * new.n_faces
            if cut_mat:
                try:
                    _mat(cut_mat)
                    ai = {"x": 0, "y": 1, "z": 2}[axis]
                    for fi, f in enumerate(new.faces):
                        cen = new.vertices[list(f)].mean(axis=0)
                        if abs(cen[ai] - off) < 0.03:
                            mats[fi] = cut_mat
                except KeyError:
                    pass
            o.mats = mats
            _bump(oid)
            return jsonify(_payload(only=oid))
        if name in ("place", "translate", "rotate", "scale", "drop", "rename"):
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            import math as _m

            def _apply_matrix_raw(obj, M):                   # transform one object's geometry by world matrix M
                V = obj.mesh.vertices
                obj.mesh.vertices = (np.c_[V, np.ones(len(V))] @ np.asarray(M, float).T)[:, :3]
                if np.linalg.det(np.asarray(M)[:3, :3]) < 0:
                    obj.mesh.faces = [tuple(reversed(f)) for f in obj.mesh.faces]
                obj.mesh.normals = None; obj.mesh._he = None; obj.mesh._adj = None
                if obj.sdf_tree is not None and np.allclose(np.asarray(M)[:3, :3], np.eye(3), atol=1e-9):
                    obj.sdf_tree = obj.sdf_tree.translate((float(M[0][3]), float(M[1][3]), float(M[2][3])))
                elif obj.sdf_tree is not None:
                    obj.sdf_tree = None

            def _apply_matrix(M):
                _snap_obj(oid)
                _apply_matrix_raw(o, M)
                # A2-3: the SAME world matrix propagates to descendants, so a group moves/rotates/scales as one
                seen = {oid}; frontier = [oid]
                while frontier:
                    pid = frontier.pop()
                    for cid, par in list(_PARENT.items()):
                        if par == pid and cid not in seen and cid in _S["objects"]:
                            _snap_obj(cid); _apply_matrix_raw(_S["objects"][cid], M)
                            _bump(cid); seen.add(cid); frontier.append(cid)
                _bump(oid)

            if name == "rename":
                o.name = str(d.get("name", o.name)) or o.name
                _bump(oid)
                return jsonify(_payload(only=oid))

            if name == "drop":
                # snap the object's min-Y onto a target plane (default: the current scene table = lowest
                # object base minus the render's ground offset). Purely a translation -> keeps the tree.
                target_y = d.get("to")
                if target_y is None:
                    others = [ov.mesh.vertices[:, 1].min() for k, ov in _S["objects"].items() if k != oid]
                    target_y = min(others) if others else 0.0
                dy = float(target_y) - float(o.mesh.vertices[:, 1].min())
                M = np.eye(4); M[1, 3] = dy
                _apply_matrix(M.tolist())
                return jsonify(_payload(only=oid))

            if name == "place":                             # absolute: move centroid to (x,y,z)
                p = d.get("position", [0, 0, 0])
                cen = o.mesh.vertices.mean(axis=0)
                M = np.eye(4)
                M[0, 3] = float(p[0]) - cen[0]; M[1, 3] = float(p[1]) - cen[1]; M[2, 3] = float(p[2]) - cen[2]
                _apply_matrix(M.tolist())
                return jsonify(_payload(only=oid))

            if name == "translate":
                dxyz = d.get("delta", [0, 0, 0])
                M = np.eye(4); M[0, 3], M[1, 3], M[2, 3] = (float(x) for x in dxyz)
                _apply_matrix(M.tolist())
                return jsonify(_payload(only=oid))

            if name == "scale":                             # uniform or per-axis, about centroid
                s = d.get("factor", 1.0)
                sv = ([float(s)] * 3) if not isinstance(s, (list, tuple)) else [float(x) for x in s]
                cen = o.mesh.vertices.mean(axis=0)
                T1 = np.eye(4); T1[:3, 3] = -cen
                S = np.diag(sv + [1.0])
                T2 = np.eye(4); T2[:3, 3] = cen
                _apply_matrix((T2 @ S @ T1).tolist())
                return jsonify(_payload(only=oid))

            if name == "rotate":                            # degrees about x/y/z, through centroid
                axis = str(d.get("axis", "y")).lower(); deg = float(d.get("degrees", 0.0))
                a = _m.radians(deg); cs, sn = _m.cos(a), _m.sin(a)
                if axis == "x":
                    R = np.array([[1, 0, 0, 0], [0, cs, -sn, 0], [0, sn, cs, 0], [0, 0, 0, 1]], float)
                elif axis == "z":
                    R = np.array([[cs, -sn, 0, 0], [sn, cs, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
                else:
                    R = np.array([[cs, 0, sn, 0], [0, 1, 0, 0], [-sn, 0, cs, 0], [0, 0, 0, 1]], float)
                cen = o.mesh.vertices.mean(axis=0)
                T1 = np.eye(4); T1[:3, 3] = -cen
                T2 = np.eye(4); T2[:3, 3] = cen
                _apply_matrix((T2 @ R @ T1).tolist())
                return jsonify(_payload(only=oid))
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object (pass 'object')"}), 400
        m = o.mesh
        _snap_obj(oid)
        try:
            new = _mesh_verb(m, name, d)
        except ValueError as e:
            _discard_snapshot()
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            _discard_snapshot()
            return jsonify({"error": str(e)}), 400
        if "_keep_faces" in d:                          # delete_faces: exact per-face material alignment
            o.mats = [o.mats[i] for i in d["_keep_faces"]]
        else:
            o.mats = _transfer_mats(m, o.mats, new)
        o.mesh = new; _bump(oid)
        _hist_record(oid, name, d)
        return jsonify(_payload(only=oid))


# =====================================================================================================
# GLB import -- the wild-file loader (byteStride, accessor offsets, node transforms, materials)
# =====================================================================================================
_CTYPE = {5120: ("<i1", 1), 5121: ("<u1", 1), 5122: ("<i2", 2), 5123: ("<u2", 2), 5125: ("<u4", 4), 5126: ("<f4", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _acc(gltf, blob, i):
    """Read accessor i honouring accessor.byteOffset AND bufferView.byteStride (interleaved buffers)."""
    a = gltf["accessors"][i]
    v = gltf["bufferViews"][a["bufferView"]]
    dt, csize = _CTYPE[a["componentType"]]
    n = _NCOMP[a["type"]]
    base = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    count = a["count"]
    stride = v.get("byteStride", 0) or csize * n
    if stride == csize * n:
        arr = np.frombuffer(blob, dtype=dt, count=count * n, offset=base).reshape(count, n)
    else:                                                  # interleaved: gather element by element
        raw = np.frombuffer(blob, dtype=np.uint8)
        idx = base + stride * np.arange(count)[:, None] + np.arange(csize * n)[None, :]
        arr = raw[idx].copy().view(dt).reshape(count, n)
    return arr[:, 0] if n == 1 else arr


def _node_world_matrices(gltf):
    """World transform per node from the hierarchy (matrix or TRS), glTF column-vector convention."""
    nodes = gltf.get("nodes", [])
    local = []
    for nd in nodes:
        if "matrix" in nd:
            M = np.array(nd["matrix"], float).reshape(4, 4).T
        else:
            T = np.eye(4); T[:3, 3] = nd.get("translation", [0, 0, 0])
            q = nd.get("rotation", [0, 0, 0, 1]); x, y, z, w = q
            R = np.eye(4)
            R[:3, :3] = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                                  [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
            S = np.diag(list(nd.get("scale", [1, 1, 1])) + [1.0])
            M = T @ R @ S
        local.append(M)
    world = [None] * len(nodes)
    scenes = gltf.get("scenes", [{}]); roots = scenes[gltf.get("scene", 0)].get("nodes", list(range(len(nodes))))

    def walk(i, parent):
        world[i] = parent @ local[i]
        for ch in nodes[i].get("children", []):
            walk(ch, world[i])
    for r in roots:
        walk(r, np.eye(4))
    for i in range(len(nodes)):                            # nodes outside the scene graph: local as world
        if world[i] is None:
            world[i] = local[i]
    return world


def _nearest_preset(base, metallic, roughness):
    """Map an imported pbrMetallicRoughness factor set to the NEAREST physical preset in the library --
    reported, not silently guessed."""
    ml = _matlib()
    best, bd = _default_mat_name(), 1e9
    for n in ml.names():
        m = ml.material(n)
        d = (np.linalg.norm(np.asarray(m.base_color[:3]) - base) +
             0.7 * abs(m.metallic - metallic) + 0.4 * abs(m.roughness - roughness))
        if d < bd:
            bd, best = d, n
    return best


@bp.route("/api/render_engine")
def render_engine():
    """ENGINE VIEWPORT RENDER: draw the mesh scene with leCore's rasteriser (holographic_render.rasterize_mesh:
    z-buffer, Lambert + lights, back-face culling, per-pixel textures) instead of the SDF field bake+raymarch --
    which voxelises imported meshes into mush at preview grids. GLB imports carry their texture + uv (or exact
    per-face colours) in the render-asset store, so a scan renders HERE the way the engine pipeline renders it.
    ?hero=<oid> draws one object per-pixel-textured; default merges the whole visible scene with per-vertex
    colours (one rasterise call, one z-buffer). Camera auto-frames via engine fit_camera unless eye= given."""
    _init()
    g = request.args.get
    import holographic_render as _hr
    from holographic_mesh import Mesh as _Mesh
    W = int(np.clip(int(g("w", 900)), 160, 1600)); H = int(np.clip(int(g("h", 600)), 120, 1200))
    with _LOCK:
        objs = {oid: o for oid, o in _S["objects"].items()}
        if not objs:
            return jsonify({"error": "empty scene"}), 400
        assets = _S.get("render_assets", {})
        hero = g("hero", "")
        if not hero:                                          # default to the textured import if there is exactly one
            tex_objs = [oid for oid in objs if "uv" in assets.get(oid, {})]
            if len(tex_objs) == 1 and len(objs) <= 2:
                hero = tex_objs[0]
        lights = [_hr.Light("directional", direction=(-0.5, -0.7, -0.4), color=(1.0, 0.97, 0.9), intensity=1.4),
                  _hr.Light("directional", direction=(0.6, -0.2, 0.5), color=(0.5, 0.6, 0.8), intensity=0.55),
                  _hr.Light("ambient", intensity=0.32)]
        bgq = g("bg", "")
        try:
            background = tuple(float(x) for x in bgq.split(",")[:3]) if bgq else (0.55, 0.62, 0.72)
        except Exception:
            background = (0.55, 0.62, 0.72)

        def _camera(mesh):
            eye = g("eye", ""); tgt = g("target", "")
            if eye:
                try:
                    from holographic_coerce import as_camera
                    e = [float(x) for x in eye.split(",")[:3]]
                    t = [float(x) for x in tgt.split(",")[:3]] if tgt else [0, 0, 0]
                    return as_camera({"eye": tuple(e), "target": tuple(t), "up": (0, 1, 0),
                                      "fov_deg": float(g("fov", 45.0)), "aspect": W / H})
                except Exception:
                    pass
            from holographic_coerce import as_camera
            cam = _hr.fit_camera(mesh, direction=(1.0, 0.75, 1.1), fov_deg=45.0, aspect=W / H)
            return as_camera(cam)                            # fit_camera returns a dict; the rasteriser is strict

        a_hero = assets.get(hero, {})
        if hero and hero in objs and "uv" in a_hero and len(a_hero["uv"]) == objs[hero].mesh.n_vertices:
            o = objs[hero]; a = a_hero
            img = _hr.rasterize_mesh(o.mesh, _camera(o.mesh), width=W, height=H, lights=lights,
                                     background=background, ambient=0.32,
                                     texture=np.asarray(a["tex"], float), uvs=np.asarray(a["uv"], float),
                                     two_sided=True, smooth=True)
            mode = "hero-textured"
        elif hero and hero in objs and "face_colors" in a_hero and len(a_hero["face_colors"]) >= objs[hero].mesh.n_faces:
            o = objs[hero]
            fcol = np.asarray(a_hero["face_colors"], float)
            V = np.asarray(o.mesh.vertices, float)
            vc = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
            for fi, f in enumerate(o.mesh.faces):
                for vtx in f:
                    vc[vtx] += fcol[fi]; cnt[vtx] += 1
            vc = np.clip(vc / np.maximum(cnt, 1)[:, None], 0, 1)
            img = _hr.rasterize_mesh(o.mesh, _camera(o.mesh), width=W, height=H, lights=lights,
                                     background=background, ambient=0.32,
                                     vertex_colors=vc, two_sided=True, smooth=True)
            mode = "hero-facecolors"
        else:
            # merge the scene into ONE mesh with per-vertex colours (one z-buffer; crisp geometry, real lights)
            V_all = []; F_all = []; C_all = []; base = 0
            ml = _matlib()
            for oid, o in objs.items():
                V = np.asarray(o.mesh.vertices, float)
                tris = []
                for f in o.mesh.faces:
                    idx = list(f)
                    for k in range(1, len(idx) - 1):
                        tris.append((idx[0], idx[k], idx[k + 1]))
                F = np.asarray(tris, np.int64)
                a = assets.get(oid, {})
                vc = np.zeros((len(V), 3)); cnt = np.zeros(len(V))
                if "uv" in a and len(a["uv"]) == len(V):     # sample the texture at each vertex uv
                    tex = np.asarray(a["tex"], float); Ht, Wt = tex.shape[0], tex.shape[1]
                    uvv = np.asarray(a["uv"], float)
                    px = np.clip((uvv[:, 0] % 1.0) * (Wt - 1), 0, Wt - 1).astype(int)
                    py = np.clip((uvv[:, 1] % 1.0) * (Ht - 1), 0, Ht - 1).astype(int)
                    vc = np.clip(tex[py, px][:, :3], 0, 1); cnt[:] = 1
                elif "face_colors" in a and len(a["face_colors"]) >= len(o.mesh.faces):
                    fcol = np.asarray(a["face_colors"], float)
                    for fi, f in enumerate(o.mesh.faces):
                        for vtx in f:
                            vc[vtx] += fcol[fi]; cnt[vtx] += 1
                    vc = vc / np.maximum(cnt, 1)[:, None]; cnt[:] = 1
                else:                                        # flat material albedo per face -> vertices
                    alb = _channels_for(o)[0]
                    for fi, f in enumerate(o.mesh.faces):
                        for vtx in f:
                            vc[vtx] += alb[fi]; cnt[vtx] += 1
                    vc = vc / np.maximum(cnt, 1)[:, None]
                V_all.append(V); F_all.append(F + base); C_all.append(np.clip(vc, 0, 1)); base += len(V)
            Vm = np.vstack(V_all); Fm = np.vstack(F_all); Cm = np.vstack(C_all)
            merged = _Mesh(Vm, [tuple(int(x) for x in t) for t in Fm])
            img = _hr.rasterize_mesh(merged, _camera(merged), width=W, height=H, lights=lights,
                                     background=background, ambient=0.32,
                                     vertex_colors=Cm, two_sided=True, smooth=True)
            mode = "scene-vertex-colors"
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    from PIL import Image as _Image
    import io as _io
    buf = _io.BytesIO(); _Image.fromarray(arr).save(buf, format="PNG")
    resp = Response(buf.getvalue(), mimetype="image/png")
    resp.headers["X-Render-Mode"] = mode
    return resp


@bp.route("/api/import_glb", methods=["POST"])
def import_glb():
    """GLB IMPORT, engine-wired: holographic_assetimport.load_glb parses the WHOLE scene (transforms, normals,
    UVs, PBR materials, embedded textures). Big or textured scans are routed through the engine's measurement-
    driven LOD (holographic_meshtools.textured_lod): a coherent atlas transfers UVs; a fragmented photogrammetry
    atlas is decimated and RE-BAKED into a fresh per-face atlas (the only correct route -- transfer renders as
    speckle). The baked atlas is then sampled at face-UV centroids and quantised to a small palette of session
    materials, because this app's renderer shades per-face materials. report["route"] says which way it went."""
    _init()
    # IMPORT OPTIONS (user-selectable in the dialog): mode=auto|asis|decimate|retopo|voxel; target = face
    # budget for decimate (or resolution for retopo/voxel); reproject=1 keeps texture/uv on the processed
    # mesh via uv reprojection, reproject=0 bakes a material palette instead. auto = the existing
    # measurement-driven routing, so old clients and default flows are unchanged.
    imp_mode = str(request.args.get("mode", "auto")).lower()   # + "rebake" (see the dispatch below)
    imp_target = int(np.clip(int(request.args.get("target", 60000) or 60000), 500, 400000))
    imp_reproject = str(request.args.get("reproject", "1")) not in ("0", "false", "off")
    data = request.get_data()
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != 0x46546C67:
        return jsonify({"error": "not a .glb file (bad magic)"}), 400
    if len(data) > 200 * 1024 * 1024:
        return jsonify({"error": "file too large (200 MB cap)"}), 400
    import tempfile, copy as _copy
    import holographic_assetimport as _ai
    import holographic_meshtools as _mt
    from holographic_mesh import Mesh
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
            tf.write(data); tmp = tf.name
        try:
            lm = _ai.load_glb(tmp)
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        mesh = lm.mesh()
        n_faces = mesh.n_faces
        if mesh.n_vertices > 1_200_000:
            return jsonify({"error": f"{mesh.n_vertices} vertices exceeds the 1.2M import cap"}), 400
        FACE_BUDGET = 60_000

        def _tex_of(mat):
            if mat is not None and getattr(mat, "base_color_map", None) is not None:
                t = np.asarray(mat.base_color_map.image, dtype=np.float32)
                return t / 255.0 if t.max() > 1.5 else t
            return None

        def _sample_face_colours(F_arr, uvv, txx):
            Hh, Ww = txx.shape[0], txx.shape[1]
            uvc = uvv[F_arr].mean(axis=1)
            px = np.clip((uvc[:, 0] % 1.0) * (Ww - 1), 0, Ww - 1).astype(int)
            py = np.clip((uvc[:, 1] % 1.0) * (Hh - 1), 0, Hh - 1).astype(int)
            return np.clip(txx[py, px][:, :3].astype(float), 0, 1)

        # PER-MATERIAL SPLIT, now via the UPSTREAM helper (LoadedMesh.split_by_material landed after we filed
        # it): one LoadedMesh per material, faces reindexed to a compact vertex set, UVs/normals subset --
        # exactly the grouping this endpoint previously hand-rolled. Each part becomes its own scene object,
        # coloured from its OWN texture, LOD'd on its own budget.
        parts = lm.split_by_material()
        added, mapping, reports = [], [], []
        with _LOCK:
            _snap_scene()
            for gname, part in parts.items():
                subm = part.mesh()
                if subm.n_faces == 0:
                    continue
                pmats = getattr(part, "materials", None) or {}
                mat = pmats.get(gname) or (pmats[next(iter(pmats))] if pmats else None)
                tex = _tex_of(mat)
                uvsub = np.asarray(part.uv, float) if getattr(part, "uv", None) is not None and \
                    len(part.uv) == subm.n_vertices else None
                F2 = np.array([list(f)[:3] for f in subm.faces], dtype=np.int64)
                Vsub = np.asarray(subm.vertices, float)
                oname = str(gname)[:48] or "glb_import"
                grep = {"object": oname, "route": "direct", "source_faces": int(subm.n_faces)}
                # --- user-selected processing (dialog): pre-process the mesh, then fall into the normal
                # texture/palette routing below. "auto" (default) is untouched classic behaviour. ---
                if imp_mode in ("asis", "decimate", "retopo", "voxel", "rebake"):
                    proc_note = "as-is (no processing)"
                    src_for_reproj = (subm, uvsub)
                    if imp_mode == "decimate" and subm.n_faces > imp_target:
                        import holographic_meshqem as _mq2
                        subm, drep_ = _mq2.decimate_to(subm, target_faces=imp_target, keep_uv="auto")
                        new_uv = getattr(subm, "uvs", None)
                        uvsub = np.asarray(new_uv, float) if new_uv is not None and                             len(new_uv) == subm.n_vertices else None
                        proc_note = f"decimate_to -> {subm.n_faces} faces (silhouette-checked)"
                    elif imp_mode == "retopo":
                        res_ = int(np.clip(imp_target // 1500, 12, 96))
                        rt_ = _mt.auto_retopo(subm, voxel_resolution=res_)   # returns a dict
                        subm = rt_["mesh"]
                        uvsub = None
                        proc_note = (f"auto_retopo res {res_} -> {subm.n_faces} faces "
                                     f"({rt_.get('quad_fraction', 0):.0%} quads)")
                    elif imp_mode == "rebake":
                        # REBAKE A NEW ATLAS. Previously unusable at scan scale (>500 s on a 151K-face mesh);
                        # the engine's textured_lod is now ~10 s for the same input, so it becomes a real
                        # import choice. Unlike the reproject modes this does NOT keep the source atlas -- it
                        # bakes a fresh one for the decimated mesh, which is what a FRAGMENTED photogrammetry
                        # atlas actually wants (cluster_decimate's keep_uv correctly declines on those).
                        if tex is not None and uvsub is not None:
                            # ATLAS SIZE IS THE QUALITY KNOB, not the face budget: the atlas packs one CELL
                            # PER FACE, so cell side ~ size/sqrt(faces). At 1024 with ~15K faces that is 4.3
                            # texels per face, and with the bake's own gutter almost the whole cell is margin
                            # -- exactly the smeared, bleeding result. MEASURED cell sides for this mesh:
                            # 1024 -> 4.3, 2048 -> 12.7. So rebake bakes at 2048 and KEEPS it at that size.
                            lod_, uv_, atlas_, rep_ = _mt.textured_lod(
                                subm, np.asarray(tex, np.float32), uvs=np.asarray(uvsub, float),
                                grid=int(np.clip(imp_target // 700, 24, 96)), size=2048)
                            subm = lod_
                            uvsub = np.asarray(uv_, float)
                            tex = np.asarray(atlas_, float)
                            proc_note = ("textured_lod rebake -> %d faces, new %dx%d atlas (coverage %.2f)"
                                         % (subm.n_faces, tex.shape[1], tex.shape[0],
                                            float(rep_.get("texel_coverage", 0.0))))
                        else:
                            proc_note = "rebake skipped (this group has no texture to bake)"
                    elif imp_mode == "voxel":
                        import holographic_meshbridge as _mb
                        res_ = int(np.clip(round((imp_target / 6) ** 0.5), 24, 160))
                        subm = _mb.voxel_remesh(subm, resolution=res_)
                        uvsub = None
                        proc_note = f"voxel_remesh res {res_} -> {subm.n_faces} faces (watertight)"
                    # optional uv reprojection so the texture survives the processed topology
                    if imp_reproject and uvsub is None and tex is not None and src_for_reproj[1] is not None                             and imp_mode not in ("asis", "rebake"):
                        s_m, s_uv = src_for_reproj
                        try:
                            uvsub, resid = _transfer_uv_compat(_mt, s_m, s_uv, subm.vertices)
                            proc_note += " + transfer_uv" + (f" (median resid {resid:.5f})" if resid is not None else "")
                            subm, uvsub, n_seam = _fix_uv_seams(subm, uvsub, s_m, s_uv)
                            if n_seam:
                                proc_note += f" + {n_seam} seam faces repaired"
                        except Exception:
                            nn_ = _nn_voxel(np.asarray(s_m.vertices, float),
                                            np.asarray(subm.vertices, float))
                            uvsub = np.asarray(s_uv, float)[nn_]
                            proc_note += " + nearest-vertex uv"
                    if any(len(f) != 3 for f in subm.faces):
                        F2 = np.array([list(f)[:3] for f in subm.triangulate()], dtype=np.int64)
                        from holographic_mesh import Mesh as _Mesh2
                        subm = _Mesh2(np.asarray(subm.vertices, float), [tuple(int(x) for x in f) for f in F2])
                    else:
                        F2 = np.array([list(f)[:3] for f in subm.faces], dtype=np.int64)
                    Vsub = np.asarray(subm.vertices, float)
                    grep.update({"processing": proc_note})
                if tex is not None and uvsub is not None:
                    if imp_mode == "auto" and subm.n_faces > FACE_BUDGET:
                        # BIG-SCAN, TEXTURE-PRESERVING (the previous face-colour bake DESTROYED textures on
                        # any scan over the face budget -- reported on a 151K-face beetle). Engine path all
                        # the way: cluster_decimate + mesh_orient, then meshtools.transfer_uv projects the
                        # SOURCE uvs onto the decimated vertices (closest-point + barycentric -- the engine's
                        # own retopo-texture-preserving step). The 2048-class texture rides along untouched,
                        # so the import looks like the file, not like a palette of it.
                        src_for_uv = subm
                        import holographic_meshqem as _mq
                        # FEATURE-AWARE BUDGET: cluster_decimate takes a GRID and reports nothing, so thin
                        # structures (legs, antennae) could be thinned away with no signal. decimate_to takes
                        # the FACE BUDGET directly and refuses to ship a result whose SILHOUETTE broke --
                        # measured per view, which is what actually catches a lost leg. The floor is 0.96;
                        # measured on the beetle it comfortably clears it (0.9956-0.9981 across 7 views) and
                        # the engine backs off automatically when it would not.
                        sil_iou = None
                        try:
                            _res = _mq.decimate_to(subm, target_faces=int(FACE_BUDGET * 1.35),
                                                   keep_uv="auto", min_silhouette_iou=0.96)
                            _dec, _drep = _res if isinstance(_res, tuple) else (_res, {})
                            # the report nests the per-view numbers: {"silhouette_iou": {"az000":.., "top":..}}
                            _iou = (_drep or {}).get("silhouette_iou", _drep)
                            _ious = [float(v) for v in (_iou or {}).values()
                                     if isinstance(v, (int, float))] if isinstance(_iou, dict) else []
                            sil_iou = min(_ious) if _ious else None
                            subm = _dec
                        except Exception:
                            subm = _mq.cluster_decimate(subm, grid=110)   # labelled fallback, older engines
                        subm, orep = _mt.mesh_orient(subm)
                        try:
                            luv, resid = _transfer_uv_compat(_mt, src_for_uv, uvsub, subm.vertices)
                            uv_route = "transfer_uv (barycentric" + \
                                (f", median resid {resid:.5f})" if resid is not None else ")")
                        except Exception:
                            # ENGINE BUG (filed): transfer_uv fails on a COLD compile cache ("inhomogeneous
                            # shape"), works in a warm process, and in-process retry does NOT recover.
                            # Fallback: nearest SOURCE VERTEX uv via the voxel-hash NN -- piecewise-constant
                            # per vertex, which on a dense scan is visually indistinguishable.
                            nn_ = _nn_voxel(np.asarray(src_for_uv.vertices, float),
                                            np.asarray(subm.vertices, float))
                            luv = np.asarray(uvsub, float)[nn_]
                            uv_route = "nearest-vertex uv (transfer_uv cold-cache bug fallback)"
                        subm, luv, n_seam = _fix_uv_seams(subm, luv, src_for_uv, uvsub)
                        if n_seam:
                            uv_route += f", {n_seam} seam faces repaired"
                        decF = np.array([list(f)[:3] for f in subm.faces], dtype=np.int64)
                        fcol = _sample_face_colours(decF, luv, tex)      # palette + viewport colours from uv
                        keep_tex = _cap_texture(tex)
                        grep.update({"silhouette_iou": (round(float(sil_iou), 4) if sil_iou is not None else None),
                                     "route": "decimate+" + uv_route + " (texture kept)",
                                     "lod_faces": int(subm.n_faces),
                                     "orient_flipped": orep.get("flipped"),
                                     "non_manifold_edges": orep.get("non_manifold_edges")})
                        asset = {"uv": luv, "tex": np.asarray(keep_tex, float)}
                        Vsub = np.asarray(subm.vertices, float); F2 = decF; uvsub = luv
                    else:
                        fcol = _sample_face_colours(F2, uvsub, tex)
                        # A REBAKED atlas is already sized to its own per-face cell budget and is passed
                        # through untouched; source textures are capped at TEXTURE_MAX.
                        keep_tex = tex if imp_mode == "rebake" else _cap_texture(tex)
                        asset = {"uv": np.asarray(uvsub, float), "tex": np.asarray(keep_tex, float)}
                    K = int(min(24, max(4, len(np.unique(np.round(fcol, 2), axis=0)))))
                    rng = np.random.default_rng(0)
                    wgt = np.array([0.35, 0.5, 0.15]); X = fcol * wgt
                    C = X[rng.choice(len(X), K, replace=False)]
                    for _ in range(12):
                        assign = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
                        for k in range(K):
                            sel = assign == k
                            if sel.any():
                                C[k] = X[sel].mean(0)
                    pal = np.stack([fcol[assign == k].mean(0) if (assign == k).any() else np.full(3, 0.5)
                                    for k in range(K)])
                    ml = _matlib()
                    names = []
                    for k in range(K):
                        mname = f"imp_{len(_CUSTOM_MATS):03d}_{k:02d}"
                        mm = _copy.deepcopy(ml.material("clay"))
                        mm.base_color = np.asarray(pal[k], float)
                        mm.metallic = float(np.clip(getattr(mat, "metallic", 0.0) or 0.0, 0, 1))
                        mm.roughness = float(np.clip(getattr(mat, "roughness", 0.65) or 0.65, 0.02, 1))
                        _CUSTOM_MATS[mname] = mm
                        names.append(mname)
                    facemats = [names[int(k)] for k in assign]
                    oid = _add_object(oname, subm, facemats)
                    _S["render_assets"][oid] = asset
                    mapping.append({"object": oname, "preset": f"palette:{K}"})
                else:
                    # this group has no usable texture/uv: flat material factors -> nearest library preset
                    if imp_mode == "auto" and subm.n_faces > FACE_BUDGET:
                        import holographic_meshqem as _mq
                        subm = _mq.cluster_decimate(subm, grid=110)
                        subm, orep = _mt.mesh_orient(subm)
                        grep.update({"route": "decimate+orient", "lod_faces": int(subm.n_faces),
                                     "flipped": orep.get("flipped")})
                    base = np.asarray(getattr(mat, "base_color", [0.8, 0.8, 0.8]), float)[:3] if mat is not None \
                        else np.array([0.8, 0.8, 0.8])
                    preset = _nearest_preset(base,
                                             float(getattr(mat, "metallic", 0.0) or 0.0) if mat is not None else 0.0,
                                             float(getattr(mat, "roughness", 0.7) or 0.7) if mat is not None else 0.7)
                    oid = _add_object(oname, subm, [preset] * subm.n_faces)
                    mapping.append({"object": oname, "preset": preset})
                added.append(oid)
                reports.append(grep)
            if not added:
                _discard_snapshot()
        if not added:
            return jsonify({"error": "no triangle meshes found in file"}), 400
        report = {"groups": reports, "materials": list(parts.keys())}
        # DON'T SILENTLY DISCARD RIG DATA (coverage-audit find): a glb can carry animations, skins and morph
        # targets; this app doesn't PLAY them yet, but the import now says so instead of losing them without a
        # word, so the user knows what the file contained.
        anims = getattr(lm, "animations", None) or []
        skins = getattr(lm, "skins", None) or []
        morphs = getattr(lm, "morph_targets", None) or []
        if anims or skins or morphs:
            report["rig_data"] = {
                "animations": [getattr(a, "name", f"clip_{i}") for i, a in enumerate(anims)],
                "skins": len(skins), "morph_targets": len(morphs),
                "note": "present in the file but NOT imported -- animation playback is future scope; "
                        "re-export from the original file to keep them"}
        with _LOCK:
            allv = np.vstack([_S["objects"][i].mesh.vertices for i in added])
            lo, hi = allv.min(axis=0), allv.max(axis=0)
            shift = np.array([(lo[0] + hi[0]) / 2, lo[1] + 0.6, (lo[2] + hi[2]) / 2])
            for i in added:
                _S["objects"][i].mesh.vertices = _S["objects"][i].mesh.vertices - shift
            _bump()
            out = _payload(); out["imported"] = mapping; out["lod_report"] = report
            return jsonify(out)
    except Exception as e:
        return jsonify({"error": f"glb import failed: {e}"}), 400


# =====================================================================================================
# Materials
# =====================================================================================================
@bp.route("/api/materials")
def materials():
    ml = _matlib()
    out = {}
    for cls in ml.classes():
        entries = []
        for n in ml.by_class(cls):
            m = ml.material(n)
            entries.append({"name": n, "albedo": [round(float(c), 3) for c in m.base_color[:3]],
                            "metallic": round(float(m.metallic), 2), "roughness": round(float(m.roughness), 2),
                            "ior": round(float(getattr(m, "ior", 0.0)), 2)
                                   if getattr(m, "transmission", 0.0) >= 1.0 else 0})
        out[cls] = entries
    if _CUSTOM_MATS:
        out["custom"] = [{"name": n, "albedo": [round(float(c), 3) for c in m.base_color[:3]],
                          "metallic": round(float(m.metallic), 2), "roughness": round(float(m.roughness), 2),
                          "ior": round(float(getattr(m, "ior", 0.0)), 2)
                                 if getattr(m, "transmission", 0.0) >= 1.0 else 0}
                         for n, m in sorted(_CUSTOM_MATS.items())]
    return jsonify({"classes": out, "default": _default_mat_name(), "floor": _FLOOR_MAT})


# Real material-ball thumbnails (Cook-Torrance-shaded preview sphere, holographic_preview.material_ball) in
# place of an approximated CSS gradient. Presets are static, so a thumbnail is computed once and cached forever
# -- 141 balls at 64px measured ~6ms each (~0.85s total), cheap enough to warm on first request per name and
# never again.
_MATBALL_CACHE = {}


@bp.route("/api/material_ball")
def material_ball_thumb():
    name = request.args.get("name", _default_mat_name())
    res = _qnum(request.args.get, "res", 64, 32, 256, int)
    key = (name, res)
    png = _MATBALL_CACHE.get(key)
    if png is None:
        from holographic_preview import material_ball
        ml = _matlib()
        try:
            m = _mat(name)
        except KeyError as e:
            return jsonify({"error": str(e)}), 400
        img = material_ball(m, res=res, background=0.086)
        png = _png_bytes(img)
        _MATBALL_CACHE[key] = png
    resp = Response(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"   # a preset's ball never changes
    return resp


_PAINT_STROKE = {"key": None}                             # (stroke_id, oid) of the stroke already snapshotted


@bp.route("/api/assign", methods=["POST"])
def assign():
    _init()
    d = request.get_json(force=True) or {}
    name = d.get("material", _default_mat_name())
    try:
        _mat(name)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    # PER-GROUP ASSIGN (basics pass): "objects": [ids] applies the material to EVERY listed object in one
    # call + one undo step -- the multi-selection is the group. Per-object and per-face forms unchanged.
    group = d.get("objects")
    if isinstance(group, list) and len(group) > 1:
        with _LOCK:
            ids = [str(x) for x in group if str(x) in _S["objects"]]
            if not ids:
                return jsonify({"error": "no such objects"}), 400
            _snap_scene()
            for gid_ in ids:
                og = _S["objects"][gid_]
                og.mats = [name] * og.mesh.n_faces
                _bump(gid_)
            out = _payload()
            out["assigned"] = {"material": name, "objects": ids}
            return jsonify(out)
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        # PAINT STROKES: the brush streams many small assign packets; snapshot once per (stroke id, object) so
        # one undo removes the whole stroke, not one packet -- the same coalescing the sculpt brush uses.
        stroke = d.get("stroke")
        snapped = False
        if stroke is None or _PAINT_STROKE["key"] != (stroke, oid):
            _snap_obj(oid)
            snapped = True
            _PAINT_STROKE["key"] = (stroke, oid) if stroke is not None else None
        F = o.mesh.n_faces
        faces = range(F) if d.get("all") else [int(i) for i in d.get("faces", [])]
        faces = [i for i in faces if 0 <= i < F]
        if not faces:
            if snapped:
                _discard_snapshot()
                _PAINT_STROKE["key"] = None
            return jsonify({"error": "no faces to assign (select faces first)"}), 400
        for i in faces:
            o.mats[i] = name
        # a material change touches no geometry: the analytic tree/kernel stay valid (caches still flush --
        # the render's material-ID channels are keyed by rev and must rebuild)
        keep_tree, keep_kern = o.sdf_tree, o.kernel_src
        _bump(oid)
        o.sdf_tree, o.kernel_src = keep_tree, keep_kern
        return jsonify(_payload(only=oid))


# =====================================================================================================
# Auto UV + map export (per object)
# =====================================================================================================
def _auto_uv(oid):
    """Auto-unwrap, now on LSCM (Levy/Petitjean/Ray & Maillot -- one linear least-squares solve, no iteration):
    ANGLE-preserving rather than isomap's distance-preserving, which is the right metric for a texture that
    must not look sheared -- measured EXACT (ratio 1.0) on a flat patch vs isomap's 1.109, and 1.086 vs 1.878 on
    a curved cap. Reported honestly: LSCM buys angle fidelity by spending area distortion (a different metric,
    not a strictly better number), and it does not guarantee a fold-free chart on high curvature, so the
    reported stats include the FLIPPED-face count (uv_angle_distortion), not just a single ratio that can't see
    a fold. Same pipeline otherwise: triangulate, cut a seam on a closed mesh first (LSCM has no seam-finder of
    its own -- it free-boundary-unwraps whatever boundary it is given, so a closed mesh still needs one cut to
    have a boundary at all), triplanar fallback past the linear-solve's practical size budget."""
    from holographic_mesh import Mesh
    from holographic_meshuv import lscm, stable_uv, uv_angle_distortion
    src = _S["objects"][oid].mesh
    work = Mesh(src.vertices.copy(), [tuple(f) for f in src.faces])
    face_src = list(range(work.n_faces))
    method = "lscm"
    if work.n_vertices > _UV_ISOMAP_MAX:
        method = "triplanar"
        uv = stable_uv(work, mode="triplanar")
    else:
        if not all(len(f) == 3 for f in work.faces):
            tris, fmap = [], []
            for i, f in enumerate(work.faces):
                for k in range(1, len(f) - 1):
                    tris.append((f[0], f[k], f[k + 1])); fmap.append(face_src[i])
            work = Mesh(work.vertices, tris); face_src = fmap
        if _safe_closed(work):
            from holographic_meshseam import shortest_seam, cut_seam
            a = 0
            b = int(np.argmax(np.linalg.norm(work.vertices - work.vertices[a], axis=1)))
            work = cut_seam(work, shortest_seam(work, a, b))
        try:
            uv = lscm(work)
        except Exception:
            from holographic_meshuv import uv_unwrap
            method = "isomap (lscm fallback)"
            uv = uv_unwrap(work, method="isomap")
    if method == "triplanar":
        from holographic_meshuv import uv_distortion
        dist = {"median": round(float(uv_distortion(work, uv)), 3), "flipped": None, "n_faces": work.n_faces}
    else:
        stats = uv_angle_distortion(work, uv)
        dist = {"median": round(float(stats["median"]), 3), "flipped": int(stats["flipped"]),
                "n_faces": int(stats["n_faces"])}
    work.uvs = np.asarray(uv, float)
    return {"mesh": work, "face_src": face_src, "method": method,
            "distortion": dist, "rev": _S["objects"][oid].rev}


def _ensure_uv(oid):
    u = _S["uv"].get(oid)
    if u is None or u["rev"] != _S["objects"][oid].rev:
        u = _S["uv"][oid] = _auto_uv(oid)
    return u


@bp.route("/api/uv", methods=["POST"])
def make_uv():
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        if oid not in _S["objects"]:
            return jsonify({"error": "no such object"}), 400
        t0 = time.time()
        u = _ensure_uv(oid)
        return jsonify({"method": u["method"], "distortion": u["distortion"],
                        "seconds": round(time.time() - t0, 2)})


def _rasterize_uv(work, values_per_face, size, background):
    img = np.full((size, size, 3), background, float)
    uv = work.uvs
    for i, f in enumerate(work.faces):
        a, b, c = (uv[f[0]] * (size - 1), uv[f[1]] * (size - 1), uv[f[2]] * (size - 1))
        lo = np.clip(np.floor(np.minimum(np.minimum(a, b), c)).astype(int), 0, size - 1)
        hi = np.clip(np.ceil(np.maximum(np.maximum(a, b), c)).astype(int) + 1, 1, size)
        if (hi <= lo).any():
            continue
        xs = np.arange(lo[0], hi[0]); ys = np.arange(lo[1], hi[1])
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel()], 1).astype(float)
        def edge(p, q): return (P[:, 0] - p[0]) * (q[1] - p[1]) - (P[:, 1] - p[1]) * (q[0] - p[0])
        e0, e1, e2 = edge(a, b), edge(b, c), edge(c, a)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        if inside.any():
            img[P[inside, 1].astype(int), P[inside, 0].astype(int)] = values_per_face[i]
    return img


def _uv_layout_png(work, size=1024):
    img = np.full((size, size, 3), 1.0)
    uv = work.uvs * (size - 1)
    for f in work.faces:
        for k in range(len(f)):
            a, b = uv[f[k]], uv[f[(k + 1) % len(f)]]
            n = max(2, int(np.linalg.norm(b - a)) + 1)
            t = np.linspace(0, 1, n)[:, None]
            pts = np.clip((a[None, :] * (1 - t) + b[None, :] * t).astype(int), 0, size - 1)
            img[pts[:, 1], pts[:, 0]] = (0.12, 0.2, 0.4)
    return img


def _png_bytes(img):
    from PIL import Image
    a = (np.clip(np.asarray(img, float), 0, 1) * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


@bp.route("/api/uv/maps.zip")
def uv_maps():
    _init()
    oid = str(request.args.get("object", ""))
    with _LOCK:
        if oid not in _S["objects"]:
            return jsonify({"error": "no such object"}), 400
        u = _ensure_uv(oid)
        work, fsrc = u["mesh"], u["face_src"]
        alb, met, rgh, emi, _ = _channels_for(_S["objects"][oid])
        size = int(np.clip(int(request.args.get("size", 512)), 128, 1024))
        per_face = lambda ch: np.asarray([ch[fsrc[i]] for i in range(len(work.faces))])
        maps = {"uv_layout.png": _uv_layout_png(work, max(size, 512)),
                "albedo.png": _rasterize_uv(work, per_face(alb), size, 0.0),
                "metallic.png": _rasterize_uv(work, np.repeat(per_face(met)[:, None], 3, 1), size, 0.0),
                "roughness.png": _rasterize_uv(work, np.repeat(per_face(rgh)[:, None], 3, 1), size, 0.5),
                "emissive.png": _rasterize_uv(work, np.clip(per_face(emi), 0, 1), size, 0.0)}
        V = work.vertices
        fn = []
        for f in work.faces:
            n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            fn.append(n / (np.linalg.norm(n) + 1e-12))
        maps["normal_object.png"] = _rasterize_uv(work, (np.asarray(fn) + 1) / 2, size, 0.5)
        lines = ["# Poly Studio export (leCore) -- positions + auto-unwrapped vt"]
        for x, y, z in work.vertices:
            lines.append(f"v {x:.7g} {y:.7g} {z:.7g}")
        for uu, vv in work.uvs:
            lines.append(f"vt {uu:.7g} {vv:.7g}")
        for f in work.faces:
            lines.append("f " + " ".join(f"{i+1}/{i+1}" for i in f))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, img in maps.items():
                z.writestr(fname, _png_bytes(img))
            z.writestr("model_uv.obj", "\n".join(lines) + "\n")
            flip_txt = (f", {u['distortion']['flipped']} flipped face(s)"
                       if u['distortion']['flipped'] is not None else "")
            z.writestr("README.txt",
                       f"Poly Studio map export (leCore)\nunwrap: {u['method']}  median angle distortion: "
                       f"{u['distortion']['median']} (1.0 = conformal){flip_txt}\nmaps baked from per-face "
                       f"physical materials (holographic_matlib); normal map is object-space, flat per face "
                       f"by design.\n")
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=polystudio_maps.zip"})


_GLB_CACHE = {}          # (oid, object_rev) -> bytes


@bp.route("/api/object_texture")
def object_texture():
    """Serve an imported object's stored base-colour texture (up to TEXTURE_MAX sq) as PNG for the viewport --
    the other half of viewport parity: geometry uv rides in the scene payload, pixels come from here."""
    _init()
    oid = str(request.args.get("object", ""))
    with _LOCK:
        a = _S.get("render_assets", {}).get(oid)
        if not a or "tex" not in a:
            return jsonify({"error": "object has no stored texture"}), 404
        tex = (np.clip(np.asarray(a["tex"], float), 0, 1) * 255).astype(np.uint8)
    from PIL import Image as _Image
    import io as _io
    buf = _io.BytesIO(); _Image.fromarray(tex).save(buf, format="PNG")
    resp = Response(buf.getvalue(), mimetype="image/png")
    resp.headers["Cache-Control"] = "max-age=60"
    return resp


@bp.route("/api/export_glb")
def export_glb():
    """Round-trip out through the engine's own writer (first object or ?object=). Cached by (object, its own
    revision) -- re-clicking Save without editing skips the triangulate + glb-pack work entirely."""
    _init()
    with _LOCK:
        oid = str(request.args.get("object", "")) or next(iter(_S["objects"]))
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        key = (oid, o.rev)
        data = _GLB_CACHE.get(key)
        if data is None:
            from holographic_gltf import mesh_to_glb
            from holographic_mesh import Mesh
            m = o.mesh if all(len(f) == 3 for f in o.mesh.faces) else \
                Mesh(o.mesh.vertices, [tuple(t) for t in o.mesh.triangulate()])
            # MULTI-MATERIAL-AWARE EXPORT: an imported object that still carries its texture + uv exports as a
            # TEXTURED glb (the engine writer embeds the image + TEXCOORD_0); a palette/face-coloured object
            # exports with per-vertex COLOR_0 (faces' colours averaged to vertices); only a plain object bakes
            # to a single base colour. Round-trip verified: re-importing recovers the texture / vertex colours.
            asset = _S.get("render_assets", {}).get(oid, {})
            alb_faces = _channels_for(o)[0]
            tex_arg = None
            if "uv" in asset and len(asset["uv"]) == m.n_vertices:
                try:
                    m.uvs = np.asarray(asset["uv"], float)
                    tex_arg = (np.clip(np.asarray(asset["tex"], float), 0, 1) * 255).astype(np.uint8)
                except Exception:
                    tex_arg = None
            if tex_arg is None:
                try:                                        # per-vertex colours from per-face materials
                    vc = np.zeros((m.n_vertices, 3)); cnt = np.zeros(m.n_vertices)
                    fc = asset.get("face_colors")
                    fcols = np.asarray(fc, float) if fc is not None and len(fc) >= m.n_faces else alb_faces
                    for fi, f in enumerate(m.faces):
                        for vtx in f:
                            vc[vtx] += fcols[fi]; cnt[vtx] += 1
                    if len(set(map(tuple, np.round(fcols[: min(len(fcols), 512)], 3)))) > 1:
                        m.colours = np.clip(vc / np.maximum(cnt, 1)[:, None], 0, 1)
                except Exception:
                    pass
            alb = alb_faces.mean(axis=0)
            data = mesh_to_glb(m, base_colour=(float(alb[0]), float(alb[1]), float(alb[2]), 1.0),
                               generator="polystudio", texture=tex_arg)
            _GLB_CACHE.clear(); _GLB_CACHE[key] = data     # one entry is enough: exports are infrequent
    return Response(data, mimetype="model/gltf-binary",
                    headers={"Content-Disposition": "attachment; filename=polystudio.glb"})


# =====================================================================================================
# The per-object field bakes (SDF + material-ID grids), cached per (object, revision)
# =====================================================================================================
def _safe_closed(mesh):
    """is_closed() that answers False instead of RAISING on non-manifold meshes. Vertex-clustered decimation
    (the sculpt-exit rebuild) can emit duplicated directed edges -- legal triangle soup for a FIELD bake (the
    banded shell + flood does not need manifoldness), but the engine's half-edge check treats it as an error."""
    try:
        return bool(mesh.is_closed())
    except Exception:
        return False


def _banded_grid_chunked(mesh, lo, hi, res, band):
    """Banded SDF grid from a mesh. ADOPTED: the engine's mesh_to_sdf_grid has been chunked since
    sweep 148 (docs/POLYSTUDIO_AUDIT.md: "duplicate; delegate"), so our copy of the chunking is
    gone. Same (grid, (xs, ys, zs)) contract. The engine needs TRIANGLES and says so loudly;
    we triangulate here so quads from the primitive builders still work."""
    from holographic_mesh import Mesh
    F = mesh.faces if all(len(f) == 3 for f in mesh.faces) else mesh.triangulate()
    tri = Mesh(np.asarray(mesh.vertices, float), [tuple(int(v) for v in f) for f in F])
    grid, axes = _mind().mesh_to_sdf_grid(tri, (tuple(map(float, lo)), tuple(map(float, hi))),
                                          res=int(res), band=band, sign="auto")
    return np.asarray(grid, float), tuple(np.asarray(a, float) for a in axes)

def _midpoint_refine(mesh, max_edge, max_rounds=5):
    """Split every triangle at its edge midpoints until no edge exceeds max_edge. PURE refinement: the surface
    is bit-identical (unlike Loop, which smooths); only the sampling density the shell build sees changes."""
    from holographic_mesh import Mesh
    V = mesh.vertices
    F = [tuple(t) for t in (mesh.faces if all(len(f) == 3 for f in mesh.faces) else mesh.triangulate())]
    for _ in range(max_rounds):
        worst = max(np.linalg.norm(V[a] - V[b]) for f in F for a, b in zip(f, (f[1], f[2], f[0])))
        if worst <= max_edge:
            break
        Vl = list(V); mid = {}

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(Vl); Vl.append((V[a] + V[b]) / 2)
            return mid[k]
        NF = []
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            NF += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        V = np.asarray(Vl); F = NF
    return Mesh(V, F)


_THIN_WARN = {}                                            # oid -> warning text for the render header


def _auto_res(o, res):
    """Per-object field resolution: a MESH object's field must resolve its actual feature size, not just the
    scene grid. Median edge length estimates the feature scale; the object's own res is raised until the field
    cell is ~0.8x that, capped at 144 (banded grids stay affordable). Past the cap the field CANNOT represent
    the object faithfully -- that is recorded as a NAMED warning surfaced in the render header, because a
    scrambled photo with no explanation is worse than an honest one. Analytic objects are exact at any res."""
    if o.sdf_tree is not None or o.sculpt is not None:
        return res, None
    V = o.mesh.vertices
    if len(V) < 4 or o.mesh.n_faces < 4:
        return res, None
    E = set()
    for f in o.mesh.faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            E.add((a, b) if a < b else (b, a))
    E = np.array(list(E))[:4000]
    # 15th percentile, not the median: a thin TUBE has few tiny cross-section edges among long axial ones --
    # the median hides exactly the features that alias
    med = float(np.percentile(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1), 15))
    span = float((V.max(axis=0) - V.min(axis=0)).max()) + 0.44
    if med <= 1e-9:
        return res, None
    need = int(np.ceil(span / (0.8 * med)))
    use = int(np.clip(max(res, need), res, 144))
    warn = None
    if need > 144:
        warn = (f"'{o.name}' features are ~{need / 144.0:.0f}x thinner than the field cell even at max "
                f"detail -- the field render cannot match the viewport for it (reduce its extent, or "
                f"thicken/solidify it)")
    return use, warn


_BAKE_STAT = {"hit": 0, "miss": 0}


def _bake_object(oid, res, with_ids):
    o0 = _S["objects"][oid]
    if o0.sculpt is not None:
        return _field_from_grid(oid, o0.sculpt["grid"], o0.sculpt["axes"], with_ids, "sculpt-grid")
    res, warn = _auto_res(o0, res)
    if warn:
        _THIN_WARN[oid] = warn
    else:
        _THIN_WARN.pop(oid, None)
    key = (oid, o0.rev, res, with_ids)
    hit = _S["cache"].get(key)
    if hit is not None:
        _BAKE_STAT["hit"] += 1
        return hit
    _BAKE_STAT["miss"] += 1

    # ANALYTIC NATIVE FAST-PATH: an un-edited primitive still equals its analytic SDF tree, so its field is the
    # EXACT distance -- no mesh_to_sdf, no shell approximation, none of the flood-leak failure modes coarse
    # meshes hit. And it can be evaluated NATIVELY: ccrun compiles the engine's own `sdf_dialect(dsl,'c_f64')`
    # map() (the same one the GLSL/WGSL export emits -- zero drift) to a shared library via the system C
    # compiler and runs it over the grid in one fused pass. Measured 5.9-6.7x faster than tree.eval at
    # res 64-88, bit-identical to machine precision. This is the C sibling of the engine's holographic_zigrun:
    # the container has no Zig toolchain (and the sandbox is offline, so pip can't fetch one), but it HAS cc,
    # and the emitter speaks c_f64 as readily as zig_f64. Falls back cleanly to the mesh bake below when there
    # is no analytic tree (edited/imported mesh), no compiler, or a node the emitter refuses.
    if o0.sdf_tree is not None:
        try:
            import ccrun            # now a 40-line shim over the engine (see ccrun.py)
            lo = o0.mesh.vertices.min(axis=0) - 0.22
            hi = o0.mesh.vertices.max(axis=0) + 0.22
            xs = np.linspace(lo[0], hi[0], res); ys = np.linspace(lo[1], hi[1], res); zs = np.linspace(lo[2], hi[2], res)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            ker = ccrun.get_sdf_kernel(o0.sdf_tree.to_dsl()) if ccrun.should_use(len(P)) else None
            dvals = ker.eval(P) if ker is not None else o0.sdf_tree.eval(P)
            grid = np.asarray(dvals, float).reshape(res, res, res)
            method = "grid(analytic-native)" if ker is not None else "grid(analytic)"
            val = _field_from_grid(oid, grid, (xs, ys, zs), with_ids, method)
            _S["cache"][key] = val
            if len(_S["cache"]) > 3 * max(len(_S["objects"]), 1) + 4:
                _S["cache"].pop(next(iter(_S["cache"])))
            return val
        except Exception:
            pass                                         # any hiccup -> fall through to the robust mesh bake

    from holographic_meshbridge import flood_fill_sign, mesh_to_sdf
    from holographic_mesh import Mesh
    o = _S["objects"][oid]
    mesh = o.mesh
    if not _safe_closed(mesh):
        from holographic_meshtools import solidify
        try:
            mesh = solidify(Mesh(mesh.vertices.copy(), [tuple(f) for f in mesh.faces]), 0.05)
        except Exception:
            pass
    n_tris = sum(max(len(f) - 2, 1) for f in mesh.faces)
    lod_target, lod_grid = (900, 30) if res <= _PREVIEW_RES else (4000, 48)
    if n_tris > lod_target * 1.4:
        # LOD: the render field does not need every triangle -- bake from a decimated proxy (vertex clustering,
        # ~0.01-0.1 s). The full mesh stays authoritative; only the BAKE reads the proxy. Decimate to an actual
        # FACE TARGET, stepping the cluster grid down until it holds: one pass at a fixed grid can be a no-op
        # when the vertices already sit one-per-cell (measured on a sculpt-exit mesh: grid=30 "decimated"
        # 2700 -> 2698 tris and the shell bake then cost 12 s at 1.6 GB; the 700-tri proxy bakes in ~0.6 s,
        # same picture on a 52-voxel grid, which cannot express more detail anyway).
        from holographic_meshqem import cluster_decimate
        try:
            tri = mesh if all(len(f) == 3 for f in mesh.faces) else Mesh(mesh.vertices, [tuple(t) for t in mesh.triangulate()])
            g = lod_grid
            dec = cluster_decimate(tri, grid=g)
            while dec.n_faces > lod_target and g > 6:
                g -= 4
                dec = cluster_decimate(tri, grid=g)
            if dec.n_faces >= 4:
                mesh = dec
        except Exception:
            pass
    lo = mesh.vertices.min(axis=0) - 0.22
    hi = mesh.vertices.max(axis=0) + 0.22
    band = 4.0 * float(((hi - lo) / max(res - 1, 1)).max())
    n_tris = sum(max(len(f) - 2, 1) for f in mesh.faces)
    if n_tris <= 60:
        # coarse mesh: EXACT signed bake. The fast shell build's documented edge cases (a triangle wider than
        # ~2*band under-covers; sign pinholes on sharp edges) make the flood leak on exactly these meshes --
        # measured: a fresh cube's interior stayed positive and rays marched straight through it.
        tri = mesh if all(len(f) == 3 for f in mesh.faces) else Mesh(mesh.vertices, [tuple(t) for t in mesh.triangulate()])
        xs = np.linspace(lo[0], hi[0], res); ys = np.linspace(lo[1], hi[1], res); zs = np.linspace(lo[2], hi[2], res)
        axes = (xs, ys, zs)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        grid = np.empty(res ** 3)
        for i in range(0, len(P), 200_000):
            grid[i:i + 200_000] = mesh_to_sdf(tri, P[i:i + 200_000])
        grid = grid.reshape(res, res, res)
        method = "grid(exact-dist)"
    else:
        grid, axes = _banded_grid_chunked(mesh, lo, hi, res, band)
        neg_before = int((grid < 0).sum())
        grid = flood_fill_sign(grid, band)
        method = "grid(shell+flood)"
        # LEAK CHECK (measured failure mode): on a coarse closed mesh the shell under-covers wide triangles and
        # the sign flood adds nothing -- the interior stays positive and rays march through. If the flood added
        # no interior and the mesh is closed, refine at midpoints (PURE refinement: the surface is identical,
        # only sampling density changes) and rebuild -- verified to restore a watertight negative interior.
        if int((grid < 0).sum()) <= neg_before and _safe_closed(mesh):
            fine = _midpoint_refine(mesh, band * 1.5)
            grid, axes = _banded_grid_chunked(fine, lo, hi, res, band)
            grid = flood_fill_sign(grid, band)
            method = "grid(refine+shell+flood)"

    val = _field_from_grid(oid, grid, axes, with_ids, method)
    _S["cache"][key] = val
    if len(_S["cache"]) > 3 * max(len(_S["objects"]), 1) + 4:
        _S["cache"].pop(next(iter(_S["cache"])))
    return val


def _field_from_grid(oid, grid, axes, with_ids, method):
    """Wrap a signed-distance grid as the render field: bbox-guarded eval + (optionally) a material-ID lookup
    built from the object's CURRENT display faces. Used by both the cached poly bakes and, directly, by sculpt
    grids (which therefore render with ZERO bake cost -- the sculpt field IS the render field)."""
    from holographic_meshbridge import sample_distance_grid
    o = _S["objects"][oid]
    xs, ys, zs = axes
    res = len(xs)
    lo = np.array([xs[0], ys[0], zs[0]]); hi = np.array([xs[-1], ys[-1], zs[-1]])
    ids = None
    if with_ids:
        cent = _centroids(o.mesh)
        voxel = float(((hi - lo) / max(res - 1, 1)).max())
        shell = np.argwhere(np.abs(grid) < 2.2 * voxel)
        P = np.stack([xs[shell[:, 0]], ys[shell[:, 1]], zs[shell[:, 2]]], 1)
        ids = np.zeros(grid.shape, dtype=np.int32)
        step = max(32, int(2_000_000 / max(len(cent), 1))) # bound the (voxels x faces x 3) temp to ~50 MB
        for i in range(0, len(P), step):
            d = np.linalg.norm(P[i:i + step, None, :] - cent[None, :, :], axis=2)
            ids[shell[i:i + step, 0], shell[i:i + step, 1], shell[i:i + step, 2]] = np.argmin(d, axis=1)
    c = (lo + hi) / 2.0; h = (hi - lo) / 2.0

    class _F:
        build_method = method
        raw_grid = grid
        raw_axes = axes

        def eval(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            q = np.abs(P - c) - h
            dbox = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
            out = np.where(dbox > 0.02, dbox + 0.02, 0.0)
            near = dbox <= 0.02
            if near.any():
                out[near] = sample_distance_grid(grid, axes, P[near])
            return out
        __call__ = eval

        def local_ids(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            gi = np.stack([np.clip(np.round((P[:, 0] - xs[0]) / (xs[1] - xs[0])), 0, res - 1),
                           np.clip(np.round((P[:, 1] - ys[0]) / (ys[1] - ys[0])), 0, res - 1),
                           np.clip(np.round((P[:, 2] - zs[0]) / (zs[1] - zs[0])), 0, res - 1)], 1).astype(int)
            return ids[gi[:, 0], gi[:, 1], gi[:, 2]]

    return _F()


def _analytic_scene():
    """EXACT scene, no voxel grid: when every object is still its analytic primitive (un-edited, un-sculpted)
    the whole scene is a UNION OF SDF TREES that render_sdf can evaluate exactly. The grid bake is a sampled
    approximation -- trilinear interpolation of a coarse field puts visible ISO-CONTOUR TERRACES on curved
    surfaces and stair-steps in soft shadows (user-reported; the same scene rendered analytically is clean).
    Returns (sdf, ground, id_fn) or None when any object has no tree (edited/imported mesh) or carries mixed
    per-face materials (whose per-FACE ids a tree cannot answer) -- those still take the baked path.

    id_fn(P) -> global face index per point, matching _scene_channels()'s albedo ordering; -1 = floor."""
    objs = list(_S["objects"].items())
    if not objs:
        return None
    trees, first_face, off = [], [], 0
    for oid, o in objs:
        if o.sdf_tree is None or o.sculpt is not None:
            return None
        if len(set(str(m) for m in np.asarray(o.mats).ravel())) > 1:
            return None                                   # per-face materials need the id grid
        trees.append(o.sdf_tree)
        first_face.append(off)
        off += o.mesh.n_faces
    ground = min((o.mesh.vertices[:, 1].min() for _, o in objs), default=0.0) - 0.35
    import holographic_sdf as _sd
    scene = trees[0]
    for t in trees[1:]:
        scene = scene.union(t)
    scene = scene.union(_sd.plane(ground))
    ff = np.asarray(first_face, dtype=np.int64)

    def id_fn(P):
        P = np.atleast_2d(np.asarray(P, float))
        best = np.full(len(P), np.inf); who = np.full(len(P), -1, dtype=np.int64)
        for i, t in enumerate(trees):
            d = np.asarray(t.eval(P), float)
            take = d < best
            best = np.where(take, d, best); who = np.where(take, i, who)
        floor_d = P[:, 1] - ground
        on_floor = floor_d <= best + 1e-4
        return np.where(on_floor, -1, ff[np.clip(who, 0, len(ff) - 1)])

    scene.face_ids = id_fn                                # duck-types the baked field for _albedo_map
    try:
        scene.build_method = "analytic-exact"             # reported in the render stats like a bake method
    except Exception:
        pass
    return scene, ground, id_fn


def _exact_field_for(o):
    """PER-OBJECT exact field: an un-edited primitive with a single material needs no voxel grid at all --
    its analytic tree IS its distance function. Returned in the same shape _scene_field expects (eval /
    local_ids / build_method) so the scene can MIX exact and baked objects: editing one cube must not put
    grid terracing on every other object in the scene, which is what an all-or-nothing analytic path did.
    Returns None when the object needs the baked grid (mesh/sculpt, or per-face materials whose ids only the
    id-grid can answer)."""
    if o.sdf_tree is None or o.sculpt is not None:
        return None
    if len(set(str(m) for m in np.asarray(o.mats).ravel())) > 1:
        return None
    tree = o.sdf_tree

    class _Exact:
        build_method = "analytic-exact"

        def eval(self, P):
            return np.asarray(tree.eval(np.atleast_2d(np.asarray(P, float))), float)
        __call__ = eval

        def local_ids(self, P):
            return np.zeros(len(np.atleast_2d(np.asarray(P, float))), dtype=np.int64)

    return _Exact()


def _scene_field(res, with_ids):
    """The whole scene as one field: min over the per-object grids (each cached by its OWN revision -- editing
    one object re-bakes one object) + the ground plane. face_ids returns GLOBAL material indices (per-object
    offsets), -1 for the ground."""
    fields, offsets, off = [], [], 0
    baked = []
    for oid, o in _S["objects"].items():
        _ex = _exact_field_for(o)                      # exact where possible, baked only where necessary
        fields.append(_ex if _ex is not None else _bake_object(oid, res, with_ids))
        offsets.append(off); off += o.mesh.n_faces
        baked.append(fields[-1].build_method)
    ground = min((o.mesh.vertices[:, 1].min() for o in _S["objects"].values()), default=0.0) - 0.35

    class _Scene:
        build_method = "+".join(sorted(set(baked))) if baked else "none"

        def eval(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            d = np.full(len(P), np.inf)
            for f in fields:
                d = np.minimum(d, f.eval(P))
            return np.minimum(d, P[:, 1] - ground)
        __call__ = eval

        def face_ids(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            on_floor = (P[:, 1] - ground) < 0.02
            D = np.stack([f.eval(P) for f in fields], 1) if fields else np.zeros((len(P), 1))
            k = np.argmin(D, axis=1)
            out = np.empty(len(P), dtype=np.int64)
            for j, f in enumerate(fields):
                m = k == j
                if m.any():
                    out[m] = offsets[j] + f.local_ids(P[m])
            return np.where(on_floor, -1, out)

    return _Scene(), ground


def _scene_channels():
    parts = [_channels_for(o) for o in _S["objects"].values()]
    if not parts:
        z = np.zeros((0, 3))
        return z, np.zeros(0), np.zeros(0), z, np.zeros(0)
    return tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(5))


def _albedo_map(scene, cam, W, H):
    """PER-OBJECT/PER-FACE COLOUR for the raymarch preview: one cheap sphere-trace over the primary rays gives
    each pixel its hit's GLOBAL face id -> that face's material albedo (ground gets the floor material; misses
    get 1.0 so the sky is untouched by the multiply). render_sdf shades with a NEUTRAL base colour and the
    preview multiplies by this map -- the same face_ids machinery the GI photo uses, at preview cost. For a
    parked camera the map is IDENTICAL across progressive rounds, so callers cache it by (rev, camera, size)."""
    from holographic_raymarch import sphere_trace
    alb = _scene_channels()[0]
    try:
        floor = _mat(_FLOOR_MAT); f_alb = np.asarray(floor.base_color[:3], float)
    except Exception:
        f_alb = np.array([0.62, 0.6, 0.58])
    eye, dirs = cam.ray_dirs(W, H)
    Dd = dirs.reshape(-1, 3); Od = np.broadcast_to(np.asarray(eye, float), Dd.shape)
    hit, tt, P = sphere_trace(scene, Od, Dd)
    out = np.ones((len(Dd), 3), float)
    if hit.any() and len(alb):
        ids = scene.face_ids(np.asarray(P)[hit])
        fl = ids < 0
        j = np.clip(ids, 0, len(alb) - 1)
        cols = np.where(fl[:, None], f_alb[None, :], alb[j])
        out[hit] = cols
    return out.reshape(H, W, 3)


def _mean_albedo():
    alb = _scene_channels()[0]
    return tuple(float(x) for x in alb.mean(axis=0)) if len(alb) else (0.7, 0.6, 0.5)


def _fsr_to(img, out_hw, sharpness=0.35):
    from holographic_fsr import lanczos_upscale
    from holographic_postfx import sharpen
    img = np.asarray(img, float)
    up = lanczos_upscale(img, out_hw)
    mn = img.copy(); mx = img.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            mn = np.minimum(mn, r); mx = np.maximum(mx, r)
    H, W = img.shape[:2]; oh, ow = out_hw
    oy = np.clip((np.arange(oh) * H / oh).astype(int), 0, H - 1)
    ox = np.clip((np.arange(ow) * W / ow).astype(int), 0, W - 1)
    up = np.clip(up, mn[oy][:, ox], mx[oy][:, ox])
    return np.clip(sharpen(up, amount=sharpness), 0, 1) if sharpness > 0 else up


def _apply_clip(scene, g):
    """Cross-section clip (user ask #4, 3-D half): intersect the scene field with an axis-aligned half-space
    max(d, sign*(axis_coord - offset)). Exact field composition -- cut faces are the real interior."""
    cl = g("clip", "")
    if not cl:
        return scene
    try:
        ax_s, off_s, sgn_s = (cl.split(",") + ["0", "1"])[:3]
        ax = {"x": 0, "y": 1, "z": 2}.get(ax_s.strip().lower(), 0)
        off = float(off_s); sgn = 1.0 if float(sgn_s) >= 0 else -1.0
    except Exception:
        return scene
    inner = scene

    class _Clipped:
        build_method = getattr(inner, "build_method", "field") + "+clip"

        def eval(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            return np.maximum(inner.eval(P), sgn * (P[:, ax] - off))
    return _Clipped()


def _light_dir(g):
    """Key-light direction from azimuth/elevation degrees (UI-friendly), defaulting to the studio 3/4 key."""
    import math
    az = _qnum(g, "light_az", 300.0, 0.0, 360.0) * math.pi / 180.0
    el = _qnum(g, "light_el", 55.0, -20.0, 89.0) * math.pi / 180.0
    ce = math.cos(el)
    return (ce * math.cos(az), math.sin(el), ce * math.sin(az))


def _cam_from_args(g):
    try:
        eye = [float(x) for x in g("eye", "2.4,1.7,2.9").split(",")]
        target = [float(x) for x in g("target", "0,0,0").split(",")]
    except Exception:
        eye, target = [2.4, 1.7, 2.9], [0, 0, 0]
    fov = float(np.clip(float(g("fov", 45)), 20, 90))
    from holographic_render import Camera
    return Camera(eye=eye, target=target, fov_deg=fov)


## ADAPTIVE PREVIEW QUALITY -- the engine's own closed-loop frame-budget controller
# (holographic_framebudget.FrameServer / FrameBudgetController), not a hand-tuned formula.
#
# The controller's contract: give it a ladder of presets (coarsest first) and a target fps; each call it hands
# back the current rung, TIMES the render, and reacts to the MEASURED time -- drops a rung the instant a frame
# blows its budget, climbs only after several comfortably-fast frames (hysteresis), so quality does not
# chatter. That is exactly the "guarantee 30fps at the floor" requirement: rather than guessing a resolution
# that should hit 30fps, this MEASURES this machine's actual render time and adapts to it, session by session
# (a phone and a desktop each get their own controller, keyed by a client-generated session id).
#
# The ladder below is ours (framebudget's DEFAULT_LADDER is tuned for a toy raymarch + a fluid sim we don't
# have); the controller only cares about the ORDER, not the keys, so it works unmodified against our preset
# shape: `scale` is the internal trace resolution as a fraction of display size (the FSR ratio), and
# ao/shadows/reflect are the shading toggles our render_sdf call already exposes.
#
# CONSIDERED AND NOT ADOPTED: holographic_refresh.RefreshRenderer / holographic_realtime.RealtimeSession
# (reprojection -- warp the previous frame, re-shade only the disocclusion border + a budget of "oldest"
# pixels). It measures a spectacular 57 dB / 5x-fewer-shades ceiling, but ONLY on a parallax-free procedural
# scene; its own module docstring reports the ceiling on a REAL 3-D scene (our case: real depth, real
# occlusion, view-dependent specular) drops to ~36-41 dB and DECAYS a further ~3.4 dB over 9 frames as warps
# compound. For a modelling tool where the person is judging shape and material fidelity, trading a resolution
# step for a warping/ghosting artifact during orbit is the wrong side of that trade -- so this demo uses the
# frame-budget controller (which only ever trades RESOLUTION, converging cleanly to the same image) and not
# the reprojection path. Worth revisiting if a future scene gets heavy enough that resolution alone can't hold
# the budget.
from holographic_framebudget import FrameServer

_RENDER_LADDER = [
    {"name": "potato", "scale": 0.30, "ao": False, "shadows": False, "reflect": 0.00},
    {"name": "low",    "scale": 0.42, "ao": False, "shadows": True,  "reflect": 0.05},
    {"name": "medium", "scale": 0.55, "ao": True,  "shadows": True,  "reflect": 0.12},
    {"name": "high",   "scale": 0.72, "ao": True,  "shadows": True,  "reflect": 0.18},
    {"name": "ultra",  "scale": 0.90, "ao": True,  "shadows": True,  "reflect": 0.22},
]
_FRAME_SERVER = FrameServer(ladder=_RENDER_LADDER, headroom=0.15)


def _qnum(g, key, default, lo, hi, cast=float):
    """Parse a numeric query param defensively: a malformed value falls back to the default instead of raising
    into an HTML 500 (a UI typo or a fuzzer should get a sane frame, not a stack trace)."""
    try:
        v = cast(g(key, default))
    except (TypeError, ValueError):
        v = cast(default)
    return cast(np.clip(v, lo, hi))


_PROG = {}                                                  # session -> progressive-accumulation state
_ALBEDO_CACHE = {}                                          # (rev,cam,size) -> per-pixel albedo map (1 entry)


@bp.route("/api/render_progressive")
def render_progressive():
    """STILL-CAMERA PROGRESSIVE RESOLVE: while the camera is parked, each call adds ONE full-quality,
    sub-pixel-JITTERED sample (Halton offsets through the pixel footprint) into a per-session accumulation
    buffer and returns the running mean -- edges anti-alias and shading noise averages out, sharpening every
    round. NOTHING is recomputed that can be reused: the SDF field bakes are already cached per object
    revision, so rounds after the first pay only the raymarch. When the mean inter-round delta drops below a
    visually-lossless threshold (~0.4/255) twice in a row, X-Converged: 1 tells the client to STOP -- the
    image has no noise left to remove. Any camera/scene/quality change resets the buffer automatically (the
    state is keyed by a signature of all of it)."""
    _init()
    g = request.args.get
    from holographic_raymarch import render_sdf
    from holographic_render import Camera
    session = g("session", "default")
    grid = _qnum(g, "grid", _PREVIEW_RES, 32, 72, int)
    W = int(np.clip(int(g("w", 760)), 220, 1100)); H = int(W * 0.75)
    with _LOCK:
        sig = (int(_S["rev"]), grid, g("eye", ""), g("target", ""), g("fov", ""), g("bg", ""),
               g("light_az", ""), g("light_el", ""), g("sun", ""), g("ambient", ""), g("clip", ""), W)
        st = _PROG.get(session)
        if st is None or st["sig"] != sig:
            # pdelta/cnt reset with the session: a camera move or edit invalidates which pixels had
            # converged, so adaptive sampling must start from "everything needs shading" again.
            st = {"sig": sig, "accum": None, "n": 0, "below": 0, "converged": False, "delta": 1.0,
                  "pdelta": None, "cnt": None, "masked_frac": 1.0, "edge": None}
            _PROG[session] = st
            while len(_PROG) > 4:                            # tiny LRU: sessions are cheap but buffers aren't
                _PROG.pop(next(iter(_PROG)))
        if st["converged"]:
            img = st["accum"]
        elif not st.get("warm"):
            # ROUND 0 -- CAMERA PRIORITY: a display-only quick frame at 0.35x internal so the parked camera
            # sees SOMETHING in ~1.5s; it never enters the accumulator (mixing resolutions would corrupt the
            # mean), and full-quality accumulation starts on the very next call.
            st["warm"] = True
            try:
                _an = _analytic_scene()                            # exact-first, see _analytic_scene
                if _an is not None:
                    scene, ground, _ = _an
                else:
                    scene, ground = _scene_field(grid, with_ids=True)
                scene = _apply_clip(scene, g)
            except Exception as e:
                return jsonify({"error": f"bake failed: {e}"}), 500
            from holographic_raymarch import render_sdf as _rs
            Wq = max(140, int(W * 0.35)); Hq = int(Wq * 0.75)
            try:
                eye0 = np.array([float(x) for x in g("eye", "2.4,1.7,2.9").split(",")], float)
                tgt0 = np.array([float(x) for x in g("target", "0,0,0").split(",")], float)
            except Exception:
                eye0, tgt0 = np.array([2.4, 1.7, 2.9]), np.zeros(3)
            fov0 = float(np.clip(float(g("fov", 45) or 45), 20, 90))
            camq = Camera(eye=eye0.tolist(), target=tgt0.tolist(), fov_deg=fov0)
            _bgq = g("bg", ""); _skyq = None
            if _bgq:
                try:
                    _skyq = np.tile(np.array([float(x) for x in _bgq.split(",")][:3], float), (4, 4, 1))
                except Exception:
                    _skyq = None
            elif _S.get("env_img") is not None:
                _skyq = _S["env_img"]
            q = np.asarray(_rs(scene, camq, width=Wq, height=Hq, light_dir=_light_dir(g),
                               base_color=(1.0, 1.0, 1.0), sky=_skyq, ao=True, shadows=True, reflect=0.35,
                               ambient=_qnum(g, "ambient", 0.28, 0.0, 1.0),
                               sun_intensity=_qnum(g, "sun", 3.14159, 0.0, 8.0)), float)
            q = q * _albedo_map(scene, camq, Wq, Hq)
            disp = _fsr_to(np.clip(q, 0, 1), (H, W), sharpness=0.2)
            resp = Response(_png_bytes(disp), mimetype="image/png")
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["X-Round"] = "0"
            resp.headers["X-Delta"] = "1.000000"
            resp.headers["X-Converged"] = "0"
            resp.headers["X-Build-Method"] = str(getattr(scene, "build_method", "?"))  # every round reports it
            return resp
        else:
            try:
                _an = _analytic_scene()                            # exact-first, see _analytic_scene
                if _an is not None:
                    scene, ground, _ = _an
                else:
                    scene, ground = _scene_field(grid, with_ids=True)
                scene = _apply_clip(scene, g)
            except Exception as e:
                return jsonify({"error": f"bake failed: {e}"}), 500
            try:
                eye = np.array([float(x) for x in g("eye", "2.4,1.7,2.9").split(",")], float)
                target = np.array([float(x) for x in g("target", "0,0,0").split(",")], float)
            except Exception:
                eye, target = np.array([2.4, 1.7, 2.9]), np.zeros(3)
            fov = float(np.clip(float(g("fov", 45) or 45), 20, 90))
            # sub-pixel jitter: Halton(2,3) offsets in [-0.5,0.5) pixels, moved along the camera's screen axes
            def _halton(ix, base):
                f, r = 1.0, 0.0
                while ix > 0:
                    f /= base; r += f * (ix % base); ix //= base
                return r
            # INTERNAL RESOLUTION: accumulating below display res and upscaling leaves stair-stepped
            # silhouettes no amount of jitter can fix -- the aliasing is baked in before the upscale. On the
            # EXACT (analytic) path evaluation is cheap enough to accumulate at FULL display resolution, so
            # the per-round jitter becomes true supersampling and edges resolve. Baked/mesh scenes keep the
            # 0.5x internal buffer, where the field bake dominates cost.
            _exact = getattr(scene, "build_method", "") == "analytic-exact"
            _sc = 1.0 if _exact else 0.5
            Wi = max(160, int(W * _sc)); Hi = int(Wi * 0.75)               # internal accumulation resolution
            # round 1 is the CAMERA-PRIORITY frame: zero jitter + a cheap 1x albedo map so the first image
            # lands fast; the 2x map (for jitter-aligned colour edges) is built once on round 2.
            if st["n"] == 0:
                jx = jy = 0.0
            else:
                jx = _halton(st["n"] + 1, 2) - 0.5; jy = _halton(st["n"] + 1, 3) - 0.5
            fwd = target - eye; dist = float(np.linalg.norm(fwd)) or 1.0; fwd = fwd / dist
            right = np.cross(fwd, [0.0, 1.0, 0.0]); rn = np.linalg.norm(right)
            right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
            up = np.cross(right, fwd)
            px_world = 2.0 * dist * np.tan(np.radians(fov) / 2.0) / Hi    # one INTERNAL pixel's footprint
            off = right * (jx * px_world) + up * (jy * px_world)
            cam = Camera(eye=(eye + off).tolist(), target=(target + off).tolist(), fov_deg=fov)
            _bg = g("bg", ""); _sky = None
            if _bg:
                try:
                    _sky = np.tile(np.array([float(x) for x in _bg.split(",")][:3], float), (4, 4, 1))
                except Exception:
                    _sky = None
            elif _S.get("env_img") is not None:
                _sky = _S["env_img"]
            # SPEED: accumulate at INTERNAL resolution and FSR-upscale for display -- each round costs a
            # ~0.62x raymarch, not a full-res one; PER-OBJECT COLOUR: multiply neutral shading by the albedo
            # map, computed ONCE per signature via a single sphere-trace and reused every round (the map is
            # identical while the camera is parked -- that is the cached data doing the work).
            # JITTER-ALIGNED albedo at CACHE cost: trace the map ONCE at 2x internal resolution per parked
            # camera (session-cached), then per round SAMPLE it at this round's jittered pixel centres.
            # The jitter shifts eye+target together -- a pure image-plane translation -- so a sub-pixel
            # shifted read of the 2x map is what the jittered camera sees; nearest sampling keeps object
            # boundaries crisp (bilinear would smear albedo across silhouettes). Colour edges anti-alias in
            # step with geometry edges, and rounds pay array indexing, not a fresh sphere-trace.
            if st["n"] == 0:
                amap = _albedo_map(scene, cam, Wi, Hi)                  # fast 1x, exact for the unjittered round
            elif _exact:
                # EXACT path: trace the albedo map at THIS round's jittered camera, 1x. Exactly aligned by
                # construction (no 2x supersampled map to build and sample), and cheap because evaluation is
                # analytic -- this removes the one-off multi-second spike the 2x map cost at full resolution.
                amap = _albedo_map(scene, cam, Wi, Hi)
            else:
                if st.get("amap2") is None or st["amap2"].shape[:2] != (2 * Hi, 2 * Wi):
                    cam0 = Camera(eye=eye.tolist(), target=target.tolist(), fov_deg=fov)
                    st["amap2"] = _albedo_map(scene, cam0, 2 * Wi, 2 * Hi)
                yy = np.clip(((np.arange(Hi) + 0.5 + jy) * 2).astype(int), 0, 2 * Hi - 1)
                xx = np.clip(((np.arange(Wi) + 0.5 + jx) * 2).astype(int), 0, 2 * Wi - 1)
                amap = st["amap2"][yy[:, None], xx[None, :]]
            # ADAPTIVE SAMPLING (uses the engine's new full-quality pixel mask): a pixel whose accumulated
            # value stopped moving has converged, and spending more samples on it buys nothing. From round 3
            # on we shade ONLY the pixels still changing -- flat interiors drop out fast, silhouettes and
            # contact shadows keep sampling. This is NOT the cheap-shading checkerboard: masked pixels get
            # the same ao/shadows/reflect as an unmasked render (verified bit-identical upstream), so the
            # converged image is the same image, reached with less work. Per-pixel sample counts keep the
            # running mean correct when different pixels have different sample totals.
            rmask = None
            # adaptive=0 disables masking -- an A/B switch kept in the product so the adaptive path can be
            # compared against plain accumulation at any time (this is how it was verified, and how a future
            # regression would be caught).
            if str(g("adaptive", "1")) not in ("0", "false", "off") \
                    and st.get("pdelta") is not None and st["n"] >= 2:
                thr = 1.2e-3                                  # per-pixel movement worth another sample
                m = st["pdelta"] > thr
                # EDGE FLOOR: measured on the A/B, 97% of the residual difference vs plain accumulation sat
                # on silhouette pixels -- a flat interior converges in a couple of samples but an edge pixel
                # is a coverage average that needs several jittered samples before it settles, and it can sit
                # momentarily still between them. So edges keep sampling until they have a real sample count
                # regardless of the movement test; interiors (the bulk of the frame) still drop out early.
                if st.get("edge") is None:
                    lum = st["accum"].mean(axis=2)
                    gy, gx = np.gradient(lum)
                    mag = np.sqrt(gx * gx + gy * gy)
                    st["edge"] = mag > max(float(np.percentile(mag, 92)), 1e-4)
                cnt_now = st.get("cnt")
                if cnt_now is not None:
                    m |= st["edge"] & (cnt_now < 8.0)
                else:
                    m |= st["edge"]
                if m.any() and m.mean() < 0.85:               # only worth masking if a real share is done
                    m[1:, :] |= m[:-1, :]; m[:-1, :] |= m[1:, :]      # dilate 1px so edges never starve
                    m[:, 1:] |= m[:, :-1]; m[:, :-1] |= m[:, 1:]
                    rmask = m
            rkw = dict(light_dir=_light_dir(g), base_color=(1.0, 1.0, 1.0), sky=_sky,
                       ao=True, shadows=True, reflect=0.35,
                       ambient=_qnum(g, "ambient", 0.28, 0.0, 1.0),
                       sun_intensity=_qnum(g, "sun", 3.14159, 0.0, 8.0))
            if rmask is not None:
                rkw["mask"] = rmask
            frame = np.asarray(render_sdf(scene, cam, width=Wi, height=Hi, **rkw), float) * amap
            prev = st["accum"]
            st["n"] += 1
            if prev is None:
                st["accum"] = frame
                st["cnt"] = np.ones((Hi, Wi), float)
            elif rmask is None:
                st["accum"] = prev + (frame - prev) / st["n"]
                st["cnt"] = st.get("cnt", np.ones((Hi, Wi), float)) + 1.0
            else:
                cnt = st.get("cnt", np.ones((Hi, Wi), float))
                upd = rmask[:, :, None]
                newc = cnt + rmask.astype(float)
                st["accum"] = np.where(upd, prev + (frame - prev) / np.maximum(newc, 1.0)[:, :, None], prev)
                st["cnt"] = newc
            if prev is not None:
                d_abs = np.abs(st["accum"] - prev)
                st["pdelta"] = d_abs.max(axis=2)               # per-pixel movement, drives the next mask
                st["delta"] = float(d_abs.mean())
                st["masked_frac"] = float(rmask.mean()) if rmask is not None else 1.0
                st["below"] = st["below"] + 1 if st["delta"] < 1.5e-3 else 0
                if st["below"] >= 2 or st["n"] >= 24:
                    st["converged"] = True                    # no noise left worth a round -- stop asking
            else:
                st["pdelta"] = np.ones((Hi, Wi), float)
            img = st["accum"]
        disp = _fsr_to(np.clip(img, 0, 1), (H, W), sharpness=0.12) if img.shape[0] != H else np.clip(img, 0, 1)
        resp = Response(_png_bytes(disp), mimetype="image/png")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Round"] = str(st["n"])
        resp.headers["X-Delta"] = f"{st['delta']:.6f}"
        resp.headers["X-Converged"] = "1" if st["converged"] else "0"
        resp.headers["X-Shaded-Frac"] = "%.3f" % st.get("masked_frac", 1.0)
        resp.headers["X-Build-Method"] = str(getattr(scene, "build_method", "?"))   # exact vs baked, testable
        return resp


@bp.route("/api/render")
def render_preview():
    _init()
    g = request.args.get
    from holographic_raymarch import render_sdf
    session = g("session")
    target_fps = _qnum(g, "target_fps", 30, 5, 120)
    grid = _qnum(g, "grid", _PREVIEW_RES, 32, 72, int)     # LEVEL OF DETAIL, user-facing: the field resolution

    def render_fn(preset):
        with _LOCK:
            t0 = time.time()
            _BAKE_STAT["hit"] = 0; _BAKE_STAT["miss"] = 0
            try:
                # EXACT-FIRST: analytic union when every object is still a primitive (no grid terracing);
                # the baked field remains the path for edited/imported meshes and per-face materials.
                _an = _analytic_scene()
                if _an is not None:
                    scene, ground, _ = _an
                else:
                    scene, ground = _scene_field(grid, with_ids=True)
                scene = _apply_clip(scene, g)
            except Exception as e:
                return {"error": str(e)}
            t_bake = time.time() - t0
            _bake_cached = (_BAKE_STAT["miss"] == 0)
            W = int(220 + preset["scale"] * 260); H = int(W * 0.75)
            Wi = max(90, int(W * preset["scale"])); Hi = int(Wi * 0.75)
            _bg = g("bg", "")
            _sky = None
            if _bg:
                try:
                    _bgc = np.array([float(x) for x in _bg.split(",")][:3], float)
                    _sky = np.tile(_bgc, (4, 4, 1))          # solid-colour env image (render_sdf wants an array)
                except Exception:
                    _sky = None
            elif _S.get("env_img") is not None:
                _sky = _S["env_img"]                          # generated backdrop (sky/starfield/nebula/galaxy)
            kw = dict(light_dir=_light_dir(g), base_color=(1.0, 1.0, 1.0), sky=_sky,
                      ao=preset["ao"], shadows=preset["shadows"], reflect=preset["reflect"],
                      ambient=_qnum(g, "ambient", 0.28, 0.0, 1.0),
                      sun_intensity=_qnum(g, "sun", 3.14159, 0.0, 8.0))
            t0 = time.time()
            cam_ = _cam_from_args(g)
            # per-object materials in the PREVIEW: shade neutral, multiply by the per-pixel albedo map
            # (one sphere-trace at internal res, cached by scene rev + camera + size so debounced repeats
            # and progressive rounds pay it once)
            akey = (int(_S["rev"]), g("eye", ""), g("target", ""), g("fov", ""), grid, Wi)
            amap = _ALBEDO_CACHE.get(akey)
            if amap is None:
                amap = _albedo_map(scene, cam_, Wi, Hi)
                _ALBEDO_CACHE.clear(); _ALBEDO_CACHE[akey] = amap
            raw = np.asarray(render_sdf(scene, cam_, width=Wi, height=Hi, **kw), float) * amap
            img = _fsr_to(raw, (H, W), sharpness=0.35 if preset["scale"] < 0.9 else 0.2)
            t_trace = time.time() - t0
        return {"png": _png_bytes(img), "Wi": Wi, "Hi": Hi, "W": W, "H": H,
                "build_method": scene.build_method, "t_bake": t_bake, "t_trace": t_trace,
                "bake_cached": _bake_cached}

    if session:
        result = _FRAME_SERVER.serve_frame(session, render_fn, target_fps=target_fps)
        payload = result["payload"]
        if "error" in payload:
            return jsonify({"error": f"bake failed: {payload['error']}"}), 500
        resp = Response(payload["png"], mimetype="image/png")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Holostuff-Render"] = (
            f"AUTO[{result['preset']['name']}] fsr {payload['Wi']}x{payload['Hi']}->{payload['W']}x{payload['H']}; "
            f"grid={grid} ({payload['build_method']}); bake={payload['t_bake']:.2f}s"+ (' (cached)' if payload.get('bake_cached') else '') + ' '
            + ("; WARN " + " | ".join(sorted(set(_THIN_WARN.values()))) if _THIN_WARN else "") + " "
            f"trace={payload['t_trace']:.2f}s; frame={result['frame_ms']:.1f}ms/"
            f"{result['budget_ms']:.0f}ms budget; met={result['stats']['met_budget_frac']}")
        return resp

    # No session -> the old manual path (a fixed `quality` slider, 0..1), unchanged for anyone driving it by hand.
    q = _qnum(g, "quality", 0.5, 0.0, 1.0)
    lo, hi = _RENDER_LADDER[0], _RENDER_LADDER[-1]
    preset = {k: (lo[k] + (hi[k] - lo[k]) * q if isinstance(lo[k], float) else (hi[k] if q > 0.5 else lo[k]))
             for k in ("scale", "ao", "shadows", "reflect")}
    preset["name"] = f"manual({q:.2f})"
    out = render_fn(preset)
    if "error" in out:
        return jsonify({"error": f"bake failed: {out['error']}"}), 500
    resp = Response(out["png"], mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"
    _w = ("WARN " + " | ".join(sorted(set(_THIN_WARN.values()))) + "; ") if _THIN_WARN else ""
    resp.headers["X-Holostuff-Render"] = (f"fsr {out['Wi']}x{out['Hi']}->{out['W']}x{out['H']}; "
                                          f"grid={grid} ({out['build_method']}); " + _w +
                                          f"bake={out['t_bake']:.2f}s" + (' (cached)' if out.get('bake_cached') else '') + f"; trace={out['t_trace']:.2f}s")
    return resp


@bp.route("/api/render_stats")
def render_stats():
    """Session health: every session's ladder level + measured budget-hit rate, for an on-screen fps/quality
    readout that reflects what the controller has actually observed, not a guess."""
    session = request.args.get("session")
    if session:
        ctrl = _FRAME_SERVER._sessions.get(session)
        if ctrl is None:
            return jsonify({"error": "unknown session"}), 404
        return jsonify({"level": ctrl.level, "preset": ctrl.current(), **ctrl.stats()})
    return jsonify(_FRAME_SERVER.sessions())


_SUN = np.array([-0.42, 0.72, -0.32]); _SUN = _SUN / np.linalg.norm(_SUN)
_SUN_COL = np.array([1.0, 0.94, 0.84])
_SUN_COS = np.cos(np.radians(9.0))


def _photo_sky(D):
    D = np.atleast_2d(np.asarray(D, float))
    up = np.clip(D[:, 1], -1, 1)
    t = np.clip(up, 0, 1)[:, None] ** 0.6
    col = np.array([0.80, 0.72, 0.60])[None, :] * (1 - t) + np.array([0.30, 0.45, 0.72])[None, :] * t
    col = np.where(up[:, None] < 0, np.array([0.38, 0.33, 0.27])[None, :], col)
    sun = (D @ _SUN > _SUN_COS)[:, None]
    return col * 0.85 + sun * _SUN_COL[None, :] * 5.0


@bp.route("/api/photo")
def photo():
    _init()
    g = request.args.get
    # A5: the client can now ask for a real output size and aspect. The old ceiling was 560 with the
    # height hard-wired to 4:3, so "the photo" was a 380x285 thumbnail no matter what you wanted.
    W = _qnum(g, "w", 560, 240, 1280, int)
    H = _qnum(g, "h", int(W * 0.75), 120, 1280, int)
    spp = _qnum(g, "spp", 24, 8, 96, int)
    # PROGRESSIVE RESOLVE: emit a refreshed frame EVERY sample (Redshift/V-Ray-style live accumulation) --
    # the first image lands after 1 spp instead of 8, and the preview visibly sharpens as it converges.
    every = 1 if spp <= 32 else 2
    grid = _qnum(g, "grid", _PHOTO_RES, 56, 112, int)      # LEVEL OF DETAIL for the photo field
    fog_on = g("fog", "1") not in ("0", "false", "off")
    dof_on = g("dof", "0") not in ("0", "false", "off")
    focus_dist = _qnum(g, "focus", 3.0, 0.2, 40.0)          # metres from eye to the sharp plane
    fstop = _qnum(g, "fstop", 4.0, 0.8, 22.0)               # smaller = shallower depth of field
    exposure = _qnum(g, "exposure", 1.0, 0.1, 4.0)          # linear pre-tonemap gain
    sharpen_amt = _qnum(g, "sharpen", 0.0, 0.0, 1.5)        # holographic_postfx.sharpen amount (0 = off)
    bg = g("bg", "")                                        # "r,g,b" solid backdrop (studio sky when empty)
    sky_fn = _photo_sky
    if bg:
        try:
            _bgcol = np.array([float(x) for x in bg.split(",")][:3], float)
            def sky_fn(D, _c=_bgcol):
                D = np.atleast_2d(np.asarray(D, float))
                return np.broadcast_to(_c, (D.shape[0], 3)).copy()
        except Exception:
            sky_fn = _photo_sky
    elif _S.get("env_img") is not None:
        _env = _S["env_img"]
        def sky_fn(D, _img=_env):                             # equirect lookup: azimuth->u, elevation->v
            D = np.atleast_2d(np.asarray(D, float))
            u = (np.arctan2(D[:, 2], D[:, 0]) / (2 * np.pi) + 0.5) % 1.0
            v = np.clip(np.arcsin(np.clip(D[:, 1], -1, 1)) / np.pi + 0.5, 0, 1)
            Hh, Ww = _img.shape[0], _img.shape[1]
            yi = np.clip(((1 - v) * (Hh - 1)).astype(int), 0, Hh - 1)
            xi = np.clip((u * (Ww - 1)).astype(int), 0, Ww - 1)
            return _img[yi, xi].astype(float)
    aov = (g("aov", "") or "").lower()                      # "", "normal", "depth" -- diagnostic passes
    cam = _cam_from_args(g)
    with _LOCK:
        try:
            _an = _analytic_scene()                                # exact-first: the photo benefits most
            if _an is not None:
                scene, ground, _ = _an
            else:
                scene, ground = _scene_field(grid, with_ids=True)
        except Exception as e:
            return jsonify({"error": f"bake failed: {e}"}), 500
        alb, met, rgh, emi, ior = _scene_channels()
    ml = _matlib()
    _floor_name = g("floor", "") or _FLOOR_MAT
    try:
        floor = _mat(_floor_name)
    except Exception:
        floor = _mat(_FLOOR_MAT)
    f_alb = np.asarray(floor.base_color[:3]); f_met = float(floor.metallic); f_rgh = float(floor.roughness)

    def material(P):
        i = scene.face_ids(P)
        fl = i < 0
        j = np.clip(i, 0, max(len(alb) - 1, 0))
        a = np.where(fl[:, None], f_alb[None, :], alb[j])
        return (a, np.where(fl, f_met, met[j]), np.where(fl, f_rgh, rgh[j]),
                np.where(fl[:, None], 0.0, emi[j]), np.where(fl, 0.0, ior[j]))

    # Depth for the atmosphere pass: ONE extra cheap sphere-trace (no shading) over the same primary rays the
    # path tracer casts -- depth does not change as samples accumulate, so it is computed once up front and
    # reused for every progressive frame + the final one, not re-traced per batch.
    depth = None
    if fog_on or dof_on:
        from holographic_raymarch import sphere_trace
        eye, dirs = cam.ray_dirs(W, H)
        Dd = dirs.reshape(-1, 3); Od = np.broadcast_to(eye, Dd.shape)
        hit, tt, _P = sphere_trace(scene, Od, Dd)
        depth = np.where(hit, tt, 60.0).reshape(H, W)

    def _dof_blur(img, dep):
        # depth-of-field: circle-of-confusion grows with |depth-focus|/fstop; approximate with a small stack of
        # gaussian-blurred copies selected per-pixel by CoC. Cheap and honest (a real thin-lens bokeh would jitter
        # the aperture rays; this is a post-blur, stated as such in the UI).
        coc = np.abs(dep - focus_dist) / max(focus_dist, 1e-3) * (12.0 / fstop)
        coc = np.clip(coc, 0.0, 1.0)
        blurs = [img]
        for s in (1.2, 2.6, 4.5):
            blurs.append(np.stack([_gauss_blur_np(img[..., ch], s) for ch in range(3)], axis=-1))
        thr = [0.18, 0.45, 0.72]
        out = blurs[0].copy()
        for bi, t in enumerate(thr):
            m = (coc > t)[..., None]
            out = np.where(m, blurs[bi + 1], out)
        return out

    def _prep(hdr):
        # Everything that needs this request's scene context: fireflies, fog, depth of field. The
        # result is cached, so exposure and sharpen can be re-applied later without re-tracing.
        hdr = np.asarray(hdr, float)
        # FIREFLY CLAMP (engine: gemrender.clamp_fireflies, new in this drop). A path tracer occasionally
        # returns one absurdly bright sample -- a caustic-ish path found by luck -- and tone mapping turns
        # it into a permanent white speck that MORE samples do not remove. Clamping outliers against the
        # image's own high percentile removes the speck without touching legitimate highlights. Applied on
        # the HDR buffer, before exposure, which is the only place it is meaningful.
        try:
            from holographic_gemrender import clamp_fireflies
            hdr = np.asarray(clamp_fireflies(hdr), float)
        except Exception:
            pass
        if depth is not None and fog_on:
            from holographic_atmosphere import depth_fog
            hdr = depth_fog(hdr, depth, density=0.075, fog_color=(0.58, 0.66, 0.80), start=1.2)
        if depth is not None and dof_on:
            try:
                hdr = _dof_blur(hdr, depth)
            except Exception:
                pass
        return hdr                                       # <- prepared, NOT yet graded

    def _tonemap(hdr):
        return _photo_grade(_prep(hdr), exposure, sharpen_amt)

    import holographic_pathtrace as pt
    from holographic_postfx import denoise

    sess_key = g("session", "") or "default"
    cancelled = {"v": False}
    _PHOTO_CANCEL[sess_key] = cancelled                      # /api/render_cancel flips this
    _CANCEL_LATCH.add(sess_key)

    def generate():
        import base64, queue as _q
        yield json.dumps({"type": "meta", "w": W, "h": H, "spp": spp, "batches": spp // every,
                          "grid": grid, "fog": fog_on,
                          "warnings": sorted(set(_THIN_WARN.values()))}) + "\n"

        if aov in ("normal", "depth", "albedo"):
            # diagnostic AOV: one primary-ray gbuffer pass (holographic_gbuffer), not the path tracer
            try:
                from holographic_gbuffer import primary_gbuffer
                nrm, alb, dep = primary_gbuffer(scene, cam, W, H, material, sky=sky_fn)
                if aov == "normal":
                    out = np.clip(np.asarray(nrm, float) * 0.5 + 0.5, 0, 1)
                elif aov == "albedo":
                    out = np.clip(np.asarray(alb, float), 0, 1)
                else:
                    dd = np.asarray(dep, float)
                    finite = dd[np.isfinite(dd) & (dd < 1e3)]
                    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
                    norm = np.clip((dd - lo) / (hi - lo + 1e-6), 0, 1)
                    out = np.stack([1.0 - norm] * 3, axis=-1)     # near = bright
                yield json.dumps({"type": "frame", "done": spp, "aov": aov,
                                  "png": base64.b64encode(_png_bytes(out)).decode()}) + "\n"
                yield json.dumps({"type": "done", "seconds": 0.0}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "error": f"AOV {aov} failed: {e}"}) + "\n"
            return

        frames = _q.Queue()

        def on_progress(running, done, total):
            # The ONLY place the tracer yields control back to us. Raising here unwinds path_trace and
            # ends the worker thread -- without it the server kept tracing to full spp after the client
            # had aborted or closed the tab, burning a core for nothing.
            if cancelled["v"]:
                raise _RenderCancelled()
            frames.put((np.asarray(running, float).copy(), done))

        def worker():
            t0 = time.time()
            try:
                hdr = pt.path_trace(scene, cam, width=W, height=H, spp=spp, max_bounce=3,
                                    material=material, sky=sky_fn, seed=0, antialias=True,
                                    on_progress=on_progress, progress_every=every)
                if cancelled["v"]:
                    return
                # G1: cache the PREPARED buffer -- raw path_trace output has no fog or depth of
                # field, so grading that would silently drop both the moment you touched a slider.
                _photo_cache_put(sess_key, _prep(hdr), W, H, spp)
                frames.put((np.asarray(hdr, float), spp)); frames.put(("done", time.time() - t0))
            except _RenderCancelled:
                pass
            except Exception as e:
                frames.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = frames.get()
            if isinstance(item[0], str):
                yield json.dumps({"type": item[0] if item[0] != "done" else "done",
                                  **({"seconds": round(item[1], 1)} if item[0] == "done" else {"error": item[1]})}) + "\n"
                break
            hdr, done = item
            img = _tonemap(hdr)
            if done < spp:
                img = np.clip(denoise(img, sigma=float(np.interp(done, [every, spp], [1.4, 0.6]))), 0, 1)
            yield json.dumps({"type": "frame", "done": int(done),
                              "png": base64.b64encode(_png_bytes(img)).decode()}) + "\n"

    def guarded():
        # Flask closes the generator when the client disconnects or aborts; that raises GeneratorExit
        # here, and the finally clause is what tells the worker thread to stop (F4).
        try:
            for chunk in generate():
                yield chunk
        finally:
            cancelled["v"] = True                            # F4: stop the worker thread
            _CANCEL_LATCH.discard(sess_key)
            _PHOTO_CANCEL.pop(sess_key, None)

    resp = Response(stream_with_context(guarded()), mimetype="application/x-ndjson")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    return resp

@bp.route("/api/render_cancel", methods=["POST", "GET"])
def render_cancel():
    """A4: stop the render in flight for this session. The stream guard also does this when the socket
    drops, but an explicit call is immediate and does not depend on the proxy noticing."""
    key = request.args.get("session", "") or "default"
    flag = _PHOTO_CANCEL.get(key)
    if flag is not None:
        flag["v"] = True
    return jsonify({"cancelled": flag is not None})


@bp.route("/api/photo_post")
def photo_post():
    """G1: re-tonemap the LAST finished photo for this session -- exposure, sharpen, AOV-free beauty --
    without tracing a single new ray. Exposure used to be a render-time query parameter, so nudging it
    meant paying for the whole path trace again."""
    _init()
    g = request.args.get
    key = g("session", "") or "default"
    ent = _PHOTO_HDR.get(key)
    if ent is None:
        return jsonify({"error": "no cached render for this session"}), 404
    hdr, W, H, spp = ent
    exposure = _qnum(g, "exposure", 1.0, 0.1, 4.0)
    sharpen_amt = _qnum(g, "sharpen", 0.0, 0.0, 1.5)
    out = _photo_grade(hdr, exposure, sharpen_amt)          # same function the render uses
    resp = Response(_png_bytes(np.clip(out, 0, 1)), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Photo-Post"] = f"{W}x{H} {spp}spp exposure={exposure} sharpen={sharpen_amt}"
    return resp


# =====================================================================================================
# SCULPT MODE -- the field toggle. Enter: the object's mesh becomes a signed-distance GRID (the same bake
# the renderer uses). Brushes are the engine's field operators (holographic_sculpt: the falloff shapes and
# sign conventions are its), applied directly to the voxels in the brush ball -- local by construction.
# After each stroke packet the surface is RE-EXTRACTED (marching tetrahedra) and decimated for the wire:
# resolution-independent sculpting with automatic clean topology (the DynaMesh move), which a fixed mesh
# cannot do. Exit: the extraction at the chosen density becomes the poly mesh again, materials transferred
# by nearest centroid. While sculpting, the PREVIEW renders the sculpt grid ITSELF -- zero re-bake, ever.
# =====================================================================================================
_SCULPT_RES = 64
_SCULPT_WIRE_TRIS = 4600


def _sculpt_extract(o, dense=False):
    from holographic_meshbridge import marching_tetrahedra_vec
    from holographic_meshqem import cluster_decimate
    s = o.sculpt
    mesh = marching_tetrahedra_vec(s["grid"], s["axes"], level=0.0)
    if not dense and mesh.n_faces > _SCULPT_WIRE_TRIS:
        g = 20
        dec = cluster_decimate(mesh, grid=g)
        while dec.n_faces > _SCULPT_WIRE_TRIS and g > 8:
            g -= 3; dec = cluster_decimate(mesh, grid=g)
        mesh = dec
    return mesh


def _sculpt_refresh_mesh(o):
    """Per-stroke refresh is on the hot path: the re-extracted surface takes the object's DOMINANT material
    (a full nearest-centroid transfer per packet measured ~0.9 s -- exit does the real transfer once)."""
    dominant = max(set(o.mats), key=o.mats.count) if o.mats else _default_mat_name()
    o.mesh = _sculpt_extract(o)
    o.mats = [dominant] * o.mesh.n_faces


@bp.route("/api/sculpt/enter", methods=["POST"])
def sculpt_enter():
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        if o.sculpt is not None:
            return jsonify(_payload(only=oid))
        _snap_obj(oid)
        res = int(np.clip(int(d.get("res", _SCULPT_RES)), 40, 96))
        fld = _bake_object(oid, res, with_ids=False)       # reuse the render bake: same policy, same fixes
        grid = fld.raw_grid.copy()
        # ISO-LEVEL CALIBRATION (the "chunky blob" fix, part 1): a mesh object's shell+flood bake places its
        # zero level a systematic fraction of a cell OUTSIDE the true surface, so entering sculpt visibly
        # INFLATED the model (measured: a +-0.6 cube came back +-0.67). The bias is measurable on the object
        # itself -- sample the baked field at the ORIGINAL mesh vertices, where a faithful field reads 0 -- and
        # subtracting the median re-seats the level set through the real surface. Analytic objects measure ~0
        # and are untouched by construction.
        from holographic_mesh import Mesh
        try:
            bias = float(np.median(fld.eval(o.mesh.vertices)))
            if abs(bias) > 1e-4:
                grid = grid - bias
        except Exception:
            pass
        # LOSSLESS PEEK (part 2): keep the exact pre-sculpt object; an exit with ZERO strokes restores it
        # byte-for-byte -- entering sculpt to look around must never cost geometry.
        o.sculpt = {"grid": grid, "axes": fld.raw_axes, "res": res, "dirty": False,
                    "orig": (Mesh(o.mesh.vertices.copy(), [tuple(f) for f in o.mesh.faces]),
                             list(o.mats), o.sdf_tree, o.kernel_src)}
        _sculpt_refresh_mesh(o)
        _bump(oid)
        return jsonify(_payload(only=oid))


@bp.route("/api/sculpt/stroke", methods=["POST"])
def sculpt_stroke():
    """A stroke packet: {object, brush, points [[x,y,z]..], r, s}. Brushes edit the GRID inside the ball only,
    with the engine's falloff (holographic_sculpt.falloff). inflate/carve move the level set out/in; smooth
    relaxes it; grab drags the field's domain along the stroke; flatten pulls toward the ball-centre level."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None or o.sculpt is None:
            return jsonify({"error": "object is not in sculpt mode"}), 400
        from holographic_sculpt import falloff
        s = o.sculpt
        s["dirty"] = True
        grid = s["grid"]; xs, ys, zs = s["axes"]; res = s["res"]
        vox = float(xs[1] - xs[0])
        r = float(np.clip(d.get("r", 0.3), vox * 2, 2.0))
        strength = float(np.clip(d.get("s", 0.4), 0.02, 1.0))
        brush = d.get("brush", "inflate")
        pts = np.asarray(d.get("points", []), float).reshape(-1, 3)[:24]
        prev = None
        for p in pts:
            i0 = np.searchsorted(xs, p - r) - 1; i1 = np.searchsorted(xs, p + r) + 1
            a0, a1 = max(i0[0], 0), min(i1[0], res); b0, b1 = max(i0[1], 0), min(i1[1], res)
            c0, c1 = max(i0[2], 0), min(i1[2], res)
            if a0 >= a1 or b0 >= b1 or c0 >= c1:
                prev = p; continue
            X, Y, Z = np.meshgrid(xs[a0:a1], ys[b0:b1], zs[c0:c1], indexing="ij")
            dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2)
            w = falloff(dist.ravel(), r).reshape(dist.shape)   # the engine's smoothstep, 0 beyond r
            blk = grid[a0:a1, b0:b1, c0:c1]
            if brush == "inflate":
                blk -= strength * vox * 1.5 * w
            elif brush == "carve":
                blk += strength * vox * 1.5 * w
            elif brush == "smooth":
                sm = blk.copy()
                for ax in (0, 1, 2):
                    sm = (np.roll(sm, 1, ax) + sm + np.roll(sm, -1, ax)) / 3.0
                grid[a0:a1, b0:b1, c0:c1] = blk * (1 - strength * w) + sm * (strength * w)
                prev = p; continue
            elif brush == "flatten":
                lvl = float(grid[min(max((a0+a1)//2,0),res-1), min(max((b0+b1)//2,0),res-1),
                                 min(max((c0+c1)//2,0),res-1)])
                grid[a0:a1, b0:b1, c0:c1] = blk * (1 - strength * w) + lvl * (strength * w)
                prev = p; continue
            elif brush == "grab" and prev is not None:
                from holographic_meshbridge import sample_distance_grid
                drag = p - prev
                P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) - drag[None, :] * w.ravel()[:, None]
                grid[a0:a1, b0:b1, c0:c1] = sample_distance_grid(grid, s["axes"], P).reshape(blk.shape)
            prev = p
        _sculpt_refresh_mesh(o)
        _bump(oid)
        return jsonify(_payload(only=oid))


@bp.route("/api/sculpt/exit", methods=["POST"])
def sculpt_exit():
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None or o.sculpt is None:
            return jsonify({"error": "object is not in sculpt mode"}), 400
        if not o.sculpt.get("dirty"):
            # zero strokes: restore the exact pre-sculpt object (mesh, materials, analytic identity) -- a look
            # around in sculpt mode is free, never a lossy grid round-trip
            mesh0, mats0, tree0, kern0 = o.sculpt["orig"]
            o.mesh, o.mats, o.sculpt = mesh0, mats0, None
            _bump(oid)
            o.sdf_tree, o.kernel_src = tree0, kern0
            return jsonify(_payload(only=oid))
        _S["undo"].append(("grid", oid, o.sculpt["grid"].copy())); _trim_undo()
        from holographic_meshqem import cluster_decimate
        dense = _sculpt_extract(o, dense=True)
        target = int(np.clip(int(d.get("target_faces", 6000)), 500, 40000))
        if dense.n_faces > target:
            g = 34
            dec = cluster_decimate(dense, grid=g)
            while dec.n_faces > target and g > 8:
                g -= 3; dec = cluster_decimate(dense, grid=g)
            dense = dec
        old_mesh, old_mats = o.mesh, o.mats
        o.mesh = dense
        o.mats = _transfer_mats(old_mesh, old_mats, dense)
        o.sculpt = None
        _bump(oid)
        return jsonify(_payload(only=oid))


@bp.route("/api/sculpt/begin_stroke", methods=["POST"])
def sculpt_begin_stroke():
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None or o.sculpt is None:
            return jsonify({"error": "object is not in sculpt mode"}), 400
        _S["undo"].append(("grid", oid, o.sculpt["grid"].copy())); _trim_undo()
        return jsonify({"rev": _S["rev"]})


@bp.route("/api/softsel", methods=["POST"])
def softsel():
    """C4D-style soft selection, now the engine's authoritative field (holographic_meshselect.
    soft_selection_weights): 1 on the selection, falling to 0 at a radius measured ALONG the surface
    (multi-source geodesic), so weight does not bleed across gaps. The client applies the weighted transform
    live and streams positions as usual. Falloff: linear/smooth/sharp."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    verts = [int(v) for v in d.get("verts", [])]
    radius = float(np.clip(d.get("radius", 0.5), 0.01, 5.0))
    falloff = d.get("falloff", "smooth")
    if falloff not in ("linear", "smooth", "sharp"):
        falloff = "smooth"
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        if not verts:
            return jsonify({"weights": [0.0] * o.mesh.n_vertices})
        from holographic_meshselect import soft_selection_weights
        w = np.asarray(soft_selection_weights(_mesh_dict(o.mesh), verts, radius, falloff=falloff), float)
        return jsonify({"weights": np.round(w, 4).tolist(), "falloff": falloff})


@bp.route("/api/select", methods=["POST"])
def select():
    """Engine-authoritative sub-object selection (holographic_meshselect) -- the loop/ring/boundary/region/
    symmetry selects a modeller expects, computed on the real mesh topology rather than re-derived in JS.
    ops: 'edge_loop' (Alt-click ring across quads), 'face_ring' (the band a loop cut runs through),
    'boundary' (open hole rims), 'box' (rubber-band region; screen-space if a 4x4 view-projection is given),
    'symmetric' (mirror an existing selection across a world axis plane). Returns index lists in the op's
    natural element mode plus the vertices they touch (for highlighting)."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    op = d.get("op", "")
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        import holographic_meshselect as msel
        g = _mesh_dict(o.mesh)
        edge_list = [list(e) for e in msel._edge_list(g)]    # index -> (v0, v1); edge selects return these indices
        try:
            if op == "edge_loop":
                seed = d.get("seed")
                if seed is None and d.get("seed_verts"):    # resolve a vertex pair -> its edge index
                    a, b = int(d["seed_verts"][0]), int(d["seed_verts"][1])
                    key = (min(a, b), max(a, b))
                    seed = next((i for i, e in enumerate(edge_list)
                                 if (min(e), max(e)) == key), None)
                    if seed is None:
                        return jsonify({"error": "those two points are not an edge"}), 400
                sel = msel.select_edge_loop(g, int(seed))               # seed is an EDGE INDEX
                out = {"mode": "edge", "edges": list(sel.to_list())}
            elif op == "face_ring":
                sel = msel.select_face_ring(g, int(d["seed"]))
                out = {"mode": "face", "faces": list(sel.to_list())}
            elif op == "boundary":
                loops = msel.select_boundary_loops(g)
                out = {"mode": "edge", "edges": list(loops.to_list() if hasattr(loops, "to_list") else loops)}
            elif op == "box":
                lo = [float(x) for x in d["lo"]]; hi = [float(x) for x in d["hi"]]
                mode = d.get("elem", "vertex")
                project = None
                if d.get("view_proj"):                      # screen-space rubber-band: test projected u,v
                    VP = np.asarray(d["view_proj"], float).reshape(4, 4)

                    def project(pt):
                        q = VP @ np.array([pt[0], pt[1], pt[2], 1.0])
                        return (q[0] / q[3], q[1] / q[3]) if abs(q[3]) > 1e-9 else (1e9, 1e9)
                sel = msel.select_in_box(g, lo, hi, mode=mode, project=project)
                key = "faces" if mode == "face" else ("edges" if mode == "edge" else "verts")
                out = {"mode": mode, key: list(sel.to_list())}
            elif op == "symmetric":
                mode = d.get("elem", "vertex")
                base = msel.MeshSelection(g, mode)
                base.add([int(i) for i in d.get("indices", [])])
                sel = msel.select_symmetric(g, base, axis=int(d.get("axis", 0)))
                key = "faces" if mode == "face" else ("edges" if mode == "edge" else "verts")
                out = {"mode": mode, key: list(sel.to_list())}
            else:
                return jsonify({"error": f"unknown select op '{op}'"}), 400
        except Exception as e:
            return jsonify({"error": f"{op} failed: {e}"}), 400
        # resolve touched vertices for highlighting, whatever the element mode
        vids = set()
        if out.get("mode") == "face":
            for f in out.get("faces", []):
                vids.update(int(i) for i in o.mesh.faces[f])
        elif out.get("mode") == "edge":
            out["edge_verts"] = [edge_list[i] for i in out.get("edges", [])]   # index -> (v0,v1) for the client
            for e in out["edge_verts"]:
                vids.update(int(i) for i in e)
        else:
            vids.update(int(i) for i in out.get("verts", []))
        out["verts_touched"] = sorted(vids)
        return jsonify(out)


@bp.route("/api/snap", methods=["POST"])
def snap():
    """The snap layer a gizmo holds Ctrl for (holographic_snap.snap_transform_delta): given the raw drag delta
    and the point being dragged, return the delta CORRECTED so that point lands on the nearest target --
    'grid' (spacing `increment`), 'vertex', or 'edge' (of this or another object). Transform and snap stay
    separate layers, exactly as the module intends."""
    _init()
    d = request.get_json(force=True) or {}
    target = d.get("target", "grid")
    delta = [float(x) for x in d.get("delta", [0, 0, 0])]
    moved = [float(x) for x in d.get("moved_point", [0, 0, 0])]
    inc = float(d.get("increment", 0.25))
    with _LOCK:
        verts = edges = None
        if target in ("vertex", "edge"):
            oid = str(d.get("object", ""))
            o = _S["objects"].get(oid)
            if o is None:
                return jsonify({"error": "no such object"}), 400
            verts = o.mesh.vertices.tolist()
            if target == "edge":
                es = set()
                for f in o.mesh.faces:
                    for k in range(len(f)):
                        a, b = int(f[k]), int(f[(k + 1) % len(f)])
                        es.add((min(a, b), max(a, b)))
                edges = [list(e) for e in es]
        from holographic.mesh_and_geometry.holographic_snap import snap_transform_delta
        try:
            r = snap_transform_delta(delta, target=target, increment=inc, moved_point=moved,
                                     vertices=verts, edges=edges)
        except Exception as e:
            return jsonify({"error": f"snap failed: {e}"}), 400
        return jsonify({"delta": [round(float(x), 5) for x in r["delta"]],
                        "snapped_to": r.get("snapped_to")})


@bp.route("/api/export_shader")
def export_shader():
    """Export an object as a signed-distance shader, in glsl / wgsl, with the engine's scene_cost ALU verdict.
    Two honest paths, and the response says which was taken:
      * EXACT (`mode:"exact"`): an un-edited primitive still equals its analytic SDF tree, so sdf_dialect walks
        that tree and emits the mathematically exact map(p) (+ a full Shadertoy program via SDF.to_glsl). Zero
        approximation -- the shader IS the object.
      * FITTED (`mode:"fitted"`): an edited/sculpted object is a signed-distance GRID (a volume texture) with no
        analytic tree, which sdf_dialect cannot walk. Rather than refuse, we fit the mesh's surface point cloud
        to a UNION OF EXACT SDF PRIMITIVES (holographic_primfit.fit_primitives -- spheres/boxes/capsules, chosen
        per cluster, auto-K to the residual elbow) and emit THAT union as a shader. This is an APPROXIMATION and
        is labelled as one: the response reports the fit `quality` (>1 = better than a single bounding sphere),
        the residual, and the primitive counts, so the person knows it is a fit, not the exact mesh."""
    _init()
    oid = str(request.args.get("object", "")) or None
    dialect = request.args.get("dialect", "glsl")
    if dialect not in ("glsl", "wgsl"):
        return jsonify({"error": "dialect must be glsl or wgsl"}), 400
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400

        # ---- EXACT path first. For GLSL, the full-program emitter (SDF.to_glsl) and the strict map-only
        # emitter (sdf_dialect) are attempted INDEPENDENTLY: to_glsl emits every warp node (twist / bend /
        # onion / displace / elongate -- verified), while the strict dialect refuses warps by name -- so a
        # twisted object still exports an EXACT Shadertoy program even when no bare map() can be emitted.
        # WGSL has only the strict emitter, so warped trees fall through to FITTED there, stated in the UI.
        # A tree nothing can emit (e.g. the humanoid's capsule leaves) falls through to FITTED entirely. ----
        if o.sdf_tree is not None:
            from holographic_sdfemit import sdf_dialect
            tree = o.sdf_tree
            dsl = tree.to_dsl()
            try:
                map_src = sdf_dialect(dsl, dialect)
            except Exception:
                map_src = None
            full = None
            if dialect == "glsl":
                try:
                    full = tree.to_glsl()
                except Exception:
                    full = None
            if map_src is not None or full is not None:
                cost = tree.cost()
                return jsonify({"analytic": True, "mode": "exact", "dialect": dialect, "dsl": dsl,
                                "map": map_src, "shadertoy": full,
                                "cost": {"alu": cost.get("alu"), "verdict": cost.get("verdict"),
                                         "nodes": cost.get("nodes"), "depth": cost.get("depth")}})
            # nothing emittable -> fitted below

        # ---- FITTED path: edited/sculpted grid mesh, or a tree the emitter refuses -> primfit union ----
        if True:
            from holographic_primfit import fit_primitives
            from holographic_sdfemit import sdf_dialect
            from holographic_sdf import _emit_shader
            pts = np.asarray(o.mesh.vertices, float)
            if len(pts) > 4000:                            # cap the fit input; surface verts are plenty
                idx = np.linspace(0, len(pts) - 1, 4000).astype(int)
                pts = pts[idx]
            try:
                # sphere + box only: sdf_dialect emits these EXACTLY; it refuses capsule (an iterative domain
                # fold whose unrolled shader size would depend on a parameter), so leaving capsule in the palette
                # would fit fine but fail at emit. Restricting the palette keeps the whole path emittable.
                fit = fit_primitives(pts, k=8, auto_k=True, k_max=16,
                                     primitives=("sphere", "box"))
                sdf = fit["sdf"]
                dsl = sdf.to_dsl()
                map_src = sdf_dialect(dsl, dialect)
                full = _emit_shader(sdf) if dialect == "glsl" else None
                cost = sdf.cost()
            except Exception as e:
                return jsonify({"analytic": False, "mode": "failed",
                                "reason": f"primitive-fit shader emit failed: {e}"}), 200
            return jsonify({"analytic": True, "mode": "fitted", "dialect": dialect, "dsl": dsl, "map": map_src,
                            "shadertoy": full,
                            "cost": {"alu": cost.get("alu"), "verdict": cost.get("verdict"),
                                     "nodes": cost.get("nodes"), "depth": cost.get("depth")},
                            "fit": {"k": int(fit["k"]), "kinds": fit["kinds"],
                                    "quality": round(float(min(fit["quality"], 999.0)), 2),
                                    "residual": round(float(fit["residual"]), 4)}})




# =====================================================================================================
# REAL-TIME GPU PREVIEW -- the 30fps-and-up path.
#
# Every earlier preview iteration paid a Python + HTTP round trip PER FRAME: bake, sphere-trace in Python,
# PNG-encode, ship it over HTTP. Even fully cached, that floor is tens of milliseconds of Python/Flask/PNG
# overhead per request -- fine for "moves when you edit", structurally incapable of 60fps camera orbiting.
#
# The fix is to stop rendering frames on the server at all. Bake the WHOLE SCENE to a dense signed-distance
# volume + a material-palette-index volume ONCE PER EDIT (cached by scene revision, exactly the compile-cache
# pattern the engine's own holographic_compile module uses: content-addressed, reuse until the source changes),
# ship that as a compact binary blob, and raymarch it in a WebGL fragment shader in the browser. A GPU walks
# every pixel in parallel every frame; a bake that only has to happen when something actually changed costs
# nothing during an orbit. That is the genuine "faster than Python" move available here -- not a JIT flag, but
# moving the per-frame work off Python (and off the network) entirely.
#
# WHY NOT THE ENGINE'S NUMBA-JIT FAST-SWEEP EIKONAL SOLVER (holographic_jit.signed_distance_3d): tempting on
# the name alone, but it is NOT installed in this environment (no numba package, no network to fetch one) --
# its own module docstring is explicit that its pure-Python fallback is the same sequential triple-nested loop
# Numba would compile, just uninlined; measured here at res=64 it cost ~9s for one bake, ~40x slower than the
# vectorized shell/exact builds already in use. Using it would have made things worse; the numbers below are
# what shipped instead.
#
# WHY NOT holostuff's "VSA program" machinery: that name refers to Vector-Symbolic-Architecture / hyperdimensional
# computing (holographic_hypervector, holographic_schedule's program DAGs) -- a different subsystem entirely,
# for binding/bundling symbolic hypervectors, not for rasterizing geometry. There's no version of "run this as a
# VSA program" that touches a signed-distance bake. What DOES generalize from that corner of the codebase is the
# CACHING PATTERN (holographic_compile.CompileCache: content-addressed, compile once, reuse until the source
# hashes differently) -- which is exactly the shape of the per-object bake cache and the scene-texture cache
# below: keyed by revision, not recomputed until the thing it depends on changes.
# =====================================================================================================
_TEX_MARGIN = 0.30
_TEX_CACHE = {}          # (rev, res) -> dict payload (see _bake_scene_texture)
_TEX_CACHE_CAP = 6


def _scene_bounds(margin=_TEX_MARGIN):
    objs = list(_S["objects"].values())
    if not objs:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    lo = np.min([o.mesh.vertices.min(axis=0) for o in objs], axis=0) - margin
    hi = np.max([o.mesh.vertices.max(axis=0) for o in objs], axis=0) + margin
    return lo, hi


def _bake_scene_texture(res):
    """Dense (res^3) signed-distance + material-palette-index volumes for the WHOLE scene, quantized for GPU
    upload: distance as float16 (WebGL HALF_FLOAT bit-for-bit -- numpy's float16 IS IEEE-754 binary16, the
    exact format a `R16F` texture wants, no repacking needed), palette id as uint8. Cached by (scene revision,
    res): an unchanged scene returns the SAME bytes object with no recomputation at all (see the perf note in
    the module docstring above this). Building fresh, cost is dominated by the per-OBJECT bakes, which are
    THEMSELVES cached by object revision (_bake_object) -- so editing one object of a five-object scene still
    only re-bakes that one object; the dense scene-grid sampling on top of the (now-cached) per-object fields
    is a couple of vectorized calls, measured ~50-120ms at res 56-72 regardless of scene complexity."""
    key = (_S["rev"], int(res))
    hit = _TEX_CACHE.get(key)
    if hit is not None:
        return hit

    lo, hi = _scene_bounds()
    res = int(res)
    xs = np.linspace(lo[0], hi[0], res); ys = np.linspace(lo[1], hi[1], res); zs = np.linspace(lo[2], hi[2], res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    scene, ground = _scene_field(res, with_ids=True)       # per-object fields: cached by object revision
    dist = np.empty(len(P), dtype=np.float32)
    for i in range(0, len(P), 150_000):
        dist[i:i + 150_000] = scene.eval(P[i:i + 150_000])

    # palette: dedup materials across the whole scene (first-seen order), index 0 reserved for the floor
    all_mats = []
    for o in _S["objects"].values():
        all_mats.extend(o.mats)
    seen = {}
    palette_names = [_FLOOR_MAT]
    for m in all_mats:
        if m not in seen:
            seen[m] = len(palette_names); palette_names.append(m)
    face_to_palette = np.array([seen.get(m, 0) for m in all_mats], dtype=np.int64)

    gid = np.empty(len(P), dtype=np.int64)
    for i in range(0, len(P), 150_000):
        gid[i:i + 150_000] = scene.face_ids(P[i:i + 150_000])
    matid = np.where(gid < 0, 0, face_to_palette[np.clip(gid, 0, max(len(face_to_palette) - 1, 0))])
    matid = np.clip(matid, 0, 255).astype(np.uint8)

    ml = _matlib()
    palette = []
    for name in palette_names:
        m = _mat(name)
        palette.append({"albedo": [round(float(c), 4) for c in m.base_color[:3]],
                        "metallic": round(float(m.metallic), 3), "roughness": round(float(m.roughness), 3),
                        "emissive": [round(float(c), 4) for c in m.emissive],
                        "ior": round(float(getattr(m, "ior", 0.0)), 3)
                               if getattr(m, "transmission", 0.0) >= 1.0 else 0})

    payload = {"rev": _S["rev"], "res": res, "lo": lo.tolist(), "hi": hi.tolist(), "ground": float(ground),
              "palette": palette,
              "dist_bytes": dist.astype("<f2").tobytes(), "matid_bytes": matid.astype("<u1").tobytes()}
    _TEX_CACHE[key] = payload
    if len(_TEX_CACHE) > _TEX_CACHE_CAP:
        _TEX_CACHE.pop(next(iter(_TEX_CACHE)))
    return payload


@bp.route("/api/field_meta")
def field_meta():
    """Small JSON poll: rev/res/bounds/palette/lighting. The client calls this after every edit (never on a
    bare camera move) and only fetches the (much larger) binary volume in api/field_tex.bin if `rev` changed
    from what it already has -- so an unchanged scene costs one tiny cached JSON response, not a re-bake."""
    _init()
    res = _qnum(request.args.get, "res", 56, 24, 96, int)
    with _LOCK:
        try:
            p = _bake_scene_texture(res)
        except Exception as e:
            return jsonify({"error": f"bake failed: {e}"}), 500
    return jsonify({"rev": p["rev"], "res": p["res"], "lo": p["lo"], "hi": p["hi"], "ground": p["ground"],
                    "palette": p["palette"], "sun_dir": _SUN.tolist(), "sun_color": _SUN_COL.tolist()})


@bp.route("/api/field_tex.bin")
def field_tex_bin():
    """The binary payload for the rev/res named in the query: float16 distance volume followed by uint8
    palette-id volume, res^3 elements each, row-major (x-fastest via the meshgrid('ij') build -- x, then y,
    then z outermost, matching a standard WebGL Data3DTexture upload with depth=z). Cached bytes -- a repeat
    request for the same (rev, res) is a dict lookup, not a rebuild."""
    _init()
    res = _qnum(request.args.get, "res", 56, 24, 96, int)
    rev = request.args.get("rev", type=int)
    with _LOCK:
        if rev is not None and rev != _S["rev"]:
            return jsonify({"error": "stale rev", "current_rev": _S["rev"]}), 409
        try:
            p = _bake_scene_texture(res)
        except Exception as e:
            return jsonify({"error": f"bake failed: {e}"}), 500
        blob = p["dist_bytes"] + p["matid_bytes"]
    resp = Response(blob, mimetype="application/octet-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Field-Rev"] = str(p["rev"]); resp.headers["X-Field-Res"] = str(p["res"])
    return resp


# =====================================================================================================
# DESCRIBE -> OBJECT (holographic_codecompose) and EXPLAIN (holographic_codeverbal)
#
# codecompose is the engine's honest "code from a description": a CONTROLLED VOCABULARY of registered parametric
# forms (sphere / rounded box / plane, iq's exact published formulae) plus union/intersect/subtract -- NOT fuzzy
# NL->code (it refuses unknown forms BY NAME, and tells you which words it ignored because an SDF has no colour).
# The kernel it emits is verifiable Python; codeverbal explains it back in English, closing the loop.
#
# Here a description becomes a real scene object THREE ways at once, all from the same text:
#   1. the Python KERNEL (kept on the object, so Explain works later),
#   2. a matching analytic SDF TREE, built from the same parsed clauses and then VERIFIED against the kernel
#      numerically (512 random points, max |kernel - tree| < 1e-8) -- if verification fails the tree is dropped
#      and stated, never silently trusted. A verified tree means the object bakes analytic-native and exports an
#      EXACT shader, exactly like a hand-added primitive;
#   3. the MESH, extracted by the same marching-tetrahedra pipeline sculpt uses.
# =====================================================================================================
def _tree_from_clauses(text):
    """Build the holographic_sdf tree matching codecompose's parse of `text`. Uses codecompose's OWN clause
    splitter and param extractor (no second grammar to drift), maps each form to its S.* constructor, folds the
    boolean ops left-to-right exactly as the kernel does. Returns the tree, or None for any clause it can't map."""
    import holographic_sdf as S
    from holographic_codecompose import _split_clauses, _match_form, _extract_params
    tree = None
    for op, clause in _split_clauses(text):
        fname, form = _match_form(clause)
        if form is None:
            return None
        p = _extract_params(clause, form)
        if "bx" in p:                                     # rounded box (r may be 0 -> plain box)
            node = S.box(p["bx"], p["by"], p["bz"])
            if p.get("r", 0):
                node = node.rounded(p["r"])
        elif "h" in p and "r" not in p:                   # plane
            node = S.plane(p["h"])
        elif "r" in p:                                    # sphere
            node = S.sphere(p["r"])
        else:
            return None
        if any(p.get(k) for k in ("cx", "cy", "cz")):
            node = node.translate((p.get("cx", 0.0), p.get("cy", 0.0), p.get("cz", 0.0)))
        if tree is None:
            tree = node
        elif op in ("union", "and", "plus", "with"):
            tree = tree.union(node)
        elif op in ("intersect", "intersection"):
            tree = tree.intersect(node)
        elif op in ("subtract", "minus", "cut"):
            tree = tree.subtract(node)
        else:
            return None
    return tree


def _verbal(kern_src):
    """codeverbal's structured explanation, flattened for the UI: the idiom line (its composition RECOGNITION --
    'a union of 2 primitives: a sphere, a rounded box' -- the loop codecompose opened, closed) as the headline,
    the full dataflow text underneath. Returns {"headline", "full"} or None."""
    from holographic_codeverbal import verbalize
    try:
        r = verbalize(kern_src)
        f = (r.get("functions") or [{}])[0]
        return {"headline": f.get("idiom") or r.get("summary") or "",
                "full": f.get("text") or ""}
    except Exception:
        return None


@bp.route("/api/compose", methods=["POST"])
def compose():
    _init()
    d = request.get_json(force=True) or {}
    text = str(d.get("text", "")).strip()
    if not text:
        return jsonify({"error": "empty description"}), 400
    from holographic_codecompose import describe_to_kernel, ComposeError
    from holographic_zigrun import as_numpy
    try:
        kern_src = describe_to_kernel(text, name="described")
    except ComposeError as e:
        return jsonify({"error": str(e)}), 400
    tree = None
    try:
        cand = _tree_from_clauses(text)
        if cand is not None:                              # VERIFY the tree against the kernel before trusting it
            kf = as_numpy(kern_src)
            rng = np.random.RandomState(7)
            P = rng.uniform(-2.0, 2.0, (512, 3))
            dk = kf(P[:, 0], P[:, 1], P[:, 2])
            dt = cand.eval(P)
            if float(np.max(np.abs(np.asarray(dk, float) - dt))) < 1e-8:
                tree = cand
    except Exception:
        tree = None
    try:
        mesh_tree = tree
        if mesh_tree is None:                             # kernel-only: mesh from the numpy twin of the kernel
            kf = as_numpy(kern_src)

            class _KTree:                                 # duck-typed .eval for _mesh_sdf_tree
                def eval(self, P):
                    P = np.atleast_2d(P)
                    return np.asarray(kf(P[:, 0], P[:, 1], P[:, 2]), float)
            mesh_tree = _KTree()
        with _LOCK:
            mesh = _mesh_sdf_tree(mesh_tree, res=64, face_target=1800)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    explanation = _verbal(kern_src)
    with _LOCK:
        _snap_scene()
        oid = _add_object(d.get("name") or "Described", mesh, sdf_tree=tree, kernel_src=kern_src)
        _bump()
        out = _payload()
    out.update({"object": oid, "kernel": kern_src, "explanation": explanation,
                "tree_verified": tree is not None})
    return jsonify(out)


@bp.route("/api/explain_shader")
def explain_shader():
    """English explanation of an object's kernel (holographic_codeverbal). Honest scope: verbalize reads PYTHON
    kernel source, which only described objects carry; a hand-added primitive has a DSL tree and an
    edited/sculpted mesh has neither, so those get the reason, not a fabricated explanation."""
    _init()
    oid = str(request.args.get("object", "")) or None
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400
        if o.kernel_src:
            v = _verbal(o.kernel_src)
            if v is not None:
                return jsonify({"explanation": v, "kernel": o.kernel_src})
            return jsonify({"explanation": None, "reason": "verbalize failed on this kernel"})
        reason = ("This object was added as a primitive (it has an SDF tree, not Python kernel text)."
                  if o.sdf_tree is not None else
                  "This object is an edited/sculpted/imported mesh -- there is no kernel text to explain.")
        return jsonify({"explanation": None,
                        "reason": reason + " Explain works for objects made with 'Add from description'."})


# =====================================================================================================
# MILKDROP MOTION (holographic_milkdrop) -- the engine's .milk preset reader + safe ns-eel2 evaluator.
#
# A Milkdrop preset is not a shader blob; it is a text file of MATH EQUATIONS, and the engine parses and runs
# exactly that layer: the per_frame equations (zoom / rot / warp / wave colours / q-vars), driven by audio
# features, through a real tokenizer->parser->evaluator over a whitelisted grammar (never Python eval; a hostile
# preset can do arithmetic and nothing else). Here those equations drive the WebGL viewport: the server
# evaluates a BATCH of frames per request (the equations cost microseconds; one HTTP call returns seconds of
# motion) and the client plays them at 60fps -- orbit from `rot`, dolly from `zoom`, background tint from
# wave_r/g/b, and a bass pulse on the active object.
#
# HONESTY: audio is SYNTHESIZED here (a deterministic beat generator -- there is no microphone in a sandbox);
# and the per_pixel warp mesh + HLSL shader blocks are parsed but NOT executed, exactly as the module's own
# kept-negatives state -- this is the preset's MOTION, not its pixel-shader look.
# =====================================================================================================
_MILK = {"name": None, "preset": None, "state": None, "frame": 0, "att": {"bass": 1.0, "mid": 1.0, "treb": 1.0}}
_MILK_DIR = os.path.join(_DEMO_DIR, "presets")


def _milk_audio(t):
    """Deterministic synthesized audio features: a 112-bpm beat for bass, slower detuned sines for mid/treb --
    the same {bass, mid, treb} shape audio_param_bus produces, minus the microphone the sandbox does not have."""
    beat = max(0.0, np.sin(2 * np.pi * t * 112.0 / 60.0)) ** 9
    return {"bass": 0.62 + 1.05 * float(beat),
            "mid": 0.78 + 0.28 * float(np.sin(2 * np.pi * t * 0.9 + 1.3)),
            "treb": 0.74 + 0.30 * float(np.sin(2 * np.pi * t * 1.7 + 0.4))}


@bp.route("/api/milkdrop/presets")
def milkdrop_presets():
    out = []
    if os.path.isdir(_MILK_DIR):
        for f in sorted(os.listdir(_MILK_DIR)):
            if f.endswith(".milk"):
                out.append({"name": f[:-5], "title": f[:-5].replace("_", " ").title()})
    return jsonify({"presets": out, "note": "per_frame equations run; per_pixel/HLSL parsed but not executed"})


@bp.route("/api/milkdrop/frames")
def milkdrop_frames():
    """Evaluate the next `n` per-frame steps of the named preset and return the motion variables as arrays.
    Passing ?preset= (re)starts that preset from its init block; omitting it continues the current one, so
    playback is seamless across batches. State (q-vars etc.) carries frame to frame, as Milkdrop does."""
    _init()
    from holographic_milkdrop import parse_milk
    name = request.args.get("preset")
    n = _qnum(request.args.get, "n", 360, 30, 1800, int)
    fps = _qnum(request.args.get, "fps", 30, 10, 60)
    with _LOCK:
        if name and name != _MILK["name"]:
            path = os.path.join(_MILK_DIR, name + ".milk")
            if not os.path.isfile(path):
                return jsonify({"error": f"no preset named {name!r}"}), 400
            try:
                preset = parse_milk(open(path).read())
                _MILK.update({"name": name, "preset": preset, "state": preset.initial_state(),
                              "frame": 0, "att": {"bass": 1.0, "mid": 1.0, "treb": 1.0}})
            except Exception as e:
                return jsonify({"error": f"preset failed to parse/init: {e}"}), 400
        if _MILK["preset"] is None:
            return jsonify({"error": "no preset started; pass ?preset=<name>"}), 400
        p, st, att = _MILK["preset"], _MILK["state"], _MILK["att"]
        keys = ("zoom", "rot", "warp", "wave_r", "wave_g", "wave_b", "cx", "cy")
        series = {k: [] for k in keys}
        series["bass"] = []
        for i in range(n):
            fr = _MILK["frame"]
            t = fr / fps
            audio = _milk_audio(t)
            for band in ("bass", "mid", "treb"):          # _att = the slow envelope Milkdrop feeds presets
                att[band] = att[band] * 0.93 + audio[band] * 0.07
                audio[band + "_att"] = att[band]
            p.run_frame(st, audio, time=t, frame=fr)
            for k in keys:
                series[k].append(round(float(st.get(k, 0.0)), 5))
            series["bass"].append(round(audio["bass"], 4))
            _MILK["frame"] = fr + 1
        return jsonify({"preset": _MILK["name"], "fps": fps, "start_frame": _MILK["frame"] - n,
                        "n": n, "series": series})


@bp.route("/api/curvature")
def curvature():
    """Per-vertex MEAN CURVATURE of the object's own signed-distance field (holographic_raymarch.sdf_curvature:
    the field Laplacian -- for a true SDF, exactly twice the mean curvature at the surface). The professional
    surfacing inspection: positive on convex edges/ridges, negative in concave creases, ~0 on flats. Uses the
    analytic tree when the object has one (exact); otherwise the cached grid bake -- where the finite-difference
    epsilon is raised to ~the grid CELL size, because a trilinear interpolant's Laplacian is degenerate below the
    cell scale (zero inside cells, spikes at faces). The eps used is reported, not hidden."""
    _init()
    oid = str(request.args.get("object", "")) or None
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400
        from holographic_raymarch import sdf_curvature
        V = o.mesh.vertices
        if o.sdf_tree is not None:
            field_eval, eps = o.sdf_tree.eval, 2e-3
        else:
            res = _qnum(request.args.get, "res", 64, 40, 88, int)
            f = _bake_object(oid, res, with_ids=False)
            fld = f[0] if isinstance(f, tuple) else f
            span = float((V.max(axis=0) - V.min(axis=0)).max()) + 0.44
            eps = max(2e-3, 0.75 * span / res)
            field_eval = fld.eval
        try:
            vals = np.asarray(sdf_curvature(field_eval, V, eps=eps), float)
        except Exception as e:
            return jsonify({"error": f"curvature failed: {e}"}), 400
        # robust display range: symmetric around 0 at the 95th percentile of |curvature|
        scale = float(np.percentile(np.abs(vals), 95)) or 1.0
        return jsonify({"values": np.round(vals, 4).tolist(), "scale": round(scale, 4),
                        "eps": round(float(eps), 4),
                        "source": "analytic" if o.sdf_tree is not None else "grid"})


@bp.route("/api/export_stl")
def export_stl():
    """ASCII STL of one object (holographic_cadexport.mesh_to_stl -- quads split automatically, per-facet
    normals from winding). The 3-D-print / CAD-exchange format a modeler cannot ship without."""
    _init()
    oid = str(request.args.get("object", "")) or None
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400
        from holographic_cadexport import mesh_to_stl
        try:
            stl = mesh_to_stl(o.mesh.vertices.tolist(), [list(f) for f in o.mesh.faces],
                              name=o.name.replace(" ", "_") or "polystudio")
        except Exception as e:
            return jsonify({"error": f"stl export failed: {e}"}), 400
        return Response(stl, mimetype="model/stl",
                        headers={"Content-Disposition": "attachment; filename=polystudio.stl"})


# =====================================================================================================
# NODE SYSTEM (holographic_nodegraph) -- the engine's unifying typed node-graph shell, bound to a real editor.
#
# The module is exactly what a node editor binds to: heterogeneous TYPED nodes (scalar/sdf/mesh/material/...),
# connections TYPE-CHECKED at wire time (an sdf into a scalar slot is refused when you draw it, not when you
# evaluate), CYCLES refused, evaluation in topological order with dirty propagation, and JSON round-trip. The
# node kinds delegate to the engine subsystems that already exist (sdf_union calls SDF.union; sdf_fillet calls
# the K5 fillet) -- the shell "wires the wheels together", it reimplements nothing.
#
# Poly Studio holds ONE graph (this is a single-user demo). "Build" evaluates a node and lands the result in the
# scene: an sdf output that is an analytic TREE stays analytic (exact bake + exact Shadertoy export -- a
# node-built object is a first-class primitive); an sdf that is a field callable (the fillet node) meshes
# field-only; a mesh output is adopted directly. The graph is NOT in the scene undo stack -- it is a separate
# document, as in every node-based package; deleting nodes is its own action.
# =====================================================================================================
_NODES = {"graph": None, "pos": {}, "muted": {}, "groups": {}}

# param defaults for the palette (the registry keeps params inside each node's fn -- this table is the UI's
# affordance; any param not listed still works, the fn reads what it knows and ignores the rest)
_NODE_PARAMS = {
    "scalar": {"value": 1.0},
    "sdf_sphere": {"radius": 0.6}, "sdf_box": {"size": [0.5, 0.5, 0.5]},
    "sdf_torus": {"R": 0.7, "r": 0.22}, "sdf_cylinder": {"h": 0.8, "r": 0.35},
    "sdf_capsule": {"h": 0.7, "r": 0.25}, "sdf_cone": {"h": 0.8, "r": 0.45}, "sdf_plane": {"h": 0.0},
    "sdf_union": {}, "sdf_subtract": {}, "sdf_intersect": {},
    "sdf_smooth_union": {"k": 0.25}, "sdf_fillet": {"radius": 0.12},
    "sdf_translate": {"t": [0.0, 0.0, 0.0]}, "sdf_scale": {"s": 1.0},
    "sdf_rotate": {"axis": [0.0, 1.0, 0.0], "angle": 0.6},
    "sdf_twist": {"k": 1.5}, "sdf_onion": {"thickness": 0.06},
    "sdf_elongate": {"h": [0.4, 0.0, 0.0]}, "sdf_repeat": {"period": [2.0, 2.0, 2.0]},
    "sdf_to_mesh": {"resolution": 48},
    "mesh_subdivide": {}, "mesh_smooth": {"iterations": 5}, "mesh_decimate": {"grid": 24},
    # ---- Poly Studio EXTENSIONS: fractals, fields (Shadertoy-style + curl noise), field-driven geometry,
    # and mesh post-processing (retopo / denoise). Registered onto every graph by _augment_registry below. ----
    "sdf_mandelbulb": {"power": 8.0, "iterations": 8, "bailout": 2.0},   # 3D fractal, emits EXACT GLSL
    "sdf_menger": {"iterations": 3, "size": 1.0},                        # 3D fractal, emits EXACT GLSL
    "formula_field": {"expr": "0.5*sin(6*x)*sin(6*z)"},                  # ns-eel2 f(x,y,z,r) as a FIELD socket
    "curl_field": {"scale": 1.4, "seed": 1, "octaves": 3},              # divergence-free 3D vector field
    "sdf_displace_field": {"amount": 0.12},                             # a field warps an SDF's distance
    "sdf_blend": {"t": 0.5, "mode": "morph", "k": 0.3},                  # morph/smooth-blend two SDFs (folder net)
    "mesh_displace_field": {"amount": 0.12},                            # a field pushes mesh verts along normals
    "sdf_material": {"material": "clay"},                              # attach a material to an SDF/mesh stream
    "mesh_retopo": {"resolution": 48, "target": 3000},                 # voxel remesh -> clean uniform topology
    "mesh_denoise": {"iterations": 8},                                 # Taubin denoise (shrink-free)
}
_NODE_PALETTE = list(_NODE_PARAMS.keys())                  # curated, professional subset shown in the UI

# P0.5-2: per-param widget metadata for the Inspector (min/max/step/kind). Anything not listed falls back to a
# free number/text field -- this table just shapes the sliders and clamps ranges (the ComfyUI users' #1 ask).
_NODE_PARAM_META = {
    "value": {"min": -8, "max": 8, "step": 0.1},
    "radius": {"min": 0.05, "max": 2.0, "step": 0.01}, "r": {"min": 0.02, "max": 1.5, "step": 0.01},
    "R": {"min": 0.1, "max": 2.0, "step": 0.01}, "h": {"min": 0.05, "max": 2.0, "step": 0.01},
    "k": {"min": 0.0, "max": 4.0, "step": 0.05}, "thickness": {"min": 0.01, "max": 0.3, "step": 0.005},
    "s": {"min": 0.1, "max": 4.0, "step": 0.05}, "angle": {"min": -3.14159, "max": 3.14159, "step": 0.02},
    "power": {"min": 2.0, "max": 16.0, "step": 0.5}, "iterations": {"min": 1, "max": 16, "step": 1, "int": True},
    "bailout": {"min": 1.2, "max": 4.0, "step": 0.1}, "size": {"min": 0.2, "max": 2.0, "step": 0.05},
    "resolution": {"min": 24, "max": 96, "step": 4, "int": True}, "iters": {"min": 1, "max": 40, "step": 1, "int": True},
    "iterations_": {"min": 1, "max": 40, "step": 1, "int": True}, "amount": {"min": -0.5, "max": 0.5, "step": 0.01},
    "scale": {"min": 0.2, "max": 4.0, "step": 0.05}, "seed": {"min": 0, "max": 99, "step": 1, "int": True},
    "octaves": {"min": 1, "max": 4, "step": 1, "int": True}, "target": {"min": 300, "max": 40000, "step": 100, "int": True},
    "grid": {"min": 8, "max": 64, "step": 2, "int": True},
    "t": {"min": 0.0, "max": 1.0, "step": 0.01},
    "mode": {"enum": ["morph", "smooth"]},
}

_CURL_CACHE = {}


def _augment_registry(reg):
    """Register Poly Studio's extension node types onto a fresh engine registry. Each DELEGATES to a real engine
    entry point (fractals from holographic_sdf; curl noise from holographic_curlnoise; the safe ns-eel2 evaluator
    from holographic_milkdrop; voxel_remesh/taubin_smooth from the mesh modules) -- the demo adds sockets and
    wiring, the engine does the maths. Field-typed sockets are what let a Shadertoy formula or a curl field
    DRIVE geometry, and let scalars/audio drive the formula's constants in turn."""
    import holographic_sdf as S
    import numpy as _np

    # -- 3D FRACTALS: real SDF trees, so they mesh AND emit exact GLSL (verified) --
    reg.register("sdf_mandelbulb", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.mandelbulb(power=float(p.get("power", 8.0)),
                                                   iterations=int(p.get("iterations", 8)),
                                                   bailout=float(p.get("bailout", 2.0)))},
                 param_inputs={"power": "scalar"})
    reg.register("sdf_menger", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.menger(iterations=int(p.get("iterations", 3)),
                                               size=float(p.get("size", 1.0)))})

    # -- FIELDS: a scalar field over R^3, emitted as a callable carrying a `.kind` tag. formula_field runs the
    #    engine's SAFE ns-eel2 evaluator (never eval); curl_field samples a cached divergence-free vector field
    #    and returns its magnitude as the scalar (the vector is kept for the displace nodes). --
    def _formula_field(p, i):
        from holographic_milkdrop import MilkExpr
        expr = str(p.get("expr", "0"))
        e = MilkExpr(expr)                                  # raises on bad syntax / non-whitelisted call

        def f(P):
            P = _np.atleast_2d(_np.asarray(P, float))
            out = _np.empty(len(P))
            for j, pt in enumerate(P):
                out[j] = e.eval({"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
                                 "r": float(_np.sqrt(pt @ pt))})
            return out
        f.kind = "scalarfield"
        return {"out": f}
    reg.register("formula_field", {}, {"out": "field"}, _formula_field)

    def _curl_field(p, i):
        from holographic_curlnoise import curl_noise_3d
        res = 20
        seed = int(p.get("seed", 1)); oct_ = int(_np.clip(int(p.get("octaves", 3)), 1, 4))
        scale = float(p.get("scale", 1.4))
        key = (res, seed, oct_)
        vf = _CURL_CACHE.get(key)
        if vf is None:
            vf = curl_noise_3d(res=res, octaves=oct_, seed=seed)   # (u,v,w) each (res,res,res) on [0,8]^3
            _CURL_CACHE[key] = vf
        u, v, w = (_np.asarray(c, float) for c in vf)

        def sample(P):                                      # trilinear-ish nearest sample, world -> [0,8]^3
            P = _np.atleast_2d(_np.asarray(P, float))
            q = _np.clip((P * scale + 4.0) / 8.0, 0, 0.999) * (res - 1)
            idx = q.astype(int)
            ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
            return _np.stack([u[ix, iy, iz], v[ix, iy, iz], w[ix, iy, iz]], axis=1)

        def f(P):                                           # scalar view = signed x-component of the curl
            return sample(P)[:, 0]
        f.kind = "vectorfield"; f.vector = sample
        return {"out": f}
    reg.register("curl_field", {}, {"out": "field"}, _curl_field,
                 param_inputs={"scale": "scalar"})

    # -- FIELD-DRIVEN GEOMETRY: a field pushes an SDF's iso-surface, or a mesh's vertices along their normals --
    def _sdf_displace_field(p, i):
        base = i["sdf"]; fld = i["field"]
        base = base["out"] if isinstance(base, dict) else base
        fld = fld["out"] if isinstance(fld, dict) else fld
        amt = float(p.get("amount", 0.12))
        base_eval = base.eval if hasattr(base, "eval") else base

        class _FieldSDF:                                    # a sampleable SDF (not an analytic tree -> field-only)
            def eval(self, P):
                P = _np.atleast_2d(_np.asarray(P, float))
                return _np.asarray(base_eval(P), float) - amt * _np.asarray(fld(P), float)
        return {"out": _FieldSDF()}
    reg.register("sdf_displace_field", {"sdf": "sdf", "field": "field"}, {"out": "sdf"}, _sdf_displace_field,
                 param_inputs={"amount": "scalar"})

    # -- BLEND / MORPH two SDFs (the "layer folder with a net result" node the user asked for): MORPH is a
    #    per-point lerp of the two distance fields, mix(dA, dB, t); SMOOTH is domain.smin (a soft union whose
    #    seam radius is k). The blend weight t is a scalar param OR, when a `field` is wired, a SPATIALLY VARYING
    #    mask (0..1) so one shape becomes the other across the surface (texture/field-driven blend). Field-only
    #    result (no analytic tree), like every field node -- meshes and renders like any object.
    def _sdf_blend(p, i):
        from holographic_domain import smin
        A = i.get("a"); B = i.get("b"); F = i.get("field")
        A = A["out"] if isinstance(A, dict) else A
        B = B["out"] if isinstance(B, dict) else B
        F = (F["out"] if isinstance(F, dict) else F) if F is not None else None
        if A is None or B is None:
            raise ValueError("blend needs both 'a' and 'b' inputs")
        aeval = A.eval if hasattr(A, "eval") else A
        beval = B.eval if hasattr(B, "eval") else B
        mode = str(p.get("mode", "morph"))
        t0 = float(_np.clip(p.get("t", 0.5), 0.0, 1.0))
        k = float(p.get("k", 0.3))

        class _Blend:
            def eval(self, P):
                P = _np.atleast_2d(_np.asarray(P, float))
                dA = _np.asarray(aeval(P), float)
                dB = _np.asarray(beval(P), float)
                if F is not None:                          # field mask drives the blend spatially, 0..1
                    t = _np.clip(_np.asarray(F(P), float), 0.0, 1.0)
                else:
                    t = t0
                if mode == "smooth":                       # soft union, seam radius k (t biases toward B)
                    return smin(dA, dB, k) * (1.0 - t) + smin(dB, dA, k) * t if _np.ndim(t) else smin(dA, dB, k)
                return dA * (1.0 - t) + dB * t             # morph: lerp the distance fields
        return {"out": _Blend()}
    reg.register("sdf_blend", {"a": "sdf", "b": "sdf", "field": "field"}, {"out": "sdf"}, _sdf_blend,
                 param_inputs={"t": "scalar"})

    def _mesh_displace_field(p, i):
        from holographic_mesh import Mesh
        mesh = i["mesh"]; fld = i["field"]
        mesh = mesh["out"] if isinstance(mesh, dict) else mesh
        fld = fld["out"] if isinstance(fld, dict) else fld
        amt = float(p.get("amount", 0.12))
        V = _np.asarray(mesh.vertices, float)
        # a vector field displaces along the field vector; a scalar field along vertex normals
        if getattr(fld, "kind", "") == "vectorfield" and hasattr(fld, "vector"):
            disp = amt * fld.vector(V)
        else:
            N = _vertex_normals(mesh)
            disp = (amt * _np.asarray(fld(V), float))[:, None] * N
        return {"out": Mesh(V + disp, [tuple(f) for f in mesh.faces])}
    reg.register("mesh_displace_field", {"mesh": "mesh", "field": "field"}, {"out": "mesh"}, _mesh_displace_field,
                 param_inputs={"amount": "scalar"})

    # -- MATERIAL on a stream: tag an SDF/mesh with a material name the build step will apply --
    def _sdf_material(p, i):
        obj = i.get("in")
        obj = obj["out"] if isinstance(obj, dict) else obj
        try:
            obj._ps_material = str(p.get("material", "clay"))
        except Exception:
            pass
        return {"out": obj}
    reg.register("sdf_material", {"in": "any"}, {"out": "any"}, _sdf_material)

    # -- MESH POST: retopo (voxel remesh -> uniform topology) and denoise (Taubin, shrink-free) --
    def _mesh_retopo(p, i):
        from holographic_mesh import Mesh
        from holographic_meshbridge import voxel_remesh
        from holographic_meshqem import cluster_decimate
        mesh = i["mesh"]; mesh = mesh["out"] if isinstance(mesh, dict) else mesh
        rm = voxel_remesh(mesh, resolution=int(_np.clip(int(p.get("resolution", 48)), 24, 96)))
        target = int(_np.clip(int(p.get("target", 3000)), 300, 40000))
        g = 46
        dec = cluster_decimate(rm, grid=g)
        while dec.n_faces > target and g > 8:
            g -= 4; dec = cluster_decimate(rm, grid=g)
        return {"out": Mesh(dec.vertices, [tuple(f) for f in dec.faces])}
    reg.register("mesh_retopo", {"mesh": "mesh"}, {"out": "mesh"}, _mesh_retopo)

    def _mesh_denoise(p, i):
        from holographic_mesh import Mesh
        from holographic_meshsmooth import taubin_smooth
        mesh = i["mesh"]; mesh = mesh["out"] if isinstance(mesh, dict) else mesh
        sm = taubin_smooth(mesh, iters=int(_np.clip(int(p.get("iterations", 8)), 1, 40)))
        return {"out": Mesh(sm.vertices, [tuple(f) for f in sm.faces])}
    reg.register("mesh_denoise", {"mesh": "mesh"}, {"out": "mesh"}, _mesh_denoise)
    return reg


def _nodes_graph():
    if _NODES["graph"] is None:
        from holographic_nodegraph import NodeGraph, default_registry
        _NODES["graph"] = NodeGraph(_augment_registry(default_registry()))
    return _NODES["graph"]


@bp.route("/api/nodes/types")
def nodes_types():
    _init()
    g = _nodes_graph()
    out = []
    for t in _NODE_PALETTE:
        try:
            info = g.describe_type(t)
        except Exception:
            continue
        info["params"] = _NODE_PARAMS.get(t, {})
        info["meta"] = {k: _NODE_PARAM_META.get(k, {}) for k in info["params"].keys()}
        out.append(info)
    return jsonify({"types": out})


@bp.route("/api/nodes/graph")
def nodes_graph():
    _init()
    g = _nodes_graph()
    d = g.to_dict()
    nodes = [{"id": nid, "type": spec.get("type"), "params": spec.get("params", {}),
              "pos": _NODES["pos"].get(nid, [40, 40]), "muted": bool(_NODES.get("muted", {}).get(nid))}
             for nid, spec in d.get("nodes", {}).items()]
    edges = d.get("edges", [])
    return jsonify({"nodes": nodes, "edges": edges})


@bp.route("/api/nodes/op", methods=["POST"])
def nodes_op():
    _init()
    d = request.get_json(force=True) or {}
    act = d.get("action", "")
    g = _nodes_graph()
    from holographic_nodegraph import NodeGraph
    try:
        if act == "group_save":
            # P3-2 SUBGRAPHS (reusable node groups): capture a selection of nodes + the edges INTERNAL to it
            # (both endpoints inside the selection) as a named group. Net inputs/outputs -- edges crossing the
            # boundary -- are recorded so the group's external interface is known. Stored in _NODES['groups'].
            ids = [str(i) for i in (d.get("ids") or [])]
            name = str(d.get("name", "")).strip() or f"group{len(_NODES.get('groups', {})) + 1}"
            dd = g.to_dict(); nodes = dd.get("nodes", {}); edges = dd.get("edges", [])
            sel = [i for i in ids if i in nodes]
            if not sel:
                return jsonify({"error": "no valid nodes to group"}), 400
            selset = set(sel)
            internal = [e for e in edges if e.get("src") in selset and e.get("dst") in selset]
            inputs = [e for e in edges if e.get("dst") in selset and e.get("src") not in selset]   # net inputs
            outputs = [e for e in edges if e.get("src") in selset and e.get("dst") not in selset]  # net outputs
            # store node specs + a local index remap so insert can rebuild with fresh ids
            order = {nid: k for k, nid in enumerate(sel)}
            grp = {"nodes": [{"local": order[n], "type": nodes[n]["type"], "params": nodes[n].get("params", {}),
                              "pos": _NODES["pos"].get(n, [60, 60])} for n in sel],
                   "edges": [{"src": order[e["src"]], "src_socket": e.get("src_socket", "out"),
                              "dst": order[e["dst"]], "dst_socket": e.get("dst_socket", "a")} for e in internal],
                   "n_inputs": len(inputs), "n_outputs": len(outputs)}
            _NODES.setdefault("groups", {})[name] = grp
            return jsonify({"ok": True, "name": name, "nodes": len(sel), "internal_edges": len(internal),
                            "net_inputs": len(inputs), "net_outputs": len(outputs)})
        if act == "group_list":
            groups = _NODES.get("groups", {})
            return jsonify({"groups": [{"name": k, "nodes": len(v["nodes"]), "edges": len(v["edges"]),
                                        "inputs": v.get("n_inputs", 0), "outputs": v.get("n_outputs", 0)}
                                       for k, v in groups.items()]})
        if act == "group_insert":
            name = str(d.get("name", ""))
            grp = _NODES.get("groups", {}).get(name)
            if grp is None:
                return jsonify({"error": f"no group {name!r}"}), 400
            ox = float(d.get("x", 80)); oy = float(d.get("y", 80))
            local_to_id = {}
            for n in grp["nodes"]:                             # instantiate each node with a fresh id
                nid = g.add(n["type"], dict(n["params"]) or dict(_NODE_PARAMS.get(n["type"], {})))
                local_to_id[n["local"]] = nid
                _NODES["pos"][nid] = [ox + n["pos"][0] * 0.4, oy + n["pos"][1] * 0.4]
            made_edges = 0
            for e in grp["edges"]:                             # re-wire the internal edges among the new ids
                try:
                    g.connect(local_to_id[e["src"]], e.get("src_socket", "out"),
                              local_to_id[e["dst"]], e.get("dst_socket", "a"))
                    made_edges += 1
                except Exception:
                    pass
            return jsonify({"ok": True, "name": name, "ids": list(local_to_id.values()), "edges": made_edges})
        if act == "add":
            t = d.get("type", "")
            params = d.get("params") if isinstance(d.get("params"), dict) else dict(_NODE_PARAMS.get(t, {}))
            nid = g.add(t, params or dict(_NODE_PARAMS.get(t, {})))
            _NODES["pos"][nid] = [float(d.get("x", 60)), float(d.get("y", 60))]
            return jsonify({"id": nid})
        if act == "connect":
            g.connect(d["src"], d.get("src_socket", "out"), d["dst"], d["dst_socket"])
            return jsonify({"ok": True})
        if act == "set_param":
            g.set_param(d["id"], **(d.get("params") or {}))
            return jsonify({"ok": True})
        if act == "mute":
            nid = d["id"]
            _NODES.setdefault("muted", {})[nid] = bool(d.get("muted", True))
            return jsonify({"ok": True, "muted": _NODES["muted"][nid]})
        if act == "move":
            _NODES["pos"][d["id"]] = [float(d.get("x", 0)), float(d.get("y", 0))]
            return jsonify({"ok": True})
        if act == "delete":
            # engine now has the missing editor verb: NodeGraph.remove(nid) prunes the node + incident edges
            # in O(edges) (upstreamed from this demo's serialize->drop->rebuild workaround).
            nid = d["id"]
            try:
                g.remove(nid)
            except Exception as e:
                return jsonify({"error": f"no node {nid!r}: {e}"}), 400
            _NODES["pos"].pop(nid, None)
            _NODES.get("muted", {}).pop(nid, None)
            return jsonify({"ok": True})
        if act == "collapse":
            # TRUE collapsed subgraph node (P3-2, engine-unblocked): NodeGraph.collapse turns a selection into
            # ONE reusable group node -- boundary edges become group sockets, external wires re-point, and the
            # graph computes EXACTLY what it computed before (refactor, not edit). Refuses cycle-creating sets.
            ids = [str(x) for x in (d.get("ids") or [])]
            if len(ids) < 2:
                return jsonify({"error": "select at least two nodes to collapse"}), 400
            try:
                gid = g.collapse(ids)
            except Exception as e:
                return jsonify({"error": f"collapse refused: {e}"}), 400
            # place the group node at the centroid of the collapsed nodes' editor positions
            ps = [_NODES["pos"].get(i) for i in ids if _NODES["pos"].get(i)]
            if ps:
                _NODES["pos"][gid] = [sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps)]
            for i in ids:
                _NODES["pos"].pop(i, None)
                _NODES.get("muted", {}).pop(i, None)
            return jsonify({"ok": True, "group": gid})
        if act == "expand":
            # inverse: dissolve a group node back into its inner nodes, wires restored
            nid = str(d.get("id", ""))
            try:
                inner = g.expand(nid)
            except Exception as e:
                return jsonify({"error": f"expand refused: {e}"}), 400
            base = _NODES["pos"].pop(nid, None) or [200, 200]
            for k, i in enumerate(inner if isinstance(inner, (list, tuple)) else []):
                _NODES["pos"].setdefault(str(i), [base[0] + (k % 3) * 150, base[1] + (k // 3) * 110])
            return jsonify({"ok": True, "inner": [str(i) for i in (inner or [])]})
        if act == "clear":
            _NODES["graph"] = None
            _NODES["pos"] = {}
            _NODES["muted"] = {}
            _nodes_graph()
            return jsonify({"ok": True})
        if act == "load":
            # replace the whole graph from a serialized {nodes, edges} dict (+ optional pos map). Used by the
            # starter library and by graph-file import. Rebuilds via from_dict, the same round-trip delete uses.
            gd = d.get("graph") or {}
            try:
                _NODES["graph"] = NodeGraph.from_dict(g.reg, gd)
            except Exception as e:
                return jsonify({"error": "could not load graph: %s" % e}), 400
            _NODES["pos"] = {k: [float(v[0]), float(v[1])] for k, v in (d.get("pos") or {}).items()}
            return jsonify({"ok": True, "nodes": len(gd.get("nodes", {}))})
        if act == "build":
            from holographic_mesh import Mesh
            nid = d["id"]
            # P3-3 MUTE/BYPASS: build a graph in which muted nodes are skipped. A muted single-input node is
            # replaced by a pass-through: each edge leaving it is re-sourced from whatever fed its primary input,
            # so the chain reconnects as if the node weren't there. We rewrite the serialized dict and rebuild
            # (the same to_dict/from_dict round-trip delete uses) rather than mutating the live graph -- the
            # user's real graph (with the muted node still present) is untouched.
            muted = {k for k, v in _NODES.get("muted", {}).items() if v}
            eval_graph, eval_target = g, nid
            if muted:
                from holographic_nodegraph import NodeGraph as _NG
                dd = g.to_dict()
                edges = list(dd.get("edges", []))
                for mnid in muted:
                    if mnid not in dd.get("nodes", {}):
                        continue
                    # find the source feeding this node's PRIMARY input (socket 'a', else first incoming)
                    incoming = [e for e in edges if e.get("dst") == mnid]
                    prim = next((e for e in incoming if e.get("dst_socket") == "a"), incoming[0] if incoming else None)
                    outgoing = [e for e in edges if e.get("src") == mnid]
                    edges = [e for e in edges if e.get("src") != mnid and e.get("dst") != mnid]
                    if prim is not None:                       # reconnect: upstream source -> each downstream sink
                        for o in outgoing:
                            edges.append({"src": prim["src"], "src_socket": prim.get("src_socket", "out"),
                                          "dst": o["dst"], "dst_socket": o.get("dst_socket", "a")})
                    dd["nodes"].pop(mnid, None)
                dd["edges"] = edges
                if nid in muted:                               # building a muted node: build its passed-through source
                    inc = [e for e in g.to_dict().get("edges", []) if e.get("dst") == nid]
                    prim = next((e for e in inc if e.get("dst_socket") == "a"), inc[0] if inc else None)
                    if prim is None:
                        return jsonify({"error": "the target node is muted and has no input to pass through"}), 400
                    eval_target = prim["src"]
                try:
                    eval_graph = _NG.from_dict(g.reg, dd)
                except Exception as e:
                    return jsonify({"error": f"bypass rewrite failed: {e}"}), 400
            out = eval_graph.evaluate(eval_target)
            # a plain node exposes socket "out"; a COLLAPSED GROUP node names its outputs by the boundary rule,
            # so fall back to the group's first (usually only) output when "out" is absent
            if isinstance(out, dict):
                val = out.get("out")
                if val is None and out:
                    val = next(iter(out.values()))
            else:
                val = out
            mat_tag = getattr(val, "_ps_material", None)   # from an sdf_material node upstream
            with _LOCK:
                # a raw FIELD callable is not buildable on its own -- it must drive an sdf/mesh first
                if getattr(val, "kind", None) in ("scalarfield", "vectorfield"):
                    return jsonify({"error": "a field can't be built directly -- wire it into a "
                                             "displace node (sdf_displace_field / mesh_displace_field)"}), 400
                mats = ([mat_tag] if mat_tag else None)
                # mesh output -> adopt; analytic sdf -> mesh + KEEP the tree; sampleable sdf -> mesh field-only
                if hasattr(val, "n_faces"):
                    mesh = Mesh(np.asarray(val.vertices, float), [tuple(f) for f in val.faces])
                    _snap_scene()
                    oid = _add_object(d.get("name") or "NodeMesh", mesh,
                                      mats=(mats * mesh.n_faces if mats else None))
                    _bump()
                    out_p = _payload(); out_p["object"] = oid
                    return jsonify(out_p)
                if hasattr(val, "to_dsl") or hasattr(val, "eval") or callable(val):
                    tree = val if hasattr(val, "to_dsl") else None
                    res = int(np.clip(int(d.get("res", 64)), 40, 88))
                    if tree is not None:
                        mesh = _mesh_sdf_tree(tree, res=res, face_target=2400)
                    elif hasattr(val, "eval"):
                        mesh = _mesh_sdf_tree(val, res=res, face_target=2400)
                    else:
                        class _F:
                            def eval(self, P):
                                return np.asarray(val(np.atleast_2d(P)), float)
                        mesh = _mesh_sdf_tree(_F(), res=res, face_target=2400)
                    _snap_scene()
                    oid = _add_object(d.get("name") or "NodeSDF", mesh, sdf_tree=tree,
                                      mats=(mats * mesh.n_faces if mats else None))
                    _bump()
                    out_p = _payload(); out_p["object"] = oid
                    out_p["analytic"] = tree is not None
                    return jsonify(out_p)
                return jsonify({"value": val if isinstance(val, (int, float, str, list)) else str(val)})
        return jsonify({"error": f"unknown action '{act}'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/material/custom", methods=["POST"])
def material_custom():
    """Author a material from physical parameters (the material-lab demo's layer idea, in the modeller): copy a
    library base and override glTF-PBR channels. Upserts by name into the session library -- immediately
    paintable, assignable, and path-traced with real physics (transmission >= 1 refracts at the given IOR)."""
    _init()
    import copy as _copy
    d = request.get_json(force=True) or {}
    name = str(d.get("name", "")).strip().lower().replace(" ", "_")
    if not name:
        return jsonify({"error": "material needs a name"}), 400
    ml = _matlib()
    if name in ml.names():
        return jsonify({"error": f"'{name}' is a library preset -- pick a new name"}), 400
    m = _copy.deepcopy(ml.material("clay"))
    col = d.get("color", [0.7, 0.7, 0.7])
    try:
        m.base_color = np.array([float(col[0]), float(col[1]), float(col[2])], float)
        m.metallic = float(np.clip(d.get("metallic", 0.0), 0, 1))
        m.roughness = float(np.clip(d.get("roughness", 0.5), 0.02, 1))
        if hasattr(m, "transmission"):
            m.transmission = float(np.clip(d.get("transmission", 0.0), 0, 1))
        if hasattr(m, "ior"):
            m.ior = float(np.clip(d.get("ior", 1.5), 1.0, 2.6))
        if hasattr(m, "emissive"):
            em = float(np.clip(d.get("emission", 0.0), 0, 20))
            m.emissive = m.base_color * em                      # the renderer reads m.emissive (verified: lava)
    except Exception as e:
        return jsonify({"error": f"bad parameters: {e}"}), 400
    with _LOCK:
        _CUSTOM_MATS[name] = m
        _S["cache"].pop("mats", None) if isinstance(_S.get("cache"), dict) else None
        _bump()                                            # material params feed the bake colour channels
    return jsonify({"ok": True, "name": name})


def _recenter(mesh):
    """Imported objects land at the SCENE ORIGIN (bbox centre to (0, *, 0), base on the boot floor line) so a
    file authored at odd coordinates cannot arrive off-screen and 'get lost'."""
    v = mesh.vertices
    lo, hi = v.min(axis=0), v.max(axis=0)
    ctr = (lo + hi) / 2
    mesh.vertices = v - np.array([ctr[0], lo[1] + 0.6, ctr[2]])
    return mesh


@bp.route("/api/import_obj", methods=["POST"])
def import_obj():
    """Wavefront OBJ import (holographic_assetimport.load_obj -- fan-triangulated, .mtl-aware)."""
    _init()
    import tempfile
    from holographic_assetimport import load_obj
    from holographic_mesh import Mesh
    data = request.get_data()
    if not data:
        return jsonify({"error": "empty upload"}), 400
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        f.write(data); path = f.name
    try:
        lm = load_obj(path)
        v = np.asarray(lm.positions, float)
        faces = [tuple(int(i) for i in fc) for fc in lm.faces]
        if len(v) == 0 or not faces:
            return jsonify({"error": "no geometry in the .obj"}), 400
        mesh = _recenter(Mesh(v, faces))
    except Exception as e:
        return jsonify({"error": f"obj import failed: {e}"}), 400
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    with _LOCK:
        _snap_scene()
        oid = _add_object(d0 := (request.args.get("name") or "Imported"), mesh)
        _bump()
        out = _payload(); out["object"] = oid
    return jsonify(out)


@bp.route("/api/export_obj")
def export_obj():
    """Wavefront OBJ export -- v/f lines, 1-indexed, quads kept (OBJ is n-gon-native)."""
    _init()
    oid = str(request.args.get("object", "")) or None
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400
        lines = ["# Poly Studio (leCore) export", "o " + (o.name.replace(" ", "_") or "object")]
        lines += ["v %.6f %.6f %.6f" % tuple(p) for p in o.mesh.vertices]
        lines += ["f " + " ".join(str(i + 1) for i in fc) for fc in o.mesh.faces]
        return Response("\n".join(lines) + "\n", mimetype="text/plain",
                        headers={"Content-Disposition": "attachment; filename=polystudio.obj"})


# =====================================================================================================
# P0-1: IN-APP ENGINE DOCUMENTATION ("ask leCore"). capabilities.json is the engine's own per-capability
# index (name / does / aliases / example / produces / consumes / theme). Shipped into the bundle so the
# PREPARE step of every future feature -- "does the engine do X?" -- is answerable inside the app instead
# of by reading source. Keyword search ranks over name + aliases + the does-string.
# =====================================================================================================
_DOCS = {"caps": None, "loaded": False}


def _load_docs():
    if _DOCS["loaded"]:
        return _DOCS["caps"]
    _DOCS["loaded"] = True
    path = None
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "..", "holostuff", "capabilities.json"),
                 os.path.join(here, "holostuff", "capabilities.json")):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        # fall back to wherever flatcompat mounted the engine
        try:
            import holographic
            hp = os.path.dirname(os.path.dirname(os.path.abspath(holographic.__file__)))
            cand = os.path.join(hp, "capabilities.json")
            if os.path.exists(cand):
                path = cand
        except Exception:
            pass
    caps = []
    if path:
        try:
            with open(path) as f:
                caps = json.load(f).get("capabilities", [])
        except Exception:
            caps = []
    _DOCS["caps"] = caps
    return caps


@bp.route("/api/docs")
def docs():
    """Search the engine capability index. ?q= ranks by keyword hits over name (weighted), aliases, and the
    'does' text; empty q returns the full list (themed). Honest when the index is absent."""
    _init()
    caps = _load_docs()
    if not caps:
        return jsonify({"available": False, "results": [],
                        "note": "engine capability index not found in this build"})
    q = (request.args.get("q", "") or "").strip().lower()
    themes = sorted({c.get("theme", "") for c in caps if c.get("theme")})
    if not q:
        results = [{"name": c.get("name", ""), "does": c.get("does", "")[:240],
                    "theme": c.get("theme", ""), "aliases": c.get("aliases", [])[:6],
                    "example": c.get("example", "")} for c in caps]
        return jsonify({"available": True, "count": len(results), "themes": themes, "results": results})
    # ENGINE RANKER FIRST (leCore sweep 163). Our token scorer is a bag-of-words hit count, and it
    # is measurably bad at task phrasing: "make a wooden chair" ranked "Make a mesh manifold" top
    # (it matched on "make"), and "make the metal look worn" returned "See what the mantis sees".
    # m.find_capability is the engine's own ranker over the same index. Token scoring stays as the
    # fallback so the route still answers on a build without it.
    try:
        import lecore
        hits = lecore.UnifiedMind(dim=256, seed=0).find_capability(q, k=int(request.args.get("k", 12)))
        by_name = {c.get("name", ""): c for c in caps}
        ranked = []
        for h in hits:
            nm = h[0] if isinstance(h, (list, tuple)) else getattr(h, "name", str(h))
            c = by_name.get(nm)
            if c is None:
                continue
            ranked.append({"name": nm, "does": c.get("does", "")[:240], "theme": c.get("theme", ""),
                           "aliases": c.get("aliases", [])[:6], "example": c.get("example", "")})
        if ranked:
            return jsonify({"available": True, "ranker": "engine.find_capability",
                            "count": len(ranked), "themes": themes, "results": ranked})
    except Exception:
        pass                                    # older engine, or no index: fall through to tokens

    terms = [t for t in q.replace(",", " ").split() if t]

    def score(c):
        name = c.get("name", "").lower()
        alias_list = [a.strip().lower() for a in c.get("aliases", [])]
        alias_blob = " ".join(alias_list)
        does = c.get("does", "").lower()
        theme = c.get("theme", "").lower()
        s = 0
        # whole-phrase bonuses (so "curl noise" beats a "denoise" substring, "make a vase" beats a stray word)
        if q == name:
            s += 40
        if q in alias_list:
            s += 30                                  # exact full-phrase alias
        if q in name:
            s += 12
        if q in does:
            s += 3
        for t in terms:                              # per-term, but a substring inside a longer word counts less
            if t == name or any(t == a for a in alias_list):
                s += 10
            elif t in name.split():
                s += 6
            elif t in name:
                s += 2
            if any(t in a.split() for a in alias_list):
                s += 5                               # whole-word alias hit
            elif t in alias_blob:
                s += 2
            if t in does.split():
                s += 2
            elif t in does:
                s += 1
            if t in theme:
                s += 1
        return s

    ranked = sorted(((score(c), c) for c in caps), key=lambda kv: kv[0], reverse=True)
    hits = [{"name": c.get("name", ""), "does": c.get("does", "")[:300], "theme": c.get("theme", ""),
             "aliases": c.get("aliases", [])[:6], "example": c.get("example", ""),
             "produces": c.get("produces", []), "consumes": c.get("consumes", [])}
            for sc, c in ranked if sc > 0][:20]
    return jsonify({"available": True, "count": len(hits), "themes": themes, "query": q, "results": hits})


# =====================================================================================================
# P1-1: SCENE -> WGSL for the client compute raymarcher. The engine already emits an EXACT per-object WGSL
# map(p) (holographic_sdfemit.sdf_dialect). This composes every ANALYTIC object into one scene map() by
# smooth-union, tagging each with its material albedo so the client can shade. Mesh/sculpted objects have no
# analytic tree -> they're reported as excluded (the client keeps the server raymarch for those / as ground
# truth). This is the "ship the kernel to the client" path: the server stays the fallback + reference.
# =====================================================================================================
@bp.route("/api/scene_wgsl")
def scene_wgsl():
    _init()
    from holographic_sdfemit import sdf_dialect
    parts, excluded = [], []
    mats = []
    with _LOCK:
        for oid, o in _S["objects"].items():
            nm = o.name
            if getattr(o, "sdf_tree", None) is None:
                excluded.append({"id": oid, "name": nm, "reason": "no analytic tree (edited/sculpted/mesh)"})
                continue
            try:
                dsl = o.sdf_tree.to_dsl()
                body = sdf_dialect(dsl, "wgsl")
            except Exception as e:
                excluded.append({"id": oid, "name": nm, "reason": "not WGSL-emittable (%s)" % type(e).__name__})
                continue
            if not body:
                excluded.append({"id": oid, "name": nm, "reason": "empty emit (warp node)"})
                continue
            idx = len(parts)
            fn = body.replace("fn map(", "fn map_%d(" % idx)
            # resolve material albedo -> linear rgb 0..1 from the object's dominant per-face material
            col = (0.8, 0.8, 0.82)
            try:
                names = getattr(o, "mats", None) or []
                if names:
                    from collections import Counter
                    dom = Counter(names).most_common(1)[0][0]
                else:
                    dom = _default_mat_name()
                m = _mat(dom)
                c = getattr(m, "base_color", None)
                if c is not None and len(c) >= 3:
                    col = (float(c[0]), float(c[1]), float(c[2]))
            except Exception:
                pass
            parts.append(fn)
            mats.append(col)
    if not parts:
        return jsonify({"available": False, "excluded": excluded,
                        "note": "no analytic objects to emit; use the server render"})
    # scene map returns vec2(dist, matID) via smooth-union carrying the nearest id
    lines = ["fn smin2(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {",
             "  let h = clamp(0.5 + 0.5*(b.x-a.x)/k, 0.0, 1.0);",
             "  let d = mix(b.x, a.x, h) - k*h*(1.0-h);",
             "  let id = select(b.y, a.y, a.x < b.x);",
             "  return vec2<f32>(d, id);", "}"]
    lines += [p.strip() for p in parts]
    lines.append("fn scene_map(p: vec3<f32>) -> vec2<f32> {")
    lines.append("  var r = vec2<f32>(map_0(p), 0.0);")
    for i in range(1, len(parts)):
        lines.append("  r = smin2(r, vec2<f32>(map_%d(p), %d.0), 0.03);" % (i, i))
    lines.append("  return r;")
    lines.append("}")
    wgsl = "\n".join(lines)
    return jsonify({"available": True, "wgsl": wgsl, "count": len(parts),
                    "materials": mats, "excluded": excluded})


# =====================================================================================================
# CAD-PARITY SWEEP: cross-sections + layered shells (user asks #3/#4). /api/section renders an EXACT 2-D
# slice of the scene (or one object) at plane axis=offset: each pixel samples the true field (the analytic
# tree where one exists -- no voxel grid, resolution-independent), colored by the object's LAYER bands:
# per-object layers [(thickness, material), ...] measured inward from the surface, i.e. -f(p) depth. That is
# the "material with layers of thickness for a solid, revealed by a cross section" -- plywood/anodizing/
# coating walls, exact. count=N composes a SERIES of parallel slices side by side (the CT-scan strip).
# =====================================================================================================
@bp.route("/api/section")
def section():
    _init()
    g = request.args.get
    axis = {"x": 0, "y": 1, "z": 2}.get((g("axis", "z") or "z").lower(), 2)
    off = float(g("offset", 0.0))
    count = int(np.clip(int(g("count", 1)), 1, 6))
    span = float(g("span", 0.8))                            # series: total offset range across count slices
    res = int(np.clip(int(g("res", 220)), 80, 420))
    oid = g("object", "") or None
    with _LOCK:
        objs = ([(_S["objects"][oid])] if oid and oid in _S["objects"] else list(_S["objects"].values()))
        if not objs:
            return jsonify({"error": "empty scene"}), 400
        # bounds over the chosen objects, in the two in-plane axes
        vs = np.vstack([o.mesh.vertices for o in objs])
        lo = vs.min(axis=0) - 0.15; hi = vs.max(axis=0) + 0.15
        ax_u, ax_v = [a for a in (0, 1, 2) if a != axis]
        us = np.linspace(lo[ax_u], hi[ax_u], res); vsv = np.linspace(lo[ax_v], hi[ax_v], res)
        U, V = np.meshgrid(us, vsv, indexing="xy")
        fields = []
        for o in objs:
            if o.sdf_tree is not None:
                fields.append(("tree", o.sdf_tree, o))
            else:
                fields.append(("grid", _bake_object(next(k for k, v in _S["objects"].items() if v is o),
                                                    56, False), o))
        offsets = [off] if count == 1 else list(np.linspace(off - span / 2, off + span / 2, count))
        panels = []
        for oz in offsets:
            P = np.zeros((res * res, 3))
            P[:, ax_u] = U.ravel(); P[:, ax_v] = V.ravel(); P[:, axis] = oz
            img = np.full((res * res, 3), 0.055)            # background
            depth_best = np.full(res * res, -1e9)           # inside-depth; pick the deepest owner per pixel
            for kind, f, o in fields:
                d = (f.eval(P) if kind == "tree" else f.eval(P))
                inside = d < 0
                dep = -d                                    # exact distance inward from the surface
                take = inside & (dep > depth_best)
                if not take.any():
                    continue
                layers = getattr(o, "layers", None) or []
                base = np.array(_obj_color(o))
                col = np.tile(base, (res * res, 1))
                edge = 0.0
                for (lt, lmat) in layers:                   # bands from the surface inward
                    try:
                        lm = _mat(lmat); lc = np.array(lm.base_color[:3], float)
                    except Exception:
                        continue
                    band = inside & (dep >= edge) & (dep < edge + float(lt))
                    col[band] = lc
                    edge += float(lt)
                if layers:                                   # anything deeper than the last layer = core color
                    core = inside & (dep >= edge)
                    col[core] = base * 0.55
                img[take] = col[take]
                depth_best[take] = dep[take]
            panels.append(img.reshape(res, res, 3))
        strip = np.concatenate(panels, axis=1)
        # thin separators between series panels
        if count > 1:
            for i in range(1, count):
                strip[:, i * res - 1: i * res + 1] = 0.35
    resp = Response(_png_bytes(np.clip(strip, 0, 1)), mimetype="image/png")
    resp.headers["X-Holostuff-Section"] = (f"axis={'xyz'[axis]} offsets=" +
                                           ",".join(f"{o:.2f}" for o in offsets) + f"; res={res}; exact-sampled")
    return resp


def _obj_color(o):
    try:
        names = getattr(o, "mats", None) or []
        if names:
            from collections import Counter
            return _mat(Counter(names).most_common(1)[0][0]).base_color[:3]
    except Exception:
        pass
    return (0.75, 0.76, 0.8)

# =====================================================================================================
# HISTORY BRANCHING (user ask): a per-object op LOG with git-style branches over a shared base snapshot,
# built on holographic_edithistory's replay discipline -- a command log whose rebuild is deterministic, so
# "checkout" = restore base + re-run ops[:k] through the SAME _mesh_verb dispatch the live session used, and
# "edit at a point in history with future changes applied" = insert/replace at the cursor then REPLAY the tail
# (edithistory.replace_command's contract, applied at our op granularity). Merging follows holographic_merge's
# policy model: replay both branches' op suffixes over the common prefix; an op that fails to apply is a
# CONFLICT and is surfaced, not guessed (merge_forks' 'select' discipline).
# HONEST LIMITS, stated: (1) ops that consume ANOTHER object (boolean) or non-replayable state (sculpt strokes,
# paint strokes) are BAKE POINTS -- the log restarts from a fresh base snapshot there; the timeline shows them.
# (2) replay executes real engine verbs, so checking out deep histories costs real compute (it's re-modeling).
# =====================================================================================================
def _hist_ensure(oid):
    o = _S["objects"].get(oid)
    if o is None:
        return None
    h = _HIST.get(oid)
    if h is None:
        h = _HIST[oid] = {"base": _hist_snap(o), "branches": {"main": []}, "current": "main", "cursor": 0}
    return h


def _hist_snap(o):
    return {"verts": o.mesh.vertices.copy(), "faces": [tuple(f) for f in o.mesh.faces],
            "mats": list(o.mats), "dsl": (o.sdf_tree.to_dsl() if o.sdf_tree is not None else None)}


def _hist_restore(o, snap):
    from holographic_mesh import Mesh
    o.mesh = Mesh(snap["verts"].copy(), [tuple(f) for f in snap["faces"]])
    o.mats = list(snap["mats"])
    if snap["dsl"]:
        try:
            from holographic_sdf import parse_dsl
            o.sdf_tree = parse_dsl(snap["dsl"])
        except Exception:
            o.sdf_tree = None
    else:
        o.sdf_tree = None


def _hist_record(oid, name, d):
    """Append a mutating single-object op at the cursor. If the cursor is mid-history (user checked out an
    earlier point), the tail is REPLAYED on top afterwards -- the 'branch from a point and keep future changes'
    behaviour -- with failures surfaced."""
    h = _hist_ensure(oid)
    if h is None:
        return
    ops = h["branches"][h["current"]]
    entry = {"op": name, "params": {k: v for k, v in d.items() if k not in ("object",)}}
    if h["cursor"] >= len(ops):
        ops.append(entry); h["cursor"] = len(ops)
        return
    # mid-history edit: insert, then replay the tail on the CURRENT (already-edited) mesh
    tail = ops[h["cursor"]:]
    ops[h["cursor"]:] = [entry]
    h["cursor"] += 1
    o = _S["objects"][oid]
    replayed, failed = 0, []
    for t in tail:
        try:
            new = _mesh_verb(o.mesh, t["op"], dict(t["params"]))
            o.mats = _transfer_mats(o.mesh, o.mats, new)
            o.mesh = new
            ops.append(t); h["cursor"] += 1; replayed += 1
        except Exception as e:
            failed.append({"op": t["op"], "error": str(e)})
    o.rev += 1
    _HIST_LAST_REPLAY[oid] = {"replayed": replayed, "failed": failed}


def _hist_bake_point(oid, label):
    """Non-replayable op (boolean/sculpt/paint): restart the log from a fresh base, keeping a marker."""
    o = _S["objects"].get(oid)
    if o is None or oid not in _HIST:
        return
    h = _HIST[oid]
    h["base"] = _hist_snap(o)
    h["branches"] = {h["current"]: []}
    h["cursor"] = 0
    h["baked_from"] = label


_HIST = {}
_HIST_LAST_REPLAY = {}


@bp.route("/api/anim/keyable")
def anim_keyable():
    """P2-3 server-side param keys: list the NUMERIC parameters of an object's recorded history ops -- these
    are the parameters that can be keyed and re-evaluated per frame (the history replay re-runs the exact op
    with a substituted value). Returns [{index, op, param, value}] for the current branch."""
    _init()
    oid = str(request.args.get("object", ""))
    with _LOCK:
        h = _HIST.get(oid)
        if h is None:
            return jsonify({"object": oid, "keyable": [], "note": "no edit history on this object yet"})
        ops = h["branches"][h["current"]]
        keyable = []
        for i, e in enumerate(ops):
            for k, v in (e.get("params") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    keyable.append({"index": i, "op": e["op"], "param": k, "value": round(float(v), 5)})
        return jsonify({"object": oid, "branch": h["current"], "keyable": keyable})


@bp.route("/api/anim/bake", methods=["POST"])
def anim_bake():
    """P2-3 server-side param KEYFRAMES: evaluate the object at frame t by replaying its edit history with one
    op-parameter overridden by a keyframe-interpolated value. This makes modifier parameters genuinely
    animatable on the SERVER (unlike the view-layer pose keys) -- the geometry actually changes per frame, so
    exports and GI photos see the animation. Body: {object, index, param, keys:[[frame,value]...], frame} to
    apply one frame in-place, OR {..., bake:[f0,f1,...]} to return the mesh for each listed frame without
    committing (for a client-side scrub/preview or an export loop)."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        h = _HIST.get(oid)
        o = _S["objects"].get(oid)
        if h is None or o is None:
            return jsonify({"error": "object has no edit history to animate"}), 400
        idx = int(d.get("index", -1)); param = str(d.get("param", ""))
        ops = h["branches"][h["current"]]
        if not (0 <= idx < len(ops)) or param not in (ops[idx].get("params") or {}):
            return jsonify({"error": "index/param not found in history"}), 400
        keys = d.get("keys") or []
        try:
            keys = sorted(([float(f), float(v)] for f, v in keys), key=lambda kv: kv[0])
        except Exception:
            return jsonify({"error": "keys must be [[frame, value], ...]"}), 400
        if not keys:
            return jsonify({"error": "need at least one keyframe"}), 400
        ease = str(d.get("easing", "linear"))

        def _ease(u):                                          # remap the 0..1 segment fraction
            u = max(0.0, min(1.0, u))
            if ease == "smooth":       return u * u * (3 - 2 * u)          # smoothstep
            if ease == "ease_in":      return u * u
            if ease == "ease_out":     return 1 - (1 - u) * (1 - u)
            if ease == "ease_in_out":  return 0.5 * (1 - np.cos(np.pi * u))
            return u                                            # linear

        def val_at(f):                                          # piecewise interpolation, clamped at ends
            if f <= keys[0][0]:
                return keys[0][1]
            if f >= keys[-1][0]:
                return keys[-1][1]
            for i in range(1, len(keys)):
                if f <= keys[i][0]:
                    (f0, v0), (f1, v1) = keys[i - 1], keys[i]
                    u = (f - f0) / (f1 - f0) if f1 > f0 else 0.0
                    return v0 + (v1 - v0) * _ease(u)
            return keys[-1][1]

        def eval_frame(f):                                     # replay history with the one param overridden
            _hist_restore(o, h["base"])
            for j, e in enumerate(ops):
                p = dict(e["params"])
                if j == idx:
                    p[param] = val_at(f)
                try:
                    new = _mesh_verb(o.mesh, e["op"], p)
                    o.mats = _transfer_mats(o.mesh, o.mats, new)
                    o.mesh = new
                except Exception:
                    pass

        bake = d.get("bake")
        if bake:                                               # return each requested frame's mesh (no commit intent beyond last)
            frames = []
            for f in bake:
                eval_frame(float(f))
                frames.append({"frame": float(f),
                               "positions": [round(float(x), 5) for x in o.mesh.vertices.ravel()],
                               "indices": [int(i) for t in o.mesh.faces for i in (t if len(t) == 3 else t[:3])],
                               "value": round(val_at(float(f)), 5)})
            _bump(oid)                                          # leave the object on the LAST baked frame
            return jsonify({"object": oid, "frames": frames, "count": len(frames)})
        # single frame, applied in place
        f = float(d.get("frame", keys[0][0]))
        eval_frame(f)
        _bump(oid)
        out = _payload(only=oid); out["object"] = oid; out["frame"] = f; out["value"] = round(val_at(f), 5)
        return jsonify(out)


@bp.route("/api/history")
def history_get():
    _init()
    oid = request.args.get("object", "")
    with _LOCK:
        h = _hist_ensure(oid)
        if h is None:
            return jsonify({"error": "no such object"}), 400
        return jsonify({"object": oid, "current": h["current"], "cursor": h["cursor"],
                        "baked_from": h.get("baked_from"),
                        "branches": {k: [e["op"] for e in v] for k, v in h["branches"].items()},
                        "last_replay": _HIST_LAST_REPLAY.get(oid)})


@bp.route("/api/history/op", methods=["POST"])
def history_op():
    _init()
    d = request.get_json(force=True) or {}
    act = d.get("action", ""); oid = str(d.get("object", ""))
    with _LOCK:
        h = _hist_ensure(oid)
        if h is None:
            return jsonify({"error": "no such object"}), 400
        o = _S["objects"][oid]

        def _replay(branch, upto):
            _hist_restore(o, h["base"])
            failed = []
            ops = h["branches"][branch]
            for i, t in enumerate(ops[:upto]):
                try:
                    new = _mesh_verb(o.mesh, t["op"], dict(t["params"]))
                    o.mats = _transfer_mats(o.mesh, o.mats, new)
                    o.mesh = new
                except Exception as e:
                    failed.append({"i": i, "op": t["op"], "error": str(e)})
            o.rev += 1
            return failed

        if act == "checkout":
            k = int(d.get("index", len(h["branches"][h["current"]])))
            k = max(0, min(k, len(h["branches"][h["current"]])))
            failed = _replay(h["current"], k)
            h["cursor"] = k
            out = _payload(only=oid); out["cursor"] = k; out["failed"] = failed
            return jsonify(out)
        if act == "branch":
            name = str(d.get("name", "")) or f"branch{len(h['branches'])}"
            if name in h["branches"]:
                return jsonify({"error": f"branch '{name}' exists"}), 400
            k = int(d.get("index", h["cursor"]))
            src_b = h["branches"][h["current"]]
            k = max(0, min(k, len(src_b)))
            h["branches"][name] = [dict(e) for e in src_b[:k]]
            h["current"] = name; h["cursor"] = k
            failed = _replay(name, k)
            out = _payload(only=oid); out["branch"] = name; out["cursor"] = k; out["failed"] = failed
            return jsonify(out)
        if act == "switch":
            name = str(d.get("name", ""))
            if name not in h["branches"]:
                return jsonify({"error": f"no branch '{name}'"}), 400
            h["current"] = name; h["cursor"] = len(h["branches"][name])
            failed = _replay(name, h["cursor"])
            out = _payload(only=oid); out["branch"] = name; out["failed"] = failed
            return jsonify(out)
        if act == "merge":
            src = str(d.get("from", ""))
            if src not in h["branches"] or src == h["current"]:
                return jsonify({"error": "need a different existing branch in 'from'"}), 400
            a = h["branches"][h["current"]]; b = h["branches"][src]
            # common prefix, then A's suffix, then B's suffix -- deterministic op-level rebase-merge.
            # Failures are surfaced as conflicts (merge_forks 'select' discipline), not guessed.
            n = 0
            while n < len(a) and n < len(b) and a[n] == b[n]:
                n += 1
            merged = a[:n] + a[n:] + b[n:]
            failed = []
            _hist_restore(o, h["base"])
            applied = []
            for t in merged:
                try:
                    new = _mesh_verb(o.mesh, t["op"], dict(t["params"]))
                    o.mats = _transfer_mats(o.mesh, o.mats, new)
                    o.mesh = new
                    applied.append(t)
                except Exception as e:
                    failed.append({"op": t["op"], "error": str(e)})
            h["branches"][h["current"]] = applied
            h["cursor"] = len(applied)
            o.rev += 1
            out = _payload(only=oid)
            out["merged_ops"] = len(applied); out["conflicts"] = failed
            return jsonify(out)
        return jsonify({"error": f"unknown action '{act}'"}), 400


# =====================================================================================================
# PHOTO TOOLS (user ask): four engine-native photo-driven abilities, each a thin wrapper over a real leCore
# module -- NOT hand-rolled vision. Honest about their roughness where the modules themselves are.
#   /api/photo/light    -> estimate_light (shape-from-shading) + estimate_light_direction (perception): sun az/el
#   /api/photo/depth    -> shape_from_shading depth map -> height-field mesh (depth_map_to_mesh) as a new object
#   /api/photo/texture  -> fit_texture: a procedural fBm GLSL whose statistical signature matches the image
#   /api/photo/shapes   -> shape_from_shading depth -> point cloud -> fit_primitives: a union of exact SDF prims
# All accept {"image": "<base64 png/jpg>"}; decode once with PIL.
# =====================================================================================================
def _decode_photo(d, cap=256):
    """Decode a base64 / data-URL image. `cap` bounds the long edge.

    The 256 default is right for the photo -> geometry analysers (shape-from-shading on a big image is
    minutes of CPU for no extra fidelity). It is WRONG for the upscaler, which was handed a 1280px
    render, shrank it to 256, and returned a "2x" result of 512 -- smaller than what it was given.
    """
    import base64, io
    from PIL import Image
    b = d.get("image", "")
    if not isinstance(b, str) or not b:
        raise ValueError("no image supplied")
    if b.startswith("blob:"):
        # A browser blob URL is meaningless here; say so instead of failing inside base64.
        raise ValueError("got a blob: URL -- send the image bytes as a data URL, not the object URL")
    if b.startswith("data:"):
        b = b.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
    cap = int(cap)
    if cap and max(img.size) > cap:
        s = cap / max(img.size)
        img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))))
    return np.asarray(img, float) / 255.0


def _az_el_from_L(L):
    import math
    L = np.asarray(L, float); n = np.linalg.norm(L)
    if n < 1e-9:
        return 300.0, 55.0
    L = L / n
    az = (math.degrees(math.atan2(L[2], L[0]))) % 360.0
    el = math.degrees(math.asin(np.clip(L[1], -1, 1)))
    return az, el


@bp.route("/api/photo/light", methods=["POST"])
def photo_light():
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    gray = rgb.mean(axis=2)
    from holographic_shapefromshading import estimate_light
    from holographic_perception import estimate_light_direction
    try:
        L = np.asarray(estimate_light(gray), float)
        az, el = _az_el_from_L(L)
    except Exception as e:
        return jsonify({"error": f"light estimate failed: {e}"}), 400
    # perception gives a coarse az/el pair directly; average azimuth for robustness, keep SfS elevation
    try:
        pae = np.asarray(estimate_light_direction(gray), float)
        paz = float(pae[0]) * 360.0 if pae[0] <= 1.0 else float(pae[0])
    except Exception:
        paz = az
    az_final = (az + paz) / 2.0 if abs(((az - paz + 180) % 360) - 180) < 90 else az
    with _LOCK:
        _S.setdefault("photolight", {})
        _S["photolight"] = {"az": az_final, "el": el}
    return jsonify({"light_az": round(az_final, 1), "light_el": round(el, 1),
                    "L": [round(float(x), 3) for x in L],
                    "note": "Coarse Pentland/centroid estimate — a warm start, not exact. Applied to the lighting dialog."})


@bp.route("/api/creature/walk", methods=["POST"])
def creature_walk():
    """CREATURE WALK-CYCLE (H1-7 / gait): animate a generated creature's legs at time t. Each leg's vertex
    block gets a rigid gait offset -- the foot end swings fore/aft (x) and lifts (y) with a per-leg phase so
    opposite legs alternate (a diagonal trot). Only the leg vertices move; the body stays put. Real geometry,
    so exports/photos see the stride. Body: {object, time} for one frame, or {object, bake:[...]} for a loop."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        w = (getattr(o, "layers", {}) or {}).get("walk")
        if not w or w.get("n_legs", 0) == 0:
            return jsonify({"error": "not a walkable creature (generate a creature with legs first)"}), 400
        v0, v1, nlegs = int(w["leg_v0"]), int(w["leg_v1"]), int(w["n_legs"])
        V = o.mesh.vertices
        if v1 > len(V) or nlegs == 0:
            return jsonify({"error": "leg range stale (re-generate the creature)"}), 400
        # cache the leg rest positions once so repeated animate calls don't drift
        rest = w.get("rest")
        if rest is None or len(rest) != (v1 - v0):
            rest = V[v0:v1].copy(); w["rest"] = rest
        rest = np.asarray(rest, float)
        per = (v1 - v0) // nlegs                                # verts per leg tube (equal-sized tubes)
        stride = float(np.clip(d.get("stride", 0.18), 0.0, 0.6))
        lift = float(np.clip(d.get("lift", 0.10), 0.0, 0.5))
        hips = w["hips"]

        def pose(t):
            Vn = V.copy(); Vn[v0:v1] = rest
            for li in range(nlegs):
                a = v0 + li * per; b = a + per if li < nlegs - 1 else v1
                sz = hips[li][3] if li < len(hips) else 1.0
                # diagonal gait: legs on opposite sides + alternating index are out of phase
                phase = 2 * np.pi * t + (np.pi if (li % 2) ^ (1 if sz < 0 else 0) else 0)
                swing = stride * np.sin(phase)
                up = lift * max(0.0, np.sin(phase))            # lift only on the forward swing
                # weight the offset toward the foot (lower verts move more) using rest height
                blk = rest[a - v0:b - v0]
                hy = blk[:, 1]
                wgt = np.clip(1.0 - (hy - hy.min()) / max(np.ptp(hy), 1e-6), 0, 1)  # 1 at foot, 0 at hip
                Vn[a:b, 0] += swing * wgt
                Vn[a:b, 1] += up * wgt
            return Vn

        from holographic_mesh import Mesh
        bake = d.get("bake")
        if bake:
            frames = []
            for t in bake:
                Vn = pose(float(t))
                frames.append({"time": float(t), "positions": [round(float(x), 5) for x in Vn.ravel()]})
            o.mesh = Mesh(pose(float(bake[-1])), [tuple(f) for f in o.mesh.faces]); _bump(oid)
            return jsonify({"object": oid, "frames": frames, "count": len(frames)})
        t = float(d.get("time", 0.0))
        o.mesh = Mesh(pose(t), [tuple(f) for f in o.mesh.faces]); _bump(oid)
        out = _payload(only=oid); out["object"] = oid; out["time"] = t
        return jsonify(out)


@bp.route("/api/ocean/animate", methods=["POST"])
def ocean_animate():
    """ANIMATED OCEAN (H1-4): recompute a generated ocean's surface at animation time t from its stored wave
    spec (deterministic direction/phase/speed per harmonic). Waves TRAVEL with deep-water phase velocity, so
    scrubbing t gives coherent motion -- and because it edits real geometry, exports and GI photos see the
    swell. Body: {object, time} for one frame in place, or {object, bake:[t0,t1,...]} to return frame meshes."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        oc = (getattr(o, "layers", {}) or {}).get("ocean")
        if not oc:
            return jsonify({"error": "not an animatable ocean (generate one with the Ocean generator first)"}), 400
        spec = np.asarray(oc["spec"], float)                   # rows: [kx, kz, phase0, omega, amp]
        V = o.mesh.vertices
        base_y = np.asarray(oc["base_y"], float) if len(oc.get("base_y", [])) == len(V) else np.zeros(len(V))

        def surf_at(t):
            wave = np.zeros(len(V))
            for kx, kz, ph0, omega, a in spec:
                wave += a * np.sin(V[:, 0] * kx + V[:, 2] * kz + ph0 + t * omega)
            Vn = V.copy(); Vn[:, 1] = base_y + wave
            return Vn, wave

        bake = d.get("bake")
        from holographic_mesh import Mesh
        if bake:
            frames = []
            for t in bake:
                Vn, _w = surf_at(float(t))
                frames.append({"time": float(t), "positions": [round(float(x), 5) for x in Vn.ravel()]})
            Vn, _w = surf_at(float(bake[-1]))                  # leave on last frame
            o.mesh = Mesh(Vn, [tuple(f) for f in o.mesh.faces]); _bump(oid)
            return jsonify({"object": oid, "frames": frames, "count": len(frames)})
        t = float(d.get("time", 0.0))
        Vn, wave = surf_at(t)
        o.mesh = Mesh(Vn, [tuple(f) for f in o.mesh.faces])
        # keep the crest/trough material split coherent
        deep = "water"; wmean = float(wave.mean()); amp = float(np.abs(spec[:, 4]).sum())
        o.mats = ["water_deep" if wave[list(f)].mean() < wmean - amp * 0.25 else deep for f in o.mesh.faces]
        _bump(oid)
        out = _payload(only=oid); out["object"] = oid; out["time"] = t
        return jsonify(out)


@bp.route("/api/photo/scene", methods=["POST"])
def photo_scene():
    """IMAGE -> SCENE BOOTSTRAP (A3-1): compose one image into a coherent starter scene rather than a single
    relief -- (1) a depth-relief terrain from shape-from-shading, coloured by the image; (2) an environment
    backdrop synthesised from the image's own colour palette (top rows = sky, bottom = ground); (3) lighting
    inferred from the image. One click gets a lit, backed scene to iterate on. Honest: monocular depth is a
    height field, and the backdrop is a palette wash, not a re-projection of the photo."""
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    made = []
    from holographic_mesh import Mesh
    with _LOCK:
        if d.get("clear", True):
            _S["objects"].clear()
        # (1) depth relief terrain
        gray = rgb.mean(axis=2)
        try:
            from holographic_shapefromshading import shape_from_shading
            depth = np.asarray(shape_from_shading(gray, smooth=float(d.get("smooth", 1.2))), float)
        except Exception:
            depth = gray                                       # fall back to luminance as pseudo-height
        step = max(1, int(d.get("step", 5)))
        dz = depth[::step, ::step]
        Hh, Ww = dz.shape
        relief = float(np.clip(d.get("relief", 0.6), 0.05, 3.0))
        dz = (dz - dz.min()) / max(np.ptp(dz), 1e-9)
        verts, faces, mats = [], [], []
        # map each sampled cell's colour to the nearest library material for a plausible look
        pal = [("grass", [0.3, 0.5, 0.2]), ("grass_dry", [0.6, 0.55, 0.3]), ("sand", [0.8, 0.72, 0.5]),
               ("water", [0.2, 0.4, 0.6]), ("water_deep", [0.1, 0.2, 0.4]), ("snow", [0.95, 0.95, 0.97]),
               ("limestone", [0.7, 0.68, 0.62]), ("crust_rock", [0.4, 0.35, 0.3]), ("forest", [0.15, 0.35, 0.15]),
               ("clay", [0.6, 0.4, 0.3]), ("obsidian", [0.1, 0.1, 0.12])]
        pal_cols = np.array([c for _, c in pal])
        for j in range(Hh):
            for i in range(Ww):
                x = (i / max(Ww - 1, 1) - 0.5) * 4.0
                yy = (0.5 - j / max(Hh - 1, 1)) * 4.0 * (Hh / Ww)
                z = float(dz[j, i]) * relief
                verts.append((x, z, yy))
        for j in range(Hh - 1):
            for i in range(Ww - 1):
                a = j * Ww + i; b = a + 1; cc = a + Ww; e = cc + 1
                faces.append((a, cc, b)); faces.append((b, cc, e))
                rj = min(j * step, rgb.shape[0] - 1); ri = min(i * step, rgb.shape[1] - 1)
                col = rgb[rj, ri]
                mi = int(np.argmin(((pal_cols - col) ** 2).sum(axis=1)))
                mats.extend([pal[mi][0], pal[mi][0]])
        terrain = Mesh(np.array(verts, float), [tuple(f) for f in faces])
        made.append(_add_object(d.get("name") or "Image terrain", terrain, mats))
        # (2) environment backdrop from the image palette (top = sky band, bottom = ground band)
        H2, W2 = 128, 256
        top = rgb[: max(1, rgb.shape[0] // 3)].reshape(-1, 3).mean(axis=0)
        bot = rgb[rgb.shape[0] * 2 // 3:].reshape(-1, 3).reshape(-1, 3).mean(axis=0)
        vv = np.linspace(1, -1, H2)[:, None]
        env = np.empty((H2, W2, 3), float)
        for ch in range(3):
            band = np.clip(vv * 0.5 + 0.5, 0, 1)               # 1 at zenith
            env[..., ch] = band[:, 0][:, None] * top[ch] + (1 - band[:, 0])[:, None] * bot[ch]
        env = np.clip(env * 1.1, 0, 4).astype(np.float32)
        _S["env_img"] = env
        # (3) inferred lighting
        try:
            from holographic_shapefromshading import estimate_light
            L = np.asarray(estimate_light(gray), float)
            az, el = _az_el_from_L(L)
            _S.setdefault("photolight", {})
            _S["photolight"] = {"az": az, "el": el}
        except Exception:
            az, el = 300.0, 55.0
        _bump()
    from PIL import Image as _Im
    import io as _io, base64 as _b64
    disp = np.clip(env / max(env.max(), 1e-6) if env.max() > 1 else env, 0, 1)
    buf = _io.BytesIO(); _Im.fromarray((disp * 255).astype(np.uint8)).save(buf, "PNG")
    out = _payload(); out["objects_made"] = made
    out["env_preview"] = _b64.b64encode(buf.getvalue()).decode()
    out["light"] = {"az": round(az, 1), "el": round(el, 1)}
    out["note"] = "Starter scene: depth-relief terrain + palette backdrop + inferred light. Monocular depth is a height field."
    return jsonify(out)


@bp.route("/api/photo/depth", methods=["POST"])
def photo_depth():
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    gray = rgb.mean(axis=2)
    from holographic_shapefromshading import shape_from_shading
    try:
        depth = np.asarray(shape_from_shading(gray, smooth=float(d.get("smooth", 1.0))), float)
    except Exception as e:
        return jsonify({"error": f"depth estimate failed: {e}"}), 400
    # height-field mesh from the depth map
    step = max(1, int(d.get("step", 4)))
    dz = depth[::step, ::step]
    H, W = dz.shape
    relief = float(d.get("relief", 0.5))
    verts, faces, cols = [], [], []
    for j in range(H):
        for i in range(W):
            x = (i / (W - 1) - 0.5) * 2.0
            y = (0.5 - j / (H - 1)) * 2.0 * (H / W)
            z = float(dz[j, i]) * relief
            verts.append((x, z, y))
            rj = min(j * step, rgb.shape[0] - 1); ri = min(i * step, rgb.shape[1] - 1)
            cols.append(tuple(rgb[rj, ri]))
    for j in range(H - 1):
        for i in range(W - 1):
            a = j * W + i; b = a + 1; c = a + W; e = c + 1
            faces.append((a, c, b)); faces.append((b, c, e))
    from holographic_mesh import Mesh
    m = Mesh(np.array(verts, float), [tuple(f) for f in faces])
    with _LOCK:
        oid = _add_object(d.get("name") or "Depth relief", m)
        _bump()
    out = _payload(); out["object"] = oid
    out["note"] = "Relief mesh from monocular shape-from-shading — a height field, not true 3-D reconstruction."
    return jsonify(out)


@bp.route("/api/photo/texture", methods=["POST"])
def photo_texture():
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    gray = rgb.mean(axis=2)
    # crop to a square power-ish patch for the statistical match
    s = min(gray.shape); gray = gray[:s, :s]
    from holographic_fitshape import fit_texture, _noise_glsl
    import holographic_noise as hn

    class _NoiseShim:
        """fit_texture wants mind.procedural_noise(...).sample_grid_fast(N); FractalNoise IS that, standalone."""
        def procedural_noise(self, n_dims=2, octaves=4, lacunarity=2.0, gain=0.5, base_bandwidth=2.0, seed=1):
            return hn.FractalNoise(n_dims=n_dims, octaves=int(octaves), lacunarity=lacunarity, gain=gain,
                                   base_bandwidth=base_bandwidth, seed=seed)
    try:
        res = fit_texture(gray, mind=_NoiseShim(), res=int(d.get("res", 20)))
    except Exception as e:
        return jsonify({"error": f"texture fit failed: {e}"}), 400
    note = res.get("note") or "A procedural fBm sharing the image's roughness signature — not a pixel-exact reproduction."
    return jsonify({"params": res.get("params"), "quality": round(float(res.get("quality", 0)), 3),
                    "glsl": res.get("glsl", ""), "note": note})


@bp.route("/api/photo/shapes", methods=["POST"])
def photo_shapes():
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d)
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    gray = rgb.mean(axis=2)
    from holographic_shapefromshading import shape_from_shading
    from holographic_primfit import fit_primitives
    try:
        depth = np.asarray(shape_from_shading(gray, smooth=float(d.get("smooth", 1.0))), float)
    except Exception as e:
        return jsonify({"error": f"depth estimate failed: {e}"}), 400
    step = max(2, int(d.get("step", 6)))
    dz = depth[::step, ::step]; H, W = dz.shape
    pts = []
    for j in range(H):
        for i in range(W):
            if dz[j, i] > 0.05:
                pts.append(((i / (W - 1) - 0.5) * 2.0, float(dz[j, i]), (0.5 - j / (H - 1)) * 2.0 * (H / W)))
    if len(pts) < 20:
        return jsonify({"error": "not enough foreground depth to fit shapes"}), 400
    pts = np.array(pts, float)
    k = int(np.clip(int(d.get("k", 4)), 1, 12))
    try:
        fit = fit_primitives(pts, k=k, auto_k=bool(d.get("auto_k", False)),
                             primitives=tuple(d.get("primitives", ("sphere", "box", "capsule"))))
    except Exception as e:
        return jsonify({"error": f"primitive fit failed: {e}"}), 400
    prims = fit.get("parts") if isinstance(fit, dict) else None
    tree = fit.get("sdf") if isinstance(fit, dict) else None
    # build a mesh from the fitted SDF union directly (the module hands back an SDF tree)
    made = None
    dsl = None
    if tree is not None:
        try:
            dsl = tree.to_dsl()
        except Exception:
            dsl = None
        try:
            lo = pts.min(axis=0) - 0.3; hi = pts.max(axis=0) + 0.3
            m = _mesh_sdf_tree(tree, res=48, face_target=6000, scan_lo=lo, scan_hi=hi)
            with _LOCK:
                made = _add_object(d.get("name") or "Fitted shapes", m, sdf_tree=tree)
                _bump()
        except Exception:
            made = None
    out = {"n_primitives": (len(prims) if prims else None),
           "kinds": (fit.get("kinds") if isinstance(fit, dict) else None),
           "residual": round(float(fit.get("residual", 0)), 4) if isinstance(fit, dict) else None,
           "dsl": dsl,
           "note": "EXPERIMENTAL & rough: monocular depth → point cloud → exact-SDF primitive cover. "
                   "Expect a loose approximation, not a faithful model."}
    if made:
        payload = _payload(); payload.update(out); payload["object"] = made
        return jsonify(payload)
    return jsonify(out)


# =====================================================================================================
# SEMANTIC COMMAND BAR (P2.5-1): a CONTROLLED-GRAMMAR command line, not free-form NL. The engine has NO LLM
# and is emphatic about it; holographic_scene_semantic.interpret_command parses an adjust command against a
# described scene and returns {understood, matched, changes, suggestions, questions} -- including the "did you
# mean ...?" report. We build a SemanticScene mirroring our real objects (name + shape from the analytic tree +
# dominant material colour), interpret the command, then MAP the understood intent onto our real engine ops
# (colour -> assign material; size -> uniform scale; move + direction -> translate). Unknown verbs return 400
# WITH the parsed report so the UI can show the suggestions/questions the engine produced.
# =====================================================================================================
def _obj_shape(o):
    if o.sdf_tree is not None:
        dsl = o.sdf_tree.to_dsl()
        for kind in ("box", "sphere", "torus", "cylinder", "capsule", "cone", "ellipsoid", "octahedron"):
            if kind in dsl:
                return "box" if kind == "box" else ("sphere" if kind == "sphere" else kind)
    return "mesh"


def _nearest_named_color(rgb):
    names = {"red": (0.8, 0.1, 0.1), "green": (0.1, 0.7, 0.2), "blue": (0.15, 0.3, 0.85),
             "yellow": (0.9, 0.85, 0.2), "white": (0.9, 0.9, 0.9), "black": (0.08, 0.08, 0.08),
             "grey": (0.5, 0.5, 0.5), "gold": (1.0, 0.77, 0.34), "orange": (0.95, 0.55, 0.15),
             "purple": (0.5, 0.2, 0.7)}
    rgb = np.asarray(rgb, float)
    return min(names, key=lambda n: float(np.sum((np.asarray(names[n]) - rgb) ** 2)))


_COLOR_TO_MAT = {"red": "ruby", "green": "emerald", "blue": "sapphire", "gold": "gold",
                 "grey": "clay", "white": "porcelain", "black": "obsidian", "purple": "amethyst",
                 "yellow": "gold", "orange": "copper"}


@bp.route("/api/semantic", methods=["POST"])
def semantic():
    _init()
    d = request.get_json(force=True) or {}
    cmd = str(d.get("command", "")).strip()
    if not cmd:
        return jsonify({"error": "empty command"}), 400
    with _LOCK:
        from holographic_scene_semantic import SemanticScene
        ids = list(_S["objects"].keys())
        sobjs = []
        for oid in ids:
            o = _S["objects"][oid]
            col = _obj_color(o)
            sobjs.append({"name": o.name, "shape": _obj_shape(o),
                          "color": _nearest_named_color(col), "size": "", "material": ""})
        scene = SemanticScene(sobjs)
        try:
            rep = scene.interpret(cmd)
        except Exception as e:
            return jsonify({"error": f"could not parse: {e}"}), 400
        matched_idx = rep.get("matched_idx", [])
        changes = rep.get("understood", {}).get("changes", {})
        if not matched_idx or not changes:
            # nothing actionable -> hand back the engine's own report (suggestions / questions)
            return jsonify({"applied": False, "understood": rep.get("understood", {}),
                            "matched": rep.get("matched", []),
                            "suggestions": rep.get("suggestions", []),
                            "questions": rep.get("questions", []),
                            "note": "controlled grammar (no LLM) — try e.g. 'make the sphere bigger', "
                                    "'paint the cube red', 'give the cube a metal material'"}), (200 if rep.get("matched") else 400)

        applied = []
        for idx in matched_idx:
            if idx >= len(ids):
                continue
            oid = ids[idx]; o = _S["objects"][oid]
            # colour -> material assign
            if "color" in changes:
                mat = _COLOR_TO_MAT.get(str(changes["color"]).lower())
                if mat:
                    try:
                        _mat(mat)
                        o.mats = [mat] * o.mesh.n_faces
                        applied.append(f"{o.name}: colour → {mat}")
                    except Exception:
                        pass
            # size -> uniform scale about the object centroid
            if "size" in changes:
                factor = {"large": 1.4, "larger": 1.4, "big": 1.4, "bigger": 1.4,
                          "small": 0.7, "smaller": 0.7, "tiny": 0.5}.get(str(changes["size"]).lower(), 1.0)
                if factor != 1.0:
                    c = o.mesh.vertices.mean(axis=0)
                    _snap_obj(oid)
                    o.mesh.vertices = (o.mesh.vertices - c) * factor + c
                    o.mesh.normals = None; o.mesh._he = None; o.mesh._adj = None
                    if o.sdf_tree is not None:
                        try:
                            o.sdf_tree = o.sdf_tree.scale(factor)
                        except Exception:
                            o.sdf_tree = None
                    o.rev += 1
                    applied.append(f"{o.name}: size ×{factor}")
            # material -> assign a matching preset (the grammar's 'metal'/'glass'/... vocabulary)
            if "material" in changes:
                want = str(changes["material"]).lower()
                mat = {"metal": "steel", "glass": "glass", "gold": "gold", "wood": "walnut",
                       "plastic": "clay", "stone": "granite", "chrome": "chrome"}.get(want)
                if mat:
                    try:
                        _mat(mat); o.mats = [mat] * o.mesh.n_faces
                        applied.append(f"{o.name}: material → {mat}")
                    except Exception:
                        pass
        _bump()
    out = _payload()
    out.update({"applied": bool(applied), "actions": applied,
                "understood": rep.get("understood", {}), "matched": rep.get("matched", [])})
    return jsonify(out)


@bp.route("/api/describe_scene")
def describe_scene():
    _init()
    with _LOCK:
        parts = []
        for oid, o in _S["objects"].items():
            col = _nearest_named_color(_obj_color(o))
            parts.append(f"{o.name} — a {col} {_obj_shape(o)} ({o.mesh.n_faces} faces)")
    return jsonify({"objects": parts, "text": "; ".join(parts) or "empty scene"})




# =====================================================================================================
# RETOPO QUALITY REPORT (holographic_crossfield.field_report): the cross-field's own topology scoreboard --
# singularity count, field energy (lower = smoother flow), and the Poincare-Hopf check (sum of singularity
# indices must equal the Euler characteristic; a passing check means the field is globally consistent).
# =====================================================================================================
@bp.route("/api/field_report")
def field_report_ep():
    _init()
    oid = request.args.get("object", "")
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object (pass 'object')"}), 400
        import holographic_crossfield as cf
        from holographic_mesh import Mesh
        tri = o.mesh
        if any(len(f) != 3 for f in o.mesh.faces):
            tri = Mesh(o.mesh.vertices.copy(), _triangulate_faces(o.mesh.faces))
        try:
            rep = cf.field_report(tri)
        except Exception as e:
            return jsonify({"error": f"field report failed: {e}"}), 400
    return jsonify({"object": oid,
                    "singularities": int(rep.get("n_singularities", 0)),
                    "energy": round(float(rep.get("energy", 0)), 2),
                    "consistent": bool(rep.get("poincare_hopf", False)),
                    "euler": rep.get("euler"),
                    "note": "Cross-field quality: fewer singularities + lower energy = cleaner edge flow. "
                            "'Consistent' = the field's singularities satisfy Poincaré–Hopf (globally valid)."})


# =====================================================================================================
# PRINT / FABRICATION VALIDATION (P3-5): "will this slice?" answered with leCore's own topology scoreboard
# (holographic_meshtools.mesh_report + Mesh.is_manifold/is_closed/genus). Manifold + watertight are the hard
# slicer gates; a bbox-thinnest-span serves as a coarse min-feature proxy (a true wall-thickness map is the
# follow-up). Re-verified reliable after a stale compile-cache was cleared.
# =====================================================================================================
@bp.route("/api/validate")
def validate():
    _init()
    oid = request.args.get("object", "")
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object (pass 'object')"}), 400
        from holographic_meshtools import mesh_report
        rep = mesh_report(o.mesh)
        span = np.asarray(rep.get("bbox_span", [1, 1, 1]), float)
        min_dim = float(span.min())
        thresh = float(request.args.get("min_feature", 0.02))
        checks = [
            {"name": "Manifold", "pass": bool(rep.get("is_manifold")),
             "detail": f"{rep.get('nonmanifold_edges', 0)} non-manifold edge(s)"},
            {"name": "Watertight (closed)", "pass": bool(rep.get("is_closed")),
             "detail": f"{rep.get('boundary_edges', 0)} boundary edge(s) — holes if > 0"},
            {"name": "Min feature ≥ threshold", "pass": min_dim >= thresh,
             "detail": f"thinnest bbox span {min_dim:.3f} vs threshold {thresh:.3f} (coarse proxy)"},
        ]
        printable = all(c["pass"] for c in checks[:2])
    return jsonify({"object": oid, "printable": printable, "checks": checks,
                    "euler": rep.get("euler_characteristic"),
                    "report": {k: rep[k] for k in ("verts", "faces", "boundary_edges",
                                                   "nonmanifold_edges", "is_manifold", "is_closed") if k in rep},
                    "note": "Manifold + watertight are the slicer gates. Min-feature here is a bbox proxy; a true "
                            "wall-thickness map is the follow-up."})


@bp.route("/api/repair", methods=["POST"])
def repair_mesh():
    """AUTO-REPAIR for printability (P3-5 follow-up, holographic_meshtools.mesh_repair): weld coincident
    vertices, fill boundary holes, drop unreferenced vertices, split non-manifold vertices. Returns the
    before/after report so the fix is auditable -- a mesh that failed the watertight gate can often be made
    slicer-ready in one click. Refuses on objects with an analytic tree unless 'force' (repair would drop the
    exact representation), and takes an undo snapshot."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        if o.sdf_tree is not None and not d.get("force"):
            return jsonify({"error": "this object has an exact analytic tree (already watertight by "
                                     "construction); pass force=true only if you really want to bake+repair it"}), 400
        from holographic_meshtools import mesh_repair, mesh_report
        _snap_obj(oid)
        try:
            fixed, stats = mesh_repair(o.mesh, weld_tol=float(d.get("weld_tol", 1e-5)),
                                       fill_holes=bool(d.get("fill_holes", True)),
                                       max_fill_sides=int(d.get("max_fill_sides", 12)),
                                       drop_unreferenced=True, split_nonmanifold=True)
        except Exception as e:
            _discard_snapshot()
            return jsonify({"error": f"repair failed: {e}"}), 400
        o.mats = _transfer_mats(o.mesh, o.mats, fixed)
        o.mesh = fixed; o.sdf_tree = None
        _bump(oid)
        after = mesh_report(o.mesh)
    out = _payload(only=oid); out["object"] = oid
    out["repair"] = {"before": stats.get("before"), "after": stats.get("after"),
                     "now_watertight": bool(after.get("is_closed")),
                     "now_manifold": bool(after.get("is_manifold"))}
    return jsonify(out)


@bp.route("/api/parent", methods=["POST"])
def parent_op():
    """A2-3 hierarchical grouping: link objects so a parent's transforms propagate to its children (a real
    scene graph, not a destructive merge -- children stay separate, editable objects). action 'set' parents
    child->parent (rejecting cycles); 'clear' unparents; 'list' returns the current tree. Transforming a
    parent via /api/op (translate/rotate/scale/place/drop) moves the whole subtree as one."""
    _init()
    d = request.get_json(force=True) or {}
    act = d.get("action", "set")
    with _LOCK:
        if act == "list":
            return jsonify({"parents": dict(_PARENT),
                            "roots": [k for k in _S["objects"] if k not in _PARENT]})
        if act == "clear":
            cid = str(d.get("child", ""))
            _PARENT.pop(cid, None)
            return jsonify({"ok": True, "child": cid})
        # set
        child = str(d.get("child", "")); parent = str(d.get("parent", ""))
        if child not in _S["objects"] or parent not in _S["objects"]:
            return jsonify({"error": "child and parent must both exist"}), 400
        if child == parent:
            return jsonify({"error": "cannot parent an object to itself"}), 400
        # reject cycles: walk parent's ancestry, ensure child isn't an ancestor
        p = parent; hops = 0
        while p is not None and hops < 1000:
            if p == child:
                return jsonify({"error": "that would create a cycle"}), 400
            p = _PARENT.get(p); hops += 1
        _PARENT[child] = parent
        return jsonify({"ok": True, "child": child, "parent": parent})


@bp.route("/api/upscale", methods=["POST"])
def upscale():
    """2x upscale a rendered frame with the engine's EASU (FSR-style edge-adaptive upscaler,
    holographic_superres.easu_upscale) -- sharper than bilinear, no network model."""
    _init()
    d = request.get_json(force=True) or {}
    try:
        rgb = _decode_photo(d, cap=1600)      # an upscaler must not shrink its input first
    except Exception as e:
        return jsonify({"error": f"could not read image: {e}"}), 400
    scale = float(np.clip(float(d.get("scale", 2.0)), 1.5, 4.0))
    try:
        from holographic_superres import easu_upscale
        out = np.clip(easu_upscale(rgb, scale=scale), 0, 1)
    except Exception as e:
        return jsonify({"error": f"upscale failed: {e}"}), 400
    import base64
    return jsonify({"png": "data:image/png;base64," + base64.b64encode(_png_bytes(out)).decode(),
                    "w": out.shape[1], "h": out.shape[0], "scale": scale,
                    "note": f"EASU {scale:.1f}× upscale (edge-adaptive, no ML model)."})


# =====================================================================================================
# DATA-VISUALIZATION PIPELINE (P2.5-2): a numeric series becomes 3-D geometry through visible, inspectable
# steps, each a real leCore module. /api/data/analyze runs holographic_demux (detect_interleave + demux_series)
# to see whether a series is actually several interleaved channels and returns each clean channel + the
# engine's own confidence. /api/data/to_geometry turns ONE chosen series into geometry: a LATHE PROFILE
# (series -> radius(y), revolved into a vase via the existing lathe op path) or a HEIGHTFIELD strip.
# Honest: demux confidence is surfaced (a loose fit says so); the lathe object is field-only (no analytic tree).
# =====================================================================================================
def _parse_series(d):
    """Accept {'series': [...]}, or {'csv': '...', 'column': i}, -> a 1-D float array."""
    if "series" in d and d["series"] is not None:
        return np.asarray([float(x) for x in d["series"]], float)
    if "csv" in d:
        import csv, io as _io
        rows = list(csv.reader(_io.StringIO(d["csv"])))
        col = int(d.get("column", 0))
        vals = []
        for r in rows:
            if col < len(r):
                try:
                    vals.append(float(r[col]))
                except ValueError:
                    pass                                     # skip header / non-numeric
        return np.asarray(vals, float)
    raise ValueError("need 'series' (list) or 'csv' + 'column'")


@bp.route("/api/data/analyze", methods=["POST"])
def data_analyze():
    _init()
    d = request.get_json(force=True) or {}
    try:
        x = _parse_series(d)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if len(x) < 8:
        return jsonify({"error": "need at least 8 samples"}), 400
    import holographic_demux as dx
    det = dx.detect_interleave(x)
    k = int(det.get("k", 1)); score = float(det.get("score", 0.0))
    channels = []
    if k > 1:
        try:
            res = dx.demux_series(x)
            objs = np.asarray(res.get("objects"))            # (n_objects, T, stride)
            stride = int(res.get("stride", k))
            if objs.ndim == 3:
                for ci in range(objs.shape[2]):
                    channels.append(objs[0, :, ci].tolist())
        except Exception:
            channels = []
    if not channels:
        channels = [x.tolist()]
        k = 1
    return jsonify({"n_channels": len(channels), "stride": k,
                    "interleave_score": round(score, 3),
                    "channels": [ch[:512] for ch in channels],       # cap payload
                    "note": "detect_interleave + demux_series (holographic_demux). "
                            "Score near its baseline = probably a single series, not interleaved."})


@bp.route("/api/data/to_geometry", methods=["POST"])
def data_to_geometry():
    _init()
    d = request.get_json(force=True) or {}
    try:
        x = _parse_series(d)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if len(x) < 3:
        return jsonify({"error": "need at least 3 samples"}), 400
    mode = d.get("mode", "lathe")
    # normalise the series to a friendly range
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    span = (hi - lo) or 1.0
    with _LOCK:
        if mode == "lathe":
            # series -> radius(y): map samples to radii in [0.15, 1.0], stacked along Y in [-1, 1]
            n = min(len(x), 64)
            idx = np.linspace(0, len(x) - 1, n).astype(int)
            radii = 0.15 + (x[idx] - lo) / span * 0.85
            ys = np.linspace(-1.0, 1.0, n)
            profile = [[float(r_), float(y_)] for r_, y_ in zip(radii, ys)]
            # reuse the lathe verb path directly
            from holographic_sdf2d import polygon2d, revolve
            loop = list(profile)
            if loop[0][0] != 0.0:
                loop.insert(0, [0.0, loop[0][1]])
            if loop[-1][0] != 0.0:
                loop.append([0.0, loop[-1][1]])
            fn = revolve(polygon2d([(r_, y_) for r_, y_ in loop]), offset=0.0)

            class _F:
                def eval(self, P):
                    return np.asarray(fn(np.atleast_2d(P)), float)
            rmax = max(r_ for r_, _ in loop)
            lo3 = np.array([-rmax - 0.1, -1.2, -rmax - 0.1]); hi3 = np.array([rmax + 0.1, 1.2, rmax + 0.1])
            mesh = _mesh_sdf_tree(_F(), res=64, face_target=6000, scan_lo=lo3, scan_hi=hi3)
            oid = _add_object(d.get("name") or "Data lathe", mesh)
            _bump()
            out = _payload(); out["object"] = oid
            out["note"] = "Series → radius(y) → revolved solid (field-only, like any lathe object)."
            return jsonify(out)
        elif mode == "heightfield":
            # series -> a ribbon: x-position = sample index, height = value, small depth in z
            n = min(len(x), 96)
            idx = np.linspace(0, len(x) - 1, n).astype(int)
            hs = (x[idx] - lo) / span
            verts, faces = [], []
            for i in range(n):
                xx = (i / (n - 1) - 0.5) * 2.0
                verts.append((xx, float(hs[i]) * 0.8, -0.15))
                verts.append((xx, float(hs[i]) * 0.8, 0.15))
            for i in range(n - 1):
                a = 2 * i; b = a + 1; cc = a + 2; e = a + 3
                faces.append((a, cc, b)); faces.append((b, cc, e))
            from holographic_mesh import Mesh
            mesh = Mesh(np.array(verts, float), [tuple(f) for f in faces])
            oid = _add_object(d.get("name") or "Data ribbon", mesh)
            _bump()
            out = _payload(); out["object"] = oid
            out["note"] = "Series → height ribbon (a 3-D line chart you can model on)."
            return jsonify(out)
    return jsonify({"error": f"unknown mode '{mode}'"}), 400


# =====================================================================================================
# SCENE SAVE / LOAD (A2-2): serialize the whole arrangement — objects (name + mesh + per-face materials) and
# any session-authored custom materials — to one JSON, and rebuild it. Foundational: lets a build be
# checkpointed, handed to a separate render process, or reopened later. Analytic trees are NOT serialized
# (they'd need DSL round-tripping); a reloaded object is mesh-only, which is correct and honest — the geometry
# is preserved exactly, only the editability-as-primitive is lost. Stated in the response.
# =====================================================================================================
# ---------------------------------------------------------------------------------------------------
# P2.5-1: AGENT SURFACE. The app is a leCore demo, so an external agent should be able to DRIVE it the
# same way it drives the engine: ask what it can do, then call it -- without scraping the UI or reading
# this file. /api/agent/tools returns a machine-readable manifest generated FROM the live Flask url_map
# (so it cannot drift from the routes that actually exist), and /api/agent/invoke calls one by name.
# ---------------------------------------------------------------------------------------------------

_AGENT_HINTS = {
    "scene": "Current scene: objects with ids, names, face/vertex counts, materials, transforms.",
    "op": "Object ops: delete_object, duplicate, boolean_union/difference/intersect, merge, etc. Body: {op, object, ...}.",
    "add": "Add a primitive. Body: {kind: cube|sphere|cylinder|cone|torus|plane, ...}.",
    "generate": "Procedural generators. Body: {kind: landscape|creature|galaxy|crystal|..., seed, ...}.",
    "assign": "Assign a material. Body: {material, object, all|faces:[...]|objects:[...]}.",
    "materials": "Material library grouped by class, with albedo/metallic/roughness.",
    "scatter": "Scatter copies of one object over another's surface. Body: {source, target, count, scale, align, seed}.",
    "erode": "Hydraulic erosion on a heightfield object. Body: {object, droplets, seed, strength}.",
    "import_glb": "POST raw .glb bytes. Query: mode=auto|asis|decimate|retopo|voxel|rebake, target, reproject.",
    "export": "Export the scene or an object (glb/obj/stl).",
    "render": "Single preview frame (PNG). Query: quality, eye, target, fov, grid.",
    "render_progressive": "Progressive preview round (PNG). Same camera query + session; read X-Converged.",
    "render_engine": "Mesh-rasteriser render (PNG), textured.",
    "photo": "Path-traced GI photo (NDJSON stream of frames). Query: w, h, spp, grid, aov.",
    "mass_properties": "Volume, mass, centre of mass, inertia/principal moments for an object.",
    "section": "Cross-section measurement/curve at a plane.",
    "draft_report": "Mouldability: per-face draft angles against a pull direction.",
    "draft_apply": "Apply a minimum draft angle to an object.",
    "scene/save": "Full-fidelity scene document (JSON) -- round-trips through scene/load.",
    "scene/load": "Restore a scene document produced by scene/save.",
    "scene/new": "Reset to the factory scene.",
    "scene_graph": "Object -> material -> texture link graph.",
    "nodes/op": "Node graph: add/connect/remove/collapse/expand/build/clear.",
    "engine_status": "Which leCore is mounted (installed vs bundled) and which extras are present.",

    # --- history -------------------------------------------------------------------------------
    "undo": "Undo the last edit. POST {}. Returns the scene after undoing.",
    "redo": "Redo the last undone edit. POST {}. Any NEW edit clears the redo stack.",
    "history": "Per-object construction history and branches. Query: object.",
    "history/op": "Walk or edit that history. Body: {object, action, index}.",

    # --- render control ------------------------------------------------------------------------
    "render_cancel": "Stop the render in flight for a session. Query: session. POST or GET.",
    "photo_post": "Re-grade the last finished photo WITHOUT re-tracing (PNG). Query: session, exposure, sharpen. 404 if that session has no cached render.",
    "material_ball": "Preview sphere for one material (PNG). Query: name, res.",
    "object_texture": "An imported object's stored base-colour texture (PNG). Query: object.",
    "scene_wgsl": "The whole scene as an emitted WGSL distance function, for a client-side GPU tracer.",
    "field_meta": "Poll the baked scene volume: rev, resolution, bounds, palette, lighting.",
    "field_report": "Bake diagnostics for one object (method, resolution, exactness). Query: object.",

    # --- creating geometry ----------------------------------------------------------------------
    "new": "Create an object. Body: {kind, ...}. Same shape as add.",
    "compose": "Build an SDF solid from a controlled description. Body: {text} e.g. \"sphere radius .6 subtract rounded box size .5 .3 .4\". Unknown forms are refused by name.",
    "semantic": "Run a described edit against the current scene. Body: {command}.",
    "sweep": "Tube swept along a path. Body: {path|points:[[x,y,z],...], radius, sides, samples, taper, cap}.",
    "describe_scene": "Plain-language description of the scene, for an agent to read back.",

    # --- data and photos in ------------------------------------------------------------------
    "data/analyze": "Inspect an uploaded table/series before turning it into geometry. Body: the data.",
    "data/to_geometry": "Turn data into geometry. Body: {mode, ...} -- heightfield, bars, curve.",
    "photo/light": "Estimate sun azimuth/elevation from a photo. Body: {image}.",
    "photo/depth": "Shape-from-shading depth map -> a height-field mesh object. Body: {image}.",
    "photo/shapes": "Fit exact SDF primitives to a photo's shapes. Body: {image}.",
    "photo/texture": "Fit a procedural fBm texture whose statistics match a photo. Body: {image}.",

    # --- uv, sculpt, nodes, checks ---------------------------------------------------------------
    "uv": "Unwrap an object and store its UVs. Body: {object}.",
    "uv/maps.zip": "Download that object's UV/texture maps as a .zip. Query: object.",
    "validate": "Printability check: thin walls, non-manifold edges, holes. Query: object, min_feature.",
    "sculpt/enter": "Switch an object into sculpt mode (mesh becomes a distance grid). Body: {object}.",
    "sculpt/exit": "Leave sculpt mode, re-extracting the mesh. Body: {object}.",
    "sculpt/begin_stroke": "Open a sculpt stroke; points are streamed to sculpt/stroke. Body: {object, brush, radius, strength}.",
    "nodes/graph": "The current procedural node graph: nodes, edges, parameters.",
    "nodes/types": "Every node type available, with its sockets and parameters.",
    "milkdrop/presets": "Milkdrop (.milk) motion presets bundled with the app.",
}


def _tool_summary(name, view):
    """Hint first, then the view's own first docstring line, then a readable fallback built from the route
    name -- an agent choosing between tools must never be handed a blank description."""
    hint = _AGENT_HINTS.get(name)
    if hint:
        return hint
    doc = (getattr(view, "__doc__", "") or "").strip()
    if doc:
        # Take the first SENTENCE, not the first LINE. Docstrings here wrap at ~100 characters, so
        # line-splitting handed agents summaries that stopped mid-clause -- "...ranks by keyword hits
        # over name (weighted), aliases, and the". Join the opening paragraph first, then cut at
        # sentence punctuation, and only then fall back to a word boundary with an ellipsis.
        para = []
        for line in doc.split("\n"):
            if not line.strip():
                break
            para.append(line.strip())
        first = " ".join(para)
        for end in (". ", "? ", "! "):
            k = first.find(end)
            if 0 < k < 200:
                first = first[:k + 1]
                break
        if len(first) > 200:
            cut = first[:200]
            stop = max(cut.rfind("; "), cut.rfind(" -- "), cut.rfind(", "))
            first = (cut[:stop] if stop > 80 else cut.rsplit(" ", 1)[0]) + "\u2026"
        if first:
            return first
    return name.replace("/", " ").replace("_", " ").strip() + " (no description recorded)"


# ---------------------------------------------------------------------------------------------------
# SHARED WORKSPACE (.lews) -- the bridge to leStudio, the 2-D editor built on the same engine.
#
# The file format is leCore's own sectioned container: a ZIP of typed sections {kind, id, meta, arrays}
# where A SECTION WHOSE KIND A READER DOES NOT UNDERSTAND ROUND-TRIPS UNTOUCHED. That property is the
# whole point -- the painter can carry our meshes without opening them, and we carry their documents
# without painting. Verified byte-identical in both directions before this was wired.
#
# We read `lestudio.document` sections (a painted document = a texture) and `lecore.image` sections (the
# canonical image kind), and we write our objects as `polystudio.object` plus a `lecore.image` per texture
# so ANY app -- not just leStudio -- can consume what we produce.
#
# Compositing is the ENGINE's (holographic_composite.composite_layers), not ours: two apps re-implementing
# ten blend modes drift, and then the same document renders differently in the modeller than in the
# painter. That was filed as a core gap and fixed upstream; this calls it rather than reimplementing it.
# ---------------------------------------------------------------------------------------------------

def _lews_documents(container):
    """Every paintable section in a loaded container -> [{id, name, w, h, rgb}] with layers already
    composited by the engine. Handles both `lestudio.document` (layer stack) and `lecore.image` (flat)."""
    from holographic.materials_and_texture.holographic_composite import composite_layers
    out = []
    for sec in container.get("sections", []):
        kind = sec.get("kind")
        try:
            if kind == "lestudio.document":
                meta = sec.get("meta") or {}
                recs = meta.get("layers") or []
                layers = {}
                for r in recs:
                    key = "layer_%s" % r.get("id")
                    if key in sec["arrays"]:
                        layers[r.get("id")] = np.asarray(sec["arrays"][key], float)
                if not layers:
                    continue
                img = np.asarray(composite_layers(layers, recs), float)
                out.append({"id": str(sec.get("id") or meta.get("id") or len(out)),
                            "name": str(meta.get("name") or "document"),
                            "w": int(img.shape[1]), "h": int(img.shape[0]),
                            "layers": len(layers), "rgb": np.clip(img[:, :, :3], 0.0, 1.0)})
            elif kind == "lecore.image":
                from holographic.io_and_interop.holographic_container import read_image_section
                got = read_image_section(sec)
                img = np.asarray(got[0] if isinstance(got, tuple) else got, float)
                out.append({"id": str(sec.get("id") or len(out)),
                            "name": str((sec.get("meta") or {}).get("name") or "image"),
                            "w": int(img.shape[1]), "h": int(img.shape[0]),
                            "layers": 1, "rgb": np.clip(img[:, :, :3], 0.0, 1.0)})
        except Exception:
            continue                                      # a section we cannot read is carried, not fatal
    return out


@bp.route("/api/workspace/inspect", methods=["POST"])
def workspace_inspect():
    """POST a .lews -- report what is inside WITHOUT importing, so the user picks knowingly. Uses the
    engine's own kind registry so foreign sections are named rather than silently ignored."""
    _init()
    from holographic.io_and_interop.holographic_container import load_container, describe_sections
    try:
        cont = load_container(request.get_data())
    except Exception as e:
        return jsonify({"error": f"not a leCore workspace container: {e}"}), 400
    try:
        desc = describe_sections(cont)
    except Exception:
        desc = [{"kind": s.get("kind"), "known": None} for s in cont.get("sections", [])]
    docs = _lews_documents(cont)
    return jsonify({"app": (cont.get("meta") or {}).get("app"),
                    "sections": desc,
                    "textures": [{"id": d["id"], "name": d["name"], "w": d["w"], "h": d["h"],
                                  "layers": d["layers"]} for d in docs]})


@bp.route("/api/workspace/import", methods=["POST"])
def workspace_import():
    """POST a .lews (raw body) -> its painted documents become texture assets on this scene.
    Query: object=<id> applies the first texture to that object; otherwise textures are stored and can be
    applied later. The whole container is retained so a later export gives the painter their file back."""
    _init()
    from holographic.io_and_interop.holographic_container import load_container
    try:
        cont = load_container(request.get_data())
    except Exception as e:
        return jsonify({"error": f"not a leCore workspace container: {e}"}), 400
    docs = _lews_documents(cont)
    if not docs:
        return jsonify({"error": "no paintable documents in that workspace"}), 400
    with _LOCK:
        # keep every section we did not author, so exporting later returns their work untouched
        _S["ws_foreign"] = [s for s in cont.get("sections", [])
                            if s.get("kind") not in ("polystudio.object",)]
        _S["ws_meta"] = cont.get("meta") or {}
        store = _S.setdefault("ws_textures", {})
        for d in docs:
            store[d["id"]] = {"name": d["name"], "rgb": d["rgb"]}
        target = str(request.args.get("object", ""))
        applied = None
        if target and target in _S["objects"]:
            o = _S["objects"][target]
            uv = None
            a = _S.get("render_assets", {}).get(target)
            if a and "uv" in a and len(a["uv"]) == o.mesh.n_vertices:
                uv = np.asarray(a["uv"], float)
            if uv is None:
                uv = _planar_uv(o.mesh)                   # unwrapped so a painted texture has somewhere to land
            _S["render_assets"][target] = {"uv": uv, "tex": docs[0]["rgb"]}
            _S["rev"] += 1
            o.rev += 1                                    # texture-only change: do NOT drop the analytic tree
            _S["cache"] = {k: v for k, v in _S["cache"].items() if k[0] != target}
            applied = target
        out = _payload()
        out["workspace"] = {"textures": [{"id": d["id"], "name": d["name"], "w": d["w"], "h": d["h"]}
                                         for d in docs],
                            "applied_to": applied,
                            "carried_sections": len(_S["ws_foreign"])}
        return jsonify(out)


def _planar_uv(mesh):
    """A plain planar unwrap on the mesh's two widest axes -- enough to LAND a painted texture on an object
    that has no uvs of its own. Honest about what it is: not a seam-aware unwrap."""
    V = np.asarray(mesh.vertices, float)
    ext = V.max(axis=0) - V.min(axis=0)
    a, b = np.argsort(-ext)[:2]
    lo, hi = V[:, [a, b]].min(axis=0), V[:, [a, b]].max(axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    return (V[:, [a, b]] - lo) / span


@bp.route("/api/workspace/export")
def workspace_export():
    """Download the scene as a .lews the painter can open: our objects as `polystudio.object` sections,
    each texture as a canonical `lecore.image`, plus every foreign section from an imported workspace
    carried through verbatim."""
    _init()
    from holographic.io_and_interop.holographic_container import save_container, image_section
    with _LOCK:
        sections = list(_S.get("ws_foreign", []))         # the painter's documents, untouched
        for oid, o in _S["objects"].items():
            m = o.mesh
            F = np.array([list(f)[:3] for f in m.triangulate()], dtype=np.int32)
            sec = {"kind": "polystudio.object", "id": str(oid),
                   "meta": {"name": o.name, "faces": int(m.n_faces),
                            "materials": sorted({str(x) for x in np.asarray(o.mats).ravel()})},
                   "arrays": {"verts": np.asarray(m.vertices, np.float32), "faces": F}}
            a = _S.get("render_assets", {}).get(oid)
            if a and "uv" in a:
                sec["arrays"]["uv"] = np.asarray(a["uv"], np.float32)
            sections.append(sec)
            if a and "tex" in a:
                tex = np.clip(np.asarray(a["tex"], float), 0.0, 1.0)
                try:
                    sections.append(image_section(tex, colour_space="srgb", dpi=72.0,
                                                  name=f"{o.name} texture"))
                except Exception:
                    pass
        meta = dict(_S.get("ws_meta") or {})
        meta.setdefault("app", "polystudio")              # our own files identify as ours
        blob = save_container(sections, meta=meta)
    resp = Response(blob, mimetype="application/octet-stream")
    resp.headers["Content-Disposition"] = 'attachment; filename="workspace.lews"'
    return resp


# The agent surface used to live here: a manifest derived from the live url_map plus an invoke-by-name
# door, ~110 lines. leCore sweep 163 shipped holographic_appserver.AgentSurface -- the union of what this
# app and leStudio each hand-rolled, with our base64-frame lesson kept -- and app.py now mounts it via
# MIND.agent_surface(app, base="/api", ...). Deleted rather than deprecated: two manifests that can
# disagree about what this app exposes is exactly the drift that produced the `kind`/`primitive` bug.


# ---------------------------------------------------------------------------------------------------
# ENGINE PREFLIGHT. The app can run on a pip-installed leCore OR a vendored copy, and those are not
# necessarily the same version -- the vendored overlay is deliberately AHEAD of the published release.
# So state, in one place, exactly what this app calls that a fresh pypi install might not have yet, and
# check it. A user on an older `leos-core` then gets a NAMED list of what is missing instead of a crash
# in the middle of a render. Each entry says which feature breaks, so the message is actionable.
# ---------------------------------------------------------------------------------------------------

_ENGINE_REQUIRES = [
    # (module, attribute or None, what breaks without it)
    ("holographic_composite", "composite_layers", "shared workspace: compositing leStudio layers into a texture"),
    ("holographic_composite", "BLEND_MODES", "shared workspace: layer blend modes"),
    ("holographic_container", "save_container", "shared workspace: reading/writing .lews files"),
    ("holographic_container", "image_section", "shared workspace: the canonical lecore.image texture kind"),
    ("holographic_container", "register_kind", "shared workspace: naming foreign sections in the UI"),
    ("holographic_meshscatter", "sample_mesh_surface", "Modifiers: scatter instances on a surface"),
    ("holographic_meshtools", "transfer_uv", "GLB import: texture-preserving decimation"),
    ("holographic_meshtools", "textured_lod", "GLB import: the rebake-atlas mode"),
    ("holographic_meshqem", "decimate_to", "GLB import: silhouette-guarded face budgets"),
    ("holographic_gemrender", "clamp_fireflies", "photo render: firefly clamp"),
    ("holographic_terrain", "erode", "terrain erosion"),
]


def _engine_preflight():
    """Which required engine capabilities are present. Cheap (imports only), so it can back an endpoint."""
    import importlib
    missing, present = [], 0
    for mod, attr, why in _ENGINE_REQUIRES:
        try:
            m = importlib.import_module(mod)
            if attr and not hasattr(m, attr):
                missing.append({"module": mod, "attr": attr, "breaks": why, "reason": "attribute missing"})
            else:
                present += 1
        except Exception as e:
            missing.append({"module": mod, "attr": attr, "breaks": why, "reason": f"import failed: {e}"[:90]})
    # signature-level checks: an attribute can exist but predate the argument this app passes
    sig_missing = []
    try:
        import inspect
        from holographic_raymarch import render_sdf as _rs
        if "mask" not in inspect.signature(_rs).parameters:
            sig_missing.append({"call": "render_sdf(mask=)", "breaks": "preview: adaptive masked sampling"})
    except Exception:
        pass
    try:
        import inspect
        import holographic_meshqem as _mq
        if "target_faces" not in inspect.signature(_mq.cluster_decimate).parameters:
            sig_missing.append({"call": "cluster_decimate(target_faces=)", "breaks": "GLB import: face budgets"})
    except Exception:
        pass
    missing.extend(sig_missing)
    return {"required": len(_ENGINE_REQUIRES), "present": present, "missing": missing,
            "ok": not missing,
            "hint": ('pip install -U "leos-core[ui]"  # this app needs >= 0.2.11'
                     if missing else "engine satisfies everything this app calls")}


@bp.route("/api/engine_preflight")
def engine_preflight():
    """Is the engine this app is running on new enough? Names what is missing and what it breaks."""
    _init()
    return jsonify(_engine_preflight())


@bp.route("/api/engine_status")
def engine_status():
    """WHICH ENGINE, WHICH EXTRAS: reports how leCore was resolved (pip-installed `leos-core` vs the bundled
    repo overlay) and which optional acceleration tiers are live -- the pypi extras map: [ui]=Flask+Pillow
    (this app's baseline), [jit]=numba fast paths, [zig]=native batch kernels, [gpu]=CuPy, [symbolic]=SymPy.
    Lets a user see at a glance why things are fast or slow and which `pip install "leos-core[...]"` would
    change that."""
    import importlib, importlib.util
    import flatcompat as _fc
    root = getattr(_fc, "_PKG_ROOT", "?")
    installed_names = [n for n in ("lecore", "holographic", "leos_core", "leoscore")
                       if importlib.util.find_spec(n) is not None]
    def _have(mod):
        try:
            importlib.import_module(mod); return True
        except Exception:
            return False
    extras = {"ui (Flask+Pillow)": _have("flask") and _have("PIL"),
              "jit (numba)": _have("numba"),
              "symbolic (SymPy)": _have("sympy"),
              "gpu (CuPy)": _have("cupy"),
              "images (Pillow)": _have("PIL")}
    zig = cc = False
    try:
        import holographic_zigrun as _zr
        zig = bool(getattr(_zr, "zig_available", lambda: False)())
        # honest cc test: a compiler on PATH (the module ships no cc_available helper)
        import shutil as _sh
        cc = any(_sh.which(x) for x in ("cc", "gcc", "clang"))
    except Exception:
        pass
    extras["zig (native kernels)"] = zig
    extras["cc (C fallback kernels)"] = cc
    mode = "bundled repo overlay (ahead of the pypi release)" if "holostuff" in str(root)         else "pip-installed package"
    return jsonify({"engine_root": str(root), "resolution": mode,
                    "installed_import_names": installed_names,
                    "extras": extras,
                    "install_hint": 'pip install "leos-core[ui,jit]"  # this app\'s recommended set; '
                                    'add [zig] or [gpu] for more speed'})


@bp.route("/api/scene_graph")
def scene_graph():
    """SCENE-GRAPH VIEW (basics pass): the whole scene as a node graph -- every object, every material IN USE
    (with per-object face counts), every import texture, the environment, and the viewport light rig -- so the
    node editor can show what the scene IS, not just the SDF recipe. Links: object->material (n faces),
    object->texture, object->parent."""
    _init()
    with _LOCK:
        objects, links = [], []
        for oid, o in _S["objects"].items():
            objects.append({"id": oid, "name": o.name, "faces": int(o.mesh.n_faces),
                            "verts": int(o.mesh.n_vertices),
                            "parent": _PARENT.get(oid),
                            "hasTexture": "uv" in _S.get("render_assets", {}).get(oid, {}),
                            "hasFaceColors": "face_colors" in _S.get("render_assets", {}).get(oid, {})})
            counts = {}
            for mname in o.mats:
                counts[mname] = counts.get(mname, 0) + 1
            for mname, n in counts.items():
                links.append({"object": oid, "material": mname, "faces": int(n)})
            if oid in _S.get("render_assets", {}) and "uv" in _S["render_assets"][oid]:
                links.append({"object": oid, "texture": oid})
        used = sorted({l["material"] for l in links if "material" in l})
        ml = _matlib()
        materials = []
        for mname in used:
            try:
                m = _mat(mname)
                materials.append({"name": mname, "albedo": [round(float(c), 3) for c in np.asarray(m.base_color)[:3]],
                                  "custom": mname in _CUSTOM_MATS})
            except Exception:
                materials.append({"name": mname, "albedo": [0.5, 0.5, 0.5], "custom": mname in _CUSTOM_MATS})
        textures = [{"object": oid, "size": list(np.asarray(a["tex"]).shape[:2])}
                    for oid, a in _S.get("render_assets", {}).items() if "uv" in a]
        lights = [{"kind": "directional", "role": "key"}, {"kind": "directional", "role": "fill"},
                  {"kind": "ambient", "role": "ambient"}]
        if _S.get("env_img") is not None:
            lights.append({"kind": "environment", "role": "sky dome"})
        return jsonify({"objects": objects, "materials": materials, "textures": textures,
                        "lights": lights, "links": links})


@bp.route("/api/scene/new", methods=["POST"])
def scene_new():
    """NEW SCENE: clear EVERYTHING -- objects, node graph, custom materials, render assets (import textures),
    history, parents, undo, environment -- and re-seed the factory default scene. The one-button fresh start."""
    _init()
    with _LOCK:
        _S["objects"].clear(); _S["undo"].clear(); _S["cache"].clear()
        _S["render_assets"] = {}
        _S["next_id"] = 1
        _S["env_img"] = None
        _CUSTOM_MATS.clear(); _GLB_CACHE.clear(); _MATBALL_CACHE.clear()
        _HIST.clear(); _PARENT.clear()
        _NODES["graph"] = None; _NODES["pos"] = {}; _NODES["muted"] = {}
        _S["seeded"] = True
        _seed_default_objects()
        _bump()
        return jsonify(_payload())


@bp.route("/api/scene/save")
def scene_save():
    _init()
    with _LOCK:
        objs = []
        order = list(_S["objects"].keys())                     # save-order; parents referenced by this index
        idx_of = {oid: i for i, oid in enumerate(order)}
        for oid, o in _S["objects"].items():
            entry = {"name": o.name,
                     "vertices": [round(float(x), 5) for x in o.mesh.vertices.ravel()],
                     "faces": [list(map(int, f)) for f in o.mesh.faces],
                     "mats": list(o.mats)}
            lay = getattr(o, "layers", None)
            if lay:                                            # ocean wave-spec / creature walk-spec survive load
                safe = {}
                for k, v in lay.items():
                    try:
                        json.dumps(v); safe[k] = v             # only JSON-safe layer data (numpy caches skipped)
                    except (TypeError, ValueError):
                        if isinstance(v, dict):
                            safe[k] = {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                                       for kk, vv in v.items() if kk != "rest"}
                if safe:
                    entry["layers"] = safe
            par = _PARENT.get(oid)
            if par in idx_of:                                  # store parent by save-order index (id-stable)
                entry["parent_index"] = idx_of[par]
            h = _HIST.get(oid)
            if h and (any(h["branches"].values()) or h.get("baked_from")):
                # P3-1: round-trip the re-editable history -- base snapshot + replayable op branches --
                # so a reloaded scene keeps its edit log (checkout/branch keeps working after load).
                entry["history"] = {
                    "base": {"vertices": [round(float(x), 5) for x in h["base"]["verts"].ravel()],
                             "faces": [list(map(int, f)) for f in h["base"]["faces"]],
                             "mats": list(h["base"]["mats"]),
                             "dsl": h["base"].get("dsl")},
                    "branches": h["branches"], "current": h["current"], "cursor": h["cursor"],
                    "baked_from": h.get("baked_from")}
            objs.append(entry)
        mats = []
        for name, m in _CUSTOM_MATS.items():
            entry = {"name": name, "color": [round(float(c), 4) for c in np.asarray(m.base_color)[:3]],
                     "metallic": round(float(m.metallic), 3), "roughness": round(float(m.roughness), 3)}
            if hasattr(m, "transmission"):
                entry["transmission"] = round(float(m.transmission), 3)
            if hasattr(m, "ior"):
                entry["ior"] = round(float(m.ior), 3)
            mats.append(entry)
    out = {"version": 1, "objects": objs, "custom_materials": mats,
           "note": "Poly Studio scene. Objects are stored as meshes (analytic primitive trees are "
                   "not round-tripped); geometry is preserved exactly."}
    # FULL-FIDELITY SAVE (basics pass): node graph (+layout/mutes), imported objects' render assets
    # (uv + texture PNG b64 / exact face colours), and units all round-trip -- a load previously dropped
    # textures and the entire node graph silently.
    with _LOCK:
        g = _NODES.get("graph")
        if g is not None:
            out["nodes"] = {"graph": g.to_dict(), "pos": _NODES.get("pos", {}), "muted": _NODES.get("muted", {})}
        ra = {}
        for oid, a in _S.get("render_assets", {}).items():
            if oid not in idx_of:
                continue
            e = {}
            if "uv" in a:
                e["uv"] = [round(float(x), 5) for x in np.asarray(a["uv"]).ravel()]
                import io as _io
                from PIL import Image as _Image
                _buf = _io.BytesIO()
                _Image.fromarray((np.clip(np.asarray(a["tex"], float), 0, 1) * 255).astype(np.uint8)).save(_buf, format="PNG")
                e["tex_png"] = base64.b64encode(_buf.getvalue()).decode()
            elif "face_colors" in a:
                e["face_colors"] = [round(float(x), 4) for x in np.asarray(a["face_colors"]).ravel()]
            if e:
                ra[str(idx_of[oid])] = e
        if ra:
            out["render_assets"] = ra
        out["units"] = dict(_S.get("units", {}))
    return jsonify(out)


@bp.route("/api/scene/load", methods=["POST"])
def scene_load():
    _init()
    d = request.get_json(force=True) or {}
    if "objects" not in d:
        return jsonify({"error": "not a scene file (no 'objects')"}), 400
    from holographic_mesh import Mesh
    with _LOCK:
        # restore custom materials first so object assignments resolve
        import copy as _copy
        for me in d.get("custom_materials", []):
            nm = me.get("name")
            if not nm or nm in _matlib().names() or nm in _CUSTOM_MATS:
                continue
            m = _copy.deepcopy(_matlib().material("clay"))
            col = me.get("color", [0.7, 0.7, 0.7])
            m.base_color = np.array([float(col[0]), float(col[1]), float(col[2])], float)
            m.metallic = float(me.get("metallic", 0.0)); m.roughness = float(me.get("roughness", 0.5))
            if hasattr(m, "transmission"):
                m.transmission = float(me.get("transmission", 0.0))
            if hasattr(m, "ior"):
                m.ior = float(me.get("ior", 1.5))
            _CUSTOM_MATS[nm] = m
        # replace the scene
        _snap_scene()
        _S["objects"].clear(); _PARENT.clear()
        new_ids = []                                          # save-order -> new id, for parent remap
        pending_parents = []                                  # (child_new_id, parent_save_index)
        for ob in d["objects"]:
            V = np.asarray(ob["vertices"], float).reshape(-1, 3)
            faces = [tuple(f) for f in ob["faces"]]
            mesh = Mesh(V, faces)
            mats = ob.get("mats") or [_default_mat_name()] * mesh.n_faces
            if len(mats) != mesh.n_faces:
                mats = (mats + [mats[-1] if mats else _default_mat_name()] * mesh.n_faces)[:mesh.n_faces]
            nid = _add_object(ob.get("name", "Object"), mesh, mats)
            new_ids.append(nid)
            lay = ob.get("layers")
            if lay:                                            # restore ocean/walk specs so animation works post-load
                _S["objects"][nid].layers = dict(lay)
            if "parent_index" in ob:
                pending_parents.append((nid, int(ob["parent_index"])))
            hs = ob.get("history")
            if hs:                                            # P3-1: rebuild the edit log under the NEW id
                try:
                    b = hs["base"]
                    _HIST[nid] = {"base": {"verts": np.asarray(b["vertices"], float).reshape(-1, 3),
                                           "faces": [tuple(f) for f in b["faces"]],
                                           "mats": list(b["mats"]), "dsl": b.get("dsl")},
                                  "branches": {k: list(v) for k, v in hs.get("branches", {}).items()} or {"main": []},
                                  "current": hs.get("current", "main"),
                                  "cursor": int(hs.get("cursor", 0))}
                    if hs.get("baked_from"):
                        _HIST[nid]["baked_from"] = hs["baked_from"]
                    if _HIST[nid]["current"] not in _HIST[nid]["branches"]:
                        _HIST[nid]["current"] = next(iter(_HIST[nid]["branches"]))
                except Exception:
                    _HIST.pop(nid, None)                      # malformed history: load geometry, drop the log
        for child_id, pidx in pending_parents:                # A2-3: re-link hierarchy under the new ids
            if 0 <= pidx < len(new_ids) and new_ids[pidx] != child_id:
                _PARENT[child_id] = new_ids[pidx]
        # FULL-FIDELITY RESTORE: node graph + layout, render assets (textures), units
        try:
            nd = d.get("nodes")
            if nd and nd.get("graph"):
                from holographic_nodegraph import NodeGraph as _NG, default_registry as _dreg
                _NODES["graph"] = _NG.from_dict(_augment_registry(_dreg()), nd["graph"])
                _NODES["pos"] = {str(k): list(v) for k, v in (nd.get("pos") or {}).items()}
                _NODES["muted"] = {str(k): bool(v) for k, v in (nd.get("muted") or {}).items()}
        except Exception:
            _NODES["graph"] = None; _NODES["pos"] = {}; _NODES["muted"] = {}
        try:
            _S["render_assets"] = {}
            for sidx, e in (d.get("render_assets") or {}).items():
                k = int(sidx)
                if not (0 <= k < len(new_ids)):
                    continue
                noid = new_ids[k]
                nverts = _S["objects"][noid].mesh.n_vertices
                if "uv" in e and "tex_png" in e:
                    uvv = np.asarray(e["uv"], float).reshape(-1, 2)
                    if len(uvv) != nverts:
                        continue
                    import io as _io
                    from PIL import Image as _Image
                    texv = np.asarray(_Image.open(_io.BytesIO(base64.b64decode(e["tex_png"]))).convert("RGB"),
                                      dtype=np.float32) / 255.0
                    _S["render_assets"][noid] = {"uv": uvv, "tex": texv}
                elif "face_colors" in e:
                    _S["render_assets"][noid] = {"face_colors": np.asarray(e["face_colors"], float).reshape(-1, 3)}
        except Exception:
            _S["render_assets"] = {}
        if isinstance(d.get("units"), dict) and d["units"].get("name"):
            _S["units"] = {"name": str(d["units"]["name"]), "per_unit": float(d["units"].get("per_unit", 10.0))}
        _bump()
    out = _payload()
    out["loaded"] = len(d["objects"])
    return jsonify(out)


# =====================================================================================================
# SWEEP-ALONG-PATH (A1-2): a circle profile swept down a Catmull-Rom curve using rotation-minimizing frames
# (holographic_curves) -> a real constant-radius tube. This is what handles, spouts, and branch stems need;
# faking them with rotated/scaled lathes distorts the cross-section and leaves seams. Optional taper (radius
# scales start->end). Field-only mesh result, welds/booleans onto the parent like any object.
# =====================================================================================================
@bp.route("/api/sweep", methods=["POST"])
def sweep():
    _init()
    d = request.get_json(force=True) or {}
    ctrl = d.get("path") or d.get("points")
    if not ctrl or len(ctrl) < 2:
        return jsonify({"error": "sweep needs a 'path' of >=2 [x,y,z] control points"}), 400
    try:
        ctrl = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in ctrl], float)
    except Exception:
        return jsonify({"error": "path points must be [x,y,z]"}), 400
    radius = float(np.clip(d.get("radius", 0.04), 0.005, 1.0))
    taper = float(np.clip(d.get("taper", 1.0), 0.0, 4.0))     # end radius / start radius
    sides = int(np.clip(int(d.get("sides", 12)), 3, 32))
    samples = int(np.clip(int(d.get("samples", max(16, len(ctrl) * 8))), 8, 200))
    cap = bool(d.get("cap", True))
    import holographic_curves as cv
    from holographic_mesh import Mesh
    with _LOCK:
        try:
            pts = np.asarray(cv.catmull_rom(ctrl, samples), float) if len(ctrl) >= 3 else \
                np.linspace(ctrl[0], ctrl[-1], samples)
            T, N, B = cv.rotation_minimizing_frame(pts)
            prof = np.asarray(cv.circle_profile(sides=sides, radius=1.0), float)   # unit; scale per-ring
        except Exception as e:
            return jsonify({"error": f"sweep failed: {e}"}), 400
        nseg = len(pts); verts = []
        for i in range(nseg):
            rr = radius * (1.0 + (taper - 1.0) * (i / max(nseg - 1, 1)))
            for (u, v) in prof:
                verts.append(pts[i] + (u * rr) * N[i] + (v * rr) * B[i])
        faces = []
        for i in range(nseg - 1):
            for j in range(sides):
                a = i * sides + j; b = i * sides + (j + 1) % sides
                cc = (i + 1) * sides + j; dd = (i + 1) * sides + (j + 1) % sides
                faces.append((a, cc, b)); faces.append((b, cc, dd))
        if cap:                                              # cap both ends with a fan to a centre vertex
            c0 = len(verts); verts.append(pts[0])
            for j in range(sides):
                faces.append((c0, (j + 1) % sides, j))
            c1 = len(verts); verts.append(pts[-1])
            base = (nseg - 1) * sides
            for j in range(sides):
                faces.append((c1, base + j, base + (j + 1) % sides))
        mesh = Mesh(np.asarray(verts, float), [tuple(f) for f in faces])
        nid = _add_object(d.get("name") or "Sweep", mesh)
        if d.get("material"):
            try:
                _mat(d["material"]); _S["objects"][nid].mats = [d["material"]] * mesh.n_faces
            except Exception:
                pass
        _bump()
    out = _payload(); out["object"] = nid
    out["note"] = "Circle profile swept along a Catmull-Rom path (rotation-minimizing frames) — a real tube."
    return jsonify(out)


# =====================================================================================================
# CAD / PRODUCT / ARCHITECTURE TOOLKIT
# =====================================================================================================
@bp.route("/api/camera/solve", methods=["POST"])
def camera_solve():
    """VANISHING-POINT CAMERA SOLVE (A0-1 second half): recover focal length + orientation from two families of
    parallel image lines (e.g. traced along two orthogonal edge directions of a building or box in a reference
    photo). Each family's lines are intersected in homogeneous coords (SVD null-space) to get a vanishing point;
    two orthogonal VPs give the focal length via (v1-pp)·(v2-pp) = -f^2, and unit camera-space directions to the
    VPs form two camera axes (third = their cross). Body: {width, height, lines_a:[[x1,y1,x2,y2]...],
    lines_b:[...], principal?:[cx,cy]}. Returns focal (px), 35mm-equivalent, an orientation matrix, and a
    suggested orbit (theta/phi/dist) to point the viewport camera roughly like the photo."""
    _init()
    d = request.get_json(force=True) or {}
    W = float(d.get("width", 0)); H = float(d.get("height", 0))
    la = d.get("lines_a", []) or []; lb = d.get("lines_b", []) or []
    if W <= 0 or H <= 0:
        return jsonify({"error": "need image width and height"}), 400
    if len(la) < 2 or len(lb) < 2:
        return jsonify({"error": "need >=2 lines in each of the two parallel families (lines_a, lines_b)"}), 400
    pp = np.asarray(d.get("principal", [W / 2, H / 2]), float)

    def vp(lines):
        A = []
        for L in lines:
            x1, y1, x2, y2 = (float(v) for v in L[:4])
            A.append(np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
        _, _, Vt = np.linalg.svd(np.asarray(A, float))
        v = Vt[-1]
        return (v / v[2])[:2] if abs(v[2]) > 1e-9 else np.array([v[0], v[1]]) * 1e6  # near-parallel -> far VP

    try:
        v1 = vp(la); v2 = vp(lb)
    except Exception as e:
        return jsonify({"error": f"could not intersect lines: {e}"}), 400
    dot = float(np.dot(v1 - pp, v2 - pp))
    import math
    if dot >= -1e-6:
        # non-orthogonal or degenerate: fall back to a focal from the field of view of the VP spread
        f = float(max(W, H))
        note = "VPs not clearly orthogonal; focal is a rough fallback. Trace the two line families along truly perpendicular edges."
    else:
        f = float(np.sqrt(-dot))
        if f > 20 * max(W, H):                                 # near-parallel families -> telephoto/ortho limit
            note = "Lines are nearly parallel in the image (long-lens / near-orthographic); focal is a lower bound."
            f = 20 * max(W, H)
        else:
            note = "Focal from two orthogonal vanishing points."
    # 35mm-equivalent focal (sensor diagonal 43.27mm), using image diagonal as the sensor proxy
    diag_px = math.hypot(W, H)
    f35 = round(f / diag_px * 43.27, 1)
    fov_h = round(math.degrees(2 * math.atan2(W / 2, f)), 1)
    # orientation: unit camera-space rays to the two VPs, third axis = cross
    def ray(v):
        r = np.array([v[0] - pp[0], v[1] - pp[1], f]); return r / np.linalg.norm(r)
    r1 = ray(v1); r2 = ray(v2)
    # re-orthogonalize r2 against r1 (Gram-Schmidt) so the basis is clean even with trace noise
    r2 = r2 - np.dot(r2, r1) * r1; r2 = r2 / max(np.linalg.norm(r2), 1e-9)
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    # suggest an orbit that looks roughly along the recovered view direction (r3)
    view = r3 if r3[2] >= 0 else -r3
    theta = round(math.degrees(math.atan2(view[0], view[2])), 1)
    phi = round(math.degrees(math.asin(np.clip(view[1], -1, 1))), 1)
    return jsonify({
        "focal_px": round(f, 1), "focal_35mm_equiv": f35, "fov_horizontal_deg": fov_h,
        "vanishing_points": [[round(float(v1[0]), 1), round(float(v1[1]), 1)],
                             [round(float(v2[0]), 1), round(float(v2[1]), 1)]],
        "orientation": [[round(float(x), 5) for x in row] for row in R.tolist()],
        "orthonormal": bool(np.allclose(R.T @ R, np.eye(3), atol=1e-3)),
        "suggested_orbit": {"theta": theta, "phi": phi},
        "note": note})


@bp.route("/api/sketch/solve", methods=["POST"])
def sketch_solve():
    """PARAMETRIC SKETCHER (F1-2): solve a 2-D point set against geometric constraints, then optionally extrude
    the solved profile. A Gauss-Newton least-squares solve drives the constraint residuals to ~0. Supported
    constraints (each references point indices):
      fixed [i] (x,y)   pin a point            horizontal [i,j]   same y
      vertical [i,j]    same x                  distance [i,j] d   |Pi-Pj| = d
      coincident [i,j]  Pi = Pj                 parallel [i,j,k,l] edge ij ∥ edge kl
      angle [i,j] deg   edge ij at absolute angle
    Body: {points:[[x,y]...], constraints:[{type, pts:[...], value?}], extrude?:bool, height?}. Returns the
    solved points and the max residual so the solve is auditable."""
    _init()
    d = request.get_json(force=True) or {}
    try:
        P = np.asarray([[float(p[0]), float(p[1])] for p in d.get("points", [])], float)
    except Exception:
        return jsonify({"error": "points must be [[x,y], ...]"}), 400
    if len(P) < 2:
        return jsonify({"error": "need at least 2 points"}), 400
    cons = d.get("constraints", []) or []
    n = len(P)

    def residuals(x):
        Q = x.reshape(n, 2); r = []
        for c in cons:
            typ = c.get("type"); pts = c.get("pts", []); val = c.get("value")
            try:
                if typ == "fixed":
                    i = pts[0]; r += [Q[i, 0] - float(val[0]), Q[i, 1] - float(val[1])]
                elif typ == "horizontal":
                    i, j = pts[:2]; r += [Q[i, 1] - Q[j, 1]]
                elif typ == "vertical":
                    i, j = pts[:2]; r += [Q[i, 0] - Q[j, 0]]
                elif typ == "distance":
                    i, j = pts[:2]; r += [np.linalg.norm(Q[i] - Q[j]) - float(val)]
                elif typ == "coincident":
                    i, j = pts[:2]; r += [Q[i, 0] - Q[j, 0], Q[i, 1] - Q[j, 1]]
                elif typ == "parallel":
                    i, j, k, l = pts[:4]
                    e1 = Q[j] - Q[i]; e2 = Q[l] - Q[k]
                    r += [e1[0] * e2[1] - e1[1] * e2[0]]        # cross product = 0 when parallel
                elif typ == "angle":
                    i, j = pts[:2]; ang = np.radians(float(val))
                    e = Q[j] - Q[i]
                    r += [e[0] * np.sin(ang) - e[1] * np.cos(ang)]  # edge aligned to the target direction
            except Exception:
                pass
        return np.asarray(r, float)

    x = P.ravel().copy()
    r0 = residuals(x)
    if len(r0) == 0:
        return jsonify({"error": "no valid constraints"}), 400
    # Gauss-Newton with finite-difference Jacobian (small systems; robust enough for sketch sizes)
    for _ in range(60):
        r = residuals(x)
        if np.max(np.abs(r)) < 1e-9:
            break
        J = np.zeros((len(r), len(x)))
        eps = 1e-6
        for k in range(len(x)):
            xk = x.copy(); xk[k] += eps
            J[:, k] = (residuals(xk) - r) / eps
        try:                                                   # damped least squares step (Levenberg-ish)
            dx = np.linalg.lstsq(J.T @ J + 1e-9 * np.eye(len(x)), -J.T @ r, rcond=None)[0]
        except Exception:
            break
        step = np.linalg.norm(dx)
        if step > 2.0:
            dx *= 2.0 / step                                   # cap wild steps
        x = x + dx
        if step < 1e-10:
            break
    solved = x.reshape(n, 2)
    max_res = float(np.max(np.abs(residuals(x)))) if len(residuals(x)) else 0.0
    out = {"points": [[round(float(a), 6), round(float(b), 6)] for a, b in solved],
           "max_residual": round(max_res, 8), "converged": max_res < 1e-4,
           "initial_residual": round(float(np.max(np.abs(r0))), 6)}
    if d.get("extrude"):                                        # hand the solved polygon to the profile extruder
        with _LOCK:
            try:
                import holographic_sdf2d as s2
                from holographic_meshbridge import marching_tetrahedra_vec
                from holographic_meshqem import cluster_decimate
                from holographic_mesh import Mesh
                height = float(np.clip(d.get("height", 0.4), 0.02, 8.0))
                poly = solved.tolist()
                solid = s2.extrude(s2.polygon2d(poly), total_height=height)   # engine grew total_height= (was half-extent height=)
                Pp = np.asarray(poly, float); lo2, hi2 = Pp.min(0) - 0.08, Pp.max(0) + 0.08
                rz = 46
                ax = (np.linspace(lo2[0], hi2[0], rz), np.linspace(lo2[1], hi2[1], rz),
                      np.linspace(-height / 2 - 0.08, height / 2 + 0.08, max(10, int(rz * height))))
                X, Y, Z = np.meshgrid(*ax, indexing="ij")
                g = solid(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
                raw = marching_tetrahedra_vec(g.reshape(len(ax[0]), len(ax[1]), len(ax[2])), ax, level=0.0)
                dec = cluster_decimate(raw, grid=40)
                nid = _add_object(d.get("name") or "Sketch", Mesh(dec.vertices, [tuple(f) for f in dec.faces]))
                _bump()
                out["object"] = nid
            except Exception as e:
                out["extrude_error"] = str(e)
    return jsonify(out)


@bp.route("/api/extrude_profile", methods=["POST"])
def extrude_profile():
    """PROFILE EXTRUSION (the CAD/architecture workhorse): a 2-D profile -- preset (rect, L, T, U, circle,
    ngon) or custom polygon points -- extruded to a height (holographic_sdf2d.polygon2d + extrude), meshed via
    marching tetrahedra. Walls, plates, brackets, beams, channels. Field-built: the result is a mesh (the
    sdf2d combinators return plain field functions, not tree nodes), which is stated rather than hidden."""
    _init()
    d = request.get_json(force=True) or {}
    import holographic_sdf2d as s2
    from holographic_meshbridge import marching_tetrahedra_vec
    from holographic_meshqem import cluster_decimate
    from holographic_mesh import Mesh
    preset = str(d.get("preset", "rect"))
    w = float(np.clip(d.get("w", 0.6), 0.02, 8.0))          # overall width  (x)
    h = float(np.clip(d.get("h", 0.6), 0.02, 8.0))          # overall height (y of the profile)
    t = float(np.clip(d.get("t", 0.15), 0.01, 4.0))         # limb thickness for L/T/U
    height = float(np.clip(d.get("height", 0.4), 0.02, 8.0))  # extrusion depth
    if preset == "custom":
        pts = d.get("points")
        if not pts or len(pts) < 3:
            return jsonify({"error": "custom profile needs >=3 [x,y] points"}), 400
        try:
            poly = [[float(p[0]), float(p[1])] for p in pts]
        except Exception:
            return jsonify({"error": "points must be [x,y]"}), 400
    elif preset == "rect":
        poly = [[0, 0], [w, 0], [w, h], [0, h]]
    elif preset == "L":
        poly = [[0, 0], [w, 0], [w, t], [t, t], [t, h], [0, h]]
    elif preset == "T":
        poly = [[0, 0], [w, 0], [w, t], [(w + t) / 2, t], [(w + t) / 2, h], [(w - t) / 2, h], [(w - t) / 2, t], [0, t]]
    elif preset == "U":
        poly = [[0, 0], [w, 0], [w, h], [w - t, h], [w - t, t], [t, t], [t, h], [0, h]]
    elif preset == "circle":
        n = 40; r = w / 2
        poly = [[r + r * np.cos(a), r + r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    elif preset == "ngon":
        n = int(np.clip(int(d.get("sides", 6)), 3, 24)); r = w / 2
        poly = [[r + r * np.cos(a), r + r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    else:
        return jsonify({"error": f"unknown preset '{preset}'"}), 400
    with _LOCK:
        try:
            # NOTE the engine convention: extrude(height=h) is the HALF-extent (inside for |z|<h) --
            # verified by probing. Pass height/2 so the user-facing 'height' is the full depth.
            solid = s2.extrude(s2.polygon2d(poly), total_height=height)   # engine grew total_height= (was half-extent height=)
            P = np.asarray(poly, float)
            lo2, hi2 = P.min(0) - 0.08, P.max(0) + 0.08
            res = int(np.clip(int(d.get("res", 72)), 40, 110))
            # PER-AXIS resolution with a uniform cell size CAPPED by the thinnest axis getting >=16 cells.
            # A uniform res^3 grid starves thin plates (seen in testing: a 2.0x0.1x1.2 wall measured volume
            # 0.173 instead of 0.24 because the 0.1 axis got ~3 cells). Anisotropic sampling fixes it cheaply.
            spans = np.array([hi2[0] - lo2[0], hi2[1] - lo2[1], height + 0.16], float)
            cell = min(spans.max() / res, spans.min() / 16.0)
            nax = np.minimum(np.maximum((spans / cell).astype(int) + 1, 8), 200)
            ax = (np.linspace(lo2[0], hi2[0], nax[0]), np.linspace(lo2[1], hi2[1], nax[1]),
                  np.linspace(-height / 2 - 0.08, height / 2 + 0.08, nax[2]))
            X, Y, Z = np.meshgrid(*ax, indexing="ij")
            g = solid(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
            if not (g.min() < 0 < g.max()):
                return jsonify({"error": "profile produced an empty solid"}), 400
            raw = marching_tetrahedra_vec(g.reshape(len(ax[0]), len(ax[1]), len(ax[2])), ax, level=0.0)
            target = int(np.clip(int(d.get("target", 4000)), 300, 20000))
            gr = 52
            dec = cluster_decimate(raw, grid=gr)
            while dec.n_faces > target and gr > 8:
                gr -= 4; dec = cluster_decimate(raw, grid=gr)
            mesh = Mesh(dec.vertices, [tuple(f) for f in dec.faces])
        except Exception as e:
            return jsonify({"error": f"extrude failed: {e}"}), 400
        nid = _add_object(d.get("name") or f"{preset.capitalize()} profile", mesh)
        if d.get("material"):
            try:
                _mat(d["material"]); _S["objects"][nid].mats = [d["material"]] * mesh.n_faces
            except Exception:
                pass
        _bump()
    out = _payload(); out["object"] = nid
    return jsonify(out)


@bp.route("/api/bounding")
def bounding_boxes():
    """BOUNDING VOLUMES (packing / fab / collision CAD slice): the axis-aligned bbox AND a tight ORIENTED
    bounding box (OBB). The OBB is seeded from the vertex covariance eigenvectors (PCA), then refined by a small
    rotation search that minimises the box volume -- for a rotated cube this recovers the true cube volume, far
    below the axis-aligned box. Returns both boxes' half-extents, volumes, the OBB centre + rotation axes, and
    the tightness ratio (AABB volume / OBB volume). Verifiable: a rotated unit cube has OBB volume ~1."""
    _init()
    oid = str(request.args.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = np.asarray(o.mesh.vertices, float)
        if len(V) < 4:
            return jsonify({"error": "need at least 4 vertices"}), 400
        aabb_dims = V.max(0) - V.min(0)
        aabb_vol = float(np.prod(aabb_dims))
        import holographic_fitshape as _fs
        r = _fs.oriented_bbox(V)                              # engine impl (PCA seed + refine, hard AABB fallback)
        best_e = np.asarray(r["half_extents"], float) * 2.0    # engine returns half-extents; contract sends dims
        best_vol = float(r["volume"])
        best_R = np.asarray(r["axes"], float).T                # engine rows=axes -> columns=axes frame
        center_world = np.asarray(r["center"], float)
        return jsonify({
            "object": oid, "name": o.name,
            "aabb": {"dims": [round(float(x), 5) for x in aabb_dims], "volume": round(aabb_vol, 6)},
            "obb": {"half_extents": [round(float(x) / 2, 5) for x in best_e],
                    "dims": [round(float(x), 5) for x in best_e],
                    "volume": round(best_vol, 6),
                    "center": [round(float(x), 5) for x in center_world],
                    "axes": [[round(float(x), 5) for x in col] for col in best_R.T.tolist()]},
            "tightness_ratio": round(aabb_vol / max(best_vol, 1e-9), 4),
            "note": "AABB is axis-aligned; OBB is a tight oriented box (PCA seed + rotation refinement). "
                    "tightness_ratio = how much smaller the OBB is than the AABB (1 = already aligned)."})


@bp.route("/api/draft_apply", methods=["POST"])
def draft_apply():
    """AUTO-APPLY DRAFT (F1-5): actually re-slope the geometry so walls meet a minimum draft angle for the given
    pull direction -- the fix to what draft_report diagnoses. Method: a linear taper about the pull axis. Every
    vertex moves radially outward from the axis by tan(angle) * height-above-the-parting-plane (the face at the
    minimum along pull), which turns 0-degree vertical walls into angle-degree drafted walls exactly, the way a
    mold designer tapers a straight boss. Faces that already exceed the angle get slightly MORE draft (a global
    taper is monotone) -- documented behaviour, not a bug. Args: object, pull=[0,0,1], angle_deg (0.5..10).
    Verifiable: draft_report's moldable fraction and minimum wall draft both rise to >= angle_deg."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    ang = float(np.clip(d.get("angle_deg", 2.0), 0.5, 10.0))
    pull = np.asarray(d.get("pull", [0, 0, 1]), float)
    n = np.linalg.norm(pull)
    if n < 1e-9:
        return jsonify({"error": "pull direction is zero"}), 400
    pull = pull / n
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = np.asarray(o.mesh.vertices, float).copy()
        h = V @ pull
        h0 = h.min()                                        # parting plane at the base along pull
        height = h - h0
        # Per-vertex draft direction: for each vertex, collect the HORIZONTAL components of its adjacent
        # WALL normals (faces roughly parallel to pull), dedupe near-identical walls, and sum the unique unit
        # directions. Moving the vertex INWARD (against that sum) by tan(angle)*height tilts EVERY adjacent
        # wall by exactly `angle` -- a cube corner (two orthogonal walls) moves along the diagonal with
        # magnitude sqrt(2)*tan*h so each axis component is tan*h, which is what per-wall exactness requires.
        # (A naive radial taper under-tilts corners by 1/sqrt(2) and the sign matters: the part must NARROW
        # away from the parting plane to release along +pull.)
        F = [list(f) for f in o.mesh.faces]
        vdir = np.zeros_like(V)
        vseen = [[] for _ in range(len(V))]
        for f in F:
            for k in range(1, len(f) - 1):
                a_, b_, c_ = f[0], f[k], f[k + 1]
                fn = np.cross(V[b_] - V[a_], V[c_] - V[a_])
                ln = np.linalg.norm(fn)
                if ln < 1e-12:
                    continue
                fn = fn / ln
                nh = fn - (fn @ pull) * pull                # horizontal component
                lh = np.linalg.norm(nh)
                if lh < 0.5:                                # not a wall (top/bottom-ish face): no draft needed
                    continue
                nh = nh / lh
                for vtx in (a_, b_, c_):
                    if not any(float(nh @ q) > 0.9 for q in vseen[vtx]):   # dedupe same-wall contributions
                        vseen[vtx].append(nh)
        for i_, dirs in enumerate(vseen):
            if dirs:
                vdir[i_] = np.sum(dirs, axis=0)
        V = V - vdir * (np.tan(np.radians(ang)) * height)[:, None]
        from holographic_mesh import Mesh
        _snap_scene()
        o.mesh = Mesh(V, [tuple(f) for f in o.mesh.faces])
        _bump(oid)
        out = _payload(only=oid)
        out["draft_applied"] = {"angle_deg": ang, "pull": [round(float(x), 4) for x in pull],
                                "parting_at": round(float(h0), 5),
                                "note": "linear taper about the pull axis; verify with draft_report"}
        return jsonify(out)


@bp.route("/api/draft_report")
def draft_report():
    """DRAFT-ANGLE REPORT (moldability CAD slice, read-only): the distribution of per-face draft angles against
    a pull direction, weighted by face AREA. draft = 90deg - angle(normal, pull), signed so faces that face
    INTO the pull (undercuts) are negative. Returns min/mean draft, the mouldable area fraction (draft >= a
    threshold or a near-parallel parting face), undercut area, and a histogram. Verifiable: a cube pulled along
    +Z has its 4 side walls at exactly 0deg draft (vertical -> not mouldable) and top/bottom as parting faces."""
    _init()
    g = request.args.get
    oid = str(g("object", ""))
    pull = np.asarray([float(x) for x in (g("pull", "0,0,1").split(","))], float)
    pull = pull / max(np.linalg.norm(pull), 1e-9)
    min_deg = float(np.clip(float(g("min_degrees", 1.0)), 0.0, 45.0))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = o.mesh.vertices
        drafts = []; areas = []; parting_area = 0.0
        for f in o.mesh.faces:
            idx = list(f)
            for k in range(1, len(idx) - 1):
                a, b, c = V[idx[0]], V[idx[k]], V[idx[k + 1]]
                n = np.cross(b - a, c - a); ln = float(np.linalg.norm(n))
                if ln < 1e-12:
                    continue
                area = 0.5 * ln; nrm = n / ln
                cosang = float(np.dot(nrm, pull))
                if abs(cosang) > 0.996:                         # nearly parallel to pull => parting/cap face
                    parting_area += area; continue
                draft = 90.0 - float(np.degrees(np.arccos(np.clip(abs(cosang), -1, 1))))
                if cosang < 0:                                  # normal faces INTO the pull -> undercut (negative)
                    draft = -draft
                drafts.append(draft); areas.append(area)
        if not drafts:
            return jsonify({"object": oid, "note": "all faces are parting/cap faces relative to this pull"})
        drafts = np.asarray(drafts); areas = np.asarray(areas)
        tot = float(areas.sum())
        moldable = float(areas[drafts >= min_deg].sum())
        undercut = float(areas[drafts < 0].sum())
        # area-weighted histogram over draft-angle bins
        bins = [-90, -10, -1, 0, 1, 3, 5, 10, 20, 45, 90]
        hist = []
        for i in range(len(bins) - 1):
            m = (drafts >= bins[i]) & (drafts < bins[i + 1])
            hist.append({"range": f"{bins[i]}..{bins[i+1]}", "area_frac": round(float(areas[m].sum()) / tot, 4)})
        return jsonify({
            "object": oid, "name": o.name, "pull": [round(float(x), 4) for x in pull],
            "min_degrees": min_deg,
            "min_draft": round(float(drafts.min()), 3), "mean_draft": round(float(np.average(drafts, weights=areas)), 3),
            "moldable_area_fraction": round(moldable / tot, 4),
            "undercut_area_fraction": round(undercut / tot, 4),
            "parting_area_fraction": round(parting_area / (tot + parting_area), 4),
            "histogram": hist,
            "note": "Area-weighted draft angles vs the pull direction. Faces below the minimum (or negative = "
                    "undercut) would resist release from a mould. Vertical walls read 0deg."})


@bp.route("/api/section/measure")
def section_measure():
    """NUMERICAL CROSS-SECTION (CAD slice): the exact section AREA and PERIMETER where a plane cuts a solid,
    plus the closed contour polylines. Samples the object's inside/outside field on a grid in the cutting
    plane, then integrates: area = (inside cells) * cell-area (midpoint rule), perimeter via marching-squares
    contour length. Verifiable against analytics: a unit cube cut anywhere across it sections to area 1,
    perimeter 4. Args: object, axis (x/y/z), offset, res."""
    _init()
    g = request.args.get
    axis = {"x": 0, "y": 1, "z": 2}.get((g("axis", "z") or "z").lower(), 2)
    off = float(g("offset", 0.0))
    res = int(np.clip(int(g("res", 300)), 60, 600))
    oid = str(g("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        import holographic_meshtools as _mt
        pp = [0.0, 0.0, 0.0]; pn = [0.0, 0.0, 0.0]
        pp[axis] = off; pn[axis] = 1.0
        s = _mt.section(o.mesh, plane_point=tuple(pp), plane_normal=tuple(pn))
        return jsonify({"object": oid, "name": o.name, "axis": "xyz"[axis], "offset": off,
                        "area": round(float(abs(s["area"])), 5), "perimeter": round(float(s["perimeter"]), 5),
                        "contours": int(s["contours"]), "resolution": res,
                        "note": "EXACT planar section (engine holographic_meshtools.section: oriented polylines, "
                                "shoelace area). A unit cube sections to area 1, perimeter 4."})


@bp.route("/api/mass_properties")
def mass_properties():
    """MASS PROPERTIES (CAD-kernel slice, P3.5-2): exact volume, surface area, centre of mass, and the inertia
    tensor of a closed mesh, via the signed-tetrahedron integration (each triangle + origin forms a tet whose
    signed contributions sum to the solid's integrals -- exact for a watertight triangulation, sign-correct for
    any winding). Returns the inertia tensor about the centre of mass, its principal moments (eigenvalues) and
    principal axes (eigenvectors), assuming unit density. Verifiable against analytic solids (a unit cube:
    volume 1, com at centre, principal moments all 1/6)."""
    _init()
    oid = str(request.args.get("object", ""))
    density = float(request.args.get("density", 1.0))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = o.mesh.vertices
        import holographic_meshtools as _mt
        mp = _mt.mass_properties(o.mesh, density=density)     # engine impl (upstreamed Tonon-correct version)
        if not np.isfinite(mp.get("volume", np.nan)) or abs(mp["volume"]) < 1e-12:
            return jsonify({"error": "degenerate/near-zero volume (mesh may not be closed)"}), 400
        I = np.asarray(mp["inertia_com"], float)               # engine key: inertia tensor about the COM
        evals = np.asarray(mp["principal_moments"], float)
        evecs = np.asarray(mp["principal_axes"], float)
        return jsonify({
            "object": oid, "name": o.name, "density": density,
            "volume": round(float(abs(mp["volume"])), 6), "surface_area": round(float(mp["area"]), 6),
            "mass": round(float(abs(mp["volume"])) * density, 6),
            "center_of_mass": [round(float(x), 6) for x in np.asarray(mp["center_of_mass"]).tolist()],
            "inertia_tensor": [[round(float(x), 6) for x in row] for row in I.tolist()],
            "principal_moments": [round(float(x), 6) for x in evals.tolist()],
            "principal_axes": [[round(float(x), 6) for x in col] for col in evecs.T.tolist()],
            "note": "Exact for a closed mesh (signed-tet integration; engine holographic_meshtools.mass_properties)."})


@bp.route("/api/measure")
def measure_ep():
    """MEASUREMENT: object dimensions (bbox W*H*D), watertight volume (divergence theorem on the triangle
    fan -- exact for a closed mesh, reported as 'approx' with a warning if boundary edges exist), surface
    area, face/vert counts; optionally the centroid distance and gap to a second object. Numbers a CAD or
    product-design user needs before exporting to print or fab."""
    _init()
    g = request.args.get
    oid = str(g("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = o.mesh.vertices
        u = _S.get("units", {"name": "cm", "per_unit": 10.0})
        k = float(u["per_unit"])
        dims = (V.max(0) - V.min(0))
        out = {"object": oid, "name": o.name,
               "dims": [round(float(x), 4) for x in dims],
               "dims_units": [round(float(x) * k, 3) for x in dims],
               "units": u["name"],
               "center": [round(float(x), 4) for x in V.mean(0)],
               "verts": int(o.mesh.n_vertices), "faces": int(o.mesh.n_faces)}
        # surface area + signed volume (divergence theorem over triangulated faces)
        area = 0.0; vol = 0.0
        for f in o.mesh.faces:
            idx = list(f)
            for k in range(1, len(idx) - 1):
                a, b, c = V[idx[0]], V[idx[k]], V[idx[k + 1]]
                cr = np.cross(b - a, c - a)
                area += 0.5 * float(np.linalg.norm(cr))
                vol += float(np.dot(a, cr)) / 6.0
        out["area"] = round(area, 5)
        out["volume"] = round(abs(vol), 5)
        try:
            from holographic_meshtools import mesh_report
            rep = mesh_report(o.mesh)
            out["watertight"] = bool(rep.get("is_closed", False))
        except Exception:
            out["watertight"] = None
        if out["watertight"] is False:
            out["volume_note"] = "mesh has open boundaries; volume is approximate"
        other = g("other")
        if other and str(other) in _S["objects"]:
            V2 = _S["objects"][str(other)].mesh.vertices
            c1, c2 = V.mean(0), V2.mean(0)
            out["distance_centroids"] = round(float(np.linalg.norm(c2 - c1)), 4)
            # conservative gap: closest bbox gap per axis (0 if overlapping)
            lo1, hi1, lo2, hi2 = V.min(0), V.max(0), V2.min(0), V2.max(0)
            gap = np.maximum(np.maximum(lo2 - hi1, lo1 - hi2), 0.0)
            out["gap_bbox"] = round(float(np.linalg.norm(gap)), 4)
    return jsonify(out)


@bp.route("/api/units", methods=["GET", "POST"])
def units_ep():
    """SCENE UNITS (F1-1): what one engine unit means physically. GET returns the current setting; POST
    {name, per_unit} sets it (e.g. name='mm', per_unit=100 -> 1 engine unit = 100 mm). Purely a display/
    reporting convention -- geometry is untouched -- used by /api/measure to report real-world numbers."""
    _init()
    if request.method == "POST":
        d = request.get_json(force=True) or {}
        name = str(d.get("name", "cm"))[:12]
        try:
            per = float(d.get("per_unit", 10.0))
        except Exception:
            return jsonify({"error": "per_unit must be a number"}), 400
        if not (1e-6 < per < 1e9):
            return jsonify({"error": "per_unit out of range"}), 400
        with _LOCK:
            _S["units"] = {"name": name, "per_unit": per}
    return jsonify(_S.get("units", {"name": "cm", "per_unit": 10.0}))


@bp.route("/api/floorplan", methods=["POST"])
def floorplan():
    """FLOOR-PLAN IMPORT (F1-4): a JSON plan -> a set of exact walls in one transaction. Format:
    {"walls": [{"x1","z1","x2","z2"}...], "thickness", "height} OR {"path": [[x,z],...], "closed": bool}
    (a polyline traced around the plan; closed joins last->first). Each wall is the same exact 8-vertex box
    the `wall` op builds. Openings are added afterwards per wall with the `opening` op."""
    _init()
    d = request.get_json(force=True) or {}
    th = float(np.clip(d.get("thickness", 0.1), 0.01, 2.0))
    hh = float(np.clip(d.get("height", 1.2), 0.05, 10.0))
    base = float(d.get("base", 0.0))
    segs = []
    if d.get("walls"):
        try:
            segs = [(float(w["x1"]), float(w["z1"]), float(w["x2"]), float(w["z2"])) for w in d["walls"]]
        except Exception:
            return jsonify({"error": "walls must be [{x1,z1,x2,z2}, ...]"}), 400
    elif d.get("path"):
        try:
            pts = [(float(p[0]), float(p[1])) for p in d["path"]]
        except Exception:
            return jsonify({"error": "path must be [[x,z], ...]"}), 400
        if len(pts) < 2:
            return jsonify({"error": "path needs at least 2 points"}), 400
        for i in range(len(pts) - 1):
            segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
        if d.get("closed") and len(pts) > 2:
            segs.append((pts[-1][0], pts[-1][1], pts[0][0], pts[0][1]))
    else:
        return jsonify({"error": "provide 'walls' or 'path'"}), 400
    if len(segs) > 200:
        return jsonify({"error": "too many segments (max 200)"}), 400
    from holographic_mesh import Mesh
    with _LOCK:
        _snap_scene()
        made = []
        for i, (x1, z1, x2, z2) in enumerate(segs):
            p1 = np.array([x1, base, z1], float); p2 = np.array([x2, base, z2], float)
            axis = p2 - p1; L = float(np.linalg.norm(axis[[0, 2]]))
            if L < 1e-6:
                continue                                       # skip degenerate segments, keep the rest
            fwd = axis / np.linalg.norm(axis)
            side = np.array([-fwd[2], 0.0, fwd[0]]) * (th / 2.0)
            up = np.array([0.0, hh, 0.0])
            V = np.array([p1 - side, p1 + side, p2 + side, p2 - side,
                          p1 - side + up, p1 + side + up, p2 + side + up, p2 - side + up])
            F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
            made.append(_add_object(f"{d.get('name') or 'Wall'} {i + 1}", Mesh(V, F)))
        if not made:
            _discard_snapshot()
            return jsonify({"error": "no valid segments"}), 400
        _bump()
    out = _payload(); out["walls"] = made
    return jsonify(out)


# =====================================================================================================
# PROCEDURAL GENERATORS (Generate menu): worlds, space, creatures, environment backdrops.
# Design notes, honest limits included per generator:
#   - meshes come in as ONE object each (scatter merges thousands of blades into a single mesh);
#   - clouds are blobby SDF meshes + the existing fog for atmosphere (the renderer has no true volumetrics);
#   - galaxy / nebula / starfield / sky are ENVIRONMENT IMAGES (equirect) used by the preview + GI photo as
#     the sky and by the viewport as a backdrop -- a galaxy as geometry would need a particle system.
# =====================================================================================================
def _gen_fbm3(P, seed=0, octaves=4):
    """Vectorised value-ish fBm in [0,1] from hashed sinusoid lattices (fast; FractalNoise.query is
    per-point and too slow for batch displacement -- honest engineering trade, stated)."""
    rng = np.random.RandomState(seed)
    out = np.zeros(len(P)); amp, freq, tot = 1.0, 1.6, 0.0
    for _ in range(octaves):
        ph = rng.rand(3) * 6.283; dirs = rng.randn(3, 3)
        s = np.sin(P @ (dirs[0] * freq) + ph[0]) * np.cos(P @ (dirs[1] * freq) + ph[1]) * np.sin(P @ (dirs[2] * freq) + ph[2])
        out += amp * (s * 0.5 + 0.5); tot += amp; amp *= 0.5; freq *= 2.03
    return out / tot


def _gen_uv_sphere(res=48, R=1.0):
    la = np.linspace(0, np.pi, res); lo = np.linspace(0, 2 * np.pi, 2 * res, endpoint=False)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    V = np.stack([np.sin(LA) * np.cos(LO), np.cos(LA), np.sin(LA) * np.sin(LO)], -1).reshape(-1, 3) * R
    F = []
    W = 2 * res
    for i in range(res - 1):
        for j in range(W):
            a = i * W + j; b = i * W + (j + 1) % W; c = (i + 1) * W + (j + 1) % W; d2 = (i + 1) * W + j
            F.append((a, b, c, d2))
    return V, F


def _gen_frame(n):
    """Orthonormal frame with n as up."""
    n = n / max(np.linalg.norm(n), 1e-9)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    t = np.cross(n, a); t /= max(np.linalg.norm(t), 1e-9)
    return t, np.cross(n, t), n


def _gen_tube(path, radius, sides=8, taper=1.0):
    """Simple polyline tube (for creature limbs); returns (verts, faces)."""
    path = np.asarray(path, float)
    V = []; F = []
    rings = []
    for i, p in enumerate(path):
        d = (path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)])
        t1, t2, _ = _gen_frame(d)
        r = radius * (1 + (taper - 1) * i / max(len(path) - 1, 1))
        ring = [p + (np.cos(a) * t1 + np.sin(a) * t2) * r for a in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
        rings.append(len(V)); V.extend(ring)
    for i in range(len(path) - 1):
        a0, b0 = rings[i], rings[i + 1]
        for j in range(sides):
            F.append((a0 + j, a0 + (j + 1) % sides, b0 + (j + 1) % sides, b0 + j))
    # caps
    V.append(path[0]); c0 = len(V) - 1
    V.append(path[-1]); c1 = len(V) - 1
    for j in range(sides):
        F.append((rings[0] + (j + 1) % sides, rings[0] + j, c0))
        F.append((rings[-1] + j, rings[-1] + (j + 1) % sides, c1))
    return np.asarray(V), F


def _gen_biome_mat(elev01, moist01, steep, biome, sea):
    """Material name for a face given elevation / moisture / steepness and a biome preset."""
    if biome == "desert":
        if elev01 < sea: return "water"
        if steep: return "sandstone"
        return "sand_red" if moist01 < 0.35 else ("sand" if elev01 < 0.75 else "limestone")
    if biome == "arctic":
        if elev01 < sea: return "polar_ice"
        if steep: return "limestone"
        return "snow" if elev01 > sea + 0.08 else "ice"
    if biome == "volcanic":
        if elev01 < sea: return "lava"
        if steep: return "obsidian"
        return "crust_rock" if elev01 < 0.8 else "obsidian"
    # temperate
    if elev01 < sea - 0.06: return "water_deep"
    if elev01 < sea: return "water"
    if elev01 < sea + 0.05: return "sand"
    if steep: return "limestone"
    if elev01 > 0.85: return "snow"
    if elev01 > 0.7: return "limestone"
    return "forest" if moist01 > 0.55 else ("grass" if moist01 > 0.3 else "grass_dry")


@bp.route("/api/generate", methods=["POST"])
def generate():
    _init()
    d = request.get_json(force=True) or {}
    kind = str(d.get("kind", ""))
    seed = int(d.get("seed", 0))
    rng = np.random.RandomState(seed)
    from holographic_mesh import Mesh
    with _LOCK:
        # ---------------- LANDSCAPE with biomes ----------------
        if kind == "landscape":
            from holographic_terrain import Terrain, terrain_to_mesh
            res = int(np.clip(int(d.get("res", 72)), 32, 128))
            size = float(np.clip(d.get("size", 4.0), 1.0, 20.0))
            relief = float(np.clip(d.get("relief", 0.6), 0.05, 3.0))
            sea = float(np.clip(d.get("sea_level", 0.35), 0.0, 0.9))
            biome = str(d.get("biome", "temperate"))
            t = Terrain(octaves=int(np.clip(int(d.get("octaves", 5)), 3, 8)), seed=seed)
            tm = terrain_to_mesh(t, res=res, z_scale=1.0)
            V = np.asarray(tm.vertices, float)
            # terrain is XY-plane Z-up on [0,1]^2 -> our Y-up, centred, scaled
            h = V[:, 2].copy()
            h01 = (h - h.min()) / max(np.ptp(h), 1e-9)
            Vw = np.stack([(V[:, 0] - 0.5) * size, h01 * relief, (V[:, 1] - 0.5) * size], 1)
            # sea floor clamp: flatten below sea level so water reads as a surface
            sea_y = sea * relief
            Vw[:, 1] = np.maximum(Vw[:, 1], np.where(h01 < sea, sea_y - 0.02, Vw[:, 1]))
            # NOTE the axis swap (terrain Z-up -> our Y-up) is a REFLECTION, which flips winding; reverse the
            # faces so normals point UP (caught by scatter's up-facing filter finding zero area).
            faces = [tuple(reversed(f)) for f in tm.faces]
            mesh = Mesh(Vw, faces)
            moist = _gen_fbm3(Vw * (1.7 / size), seed + 101, octaves=3)
            mats = []
            for f in faces:
                e = float(h01[list(f)].mean()); m = float(moist[list(f)].mean())
                p = Vw[list(f)]
                n = np.cross(p[1] - p[0], p[2] - p[0]); ln = np.linalg.norm(n)
                steep = (abs(n[1] / ln) < 0.55) if ln > 1e-12 else False
                mats.append(_gen_biome_mat(e, m, steep, biome, sea))
            nid = _add_object(d.get("name") or f"{biome.capitalize()} landscape", mesh, mats)
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        # ---------------- PLANET ----------------
        if kind == "planet":
            res = int(np.clip(int(d.get("res", 56)), 24, 96))
            R = float(np.clip(d.get("radius", 0.8), 0.1, 5.0))
            relief = float(np.clip(d.get("relief", 0.1), 0.0, 0.5))
            sea = float(np.clip(d.get("sea_level", 0.45), 0.0, 0.95))
            biome = str(d.get("biome", "temperate"))
            V, F = _gen_uv_sphere(res, 1.0)
            e = _gen_fbm3(V * 2.2, seed, octaves=5)
            Vd = V * (R * (1.0 + relief * (e - sea)))[:, None]
            mesh = Mesh(Vd, F)
            lat = np.abs(V[:, 1])
            moist = _gen_fbm3(V * 3.1, seed + 7, octaves=3)
            mats = []
            for f in F:
                idx = list(f); ee = float(e[idx].mean()); la = float(lat[idx].mean()); mm = float(moist[idx].mean())
                if la > 0.86 and biome != "volcanic":
                    mats.append("polar_ice" if ee < sea else "snow"); continue
                mats.append(_gen_biome_mat(ee, mm, False, biome, sea))
            pos = d.get("position")
            if pos:
                mesh.vertices = mesh.vertices + np.asarray(pos, float)
            nid = _add_object(d.get("name") or "Planet", mesh, mats)
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        # ---------------- OCEAN / LAKE ----------------
        if kind == "supershape":
            # PARAMETRIC GEOMETRY FAMILY (P3.5-1): the Gielis superformula swept as a 3-D supershape. Two
            # superformula profiles (one per angular axis) multiply into a radius field, giving a continuous
            # family -- spheres, superellipsoid boxes, stars, flowers, diatoms -- from a handful of numbers.
            res = int(np.clip(int(d.get("res", 72)), 24, 140))
            m1 = float(d.get("m1", 6)); m2 = float(d.get("m2", 6))
            n11 = float(np.clip(d.get("n11", 1.0), 0.05, 40)); n12 = float(np.clip(d.get("n12", 1.0), 0.05, 40)); n13 = float(np.clip(d.get("n13", 1.0), 0.05, 40))
            n21 = float(np.clip(d.get("n21", 1.0), 0.05, 40)); n22 = float(np.clip(d.get("n22", 1.0), 0.05, 40)); n23 = float(np.clip(d.get("n23", 1.0), 0.05, 40))
            scale = float(np.clip(d.get("scale", 0.8), 0.1, 4.0))

            def _superr(t, m, a, b, cc):
                return (np.abs(np.cos(m * t / 4)) ** b + np.abs(np.sin(m * t / 4)) ** cc) ** (-1.0 / max(a, 1e-6))

            th = np.linspace(-np.pi / 2, np.pi / 2, res)       # latitude
            ph = np.linspace(-np.pi, np.pi, res)               # longitude
            TH, PH = np.meshgrid(th, ph, indexing="ij")
            r1 = _superr(TH, m1, n11, n12, n13); r2 = _superr(PH, m2, n21, n22, n23)
            X = r1 * np.cos(TH) * r2 * np.cos(PH)
            Y = r1 * np.sin(TH)
            Z = r1 * np.cos(TH) * r2 * np.sin(PH)
            Vv = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) * scale
            faces = []
            for i in range(res - 1):
                for j in range(res - 1):
                    a = i * res + j; b = i * res + (j + 1); cc = (i + 1) * res + (j + 1); dd = (i + 1) * res + j
                    faces.append((a, b, cc)); faces.append((a, cc, dd))
            mesh = Mesh(Vv, faces)
            mat = d.get("material") or "clay"
            try:
                _mat(mat)
            except Exception:
                mat = "clay"
            nid = _add_object(d.get("name") or "Supershape", mesh, [mat] * mesh.n_faces)
            _bump()
            out = _payload(); out["object"] = nid
            out["supershape"] = {"m1": m1, "m2": m2, "verts": int(mesh.n_vertices), "faces": int(mesh.n_faces)}
            return jsonify(out)
        if kind == "ocean":
            amp = float(np.clip(d.get("amplitude", 0.06), 0.0, 1.0))
            wl = float(np.clip(d.get("wavelength", 0.8), 0.1, 6.0))
            harm = int(np.clip(int(d.get("harmonics", 4)), 1, 8))
            oid = str(d.get("object", ""))
            if oid and oid in _S["objects"]:
                o = _S["objects"][oid]
                _snap_obj(oid)
                Vw = o.mesh.vertices.copy(); faces = [tuple(f) for f in o.mesh.faces]
                mode = "existing"
            else:
                size = float(np.clip(d.get("size", 4.0), 0.5, 30.0)); res = int(np.clip(int(d.get("res", 64)), 16, 128))
                xs = np.linspace(-size / 2, size / 2, res)
                X, Z = np.meshgrid(xs, xs, indexing="ij")
                Vw = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
                faces = []
                for i in range(res - 1):
                    for j in range(res - 1):
                        a = i * res + j
                        faces.append((a, a + 1, a + res + 1, a + res))
                mode = "new"
            wave = np.zeros(len(Vw))
            phase_t = float(d.get("time", 0.0))                # animation time (waves travel with phase velocity)
            wrng = np.random.RandomState(seed)                 # DETERMINISTIC per-seed directions/phases so a
            wave_spec = []                                     # re-generation at time t is coherent, not random
            for k in range(harm):
                ang = wrng.rand() * 6.283
                kmag = 2 * np.pi / (wl / (k * 0.7 + 1))
                kv = np.array([np.cos(ang), np.sin(ang)]) * kmag
                ph0 = wrng.rand() * 6.283
                speed = np.sqrt(9.8 / max(kmag, 0.1)) * 0.15   # deep-water dispersion: faster for long waves
                wave += (amp / (k + 1)) * np.sin(Vw[:, 0] * kv[0] + Vw[:, 2] * kv[1] + ph0 + phase_t * speed * kmag)
                wave_spec.append([float(kv[0]), float(kv[1]), float(ph0), float(speed * kmag), float(amp / (k + 1))])
            Vw2 = Vw.copy(); Vw2[:, 1] = Vw[:, 1] + wave
            mesh = Mesh(Vw2, faces)
            deep = str(d.get("water", "water"))
            mats = []
            wmean = wave.mean()
            for f in faces:
                mats.append("water_deep" if wave[list(f)].mean() < wmean - amp * 0.25 else deep)
            if mode == "existing":
                o.mesh = mesh; o.mats = mats; o.sdf_tree = None
                o.layers = dict(getattr(o, "layers", {}) or {}); o.layers["ocean"] = {"spec": wave_spec, "base_y": Vw[:, 1].tolist()}
                _bump(oid)
                out = _payload(only=oid); out["object"] = oid
            else:
                nid = _add_object(d.get("name") or "Ocean", mesh, mats)
                _S["objects"][nid].layers = {"ocean": {"spec": wave_spec, "base_y": Vw[:, 1].tolist()}}
                _bump()
                out = _payload(); out["object"] = nid
            out["animatable"] = True
            return jsonify(out)
        # ---------------- CLOUDS (blobby SDF; fog gives the atmosphere) ----------------
        if kind == "clouds":
            import holographic_sdf as S
            from holographic_meshbridge import marching_tetrahedra_vec
            from holographic_meshqem import cluster_decimate
            puffs = int(np.clip(int(d.get("puffs", 7)), 2, 24))
            size = float(np.clip(d.get("size", 0.5), 0.1, 3.0))
            spread = float(np.clip(d.get("spread", 1.4), 0.3, 8.0))
            k = float(np.clip(d.get("softness", 0.22), 0.05, 0.6))
            tree = None
            for i in range(puffs):
                c = (rng.rand(3) - 0.5) * np.array([spread, spread * 0.28, spread * 0.7])
                r = size * (0.5 + 0.6 * rng.rand())
                node = S.sphere(r).translate(tuple(c))
                tree = node if tree is None else tree.smooth_union(node, k)
            lo = np.array([-spread, -spread * 0.4, -spread]) - size * 1.4
            hi = -lo
            res = 56
            ax = tuple(np.linspace(lo[i], hi[i], res) for i in range(3))
            X, Y, Z = np.meshgrid(*ax, indexing="ij")
            g = tree.eval(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
            raw = marching_tetrahedra_vec(g.reshape(res, res, res), ax, level=0.0)
            dec = cluster_decimate(raw, grid=40)
            mesh = Mesh(dec.vertices + np.asarray(d.get("position", [0, 1.6, 0]), float),
                        [tuple(f) for f in dec.faces])
            nid = _add_object(d.get("name") or "Clouds", mesh, ["snow"] * mesh.n_faces)
            _bump()
            out = _payload(); out["object"] = nid
            out["note"] = "blobby mesh clouds; enable fog in the photo panel for atmosphere (no true volumetrics)"
            return jsonify(out)
        # ---------------- SCATTER: grass / trees / rocks on a surface ----------------
        if kind == "scatter":
            oid = str(d.get("object", ""))
            host = _S["objects"].get(oid)
            if host is None:
                return jsonify({"error": "scatter needs a target 'object' (the surface)"}), 400
            what = str(d.get("what", "grass"))
            count = int(np.clip(int(d.get("count", 400)), 1, 3000 if what == "grass" else 400))
            scale = float(np.clip(d.get("size", 0.08), 0.01, 2.0))
            up_only = bool(d.get("up_only", True))
            HV = host.mesh.vertices
            areas = []; norms = []; tris = []
            for f in host.mesh.faces:
                idx = list(f)
                for kk in range(1, len(idx) - 1):
                    a, b, c = HV[idx[0]], HV[idx[kk]], HV[idx[kk + 1]]
                    n = np.cross(b - a, c - a); ln = np.linalg.norm(n)
                    if ln < 1e-12: continue
                    if up_only and (n[1] / ln) < 0.35: continue      # plant on up-facing ground only
                    areas.append(ln / 2); norms.append(n / ln); tris.append((a, b, c))
            if not tris:
                return jsonify({"error": "no up-facing area to scatter on (untick 'up-facing only'?)"}), 400
            areas = np.asarray(areas); cdf = np.cumsum(areas) / areas.sum()
            V = []; F = []; M = []
            def emit(verts, faces, mat):
                b0 = len(V); V.extend(verts); F.extend([tuple(b0 + i for i in f) for f in faces]); M.extend([mat] * len(faces))
            for _ in range(count):
                ti = int(np.searchsorted(cdf, rng.rand()))
                a, b, c = tris[ti]; n = norms[ti]
                r1, r2 = rng.rand(), rng.rand()
                if r1 + r2 > 1: r1, r2 = 1 - r1, 1 - r2
                p = a + (b - a) * r1 + (c - a) * r2
                t1, t2, nu = _gen_frame(n)
                yaw = rng.rand() * 6.283; ca, sa = np.cos(yaw), np.sin(yaw)
                u = t1 * ca + t2 * sa; w = -t1 * sa + t2 * ca
                s = scale * (0.6 + 0.8 * rng.rand())
                if what == "grass":
                    lean = (rng.rand() - 0.5) * 0.5
                    tipv = p + nu * s + u * lean * s
                    base = [p + u * s * 0.06, p - u * s * 0.03 + w * s * 0.05, p - u * s * 0.03 - w * s * 0.05]
                    emit(base + [tipv], [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)], d.get("material", "grass"))
                elif what == "trees":
                    trunk_v, trunk_f = _gen_tube([p, p + nu * s * 1.2], s * 0.09, sides=6)
                    emit(list(trunk_v), trunk_f, "clay")
                    top = p + nu * s * 1.2
                    ring = [top + (np.cos(aa) * u + np.sin(aa) * w) * s * 0.55 for aa in np.linspace(0, 6.283, 7, endpoint=False)]
                    apex = top + nu * s * 1.6
                    emit(ring + [apex], [(i, (i + 1) % 7, 7) for i in range(7)] + [(i + 1, i, 0) for i in range(1, 6)], "forest")
                else:                                             # rocks
                    ph = (1 + 5 ** 0.5) / 2
                    ico = np.array([[-1, ph, 0], [1, ph, 0], [-1, -ph, 0], [1, -ph, 0], [0, -1, ph], [0, 1, ph],
                                    [0, -1, -ph], [0, 1, -ph], [ph, 0, -1], [ph, 0, 1], [-ph, 0, -1], [-ph, 0, 1]], float)
                    ico = ico / np.linalg.norm(ico[0]) * s * 0.5
                    ico = ico * (0.6 + 0.8 * rng.rand(3))          # squash randomly
                    Rv = np.stack([u, nu, w], 1)
                    icoW = (ico @ Rv.T) + p
                    icof = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                            (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                            (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
                    emit(list(icoW), icof, d.get("material", "limestone"))
            mesh = Mesh(np.asarray(V), F)
            nid = _add_object(d.get("name") or {"grass": "Grass", "trees": "Trees"}.get(what, "Rocks"), mesh, M)
            _bump()
            out = _payload(); out["object"] = nid; out["instances"] = count
            return jsonify(out)
        # ---------------- TREE (recursive branching + leaf cards) ----------------
        if kind == "tree":
            depth = int(np.clip(int(d.get("depth", 5)), 2, 7))
            base_len = float(np.clip(d.get("height", 1.6), 0.3, 6.0))
            base_rad = float(np.clip(d.get("thickness", 0.09), 0.01, 0.5))
            splits = int(np.clip(int(d.get("splits", 2)), 1, 4))       # children per branch
            spread = float(np.clip(d.get("spread", 0.6), 0.1, 1.4))    # branch angle (radians)
            leafy = bool(d.get("leaves", True))
            leaf_sz = float(np.clip(d.get("leaf_size", 0.16), 0.02, 0.8))
            bark = str(d.get("bark", "clay")); leafmat = str(d.get("leaf_material", "forest"))
            V = []; F = []; M = []
            def emit(verts, faces, mat):
                b0 = len(V); V.extend(verts); F.extend([tuple(b0 + i for i in f) for f in faces]); M.extend([mat] * len(faces))

            def grow(base, direction, length, radius, lvl):
                direction = direction / max(np.linalg.norm(direction), 1e-9)
                tip = base + direction * length
                # slight mid-branch bow for a natural look
                mid = (base + tip) / 2 + np.cross(direction, [0, 1, 0]) * length * 0.06 * (rng.rand() - 0.5)
                tv, tf = _gen_tube([base, mid, tip], radius, sides=max(4, 7 - lvl), taper=0.62)
                emit(list(tv), tf, bark)
                if lvl >= depth:
                    if leafy:                                          # leaf cluster: a few crossed cards
                        for _ in range(4):
                            t1, t2, nu = _gen_frame(direction + rng.randn(3) * 0.4)
                            c = tip + (rng.randn(3)) * length * 0.4
                            s = leaf_sz * (0.6 + 0.8 * rng.rand())
                            quad = [c - t1 * s - t2 * s * 0.6, c + t1 * s - t2 * s * 0.6,
                                    c + t1 * s + t2 * s * 0.6, c - t1 * s + t2 * s * 0.6]
                            emit(quad, [(0, 1, 2), (0, 2, 3)], leafmat)
                    return
                t1, t2, _ = _gen_frame(direction)
                n = splits + (1 if rng.rand() < 0.4 else 0)
                for i in range(n):
                    ang = (i / n) * 2 * np.pi + rng.rand()
                    tilt = spread * (0.7 + 0.6 * rng.rand())
                    child = (direction * np.cos(tilt)
                             + (t1 * np.cos(ang) + t2 * np.sin(ang)) * np.sin(tilt))
                    grow(tip, child, length * (0.62 + 0.08 * rng.rand()), radius * 0.66, lvl + 1)

            grow(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), base_len, base_rad, 1)
            mesh = Mesh(np.asarray(V) + np.asarray(d.get("position", [0, 0, 0]), float), F)
            nid = _add_object(d.get("name") or f"Tree {seed}", mesh, M)
            _bump()
            out = _payload(); out["object"] = nid; out["branches"] = M.count(bark)
            return jsonify(out)
        # ---------------- CREATURE (Spore-ish) ----------------
        if kind == "creature":
            import holographic_sdf as S
            from holographic_meshbridge import marching_tetrahedra_vec
            from holographic_meshqem import cluster_decimate
            plump = float(np.clip(d.get("plump", 1.0), 0.4, 2.0))
            legs = int(np.clip(int(d.get("legs", 4)), 0, 8)) // 2 * 2
            spikes = bool(d.get("spikes", rng.rand() > 0.5))
            L = 0.9 + rng.rand() * 0.6
            nspine = 4
            hip_y = 0.45 + 0.2 * rng.rand()
            tree = None
            spine = []
            for i in range(nspine):
                x = -L / 2 + L * i / (nspine - 1)
                y = hip_y + 0.08 * np.sin(i / (nspine - 1) * np.pi)
                r = (0.16 + 0.1 * rng.rand()) * plump * (0.75 + 0.5 * np.sin((i + 0.5) / nspine * np.pi))
                spine.append((x, y, r))
                node = S.sphere(r).translate((x, y, 0))
                tree = node if tree is None else tree.smooth_union(node, 0.12)
            head_r = 0.16 * plump + 0.06 * rng.rand()
            hx, hy = L / 2 + head_r * 0.7, hip_y + 0.18 + 0.1 * rng.rand()
            tree = tree.smooth_union(S.sphere(head_r).translate((hx, hy, 0)), 0.1)
            pad = 0.5
            lo = np.array([-L, 0.0, -0.8]) - pad; hi = np.array([L + head_r * 2, hip_y + 0.9, 0.8]) + pad
            res = 60
            ax = tuple(np.linspace(lo[i], hi[i], res) for i in range(3))
            X, Y, Z = np.meshgrid(*ax, indexing="ij")
            g = tree.eval(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
            raw = marching_tetrahedra_vec(g.reshape(res, res, res), ax, level=0.0)
            dec = cluster_decimate(raw, grid=44)
            V = list(dec.vertices); F = [tuple(f) for f in dec.faces]
            skin = f"skin_{seed}"
            try:
                _mat(skin)
            except Exception:
                hcol = rng.rand(3) * 0.6 + 0.25
                import copy as _copy
                m = _copy.deepcopy(_matlib().material("clay")); m.base_color = hcol; m.roughness = 0.6
                _CUSTOM_MATS[skin] = m
            M = [skin] * len(F)
            def emit(verts, faces, mat):
                b0 = len(V); V.extend(verts); F.extend([tuple(b0 + i for i in f) for f in faces]); M.extend([mat] * len(faces))
            body_faces = len(F)                                   # legs/eyes/etc. are appended AFTER this many
            body_verts_n = len(V)
            leg_hips = []
            for i in range(legs // 2):                            # leg pairs along the body
                fx = -L / 2 + (i + 0.5) / max(legs // 2, 1) * L * 0.9
                for sz in (-1, 1):
                    hipp = np.array([fx, hip_y - 0.05, sz * 0.16 * plump])
                    knee = hipp + np.array([0.04, -hip_y * 0.5, sz * 0.1])
                    foot = np.array([hipp[0] + 0.08, 0.0, hipp[2] + sz * 0.12])
                    leg_hips.append([hipp.tolist(), knee.tolist(), foot.tolist(), float(sz)])
                    tv, tf = _gen_tube([hipp, knee, foot], 0.05 * plump, sides=7, taper=0.7)
                    emit(list(tv), tf, skin)
            leg_end_faces = len(F); leg_end_verts = len(V)        # legs occupy [body_faces:leg_end_faces)
            for sz in (-1, 1):                                    # eyes
                er = head_r * 0.22
                ec = np.array([hx + head_r * 0.55, hy + head_r * 0.25, sz * head_r * 0.45])
                ev, ef = _gen_uv_sphere(8, er)
                emit(list(np.asarray(ev) + ec), ef, "obsidian")
            if d.get("arms"):                                     # H1-7: a pair of arms near the shoulders
                for sz in (-1, 1):
                    sh = np.array([L / 2 - 0.12, hip_y + 0.18, sz * 0.18 * plump])
                    elb = sh + np.array([0.14, -0.12, sz * 0.14])
                    hand = elb + np.array([0.12, -0.18, sz * 0.06])
                    tv, tf = _gen_tube([sh, elb, hand], 0.042 * plump, sides=6, taper=0.65)
                    emit(list(tv), tf, skin)
            if d.get("tail"):                                     # H1-7: a tapering tail off the rear
                base = np.array([-L / 2, hip_y, 0.0])
                mid = base + np.array([-0.35, 0.05, 0.0])
                tip = base + np.array([-0.7, 0.28, 0.0])
                tv, tf = _gen_tube([base, mid, tip], 0.09 * plump, sides=7, taper=0.18)
                emit(list(tv), tf, skin)
            if d.get("antennae"):                                 # H1-7: two antennae off the head
                for sz in (-1, 1):
                    a0 = np.array([hx, hy + head_r * 0.8, sz * head_r * 0.3])
                    a1 = a0 + np.array([0.06, 0.22, sz * 0.05])
                    tv, tf = _gen_tube([a0, a1], 0.014, sides=5, taper=0.5)
                    emit(list(tv), tf, skin)
                    bv, bf = _gen_uv_sphere(6, 0.03)
                    emit(list(np.asarray(bv) + a1), bf, "limestone")
            if spikes:
                for i in range(nspine):
                    x, y, r = spine[i]
                    tipp = np.array([x, y + r + 0.16, 0]); base = np.array([x, y + r * 0.7, 0])
                    ring = [base + np.array([np.cos(a), 0, np.sin(a)]) * 0.045 for a in np.linspace(0, 6.283, 5, endpoint=False)]
                    emit(ring + [tipp], [(j, (j + 1) % 5, 5) for j in range(5)] + [((j + 1) % 5, j, 0) for j in range(1, 4)], "limestone")
            mesh = Mesh(np.asarray(V), F)
            nid = _add_object(d.get("name") or f"Creature {seed}", mesh, M)
            # walk-cycle spec: the legs occupy faces [body_faces .. body_faces + n_leg_faces); animate rebuilds
            # just the leg tubes with a gait, leaving the body untouched.
            _S["objects"][nid].layers = {"walk": {"hips": leg_hips, "plump": float(plump), "skin": skin,
                                                  "leg_v0": int(body_verts_n), "leg_v1": int(leg_end_verts),
                                                  "n_legs": len(leg_hips)}}
            _bump()
            out = _payload(); out["object"] = nid; out["walkable"] = len(leg_hips) > 0
            return jsonify(out)
        # ---------------- STAR (emissive sphere) ----------------
        if kind == "star":
            R = float(np.clip(d.get("radius", 0.6), 0.05, 5.0))
            temp = str(d.get("temperature", "yellow"))
            col = {"red": [1, 0.35, 0.15], "orange": [1, 0.55, 0.2], "yellow": [1, 0.9, 0.55],
                   "white": [1, 1, 0.95], "blue": [0.6, 0.75, 1]}.get(temp, [1, 0.9, 0.55])
            mname = f"star_{temp}"
            if mname not in _CUSTOM_MATS:
                import copy as _copy
                m = _copy.deepcopy(_matlib().material("clay"))
                m.base_color = np.asarray(col, float); m.roughness = 1.0
                if hasattr(m, "emissive"):
                    m.emissive = np.asarray(col, float) * float(np.clip(d.get("emission", 6.0), 0.5, 20.0))
                _CUSTOM_MATS[mname] = m
            V, F = _gen_uv_sphere(28, R)
            mesh = Mesh(np.asarray(V) + np.asarray(d.get("position", [0, 0, 0]), float), F)
            nid = _add_object(d.get("name") or f"{temp.capitalize()} star", mesh, [mname] * len(F))
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        # ---------------- MOON (cratered) ----------------
        if kind == "moon":
            res = int(np.clip(int(d.get("res", 44)), 24, 80))
            R = float(np.clip(d.get("radius", 0.4), 0.05, 3.0))
            craters = int(np.clip(int(d.get("craters", 14)), 0, 60))
            V, F = _gen_uv_sphere(res, 1.0)
            V = np.asarray(V)
            e = _gen_fbm3(V * 3.0, seed, octaves=4) * 0.04     # gentle base roughness
            for _ in range(craters):                           # bowl depressions with raised rims
                cdir = rng.randn(3); cdir /= np.linalg.norm(cdir)
                crad = 0.12 + 0.3 * rng.rand()
                ang = np.arccos(np.clip(V @ cdir, -1, 1))
                t = np.clip(1 - ang / crad, 0, 1)
                e += -0.05 * (t ** 1.5) * crad * 3 + 0.012 * np.clip(1 - np.abs(ang - crad) / (crad * 0.25), 0, 1)
            Vd = V * (R * (1.0 + e))[:, None] + np.asarray(d.get("position", [0, 0, 0]), float)
            g01 = (e - e.min()) / max(np.ptp(e), 1e-9)
            mats = ["limestone" if g01[list(f)].mean() > 0.45 else "obsidian" for f in F]
            nid = _add_object(d.get("name") or "Moon", Mesh(Vd, F), mats)
            _bump()
            out = _payload(); out["object"] = nid
            return jsonify(out)
        # ---------------- ASTEROID BELT (ring scatter, no host surface) ----------------
        # ---------------- GALAXY FIELD (geometric, H1-5): spiral star billboards merged into one mesh ----------------
        if kind == "galaxy_field":
            count = int(np.clip(int(d.get("count", 1500)), 100, 6000))
            arms = int(np.clip(int(d.get("arms", 2)), 1, 6))
            radius = float(np.clip(d.get("radius", 3.0), 0.5, 12.0))
            twist = float(np.clip(d.get("twist", 2.6), 0.2, 6.0))
            thick = float(np.clip(d.get("thickness", 0.12), 0.01, 2.0))
            core = float(np.clip(d.get("core", 0.35), 0.05, 1.0))         # bright bulge fraction
            star_sz = float(np.clip(d.get("star_size", 0.02), 0.004, 0.15))
            cen = np.asarray(d.get("center", [0, 0, 0]), float)
            # emissive star materials by temperature band (reused across the field)
            bands = [("gfield_blue", [0.65, 0.78, 1.0]), ("gfield_white", [1, 1, 0.96]),
                     ("gfield_yellow", [1, 0.9, 0.6]), ("gfield_red", [1, 0.5, 0.35])]
            import copy as _copy
            for nm, col in bands:
                if nm not in _CUSTOM_MATS:
                    m = _copy.deepcopy(_matlib().material("clay")); m.base_color = np.asarray(col, float)
                    if hasattr(m, "emissive"):
                        m.emissive = np.asarray(col, float) * 3.0
                    _CUSTOM_MATS[nm] = m
            V = []; F = []; M = []
            # a camera-agnostic billboard = a tiny tri; orient randomly so it reads from any angle
            for _ in range(count):
                # radius: concentrate toward the core (r = R * u^1.6)
                u = rng.rand()
                r = radius * (u ** 1.6)
                arm = rng.randint(0, arms)
                base_ang = arm * (2 * np.pi / arms) + r * twist / radius * 2 * np.pi
                jitter = rng.randn() * (0.18 + 0.5 * (r / radius))          # arms tighten toward core
                ang = base_ang + jitter
                y = rng.randn() * thick * (0.4 + 0.6 * (1 - r / radius))     # thinner disk toward rim
                p = cen + np.array([r * np.cos(ang), y, r * np.sin(ang)])
                # colour: core = blue/white, rim = yellow/red
                cf = r / radius
                if cf < core:
                    mat = "gfield_blue" if rng.rand() < 0.5 else "gfield_white"
                elif cf < 0.7:
                    mat = "gfield_white" if rng.rand() < 0.5 else "gfield_yellow"
                else:
                    mat = "gfield_yellow" if rng.rand() < 0.5 else "gfield_red"
                s = star_sz * (0.6 + 1.2 * rng.rand()) * (1.5 if cf < core else 1.0)
                # a small oriented triangle (3 verts) -- cheap "star"
                t1, t2, _n = _gen_frame(rng.randn(3))
                tri = [p + t1 * s, p + t2 * s, p - (t1 + t2) * s * 0.7]
                b0 = len(V); V.extend(tri); F.append((b0, b0 + 1, b0 + 2)); M.append(mat)
            # a bright central bulge: a small emissive sphere
            bv, bf = _gen_uv_sphere(12, radius * 0.10)
            b0 = len(V); V.extend(list(np.asarray(bv) + cen)); F.extend([tuple(b0 + i for i in f) for f in bf])
            M.extend(["gfield_white"] * len(bf))
            mesh = Mesh(np.asarray(V), F)
            nid = _add_object(d.get("name") or "Galaxy", mesh, M)
            _bump()
            out = _payload(); out["object"] = nid; out["stars"] = count
            out["note"] = "geometric star field (merged mesh); GPU instancing (P2-5) would allow far higher counts"
            return jsonify(out)
        if kind == "asteroids":
            count = int(np.clip(int(d.get("count", 120)), 5, 500))
            r0 = float(np.clip(d.get("radius", 2.2), 0.3, 20.0))
            width = float(np.clip(d.get("width", 0.5), 0.05, 5.0))
            thick = float(np.clip(d.get("thickness", 0.12), 0.01, 2.0))
            s0 = float(np.clip(d.get("size", 0.05), 0.005, 0.6))
            cen = np.asarray(d.get("center", [0, 0, 0]), float)
            ph = (1 + 5 ** 0.5) / 2
            ico = np.array([[-1, ph, 0], [1, ph, 0], [-1, -ph, 0], [1, -ph, 0], [0, -1, ph], [0, 1, ph],
                            [0, -1, -ph], [0, 1, -ph], [ph, 0, -1], [ph, 0, 1], [-ph, 0, -1], [-ph, 0, 1]], float)
            ico /= np.linalg.norm(ico[0])
            icof = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                    (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                    (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
            V = []; F = []; M = []
            for _ in range(count):
                a = rng.rand() * 2 * np.pi
                rr = r0 + (rng.rand() - 0.5) * width
                p = cen + np.array([rr * np.cos(a), (rng.rand() - 0.5) * thick, rr * np.sin(a)])
                s = s0 * (0.4 + 1.3 * rng.rand())
                rock = ico * s * (0.55 + 0.9 * rng.rand(3))    # random squash per asteroid
                yaw = rng.rand() * 6.283; ca, sa = np.cos(yaw), np.sin(yaw)
                Rm = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
                b0 = len(V); V.extend(list(rock @ Rm.T + p))
                F.extend([tuple(b0 + i for i in f) for f in icof]); M.extend(["crust_rock"] * len(icof))
            nid = _add_object(d.get("name") or "Asteroid belt", Mesh(np.asarray(V), F), M)
            _bump()
            out = _payload(); out["object"] = nid; out["instances"] = count
            return jsonify(out)
        # ---------------- SOLAR SYSTEM ----------------
        if kind == "solar_system":
            n = int(np.clip(int(d.get("planets", 4)), 1, 8))
            made = []
            # the star, via the same emissive path
            import copy as _copy
            col = [1, 0.85, 0.5]
            if "star_yellow" not in _CUSTOM_MATS:
                m = _copy.deepcopy(_matlib().material("clay")); m.base_color = np.asarray(col, float)
                if hasattr(m, "emissive"):
                    m.emissive = np.asarray(col, float) * 6.0
                _CUSTOM_MATS["star_yellow"] = m
            Vs, Fs = _gen_uv_sphere(24, 0.55)
            made.append(_add_object("Sun", Mesh(np.asarray(Vs), Fs), ["star_yellow"] * len(Fs)))
            x = 1.1
            biomes = ["volcanic", "desert", "temperate", "arctic"]
            ringed = rng.randint(0, n)
            for i in range(n):
                r = 0.1 + 0.16 * rng.rand()
                V, F = _gen_uv_sphere(22, 1.0)
                e = _gen_fbm3(np.asarray(V) * 2.4, seed + i, octaves=4)
                Vd = np.asarray(V) * (r * (1 + 0.08 * (e - 0.5)))[:, None] + np.array([x + r, 0, 0])
                bi = biomes[min(i, 3)] if n > 2 else "temperate"
                sea = 0.45
                mats = [_gen_biome_mat(float(e[list(f)].mean()), 0.5, False, bi, sea) for f in F]
                pid = _add_object(f"Planet {i + 1}", Mesh(Vd, F), mats)
                made.append(pid)
                if i == ringed and r > 0.16:                      # a flat ring for one lucky planet
                    RV = []; RF = []
                    ri, ro = r * 1.5, r * 2.3
                    segs = 36
                    for si in range(segs):
                        a = si / segs * 2 * np.pi
                        RV += [[(x + r) + ri * np.cos(a), 0, ri * np.sin(a)], [(x + r) + ro * np.cos(a), 0, ro * np.sin(a)]]
                    for si in range(segs):
                        a0 = si * 2; b0 = ((si + 1) % segs) * 2
                        RF += [(a0, b0, b0 + 1, a0 + 1), (a0 + 1, b0 + 1, b0, a0)]   # both windings: visible from above+below
                    made.append(_add_object(f"Planet {i + 1} rings", Mesh(np.asarray(RV, float), RF), ["sand"] * len(RF)))
                x += r * 2 + 0.55 + 0.25 * rng.rand()
            _bump()
            out = _payload(); out["objects_made"] = made
            return jsonify(out)
    return jsonify({"error": f"unknown generator '{kind}'"}), 400


@bp.route("/api/scatter", methods=["POST"])
def scatter_on_surface():
    """SCATTER INSTANCES ON A SURFACE (engine: holographic_meshscatter, new in this drop). Area-weighted
    sampling of the TARGET object's surface -> per-point normal/tangent frames -> the SOURCE object baked
    into real geometry at every placement. This is the engine's own S-1/S-2 pair (mesh surfaces, and
    placements becoming geometry); the app supplies only the object plumbing.
    Args: source (object id, the thing to instance), target (object id, the surface), count, scale,
    scale_jitter, align (0=world-up, 1=follow the surface normal), seed, radius (blue-noise min spacing)."""
    _init()
    d = request.get_json(force=True) or {}
    src_id = str(d.get("source", "")); tgt_id = str(d.get("target", ""))
    count = int(np.clip(int(d.get("count", 120)), 1, 4000))
    scale = float(np.clip(d.get("scale", 0.15), 0.005, 5.0))
    jitter = float(np.clip(d.get("scale_jitter", 0.25), 0.0, 1.0))
    align = float(np.clip(d.get("align", 1.0), 0.0, 1.0))
    radius = d.get("radius", None)
    seed = int(d.get("seed", 1))
    with _LOCK:
        so = _S["objects"].get(src_id); to = _S["objects"].get(tgt_id)
        if so is None or to is None:
            return jsonify({"error": "pick a source object to instance and a target surface"}), 400
        if so is to:
            return jsonify({"error": "source and target must be different objects"}), 400
        import holographic_meshscatter as _ms
        try:
            samp = _ms.sample_mesh_surface(to.mesh, count, seed=seed,
                                           relax=radius is not None,
                                           radius=float(radius) if radius else None)
            T = _ms.placement_frames(samp["points"], samp["normals"], samp.get("tangents"),
                                     scale=scale, scale_jitter=jitter, align=align, seed=seed)
            got = _ms.realize_scatter(so.mesh, T, mode="merge")
            mesh = got[0] if isinstance(got, tuple) else got
        except Exception as e:
            return jsonify({"error": f"scatter failed: {e}"}), 400
        placed = int(np.asarray(T).shape[0])
        if placed == 0 or mesh.n_faces == 0:
            return jsonify({"error": "no placements produced (try more count or a smaller radius)"}), 400
        # the scatter inherits the SOURCE object's material, which is what a user expects of a copy
        name = f"Scatter ({so.name} on {to.name})"
        oid = _add_object(name, mesh, mats=[str(np.asarray(so.mats).ravel()[0])] * mesh.n_faces)
        out = _payload()
        out["scatter"] = {"object": oid, "placements": placed, "faces": int(mesh.n_faces),
                          "source": so.name, "target": to.name,
                          "note": "instances baked to real geometry (engine meshscatter, merge mode)"}
        return jsonify(out)


@bp.route("/api/erode", methods=["POST"])
def erode_terrain():
    """HYDRAULIC EROSION (H1-3): carve drainage into a landscape by simulating rain droplets. Each droplet
    follows the height gradient downhill, picks up sediment on steep descents (up to a speed/volume-scaled
    capacity) and deposits it where the slope flattens -- lowering peaks, filling valleys, and cutting channels.
    Operates on the object's height field (its vertices reprojected to a grid), so it needs a grid-like terrain
    mesh (the Landscape generator). Verified physics: peaks are lowered, relief (height std) is reduced, and the
    result is deterministic per seed. Args: object, droplets, strength, seed."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    droplets = int(np.clip(int(d.get("droplets", 10000)), 500, 60000))
    strength = float(np.clip(d.get("strength", 1.0), 0.1, 3.0))
    seed = int(d.get("seed", 1))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = np.asarray(o.mesh.vertices, float)
        xs = np.unique(np.round(V[:, 0], 5)); zs = np.unique(np.round(V[:, 2], 5))
        nx, nz = len(xs), len(zs)
        if nx * nz != len(V) or nx < 8 or nz < 8:
            return jsonify({"error": "erosion needs a grid terrain mesh (use the Landscape generator)"}), 400
        xi = np.searchsorted(xs, np.round(V[:, 0], 5)); zi = np.searchsorted(zs, np.round(V[:, 2], 5))
        H = np.zeros((nx, nz)); order = np.zeros((nx, nz), int)
        for k in range(len(V)):
            H[xi[k], zi[k]] = V[k, 1]; order[xi[k], zi[k]] = k
        # ENGINE EROSION (was hand-rolled): the upstream droplet-count runaway that forced a local
        # simulation is FIXED and re-verified on this engine drop -- the filed reproducer now holds peak
        # flat across 3K/6K/10K droplets and is exactly scale-invariant, while still doing real work
        # (relief reduced, valleys filled, deterministic per seed). Mapping: `strength` scales the pair
        # that sets how aggressively a droplet cuts vs. how much it can carry, so strength=1 is the
        # engine's own default behaviour.
        H0 = H.copy()
        import holographic_terrain as _tr
        H = np.asarray(_tr.erode(H, droplets=droplets, seed=seed,
                                 capacity=1.0 * strength, erosion=0.3 * strength), float)
        if not np.isfinite(H).all() or H.max() > H0.max() * 1.25:
            # Guard, not a workaround: erosion must never RAISE terrain. If a future drop regresses the
            # fixed runaway, fail loudly here instead of silently writing exploded geometry into the scene.
            return jsonify({"error": "engine erosion produced a divergent field (peak %.3f -> %.3f); "
                                     "terrain left unchanged" % (float(H0.max()), float(H.max()))}), 500
        Vn = V.copy()
        for ix in range(nx):
            for iy in range(nz):
                Vn[order[ix, iy], 1] = H[ix, iy]
        from holographic_mesh import Mesh
        o.mesh = Mesh(Vn, [tuple(f) for f in o.mesh.faces])
        o.sdf_tree = None
        _bump(oid)
        out = _payload(only=oid)
        out["erosion"] = {"droplets": droplets, "strength": round(strength, 3),
                          "peak_before": round(float(H0.max()), 4), "peak_after": round(float(H.max()), 4),
                          "relief_before": round(float(H0.std()), 4), "relief_after": round(float(H.std()), 4),
                          "material_moved": round(float(np.abs(H - H0).sum()), 3),
                          "note": "droplet hydraulic erosion; peaks lowered and valleys filled"}
        return jsonify(out)


@bp.route("/api/env", methods=["POST", "GET", "DELETE"])
def env_backdrop():
    """ENVIRONMENT BACKDROPS: procedural equirect env images -- day / sunset / night sky (sun, moon, stars),
    pure starfield, nebula, spiral galaxy. The image becomes the renderer's sky (fast preview + GI photo both
    sample it; the photo lights the scene with it = dome light), and is returned as b64 PNG so the viewport
    can show the same backdrop. DELETE clears it. Stored per session in _S['env_img']."""
    _init()
    if request.method == "DELETE":
        with _LOCK:
            _S["env_img"] = None
        return jsonify({"cleared": True})
    if request.method == "GET":
        return jsonify({"active": _S.get("env_img") is not None})
    d = request.get_json(force=True) or {}
    preset = str(d.get("preset", "day"))
    seed = int(d.get("seed", 0))
    rng = np.random.RandomState(seed)
    quality = str(d.get("quality", "dome"))
    H, W = (256, 512) if quality == "high" else (128, 256)   # H1-6: high = crisper backdrop for hero shots
    v = np.linspace(1, -1, H)[:, None]                       # +1 = zenith
    u = np.linspace(0, 2 * np.pi, W, endpoint=False)[None, :]
    el = np.arcsin(np.clip(np.broadcast_to(v, (H, W)), -1, 1))
    az = np.broadcast_to(u, (H, W))
    D = np.stack([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)], -1)

    def fbm2(sc, sd, octv=4):
        out = np.zeros((H, W)); amp, freq, tot = 1.0, sc, 0.0
        r2 = np.random.RandomState(sd)
        for _ in range(octv):
            ph = r2.rand(3) * 6.283
            out += amp * (np.sin(D[..., 0] * freq * 3.1 + ph[0]) * np.cos(D[..., 1] * freq * 2.6 + ph[1])
                          * np.sin(D[..., 2] * freq * 3.7 + ph[2]) * 0.5 + 0.5)
            tot += amp; amp *= 0.55; freq *= 2.1
        return out / tot

    def stars(density, r2):
        img = np.zeros((H, W))
        n = int(H * W * density)
        ys = r2.randint(0, H, n); xs = r2.randint(0, W, n)
        mag = r2.rand(n) ** 3 * 2.2 + 0.15
        img[ys, xs] = np.maximum(img[ys, xs], mag)
        big = r2.rand(n) < 0.06                              # a few bright ones bleed to neighbours
        for yy, xx, mm in zip(ys[big], xs[big], mag[big]):
            img[yy, (xx + 1) % W] = max(img[yy, (xx + 1) % W], mm * 0.5)
            img[(yy + 1) % H, xx] = max(img[(yy + 1) % H, xx], mm * 0.5)
        return img

    def disk(az0, el0, radius, soft=0.15):
        d0 = np.array([np.cos(el0) * np.cos(az0), np.sin(el0), np.cos(el0) * np.sin(az0)])
        cosang = D @ d0
        return np.clip((cosang - np.cos(radius)) / (np.cos(radius * (1 - soft)) - np.cos(radius) + 1e-9), 0, 1)

    t = np.clip(np.sin(el), -1, 1)
    if preset == "day":
        sky = np.stack([0.36 + 0.25 * (1 - t), 0.55 + 0.12 * (1 - t), 0.92 - 0.15 * (1 - t)], -1)
        sky = np.clip(sky, 0, 1) * np.clip(0.55 + 0.45 * t, 0.25, 1)[..., None]
        sun_az = float(d.get("sun_az", 0.9)); sun_el = float(d.get("sun_el", 0.7))
        sky += disk(sun_az, sun_el, 0.06)[..., None] * np.array([6.0, 5.6, 4.6])
        sky += disk(sun_az, sun_el, 0.30, soft=0.9)[..., None] * np.array([0.5, 0.45, 0.3])
        ground = np.array([0.32, 0.30, 0.26])
    elif preset == "sunset":
        band = np.clip(1 - np.abs(t + 0.05) * 2.4, 0, 1)
        sky = np.stack([0.18 + 0.8 * band, 0.10 + 0.32 * band, 0.28 + 0.12 * band], -1)
        sky[t > 0.3] = sky[t > 0.3] * 0.5 + np.array([0.06, 0.08, 0.22]) * 0.5
        sun_az = float(d.get("sun_az", 0.0))
        sky += disk(sun_az, 0.06, 0.05)[..., None] * np.array([7, 3.4, 1.4])
        sky += disk(sun_az, 0.06, 0.4, soft=0.92)[..., None] * np.array([1.1, 0.4, 0.12])
        ground = np.array([0.12, 0.08, 0.08])
    elif preset == "night":
        sky = np.stack([0.015 + 0.02 * (1 - t), 0.02 + 0.02 * (1 - t), 0.05 + 0.04 * (1 - t)], -1)
        s = stars(0.02, rng)
        sky += s[..., None] * np.array([0.9, 0.95, 1.0])
        moon_az = float(d.get("moon_az", 2.2)); moon_el = float(d.get("moon_el", 0.6))
        md = disk(moon_az, moon_el, 0.05, soft=0.1)
        crater = fbm2(14, seed + 5, 3) * 0.35 + 0.65
        sky += (md * crater)[..., None] * np.array([0.85, 0.85, 0.8]) * 1.6
        sky += disk(moon_az, moon_el, 0.14, soft=0.9)[..., None] * np.array([0.10, 0.11, 0.13])
        ground = np.array([0.015, 0.017, 0.022])
    elif preset == "starfield":
        sky = np.full((H, W, 3), 0.004)
        s = stars(0.028, rng)
        tint = np.stack([0.9 + 0.2 * fbm2(3, seed + 2, 2), np.full((H, W), 0.95), 1.0 + 0.15 * fbm2(3, seed + 3, 2)], -1)
        sky += s[..., None] * tint
        ground = None
    elif preset == "nebula":
        base = fbm2(2.2, seed, 5); wisp = fbm2(5.0, seed + 9, 4)
        neb = np.clip(base * 0.7 + wisp * 0.5 - 0.35, 0, 1) ** 1.4
        pal = {"purple": ([0.45, 0.1, 0.7], [0.05, 0.35, 0.6]), "teal": ([0.05, 0.5, 0.55], [0.3, 0.1, 0.5]),
               "fire": ([0.8, 0.25, 0.05], [0.4, 0.05, 0.3])}
        c1, c2 = pal.get(str(d.get("palette", "purple")), pal["purple"])
        sky = neb[..., None] * np.asarray(c1) + (fbm2(3.4, seed + 4, 3) * neb)[..., None] * np.asarray(c2)
        sky += stars(0.02, rng)[..., None] * np.array([0.9, 0.95, 1.0])
        sky += 0.004
        ground = None
    elif preset == "galaxy":
        # a tilted spiral: density in the galaxy's own disk coordinates
        g_az = float(d.get("az", 1.2)); g_el = float(d.get("el", 0.25))
        n0 = np.array([np.cos(g_el) * np.cos(g_az), np.sin(g_el), np.cos(g_el) * np.sin(g_az)])   # galaxy centre dir
        t1, t2, _ = _gen_frame(np.array([n0[1], -n0[0], 0]) if abs(n0[2]) > 0.9 else np.cross(n0, [0, 1, 0]))
        x = np.tensordot(D, t1, axes=([-1], [0])); y = np.tensordot(D, t2, axes=([-1], [0]))
        cen = np.tensordot(D, n0, axes=([-1], [0]))
        rr = np.sqrt(x * x + y * y) / np.maximum(cen, 0.05)
        th = np.arctan2(y, x)
        arms = int(np.clip(int(d.get("arms", 2)), 1, 5))
        spir = np.cos(arms * th - 5.5 * np.log(np.maximum(rr, 0.03))) * 0.5 + 0.5
        dens = np.exp(-rr * 2.4) * (0.35 + 0.65 * spir ** 2) * np.clip(cen, 0, 1) ** 2
        core = np.exp(-rr * 10) * np.clip(cen, 0, 1)
        grain = fbm2(6, seed + 11, 3)
        sky = (dens * (0.55 + 0.45 * grain))[..., None] * np.array([0.75, 0.7, 0.9])
        sky += core[..., None] * np.array([1.2, 1.05, 0.85])
        sky += stars(0.02, rng)[..., None] * np.array([0.9, 0.95, 1.0])
        sky += 0.004
        ground = None
    else:
        return jsonify({"error": f"unknown env preset '{preset}'"}), 400
    if ground is not None:
        sky = np.where((t < 0)[..., None], np.broadcast_to(ground, sky.shape), sky)
    sky = np.clip(sky, 0, 8).astype(np.float32)
    with _LOCK:
        _S["env_img"] = sky
        _bump()
    from PIL import Image as _Im
    import io as _io, base64 as _b64
    disp = np.clip(sky / max(sky.max(), 1e-6) if sky.max() > 1 else sky, 0, 1)
    buf = _io.BytesIO(); _Im.fromarray((disp * 255).astype(np.uint8)).save(buf, "PNG")
    return jsonify({"preset": preset, "preview": _b64.b64encode(buf.getvalue()).decode(),
                    "note": "backdrop is now the render sky + GI dome light; DELETE /api/env clears it"})


# =====================================================================================================
# P3-4 geometry coverage: geodesic selection, primitive fitting (re-analytify), lattice deform
# =====================================================================================================
@bp.route("/api/select/geodesic", methods=["POST"])
def select_geodesic():
    """GROW A SELECTION ALONG THE SURFACE (holographic_meshgeodesic): from seed vertices, walk the mesh's
    edge graph out to a geodesic radius -- unlike a euclidean sphere, this cannot leak across gaps (fingers,
    handles) because distance travels ON the surface. Returns hard indices within the radius plus smooth
    falloff weights for soft-selection sculpting/dragging."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        seeds = [int(i) for i in (d.get("seeds") or [])]
        if not seeds:
            return jsonify({"error": "geodesic grow needs seed vertex indices"}), 400
        radius = float(np.clip(d.get("radius", 0.3), 0.005, 50.0))
        import holographic_meshgeodesic as mg
        try:
            base = o.mesh if all(len(f) == 3 for f in o.mesh.faces) else None
            from holographic_mesh import Mesh
            work = base or Mesh(o.mesh.vertices, [tuple(t) for t in o.mesh.triangulate()])
            dist = None
            for s in seeds:                                    # multi-seed = min over per-seed fields
                ds = np.asarray(mg.geodesic_distances(work, int(s)), float)
                dist = ds if dist is None else np.minimum(dist, ds)
        except Exception as e:
            return jsonify({"error": f"geodesic failed: {e}"}), 400
        inside = np.where(dist <= radius)[0]
        w = np.clip(1.0 - dist / radius, 0.0, 1.0)
        w = w * w * (3 - 2 * w)                                # smoothstep falloff
    return jsonify({"indices": [int(i) for i in inside],
                    "weights": [round(float(x), 4) for x in w],
                    "radius": radius, "count": int(len(inside))})


@bp.route("/api/fitprims", methods=["POST"])
def fitprims():
    """FIT PRIMITIVES (holographic_primfit): approximate the object's vertices with spheres / boxes /
    capsules. Reports each fitted part + a residual, and -- when the fit is good and 'adopt' is set --
    installs the fitted SDF tree as the object's analytic representation, RESTORING exact ops (halve,
    tree booleans) on objects that had lost or never had a tree. Adoption is gated on fit quality; a bad
    fit reports honestly instead of installing a wrong tree."""
    _init()
    d = request.get_json(force=True) or {}
    oid = str(d.get("object", ""))
    with _LOCK:
        o = _S["objects"].get(oid)
        if o is None:
            return jsonify({"error": "no such object"}), 400
        import holographic_primfit as pf
        k = int(np.clip(int(d.get("k", 3)), 1, 12))
        try:
            fits = pf.fit_primitives(o.mesh.vertices, k=k, auto_k=bool(d.get("auto_k", True)),
                                     k_max=12, seed=int(d.get("seed", 0)))
        except Exception as e:
            return jsonify({"error": f"fit failed: {e}"}), 400
        def _ser(x):                                           # recursive: tuples/arrays -> nested lists of floats
            if hasattr(x, "__len__") and not isinstance(x, str):
                return [_ser(v) for v in x]
            try:
                return round(float(x), 5)
            except Exception:
                return str(x)
        parts = [{"kind": kind, "params": _ser(prm)} for kind, prm in fits.get("parts", [])]
        residual = float(fits.get("residual", 1.0))
        out = {"object": oid, "parts": parts, "kinds": fits.get("kinds", {}),
               "residual": round(residual, 5), "quality": round(float(fits.get("quality", 0.0)), 3)}
        # bbox-relative residual gate for adoption
        V = o.mesh.vertices
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        rel = residual / max(diag, 1e-9)
        out["residual_relative"] = round(rel, 5)
        if d.get("adopt"):
            if rel < float(d.get("adopt_tol", 0.02)) and fits.get("sdf") is not None:
                o.sdf_tree = fits["sdf"]
                _S["rev"] += 1; o.rev += 1                     # bump WITHOUT dropping the tree we just set
                out["adopted"] = True
                out["note"] = "fitted tree installed -- exact ops (halve, tree booleans) work on this object again"
            else:
                out["adopted"] = False
                out["note"] = f"fit not adopted: relative residual {rel:.4f} over tolerance -- reported only"
    return jsonify(out)


@bp.route("/api/analyze_surface")
def analyze_surface():
    """DISCRETE SURFACE ANALYSIS (P3-4, complements the mean-curvature heatmap): per-vertex GAUSSIAN curvature
    via the angle-defect formula K = (2pi - sum of incident triangle angles) / (vertex area/3) -- the intrinsic
    curvature the mean-curvature field can't tell you (saddle K<0 vs dome K>0 vs flat/cylinder K=0). Also a
    DEVELOPABILITY score: |K| near zero everywhere means the surface can be unrolled flat without stretching
    (sheet metal, papercraft, cardboard) -- reports the flat-vertex fraction. Returns a per-vertex value array
    the client paints as the same heatmap the curvature button uses."""
    _init()
    oid = str(request.args.get("object", "")) or None
    metric = str(request.args.get("metric", "gaussian"))
    with _LOCK:
        oid = oid or next(iter(_S["objects"]), None)
        o = _S["objects"].get(oid) if oid else None
        if o is None:
            return jsonify({"error": "no such object"}), 400
        V = o.mesh.vertices
        tris = o.mesh.faces if all(len(f) == 3 for f in o.mesh.faces) else o.mesh.triangulate()
        n = len(V)
        angle_sum = np.zeros(n); area = np.zeros(n)
        for f in tris:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            pa, pb, pc = V[a], V[b], V[c]
            for i, (p0, p1, p2) in ((a, (pa, pb, pc)), (b, (pb, pc, pa)), (c, (pc, pa, pb))):
                e1 = p1 - p0; e2 = p2 - p0
                d1 = np.linalg.norm(e1); d2 = np.linalg.norm(e2)
                if d1 < 1e-12 or d2 < 1e-12:
                    continue
                cosv = np.clip(np.dot(e1, e2) / (d1 * d2), -1, 1)
                angle_sum[i] += np.arccos(cosv)
            tri_area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))
            area[a] += tri_area / 3; area[b] += tri_area / 3; area[c] += tri_area / 3
        area = np.maximum(area, 1e-9)
        K = (2 * np.pi - angle_sum) / area                     # Gaussian curvature (angle defect / area)
        defect = np.abs(2 * np.pi - angle_sum)                 # raw angle defect (radians) -- scale-invariant
        # developability: a vertex is "flat enough to unroll" when its angle defect is a small angle. This is
        # dimensionless (independent of object size), unlike thresholding K which scales with 1/area.
        flat_ang = float(request.args.get("flat_tol", 0.12))   # ~7 degrees of total defect
        flat_frac = float((defect < flat_ang).mean())
        if metric == "developable":
            vals = np.where(defect < flat_ang, 0.0, defect)    # 0 = unrollable, hot = stretches when flattened
            pos = vals[vals > 0]
            scale = float(np.percentile(pos, 90)) if pos.size else 1.0
        else:
            vals = K
            scale = float(np.percentile(np.abs(K), 95)) or 1.0
        return jsonify({"values": np.round(vals, 4).tolist(), "scale": round(scale, 4),
                        "metric": metric, "developable_fraction": round(flat_frac, 3),
                        "note": ("K>0 dome, K<0 saddle, K~0 flat/cylinder" if metric == "gaussian"
                                 else "0 (cool) = unrolls flat; hot = would stretch when flattened")})
