"""lestudio.server -- the browser UI + JSON API for leStudio.

    python -m lestudio            # or: lestudio  (console script)
    -> http://127.0.0.1:5050

Requires the [ui] extra (Flask + Pillow):  pip install "lestudio"
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

import json
import os
import threading
import time
import uuid

from . import (OPS, Document, NodeGraph, accel_status, decode_image, image_dpi,
               parallel_advice, load_workspace,
               mind, op_catalog, png_bytes, save_workspace, sdf_to_glsl,
               BLEND_MODES, _MATERIALS, _PAPERS)

try:
    from flask import Flask, Response, jsonify, request, send_file
except ImportError as e:  # pragma: no cover
    raise SystemExit('leStudio needs Flask + Pillow: pip install "lestudio"') from e

app = Flask(__name__)


class _WS:
    """The workspace: several documents, one active. `DOC`/`GRAPH` below keep the
    whole existing endpoint surface working against the active document."""

    def __init__(self):
        d = Document(768, 512)
        self.docs = {d.id: d}
        self.graphs = {d.id: NodeGraph(d)}
        self.active = d.id
        self.extras = []           # foreign workspace sections, carried verbatim
        self._wire()

    def _wire(self):
        for g in self.graphs.values():
            g.resolver = self.docs.get
            if "MEDIA" in globals():
                g.media = MEDIA.hook

    @property
    def doc(self):
        return self.docs[self.active]

    @property
    def graph(self):
        return self.graphs[self.active]

    def add(self, w, h, name=None, background=(1.0, 1.0, 1.0)):
        d = Document(w, h, name, background=background)
        self.docs[d.id] = d
        self.graphs[d.id] = NodeGraph(d)
        self._wire()
        return d

    def close(self, did):
        if len(self.docs) <= 1:
            return False
        self.docs.pop(did, None)
        self.graphs.pop(did, None)
        if self.active not in self.docs:
            self.active = next(iter(self.docs))
        return True


WS = _WS()


class _Active:
    """A live proxy to the ACTIVE document or graph."""

    def __init__(self, attr):
        object.__setattr__(self, "_attr", attr)

    def __getattr__(self, name):
        return getattr(getattr(WS, self._attr), name)

    def __setattr__(self, name, value):
        # Without this, `DOC.dpi = 300` quietly created an attribute on the
        # PROXY that shadowed the real document -- reads then came back from
        # the proxy and the document never changed. Writes must land on the
        # object being proxied.
        if name == "_attr":
            object.__setattr__(self, name, value)
        else:
            setattr(getattr(WS, self._attr), name, value)


class _Gone(Exception):
    """A user error the app should explain, not a server fault."""


def _finite(v, name, lo=None, hi=None, default=None):
    """One number, checked. NaN and infinity are the dangerous ones: they do
    not raise, they PROPAGATE -- a NaN colour returned 200 and then spread
    through the layer's pixels and height map, corrupting the document
    silently, which is worse than any crash."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise _Gone("%s must be a number" % name)
    if f != f or f in (float("inf"), float("-inf")):
        raise _Gone("%s must be a real number" % name)
    if lo is not None:
        f = max(lo, f)
    if hi is not None:
        f = min(hi, f)
    return f


def _clean_points(pts):
    """A stroke path, checked. Every one of these came back a 500 from the
    fuzz pass: a string instead of a list, a point with one number, missing
    entirely, NaN or infinite coordinates."""
    if pts is None:
        raise _Gone("this stroke has no points")
    if isinstance(pts, (str, bytes)) or not isinstance(pts, (list, tuple)):
        raise _Gone("points must be a list of [x, y] pairs")
    out = []
    for p in pts:
        if isinstance(p, (str, bytes)) or not isinstance(p, (list, tuple)) \
                or len(p) < 2:
            raise _Gone("every point must be an [x, y] pair")
        x = _finite(p[0], "x")
        y = _finite(p[1], "y")
        rest = [_finite(v, "pressure", 0.0, 4.0) for v in p[2:3]]
        out.append([x, y] + rest)
    return out


def _clean_paint(d):
    """Validate a paint payload ONCE, at the entrance, rather than letting bad
    numbers reach the engine."""
    d["points"] = _clean_points(d.get("points"))
    col = d.get("color")
    if col is not None:
        if isinstance(col, (str, bytes)) or not isinstance(col, (list, tuple)):
            raise _Gone("colour must be [r, g, b] numbers from 0 to 1")
        d["color"] = [_finite(v, "colour", 0.0, 1.0) for v in list(col)[:3]]
        if len(d["color"]) < 3:
            raise _Gone("colour must be [r, g, b] numbers from 0 to 1")
    for k, lo, hi in (("radius", 0.05, 8000.0), ("opacity", 0.0, 1.0),
                      ("load", 0.0, 40.0), ("mix", 0.0, 1.0),
                      ("hardness", 0.0, 1.0), ("taper", 0.0, 1.0)):
        if k in d and d[k] is not None:
            d[k] = _finite(d[k], k, lo, hi)
    return d


DOC = _Active("doc")

# Compositing reads the document's SIZE and then its LAYERS. Those are two
# separate reads, so a resize landing between them composites layers of the
# old shape into a frame of the new one:
#   "operands could not be broadcast together with shapes (350,500,3) (200,300,3)"
# -- a 500 on /api/composite.png, reproduced 5 times in 40 runs of the
# concurrency test and unexplained for four sightings before this. Re-entrant
# because composite handlers call back into other guarded helpers.
_DOC_LOCK = threading.RLock()
GRAPH = _Active("graph")


def _png(img):
    """PNG by default; ?fmt=jpeg&w=<px> serves a downscaled JPEG -- ~5-10x cheaper
    to encode and transfer, which is what live/preview frames want. PNG (with
    alpha) remains the default for anything the user keeps. ?fmt=auto picks
    JPEG only when the image is fully opaque, so display paths get the cheap
    encoding without ever losing real transparency."""
    import numpy as np
    from PIL import Image as PImage
    fmt = request.args.get("fmt", "png")
    wq = request.args.get("w")
    a = np.asarray(img)
    if fmt == "auto":
        # "cheapest encoding that is still CORRECT". PNG must be kept whenever
        # the image carries real transparency (the canvas draws it over a
        # checkerboard), but most working documents sit on an opaque background
        # -- and there JPEG is ~3x faster to encode and ~4x smaller. Measured at
        # 1920x1080 with real artwork: 592 ms / 1203 KB -> 193 ms / 329 KB.
        opaque = not (a.ndim == 3 and a.shape[2] == 4 and float(a[..., 3].min()) < 0.999)
        fmt = "jpeg" if opaque else "png"
    if wq:
        try:
            tw = max(16, min(2048, int(wq)))
            if a.shape[1] > tw:
                from . import _resize
                a = _resize(a, max(int(a.shape[0] * tw / a.shape[1]), 8), tw)
        except ValueError:
            pass
    if fmt == "jpeg":
        rgb = a[..., :3] if a.ndim == 3 and a.shape[2] >= 3 else np.stack([a] * 3, -1)
        buf = io.BytesIO()
        PImage.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).save(
            buf, "JPEG", quality=82)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    return send_file(io.BytesIO(png_bytes(a)), mimetype="image/png")


@app.get("/")
def index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return send_file(path)


JOBS = {}

# ------------------------------------------------------------------------------------------------
# Multiplayer: every mutation bumps a revision; clients subscribe to an SSE feed
# and refresh when someone ELSE changed the workspace. Presence = active feeds.
# ------------------------------------------------------------------------------------------------
# clients: tab-id -> last_seen.  tabuser: tab-id -> user-id.  A USER is a
# person (persistent id in their browser's localStorage, shared by all their
# tabs); a CLIENT is one tab. Presence, host and kick are all per-user, so a
# refresh or a second tab never shows a phantom second editor. joined is
# per-user and never reaped while the process lives: the host is the
# earliest-seen user still present, so the role cannot flap during a reload.
SYNC = {"rev": 0, "src": "", "clients": {}, "tabuser": {}, "names": {},
        "joined": {}, "kicked": set(), "lock": threading.Lock()}
INVITES = {"pending": [], "joined": []}    # session-scoped guest bookkeeping


@app.after_request
def _bump_rev(resp):
    if request.method in ("POST", "PATCH") and request.path.startswith("/api/") \
            and request.path not in ("/api/graph/run", "/api/live",
                                      "/api/autosave") \
            and not request.path.startswith("/api/job/") \
            and resp.status_code < 400:
        with SYNC["lock"]:
            SYNC["rev"] += 1
            SYNC["src"] = request.headers.get("X-Client", "")
    return resp


@app.get("/api/schema")
def schema():
    """Machine-readable surface for agents: every endpoint (method, path, doc)
    plus the full node-op catalog with parameter specs. An agent needs nothing
    else to drive the app -- see AGENT.md."""
    routes = []
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api"):
            continue
        fn = app.view_functions[rule.endpoint]
        routes.append({"path": rule.rule,
                       "methods": sorted(m for m in rule.methods
                                         if m in ("GET", "POST")),
                       "doc": (fn.__doc__ or "").strip().split(chr(10))[0]})
    return jsonify(routes=sorted(routes, key=lambda r: r["path"]),
                   ops=op_catalog(),
                   hints={"state": "GET /api/state is the complete truth",
                          "identify": "send X-Client (your tab/agent run) AND "
                                      "X-User (your persistent identity) on "
                                      "POSTs -- presence, the host role and "
                                      "kick are per X-User",
                          "watch": "GET /api/events?client=..&user=..&name=.. "
                                   "is an SSE change feed; holding it open is "
                                   "presence",
                          "params_on_wires": "graph inputs accept 'NID', "
                                             "'NID.socket', or [id, socket]; "
                                             "'param:<name>' input keys drive "
                                             "any numeric parameter",
                          "impasto": "paint with media oil|acrylic|water and "
                                     "load to build a paint body with gravity "
                                     "and relief light"})


@app.get("/api/events")
def events():
    """Server-sent events: {rev, src, editors}. A client refreshes when rev moves
    and src is not itself. Holding the stream open IS presence."""
    cid = request.args.get("client", uuid.uuid4().hex[:8])
    uid = request.args.get("user") or cid       # old clients: tab id = user id
    nm = (request.args.get("name") or "").strip()[:24]
    if nm:
        SYNC["names"][uid] = nm

    def gen():
        SYNC["clients"][cid] = time.time()
        SYNC["tabuser"][cid] = uid
        SYNC["joined"].setdefault(uid, time.time())
        last = -1
        beat = 0.0
        try:
            while True:
                if uid in SYNC["kicked"]:
                    # a terminal event, then end the stream (EVERY tab of the
                    # kicked user gets this); the finally reaps
                    yield ("data: " + json.dumps({"kicked": True})
                           + chr(10) + chr(10))
                    return
                SYNC["clients"][cid] = time.time()
                now = time.time()
                live_uids = {SYNC["tabuser"].get(c, c)
                             for c, t in SYNC["clients"].items() if now - t < 10}
                editors = len(live_uids)        # USERS, not tabs
                if SYNC["rev"] != last:
                    last = SYNC["rev"]
                    active = [SYNC["names"].get(u2, "") for u2 in live_uids]
                    yield ("data: " + json.dumps(
                        {"rev": last, "src": SYNC["src"], "editors": editors,
                         "names": sorted(n for n in active if n)})
                        + chr(10) + chr(10))
                elif now - beat >= 2.0:
                    # THE GHOST FIX. This generator only wrote to the socket
                    # when rev changed -- on an idle document, never. A closed
                    # tab's connection is only discovered when a WRITE fails,
                    # so dead streams looped forever, refreshing their own
                    # presence timestamp every 250 ms: immortal ghost editors,
                    # and every page refresh added one more ("6 editors",
                    # close tabs, refresh, "7 editors"). A comment ping every
                    # 2 s makes a dead socket raise within seconds, and the
                    # finally below actually runs. EventSource ignores
                    # comment lines, so clients see nothing.
                    beat = now
                    yield ": ping" + chr(10) + chr(10)
                time.sleep(0.25)
        finally:
            SYNC["clients"].pop(cid, None)
            SYNC["tabuser"].pop(cid, None)
            # joined is deliberately NOT reaped: it is per-user and keeping it
            # makes the host role sticky across reloads (host = earliest-seen
            # user still live, not "whoever's newest stream happens to be
            # oldest right now")
    return app.response_class(gen(), mimetype="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


def _editor_roster(me=""):
    """Live USERS (any tab fresh within 10 s), oldest first. One person with
    five tabs is one row with tabs=5. The HOST is the earliest-seen user
    still present -- the role survives the host's reloads (their join time is
    per-user and never reaped) and passes to the next-oldest only when they
    are genuinely gone."""
    now = time.time()
    tabs = {}
    for c, t in SYNC["clients"].items():
        if now - t < 10:
            u2 = SYNC["tabuser"].get(c, c)
            tabs[u2] = tabs.get(u2, 0) + 1
    live = sorted((SYNC["joined"].get(u2, now), u2) for u2 in tabs)
    host = live[0][1] if live else ""
    return [{"id": u2, "name": SYNC["names"].get(u2, ""),
             "tabs": tabs[u2],
             "joined": round(now - j, 1), "you": u2 == me, "host": u2 == host}
            for j, u2 in live]


def _req_uid():
    """The requesting USER: the persistent id if the client sends one, else
    the tab id (old clients and agents degrade to per-tab identity)."""
    return (request.headers.get("X-User")
            or request.headers.get("X-Client")
            or request.args.get("user") or request.args.get("client") or "")


@app.get("/api/editors")
def editors_list():
    """Who is here: id, optional display name, seconds connected, and which
    one is the host (the longest-connected editor)."""
    me = _req_uid()
    return jsonify(ok=True, editors=_editor_roster(me),
                   kicked=[{"id": k, "name": SYNC["names"].get(k, "")}
                           for k in sorted(SYNC["kicked"])])


@app.post("/api/editors/kick")
def editors_kick():
    """Host-only: disconnect another editor. Their event stream ends with a
    terminal {kicked} message and every later edit they attempt is refused."""
    me = _req_uid()
    roster = _editor_roster(me)
    host = next((e["id"] for e in roster if e["host"]), "")
    target = str((request.json or {}).get("id", ""))
    if me != host:
        return jsonify(error="only the host (the longest-connected editor) "
                             "can remove people"), 403
    if target == me:
        return jsonify(error="you cannot kick yourself"), 400
    if not any(e["id"] == target for e in roster):
        return jsonify(error="no such editor (they may have left already)"), 404
    SYNC["kicked"].add(target)
    return jsonify(ok=True)


@app.post("/api/editors/allow")
def editors_allow():
    """Host-only: let a kicked user back in. Until this, every request from
    the kicked id is refused, so a kick is a real removal rather than a
    disconnect they can undo by reloading."""
    me = _req_uid()
    roster = _editor_roster(me)
    host = next((e["id"] for e in roster if e["host"]), "")
    if me != host:
        return jsonify(error="only the host can allow people back"), 403
    target = str((request.json or {}).get("id", ""))
    if target not in SYNC["kicked"]:
        return jsonify(error="that id is not kicked"), 404
    SYNC["kicked"].discard(target)
    return jsonify(ok=True)


@app.post("/api/paint_run")
def paint_run():
    """Wet paint runs under gravity: {"layer": id, "steps": n,
    "gx","gy","gz"}. gz presses paint downhill along the layer's surface
    (tilt, curve, dome, relief); gx/gy pull it laterally."""
    from . import run_paint
    d = request.json or {}
    try:
        run_paint(DOC, d.get("layer", ""), steps=int(d.get("steps", 12)),
                  gx=float(d.get("gx", 0.0)), gy=float(d.get("gy", 0.0)),
                  gz=float(d.get("gz", 1.0)))
    except KeyError:
        return jsonify(error="no such layer"), 404
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.get("/api/timeline")
def timeline_get():
    """The global timeline: {frame, fps, range, tracks} where tracks maps
    "kind:id:prop" -> [[t, v], ...]. Every keyframed property is
    evaluated at the playhead; media layers advance by elapsed frames
    times their keyable media_rate."""
    return jsonify(frame=float(getattr(DOC, "frame", 0.0)),
                   fps=float(getattr(DOC, "fps", 24.0)),
                   range=list(getattr(DOC, "frame_range", [0.0, 96.0])),
                   tracks=getattr(DOC, "tracks", {}))


@app.post("/api/timeline")
def timeline_post():
    """Drive the timeline: {"action": "frame"|"key"|"delkey"|"range"}.
    frame takes t (scrubbing; not undoable). key/delkey take kind
    ("layer"|"light"), id, prop, and optional t/v -- both default to the
    playhead and the live value; both are undoable. range takes lo, hi.
    Animatable props: layer opacity/z_off/tilt_x/tilt_y/thickness/
    emissive/reflect/dispersion/media_rate; light intensity/azimuth/
    elevation/x/y/z/cone."""
    d = request.json or {}
    act = d.get("action")
    try:
        if act == "frame":
            t = DOC.set_frame(float(d.get("t", 0.0)))
            return jsonify(ok=True, frame=t)
        if act == "key":
            ks = DOC.set_key(d["kind"], d["id"], d["prop"],
                             t=d.get("t"), v=d.get("v"))
            return jsonify(ok=True, keys=ks)
        if act == "delkey":
            DOC.del_key(d["kind"], d["id"], d["prop"], t=d.get("t"))
            return jsonify(ok=True)
        if act == "range":
            DOC.frame_range = [float(d.get("lo", 0.0)),
                               float(d.get("hi", 96.0))]
            from . import _MUT_REV
            _MUT_REV[0] += 1
            return jsonify(ok=True)
    except KeyError as ex:
        return jsonify(error="no such target/prop: %s" % ex), 404
    return jsonify(error="unknown action"), 400


@app.get("/api/perspective")
def perspective_get():
    """The document's perspective state: {enabled, vps, horizon, snap,
    ground:{enabled, grid, opacity}}."""
    return jsonify(persp=getattr(DOC, "persp", {}))


@app.post("/api/perspective")
def perspective_post():
    """Perspective assistance: {"action": "estimate"|"set"|"clear"}.
    estimate reads vanishing points out of a layer's drawing ({"layer"})
    via line-intersection voting and stores them (guides turn on); set
    patches any of enabled/vps/horizon/snap/ground; clear wipes it all.
    The ground is an infinite shadow-catcher plane under the scene."""
    from . import estimate_perspective
    d = request.json or {}
    act = d.get("action", "set")
    P = getattr(DOC, "persp", {})
    if act == "estimate":
        try:
            est = estimate_perspective(DOC, d.get("layer", ""))
        except KeyError:
            return jsonify(error="no such layer"), 404
        except Exception as ex:
            return jsonify(error=str(ex)), 400
        DOC.record("Estimate perspective", only=[])
        P.update(vps=est["vps"], horizon=est["horizon"], enabled=True)
        from . import _MUT_REV
        _MUT_REV[0] += 1
        return jsonify(ok=True, persp=P, confidence=est["confidence"])
    if act == "set":
        DOC.record("Perspective", only=[])
        for k in ("enabled", "vps", "horizon", "snap"):
            if k in d:
                P[k] = d[k]
        if "ground" in d:
            P.setdefault("ground", {}).update(d["ground"])
        from . import _MUT_REV
        _MUT_REV[0] += 1
        return jsonify(ok=True, persp=P)
    if act == "clear":
        DOC.record("Clear perspective", only=[])
        DOC.persp = {"enabled": False, "vps": [], "horizon": None,
                     "snap": False,
                     "ground": {"enabled": False, "grid": False,
                                "opacity": 0.5}}
        from . import _MUT_REV
        _MUT_REV[0] += 1
        return jsonify(ok=True, persp=DOC.persp)
    return jsonify(error="unknown action"), 400


@app.get("/api/lights")
def lights_list():
    """The environment's lights: [{id, kind, color, intensity, azimuth,
    elevation, x, y, z, enabled}]."""
    return jsonify(lights=[dict(li) for li in getattr(DOC, "lights", [])])


@app.get("/api/fields")
def get_fields():
    """List force-field objects: [{id, kind: point|direct|vortex, layer, x, y, radius, strength, angle}]. Fields parented to a layer shape that layer's living media every timeline step."""
    return jsonify(fields=[dict(f) for f in getattr(DOC, "fields", [])])


@app.post("/api/field")
def field_route():
    """Field CRUD: {"action": "add"|"edit"|"delete", "id"?, "kind"?, "layer"?, "x"?, "y"?, "radius"?, "strength"?, "angle"?}. add returns {ok, field}; point strength>0 repels, <0 attracts; direct pushes along angle degrees; vortex swirls. Fields die with their layer."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        f = DOC.add_field(kind=d.get("kind", "point"),
                          layer=d.get("layer"),
                          x=d.get("x"), y=d.get("y"),
                          radius=float(d.get("radius", 120.0)),
                          strength=float(d.get("strength", 1.0)),
                          angle=float(d.get("angle", 0.0)))
        return jsonify(ok=True, field=f)
    if act == "edit":
        try:
            f = DOC.edit_field(d["id"],
                               **{k: d.get(k) for k in
                                  ("kind", "layer", "x", "y", "radius",
                                   "strength", "angle")})
        except KeyError:
            return jsonify(error="no such field"), 404
        return jsonify(ok=True, field=f)
    if act == "delete":
        return jsonify(ok=DOC.delete_field(d.get("id", "")))
    return jsonify(error="unknown action"), 400


@app.post("/api/light")
def light_edit():
    """Manage lights: {"action": "add"|"edit"|"remove"|"preset", ...}.
    preset takes name ("sun"|"studio"|"three_point"|"dome") and replaces
    the whole rig. Kinds: view, directional, point, spot (x,y,z with
    aim_x/aim_y + cone/soft), dome (color=sky, color2=ground). add takes
    kind ("view"|"directional"|"point"), color [r,g,b] (channels may
    exceed 1), intensity, azimuth, elevation, x, y, z; edit takes id plus
    any of those; remove takes id. Multiple lights sum; directionals cast
    height-field shadows across the canvas."""
    d = request.json or {}
    # A NaN here is not caught anywhere downstream and makes the whole LIT
    # RENDER non-finite -- the picture is corrupted and it is invisible until
    # you look at the output. Numbers get checked before they reach the rig.
    try:
        for k in ("intensity", "azimuth", "elevation", "x", "y", "z",
                  "aim_x", "aim_y", "cone", "soft", "radius"):
            if d.get(k) is not None:
                d[k] = _finite(d[k], k, -1e6, 1e6)
        for k in ("color", "color2"):
            col = d.get(k)
            if col is not None:
                if isinstance(col, (str, bytes)) or not isinstance(
                        col, (list, tuple)) or len(col) < 3:
                    raise _Gone("%s must be [r, g, b] numbers" % k)
                d[k] = [_finite(v, k, 0.0, 64.0) for v in list(col)[:3]]
    except _Gone as e:
        return jsonify(error=str(e)), 400
    act = d.get("action", "add")
    try:
        if act == "add":
            li = DOC.add_light(kind=d.get("kind", "directional"),
                               color=d.get("color", [1, 1, 1]),
                               intensity=float(d.get("intensity", 1.0)),
                               azimuth=float(d.get("azimuth", 315.0)),
                               elevation=float(d.get("elevation", 45.0)),
                               x=d.get("x"), y=d.get("y"),
                               z=float(d.get("z", 60.0)),
                               aim_x=d.get("aim_x"), aim_y=d.get("aim_y"),
                               cone=float(d.get("cone", 30.0)),
                               soft=float(d.get("soft", 0.5)),
                               color2=d.get("color2", [0.25, 0.22, 0.18]),
                               layer=d.get("layer"),
                               scale=float(d.get("scale", 1.0)))
            return jsonify(ok=True, light=dict(li))
        if act == "preset":
            try:
                DOC.light_preset(d.get("name", ""))
            except KeyError:
                return jsonify(error="unknown preset"), 400
            return jsonify(ok=True,
                           lights=[dict(li) for li in DOC.lights])
        if act == "edit":
            li = DOC.edit_light(d["id"], **{k: v for k, v in d.items()
                                            if k not in ("action", "id")})
            return jsonify(ok=True, light=dict(li))
        if act == "remove":
            DOC.remove_light(d["id"])
            return jsonify(ok=True)
    except KeyError:
        return jsonify(error="no such light"), 404
    return jsonify(error="unknown action"), 400


@app.get("/api/stamps")
def stamps_list():
    """The sticker shelf: [{id, name, w, h}]."""
    return jsonify(stamps=[{"id": s.id, "name": s.name,
                            "w": int(s.pixels.shape[1]),
                            "h": int(s.pixels.shape[0])}
                           for s in getattr(DOC, "stamps", [])])


@app.get("/api/stamp/<sid>.png")
def stamp_png(sid):
    """A sticker's pixels as PNG (transparent background preserved)."""
    try:
        s = DOC.stamp_by_id(sid)
    except KeyError:
        return jsonify(error="no such stamp"), 404
    return _png(s.pixels)


@app.post("/api/stamp/create")
def stamp_create():
    """Capture a sticker: {"layer": id, "selection"?: sel_id, "name"?}.
    The layer's alpha bbox, cut through the selection if given."""
    d = request.json or {}
    try:
        s = DOC.make_stamp_from(d.get("layer", ""),
                                sel=d.get("selection") or None,
                                name=d.get("name") or None)
    except KeyError:
        return jsonify(error="no such layer"), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, id=s.id, name=s.name,
                   w=int(s.pixels.shape[1]), h=int(s.pixels.shape[0]))


@app.post("/api/stamp/place")
def stamp_place():
    """Press a sticker onto a layer: {"layer", "stamp", "x", "y",
    "scale"?, "rotation"?, "opacity"?, "record"?}."""
    d = request.json or {}
    try:
        DOC.place_stamp(d.get("layer", ""), d.get("stamp", ""),
                        float(d.get("x", 0)), float(d.get("y", 0)),
                        scale=float(d.get("scale", 1.0)),
                        rotation=float(d.get("rotation", 0.0)),
                        opacity=float(d.get("opacity", 1.0)),
                        record=bool(d.get("record", True)))
    except KeyError:
        return jsonify(error="no such layer or stamp"), 404
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.delete("/api/stamp/<sid>")
def stamp_delete(sid):
    """Remove a sticker from the shelf."""
    try:
        DOC.remove_stamp(sid)
    except KeyError:
        return jsonify(error="no such stamp"), 404
    return jsonify(ok=True)


@app.post("/api/contact_print")
def contact_print_route():
    """Screen printing between slabs: {"layer": top_id}. Wherever that
    layer's (tilted, lowered) base penetrates the surface of the layer
    below, pigment transfers -- weighted by penetration depth. Returns the
    printed pixel count. Tilt/lower first via /api/layer edit (z_off,
    tilt_x, tilt_y)."""
    from . import contact_print
    d = request.json or {}
    try:
        n = contact_print(DOC, d.get("layer", ""))
    except (StopIteration, KeyError):
        return jsonify(error="no such layer"), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, printed=int(n))


@app.post("/api/media/step")
def media_step():
    """Stir a dynamic media slab: {"layer": id, "steps": n}. Runs the
    layer's fluid forward -- smoke keeps climbing, fire burns down, ink
    keeps curling. The canvas refresh after this shows the new state, so
    repeated calls animate."""
    from . import _media_slab_step, _MEDIA_KINDS
    d = request.json or {}
    try:
        l = DOC.layer(d.get("layer", ""))
    except KeyError:
        return jsonify(error="no such layer"), 404
    if getattr(l, "vol_kind", "none") not in _MEDIA_KINDS:
        return jsonify(error="layer is not a dynamic medium "
                             "(vol_kind inkwater|smoke|fire)"), 400
    DOC.record("Media step", only=[l.id])
    _media_slab_step(DOC, l, int(d.get("steps", 12)))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/view3d")
def view3d():
    """The document camera: {"mode": "flat"|"ortho"|"persp", "vantage":
    "above"|"below"}. flat is the classic pipeline (cached, patched);
    ortho/persp switch the composite to the volumetric slab stack --
    explicit opt-in, so the realtime caches stay honest for the flat path
    everyone paints in. vantage picks which SIDE of the refracting sheets
    you are standing on: above sees the surface and what it reflects and
    refracts; below sees the ceiling ripple, dispersion, and caustics
    swimming toward the eye."""
    d = request.json or {}
    mode = d.get("mode", DOC.view3d if hasattr(DOC, "view3d") else "flat")
    if mode not in ("flat", "ortho", "persp"):
        return jsonify(error="mode must be flat|ortho|persp"), 400
    from . import _MUT_REV
    DOC.view3d = mode
    if "vantage" in d:
        van = d.get("vantage") or "above"
        if van not in ("above", "below"):
            return jsonify(error="vantage must be above|below"), 400
        DOC.vantage = van
    _MUT_REV[0] += 1
    return jsonify(ok=True, mode=mode,
                   vantage=getattr(DOC, "vantage", "above"))


@app.get("/api/layers/duplicates")
def layer_duplicates():
    """Near-duplicate layers via leCore's image_signature (probed: separates
    real content at 0.9997 near-dup vs 0.788 different, ~18 ms/layer).
    Signature similarity above 0.995 AND alpha coverage within 20% of each
    other counts as a duplicate pair -- both gates, because the signature is
    global statistics and an empty layer resembles another empty layer."""
    import numpy as np
    from . import mind
    m = mind()
    sigs, cov = {}, {}
    for l in DOC.layers:
        a = l.pixels
        cov[l.id] = float(a[..., 3].mean())
        rgb = a[..., :3] * a[..., 3:4]
        try:
            sigs[l.id] = np.asarray(m.image_signature(rgb),
                                    np.float32).ravel()
        except Exception:
            sigs[l.id] = None
    pairs = []
    ids = [l.id for l in DOC.layers]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = sigs[ids[i]], sigs[ids[j]]
            if a is None or b is None:
                continue
            if cov[ids[i]] < 1e-4 and cov[ids[j]] < 1e-4:
                continue                       # two empty layers match trivially
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            sim = float(np.dot(a, b) / denom)
            cmax = max(cov[ids[i]], cov[ids[j]], 1e-6)
            if sim > 0.995 and abs(cov[ids[i]] - cov[ids[j]]) / cmax < 0.2:
                pairs.append({"a": ids[i], "b": ids[j],
                              "a_name": DOC.layer(ids[i]).name,
                              "b_name": DOC.layer(ids[j]).name,
                              "similarity": round(sim, 4)})
    return jsonify(ok=True, pairs=pairs)


@app.post("/api/editors/name")
def editors_name():
    """Set the requesting user's display name (shown in the roster instead of
    'editor a1b2')."""
    me = _req_uid()
    if not me:
        return jsonify(error="no user id on the request"), 400
    nm = str((request.json or {}).get("name", "")).strip()[:24]
    if nm:
        SYNC["names"][me] = nm
    else:
        SYNC["names"].pop(me, None)
    return jsonify(ok=True, name=nm)


@app.before_request
def _refuse_kicked():
    """A kicked USER's edits are refused server-side, not just hidden: the
    persistent user id rides on every POST (X-User, falling back to
    X-Client), so mutations from a kicked person get a clear 403 from every
    one of their tabs instead of silently landing."""
    if request.method in ("POST", "PATCH", "DELETE"):
        uid = _req_uid()
        if uid and uid in SYNC["kicked"]:
            return jsonify(error="you were removed from this session by the "
                                 "host"), 403


# ------------------------------------------------------------------------------------------------
# External media: files, URLs, streams, test signals -- feeding "Media in" nodes.
# ------------------------------------------------------------------------------------------------
try:
    from holographic.io_and_interop.holographic_framesource import (
        FrameSource as _CoreFrameSource, is_frame_source)
except ImportError:                       # engine < 0.2.2
    class _CoreFrameSource:
        seekable = pausable = False
    def is_frame_source(o):
        return hasattr(o, "get")


class _MediaSource(_CoreFrameSource):
    """One live frame supplier, conforming to leCore's FrameSource contract
    (get() -> (frame, seq); seekable/pausable flags). Kinds: test pattern,
    still image file (mtime-watched), or anything cv2.VideoCapture accepts
    (video files loop; network streams -- MJPEG/RTSP/HTTP -- follow live)."""

    seekable = True                       # video files honour seek; streams no-op
    pausable = True

    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

    def __init__(self, source, fps):
        self.source = source
        self.fps = max(float(fps or 10), 0.5)
        self.frame = None          # float32 RGB
        self.seq = 0
        self._stop = threading.Event()
        self._mtime = None
        self.status = "connecting…"
        self.play = True
        self.pos = None            # requested seek position 0..1 (files only)
        self._seek = None          # pending seek
        self.nframes = 0
        if source.startswith("test:"):
            self.kind = "test"
            self.status = "ok — built-in test signal"
        elif os.path.splitext(source.split("?")[0])[1].lower() in self.IMG_EXT                 and "://" not in source:
            self.kind = "image"
        else:
            self.kind = "capture"
            threading.Thread(target=self._capture_loop, daemon=True).start()

    def _resolve(self):
        """Turn page URLs (YouTube etc.) into direct stream URLs when possible."""
        src = self.source
        page_hosts = ("youtube.com", "youtu.be", "twitch.tv", "vimeo.com")
        if "://" in src and any(h in src for h in page_hosts):
            try:
                import yt_dlp
            except ImportError:
                self.status = ("error: this is a video PAGE, not a direct stream. "
                               "Install the resolver (pip install yt-dlp, or the "
                               "app's [media] extra) or paste a direct .mp4 / .m3u8 "
                               "/ RTSP URL.")
                return None
            try:
                with yt_dlp.YoutubeDL({"quiet": True,
                                       "format": "best[height<=720]"}) as y:
                    return y.extract_info(src, download=False)["url"]
            except Exception as e:
                self.status = f"error: could not resolve page URL ({e})"
                return None
        if "://" not in src and not os.path.exists(src):
            self.status = f"error: file not found: {src}"
            return None
        return src

    def _capture_loop(self):
        try:
            import cv2
        except ImportError:
            self.status = ("error: opencv-python-headless is not installed — "
                           "reinstall the app (pip install -e .) to pick it up")
            return
        src = self._resolve()
        if src is None:
            return
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status = ("error: cannot open source. For network URLs it must be "
                           "a direct video stream (mp4 / m3u8 / RTSP / MJPEG); for "
                           "files, the path must be readable by the server.")
            return
        self.status = "ok — capturing"
        try:
            self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            self.nframes = 0
        fails = 0
        while not self._stop.is_set():
            if not self.play:
                # paused: hold the frame; honour seeks on seekable files
                if self._seek is not None and self.nframes > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            int(self._seek * max(self.nframes - 1, 0)))
                    ok, bgr = cap.read()
                    if ok:
                        self.frame = bgr[..., ::-1].astype("float32") / 255.0
                        self.seq += 1
                    self._seek = None
                    self.status = "ok — paused (seek)"
                else:
                    self.status = "ok — paused"
                time.sleep(0.1)
                continue
            ok, bgr = cap.read()
            if not ok:
                # a file that ended: loop it; a stream: retry
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = cap.read()
                if not ok:
                    fails += 1
                    self.status = ("reconnecting…" if fails < 20
                                   else "error: source stopped producing frames")
                    time.sleep(0.5)
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    continue
            fails = 0
            self.status = "ok — capturing"
            rgb = bgr[..., ::-1].astype("float32") / 255.0
            self.frame = rgb
            self.seq += 1
            time.sleep(1.0 / self.fps)
        cap.release()

    def _tick_test(self):
        import numpy as np
        t = time.time()
        seq = int(t * self.fps)
        if seq != self.seq or self.frame is None:
            h, w = 288, 512
            ys, xs = np.mgrid[0:h, 0:w].astype("float32")
            ph = (t % 10) / 10.0
            r = 0.5 + 0.5 * np.sin(xs / 40 + ph * 6.283)
            g = 0.5 + 0.5 * np.sin(ys / 30 - ph * 6.283)
            b = 0.5 + 0.5 * np.sin((xs + ys) / 60 + ph * 12.566)
            f = np.stack([r, g, b], -1)
            bar = int(ph * w)
            f[:, max(0, bar - 6):bar + 6] = 1.0        # sweeping clock bar
            self.frame = f.astype("float32")
            self.seq = seq
        return self.frame

    def _tick_image(self):
        try:
            mt = os.path.getmtime(self.source)
        except OSError:
            self.status = f"error: file not found: {self.source}"
            return self.frame
        if mt != self._mtime:
            self._mtime = mt
            with open(self.source, "rb") as f:
                self.frame = decode_image(f.read())[..., :3]
            self.seq += 1
            self.status = "ok — image (reloads on change)"
        return self.frame

    def get(self):
        if self.kind == "test":
            return self._tick_test(), self.seq
        if self.kind == "image":
            return self._tick_image(), self.seq
        return self.frame, self.seq

    def close(self):
        self._stop.set()


class _MediaManager:
    def __init__(self):
        self.sources = {}          # node_id -> _MediaSource

    def hook(self, nid, params, want):
        p = params or {}
        src = p.get("source", "") or ""
        if not src.strip():
            return 0 if want == "seq" else None
        cur = self.sources.get(nid)
        if cur is None or cur.source != src:
            if cur:
                cur.close()
            cur = self.sources[nid] = _MediaSource(src, p.get("fps", 10))
        play = bool(int(float(p.get("play", 1))))
        pos = float(p.get("pos", 0))
        if play != cur.play:
            cur.play = play
        if not play and pos != (cur.pos if cur.pos is not None else -1):
            cur.pos = pos
            cur._seek = pos
        frame, seq = cur.get()
        return seq if want == "seq" else frame


    def statuses(self):
        return {nid: {"status": s.status, "has_frame": s.frame is not None}
                for nid, s in self.sources.items()}


MEDIA = _MediaManager()
WS._wire()                     # attach the media hook to graphs created before this point


# ------------------------------------------------------------------------------------------------
# Live output: evaluate the active graph on a clock, publish as an MJPEG stream.
# ------------------------------------------------------------------------------------------------
LIVE = {"on": False, "fps": 10.0, "jpeg": None, "seq": 0,
        "cond": threading.Condition(), "error": None}


def _live_loop():
    from PIL import Image as PImage
    import numpy as np
    while LIVE["on"]:
        t0 = time.time()
        try:
            GRAPH.ensure_default()
            img = np.asarray(_renderable(GRAPH.evaluate(GRAPH.output_node())))
            arr = (np.clip(img[..., :3], 0, 1) * 255).astype("uint8")   # JPEG: no alpha
            buf = io.BytesIO()
            PImage.fromarray(arr).save(buf, "JPEG", quality=85)
            with LIVE["cond"]:
                LIVE["jpeg"] = buf.getvalue()
                LIVE["seq"] += 1
                LIVE["cond"].notify_all()
            LIVE["error"] = None
        except Exception as e:
            LIVE["error"] = str(e)
        time.sleep(max(0.0, 1.0 / LIVE["fps"] - (time.time() - t0)))


@app.post("/api/live")
def live_ctl():
    """Toggle live stroke streaming for this session: {"on": bool}."""
    d = request.json or {}
    if d.get("action") == "start":
        LIVE["fps"] = max(0.5, min(30.0, float(d.get("fps", 10))))
        if not LIVE["on"]:
            LIVE["on"] = True
            threading.Thread(target=_live_loop, daemon=True).start()
    elif d.get("action") == "stop":
        LIVE["on"] = False
    return jsonify(ok=True, live=LIVE["on"], fps=LIVE["fps"], error=LIVE["error"])


@app.get("/api/stream.mjpg")
def stream_mjpg():
    """The composited graph output as multipart MJPEG -- point OBS (or any
    streaming software's browser/media source) at this URL."""
    if not LIVE["on"]:
        return jsonify(error="start a live session first (Live button)"), 409

    def gen():
        last = -1
        while LIVE["on"]:
            with LIVE["cond"]:
                LIVE["cond"].wait(timeout=2.0)
                if LIVE["seq"] == last or LIVE["jpeg"] is None:
                    continue
                last, data = LIVE["seq"], LIVE["jpeg"]
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                   + data + b"\r\n")
    return app.response_class(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/sdf/shader")
def sdf_shader():
    """Compile an SDF DSL to GLSL. Since leCore 0.2.2 the emitter provides the
    uniform-driven camera natively (camera="uniforms") -- no host patching."""
    d = request.json or {}
    try:
        from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
        from . import mind
        tree = parse_dsl(d.get("dsl", "(sphere 0.8)"))
        glsl = mind().to_shadertoy(tree, camera="uniforms")
    except Exception as e:
        return jsonify(error=str(e)), 400
    body = glsl.replace("mainImage", "st_mainImage")
    nl = chr(10)
    decls = "".join(                            # only declare what the emitter didn't
        "uniform " + t + " " + n + ";" for t, n in
        [("vec3", "iResolution"), ("float", "iTime"), ("float", "uAngle"),
         ("float", "uHeight"), ("float", "uDist")]
        if "uniform " + t + " " + n not in body)
    wrapped = (
        "#version 300 es" + nl + "precision highp float;" + nl + decls + nl +
        "out vec4 fragOut;" + nl + body + nl +
        "void main(){ st_mainImage(fragOut, gl_FragCoord.xy); }" + nl)
    return jsonify(ok=True, glsl=glsl, wrapped=wrapped, orbit=True)


@app.post("/api/postfx/shader")
def postfx_shader():
    """Compile a Post FX node's settings into a fragment shader (leCore 0.2.2
    PostChain.to_glsl): the exact colour pipeline the node applies, but running
    on the viewer's GPU -- live grading at display rate."""
    from . import _postfx_steps, mind
    p = (request.json or {}).get("params", {})
    try:
        steps = _postfx_steps(p)
        if not steps:
            steps = [("exposure", {"ev": 0.0})]     # neutral but valid chain
        chain = mind().postfx_chain(*steps)
        glsl = chain.to_glsl(skip_unsupported=True)
    except Exception as e:
        return jsonify(error=str(e)), 400
    pointwise = {"exposure", "reinhard", "aces", "gamma", "color_grade",
                 "vignette", "pbr_neutral"}
    skipped = sorted({k for k, _ in steps} - pointwise)
    body = glsl.replace("mainImage", "fx_mainImage")
    nl = chr(10)
    decls = "".join(
        "uniform " + t + " " + n + ";" for t, n in
        [("vec3", "iResolution"), ("sampler2D", "iChannel0")]
        if "uniform " + t + " " + n not in body)
    wrapped = (
        "#version 300 es" + nl + "precision highp float;" + nl + decls + nl +
        "out vec4 fragOut;" + nl + body + nl +
        "void main(){ fx_mainImage(fragOut, vec2(gl_FragCoord.x, "
        "iResolution.y - gl_FragCoord.y)); }" + nl)
    return jsonify(ok=True, glsl=glsl, wrapped=wrapped, skipped=skipped)


_ST_UNIFORMS = ("vec3 iResolution", "float iTime", "vec4 iMouse",
                "int iFrame", "float iTimeDelta", "vec4 iDate",
                "sampler2D iChannel0", "sampler2D iChannel1")


def _wrap_shadertoy(source):
    """Wrap raw Shadertoy GLSL (a `mainImage` function, plus any helpers the
    user pasted) into a complete WebGL2 / GLSL ES 3.00 fragment program the
    browser can compile: version + precision, the Shadertoy uniforms, an output
    variable, the user's body verbatim, and a main() that calls mainImage.

    Done in the app so the Shadertoy node never depends on a specific leCore
    build. If the installed leCore exposes wrap_webgl2 we use it (identical
    output); otherwise we assemble it ourselves."""
    fn = getattr(mind(), "wrap_webgl2", None)
    if callable(fn):
        try:
            return fn(source, uniforms=_ST_UNIFORMS)
        except Exception:
            pass                                        # fall through to app wrap
    nl = "\n"
    decls = nl.join("uniform " + u + ";" for u in _ST_UNIFORMS)
    return (
        "#version 300 es" + nl + "precision highp float;" + nl +
        decls + nl + "out vec4 fragOut;" + nl + nl +
        source + nl + nl +
        "void main(){ mainImage(fragOut, gl_FragCoord.xy); }" + nl + nl)


@app.post("/api/shader/match")
def shader_match():
    """Match a layer (or the composite) with a PROCEDURAL SHADER: leCore's
    fit_shape reads the image's roughness/detail signature and we compose a
    complete, runnable Shadertoy source from it -- a starting point to paste
    into a Shadertoy node and tweak.

    Body: {layer?}. Returns {source, quality, baseline, ratio, note}. The note
    is leCore's own and says plainly that this is a same-family statistical
    match, NOT a pixel match -- we pass it through so the artist isn't sold
    something the fit doesn't do."""
    from . import shader_from_image
    d = request.json or {}
    lid = d.get("layer") or None
    try:
        img = DOC.layer(lid).pixels[..., :3] if lid else DOC.composite()[..., :3]
        r = shader_from_image(np.asarray(img, np.float32))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, **r)


@app.post("/api/export/svg")
def export_svg():
    """Trace the graph output (or a chosen node) into a layered SVG poster --
    scalable vector art from any picture. Body: {node?, levels, simplify, w?}.
    Needs scikit-image; the capability flag says whether this build has it."""
    from . import vectorize_svg
    d = request.json or {}
    GRAPH.ensure_default()
    nid = d.get("node") or GRAPH.output_node()
    try:
        w = int(d.get("w") or 0)
        img = (_renderable(GRAPH.render_at(nid, w, int(w * DOC.height /
                                                       max(DOC.width, 1))))
               if w else _renderable(GRAPH.evaluate(nid)))
        svg = vectorize_svg(img, levels=int(d.get("levels", 6)),
                            simplify=float(d.get("simplify", 1.2)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return app.response_class(svg, mimetype="image/svg+xml",
        headers={"Content-Disposition": "attachment; filename=lestudio.svg"})


@app.get("/api/shader/presets")
def shader_presets():
    """A small library of ready-to-run shaders: clouds, water, fire, smoke,
    embers, terrain, grass, foliage, plus glow and depth-of-field post effects.

    All original work for leStudio on top of MIT-licensed simplex noise --
    NOT copied from Shadertoy, whose default licence (CC BY-NC-SA) would
    forbid shipping them here. See shader_presets.ATTRIBUTION."""
    from .shader_presets import catalogue, ATTRIBUTION
    return jsonify(ok=True, presets=catalogue(), attribution=ATTRIBUTION)


@app.get("/api/shader/palette")
def shader_palette():
    """iq's cosine palette as a GLSL function (leCore cosine_palette_to_glsl):
    the standard way to colour a greyscale shader. The UI's 🎨 Palette button
    inserts this into the Shadertoy editor -- one click after ✨ Match canvas
    turns the matched greyscale fbm into colour."""
    from . import have
    if not have("cosine_palette_to_glsl"):
        return jsonify(error="this leCore build has no cosine_palette_to_glsl"), 400
    try:
        g = mind().cosine_palette_to_glsl(
            a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5),
            c=(1.0, 1.0, 1.0), d=(0.0, 0.33, 0.67))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, glsl=g)


@app.get("/api/shadertoy/pending")
def st_pending():
    """Render requests for the browser's GPU: each spec carries the wrapped
    WebGL2 source (leCore wrap_webgl2 -- the one true Shadertoy preamble),
    plus channel textures as PNG data URLs."""
    from . import SHADER_PENDING
    import base64, io as _io
    from PIL import Image as _PImage
    out = []
    for spec in list(SHADER_PENDING.values()):
        try:                                    # one bad spec must not 500 them all
            chans = []
            for c in spec["channels"]:
                if c is None:
                    chans.append(None)
                else:
                    buf = _io.BytesIO()
                    _PImage.fromarray(
                        (np.clip(c, 0, 1) * 255).astype("uint8")).save(buf, "PNG")
                    chans.append("data:image/png;base64," +
                                 base64.b64encode(buf.getvalue()).decode())
            wrapped = _wrap_shadertoy(spec["source"])
            out.append({"key": spec["key"], "fragment": wrapped,
                        "time": spec["time"], "mouse_x": spec["mouse_x"],
                        "mouse_y": spec["mouse_y"], "width": spec["width"],
                        "height": spec["height"], "channels": chans})
        except Exception as e:                  # surface as a node error, keep going
            from . import SHADER_ERRORS, _mut
            SHADER_ERRORS[spec["key"]] = "could not prepare shader: " + str(e)
            _mut()
    return jsonify(ok=True, pending=out)


# /api/shadertoy/frame stays in the blanket after_request bump: engine-side
# _mut() below invalidates EVAL caches only (it never touches SYNC), so the
# blanket hook is this endpoint's one and only SYNC bump -- other clients rely
# on it to learn a fresh shader frame exists. (I briefly excluded it on the
# false assumption of double-counting; the verification caught it.)
@app.post("/api/shadertoy/frame")
def st_frame():
    """The browser posts back a rendered frame ({key, pixels: base64 RGBA
    bytes at width*height*4}) or a GLSL error ({key, error}). Frames become
    the node's output; errors surface on the node verbatim."""
    from . import SHADER_FRAMES, SHADER_ERRORS, SHADER_PENDING, SHADER_GEN, _mut
    import base64
    d = request.json or {}
    key = d.get("key")
    if not key:
        return jsonify(error="missing key"), 400
    if d.get("error"):
        SHADER_ERRORS[key] = str(d["error"])[:600]
        SHADER_PENDING.pop(key, None)
        SHADER_GEN[0] += 1; _mut()                   # invalidate eval caches
        return jsonify(ok=True, stored="error")
    spec = SHADER_PENDING.get(key)
    w = int(d.get("width") or (spec or {}).get("width") or 0)
    h = int(d.get("height") or (spec or {}).get("height") or 0)
    raw = base64.b64decode(d.get("pixels", ""))
    if not (w and h) or len(raw) != w * h * 4:
        return jsonify(error="pixels do not match width*height*4"), 400
    arr = np.frombuffer(raw, np.uint8).reshape(h, w, 4).astype(np.float32) / 255
    arr = arr[::-1]                                  # WebGL reads bottom-up
    SHADER_FRAMES[key] = np.ascontiguousarray(arr)
    SHADER_ERRORS.pop(key, None)
    SHADER_PENDING.pop(key, None)
    SHADER_GEN[0] += 1; _mut()                       # invalidate eval caches
    if len(SHADER_FRAMES) > 64:                      # bound the cache
        for k in list(SHADER_FRAMES)[:-48]:
            SHADER_FRAMES.pop(k, None)
    return jsonify(ok=True, stored="frame")


@app.get("/api/assets")
def assets_list():
    """List uploaded media assets (id, name, kind, size)."""
    from . import ASSETS, _ensure_sample_asset
    _ensure_sample_asset()
    return jsonify(ok=True, assets=[{"id": k, "name": a["name"]}
                                    for k, a in ASSETS.items()])


@app.post("/api/assets/upload")
def assets_upload():
    """Upload a 3-D model (.obj, .glb, .gltf; an .obj may bring its .mtl in
    the same request). Registers it for the 3D model node and returns its id."""
    from . import register_asset
    import os as _os
    files = request.files.getlist("file")
    if not files:
        return jsonify(error="no file in the upload"), 400
    main = next((f for f in files
                 if _os.path.splitext(f.filename)[1].lower()
                 in (".obj", ".glb", ".gltf")), None)
    if main is None:
        return jsonify(error="upload a .obj, .glb or .gltf "
                             "(got: %s)" % ", ".join(f.filename for f in files)), 400
    ext = _os.path.splitext(main.filename)[1].lower()
    data = main.read()
    if ext == ".obj":                                # keep any .mtl companion
        for f in files:
            if f.filename.lower().endswith(".mtl"):
                pass                                 # obj importer reads by path;
                                                     # materials default without it
    aid = register_asset(main.filename, data, ext)
    try:
        from . import asset_mesh
        asset_mesh(aid)                              # validate it imports NOW
    except Exception as e:
        from . import ASSETS
        ASSETS.pop(aid, None)
        return jsonify(error="could not read that model: %s" % e), 400
    return jsonify(ok=True, id=aid, name=main.filename)


@app.post("/api/media/upload")
def media_upload():
    """Upload media (multipart file) for Media in / Brush tip use; returns {id}."""
    f = request.files["file"]
    mdir = "/tmp/lestudio_media"
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f.filename)
    f.save(path)
    return jsonify(ok=True, path=path)


# What each optional accelerator buys, in the engine's own measured terms.
# We never auto-install these: they are large (the Zig toolchain wheel is
# ~45 MB) and CuPy must match your CUDA. We just say plainly what is missing.
_ACCEL_HINTS = {
    "ziglang": ("pip install -e .[accel]",
                "native batch kernels + raymarcher (2-5x on kernels, "
                "3.8x raymarch -- speeds up Clouds/SDF)"),
    "numba":   ("pip install -e .[accel]",
                "JIT fast paths for SDF render and codegen"),
    "pyfftw":  ("pip install -e .[accel]",
                "FFTW-backed FFT with plan caching (spectral nodes)"),
    "cupy":    ("pip install cupy-cuda12x   (match your CUDA)",
                "GPU backend for the whole engine"),
}


@app.get("/api/status")
def status():
    """Engine status: gpu availability, leCore version, faculty report."""
    a = accel_status()
    have_map = a.get("accel") or {}
    missing = [{"name": n, "install": _ACCEL_HINTS[n][0],
                "unlocks": _ACCEL_HINTS[n][1]}
               for n in ("ziglang", "numba", "pyfftw", "cupy")
               if n in have_map and not have_map[n]]
    # `gpu` stays a BOOLEAN -- the chip reads it directly, and accel_status()
    # now also returns a gpu_report DICT under the same word. Keeping the flag
    # separate from the report avoids a truthy dict silently reading as "GPU
    # available" on a machine with none.
    report = a.get("gpu") if isinstance(a.get("gpu"), dict) else None
    gpu_flag = bool(report.get("any_available")) if report else bool(a.get("gpu"))
    # WHAT IS ACTUALLY ACCELERATED. `gpu` means leCore found a device it can
    # use for SIMULATION and node work. The painting engine -- deposit, flow,
    # bristle tracks, blurs -- is pure numpy on the CPU, so a chip reading
    # "GPU" told a painter their brush was accelerated when it was not.
    # Report it per subsystem instead of as one flag.
    subsystems = {
        "painting": {"device": "cpu",
                     "note": "brush, impasto and media run on the CPU"},
        "simulation": {"device": "gpu" if gpu_flag else "cpu",
                       "note": ("fluid, fields and node work use the GPU"
                                if gpu_flag else
                                "no GPU found - running on the CPU")},
        "shaders": {"device": "gpu", "note": "shader previews run in your "
                                             "browser's GPU"},
    }
    return jsonify(gpu=gpu_flag, jit=a["jit"], live=LIVE["on"],
                   live_error=LIVE["error"], accel=have_map,
                   accel_missing=missing, subsystems=subsystems,
                   threads=int(os.environ.get("LESTUDIO_THREADS", "0")) or None,
                   gpu_report=report, advice=a.get("advice") or [],
                   determinism=a.get("determinism"))


@app.post("/api/graph/run")
def graph_run():
    """Evaluate a node (default: the Output node) as a background JOB with
    progress reporting and cancellation."""
    d = request.json or {}
    GRAPH.ensure_default()
    nid = d.get("id") or GRAPH.output_node()
    jid = uuid.uuid4().hex[:10]
    total = max(len(GRAPH.upstream_ids(nid)), 1)
    job = {"progress": 0.0, "done": False, "error": None,
           "cancel": threading.Event(), "count": 0, "total": total}
    JOBS[jid] = job

    def run():
        def tick(_nid):
            job["count"] += 1
            job["progress"] = min(job["count"] / job["total"], 1.0)
        GRAPH.progress_cb = tick
        GRAPH.cancel_event = job["cancel"]
        try:
            GRAPH.evaluate(nid)
            GRAPH.commit_layer_outputs()
            job["progress"] = 1.0
        except Exception as e:
            job["error"] = str(e)
        finally:
            GRAPH.progress_cb = None
            GRAPH.cancel_event = None
            job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True, job=jid, node=nid)


def _skimage_available():
    try:
        import skimage                                   # noqa: F401
        return True
    except ImportError:
        return False


_MP4_OK = None


def _mp4_available():
    """MP4 needs imageio-ffmpeg AND its bundled ffmpeg binary. get_ffmpeg_exe()
    tries to download the binary when missing, so we check once and cache --
    an import succeeding is not the same as ffmpeg actually being runnable."""
    global _MP4_OK
    if _MP4_OK is None:
        try:
            import imageio_ffmpeg
            _MP4_OK = bool(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            _MP4_OK = False
    return _MP4_OK


@app.post("/api/render/animation")
def render_animation():
    """Render the graph as an ANIMATION by sweeping one node's dial across a
    range -- Water's `time`, a Shadertoy's `time`, a Value node, any float/int
    param. Runs as a cancellable JOB (progress = frames done).

    Body: {node, param, from, to, frames<=120, fps, w?, h?, format}
    GIF always works (Pillow encodes it, no extra installs). MP4 needs the
    imageio-ffmpeg package; when it's absent we say so plainly instead of
    guessing at codecs. Fetch the file from /api/job/<id>/result."""
    d = request.json or {}
    nid = d.get("node")
    pname = d.get("param")
    n = GRAPH.nodes.get(nid) if nid else None
    if not n:
        return jsonify(error="unknown node %r" % nid), 400
    meta = OPS.get(n["type"], {})
    if pname not in {q["name"] for q in meta.get("params", [])}:
        return jsonify(error="node %r has no param %r" % (n["type"], pname)), 400
    fmt = (d.get("format") or "gif").lower()
    if fmt == "mp4" and not _mp4_available():
        return jsonify(error="MP4 export needs the imageio-ffmpeg package -- "
                             "pip install imageio-ffmpeg (GIF works now, "
                             "no install needed)"), 400
    v0, v1 = float(d.get("from", 0.0)), float(d.get("to", 1.0))
    frames = max(2, min(int(d.get("frames", 16)), 120))
    fps = max(1, min(int(d.get("fps", 8)), 30))
    w = int(d.get("w") or DOC.width)
    h = int(d.get("h") or DOC.height)
    if w * h > 1280 * 720:
        return jsonify(error="animation frames are capped at 1280x720"), 400
    snapped = None
    if fmt == "mp4":
        # H.264 macroblocks are 16x16; imageio's writer silently RESIZES any
        # other size (96x72 -> 96x80 = ~11% vertical stretch). Snap the render
        # size up front instead: frames are rendered at exactly the encoded
        # size, so nothing is stretched -- at most an ~8 px dimension change,
        # which we report back.
        sw = max(16, int(round(w / 16.0)) * 16)
        sh = max(16, int(round(h / 16.0)) * 16)
        if (sw, sh) != (w, h):
            snapped = (sw, sh)
        w, h = sw, sh
    jid = uuid.uuid4().hex[:10]
    job = {"progress": 0.0, "done": False, "error": None,
           "cancel": threading.Event(), "count": 0, "total": frames,
           "result": None, "mime": None, "filename": None}
    JOBS[jid] = job
    kind = next(q["kind"] for q in meta["params"] if q["name"] == pname)
    original = n["params"].get(pname)

    def run():
        import io as _io
        from PIL import Image as _Img
        outs = []
        try:
            for i in range(frames):
                if job["cancel"].is_set():
                    raise RuntimeError("cancelled")
                t = v0 + (v1 - v0) * (i / (frames - 1))
                n["params"][pname] = int(round(t)) if kind == "int" else t
                # no explicit invalidation needed: the graph cache keys on each
                # node's param signature, so the mutation re-evaluates naturally
                arr = _renderable(GRAPH.render_at(GRAPH.output_node(), w, h))
                outs.append(_Img.fromarray(
                    (np.clip(arr, 0, 1) * 255).astype("uint8")))
                job["count"] = i + 1
                job["progress"] = (i + 1) / frames
            if fmt == "mp4":
                import tempfile, imageio_ffmpeg
                path = tempfile.mktemp(suffix=".mp4")
                gen = imageio_ffmpeg.write_frames(          # dims are already
                    path, (w, h), pix_fmt_in="rgb24", fps=fps)  # 16-multiples
                gen.send(None)
                for fimg in outs:
                    fr8 = np.asarray(fimg.convert("RGB"))
                    gen.send(np.ascontiguousarray(fr8).tobytes())
                gen.close()
                job["result"] = open(path, "rb").read()
                os.unlink(path)
                job["mime"] = "video/mp4"
                job["filename"] = "animation.mp4"
                if snapped:
                    job["note"] = ("size snapped to %dx%d for H.264 "
                                   "macroblocks (no stretching)" % snapped)
            else:
                buf = _io.BytesIO()
                outs[0].save(buf, format="GIF", save_all=True,
                             append_images=outs[1:], duration=int(1000 / fps),
                             loop=0, optimize=True)
                job["result"] = buf.getvalue()
                job["mime"] = "image/gif"
                job["filename"] = "animation.gif"
        except Exception as e:
            job["error"] = str(e)
        finally:
            if original is None:
                n["params"].pop(pname, None)
            else:
                n["params"][pname] = original            # leave the graph as found
            job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True, job=jid, frames=frames)


@app.get("/api/job/<jid>/result")
def job_result(jid):
    """Fetch a finished job's rendered PNG."""
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="unknown job"), 404
    if not j.get("result"):
        return jsonify(error="no result (job unfinished, failed, or not a "
                             "render job)"), 404
    return app.response_class(j["result"], mimetype=j["mime"],
        headers={"Content-Disposition":
                 "attachment; filename=" + (j["filename"] or "result.bin")})


@app.get("/api/job/<jid>")
def job_status(jid):
    """Poll a background job: {done, progress, error?}."""
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="unknown job"), 404
    return jsonify(progress=j["progress"], done=j["done"], error=j["error"],
                   note=j.get("note"))


@app.post("/api/job/<jid>/cancel")
def job_cancel(jid):
    """Cancel a running background job."""
    j = JOBS.get(jid)
    if j:
        j["cancel"].set()
    return jsonify(ok=True)


@app.get("/api/state")
def state():
    """THE complete truth: doc, layers (with alpha_lock/clip), masks, selections, splines, brushes, graph, ops catalog."""
    return jsonify({
        "capabilities": _capabilities(),
        "docs": [{"id": d.id, "name": d.name,
                  # unsaved-work signal: closing a dirty document should warn
                  "dirty": bool(getattr(d, "_undo", None)), "active": d.id == WS.active,
                  "layers": [{"id": l.id, "name": l.name} for l in d.layers],
                  "groups": [{"id": g["id"], "name": g["name"]} for g in d.groups],
                  "masks": [{"id": m.id, "name": m.name} for m in d.masks]}
                 for d in WS.docs.values()],
        "active_doc": WS.active,
        "dpi": float(getattr(DOC, "dpi", 72.0)),
        # Who else is here, and what each client is looking at. Presence was
        # tracked but never surfaced, so a collaborator was invisible until
        # their edits appeared out of nowhere.
        "peers": [{"id": cid, "name": nm or cid[:6],
                   "doc": SYNC.get("viewing", {}).get(cid),
                   "me": cid == request.headers.get("X-Client", "")}
                  for cid, nm in sorted(SYNC.get("names", {}).items())],
        "width": DOC.width, "height": DOC.height,
        "layers": [dict(l.meta(),
                        has_strokes=any(k["layer"] == l.id for k in DOC.strokes))
                   for l in DOC.layers],
        "groups": DOC.groups,
        "masks": [m.meta() for m in DOC.masks],
        # saved selections plus the unsaved working one, flagged so the UI can
        # show it apart and offer to keep it
        "selections": [dict(x.meta(), saved=True) for x in DOC.selections]
                      + ([dict(DOC._scratch_sel.meta(), saved=False)]
                         if getattr(DOC, "_scratch_sel", None) is not None else []),
        "splines": [p.meta() for p in DOC.splines],
        # Recorded brush strokes: id + a summary only. The full point lists can
        # be large and the UI just needs to offer them in a picker.
        "strokes": [{"id": k["id"], "layer": k["layer"],
                     "points": len(k["points"]),
                     "ends": [[round(float(k["points"][0][0]), 1),
                               round(float(k["points"][0][1]), 1)],
                              [round(float(k["points"][-1][0]), 1),
                               round(float(k["points"][-1][1]), 1)]]
                     if k["points"] else None,
                     "erase": bool(k["brush"].get("erase"))}
                    for k in getattr(DOC, "strokes", [])][-64:],
        "media_status": MEDIA.statuses(),
        "brushes": [b.meta() for b in DOC.brushes],
        "graph": list(GRAPH.ensure_default().values()),
        "output_node": GRAPH.output_node(),
        "ops": op_catalog(),
        "blend_modes": list(BLEND_MODES),
        # the stock everything is painted on: it decides where thin
        # paint catches, where a wash pools, and how much it granulates
        "paper": str(getattr(DOC, "paper", "canvas")),
        # spill a full layer onto a fresh stratum instead of flattening
        "auto_stratum": bool(getattr(DOC, "auto_stratum", False)),
        # so the app can point at / hide the palette without guessing
        # the palette lives on its OWN surface; the app talks to it through
        # /api/palette/* rather than by painting a layer of the picture
        "palette_ready": DOC.palette_doc(create=False) is not None,
        "papers": sorted(_PAPERS),
        # what is physically on the brush right now, for the charge meter
        "brush_state": DOC.brush_state(),
        "stroke_groups": [dict(g, strokes=list(g["strokes"]))
                          for g in DOC.stroke_groups],
        # the materials catalog rides in state so the client builds its menu
        # from the ENGINE's table -- one source of truth, and a custom or
        # future preset appears in the UI without a client edit
        "materials": [{"name": k, "rough": v["rough"], "metal": v["metal"],
                       "color": (list(v["color"]) if v.get("color") else None)}
                      for k, v in _MATERIALS.items()],
        "can_undo": bool(DOC._undo), "can_redo": bool(DOC._redo),
    })


@app.post("/api/new")
def new_doc():
    """Create a document: {"name", "width", "height", "background"?: [r,g,b] or null for transparent}. Activates it."""
    # The DOCSTRING BELONGS ON THE ROUTE HANDLER. Wrapping this for the
    # document lock moved it to the inner function, and the route went
    # undocumented -- invisible to the agent-facing schema. Caught only when
    # the suite finally ran against real leCore; the stub environment had
    # that test in its noise.
    # (creating a document SWITCHES the active one, which is the same
    # shape-change race the composite is guarded against)
    with _DOC_LOCK:
        return _new_doc_locked()


def _new_doc_locked():
    d = request.json or {}
    bg = d.get("background", (1.0, 1.0, 1.0))   # null -> transparent
    try:
        w, h = int(d.get("width", 768)), int(d.get("height", 512))
    except (TypeError, ValueError):
        return jsonify(error="width and height must be numbers"), 400
    if w < 8 or h < 8:
        return jsonify(error="minimum canvas size is 8x8"), 400
    if w > 16384 or h > 16384 or w * h > 80_000_000:
        return jsonify(error="that canvas is too large (limit 80 megapixels, "
                             "16384 px per side)"), 400
    doc = WS.add(w, h, d.get("name"), background=bg)
    # a new document declares its resolution AND its physical scale
    try:
        dpi = float(d.get("dpi", 72.0))
    except (TypeError, ValueError):
        dpi = 72.0
    doc.dpi = min(2400.0, max(1.0, dpi))
    WS.active = doc.id
    return jsonify(ok=True, doc={"id": doc.id, "name": doc.name,
                                 "width": w, "height": h, "dpi": doc.dpi})


@app.post("/api/doc")
def doc_ops():
    """Document ops: {"action": "activate"|"rename"|"close"|"settings", "id", ...}. close needs force:true if unsaved."""
    # resize and close change the document's SHAPE under any composite that
    # is mid-flight -- the other half of the race guarded in comp_png
    with _DOC_LOCK:
        return _doc_ops_locked()


def _doc_ops_locked():
    d = request.json or {}
    act = d.get("action")
    if act == "activate":
        if d["id"] in WS.docs:
            WS.active = d["id"]
            # Remember who is looking at what, so the UI can say "Bob is on
            # Second" instead of silently yanking everyone to the same doc.
            cid = request.headers.get("X-Client", "")
            if cid:
                SYNC.setdefault("viewing", {})[cid] = d["id"]
    elif act == "close":
        doc = WS.docs.get(d["id"])
        if doc is not None and getattr(doc, "_undo", None) and not d.get("force"):
            # Closing threw away every edit with no warning at all. Report it
            # and let the client confirm rather than deciding for the user.
            return jsonify(ok=False, needs_confirm=True,
                           name=doc.name, edits=len(doc._undo)), 409
        WS.close(d["id"])
    elif act == "rename":
        WS.docs[d["id"]].name = d.get("name") or WS.docs[d["id"]].name
    elif act == "settings":
        doc = WS.docs.get(d.get("id"), WS.doc)
        if d.get("name"):
            doc.name = d["name"]
        try:
            w = int(d.get("width", doc.width))
            h = int(d.get("height", doc.height))
        except (TypeError, ValueError):
            return jsonify(error="width and height must be numbers"), 400
        # A mistyped size tried to allocate the array and took the whole app
        # down -- 99999x99999 is 149 GiB. Refuse politely instead, in the same
        # terms the user typed.
        if w < 8 or h < 8:
            return jsonify(error="minimum canvas size is 8x8"), 400
        if w > 16384 or h > 16384:
            return jsonify(error="maximum canvas dimension is 16384 px"), 400
        if w * h > 80_000_000:
            return jsonify(
                error="%dx%d is %.0f megapixels; the limit is 80" % (
                    w, h, w * h / 1e6)), 400
        if d.get("dpi") is not None:
            try:
                dv = float(d["dpi"])
            except (TypeError, ValueError):
                return jsonify(error="dpi must be a number"), 400
            if not 1 <= dv <= 2400:
                return jsonify(error="dpi must be between 1 and 2400"), 400
            doc.dpi = dv
        if (w, h) != (doc.width, doc.height):
            try:
                # "resample" = IMAGE size (the picture stays, pixel count
                # changes). "canvas" = CANVAS size (the frame changes, content
                # keeps its pixels and is cropped or padded). Two different
                # operations that were hiding behind one field.
                doc.resize(w, h, d.get("mode", "resample"))
            except MemoryError:
                return jsonify(error="not enough memory for %dx%d" % (w, h)), 400
    return jsonify(ok=True)


def _fill_content(src, h, w):
    """Build the (h, w, 3) content image a fill stamps: solid colour, linear
    gradient, leCore pattern, or any Fill-out node's image."""
    import numpy as np
    from . import mind, _resize, _rgb
    kind = (src or {}).get("type", "color")
    if kind == "color":
        return np.full((h, w, 3), np.asarray(src.get("color", [0, 0, 0]),
                                             np.float32)[None, None, :3])
    if kind == "gradient":
        a = np.asarray(src.get("a", [0, 0, 0]), np.float32)[:3]
        b = np.asarray(src.get("b", [1, 1, 1]), np.float32)[:3]
        ang = np.deg2rad(float(src.get("angle", 0)))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (xs / max(w - 1, 1)) * np.cos(ang) + (ys / max(h - 1, 1)) * np.sin(ang)
        t = (t - t.min()) / max(np.ptp(t), 1e-9)
        return a[None, None] * (1 - t[..., None]) + b[None, None] * t[..., None]
    if kind == "pattern":
        pat = mind().pattern_field(src.get("kind", "noise"),
                                   seed=int(src.get("seed", 0)))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([xs / w - 0.5, ys / h - 0.5,
                        np.zeros_like(xs)], -1).reshape(-1, 3)
        pts = pts * float(src.get("scale", 6.0))
        return _rgb(np.asarray(pat(pts)).reshape(h, w).astype(np.float32))
    if kind == "node":
        GRAPH.ensure_default()
        img = GRAPH.evaluate(str(src.get("node", "")))
        return _resize(np.asarray(img, np.float32)[..., :3], h, w)
    raise ValueError(f"unknown fill source type {kind!r}")


@app.get("/api/fonts")
def fonts():
    """Font names the text tool can use, sorted, with the default first."""
    from . import list_fonts
    names = sorted(list_fonts())
    for pref in ("DejaVuSans",):
        if pref in names:
            names.remove(pref); names.insert(0, pref)
    return jsonify(ok=True, fonts=names)


@app.post("/api/text")
def text():
    """Rasterise text onto a layer: {layer, text, x, y, size, font, color,
    spline?, letter_spacing?, shadow?{dx,dy,blur,opacity,color}}. With a
    spline id the text rides the path, glyphs rotated to the tangent."""
    d = request.json or {}
    try:
        n = DOC.add_text(d["layer"], str(d.get("text", "")),
                         x=int(d.get("x", 0)), y=int(d.get("y", 0)),
                         size=int(d.get("size", 48)),
                         color=d.get("color", [1, 1, 1]),
                         font=d.get("font") or None,
                         spline=d.get("spline") or None,
                         letter_spacing=float(d.get("letter_spacing", 0)),
                         shadow=d.get("shadow") or None)
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, glyphs=n)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/fill")
def fill():
    """Flood fill: {layer, x, y, tolerance, contiguous, source}. Source types:
    color | gradient | pattern | node (any Fill-out node's image)."""
    d = request.json or {}
    try:
        content = _fill_content(d.get("source"), DOC.height, DOC.width)
        n = DOC.flood_fill(d["layer"], int(d["x"]), int(d["y"]), content,
                           tolerance=float(d.get("tolerance", 0.12)),
                           contiguous=bool(d.get("contiguous", True)),
                           selection=d.get("selection") or None)
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, filled=n)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/transform/meta")
def transform_meta():
    """Bounding box + validity for the interactive transform tool's target."""
    kind = request.args.get("kind", "layer")
    oid = request.args.get("id", "")
    layer = request.args.get("layer") or None      # selection auto-shrink hint
    try:
        return jsonify(ok=True, bbox=DOC.content_bbox(kind, oid, layer=layer),
                       width=DOC.width, height=DOC.height)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/transform/content.png")
def transform_content():
    """RGBA image of just the target (layer content, or mask/selection coverage
    tinted) -- the transform tool drags THIS around as a live preview."""
    import numpy as np
    kind = request.args.get("kind", "layer")
    oid = request.args.get("id", "")
    if kind == "strokes":
        img = DOC.render_strokes_rgba([s for s in oid.split(",") if s])
    elif kind == "layer":
        img = DOC.layer(oid).pixels
    else:
        f = (DOC.mask_by_id(oid).data if kind == "mask"
             else DOC.selection_by_id(oid).data)
        img = np.zeros((DOC.height, DOC.width, 4), np.float32)
        img[..., 0] = 0.33; img[..., 1] = 0.88; img[..., 2] = 0.83
        img[..., 3] = f * 0.65
    return _png(img)


_CAPS_CACHE = []


def _engine_has_walls():
    """Can the installed engine CONTAIN a simulation at the canvas edge?

    have() answers "does fluid_step exist", which was true long before
    it could do anything but wrap. The question an artist cares about is
    whether ink pushed off one edge comes back on the other, and that is
    a leCore 0.2.9 argument, not a faculty name -- so inspect the
    signature. Reported in /api/state so the UI can say so plainly
    rather than letting someone discover it by painting."""
    try:
        import inspect
        from . import mind
        return "boundary" in inspect.signature(mind().fluid_step).parameters
    except Exception:
        return False


def _capabilities():
    """What the installed leCore can do -- the UI hides features accordingly.

    Uses the engine's own `features()` manifest (leCore 0.2.4 / their C14) via
    lestudio.have(), which falls back to hasattr on older builds. Asking the
    engine beats hard-coding a list that rots silently: a renamed faculty and a
    missing one look identical from outside, which is how the wrap_webgl2 500
    reached a user."""
    # Computed ONCE. The installed engine cannot change while the server runs,
    # but this is called from /api/state, which the client polls -- and
    # engine_version() re-imports lecore each time. Measured at 90 ms per call
    # (561 module imports), i.e. the whole cost of a "cheap" status poll.
    if _CAPS_CACHE:
        return _CAPS_CACHE[0]
    from . import have, engine_version
    caps = {"media_walls": _engine_has_walls(),
            "invite": have("create_invite_link", "join_from_link"),
            "obs": have("obs_capture_profile"),
            "tighten_selection": have("tighten_selection"),
            # nodes that need faculties only newer leCore ships
            "proctex": have("texture_image"),
            "colour_ramp": have("ramp"),
            "refract": have("mask_refraction"),
            "clouds": have("cloud_scene"),
            "water": have("render_water"),
            "sampler": have("sample_image", "values_to_texture"),
            "shader_match": have("fit_shape"),
            "shader_palette": have("cosine_palette_to_glsl"),
            "anim_gif": True,
            "vectorize": _skimage_available(),
            "anim_mp4": _mp4_available(),
            "engine": engine_version()}
    _CAPS_CACHE.append(caps)
    return caps


@app.post("/api/invite")
def invite():
    """Mint a SINGLE-USE invite link for this session (leCore create_invite_link):
    the returned link is this server's URL with ?join=<code>; opening it joins
    automatically. Mint one link per guest -- codes are consumed on join."""
    if not _capabilities()["invite"]:
        return jsonify(error="Invites need leos-core >= 0.2.3 "
                             "(pip install -U leos-core)"), 400
    try:
        base = request.host_url
        r = mind().create_invite_link(base_url=base)
        INVITES["pending"].append({"code": r["code"], "at": time.time()})
        return jsonify(ok=True, code=r["code"], link=r["link"],
                       joined=[{"name": x["name"]} for x in INVITES["joined"]])
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/join")
def join():
    """Redeem an invite code (leCore join_from_link/admit). Send X-Client and an
    optional display name; on success the name shows in everyone's presence chip."""
    d = request.json or {}
    cid = request.headers.get("X-Client", "") or uuid.uuid4().hex[:8]
    code = str(d.get("code", "")).strip()
    name = str(d.get("name", "")).strip()[:24] or ("guest-" + cid[:4])
    if not _capabilities()["invite"]:
        return jsonify(error="Joining needs leos-core >= 0.2.3 on the host"), 400
    try:
        principal = mind().join_from_link(code, actor_id=cid)
        SYNC["names"][cid] = name
        INVITES["pending"] = [p for p in INVITES["pending"]
                              if p.get("code") != code]
        INVITES["joined"].append({"client": cid, "name": name, "at": time.time()})
        return jsonify(ok=True, name=name, actor=str(getattr(principal, "id", cid)))
    except Exception as e:
        return jsonify(error="That invite isn't valid (codes are single-use): "
                             + str(e)), 400


@app.post("/api/presence/name")
def presence_name():
    """Set the display name for this client (no invite needed on the host side)."""
    d = request.json or {}
    cid = request.headers.get("X-Client", "")
    if not cid:
        return jsonify(error="send an X-Client header"), 400
    SYNC["names"][cid] = str(d.get("name", "")).strip()[:24]
    SYNC.setdefault("viewing", {}).setdefault(cid, WS.active)
    return jsonify(ok=True)


@app.get("/api/obs")
def obs_profile():
    """OBS Browser-Source settings for putting this canvas on a stream (leCore
    obs_capture_profile), retargeted at OUR capture page /obs: width/height/fps
    to match the OBS canvas, the custom CSS, and step-by-step instructions."""
    preset = request.args.get("preset", "1080p")
    fps = int(request.args.get("fps", 30))
    transparent = request.args.get("transparent", "0") in ("1", "true")
    if not _capabilities()["obs"]:
        return jsonify(error="OBS profiles need leos-core >= 0.2.3 "
                             "(pip install -U leos-core)"), 400
    try:
        base = request.host_url
        p = mind().obs_capture_profile(base_url=base, preset=preset, fps=fps,
                                       transparent=transparent)
        ours = base + "obs?fps=%d%s" % (fps, "&transparent=1" if transparent else "")
        p["obs_steps"] = [s.replace(p["url"], ours) for s in p["obs_steps"]]
        p["url"] = ours
        return jsonify(ok=True, **p)
    except Exception as e:
        return jsonify(error=str(e)), 400


_STREAM_HEALTH = {}


def _png_bytes_for_live():
    """One live frame through the same path the stream uses."""
    import io as _io
    import numpy as _np
    from PIL import Image as _Img
    a = _np.clip(_np.asarray(DOC.composite()), 0, 1)
    im = _Img.fromarray((a * 255).astype("uint8"), "RGBA").convert("RGB")
    b = _io.BytesIO()
    im.save(b, "JPEG", quality=80)
    return b.getvalue()


@app.get("/api/stream/health")
def stream_health():
    """What frame rate this machine can actually sustain for the CURRENT canvas.

    The OBS dialog offers up to 4k60, but a frame costs what it costs: measured
    here so the UI can say "this canvas sustains about 4 fps" instead of letting
    someone configure 60 and wonder why their stream stutters. Timed on the real
    encode path, once, and cached against the mutation counter."""
    import time as _t
    from . import _MUT_REV
    key = (DOC.width, DOC.height, _MUT_REV[0])
    if _STREAM_HEALTH.get("key") != key:
        t0 = _t.time()
        _png_bytes_for_live()
        ms = (_t.time() - t0) * 1000.0
        _STREAM_HEALTH.clear()
        _STREAM_HEALTH.update(key=key, ms=ms)
    ms = _STREAM_HEALTH["ms"]
    fps = 1000.0 / max(ms, 1e-6)
    return jsonify(ok=True, frame_ms=round(ms, 1),
                   sustainable_fps=round(min(fps, 60.0), 1),
                   width=DOC.width, height=DOC.height,
                   live=bool(LIVE["on"]), clients=LIVE.get("clients", 0),
                   advice=("this canvas streams comfortably" if fps >= 24 else
                           "lower the canvas size or fps for a smoother stream"))


@app.get("/obs")
def obs_page():
    """The chromeless capture page OBS loads as a Browser Source: just the live
    output, edge to edge. transparent=1 polls PNG frames (alpha preserved);
    otherwise it rides the MJPEG live stream. Turns Live mode on automatically."""
    fps = max(1, min(int(request.args.get("fps", 30)), 60))
    transparent = request.args.get("transparent", "0") in ("1", "true")
    if not LIVE["on"]:                     # same start discipline as /api/live
        LIVE["on"] = True
        LIVE["fps"] = max(LIVE["fps"], min(float(fps), 30.0))
        threading.Thread(target=_live_loop, daemon=True).start()
    body_bg = "rgba(0,0,0,0)" if transparent else "#000"
    if transparent:
        # Chain each request off the previous frame instead of firing on a
        # fixed timer: if a frame takes longer than the interval (it does at
        # 1080p and above) a timer just queues requests the server cannot
        # answer. Back off on error and keep trying, so a hiccup does not
        # freeze the overlay for the rest of the stream.
        inner = ('<canvas id="c"></canvas><script>'
                 'const c=document.getElementById("c"),x=c.getContext("2d");'
                 'const gap=%d;let img=new Image(),fails=0;'
                 'img.onload=()=>{c.width=img.width;c.height=img.height;'
                 'x.clearRect(0,0,c.width,c.height);x.drawImage(img,0,0);'
                 'fails=0;setTimeout(next,gap);};'
                 'img.onerror=()=>{fails=Math.min(fails+1,6);'
                 'setTimeout(next,gap*Math.pow(2,fails));};'
                 'function next(){img.src="/api/graph/output.png?t="+Date.now();}'
                 'next();</script>' % int(1000 / fps))
    else:
        inner = '<img src="/api/stream.mjpg" alt="">'
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            'html,body{margin:0;padding:0;background:%s;overflow:hidden;'
            'width:100vw;height:100vh}'
            'img,canvas{width:100vw;height:100vh;object-fit:contain;display:block}'
            '</style></head><body>%s</body></html>' % (body_bg, inner))


@app.post("/api/transform")
def transform():
    """Transform content: {"kind": "layer"|"selection"|"strokes", "id"|"ids", "sx","sy","deg","dx","dy","pivot"?}."""
    d = request.json or {}
    try:
        DOC.transform(d["kind"], d["id"],
                      sx=float(d.get("sx", 1)), sy=float(d.get("sy", 1)),
                      deg=float(d.get("deg", 0)),
                      dx=float(d.get("dx", 0)), dy=float(d.get("dy", 0)),
                      layer=d.get("layer") or None)
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/histogram")
def histogram():
    """Per-channel tone distribution of the composite (or one layer).

    Sampled, not exhaustive: a display histogram only needs the SHAPE, and
    striding a 1920x1080 composite down to ~250k pixels gives a curve that is
    visually identical while costing a fraction of the time. Reported so the
    caller knows what it is looking at.
    """
    bins = max(16, min(256, int(request.args.get("bins", 64))))
    lid = request.args.get("layer") or None
    try:
        img = DOC.layer(lid).pixels[..., :3] if lid else DOC.composite()[..., :3]
    except Exception as e:
        return jsonify(error=str(e)), 400
    a = np.asarray(img, np.float32)
    step = max(1, int((a.shape[0] * a.shape[1] // 250000) ** 0.5))
    a = a[::step, ::step]
    out = {}
    for i, ch in enumerate("rgb"):
        out[ch] = np.histogram(a[..., i], bins=bins, range=(0.0, 1.0))[0].tolist()
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    out["lum"] = np.histogram(lum, bins=bins, range=(0.0, 1.0))[0].tolist()
    return jsonify(ok=True, bins=bins, sampled=(step > 1), step=step,
                   clipped_black=float((lum <= 0.002).mean()),
                   clipped_white=float((lum >= 0.998).mean()), **out)


@app.post("/api/strokes/select")
def strokes_select():
    """Pick strokes under a point and grow the set along paint order.

    Body: {x, y, layer?, anchors?, forward, back, ignore[], add}
    Returns the resolved set with each stroke's metadata AND its path, so the
    client can outline them on the canvas without another round trip."""
    d = request.json or {}
    lay = d.get("layer") or None
    anchors = list(d.get("anchors") or [])
    if d.get("x") is not None and d.get("y") is not None:
        hit = DOC.strokes_at(float(d["x"]), float(d["y"]), layer=lay)
        if hit:
            if d.get("add"):
                anchors = anchors + [h for h in hit[:1] if h not in anchors]
            else:
                anchors = hit[:1]
        elif not d.get("add"):
            anchors = []
    try:
        ids = DOC.resolve_stroke_selection(anchors,
                                           forward=int(d.get("forward", 0)),
                                           back=int(d.get("back", 0)),
                                           ignore=d.get("ignore") or (),
                                           layer=lay)
        return jsonify(ok=True, anchors=anchors,
                       strokes=[DOC.stroke_meta(s) for s in ids],
                       paths={s: DOC.stroke_by_id(s)["points"] for s in ids})
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/strokes/width")
def strokes_width():
    """Per-point width. {id, index, w, spread} sets one joint (spread blends
    back toward 1.0 over that many neighbours); {id, taper:{tip,root}} tapers
    the whole stroke root-to-tip."""
    d = request.json or {}
    try:
        if d.get("taper") is not None:
            t = d["taper"] or {}
            n = DOC.taper_stroke(d["id"], tip=float(t.get("tip", 0.15)),
                                 root=float(t.get("root", 1.0)))
            return jsonify(ok=True, points=n)
        w = DOC.set_point_width(d["id"], int(d["index"]), float(d["w"]),
                                spread=int(d.get("spread", 0)))
        return jsonify(ok=True, widths=w)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/strokes/rig")
def strokes_rig():
    """Turn a stroke into an armature: points become joints, segments become
    bones with a rest length. {id, pins:[joint indices]}"""
    d = request.json or {}
    try:
        return jsonify(ok=True, **DOC.rig_stroke(d["id"], d.get("pins")))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/strokes/simulate")
def strokes_simulate():
    """Advance a rigged stroke. Bone lengths are held, pinned joints stay put.
    {id, steps, gravity:[x,y], wind, damping, stiffness, seed}"""
    d = request.json or {}
    try:
        n = DOC.simulate_stroke(d["id"], steps=int(d.get("steps", 1)),
                                gravity=tuple(d.get("gravity", (0.0, 60.0))),
                                wind=float(d.get("wind", 0.0)),
                                damping=float(d.get("damping", 0.02)),
                                stiffness=float(d.get("stiffness", 1.0)),
                                seed=int(d.get("seed", 0)),
                                wind_detail=int(d.get("wind_detail", 12)),
                                record=bool(d.get("record", False)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, joints=n)


@app.post("/api/strokes/key")
def strokes_key():
    """Keyframe a stroke's pose, or apply the keyed animation at a time.
    {id, t, apply?} -- apply=true poses it, otherwise the pose is stored."""
    d = request.json or {}
    try:
        if d.get("apply"):
            times = DOC.apply_stroke_keys(d["id"], float(d.get("t", 0)),
                                          interp=str(d.get("interp", "linear")))
        else:
            times = DOC.key_stroke(d["id"], float(d.get("t", 0)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, keys=times)


@app.post("/api/strokes/points")
def strokes_points():
    """Point-level selection: [[stroke_id, index], ...] near (x, y), spanning
    strokes. The points are the real data; a stroke is the shape over them."""
    d = request.json or {}
    sel = DOC.points_at(float(d.get("x", 0)), float(d.get("y", 0)),
                        radius=float(d.get("radius", 12)),
                        layer=d.get("layer") or None)
    return jsonify(ok=True, points=[[s, i] for s, i in sel])


@app.post("/api/strokes/move")
def strokes_move():
    """Move an explicit set of points -- the primitive under nudging."""
    d = request.json or {}
    sel = [(p[0], int(p[1])) for p in (d.get("points") or [])]
    try:
        n = DOC.move_points(sel, float(d.get("dx", 0)), float(d.get("dy", 0)),
                            falloff=float(d.get("falloff", 0)),
                            strength=float(d.get("strength", 1)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, moved=n)


_STROKE_CLIP = []                     # deep-copied records, detached from any doc


@app.post("/api/strokes/clipboard")
def strokes_clipboard():
    """Copy / cut / paste whole strokes.

    The clip holds DEEP COPIES of the records, detached from the document --
    so cut can delete the originals (ink and all) and paste can re-create
    them later, on any layer, even after undo has rewritten history."""
    import copy as _copy
    d = request.json or {}
    act = d.get("action")
    try:
        if act in ("copy", "cut"):
            ids = d.get("ids") or []
            recs = [_copy.deepcopy(DOC.stroke_by_id(s)) for s in ids]
            if act == "cut":
                DOC.delete_strokes(ids)          # guard applies; refuses dirty
            del _STROKE_CLIP[:]
            _STROKE_CLIP.extend(recs)
            GRAPH.commit_layer_outputs()
            return jsonify(ok=True, count=len(recs))
        if act == "paste":
            if not _STROKE_CLIP:
                return jsonify(error="the stroke clipboard is empty"), 400
            tgt = d.get("layer") or _STROKE_CLIP[0]["layer"]
            DOC.layer(tgt)                       # raises on a bad id
            dx = float(d.get("dx", 14)); dy = float(d.get("dy", 14))
            DOC.record("Paste strokes", only=[tgt])
            if not any(k["layer"] == tgt for k in DOC.strokes):
                DOC._capture_replay_base(tgt)
            faithful = DOC.replay_is_faithful(tgt)
            out = []
            for rec in _STROKE_CLIP:
                DOC._stroke_n = getattr(DOC, "_stroke_n", len(DOC.strokes)) + 1
                nk = {"id": "K%d" % DOC._stroke_n, "layer": tgt,
                      "points": [[p[0] + dx, p[1] + dy] + list(p[2:])
                                 for p in rec["points"]],
                      "brush": dict(rec["brush"])}
                DOC.strokes.append(nk)
                out.append(nk["id"])
            if faithful:
                DOC._rebuild_after_stroke_edit(tgt)
            else:
                for sid in out:
                    k = DOC.stroke_by_id(sid); b = k["brush"]
                    DOC.paint(tgt, [tuple(p[:2]) for p in k["points"]],
                              color=tuple(b.get("color", (0, 0, 0))),
                              radius=float(b.get("radius", 8.0)),
                              opacity=float(b.get("opacity", 1.0)),
                              erase=bool(b.get("erase")),
                              hardness=float(b.get("hardness", 0.7)),
                              record=False, stroke_new=False)
            GRAPH.commit_layer_outputs()
            return jsonify(ok=True, ids=out)
        return jsonify(error="action must be copy, cut or paste"), 400
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/strokes/duplicate")
def strokes_duplicate():
    """Copy strokes, optionally to another layer -- the paste primitive."""
    d = request.json or {}
    try:
        ids = DOC.duplicate_strokes(d.get("ids") or [],
                                    dx=float(d.get("dx", 14)),
                                    dy=float(d.get("dy", 14)),
                                    layer=d.get("layer") or None)
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, ids=ids)


@app.post("/api/strokes/delete")
def strokes_delete():
    """Remove strokes and their ink (needs a faithful replay)."""
    d = request.json or {}
    try:
        n = DOC.delete_strokes(d.get("ids") or [])
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, deleted=n)


@app.post("/api/strokes/tolayer")
def strokes_tolayer():
    """Move strokes to another layer, ink and all."""
    d = request.json or {}
    try:
        n = DOC.strokes_to_layer(d.get("ids") or [], d["layer"])
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, moved=n)


@app.post("/api/strokes/pull")
def strokes_pull():
    """Thread pull: {id, index, x, y}. The joint goes to the point and the
    stroke follows under its segment-length constraints; rigged strokes keep
    their pins and rest lengths."""
    d = request.json or {}
    try:
        n = DOC.pull_stroke(d["id"], int(d["index"]),
                            float(d["x"]), float(d["y"]))
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, points=n)


@app.post("/api/strokes/smooth")
def strokes_smooth():
    """Laplacian-relax stroke points -- rounds off shaky freehand corners."""
    d = request.json or {}
    try:
        n = DOC.smooth_strokes(d.get("ids") or [],
                               amount=float(d.get("amount", 0.5)),
                               iterations=int(d.get("iterations", 2)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, points=n)


@app.post("/api/strokes/split")
def strokes_split():
    """Cut one stroke into two at a point index."""
    d = request.json or {}
    try:
        parts = DOC.split_stroke(d["id"], int(d["index"]))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, parts=parts)


@app.post("/api/strokes/join")
def strokes_join():
    """Merge compatible strokes into one, in paint order. Refuses when the
    brush settings differ -- a join that discarded a colour or radius would be
    losing work silently."""
    d = request.json or {}
    try:
        sid = DOC.join_strokes(d.get("ids") or [],
                               gap=float(d.get("gap", 1e9)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, id=sid)


@app.post("/api/nudge")
def nudge():
    """Push recorded stroke PATHS around instead of smearing pixels.

    Body: {layer, points:[[x,y]...], radius, strength}. Returns how many
    stroke points moved; 0 means the layer could not be replayed faithfully
    from its strokes (something else contributed pixels), so nothing was
    touched rather than risking the artwork."""
    d = request.json or {}
    try:
        n = DOC.nudge_strokes(d["layer"], d.get("points") or [],
                              radius=float(d.get("radius", 40)),
                              strength=float(d.get("strength", 1.0)),
                              record=bool(d.get("record", True)))
    except Exception as e:
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    # NB: do not re-verify here -- nudge already rebuilt the layer from the
    # strokes, so it is replayable by construction and another check would
    # cost a full replay for nothing.
    return jsonify(ok=True, moved=n, replayable=(n > 0))


@app.errorhandler(Exception)
def _api_error(e):
    """Any unhandled failure under /api returns JSON the UI can show.

    An empty 500 gave the user a silent no-op -- the click just did nothing,
    with the reason only in the server log they never see."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    if request.path.startswith("/api"):
        app.logger.exception("unhandled error on %s", request.path)
        msg = str(e) or e.__class__.__name__
        if isinstance(e, KeyError):
            msg = "no such item: %s" % msg.strip("'")
        elif isinstance(e, MemoryError):
            msg = "not enough memory for that operation"
        return jsonify(error=msg), 500
    raise e


_PREFS = {}


@app.route("/api/prefs", methods=["GET", "POST"])
def prefs():
    """Small key/value store for UI preferences that should outlive a reload --
    currently just whether the first-run hints have been dismissed."""
    if request.method == "POST":
        _PREFS.update({str(k): v for k, v in (request.json or {}).items()})
    return jsonify(ok=True, prefs=_PREFS)


@app.post("/api/layer/resource")
def layer_resource():
    """Re-render a PLACED image layer from its original pixels. {layer}

    An import fitted to the canvas is lossy; keeping the file's own pixels
    means a later resize can recover the detail instead of upscaling."""
    d = request.json or {}
    lid = d.get("layer") or (DOC.layers[-1].id if DOC.layers else None)
    try:
        ok = DOC.replace_from_source(lid)
    except Exception as e:
        return jsonify(error=str(e)), 400
    if not ok:
        return jsonify(ok=False,
                       reason="this layer was not placed from an image file, "
                              "so there is no original to re-render from"), 200
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/layer/compare")
def layer_compare():
    """Perceptual similarity between a layer and the autosave's copy of it.

    "Has this changed since the last autosave" is answered uselessly by byte
    equality -- every stroke changes bytes. This says whether the change is
    one anyone would SEE: an imperceptible tonal shift scores ~0.99, half the
    image going black scores ~0.62."""
    d = request.json or {}
    lid = d.get("layer") or (DOC.layers[-1].id if DOC.layers else None)
    try:
        cur = DOC.layer(lid).pixels
    except Exception as e:
        return jsonify(error=str(e)), 400
    src = getattr(DOC.layer(lid), "source", None)
    if src is None:
        return jsonify(ok=False,
                       reason="nothing to compare against: this layer was not "
                              "placed from an image file"), 200
    from . import image_similarity
    sim = image_similarity(cur, src)
    if sim is None:
        return jsonify(ok=False,
                       reason="this engine build cannot compare images"), 200
    # Thresholds CALIBRATED against the metric rather than guessed: on known
    # differences it scores +1/255 = 0.998, +2% exposure = 0.993, +10% = 0.964,
    # a 4 px roll = 0.691, half the image blacked = 0.659, flat grey = 0.451.
    # So it is strict about structure and forgiving about tone -- and a first
    # guess of "0.9 = close" would have called an untouched placed layer
    # "clearly different", which is how this was caught.
    return jsonify(ok=True, similarity=round(sim, 4),
                   verdict=("visually identical" if sim >= 0.995 else
                            "barely changed" if sim >= 0.96 else
                            "noticeably changed" if sim >= 0.85 else
                            "very different"))


@app.post("/api/layer/revector")
def layer_revector():
    """Re-render a layer's strokes at the CURRENT resolution. {layer}

    After a resize the pixels are an upscale, but the stroke paths are exact --
    repainting them lands bit-identical to a native-resolution render."""
    d = request.json or {}
    lid = d.get("layer") or (DOC.layers[-1].id if DOC.layers else None)
    try:
        ok = DOC.revector_layer(lid)
    except Exception as e:
        return jsonify(error=str(e)), 400
    if not ok:
        return jsonify(ok=False,
                       reason="this layer has content that was not painted as "
                              "strokes, so it cannot be re-rendered"), 200
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/selection/keep")
def selection_keep():
    """Promote the working selection into the saved list. {id, name?}

    Selections are momentary by default -- drag a marquee, paint inside it,
    move on -- so only the ones a user deliberately keeps get a permanent
    entry."""
    d = request.json or {}
    try:
        sel = DOC.keep_selection(d["id"], d.get("name"))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, selection=sel.meta())


@app.post("/api/reorient")
def reorient():
    """Rotate or flip the whole document: {op: rot90|rot270|rot180|fliph|flipv}."""
    d = request.json or {}
    try:
        DOC.reorient(str(d.get("op", "")))
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, width=DOC.width, height=DOC.height)


@app.post("/api/crop")
def crop():
    """Crop the document to {"x0","y0","x1","y1"} (all layers, masks, selections, strokes and impasto follow)."""
    d = request.json or {}
    try:
        if d.get("selection"):
            box = DOC.selection_bbox(d["selection"])
        elif all(k in d for k in ("x0", "y0", "x1", "y1")):
            box = (d["x0"], d["y0"], d["x1"], d["y1"])
        else:
            # the likeliest first attempt: crop with nothing selected. "'x0'"
            # told the user nothing about what to do.
            return jsonify(error="select an area first, then crop to it"), 400
        DOC.crop(*box)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/workspace.lews")
def workspace_save():
    """Download the whole workspace as a .lews file (all docs, graphs, strokes, impasto height, replay bases)."""
    from . import ASSETS
    asset_secs = [{"kind": "lestudio.asset", "id": k,
                   "meta": {"name": a["name"], "ext": a["ext"]},
                   "arrays": {"data": np.frombuffer(a["data"], np.uint8)}}
                  for k, a in ASSETS.items()]
    data = save_workspace(WS.docs, WS.graphs, WS.active,
                          extras=list(WS.extras) + asset_secs)
    return send_file(io.BytesIO(data), mimetype="application/octet-stream",
                     as_attachment=True, download_name="workspace.lews")


def _load_workspace_bytes(data):
    """Shared loader for workspace_open and autosave restore."""
    from . import register_asset
    docs, graphs, active, extras = load_workspace(data)
    ours, foreign = [], []
    for sec in extras:
        (ours if sec.get("kind") == "lestudio.asset" else foreign).append(sec)
    WS.docs, WS.graphs, WS.active, WS.extras = docs, graphs, active, foreign
    for sec in ours:                                # uploaded models ride along
        register_asset(sec["meta"]["name"], sec["arrays"]["data"].tobytes(),
                       sec["meta"]["ext"], aid=sec.get("id"))
    WS._wire()


@app.post("/api/workspace/open")
def workspace_open():
    """Open a .lews workspace (multipart file). Replaces the current workspace."""
    _load_workspace_bytes(request.files["file"].read())
    return jsonify(ok=True)


_AUTOSAVE_PATH = os.path.expanduser("~/.lestudio_autosave.lews")


@app.post("/api/autosave")
def autosave_write():
    """Write the whole workspace to a fixed sidecar file. The client calls
    this on a timer while there are unsaved changes; an explicit Save is
    still the person's own file -- this is just the crash net."""
    from . import ASSETS
    asset_secs = [{"kind": "lestudio.asset", "id": k,
                   "meta": {"name": a["name"], "ext": a["ext"]},
                   "arrays": {"data": np.frombuffer(a["data"], np.uint8)}}
                  for k, a in ASSETS.items()]
    try:
        data = save_workspace(WS.docs, WS.graphs, WS.active,
                              extras=list(WS.extras) + asset_secs)
    except Exception as e:
        # Autosave is the crash net. If it cannot run, the person needs to
        # know their work is NOT being kept behind them and to save it
        # themselves -- a 500 carrying a raw Python error (this failed with
        # "No module named 'holographic'") tells them neither.
        return jsonify(ok=False, saved=False,
                       error="autosave is not working, so nothing is being "
                             "kept for you in the background - save your work "
                             "yourself (%s)" % type(e).__name__), 200
    # excluded from the after_request rev bump: an autosave changes NOTHING in
    # the workspace, and bumping made every other client run its full
    # foreign-edit refresh -- with several editors' timers that read as the UI
    # "blinking while idle".
    tmp = _AUTOSAVE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, _AUTOSAVE_PATH)                 # atomic: never half a file
    return jsonify(ok=True, bytes=len(data))


@app.get("/api/autosave")
def autosave_info():
    """Autosave status: {exists, time, size}."""
    try:
        st = os.stat(_AUTOSAVE_PATH)
    except FileNotFoundError:
        return jsonify(ok=True, exists=False)
    # Offering a recovery that then fails is worse than offering none: it is
    # shown right after a crash, when the user most wants to believe it. A
    # corrupt or truncated file (a crash mid-write, a full disk) reports
    # `usable: False` so the banner can say so instead of promising a restore
    # and dying on "File is not a zip file".
    import zipfile as _zip
    usable, why = True, None
    try:
        # A magic-byte check is not enough: a file truncated mid-write keeps
        # its PK header and still reads as fine. Verify the ARCHIVE, which is
        # what a restore actually needs -- cheap (a directory read, no
        # decompression) and it catches the truncation case that matters.
        with _zip.ZipFile(_AUTOSAVE_PATH) as z:
            if z.testzip() is not None or not z.namelist():
                usable, why = False, "the autosave file is damaged"
    except (_zip.BadZipFile, EOFError):
        usable, why = False, "the autosave file is damaged or incomplete"
    except OSError as e:
        usable, why = False, str(e)
    return jsonify(ok=True, exists=True, usable=usable, why=why,
                   age_s=round(time.time() - st.st_mtime), bytes=st.st_size)


@app.post("/api/autosave/restore")
def autosave_restore():
    """Restore the autosaved workspace."""
    try:
        with open(_AUTOSAVE_PATH, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return jsonify(error="no autosave on disk"), 404
    try:
        _load_workspace_bytes(raw)
    except Exception as e:
        # Say it in the user's terms and keep the current document intact --
        # a failed recovery must not also cost them what they have now.
        app.logger.warning("autosave restore failed: %s", e)
        return jsonify(error="that autosave is damaged and cannot be "
                             "recovered; your current work is untouched"), 400
    return jsonify(ok=True)


@app.post("/api/paste")
def paste_image():
    """Paste external image data as a new PLACED layer: {"png": base64}
    (any format PIL reads). The full source is kept at native scale --
    content larger than the document hangs off the canvas untrimmed;
    /api/place adjusts its centre, scale, and rotation and re-rasterises
    from the source, so nothing is ever lost to the edges."""
    import base64 as _b64
    import io as _io
    from PIL import Image as _PImage
    d = request.json or {}
    try:
        raw = _b64.b64decode(d["png"].split(",")[-1])
        im = _PImage.open(_io.BytesIO(raw)).convert("RGBA")
    except Exception as ex:
        return jsonify(error="could not read image data: %s" % ex), 400
    arr = np.asarray(im, np.float32) / 255.0
    l = DOC.add_layer(d.get("name") or "Pasted", pixels=arr, placed=True)
    if getattr(l, "source", None) is None:
        l.source = arr                      # same-size pastes place too
    DOC.place_source(l.id, record=False)
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, id=l.id, w=int(arr.shape[1]),
                   h=int(arr.shape[0]),
                   oversize=bool(arr.shape[0] > DOC.height
                                 or arr.shape[1] > DOC.width))


@app.post("/api/place")
def place_layer():
    """Adjust a placed layer's transform: {"id", "x"?, "y"?, "scale"?,
    "rot"?}. Re-rasterises from the retained source (undoable); 404 if
    the layer has no source."""
    d = request.json or {}
    try:
        ok = DOC.place_source(d["id"], x=d.get("x"), y=d.get("y"),
                              scale=d.get("scale"), rot=d.get("rot"))
    except KeyError:
        return jsonify(error="no such layer"), 404
    if not ok:
        return jsonify(error="layer has no retained source"), 404
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, place=DOC.layer(d["id"]).place)


@app.post("/api/open")
def open_image():
    """Open an image file (multipart) as a new document; adopts its resolution and DPI."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file was uploaded"), 400
    raw = f.read()
    if not raw:
        return jsonify(error="that file is empty"), 400
    if len(raw) > 80 * 1024 * 1024:
        return jsonify(error="that file is %.0f MB; the limit is 80 MB"
                             % (len(raw) / 1e6)), 400
    try:
        img = decode_image(raw)
        src_dpi = image_dpi(raw)
    except Exception:
        # Pillow's message embeds a BytesIO repr -- "cannot identify image file
        # <_io.BytesIO object at 0x7f..>" told the user nothing about their file.
        name = f.filename or "that file"
        return jsonify(error="could not read %s — is it a PNG, JPEG, WebP or "
                             "similar image?" % name), 400
    ih, iw = img.shape[0], img.shape[1]
    # Adopt the image's own resolution when the document is still untouched.
    # Fitting a 2400x1600 photo into a default 768x512 canvas silently threw
    # away two thirds of the user's pixels, unrecoverably and with no notice.
    untouched = (len(DOC.layers) == 1 and not DOC.layers[0].pixels[..., 3].any()
                 and not DOC.strokes and not DOC._undo)
    adopted = False
    if untouched and (iw, ih) != (DOC.width, DOC.height):
        if iw * ih > 80_000_000:
            return jsonify(error="that image is %.0f megapixels; the limit is 80"
                                 % (iw * ih / 1e6)), 400
        DOC.resize(iw, ih, "canvas")
        if src_dpi:
            DOC.dpi = float(src_dpi)
        adopted = True
    elif (iw, ih) != (DOC.width, DOC.height):
        # existing work: say what happened rather than resampling in silence
        pass
    DOC.add_layer(f.filename or "Imported", img, placed=True)
    return jsonify(ok=True, adopted=adopted, width=DOC.width, height=DOC.height,
                   dpi=DOC.dpi, source=[iw, ih],
                   fitted=(not adopted and (iw, ih) != (DOC.width, DOC.height)))


@app.get("/api/composite.png")
def comp_png():
    """The composited canvas as PNG. ?fmt=auto serves JPEG when fully opaque; ?maxw= caps width. Window-patched cache: cheap after strokes."""
    with _DOC_LOCK:
        return _comp_png_locked()


def _comp_png_locked():
    # ?maxw= is the width the CANVAS is actually showing. The composite is then
    # built at the smallest power-of-two reduction that still has at least that
    # many pixels -- never fewer than the screen displays, and skipped entirely
    # when a visible layer carries a mask (see composite_display).
    from . import composite_display
    try:
        maxw = int(request.args.get("maxw", 0)) or None
    except ValueError:
        maxw = None
    from . import composite_cached, _MUT_REV
    vmode = getattr(DOC, "view3d", "flat")
    from . import composite_lit, _doc_emission
    lit = (any(li.get("enabled") for li in getattr(DOC, "lights", []))
           or _doc_emission(DOC) is not None
           or getattr(DOC, "vantage", "above") == "below")
    van = getattr(DOC, "vantage", "above")
    if vmode in ("ortho", "persp"):
        c = composite_lit(DOC, vmode, vantage=van)
        if maxw and maxw < DOC.width:
            from . import _resize as _rz
            c = _rz(c, max(1, int(round(DOC.height * maxw / DOC.width))),
                    maxw)
        return _png(c)
    key = (_MUT_REV[0], maxw, request.args.get("fmt", ""), DOC.id, van)
    hit = getattr(DOC, "_png_memo", None)
    if hit and hit[0] == key:
        # same frame, same request: rebuild a FRESH response from the cached
        # bytes -- the first version cached the Response object itself, and a
        # send_file stream is single-use: the second GET replayed a closed
        # file and 500'd (found by the browser E2E, not the unit tests, which
        # never fetched the same frame twice)
        return app.response_class(hit[1], mimetype=hit[2])
    if lit:
        # lights or emission are live: the lit composite (falls through to
        # the cached one byte-exactly when both are absent, so this branch
        # only runs when it changes the pixels)
        c = composite_lit(DOC, "flat", vantage=van)
        if maxw and maxw < DOC.width:
            from . import _resize as _rz
            c = _rz(c, max(1, int(round(DOC.height * maxw / DOC.width))),
                    maxw)
    elif maxw is None or maxw >= DOC.width:
        # full-resolution serve: the window-patched composite cache. A brush
        # stroke re-blends only its own rectangle (composite_patch), so this
        # path skips the 325 ms full composite the old route paid per stroke.
        c = composite_cached(DOC)
    else:
        # DOWNSCALED serve (pane narrower than the document -- the normal
        # case). composite_display re-composites at DISPLAY resolution and
        # deliberately does NOT use the full-res composite cache.
        #
        # MEASURED, and recorded because it looks like an obvious win and
        # is not: routing this through the cache so the media window-patch
        # path could help made playback WORSE -- 379 ms/frame against 210.
        # By mid-simulation the ink's dirty window covers a large fraction
        # of the canvas, so the "patch" is nearly a full composite, and it
        # runs at full resolution (1.5x the pixels of the display buffer)
        # and then still needs a downscale. Compositing straight at display
        # size wins. Left alone on purpose.
        # canvas_layers(), not layers: a layer standing on a WALL is not
        # canvas content. This path had the raw list, so a wall's paint
        # was still drawn flat on the picture AND projected as light --
        # the same strokes twice, which is what made the wall look like
        # an overlay with embossing.
        c = composite_display(DOC.canvas_layers(), DOC.height, DOC.width,
                              {m.id: m for m in DOC.masks}, maxw)
    # checker-through-alpha handled client-side; export straight RGBA
    resp = _png(c)
    try:
        resp.direct_passthrough = False
        DOC._png_memo = (key, resp.get_data(), resp.mimetype)
    except Exception:
        pass
    return resp


@app.get("/api/layer/<lid>.png")
def layer_png(lid):
    """One layer's pixels as PNG (thumbnail-cacheable via ?t=)."""
    return _png(DOC.layer(lid).pixels)


@app.post("/api/layer")
def layer_edit():
    """Layer ops: {"action": "add"|"delete"|"edit"|"duplicate"|"merge_down"|"move", "id"?, plus edit props: name, visible, opacity, blend, mask, mask_invert, alpha_lock, clip}."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        _nl = DOC.add_layer(d.get("name"), below=d.get("below"))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, id=_nl.id)
    elif act == "duplicate":
        DOC.duplicate_layer(d["id"])
        GRAPH.commit_layer_outputs()
    elif act == "merge_down":
        DOC.merge_layer_down(d["id"])
        GRAPH.commit_layer_outputs()
    elif act == "merge":
        DOC.merge_layers(d.get("ids", []))
        GRAPH.commit_layer_outputs()
    elif act == "merge_visible":
        DOC.merge_visible_layers()
        GRAPH.commit_layer_outputs()
    elif act == "clear":
        DOC.clear(d["id"], selection=d.get("selection") or None,
                  sel_invert=bool(d.get("sel_invert")))
    elif act in ("remove", "delete"):
        # both verbs: the docstring said "delete" for years while the code
        # only matched "remove", and an unknown action fell through to
        # ok:True -- a SILENT no-op that a dup-cleanup click hit in E2E
        # A document with NO layers is a dead end: painting into it 500s with
        # "no such item", and the app offers nothing to paint on. Keep the
        # last one rather than letting a click strand the user.
        if len([l for l in DOC.canvas_layers()]) <= 1:
            return jsonify(error="that is the only layer - add another before "
                                 "deleting this one, or clear it instead"), 400
        DOC.remove_layer(d["id"])
    elif act == "fill":
        try:
            DOC.fill_layer(d["id"], d.get("content") or
                           {"kind": "solid", "color": [1, 1, 1, 1]},
                           respect_alpha=bool(d.get("respect_alpha")))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        GRAPH.commit_layer_outputs()
    elif act == "flip":
        DOC.flip_layer(d["id"], axis=d.get("axis", "x"))
        GRAPH.commit_layer_outputs()
    elif act == "move":
        DOC.move_layer(d["id"], d["index"])
    elif act == "edit":
        props = {k: d.get(k) for k in ("name", "visible", "opacity", "blend",
                                       "mask_invert", "alpha_lock", "clip",
                                       "thickness", "vol_kind", "vol_ior",
                                       "vol_density", "absorbency", "emissive",
                                       "emissive_color", "reflect",
                                       "dispersion", "media_rate",
                                       "thickness",
                                       "z_off",
                                       "tilt_x", "tilt_y", "curve", "dome",
                                       "field",
                                       "field_mode", "field_strength", "curve_axis", "curve_profile", "dome_profile", "locked", "relief", "gravity", "gravity_angle", "optical", "media_res",
                                  "media_time")}
        if "mask" in d:
            props["mask"] = d.get("mask")
        if "bg" in d:                    # None -> transparent sheet
            props["bg"] = d.get("bg")
        if not d.get("id"):
            # a missing id 500'd with "no such item: None"
            return jsonify(error="which layer? this needs a layer id"), 400
        try:
            DOC.layer(d["id"])
        except KeyError:
            return jsonify(error="layer %s is gone - pick another in the "
                                 "layer list" % d["id"]), 400
        DOC.edit_layer(d["id"], **props)
        GRAPH.commit_layer_outputs()
    elif act:
        return jsonify(error="unknown layer action: %r" % act), 400
    return jsonify(ok=True)


@app.post("/api/group")
def group_edit():
    """Layer group ops: {"action": "add"|"delete"|"edit"|"assign", ...}."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        g = DOC.add_group(d.get("name"), d.get("layers", []))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, group=g)
    if act == "remove":
        DOC.remove_group(d["id"])
    elif act == "edit":
        DOC.edit_group(d["id"], d.get("name"), d.get("layers"))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/mask")
def mask_edit():
    """Mask ops: {"action": "add"|"delete"|"edit"|"from_selection", ...}. Attach to a layer via /api/layer edit {mask: id}."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        m = DOC.add_mask(d.get("name"))
        return jsonify(ok=True, mask=m.meta())
    if act == "duplicate":
        m = DOC.duplicate_mask(d["id"])
        return jsonify(ok=True, mask=m.meta())
    if act == "remove":
        DOC.remove_mask(d["id"])
    elif act == "edit":
        DOC.edit_mask(d["id"], d.get("name"))
    elif act == "move":
        DOC.move_mask(d["id"], d["index"])
    elif act == "merge":
        DOC.merge_masks(d.get("ids") or None)
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/select")
def select():
    """Make a selection: {"tool": "rect"|"ellipse"|"wand"|"lum"|"obj", "params": {...}, "mode": "new"|"add"|"sub"}."""
    d = request.json or {}
    # A selection built from NaN silently produced a mask nothing could use,
    # and a missing corner reported just "'x0'" -- true, and useless.
    prm = d.get("params") or {}
    if str(d.get("tool")) in ("rect", "ellipse"):
        try:
            for k in ("x0", "y0", "x1", "y1"):
                if k not in prm:
                    return jsonify(error="a %s selection needs x0, y0, x1 and "
                                         "y1" % d.get("tool")), 400
                prm[k] = _finite(prm[k], k, -1e6, 1e6)
        except _Gone as e:
            return jsonify(error=str(e)), 400
        d["params"] = prm
    try:
        sel = DOC.select(d["tool"], d.get("params", {}), mode=d.get("mode", "new"),
                         target=d.get("target"), name=d.get("name"),
                         feather=float(d.get("feather", 0)))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, selection=sel.meta())
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/selection")
def selection_edit():
    """Selection ops: {"action": "clear"|"invert"|"delete"|"keep", "id"?}."""
    d = request.json or {}
    act = d.get("action")
    if act == "remove":
        DOC.remove_selection(d["id"])
    elif act == "edit":
        DOC.edit_selection(d["id"], d.get("name"))
    elif act == "move":
        DOC.move_selection(d["id"], d["index"])
    elif act == "merge":
        DOC.merge_selections(d.get("ids") or None)
    elif act == "modify":
        DOC.modify_selection(d["id"], d["op"], d.get("amount", 1))
    elif act == "to_mask":
        m = DOC.selection_to_mask(d["id"], d.get("name"))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, mask=m.meta())
    return jsonify(ok=True)


@app.get("/api/selection/<sid>.png")
def selection_png(sid):
    """A selection's coverage as PNG."""
    return _png(DOC.selection_by_id(sid).data)


@app.post("/api/spline")
def spline_edit():
    """Spline ops: {"action": "add"|"edit"|"delete", "points": [{x,y,hx,hy}], ...}. Splines rail brushes and feed Stroke FX."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        p = DOC.add_spline(d.get("name"), d.get("points"), bool(d.get("closed")))
        return jsonify(ok=True, spline=p.meta())
    if act == "remove":
        DOC.remove_spline(d["id"])
    elif act == "edit":
        DOC.edit_spline(d["id"], d.get("name"), d.get("points"), d.get("closed"),
                        record=bool(d.get("record")))
    elif act == "stroke":
        DOC.stroke_spline(d["layer"], d["id"],
                          color=d.get("color", [0, 0, 0]),
                          radius=float(d.get("radius", 8)),
                          opacity=float(d.get("opacity", 1)),
                          hardness=float(d.get("hardness", 0.7)),
                          erase=bool(d.get("erase")),
                          selection=d.get("selection"),
                          sel_invert=bool(d.get("sel_invert")),
                          brush=d.get("brush"))
        GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/brush")
def brush_edit():
    """Custom brush ops: {"action": "add"|"edit"|"delete", tip image by asset id, spacing, dynamics}."""
    d = request.json or {}
    act = d.get("action")
    if act == "add":
        b = DOC.add_brush(d.get("name"))
        return jsonify(ok=True, brush=b.meta())
    if act == "remove":
        DOC.remove_brush(d["id"])
    elif act == "edit":
        DOC.edit_brush(d["id"], d.get("name"), d.get("spacing"), d.get("follow"),
                       d.get("j_angle"), d.get("j_size"), d.get("j_scatter"))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.get("/api/brush/<bid>.png")
def brush_png(bid):
    """A brush tip preview as PNG."""
    return _png(DOC.brush_by_id(bid).tip)


@app.get("/api/mask/<mid>.png")
def mask_png(mid):
    """A mask's data as PNG."""
    return _png(DOC.mask_by_id(mid).data)


@app.post("/api/paint")
def paint():
    """Paint a stroke: {"layer", "points": [[x,y,widthFactor?]...], "color", "radius", "opacity", "hardness", "erase"?, "media"?: "oil"|"acrylic"|"water" (impasto body + gravity), "material"?: a preset name from /api/state materials (gold, chrome, chalk, ...) or {"preset"?, "rough" 0..1, "metal" 0..1, "grain", "hold", "flow", "iters"} -- the stroke lays a PBR surface (per-pixel roughness/metalness lit by the composite; the brush colour is the albedo, so gold gleams in YOUR gold) plus a paint body, "load"?, "mode"?: "brush"|"knife" (the PALETTE KNIFE: shapes the paint already there instead of adding more, working the whole paint COLUMN across every stratum, with "knife": "smooth" to level a surface and cure stepping between layers, "push" to plough a ridge with volume conserved, "scrape" to take the tops off, "spread" to drag it into a thin film)|"blend" (the BLENDER: carries no pigment, softens and drags the WET paint already on canvas, gated by paint body -- and unlike smudge it is a recorded stroke, so the layer keeps full stroke editing and a blend re-derives when you nudge the colours under it)|"smudge"|"clone"|"heal"|"erase_strokes" (whole strokes under the path)|"erase_top" (only the topmost stroke)|"erase_undo" (restore the area to its pre-stroke base)|"erase_depth" (carve the impasto body first)|"node", "selection"?, "brush"?, "live"?, "record"}. Returns {ok, sid}. Layer alpha_lock is honoured and recorded."""
    d = request.json or {}
    # FIRST, before anything reads the payload. The bounding box below is
    # computed straight from the points, so NaN or a string in there crashed
    # with a 500 long before the engine was reached.
    try:
        d = _clean_paint(d)
    except _Gone as e:
        return jsonify(error=str(e)), 400
    mode = d.get("mode", "brush")
    # WHY DIDN'T THAT PAINT? Professional trust: a stroke that can have
    # no visible effect gets a diagnosis, never a silent no-op. These
    # states all bit during dogfooding or will bite an artist mid-flow.
    warn = None
    try:
        _l = DOC.layer(d.get("layer", ""))
    except Exception:
        _l = None
    if _l is not None and mode in ("brush", "knife", "blend", "smudge", "clone", "heal",
                                   "node"):
        import numpy as _np
        pts = d.get("points") or []
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            r = float(d.get("radius", 8)) + 2
            x0 = max(0, int(min(xs) - r))
            x1 = min(DOC.width, int(max(xs) + r) + 1)
            y0 = max(0, int(min(ys) - r))
            y1 = min(DOC.height, int(max(ys) + r) + 1)
        else:
            x0 = y0 = 0
            x1, y1 = DOC.width, DOC.height
        if not _l.visible:
            warn = "that landed on a HIDDEN layer — toggle its eye to see it"
        elif float(getattr(_l, "opacity", 1.0)) <= 0.02:
            warn = "this layer's opacity is ~0 — the stroke is there but invisible"
        elif getattr(_l, "alpha_lock", False)                 and float(_l.pixels[y0:y1, x0:x1, 3].max(initial=0.0)) < 0.01:
            warn = ("alpha lock recolours EXISTING pixels only, and this "
                    "area of the layer is empty — nothing to recolour")
        elif getattr(_l, "clip", False):
            idx = next((i for i, x in enumerate(DOC.layers)
                        if x.id == _l.id), 0)
            base = DOC.layers[idx - 1] if idx > 0 else None
            j = idx - 1
            while base is not None and getattr(base, "clip", False):
                j -= 1
                base = DOC.layers[j] if j >= 0 else None
            if base is not None                     and float(base.pixels[y0:y1, x0:x1, 3]
                              .max(initial=0.0)) < 0.01:
                warn = ("this layer clips onto %r, which is empty here — "
                        "the paint shows only where the base has pixels"
                        % base.name)
    try:
        try:
            resp = _paint_dispatch(d, mode)
        except _Gone as e:
            # a stale or hidden layer is a user error the app can explain,
            # not a server fault
            return jsonify(error=str(e)), 400
        if warn is not None:
            body = resp.get_json(silent=True) or {}
            body["warning"] = warn
            return jsonify(**body)
        return resp
    except ValueError as e:
        # a guard refusal (locked layer, stroke-content guard): the
        # client shows this as a toast instead of a silent no-op
        return jsonify(error=str(e)), 400


def _paint_dispatch(d, mode):
    DOC._edited_palette_last = False   # this edit was on the picture
    lid = d.get("layer")
    if lid is not None:
        try:
            _l = DOC.layer(lid)
        except KeyError:
            # deleting a layer and painting into it 500'd with a bare
            # "no such item"; it is a stale reference, not a server fault
            # name the layer as well as explaining: an error that identifies
            # the offending item is what makes a bug report actionable, and a
            # friendly message that drops the id trades one kind of useless
            # for another
            raise _Gone("layer %s is gone - pick another in the layer list"
                        % lid)
        # NOTE: a HIDDEN layer is deliberately NOT refused here. The stroke
        # lands and the response carries a warning ("that landed on a HIDDEN
        # layer -- toggle its eye to see it"), which does what the painter
        # asked AND explains it. I briefly made this a 400; that threw the
        # stroke away and broke the existing design. My user-test only read
        # the status code and the pixels, never the `warning` field, and
        # reported a silent failure that was not one.
    if mode == "knife":
        # the PALETTE KNIFE shapes existing paint across the whole stratum
        # chain rather than adding more
        DOC.knife(d["layer"], d["points"],
                  mode=str(d.get("knife", "smooth")),
                  radius=float(d.get("radius", 26)),
                  strength=float(d.get("opacity", 0.7)),
                  record=bool(d.get("record", True)),
                  stroke_new=bool(d.get("record", True)))
    elif mode == "blend":
        # the BLENDER: no pigment, works the wet paint already there. Unlike
        # smudge this is a recorded stroke, so the layer keeps stroke editing
        DOC.blend_stroke(d["layer"], d["points"],
                         radius=float(d.get("radius", 18)),
                         strength=float(d.get("opacity", 0.6)),
                         brush=d.get("brush"),
                         record=bool(d.get("record", True)),
                         stroke_new=bool(d.get("record", True)))
    elif mode == "smudge":
        DOC.smudge(d["layer"], d["points"], radius=float(d.get("radius", 12)),
                   strength=float(d.get("opacity", 0.6)), brush=d.get("brush"),
                   record=bool(d.get("record", True)))
    elif mode in ("erase_strokes", "erase_top"):
        n = DOC.erase_strokes(d["layer"], d["points"],
                              radius=float(d.get("radius", 12)),
                              topmost=(mode == "erase_top"))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, removed=int(n))
    elif mode == "erase_undo":
        DOC.erase_restore(d["layer"], d["points"],
                          radius=float(d.get("radius", 12)),
                          record=bool(d.get("record", True)))
    elif mode == "erase_depth":
        DOC.erase_depth(d["layer"], d["points"],
                        radius=float(d.get("radius", 12)),
                        strength=float(d.get("opacity", 1.0)),
                        record=bool(d.get("record", True)))
    elif mode == "heal":
        DOC.heal(d["layer"], d["points"], radius=float(d.get("radius", 14)),
                 record=bool(d.get("record", True)))
    elif mode == "clone":
        DOC.clone(d["layer"], d["points"], d["source"],
                  radius=float(d.get("radius", 12)),
                  opacity=float(d.get("opacity", 1)), brush=d.get("brush"),
                  record=bool(d.get("record", True)), origin=d.get("origin"))
    elif mode == "node":
        # Node paint: pigment from a Paint-out node's image. Evaluated through
        # the graph's memoised cache, so mid-stroke flushes cost one lookup.
        GRAPH.ensure_default()
        nid = str(d.get("node", ""))
        n = GRAPH.nodes.get(nid)
        if n is None or n.get("type") != "Paint out":
            return jsonify(error="pick a Paint out node first"), 400
        if not (n.get("inputs") or {}).get("image"):
            return jsonify(error="wire an image into the Paint out node first"), 400
        img = GRAPH.evaluate(nid)
        DOC.paint_image(d["layer"], d["points"], img,
                        radius=float(d.get("radius", 12)),
                        opacity=float(d.get("opacity", 1)),
                        hardness=float(d.get("hardness", 0.7)),
                        brush=d.get("brush"),
                        record=bool(d.get("record", True)),
                        selection=d.get("selection") or None,
                        sel_invert=bool(d.get("sel_invert")))
    elif d.get("live"):
        # live streaming: the client sends the FULL point list each flush and
        # the document repaints the whole stroke as one call, so the pixels
        # equal a single-call resolve exactly (see Document.paint_live)
        DOC.paint_live(d["layer"], d["points"],
                       first=bool(d.get("record", True)),
                       color=d.get("color", [0, 0, 0]),
                       radius=float(d.get("radius", 8)),
                       opacity=float(d.get("opacity", 1)),
                       erase=bool(d.get("erase")),
                       hardness=float(d.get("hardness", 0.7)),
                       target_mask=d.get("target_mask"),
                       selection=d.get("selection"),
                       sel_invert=bool(d.get("sel_invert")),
                       brush=d.get("brush"),
                       media=(d.get("media") or None),
                       material=(d.get("material") or None),
                       mix=float(d.get("mix", 0.0)),
                       real_brush=bool(d.get("real_brush", False)),
                       load=float(d.get("load", 0.6)),
                       taper=float(d.get("stroke_taper", 0.0)))
    else:
        DOC.paint(d["layer"], d["points"], color=d.get("color", [0, 0, 0]),
                  radius=float(d.get("radius", 8)), opacity=float(d.get("opacity", 1)),
                  erase=bool(d.get("erase")), hardness=float(d.get("hardness", 0.7)),
                  target_mask=d.get("target_mask"),
                  record=bool(d.get("record", True)),
                  selection=d.get("selection"), sel_invert=bool(d.get("sel_invert")),
                  brush=d.get("brush"),
                  media=(d.get("media") or None),
                  material=(d.get("material") or None),
                  mix=float(d.get("mix", 0.0)),
                  real_brush=bool(d.get("real_brush", False)),
                  load=float(d.get("load", 0.6)),
                  taper=float(d.get("stroke_taper", 0.0)))
    GRAPH.commit_layer_outputs()
    # the FX brush needs the id of the stroke it just painted, so it can hand
    # the path to its Stroke FX node without a second round trip
    sid = DOC.strokes[-1]["id"] if DOC.strokes else None
    patch = None
    if d.get("patch_ok"):
        # region-patch delivery: instead of the client re-fetching (and this
        # server re-encoding) the WHOLE frame after every stroke, hand back
        # just the dirty window of the patched composite cache. The full-frame
        # PNG encode was the last ~85 ms of per-stroke latency.
        from . import composite_cached, _MUT_REV
        rect = getattr(DOC, "_last_paint_rect", None)
        cc = getattr(DOC, "_ccache", None)
        if rect and cc is not None and cc["rev"] == _MUT_REV[0]:
            x0, y0, x1, y1 = rect
            if x1 > x0 and y1 > y0:
                import base64
                win = cc["buf"][y0:y1, x0:x1]
                patch = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                         "png": base64.b64encode(
                             png_bytes(win)).decode("ascii")}
    return jsonify(ok=True, sid=sid, patch=patch)


def _last_surface():
    """Where the last edit happened -- the picture, or the palette.

    Undo has to undo THE LAST THING YOU DID, wherever you did it. Mixing a
    colour on the palette and pressing Ctrl+Z did nothing at all, because
    undo only ever spoke to the picture while the palette kept its own stack.
    """
    pd = DOC.palette_doc(create=False)
    if pd is not None and getattr(DOC, "_edited_palette_last", False):
        return pd
    return DOC


@app.post("/api/undo")
def undo():
    """Undo the last operation, on the picture or the palette -- whichever was
    edited last. Includes impasto height."""
    with _DOC_LOCK:
        surf = _last_surface()
        ok = surf.undo()
        if not ok and surf is not DOC:
            DOC._edited_palette_last = False       # fall back to the picture
            ok = DOC.undo()
    return jsonify(ok=ok)


@app.post("/api/redo")
def redo():
    """Redo, on whichever surface was edited last."""
    with _DOC_LOCK:
        surf = _last_surface()
        ok = surf.redo()
        if not ok and surf is not DOC:
            ok = DOC.redo()
    return jsonify(ok=ok)


@app.post("/api/graph")
def set_graph():
    """Replace the node graph: {"nodes": [{id, type, params, inputs, x, y}]}. Inputs: "NID", "NID.socket", or [id, socket]; "param:<name>" keys wire values into parameters. Commits Layer out nodes."""
    GRAPH.set_graph((request.json or {}).get("nodes", []))
    n = GRAPH.commit_layer_outputs()
    return jsonify(ok=True, committed=n,
                   conflicts=[DOC.layer(c).name for c in getattr(GRAPH, "last_conflicts", [])
                              if any(l.id == c for l in DOC.layers)])


@app.patch("/api/graph/node/<nid>")
def patch_graph_node(nid):
    """Cheap iteration: update ONE node's params/inputs without re-POSTing the
    whole graph. Body: {params?:{}, inputs?:{socket: "node[.sock]"|null},
    pos?:[x,y]}. Bumps the sync rev exactly once (via after_request)."""
    GRAPH.ensure_default()
    d = request.json or {}
    try:
        node = GRAPH.patch_node(nid, params=d.get("params"),
                                inputs=d.get("inputs"), pos=d.get("pos"))
    except KeyError:
        return jsonify(error="no node '%s' in the graph" % nid), 404
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, node=node)


@app.get("/api/graph/output.png")
def graph_output():
    """The Output node's render as PNG."""
    if LIVE["on"] and LIVE["jpeg"] is not None and request.args.get("fmt") == "jpeg":
        # the live loop already evaluated + encoded this frame: zero extra work
        return send_file(io.BytesIO(LIVE["jpeg"]), mimetype="image/jpeg")
    GRAPH.ensure_default()
    nid = GRAPH.output_node()
    try:
        return _png(_renderable(GRAPH.evaluate(nid)))
    except Exception:
        c = DOC.composite()
        return _png(c[..., :3] * c[..., 3:4])


@app.get("/api/graph/sigs")
def graph_sigs():
    """{node_id: signature} for the whole graph (plus __output). Signatures move
    exactly when a node's result would -- new video frame, edited layer, changed
    param -- so a client polls THIS tiny JSON and re-fetches only what moved.
    A paused video costs zero image traffic."""
    GRAPH.ensure_default()
    out = {}
    for nid in list(GRAPH.nodes):
        try:
            out[nid] = GRAPH._sig(nid)[:16]
        except Exception:
            out[nid] = "err"
    onode = GRAPH.output_node()
    out["__output"] = out.get(onode, "none")
    out["__rev"] = SYNC["rev"]
    return jsonify(out)


def _renderable(val):
    """Graph values can be numbers now: render them as a labelled swatch."""
    import numpy as np
    if not isinstance(val, (int, float)):
        return val
    g = float(np.clip(val, 0, 1))
    img = np.full((96, 160, 3), g, np.float32)
    try:
        from PIL import Image as PImage, ImageDraw
        im = PImage.fromarray((img * 255).astype("uint8"))
        d = ImageDraw.Draw(im)
        txt = f"{val:.3g}"
        d.text((6, 6), txt, fill=(255, 80, 120) if g > 0.5 else (255, 200, 220))
        img = np.asarray(im).astype(np.float32) / 255.0
    except Exception:
        pass
    return img


@app.get("/api/graph/timings")
def graph_timings():
    """Seconds each node's LAST actual compute took (cache hits keep the old
    number). The UI shows a small clock chip on anything slow, so people can
    see where render time goes and reach for the detail dials."""
    t = {k: round(v, 3) for k, v in GRAPH.timings.items()}
    # If several nodes are individually slow, say whether spreading them across
    # cores would actually help HERE -- leCore knows the usable core count and
    # answers with a reason, which beats implying that more hardware is the fix.
    slow = [k for k, v in t.items() if v >= 0.8]
    advice = None
    if len(slow) >= 2:
        a = parallel_advice(n_jobs=len(slow),
                            ms_each=max(t[k] for k in slow) * 1000.0)
        advice = dict(a, slow_nodes=slow)
    return jsonify(ok=True, timings=t, parallel=advice)


@app.get("/api/graph/preview/<nid>.png")
def graph_preview(nid):
    """One node's render as PNG (memoised per signature)."""
    try:
        return _png(_renderable(GRAPH.evaluate(nid, request.args.get("sock", "out"))))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/graph/group")
def graph_group():
    """Bundle a set of existing nodes into one Group node (P1#8). Body:
    {ids:[...], label?}. Wires crossing the boundary become the group's
    imports (external → internal) and the group's own inputs inherit the
    outer sources; the downstream-most selected node becomes the output.
    Returns the new group node. The subgraph rides inside the graph/.lews."""
    d = request.json or {}
    ids = [i for i in (d.get("ids") or []) if i in GRAPH.nodes]
    if len(ids) < 2:
        return jsonify(error="select at least two nodes to group"), 400
    idset = set(ids)
    members = [json.loads(json.dumps(GRAPH.nodes[i])) for i in ids]
    # external wires: any member input whose source is OUTSIDE the selection
    imports, ext_inputs, k = {}, {}, 0
    for m in members:
        for sock, ref in list((m.get("inputs") or {}).items()):
            src_id = str(ref).split(".")[0]
            if src_id not in idset:                  # crosses the boundary
                ext = "in%d" % k; k += 1
                imports[ext] = m["id"]
                ext_inputs[ext] = ref
                m["inputs"].pop(sock)                # inner node fed via import
    # output = a selected node nothing else selected consumes
    consumed = set()
    for m in members:
        for ref in (m.get("inputs") or {}).values():
            consumed.add(str(ref).split(".")[0])
    sinks = [i for i in ids if i not in consumed]
    out_id = sinks[-1] if sinks else ids[-1]
    gid = "grp%d" % (max([int("".join(filter(str.isdigit, i)) or 0)
                          for i in GRAPH.nodes] + [0]) + 1)
    xs = [GRAPH.nodes[i].get("x", 0) for i in ids]
    ys = [GRAPH.nodes[i].get("y", 0) for i in ids]
    group = {"id": gid, "type": "Group",
             "params": {"label": d.get("label") or "Group",
                        "subgraph": members, "output": out_id,
                        "imports": imports},
             "inputs": ext_inputs,
             "x": sum(xs) / len(xs), "y": sum(ys) / len(ys),
             "title": d.get("label") or "Group"}
    # rewrite outer graph: drop members, add group, redirect refs to the group
    remaining = {i: n for i, n in GRAPH.nodes.items() if i not in idset}
    for n in remaining.values():
        for sock, ref in list((n.get("inputs") or {}).items()):
            sid, _, ssock = str(ref).partition(".")
            if sid in idset:
                n["inputs"][sock] = gid          # consumers point at the group
    remaining[gid] = group
    GRAPH.set_graph(list(remaining.values()))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, group=group)


@app.post("/api/graph/ungroup")
def graph_ungroup():
    """Explode a Group back into its member nodes (inverse of /group)."""
    d = request.json or {}
    gid = d.get("id")
    g = GRAPH.nodes.get(gid)
    if not g or g.get("type") != "Group":
        return jsonify(error="not a group node"), 400
    gp = g.get("params") or {}
    members = {m["id"]: json.loads(json.dumps(m)) for m in gp.get("subgraph", [])}
    imports = gp.get("imports") or {}
    for ext, inner_id in imports.items():            # restore external wires
        src = (g.get("inputs") or {}).get(ext)
        if src is not None and inner_id in members:
            itype = members[inner_id]["type"]
            from lestudio import OPS
            isock = (OPS[itype]["inputs"][0]
                     if itype in OPS and OPS[itype]["inputs"] else "image")
            members[inner_id].setdefault("inputs", {})[isock] = src
    out_id = gp.get("output")
    rest = {i: n for i, n in GRAPH.nodes.items() if i != gid}
    for n in rest.values():                          # consumers of the group -> its output
        for sock, ref in list((n.get("inputs") or {}).items()):
            if str(ref).split(".")[0] == gid:
                n["inputs"][sock] = out_id
    rest.update(members)
    GRAPH.set_graph(list(rest.values()))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, ids=list(members))


@app.get("/api/graph/render.png")
def graph_render_at():
    """Render the graph (default: the Output node) at an arbitrary resolution,
    not the document size -- procedural graphs gain real detail when scaled up.
    Query: w, h, node (optional), sock (optional). Area is capped server-side."""
    GRAPH.ensure_default()
    try:
        w = int(request.args.get("w", DOC.width))
        h = int(request.args.get("h", DOC.height))
    except ValueError:
        return jsonify(error="w and h must be integers"), 400
    if w * h > 8192 * 8192:
        return jsonify(error="requested render is too large (max ~67 MP)"), 400
    nid = request.args.get("node") or GRAPH.output_node()
    sock = request.args.get("sock", "out")
    try:
        return _png(_renderable(GRAPH.render_at(nid, w, h, sock)))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/analyze")
def analyze():
    """Measure regions of a rendered node -- the 'is the sky actually blue?'
    door. Body: {node?, sock?, regions:[{name, box:[x0,y0,x1,y1] in 0..1,
    metrics:[...]}]}. Metrics: mean_rgb, dominant_hue, brightness,
    fraction_matching (needs {hue:[lo,hi]} or {darker_than:v}). Powered by the
    same maths the meadow rescue used, so a script or CI can verify a .lews
    without pulling PNGs apart."""
    import colorsys
    GRAPH.ensure_default()
    d = request.json or {}
    nid = d.get("node") or GRAPH.output_node()
    try:
        img = np.asarray(_renderable(GRAPH.evaluate(nid, d.get("sock", "out"))),
                         np.float32)
    except Exception as e:
        return jsonify(error=str(e)), 400
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    H, W = img.shape[:2]

    def dom_hue(reg):
        hs = []
        flat = reg.reshape(-1, 3)
        for px in flat[:: max(1, len(flat) // 800)]:
            hh, ss, vv = colorsys.rgb_to_hsv(*np.clip(px, 0, 1))
            if ss > 0.15 and vv > 0.12:
                hs.append(hh * 360)
        return round(float(np.median(hs)), 1) if hs else None

    out = []
    for r in d.get("regions", []):
        box = r.get("box", [0, 0, 1, 1])
        x0, y0, x1, y1 = [int(box[0] * W), int(box[1] * H),
                          int(box[2] * W), int(box[3] * H)]
        reg = img[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
        res = {"name": r.get("name", "")}
        for metric in r.get("metrics", ["mean_rgb"]):
            if metric == "mean_rgb":
                res["mean_rgb"] = [round(float(reg[..., i].mean()), 4)
                                   for i in range(3)]
            elif metric == "dominant_hue":
                res["dominant_hue"] = dom_hue(reg)
            elif metric == "brightness":
                res["brightness"] = round(float(reg.mean()), 4)
            elif metric == "fraction_matching":
                spec = r.get("match", {})
                if "hue" in spec:
                    lo, hi = spec["hue"]
                    hsv = np.array([[colorsys.rgb_to_hsv(*np.clip(px, 0, 1))
                                     for px in row] for row in reg[::4, ::4]])
                    hh = hsv[..., 0] * 360
                    sel = (hh >= lo) & (hh <= hi) & (hsv[..., 1] > 0.15)
                    res["fraction_matching"] = round(float(sel.mean()), 4)
                elif "darker_than" in spec:
                    res["fraction_matching"] = round(
                        float((reg.mean(-1) < spec["darker_than"]).mean()), 4)
        out.append(res)
    return jsonify(ok=True, node=nid, size=[W, H], regions=out)


@app.post("/api/graph/apply")
def graph_apply():
    """Bake a node's output into a layer: {"node", "layer"}."""
    d = request.json or {}
    try:
        l = GRAPH.apply_to_layer(d["id"], d.get("name"), d.get("layer"))
        return jsonify(ok=True, layer=l.meta())
    except Exception as e:
        return jsonify(error=str(e)), 400


CLIPBOARD = {"clip": None}


@app.post("/api/clipboard")
def clipboard():
    """Copy / cut / paste for the active layer. {action: copy|cut|paste,
    layer, selection?, x?, y?}. copy/cut confine to the selection when given
    (Photoshop semantics) and report the clip size; paste creates a NEW layer
    at the copied position (or x,y). One clipboard per workspace -- it
    survives switching documents."""
    d = request.json or {}
    act = d.get("action")
    try:
        if act in ("copy", "cut"):
            fn = DOC.copy_region if act == "copy" else DOC.cut_region
            clip = fn(d["layer"], selection=d.get("selection") or None)
            if clip is None:
                return jsonify(error="nothing to copy: the region is empty"), 400
            CLIPBOARD["clip"] = clip
            if act == "cut":
                GRAPH.commit_layer_outputs()
            h2, w2 = clip["pixels"].shape[:2]
            return jsonify(ok=True, width=w2, height=h2, x=clip["x"], y=clip["y"])
        if act == "paste":
            clip = CLIPBOARD["clip"]
            if clip is None:
                return jsonify(error="clipboard is empty: copy or cut first"), 400
            l = DOC.paste(clip, x=d.get("x"), y=d.get("y"),
                          name=d.get("name") or "Pasted")
            GRAPH.commit_layer_outputs()
            return jsonify(ok=True, layer=l.meta())
        return jsonify(error=f"unknown clipboard action '{act}'"), 400
    except KeyError as e:
        return jsonify(error=f"missing or unknown id: {e}"), 400


# NOTE: /api/export/relief.glb and /api/export/splats.ply were removed --
# depth-mesh and Gaussian-splat EXPORT are 3D-modelling-tool features that
# produced files this image editor can't display, on slow (quadratic-in-pixels)
# operations that stalled the UI. The in-graph Splatify node stays: it renders
# splats back to a normal image the canvas shows. (leCore still exposes the
# underlying faculties for the separate 3D app.)


_MIND_ALLOW = {
    # discovery + docs: the agent door (read-only faculties only)
    "find_capability", "find_scored", "suggest", "describe_skill",
    "complete_method", "capabilities", "features", "version",
    "seam_continuity", "compare_images", "est_dx", "vanishing_point",
    "image_colours", "image_signature",
}


@app.post("/api/mind")
def mind_invoke():
    """Allowlisted pass-through to leCore's own capability door:
    {"name": faculty, "args": {...}} -> its JSON-safe result. Discovery and
    read-only analysis faculties only -- nothing that mutates the mind."""
    d = request.json or {}
    name = str(d.get("name", ""))
    if name not in _MIND_ALLOW:
        import inspect as _ins
        allowed = {n: str(_ins.signature(getattr(mind(), n)))
                   for n in sorted(_MIND_ALLOW)}
        return jsonify(error=f"'{name}' is not on the /api/mind allowlist",
                       allowed=allowed), 400
    try:
        args = d.get("args") or {}
        for k2, v2 in list(args.items()):
            if isinstance(v2, list) and v2 and isinstance(v2[0], list):
                args[k2] = np.asarray(v2, np.float32)
        r = mind().invoke(name, args)
        def safe(x):
            if isinstance(x, np.ndarray):
                return x.tolist() if x.size <= 4096 else                     {"shape": list(x.shape), "note": "array too large; truncated",
                     "sample": x.ravel()[:16].tolist()}
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            if isinstance(x, dict):
                return {k3: safe(v3) for k3, v3 in x.items()}
            if isinstance(x, (list, tuple)):
                return [safe(v3) for v3 in x]
            return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)
        return jsonify(ok=True, result=safe(r))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/walls")
def walls_get():
    """The four perpendicular planes: {walls: {front, back, left, right},
    editing: side|None}. Empty by default -- a document is a flat canvas
    until an artist puts a layer on a wall."""
    return jsonify(wall_scale=dict(getattr(DOC, "wall_scale", {}) or {}),
                   stack_height=DOC.stack_height(),
                   walls=dict(getattr(DOC, "walls", {})),
                   editing=getattr(DOC, "wall_edit", None))


@app.post("/api/wall")
def wall_post():
    """Manage the planes: {"action": "assign"|"clear"|"edit", "side":
    front|back|left|right, "layer"?}. assign puts an ordinary layer on
    that side (it keeps its strokes, thickness, volume, fields and
    lights, and stands on exactly one wall); clear gives it back to the
    canvas; edit opens a side for painting, where it lies flat and every
    ordinary tool works on it unchanged -- pass side null to close.
    scale sets that side's VERTICAL scale (0.05-20): a wall's own
    vertical axis is height above the canvas, and how far that height
    reaches across the floor depends on how deep the layer stack is.
    The stack's depth normalises it automatically; this is the artist's
    multiplier on top -- raise it for a cathedral window, lower it for
    a slide under glass."""
    d = request.json or {}
    act = d.get("action")
    try:
        if act == "assign":
            w = DOC.assign_wall(d.get("side"), d.get("layer"))
            return jsonify(ok=True, walls=w)
        if act == "clear":
            return jsonify(ok=True, walls=DOC.clear_wall(d.get("side")))
        if act == "scale":
            sc = DOC.set_wall_scale(d.get("side"), d.get("scale", 1.0))
            return jsonify(ok=True, wall_scale=sc,
                           stack_height=DOC.stack_height())
        if act == "edit":
            s = DOC.edit_wall(d.get("side"))
            return jsonify(ok=True, editing=s,
                           walls=dict(getattr(DOC, "walls", {})))
    except KeyError:
        return jsonify(error="no such layer"), 404
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(error="unknown action"), 400


@app.post("/api/shape/recognise")
def shape_recognise():
    """What shape was that stroke? {"points": [[x, y], ...]} ->
    {shape: "circle"|"rectangle"|"line"|null, confident, why}. Two
    independent opinions (leCore's HRNN trajectory readout and a
    circularity test) must AGREE before a shape is claimed; when they
    disagree the answer is null with the reason, because an unasked-for
    shape replacement that guesses wrong is worse than no feature."""
    from . import recognise_shape
    d = request.json or {}
    pts = d.get("points") or []
    if not isinstance(pts, list) or len(pts) < 2:
        return jsonify(error="points must be a list of [x, y]"), 400
    try:
        return jsonify(ok=True, **recognise_shape(pts))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/media/cook")
def media_cook():
    """Let simulations COOK without moving the playhead: {"steps": N,
    "layer": id?, "until": "settled"?}. With until="settled" the step
    count is decided by the SIMULATION rather than guessed: leCore's
    HRNN regime detection watches the medium's own change-per-step
    signal and stops when it enters a final quiet regime, reporting
    {steps, settled, why} per layer so a cap-stop is distinguishable
    from a real settle. The cooked state becomes the layer's starting state at
    the current frame, so a puff of smoke can already be drifting when
    the timeline starts instead of being a hard-edged blob at frame 0.
    Returns {ok, cooked} -- the number of layers advanced."""
    from . import cook_media
    d = request.json or {}
    try:
        steps = int(d.get("steps", 24))
    except (TypeError, ValueError):
        return jsonify(error="steps must be an integer"), 400
    # negatives are meaningful now: they UN-cook
    if str(d.get("until", "")) == "settled":
        from . import cook_until_settled
        r = cook_until_settled(DOC, layer=d.get("layer") or None)
        if r["cooked"] == 0:
            return jsonify(ok=True, cooked=0,
                           warning="nothing to cook -- paint into a "
                                   "media layer (ink, smoke, fire) "
                                   "first"), 200
        return jsonify(ok=True, **r)
    if steps == 0:
        return jsonify(error="steps must not be zero"), 400
    if abs(steps) > 2400:
        return jsonify(error="that is %d steps; the limit is 2400"
                             % abs(steps)), 400
    r = cook_media(DOC, steps=steps, layer=d.get("layer") or None)
    if r["cooked"] == 0:
        return jsonify(ok=True, cooked=0,
                       warning="nothing to cook -- paint into a media "
                               "layer (ink, smoke, fire) first"), 200
    return jsonify(ok=True, **r)


@app.get("/api/export/frames.zip")
def export_frames():
    """Export an ANIMATION as a zip of numbered PNGs: ?from=&to=&step=
    (frames, defaults to the document's play range), ?w=&h= to render at
    another size, ?fps= recorded in a small README for whoever assembles
    the sequence. Simulated media are stepped by the same timeline the
    canvas uses, so the exported frames are exactly what playback showed.
    The artist's playhead is restored afterwards."""
    import io as _io
    import zipfile as _zip
    from . import composite_lit, _doc_emission, _resize as _rz
    try:
        lo = float(request.args.get("from", DOC.frame_range[0]))
        hi = float(request.args.get("to", DOC.frame_range[1]))
        step = float(request.args.get("step", 1.0))
    except ValueError:
        return jsonify(error="from, to and step must be numbers"), 400
    if step <= 0:
        return jsonify(error="step must be positive"), 400
    n = int(np.floor((hi - lo) / step)) + 1
    if n < 1:
        return jsonify(error="that range contains no frames"), 400
    if n > 600:
        return jsonify(error="that is %d frames; the limit is 600 -- "
                             "raise the step or narrow the range" % n), 400
    try:
        w = int(request.args.get("w", DOC.width))
        h = int(request.args.get("h", DOC.height))
    except ValueError:
        return jsonify(error="w and h must be integers"), 400
    fps = request.args.get("fps") or getattr(DOC, "fps", 24)
    keep = float(DOC.frame)
    van = getattr(DOC, "vantage", "above")
    vmode = getattr(DOC, "view3d", "flat")
    buf = _io.BytesIO()
    try:
        with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as z:
            for i in range(n):
                t = lo + i * step
                DOC.set_frame(t)
                lit = (any(li.get("enabled")
                           for li in getattr(DOC, "lights", []))
                       or _doc_emission(DOC) is not None
                       or van == "below")
                if vmode in ("ortho", "persp") or lit:
                    c = composite_lit(DOC, vmode if vmode in
                                      ("ortho", "persp") else "flat",
                                      vantage=van)
                else:
                    c = DOC.composite()
                if (h, w) != (DOC.height, DOC.width):
                    c = _rz(c, h, w)
                flat = c[..., :3] * c[..., 3:4] + 1.0 * (1 - c[..., 3:4])
                # encode straight to bytes: _png returns a streaming
                # send_file response and reading it back raises
                # "direct passthrough mode"
                from PIL import Image as _PI
                arr = (np.clip(flat, 0, 1) * 255).astype(np.uint8)
                fb = _io.BytesIO()
                _PI.fromarray(arr).save(fb, "PNG")
                z.writestr("frame_%04d.png" % i, fb.getvalue())
            z.writestr("README.txt",
                       "leStudio frame sequence\n"
                       "frames %g..%g step %g (%d files)\n"
                       "fps %s\n"
                       "assemble e.g.: ffmpeg -framerate %s "
                       "-i frame_%%04d.png out.mp4\n"
                       % (lo, hi, step, n, fps, fps))
    finally:
        DOC.set_frame(keep)          # never move the artist's playhead
    buf.seek(0)
    return app.response_class(
        buf.getvalue(), mimetype="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="frames.zip"'})


@app.get("/api/export.png")
def export():
    """Export the composite as PNG. ?w=&h= renders AT that size (procedural nodes synthesize detail); ?layer= exports one layer."""
    c = DOC.composite()
    flat = c[..., :3] * c[..., 3:4] + 1.0 * (1 - c[..., 3:4])
    return _png(np.clip(flat, 0, 1))


@app.get("/api/health")
def health():
    """Liveness and readiness, for a load balancer or container probe.

    Deliberately cheap and side-effect free: it must not composite, touch
    leCore, or take `_DOC_LOCK`, or a health check would queue behind a slow
    stroke and the orchestrator would kill a working process.
    """
    return jsonify(ok=True, service="lestudio",
                   docs=len(getattr(WS, "docs", {}) or {}),
                   live=LIVE["on"])


@app.get("/api/ready")
def ready():
    """Readiness: can this process actually serve a request? Unlike health
    this touches the workspace, so it fails while the app is still starting
    or has been wedged."""
    try:
        with _DOC_LOCK:
            w, h = int(DOC.width), int(DOC.height)
        return jsonify(ok=True, canvas=[w, h])
    except Exception as e:
        return jsonify(ok=False, error=type(e).__name__), 503


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def apply_runtime_limits():
    """Fit the process to the machine it is on.

    On a small box (a phone-class VM, a shared container) numpy's BLAS will
    happily spawn a thread per core and thrash. LESTUDIO_THREADS caps that.
    Set BEFORE numpy is imported to take effect, which is why this is called
    from the entry point rather than lazily.
    """
    n = _env_int("LESTUDIO_THREADS", 0)
    if n > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, str(n))
    return n


def serve(host=None, port=None, debug=False):
    """Run the app.

    Configurable from the environment so the same image can be run locally or
    in a container without editing code: LESTUDIO_HOST, LESTUDIO_PORT,
    LESTUDIO_THREADS.

    NOTE FOR HOSTING: this uses Flask's development server, which is fine for
    one painter on one machine and is NOT a production server. Behind a real
    one, mind two things about this app: the workspace is a SINGLE SHARED
    STUDIO (see `/api/health` docs) rather than one canvas per visitor, and
    the engine keeps its state in memory in this process -- so it must run as
    ONE worker. Multiple workers would each hold a different painting and
    serve whichever one the load balancer happened to pick.
    """
    apply_runtime_limits()
    host = host or os.environ.get("LESTUDIO_HOST", "127.0.0.1")
    port = int(port or _env_int("LESTUDIO_PORT", 5050))
    print(f"leStudio -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


@app.post("/api/brush_load")
def brush_load():
    """Dip the brush in the palette: {"color"?: [r,g,b], "amount"?: 0..1}.
    Real brush mode lets paint run out; this is how you fill it back up
    without going to find a thick passage to scrape."""
    d = request.get_json(force=True, silent=True) or {}
    try:
        # a NaN charge and a string colour both got through: one crashed,
        # the other quietly put NaN on the brush
        amt = _finite(d.get("amount", 1.0), "amount", 0.0, 1.0, 1.0)
        col = d.get("color")
        if col is not None:
            if isinstance(col, (str, bytes)) or not isinstance(
                    col, (list, tuple)) or len(col) < 3:
                raise _Gone("colour must be [r, g, b] numbers from 0 to 1")
            col = [_finite(v, "colour", 0.0, 1.0) for v in list(col)[:3]]
    except _Gone as e:
        return jsonify(error=str(e)), 400
    st = DOC.load_brush(color=col, amount=amt)
    return jsonify(ok=True, brush_state=st)


@app.post("/api/stroke_group")
def stroke_group():
    """Bundle a blended passage into one editable object:
    {"action": "create"|"dissolve", "strokes"?: [sid|gid, ...], "name"?,
     "group"?}. A group id is accepted anywhere a stroke id is, so the whole
    passage transforms as a unit while every member stays individually
    editable."""
    d = request.get_json(force=True, silent=True) or {}
    act = str(d.get("action", "create"))
    try:
        if act == "create":
            gid = DOC.group_strokes(d.get("strokes") or [], name=d.get("name"))
            if gid is None:
                return jsonify(error="no strokes to group"), 400
            return jsonify(ok=True, group=gid)
        if act == "dissolve":
            return jsonify(ok=True, strokes=DOC.ungroup_strokes(d["group"]))
    except KeyError as e:
        return jsonify(error="no such group: %s" % e), 404
    return jsonify(error="unknown action %r" % act), 400


@app.post("/api/palette")
def palette_squeeze():
    """Squeeze mounds of thick paint onto a layer:
    {"colors": [[r,g,b], ...], "layer"?, "x"?, "y"?, "size"?, "media"?}.
    Without "layer" the paint goes onto a dedicated Palette layer, so the
    mounds sit beside the picture instead of in it.

    A palette is not a picker -- it is real paint. The mounds are ordinary
    strokes at a heavy load, thick enough to count as a pile the brush can
    reload from, so dipping, carrying two colours at once and scraping a
    mound thinner all come from the physics that is already there."""
    d = request.get_json(force=True, silent=True) or {}
    cols = d.get("colors") or []
    if not cols:
        return jsonify(error="no colours to squeeze out"), 400
    try:
        if isinstance(cols, (str, bytes)) or not isinstance(cols, (list, tuple)):
            raise _Gone("colours must be [r, g, b] numbers from 0 to 1")
        rgb = []
        for one in cols:
            if isinstance(one, (str, bytes)) or not isinstance(
                    one, (list, tuple)) or len(one) < 3:
                raise _Gone("colours must be [r, g, b] numbers from 0 to 1")
            rgb.append(tuple(_finite(v, "colour", 0.0, 1.0) for v in one[:3]))
        for k in ("x", "y", "size"):
            if d.get(k) is not None:
                d[k] = _finite(d[k], k, 1.0, 20000.0)
    except _Gone as e:
        return jsonify(error=str(e)), 400
    try:
        # the palette is its own SURFACE now -- not a layer of the picture,
        # so it can never be nudged, exported, or spill into strata
        pd = DOC.palette_doc()
        # sized for the palette surface, not derived from a canvas's
        # dimensions -- the default made pea-sized mounds you could not dip in
        n = max(len(rgb), 1)
        size = float(d.get("size") or min(34.0, (pd.width - 60.0) / (n * 2.7)))
        spots = pd.lay_palette(
            d.get("layer"), rgb,
            x=d.get("x", size * 1.5), y=d.get("y", pd.height * 0.5),
            size=size, media=str(d.get("media", "oil")))
    except (KeyError, ValueError, IndexError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True,
                   spots=[[float(a), float(b)] for a, b in spots])


@app.post("/api/paper")
def set_paper():
    """Choose the stock: {"paper": "canvas"|"rough"|"cold_press"|
    "hot_press"|"smooth"|"linen"}. The substrate decides where thin paint
    catches (stiff paint on the risen threads), where a wash pools (in the
    dips), how hard dry-brush breaks up and how strongly watercolour
    granulates."""
    d = request.get_json(force=True, silent=True) or {}
    try:
        name = DOC.set_paper(str(d.get("paper", "canvas")))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, paper=name)


@app.post("/api/stratum")
def set_stratum():
    """Spill a full layer onto a new one: {"on": true|false}.

    A layer holds a finite amount of paint; past that the height field used
    to be clipped, so a worked passage saturated after about two loaded
    passes and flattened to a plateau. With this on, the excess starts a
    fresh stratum from zero and the build-up keeps going."""
    d = request.get_json(force=True, silent=True) or {}
    DOC.auto_stratum = bool(d.get("on", True))
    return jsonify(ok=True, auto_stratum=DOC.auto_stratum)


@app.get("/api/palette.png")
def palette_png():
    """The palette, cropped to the paint, for the dock beside the canvas.

    The palette is excluded from `canvas_layers()`, so it appears in neither
    the canvas view nor an export -- it is a surface you work beside the
    picture, and this is where it is drawn. The header carries the region it
    occupies on the document so the app can map a dip back to real
    coordinates."""
    with _DOC_LOCK:
        pd = DOC.palette_doc(create=False)
        got = pd.palette_png() if pd is not None else None
    if got is None:
        return jsonify(error="no palette yet - squeeze some paint out first"), 404
    img, box = got
    r = Response(png_bytes(img), mimetype="image/png")
    r.headers["X-Palette-Box"] = ",".join(str(int(v)) for v in box)
    r.headers["Cache-Control"] = "no-store"
    return r


@app.post("/api/palette/paint")
def palette_paint():
    """Paint ON the palette: {"points": [[x,y],...], "color", "radius",
    "media"?, "load"?, "mix"?, "real_brush"?, "mode"?: "blend"|"knife"}.

    This is how you dip and how you mix. It is ORDINARY painting on the
    palette surface -- same physics, and the same brush reservoir as the
    picture, so a dip here loads the brush you go on to paint with. `mode`
    lets the blender and knife work there too, which is what mixing on a
    palette actually is."""
    d = request.get_json(force=True, silent=True) or {}
    try:
        d = _clean_paint(d)          # the same gate the picture gets
    except _Gone as e:
        return jsonify(error=str(e)), 400
    pts = d["points"]
    if len(pts) < 1:
        return jsonify(error="a dip needs somewhere to go"), 400
    with _DOC_LOCK:
        pd = DOC.palette_doc()
        lid = (pd.palette_layer(create=False) or pd.layers[-1]).id
        mode = str(d.get("mode", "brush"))
        try:
            if mode == "blend":
                pd.blend_stroke(lid, pts, radius=float(d.get("radius", 22)),
                                strength=float(d.get("opacity", 0.6)))
            elif mode == "knife":
                pd.knife(lid, pts, mode=str(d.get("knife", "smooth")),
                         radius=float(d.get("radius", 26)),
                         strength=float(d.get("opacity", 0.7)))
            else:
                pd.paint(lid, pts,
                         color=tuple(d.get("color", (0.5, 0.5, 0.5))),
                         radius=float(d.get("radius", 14)),
                         opacity=float(d.get("opacity", 1.0)),
                         media=(d.get("media") or None),
                         load=float(d.get("load", 1.2)),
                         mix=float(d.get("mix", 1.0)),
                         real_brush=bool(d.get("real_brush", True)))
        except (KeyError, ValueError) as e:
            return jsonify(error=str(e)), 400
    DOC._edited_palette_last = True
    return jsonify(ok=True, brush_state=DOC.brush_state())


@app.post("/api/palette/clear")
def palette_clear():
    """Scrape the palette back to bare board -- UNDOABLY.

    Dropping the surface outright destroyed a session of mixing on one click
    with nothing to press afterwards. Clearing the paint through the palette's
    own history keeps it on the undo stack like any other edit."""
    with _DOC_LOCK:
        pd = DOC.palette_doc(create=False)
        if pd is None:
            return jsonify(ok=True)
        pd.record("Scrape palette")
        for l in pd.layers:
            l.pixels[...] = 0.0
            if getattr(l, "height_map", None) is not None:
                l.height_map[...] = 0.0
        pd.strokes = [k for k in pd.strokes if False]
        DOC._edited_palette_last = True      # so Ctrl+Z reaches it
    return jsonify(ok=True)


# ------------------------------------------------------------------------------------------------
# Entry point. THIS MUST STAY AT THE END OF THE FILE: every @app.route below
# the __main__ guard is never registered when the module is run directly
# (`python server.py`), because main() fires before those lines execute. Eight
# routes -- the whole palette and paint-setup surface -- were unreachable that
# way. `python -m lestudio` was unaffected, which is exactly why it went
# unnoticed: the documented path imports the module fully first.
# ------------------------------------------------------------------------------------------------
def main():  # console entry point
    serve()


if __name__ == "__main__":
    main()
