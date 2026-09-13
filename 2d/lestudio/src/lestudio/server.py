"""lestudio.server -- the browser UI + JSON API for leStudio.

    python -m lestudio            # or: lestudio  (console script)
    -> http://127.0.0.1:5050

Requires the [ui] extra (Flask + Pillow):  pip install "lestudio"
"""
from __future__ import annotations

import collections
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
               BLEND_MODES, _MATERIALS, _PAPERS, _doc_has_work)

try:
    from flask import Flask, Response, g, jsonify, request, send_file
except ImportError as e:  # pragma: no cover
    raise SystemExit('leStudio needs Flask + Pillow: pip install "lestudio"') from e

app = Flask(__name__)


class _WS:
    """The workspace: several documents, one active. `DOC`/`GRAPH` below keep the
    whole existing endpoint surface working against the active document."""

    def _mint_did(self):
        """Workspace-scoped document ids. R57: minted by the ENGINE's
        persisted per-prefix counter on the shared .lews directory
        (`Workspace.mint("D")` -- the lews_mint law itself, not our imitation
        of it), so no app on this workspace can ever issue the same id, even
        from another process. The scan fallback keeps an engine without the
        Workspace class booting; the collision guard covers ids that predate
        the counter (the boot doc's explicit "D1", restored files)."""
        try:
            ws = _lews_ws()
            if ws is not None:
                did = ws.mint("D")
                while did in getattr(self, "docs", {}):
                    did = ws.mint("D")
                return did
        except Exception:
            pass
        import re as _re
        n = 0
        for did in getattr(self, "docs", {}):
            m = _re.match(r"^D(\d+)$", str(did))
            if m:
                n = max(n, int(m.group(1)))
        return "D%d" % (n + 1)

    def __init__(self):
        d = Document(768, 512, id="D1")
        self.docs = {d.id: d}
        self.graphs = {d.id: NodeGraph(d)}
        self.active = d.id
        self.extras = []           # foreign workspace sections, carried verbatim
        self._wire()

    def _wire(self):
        for did, g in self.graphs.items():
            g.resolver = self.docs.get
            if "MEDIA" in globals():
                # media sources are keyed (doc_id, node_id): with a plain
                # node id, doc A's N1 and doc B's N1 fought over ONE video
                # slot (both graphs seed ids from the same counter, so the
                # collision is the common case, not the corner)
                g.media = _media_hook_for(did)

    @property
    def doc(self):
        return self.docs[self.active]

    @property
    def graph(self):
        return self.graphs[self.active]

    def add(self, w, h, name=None, background=(1.0, 1.0, 1.0)):
        d = Document(w, h, name, background=background,
                     id=self._mint_did())
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
        # A closed doc must not linger in presence or media bookkeeping:
        # stale viewing entries made peers show "on another document" for a
        # doc that no longer exists (and, worse, per-client doc resolution
        # would have had to re-check liveness on every request).
        if "SYNC" in globals():
            vw = SYNC.get("viewing", {})
            for uid in [u for u, v in vw.items() if v not in self.docs]:
                vw.pop(uid, None)
        if "MEDIA" in globals():
            MEDIA.purge_doc(did)
        return True


WS = _WS()


def _media_hook_for(did):
    """A media hook bound to one document id (see _WS._wire)."""
    def hook(nid, params, want):
        return MEDIA.hook((did, nid), params, want)
    return hook


def _viewing_doc_id():
    """The document id THIS request operates on.

    The active document used to be one global: any client switching docs
    redirected everyone else's edits mid-stroke. Each user's choice now lives
    in SYNC["viewing"] (keyed by user id); WS.active remains the DEFAULT for
    clients that never activated anything themselves -- which is exactly the
    old behaviour for a single user, so nothing single-user changes."""
    try:
        from flask import has_request_context
        if not has_request_context():
            return WS.active                  # background threads, tests
        did = SYNC.get("viewing", {}).get(_req_uid())
        if did and did in WS.docs:
            return did
    except Exception:
        pass
    return WS.active


class _Active:
    """A live proxy to the requesting user's ACTIVE document or graph
    (falling back to the workspace-global active outside request context)."""

    def __init__(self, attr):
        object.__setattr__(self, "_attr", attr)

    def _target(self):
        attr = self._attr
        if attr == "doc":
            return WS.docs[_viewing_doc_id()]
        if attr == "graph":
            return WS.graphs[_viewing_doc_id()]
        return getattr(WS, attr)

    def __getattr__(self, name):
        return getattr(self._target(), name)

    def __setattr__(self, name, value):
        # Without this, `DOC.dpi = 300` quietly created an attribute on the
        # PROXY that shadowed the real document -- reads then came back from
        # the proxy and the document never changed. Writes must land on the
        # object being proxied.
        if name == "_attr":
            object.__setattr__(self, name, value)
        else:
            setattr(self._target(), name, value)


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
        "joined": {}, "kicked": set(), "lock": threading.Lock(),
        # per-USER id (never tab id -- mixing the two gave people their own
        # ghost chip): viewing = which doc each user has active, activity =
        # their latest {tool, layer} ping for the presence chips
        "viewing": {}, "activity": {},
        # R63: "have we heard from this uid at all, recently" -- for the
        # layer-ownership stale-lock check. `clients`/`tabuser` answer "is a
        # tab open RIGHT NOW" but get POPPED the instant a tab closes
        # (see /api/events' `finally`), losing exactly the timestamp a
        # staleness check needs; `joined` never updates after the first
        # sighting. last_seen is stamped by the ownership gate on every
        # identified request AND by the /api/events heartbeat, so a uid
        # working purely through the JSON API (a script, an agent, most of
        # this test suite) counts as present without ever opening a stream.
        "last_seen": {}}
INVITES = {"pending": [], "joined": []}    # session-scoped guest bookkeeping

# R56: presence is MIRRORED into the shared .lews workspace directory via
# the engine's live-session contract (lews_touch keyed by the PERSON --
# the ghost-editor law is the engine's now), so any other app on this
# machine (Poly Studio, an agent shell) sees leStudio's participants
# through lews_presence / the engine's /api/presence door. The in-app
# SYNC table stays the POLICY layer (host, kick, invites, per-doc
# viewing) -- APP_FOUNDATION §8 keeps hosting policy app-side.
_LEWS_TOUCH = {"last": {}}

# R57: the shared live workspace DIRECTORY (engine Workspace: locked, atomic,
# journalled container + persisted id mints). One root for everything that
# touches it -- agent_surface mounts on it, presence mirrors into it, and the
# document sections now autosave into it -- so any other app or agent holding
# the same directory open sees leStudio's documents, ids and presence as one
# coherent workspace instead of three half-overlapping ones.
_WS_ROOT = os.environ.get("LESTUDIO_WS") \
    or os.path.expanduser("~/.lestudio_agent_ws")
_LEWS = {"ws": None, "sha": {}}


def _lews_ws():
    """The engine's live .lews Workspace on `_WS_ROOT`, or None when the
    engine on PYTHONPATH predates it. Cached: the object is cheap but the
    guard import is not free per request."""
    if _LEWS["ws"] is None:
        try:
            from holographic.io_and_interop.holographic_lews import Workspace
            os.makedirs(_WS_ROOT, exist_ok=True)
            _LEWS["ws"] = Workspace(_WS_ROOT, app="lestudio")
        except Exception:
            return None
    return _LEWS["ws"]


def _lews_mirror_touch(uid, activity=None):
    try:
        now = time.time()
        if now - _LEWS_TOUCH["last"].get(uid, 0) < 2.0:
            return
        _LEWS_TOUCH["last"][uid] = now
        from . import mind as _mind_fn
        m = _mind_fn()
        if not hasattr(m, "lews_touch"):
            return
        root = _WS_ROOT
        os.makedirs(root, exist_ok=True)
        m.lews_touch(root, str(uid), activity=dict(activity or {}),
                     name=SYNC["names"].get(uid) or str(uid),
                     app="lestudio")
    except Exception:
        pass                                    # presence mirror is best-effort


@app.after_request
def _bump_rev(resp):
    if request.method in ("POST", "PATCH") and request.path.startswith("/api/") \
            and request.path not in ("/api/graph/run", "/api/live",
                                      "/api/autosave",
                                      # an activity ping changes nothing in
                                      # the workspace; bumping would make
                                      # every client refresh on every ping
                                      # (same reasoning as /api/autosave)
                                      "/api/presence/activity") \
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
        _lews_mirror_touch(uid, SYNC["activity"].get(uid))
        last = -1
        beat = 0.0
        # R57: cross-APP awareness. The shared .lews workspace journals every
        # put/note from every app on the directory; this stream forwards the
        # ones that are NOT ours (exclude="lestudio" -- our own writes already
        # reached this client as SYNC revs), so the UI can say "Poly Studio
        # changed the workspace" without polling anything itself.
        lws = _lews_ws()
        try:
            lews_last = lws.rev() if lws is not None else 0
        except Exception:
            lws, lews_last = None, 0

        def lews_foreign():
            nonlocal lews_last
            if lws is None:
                return None
            try:
                # changes_since (not since): the cursor must advance over our
                # OWN excluded entries too, or every poll re-reads them
                ent = lws.changes_since(lews_last)
                if ent:
                    lews_last = max(e.get("rev", lews_last) for e in ent)
                ent = [e for e in ent if e.get("app") != "lestudio"]
                if not ent:
                    return None
                return [{"rev": e.get("rev"), "op": e.get("op"),
                         "id": e.get("id"), "kind": e.get("kind"),
                         "app": e.get("app")} for e in ent[-20:]]
            except Exception:
                return None
        try:
            while True:
                if uid in SYNC["kicked"]:
                    # a terminal event, then end the stream (EVERY tab of the
                    # kicked user gets this); the finally reaps
                    yield ("data: " + json.dumps({"kicked": True})
                           + chr(10) + chr(10))
                    return
                SYNC["clients"][cid] = time.time()
                # R63: a tab sitting open with no other traffic is still
                # PRESENT -- without this, an owner who parked their cursor
                # and stopped clicking (but never closed the tab) would look
                # "absent" to the 120s stale-lock check the moment 120s of
                # silence passed, and lose their layer mid-thought.
                SYNC["last_seen"][uid] = time.time()
                now = time.time()
                live_uids = {SYNC["tabuser"].get(c, c)
                             for c, t in SYNC["clients"].items() if now - t < 10}
                editors = len(live_uids)        # USERS, not tabs
                if SYNC["rev"] != last:
                    last = SYNC["rev"]
                    active = [SYNC["names"].get(u2, "") for u2 in live_uids]
                    yield ("data: " + json.dumps(
                        {"rev": last, "src": SYNC["src"], "editors": editors,
                         "names": sorted(n for n in active if n),
                         # what each live user is doing (tool/layer pings
                         # via /api/presence/activity), for presence chips
                         "activity": {u2: SYNC["activity"][u2]
                                      for u2 in live_uids
                                      if u2 in SYNC["activity"]}})
                        + chr(10) + chr(10))
                elif now - beat >= 2.0:
                    foreign = lews_foreign()
                    if foreign:
                        # another APP wrote the shared workspace: forward the
                        # journal entries themselves (rev unchanged -- this is
                        # not a leStudio edit, so no self-refresh storm)
                        beat = now
                        yield ("data: " + json.dumps(
                            {"rev": last, "src": SYNC["src"],
                             "workspace": foreign})
                            + chr(10) + chr(10))
                        continue
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
    # R64: this list stays TAB-based on purpose. It is what elects the host
    # and authorises a kick, and both are properties of a browser session --
    # a script that painted once must not become host and start removing
    # people. Scripts and agents ARE here and DO show up: they arrive
    # through /api/state's `peers` (R63), and the presence popup renders
    # both lists. Two questions, two answers, rather than one list forced
    # to mean both.
    live = sorted((SYNC["joined"].get(u2, now), u2) for u2 in tabs)
    host = live[0][1] if live else ""
    # R64: one naming rule everywhere. This list said "" for anyone who had
    # not registered a display name while /api/state's peers said what
    # _display_name says, so the same agent appeared named in one roster
    # and anonymous in the other.
    return [{"id": u2, "name": SYNC["names"].get(u2) or _display_name(u2),
             "tabs": tabs[u2],
             "activity": SYNC["activity"].get(u2),
             "joined": round(now - j, 1), "you": u2 == me, "host": u2 == host}
            for j, u2 in live]


def _req_uid():
    """The requesting USER: the persistent id if the client sends one, else
    the tab id (old clients and agents degrade to per-tab identity)."""
    return (request.headers.get("X-User")
            or request.headers.get("X-Client")
            or request.args.get("user") or request.args.get("client") or "")


def _owner_uid():
    """The identity LAYER OWNERSHIP keys off -- X-User specifically (or its
    query-param twin), never the X-Client/tab-id fallback _req_uid() uses
    for presence and kick. A tab id is reborn every reload and every new
    tab; keying a LOCK to one would mean reloading your own browser tab
    loses your own layer, and would let a script that never identifies
    itself as a person rack up a permanent claim just by painting once.
    CONTRACT.md section 1's "U == '' (no identity sent)" means exactly
    this: no X-User -- which is also, not coincidentally, why this test
    suite's many X-Client-only / header-free paint calls keep working:
    they were never claiming anything to begin with."""
    return request.headers.get("X-User") or request.args.get("user") or ""


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
    from . import drip_paint, run_paint
    d = request.json or {}
    try:
        if d.get("mode", "drips") == "drips":
            # R6 default: droplets that walk the gravity direction and
            # leave trails -- what "make the paint run" means to a person.
            # The old whole-sheet advection stays as mode="sheet".
            made = drip_paint(DOC, d.get("layer", ""),
                              direction_deg=float(d.get("direction", 90.0)),
                              strength=float(d.get("strength", 1.0)),
                              drops=int(d.get("drops", 24)),
                              seed=int(d.get("seed", 0)))
            GRAPH.commit_layer_outputs()
            return jsonify(ok=True, drips=made)
        run_paint(DOC, d.get("layer", ""), steps=int(d.get("steps", 12)),
                  gx=float(d.get("gx", 0.0)), gy=float(d.get("gy", 0.0)),
                  gz=float(d.get("gz", 1.0)))
    except KeyError:
        return jsonify(error="no such layer"), 404
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.post("/api/anim/flipbook")
def anim_flipbook():
    """R6 flipbook: turn a layer GROUP into an animation by generating
    hold-interpolated visibility keys -- frames are layers, exactly the
    Procreate model, and playback/export ride the existing timeline.
    Body: {"layers": [ids bottom-to-top = frame order], "fps"?: 12,
    "mode"?: "loop"|"pingpong"|"once", "holds"?: {id: ticks}}.
    Rebuilds the visibility tracks from scratch each call (the strip UI
    calls it after every edit); sets the frame range to one cycle for
    "once", leaves it for loop/pingpong (playback loops the range).
    Returns {frames, range, fps}."""
    d = request.json or {}
    ids = [str(x) for x in (d.get("layers") or [])]
    if not ids:
        return jsonify(error="pass layers: the frame order"), 400
    try:
        for lid in ids:
            DOC.layer(lid)
    except KeyError as e:
        return jsonify(error="no such layer: %s" % e), 400
    holds = {str(k): max(1, int(v)) for k, v in (d.get("holds") or {}).items()}
    mode = d.get("mode", "loop")
    fps = float(d.get("fps", 12.0))
    order = list(ids) if mode != "pingpong" else ids + ids[-2:0:-1]
    # wipe previous visibility tracks for these layers, then lay hold keys
    for lid in ids:
        DOC.tracks.pop("layer:%s:visible" % lid, None)
    t = 0.0
    spans = []
    for lid in order:
        n = holds.get(lid, 1)
        spans.append((lid, t, t + n))
        t += n
    total = t
    for lid in ids:
        keys = []
        vis_spans = [(a, b) for (l2, a, b) in spans if l2 == lid]
        cur = None
        for f in range(int(total)):
            on = any(a <= f < b for a, b in vis_spans)
            if on != cur:
                keys.append([float(f), 1.0 if on else 0.0, 1])
                cur = on
        DOC.tracks["layer:%s:visible" % lid] = keys
    DOC.fps = fps
    DOC.frame_range = [0.0, max(total - 1.0, 1.0)]
    from . import _MUT_REV
    _MUT_REV[0] += 1
    DOC.set_frame(0.0)
    return jsonify(ok=True, frames=len(order), range=DOC.frame_range,
                   fps=fps)


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
            # R6: interp="hold" makes the value STEP at the next key --
            # what a flipbook frame or a visibility switch needs
            ks = DOC.set_key(d["kind"], d["id"], d["prop"],
                             t=d.get("t"), v=d.get("v"),
                             interp=d.get("interp", "linear"))
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
        from . import _Estimate
        try:
            est = estimate_perspective(DOC, d.get("layer", ""))
        except KeyError:
            return jsonify(error="no such layer"), 404
        except _Estimate as ex:
            # R59: an honest "nothing to estimate from", written for a
            # person -- not the raw exception text the catch-all produced
            return jsonify(error=str(ex)), 400
        except Exception as ex:
            app.logger.warning("perspective estimate failed: %r", ex)
            return jsonify(error="could not read perspective from that "
                                 "layer — draw some straight edges and try "
                                 "again, or place the vanishing points by "
                                 "hand"), 400
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
    DOC.media_step(l.id, int(d.get("steps", 12)),
                   selection=d.get("selection"),
                   sel_invert=bool(d.get("sel_invert")))
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True)


@app.get("/api/media/vectors")
def media_vectors():
    """R5 #20: the simulation's velocity field, coarsened for an overlay.
    ?layer=<id> returns the layer's media velocity block-averaged onto a
    ~24x18 grid ({vx, vy, gw, gh} as row-major lists, in canvas px/step),
    so the client can draw sparse arrows instead of guessing what the
    black box is doing. 404 when the layer has no living medium yet --
    the read NEVER creates simulation state."""
    import numpy as np
    from . import _MEDIA_KINDS
    try:
        l = DOC.layer(request.args.get("layer", ""))
    except KeyError:
        return jsonify(error="no such layer"), 404
    st = getattr(l, "_media", None)
    if st is None or getattr(l, "vol_kind", "none") not in _MEDIA_KINDS:
        return jsonify(error="layer has no living medium (vol_kind "
                             "inkwater|smoke|fire, and it must have been "
                             "painted or stepped at least once)"), 404
    vx, vy = np.asarray(st["vx"]), np.asarray(st["vy"])
    sh, sw = vx.shape
    gw, gh = min(24, sw), min(18, sh)

    def coarsen(a):
        # deterministic block mean: trim to a multiple of the target grid,
        # then average each block (the trim is at most one block's width)
        ty, tx = (sh // gh) * gh, (sw // gw) * gw
        b = a[:ty, :tx].reshape(gh, sh // gh, gw, sw // gw)
        return b.mean(axis=(1, 3))

    # velocities live on the sim grid; scale to CANVAS px per step so the
    # arrows mean the same thing at every media_res
    kx, ky = DOC.width / float(sw), DOC.height / float(sh)
    return jsonify(ok=True, gw=gw, gh=gh,
                   vx=[[round(float(v) * kx, 3) for v in row]
                       for row in coarsen(vx)],
                   vy=[[round(float(v) * ky, 3) for v in row]
                       for row in coarsen(vy)])


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
    _touch_presence(me)         # announcing yourself IS being here (R63)
    nm = str((request.json or {}).get("name", "")).strip()[:24]
    if nm:
        SYNC["names"][me] = nm
    else:
        SYNC["names"].pop(me, None)
    return jsonify(ok=True, name=nm)


@app.before_request
def _stamp_author():
    """Remember WHO is about to mutate the caller's active document: record()
    copies this onto each undo entry, so a collaborator's Ctrl+Z can say
    whose change it is about to revert instead of silently reverting it.
    Cheap (one dict write); reads and unidentified callers cost nothing."""
    if request.method in ("POST", "PATCH", "DELETE") \
            and request.path.startswith("/api/"):
        try:
            WS.docs[_viewing_doc_id()]._last_author = _req_uid()
        except Exception:
            pass


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
# R63: live-collaboration layer ownership (CONTRACT.md sections 1-3).
#
# A layer's `owner`/`shared` are a MULTIPLAYER COURTESY, not `locked`
# (Document._locked_guard's freeze, a DIFFERENT and older feature): `locked`
# blocks everyone including its own owner; `owner` exists purely so two
# connected people don't paint over each other on the same layer, and must
# never get in a single, unidentified user's way -- see the permission table
# in _layer_permission below, which is the ONE place it is decided.
# ------------------------------------------------------------------------------------------------
STALE_S = 120.0    # CONTRACT.md: "a document must never be frozen by
                    # someone who shut their laptop"


def _touch_presence(uid):
    """Stamp `uid` as heard-from right now. Called from the ownership gate
    on every identified request (mutating or not) and from the /api/events
    heartbeat -- together they're what makes "absent > 120s" mean anything
    for a uid that never opens a live stream at all (a script, an agent, most
    of this test suite painting with a bare X-User header)."""
    if uid:
        SYNC["last_seen"][uid] = time.time()


def _presence_absent_s(uid):
    """Seconds since `uid` was last heard from. No record at all (this
    PROCESS has never heard from them -- e.g. a fresh boot reloaded a .lews
    whose layer owner was set by a run that no longer exists) reads as
    infinitely absent: a restart must not let a long-gone owner freeze a
    layer just because nobody has said hello to this process yet."""
    ts = SYNC["last_seen"].get(uid)
    return float("inf") if ts is None else max(0.0, time.time() - ts)


def _display_name(uid):
    """Best-effort human name for a uid, matching the convention already
    used for undo's cross-author warning (_foreign_top_entry).

    An agent's uid is "agent:<name>" (CONTRACT.md section 4), so the plain
    uid[:6] fallback rendered EVERY agent as the string "agent:" -- which
    would have made two agents in one document indistinguishable, and read
    as a truncation bug wherever it appeared. The name after the colon is
    the name it chose; use it."""
    if not uid:
        return uid
    nm = SYNC["names"].get(uid)
    if nm:
        return nm
    if uid.startswith("agent:"):
        return uid[6:] or "agent"
    # R64 (dogfooded): uid[:6] chopped "u_devin" to "u_devi" and put THAT
    # in the sentence a human reads -- "ask u_devi for access?" reads as a
    # truncation bug, not as a fallback. An unregistered id is shown whole:
    # it is at least the real handle, and a UI that wants it shorter can
    # shorten it knowing what it started from.
    return uid


def _layer_refusal(lid, layer):
    """The CONTRACT's one refusal shape. Every route that turns down a
    write for OWNERSHIP reasons must return exactly this (403) -- the client
    has one renderer for it ("That layer is Priya's -- ask her for
    access?" + a button), and a second shape would need a second renderer,
    or silently fail to show the ask-for-access button at all."""
    owner = getattr(layer, "owner", "") or ""
    name = _display_name(owner) or owner
    return {"error": "that layer is %s's — ask %s for access?"
                     % (name, name),
            "layer": lid, "owner": owner, "owner_name": name,
            "can_request": True}


def _layer_permission(lid, uid):
    """PERMISSION to mutate layer `lid` as `uid` -- CONTRACT.md section 1,
    checked in the order it is written there (first match decides). `uid`
    must be _owner_uid()'s notion of identity (X-User only), not _req_uid()'s
    -- see _owner_uid's docstring for why the tab-id fallback doesn't count:

        unowned            -> yes, and `uid` CLAIMS it here (auto-lock on
                               first write; a no-op when `uid` is itself ""
                               -- an anonymous write to an unowned layer
                               does not conjure an owner out of nothing).
                               Checking permission and taking the lock are
                               the SAME decision on purpose: if they were
                               two steps, two requests racing on a freshly-
                               unowned layer could both see "yes" and both
                               end up believing they own it.
        uid is the owner    -> yes
        uid in shared       -> yes (the owner granted them in)
        owner is an agent   -> yes (agent:* never locks a human out -- the
                               document belongs to the human, not the agent)
        uid is anonymous    -> refused (no X-User at all; an OWNED layer
                               stays protected even from a client sending no
                               persistent identity, or the lock would be
                               worth nothing -- but note an UNOWNED layer
                               already returned "yes" above, which is
                               exactly how this suite's many X-User-free
                               paint calls keep working)
        owner absent > 120s -> yes, and the lock YIELDS to `uid`
        otherwise           -> refused, contract shape

    Returns (ok, extra). `extra` is {} when ok with no side note, or
    {"yielded_from": <prior owner>} when ok via a stale-lock yield, or (when
    not ok) the refusal dict itself. An unknown layer id is waved through --
    manufacturing a 403 for it here would bury the route's own, more useful
    "no such layer" 400 under an ownership error that names no one."""
    try:
        layer = DOC.layer(lid)
    except Exception:
        return True, {}
    owner = getattr(layer, "owner", "") or ""
    if owner == "":
        layer.owner = uid                              # auto-claim
        return True, {}
    if uid and uid == owner:
        return True, {}
    if uid and uid in (getattr(layer, "shared", None) or []):
        return True, {}
    if owner.startswith("agent:"):
        return True, {}
    if not uid:
        return False, _layer_refusal(lid, layer)
    if _presence_absent_s(owner) > STALE_S:
        prior = owner
        layer.owner = uid                              # the lock yields
        return True, {"yielded_from": prior}
    return False, _layer_refusal(lid, layer)


# Routes the ownership gate must NEVER touch, and why: these either have
# their own separate authority (undo/redo's cross-author warning, host
# moderation), or are the plumbing that presence and the request flow
# themselves run on -- gating THOSE would be circular (you'd need access to
# ask for access) or would break multiplayer awareness outright (you'd need
# a lock to say you're still here).
_GATE_EXCLUDED_PATHS = {
    "/api/undo", "/api/redo",                 # _foreign_top_entry already
                                               # warns about another
                                               # author's work; a SEPARATE,
                                               # older feature from the
                                               # ownership lock
    "/api/events",                            # presence itself -- holding
                                               # this stream open IS how a
                                               # user is "here" at all
    "/api/presence/name", "/api/presence/activity",   # presence pings
    "/api/autosave", "/api/autosave/restore",  # background persistence,
                                               # never a deliberate user edit
    "/api/editors/kick", "/api/editors/allow",  # host moderation is a
                                               # different authority than
                                               # layer ownership
    "/api/access",                            # the request FLOW -- asking
                                               # for access must never itself
                                               # require access
}


def _mutating_layer_targets():
    """The layer ids this request would touch, by the body shapes actually
    in use (grepped across every POST route, not guessed): almost everything
    that targets a layer sends it as a top-level "layer" string, the one
    exception being /api/layer itself (which sends "id", and only for the
    actions that touch an EXISTING layer's content -- "add" makes a new one,
    "release"/"grant"/"revoke" are owner-administration handled by their own
    stricter check inside layer_edit(), not this generic gate) and
    /api/paint_batch (many strokes, each carrying its own "layer")."""
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        return set()
    ids = set()
    lay = d.get("layer")
    if isinstance(lay, str) and lay:
        ids.add(lay)
    if request.path == "/api/paint_batch":
        for it in (d.get("strokes") or ()):
            if isinstance(it, dict) and isinstance(it.get("layer"), str) \
                    and it["layer"]:
                ids.add(it["layer"])
    elif request.path == "/api/layer":
        act = d.get("action")
        # R64 (dogfooded): VISIBILITY is not destruction. Two painters left
        # the construction underdrawing sitting over the finished picture
        # because the only person who could switch it off was scoped
        # elsewhere, and hiding someone else's layer was refused. The lock
        # exists so people cannot DESTROY each other's work; hiding changes
        # no pixels, is instantly reversible by anyone, and the alternative
        # is a document nobody present is able to finish. An edit that
        # touches nothing but `visible` is therefore ungated -- every other
        # property (name, opacity, blend, clip, alpha_lock, mask, locked)
        # changes the picture and stays owner's-only.
        if act == "edit" and set(d) <= {"action", "id", "visible"}:
            return set()
        if act in ("edit", "delete", "remove", "duplicate", "merge_down",
                   "clear", "fill", "flip", "move", "claim"):
            lid = d.get("id")
            if isinstance(lid, str) and lid:
                ids.add(lid)
        elif act == "merge":
            ids.update(x for x in (d.get("ids") or ()) if isinstance(x, str))
        elif act == "merge_visible":
            try:
                ids.update(l.id for l in DOC.layers if l.visible)
            except Exception:
                pass
    return ids


@app.before_request
def _layer_ownership_gate():
    """THE choke point (CONTRACT.md sections 1-2): every mutating request is
    inspected HERE, once, rather than each of the ~90 POST routes carrying
    its own copy of the ownership check -- a guard sprinkled through thirty
    call sites is exactly how the one route someone forgets becomes the
    hole.
    """
    if request.method != "POST" or not request.path.startswith("/api/"):
        return None
    if request.path in _GATE_EXCLUDED_PATHS \
            or request.path.startswith("/api/agent/") \
            or request.path.startswith("/api/job/"):
        return None
    uid = _owner_uid()
    _touch_presence(uid)
    targets = _mutating_layer_targets()
    if not targets:
        return None
    yields = []
    for lid in sorted(targets):
        ok, extra = _layer_permission(lid, uid)
        if not ok:
            return jsonify(extra), 403
        if extra.get("yielded_from"):
            yields.append({"layer": lid, "yielded_from": extra["yielded_from"]})
    if yields:
        # picked up by _layer_yield_notice below -- the route handler itself
        # doesn't need to know a yield happened to report it, which is the
        # point of doing this at the gate instead of in every route
        g._layer_yields = yields
    return None


@app.after_request
def _layer_yield_notice(resp):
    """Folds `yielded_from` into a successful mutation's own JSON body, per
    CONTRACT.md ("the lock YIELDS to U, with yielded_from in the response").
    A separate hook rather than each route reporting it itself, for the same
    reason the permission check is a separate hook: one place, not thirty."""
    y = getattr(g, "_layer_yields", None)
    if not y or resp.status_code >= 400 or not resp.is_json:
        return resp
    try:
        body = resp.get_json()
        if isinstance(body, dict):
            body["yielded_from"] = y[0]["yielded_from"] if len(y) == 1 else \
                {e["layer"]: e["yielded_from"] for e in y}
            resp.set_data(json.dumps(body))
    except Exception:
        pass                                    # never let a notice break a real response
    return resp


# ------------------------------------------------------------------------------------------------
# R63: the reactive agent loop (CONTRACT.md section 2, /api/agent/brief +
# /api/agent/tick). Built ON section 1-3's ownership gate above, not beside
# it -- the agent gets no bypass of its own; _layer_permission's existing
# "owner is an agent -> yes" / "otherwise -> refused" rows are what make
# "an agent never touches a user's layer" true, so this section only has to
# report state, never enforce it.
# ------------------------------------------------------------------------------------------------
PAUSE_MS = 1800.0    # CONTRACT.md: may_act needs this much HUMAN idle time

# Devin's brief for an unset document (CONTRACT.md section 2's
# "default_brief"). This is not a placeholder -- it IS the instruction an
# agent follows when nobody has written one, so its wording is load-bearing:
# support, infer, do the unglamorous work, never invent, never recompose,
# smallest change, and when unsure, do nothing at all.
DEFAULT_AGENT_BRIEF = (
    "Support what the painter is doing. Infer their intent from their most "
    "recent strokes, not from a plan they never stated. Do the unglamorous "
    "supporting work: contact shadows, edge cleanup, continuity between "
    "strokes, keeping the established light consistent. Never invent "
    "subject matter, and never change the composition -- add to what is "
    "already there, don't redirect it. Prefer the smallest change that "
    "helps. If you are unsure whether something would help, do nothing."
)

# What happened, per document, since the last _MUT_REV bump: a bounded log
# of {rev, lid, layer_name, by, by_name, kind, ts}, keyed by document id so
# switching pictures doesn't leak one picture's activity into another's
# tick. Kept in server memory like ACCESS_REQUESTS above, for the same
# reason -- this is conversation about the live session, not part of the
# picture, and must not resurrect itself out of a reopened .lews.
_AGENT_ACTIVITY = {}
_ACTIVITY_MAXLEN = 500

# The last time a NON-agent uid successfully changed a layer, per document.
# Deliberately separate from SYNC["last_seen"] (which is presence -- "is
# this uid's tab/script still around") and from _MUT_REV (which is global
# across every document): human_idle_ms means "since this document was last
# PAINTED ON by a person", and a stale-but-present peer, or a mutation on a
# different picture, must not reset it.
_LAST_HUMAN_MUTATION = {}
# Sentinel idle reading for "no human mutation recorded this process" (a
# freshly booted server, or a document nobody has touched yet) -- large
# enough that PAUSE_MS's comparison always passes, but finite, because
# float("inf") serialises to invalid JSON ("Infinity") over jsonify.
_NEVER_MS = 24 * 3600 * 1000.0


def _still_busy(body):
    """R64, dogfooded. "It acts WHEN THE USER PAUSES, never mid-stroke" was
    only ever ADVISORY: `may_act` is computed when the agent POLLS, and the
    agent then spends a second or two looking at the picture before it
    writes. In the test painting two of its four assists landed 0.19 s and
    0.14 s after a human burst ENDED -- decided in a real pause, written
    after the painter had already started and finished the next one. A
    slower analysis or a longer burst and it would have painted straight
    through a live stroke, which is the one thing the cadence promise says
    it will never do.

    No amount of client-side re-polling closes that: the gap between
    "check" and "write" is exactly where the race lives. So the check moves
    INTO the write, under _DOC_LOCK, as an HTTP-conditional: a caller that
    passes `if_human_idle_ms` is saying "only apply this if the painter has
    been quiet at least this long", and gets 409 if they have not. OPT-IN,
    so nothing that does not ask for it changes behaviour -- but the
    reference agent always asks, and any agent that wants the guarantee to
    be real rather than polite can have it for one field."""
    try:
        want = float((body or {}).get("if_human_idle_ms") or 0)
    except (TypeError, ValueError):
        return None
    if want <= 0:
        return None
    last = _LAST_HUMAN_MUTATION.get(_viewing_doc_id())
    idle = _NEVER_MS if last is None else max(0.0, (time.time() - last) * 1000.0)
    if idle >= want:
        return None
    return jsonify(error="the painter is still working -- %d ms since their "
                         "last change, and you asked to wait for %d"
                         % (int(idle), int(want)),
                   human_idle_ms=idle, wanted_idle_ms=want,
                   retry_when_idle=True), 409


def _log_activity(did, lid, layer_name, by, kind):
    """Record one layer-touching mutation for /api/agent/tick's `changed`,
    and -- for a non-agent `by` -- reset how long this document has been
    quiet. Called from _record_agent_activity below, once per touched
    layer, AFTER the mutation actually succeeded."""
    from . import _MUT_REV
    log = _AGENT_ACTIVITY.setdefault(did, collections.deque(maxlen=_ACTIVITY_MAXLEN))
    # R64: `by_name` is deliberately NOT stamped here. A painter who
    # registers a display name after their first stroke would otherwise
    # appear in one feed under two names ("u_devin" for the early entries,
    # "Devin" for the later ones) and a presence chip fed from this would
    # show one collaborator twice. The uid is the fact; the name is a
    # lookup, and agent_tick does it at read time.
    log.append({"rev": _MUT_REV[0], "lid": lid, "layer_name": layer_name,
               "by": by, "kind": kind, "ts": time.time()})
    if not (by or "").startswith("agent:"):
        _LAST_HUMAN_MUTATION[did] = time.time()


@app.after_request
def _record_agent_activity(resp):
    """THE most important line in this file, per CONTRACT.md: an agent's
    own writes must never look like human activity, or a tick that follows
    one of its own strokes would see "something changed" forever and
    trigger on itself in a tight loop. The fix lives in ONE place --
    _log_activity's `by`-startswith-"agent:" check above -- rather than
    every call site remembering to exclude itself.

    Reuses _mutating_layer_targets() (the same body-shape parser the
    ownership gate uses) so this needs no second list of "which routes
    touch a layer" to keep in sync with the real one. Runs for every
    successful mutating POST, gated or not -- /api/paint on an UNOWNED or
    already-mine layer never even reaches the ownership gate's targets
    check with something to refuse, but it still must show up as
    "changed" for a watching agent."""
    if request.method != "POST" or not request.path.startswith("/api/") \
            or resp.status_code >= 400 or not resp.is_json:
        return resp
    try:
        targets = _mutating_layer_targets()
    except Exception:
        targets = set()
    if not targets:
        return resp
    kind = ("paint" if request.path in ("/api/paint", "/api/paint_batch")
            else "layer" if request.path == "/api/layer" else "other")
    try:
        uid = _owner_uid()
        did = _viewing_doc_id()
        doc = WS.docs.get(did)
        if kind == "paint" and (uid or "").startswith("agent:"):
            body = request.get_json(silent=True) or {}
            rj = resp.get_json() or {}
            sids = rj.get("sids") or ([rj["sid"]] if rj.get("sid") else [])
            _record_assist(did, uid, str(body.get("note") or "")[:160],
                           targets, sids)
        for lid in targets:
            try:
                name = doc.layer(lid).name if doc is not None else lid
            except Exception:
                name = lid          # deleted mid-request, or an unknown id
            _log_activity(did, lid, name, uid, kind)
    except Exception:
        pass                        # a logging bug must never break a real response
    return resp


def _layer_locked_for(l, uid):
    """Read-only mirror of _layer_permission's table (section 1), for
    /api/agent/tick's `locked` list. Deliberately NOT _layer_permission
    itself: that function's job is to DECIDE and, on an unowned or
    stale-owned layer, CLAIM or YIELD it as a side effect -- exactly right
    for a route that is actually about to write, and exactly wrong for a
    GET that only wants to report status. Calling the real one here would
    make merely polling the agent loop silently seize locks."""
    owner = getattr(l, "owner", "") or ""
    if owner == "" or (uid and uid == owner) \
            or (uid and uid in (getattr(l, "shared", None) or [])) \
            or owner.startswith("agent:"):
        return False
    if not uid:
        return True
    return _presence_absent_s(owner) <= STALE_S


# R64: what each connected agent says it UNDERSTOOD the prompt to mean.
# A document prompt whose effect is invisible is indistinguishable from one
# that was ignored -- which is what R63 shipped, and the R64 test painting
# proved it: the brief said "leave the window alone" and the agent grounded
# the window, with nothing anywhere to show the painter that their words had
# not landed. Server memory, like ACCESS_REQUESTS: a reading is a remark
# about this session, not part of the picture.
AGENT_READINGS = {}


# R65: what the agent DID, as the painter sees it -- one entry per assist,
# with the stroke ids so one click takes it back. The layer list already
# lets you hide or delete the agent's whole layer; this is finer: "that
# one, no". Session memory, per document, like the activity ring.
AGENT_ASSISTS = {}
_ASSIST_SEQ = [0]


def _record_assist(did, by, note, lids, sids):
    log = AGENT_ASSISTS.setdefault(did, collections.deque(maxlen=60))
    _ASSIST_SEQ[0] += 1
    log.append({"id": "A%d" % _ASSIST_SEQ[0], "by": by, "note": note,
                "layers": list(lids), "sids": list(sids), "at": time.time(),
                "undone": False})


@app.get("/api/agent/assists")
def agent_assists():
    """What an agent has DONE here, newest first: one entry per assist with
    the note the agent wrote, the layers it touched and the ids of exactly
    the strokes it painted. POST {"action":"undo","id"} takes one back."""
    did = _viewing_doc_id()
    out = []
    for a in AGENT_ASSISTS.get(did, ()):
        out.append(dict(a, by_name=_display_name(a["by"]) or a["by"],
                        layer_names=[_layer_name(l) for l in a["layers"]]))
    return jsonify(ok=True, assists=list(reversed(out)))


def _layer_name(lid):
    try:
        return DOC.layer(lid).name
    except Exception:
        return lid


@app.post("/api/agent/assists")
def agent_assist_act():
    """{action:"undo", id} -- take ONE assist back. Deletes exactly the
    strokes that assist painted (needs a faithful replay, which an agent's
    own journal-only layer always has). Anyone may do this: the assist is
    on the agent's layer, and an agent never locks a human out."""
    d = request.json or {}
    did = _viewing_doc_id()
    if d.get("action") != "undo":
        return jsonify(error="unknown action: %r" % d.get("action")), 400
    ent = next((a for a in AGENT_ASSISTS.get(did, ()) if a["id"] == d.get("id")), None)
    if ent is None:
        return jsonify(error="no such assist"), 404
    if ent["undone"]:
        return jsonify(ok=True, already=True)
    try:
        with _DOC_LOCK:
            n = DOC.delete_strokes([s_ for s_ in ent["sids"] if s_])
    except Exception as e:
        return jsonify(error="could not take that back: %s" % e), 400
    ent["undone"] = True
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, deleted=n)


@app.get("/api/agent/brief")
def agent_brief_get():
    """The document prompt (CONTRACT.md section 2). `default_brief` rides
    along on every GET so the UI can show what the agent actually follows
    even when the document has never had one written -- an empty brief
    field with no explanation reads as "nothing is happening", not "the
    quiet-support default is happening"."""
    now = time.time()
    return jsonify(brief=str(getattr(DOC, "agent_brief", "") or ""),
                   paused=bool(getattr(DOC, "agent_paused", False)),
                   updated_by=str(getattr(DOC, "agent_brief_by", "") or ""),
                   updated_at=float(getattr(DOC, "agent_brief_at", 0.0) or 0.0),
                   default_brief=DEFAULT_AGENT_BRIEF,
                   # only agents still around: a reading from a run that
                   # ended is a claim about nothing
                   readings=[dict(v, uid=k, name=_display_name(k))
                             for k, v in sorted(AGENT_READINGS.items())
                             if _presence_absent_s(k) <= STALE_S
                             and now - v.get("at", 0) < 3600])


@app.post("/api/agent/brief")
def agent_brief_set():
    """Save the document prompt. Excluded from the ownership gate (its path
    starts with /api/agent/, see _layer_ownership_gate) -- the brief is a
    property of the whole picture, not of any one layer, so it is never
    something a layer lock could refuse."""
    d = request.json or {}
    me = _owner_uid()
    _touch_presence(me)
    if "paused" in d and "brief" not in d:
        # R65: "not right now". Sometimes you do not want help, and the
        # only ways to say so were to disconnect the agent or write a brief
        # telling it to do nothing. A document-level switch; the tick
        # reports it, and may_act is false while it is set.
        DOC.agent_paused = bool(d.get("paused"))
        return jsonify(ok=True, paused=DOC.agent_paused)
    if "reading" in d and "brief" not in d:
        # an AGENT reporting how it read the prompt -- never a rewrite of
        # the prompt itself, which belongs to the human who wrote it
        if not me:
            return jsonify(error="a reading needs an X-User"), 400
        AGENT_READINGS[me] = {"reading": str(d.get("reading") or "")[:600],
                              "at": time.time()}
        return jsonify(ok=True)
    DOC.agent_brief = str(d.get("brief") or "")[:4000]
    DOC.agent_brief_by = me
    DOC.agent_brief_at = time.time()
    return jsonify(ok=True)


@app.get("/api/agent/tick")
def agent_tick():
    """The reactive loop, one call (CONTRACT.md section 2). An agent polls
    this with the last `rev` it saw; `may_act` is decided HERE, server-side,
    so the pause rule is testable without a browser and identical for every
    client. `pause_ms` overrides PAUSE_MS -- query-param only, for tests
    that cannot wait 1.8 real seconds; no client-facing control offers it."""
    # Polling the tick IS the agent being here: while it waits for the
    # painter to pause it makes no POSTs at all, so without this stamp an
    # attentive, well-behaved agent aged out of the roster after STALE_S
    # and the panel said "none connected" about an agent that was watching
    # every stroke.
    _touch_presence(_owner_uid())
    did = _viewing_doc_id()
    doc = WS.docs.get(did)
    from . import _MUT_REV
    rev = _MUT_REV[0]
    try:
        since = int(request.args.get("since", 0))
    except (TypeError, ValueError):
        since = 0
    try:
        pause_ms = float(request.args.get("pause_ms", PAUSE_MS))
    except (TypeError, ValueError):
        pause_ms = PAUSE_MS

    entries = [e for e in _AGENT_ACTIVITY.get(did, ())
              if e["rev"] > since]
    # group by layer -- an agent responds to WHAT happened, not a raw
    # per-mutation diary it would have to re-derive that from every time
    by_lid = {}
    for e in entries:
        agg = by_lid.setdefault(e["lid"], {"lid": e["lid"], "n": 0})
        agg["layer_name"] = e["layer_name"]      # last-known name wins
        agg["by"] = e["by"]
        agg["by_name"] = _display_name(e["by"]) or e["by"]   # R64: live
        agg["kind"] = e["kind"]
        agg["n"] += 1
    changed = [by_lid[k] for k in sorted(by_lid)]

    last_human = _LAST_HUMAN_MUTATION.get(did)
    human_idle_ms = (_NEVER_MS if last_human is None
                     else max(0.0, (time.time() - last_human) * 1000.0))
    # "something new since `since` that a human did" -- an agent's OWN
    # entries in `changed` must never satisfy this, or a tick that follows
    # its own paint would see may_act stay true and loop on itself forever.
    human_did_something = any(not (e["by"] or "").startswith("agent:")
                              for e in entries)
    paused = bool(getattr(doc, "agent_paused", False)) if doc is not None else False
    may_act = human_idle_ms >= pause_ms and human_did_something and not paused

    uid = _owner_uid()
    mine, locked = [], []
    if doc is not None:
        for l in doc.layers:
            owner = getattr(l, "owner", "") or ""
            if uid and owner == uid:
                mine.append(l.id)
            elif _layer_locked_for(l, uid):
                locked.append({"id": l.id, "owner": owner,
                               "owner_name": _display_name(owner)})

    brief = str(getattr(doc, "agent_brief", "") or "") if doc is not None else ""
    return jsonify(rev=rev, changed=changed, human_idle_ms=human_idle_ms,
                   may_act=may_act, brief=brief or DEFAULT_AGENT_BRIEF,
                   mine=mine, locked=locked, paused=paused,
                   # R64: an agent watching a different picture from the
                   # painter sees a document where nothing ever happens and
                   # has no way to tell that from a painter who has stopped
                   doc=did, workspace_active_doc=WS.active,
                   following_workspace=did == WS.active)


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
        elif source.startswith("sim:"):
            # R8 (leCore 0.2.20): engine-simulated clips. sim:smoke and
            # sim:particles render a deterministic frame list ONCE via the
            # engine's smoke_animation / particle_animation and loop it --
            # an animated source with no file and no network.
            self.kind = "sim"
            self._sim_frames = None
            self.status = "rendering simulation…"
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

    def _tick_sim(self):
        import numpy as np
        if self._sim_frames is None:
            what = self.source.split(":", 1)[1].strip() or "smoke"
            try:
                m = mind()
                if what.startswith("particle"):
                    if not hasattr(m, "particle_animation"):
                        raise AttributeError("particle_animation")
                    frames = m.particle_animation(n=400, steps=48,
                                                  width=256, height=192)
                else:
                    if not hasattr(m, "smoke_animation"):
                        raise AttributeError("smoke_animation")
                    frames = m.smoke_animation(steps=48, shape=(96, 96))
                self._sim_frames = [np.asarray(f, np.float32)[..., :3]
                                    for f in frames]
                self.status = ("ok — %d-frame engine simulation (loops)"
                               % len(self._sim_frames))
            except AttributeError as e:
                self._sim_frames = []
                self.status = ("error: this engine build has no %s — "
                               "sim: sources need leCore >= 0.2.20" % e)
            except Exception as e:
                self._sim_frames = []
                self.status = "error: simulation failed (%s)" % str(e)[:80]
        if not self._sim_frames:
            return self.frame
        t = time.time()
        seq = int(t * self.fps) if self.play else self.seq
        idx = seq % len(self._sim_frames)
        if seq != self.seq or self.frame is None:
            self.frame = self._sim_frames[idx]
            self.seq = seq
        return self.frame

    def get(self):
        if self.kind == "sim":
            return self._tick_sim(), self.seq
        if self.kind == "test":
            return self._tick_test(), self.seq
        if self.kind == "image":
            return self._tick_image(), self.seq
        return self.frame, self.seq

    def close(self):
        self._stop.set()


class _MediaManager:
    def __init__(self):
        # (doc_id, node_id) -> _MediaSource. Keying by node id alone made
        # doc A's N1 and doc B's N1 share one video slot -- and node ids
        # collide across docs by construction (every graph counts from N1).
        self.sources = {}

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


    def statuses(self, did=None):
        """Per-node statuses, keyed by NODE id for the given doc (the UI
        looks nodes up by their id in the visible graph; other docs' sources
        are not its business). did=None returns everything, node-keyed, for
        introspection."""
        return {(nid[1] if isinstance(nid, tuple) else nid):
                {"status": s.status, "has_frame": s.frame is not None}
                for nid, s in self.sources.items()
                if did is None
                or (isinstance(nid, tuple) and nid[0] == did)}

    def purge_doc(self, did):
        """Stop and drop every source belonging to a closed document --
        otherwise its capture threads keep pulling frames forever."""
        for k in [k for k in self.sources
                  if isinstance(k, tuple) and k[0] == did]:
            try:
                self.sources[k].close()
            except Exception:
                pass
            self.sources.pop(k, None)


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


@app.post("/api/export/lut")
def export_lut():
    """Bake a node's COLOUR transform into a .cube 3D LUT (R4). Walks the
    single-image-input chain upstream of the chosen node (each hop a colour
    op), applies it to an identity lattice, and writes the mapped lattice as
    a .cube -- so a look built from Wheels/Levels/Curves/Post FX travels to
    Resolve/Premiere/OBS. HONESTY GUARD: the chain is run twice with the
    lattice laid out in two different spatial arrangements; if the mapped
    colours disagree, a stage is position-dependent (vignette, blur...) and
    the header says so instead of silently baking one arrangement's answer.
    Body: {node?, size? (17/33/65), stop? (node id to treat as source)}."""
    d = request.json or {}
    GRAPH.ensure_default()
    nid = d.get("node") or GRAPH.output_node()
    size = int(d.get("size", 33))
    if size not in (17, 33, 65):
        return jsonify(error="size must be 17, 33 or 65"), 400
    stop = d.get("stop") or None

    # collect the chain: nid up through single-image-input colour ops
    chain = []
    cur = nid
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        n = GRAPH.nodes.get(cur)
        if n is None:
            break
        meta = OPS.get(n["type"])
        if meta is None or not meta["inputs"]:
            break                          # a generator: the source, excluded
        chain.append(n)
        if cur == stop:
            break
        first = meta["inputs"][0]
        src = (n.get("inputs") or {}).get(first)
        if not src:
            break
        cur = str(src).split(".", 1)[0]
    if not chain:
        return jsonify(error="node has no image-input chain to bake"), 400
    chain.reverse()

    idx = np.linspace(0.0, 1.0, size)
    lat = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape(-1, 3)

    def run(pixels, h, w):
        img = pixels.reshape(h, w, 3).astype(np.float32)
        for n in chain:
            meta = OPS[n["type"]]
            params = {p["name"]: p["default"] for p in meta["params"]}
            params.update(n.get("params") or {})
            src_img = img
            if meta.get("rgba"):
                src_img = np.concatenate(
                    [img, np.ones(img.shape[:2] + (1,), np.float32)], -1)
            ins = {meta["inputs"][0]: src_img}
            for extra in meta["inputs"][1:]:
                ins[extra] = None
            out = meta["fn"]((h, w), ins, params)
            if isinstance(out, dict):
                out = out.get("out")
            img = np.asarray(out, np.float32)[..., :3]
        return img.reshape(-1, 3)

    h1, w1 = size, size * size
    a = run(lat, h1, w1)
    b = run(lat[::-1], h1, w1)[::-1]      # same colours, different positions
    spatial = float(np.abs(a - b).max())
    lines = ["# generated by leStudio /api/export/lut",
             "# chain: " + " -> ".join(n["type"] for n in chain)]
    if spatial > 1e-3:
        lines.append("# WARNING: chain is position-dependent (max colour "
                     "disagreement %.4f between two lattice layouts) -- a "
                     "spatial stage (vignette/blur/grain?) is baked from ONE "
                     "arrangement and will not travel faithfully" % spatial)
    lines.append("LUT_3D_SIZE %d" % size)
    out = np.clip((a + b) * 0.5, 0, 1).reshape(size, size, size, 3)
    for bb in range(size):
        for gg in range(size):
            for rr in range(size):
                v = out[rr, gg, bb]
                lines.append("%.6f %.6f %.6f" % (v[0], v[1], v[2]))
    body = "\n".join(lines) + "\n"
    return app.response_class(body, mimetype="text/plain", headers={
        "Content-Disposition": "attachment; filename=lestudio.cube"})


@app.post("/api/export/glsl")
def export_glsl():
    """Compile a Post FX node's POINTWISE chain to a Shadertoy fragment via
    leCore's postfx_to_glsl (R4). Non-pointwise stages (bloom, glare, grain,
    chromatic aberration -- they need neighbours or multiple passes) are
    emitted as '// skipped' comments, not silently dropped; agx is likewise
    reported (the engine's GLSL algebra doesn't carry it yet).
    Body: {node} -- must be a Post FX node."""
    from . import _postfx_steps
    d = request.json or {}
    nid = d.get("node")
    n = GRAPH.nodes.get(nid or "")
    if n is None or n.get("type") != "Post FX":
        return jsonify(error="pass the id of a Post FX node"), 400
    meta = OPS["Post FX"]
    params = {p["name"]: p["default"] for p in meta["params"]}
    params.update(n.get("params") or {})
    steps = _postfx_steps(params)
    if params.get("tonemap") == "agx":
        steps = steps + [("agx", {})]
    if not steps:
        return jsonify(error="this Post FX node is all defaults -- nothing "
                             "to export"), 400
    try:
        import holographic.rendering.holographic_postfx as _pf
        src = _pf.chain_to_glsl(steps, name="lestudio_grade",
                                skip_unsupported=True)
    except Exception as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True, glsl=src,
                   note="stages needing multi-pass are '// skipped' comments "
                        "in the source; everything else matches the node to "
                        "float precision")


@app.get("/api/materials/substances")
def materials_substances():
    """R8 (leCore 0.2.20): the engine's physical-materials database, filtered
    to entries with a REFRACTIVE INDEX -- the ones a layer's optics can
    honestly imitate. The Layer options dialog offers them as one-click IOR
    presets ("honey", "diamond", "seawater"...)."""
    m = mind()
    if not hasattr(m, "material_data"):
        return jsonify(ok=True, substances=[],
                       note="engine build has no material_data")
    cats = ("liquid", "glass", "mineral", "polymer", "gas")
    out = []
    for cat in cats:
        try:
            listing = m.material_data(category=cat)
        except Exception:
            continue
        for name in (listing or {}).get("materials", []):
            try:
                d = m.material_data(name)
            except Exception:
                continue
            if not d or not d.get("found") or "refractive" not in d:
                continue
            out.append({"name": name, "category": cat,
                        "refractive": float(d["refractive"]),
                        "density": float(d.get("density", 0.0))})
    out.sort(key=lambda x: (x["category"], x["name"]))
    return jsonify(ok=True, substances=out)


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


# R59: /api/status cost 10.4 SECONDS per call -- accel_status() 7.0s and
# engine_status() 3.4s, both recomputed from scratch every time. The page
# asks for it on load, and (worse) a slow status call saturates the server,
# so an unrelated stroke or selection queued behind it: the app froze for
# ten seconds at exactly the moment a person first touched it, and that is
# the "stroke round-trip 1.2s under load" note in the UX backlog too.
# Both answers are PROCESS-STATIC -- which optional wheels are importable
# and which faculties this engine build has cannot change while we run --
# so they are computed once, warmed in the background at boot so the first
# request never pays, and re-checkable with ?fresh=1 for a developer who
# just pip-installed something.
_STATUS = {"accel": None, "engine": None, "lock": threading.Lock()}


def _status_parts(fresh=False):
    if fresh:
        _STATUS["accel"] = _STATUS["engine"] = None
    if _STATUS["accel"] is None or _STATUS["engine"] is None:
        with _STATUS["lock"]:            # one worker pays, the rest wait once
            if _STATUS["accel"] is None:
                _STATUS["accel"] = accel_status()
            if _STATUS["engine"] is None:
                eng = None
                try:
                    from . import mind as _mind_fn
                    _m = _mind_fn()
                    if hasattr(_m, "engine_status"):
                        eng = _m.engine_status()
                except Exception:
                    eng = None
                _STATUS["engine"] = {"v": eng}
    return _STATUS["accel"], _STATUS["engine"]["v"]


def _warm_status():
    try:
        _status_parts()
    except Exception:
        pass


threading.Thread(target=_warm_status, daemon=True).start()


@app.get("/api/status")
def status():
    """Engine status: gpu availability, leCore version, faculty report. Cached (process-static); ?fresh=1 recomputes."""
    a, _eng_cached = _status_parts(fresh=request.args.get("fresh"))
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
    # R55: the engine's own status panel (version, extras, determinism
    # policy, budget) rides along when the build provides it -- the
    # APP_FOUNDATION rule is to gate on the build in front of us, never
    # keep a client-side list of what to check. R59: cached, see above.
    eng = _eng_cached
    # R57: the shared live workspace, as the engine describes it -- rev,
    # sections and which apps wrote them -- so a status call answers "is
    # anyone else working in this workspace?" without a second protocol.
    lews = None
    try:
        _ws = _lews_ws()
        if _ws is not None:
            lews = _ws.describe()
            lews["root"] = _WS_ROOT
    except Exception:
        lews = None
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
    return jsonify(engine=eng, lews=lews, gpu=gpu_flag, jit=a["jit"],
                   live=LIVE["on"],
                   live_error=LIVE["error"], accel=have_map,
                   accel_missing=missing, subsystems=subsystems,
                   threads=int(os.environ.get("LESTUDIO_THREADS", "0")) or None,
                   gpu_report=report, advice=a.get("advice") or [],
                   determinism=a.get("determinism"))


@app.post("/api/graph/run")
def graph_run():
    """Evaluate a node (default: the Output node) as a background JOB with
    progress reporting and cancellation.

    R5 #5: an optional `w` in the body renders at that WIDTH via render_at
    instead of full canvas res (measured 473 ms vs 1618 ms full at 512^2 --
    3.4x), for interactive param-commit previews; the PNG lands in the job's
    result (/api/job/<id>/result). Reduced-res runs do NOT commit Layer out /
    Mask out targets -- baking document pixels at preview resolution would
    corrupt them; full-res (no `w`) keeps today's evaluate+commit behaviour."""
    d = request.json or {}
    GRAPH.ensure_default()
    nid = d.get("id") or GRAPH.output_node()
    try:
        want_w = int(d["w"]) if d.get("w") else None
    except (TypeError, ValueError):
        return jsonify(error="w must be an integer width"), 400
    if want_w is not None and not (16 <= want_w <= 4096):
        return jsonify(error="w must be between 16 and 4096"), 400
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
            if want_w is not None:
                h = max(int(round(want_w * DOC.height / max(DOC.width, 1))), 8)
                img = _renderable(GRAPH.render_at(nid, want_w, h))
                # encode here, not via _png(): this thread has no flask
                # request context
                import numpy as _np
                from PIL import Image as _PImage
                a = _np.clip(_np.asarray(img, _np.float32), 0, 1)
                if a.ndim == 2:
                    a = _np.stack([a] * 3, -1)
                buf = io.BytesIO()
                _PImage.fromarray((a * 255).astype("uint8")).save(buf, "PNG")
                job["result"] = buf.getvalue()
                job["mime"] = "image/png"
                job["filename"] = "graph_preview.png"
            else:
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


def _doc_dirty(d):
    """Unsaved-work signal. `bool(d._undo)` stayed True forever after the
    first edit -- saving never cleared it, so every close warned. A save
    (explicit download or autosave) records the undo length it saw; dirty is
    'the history moved since then'. Undoing back TO the saved point also
    reads clean, which is what people expect."""
    return len(getattr(d, "_undo", []) or []) != \
        int(getattr(d, "_saved_undo_len", 0))


def _mark_saved():
    """Every doc's current undo depth becomes the 'saved' waterline."""
    for d in WS.docs.values():
        d._saved_undo_len = len(getattr(d, "_undo", []) or [])


@app.get("/api/state")
def state():
    """THE complete truth: doc, layers (with alpha_lock/clip), masks, selections, splines, brushes, graph, ops catalog."""
    active = _viewing_doc_id()               # the CALLER's active document
    return jsonify({
        "capabilities": _capabilities(),
        "docs": [{"id": d.id, "name": d.name,
                  # unsaved-work signal: closing a dirty document should warn
                  "dirty": _doc_dirty(d), "active": d.id == active,
                  "layers": [{"id": l.id, "name": l.name} for l in d.layers],
                  "groups": [{"id": g["id"], "name": g["name"]} for g in d.groups],
                  "masks": [{"id": m.id, "name": m.name} for m in d.masks]}
                 for d in WS.docs.values()],
        "active_doc": active,
        "dpi": float(getattr(DOC, "dpi", 72.0)),
        # Who else is here, and what each USER is looking at. Everything here
        # keys by user id -- names, viewing and the caller comparison. When
        # `me` compared _req_uid-keyed names against the TAB id, users saw
        # their own ghost chip and peers' docs showed wrongly.
        # Who is here. Named editors (a browser tab announces its name on
        # the SSE stream) UNION anyone else this process has heard from
        # inside the stale window -- R63: an agent, or any script, that
        # paints with an X-User but never announces a display name was
        # invisible in the roster, so the one thing CONTRACT.md section 4
        # promises ("it appears in presence like anyone else, so the user
        # can see it is there") was not true of an agent that simply got
        # to work. _touch_presence already stamps every identified request,
        # so the server knew all along; only this list did not say so.
        # Bounded by STALE_S for the same reason the lock yields at it: a
        # script that pinged once an hour ago is not "here".
        "peers": [{"id": uid, "name": _display_name(uid),
                   # R64: a peer who has never activated a document is
                   # following the workspace, not sitting nowhere -- the
                   # old None here read as "this person is unplaced"
                   "doc": SYNC["viewing"].get(uid) or WS.active,
                   "activity": SYNC["activity"].get(uid),
                   "agent": uid.startswith("agent:"),
                   "me": uid == _req_uid()}
                  for uid in sorted(set(SYNC.get("names", {}))
                                    | set(SYNC.get("last_seen", {})))
                  if _presence_absent_s(uid) <= STALE_S],
        "width": DOC.width, "height": DOC.height,
        # R63: owner_name/mine/agent ride on TOP of l.meta()'s bare
        # owner/shared -- they need the requesting user's id and the
        # presence table, neither of which a Layer object has, so they
        # belong here rather than growing Layer.meta() a request parameter.
        # R64 (dogfooded): an access request had NO notification path.
        # /api/access returned {ok,id} and then nothing happened anywhere --
        # the owner learned they had been asked only by independently
        # POSTing {"action":"list"}, which they have no reason to do. The
        # refusal tells the REQUESTER exactly how to ask and nothing at all
        # told the person being asked. This is the count every client is
        # already polling for, so a chip can light up without a second
        # request; the detail still comes from /api/access.
        # R64: which document the workspace considers current, next to the
        # one THIS user is on. They are normally the same; when they are
        # not, the client can say so instead of leaving someone painting
        # into a picture nobody else can see.
        "workspace_active_doc": WS.active,
        "following_workspace": _viewing_doc_id() == WS.active,
        "access_pending": sum(
            1 for r in ACCESS_REQUESTS.get(_viewing_doc_id(), ())
            if r["to"] == _owner_uid() and r["state"] == "pending"),
        # R64: the live `strokes` list is a WINDOW -- older segments spool
        # to disk (_journal_spool) and only _journal_spooled remembers
        # them. Asking the window alone said "no strokes" about the most
        # heavily painted layers in the document, which is exactly
        # backwards, and it is what R59 gates "Re-render strokes" on.
        "layers": [dict(l.meta(),
                        has_strokes=(l.id in getattr(DOC, "_journal_spooled", {})
                                     or any(k["layer"] == l.id
                                            for k in DOC.strokes)),
                        owner_name=_display_name(getattr(l, "owner", "") or ""),
                        mine=bool(_owner_uid())
                             and (getattr(l, "owner", "") == _owner_uid()
                                  or _owner_uid() in (getattr(l, "shared", None) or [])),
                        agent=str(getattr(l, "owner", "") or "").startswith("agent:"))
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
        # R64: this array is the TAIL, not the journal (see the slice at
        # its end). Callers derived per-layer counts from it and got them
        # wrong; the totals below say so out loud.
        "strokes_total": (len(DOC.strokes)
                          + sum((getattr(DOC, "_journal_spooled", {}) or {}).values())),
        "strokes_are_a_tail": True,
        "strokes": [{"id": k["id"], "layer": k["layer"],
                     "points": len(k["points"]),
                     "ends": [[round(float(k["points"][0][0]), 1),
                               round(float(k["points"][0][1]), 1)],
                              [round(float(k["points"][-1][0]), 1),
                               round(float(k["points"][-1][1]), 1)]]
                     if k["points"] else None,
                     "erase": bool(k["brush"].get("erase"))}
                    for k in getattr(DOC, "strokes", [])][-64:],
        "media_status": MEDIA.statuses(active),
        "brushes": [b.meta() for b in DOC.brushes],
        "graph": list(GRAPH.ensure_default().values()),
        # the graph's revision: clients echo it as base_rev on whole-graph
        # POSTs so a stale write 409s instead of erasing someone's nodes
        "grev": int(getattr(GRAPH, "grev", 0)),
        "output_node": GRAPH.output_node(),
        "ops": op_catalog(),
        "blend_modes": list(BLEND_MODES),
        # R67: how many documents are RESIDENT, and how many of those are
        # blanks nobody painted in. Nothing closes a document -- /api/new
        # switches the active one and leaves the old one alive -- so a
        # session that re-runs a script accumulates full paintings with no
        # cap and nothing to notice it. The writer stops saving the blanks;
        # this is so the app can offer to close the rest rather than the
        # person discovering them in a 160 MB file.
        "docs_resident": len(WS.docs),
        "docs_blank": [k for k, v in WS.docs.items()
                       if k != WS.active and not _doc_has_work(v)],
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
    me = _req_uid()
    if me:
        # creating a document activates it FOR ITS CREATOR (and, via
        # WS.active, for clients that never picked a doc) -- not for peers
        # who have their own viewing entry
        SYNC["viewing"][me] = doc.id
    return jsonify(ok=True, doc={"id": doc.id, "name": doc.name,
                                 "width": w, "height": h, "dpi": doc.dpi})


def _recommit_foreign_bakes(did):
    """Cross-doc Layer-out bakes go STALE while their source document is
    edited elsewhere: the baked pixels only refresh when this graph commits.
    Re-run the commit path when this doc becomes active, but only for graphs
    that both READ a foreign doc and WRITE a bake target -- everything else
    activates with zero extra work (and the commit path itself is signature-
    cached, so an unchanged upstream costs one evaluation of small graphs)."""
    g = WS.graphs.get(did)
    if g is None:
        return
    reads_foreign = any(
        n.get("type") in ("Layer", "Layer group", "Mask")
        and (n.get("params") or {}).get("doc") not in ("", None, did)
        for n in g.nodes.values())
    writes_bake = any(n.get("type") in ("Layer out", "Mask out")
                      for n in g.nodes.values())
    if reads_foreign and writes_bake:
        try:
            g.commit_layer_outputs()
        except Exception:
            pass                    # a broken graph must not block activate


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
    # every id-taking action shares one guard: a missing id is a client bug
    # (400), an unknown id means the doc is gone (404) -- both used to 500
    # (KeyError) or, worse, silently act on the WRONG document
    if act in ("activate", "close", "rename"):
        did = d.get("id")
        if not did:
            return jsonify(error="which document? '%s' needs an id" % act), 400
        if did not in WS.docs:
            return jsonify(error="no such document: %s (it may have been "
                                 "closed)" % did), 404
    if act == "activate":
        # PER USER: only the caller's viewing entry moves. WS.active is kept
        # as the default for clients that never activate anything (agents,
        # old clients), so single-user behaviour is unchanged -- but one
        # person switching docs no longer redirects everyone else's edits.
        me = _req_uid()
        if me:
            SYNC["viewing"][me] = d["id"]
        WS.active = d["id"]
        _recommit_foreign_bakes(d["id"])
    elif act == "close":
        doc = WS.docs[d["id"]]
        if _doc_dirty(doc) and not d.get("force"):
            # Closing threw away every edit with no warning at all. Report it
            # and let the client confirm rather than deciding for the user.
            return jsonify(ok=False, needs_confirm=True,
                           name=doc.name, edits=len(doc._undo)), 409
        # cross-doc references: another doc's graph reading THIS doc turns
        # into ':gone' transparent zeros the moment it closes -- warn first
        refs = sorted({WS.docs[oid].name for oid, g in WS.graphs.items()
                       if oid != d["id"] and oid in WS.docs
                       and any((n.get("params") or {}).get("doc") == d["id"]
                               for n in g.nodes.values())})
        if refs and not d.get("force"):
            return jsonify(ok=False, needs_confirm=True, name=doc.name,
                           referrers=refs,
                           error="other documents read this one through "
                                 "their node graphs: %s -- closing blanks "
                                 "those nodes (pass force to close anyway)"
                                 % ", ".join(refs)), 409
        if not WS.close(d["id"]):
            # WS.close refuses to strand the app with zero documents; that
            # used to be swallowed as ok:True
            return jsonify(ok=False,
                           error="cannot close the last document"), 200
    elif act == "rename":
        WS.docs[d["id"]].name = d.get("name") or WS.docs[d["id"]].name
    elif act == "settings":
        if d.get("id") and d["id"] not in WS.docs:
            # a STALE id silently resized the ACTIVE document -- the exact
            # wrong-target class of bug. Only fall back when no id was given.
            return jsonify(error="no such document: %s (it may have been "
                                 "closed)" % d["id"]), 404
        doc = WS.docs[d["id"]] if d.get("id") else WS.docs[_viewing_doc_id()]
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
        # The STOCK is a document setting, and this is where a caller looks
        # for document settings. It used to live only on /api/paper, so
        # {"action":"settings","paper":"smooth"} answered ok:true and
        # changed nothing -- and a whole picture came back painted on
        # heavy canvas weave, because nothing had said no.
        if d.get("paper") is not None:
            try:
                doc.set_paper(str(d["paper"]))
            except ValueError as e:
                return jsonify(error=str(e)), 400
        # ...and REFUSE anything this route does not understand, for the
        # same reason: a setting that is quietly dropped is worse than one
        # that is rejected, because the caller goes on believing it took.
        unknown = sorted(set(d) - _SETTINGS_KEYS)
        if unknown:
            return jsonify(error="unknown document setting%s: %s (this route "
                                 "takes %s)" % ("s" if len(unknown) > 1 else "",
                                                ", ".join(unknown),
                                                ", ".join(sorted(_SETTINGS_KEYS
                                                                 - {"action", "id"})))), 400
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
    color | gradient | pattern | node (any Fill-out node's image) --
    or a GENERATED source (R48): {"type": "generated", "style": "scribble"|
    "line"|"hatch"|"both", ...generator params, "sample": "layer"|
    "composite", "softness"} fills the bucket region with scribble/hatch
    strokes instead of flat content (every stroke journaled as paint)."""
    d = request.json or {}
    try:
        src = d.get("source") or {}
        if src.get("type") == "generated":
            kw = {}
            for k in ("curl", "thickness", "density", "angle", "spacing",
                      "opacity", "wobble", "hardness", "depth", "horizon",
                      "perspective", "size", "size_jitter", "lean",
                      "wind"):
                if k in src:
                    kw[k] = float(src[k])
            if "weave" in src:
                kw["weave"] = str(src["weave"])
            if "element" in src:
                kw["element"] = str(src["element"])
            if "color2" in src:
                kw["color2"] = tuple(src["color2"])
            if "custom" in src:
                kw["custom"] = src["custom"]
            if "color" in src:
                kw["color"] = tuple(src["color"])
            _rpt = {}
            n = DOC.fill_generated(
                d["layer"], int(d["x"]), int(d["y"]),
                style=str(src.get("style", "both")),
                tolerance=float(d.get("tolerance", 0.12)),
                contiguous=bool(d.get("contiguous", True)),
                selection=d.get("selection") or None,
                sel_invert=bool(d.get("sel_invert")),
                softness=float(src.get("softness", 2.0)),
                sample=str(src.get("sample", "layer")),
                seed=src.get("seed"), _report=_rpt, **kw)
            GRAPH.commit_layer_outputs()
            # WHY DIDN'T THAT PAINT? (generated fills): `filled` used to be
            # hard-coded 0 on this path -- it never meant anything, so the
            # client's "filled 0 px" fallback line was always a lie. Report
            # the real seed-region coverage instead. And a generator that
            # laid zero strokes is the same silent no-op as any other tool:
            # either the bucket found nothing to flood at that point (empty
            # coverage), or the region existed but was too sparse/thin for
            # the current density/spacing to place even one stroke -- both
            # read as "stitched 0 strokes" with no clue why (R58 UX open #3).
            covered = int(_rpt.get("covered", 0))
            gwarn = None
            if int(n) == 0:
                if covered == 0:
                    gwarn = ("nothing to fill here — the flood found no "
                             "matching pixels at that point; click inside a "
                             "shape's edges, or raise the tolerance")
                else:
                    gwarn = ("the region here is too thin or sparse for "
                             "these settings — no strokes fit; try a lower "
                             "spacing or a smaller size/thickness")
            resp = dict(ok=True, filled=covered, strokes=int(n))
            if gwarn is not None:
                resp["warning"] = gwarn
            return jsonify(**resp)
        content = _fill_content(src or None, DOC.height, DOC.width)
        n = DOC.flood_fill(d["layer"], int(d["x"]), int(d["y"]), content,
                           tolerance=float(d.get("tolerance", 0.12)),
                           contiguous=bool(d.get("contiguous", True)),
                           selection=d.get("selection") or None,
                           sel_invert=bool(d.get("sel_invert")),
                           spec=(d.get("source") or {"type": "color"})
                           if (d.get("source") or {}).get("type", "color")
                           in ("color", "gradient", "pattern") else None)
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
    """Set the display name for this USER (alias of /api/editors/name; no
    invite needed on the host side)."""
    # This route used to key names AND viewing by the TAB id while the
    # roster and /api/editors/name keyed by USER id -- so a person who set
    # their name here saw a ghost chip of themselves (their uid row unnamed,
    # their tab id row named). One map, one key: the user id.
    d = request.json or {}
    me = _req_uid()
    if not me:
        return jsonify(error="send an X-User (or X-Client) header"), 400
    _touch_presence(me)         # announcing yourself IS being here (R63)
    SYNC["names"][me] = str(d.get("name", "")).strip()[:24]
    # R64, dogfooded, and the worst bug this round found: this line used to
    # be `SYNC["viewing"].setdefault(me, WS.active)` -- saying your NAME
    # pinned you to whatever document happened to be open at that moment,
    # permanently. A second painter who introduced herself, and then kept
    # working while the picture was replaced, spent an entire session
    # painting a complete wall, window, bottle, bowl and lemons into a
    # document nobody was looking at. Every write returned 200. Nothing
    # anywhere said the two of them were on different pictures.
    #
    # A name is not a choice of document. Only /api/doc activate is, and a
    # user who has never made that choice keeps FOLLOWING the workspace's
    # active document (_viewing_doc_id's fallback) -- which is what a
    # collaborator joining a shared studio, and what an agent connecting to
    # watch someone paint, both actually want.
    return jsonify(ok=True)


@app.post("/api/presence/activity")
def presence_activity():
    """What the requesting user is doing right now: {"tool", "layer"}.
    Stored per user and echoed in the SSE feed and /api/editors so presence
    chips can say 'painting on Leaves' instead of just a name. Excluded from
    the rev bump (see _bump_rev): a ping changes nothing in the workspace."""
    d = request.json or {}
    me = _req_uid()
    if not me:
        return jsonify(error="send an X-User (or X-Client) header"), 400
    _touch_presence(me)         # an activity ping IS being here (R63)
    SYNC["activity"][me] = {"tool": str(d.get("tool", ""))[:24],
                            "layer": str(d.get("layer", ""))[:24],
                            "at": time.time()}
    _lews_mirror_touch(me, SYNC["activity"][me])
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


@app.post("/api/scribble")
def scribble():
    """R47 scribble brush: {"layer", "x", "y", "radius", "curl" 0..1,
    "thickness", "color", "opacity", "hardness", "density", "length",
    "seed"?, "selection"?, "sel_invert"?, "feather"?, "record"}. Curl-noise
    strands painted as ordinary journaled strokes; a (feathered) selection
    shapes and fades the scribble. Returns {ok, strokes}."""
    d = request.json or {}
    try:
        n = DOC.scribble(
            d["layer"], float(d["x"]), float(d["y"]),
            radius=float(d.get("radius", 60)),
            curl=float(d.get("curl", 0.5)),
            thickness=float(d.get("thickness", 1.6)),
            color=tuple(d.get("color", (0, 0, 0))),
            opacity=float(d.get("opacity", 0.85)),
            hardness=float(d.get("hardness", 0.7)),
            density=float(d.get("density", 1.0)),
            length=float(d.get("length", 1.0)),
            seed=d.get("seed"),
            selection=d.get("selection") or None,
            sel_invert=bool(d.get("sel_invert")),
            feather=float(d.get("feather", 0.0)),
            poly=d.get("poly"),
            record=bool(d.get("record", True)))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, strokes=int(n))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/hatchfill")
def hatchfill():
    """R47 line/hatch shading brush: {"layer", "x"?, "y"?, "radius",
    "angle", "spacing", "thickness", "mode": "line"|"hatch"|"both"|
    "weave"|"cross"|"stitch" (R49 textile modes; "weave" also takes a
    "weave": "plain"|"twill"|"satin"|"basket" interlacement),
    "color", "opacity", "hardness", "wobble", "cross_angle"?, "seed"?,
    "selection"?, "sel_invert"?, "feather"?, "area": "brush"|"selection",
    "record"}. 'both' is value-aware: composite darks cross-hatch, lights
    fade to sparse broken lines. area='selection' shades the whole
    (feathered) gate. Returns {ok, strokes}."""
    d = request.json or {}
    try:
        n = DOC.hatch_fill(
            d["layer"], x=d.get("x"), y=d.get("y"),
            radius=float(d.get("radius", 80)),
            angle=float(d.get("angle", 45)),
            spacing=float(d.get("spacing", 7)),
            thickness=float(d.get("thickness", 1.2)),
            mode=d.get("mode", "both"),
            color=tuple(d.get("color", (0, 0, 0))),
            opacity=float(d.get("opacity", 0.9)),
            hardness=float(d.get("hardness", 0.75)),
            wobble=float(d.get("wobble", 0.6)),
            cross_angle=d.get("cross_angle"),
            weave=d.get("weave", "twill"),
            depth=float(d.get("depth", 0.0)),
            poly=d.get("poly"),
            seed=d.get("seed"),
            selection=d.get("selection") or None,
            sel_invert=bool(d.get("sel_invert")),
            feather=float(d.get("feather", 0.0)),
            area=d.get("area", "brush"),
            record=bool(d.get("record", True)))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, strokes=int(n))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/scatter")
def scatter():
    """R53 perspective scatter: populate a region with vegetation, rocks,
    water ripples, or custom elements. {"layer", element: "grass"|
    "flowers"|"rocks"|"pebbles"|"reeds"|"ripples", or "custom": [unit-
    space strokes], region via "poly": [[x,y]..] (+"feather") -- the
    ATOMIC, race-free way -- or "selection"/"x","y","radius";
    "horizon" (screen y), "perspective" 0..1 (0 = top-down: fewer,
    uniform; 1 = across a field: many small far, few large near),
    "density", "size", "size_jitter", "color", "color2", "opacity",
    "lean", "wind", "depth", "seed"}. Returns {ok, strokes}."""
    d = request.json or {}
    try:
        kw = {}
        for k in ("horizon", "perspective", "density", "size",
                  "size_jitter", "opacity", "lean", "wind", "depth",
                  "feather", "radius"):
            if d.get(k) is not None:
                kw[k] = float(d[k])
        n = DOC.scatter_fill(
            d["layer"], x=d.get("x"), y=d.get("y"),
            element=d.get("element", "grass"),
            area=d.get("area", "selection" if (d.get("poly")
                                               or d.get("selection"))
                       else "brush"),
            selection=d.get("selection") or None,
            sel_invert=bool(d.get("sel_invert")),
            poly=d.get("poly"),
            color=tuple(d.get("color", (0.30, 0.40, 0.24))),
            color2=(tuple(d["color2"]) if d.get("color2") else None),
            custom=d.get("custom"),
            seed=d.get("seed"), record=bool(d.get("record", True)), **kw)
        GRAPH.commit_layer_outputs()
        return jsonify(ok=True, strokes=int(n))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.get("/api/textile/preview.png")
def textile_preview():
    """Live swatch for the textile tool (R50): renders the current thread/
    pattern settings on a scratch document with the REAL generators --
    what you see is exactly what a click will lay down. Query params:
    mode, weave, angle, spacing, thickness, depth, color (rrggbb hex),
    seed. Never touches the workspace or the journal."""
    import io as _io
    from PIL import Image as _Img
    q = request.args
    d = Document(190, 140)
    d.layers[0].pixels[..., :3] = np.float32(0.955)
    d.layers[0].pixels[..., 3] = 1.0
    l = d.add_layer("swatch")
    cx = q.get("color", "26243f")
    col = tuple(int(cx[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    depth = float(q.get("depth", 0))
    try:
        d.hatch_fill(l.id, 95, 70, radius=62,
                     mode=q.get("mode", "weave"),
                     weave=q.get("weave", "twill"),
                     angle=float(q.get("angle", 0)),
                     spacing=float(q.get("spacing", 7)),
                     thickness=float(q.get("thickness", 1.4)),
                     depth=depth, color=col,
                     opacity=float(q.get("opacity", 0.95)),
                     seed=int(q.get("seed", 7)), record=False)
    except Exception as e:
        return jsonify(error=str(e)), 400
    if depth > 0:
        l.relief = 0.5 + depth
    comp = np.clip(d.composite(), 0, 1)
    im = _Img.fromarray((comp[..., :3] * 255).astype(np.uint8))
    buf = _io.BytesIO()
    im.save(buf, "PNG")
    resp = app.response_class(buf.getvalue(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/pwarp")
def pwarp():
    """R47 perspective warp: {"layer", "quad": [[x,y] TL,TR,BR,BL],
    "bbox"?: [x0,y0,x1,y1] (default: selection bbox, else layer content),
    "selection"?, "sel_invert"?, "feather"?}. Cuts the (gated) source
    region and re-projects it so its corners land on the quad — a
    journaled pixel-free op, replayed by the same applier."""
    d = request.json or {}
    try:
        ok = DOC.warp_perspective(
            d["layer"], d["quad"], bbox=d.get("bbox"),
            selection=d.get("selection") or None,
            sel_invert=bool(d.get("sel_invert")),
            feather=float(d.get("feather", 0.0)),
            record=bool(d.get("record", True)))
        GRAPH.commit_layer_outputs()
        return jsonify(ok=bool(ok))
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


@app.post("/api/strokes/restyle")
def strokes_restyle():
    """Edit past strokes' recorded brush and re-render them.

    Body: {ids, media?: "oil"|"acrylic"|"water"|"none", material?: name or
    dict or "none", color?: [r,g,b], radius?, opacity?, load?, mix?}. The
    medium is PER STROKE (changing the brush panel never touches what is
    already painted); this is the one door for changing it after the fact."""
    d = request.json or {}
    ids = list(d.get("ids") or [])
    unknown = [s for s in ids if not any(k["id"] == s for k in DOC.strokes)]
    if unknown:
        # name the ids rather than KeyError-ing on the first: a stale panel
        # selection after an undo is the normal way to get here
        return jsonify(error="no such stroke(s): %s"
                       % ", ".join(unknown)), 400
    try:
        r = DOC.restyle_strokes(ids,
                                **{k: d.get(k) for k in DOC.RESTYLE_KEYS
                                   if d.get(k) is not None})
    except (ValueError, KeyError) as e:
        # ValueError carries the engine's own wording -- including the
        # faithfulness gate's "content that was not painted as strokes"
        return jsonify(error=str(e)), 400
    GRAPH.commit_layer_outputs()
    return jsonify(ok=True, **r)


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
                              record=bool(d.get("record", True)),
                              selection=d.get("selection"),
                              sel_invert=bool(d.get("sel_invert")))
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
                          extras=list(WS.extras) + asset_secs,
                          # R67: journal-first is the DEFAULT now -- a layer
                          # that can prove it replays does not ship its
                          # pixels. ?light=1 still forces the smallest file
                          # (every layer with a base, gate or no gate);
                          # ?pixels=1 forces the old fat file for a caller
                          # that wants pixels no matter what.
                          cache_pixels=(True if request.args.get("pixels")
                                        else (False if request.args.get("light")
                                              else None)),
                          # ...and the budget is a DIAL, because how much
                          # rebuild time a file may cost on Open is a
                          # judgement about the document, not a constant:
                          # ?budget=0 is the smallest file, a big number
                          # keeps opens instant. See Document._replay_first_set
                          replay_budget=_int_arg("budget"))
    _mark_saved()                # everything on disk: docs read clean now
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
    SYNC["viewing"].clear()      # every per-user doc choice is now stale
    for sec in ours:                                # uploaded models ride along
        register_asset(sec["meta"]["name"], sec["arrays"]["data"].tobytes(),
                       sec["meta"]["ext"], aid=sec.get("id"))
    WS._wire()


@app.post("/api/workspace/open")
def workspace_open():
    """Open a .lews workspace (multipart file). Replaces the current workspace."""
    data = request.files["file"].read()
    _load_workspace_bytes(data)
    # R57: the opened file replaces the LIVE workspace too, through the
    # engine's import (`Workspace.from_file`): every section carried
    # verbatim, one journal line recording where the content came from, so
    # a late-joining app can tell "opened a file" from "edited in place".
    try:
        ws = _lews_ws()
        if ws is not None:
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".lews")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                from holographic.io_and_interop.holographic_lews import Workspace
                _LEWS["ws"] = Workspace.from_file(tmp, _WS_ROOT,
                                                  app="lestudio")
                _LEWS["sha"].clear()      # hashes describe the OLD container
            finally:
                os.unlink(tmp)
    except Exception as e:
        app.logger.warning("lews import skipped: %s", e)
    return jsonify(ok=True)


_AUTOSAVE_PATH = os.path.expanduser("~/.lestudio_autosave.lews")


_AUTOSAVE_LAST_REV = [-1]


def _autosave_tick():
    """R65: the crash net used to live in the BROWSER PAGE (a setInterval
    that POSTs /api/autosave), so a session driven entirely by scripts and
    agents never autosaved at all -- restarting the server mid-round lost a
    whole painting, and it was only recoverable because the painters'
    scripts happened to be deterministic. The server now runs the timer
    itself: every 90 s, if anything has changed since the last write, the
    same sidecar is written whether or not a tab is open. The browser's own
    timer is left in place; two writers of the same file at 90 s cadence
    cost nothing and a second net catches what the first misses."""
    import threading
    def loop():
        while True:
            time.sleep(90)
            try:
                from . import _MUT_REV
                rev = _MUT_REV[0]
                if rev == _AUTOSAVE_LAST_REV[0]:
                    continue
                with app.test_request_context("/api/autosave", method="POST"):
                    r = autosave_write()
                _AUTOSAVE_LAST_REV[0] = rev
            except Exception:
                pass                 # the net must never take the server down
    t = threading.Thread(target=loop, name="lestudio-autosave", daemon=True)
    t.start()
    return t


@app.post("/api/autosave")
def autosave_write():
    """Write the whole workspace to a fixed sidecar file. The client calls
    this on a timer while there are unsaved changes; an explicit Save is
    still the person's own file -- this is just the crash net. Since R65
    the server runs the same timer itself (_autosave_tick), so a session
    with no browser tab open is covered too."""
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
    _mark_saved()                # the crash net holds this state: docs clean
    # R57: the live .lews Workspace is the CROSS-APP backing -- each document
    # goes in as its own section through the engine's locked, journalled
    # put(), so other apps holding this directory open see leStudio's
    # documents by rev, not by racing us for a sidecar file. The sidecar
    # above stays: it is the single-file crash net and needs no engine.
    # Best-effort by design (autosave already succeeded either way).
    lews_rev = _lews_publish()
    return jsonify(ok=True, bytes=len(data),
                   **({"lews_rev": lews_rev} if lews_rev else {}))


def _lews_publish():
    """Mirror WS into the live workspace directory: one 'lestudio.document'
    section per doc (unchanged docs skipped by content hash -- a put rewrites
    the whole container and journals a line, so no-op churn would spam every
    other app's change feed), one 'lestudio.state' section for the active id,
    and sections for docs closed since are deleted. Returns the last rev
    written, or None (nothing changed, or no engine Workspace)."""
    try:
        ws = _lews_ws()
        if ws is None:
            return None
        from . import _doc_section
        from holographic.io_and_interop.holographic_lews import section_hash
        rev = None
        for did, d in WS.docs.items():
            dm, arrays = _doc_section(d, WS.graphs.get(did))
            sec = {"kind": "lestudio.document", "id": did,
                   "meta": dm, "arrays": arrays}
            sha = section_hash(sec)
            if _LEWS["sha"].get(did) == sha:
                continue
            rev = ws.put(sec)
            _LEWS["sha"][did] = sha
        st = {"kind": "lestudio.state", "id": "lestudio-state",
              "meta": {"active": WS.active}, "arrays": {}}
        if _LEWS["sha"].get("__state") != WS.active:
            rev = ws.put(st)
            _LEWS["sha"]["__state"] = WS.active
        for sec in ws.sections("lestudio.document", upgrade=False):
            if sec.get("id") not in WS.docs:
                rev = ws.delete(sec["id"])
                _LEWS["sha"].pop(sec.get("id"), None)
        return rev
    except Exception as e:
        app.logger.warning("lews publish skipped: %s", e)
        return None


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


@app.get("/api/layer/<lid>/below.png")
def layer_below_png(lid):
    """Everything UNDER this layer, composited: the surface you are actually
    painting on. R65, dogfooded: a painter had no way to see what was
    beneath her own layer without rendering the whole document -- which
    includes the layers ABOVE hers -- so she sampled the full composite to
    repair a halo and painted a flat, wall-coloured rectangle onto her
    OBJECT layer. That rectangle is the artifact Devin saw behind the bowl.
    Two of her five fix rounds existed only to work around this."""
    from . import composite
    with _DOC_LOCK:
        DOC.layer(lid)                              # 404 the honest way
        ls = DOC.canvas_layers()
        idx = next((i for i, l in enumerate(ls) if l.id == lid), None)
        if idx is None:
            return jsonify(error="that layer is not on the canvas"), 400
        return _png(composite(ls[:idx], DOC.height, DOC.width, DOC.mask_map()))


@app.get("/api/layer/<lid>/above.png")
def layer_above_png(lid):
    """Everything OVER this layer, composited -- what will be drawn on top
    of anything you paint here. The other half of below.png: a highlight
    that lands under someone's glaze is not the highlight you painted."""
    from . import composite
    with _DOC_LOCK:
        DOC.layer(lid)
        ls = DOC.canvas_layers()
        idx = next((i for i, l in enumerate(ls) if l.id == lid), None)
        if idx is None:
            return jsonify(error="that layer is not on the canvas"), 400
        return _png(composite(ls[idx + 1:], DOC.height, DOC.width, DOC.mask_map()))


@app.post("/api/layer")
def layer_edit():
    """Layer ops: {"action": "add"|"delete"|"edit"|"duplicate"|"merge_down"|"move"|"claim"|"release"|"grant"|"revoke", "id"?, plus edit props: name, visible, opacity, blend, mask, mask_invert, alpha_lock, clip}. claim/release take just {id}; grant/revoke take {id, to} and are owner-only -- see CONTRACT.md section 1-2."""
    d = request.json or {}
    act = d.get("action")
    # ONE guard for every id-taking action. Before this, each action failed
    # its own way: delete of an unknown id silently no-opped (after burning
    # an undo snapshot), move KeyError'd to a 500, edit had a private check.
    if act in ("duplicate", "merge_down", "clear", "remove", "delete",
               "fill", "flip", "move", "edit",
               "claim", "release", "grant", "revoke"):
        lid = d.get("id")
        if not lid:
            return jsonify(error="which layer? '%s' needs a layer id"
                                 % act), 400
        try:
            DOC.layer(lid)
        except KeyError:
            return jsonify(error="no such layer: %s - pick another in the "
                                 "layer list" % lid), 400
    if act == "move" and d.get("index") is None:
        return jsonify(error="move needs an index (0 = bottom of the "
                             "stack)"), 400
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
        reg = d.get("region")
        if reg:
            # R65: a plain rectangle, for a painter who needs to take a
            # mistake OUT rather than paint over it. Painting over is how
            # the flat patch behind the bowl happened; nobody erases when
            # erasing needs a selection object first.
            try:
                x0, y0, x1, y1 = [int(v) for v in reg]
            except Exception:
                return jsonify(error="region must be [x0, y0, x1, y1]"), 400
            x0, x1 = max(0, min(x0, x1)), min(DOC.width, max(x0, x1))
            y0, y1 = max(0, min(y0, y1)), min(DOC.height, max(y0, y1))
            if x1 <= x0 or y1 <= y0:
                return jsonify(error="empty region"), 400
            DOC.clear_region(d["id"], x0, y0, x1, y1)   # journal-first
        else:
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
                                  "media_time", "paint_gloss")}
        if "mask" in d:
            props["mask"] = d.get("mask")
        if "bg" in d:                    # None -> transparent sheet
            props["bg"] = d.get("bg")
        # id existence is covered by the shared guard above
        try:
            DOC.edit_layer(d["id"], **props)
        except ValueError as e:
            # a string opacity used to return ok, poison every composite
            # with a 500 AND save into the .lews -- the model now validates
            # and this surfaces its (helpful) message as a client error
            return jsonify(error=str(e)), 400
        GRAPH.commit_layer_outputs()
        # WHY DIDN'T THAT GLINT? "Water reflection" and "Dispersion fringes"
        # (layer Actions ▾) dial `reflect`/`dispersion`, which every layer
        # model carries -- but the renderer only ever READS them off a
        # WATER/GLASS volume slab (composite_volumetric's refraction
        # branch), and even there only paints them in the Ortho/Persp 3D
        # view; the Flat composite (composite_cached / composite_lit's
        # flat branch) never calls that code at all. Dialling either up on
        # a `Flat paint (no volume)` layer -- or on a real water/glass slab
        # while still in Flat view -- reported ok:True over a pixel-
        # identical composite (checked both ways with a real render diff).
        # Emissive glow (vol_glow) is NOT in this boat: it lights the Flat
        # composite directly (_doc_emission), so it needs no message here.
        _refl_on = d.get("reflect") not in (None, 0, 0.0)
        _disp_on = d.get("dispersion") not in (None, 0, 0.0)
        if _refl_on or _disp_on:
            _labels = [n for n, on in (("water reflection", _refl_on),
                                       ("dispersion", _disp_on)) if on]
            _what = " and ".join(_labels)
            _is, _do = (("is a property", "renders") if len(_labels) == 1
                        else ("are properties", "render"))
            _l = DOC.layer(d["id"])
            _kind = getattr(_l, "vol_kind", "none")
            _optics_kind = {"inkwater": "water", "smoke": "fog",
                            "fire": "fog"}.get(_kind, _kind)
            warn = None
            if _optics_kind not in ("water", "glass"):
                warn = ("%s %s of a WATER or GLASS slab, and this layer's "
                        "Type is %s -- set Type to Water or Glass (Layer "
                        "options ▸ Optics & material) first, then it "
                        "does something"
                        % (_what[0].upper() + _what[1:], _is,
                           "'Flat paint (no volume)'" if _kind == "none"
                           else repr(_kind)))
            elif getattr(DOC, "view3d", "flat") == "flat":
                warn = ("done -- but %s only %s in Ortho/Persp 3D, not "
                        "Flat; the Flat view has no waterline to show it "
                        "on -- switch the camera (View ▸ Ortho/Persp) "
                        "to see it" % (_what, _do))
            if warn is not None:
                return jsonify(ok=True, warning=warn)
    elif act == "claim":
        # The ownership GATE (before_request, above) already ran the full
        # CONTRACT.md permission table for this id and 403'd if it refused
        # -- an unowned or stale-owned layer was just auto-claimed/yielded
        # to `me` as a SIDE EFFECT of that check passing, and _layer_yield_
        # notice already folded `yielded_from` into this response if a
        # stale lock just yielded. Reaching here at all means claim already
        # happened; there is nothing left to do but say so.
        return jsonify(ok=True, owner=DOC.layer(d["id"]).owner)
    elif act == "release":
        # Deliberately NOT run through the generic gate (see
        # _mutating_layer_targets): "release" means "give up MY OWN lock",
        # which is the opposite of what the gate's unowned-auto-claim would
        # do if it ran on a layer that turns out to already be unowned.
        l = DOC.layer(d["id"])
        me = _owner_uid()
        owner = getattr(l, "owner", "") or ""
        if owner == "":
            return jsonify(ok=True)          # nothing to release, no-op
        if owner != me:
            return jsonify(_layer_refusal(d["id"], l)), 403
        l.owner = ""
        return jsonify(ok=True)
    elif act in ("grant", "revoke"):
        # OWNER ONLY -- deliberately stricter than the generic mutate-layer
        # permission (which also admits `shared` collaborators): letting a
        # collaborator you shared with re-grant or revoke OTHER people would
        # turn "I trust Priya with this layer" into "I trust Priya with who
        # else gets to touch it", which is not what a grant means.
        to = str(d.get("to", "")).strip()
        if not to:
            return jsonify(error="grant/revoke needs 'to' -- whose access "
                                 "are you changing?"), 400
        l = DOC.layer(d["id"])
        me = _owner_uid()
        owner = getattr(l, "owner", "") or ""
        if owner == "":
            return jsonify(error="that layer has no owner yet -- claim it "
                                 "first, then you can share it"), 400
        if owner != me:
            return jsonify(_layer_refusal(d["id"], l)), 403
        shared = list(getattr(l, "shared", None) or [])
        if act == "grant":
            if to not in shared:
                shared.append(to)
        else:
            shared = [u for u in shared if u != to]
        l.shared = shared
        return jsonify(ok=True, shared=shared)
    elif act:
        return jsonify(error="unknown layer action: %r" % act), 400
    return jsonify(ok=True)


# R63: pending access requests live in server MEMORY, keyed per document id
# -- not in the .lews. A request is a conversation about the picture, not
# part of it: saving/loading a file must not resurrect "may I paint on your
# sky?" from a session that ended, and a request naming a uid who never
# reconnects should just fade with the process rather than haunt a reopened
# document forever. Keyed per document so switching pictures doesn't leak
# one picture's inbox into another's.
ACCESS_REQUESTS = {}
_ACCESS_SEQ = [0]


# R65: named passes already painted, per (document, user, pass). Session
# memory like ACCESS_REQUESTS: a pass name is a fact about one painting
# session, not about the picture.
PASSES_SEEN = {}


def _access_bucket():
    return ACCESS_REQUESTS.setdefault(_viewing_doc_id(), [])


@app.get("/api/access")
def access_list_get():
    """R64 (dogfooded): reading your own requests used to require
    POST {"action":"list"} and a plain GET answered 405, which is the wrong
    answer to the most obvious thing anyone tries. Same payload, no verb."""
    me = _owner_uid()
    reqs = _access_bucket()
    return jsonify(ok=True,
                   incoming=[_access_view(r) for r in reqs if r["to"] == me],
                   outgoing=[_access_view(r) for r in reqs if r["from"] == me])


def _access_view(r):
    """A request as the client should see it: `from_name`/`to_name` are
    resolved NOW, not stamped when it was filed. R64 -- a painter who
    registered a display name after asking stayed "u_devi" in their own
    pending request forever."""
    return dict(r, from_name=_display_name(r["from"]) or r["from"],
                to_name=_display_name(r["to"]) or r["to"])


@app.post("/api/access")
def access_route():
    """The access-request flow (CONTRACT.md section 2):
    {"action":"request","layer","note"?} -> {ok,id} (requester -> owner)
    {"action":"list"} -> {ok,incoming:[...],outgoing:[...]} for the caller
    {"action":"grant","id"} -> owner answers yes (adds requester to shared)
    {"action":"deny","id","reason"?} -> owner answers no
    Deliberately excluded from the ownership gate (_GATE_EXCLUDED_PATHS):
    asking for access must never itself require access."""
    d = request.json or {}
    act = d.get("action")
    me = _owner_uid()    # requests are about real identities, same as ownership
    reqs = _access_bucket()
    if act == "request":
        lid = d.get("layer")
        if not lid:
            return jsonify(error="which layer are you asking for?"), 400
        try:
            l = DOC.layer(lid)
        except KeyError:
            return jsonify(error="no such layer: %s" % lid), 400
        owner = getattr(l, "owner", "") or ""
        if not owner:
            return jsonify(error="that layer has no owner -- just paint on "
                                 "it, no need to ask"), 400
        if owner == me or me in (getattr(l, "shared", None) or []):
            return jsonify(error="you already have access to that layer"), 400
        _ACCESS_SEQ[0] += 1
        rid = "AR%d" % _ACCESS_SEQ[0]
        reqs.append({"id": rid, "layer": lid, "layer_name": l.name,
                     "from": me, "to": owner,
                     "note": str(d.get("note") or "")[:280],
                     "state": "pending", "asked_at": time.time(),
                     "answered_at": None})
        return jsonify(ok=True, id=rid)
    if act == "list":
        return jsonify(ok=True,
                       incoming=[_access_view(r) for r in reqs if r["to"] == me],
                       outgoing=[_access_view(r) for r in reqs if r["from"] == me])
    if act in ("grant", "deny"):
        rid = d.get("id")
        ent = next((r for r in reqs if r["id"] == rid), None)
        if ent is None:
            return jsonify(error="no such request (it may already have "
                                 "been answered)"), 404
        if ent["to"] != me:
            return jsonify(error="that request is not addressed to you"), 403
        if ent["state"] != "pending":
            return jsonify(error="that request was already %s"
                                 % ent["state"]), 400
        if act == "grant":
            try:
                l = DOC.layer(ent["layer"])
            except KeyError:
                return jsonify(error="that layer no longer exists"), 400
            shared = list(getattr(l, "shared", None) or [])
            if ent["from"] not in shared:
                shared.append(ent["from"])
            l.shared = shared
        else:
            if d.get("reason"):
                ent["reason"] = str(d["reason"])[:280]
        ent["state"] = "granted" if act == "grant" else "denied"
        ent["answered_at"] = time.time()
        return jsonify(ok=True)
    return jsonify(error="unknown action: %r" % act), 400


@app.post("/api/group")
def group_edit():
    """Layer group ops: {"action": "add"|"delete"|"edit"|"assign", ...}."""
    d = request.json or {}
    act = d.get("action")
    if act in ("remove", "edit"):
        gid = d.get("id")
        if not gid or not any(g["id"] == gid for g in DOC.groups):
            return jsonify(error="no such group: %s" % gid), 400
    if act == "add":
        want = d.get("layers", []) or []
        known = {l.id for l in DOC.layers}
        if want and not any(x in known for x in want):
            # a group of only unknown layer ids is an empty group nobody
            # asked for -- the ids are stale (undo, another user's delete)
            return jsonify(error="none of those layers exist any more: %s"
                                 % ", ".join(map(str, want))), 400
        g = DOC.add_group(d.get("name"), want)
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
    # same shared guard as /api/layer: unknown ids answered plainly, not 500
    if act in ("duplicate", "remove", "edit", "move"):
        mid = d.get("id")
        if not mid:
            return jsonify(error="which mask? '%s' needs a mask id"
                                 % act), 400
        try:
            DOC.mask_by_id(mid)
        except KeyError:
            return jsonify(error="no such mask: %s" % mid), 400
    if act == "move" and d.get("index") is None:
        return jsonify(error="move needs an index"), 400
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
        meta = sel.meta()
        # WHY DIDN'T THAT SELECT ANYTHING? object/wand/lum can silently
        # produce an essentially empty mask: clicking bare paper with the
        # object selector (no segment under the point covers the click), or
        # a wand/lum tolerance too tight to match even the antialiased
        # pixels next door. The selection is still made -- naming and undo
        # history stay honest -- it is just told plainly there is nothing
        # here to paint into (R58 UX open #3).
        coverage = float(sel.data.mean())
        meta["coverage"] = coverage
        warn = None
        if d.get("tool") == "object" and coverage < 1e-4:
            warn = ("that selected nothing — no shape was found under the "
                    "click; try a spot with a clearer edge")
        elif d.get("tool") in ("color", "brightness") and coverage < 1e-4:
            warn = ("that matched almost nothing — the tolerance is too "
                    "tight for this area; raise it and try again")
        resp = dict(ok=True, selection=meta)
        if warn is not None:
            resp["warning"] = warn
        return jsonify(**resp)
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
    # R58: strokes are SERIALISED against composites. This route mutated the
    # document with no lock while GET /api/composite.png rendered under one;
    # a stroke landing mid-render left every cache stamped current over
    # pre-stroke pixels, and the person saw one dab where they had dragged
    # a whole line. RLock: nested takes by helpers stay fine.
    with _DOC_LOCK:
        return _paint_locked()


def _paint_locked():
    d = request.json or {}
    busy = _still_busy(d)          # R64: the cadence promise, made atomic
    if busy is not None:
        return busy
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
        if warn is None and mode == "knife" and pts:
            # R58 UX sweep: the palette knife shapes paint BODY. On flat paint
            # (no impasto under the path) it is a silent no-op -- the drag
            # does nothing and nothing says why.
            _hm = getattr(_l, "height_map", None)
            if _hm is None or float(_hm[y0:y1, x0:x1].max(initial=0.0)) < 1e-4:
                warn = ("the knife shapes paint BODY, and there is none here — "
                        "paint with an oil/acrylic/water medium first (Brush ▸ "
                        "Media), then knife it")
        if warn is None:
            # STROKES UNDER OPAQUE PAINT (R11, found swarm-painting a
            # portrait): a fix painted on a layer BELOW the flaw looks like
            # a silent no-op — the stroke succeeds, the picture does not
            # change, and nothing says why. If some visible, normal-blend
            # layer above covers the whole stroke area opaquely, say so.
            _idx = next((i for i, x in enumerate(DOC.layers)
                         if x.id == _l.id), -1)
            for _up in DOC.layers[_idx + 1:]:
                if (not _up.visible
                        or float(getattr(_up, "opacity", 1.0)) < 0.98
                        or str(getattr(_up, "blend", "normal")) != "normal"):
                    continue        # translucent / blended layers show through
                _a = _up.pixels[y0:y1:4, x0:x1:4, 3]
                if _a.size and float(_a.min(initial=1.0)) > 0.92:
                    warn = ("that landed UNDER %r, which covers this area "
                            "opaquely — the stroke is saved but hidden; "
                            "paint on %r or a layer above it to see it"
                            % (_up.name, _up.name))
                    break
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
                  stroke_new=bool(d.get("record", True)),
                  selection=d.get("selection"),
                  sel_invert=bool(d.get("sel_invert")))
    elif mode == "blend":
        # the BLENDER: no pigment, works the wet paint already there. Unlike
        # smudge this is a recorded stroke, so the layer keeps stroke editing
        DOC.blend_stroke(d["layer"], d["points"],
                         radius=float(d.get("radius", 18)),
                         strength=float(d.get("opacity", 0.6)),
                         brush=d.get("brush"),
                         record=bool(d.get("record", True)),
                         stroke_new=bool(d.get("record", True)),
                         selection=d.get("selection"),
                         sel_invert=bool(d.get("sel_invert")))
    elif mode == "smudge":
        DOC.smudge(d["layer"], d["points"], radius=float(d.get("radius", 12)),
                   strength=float(d.get("opacity", 0.6)), brush=d.get("brush"),
                   record=bool(d.get("record", True)),
                   selection=d.get("selection"),
                   sel_invert=bool(d.get("sel_invert")))
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
                 record=bool(d.get("record", True)),
                 selection=d.get("selection"),
                 sel_invert=bool(d.get("sel_invert")))
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


# --------------------------------------------------------------------------
# R17: the leCore bridge -- style transfer, dream seeds, the studio sage,
# and a GLSL export of the grade chain. All lazy: leStudio runs fine
# without leCore on the path; these endpoints then answer 503 honestly.

def composite_png_array():
    """The flattened picture as float RGBA (white ground), shared by the
    leCore bridge endpoints."""
    c = DOC.composite()
    flat = c[..., :3] * c[..., 3:4] + 1.0 * (1 - c[..., 3:4])
    return np.dstack([np.clip(flat, 0, 1).astype(np.float32),
                      np.ones(flat.shape[:2], np.float32)])


_LECORE = {"tried": False}
_LECORE_LOCK = threading.Lock()


def _lecore():
    """Import the leCore pieces once; None (with a reason) if unavailable."""
    with _LECORE_LOCK:
        if not _LECORE["tried"]:
            _LECORE["tried"] = True
            try:
                from holographic.materials_and_texture.holographic_colortransfer \
                    import color_transfer
                from holographic.rendering.holographic_postfx import chain_to_glsl
                from holographic.sampling_and_signal.holographic_hdrift \
                    import train_image_drift, generate_images
                _LECORE.update(ct=color_transfer, glsl=chain_to_glsl,
                               train=train_image_drift, gen=generate_images)
            except Exception as e:                      # not installed: honest 503
                _LECORE["err"] = str(e)
        return _LECORE


def _sage():
    """The studio's own leCore mind, sharing the SAME memory store the
    painting sessions teach ('lecore_memory') -- so every law learned while
    dogfooding is on tap for agents (/api/advise) and the Sage panel."""
    with _LECORE_LOCK:
        if "mind" not in _LECORE and "mind_err" not in _LECORE:
            try:
                import os as _os
                from lecore import autoboot
                # the store is found by PARTITION (an absolute path wins over
                # whatever directory the server happens to run from): env,
                # then ./lecore_memory, then the house partition
                part = _os.environ.get("LECORE_PARTITION")
                if not part:
                    for cand in ("lecore_memory", "/root/work/lecore_memory"):
                        if _os.path.isdir(cand):
                            part = _os.path.abspath(cand)
                            break
                _LECORE["mind"] = autoboot(partition=part)
                _LECORE["mind_part"] = part
            except Exception as e:
                _LECORE["mind_err"] = str(e)
        return _LECORE.get("mind")


def _ref_image_from(d):
    """A reference image as float RGB HxWx3 from one of: image_b64 (data
    URL or raw base64 PNG), path (server-side file), layer (a layer id)."""
    import base64 as _b64
    import io as _io
    from PIL import Image as _PImage
    if d.get("image_b64"):
        raw = _b64.b64decode(d["image_b64"].split(",")[-1])
        im = _PImage.open(_io.BytesIO(raw)).convert("RGB")
        return np.asarray(im, np.float32) / 255.0
    if d.get("path"):
        im = _PImage.open(d["path"]).convert("RGB")
        return np.asarray(im, np.float32) / 255.0
    if d.get("layer"):
        l = DOC.layer(d["layer"])
        a = l.pixels
        return (a[..., :3] * a[..., 3:4]).astype(np.float32)
    raise ValueError("give a reference: image_b64, path, or layer")


@app.post("/api/style/match")
def style_match():
    """MATCH THE MOOD: grade this painting toward a reference image's colour
    statistics (leCore colour transfer, Reinhard/covariance) and land the
    result as a NEW top layer, so the original stays untouched underneath.
    {"image_b64"|"path"|"layer", "mode": "covariance"|"meanstd",
    "strength": 0..1, "name"?}. Moves colour, never content."""
    lc = _lecore()
    if "ct" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    d = request.json or {}
    try:
        ref = _ref_image_from(d)
    except (ValueError, KeyError, OSError) as e:
        return jsonify(error=str(e)), 400
    comp = composite_png_array()
    graded = lc["ct"](comp[..., :3], ref,
                      mode=str(d.get("mode", "covariance")),
                      strength=float(d.get("strength", 1.0)))
    out = np.dstack([np.clip(graded, 0, 1).astype(np.float32),
                     np.ones(graded.shape[:2], np.float32)])
    l = DOC.add_layer(d.get("name") or "Match",
                      pixels=_gated_output(out, d), asset=True)
    return jsonify(ok=True, id=l.id)


_DREAMS = []
_DREAM_PTS = []        # raw-space feature vectors of the last batch (R21)


def _taste_path():
    import os as _os
    part = _LECORE.get("mind_part") or "/root/work/lecore_memory"
    return _os.path.join(part, "lestudio_taste.json")


def _taste_load():
    import json as _json
    import os as _os
    try:
        if _os.path.exists(_taste_path()):
            return _json.load(open(_taste_path()))
    except Exception:
        pass
    return []


def _taste_save(vecs):
    import json as _json
    try:
        _json.dump(vecs, open(_taste_path(), "w"))
    except Exception:
        pass                                   # taste is best-effort


@app.post("/api/dream/fave")
def dream_fave():
    """TASTE (R21): star dream #i from the last /api/dream. The dream's
    raw splat-feature vector is recorded as a SUCCESS (the semantic-compass
    pattern: remember what worked, bias future candidates toward it) and
    persisted with the leCore partition, so future /api/dream batches lean
    toward the starred look -- across sessions, for humans and agents
    alike. {"i"}."""
    d = request.json or {}
    try:
        vec = _DREAM_PTS[int(d.get("i", 0))]
    except (IndexError, ValueError, TypeError):
        return jsonify(error="no such dream -- run /api/dream first"), 400
    taste = _taste_load()
    taste.append([round(float(v), 5) for v in vec])
    _taste_save(taste[-64:])                   # keep the last 64 stars
    return jsonify(ok=True, stars=len(taste))



def _splat_code(img, K):
    """The compact splat code of a color image (R18): K luminance-placed
    Gaussians (matching pursuit + joint refit) with PER-CHANNEL amplitudes
    solved jointly against the shared basis. ~7 floats per splat instead of
    a pixel buffer -- Devin's 'generate instead of store': the code is small
    enough to pass around, and rendering it is deterministic anywhere.

    R20: placement is COLOUR-AWARE adaptively. Luminance placement is blind
    to passages where colour differs at equal brightness, so after the
    first solve a share of the budget goes to splats placed on the COLOUR
    residual -- sized by how much of the remaining error actually lives in colour
    (measured: +1.6 dB on a portrait, and the adaptive share avoids the
    -0.5 dB a fixed share cost on a luminance-dominant abstract).
    Returns (splats [(cy,cx,amp,sigma)...], A (K,3) channel amps)."""
    from holographic.rendering.holographic_splat import splat_fit, _gaussian
    K = int(K)
    lum = img.mean(-1).astype(float)
    H, W = lum.shape

    def basis(ss):
        return np.stack([_gaussian((H, W), cy, cx, sg).ravel()
                         for (cy, cx, _, sg) in ss], axis=1)

    k1 = max(8, int(K * 0.8))
    s = splat_fit(lum, k1, refit=True)
    G = basis(s)
    A = np.linalg.lstsq(G, img.reshape(-1, 3), rcond=None)[0]
    if K > k1:
        R = img.reshape(-1, 3) - G @ A
        # how much of the remaining error is COLOUR (not luminance)?
        lum_res = np.abs(R.mean(1)).sum()
        col_res = np.abs(R - R.mean(1, keepdims=True)).sum()
        share = float(col_res / max(col_res + lum_res, 1e-9))
        k_col = int(round((K - k1) * min(1.0, share * 1.6)))
        extra = []
        if k_col > 0:
            rfield = np.abs(R).max(1).reshape(H, W)
            extra += splat_fit(rfield, k_col, refit=False)
        if (K - k1 - k_col) > 0:
            lfield = np.abs(R.mean(1)).reshape(H, W)
            extra += splat_fit(lfield, K - k1 - k_col, refit=False)
        s = s + extra
        G = basis(s)
        A = np.linalg.lstsq(G, img.reshape(-1, 3), rcond=None)[0]
    return s, A


def _splat_render_color(splats, A, shape):
    from holographic.rendering.holographic_splat import _gaussian
    out = np.zeros(shape + (3,))
    for j, (cy, cx, _, sg) in enumerate(splats):
        out += _gaussian(shape, cy, cx, sg)[..., None] * A[j][None, None, :]
    return np.clip(out, 0, 1)




def _gated_output(out, d):
    """Phase G3: an active selection FOCUSES a generated layer -- output
    alpha multiplies by the gate, so generation lands only inside the
    boundary the artist chose (and the baked asset is born gated)."""
    sel = d.get("selection")
    if not sel:
        return out
    g = DOC._resolve_gate(sel, bool(d.get("sel_invert")),
                          feather=float(d.get("sel_feather", 0.0)))
    if g is None:
        return out
    out = out.copy()
    out[..., 3] = out[..., 3] * g
    return out

@app.post("/api/splatify")
def splatify():
    """SPLATIFY (R18): re-render this painting as K colour Gaussian splats
    -- an abstraction dial (K=32 pointillist mood, K=160 dreamy, K=400
    mosaic-faithful; audited leCore splat suite, matching pursuit + joint
    per-channel refit). Lands as a NEW top layer and returns the COMPACT
    SPLAT CODE ({splats, colors}, ~7 floats each), which regenerates the
    layer deterministically anywhere -- pass the code, not the pixels.
    {"k": 16..600, "name"?, "opacity"?}."""
    lc = _lecore()
    if "ct" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    from PIL import Image as _PImage
    d = request.json or {}
    K = max(8, min(600, int(d.get("k", 160))))
    comp = composite_png_array()[..., :3]
    sw = 180
    sh = max(2, int(round(DOC.height * sw / max(DOC.width, 1))))
    small = np.asarray(_PImage.fromarray(
        (comp * 255).astype(np.uint8)).resize((sw, sh)), np.float32) / 255.0
    splats, A = _splat_code(small.astype(float), K)
    r = _splat_render_color(splats, A, (sh, sw))
    up = np.asarray(_PImage.fromarray((r * 255).astype(np.uint8)).resize(
        (DOC.width, DOC.height), _PImage.LANCZOS), np.float32) / 255.0
    out = np.dstack([up, np.ones(up.shape[:2], np.float32)])
    l = DOC.add_layer(d.get("name") or ("Splats x%d" % K),
                      pixels=_gated_output(out, d), asset=True)
    if d.get("opacity") is not None:
        DOC.edit_layer(l.id, opacity=float(d["opacity"]))
    return jsonify(ok=True, id=l.id, k=K, code={
        "shape": [sh, sw],
        "splats": [[round(float(cy), 3), round(float(cx), 3),
                    round(float(sg), 3)] for (cy, cx, _, sg) in splats],
        "colors": [[round(float(v), 5) for v in row] for row in A]})


@app.post("/api/splats/morph")
def splats_morph():
    """SPLAT MORPH (R20): the painting flows into a reference image and
    back, as Gaussian splats in motion. Both pictures become K-splat codes
    at the same grid; splats are matched greedily (nearest centre, largest
    energies first) and interpolated with an ease curve; every frame is
    the same closed form the browser parity-renders, so the whole
    animation is DETERMINISTIC and could be regenerated client-side from
    the two codes alone. {"image_b64"|"path", "k"?, "frames"?, "fps"?,
    "boomerang"?: true}. Returns the GIF."""
    lc = _lecore()
    if "ct" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    from PIL import Image as _PImage
    d = request.json or {}
    K = max(16, min(300, int(d.get("k", 120))))
    frames = max(6, min(72, int(d.get("frames", 30))))
    fps = max(4, min(30, int(d.get("fps", 14))))
    try:
        ref = _ref_image_from(d)
    except (ValueError, KeyError, OSError) as e:
        return jsonify(error=str(e)), 400
    sw = 200
    sh = max(2, int(round(DOC.height * sw / max(DOC.width, 1))))
    comp = composite_png_array()[..., :3]
    a_img = np.asarray(_PImage.fromarray((comp * 255).astype(np.uint8))
                       .resize((sw, sh)), np.float32) / 255.0
    b_img = np.asarray(_PImage.fromarray(
        (np.clip(ref, 0, 1) * 255).astype(np.uint8)).resize((sw, sh)),
        np.float32) / 255.0
    sa, Aa = _splat_code(a_img.astype(float), K)
    sb, Ab = _splat_code(b_img.astype(float), K)
    # peak-space parameters (the closed form both ends share)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(float)

    def peaks(ss, A):
        out = []
        for j, (cy, cx, _, sg) in enumerate(ss):
            g = np.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / (sg * sg))
            nrm = float(np.sqrt((g * g).sum())) + 1e-12
            out.append((float(cy), float(cx), float(sg), A[j] / nrm))
        return out
    pa, pb = peaks(sa, Aa), peaks(sb, Ab)
    # match: biggest energies first, each takes its nearest unused partner
    order = np.argsort([-float(np.abs(p[3]).sum()) for p in pa])
    usedb = np.zeros(len(pb), bool)
    pairs = []
    for i in order:
        cy, cx = pa[i][0], pa[i][1]
        dists = [((pb[j][0] - cy) ** 2 + (pb[j][1] - cx) ** 2)
                 if not usedb[j] else 1e18 for j in range(len(pb))]
        j = int(np.argmin(dists))
        usedb[j] = True
        pairs.append((pa[i], pb[j]))
    seq = list(range(frames))
    if d.get("boomerang", True):
        seq = seq + seq[-2:0:-1]
    ims = []
    for f in seq:
        t = f / max(frames - 1, 1)
        t = t * t * (3 - 2 * t)                       # smoothstep ease
        out = np.zeros((sh, sw, 3))
        for (a, b) in pairs:
            cy = a[0] + (b[0] - a[0]) * t
            cx = a[1] + (b[1] - a[1]) * t
            sg = max(a[2] + (b[2] - a[2]) * t, 0.5)
            col = a[3] + (b[3] - a[3]) * t
            e = np.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / (sg * sg))
            out += e[..., None] * col[None, None, :]
        ims.append(_PImage.fromarray(
            (np.clip(out, 0, 1) * 255).astype(np.uint8)).resize(
            (sw * 2, sh * 2), _PImage.LANCZOS))
    import io as _io2
    buf = _io2.BytesIO()
    ims[0].save(buf, "GIF", save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)
    from flask import Response
    return Response(buf.getvalue(), mimetype="image/gif")


@app.get("/api/splatify/code")
def splatify_code():
    """The COMPACT SPLAT CODE of the current painting, no layer created
    (R19): ?k= splats (default 160). Alongside the unit-norm-basis
    `colors`, the response carries `peak` colours -- closed-form
    per-splat peak amplitudes, so ANY client can regenerate the picture
    with pixel(x,y) = sum_j peak_j * exp(-0.5*((x-cx)^2+(y-cy)^2)/sigma^2)
    at `shape` resolution: the front end renders exactly what the back end
    authored, and only the ~4 KB code crosses the wire."""
    lc = _lecore()
    if "ct" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    from PIL import Image as _PImage
    K = max(8, min(600, int(request.args.get("k", 160))))
    comp = composite_png_array()[..., :3]
    sw = 180
    sh = max(2, int(round(DOC.height * sw / max(DOC.width, 1))))
    small = np.asarray(_PImage.fromarray(
        (comp * 255).astype(np.uint8)).resize((sw, sh)), np.float32) / 255.0
    splats, A = _splat_code(small.astype(float), K)
    # peak-space colours: A is per unit-L2-norm basis; the closed form a
    # client evaluates has peak 1, so divide by the numeric L2 norm of the
    # RAW gaussian (edge-truncated splats included -- computed, not the
    # sqrt(pi sigma^2) interior approximation)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(float)
    peak = []
    for j, (cy, cx, _, sg) in enumerate(splats):
        g = np.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / (sg * sg))
        nrm = float(np.sqrt((g * g).sum())) + 1e-12
        peak.append([float(v) / nrm for v in A[j]])
    return jsonify(ok=True, k=K, code={
        "shape": [sh, sw],
        "splats": [[round(float(cy), 4), round(float(cx), 4),
                    round(float(sg), 4)] for (cy, cx, _, sg) in splats],
        "colors": [[round(float(v), 6) for v in row] for row in A],
        "peak": [[round(float(v), 8) for v in row] for row in peak]})


def _splats_export_layers(K):
    """R22: DEPTH-AWARE splat export. Each visible painted layer becomes
    its own splat code (fit on its alpha-weighted content), lifted to that
    pane's real depth (accumulated thickness + z_off, scaled by ?z_scale,
    default 10) -- a layered glass painting leaves as a TRUE 3D splat
    sculpture: parallax between panes survives in any 3DGS viewer.
    ?fmt=json returns three.js records with per-splat z instead."""
    import tempfile
    from PIL import Image as _PImage
    m = _sage()
    if m is None:
        return jsonify(error="sage unavailable: %s"
                             % _LECORE.get("mind_err", "")), 503
    try:
        z_scale = float(request.args.get("z_scale", 10.0))
    except ValueError:
        return jsonify(error="z_scale must be a number"), 400
    sw = 180
    sh = max(2, int(round(DOC.height * sw / max(DOC.width, 1))))
    painted = [l for l in DOC.layers
               if l.visible and float(l.pixels[..., 3].max()) > 0]
    if not painted:
        return jsonify(error="nothing painted to export"), 400
    per = max(12, K // len(painted))
    records, colors = [], []
    depth = 0.0
    from holographic.rendering.holographic_splat import splat_fit, _gaussian
    for l in painted:
        z = (depth + float(getattr(l, "z_off", 0.0))) * z_scale
        depth += float(getattr(l, "thickness", 1.0) or 1.0)
        rgba = np.asarray(_PImage.fromarray(
            (np.clip(l.pixels, 0, 1) * 255).astype(np.uint8)).resize(
            (sw, sh)), np.float32) / 255.0
        a = rgba[..., 3]
        if float(a.max()) <= 0:
            continue
        field = (rgba[..., :3].mean(-1) * a).astype(float)
        s = splat_fit(field, per, refit=True)
        G = np.stack([_gaussian((sh, sw), cy, cx, sg).ravel()
                      for (cy, cx, _, sg) in s], axis=1)
        A = np.linalg.lstsq(G, (rgba[..., :3] * a[..., None]
                                ).reshape(-1, 3), rcond=None)[0]
        cols = np.clip(np.abs(A) / (np.abs(A).max(axis=1, keepdims=True)
                                    + 1e-9), 0, 1)
        for j, (cy, cx, amp, sg) in enumerate(s):
            L3 = np.eye(3) / max(float(sg), 0.5)
            records.append((np.array([float(cx), float(cy), z]),
                            float(abs(amp)) + 1e-4, L3))
            colors.append([float(v) for v in cols[j]])
    if not records:
        return jsonify(error="nothing painted to export"), 400
    if request.args.get("fmt") == "json":
        js = m.export_splats(records, fmt="json", colors=colors)
        from flask import Response
        return Response(js, mimetype="application/json")
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        pth = f.name
    m.export_splats(records, path=pth, fmt="ply", colors=colors)
    from flask import send_file
    return send_file(pth, mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name="painting_splats_3d.ply")


@app.get("/api/splats/export.ply")
def splats_export_ply():
    """Export this painting as a STANDARD 3D-Gaussian-Splatting .ply
    (?k=, default 300) -- opens in any 3DGS viewer. 2-D splats lifted to
    the z=0 plane by leCore's exporter; colours are the per-splat channel
    amplitudes, normalised. ?fmt=json returns the three.js billboard JSON
    instead (the seed of a GPU front end that renders what the back end
    authored -- same code, same picture)."""
    lc = _lecore()
    if "ct" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    from PIL import Image as _PImage
    K = max(8, min(800, int(request.args.get("k", 300))))
    if request.args.get("scope") == "layers":
        return _splats_export_layers(K)
    comp = composite_png_array()[..., :3]
    sw = 180
    sh = max(2, int(round(DOC.height * sw / max(DOC.width, 1))))
    small = np.asarray(_PImage.fromarray(
        (comp * 255).astype(np.uint8)).resize((sw, sh)), np.float32) / 255.0
    splats, A = _splat_code(small.astype(float), K)
    cols = np.clip(np.abs(A) / (np.abs(A).max(axis=1, keepdims=True) + 1e-9),
                   0, 1)
    import tempfile
    if request.args.get("fmt") == "json":
        from lecore import autoboot as _ab           # exporter lives on the mind
        m = _sage()
        if m is None:
            return jsonify(error="sage unavailable: %s"
                                 % _LECORE.get("mind_err", "")), 503
        js = m.export_splats_2d(splats, fmt="json", colors=cols.tolist())
        from flask import Response
        return Response(js, mimetype="application/json")
    m = _sage()
    if m is None:
        return jsonify(error="sage unavailable: %s"
                             % _LECORE.get("mind_err", "")), 503
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        pth = f.name
    m.export_splats_2d(splats, path=pth, fmt="ply", colors=cols.tolist())
    from flask import send_file
    return send_file(pth, mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name="painting_splats.ply")


@app.post("/api/dream")
def dream():
    """DREAM SEEDS: train leCore's holographic drift model (HDRIFT) on this
    painting -- the composite plus each visible layer as its own view -- and
    generate soft light-and-colour compositions in its mood. Honest scope:
    HDRIFT v1 drifts in splat space, so dreams are bokeh-soft mood fields,
    not pictures -- underpaintings, colour studies, backdrops. {"n": <=8,
    "seed"?, "k"? splats}. Returns thumbnails; /api/dream/place lands one
    as a layer."""
    lc = _lecore()
    if "train" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    import base64 as _b64
    import io as _io
    from PIL import Image as _PImage
    d = request.json or {}
    n = max(1, min(8, int(d.get("n", 6))))
    seed = int(d.get("seed", 0) or 0)
    k = max(4, min(24, int(d.get("k", 12))))
    comp = composite_png_array()[..., :3]
    views = []
    from PIL import Image as _PI2
    for pth in (d.get("paths") or [])[:8]:
        try:
            views.append(np.asarray(_PI2.open(pth).convert("RGB"),
                                    np.float32) / 255.0)
        except OSError as e:
            return jsonify(error="could not read %s: %s" % (pth, e)), 400
    views.append(comp)
    for l in DOC.layers:
        if not l.visible:
            continue
        a = l.pixels
        if float(a[..., 3].max()) <= 0:
            continue
        views.append(np.clip(a[..., :3] * a[..., 3:4]
                             + comp * (1 - a[..., 3:4]), 0, 1))
    small = []
    for v in views[:8]:
        im = _PImage.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8))
        small.append(np.asarray(im.resize((96, 64)), np.float32) / 255.0)
    if len(small) < 2:
        small = small * 2
    # R18, after the splat audit: dreams drift in COLOUR SPLAT space now.
    # Per view: K luminance splats (matching pursuit + joint refit) with
    # per-channel amplitudes -- (cy, cx, sigma, aR, aG, aB) x K, whitened
    # (mixed units drift badly raw), and generated points are clamped to
    # the training range (off-manifold amplitudes blow out: measured).
    # Decode tone-maps (Reinhard) so overlap keeps drama without clipping
    # to white. Deterministic in seed -- pass the seed around, not pixels.
    from holographic.sampling_and_signal.holographic_hdrift import (
        build_drift_model, drift_sample)
    dshape = (64, 96)
    # kk is FIXED at 48 by default so taste vectors recorded today still
    # match tomorrow's feature space; an explicit k changes it (and taste
    # steering silently skips when dimensions disagree)
    kk = 48 if int(d.get("k", 12)) == 12 else max(16, min(96, k * 4))
    feats = []
    for v in small:
        s2, A2 = _splat_code(v.astype(float), kk)
        row = [(s2[j][0], s2[j][1], s2[j][3],
                A2[j, 0], A2[j, 1], A2[j, 2]) for j in range(kk)]
        row.sort(key=lambda f: (round(f[0], 3), round(f[1], 3)))
        feats.append(np.asarray(row, float).ravel())
    raw = np.stack(feats)
    mu, sd = raw.mean(0), raw.std(0) + 1e-6
    lo, hi = raw.min(0), raw.max(0)
    try:
        model = build_drift_model((raw - mu) / sd, dim=1024, seed=seed)
    except ValueError as e:
        # leCore REFUSES degenerate data rather than generating the mean --
        # relay that honestly instead of a 500
        return jsonify(error="not enough distinct views to dream from: %s "
                             "-- paint more layers, or pass paths of other "
                             "images" % e), 400
    X = drift_sample(model, n=n, seed=seed or 1, steps=60)
    # R21 TASTE: starred dreams (see /api/dream/fave) pull new candidates
    # toward what the user loved -- a scale-preserving centroid blend in
    # the stable RAW feature space (compass.steer renormalises to unit
    # length, which would crush these mixed-unit vectors -- measured), then
    # the usual clamp keeps everything on-manifold. Deterministic: same
    # views + seed + stars = same dreams.
    _sv = d.get("steer", 0.35)
    steer_step = (0.35 if _sv is True else 0.0 if _sv in (False, None)
                  else float(_sv))
    taste = [t for t in _taste_load() if len(t) == raw.shape[1]] \
        if steer_step > 0 else []
    centroid = np.mean(np.asarray(taste, float), axis=0) if taste else None
    del _DREAMS[:]
    del _DREAM_PTS[:]
    thumbs = []
    from holographic.rendering.holographic_splat import _gaussian
    for x in X:
        raw_pt = np.asarray(x) * sd + mu
        if centroid is not None:
            raw_pt = (1.0 - steer_step) * raw_pt + steer_step * centroid
        p = np.clip(raw_pt, lo, hi)
        _DREAM_PTS.append([float(v) for v in p])
        p = p.reshape(-1, 6)
        out = np.zeros(dshape + (3,))
        for row in p:
            g = _gaussian(dshape, row[0], row[1], max(row[2], 0.6))
            out += g[..., None] * row[3:6][None, None, :]
        out = np.maximum(out, 0.0)
        out = np.clip(out / (1.0 + out * 0.55), 0, 1)
        _DREAMS.append(out.astype(np.float32))
        im = _PImage.fromarray((out * 255).astype(np.uint8)).resize(
            (240, 160), _PImage.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, "PNG")
        thumbs.append(_b64.b64encode(buf.getvalue()).decode("ascii"))
    return jsonify(ok=True, count=len(thumbs), thumbs=thumbs,
                   seed=int(seed or 1),
                   note="deterministic in seed: the same views + seed "
                        "regenerate these exact dreams")


@app.post("/api/dream2")
def dream2():
    """DREAM v2 (R28): high-capacity anisotropic COLOUR-splat drift.

    The v1 dream drifts 48 isotropic splats -- bokeh by construction.
    v2 fits each view with K anisotropic colour splats (coarse-to-fine
    matching pursuit + ridge-refit, lestudio.hdrift_aniso), whitens the
    codes, PCA-projects to m components and drifts THERE with leCore --
    samples stay on the collection's manifold, so they decode into
    tangible structured compositions, not dots. {"n"<=8, "seed",
    "k"?=160, "m"?=10, "steps"?=30, "noise0"?=0.35, "latitude"?=0.12,
    "paths": [>=3 reference images]}. The doc composite is always the
    first view. Results land in the same cache as /api/dream, so
    /api/dream/place places them."""
    lc = _lecore()
    if "train" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    import base64 as _b64
    import io as _io
    from PIL import Image as _PImage
    from .hdrift_aniso import fit_color_splats, render_splats
    d = request.json or {}
    n = max(1, min(8, int(d.get("n", 6))))
    seed = int(d.get("seed", 1) or 1)
    K = max(48, min(320, int(d.get("k", 160))))
    m = max(3, min(40, int(d.get("m", 10))))
    steps = max(5, min(120, int(d.get("steps", 30))))
    noise0 = float(d.get("noise0", 0.35))
    lat = float(d.get("latitude", 0.12))
    comp = composite_png_array()[..., :3]
    views = [comp]
    for pth in (d.get("paths") or [])[:12]:
        try:
            views.append(np.asarray(_PImage.open(pth).convert("RGB"),
                                    np.float32) / 255.0)
        except OSError as e:
            return jsonify(error="could not read %s: %s" % (pth, e)), 400
    if len(views) < 4:
        return jsonify(error="dream2 needs at least 3 reference paths "
                             "(the composite is the 4th view) -- its PCA "
                             "manifold is meaningless with fewer"), 400
    # common working size, portrait/landscape aware, ~97k px
    ar = comp.shape[0] / comp.shape[1]
    ww = int(round((97000 / ar) ** 0.5))
    hh = int(round(ww * ar))
    small = []
    for v in views:
        im = _PImage.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8))
        small.append(np.asarray(im.resize((ww, hh), _PImage.LANCZOS),
                                np.float32) / 255.0)
    rows = []
    for i, v in enumerate(small):
        F, _ = fit_color_splats(v, K=K,
                                jitter=np.random.RandomState(seed + i))
        rows.append(F.ravel())
    raw = np.stack(rows)
    mu, sd = raw.mean(0), raw.std(0) + 1e-6
    lo, hi = raw.min(0), raw.max(0)
    Z = (raw - mu) / sd
    zm = Z.mean(0)
    U, S, Vt = np.linalg.svd(Z - zm, full_matrices=False)
    m = min(m, len(small) - 1)
    C = U[:, :m] * S[:m]
    from holographic.sampling_and_signal.holographic_hdrift import (
        build_drift_model, drift_sample)
    try:
        model = build_drift_model(C, dim=512, seed=seed)
    except ValueError as e:
        return jsonify(error="not enough distinct views to dream from: "
                             "%s" % e), 400
    X = drift_sample(model, n=n, seed=seed, steps=steps, noise0=noise0)
    lo_c, hi_c = C.min(0), C.max(0)
    span_c = np.where(hi_c - lo_c < 1e-9, 1.0, hi_c - lo_c)
    del _DREAMS[:]
    del _DREAM_PTS[:]
    thumbs = []
    for x in X:
        c = np.clip(np.asarray(x), lo_c - lat * span_c, hi_c + lat * span_c)
        p = np.clip((zm + c @ Vt[:m]) * sd + mu, lo, hi)
        _DREAM_PTS.append([float(v) for v in p])
        out = np.clip(render_splats(p.reshape(-1, 8), (hh, ww)), 0, 1)
        _DREAMS.append(out.astype(np.float32))
        im = _PImage.fromarray((out * 255).astype(np.uint8)).resize(
            (240, int(240 * ar)), _PImage.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, "PNG")
        thumbs.append(_b64.b64encode(buf.getvalue()).decode("ascii"))
    return jsonify(ok=True, count=len(thumbs), thumbs=thumbs, seed=seed,
                   k=K, m=int(m),
                   note="dream2: anisotropic colour-splat PCA drift; "
                        "deterministic in (views, seed); place with "
                        "/api/dream/place")


@app.post("/api/dream/place")
def dream_place():
    """Land dream #i from the last /api/dream as a new layer, scaled to the
    canvas. {"i", "name"?, "opacity"?}"""
    from PIL import Image as _PImage
    d = request.json or {}
    try:
        arr = _DREAMS[int(d.get("i", 0))]
    except (IndexError, ValueError, TypeError):
        return jsonify(error="no such dream -- run /api/dream first"), 400
    im = _PImage.fromarray((arr * 255).astype(np.uint8)).resize(
        (DOC.width, DOC.height), _PImage.LANCZOS)
    rgb = np.asarray(im, np.float32) / 255.0
    out = np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])
    l = DOC.add_layer(d.get("name") or "Dream",
                      pixels=_gated_output(out, d), asset=True)
    if d.get("opacity") is not None:
        DOC.edit_layer(l.id, opacity=float(d["opacity"]))
    return jsonify(ok=True, id=l.id)


_APP_SUBSTRATES = {}


@app.post("/api/memory")
def user_memory():
    """PER-USER memory (R56, APP_FOUNDATION §6): each painter gets their
    own physically separate leCore partition via m.app_substrate --
    remember/recall with provenance (taught vs model-cached), observe/
    suggest/habits (procedures mined from what THIS user actually does),
    and forget (the veto). Keyed by the X-User identity the agent
    surface already carries; the shared studio doctrine stays with the
    sage (/api/advise). {"action": "remember"|"recall"|"observe"|
    "suggest"|"habits"|"forget", plus q/a/goal/steps as the action
    needs}."""
    d = request.json or {}
    uid = _req_uid()
    if not uid:
        return jsonify(error="send an X-User header -- memory is per "
                             "person"), 400
    from . import mind as _mind_fn
    m = _mind_fn()
    if not hasattr(m, "app_substrate"):
        return jsonify(error="this leCore build has no app_substrate -- "
                             "update leos-core for per-user memory"), 503
    sub_ = _APP_SUBSTRATES.get(uid)
    if sub_ is None:
        sub_ = _APP_SUBSTRATES[uid] = m.app_substrate("lestudio", user=uid)
    act = d.get("action") or "recall"
    try:
        if act == "remember":
            r = sub_.remember(d["q"], d["a"], topic=d.get("topic"))
        elif act == "recall":
            r = sub_.recall(d["q"],
                            established_only=bool(d.get("established_only")))
        elif act == "observe":
            r = sub_.observe(d["goal"], list(d.get("steps") or []))
        elif act == "suggest":
            r = sub_.suggest(d["goal"])
        elif act == "habits":
            r = sub_.habits()
        elif act == "forget":
            r = sub_.forget(d["q"])
        else:
            return jsonify(error="unknown action %r" % act), 400
        try:
            if hasattr(sub_, "save"):
                sub_.save()
        except Exception:
            pass
        return jsonify(ok=True, result=r)
    except KeyError as e:
        return jsonify(error="missing field for %s: %s" % (act, e)), 400
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post("/api/advise")
def advise():
    """THE STUDIO SAGE: ask the leCore memory that every painting session
    has been teaching ('hair is a mass before strands', the banding cure,
    the replay laws...). {"q": question} -> {answer, tier, via}; or
    {"teach": {"q", "a"}} adds a lesson and saves the store -- the swarm's
    blackboard, productised. Agents and the Sage panel share one mind."""
    m = _sage()
    if m is None:
        return jsonify(error="leCore is not available here: %s"
                             % _LECORE.get("mind_err", "unknown")), 503
    d = request.json or {}
    t = d.get("teach")
    if t:
        if not (t.get("q") and t.get("a")):
            return jsonify(error="teach needs q and a"), 400
        with _LECORE_LOCK:
            m.teach(str(t["q"]), str(t["a"]))
            try:
                m.learning_save(_LECORE.get("mind_part") or "lecore_memory")
            except Exception:
                pass
        return jsonify(ok=True, taught=True)
    q = str(d.get("q") or "").strip()
    if not q:
        return jsonify(error="ask something: {\"q\": ...}"), 400
    with _LECORE_LOCK:
        r = m.ask(q) or {}
        # R54: the ladder's exact tiers miss any PARAPHRASE of a taught
        # lesson (an audit found 723 taught pairs and every reworded
        # question coming back empty). Fall back to the mind's own
        # rare-token-weighted search over the taught log and serve the
        # best hit -- with the matched question, so the caller can see
        # what the sage actually recalled.
        if not (r.get("answer") or "").strip():
            try:
                got = m.session_search(q, sessions="all", k=6)
                qn = " ".join(q.lower().split())
                hits = [h for h in (got or {}).get("hits", [])
                        if h.get("score", 0) >= 0.18
                        and (h.get("answer") or "").strip()
                        and " ".join(str(h.get("question", ""))
                                     .lower().split()) != qn][:3]
            except Exception:
                hits = []
            if hits:
                top = hits[0]
                return jsonify(ok=True, answer=top["answer"],
                               tier="T1s", via="taught-log search",
                               matched=top["question"],
                               score=top["score"],
                               also=[{"q": h["question"],
                                      "score": h["score"]}
                                     for h in hits[1:]])
    return jsonify(ok=True, answer=r.get("answer") or "",
                   tier=r.get("tier"), via=r.get("via"))


@app.get("/api/graph/export.glsl")
def graph_export_glsl():
    """Take your grade to the GPU: compile this document's pointwise grade
    nodes (grade / colour wheels / vignette) to a Shadertoy-style fragment
    shader via leCore's postfx emitter. Neighbourhood nodes (glow, clarity,
    grain) cannot be a single-pass fragment and are listed as skipped in
    the header comment. The mapping is an approximation, and says so."""
    lc = _lecore()
    if "glsl" not in lc:
        return jsonify(error="leCore is not available here: %s"
                             % lc.get("err", "not on PYTHONPATH")), 503
    nodes = {nd["id"]: nd for nd in GRAPH.to_list()}
    steps, skipped = [], []
    # walk the chain from the output backwards, then reverse
    out = next((nd for nd in nodes.values()
                if nd.get("type") in ("output", "Output")), None)
    chain = []
    seen = set()
    cur = out
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        ins = cur.get("inputs") or {}
        nxt = None
        for v in ins.values():
            key = v[0] if isinstance(v, (list, tuple)) else str(v).split(".")[0]
            if key in nodes:
                nxt = nodes[key]
                break
        cur = nxt
    _ALIAS = {"Grade": "grade", "Color wheels": "color_wheels",
              "Vignette": "vignette", "Media in": "media",
              "Output": "output", "Clarity": "clarity", "Glow": "glow",
              "Grain": "grain"}
    for nd in reversed(chain):
        t, p = nd.get("type"), nd.get("params") or {}
        t = _ALIAS.get(t, t)
        if t == "grade":
            bp = float(p.get("blackpoint", 0.0))
            wp = float(p.get("whitepoint", 1.0))
            g = float(p.get("gamma", 1.0))
            steps.append(("color_grade",
                          {"lift": -bp,
                           "contrast": 1.0 / max(wp - bp, 1e-3)}))
            if abs(g - 1.0) > 1e-6:
                steps.append(("gamma", {"g": 2.2 * g}))
        elif t == "color_wheels":
            steps.append(("color_grade", {
                "temperature": float(p.get("gain_r", 0.0))
                               - float(p.get("gain_b", 0.0)),
                "tint": float(p.get("gain_g", 0.0)),
                "saturation": 1.0}))
        elif t == "vignette":
            steps.append(("vignette",
                          {"strength": float(p.get("amount", 0.4)),
                           "radius": float(p.get("radius", 1.0))}))
        elif t in ("media", "output", None):
            pass
        else:
            skipped.append(t)
    if not steps:
        return jsonify(error="no pointwise grade nodes (grade, color "
                             "wheels, vignette) in this graph"), 400
    try:
        sh = lc["glsl"](steps, name="lestudio_grade")
    except Exception as e:
        return jsonify(error="GLSL emit failed: %s" % e), 500
    head = ("// exported from leStudio -- APPROXIMATE mapping of the "
            "document grade chain\n")
    if skipped:
        head += ("// skipped (multi-pass, not fragment-emittable): %s\n"
                 % ", ".join(sorted(set(skipped))))
    from flask import Response
    return Response(head + sh, mimetype="text/plain")


@app.post("/api/paint_batch")
def paint_batch():
    """Many strokes in ONE request -- the agent/swarm fast path (R16).

    {"strokes": [<same payload as /api/paint>...]} -- modes "brush"
    (default), "knife" and "blend"; masks/selections/live are not batchable.
    The whole batch applies atomically under the document lock, records ONE
    undo entry ("Brush xN" -- ctrl+Z reverts the batch), and every stroke is
    still its own replay record, so the timelapse shows each mark. Per-stroke
    diagnosis warnings are skipped: a script wants throughput, and can probe
    with a single /api/paint when it cares. Returns {ok, count, sids}.

    Why it exists: a swarm painting a portrait made ~8000 /api/paint calls;
    HTTP + JSON + per-stroke undo bookkeeping dominated wall time. One batch
    of 50 strokes costs one round trip and one snapshot."""
    with _DOC_LOCK:                 # R58: see /api/paint
        return _paint_batch_locked()


def _paint_batch_locked():
    d = request.json or {}
    items = d.get("strokes")
    if not isinstance(items, list) or not items:
        return jsonify(error="strokes must be a non-empty list of "
                             "/api/paint payloads"), 400
    if len(items) > 512:
        return jsonify(error="at most 512 strokes per batch -- split it"), 400
    cleaned = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            return jsonify(error="stroke %d is not an object" % i), 400
        mode = it.get("mode", "brush")
        if mode not in ("brush", "knife", "blend"):
            return jsonify(error="stroke %d: mode %r is not batchable -- "
                                 "use /api/paint for it" % (i, mode)), 400
        if it.get("target_mask") or it.get("live"):
            return jsonify(error="stroke %d: mask/live strokes are not "
                                 "batchable" % i), 400
        try:
            cleaned.append((_clean_paint(dict(it)), mode))
        except _Gone as e:
            return jsonify(error="stroke %d: %s" % (i, e)), 400
    with _DOC_LOCK:
        busy = _still_busy(d)      # R64: the cadence promise, made atomic
        if busy is not None:
            return busy
        # R65: a NAMED pass is applied once. A painter re-ran a script by
        # accident and every semi-transparent build-up pass doubled -- the
        # window bloom washed the frame out and it took two fix rounds to
        # find out why. Low-opacity passes have no way to tell they have
        # already happened, so the server remembers for them: same user,
        # same document, same pass name -> 409, unless `force`.
        pname = str(d.get("pass") or "").strip()
        prun = str(d.get("pass_run") or "").strip()
        if pname and not d.get("force"):
            key = (_viewing_doc_id(), _owner_uid(), pname)
            prev = PASSES_SEEN.get(key)
            # A pass bigger than one batch arrives as several calls, and
            # they all belong to the SAME pass: `pass_run` is the client's
            # id for this instance of it, so continuing chunks are let
            # through and only a genuinely NEW run of an applied pass is
            # refused. (Found immediately: the first chunk of the cast
            # shadows registered the name and the server then refused the
            # rest of the same pass.)
            if prev is not None and prev != prun:
                return jsonify(error="pass %r was already painted in this "
                                     "document by you -- send force:true to "
                                     "paint it again on purpose" % pname,
                               pass_already_applied=True, **{"pass": pname}), 409
        DOC._edited_palette_last = False
        lids = []
        for it, _m in cleaned:
            lid = it.get("layer")
            try:
                DOC.layer(lid)
            except KeyError:
                return jsonify(error="no such layer: %r" % lid), 400
            if lid not in lids:
                lids.append(lid)
        # one undo record for the batch: region = union box when every
        # target layer has had its first (layer-wide hygiene) stroke
        reg = None
        if all(getattr(DOC.layer(l), "_hyg_filled", False) for l in lids):
            xs, ys, pad = [], [], 3.0
            for it, _m in cleaned:
                r = float(it.get("radius", 8)) + pad
                for pt in (it.get("points") or []):
                    xs += [float(pt[0]) - r, float(pt[0]) + r]
                    ys += [float(pt[1]) - r, float(pt[1]) + r]
            if xs:
                rx0, ry0 = max(0, int(min(xs))), max(0, int(min(ys)))
                rx1 = min(DOC.width, int(max(xs)) + 1)
                ry1 = min(DOC.height, int(max(ys)) + 1)
                if rx1 > rx0 and ry1 > ry0:
                    reg = (rx0, ry0, rx1, ry1)
        # journaled=True is the R16 batch-record law, and it was MISSING.
        # Without it record() takes the "any non-stroke edit" branch and sets
        # layer._replay_ok = False on every target -- so a layer painted
        # through the swarm fast path was never replay-faithful again, and a
        # .lews had to store its pixels instead of rebuilding it from the
        # journal. Measured on the R61 painting: 7 of 13 layers demoted, 93 MB
        # of float32 saved for a picture that displays as a 1.3 MB PNG. Every
        # stroke in the batch IS in the replay log, which is exactly what
        # journaled=True asserts.
        DOC.record("Brush x%d" % len(cleaned), only=lids, region=reg,
                   journaled=True)
        sids = []
        try:
            for it, mode in cleaned:
                # DIP IN BAND. `real_brush` models a finite charge, which is
                # correct physics and a trap for anything without a hand:
                # measured, a loaded brush is dry after three long strokes
                # and the next seven change not one pixel, silently, at 200.
                # The only refill was POST /api/brush_load, so a script had
                # to leave the batch to dip -- which meant a real-brush pass
                # could not be batched AT ALL. The R66 accents pass spent
                # twelve minutes on two round trips per stroke for that
                # reason alone. `dip` on a batch item reloads the brush
                # immediately before that stroke: true for a full charge in
                # the stroke's own colour, or {"color", "amount"}.
                d_ = it.get("dip")
                if d_:
                    if d_ is True:
                        DOC.load_brush(color=it.get("color"), amount=1.0)
                    elif isinstance(d_, dict):
                        DOC.load_brush(color=d_.get("color", it.get("color")),
                                       amount=float(d_.get("amount", 1.0)))
                    else:
                        raise ValueError("dip must be true or "
                                         "{color?, amount?}")
                if mode == "knife":
                    sid = DOC.knife(it["layer"], it["points"],
                                    mode=str(it.get("knife", "smooth")),
                                    radius=float(it.get("radius", 26)),
                                    strength=float(it.get("opacity", 0.7)),
                                    record=False, stroke_new=True)
                elif mode == "blend":
                    sid = DOC.blend_stroke(
                        it["layer"], it["points"],
                        radius=float(it.get("radius", 18)),
                        strength=float(it.get("opacity", 0.6)),
                        brush=it.get("brush"),
                        record=False, stroke_new=True)
                else:
                    sid = DOC.paint(
                        it["layer"], it["points"],
                        color=it.get("color", [0, 0, 0]),
                        radius=float(it.get("radius", 8)),
                        opacity=float(it.get("opacity", 1)),
                        erase=bool(it.get("erase")),
                        hardness=float(it.get("hardness", 0.7)),
                        record=False, stroke_new=True,
                        selection=it.get("selection"),
                        sel_invert=bool(it.get("sel_invert")),
                        brush=it.get("brush"),
                        media=(it.get("media") or None),
                        material=(it.get("material") or None),
                        mix=float(it.get("mix", 0.0)),
                        real_brush=bool(it.get("real_brush", False)),
                        load=float(it.get("load", 0.6)),
                        taper=float(it.get("stroke_taper", 0.0)))
                if sid is None and DOC.strokes:
                    # paint() only hands back the id for record=True; the
                    # replay record exists either way, so name it honestly
                    sid = DOC.strokes[-1]["id"]
                sids.append(sid)
        except ValueError as e:
            return jsonify(error=str(e), applied=len(sids),
                           sids=sids), 400
        # a recorded stroke would stamp replay-ok forward itself; these are
        # record=False (one undo entry for the batch) but every stroke IS
        # recorded in the replay log, so the layers stay faithful by
        # construction -- stamp them like paint() would have.
        #
        # NOTE, because this block used to look like it was undoing the
        # damage above and was not: _mark_replay_ok writes the DOCUMENT's
        # _replay_ok DICT (the "is this verdict still current" cache), while
        # the demotion above cleared the LAYER's _replay_ok ATTRIBUTE (the
        # verdict itself). Two different things sharing one name, so the
        # repair silently repaired nothing for years of agent painting. The
        # real fix is journaled=True above; this stamp stays because it is
        # still the right thing for the cache.
        for lid in lids:
            if DOC._replay_ok_cached(lid):
                DOC._mark_replay_ok(lid)
        GRAPH.commit_layer_outputs()
        if pname:
            PASSES_SEEN[(_viewing_doc_id(), _owner_uid(), pname)] = prun
    return jsonify(ok=True, count=len(sids), sids=sids)


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


def _foreign_top_entry(stack):
    """(author, display_name) of the stack's top entry when it belongs to a
    DIFFERENT identified user than the caller, else None. Entries recorded
    outside a request (tests, scripts) carry author '' and are unowned."""
    if not stack:
        return None
    ent = stack[-1]
    author = ent[2] if len(ent) > 2 else ""
    me = _req_uid()
    if author and me and author != me:
        return author, (SYNC["names"].get(author) or author[:6])
    return None


@app.post("/api/undo")
def undo():
    """Undo the last operation, on the picture or the palette -- whichever was
    edited last. Includes impasto height. If the last change belongs to a
    DIFFERENT user this refuses with 409 (pass force:true to override):
    silently reverting a collaborator's stroke reads as data loss to them."""
    # a body-less POST (older clients, scripts) must still work:
    # request.json raises 415 without a JSON content-type
    d = request.get_json(silent=True) or {}
    with _DOC_LOCK:
        foreign = _foreign_top_entry(DOC._undo)
        if foreign and not d.get("force"):
            return jsonify(error="the last change is %s's -- pass force to "
                                 "undo it anyway" % foreign[1],
                           author=foreign[0]), 409
        surf = _last_surface()
        ok = surf.undo()
        if not ok and surf is not DOC:
            DOC._edited_palette_last = False       # fall back to the picture
            ok = DOC.undo()
    return jsonify(ok=ok)


@app.post("/api/redo")
def redo():
    """Redo, on whichever surface was edited last. Refuses (409) when the
    entry to redo is another user's, unless force:true -- symmetric with
    /api/undo."""
    # a body-less POST (older clients, scripts) must still work:
    # request.json raises 415 without a JSON content-type
    d = request.get_json(silent=True) or {}
    with _DOC_LOCK:
        foreign = _foreign_top_entry(DOC._redo)
        if foreign and not d.get("force"):
            return jsonify(error="that change is %s's -- pass force to "
                                 "redo it anyway" % foreign[1],
                           author=foreign[0]), 409
        surf = _last_surface()
        ok = surf.redo()
        if not ok and surf is not DOC:
            ok = DOC.redo()
    return jsonify(ok=ok)


@app.post("/api/graph")
def set_graph():
    """Replace the node graph: {"nodes": [{id, type, params, inputs, x, y}], "base_rev"?, "force"?}. Inputs: "NID", "NID.socket", or [id, socket]; "param:<name>" keys wire values into parameters. Commits Layer out nodes. With base_rev (the grev you loaded), a concurrent edit 409s with the current nodes so you can rebase instead of erasing it."""
    d = request.json or {}
    base = d.get("base_rev")
    cur = int(getattr(GRAPH, "grev", 0))
    if base is not None and not d.get("force") and int(base) != cur:
        # Whole-graph POST was last-write-wins: two editors, and whoever
        # saved second silently DELETED the other's new nodes (probe: an
        # added node vanished). The 409 carries the live nodes so the client
        # can rebase its one local op and re-post. No base_rev (older
        # clients, agents) keeps the old unconditional behaviour.
        return jsonify(error="the graph changed under you (rev %d, yours "
                             "was %s) -- rebase onto the returned nodes and "
                             "re-post, or pass force to overwrite"
                             % (cur, base),
                       grev=cur, nodes=GRAPH.to_list()), 409
    GRAPH.set_graph(d.get("nodes", []))
    n = GRAPH.commit_layer_outputs()
    # R24 (found grading a painting): a node with an unknown TYPE evaluates
    # to an error, and output.png used to silently fall back to the raw
    # composite -- three rounds of "graded" finals were never graded. Name
    # the strangers at the door.
    from . import OPS as _OPS
    unknown = sorted({nd.get("type") for nd in d.get("nodes", [])
                      if nd.get("type") and nd.get("type") not in _OPS})
    body = dict(ok=True, committed=n, grev=int(getattr(GRAPH, "grev", 0)),
                conflicts=[DOC.layer(c).name for c in getattr(GRAPH, "last_conflicts", [])
                           if any(l.id == c for l in DOC.layers)])
    if unknown:
        body["warning"] = ("unknown node type(s) %s -- the graph cannot "
                           "evaluate them; valid types are capitalised "
                           "(Grade, Glow, Media in...); see /api/mind"
                           % ", ".join(repr(u) for u in unknown))
        body["unknown"] = unknown
    # R26 (found grading a painting): a client sent wiring as a separate
    # 'wires' list, which this API does not read -- every node sat
    # unreachable, and output.png served the raw composite behind an
    # ok:true. Wires live in each node's 'inputs' {socket: "nid"}; name
    # the orphans so the caller notices before trusting the render.
    orphans = GRAPH.unreachable_nodes()
    if orphans:
        body["warning"] = (body.get("warning", "") + (" " if unknown else "")
                          + "node(s) %s cannot reach the Output node -- "
                            "wire nodes with per-node inputs "
                            "{\"image\": \"<node id>\"}; a separate "
                            "'wires' list is not read"
                          % ", ".join(repr(o) for o in orphans))
        body["unreachable"] = orphans
    return jsonify(**body)


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
    return jsonify(ok=True, node=node, grev=int(getattr(GRAPH, "grev", 0)))


@app.get("/api/graph/output.png")
def graph_output():
    """The Output node's render as PNG."""
    if LIVE["on"] and LIVE["jpeg"] is not None and request.args.get("fmt") == "jpeg":
        # the live loop already evaluated + encoded this frame: zero extra work
        return send_file(io.BytesIO(LIVE["jpeg"]), mimetype="image/jpeg")
    GRAPH.ensure_default()
    nid = GRAPH.output_node()
    try:
        wq = request.args.get("w")
        if wq:
            # R5 #5: cap the EVALUATION, not just the encode -- render_at at
            # display width is the 3.4x interactive path
            w = max(16, min(int(wq), 4096))
            h = max(int(round(w * DOC.height / max(DOC.width, 1))), 8)
            return _png(_renderable(GRAPH.render_at(nid, w, h)))
        return _png(_renderable(GRAPH.evaluate(nid)))
    except Exception as e:
        # R24: this fallback used to be SILENT for every failure, so a graph
        # full of unknown node types served the raw composite and the caller
        # believed their grade applied (three rounds of finals, ungraded).
        # An explicitly-built graph now fails honestly; only the untouched
        # default graph keeps the composite convenience.
        if len(GRAPH.nodes) > 1:
            return jsonify(error="the graph did not evaluate: %s" % e), 409
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
    """One node's render as PNG (memoised per signature). R5 #5: `?w=` EVALUATES
    at that width via render_at (before, w only downscaled the encode of a
    full-res evaluate -- all the compute, none of the savings)."""
    try:
        sock = request.args.get("sock", "out")
        wq = request.args.get("w")
        if wq:
            w = max(16, min(int(wq), 4096))
            h = max(int(round(w * DOC.height / max(DOC.width, 1))), 8)
            return _png(_renderable(GRAPH.render_at(nid, w, h, sock)))
        return _png(_renderable(GRAPH.evaluate(nid, sock)))
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
        l = GRAPH.apply_to_layer(d["id"], d.get("name"), d.get("layer"),
                                 selection=d.get("selection"),
                                 sel_invert=bool(d.get("sel_invert")),
                                 sel_feather=float(d.get("sel_feather", 0)))
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


@app.get("/api/replay/info")
def replay_info():
    """How much of the painting can PLAY BACK: recorded stroke count and,
    per painted layer, whether a replay base exists (a layer with no base
    starts fully formed in the timelapse instead of growing)."""
    with _DOC_LOCK:
        bases = getattr(DOC, "_replay_base", {})
        per = {}
        for k in DOC.strokes:
            per[k["layer"]] = per.get(k["layer"], 0) + 1
        return jsonify(ok=True, strokes=len(DOC.strokes),
                       history=DOC.history_len(),
                       history_spooled=len(getattr(DOC, "_history_spool",
                                                   []) or []),
                       layers=[{"id": lid, "strokes": n,
                                "from_base": lid in bases}
                               for lid, n in per.items()])


@app.get("/api/replay/timelapse.gif")
def replay_timelapse():
    """WATCH THE PAINTING BEING PAINTED: an animated GIF of the document
    rebuilt stroke by stroke, in the order the marks were actually made
    (globally chronological, not layer by layer). ?frames= snapshots
    (default 60, max 240), ?w= output width (default 640), ?fps=
    (default 12), ?hold= extra copies of the final frame (default 10) so
    the loop rests on the finished picture. The last frame is always the
    TRUE current composite -- unrecorded touch-ups appear there rather
    than being pretended into the history."""
    from PIL import Image as PImage
    frames = max(2, min(240, int(request.args.get("frames", 60))))
    w = max(64, min(DOC.width, int(request.args.get("w", 640))))
    fps = max(1, min(50, int(request.args.get("fps", 12))))
    hold = max(0, min(100, int(request.args.get("hold", 10))))
    imgs = []
    with _DOC_LOCK:                 # the rebuild borrows the live layers
        h = max(1, round(DOC.height * w / DOC.width))
        for c in DOC.timelapse_frames(frames):
            flat = c[..., :3] * c[..., 3:4] + 1.0 * (1 - c[..., 3:4])
            im = PImage.fromarray(
                (np.clip(flat, 0, 1) * 255).astype("uint8"))
            if w < DOC.width:
                im = im.resize((w, h), PImage.LANCZOS)
            imgs.append(im)
    if hold:
        imgs += [imgs[-1]] * hold
    buf = io.BytesIO()
    imgs[0].save(buf, "GIF", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    buf.seek(0)
    return send_file(buf, mimetype="image/gif",
                     download_name="timelapse.gif")


_REPLAY_JOB = {"state": "idle"}


def _replay_render_worker(frames, w, fps, hold, mode="strokes"):
    """Background GIF render. Holds the document lock while rebuilding (the
    timelapse borrows the live layers), so the job reports stroke-level
    progress the UI can show instead of a silent frozen canvas. GIF encoding
    happens after the lock is released.

    mode "history" (R33): frames come from Document.history_frames -- the
    undo history walked back to the beginning, newest-first -- then the
    captured frames are REVERSED so the film progresses forward in time.
    That covers pastes, fills, clears and layer ops, which the stroke
    replay showed fully formed at frame one."""
    from PIL import Image as PImage
    J = _REPLAY_JOB
    gen = None
    try:
        imgs = []
        with _DOC_LOCK:
            if mode == "history":
                total = max(1, DOC.history_len())
                step = 1                  # the generator subsamples itself
                gen = DOC.history_frames(frames)
            else:
                total = max(1, len([k for k in DOC.strokes if k["points"]]))
                step = max(1, -(-total // frames))
                gen = DOC.timelapse_frames(frames)
            J["total"] = total
            h = max(1, round(DOC.height * w / DOC.width))
            done = 0
            for c in gen:
                if J.get("cancel"):
                    # close() raises GeneratorExit inside the generator, so
                    # its finally block restores the live layers before we
                    # let go of the lock
                    gen.close()
                    J["state"] = "cancelled"
                    return
                flat = c[..., :3] * c[..., 3:4] + 1.0 * (1 - c[..., 3:4])
                im = PImage.fromarray(
                    (np.clip(flat, 0, 1) * 255).astype("uint8"))
                if w < DOC.width:
                    im = im.resize((w, h), PImage.LANCZOS)
                imgs.append(im)
                done = min(total, done + step)
                J["done"] = done
        if mode == "history":
            imgs.reverse()                # backward walk -> forward film
        if hold:
            imgs += [imgs[-1]] * hold
        buf = io.BytesIO()
        imgs[0].save(buf, "GIF", save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(1, fps)), loop=0)
        J["gif"] = buf.getvalue()
        J["state"] = "done"
    except Exception as e:                          # pragma: no cover
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        J["state"] = "error"
        J["err"] = str(e)


@app.post("/api/replay/render")
def replay_render_start():
    """Start rendering the painting timelapse as a background job:
    {"frames"?: 110, "w"?: 720, "fps"?: 12, "hold"?: 14}. One at a time --
    a second start while one runs returns 409. Poll
    /api/replay/render/status, then download /api/replay/render/result.gif.
    The rebuild holds the document while it runs (it borrows the live
    layers), which is exactly why it reports progress."""
    d = request.json or {}
    if _REPLAY_JOB.get("state") == "running":
        return jsonify(error="a replay render is already running"), 409
    mode = str(d.get("mode", "auto"))
    if mode not in ("auto", "history", "strokes"):
        return jsonify(error="mode must be auto|history|strokes"), 400
    if mode == "auto":
        # R33: the undo-history replay is the truthful one (it carries
        # pastes, fills and layer ops); fall back to strokes only when
        # there is no retained history at all
        mode = "history" if DOC.history_len() > 0 else "strokes"
    if mode == "history" and DOC.history_len() == 0:
        return jsonify(error="no retained history to replay yet"), 400
    if mode == "strokes" and not DOC.strokes:
        return jsonify(error="nothing recorded to replay yet"), 400
    frames = max(2, min(240, int(d.get("frames", 110))))
    w = max(64, min(DOC.width, int(d.get("w", 720))))
    fps = max(1, min(50, int(d.get("fps", 12))))
    hold = max(0, min(100, int(d.get("hold", 14))))
    _REPLAY_JOB.clear()
    _REPLAY_JOB.update(state="running", done=0, total=1, cancel=False,
                       mode=mode)
    threading.Thread(target=_replay_render_worker,
                     args=(frames, w, fps, hold, mode), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/replay/render/status")
def replay_render_status():
    """{state: idle|running|done|cancelled|error, done, total, err?}.
    Lock-free on purpose: it must answer while the render holds the
    document."""
    J = _REPLAY_JOB
    return jsonify(state=J.get("state", "idle"), done=J.get("done", 0),
                   total=J.get("total", 0), err=J.get("err"))


@app.post("/api/replay/render/cancel")
def replay_render_cancel():
    """Ask the running render to stop at the next frame boundary."""
    if _REPLAY_JOB.get("state") != "running":
        return jsonify(error="no render running"), 400
    _REPLAY_JOB["cancel"] = True
    return jsonify(ok=True)


@app.get("/api/replay/render/result.gif")
def replay_render_result():
    """The finished timelapse GIF (kept until the next render starts)."""
    if _REPLAY_JOB.get("state") != "done" or not _REPLAY_JOB.get("gif"):
        return jsonify(error="no finished render -- start one and poll "
                             "status until it is done"), 404
    return send_file(io.BytesIO(_REPLAY_JOB["gif"]), mimetype="image/gif",
                     as_attachment=bool(request.args.get("download")),
                     download_name="painting_timelapse.gif")


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
    _autosave_tick()             # R65: the crash net runs here, not in a tab
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


_SETTINGS_KEYS = {"action", "id", "name", "width", "height", "dpi", "mode",
                  "paper"}


def _int_arg(name):
    """A non-negative integer query arg, or None when absent/nonsense."""
    v = request.args.get(name)
    if v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


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


# --------------------------------------------------------------- agent surface ----
# R55 (leCore 0.2.21 / APP_FOUNDATION): mount the ENGINE'S standard agent
# doors -- /api/agent/tools (manifest), /api/agent/invoke (call by name,
# image routes return data: URLs), /api/engine (engine_status), and the
# engine's /api/presence -- beside our own doors rather than deriving a
# second manifest by hand (the app_lint 'own_tool_manifest' hit). Our
# pre-existing /api/mind, /api/schema and SSE /api/events stay: first
# registration wins for duplicate rules, and agents get the union.
# Guarded: an older engine without agent_surface still boots the app.
def _mount_agent_surface():
    try:
        from . import mind as _mind_fn
        m = _mind_fn()
    except Exception:
        return
    if not hasattr(m, "agent_surface"):
        return
    try:
        root = _WS_ROOT
        os.makedirs(root, exist_ok=True)
        before = set(app.view_functions)
        m.agent_surface(app, base="/api", app_name="lestudio",
                        workspace_root=root,
                        image_routes=("composite.png", "graph/render.png",
                                      "textile/preview.png"))
        # our /api/schema manifest documents routes from their view
        # docstrings; give the engine's mounted views one where the
        # engine did not, so they stay visible to agents through BOTH
        # manifests (the doc-coverage pin holds)
        for ep in set(app.view_functions) - before:
            fn = app.view_functions[ep]
            if not (fn.__doc__ or "").strip():
                fn.__doc__ = ("Engine-mounted agent door (leCore "
                              "agent_surface): see GET /api/agent/tools "
                              "for the manifest.")
    except Exception as e:
        app.logger.warning("agent_surface mount skipped: %s", e)


def _lews_boot():
    """R57 boot: the live workspace directory is the document backing. If it
    already holds leStudio documents (a previous run's, another app's, an
    agent's), REBUILD the in-process workspace from them -- restarting the
    server must not orphan the shared truth. Otherwise seed the directory
    from the boot state so other apps see us from the first request. Guarded
    to nothing on an engine without the Workspace class."""
    try:
        ws = _lews_ws()
        if ws is None:
            return
        secs = ws.sections("lestudio.document")
        if not secs:
            _lews_publish()
            return
        from . import _doc_from_section
        docs, graphs = {}, {}
        for sec in secs:
            d, g = _doc_from_section(sec["meta"], sec["arrays"])
            docs[d.id], graphs[d.id] = d, g
        if not docs:
            return
        st = ws.get("lestudio-state")
        active = (st or {}).get("meta", {}).get("active")
        WS.docs, WS.graphs = docs, graphs
        WS.active = active if active in docs else next(iter(docs))
        WS._wire()
        from holographic.io_and_interop.holographic_lews import section_hash
        for sec in secs:
            _LEWS["sha"][sec["id"]] = section_hash(sec)
        _LEWS["sha"]["__state"] = WS.active
    except Exception as e:
        app.logger.warning("lews boot restore skipped: %s", e)


try:
    _mount_agent_surface()
    _lews_boot()
except Exception:
    pass


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
