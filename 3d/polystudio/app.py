"""Poly Studio -- standalone.

A single-app launcher: the same Poly Studio backend the gallery mounts, served on its own with no sidebar.
Run:  python app.py   ->  http://127.0.0.1:5000/

The leCore engine is bundled under holostuff/ (flat holographic_* module names via flatcompat), so this folder
gets its engine from pypi: `pip install -r requirements.txt` pulls `leos-core` along with Flask/Pillow.
A vendored copy under holostuff/holographic, if present, is used INSTEAD as a development overlay -- see
Help > Engine status, which reports which one is live and whether it has everything this app calls.
"""
import os
import sys
import importlib.util

# Friendly dependency guard: name what's missing and the one-line fix, instead of a stack trace on launch.
_missing = [m for m, pkg in (("flask", "flask"), ("numpy", "numpy"), ("PIL", "pillow"))
            if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit("Missing packages: %s\nFix:  pip install -r requirements.txt" %
             ", ".join("pillow" if m == "PIL" else m for m in _missing))

from flask import Flask, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "holostuff"))     # bundled leCore engine
sys.path.insert(0, ROOT)                                # so backend.py finds its ccrun sibling

import flatcompat
_n = flatcompat.install()
print(f"  [engine] leCore mounted ({_n} modules)")

import backend                                           # the Poly Studio Blueprint (routes are /api/...)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
# Register at the root: the blueprint's own rules are "/api/...", which is exactly what app.js fetches with
# relative "api/..." URLs. No url_prefix -> no gallery path segment.
app.register_blueprint(backend.bp)

# ---------------------------------------------------------------------------------------------------
# THE FOUNDATION (leCore docs/APP_FOUNDATION.md, sweep 163). One mind, held as a singleton and gated on
# features() rather than a version pin -- a missing faculty and a renamed one both look like an absent
# attribute at call time. Then the engine's own doors, not our re-derivations of them.
import lecore

MIND = lecore.UnifiedMind(dim=256, seed=0)
HAVE = MIND.features(["agent_surface", "lews_open", "render_quality_gate",
                      "mesh_uv_unwrap", "mesh_catmull_clark", "mesh_to_sdf_grid"])
print("  [engine] " + MIND.engine_status().get("engine", "?") +
      "  features: " + ", ".join(k for k, v in HAVE.items() if v))

# A live .lews workspace so a second leCore app on the same directory sees our edits and our users.
WORKSPACE_ROOT = os.environ.get("POLYSTUDIO_WORKSPACE")
if HAVE.get("lews_open") and WORKSPACE_ROOT:
    try:
        MIND.lews_open(WORKSPACE_ROOT, app="polystudio", app_version=open(
            os.path.join(ROOT, "VERSION")).read().strip())
        print(f"  [lews] live workspace {WORKSPACE_ROOT}")
    except Exception as e:
        print(f"  [lews] workspace unavailable ({type(e).__name__}: {e})")

# The standard agent surface: manifest + invoke (base64 frames for image routes), the allow-listed /mind
# discovery door, /engine status, the SSE change feed with its 2s ping, and presence keyed by X-User.
# Our own re-derivation is gone; this replaces it at the same URLs.
if HAVE.get("agent_surface"):
    MIND.agent_surface(app, base="/api", app_name="polystudio",
                       workspace_root=WORKSPACE_ROOT,
                       image_routes=("render", "render_progressive", "render_engine",
                                     "object_texture"),
                       stream_routes=("photo",))
    print("  [agent] /api/agent/{tools,invoke} + /api/{mind,engine,events,presence}")
else:
    print("  [agent] engine has no agent_surface; upgrade leos-core")



@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/app.js")
def appjs():
    return send_from_directory(ROOT, "app.js")


@app.route("/<path:fname>")
def static_file(fname):
    # serve any other sibling asset (favicon, etc.) but never traverse out of the app folder
    safe = os.path.normpath(fname)
    if safe.startswith("..") or os.path.isabs(safe):
        return "no", 404
    full = os.path.join(ROOT, safe)
    if os.path.isfile(full):
        return send_from_directory(ROOT, safe)
    return "not found", 404


THREE_URLS = [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "https://unpkg.com/three@0.128.0/build/three.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
]


def _ensure_three():
    """Cache three.js LOCALLY on first run.

    The viewport is three.js, and index.html used to load it from a CDN only. On a machine that is offline
    or behind a proxy that blocks that CDN, the script never arrives -- and because app.js builds materials
    at module scope, an undefined THREE kills the WHOLE script: menus dead, viewport dead, no error shown.
    That is a total failure of the app for a reason that has nothing to do with the app.

    So: fetch it once into vendor/ and serve it ourselves from then on. Failure here is NOT fatal -- the
    page falls back to the CDN chain and, failing that, shows an actionable banner.
    """
    dest = os.path.join(ROOT, "vendor", "three.min.js")
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000:
        return "local"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    import urllib.request
    for url in THREE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                data = r.read()
            if len(data) > 100_000:
                with open(dest, "wb") as f:
                    f.write(data)
                print("  cached three.js locally (vendor/three.min.js) -- offline from now on")
                return "downloaded"
        except Exception:
            continue
    print("  NOTE: could not fetch three.js. The viewport needs it.")
    print("        On a networked machine run this once, or drop three.min.js (r128) into vendor/.")
    return "missing"


if __name__ == "__main__":
    import threading
    import webbrowser
    _ensure_three()
    url = "http://127.0.0.1:5000/"
    print("  Poly Studio -> " + url)
    if "--no-browser" not in sys.argv:
        # open the UI for the user instead of making them copy a URL out of the terminal
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
