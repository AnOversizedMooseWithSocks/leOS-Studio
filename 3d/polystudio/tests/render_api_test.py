"""Call the render routes over real HTTP, against tests/fake_engine.

This is the harness that would have caught the shipped `/api/photo_post` bug: it called `_tonemap`,
a function defined only inside `photo()`, so every request 500'd. `py_compile` passed. `node --check`
passed. Every browser harness passed. Only *calling the route* finds that.

It is plumbing verification, not rendering verification: the fake tracer returns a gradient, so a
green run says nothing about image quality. `quality_gate.py` against the real engine remains the
gate that matters before shipping a render change.

    python3 tests/render_api_test.py
"""
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "fake_engine"))   # flat holographic_* modules
sys.path.insert(0, ROOT)                                # backend.py and its ccrun sibling

import backend                                          # noqa: E402
from flask import Flask                                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("ok  " if cond else "FAIL ") + name + ((" -- " + detail) if detail and not cond else ""))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve():
    app = Flask(__name__)
    app.register_blueprint(backend.bp)
    port = free_port()
    t = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port,
                                                threaded=True, use_reloader=False),
                         daemon=True)
    t.start()
    for _ in range(100):                                # wait for the socket to accept
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/scene", timeout=1).read()
            return port
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("server did not come up")


def get(port, path, timeout=30):
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout)


def ndjson(resp, stop_after=None):
    """Read an NDJSON stream, optionally hanging up early (to test client disconnect)."""
    out, buf = [], b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                out.append(json.loads(line))
                if stop_after is not None and len(out) >= stop_after:
                    resp.close()
                    return out
    return out


def main():
    port = serve()
    print(f"server up on 127.0.0.1:{port}\n")

    # --- the scene seeds itself, so there is something to trace -------------------------------
    scene = json.loads(get(port, "/api/scene").read())
    check("scene seeds a default object", len(scene.get("objects", [])) >= 1,
          f"got {len(scene.get('objects', []))}")

    # --- A5: the size and sample controls the UI now exposes must reach the renderer ----------
    frames = ndjson(get(port, "/api/photo?w=320&h=180&spp=8&session=t1&fog=1"))
    meta = next((f for f in frames if f.get("type") == "meta"), None)
    check("/api/photo streams a meta header", meta is not None)
    if meta:
        check("A5: requested width honoured", meta["w"] == 320, f"got {meta['w']}")
        check("A5: requested height honoured (not forced 4:3)", meta["h"] == 180, f"got {meta['h']}")
        check("A5: requested spp honoured", meta["spp"] == 8, f"got {meta['spp']}")
    prog = [f for f in frames if f.get("type") == "frame"]
    done = [f for f in frames if f.get("type") == "done"]
    errs = [f for f in frames if f.get("type") == "error"]
    check("progressive frames arrive", len(prog) >= 2, f"{len(prog)} frames")
    check("stream terminates with done", len(done) == 1 and not errs, f"done={len(done)} err={errs}")
    check("frames carry PNG payloads", all(f.get("png") for f in prog))

    # --- clamps: a hostile request must fold, not explode -------------------------------------
    m2 = next(f for f in ndjson(get(port, "/api/photo?w=99999&h=1&spp=999&session=t2"))
              if f["type"] == "meta")
    check("out-of-range size folds to the clamp", m2["w"] == 1280 and m2["h"] == 120,
          f"got {m2['w']}x{m2['h']}")
    check("out-of-range spp folds to the clamp", m2["spp"] == 96, f"got {m2['spp']}")

    # --- G1: THE REGRESSION. post-without-retrace must actually answer -------------------------
    try:
        r = get(port, "/api/photo_post?session=t1&exposure=1.8&sharpen=0.2")
        body = r.read()
        check("G1: /api/photo_post returns 200 (this route shipped as a 500)", r.status == 200,
              f"status {r.status}")
        check("G1: post returns a PNG", body[:8] == b"\x89PNG\r\n\x1a\n", f"first bytes {body[:8]!r}")
        check("G1: post reports what it did", bool(r.headers.get("X-Photo-Post")),
              "missing X-Photo-Post header")
    except urllib.error.HTTPError as e:
        check("G1: /api/photo_post returns 200 (this route shipped as a 500)", False,
              f"HTTP {e.code}: {e.read()[:200]!r}")

    # exposure must actually change the bytes, or the slider is decorative
    a = get(port, "/api/photo_post?session=t1&exposure=0.4").read()
    b = get(port, "/api/photo_post?session=t1&exposure=3.5").read()
    check("G1: exposure changes the image", a != b, "identical PNGs at 0.4 and 3.5")

    # an unknown session must 404, not 500
    try:
        get(port, "/api/photo_post?session=nope")
        check("post with no cached render 404s", False, "expected 404")
    except urllib.error.HTTPError as e:
        check("post with no cached render 404s", e.code == 404, f"got {e.code}")

    # --- A4/F4: cancellation must stop the server, not just the client ------------------------
    traced = {"n": 0}
    real = backend.pt.path_trace if hasattr(backend, "pt") else None
    import holographic_pathtrace as fake_pt
    original = fake_pt.path_trace

    def counting(*a, **k):
        on_p = k.get("on_progress")

        def wrapped(img, done, total):
            traced["n"] = max(traced["n"], done)
            return on_p(img, done, total) if on_p else None
        k["on_progress"] = wrapped
        return original(*a, **k)
    fake_pt.path_trace = counting
    try:
        resp = get(port, "/api/photo?w=240&h=180&spp=96&session=cancel1")
        ndjson(resp, stop_after=3)                       # read a little, then hang up
        early = traced["n"]
        time.sleep(1.2)                                  # give a runaway trace time to show itself
        after = traced["n"]
        check("F4: client disconnect stops the trace server-side",
              after - early <= 6, f"kept tracing: sample {early} -> {after} after hangup")

        # explicit cancel endpoint
        traced["n"] = 0
        resp = get(port, "/api/photo?w=240&h=180&spp=96&session=cancel2")
        ndjson(resp, stop_after=2)
        urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/api/render_cancel?session=cancel2",
                                   method="POST"), timeout=5).read()
        n1 = traced["n"]; time.sleep(0.8); n2 = traced["n"]
        check("A4: /api/render_cancel stops an in-flight render", n2 - n1 <= 6,
              f"sample {n1} -> {n2} after cancel")
    finally:
        fake_pt.path_trace = original

    # --- the diagnostic passes still route to the gbuffer path ---------------------------------
    for aov in ("normal", "depth", "albedo"):
        fs = ndjson(get(port, f"/api/photo?w=160&h=120&spp=8&aov={aov}&session=a_{aov}"))
        ok = any(f.get("type") == "frame" and f.get("png") for f in fs)
        check(f"AOV '{aov}' renders", ok, str([f.get("type") for f in fs]))

    # --- undo/redo over HTTP (F1) --------------------------------------------------------------
    before = len(json.loads(get(port, "/api/scene").read())["objects"])
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/new", method="POST",
                                 data=json.dumps({"kind": "cube"}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()
    added = len(json.loads(get(port, "/api/scene").read())["objects"])
    post_undo = json.loads(urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{port}/api/undo", method="POST",
                               data=b"{}", headers={"Content-Type": "application/json"}),
        timeout=10).read())
    post_redo = json.loads(urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{port}/api/redo", method="POST",
                               data=b"{}", headers={"Content-Type": "application/json"}),
        timeout=10).read())
    check("F1: add then undo removes it", added == before + 1 and len(post_undo["objects"]) == before,
          f"{before} -> {added} -> {len(post_undo['objects'])}")
    check("F1: redo brings it back over HTTP", len(post_redo["objects"]) == added,
          f"got {len(post_redo['objects'])}")

    # --- the other three routes the Render View calls, never executed until now ----------------
    r = get(port, "/api/render?quality=0.5&w=200&eye=2.4,1.7,2.9&target=0,0,0&fov=45&grid=40")
    png = r.read()
    check("/api/render returns a PNG", png[:8] == b"\x89PNG\r\n\x1a\n", f"{png[:16]!r}")
    check("/api/render reports its stats header", bool(r.headers.get("X-Holostuff-Render")),
          "missing X-Holostuff-Render (the client parses this for warnings and perf)")

    # The client's DEFAULT preview is the ADAPTIVE branch (session= + target_fps=), which routes
    # through the frame-budget controller instead of a fixed quality. The route sweep found it had
    # never been executed -- the earlier check above only exercised the manual-quality branch.
    r = get(port, "/api/render?session=adapt1&target_fps=30&eye=2.4,1.7,2.9&target=0,0,0&fov=45&grid=40")
    png = r.read()
    hdr = r.headers.get("X-Holostuff-Render") or ""
    check("/api/render adaptive branch returns a PNG", png[:8] == b"\x89PNG\r\n\x1a\n")
    check("adaptive branch reports its ladder rung", hdr.startswith("AUTO["), f"header was {hdr[:60]!r}")
    stats = json.loads(get(port, "/api/render_stats?session=adapt1").read())
    check("render_stats knows the session the controller just served", "level" in stats, str(stats)[:120])

    # /api/materials is the FIRST call the client makes at boot -- a 500 here is a blank app.
    mats = json.loads(get(port, "/api/materials").read())
    check("/api/materials returns classed materials", bool(mats.get("classes")) or bool(mats),
          str(mats)[:120])

    r = get(port, "/api/render_progressive?session=p1&w=240&eye=2.4,1.7,2.9&target=0,0,0&fov=45&grid=40")
    png = r.read()
    check("/api/render_progressive returns a PNG", png[:8] == b"\x89PNG\r\n\x1a\n")
    for h in ("X-Round", "X-Delta", "X-Converged"):
        check(f"progressive reports {h}", r.headers.get(h) is not None,
              "the client stops resolving on these")

    r = get(port, "/api/render_engine?w=320&h=180&eye=2.4,1.7,2.9&target=0,0,0&fov=45")
    png = r.read()
    check("/api/render_engine returns a PNG", png[:8] == b"\x89PNG\r\n\x1a\n")
    check("/api/render_engine reports its mode", bool(r.headers.get("X-Render-Mode")),
          "the client shows this in the meta line")

    # /api/upscale: B3 turned every render into a blob URL, and this decoder wants base64. Posting
    # img.src sent it the literal string "blob:null/<uuid>" and the 2x button failed.
    import base64 as _b64
    import io as _io
    from PIL import Image as _Image
    _buf = _io.BytesIO()
    _Image.new("RGB", (4, 4), (120, 30, 200)).save(_buf, format="PNG")
    small = _b64.b64encode(_buf.getvalue()).decode()

    def post_json(path, obj):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                     data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except urllib.error.HTTPError as e:          # 4xx carries a JSON error body worth reading
            try:
                return json.loads(e.read())
            except Exception:
                return {"error": f"HTTP {e.code}"}
    up = post_json("/api/upscale", {"image": "data:image/png;base64," + small, "scale": 2.0})
    check("/api/upscale accepts a data URL", "png" in up, str(up)[:140])
    if "png" in up:
        check("/api/upscale actually enlarges", up["w"] == 8 and up["h"] == 8,
              f"4x4 at scale 2 should be 8x8, got {up.get('w')}x{up.get('h')}")

    # The input must be BIGGER than the decoder's default 256px cap, or this proves nothing: the
    # shared decoder shrank every image to 256 first, so "2x" on a 1280px render returned 512 --
    # smaller than what it was handed. A 4x4 probe sails straight past that.
    _big = _io.BytesIO()
    _Image.new("RGB", (400, 300), (10, 200, 90)).save(_big, format="PNG")
    upb = post_json("/api/upscale", {"image": "data:image/png;base64,"
                                     + _b64.b64encode(_big.getvalue()).decode(), "scale": 2.0})
    check("/api/upscale does not shrink a large render before enlarging it",
          upb.get("w") == 800 and upb.get("h") == 600,
          f"400x300 at scale 2 should be 800x600, got {upb.get('w')}x{upb.get('h')}")
    blob = post_json("/api/upscale", {"image": "blob:null/2f1c-not-a-real-image", "scale": 2.0})
    check("/api/upscale names the blob-URL mistake instead of failing cryptically",
          "blob" in str(blob.get("error", "")).lower(), str(blob)[:140])

    # --- the agent manifest: this app's whole point is being drivable by an agent ---------------
    import re as _re
    man = json.loads(get(port, "/api/agent/tools").read())
    tools = {t["name"]: t for t in man["tools"]}
    check("agent manifest lists tools", man.get("count", 0) > 50, f"count={man.get('count')}")

    # routes added in 1.1.x must be discoverable, or an agent cannot reach them
    for n in ("redo", "render_cancel", "photo_post"):
        check(f"manifest exposes '{n}'", n in tools, "missing from /api/agent/tools")

    # An agent picking between 94 tools is choosing on these strings alone. They were, variously:
    # blank ("undo (no description recorded)"), prefixed with an internal ticket id ("G1: re-tonemap
    # the LAST..."), or the first WRAPPED LINE of a docstring, stopping mid-clause.
    TICKET = _re.compile(r"^\s*(?:[A-G]\d+|P\d-\d+)\s*[:.]")
    bad = []
    for n, t in tools.items():
        sm = (t.get("summary") or "").strip()
        if not sm or "(no description recorded)" in sm:
            bad.append(f"{n}: no description")
        elif TICKET.match(sm):
            bad.append(f"{n}: leaks an internal ticket id")
        elif sm.rstrip().endswith(("--", "and", "the", "with", "of", "in", "to", ",")):
            bad.append(f"{n}: stops mid-sentence")
    check("every tool has a usable agent-facing summary", not bad,
          f"{len(bad)} poor: " + "; ".join(bad[:4]))

    # /invoke must actually drive a tool by name
    body = json.dumps({"tool": "scene", "args": {}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/agent/invoke", data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    inv = json.loads(urllib.request.urlopen(req, timeout=15).read())
    check("agent/invoke drives a tool by name", "objects" in json.dumps(inv)[:4000],
          str(inv)[:160])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    print("\nNOTE: the fake engine returns a gradient, not a render. This proves plumbing only --")
    print("      run quality_gate.py against the real engine before shipping a render change.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
