"""tests/test_r72.py -- the busy indicator has to come DOWN.

Devin, on R71's new indicator:

    the indicator doesn't always clear after the pending work completes.
    Make sure that clears out after a tool completes so we know when it's
    done working.

It never cleared, and in fact it was never hidden at all: the badge sat on
screen from the moment the page loaded, reading "working...", because
`el.hidden = true` hides an element through the USER-AGENT stylesheet's
`[hidden]{display:none}` -- which any author rule with an ID selector
outranks, and `#busyBadge{...display:flex...}` is exactly that. The
attribute was being set correctly the whole time; the pixels never moved.

R71's own pin asserted on `el.hidden` -- the attribute -- so it passed
while the bug shipped. THE LESSON, and the reason these tests are written
the way they are: an indicator test must measure what the person sees. Two
of the three tests below drive a real browser and read
`getBoundingClientRect()`; the source-level test exists only to keep the
CSS rule that makes `hidden` mean what it says.

Two smaller leaks were fixed at the same time, both of which would have
stuck the badge until the watchdog:
  * `await r.json()` throws on an empty, truncated or not-really-JSON
    body, and the `busyStop()` after it never ran. It is in a `finally`
    now, so the badge comes down on every exit from api(), including the
    ones that throw.
  * the watchdog itself was 90 seconds -- long enough to conclude the app
    has hung. It is 20 s, past any real tool (the slowest measured is a
    ~3 s textile fill), and it says what happened rather than vanishing.
"""
import os
import socket
import threading
import time

UI = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                  "static", "index.html")


def _serve():
    from werkzeug.serving import make_server
    from lestudio.server import app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.7)
    return srv, port


def _ready(pg):
    for _ in range(400):
        if pg.evaluate("typeof canvasReady==='function'&&canvasReady()"):
            break
        pg.wait_for_timeout(50)
    pg.wait_for_timeout(500)


# what the PERSON sees: the rendered box, not the attribute
_SHOWN = """()=>{const e=$('busyBadge');const r=e.getBoundingClientRect();
  return {n:BUSY.n, px:Math.round(r.width*r.height),
          display:getComputedStyle(e).display,
          cur:$('canvasPane').classList.contains('busy')};}"""


def test_r72_a_the_badge_is_invisible_at_rest_and_after_the_work():
    """The bug, measured the only way that would have caught it. At rest
    the badge must occupy NO PIXELS -- `hidden` set on an element an author
    rule gives `display:flex` does nothing at all."""
    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    srv, port = _serve()
    try:
        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            try:
                pg = b.new_page(viewport={"width": 1280, "height": 800})
                pg.goto("http://127.0.0.1:%d" % port)
                _ready(pg)

                rest = pg.evaluate(_SHOWN)
                assert rest["px"] == 0 and rest["display"] == "none", \
                    "the badge is on screen before anything has happened: %r" % rest
                assert not rest["cur"], "the progress cursor is on at rest"

                pg.evaluate("()=>{busyStart('/api/hatchfill','stitching');busyShow();}")
                pg.wait_for_timeout(150)
                on = pg.evaluate(_SHOWN)
                assert on["px"] > 500 and on["cur"], \
                    "the badge does not actually appear: %r" % on

                pg.evaluate("()=>busyStop()")
                pg.wait_for_timeout(250)
                off = pg.evaluate(_SHOWN)
                assert off["px"] == 0 and off["n"] == 0 and not off["cur"], \
                    "the badge did not come down: %r" % off
            finally:
                b.close()
    finally:
        srv.shutdown()


def test_r72_b_every_tool_leaves_the_badge_clear():
    """Devin's report is about tools, so this drives them. After each one
    settles the badge must be gone, the counter at zero and the progress
    cursor off -- including on the paths that used to leak: a 400 response
    and a body that cannot be parsed."""
    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    srv, port = _serve()
    stuck = []
    try:
        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            try:
                pg = b.new_page(viewport={"width": 1280, "height": 800})
                pg.goto("http://127.0.0.1:%d" % port)
                _ready(pg)
                box = pg.eval_on_selector(
                    "#view", "e=>{const r=e.getBoundingClientRect();"
                             "return{x:r.x,y:r.y,w:r.width,h:r.height}}")

                def drag(frac, n=10):
                    x0 = box["x"] + box["w"] * (0.22 + frac)
                    y0 = box["y"] + box["h"] * 0.3
                    pg.mouse.move(x0, y0)
                    pg.mouse.down()
                    for k in range(1, n):
                        pg.mouse.move(x0 + k * 7, y0 + k * 7)
                        pg.wait_for_timeout(8)
                    pg.mouse.up()

                def settled(tag, wait):
                    pg.wait_for_timeout(wait)
                    st = pg.evaluate(_SHOWN)
                    if not (st["n"] == 0 and st["px"] == 0 and not st["cur"]):
                        stuck.append((tag, st))
                    pg.evaluate("()=>{BUSY.n=0;busyHide();}")   # don't mask the rest

                pg.evaluate("()=>{$('addLayer').click();}")
                settled("add layer", 1200)
                for tag, setup, frac, wait in (
                        ("brush", "()=>setTool('brush')", 0.00, 2500),
                        ("erase", "()=>setTool('erase')", 0.03, 2500),
                        ("scribble", "()=>setTool('scribble')", 0.08, 5000),
                        ("hatch", "()=>setTool('hatch')", 0.14, 5000),
                        ("textile",
                         "()=>{setTool('textile');$('txtRegion').checked=false;"
                         "txtRegionRows();}", 0.20, 9000)):
                    pg.evaluate(setup)
                    pg.wait_for_timeout(200)
                    drag(frac)
                    settled(tag, wait)

                # the two paths that used to leak the counter
                pg.evaluate("()=>api('/api/timeline',{...J,body:JSON.stringify("
                            "{action:'range',lo:0,hi:1e9})})")
                settled("a 400 response", 2500)
                pg.evaluate("""()=>{const of=window.fetch;window.fetch=function(u,o){
                    if((''+u).includes('/api/layer'))return Promise.resolve(
                      new Response('not json',{status:200,
                        headers:{'content-type':'application/json'}}));
                    return of.apply(this,arguments);};}""")
                pg.evaluate("()=>api('/api/layer',{...J,body:JSON.stringify("
                            "{action:'add'})}).catch(()=>{})")
                settled("an unparseable body", 2500)
            finally:
                b.close()
    finally:
        srv.shutdown()
    assert not stuck, "the badge was still showing after: %r" % (stuck,)


def test_r72_c_overlapping_requests_retire_the_right_one():
    """R71 counted requests but kept only ONE label, set when the count went
    0 -> 1. So with two in flight, the first finishing left the badge naming
    work that was already done while the second was still running -- which
    is its own version of "it doesn't clear": the text is stale even though
    the spinner is honest. Each request carries its own token now."""
    import pytest
    pw = pytest.importorskip("playwright.sync_api")
    srv, port = _serve()
    try:
        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            try:
                pg = b.new_page(viewport={"width": 1280, "height": 800})
                pg.goto("http://127.0.0.1:%d" % port)
                _ready(pg)
                r = pg.evaluate("""()=>{
                  const a=busyStart('/api/fill','filling');
                  const c=busyStart('/api/hatchfill','stitching');
                  busyShow();
                  const two=$('busyText').textContent;
                  busyStop(a);                      // the FIRST one finishes
                  const one=$('busyText').textContent;
                  const stillUp=!$('busyBadge').hidden;
                  busyStop(c);
                  const px=(()=>{const q=$('busyBadge').getBoundingClientRect();
                                 return Math.round(q.width*q.height);})();
                  return {two,one,stillUp,n:BUSY.n,px};}""")
                assert r["two"].startswith("stitching"), r
                assert r["one"].startswith("stitching"), \
                    "the badge named work that had already finished: %r" % r
                assert r["stillUp"], "it came down while work was still running"
                assert r["n"] == 0 and r["px"] == 0, r
            finally:
                b.close()
    finally:
        srv.shutdown()


def test_r72_d_hidden_actually_hides_it():
    """The one-line cause, kept honest. `[hidden]` is a UA-stylesheet rule
    and `#busyBadge{display:flex}` outranks it, so the element needs its
    own rule or `el.hidden` is decoration."""
    ui = open(UI).read()
    assert "#busyBadge[hidden]{display:none !important}" in ui
    # and the badge must come down on EVERY exit from api(), not just the
    # happy one: r.json() throws on a body that is not what it claims
    api = ui[ui.index("async function api(p,opt){"):]
    api = api[:api.index("\nconst apiQuiet=")]
    assert "}finally{" in api and "busyStop(_btok)" in api
    assert api.count("busyStop(_btok)") >= 2, \
        "only one exit from api() lowers the badge"
    # a watchdog someone would still be waiting for is no watchdog
    w = ui[ui.index("function busyWatchdog(){"):]
    w = w[:w.index("\nfunction ")]
    ms = int(w.split("}, ")[1].split(")")[0])
    assert 5000 <= ms <= 30000, "the watchdog waits %d ms" % ms
