"""tests/test_r19.py -- front-end/back-end splat parity (R19).

Devin: 'there are some things that might go faster if the front end was
able to produce identical output to the back end, and we could just pass
seeds around... anything we can generate instead of store should help.'
The splat CODE is that wire format: ~4 KB regenerates the picture. Pinned
here: the closed-form (peak-colour) render equals the server's unit-norm
basis render, and the BROWSER (the UI's own renderSplatCode) reproduces
the server's pixels from the code alone."""
import numpy as np
import pytest


def _need_lecore():
    try:
        from holographic.rendering.holographic_splat import splat_fit  # noqa
    except Exception:
        pytest.skip("leCore not on the path")


def _painted_client():
    import lestudio.server as srv
    c = srv.app.test_client()
    c.post("/api/new", json={"name": "pc", "width": 200, "height": 140,
                             "background": [0.85, 0.88, 0.92]})
    lid = srv.DOC.layers[0].id
    for i in range(3):
        c.post("/api/paint", json={"layer": lid,
                                   "points": [[12, 22 + 34 * i],
                                              [188, 30 + 34 * i]],
                                   "color": [0.15 + 0.25 * i, 0.45,
                                             0.85 - 0.25 * i],
                                   "radius": 11, "record": True})
    return srv, c


def test_r19_peak_code_is_the_same_picture():
    """The closed form any client evaluates must equal the server's
    unit-norm basis render to float precision -- no approximation."""
    _need_lecore()
    pytest.importorskip("flask")
    srv, c = _painted_client()
    j = c.get("/api/splatify/code?k=48").get_json()
    assert j.get("ok")
    code = j["code"]
    sh, sw = code["shape"]
    # server-side truth: unit-norm basis
    from holographic.rendering.holographic_splat import _gaussian
    truth = np.zeros((sh, sw, 3))
    for (cy, cx, sg), col in zip(code["splats"], code["colors"]):
        truth += _gaussian((sh, sw), cy, cx, sg)[..., None] * \
            np.asarray(col)[None, None, :]
    # client closed form: peak colours, no basis normalisation needed
    ys, xs = np.mgrid[0:sh, 0:sw].astype(float)
    client = np.zeros((sh, sw, 3))
    for (cy, cx, sg), pk in zip(code["splats"], code["peak"]):
        e = np.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / (sg * sg))
        client += e[..., None] * np.asarray(pk)[None, None, :]
    assert float(np.abs(truth - client).max()) < 1e-3, \
        "peak-space code must be the same picture as the basis render"


def test_r19_browser_renders_the_code_to_the_servers_pixels():
    """THE parity pin: the UI's own renderSplatCode, in real Chromium,
    fed only the ~4 KB code, must reproduce the server's render."""
    _need_lecore()
    pytest.importorskip("flask")
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        pytest.skip("playwright not installed")
    import socket
    import threading
    import time
    import lestudio.server as srv
    srv.app.logger.disabled = True
    _, c = _painted_client()
    j = c.get("/api/splatify/code?k=40").get_json()
    code = j["code"]
    sh, sw = code["shape"]
    ys, xs = np.mgrid[0:sh, 0:sw].astype(float)
    truth = np.zeros((sh, sw, 3))
    for (cy, cx, sg), pk in zip(code["splats"], code["peak"]):
        e = np.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / (sg * sg))
        truth += e[..., None] * np.asarray(pk)[None, None, :]
    truth8 = np.clip(np.round(np.clip(truth, 0, 1) * 255), 0, 255)

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    th = threading.Thread(target=lambda: srv.app.run(port=port,
                                                     use_reloader=False),
                          daemon=True)
    th.start()
    time.sleep(1.0)
    import json as _json
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        try:
            pg = b.new_page()
            pg.goto("http://127.0.0.1:%d" % port)
            pg.wait_for_timeout(900)
            got = pg.evaluate("""(code)=>{
                const cv=renderSplatCode(code);
                const d=cv.getContext('2d')
                          .getImageData(0,0,cv.width,cv.height).data;
                const out=[];
                for(let i=0;i<d.length;i+=4) out.push(d[i],d[i+1],d[i+2]);
                return out; }""", code)
        finally:
            b.close()
    got = np.asarray(got, float).reshape(sh, sw, 3)
    diff = np.abs(got - truth8)
    assert float(diff.max()) <= 2.0, \
        "browser render diverged from the server (max %d/255)" % diff.max()


def test_r19_ui_has_the_preview():
    import os
    import lestudio.server as srv
    ui = open(os.path.join(os.path.dirname(srv.__file__), "static",
                           "index.html")).read()
    for needle in ("renderSplatCode", "renderSplatCodeGL",
                   "/api/splatify/code", "lcSplPrev"):
        assert needle in ui, needle
