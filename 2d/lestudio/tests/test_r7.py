"""R7 pins: the Layer options dialog.

The layers panel's "Selected layer" area had grown to ~25 rows -- blend,
opacity and type sitting shoulder to shoulder with IOR sliders, cook
buttons and herding fields. R7 moves the advanced rows into a tabbed
dialog (#layerDlg: Shape & pose / Optics & material / Living media /
Walls) and leaves the panel with the everyday controls plus one ⚙
button. The rows were MOVED, never cloned: same DOM ids, so every
id-bound handler, refresh() sync and older test pin keeps working.

The dialog header carries a horizontal strip of engine-rendered
previews of all 11 layer types (inlined from type_previews.json);
clicking one drives the shipped lVol change handler.
"""
import os

UI_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                       "static", "index.html")


def _ui():
    return open(UI_PATH).read()


def test_r7_layer_dialog_markup():
    """(a) The dialog, its four tabs, and the panel's ⚙ entry point."""
    ui = _ui()
    for frag in ('id="layerDlg"', 'id="layerDlgBack"', 'id="layerDlgBtn"',
                 'id="layerDlgTabs"', 'id="layerDlgBody"', 'id="layerDlgClose"',
                 'data-ltab="shape"', 'data-ltab="optics"',
                 'data-ltab="media"', 'data-ltab="walls"',
                 ">Shape &amp; pose</button>", ">Optics &amp; material</button>",
                 ">Living media</button>", ">Walls</button>",
                 "⚙ Layer options…"):
        assert frag in ui, frag
    # modal-scoped Esc, the miniForm way, and a backdrop that closes
    assert "layerDlgEsc" in ui and "addEventListener('keydown',layerDlgEsc,true)" in ui
    assert "removeEventListener('keydown',layerDlgEsc,true)" in ui
    # the play controls stay in the panel: Cook row before the dialog markup
    assert ui.index('id="rowCook"') < ui.index('id="layerDlgBack"')
    assert "they're the play controls" in ui
    # empty state for non-media layers
    assert 'id="mediaEmpty"' in ui and "no living medium" in ui


def test_r7_moved_ids_exactly_once():
    """(b) Every moved control keeps its id, and exactly one copy exists --
    a clone would leave two elements answering one id and the handlers
    editing the invisible twin."""
    ui = _ui()
    for cid in ("lThickMM", "lBgKind", "lBgColor", "lBgTex",
                "plX", "plY", "plScale", "plRot",
                "lZOff", "lFlipX", "lFlipY", "lTiltX", "lTiltY",
                "lCurve", "lCurveAxis", "curveRamp", "lDome", "domeRamp",
                "lIor", "lDen", "lSoak", "lPaintGloss", "lEmit", "lEmitC",
                "lOptical", "lRelief", "lRefl", "lDisp",
                "lMediaRes", "lMediaTime", "lMediaRate",
                "lFieldSrc", "lFieldMode", "lFieldStr", "fieldRows",
                "wallSlots", "layerDlgBtn",
                "lCookN", "lLive", "lFlowVis"):
        n = ui.count('id="%s"' % cid)
        assert n == 1, "%s appears %d times" % (cid, n)


def test_r7_type_previews_inlined():
    """(c, static half) All 11 engine-rendered type previews are inlined
    and the strip drives the shipped lVol handler (set value + dispatch
    change -- no second code path for changing a layer's type)."""
    import json
    import re
    ui = _ui()
    m = re.search(r"const TYPE_PREVIEWS=(\{.*?\});\n", ui)
    assert m, "TYPE_PREVIEWS object missing"
    d = json.loads(m.group(1))
    assert sorted(d) == sorted(["none", "water", "glass", "absorb", "fog",
                                "puff", "soak", "air", "inkwater", "smoke",
                                "fire"]), sorted(d)
    for k, v in d.items():
        assert v.startswith("data:image/png;base64,"), k
    # the json source of truth stays in the repo for regeneration
    src = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                      "type_previews.json")))
    assert src == d, "inlined previews drifted from type_previews.json"
    # vol_kind → lVol option-value mapping and the change dispatch
    assert "LVOL_OF_KIND={none:'',inkwater:'ink'}" in ui
    assert "vol.dispatchEvent(new Event('change'))" in ui


def test_r7_dialog_live_in_browser():
    """(c, live half) The strip renders 11 entries, clicking one changes
    the layer's type through the real handler and the highlight follows;
    a living-media layer auto-opens on the Living media tab; the
    non-media empty state shows; the panel keeps room to spare."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("SKIP: playwright not installed in this interpreter")
        return
    import threading
    import time
    import socket
    import warnings
    warnings.filterwarnings("ignore")
    from lestudio.server import app, WS
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    th = threading.Thread(target=lambda: app.run(port=port,
                                                 use_reloader=False),
                          daemon=True)
    th.start()
    time.sleep(1.0)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        try:
            pg = b.new_page(viewport={"width": 1366, "height": 768})
            pg.goto("http://127.0.0.1:%d" % port)
            pg.wait_for_timeout(1300)
            lid = WS.doc.layers[0].id
            pg.evaluate("sel='%s'; refresh()" % lid)
            pg.wait_for_timeout(500)
            pg.evaluate("$('layerDlgBtn').click()")
            pg.wait_for_timeout(250)
            # 11 previews, one per type, everyone titled
            n = pg.evaluate("document.querySelectorAll('#typeStrip .tprev')"
                            ".length")
            assert n == 11, n
            assert pg.evaluate("[...document.querySelectorAll("
                               "'#typeStrip .tprev')]"
                               ".every(c=>c.title.length>10)")
            # flat layer: current type highlighted, empty state on media tab
            assert pg.evaluate("document.querySelector('#typeStrip .tprev.on')"
                               ".dataset.v") == ""
            pg.evaluate("[...document.querySelectorAll("
                        "'#layerDlgTabs button')]"
                        ".find(b=>b.dataset.ltab==='media').click()")
            pg.wait_for_timeout(100)
            assert pg.evaluate("$('mediaEmpty').offsetWidth>0")
            assert pg.evaluate("$('lMediaRate').offsetWidth===0")
            # click the water preview: the REAL lVol handler runs
            pg.evaluate("[...document.querySelectorAll("
                        "'#typeStrip .tprev')]"
                        ".find(c=>c.dataset.v==='water').click()")
            for _ in range(40):                      # poll, not a fixed nap
                if WS.doc.layer(lid).vol_kind == "water":
                    break
                pg.wait_for_timeout(150)
            assert WS.doc.layer(lid).vol_kind == "water", \
                WS.doc.layer(lid).vol_kind
            pg.wait_for_timeout(300)                 # let refresh() repaint
            assert pg.evaluate("document.querySelector('#typeStrip .tprev.on')"
                               ".dataset.v") == "water", \
                "the highlight follows the type"
            # a living medium: empty state yields to the rows, and a fresh
            # session auto-opens on the Living media tab
            pg.evaluate("[...document.querySelectorAll("
                        "'#typeStrip .tprev')]"
                        ".find(c=>c.dataset.v==='ink').click()")
            for _ in range(40):
                if WS.doc.layer(lid).vol_kind == "inkwater":
                    break
                pg.wait_for_timeout(150)
            pg.wait_for_timeout(600)                 # refresh() + row gating
            assert pg.evaluate("$('mediaEmpty').offsetWidth===0")
            assert pg.evaluate("$('lMediaRate').offsetWidth>0")
            assert pg.evaluate("$('lLive').offsetWidth>0"), \
                "the Cook/Live row stays in the PANEL, visible for media"
            pg2 = b.new_page(viewport={"width": 1366, "height": 768})
            pg2.goto("http://127.0.0.1:%d" % port)
            pg2.wait_for_timeout(1300)
            pg2.evaluate("sel='%s'; refresh()" % lid)
            pg2.wait_for_timeout(500)
            pg2.evaluate("$('layerDlgBtn').click()")
            pg2.wait_for_timeout(200)
            assert pg2.evaluate("document.querySelector("
                                "'#layerDlgTabs button.on').dataset.ltab") \
                == "media", "living-media layers open on their tab"
            pg2.close()
            pg.close()
        finally:
            b.close()
