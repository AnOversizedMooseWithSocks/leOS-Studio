"""tests/test_r70.py -- "Each layer is separate."

Devin, working on a multi-layer document:

    I noticed that the flood fill for at least some of the tools, were
    reacting to data on other layers. Light needs to consider all layers
    that it passes through, but paint related things should not be aware
    of other layers by default. Each layer is separate. There are blending
    and other things that involve combining layers, and light as I said
    passes through layers, but as far as painting and filling goes though
    it should be sandboxed on a per layer basis.

He is right, and the audit found it was not one tool but five, with the
defaults disagreeing across the three layers of the stack:

  * `fill_generated` -- engine default "layer", server default "layer",
    and the CLIENT shipping its `all layers` checkbox pre-checked. Every
    real user got composite; only a direct API caller got the sandbox.
  * the textile tool's region fill -- `sample:'composite'` hard-coded in
    the client, with no control at all.
  * `hatch_fill` mode "both" -- the value-aware mode read the flattened
    picture to find the darks, unconditionally, with no parameter. So
    turning the checkbox off made the REGION layer-local while the value
    weighting still came from every layer.
  * `clone` -- always composite, no toggle anywhere. Photoshop's default
    for the clone stamp is Current Layer.
  * the wand / luminance / object selection tools -- always composite. In
    every other editor this is "Sample All Layers", and it ships OFF.

They now share one answer (`Document._sample_px`) and one control, and it
defaults to the layer being painted. What did NOT change: the relief and
lighting stack still sees the whole document, because light does pass
through -- pinned below so a later sweep does not "fix" it.
"""
import os

import numpy as np

UI = os.path.join(os.path.dirname(__file__), "..", "src", "lestudio",
                  "static", "index.html")


def _stack():
    """A red square on the BOTTOM layer, an empty layer above it. Every
    test below asks a tool on the empty layer what it can see."""
    from lestudio import Document
    d = Document(160, 120)
    for l in list(d.layers):            # a new Document ships an opaque
        d.remove_layer(l.id)            # white background; this test is
    base = d.add_layer("base")          # about what a tool can SEE
    base.pixels[30:90, 30:90, :3] = [1.0, 0.0, 0.0]
    base.pixels[30:90, 30:90, 3] = 1.0
    top = d.add_layer("top")
    return d, base, top


# ------------------------------------------------------- the shared primitive

def test_r70_a_sample_px_is_one_answer_with_three_settings():
    d, base, top = _stack()
    lay = d._sample_px(top.id, "layer")
    below = d._sample_px(top.id, "below")
    comp = d._sample_px(top.id, "composite")
    assert lay[60, 60, 3] == 0.0, "the layer sees its own (empty) pixels"
    assert below[60, 60, 0] > 0.9 and below[60, 60, 3] > 0.9, \
        "'below' must see the layer underneath"
    assert comp[60, 60, 0] > 0.9, "'composite' must see the whole stack"
    # and the base layer has nothing below it
    assert d._sample_px(base.id, "below")[60, 60, 3] == 0.0


def test_r70_b_an_unknown_sample_value_means_the_layer():
    d, base, top = _stack()
    for junk in (None, "", "all", "everything", 7):
        assert d._sample_px(top.id, junk)[60, 60, 3] == 0.0, junk


# --------------------------------------------------------------- the bucket

def test_r70_c_a_generated_fill_does_not_see_the_layer_below():
    """The reported bug, at its smallest. Filling the empty top layer must
    flood the WHOLE empty layer, not the red square underneath it."""
    d, base, top = _stack()
    d.fill_generated(top.id, 60, 60, style="hatch", spacing=6,
                     color=(0, 0, 1), seed=2)
    a = np.asarray(top.pixels)[..., 3]
    # ink well outside the red square: the region was the whole empty layer
    assert a[5:20, 5:20].max() > 0.03, \
        "the fill stopped at the shape on the layer below"


def test_r70_d_asking_for_composite_still_works():
    """The wider settings are not removed -- shading onto a clean layer
    over lineart is a real technique. It just has to be asked for."""
    d, base, top = _stack()
    d.fill_generated(top.id, 60, 60, style="hatch", spacing=6,
                     color=(0, 0, 1), sample="composite", seed=2)
    a = np.asarray(top.pixels)[..., 3] > 0.03
    assert a[30:90, 30:90].sum() > 50, "nothing inside the sampled shape"
    assert a[:20, :20].sum() == 0, \
        "sample='composite' should have confined the fill to the red square"


def test_r70_e_the_value_aware_hatch_reads_its_own_layer_too():
    """mode='both' decides dark/mid/light from the pixels it reads. Pre-fix
    that was the composite with no parameter, so the same shading stroke on
    the same layer came out differently depending on other layers."""
    from lestudio import Document

    def hatch(with_base):
        d = Document(160, 120)
        b = d.add_layer("base")
        if with_base:
            b.pixels[..., :3] = 0.0
            b.pixels[..., 3] = 1.0        # solid black under everything
        t = d.add_layer("top")
        d.hatch_fill(t.id, 80, 60, radius=40, mode="both", spacing=7,
                     color=(0, 0, 1), seed=4)
        return np.asarray(t.pixels).copy()

    assert np.array_equal(hatch(True), hatch(False)), \
        "what the hatch painted changed because of a layer it does not paint"


def test_r70_f_the_clone_stamp_defaults_to_the_current_layer():
    d, base, top = _stack()
    # clone from inside the red square to somewhere else on the EMPTY layer
    d.clone(top.id, [[110.0, 60.0], [120.0, 60.0]], [60.0, 60.0], radius=8)
    assert np.asarray(top.pixels)[..., 3].max() < 0.03, \
        "the clone stamp lifted pixels off a layer it was not painting"
    d.clone(top.id, [[110.0, 60.0], [120.0, 60.0]], [60.0, 60.0], radius=8,
            sample="below")
    assert np.asarray(top.pixels)[..., 3].max() > 0.3, \
        "sample='below' must still clone what is underneath"


def test_r70_g_the_wand_selects_the_shape_on_the_active_layer():
    d, base, top = _stack()
    on_top = d.select("color", {"x": 60, "y": 60, "tolerance": 0.1,
                                "layer": top.id})
    # the top layer is uniformly empty, so a wand click takes all of it
    assert float(np.asarray(on_top.data).mean()) > 0.9, \
        "the wand traced a shape that is not on this layer"
    wide = d.select("color", {"x": 60, "y": 60, "tolerance": 0.1,
                              "layer": top.id, "sample": "composite"})
    f = np.asarray(wide.data)
    assert f[60, 60] > 0.5 and f[5, 5] < 0.5, \
        "sample='composite' must still trace the picture"


# ----------------------------------------------------- light is NOT sandboxed

def test_r70_h_light_still_passes_through_every_layer():
    """Devin's own carve-out, pinned so a later sweep does not sandbox it:
    'Light needs to consider all layers that it passes through.'"""
    from lestudio import Document, composite_lit
    d = Document(80, 60)
    b = d.add_layer("base")
    b.pixels[..., :3] = [0.9, 0.9, 0.9]
    b.pixels[..., 3] = 1.0
    t = d.add_layer("glass")
    t.pixels[20:40, 20:40, :3] = [0.2, 0.4, 0.9]
    t.pixels[20:40, 20:40, 3] = 0.5
    lit = composite_lit(d)
    assert lit is not None and np.asarray(lit).shape[:2] == (60, 80)
    # the lit result over the translucent patch must carry BOTH layers
    px = np.asarray(lit)[30, 30, :3]
    assert px.max() > 0.05, "the lighting pass lost the stack"


def test_r70_i_the_composite_is_still_the_composite():
    """Layer combination (blending, the composite itself) is untouched."""
    d, base, top = _stack()
    top.pixels[..., :3] = [0.0, 0.0, 1.0]
    top.pixels[..., 3] = 1.0
    assert np.asarray(d.composite())[60, 60, 2] > 0.9


# -------------------------------------------------------- routes and the UI

def test_r70_j_the_routes_default_to_the_layer_and_refuse_junk():
    from lestudio.server import _sample_mode
    assert _sample_mode(None) == "layer"
    assert _sample_mode("") == "layer"
    assert _sample_mode("composite") == "composite"
    assert _sample_mode("BELOW") == "below"
    for junk in ("all", "everything", 7, {"a": 1}, ["composite"]):
        assert _sample_mode(junk) == "layer", junk


def test_r70_k_the_client_has_one_control_and_it_starts_on_the_layer():
    ui = open(UI).read()
    assert 'id="fiComp"' not in ui, "the old pre-checked all-layers box is back"
    assert "sample:'composite'" not in ui, "a tool is hard-coding composite again"
    i = ui.index('id="bSample"')
    block = ui[i:i + 200]
    assert 'data-value="layer"' in block, \
        "the sample control does not default to the active layer"
    # and the cycle starts there too, so a reload always lands on the layer
    cyc = ui[ui.index("const SAMPLE_CYCLE="):]
    cyc = cyc[:cyc.index("]]") + 2]
    assert cyc.index("'layer'") < cyc.index("'below'") < cyc.index("'composite'")
    assert "function sampleMode()" in ui
    # and every sampling tool reads it
    assert ui.count("sampleMode()") >= 6, ui.count("sampleMode()")
    assert "body.sample=sampleMode()" in ui           # selection tools
    assert "sample:tool==='clone'?sampleMode():undefined" in ui


def test_r70_l_a_wider_setting_is_announced_in_the_status_bar():
    ui = open(UI).read()
    ctx = ui[ui.index("function paintContext()"):]
    ctx = ctx[:ctx.index("\nfunction ")]
    assert "reading ALL layers" in ctx and "reading this layer + below" in ctx


def test_r70_m_the_fill_route_honours_the_sample_it_is_given():
    from lestudio.server import app
    c = app.test_client()
    c.post("/api/new", json={"width": 160, "height": 120})
    st = c.get("/api/state").get_json()
    lid = st["doc"]["layers"][-1]["id"] if "doc" in st else st["layers"][-1]["id"]
    r = c.post("/api/fill", json={
        "layer": lid, "x": 60, "y": 60, "tolerance": 0.12, "contiguous": True,
        "source": {"type": "generated", "style": "hatch", "spacing": 6,
                   "color": [0, 0, 1], "sample": "nonsense"}})
    assert r.status_code == 200, r.get_data(as_text=True)
