"""tests/test_r34.py -- P1.4 (layer transform as an op) and P0.6 (float
policy pinned by a cross-process golden render).

A layer transform is a pure affine {sx, sy, deg, dx, dy} with the pivot
frozen at journal time -- it journals pixel-free now. Its old record
snapshotted and dirtied the ENTIRE document.

Float policy (P0.6): bit-exact same-machine replay is REQUIRED; the pin
is a golden document -- material stroke, media stroke, fill, transform,
smudge -- rendered in two SEPARATE processes with no PYTHONHASHSEED,
compared by crc32. Cross-machine gets a tolerance suite, not this pin.
"""
import numpy as np


def test_r34_layer_transform_is_a_journaled_op():
    from lestudio import Document
    d = Document(160, 120)
    d.layers[0].pixels[...] = 0.0
    lid = d.layers[0].id
    d.paint(lid, [[30, 30], [120, 80]], color=(1, 0.4, 0.1), radius=8.0,
            media="oil", load=0.7)
    d.transform("layer", lid, sx=0.8, sy=0.8, deg=20, dx=10, dy=-5)
    ent = d._undo[-1]
    assert ent[0] == "Transform layer" and ent[1].get("rerender") == [lid]
    assert all(r[7] is None for r in ent[1].get("layers", [])), \
        "a layer transform must journal, not snapshot the document"
    k = d.strokes[-1]
    assert k["brush"].get("op") == "xform" and "px" in k["brush"], \
        "the op must freeze its pivot at journal time"
    assert d.replay_is_faithful(lid)
    before = d.layer(lid).pixels.copy()
    hm = d.layer(lid).height_map.copy()
    assert d.undo() and d.redo()
    after = d.layer(lid).pixels
    cov = np.maximum(before[..., 3:4], after[..., 3:4])
    assert float(max(np.abs((before[..., :3] - after[..., :3]) * cov).max(),
                     np.abs(before[..., 3] - after[..., 3]).max())) <= 2e-3
    assert np.allclose(d.layer(lid).height_map, hm, atol=1e-5), \
        "the body must travel with the pigment through undo/redo"


def test_r34_other_layers_survive_a_transform_undo():
    """The old full-doc snapshot hid a class of bug the scoped op must not
    reintroduce: transforming one layer then undoing must leave every
    OTHER layer untouched, and must not dirty them."""
    from lestudio import Document
    d = Document(120, 90)
    d.layers[0].pixels[...] = 0.0
    a = d.layers[0].id
    lb = d.add_layer("b")
    d.paint(a, [[10, 20], [100, 20]], color=(1, 0, 0), radius=5.0)
    d.paint(lb.id, [[10, 60], [100, 60]], color=(0, 0, 1), radius=5.0)
    other = d.layer(lb.id).pixels.copy()
    d.transform("layer", a, deg=30)
    assert d.undo()
    assert np.array_equal(d.layer(lb.id).pixels, other)
    assert d.replay_is_faithful(lb.id), \
        "transforming layer A must not dirty layer B"


def test_r34_golden_render_is_process_independent():
    """P0.6/P0.1 acceptance, end to end: the SAME edits in two separate
    python processes -- each with its own random hash salt -- produce
    byte-identical pixels. Textured material, living media, a fill, a
    transform and a smudge all run through their seeds."""
    import subprocess
    import sys
    import os
    prog = r'''
import sys, zlib
import numpy as np
sys.path.insert(0, %r); sys.path.insert(0, %r)
from lestudio import Document
d = Document(128, 96)
d.layers[0].pixels[...] = 0.0
lid = d.layers[0].id
d.paint(lid, [[20, 20], [100, 60]], color=(0.8, 0.6, 0.2), radius=9.0,
        material="gold", load=0.7)
l2 = d.add_layer("wash")
d.edit_layer(l2.id, vol_kind="smoke", thickness=5.0)
d.paint(l2.id, [[40, 50], [44, 52]], color=(0.9, 0.9, 0.9), radius=10.0)
d.media_step(l2.id, 2)
content = np.zeros((96, 128, 4), np.float32); content[...] = (0, 0.5, 0.9, 1)
d.flood_fill(lid, 2, 90, content, tolerance=0.05,
             spec={"type": "color", "color": [0, 0.5, 0.9]})
d.transform("layer", lid, deg=15, dx=4, dy=2)
d.smudge(lid, [[30, 40], [90, 44]], radius=12.0, strength=0.7)
comp = d.composite()
print(zlib.crc32(np.ascontiguousarray(comp).tobytes()))
''' % (os.environ.get("LECORE_PATH", "/root/work/lecore_main"), "src")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    outs = [subprocess.check_output([sys.executable, "-c", prog],
                                    env=env, cwd=os.path.join(
                                        os.path.dirname(__file__), ".."))
            for _ in range(2)]
    assert outs[0].strip() == outs[1].strip(), \
        "two fresh processes rendered different pixels: %s vs %s" % (
            outs[0].strip(), outs[1].strip())
