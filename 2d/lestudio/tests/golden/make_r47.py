"""Author golden_r47.lews: a journal-first corpus for the R47 ops.

scribble/hatch_fill land as ordinary `paint` strokes (already covered),
so what this corpus PINS is `pwarp` -- plus a gated scribble and a
value-aware hatch so their generated stroke sets stay bit-stable too.
Run from the repo root:

    PYTHONPATH=/root/work/lecore_main:src python3 tests/golden/make_r47.py
"""
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, "src")
from lestudio import Document, NodeGraph, save_workspace  # noqa: E402

GOLD = os.path.dirname(os.path.abspath(__file__))

d = Document(200, 150)
L = d.add_layer("gen").id
sel = d.select("ellipse", {"x0": 30, "y0": 25, "x1": 170, "y1": 125},
               feather=10)
d.scribble(L, 100, 75, radius=60, curl=0.6, thickness=1.4,
           color=(0.1, 0.1, 0.1), seed=47, selection=sel.id)
d.hatch_fill(L, 100, 75, radius=55, mode="both", spacing=6, seed=48)
Lw = d.add_layer("warp").id
d.paint(Lw, [[50, 40, 1], [150, 40, 1], [150, 110, 1], [50, 110, 1],
             [50, 40, 1]], color=(0.15, 0.1, 0.1), radius=3)
d.warp_perspective(Lw, [60, 30, 160, 45, 150, 120, 45, 105])

blob = save_workspace({d.id: d}, {d.id: NodeGraph(d)}, d.id,
                      cache_pixels=False)
open(os.path.join(GOLD, "golden_r47.lews"), "wb").write(blob)

# pin the crc THROUGH a reload (what the test does), not the live doc
from lestudio import load_workspace  # noqa: E402
docs, graphs, active, extras = load_workspace(blob)
comp = docs[active].composite()
crc = zlib.crc32(np.ascontiguousarray(comp).tobytes())

man_p = os.path.join(GOLD, "manifest.json")
man = json.load(open(man_p))
man["golden_r47.lews"] = {"composite_crc": crc,
                          "ops": ["paint", "pwarp"], "w": 200, "h": 150}
json.dump(man, open(man_p, "w"), indent=1)
print("golden_r47.lews written, crc", crc)
