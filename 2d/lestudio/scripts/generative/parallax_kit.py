"""Parallax GIF the stable way (R28).

Devin: 'stuff pops in and out of visibility -- the layers should just
move, not the contents.' The engine was innocent (full-res persp frames
differ smoothly); the popping came from the GIF pipeline: the server's
quick display resize plus PIL quantizing EVERY frame to its own 256-color
palette, so dim 1-2px window lights fell in and out of each frame's
palette. The cure: fetch frames at FULL resolution, LANCZOS-downscale
client-side, and quantize every frame with ONE shared palette built from
a middle frame. Phases avoid duplicate endpoints (no dedup stutter)."""
import io
import json
import math
import urllib.request

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:5050"


def _get(p):
    return urllib.request.urlopen(BASE + p, timeout=300).read()


def _post(p, o):
    r = urllib.request.Request(BASE + p, json.dumps(o).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def parallax_gif(out_path, n_frames=24, amp=90.0, width=560, duration=90):
    layers = json.loads(_get("/api/state"))["layers"]
    # Depth from each layer's accumulated TOP-surface z, not its list
    # index: a zero-thickness layer (an underpainting riding the pane
    # beneath) then gets exactly its pane's offset and can never slide
    # behind it. Pane order is preserved for any |ph|*amp < total z.
    tops, z = [], 0.0
    for l in layers:
        z += float(l.get("thickness", 0.0) or 0.0)
        tops.append(z)
    ztot = max(z, 1.0)
    if amp >= ztot:
        amp = 0.9 * ztot                    # never allow a pane reorder
    _post("/api/view3d", {"mode": "persp"})
    frames = []
    try:
        for i in range(n_frames):
            # offset half a step: no frame sits exactly at phase 0 twice
            ph = math.sin(2 * math.pi * (i + 0.5) / n_frames)
            for k, l in enumerate(layers):
                depth = tops[k] / ztot
                _post("/api/layer", {"action": "edit", "id": l["id"],
                                     "z_off": ph * amp * (depth - 0.5)})
            im = Image.open(io.BytesIO(_get("/api/composite.png")))
            im = im.convert("RGB")
            h = int(round(im.height * width / im.width))
            frames.append(im.resize((width, h), Image.LANCZOS))
    finally:
        for l in layers:
            _post("/api/layer", {"action": "edit", "id": l["id"],
                                 "z_off": 0.0})
        _post("/api/view3d", {"mode": "flat"})
    # ONE palette for every frame, learned from a representative frame
    master = frames[len(frames) // 4].quantize(
        colors=255, method=Image.MEDIANCUT)
    pal = [f.quantize(palette=master, dither=Image.NONE)
           for f in frames]
    pal[0].save(out_path, save_all=True, append_images=pal[1:],
                duration=duration, loop=0)
    return len(pal)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/parallax.gif"
    print("frames", parallax_gif(out))
