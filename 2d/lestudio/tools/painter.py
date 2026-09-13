#!/usr/bin/env python3
"""tools/painter.py -- the painter's kit for agents and swarm painters.

Every scripted painter in R51, R61 and R64 re-wrote its own brush module,
and every one of them re-hit the same traps. This is the one to import
instead. The laws it enforces were each paid for with a ruined pass:

  COVERAGE   Never lay a mass as world-spaced strokes: world spacing does
             not guarantee SCREEN overlap and the underpainting comes out
             striped -- banding that bleeds through every layer above it.
             fill() lays scanline runs with step < radius, and refuses a
             step that is not.
  GRADIENTS  A colour function that only sees a run's midpoint cannot make
             a horizontal gradient inside one run, so painters sliced every
             mass into vertical bands by hand and got scallops where the
             bands' soft caps met. fill() splits runs into short segments
             and asks color_at(x, y) for EACH, so a gradient is a gradient.
  ONCE       Low-opacity build-up passes have no way to tell they already
             happened; a script re-run by accident doubled every one and
             washed out a window. passes are NAMED and applied once.
  CONTEXT    You paint ON something. below() is what is under your layer;
             above() is what will be drawn over it. Sampling the whole
             composite to repair your own layer paints the layers above
             yours into it -- that is how a flat wall-coloured rectangle
             ended up behind a bowl of lemons.
  ERASE      Take a mistake OUT (erase_rect) instead of painting over it.
  OWNERSHIP  A 403 with can_request is the R63 contract: never retried,
             recorded in .refusals, and ask_access() is how you respond.

The sage's painting law travels with it: masses back to front, one
committed light, darks warmer and more saturated than the lights,
near-white reserved for the focal, and dark-on-dark cannot read.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


class Refused(Exception):
    """A write the ownership gate turned down (R63 shape in .refusal)."""

    def __init__(self, refusal):
        super().__init__(refusal.get("error", "refused"))
        self.refusal = refusal


class Painter:
    def __init__(self, who, base="http://127.0.0.1:5050", name=None, seed=0,
                 canvas=None, dry=False):
        """`dry=True` REHEARSES: every colour function is called, every
        stroke is built and validated, and nothing is sent or painted.

        A painting pass is minutes of engine time, so a colour function
        that raises on one awkward coordinate costs the whole pass to find
        -- and fill() deliberately asks for points outside the shape and
        past the edges of the canvas, which is exactly where a hand-written
        value function falls over. Rehearsing the whole picture takes about
        twenty seconds and answers the only question worth asking first:
        does every stroke in it survive being built? `p.counts` holds the
        strokes per layer afterwards, and `p.sent` stays 0."""
        self.base = base.rstrip("/")
        self.who = who
        self.rnd = random.Random(seed or abs(hash(who)) % 10 ** 6)
        self.batch = []
        self.refusals = []
        self.skipped_passes = []
        self._pass = None
        self._force = False
        self._pass_run = None
        self._paint = {}                       # the medium in force
        self.dry = bool(dry)
        self.counts = {}
        self.sent = 0
        if self.dry:
            w, h = canvas or (1000, 700)
            self.W, self.H = int(w), int(h)
            self.lids = {}
            return
        if name:
            self.post("/api/presence/name", {"name": name})
        st = self.state()
        self.W = int(st["width"])
        self.H = int(st["height"])
        self.lids = {l["name"]: l["id"] for l in st["layers"]}

    # ----------------------------------------------------------- plumbing
    def _call(self, method, path, body=None, raw=False):
        req = urllib.request.Request(
            self.base + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-User": self.who, "X-Client": self.who + "-kit",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = r.read()
                return r.status, (data if raw else json.loads(data or b"{}"))
        except urllib.error.HTTPError as e:
            data = e.read()
            try:
                return e.code, json.loads(data or b"{}")
            except Exception:
                return e.code, {"error": data[:200].decode("utf-8", "replace")}

    def get(self, path):
        if self.dry:
            return {}
        return self._call("GET", path)[1]

    def post(self, path, body):
        if self.dry:
            return {}
        self.sent += 1
        status, j = self._call("POST", path, body)
        if status == 403 and j.get("can_request"):
            self.refusals.append(j)
            raise Refused(j)
        if status == 409 and j.get("pass_already_applied"):
            return j
        if status >= 400:
            raise RuntimeError("%s -> %s: %s" % (path, status, j.get("error", j)))
        return j

    def state(self):
        if self.dry:
            return {"width": self.W, "height": self.H, "layers": []}
        return self.get("/api/state")

    def layer(self, name_or_id):
        return self.lids.get(name_or_id, name_or_id)

    def add_layer(self, name, below=None):
        r = self.post("/api/layer", {"action": "add", "name": name,
                                     **({"below": below} if below else {})})
        self.lids[name] = r.get("id", name) if self.dry else r["id"]
        return self.lids[name]

    # ----------------------------------------------------------- seeing
    def _image(self, path):
        if self.dry:
            # a plausible mid-grey surface, so a pass that READS what is
            # under it (a glaze judging the value it is glazing, broken
            # colour sampling the ground) rehearses instead of crashing
            return np.full((self.H, self.W, 4), 0.45, np.float32)
        status, data = self._call("GET", path, raw=True)
        if status != 200:
            return None
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")
                          ).astype(np.float32) / 255.0

    def composite(self):
        return self._image("/api/composite.png")

    def layer_image(self, layer):
        return self._image("/api/layer/%s.png" % self.layer(layer))

    def below(self, layer):
        """What is UNDER this layer -- the surface you are painting on."""
        return self._image("/api/layer/%s/below.png" % self.layer(layer))

    def above(self, layer):
        """What is OVER this layer -- what will be drawn on top of you."""
        return self._image("/api/layer/%s/above.png" % self.layer(layer))

    @staticmethod
    def sample(img, x, y, r=2):
        """Mean RGB of a small window, or None off-canvas."""
        if img is None:
            return None
        H, W = img.shape[:2]
        x, y = int(x), int(y)
        if not (0 <= x < W and 0 <= y < H):
            return None
        p = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1, :3]
        return [float(v) for v in p.reshape(-1, 3).mean(axis=0)]

    # ----------------------------------------------------------- passes
    @contextlib.contextmanager
    def pass_(self, name, force=False):
        """A NAMED pass: everything painted inside is one batch, sent once.
        A second run of the same pass in the same document is refused by
        the server and recorded in .skipped_passes instead of doubling."""
        self.flush()
        # a fresh run id per ENTRY to the pass: a pass larger than one batch
        # is several calls that must all be let through, while a second run
        # of the same named pass is the accident we are guarding against
        self._pass, self._force = name, force
        self._pass_run = "%s-%d" % (self.who, self.rnd.getrandbits(48))
        try:
            yield
        finally:
            self.flush()
            self._pass, self._force, self._pass_run = None, False, None

    def dip(self, color=None, amount=1.0):
        """Reload the brush from the palette.

        leStudio models a REAL RESERVOIR: with `real_brush` on, the brush
        holds a finite charge, spends it as it paints, and when it is empty
        every further stroke deposits NOTHING -- silently, at HTTP 200.
        Measured: a loaded brush is dry after about three long strokes, and
        the next seven changed not one pixel. That is correct physics and a
        terrible trap for a script, which has no hand to feel the brush go
        dry with. So the kit dips for you (see `flush`), and this is the
        manual door for when you want to dip in a particular colour."""
        body = {"amount": float(amount)}
        if color is not None:
            body["color"] = [float(c) for c in color]
        try:
            self.post("/api/brush_load", body)
        except Exception:
            pass                      # dipping must never break a pass

    def flush(self):
        # A real-brush stroke spends the reservoir, so each carries its own
        # DIP -- `dip: true` on the batch item, which the engine honours
        # immediately before that stroke. The kit used to leave the batch
        # to POST /api/brush_load between strokes, which meant a real-brush
        # pass could not be batched at all: two round trips per mark, and
        # the R66 accents pass sat there for twelve minutes. It batches now,
        # and the physics is identical -- the dip still happens, in order,
        # in the stroke's own colour.
        for st_ in self.batch:
            if st_.get("real_brush") and "dip" not in st_:
                st_["dip"] = True
        if self.dry:
            for st_ in self.batch:
                self.counts[st_["layer"]] = self.counts.get(st_["layer"], 0) + 1
            self.batch = []
            return
        while self.batch:
            chunk, self.batch = self.batch[:250], self.batch[250:]
            body = {"strokes": chunk}
            if self._pass:
                body["pass"] = self._pass
                body["pass_run"] = self._pass_run
                if self._force:
                    body["force"] = True
            r = self.post("/api/paint_batch", body)
            if r.get("pass_already_applied"):
                self.skipped_passes.append(self._pass)
                self.batch = []
                return

    # ----------------------------------------------------------- the medium
    @contextlib.contextmanager
    def medium(self, media=None, **kw):
        """Paint with real PAINT for everything inside this block.

        leStudio's engine is not a flat-colour compositor: `media` runs a
        physical paint model (oil holds a ridge and takes a satin specular
        off its own slopes, acrylic holds a bristle comb, water wicks into
        the paper and leaves a dark rim as it dries), `load` is how much
        paint is on the brush, `mix` is PICKUP -- how much of what is
        already on the canvas the brush drags into the stroke -- and
        `material` is a PBR surface (rough/metal) the light answers to.

        The first version of this kit exposed none of it, so every painter
        that used it laid flat colour at a uniform radius. That is the
        single biggest reason the R64/R65 still life read as a cartoon:
        "uniformly flat, as though applied with a roller" is the textbook
        description of the fault, and it was the only thing the API could
        express. Settings nest, so an inner block can override one field
        of an outer one.

            with p.medium("oil", load=0.6, real_brush=True):
                with p.medium(mix=0.55):        # keeps oil + load
                    p.line(...)
        """
        prev = dict(self._paint)
        if media is not None:
            self._paint["media"] = media
        self._paint.update({k: v for k, v in kw.items() if v is not None})
        try:
            yield self
        finally:
            self._paint = prev

    PAINT_KEYS = ("media", "material", "load", "mix", "real_brush", "brush",
                  "erase", "stroke_taper")

    # ----------------------------------------------------------- marks
    def st(self, layer, pts, color, radius, opacity=1.0, hardness=0.25,
           erase=False, **paint):
        if len(pts) < 2:
            pts = [pts[0], [pts[0][0] + 0.01, pts[0][1]]]
        d = {"layer": self.layer(layer),
             "points": [[float(p[0]), float(p[1]), 1.0] for p in pts],
             "color": [min(1.0, max(0.0, float(c))) for c in color],
             "radius": float(radius), "opacity": float(opacity),
             "hardness": float(hardness)}
        d.update(self._paint)                  # the medium in force
        for k, v in paint.items():             # per-stroke overrides
            if k not in self.PAINT_KEYS:
                raise TypeError("unknown paint setting %r -- one of %s"
                                % (k, ", ".join(self.PAINT_KEYS)))
            if v is not None:
                d[k] = v
        if erase:
            d["erase"] = True
            for k in ("media", "material", "mix", "real_brush"):
                d.pop(k, None)                 # an eraser carries no paint
        self.batch.append(d)
        if len(self.batch) >= 250:
            self.flush()

    def line(self, layer, a, b, color, radius, opacity=1.0, hardness=0.25,
             wobble=0.0, n=None, **paint):
        """A stroke from a to b, optionally hand-wobbled, with points spaced
        below the radius so the stamps overlap on screen."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy) or 1.0
        if n is None:
            n = max(2, int(ln / max(1.5, radius * 0.45)) + 1)
        px, py = -dy / ln, dx / ln
        pts = [[a[0] + dx * i / (n - 1.0) + px * self.rnd.uniform(-wobble, wobble),
                a[1] + dy * i / (n - 1.0) + py * self.rnd.uniform(-wobble, wobble)]
               for i in range(n)]
        self.st(layer, pts, color, radius, opacity, hardness, **paint)

    # ----------------------------------------------------------- masses
    def fill(self, layer, poly, color_at, step=None, radius=9.0, opacity=1.0,
             hardness=0.22, seg=None, jitter=None, erase=False, edge=None,
             **paint):
        """Lay a MASS. `color_at(x, y)` is asked once per horizontal
        segment, so it can be a real 2-D gradient -- but only at the
        resolution `seg` gives it. A 40 px default turned a 80 px lemon
        into two flat blocks with a step down the middle, so `seg` now
        defaults to ~`radius`, and small shapes get a proportionally
        smaller one: a mass is never sampled at fewer than ~12 columns
        across, however small it is.

        `step` (the scanline pitch) must be below `radius` -- the coverage
        law -- and defaults to radius * 0.6. `radius` is also CAPPED to a
        fraction of the shape's smaller dimension, because a dab wider
        than the feature it is painting inflates the silhouette: a 10 px
        highlight painted with a radius-4 brush comes out 18 px wide and
        rectangular. Return None from color_at to leave a spot unpainted."""
        px = np.asarray(poly, dtype=float)
        if len(px) >= 3:
            pw = float(px[:, 0].max() - px[:, 0].min())
            ph = float(px[:, 1].max() - px[:, 1].min())
            radius = max(0.8, min(radius, max(1.0, min(pw, ph)) * 0.34))
            if seg is None:
                seg = max(2.0, min(radius * 1.1, pw / 12.0))
        elif seg is None:
            seg = max(2.0, radius)
        if step is None:
            step = max(1.0, radius * 0.6)
        if jitter is None:
            # Scanlines on a RULER beat against the gradient they are
            # laying: the lemons came back with a clean period-5 ripple of
            # +/-0.018 across every sphere, which on a smooth form reads as
            # contour banding. The same fill at a soft hardness measured
            # +/-0.001, so it is the regular pitch, not the coverage. A
            # painter's strokes are not on a ruler, and breaking the pitch
            # into noise costs nothing and is truer to the hand.
            jitter = step * 0.35
        if step >= radius:
            raise ValueError("fill(): step %.1f must be smaller than radius "
                             "%.1f, or the mass comes out striped" % (step, radius))
        # Rasterise the polygon into a mask PADDED by a radius, and walk
        # that -- never the canvas. Clipping the mask to the canvas first
        # makes the canvas edge look like the shape's edge, so the first
        # and last scanline get coverage from one side only: row 0 came
        # back at alpha 0.52 against 0.95 in the middle, and the dark
        # ground showed through as a ragged frame round the whole picture.
        # A wall that runs off the canvas must be PAINTED off the canvas.
        # A shape that really does end inside it is unaffected -- the pad
        # adds no scanlines where the polygon has no area.
        P = int(math.ceil(radius)) + 2
        m = Image.new("1", (self.W + 2 * P, self.H + 2 * P), 0)
        ImageDraw.Draw(m).polygon([(float(q[0]) + P, float(q[1]) + P)
                                   for q in poly], fill=1)
        a = np.asarray(m)
        ys = np.where(a.any(axis=1))[0]
        if not len(ys):
            return 0
        n = 0
        # Where the polygon is CUT OFF by the padded mask it has not ended
        # -- it goes on. Walk a further radius out on that side and keep
        # reading the last real row, or the boundary scanline lands on a
        # different phase of the pitch from the interior and picks up one
        # pass fewer: measured, rows 0-3 came back 0.05 darker than row 10
        # across the whole picture. A shape that ends INSIDE the mask is
        # untouched, so a lemon never smears outward.
        ylo, yhi = float(ys.min()), float(ys.max())
        mh, mw = a.shape
        wlo = ylo - radius if ys.min() == 0 else ylo
        whi = yhi + radius if ys.max() == mh - 1 else yhi
        y = wlo
        while y <= whi:
            row = np.where(a[int(min(max(y, ylo), yhi))])[0]
            if len(row):
                for run in np.split(row, np.where(np.diff(row) > 1)[0] + 1):
                    if len(run) < 2:
                        continue
                    x0, x1 = float(run[0]) - P, float(run[-1]) - P
                    if run[0] == 0:                  # cut off on the left
                        x0 -= radius
                    if run[-1] == mw - 1:            # ...and on the right
                        x1 += radius
                    k = max(1, int((x1 - x0) / seg))
                    # STAGGER the segment boundaries row by row. A real
                    # medium shoves paint sideways into a rim at the end of
                    # every stroke (oil's `berm`), so segment boundaries
                    # that line up from row to row build a RIDGE -- and a
                    # whole picture laid this way came back covered in
                    # vertical corduroy. Offsetting each row by a fraction
                    # of a segment scatters those rims instead of stacking
                    # them, and costs nothing.
                    ry = y - P                      # back in canvas space
                    off = ((ry * 0.6180339887) % 1.0) - 0.5
                    for i in range(k):
                        sx0 = x0 + (x1 - x0) * (i + off * 0.9) / k
                        sx1 = x0 + (x1 - x0) * (i + 1 + off * 0.9) / k
                        sx0, sx1 = max(x0, sx0), min(x1, sx1)
                        if sx1 - sx0 < 0.5:
                            continue
                        col = color_at((sx0 + sx1) * 0.5, ry)
                        if col is None:
                            continue
                        yy = ry + (self.rnd.uniform(-jitter, jitter) if jitter else 0.0)
                        # and overlap by a FULL radius each side, not half:
                        # half left the soft cap of each dab visible as a
                        # seam even before the berm piled on it
                        self.st(layer, [[max(x0, sx0 - radius), yy],
                                        [min(x1, sx1 + radius), yy]],
                                col if not erase else [0, 0, 0],
                                radius, opacity, hardness, erase=erase,
                                **paint)
                        n += 1
            y += step
        n += self._contour(layer, px, color_at, radius, opacity, hardness,
                           erase, edge, paint)
        return n

    def _contour(self, layer, px, color_at, radius, opacity, hardness,
                 erase, edge, paint):
        """Walk the silhouette with a small brush.

        A scanline fill quantises a curved edge to its own pitch: at
        step 6 the jug came back with a visible 6 px staircase down both
        sides, and a staircase reads as a CUT-OUT -- which is the same
        fault as a hard black outline, arrived at from the other
        direction. Every source on why a picture looks like a cartoon
        puts edges first, and an edge is where a painter spends the most
        attention, so the kit now spends some too: the contour is dabbed
        at ~1.5 px with a soft brush carrying the colour sampled just
        INSIDE it, which fills the notches without inventing a rim.

        Off by default for a GROUND -- a wall or a wash has no silhouette
        worth drawing, and outlining one would be absurd."""
        if edge is False or erase or len(px) < 3:
            return 0
        w = float(px[:, 0].max() - px[:, 0].min())
        h = float(px[:, 1].max() - px[:, 1].min())
        if edge is None:
            if w * h > self.W * self.H * 0.25:
                return 0                       # a ground has no contour
            if min(w, h) < 6.0 or radius < 2.5:
                return 0                       # already finer than the pitch
            if hardness < 0.08:
                return 0                       # a soft mass KEEPS its soft
                # edge. A shadow pool, a bloom, a scumble -- anything laid
                # with a hardness this low was asked for without a boundary,
                # and sharpening its contour would turn a cast shadow back
                # into the pasted-on disc it is trying not to be.
        cx, cy = float(px[:, 0].mean()), float(px[:, 1].mean())
        r = max(1.2, min(radius * 0.34, 3.2))
        pitch = max(1.2, r * 0.5)
        n = 0
        for i in range(len(px)):
            ax, ay = px[i]
            bx, by = px[(i + 1) % len(px)]
            seglen = math.hypot(bx - ax, by - ay)
            for j in range(max(1, int(seglen / pitch)) + 1):
                t = min(1.0, j * pitch / max(1e-6, seglen))
                x, y = ax + (bx - ax) * t, ay + (by - ay) * t
                dx, dy = cx - x, cy - y
                d = math.hypot(dx, dy) or 1.0
                col = None
                for inward in (2.5, 5.0, 8.0):   # far enough in to be honest
                    col = color_at(x + dx / d * inward, y + dy / d * inward)
                    if col is not None:
                        break
                if col is None:
                    continue
                self.st(layer, [[x, y], [x + 0.01, y]], col, r,
                        opacity * 0.85, min(hardness, 0.35), **paint)
                n += 1
        return n

    def ellipse(self, layer, cx, cy, rx, ry, color_at, **kw):
        """An elliptical mass. Extra keywords reach fill(), including the
        paint settings, so `p.ellipse(..., media="oil", mix=0.5)` works."""
        n = 96
        return self.fill(layer, [[cx + rx * math.cos(2 * math.pi * i / n),
                                  cy + ry * math.sin(2 * math.pi * i / n)]
                                 for i in range(n)], color_at, **kw)

    def masses(self, layer, k=4, thresh=0.15, min_frac=0.0004, max_frac=0.45):
        """The SEPARATE painted masses on a layer, largest first, as
        full-resolution (x0, y0, x1, y1). A mass covering more than
        `max_frac` of the canvas is a GROUND -- a wall, a wash, a sky --
        and is dropped: a ground is not an object and nothing about it
        wants a contact shadow. 8-way connected components on a /k grid."""
        a = self.layer_image(layer)
        if a is None:
            return []
        small = a[::k, ::k, 3] > thresh
        if not small.any():
            return []
        h, w = small.shape
        lab = np.zeros((h, w), np.int32)
        out, nxt = [], 0
        for sy, sx in np.argwhere(small):
            if lab[sy, sx]:
                continue
            nxt += 1
            stack = [(int(sy), int(sx))]
            lab[sy, sx] = nxt
            y0 = y1 = int(sy)
            x0 = x1 = int(sx)
            n = 0
            while stack:
                cy, cx = stack.pop()
                n += 1
                y0, y1 = min(y0, cy), max(y1, cy)
                x0, x1 = min(x0, cx), max(x1, cx)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w \
                                and small[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = nxt
                            stack.append((ny, nx))
            frac = n / float(h * w)
            if min_frac <= frac <= max_frac:
                out.append((n, (x0 * k, y0 * k, (x1 + 1) * k, (y1 + 1) * k)))
        out.sort(key=lambda t: -t[0])
        return [b for _, b in out]

    def footprint(self, layer, bbox=None, rows=10, thresh=0.15):
        """WHERE an object meets the surface: a small polygon along the
        bottom `rows` of its silhouette. This is the thing a cast shadow is
        thrown from and the thing a contact shadow sits under, and every
        painter was eyeballing it from nominal geometry that the painting
        had long since drifted away from. Returns [] if nothing is there."""
        a = self.layer_image(layer)
        if a is None:
            return []
        m = a[..., 3] > thresh
        if bbox:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            keep = np.zeros_like(m)
            keep[y0:y1, x0:x1] = True
            m = m & keep
        if not m.any():
            return []
        ys = np.where(m.any(axis=1))[0]
        ybot = int(ys.max())
        top = max(int(ys.min()), ybot - rows)
        left, right = [], []
        for y in range(top, ybot + 1):
            row = np.where(m[y])[0]
            if not len(row):
                continue
            left.append([float(row.min()), float(y)])
            right.append([float(row.max()), float(y)])
        if len(left) < 2:
            return []
        return left + right[::-1]

    def settle_edge(self, layer, bbox=None, inset=6, width=1, radius=3.5,
                    opacity=0.55, step=1, thresh=0.35, erase_fringe=0):
        """Repair the EDGE of a mass without eating its silhouette.

        Two separate faults live at a painted contour, and a picture
        usually has both:

          the HAIRLINE -- a mass laid as scanline runs ends up with a ring
          of the wrong value right on its boundary, and that one-pixel line
          is what makes a painted object read as a vector shape with an
          outline on it;

          the HALO -- a soft anti-aliased skirt a few pixels OUTSIDE the
          boundary, carrying whatever colour the brush had. On the jug this
          was a 0.52 grey fringe against a 0.38 body, glowing all round it.

        `width` repairs a BAND that many pixels inward (an outline drawn
        just inside the contour needs this, not just the boundary row), and
        `erase_fringe` erases that many pixels outward. Both take their
        colour from the layer's own pixels `inset` in, so the edge keeps
        its shape and loses its line."""
        a = self.layer_image(layer)
        if a is None:
            return 0
        m = a[..., 3] > thresh
        if bbox:
            bx0, by0, bx1, by1 = [int(v) for v in bbox]
            keep = np.zeros_like(m)
            keep[by0:by1, bx0:bx1] = True
            m = m & keep
        if not m.any():
            return 0
        inner = (m & np.roll(m, 1, 0) & np.roll(m, -1, 0)
                 & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        ys, xs = np.where(m & ~inner)
        if not len(ys):
            return 0
        H, W = m.shape
        gy, gx = np.gradient(m.astype(np.float32))
        n = 0
        # `step` subsamples the FLAT boundary list, and a contour
        # contributes only two points per row -- so any step above 1 drops
        # whole rows of the edge, in a pattern that shifts wherever the
        # silhouette has a handle or a notch. It left the jug's fringe
        # repaired on some scanlines and untouched on others. Default 1.
        for i in range(0, len(ys), max(1, step)):
            y, x = int(ys[i]), int(xs[i])
            dx, dy = float(gx[y, x]), float(gy[y, x])
            ln = math.hypot(dx, dy)
            if ln < 1e-3:
                continue
            ux, uy = dx / ln, dy / ln            # points INTO the mass
            sx, sy = int(round(x + ux * inset)), int(round(y + uy * inset))
            if not (0 <= sx < W and 0 <= sy < H) or a[sy, sx, 3] < thresh:
                continue
            col = [float(v) for v in a[sy, sx, :3]]
            for k in range(max(1, width)):
                px, py = x + ux * k, y + uy * k
                self.st(layer, [[px - 0.4, py], [px + 0.4, py]], col,
                        radius, opacity, 0.30)
                n += 1
            for k in range(1, erase_fringe + 1):
                px, py = x - ux * k, y - uy * k
                self.st(layer, [[px - 0.4, py], [px + 0.4, py]], [0, 0, 0],
                        max(1.2, radius * 0.55), 1.0, 0.85, erase=True)
                n += 1
        return n

    def soften(self, layer, bbox, radius=9.0, opacity=0.30, step=5,
               axis="y", span=None, thresh=0.55):
        """Average a mass along one axis and glaze the result back.

        Two jobs, one tool. axis="y" takes the STEPPING out of a mass laid
        in scanline runs -- the rings a coarse step leaves across a
        cylinder. axis="x" flattens a vertical LINE that should not be
        there: the pale hairline a previous painter left just inside a
        silhouette reads as an outline, and blurring across it is how you
        lose the line without losing the edge.

        It only writes where the layer is ALREADY opaque (`thresh`), so it
        can never spill past the silhouette it is repairing -- which is
        exactly how a contour repair turned into a chewed edge once."""
        a = self.layer_image(layer)
        if a is None:
            return 0
        x0, y0, x1, y1 = [int(v) for v in bbox]
        H, W = a.shape[:2]
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        w = int(span if span is not None else radius)
        n = 0
        for y in range(y0, y1, step):
            for x in range(x0, x1, step):
                if a[y, x, 3] < thresh:
                    continue
                if axis == "y":
                    win = a[max(y0, y - w):min(y1, y + w + 1), x:x + 1]
                else:
                    win = a[y:y + 1, max(x0, x - w):min(x1, x + w + 1)]
                sel = win[win[..., 3] > thresh]
                if len(sel) < 3:
                    continue
                col = [float(v) for v in sel[:, :3].mean(axis=0)]
                self.st(layer, [[x - 0.4, y], [x + 0.4, y]],
                        col, radius, opacity, 0.30)
                n += 1
        return n

    def erase_rect(self, layer, x0, y0, x1, y1):
        """Take a rectangle OUT of your layer (journal-first on the server)."""
        self.flush()
        return self.post("/api/layer", {"action": "clear", "id": self.layer(layer),
                                        "region": [int(x0), int(y0), int(x1), int(y1)]})

    def cast_shadow(self, layer, footprint, light, length, color_at,
                    radius=10.0, opacity=0.5, hardness=0.06, falloff=1.6,
                    squash=0.17, spread=0.9, steps=None):
        """Throw a shadow from a FOOTPRINT onto the surface it stands on.

        The shadow is a POOL, not a sliver. A first version swept the
        footprint polygon along the light direction and produced a ~19 px
        band that read as a smudge: a footprint is the few rows where the
        object meets the surface, and the shadow of a standing object on a
        horizontal plane is that contact ELLIPSE stretched away from the
        light and growing softer and wider as it goes. So: an ellipse at
        the contact (width from the footprint, height = width * `squash`,
        the foreshortening of a circle seen at this angle), repeated along
        the throw with the radius growing by `spread` and the opacity
        falling by `falloff`.

        `light` is (dx, dy) pointing FROM the source TOWARD the object, so
        the shadow goes the same way. `color_at(x, y, t)` gets the
        normalised distance along the throw, so the near end can be tight,
        warm and dark and the far end soft and faint."""
        pts = np.asarray(footprint, dtype=float)
        if len(pts) < 3:
            return 0
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        ybot = float(pts[:, 1].max())
        cx, w = (x0 + x1) * 0.5, (x1 - x0)
        if w < 4:
            return 0
        rx0, ry0 = w * 0.52, max(3.0, w * squash)
        lx, ly = light
        ln = math.hypot(lx, ly) or 1.0
        lx, ly = lx / ln, ly / ln
        if steps is None:
            steps = max(4, int(length / max(3.0, rx0 * 0.30)))
        total = 0
        for i in range(steps):
            t = i / float(steps - 1)
            ex = cx + lx * length * t
            ey = ybot - ry0 * 0.35 + ly * length * t * 0.55
            rx = rx0 * (1.0 + spread * t)
            ry = ry0 * (1.0 + spread * 0.75 * t)
            fade = (1.0 - t) ** falloff
            if fade <= 0.01:
                continue
            total += self.ellipse(
                layer, ex, ey, rx, ry,
                lambda x, y, t=t: color_at(x, y, t),
                radius=radius * (1.0 + t * 0.8),
                opacity=min(0.85, opacity * fade * (1.6 / steps) * 6.0),
                hardness=hardness)
        return total

    # ----------------------------------------------------------- ownership
    def ask_access(self, layer, note=""):
        return self.post("/api/access", {"action": "request",
                                         "layer": self.layer(layer), "note": note})


def pw(base, e):
    """`base ** e` that cannot return a COMPLEX number.

    A fractional exponent on a negative base is complex in Python, and a
    colour function reaching one pixel outside the shape it was written for
    is the normal way to get there -- `(v * 1.12) ** 1.35` with v = -0.03
    took down a whole pass. It has bitten this project twice now, once as a
    complex number serialised into a paint request. Falls off symmetrically
    for a negative base so a curve through zero stays continuous."""
    if base >= 0.0:
        return base ** e
    return -((-base) ** e)


def lerp(a, b, t):
    """Interpolate two colours -- or two plain NUMBERS.

    The scalar case is here because value functions are written in
    luminance and then turned into colour, so half the calls are
    `lerp(0.4, 0.7, t)`. Without it those had to be written
    `lerp([a]*3, [b]*3, t)[0]`, the `[0]` got forgotten, and a list
    reached arithmetic as `v *= ...` -> "can't multiply sequence by
    non-int of type 'float'" in the middle of a pass."""
    if isinstance(t, complex) or t != t:
        # A fractional power of a negative base is COMPLEX in Python, and
        # fill() now paints a ground past the canvas edge, so a colour
        # function written with a bare `**` really does hand one over. The
        # comparison below would fail as "'<' not supported between
        # instances of 'complex' and 'int'", forty frames from the line
        # that wrote it. Say what happened and what to use instead.
        raise TypeError(
            "lerp() got t=%r. A fractional power of a negative base is a "
            "complex number -- use pw(base, e) rather than base ** e in a "
            "colour function, which is asked for points OUTSIDE the shape "
            "and past the edges of the canvas." % (t,))
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def mul(c, k):
    return [min(1.0, max(0.0, v * k)) for v in c]


def warm(c, k=0.06):
    """Push a colour warmer without changing its value much."""
    return [min(1.0, c[0] + k), c[1], max(0.0, c[2] - k)]
