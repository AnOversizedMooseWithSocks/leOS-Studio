# Driving leStudio as an agent

Everything the UI does goes through the same JSON HTTP API -- the browser is
just one client. An agent is another.

## Orientation
1. `GET /api/schema` -- every endpoint + the full node-op catalog (names,
   inputs, parameter kinds/ranges/choices, docs).
2. `GET /api/state` -- the complete current truth: documents, layers, masks,
   selections, splines, brushes, graph, media status.

## Conventions
- Send TWO headers on POSTs: `X-Client: <this run>` (echo suppression) and
  `X-User: <your persistent identity>`. Presence, the roster, the HOST role
  and kick/allow are all per X-User -- reuse the same X-User across runs and
  you are one editor, not a crowd of ghosts. The SSE feed
  (`GET /api/events?client=..&user=..&name=YourName`) reports
  `{rev, src, editors, names}`; refresh your view when `rev` moves and `src`
  is not your X-Client. Holding the stream open IS presence; kicked users
  get a terminal `{kicked:true}` event and 403s until the host allows them
  back (`/api/editors`, `/api/editors/kick`, `/api/editors/allow`,
  `/api/editors/name`).
- Long evaluations: `POST /api/graph/run` -> `{job}`, poll `GET /api/job/<id>`,
  cancel with `POST /api/job/<id>/cancel`.
- Everything is undoable: `POST /api/undo` / `POST /api/redo`.

## A worked example
```
POST /api/new                {"width": 1024, "height": 576, "background": null}
POST /api/paint              {"layer": "L2", "points": [[100,100],[500,300]],
                              "radius": 30, "color": [0.9,0.2,0.1]}
POST /api/select             {"tool": "object", "params": {"x": 300, "y": 200}}
POST /api/selection          {"action": "to_mask", "id": "S1"}
POST /api/graph              {"nodes": [... incl {"type":"Media in",
                              "params":{"source":"test:clock"}} wired to Output ...]}
POST /api/live               {"action": "start", "fps": 10}
GET  /api/stream.mjpg        <- the composited result, streamable
GET  /api/workspace.lews     <- save everything
```

The node graph is plain JSON (`nodes` with `type`, `params`, `inputs`
addressed as `"nodeId"`, `"nodeId.socket"`, or `[id, socket]`), so an agent
can synthesise whole pipelines -- including cross-document reads
(`params.doc`) and leCore-generated GLSL (`POST /api/sdf/shader`).
`"param:<name>"` input keys wire any node output into any NUMERIC parameter:
a `Value` node drives it directly, `Light direction` exposes x/y/angle value
sockets, and an image input drives the parameter by mean luminance.

## The deposit model (what makes a stroke look like paint)
Height is NOT a rescaled alpha mask. `_deposit()` turns coverage into a paint
surface via four mechanisms, each keyed to stroke-local coordinates from
`_stroke_frame()` (`u` = signed distance across the stroke, `s` = distance
along it): **load depletion** (the brush empties as it travels, so long
strokes thin and run dry), **bristle comb** (`_bristle_comb`, fixed lanes of
furrow running the stroke's length -- lane pitch is ~4px at ANY brush size,
and lane width is a fraction of lane SPACING so furrows stay separate on
small brushes), **canvas tooth** (`_canvas_tooth`, an fbm+weave substrate
seeded by a fixed constant so the same stroke lands identically in any
document; thin paint catches only on the peaks), and **berm** (bristles bank
paint sideways into two soft rims with a trough between, redistributed with
volume conserved). All four reach COVERAGE as well as height -- dry-brush is
a hole in the paint film, not just an embossed texture -- but only where the
brush is genuinely starved (`cover` is pinned at 1 for a loaded brush).
`_RELIEF_SLOPE` converts height units into the slope the light sees; without
it `np.gradient` on a ridge yields near-flat normals and everything reads
airbrushed. Media carry `bristle`/`berm` character; materials derive theirs
from `hold`.

The canvas has relief of its own: `_CANVAS_RELIEF` adds the weave to every
shaded layer's height, so paint sits IN a surface rather than floating on
glass. This was the fix for strokes reading as extrusions; it helps but does
not fully solve small strokes (see the gap below).

## Blending, and why mixing does NOT cost you stroke editing
`blend_stroke()` / `POST /api/paint {"mode":"blend"}` is the BLENDER -- a
clean brush carrying no pigment that softens and drags the wet paint already
on the canvas, gated by paint BODY (you cannot blend what is not there) and
knocking the ridges down as it passes. Bob Ross's second brush.

The important part is that it is a RECORDED, REPLAYABLE STROKE. `smudge`
mutates pixels and is explicitly not recorded, so smudging cost the layer its
stroke editing -- and blending is exactly the gesture you most want to nudge,
because blending is where the picture gets made.

This means leStudio does NOT need a "derived third stroke with the parents
hidden" to make mixing non-destructive: the replay pipeline already IS the
derived render. A layer rebuilds by replaying paint, paint, blend IN ORDER,
so the parents stay individually selectable and editable, and nudging a
colour underneath re-derives the blend automatically. Verified in
`test_blend_stroke_is_recorded_and_replayable`. What a group entity would
still add is the ability to select and move a blended passage as ONE unit --
organisation, not semantics. Not built yet.

## Pressure and speed
Both reach the PAINT, not just the radius. Pressure arrives as the per-point
width channel (`[x, y, w]`) -- the engine always had it, the UI simply never
fed it, so every stroke landed at a flat 1.0. Leaning squeezes more paint
out, splays the bristles so the comb bites HARDER (press hard and you want
bristle marks to show), and banks more into the berm. Speed thins the
deposit, which trips the canvas-tooth gate by itself -- dry-brush from a
flick falls out for free with no separate rule.

`penPressure()` gates on `pointerType === 'pen'`: a mouse reports 0.5, or 0
while hovering, so treating that as pressure paints every mouse stroke at
half weight. Mice always paint at full pressure. Watch the lazy-brush EMA --
it rebuilt the point as a bare `[x, y]` and silently dropped pressure.

SPEED IS INFERRED from raw point spacing (the client polls at a fixed rate,
so spacing is speed; recorded points keep their spacing, so replays
re-derive it identically -- no timestamps to plumb). But spacing only means
speed for a path a HAND sampled. A two-point straight line from the API has
one enormous gap and was read as a maximum-speed flick, making programmatic
strokes three times too thin. Below 6 points, speed is unknown, and unknown
means NEUTRAL, not fast. If you ever add another input path, check it clears
that bar or make speed explicit.

## Building up a brush ACROSS ITS WIDTH (WIP -- NOT GREEN)
`_BRUSH_LANES` (7) bands across the tuft each carry their own colour and
charge. Loading one edge with blue and the other with white lays a variegated
band in ONE stroke -- the core Bob Ross move, which a scalar reservoir cannot
express because it averages the two into a flat mix before the stroke is
laid. Each lane samples the canvas under its own part of the tuft (offset
along the path normal), so dragging half the brush through a mound loads only
that half. `load_brush(color=[...7 colours...])` loads lanes directly;
dipping builds them up naturally. Per-pixel colour is looked up by BOTH arc
position and lane (from `u`). Verified working:
`test_brush_holds_different_colours_across_its_width` and
`test_dipping_one_edge_builds_a_two_colour_brush` both pass, and the demo
render shows a blue-to-white single stroke.

TWO FAILURES, do not ship until fixed:
ATTEMPTED AND REVERTED: freezing the full per-lane charge and colour into the
record. The idea is right but the record is written AFTER the reservoir has
already been updated, so the values must be captured where `c0`/`bc` are
resolved, and a real-brush stroke with NO medium never enters that branch at
all -- my defaults for that case were wrong and took the failure count from 2
to 5. Redo it by capturing into locals at the top of `paint()`, before any
branch, and check the no-medium path explicitly.

1. REPLAY DETERMINISM (`test_real_brush_runs_out_reloads_and_replays`). This
   is the important one. `charge0` is frozen into the stroke record as a
   SCALAR, and the brush colour at stroke start is taken from the `color`
   argument -- but the reservoir is now per-lane, so a replay restarts every
   lane equal and diverges from the original run. FIX: record the full
   per-lane charge AND per-lane colour in the stroke record (rec key
   `charge0` becomes a list of 7, plus a new `lanes0`), and thread them
   through `replay_layer`. The freezing principle is unchanged -- only the
   shape of what gets frozen.
2. `test_palette_mounds_can_be_dipped_and_mixed` asserts the mound is scraped
   down; with lanes the take is split seven ways so no single lane lifts as
   much, and the stroke still deposits onto the mound. Likely needs the
   scrape summed across lanes rather than sampled per pixel from one lane.

Also note `brush_color`/`brush_charge` are now the MEAN over lanes, kept for
API compatibility; `brush_lanes`/`brush_charges` hold the real state.

## Performance notes (measured, not assumed)
Live-flush latency at 1080p, 140-point stroke: oil r12 60ms / r28 79ms,
water r12 114ms / r28 149ms. Water is the slow case and always will be --
26 flow iterations against oil's 10.

Two optimisations already taken, do not undo them:
- `_gauss_small()` is a DIRECT separable kernel for the small sigmas the
  paint code uses. `_gauss_blur` convolves via FFT, which is right for big
  postfx kernels and badly wrong for a sigma-2 blur on a stroke window: an
  FFT pair per channel was 183ms of a 578ms watercolour stroke, 20
  transforms for one mark. Anything in the paint path wanting sigma < ~6
  should use `_gauss_small`.
- `_paint_flow`'s loop is allocation-free (reused `excess`/`frac`/`moved`
  buffers, `out=` everywhere). It runs up to 26 times per stroke and was
  allocating four full-window arrays each pass, one of them 4-channel;
  allocation, not arithmetic, was most of the cost.

STILL SLOW: water at ~150ms median flush on a big brush. The remaining cost
is spread across the flow iterations themselves. The untried lever with real
headroom is running the flow at half resolution and upsampling -- it is a
smooth diffusion-like process, so the visual cost should be small, but it
changes results and needs the byte-parity tests re-baselined deliberately.
The other structural win is incremental live painting (every flush repaints
the whole stroke from `before`, so cost grows with stroke length).

## Gravity is a SURFACE PROPERTY, not an assumption
`_flow_dir(doc, l)` returns (strength, dx, dy) from `l.gravity` /
`l.gravity_angle`, falling back to the document's, default (1.0, 90deg) so
existing behaviour is unchanged. Wet paint used to run screen-down no matter
what -- correct only for a canvas on an easel. A layer standing on one of the
room's walls runs down THAT wall; a canvas lying flat on a table has no
in-plane gravity and a puddle there LEVELS outward instead of running, which
is a different code path (spread to all four neighbours), not "flow off".
The flow window opens in the direction paint will actually travel, and the
step is split between the two axes so drips follow the real angle rather than
snapping to 8 compass points and staircasing. Serialised in snapshots and
.lews, editable via `/api/layer {"action":"edit"}`.

## Watercolour: a fluid IN paper, not a film on a surface
`_watercolour()` runs after `_paint_flow` for any medium with `absorb > 0`
and implements three of the effects Curtis et al. (SIGGRAPH 97) identify as
what makes watercolour read as watercolour. Before this, "water" was oil with
a low hold and more gravity -- a runny film on a surface, with none of them.
  WICKING: the paper drinks the wash sideways along its fibres, so the mark
  ends up larger and softer than the brush that made it.
  EDGE DARKENING: pigment carried to the perimeter as the water evaporates
  and stranded there. It must MOVE pigment (take from the interior, deposit
  on the rim), not amplify it -- scaling a thin rim can never beat a thick
  centre, and adding alpha at the damp boundary lands it on bare paper and
  haloes the mark with edge LIGHTENING, the exact inverse.
  GRANULATION: pigment is heavier than water and settles into the paper's
  dips, strongest where wettest.

`settle` flips which way the substrate works and is the key idea: stiff paint
dragged by a brush is scraped off the risen threads and catches on their
tops; a fluid wash runs off the tops and POOLS IN THE DIPS. Same tooth field,
opposite sign. Using the stiff-paint rule for watercolour was putting the
wash exactly where the water would have run out of.

TESTING NOTE: do not test granulation by comparing raw variance against oil.
Oil has plenty of variance from its bristle hairs and in fact measures
HIGHER. What is specific to granulation is that pigment anti-correlates with
the tooth -- test that.

## Everything added for painting is DOCUMENT STATE -- save it
Audited the save path and found six things silently dropped on save/reopen:
the paper, `auto_stratum`, the palette flag, the whole stratum chain
(`stratum_of` / `_next` / `_root`), `height_below`, and the brush reservoir.
A reopened picture would have sat on a different substrate, stopped building
past the layer ceiling, shaded its deep paint as stepped slabs again, and put
the palette back INTO the picture. All now written by `_doc_section` and read
by `_doc_from_section`, pinned by `test_the_studio_survives_a_save_and_reopen`.

AUDITING NOTE: `save_workspace` and `load_workspace` are THIN -- 816 and 976
chars. The real work is in `_doc_section` / `_doc_from_section`. Grepping the
public functions for a field, or slicing a fixed line window, gives false
"GAP" answers; find the helper that actually holds the meta. I got a false
result three times before checking where the code really lives.

RULE OF THUMB for anything added from here: if it lives on a Document or a
Layer and a painter would notice it missing, it needs a line in BOTH helpers
and a line in the round-trip test.

## Release pass
Checked the things a RELEASE breaks on rather than more feature behaviour.

1. THE ENTRY POINT SAT MID-FILE. `main()` and the `__main__` guard were
   ~200 lines from the end, with EIGHT routes defined after them -- the whole
   palette and paint-setup surface, all of it appended by me. Moved to the
   end, pinned by a test. Be precise about severity: this was LATENT, not
   live. `python -m lestudio` and the console script both import the module
   fully first, and running `server.py` directly fails on its relative
   imports anyway. But appending a route to the end of a file is the natural
   thing to do, so the structure was a trap.
2. THE LAUNCHERS HARDCODED 5050 while `serve()` reads LESTUDIO_PORT, so
   setting the port made run.sh announce -- and open a browser at -- a URL
   nothing was listening on. Both launchers now read the same environment.
3. THE README DOCUMENTED NONE OF IT. Palette, knife, real brush, build-up,
   paper, watercolour, gravity, the hosting knobs: zero mentions. A release
   that ships undocumented headline features ships features nobody finds.
   Added a "Painting with real media" section and a "Running it as a service"
   section, the latter stating plainly that the workspace is a SINGLE SHARED
   STUDIO and must run as one worker.

VERIFIED, not assumed: the packaged zip is clean (no __pycache__ or .pyc --
the ones I first saw were created by running tests in the extraction dir),
`static/index.html` is the only static asset and is declared in package-data,
and the app serves `/`, `/api/health`, `/api/state` and `/api/composite.png`
over real HTTP from a fresh unzip.

One more test pinned presentation: `test_accelerators_are_optional_with_fallbacks`
asserted the literal launcher URL. Re-pinned to the behaviour.

## leCore upgrade check (second snapshot)
A newer leCore tree was supplied. **Clean upgrade: 0 failures on all four
chunks, all three JS gates green, no code change needed.** Features went
1973 -> 2057, and all 21 faculties leStudio actually depends on are present;
the wall-boundary advect path still takes the fast route and keeps 100% of
the mass.

NOTHING IN IT IS WORTH ADOPTING FOR THIS APP, and that is the honest answer
rather than a shrug. The 84 new faculties are almost entirely 3D creature
work (face/head specs, fur shells, groom maps, tet meshes, template wrap,
FEM, morphogenesis) plus a Lean logic layer. Two looked applicable to 2D and
were MEASURED before being dismissed:
- `skin_sss_shade(base_rgb, ndl, thickness, ...)` -- subsurface scattering,
  and paint really does scatter (wax, thick glazes). But: 349 ms per call at
  1080p against a ~75 ms whole-stroke budget, it darkens thin paint to near
  black (0.029 where plain shading gives 0.780) because it models skin, and
  the difference at thickness 3 is 0.793 vs 0.780. Wrong model, wrong price.
- `sfs_debas_relief(depth, mask)` -- removes the bas-relief flattening/tilting
  ambiguity from ESTIMATED depth. Our height map is known exactly, so there is
  no ambiguity to remove; and the one place leStudio estimates depth is
  `_depthfog`, where a global flattening does not change the result.

WHEN A NEW ENGINE ARRIVES: run all four chunks, check `have()` for the 21
used faculties, exercise `_advect_walled` (its gate is a direct module import
with a fallback, not a feature name, so a rename would silently downgrade
the medium rather than fail), and diff `features()` against the old build --
that diff is the only reliable list of what actually changed.

## VERIFIED AGAINST REAL leCORE (first time)
A leCore source tree was supplied and the whole suite ran against it:
**0 failures in all four chunks**, and the stub environment still shows the
same 91 and no more. So the long-standing "the 91 are environmental" claim
was broadly right -- but the stub was ALSO MASKING FIVE REAL BUGS, four of
them mine. Never treat a stubbed failure list as noise; it is a blindfold.

WHAT THE STUB HID:
1. `/api/new` LOST ITS DOCSTRING. Wrapping the route for `_DOC_LOCK` moved it
   to the inner function and the route went undocumented -- invisible to the
   agent-facing schema. Checked the other two lock wrappers keep theirs.
2. COMPOSITE CACHE DRIFT OF 0.127 (mine, the serious one). When a layer gains
   its FIRST height map, canvas tooth starts lighting EVERY painted pixel,
   not just the new stroke, so a window patch describes a picture that no
   longer exists. Baseline drifted 0.0015 and got away with it; deepening the
   canvas relief made the same gap visible.
   THE FIX HAS TWO HALVES -- the first attempt was correct and 4x too slow.
   Skipping the patch left `_shade_rev` unset, so the layer never counted as
   "already relief" and every later stroke fell back to a full rebuild
   (measured 1365 ms). Re-light the layer ONCE in full, patch the composite
   over its whole extent, then resume window patching. Now 0.000 drift AND
   the budget passes.
3. The mirror labels from the previous round TRUNCATE in that narrow select --
   which reads as broken rather than terse, i.e. worse than the glyphs they
   replaced. Short words instead ("across", "down", "3-way"); the row's own
   "mirror" label carries the meaning.
4. I WAS WRONG ABOUT HIDDEN LAYERS. My user-test reported "painting a hidden
   layer succeeds silently" and I made it a 400. The app ALREADY warned
   ("that landed on a HIDDEN layer -- toggle its eye to see it"); my scenario
   read only the status code and the pixels, never the `warning` field. The
   400 threw the stroke away and broke a deliberate design. Reverted.
5. Three tests pinned CONSTANTS or SOURCE TEXT rather than intent: an oil
   gloss of 0.55 (the medium moved to 0.34), a literal `strokeBody` line, and
   the literal `_replaying` condition. All re-pinned to the guarantee.

ENGINE NOTES FOR THIS BUILD:
- It reports version "0.0.0": leCore's VERSION file is CI-owned and excluded
  from source archives. Functionally harmless -- availability is decided by
  `have()`, never by comparing the string -- but it DISPLAYED as an ancient
  engine. `engine_version()` now says "development build" for the sentinel.
- `have("advect")` and `have("boundary_wall")` read FALSE on this build: the
  faculty is named `advect_field`, and `boundary="wall"` is an ARGUMENT, not
  a faculty. Not a problem -- `_advect_walled` imports the module directly
  and falls back -- and verified working here: wall path taken, 100% of the
  mass kept. This is exactly why the pin comment says a version number cannot
  answer "is THIS faculty present".

## Acceleration, and what it would take to host this
### What is actually accelerated (measured, not assumed)
The PAINTING engine -- `_deposit`, `_paint_flow`, `_bristle_tracks`,
`_gauss_small`, `_stroke_frame` -- is PURE NUMPY ON THE CPU. Checked each
function: none touches leCore or any accelerator. leCore's GPU support covers
SIMULATION and node work only.

So the status chip reading "GPU" whenever leCore found a device was a
misreport: a painter with a good card was told their brush was accelerated
when every stroke ran on the CPU. `/api/status` now returns a `subsystems`
map (painting: cpu, simulation: gpu-or-cpu, shaders: browser gpu) and the
chip reads "paint: CPU - sim: GPU". Pinned.

### NOT DONE, and why: GPU painting
Porting the paint path to CuPy is a genuine project, not a sweep item. CuPy
is close to a drop-in for the array ops used here, but the deposit returns
arrays that are written straight into numpy layer buffers, so the seam has to
be drawn carefully or every stroke pays a host<->device copy that costs more
than it saves. This container has NO cupy, NO GPU and NO leCore, so anything
written here would be untested and unmeasured -- shipping it would be a claim
rather than a capability. Do it against real hardware, profile the copies
first, and keep numpy as the default path.

### Hosting: what exists now
- `GET /api/health` -- liveness. Deliberately cheap, takes NO lock and does
  not composite: measured 0.3 ms median while the engine was saturated with
  watercolour strokes. A probe that queues behind a slow stroke gets a
  working process killed by its orchestrator.
- `GET /api/ready` -- readiness; takes the lock and reads the canvas, so it
  fails while starting or wedged.
- `LESTUDIO_HOST` / `LESTUDIO_PORT` / `LESTUDIO_THREADS`. The thread cap sets
  the BLAS variables, which on a shared or phone-class box otherwise spawn a
  thread per core and thrash. It must run before numpy is imported, hence
  from the entry point.

### Hosting: the constraints a host MUST know
1. THE WORKSPACE IS ONE SHARED STUDIO, not a canvas per visitor. `WS` and
   `DOC` are module globals and the invite/presence machinery is built around
   collaboration. Expose it publicly and everyone paints on the same picture.
   Per-user isolation would mean keying the workspace by session and is a
   substantial change.
2. IT MUST RUN AS ONE WORKER. State lives in memory in the process; multiple
   workers would each hold a different painting.
3. `app.run()` is Flask's development server. Fine for one painter; put a
   real server in front for anything else.

## Usability test: 7 tasks walked as a first-time user
Task-based this time rather than fuzzing -- "paint in oil", "load your brush
from the palette", "mix two colours", "undo a mistake", "make the paint
thicker", "start over", "find out what a control does". Two failures, both in
the FLOW rather than the code.

1. A DIP DID NOTHING UNLESS "runs out" WAS ON. Painting always sends the
   COLOUR SWATCH, and the reservoir is only consulted in real-brush mode --
   so a user drags through the red mound, paints, and gets their old colour
   with no hint why. Nothing anywhere showed what was on the brush either.
   ONE FIX FOR BOTH: a dip now writes the loaded colour into the swatch, so
   it works in either mode and the swatch is the confirmation. (Mixing must
   NOT do this -- you are combining paint, not choosing a colour.)
2. "BUILD UP" SILENTLY ADDED LAYERS CALLED "p ~2". That cryptic name was the
   mystery layer in the very first user report; I fixed the palette case then
   and left the general one. Strata are now "sky - build-up 2" and switching
   the feature on says what will happen.

PASSED: the oil setup and its one-line instruction; mixing (button and
shift-drag, with a permanent hint); undo on the palette; the protected last
layer; and every brush control now carries a tooltip.

NOTE ON TESTS: four assertions pinned the stratum NAME ("~" in l.name), so a
clarity rename broke them -- exactly the "pin intent, not presentation" trap.
They now check `stratum_of`, which is the actual relationship.

## User-test scenarios, round 6 (stateful ops, and NaN again)
Applied round 5's fuzzing to a different axis -- the STATEFUL endpoints
(groups, stroke transforms, selections, lights, documents) plus concurrency
against the new palette surface.

NO CRASHES this time; the round-5 gate holds. But NaN got in through two
doors the paint validation does not reach:
1. A NaN LIGHT INTENSITY was accepted with a 200 and nothing downstream
   caught it, so `composite_lit` came back NON-FINITE -- the whole picture
   corrupted, invisibly, until you look at the output. Lights now check every
   number and both colours.
2. A NaN RECT built a selection mask nothing could use, and a missing corner
   reported just "'x0'" -- true, and useless. Rect/ellipse corners are
   checked and the message names what is missing.

THE PATTERN WORTH GENERALISING: NaN is the recurring defect in this codebase,
across three rounds now (paint colour, brush charge, light intensity,
selection corners). It never raises, it PROPAGATES, and every instance
returned 200. Any new endpoint that takes a number needs `_finite()`; that is
now the first thing to check when adding one.

CONCURRENCY ON THE PALETTE IS CLEAN: clearing and squeezing while the dock
reads it, palette churn while compositing the picture, and an undo/redo storm
while reading state all ran without a single 500 -- `_DOC_LOCK` covers the
palette because it hangs off the document.

## User-test scenarios, round 5 (hostile input)
Fuzzed the endpoints with malformed and extreme payloads -- the richest round
yet: TWELVE real defects.

/api/paint returned 500 for NaN coordinates, infinite coordinates, a NaN
radius, string coordinates, points that were not a list, missing points, and
a point with one number. Five more across /api/palette, /api/palette/paint,
/api/brush_load and /api/layer.

WORSE THAN THE CRASHES: NaN returned 200 and PROPAGATED. A NaN colour spread
into the layer's pixels and height map; a NaN amount put NaN on the brush
charge. NaN does not raise, it spreads, and it corrupts the document silently
-- strictly worse than a crash, which at least tells you something happened.

FIX: `_finite()` / `_clean_points()` / `_clean_paint()` validate ONCE AT THE
ENTRANCE. Placement matters -- my first attempt validated just before the
engine call, but the endpoint computes a bounding box straight from the
points, so bad values still crashed above it. Validation goes first, before
anything reads the payload.

POLICY, worth keeping consistent: values that are the wrong SHAPE or not real
numbers are refused with a 400 that says which field; values that are real
but out of range (a colour of 5, a negative radius) CLAMP. A caller being
sloppy is not a caller being wrong.

Two of these needed a second attempt because the handler I patched was not the
one serving the route: `/api/layer` edits go through `layer_edit`, not
`layer_ops`. Check which function actually runs before assuming the patch
landed.

## User-test scenarios, round 4 (presentation, long sessions, recovery)
Went after the browser side and the returning user -- areas never tested.

1. THE PALETTE WAS DRAWN STRETCHED. A canvas cannot letterbox, it STRETCHES,
   and the dock drew a content-cropped image into a fixed 76px strip:
   measured up to 33% vertical distortion, so round mounds rendered as ovals.
   Fixed on both sides -- the crop is now shaped for the strip it will be
   shown in (>=4.6:1, widening sideways and only then trimming), and the strip
   takes its height from the image instead of a constant.
2. AUTOSAVE 500'd AND LEAKED A PYTHON ERROR. Autosave is the crash net; when
   it cannot run the person needs to know their work is NOT being kept behind
   them. It now returns 200 with ok:false and says to save manually. (The
   failure here is the missing `holographic` module, i.e. environmental --
   but the HANDLING was the bug.)

HELD UP: recovery after a run of rejected calls; undo stack bounded at 120
strokes; state payload stays small after 80 media strokes; a dozen squeezes
make one palette layer; five documents opened and closed cleanly; export and
composite variants; the palette stays out of the layer list.

FLAKY, not a regression: `test_sized_export_and_render_at` failed once under
full-suite load and passes in isolation -- the third timing-sensitive test
found this way, alongside `test_deposit_costs_stay_live_friendly` and the
concurrency one (since fixed).

## User-test scenarios, round 3 (correctness under the new features)
Went after the deepest property instead of more edge cases, and it was the
right call -- the worst bug of the three rounds was here.

1. STRATA BROKE REPLAY DETERMINISM -- the only thing in the engine that did.
   Isolated by bisecting features one at a time: plain / real brush / mix /
   blend / knife all replayed exactly; anything with strata diverged by 0.33.
   Cause was mine: spilling was SUPPRESSED during replay (to stop it breeding
   layers), so a rebuild kept the overflow on the base instead of moving it
   up. The suppression was unnecessary -- `replay_layer` clears the whole
   chain first and `_stratum_for` reuses the existing link. Now exact, and
   layers do not grow across repeated replays.
2. A CLIPPED GLAZE THAT SPILLED PRODUCED UNCLIPPED STRATA, so the glaze
   escaped its base -- the exact workflow clipping exists to support. A
   stratum now inherits `clip`, `blend`, `opacity` and `alpha_lock`: it is
   the same mark continued, so it must meet the picture the same way.
3. `paint()` RETURNED NOTHING while `knife` and `blend_stroke` return their
   stroke id, so a caller could not name what it had just painted and the
   server was reaching into `strokes[-1]`. It returns the id now.

HELD UP: replaying twice is byte-identical; live stroke then commit then undo;
deleting every stroke of a column clears it; moving a stroke rebuilds the
strata beneath it.

HARNESS ERRORS AGAIN (5 across three rounds now, vs 7 real bugs): `paint`
appeared to "return None" for a reason that was really the missing return;
and my new test painted an UNRECORDED stroke on the layer it then tried to
stroke-edit, which the engine correctly refuses -- the guard was right and the
test was wrong. Also note `test_deposit_costs_stay_live_friendly` is timing
sensitive: it fails under full-suite load and passes in isolation.

## User-test scenarios, round 2 (feature combinations)
Targeted COMBINATIONS rather than single features -- that is where the one
real bug was.

REAL BUG FIXED: THE ERASER LEFT PAINT ON THE STRATA. A passage that built
past a layer's ceiling lives on several layers, and erasing cleared only the
base -- you wiped a mark and it was still there on the layers above. The
eraser now goes through the whole column (pigment, height, height_below and
material), because the strata ARE one body of paint. Note when testing this:
measure the CORE of the erased band; a soft brush leaves residue at its edges
by design and that is not the bug.

HELD UP under combination: brush state and palettes do not leak between
documents; knife on a stratum column then undo; material + real brush + wet
mix together; switching medium repeatedly on one layer; undo/paint/redo out
of order; 60 strokes with no measurable slowdown; export with a palette
present; a deep column then composite; 0x0 and 40000x40000 documents refused;
undoing 30x past a layer's own creation and redoing 30x rebuilds exactly.

MY HARNESS WAS WRONG THREE TIMES, and each looked like a bug:
- `/api/select` takes `tool` + `params` with `x0,y0,x1,y1`, not `action`/`x,y,w,h`.
  With the wrong shape no selection is made, so paint "escaped" it. Selection
  clipping is correct.
- Undoing 30 times undoes the LAYER'S CREATION, so a later `layer(id)` lookup
  raises KeyError -- in the test, not the app.
- (round 1) measuring alpha on the opaque Background layer reads every pixel
  as painted.
CHECK THE HARNESS BEFORE FILING THE BUG: so far more scenario "failures" have
been my own test code than real defects.

## Findings from running user-test scenarios
Behaved like a tester rather than a developer -- ran the awkward things a real
user does (spam undo, delete layers, absurd brush sizes, paint off-canvas,
squeeze twice, resize mid-session) and kept only what actually broke.

REAL BUGS FIXED:
1. DELETING THE ONLY LAYER stranded the document with none. Painting into it
   then 500'd with a bare "no such item", and the app offered nothing to
   paint on -- a dead end reached in one click. The last canvas layer is now
   protected with an explanation.
2. PAINTING A HIDDEN LAYER SUCCEEDED SILENTLY. The user sees nothing happen
   and has no way to know why. Now a 400 saying the layer is hidden.
3. PAINTING A DELETED LAYER 500'd. Now a 400 -- a stale reference is a user
   error, not a server fault.

FALSE ALARMS -- my harness, not the app (worth recording so they are not
"fixed" again):
- "brush ran dry but still painted the far end": I measured alpha on the
  opaque BACKGROUND layer, where every pixel reads as painted. On a
  transparent layer the stroke stops dead where the charge hits zero.
- Three "crashes" were `layers[0]` in my own scenario code after a previous
  scenario had deleted the only layer.

AND ONE LESSON: my friendlier "that layer is gone" message broke
`test_errors_are_actionable_and_never_silent`, which asserts the error NAMES
the offending item. A friendly message that drops the identifier trades one
kind of useless for another -- say both.

## Making the palette frictionless
- SCRAPING WAS UNRECOVERABLE. `/api/palette/clear` dropped the whole surface,
  so one click destroyed a session of mixing with nothing to press
  afterwards. It now clears the paint through the palette's own history and
  stays on the undo stack; the button reads "scrape" and says Ctrl+Z brings
  it back.
- Fixing that exposed a second bug: THE UNDO SNAPSHOT DROPPED THE `palette`
  FLAG (it is a separate code path from save/load, which already carried it),
  so undoing anything on the palette turned it back into an ordinary layer
  and its dock 404'd. The snapshot now carries `palette` and the stratum
  chain too -- an undo used to sever that as well.
- MIXING NEEDS NO MODE. Shift-drag mixes whatever state the toggle is in, so
  you never have to remember what the palette is set to. The toggle stays for
  discoverability and the hint reads "drag to load the brush, shift-drag to
  mix".
- IT WAS INVISIBLE UNTIL IT EXISTED. Choosing a medium with no palette yet
  showed nothing at all, so anyone who picked oil without running a setup
  never learned the palette was there. The strip now appears with an
  invitation to squeeze paint out.
- The strip is 76px rather than 54px: mixing in a 54px band is cramped, and
  the layout budget had room.

## Polish sweep after the palette-surface change
Two real bugs, both found by asking what the ARCHITECTURE CHANGE broke rather
than by looking for new features to add:

1. THE PALETTE WAS LOST ON SAVE. It is a new Document and nothing serialised
   it, so mixing a set of colours and reopening lost them -- exactly the
   silent loss the save/load rule exists to prevent, and I broke my own rule
   the moment I added new state. It now saves with the picture. Its arrays go
   under a "pal_" PREFIX: both documents number their layers from L1, so
   without it the restore died with KeyError: 'layer_L2'.
2. CTRL+Z AFTER MIXING DID NOTHING. The palette keeps its own undo stack but
   `/api/undo` only ever spoke to the picture. Undo and redo now follow
   whichever surface was edited last (`_edited_palette_last`, set by the
   palette paint route and cleared by the picture's), falling back to the
   picture when the palette has nothing left. Undo must undo the last thing
   you did, wherever you did it.

CHECKLIST for the next architecture change: does the new state SAVE? does
UNDO reach it? does it show up in the layer list when it should not? is it
excluded from EXPORT? Each of those has been a real bug here at least once.

## The palette is its OWN SURFACE, not a layer
`Document.palette_doc()` is a companion Document (560x150), not a layer of
the picture. It cannot be nudged by editing the painting, is never exported,
never appears in the layer list, and never spills into strata -- which is
what bred "Palette ~2 ... ~6" when it lived in the document.

THE KEY PIECE is `_res()` / `_brush_host`: the palette borrows its OWNER's
brush reservoir, so a dip on the palette loads the brush you then paint the
picture with. Two reservoirs would silently disagree and a dip would appear
to do nothing. Every reservoir read and write in `paint()`, `load_brush()`
and `brush_state()` goes through the host.

Endpoints: `POST /api/palette` squeezes mounds (sized for the palette
surface -- the canvas-derived default made pea-sized blobs you could not dip
in), `GET /api/palette.png` renders it with an `X-Palette-Box` header,
`POST /api/palette/paint` is BOTH dipping and mixing, `POST /api/palette/clear`
scrapes it. Mixing is ordinary painting on that surface with `mode: blend`,
so the blender, knife and wet-on-wet behave exactly as on the canvas. The app
has a "mix here" toggle that switches a drag between dipping and mixing.

WHEN CHANGING THIS: several tests encoded the old layer-era contract
(`state.palette_layer`, passing a picture layer id to `/api/palette`,
checking the picture's layer list for a Palette). Architecture changes make
tests that pin the OLD shape fail for the right reason -- rewrite them to the
new contract rather than patching the assertion.

## Brush controls say what they DO, in words
Second user report: "some of the settings are just a single character or
symbol without explanation". A glyph plus a tooltip is still a guess -- you
have to hover the right thing to find out what it is. Renamed:
  ≡ -> "build up"      🖌 -> "runs out"     🎨 -> "palette"
  ⇋ -> "mirror"        ◎ -> "snap"         ⬡ -> "shapes"
  "—" -> "paint 62%" / "paint ∞"   (a bare number names nothing)
  bMirror options ↔ ↕ 3× 6× 8× -> left-right / up-down / radial 3 / 6 / 8
The paint row had grown to SIX controls including two bare glyphs; it is now
two labelled rows ("Load / Mix", then "Brush: runs out | paint ∞ | build up"),
both hidden until a medium is chosen so the default panel is unchanged.

`test_no_brush_control_is_labelled_only_by_a_symbol` scans the panel for
symbol-only buttons, bare-glyph labels, and any button or select with no
tooltip. It immediately found two more I had not noticed -- `bMirror` and
`bSel` had NO tooltip at all -- which is the point of writing it as a scan
rather than fixing the six I knew about.

TWO TRAPS while doing this:
- Building a replacement tag with `rindex('>')` finds the CLOSING `</button>`,
  not the end of the opening tag, and silently produces a doubled close. The
  markup then parses "fine" but the DOM shape is wrong and the toolbar gate
  fails with something unrelated-looking ("inputs=1"). Balance-check tags
  after editing markup: `<button` count vs `</button>` count.
- `test_backlog_quickshape_mandala_grow` asserted the literal option text
  `<option value="6">6×</option>`. Renaming the label for clarity broke a
  test that only cared that the option exists. Pin the VALUE; the visible
  text is presentation.

## USER-REPORTED CONFUSION, and what caused it
A first-time user ran a studio setup and got "Palette, Palette ~2 ... ~6" in
the layer list, could not tell how to load the brush, and could not find the
thing that makes paint run out. Four causes, all mine:

1. THE PALETTE SPILLED INTO STRATA. Mounds are laid heavily on purpose (they
   must be a pile you can dip into), and the oil setup switched `auto_stratum`
   on, so they overflowed and bred a layer per stratum. `lay_palette` now
   suppresses spilling and restores the setting, and a palette layer never
   spills even if painted directly.
2. A SETUP SWITCHED ON AN ADVANCED FEATURE SILENTLY. Strata are removed from
   the setups. A first-time painter did not ask for them, and they are what
   produced the junk layers.
3. THE DOCKED PALETTE WAS ALSO IN THE LAYER LIST, where it read as
   uninterpretable junk. `drawLayers` now filters it out -- it is docked, it
   is not part of the picture.
4. THE CONTROLS WERE GLYPHS. A lone brush icon and a bare "100%" tell you
   nothing. The toggle now reads "runs out", the meter reads "paint 62%" /
   "paint INF", and the dock carries a permanent line: "palette - drag across
   a colour to load the brush". A toast is not documentation; it vanishes.

LESSON: every one of these passed its own tests. The tests checked that the
features WORKED, not that the result was interpretable. Look at the layer
list and the panel after running a setup, not just at the endpoint responses.

## Studio setups: one action instead of five switches
Painting properly meant finding FIVE separate switches scattered across three
panels -- a medium (Brush), real brush (Brush), the paper (Layer), a palette
to dip in (Colour), and sometimes strata (Brush). Each is sensible on its own
and together they are not a workflow.

`STUDIOS` + `applyStudio()` offer "oil painting", "watercolour" and "ink
drawing" AT THE TOP OF THE MEDIA MENU, which is where the intent forms:
someone opening that menu to try oil is exactly the person who wants the rest
of it. A setup sets the medium, the paper, real brush, strata and mix, seeds
the swatches, squeezes a matching palette into the dock, and says in one line
what it did and what to do next.

Two things worth keeping:
- It drives the ordinary endpoints (`/api/paper`, `/api/palette`,
  `/api/stratum`) and clicks the ordinary toggles rather than having a
  privileged path, so a setup can never drift from what the controls do.
- Every setup must touch EVERY axis. A setup that only picks a medium is a
  medium picker wearing a setup's name; `test_studio_setups_are_offered...`
  checks each axis is set by all three.

NOTE for whoever edits the bMedia handler: `tests/test_material_ui.js` grabs
it by regex, and it is now multi-line -- a single-line pattern silently
grabs half a function and the gate dies with "Unexpected end of input".

## Release-polish pass
- AN EMPTY BRUSH USED TO KEEP PAINTING. `film` carries a 0.5 floor so a
  merely tired brush still puts colour down, but that floor ignored the
  reservoir -- a brush at zero charge laid a 50% film and real-brush mode
  meant nothing. `dry_gate` fades coverage out over the last ~18% of charge
  (a hard cut-off would end a stroke with an edge in mid-air). Verified
  across the range: 0.00 charge lays nothing, 0.05 ghosts, 1.00 unchanged.
- The app now says WHY nothing appeared: `noteBrushCharge` toasts once per
  dry spell rather than letting the painter keep stroking air. A stroke that
  lays nothing with no explanation is the worst thing this app can do.
- User-facing errors must not be Python errors. `/api/palette` was returning
  "could not convert string to float: 'red'". Pinned by
  `test_user_facing_errors_are_not_python_errors`, which scans for Traceback,
  NoneType, "object has no attribute" and friends in any 4xx body.
- CHECKED AND ALREADY FINE, do not redo: 99 `api()` call sites never inspect
  `r.error`, which looks alarming until you read `api()` -- it surfaces every
  failure centrally through `friendlyError` and a toast. Verify the choke
  point before "fixing" the call sites.
- A full first session (paper, squeeze a palette, load, paint in oil, blend,
  knife, undo, redo, composite, refresh) runs in 534 ms with every step under
  150 ms and no failures.
- ~~palette lands in the picture~~ DONE: `palette_layer()` find-or-creates a
  dedicated "Palette" layer and `lay_palette()` defaults to it, so squeezing
  no longer touches the layer you are painting on (pinned by comparing the
  art layer's pixels before and after). Squeezing again REUSES the layer
  rather than breeding one per squeeze, `palette_layer` is exposed in
  `/api/state` so the app can find or hide it, and it is an ordinary layer
  otherwise -- hide or delete it with the normal controls.
- ~~palette docked~~ DONE. `canvas_layers()` now excludes palette layers the
  same way it excludes wall layers, so the palette is in NEITHER the canvas
  view nor an export. `palette_png()` / `GET /api/palette.png` renders it
  cropped to its paint and returns the document region it occupies in an
  `X-Palette-Box` header; the app draws that in a strip under the colour
  row and maps a drag there back to real document coordinates. So a dip in
  the dock is an ORDINARY stroke on the palette layer -- same paint, same
  physics, same brush reservoir -- rather than a second code path that
  would drift. The strip is hidden until there is paint to show, so it
  costs no height in the panel's no-scroll budget until it is used.
- STILL OPEN: 98 controls carry no tooltip; most have self-explanatory
  labels, so this wants human judgement rather than a script.

## The intermittent concurrency failure: FOUND AND FIXED
Four sightings across the sessions, never reproduced, and it was quietly
eroding the before/after failure diff every change here is judged by.

REPRODUCED by running the test 40x IN-PROCESS rather than through the runner
-- it fails ~5/40, and only when server state accumulates across runs, which
is why one-off isolated runs always passed and made it look like noise.

ROOT CAUSE: compositing reads the document's SIZE and then its LAYERS. Those
are two separate reads, so a resize landing between them composites layers of
the old shape into a frame of the new one:
  "operands could not be broadcast together with shapes (350,500,3) (200,300,3)"
Not a numerical fluke, not load -- a plain unguarded read-modify race.

FIX: `_DOC_LOCK` (RLock, re-entrant because composite handlers call back into
other guarded helpers) around `comp_png`, `doc_ops` and `new_doc`. Guarding
only the first two took it from 5/40 to 1/120; `/api/new` switches the active
document and was the second door. With all three: 0/120. No measurable cost
(a heavy stroke is unchanged). Pinned by
`test_compositing_is_atomic_against_a_resize`.

LESSON: an intermittent test is worth chasing properly rather than re-running
until it passes. Getting the failing RESPONSE BODY was what cracked it --
the exception never reached `app.logger` or stderr, so logging hooks found
nothing; wrapping the test client to capture 500 bodies did.

## UI/UX pass findings
- A DUPLICATE KEY IN THE TOOL MAP IS SILENT. JS keeps the later one and the
  earlier binding simply stops working. This has now bitten TWICE -- the
  blender took Nudge's `n`, the knife took Select-strokes' `k` -- both times
  with no error and no visible symptom unless you happen to press the key.
  `test_no_shortcut_is_silently_double_bound` now scans the map. Knife is `q`.
- A RIGHT-CLICK-ONLY CONTROL WITH NO VISIBLE STATE IS BURIED. The knife's
  blade was cycled by right-clicking the tool and shown nowhere, so there was
  no way to learn it existed or to tell which blade was active. It now shows
  the blade as its glyph, carries `data-blade`, and clicking the tool while
  it is ALREADY active cycles it -- discoverable without knowing the trick.
- Dead `$()` refs are not necessarily bugs: `toast`, `dupsPop`, `addMenuList`
  and `addSearch` are all created dynamically. Check before "fixing" them.
- Every control added across the paint work was exercised end-to-end through
  its endpoint (13/13) and is pinned by
  `test_every_new_control_reaches_its_endpoint`, which fails both ways: an
  endpoint with no control is buried, a control with no endpoint is dead.
- STILL OPEN: 98 controls carry no `title` tooltip. Most are tabs and menu
  items whose label says enough, but it is worth a pass with fresh eyes.

## The palette knife, and curing the stepping between strata
Two halves of one problem: paint could span several strata, nothing could
shape it, and each stratum LIT AS ITS OWN SLAB.

STEPPING: `_shaded_pixels` used only `lyr.height_map`, so a stratum read as a
thin sheet resting on a plateau. Each stratum now carries `height_below` --
what it sits on, recorded as it is created -- and shades on the TOTAL. A
stratum is the top of a column, not a separate sheet.

`knife()` / `POST /api/paint {"mode":"knife","knife":...}` shapes existing
paint. It reads the whole column via `_stratum_chain` + `_column`, applies
the blade, and pours the result back with `_refill_column` (fill each layer
to the cap before starting the next). That refill is itself the cure for
stepping, because the body is re-levelled as one mass rather than a stack.
Modes: `smooth` (level, and de-step), `push` (plough a ridge, volume
conserved -- pinned by test at >90% retained), `scrape` (take the tops off),
`spread` (drag into a thin film). Recorded and replayed in order like a
blend. Tool `K`; right-click the tool to cycle the blade.

DO NOT PUT AN EMOJI IN A PATCH STRING for index.html. A surrogate pair like
"\ud83d\udd2a" is not encodable and the write dies mid-file, truncating it to
ZERO BYTES. This has now happened TWICE. Use an ASCII glyph, or "\U0001F52A"
form, and always read back and assert on length after writing.

## Paint strata: building past the layer ceiling
A layer holds `_HEIGHT_CAP` (4.0) of paint. Past that the height field was
just CLIPPED, so a worked passage saturated after about TWO loaded passes and
every later stroke added nothing -- measured: pass 2 hits 4.00 and pass 12 is
still 4.00, the mark flattened to a plateau at exactly the cap. That is the
inorganic stepped look.

`Document.auto_stratum` (opt-in, `POST /api/stratum`, button in the brush
panel) spills the excess onto a fresh layer that starts from zero, the way a
painter builds heavy impasto in campaigns. Measured: 9 loaded passes give
22.1 units of relief across 6 strata instead of a flat 4.0.

Three things to know:
- IT MUST CASCADE. Spilling once only defers the ceiling -- the new stratum
  filled and flattened in its turn (measured 28.9 units on a layer whose cap
  is 4, because nothing capped IT). The spill loops until the excess is gone.
- Name strata from the ROOT of the chain or a deep build reads
  "p ~2 ~2 ~2 ~2 ~2 ~2". `stratum_root` carries it.
- A stratum is DERIVED from the base layer's strokes, so `replay_layer` must
  clear the WHOLE chain first or every rebuild stacks another copy.

OPT-IN ON PURPOSE: it changes how a worked passage builds, and a stroke onto
a full layer costs ~815ms (vs ~240ms) because each stratum is another full
composite pass. It also broke the real-brush reload test when on by default,
since the pile a dry brush scrapes no longer sits where that test expects.

## Using strata in a painting (dogfooded, mixed result)
Impasto in the lights and thin paint in the shadows is the standard use for
paint body, and strata make it physical. Three things learned putting it in
the still life, all of which cost a render:
- `auto_stratum` is DOCUMENT-WIDE, so it must be toggled around the passage
  that wants it. Left on for the whole painting, the watercolour ground
  spilled too and the wash stacked as visible horizontal STRIPES.
- A clipped glaze binds to the layer directly beneath it. Building strata on
  the base BEFORE adding the glazes left the glazes clipped to a partial
  stratum, so they stopped working and the forms went flat again. Order:
  base -> glazes -> impasto on its own layer above.
- Impasto is BODY, not a patch of different colour. Thirty fat opaque dabs in
  a narrow arc piled into one hard-edged crust; many small touches close to
  the local hue, spread over the whole lit half, is the right shape.

HONEST RESULT: the strata engage (14.1 units of relief against a 4.0 cap) but
the painting is not clearly better for it -- the thickened region reads as a
crusty patch rather than as brushwork, because short thick strokes at high
relief have blocky ends. The glaze-only version is arguably the better
picture. The mechanism is sound; what it needs is a deposit whose stroke ENDS
taper, which is the same frayed-cap problem noted under the mark generator.

## Glazing: shadow and light on SEPARATE layers
The thing that finally made forms turn. Local colour on a base layer, then a
shadow glaze on its own MULTIPLY layer and a light scumble on a SCREEN layer,
both `clip=True` to the base, and the specular on a small ADD layer. A glaze
darkens without repainting, so the turn is a true gradient and the local
colour still reads through it -- where stroke-by-stroke modelling kept
landing as concentric ripple no matter how it was blended.

Three parameter traps, all of which cost a render each:
- The mass must be an OFFSET BLOB, not a ring. Sweeping a band inward while
  sweeping the angle draws a spiral, which lands as a vignette round the
  whole rim rather than darkness on one side. A form turns because the
  shadow SITS somewhere: give the mass a centre offset toward the shadow.
- Glazes need real strength. At opacity ~0.24 (alpha mean 0.235 measured) a
  multiply glaze barely darkens anything and the form stays flat; ~0.55 is
  where the turn reads. Diagnose this by checking the glaze layer's own
  alpha and centroid, not by squinting at the composite.
- Keep the light mass TIGHTER than the shadow mass, or the two cancel.

## Painting realistically: what earlier blocked it
Three attempts at a realistic still life did not get there. The forms stay
flat -- saturated poster shapes with no turn from light into shadow. What was
ruled out BY MEASUREMENT, so nobody repeats it:
- NOT the palette recipes. They separate properly: measured luminance runs
  0.119 (core) / 0.163 (dark) / 0.221 (mid) / 0.354 (light) / 0.869 (spec).
- NOT `alpha_lock`, and NOT wet pickup. Six opaque passes over a dark ground
  reached only ~60% of the intended colour, and `mix` at 0.0 / 0.14 / 0.50
  gave BYTE-IDENTICAL results -- so pickup was never involved.
- IT WAS THE BRUSH RUNNING DRY. `real_brush` depletes across strokes, so
  charging once per value family left the later strokes in that family laying
  almost nothing. Correct physics; the mistake was painting as if the brush
  were infinite. Fixed by reloading whenever charge drops below ~0.55, which
  is what a painter actually does.
- A REAL BUG WAS ALSO FIXED HERE: the lit families were swept from pi to
  TAU*1.02, which wraps right around onto the shadow side and repaints it.
  Screen y is down, so a key from the upper left lights roughly
  [pi*0.86, pi*1.62] and nothing beyond.

STILL FLAT after all of that. The remaining suspects, untested: bands painted
at fixed radii read as concentric ripples rather than a gradient (the same
trap as the earlier vinyl-record spiral, in a subtler form); and the value
families may need to be laid as large soft masses first and only then broken
into strokes, rather than built stroke-by-stroke from the start.

## Multiplane: what actually works, and what does not
CORRECTION to my earlier note: I claimed `view="persp"` was "markedly more
convincing" than flat. MEASURED, IT IS NOT. Flat vs persp differ by ~2.4%
mean, ~19% of pixels -- and that number DID NOT CHANGE when I raised the
pane separation from 26 units to 88 (0.0258/18.3% -> 0.0241/19.3%). Most of
that difference was the palette artifact below, not parallax. Perspective is
close to inert for this content; if real multiplane parallax is wanted,
`composite_volumetric`'s persp path needs looking at.

WHAT DOES WORK, and it is most of the win: a directional light with
`shadows=True` over layers at different `z_off` throws each pane's shadow
onto the one behind it. That is real computed depth and it reads immediately.
Big separations (wall 0, table 6, back fruit 46, front fruit 88, stem 104)
give clearly offset cast shadows.

- `z_off` buys the depth; `thickness` is the slab's own edge. At thickness 8
  every object was rimmed with its own cut-out shadow like a sticker.
- Light intensity comes WAY down vs the unlit composite (there is a 0.30
  ambient floor already): key ~0.6, cool fill ~0.2, no view light. Key 1.15
  blew the background to white.
- Per-layer `relief` ~0.35 on painted panes, or every impasto ridge lights
  like a crater.
- PALETTE ARTIFACT: a layer sitting ON TOP of deep panes renders as grey
  embossing with no pigment in the volumetric view. Colour survives fine when
  the same layer is tested in isolation, so it is about stack position, not
  the layer. Workaround: keep the palette at the BOTTOM of the stack.

## Multiplane: painting on panes of glass
The engine already had everything for this and it was never used together.
Put each element on its OWN layer with a `z_off`, add a directional light
with `shadows=True`, and render with `composite_lit(doc, view="persp")`:
the key light throws each pane's shadow onto the one behind it, so the
depth is COMPUTED rather than painted. The perspective view is markedly
more convincing than flat -- the forms read as volumes.

Three things learned setting it up:
- `z_off` BUYS THE DEPTH (it sets how far a pane's shadow falls behind it);
  `thickness` is the slab's own edge. At thickness 8 every object was rimmed
  with its own cut-out shadow like a sticker. Thin panes, well separated.
- Light intensity has to come WAY down versus the unlit composite: the first
  attempt at key 1.15 blew the background to white. Key ~0.6, cool fill
  ~0.2, no view light. There is a 0.30 ambient floor already.
- Per-layer `relief` needs easing (~0.35) on painted panes or every impasto
  ridge lights like a crater.

ARTIFACT, not chased: in `view="persp"` the palette mounds lose their colour
and render as embossed grey ghosts. Layers with no meaningful depth setup
seem to drop their pigment in the volumetric composite. Worth a look.

## Dogfooding: the apple still life (second time)
Repainted the still life using the media rather than plain dabs: watercolour
ground blended flat, oil apples built from a palette with real-brush mixing,
soft washes for the cast shadows, ink only for a few accents. Four real
findings, all now fixed or pinned:

- BUG, FOUND ONLY BY PAINTING: the reservoir write-back was gated on
  `record`. That conflates "should this stroke be undoable" with "did the
  brush pick anything up", so dabs passed `record=False` -- the normal way to
  lay texture without a stroke record per dab -- silently did not load the
  brush. Dipping into a palette in a loop left the brush exactly the colour
  it started, and my apples came out grey. Now gated on `charge0 is None`
  (i.e. not a replay). Its twin: a live stroke repaints from `before` each
  flush, so the reservoir must REWIND with the pixels or the brush empties
  once per flush. Both pinned.
- LAYER PER OBJECT, and lock the silhouette. Painting both apples on one
  layer let the blender drag one form's colour into the other, and scumble
  strokes sprayed past the edges as a spiky fringe. Laying the silhouette
  solid, then `edit_layer(lid, alpha_lock=True)` before any modelling, keeps
  every later stroke INSIDE the form -- the fringe disappears entirely and
  each object stays independently editable. This is the workflow to reach
  for, not a workaround.
- MIX BELONGS AT THE PALETTE, NOT ON THE WORK. Modelling with mix=1.0 makes
  the brush re-pick the canvas on every dab and the whole form converges to
  one muddy average. Mix while charging the brush; model at mix~0.2.
- The BLENDER AVERAGES. ~108 blend passes over one apple flattened it to a
  pale even field. Coverage was never the problem (measured: a single heavy
  oil pass reaches 0.988 alpha and lays the exact colour asked for) -- it is
  purely blend count. Blend sparingly and only where two values meet.
- Dipping MIXES, it does not replace. Starting a brush at grey and dipping
  red gives dirty red, which is correct physics and a UX trap: load from the
  pile first (`load_brush(color=...)`), then dip to modify.
- The blender CARRIES paint, so a long blend stroke is a transport, not a
  soften: full-diameter radial passes dragged the highlight clear across a
  form and left a starburst. Blend with many SHORT local strokes.
- Painting concentric rings AND blending concentrically gives every form a
  vinyl-record spiral. Scumble with scattered short strokes and blend ACROSS
  the form.

## Paper and canvas stocks
`_PAPERS` (canvas, rough, cold_press, hot_press, smooth, linen) parameterise
the substrate: `grain` scales feature size, `weave` how much crossed-thread
structure shows over the fbm, `depth` the amplitude, `drink` how far a wash
wicks. `Document.set_paper()` / `POST /api/paper`, select in the Layer panel.
The stock is stamped onto each LAYER as it is painted, because the shading
paths only receive a Layer and never the Document.

MEASURING IT: use how strongly pigment tracks the paper's DIPS
(corr(tooth, alpha), negative = settling). Smooth reads +0.10, rough -0.21,
linen -0.23. Do NOT use variance proxies -- the wash alpha saturates near 1
over most of a loaded mark so the mottle is clipped out of any std, and a
high-pass misses it unless the cutoff happens to match that stock's grain.
I got both wrong before measuring the mechanism directly, and the tooth
fields were correct the whole time (std 0.018 smooth to 0.183 rough).

## Sweep findings (third pass)
- NEAR MISS, read this before editing index.html: a patch script that read
  the file WITHOUT an explicit encoding and wrote it back truncated it to
  ZERO BYTES when a non-ASCII character hit the write. Always
  `io.open(p, encoding="utf-8")` both directions, and always read back and
  assert on length + expected content after writing. Recovered from the
  packaged zip, which is the argument for packaging often.
- The Colour row has a WIDTH budget as well as the panel having a height
  budget. A text label ("Squeeze out") wrapped the row and broke
  `test_layout_fits_the_viewport`; a 22px icon button fits. Same trap as the
  Material row, in the other axis.
- CORRECTION to the second-pass note: I claimed the flaky tests were
  "order-dependent, not random". Three consecutive full runs of their chunk
  did NOT reproduce a failure. 0/6 in isolation plus ~1 failure in 6 suite
  runs is equally consistent with simply RARE. The honest state is: cause
  unknown, frequency low, still worth a dedicated look because it erodes the
  before/after failure diff. Do not treat the order-dependence claim as
  established.

## Sweep findings (second pass)
- `_bristle_tracks` built its lookup with a PYTHON LOOP over up to 30 hairs,
  one exp() per hair over the whole table -- ~2M exponentials per live flush
  for what is pure setup. Now one broadcast exp(), and the table is 24x512
  rather than 64x1024: both axes are read with linear interpolation and the
  functions sampled are slow (contact flicker is under one cycle along the
  whole stroke), so the big table cost 4x the work for detail interpolation
  was reconstructing anyway. Verified no quality loss by render.
- THE FLAKY TESTS ARE ORDER-DEPENDENT, NOT RANDOM. `test_concurrent_requests`
  and `test_volumetric_layers` each failed 0/6 times in isolation but fail
  intermittently inside a full chunk. That means state leaking between tests,
  not timing noise. Module globals are the suspects (`_TOOTH_CACHE`, which is
  REBOUND rather than mutated, `_MUT_REV`, the shared server `DOC`). Worth a
  dedicated look: two flaky tests quietly erode the before/after failure diff
  that every change in this project is judged by.
- `edit_layer` has its own property whitelist, separate from the server's.
  Adding a layer property needs BOTH or the API returns 200 and silently
  discards the value. And the `is not None` gate means a property whose zero
  is meaningful (gravity 0 = flat) must be handled deliberately.
- `test_control_coverage_sweep` scans quoted strings inside `edit_layer`, so
  a comment containing a quoted word registers as a phantom property.

## UX GAPS (engine-only features with no UI)
These work through the API but have no control in the app, and should get one
before the next feature:
- ~~palette UI~~ DONE: a palette button in the Colour row squeezes your
  SWATCH colours out as real mounds (`POST /api/palette`). It reads
  `recentCols`, the same list the swatch strip draws, so there is no second
  source of truth to drift. Mounds still land in the picture; a docked
  palette strip beside the canvas is still the nicer end state.
- ~~per-layer gravity~~ DONE: a named surface select (easel / flat / up /
  left / right) sits in the Layer panel's Relief row. Named surfaces, not
  gravity degrees -- nobody reasons about their canvas in degrees, and two
  raw sliders also broke the panel's no-scroll budget.
- watercolour's `absorb`/`edge_dark`/`granulate` are fixed per medium; no way
  to pick a rougher paper or a more granulating pigment.
- loading the brush per-lane (`load_brush(color=[...])`) is API-only; the
  charge button only fills uniformly.
Note `/api/analyze`, `/api/mind` and `/api/schema` are intentionally
API-only and are exempted in the reachability sweep.

## The palette
`lay_palette()` squeezes mounds of thick paint onto the canvas. Deliberately
NOT a new mechanism: a palette is just paint, piled high enough to clear the
"this is a pile you can dip into" threshold the reload model already uses, so
dipping, gathering two colours on one brush, and scraping a mound thinner all
fall out of the existing physics.

Two couplings that bit when tuning it:
- A full brush must still EXCHANGE paint at a mound. Gating reload on "has
  room" meant a freshly loaded brush dipped in a second colour only got the
  slow trickle, so red into white stayed red -- which defeats the point.
- The SCRAPE must scale with the reload rate. The stroke deposits onto the
  mound as well as lifting from it; if lifting does not keep pace the mound
  never goes down and paint stops being conserved exactly where dipping
  happens. reload_rate and the scrape coefficient move together.

## Real brush mode
`paint(..., real_brush=True)` makes the brush a PHYSICAL OBJECT holding a
finite amount of paint (`Document.brush_charge` / `brush_color`, exposed as
`brush_state` in `/api/state`). It empties as you work and eventually lays
nothing. It recharges by dragging through paint that has genuine BODY -- and
what it lifts, the canvas loses, because paint is conserved. `load_brush()`
/ `POST /api/brush_load` is the palette dip that fills it right back up.

Two things that were wrong on the first cut and matter if you touch this:
- Recharge must be gated on a PILE, not on `body`, which saturates at 1. A
  normal stroke and a six-stroke mound looked identical to it, so the brush
  recharged off its own last mark and charge went UP while painting. Only
  height above what one loaded stroke leaves (`raw - 2.2`) counts.
- `charge0` is FROZEN into the stroke record. The reservoir is stateful
  across strokes, so re-deriving it at replay would mean nudging stroke three
  of forty retroactively changed how much paint stroke forty had. Within a
  stroke, depletion and pickup are recomputed from canvas state and stay
  deterministic; the starting charge is history. A replay never spends live
  paint (pinned by test).

`_brush_walk()` does mixing, depletion and recharge in ONE 1-D pass along the
path, looked up per pixel by arc length, so none of it costs more as the
brush grows.

## Stroke groups
`group_strokes()` / `POST /api/stroke_group` bundle a blended passage into one
object. `_expand_strokes()` resolves group ids to member ids and every stroke
editor runs through it, so a group id works anywhere a stroke id does while
members stay individually editable. Selections are de-duplicated: a group and
one of its own members together must not transform that member twice. UI:
Group/Ungroup in the stroke-select panel, Ctrl+G / Ctrl+Shift+G.

## Wet-on-wet mixing
`paint(..., mix=0..1)` makes the brush PICK UP the wet paint it crosses and
carry it forward, so a stroke changes colour from the crossing onward and
muddies as it gathers more -- the asymmetry is the tell, since a stroke that
mixes evenly along its length reads as a gradient, not as paint. `_wet_mix()`
walks a reservoir ALONG the path (a 1-D recurrence over a few hundred points)
and the per-pixel colour is a lookup by arc length, so cost is independent of
brush size. Pickup is gated by canvas alpha AND paint body: bare canvas and
dry stain give nothing back. The rate is PER PIXEL TRAVELLED (0.012), not per
crossing -- at a per-step 0.55 a brush crossing one 34px bar compounded
(1-0.55)^34 and came out pure red. `mix` rides the stroke record, or a replay
unmixes the painting.

## Performance
Measured, not assumed (`test_deposit_costs_stay_live_friendly` pins it).
Live-flush latency vs the pre-deposit baseline: 1080p r12 41->59ms median,
1080p r28 50->56ms, 4K r28 96->93ms. Mixing is free (it is a 1-D recurrence).
The dominant cost is `_paint_flow`, which is PRE-EXISTING and unrelated to
the deposit model; profile before optimising anything else. `_bristle_comb`
is a 1-D LUT gather, not up-to-26 exp() passes over the window -- that was
the single most expensive thing the deposit first added. The real structural
win available is incremental live painting: every flush currently repaints
the whole stroke from `before`, so cost grows with stroke length.

KNOWN GAP -- THE BIG ONE, and the architecture is the reason. Strokes still
read as extruded glossy bars rather than paint. Ruled out BY MEASUREMENT (do
not re-litigate these): the piped rim outline is NOT the berm and NOT gloss
-- rendering with berm=0 and gloss=0.04 leaves the outline unchanged. It is
inherent to a uniform slab with a smooth boundary sitting on flat canvas.

The root cause is how the mark is GENERATED. leStudio stamps discs along the
path and modulates the result afterward, so the silhouette is always a
capsule and every fix (comb, tooth, edge bite, load wobble) is decoration on
top of that capsule -- which keeps showing through. WetBrush (SIGGRAPH Asia
2015) and Adobe's bristle-brush patents do the opposite: each BRISTLE is
rasterised as its own swept quad and the stroke silhouette is the UNION of
those tracks. That is why their marks have ragged edges, interior gaps and
frayed ends without any of it being added afterward.

RECOMMENDED NEXT STEP: replace the mark generator. Give each bristle its own
lateral offset, its own load, and its own contact on/off along the path, and
build the mask as the union of bristle tracks. Then the comb, the ragged
boundary, dry-brush skips and chisel ends all emerge from ONE mechanism
instead of four bolted on. This is a rewrite of the mask loop, not a tweak.

Also still open: small isolated strokes (radius < ~15) still read as glossy tubes.
Ruled out by measurement: specular level (identical at gloss 0.34/0.14/0.04),
film alpha (0.98-1.0 in core, so not translucency), comb lane separation, and
canvas relief (helped, did not solve). Remaining untested suspects: strokes
are uniform in cross-section along their length, and the demos used no
`taper`, so ends are hemispherical caps rather than a lifted brush.

## Painting with a body (impasto)
`POST /api/paint` with `"media": "oil"|"acrylic"|"water"` and `"load": 0..1.5`
builds real paint: height accumulates per layer, gravity moves fresh excess
downhill carrying pigment (oil holds glossy ridges, water runs), and the
composite is relief-lit non-destructively. Erase carves the body. Layer flags
via `/api/layer {"action":"edit", ...}`: `alpha_lock` (recolor existing
pixels only; frozen transparency, recorded per stroke so replays match) and
`clip` (show only where the layer below has pixels; consecutive clipped
layers share one base). The `Paint relief 3D` node renders a layer's paint
body as a lit mesh -- tilt it to judge build-up.

## Painting with stuff (PBR materials)
`"material": "gold"` (catalog in `/api/state` -> `materials`; or a dict
`{"preset"?, "rough" 0..1, "metal" 0..1, "grain", "hold", "flow", "iters"}`)
lays a per-pixel SURFACE with the stroke: roughness and metalness, lit by the
composite. The brush colour is the albedo -- gold gleams in YOUR gold; metals
tint their highlight with it and give up diffuse, matte materials stay dead
flat, and grainy ones (brushed_steel, chalk) deposit a position-stable
micro-tooth that replays identically. Materials build a paint body like media
do (one body per stroke: a material wins over a media if both are sent).
Erase and the depth eraser take the stuff with what they remove; undo,
stroke edits, transforms and .lews save/load all carry the material map.

## Editing strokes after the fact
Every brush stroke is a live record. `POST /api/paint` returns `{sid}`.
Then: `/api/strokes/select`, `/move` (`falloff` px of arc-length softness +
`strength`), `/pull` (grab one joint, the stroke follows like a thread --
rigs and pins honoured), `/transform`, `/duplicate`, `/smooth`, `/tolayer`,
`/delete`, `/clipboard`. Layers refuse stroke edits when hand-painted pixels
would be lost (a clear error names the layer).

## Paint Effects (the FX-brush recipe, agent style)
1. Paint an INKLESS recorded stroke: `opacity: 0, record: true` -> `{sid}`.
2. Build the chain: a `Stroke FX` node (`spline: sid` -- comma-join several,
   `mode: "tubes"` for the 2.5-D mesh growth with a `depth` output socket)
   feeding a `Layer out` on a fresh layer.
3. Retune the node any time; strokes stay procedural. `Fluid` (dye + real
   solver, scrub `steps` to animate) and `Light direction` are close friends.


## /api/mind — leCore discovery door (agents)
POST /api/mind {"name": faculty, "args": {...}} for a small allowlist of
read-only leCore faculties: find_capability / suggest / describe_skill /
complete_method (discovery + docs) and analysis helpers (compare_images,
seam_continuity, est_dx, vanishing_point, image_colours, image_signature).
A rejected name returns the full allowlist WITH python signatures, so one
failed call teaches the correct usage. Mutating faculties are not exposed.

## Paint LEVELS: the weave is filled, not printed on top
Paint is a fluid. It fills the cavities of the canvas and levels off, so a
thick passage has a smooth globby top surface with NO trace of the substrate
in it; the weave only reads where the film is thin -- paint sitting in the
valleys, scraped bare off the risen threads, which is what dry-brush IS.
`_shaded_pixels`/`_shade_patch` therefore attenuate the canvas relief by
paint thickness (`exp(-h / _PAINT_LEVEL)`). Adding the tooth to every pixel's
height regardless of thickness printed canvas texture onto the top of
impasto, which no real painting does -- it was the single biggest reason
thick strokes read as fabric rather than paint. Confirmed against reference:
thin paint adheres selectively to the raised weave and leaves valleys bare,
while freer-flowing paint fills the surface texture instead of catching only
the peaks.

Bristle definition levels off the SAME way. A fluid film flows back into the
brush's own furrows, so a heavy passage is smoother and globbier than a thin
one -- how much survives depends on how clay-like the medium is (`hold`): a
stiff paint keeps its ridges, which is impasto, and a watery one closes them
almost entirely. PRESSURE feeds the same term: it already sets the stroke's
diameter, and it also sets how far the hairs are driven into the film, so a
light touch leaves a smooth mark and leaning on the brush leaves a raked one.
Both live in `depth` in `_deposit`, which drives the furrows AND the film's
breakup, so the two cannot disagree.

BANDING WATCH: the load wobble is a per-arc value applied across the full
width, i.e. literally a vertical band, and `levelling` amplifies any
variation in `dep`. Its bucket count must follow the stroke's PHYSICAL length
(~55px per bucket), never a fixed K -- at a fixed count the wavelength scales
with the stroke and short marks band. This has now regressed twice.

Note the two halves must stay consistent: `grip` in `_deposit` decides WHERE
thin paint lands (on the peaks), and `_PAINT_LEVEL` decides when the weave
stops showing in the SHADING. Change one and check the other.

## Bristle-track mark generator
`_bristle_tracks()` replaced the disc-stamp-then-modulate mark generator. A
hair covers a pixel when |u - offset_b(s)| < its half-width, evaluated
analytically from the stroke frame as a 2-D LUT read with one gather, so cost
is flat in bristle count (stamping ~30 hairs x 200 path points would have been
thousands of small array ops). Coverage IS the union of tracks, so the
silhouette is genuinely ragged. ~240ms for a 1080p 200-point stroke.

Four artifacts found and fixed, all worth knowing:
- STEPPING down a stroke whose width changed: `_stroke_frame` normalised `u`
  by one width per ~radius-long stride, so u's scale was piecewise-constant
  and the stride boundaries showed as hard seams. Width is now interpolated
  PER PIXEL along the segment.
- BANDING along the arc: the hair LUT was read nearest-bucket, so the pattern
  jumped in S discrete steps. Both the hair table and the load wobble are now
  linearly interpolated in s.
- HAIRS AS RIBBONS: a 1-d**3 profile is a plateau, so neighbouring hairs
  merged into a few thick bands. A Gaussian thread profile at ~1.6px pitch
  (n = radius*1.25) reads as many fine hairs.
- CAPSULE ENDS: `_stroke_frame` now also returns `ov`, the axial overshoot
  past the path ends, so each hair is cut at its own reach and the mark ends
  chisel-shaped and frayed. The solid-core floor must be gated by `ov` too or
  it fills the round cap back in behind the hairs.

COUPLING TO WATCH: several models read ABSOLUTE height off the canvas (the
pile a dry brush reloads from, wet-pickup wetness, the tooth's thin-paint
reference). The tracks lay less paint than the disc did, which silently
weakened all of them until a near-solid core floor restored density. Scaling
the deposit constant to compensate was tried and REJECTED -- it inflated the
piles faster than it restored pickup and made things worse. If you change the
mark generator again, make those thresholds relative to the deposit scale.

Also fixed here: `_canvas_tooth`'s cache did clear()-then-insert on a shared
dict, which raced composites served concurrently (500s under load). It now
rebinds to a fresh dict instead of mutating in place.

STILL OPEN: the piped rim tracing the silhouette survives even the rewrite,
so the capsule was not its cause. Leading untested suspect: the berm's rim
sits at |u|~0.70, a continuous curve right around the stroke including the
caps. The earlier berm=0 A/B that seemed to clear it ran against the old
floored coverage and is invalid; repeat it against the track field.
