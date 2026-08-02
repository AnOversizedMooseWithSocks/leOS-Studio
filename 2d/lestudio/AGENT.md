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
