# Poly Studio test harnesses

Added with the 1.1.0 UX pass, because most of that work was client-side and "it parses" is not
verification. None of these need a running server; the browser tests mock `/api/*` and stub three.js.

| file | needs | what it proves |
|---|---|---|
| `id_audit.py` | python3 | every `$('id')` in app.js resolves in index.html, or is guarded |
| `scope_audit.py` | python3 | no route calls a helper that only exists inside another function (the `_tonemap` bug class: parses clean, 500s at runtime) |
| `undo_redo_test.py` | python3 + numpy | undo/redo symmetry, HDR cache cap, cancel unwinding, size clamps |
| `render_api_test.py` | python3 + numpy + flask | **boots the real backend against `fake_engine/` and calls the render routes over HTTP**: photo streaming and NDJSON framing, w/h/spp honoured and clamped, `/api/photo_post` answers with a PNG that exposure actually changes, client-disconnect and explicit cancel both stop the trace, AOVs, undo/redo, `/api/render`, `/api/render_progressive`, `/api/render_engine` and their headers, `/api/upscale` (data URLs, no shrink-before-enlarge, blob URLs refused by name), and the agent manifest (every tool has a usable summary, new routes are discoverable, `/invoke` works) |
| `route_sweep.py` | python3 + numpy + flask | calls **every** route and asserts none fails inside app code — a route may 200 or refuse with 4xx, but never 500 in its own logic |
| `ui_smoke.js` | node + puppeteer + chrome | image ownership, preview gate, typing guards, palette, a11y roles, touch-action, job chips, modals, blob lifetimes |
| `keymap_test.js` | node + puppeteer + chrome | dispatches real key events: bindings fire once, Ctrl combos stay with the browser, the shortcuts dialog matches the table |
| `touch_test.js` | node + puppeteer + chrome | one finger orbits, pinch zooms, tap selects once; plus a source check that the mouse path bails on touch |
| `dialog_test.js` | node + puppeteer + chrome | dialogs cascade and stay on screen, centred ones stay centred, `aria-modal` actually contains Tab |
| `zip_roundtrip.js` | node + puppeteer + chrome | dumps a browser-written .zip for `zipfile` to verify |

```sh
python3 tests/id_audit.py
python3 tests/scope_audit.py
python3 tests/undo_redo_test.py
python3 tests/render_api_test.py
python3 tests/route_sweep.py
CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/ui_smoke.js
CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/keymap_test.js
CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/touch_test.js
CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/dialog_test.js
CHROME=... PUPPETEER_PATH=... node tests/zip_roundtrip.js && \
  python3 -c "import zipfile; z=zipfile.ZipFile('/tmp/ps_turntable_test.zip'); print(z.namelist(), z.testzip())"
```

## The fake engine (`tests/fake_engine/`)

`leos-core` cannot be installed on every build machine (this one has no network at all), and without an
engine the render routes cannot be *called* — which is exactly how `/api/photo_post` shipped returning
500 on every request. `fake_engine/` is the smallest set of flat `holographic_*` modules those routes
touch, with the real call signatures: `Mesh`, an SDF tree with `union`/`eval`, `Camera.ray_dirs`,
`sphere_trace`, a `path_trace` that honours `on_progress`, a small material library, and the post-effect
helpers.

**It is not a renderer.** `path_trace` returns a deterministic gradient. `render_api_test.py` proves
plumbing — parameters, framing, cancellation, cache behaviour, route wiring, agent-manifest quality — and says nothing about
whether a render looks right.

## What these harnesses are worth

They were mutation-tested: fixed bugs were deliberately reintroduced to confirm the tests actually fail.
`keymap_test.js` caught both key-dispatch mutants by name. Reintroducing the exact `/api/photo_post`
bug that shipped — `py_compile` passes, which is why it got out — is caught by name by `scope_audit.py`
(*"photo_post() references '_tonemap' -- only defined inside photo()"*) and as a real HTTP 500 by
`render_api_test.py`. The touch double-fire mutant was **not**
caught at runtime — Chromium runs at-target capture listeners before bubble listeners, so the touch
module already wins there and the guard is defence for browsers that order by registration instead.
That case is covered by a source assertion instead, and the honest limitation is written into the test.

The lesson generalises: a green harness proves the mutants it was tested against, not correctness.

**These do not replace `quality_gate.py`.** That one measures the actual rendered image against absolute
thresholds and needs the real engine installed. Run it before shipping anything that touches a render
path — none of the harnesses here can see a render that is merely ugly.
