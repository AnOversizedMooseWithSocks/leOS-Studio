# leStudio3d 1.8.1 - the agent skill, two fixes it found

* **`lestudio3d.skill`** ships in the app folder: the agent manual (install, headers, discovery, objects/ops/
  materials, rendering, the `.lews` contract, `/engine/`, multi-agent patterns, measured sharp edges), the
  sibling of leStudio's `lestudio.skill`.
* Writing it found two defects: a described object (`/api/scene/describe`) carried the Scene document's own
  SDF classes as its `sdf_tree`, which lack `to_dsl` -- `/api/semantic`, shader export and publishing then
  500'd; such objects are now mesh-only (honest: no exact tree). And `GET /api/scene?view=summary` was
  documented but not wired; it is now (ids / names / bbox / counts / materials only).
* `tests/scene_doc_test.py` 20/20 covers both.

---

# leStudio3d 1.8.0 - no legacy

Nothing in this app exists to support an older version of itself any more. Removed, with what replaced it:

* **`holostuff/flatcompat.py` and the "bundled engine overlay"** -- the import hook that made the packaged engine
  answer to the flat `holographic_*` names the app was first written against, plus `LECORE_ENGINE=bundled|installed`
  and the "vendored copy wins" rule. Every import is now the engine's packaged path
  (`holographic.mesh_and_geometry.holographic_mesh` ...); `leos-core` from PyPI is the one engine.
* **`tests/fake_engine/`** -- a stub for a build machine that could not install the engine. That machine is gone
  and the engine is a hard dependency; `render_api_test` and `route_sweep` now run on the real engine.
  The sweep used to classify 30 of its 500s as "stub gaps"; on the real engine it is 114 calls, **0** failures.
* **`/api/engine_preflight` and `_ENGINE_REQUIRES`** -- a hand-kept list of what an older `leos-core` might lack.
  Help > Engine status is the engine's `engine_status()` plus the `features()` gate from the mount.
* **The bundled `capabilities.json` and the token-scoring fallback in `/api/docs`** -- the catalog is the engine's
  (`find_capability` / `browse_capabilities`), in-process.
* **`_transfer_uv_compat`** (a 0.2.8 return-shape shim), the **"older engines" decimate fallback**, the
  **`holographic_materials` import that never existed**, **`_trim_undo()`** (a no-op kept as a call target),
  **`buildMatTabs()`** (same, client side), the `#legacy` element id.
* **Save files key parents and render assets by object id**, not by save-order index (that scheme existed only
  because ids used to be reassigned on load).

`tests/run_all.sh`: all green on the real engine -- render api 44/44, route sweep 114/114 clean, lews bridge 33/33,
scene document 19/19.

---

# leStudio3d 1.7.0 - the name, the whole engine, the whole .lews contract

## leStudio3d, fully

The app is **leStudio3d** everywhere: window title and menubar, welcome guide, download filenames,
`app="lestudio3d"` on the workspace and the agent surface, the private kind `lestudio3d.scene` (registered with
a schema), section ids `lestudio3d:mesh:<id>` / `:sdf:` / `:journal:` / `:tex:` / `:mat:` / `:camera` / `:env`,
the preset target `lestudio3d.render`, the memory partition, `LESTUDIO3D_WORKSPACE`, every doc, test and the
zip. No compatibility shims for the old name exist (nothing ever used it). The only remaining spelling of the
old name in the tree is the filename of the engine's own audit document (`docs/POLYSTUDIO_AUDIT.md` in leCore),
cited as a source; that file lives in the other repo.

## The whole engine, beside the app (`/engine/`)

APP_FOUNDATION §3 says to mount the engine's own service beside the app rather than proxy it. Done:
`holographic_service.Service` on THIS process's mind, every route it has under `/engine/` -- `GET /engine/tools`
(2,408 faculties), `POST /engine/invoke` (JSON args, object handles `ref:Type:N` minted on the way out and
resolved on the way in, `budget=` to bound big results), jobs, documents, skills, capability search, bus, sql.
`GET /api/scene/doc` now returns a `ref` for this scene's Scene document, so any faculty can be run on it.
`Help > Engine console` searches the catalog, invokes, and shows the result.

## The whole .lews contract

* Published on save: `lecore.mesh` per object **and `lecore.sdf` for every analytic object** (exact geometry
  beside the mesh), `lecore.material`, `lecore.image` textures, one `lecore.scene`, **`lecore.camera`** (the
  last render camera), **`lecore.journal`** per edited object (the op history, replayable), the environment as a
  content-addressed **`lecore.asset`** referenced from a private section so GC keeps it.
* Read on import: `lestudio.document` / `lecore.image` (textures), `lecore.mesh`, **`lecore.sdf`** (meshed,
  tree kept), `lecore.scene` bindings (names, materials via `overrides.matlib`), **`lecore.camera`**.
* Files: export is `Workspace.export_bytes()` of the live directory; import of a file goes through
  `Workspace.from_file` (journalled as "opened a file"). Bare `save_container` only when no workspace is open.
* Session: `/api/workspace` reports presence, kinds, assets; `/api/workspace/note` (`lews_note`),
  `/api/workspace/wait` (`lews_wait` long-poll), `/api/workspace/gc` (`gc_assets`), `/api/invite` and
  `/api/join` (`create_invite_link` / `join_from_link`); `Help > Workspace` shows all of it.

## Verified

`tests/run_all.sh` green: render api 44/44, route sweep 0 failed, `lews_bridge_test` **33/33**, `scene_doc_test`
19/19. Engine gaps hit: LC-8 (`scene_section` drops `sdf` and `name` bindings) added to LECORE_CORE_BACKLOG.md.

---

# leStudio3d 1.6.0 - leCore 0.2.22 and leStudio R72

Engine floor 0.2.11 -> **0.2.22** (480 new catalog cards since our snapshot, 0 removed; snapshot refreshed).
Every new door is gated on `m.features()` at mount; `Help > Engine status` names what an older build lacks.

## The foundation, fixed

* **One mind.** 1.5.0 built a `UnifiedMind` in `app.py` and a second one lazily in the backend
  (APP_FOUNDATION rule 1). `backend.mount_foundation(app)` now does the whole mount -- mind, features,
  live `.lews`, agent surface -- for the launcher AND the harnesses. `render_api_test` had been failing on
  `/api/agent/tools` (404) because the tests booted the blueprint without the surface; it is green again.
* **Ids from the engine.** `Workspace.mint("O")` when a workspace is open; ids are saved in the scene file
  and kept on load (an agent no longer re-reads the scene after every open).
* **Canonical kinds in the live workspace.** Every save also puts `lecore.mesh` per object, `lecore.image`
  per texture, `lecore.material` per used material and one `lecore.scene` of bindings by section id,
  unchanged sections skipped by `section_hash`, deleted objects' sections deleted. The private blob is now
  honestly `lestudio3d.scene` (1.5.0 filed it under the canonical kind).
* **Agent manifest summaries** go through `agent_surface(hints=)`: our first-SENTENCE rule, not the engine's
  first line (44/44 in render_api_test, 0 "stops mid-sentence").

## leStudio R72 compatibility (see LESTUDIO_COMPAT_REPORT.md)

* leStudio's R67 journal-first documents -- arrays hoisted into `lecore.asset` (`array_refs`), layers saved
  with NO pixels -- composited to zero layers and were silently skipped. Fixed: assets resolved, sections
  upgraded, replay base used for journal-only layers and reported as approximate.
* `File > Pull from live workspace` (or `POST /api/workspace/import {"from":"workspace"}`): painted
  documents and any app's `lecore.mesh` sections straight from the directory, no file round-trip.

## The Scene document (agentic parity)

The engine's Scene document is where every parity faculty landed; our objects now live there too
(`scene_add` takes anything with `.eval` -- our exact trees and baked grids qualify):

* `Create from > Describe a scene` / `POST /api/scene/describe {text}` -- `describe_to_scene`: words ->
  objects, exact SDF trees kept (still export as shaders), colour words -> the same materials
  `/api/semantic` picks (red -> ruby), the engine's `unknown` / `suggestions` passed through.
* `GET /api/scene/doc` -- `scene_info`: what is in my scene, as the engine reads it (handle -> object id).
* Render view **Quick look** / `GET /api/scene_preview` -- `render_preview`, ~0.2 s at 120x90.
* Render view **Frame all** / `GET /api/camera/fit` -- `fit_camera`, exact and aspect-aware.
* `POST /api/scene/animate {keys}` -- `render_animation` -> GIF, keyframes by object id.
* `refine_scene` could not be wired: it takes `build_scene`'s SemanticScene, not the document (LC-3).

## New from the engine

* Mesh > Remesh: **Retopologise** (`surface_retopo`, orient + manifold cleanup applied first, as the
  catalog's precondition says), **CVT remesh**, **Manifold cleanup**, **Orient faces** -- each reports faces
  before/after and the engine's own report.
* Generate: **Studio lighting rig** (`studio_sky`), **Parametric sky** (`sky_model`), plus `{"exr": path}`
  on `/api/env` (`load_exr`) -- rasterised into the same dome the renderer already samples.
* `/api/presets` -- `lecore.preset` sections (render / material recipes) shared with any app; `Help > Presets`.
* `/api/memory` -- per-user memory through `app_substrate` (remember / recall / observe / suggest / habits).

## Verified

`tests/run_all.sh`: syntax, id/scope audits, backend stubs, render api 44/44, route sweep 0 failed inside
the app, **new** `lews_bridge_test` 15/15 and `scene_doc_test` 19/19 (both need the real engine; skip
without it). `tools/app_lint.py` 8/16 (was 9: the one new hit is the `hints=` table built from `url_map` -- LC-7 -- not a hand-rolled manifest; the manifest is the engine's). `quality_gate.py` now RUNS from the standalone (it only ran under the gallery mount before): terracing and fringe pass, **`edge_tones` fails at 0.5985 (limit 0.90) -- identically on untouched 1.5.0**, so it predates this release and is now measurable; filed as PS-1 in LESTUDIO3D_BACKLOG.md.


---

# leStudio3d 1.5.0 - full adoption

Citizenship by the engine's lint: 4/16 (1.2.0) -> 8 -> 9 -> **this**. Everything that had an
engine twin is now a shim over the engine door, and two engine capabilities we never had are
in the app.

## Adopted this release

* **Undo/redo is entirely the engine's `EditHistory`.** 1.4.0 drained an app-side list onto
  it; 1.5.0 removes the list. Every producer calls `_push_snapshot(entry)`, which does
  `history.do(None, _SnapshotCommand(entry))`. The engine truncates the redo tail on a fork;
  the app keeps no redo list at all. `_trim_undo()` survives as a no-op call target.
* **`quality_gate.py` metrics are the engine's.** `terracing`, `edge_tones`, `fringe_ratio`
  now come from `m.render_quality_gate` (the faculty the audit says this file caused). What
  stays ours is the harness: which of THIS app's routes to render and which frames to compare.
* **`_planar_uv` asks `mesh_uv_unwrap` first**; the two-widest-axes projection is only the
  fallback for a mesh the unwrapper refuses.

## New capabilities, from the engine

* **HDRI environments.** `POST /api/env {"hdr": "<path or data URL>", "exposure": 1.0}` reads
  a Radiance .hdr/.pic through `m.load_hdr` -> unbounded linear radiance -> the same dome
  light the procedural presets feed. The backlog had this filed as C1 "no HDRI import". It
  existed; nobody had asked the engine.
* **Live `.lews` workspace.** With `LESTUDIO3D_WORKSPACE=<dir>`: `GET /api/scene/save` is also
  a `Workspace.put` of section `lestudio3d.scene` (kind `lecore.scene`); `POST /api/scene/load
  {"from":"workspace"}` reads it back; `GET /api/workspace?since=<rev>` shows roster and the
  journal. A second leCore app on the same directory sees leStudio3d's scene and its users.
  VERIFIED: save -> rev 1, journal records the put with its sha, load restores the shared scene
  and drops the unsaved local object.

## Kept on purpose (reason is in the code at each site)

`_auto_uv` (already on the engine's LSCM), `_midpoint_refine` (pure refinement; Catmull-Clark
smooths), `fbm2`/`erode_terrain` (procedural_noise returns a field encoding, not the (H,W)
array the terrain composes), `/api/photo` streaming (LC-2: fold-error gate first), `_tonemap`
(the lint's nearest card was load_exr, which is not a tone mapper).

## Verified

route sweep: 0 failed inside the app. undo_redo_test: green on the engine contract.


---

