# Poly Studio 1.5.0 - full adoption

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
* **Live `.lews` workspace.** With `POLYSTUDIO_WORKSPACE=<dir>`: `GET /api/scene/save` is also
  a `Workspace.put` of section `polystudio.scene` (kind `lecore.scene`); `POST /api/scene/load
  {"from":"workspace"}` reads it back; `GET /api/workspace?since=<rev>` shows roster and the
  journal. A second leCore app on the same directory sees Poly Studio's scene and its users.
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

