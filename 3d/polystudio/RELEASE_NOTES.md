# Poly Studio 1.4.0 - full adoption of the leCore foundation

Every hand-rolled duplicate of an engine faculty is now a thin shim over the engine door, so
this app stays on core development and can talk to other leCore apps. Citizenship measured
by the engine's own lint: 1.2.0 **4/16** -> 1.3.0 **8/16** -> 1.4.0 **9/16**.

## What moved onto the engine

| was ours | now | how |
|---|---|---|
| `ccrun.py` (200 lines: C emission, compile cache, DSL glue) | `holographic_ccrun` + `sdf_dialect(c_f64)` | 40-line shim keeping only the app's size-threshold POLICY |
| `_banded_grid_chunked` (chunked banded SDF bake) | `m.mesh_to_sdf_grid` (chunked since sweep 148) | shim; same `(grid, (xs,ys,zs))` contract; triangulates quads first |
| `_gauss_blur_np` | `m.blur_image` | shim |
| `_S["undo"]` / `_S["redo"]` snapshot lists | `m.edit_history()` | see below |
| `/api/agent/*` manifest + invoke | `m.agent_surface()` (1.3.0) | deleted |
| `/api/docs` token scorer | `find_capability` (1.3.0) | engine first, tokens as fallback |
| `quality_gate.py` thresholds | `m.render_quality_gate` (1.3.0) | delegates; marked for UPSTREAM by the audit |

One `UnifiedMind` per process (`_mind()`), built lazily - APP_FOUNDATION rule 1.

## Undo/redo on the engine's EditHistory, behaviour byte-identical

The app's undo is snapshot-based; EditHistory is command-based (`do / undo / redo` with
`invert`). Rather than rewrite 18 producers, a `_SnapshotCommand` whose `apply()` is a no-op
on first application and a restore on redo, and whose `invert()` restores the captured
snapshot, lets the existing snapshots ride the engine's stack. `_trim_undo()` - the one choke
point every producer already calls - drains the append target onto the history. Fourteen
direct `_S["undo"].pop()` discards became `_discard_snapshot()`, which knows the entry may
already have moved.

VERIFIED against the live app: add, add, undo, undo, redo, then a new edit truncates the
redo tail. `tests/undo_redo_test.py` updated to the engine contract and green; route sweep
0 failed inside the app.

## What I deliberately kept, with the reason in the code

  * `_auto_uv`: already on the engine's `lscm`; the lint matched a name.
  * `_midpoint_refine`: PURE refinement (bit-identical surface). `mesh_catmull_clark` smooths.
  * `fbm2`, `erode_terrain`: the engine's `procedural_noise` returns a field encoding, not the
    `(H, W)` array the terrain generator composes; a swap changes the output. Needs a card.
  * `/api/photo` streaming on `m.job_submit`: LC-2 in the audit is an open question - fold
    error measurement is the gate. Not taken blind.
  * `_tonemap`: the lint's nearest card was `load_exr`, which is not a tone mapper. Kept.

## Remaining lint hits

`own_undo_stack` still shows 18 because the producers still say `_S["undo"].append` - that
is the shim's append target, not a second stack. `pil_in_core_paths` (9) and `wall_clock`
(9) are measurement and file I/O, not core paths.


---

