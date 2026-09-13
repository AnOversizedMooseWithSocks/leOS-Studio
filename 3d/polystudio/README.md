# Poly Studio (standalone)

A C4D/Blender-style polygon **and** field modeller on the leCore engine — poly editing, sculpting,
material painting, CAD booleans, a procedural node graph, and path-traced photos, in one app with no gallery.

## Run it
**Windows:** double-click `run.bat`.
**macOS / Linux:** `./run.sh` (creates a venv and installs everything on first run), or by hand:
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs the leCore engine (leos-core) + Flask/Pillow
python app.py
```

### The viewport needs three.js
`python app.py` caches three.js into `vendor/` on first run, so the app works offline afterwards. If this
machine cannot reach the internet at all, drop `three.min.js` (r128) into `vendor/` yourself — the page
loads that copy first and only then tries CDNs. If neither is available the page now says so clearly
instead of showing a dead viewport.

**Offline / blocked CDN**: see above — this used to silently break the whole UI.

### Which engine am I running?
This app builds on **leCore**, installed from pypi as `leos-core`. Two download flavours:

| download | engine |
|---|---|
| `polystudio_standalone.zip` (~0.4 MB) — **the release** | none bundled; `pip install -r requirements.txt` fetches `leos-core` |
| `polystudio_standalone_bundled.zip` (~5.8 MB) | engine vendored, for offline / air-gapped machines only |

If both an installed engine and a vendored copy are present, the **vendored copy wins** — it is the
development overlay for building ahead of the next release. Force either with
`LECORE_ENGINE=installed` or `LECORE_ENGINE=bundled`. **Help ▸ Engine status** always shows which one is
live, its import name, and which optional extras (`jit`, `zig`, `gpu`) are present.
Opens at http://127.0.0.1:5000/.

## Layout (professional-app style)
- **Menu bar** (top): File / Create from / Add / Generate / Select / Mesh / Modifiers / Window / Help.
- **Icon toolbar** (left): modes (Object/Point/Poly/Sculpt/Paint), transform tools, snap; click **☰** to expand it to labelled buttons.
- **Floating dialogs**, draggable, opened from the menus or by hotkey:
  - **N** — Objects & attributes (scene list, per-object info, apply material).
  - **M** — Material editor (141 presets + author your own physical PBR material).
  - **P** — Render view: the framebuffer. Toolbar with renderer picker, Render/Cancel, zoom
    (fit · 1:1 · wheel · drag · double-click), render history with A/B wipe compare, turntable,
    Save, 2x upscale. It owns its image: a finished render is never overwritten by the live preview,
    and no preview traffic runs while it is closed.
  - **Shift+P** — Render settings: output size and aspect, samples per pixel, field detail, backdrop,
    exposure/sharpen (re-tonemapped from the cached result, no re-trace), diagnostic AOVs, live-preview
    controls.
  - **Ctrl+K** — Command palette: fuzzy-finds every menu action, and queries the leCore capability
    index for anything it doesn't recognise.
  - plus Shader export, Boolean, Lathe, Shader-FX, Milkdrop, UV, Reduce — each its own dialog.
- **Node editor** (bottom): procedural graph — see below.

## Input
Mouse: Alt+drag or right-drag orbits, MMB pans, Alt+right-drag dollies, wheel zooms.
Touch: one finger orbits, two fingers pinch-zoom and pan, tap selects, long press opens the context
menu. In Sculpt and Paint one finger paints and two fingers navigate.
Keyboard: Ctrl+Z / Ctrl+Shift+Z undo and redo, F2 renames the active object, arrow keys nudge the
selection (so transforms are not drag-only), and `?` lists everything else.

## Working with other apps

- **Shared workspace (`.lews`).** *File ▸ Open shared workspace .lews (leStudio)…* opens a workspace written
  by **leStudio**, the 2-D image editor built on the same engine. Poly Studio reads the painter's documents,
  composites their layers with the engine's own compositor (so the texture matches what the painter sees),
  and offers each one as a texture — optionally applying it to the selected object on import. The dialog
  lists everything in the file first, including sections this app can't open.
  *File ▸ Export shared workspace .lews* writes your objects back into the same file as `polystudio.object`
  sections plus a canonical `lecore.image` per texture, and carries the other app's sections through
  untouched, so the painter gets their work back exactly as they left it.

- **Agent access.** `GET /api/agent/tools` returns a machine-readable manifest of every endpoint (generated
  from the live routing table, so it can't drift), and `POST /api/agent/invoke` calls one by name. Renders
  and other binary endpoints are called directly; the manifest says which.

## Import options

Choosing a `.glb` opens an options dialog rather than guessing:

| mode | what it does |
|---|---|
| **Auto** | measurement-driven; decimates big scans with a silhouette guarantee and reprojects the original texture |
| **As-is** | no processing, full resolution |
| **Decimate** | to a face budget, silhouette-checked |
| **Retopology** | clean quad-dominant cage |
| **Voxelize** | watertight uniform remesh |
| **Rebake atlas** | decimate and bake a *fresh* texture atlas (right for fragmented photogrammetry atlases) |

## Modifiers worth knowing

- **Scatter instances on a surface** — copies one object across another's surface (count, scale, jitter,
  follow-surface, even spacing) and bakes the result to real geometry.

## Checking a build

`python3 quality_gate.py --write` renders the reference scenes and measures them against absolute
thresholds (terracing, edge anti-aliasing, colour fringing, per-object materials, and that the exact render
path is actually being taken). It fails loudly and names the defect. Run it after any change that touches a
render path — and still look at the frames it writes.

## What it can do
- **Poly modeling:** extrude, inset, bevel (multi-segment), loop cut, dissolve, poke, subdivide, smooth, solidify, fill holes, bridge, triangulate.
- **Sculpting:** field brushes (inflate/carve/smooth/flatten/grab), DynaMesh-style clean re-mesh; entering sculpt to look around and leaving without a stroke restores your mesh exactly.
- **Materials:** paint directly on the model, assign by selection, or drive by formula; author custom physical materials (metallic/roughness/transmission/IOR/emission).
- **CAD:** booleans with exact constant-radius fillets, lathe (profile → solid of revolution), curvature inspection, STL export.
- **Import/Export:** `.glb/.gltf`, `.obj`, `.stl` — imports land centered at the origin so they can't get lost.
- **Shadertoy:** export any analytic object as an exact GLSL/WGSL SDF program; apply iq's operators (twist/bend/onion/…); safe ns-eel2 formulas as geometry/material inputs.
- **Node graph:** typed sockets, connect-time type checking, cycles refused. Sources include primitives, **3-D fractals (Mandelbulb, Menger — exact GLSL)**, **fields (ns-eel2 formulas and 3-D curl noise)**; fields **drive** geometry (displace an SDF or a mesh) and scalars drive field constants; mesh post nodes do **retopo** (voxel remesh) and **denoise** (Taubin). Build any node to a scene object.

The leCore engine is bundled under `holostuff/`; this folder is self-contained.
