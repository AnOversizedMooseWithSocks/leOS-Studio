#!/usr/bin/env python3
"""tools/lecore_plan.py -- the R74 UX task, held in leCore's external memory.

    python tools/lecore_plan.py teach     # (re)fill lecore_memory/ from the backlog
    python tools/lecore_plan.py           # boot the partition and print the plan back
    python tools/lecore_plan.py ask "where does the lasso tool get its gate?"

The partition is `2d/lestudio/lecore_memory/` -- the conventional per-repo
path lecore.autoboot() finds on its own, and the one the leStudio server
mounts at boot. Session `lestudio-r74-plan` holds everything about this task:
the audit of what exists, what is missing and why it matters, the acceptance
criteria, the engineering rules that bind any change, and the ordered plan
(`learn_plan("r74-ux", ...)`) so an agent can ask "what comes before the
gradient tool?" and get an exact answer.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PART = os.path.join(ROOT, "lecore_memory")
SESSION = "lestudio-r74-plan"
BACKLOG = os.path.join(ROOT, "docs", "BACKLOG-r74-ux.md")

PLAN = [
    "01 lasso and polygon lasso on _poly_gate; L is lasso, creature moves to Shift+L",
    "02 gradient tool as a journaled {op:gradient} record; linear radial angle",
    "03 stroke selection and fill selection; Shift-drag straight line",
    "04 menu bar Edit Image Layer Select Filter View mirroring the side tabs with shortcuts",
    "05 selection modifier keys Shift add Alt subtract Shift+Alt intersect; Ctrl+A Ctrl+D",
    "06 foreground/background swatches, X swap, D defaults, HSV wheel, Alt eyedropper",
    "07 canvas rotation R-drag and flip view",
    "08 blend modes hard light color dodge color burn hue saturation color luminosity",
    "09 dodge burn sponge brush journaled like blend_stroke",
    "10 crop tool with handles calling crop to selection",
    "11 brush tips and pressure to size and opacity toggles",
    "12 history panel over the undo stack",
    "13 fold layer and brush physics behind Advanced disclosures",
    "14 split the rail into Standard and Studio groups; first-use hints; plain names",
    "15 node editor Tab quick-add at cursor",
    "16 node editor drop node on wire and drag wire to add",
    "17 node editor mute bypass frame all minimap",
    "18 node editor frames favourites recent",
    "19 open layer in graph button",
]

# what an agent picking this up needs to know, as question -> answer
FACTS = [
    ("what is the R74 task for leStudio",
     "Close the gap between leStudio and traditional editors (Photoshop, GIMP, Krita, Procreate): add the standard tools that are missing and align the UI with conventions, without changing any existing stroke's pixels. Backlog: docs/BACKLOG-r74-ux.md; plan: learn_plan r74-ux."),
    ("where is the leStudio source",
     "GitHub AnOversizedMooseWithSocks/leOS-Studio, subtree 2d/lestudio on main. Backend src/lestudio/__init__.py (Document, ~20k lines) and server.py (Flask routes); single-file frontend src/lestudio/static/index.html; tests/run.py runs tests/test_*.py; JS gates node tests/test_toolbar.js test_popups.js test_creature_ui.js."),
    ("how are changes delivered to Devin",
     "As a zip of only the changed files with repo-root-relative paths (2d/lestudio/...), so he extracts in the repo root, reviews with git diff, then commits. Never a full-tree zip."),
    ("which tools exist in leStudio today",
     "26: transform, fill, text, brush, erase, heal, stamp, smudge, blend, knife, clone, spline, pick, nudge, strokesel, nodepaint, fx, rect, ellipse, wand, lum, obj, scribble, hatch, textile, creature."),
    ("which tool hotkeys exist in leStudio",
     "V transform, G fill, T text, B brush, E erase, H heal, U stamp, S smudge, Y blend, Q knife, C clone, P spline, I pick, N nudge, K strokesel, J nodepaint, F fx, M rect, W wand, L creature. Ctrl+Z/Y undo redo, Ctrl+C/V, Ctrl+D, Ctrl+G, Ctrl+J, Ctrl+Shift+I invert, + - 0 zoom, [ ] brush size, Space pan. Defined in TOOLKEY in index.html."),
    ("which menus exist in leStudio",
     "Only File, Share, Lights, Persp in the bar. Layers, Select, Masks, Splines, Brush, Tool, Colour, Image, Nodes are side tabs. Traditional apps expect Edit Image Layer Select Filter View in the bar -- backlog item 4."),
    ("which selection tools exist in leStudio",
     "rect (M), ellipse, wand (W), brightness (lum), object (leCore segmentation). No lasso, although the brush panel tooltip promises one. Modes add subtract intersect are a dropdown selMode, not modifier keys. Expand contract feather invert to-mask keep crop-to-selection exist."),
    ("which blend modes exist in leStudio",
     "BLEND_MODES in __init__.py: normal, multiply, screen, overlay, add, subtract, difference, darken, lighten, softlight. Missing: hard light, color dodge, color burn, hue, saturation, color, luminosity -- backlog item 8."),
    ("which standard tools is leStudio missing",
     "Lasso/polygon lasso, gradient tool, stroke/fill selection (shapes), crop tool, dodge burn sponge, foreground/background swatches with X and D, HSV wheel, Alt eyedropper, canvas rotation and flip view, brush tip shapes, pressure to size/opacity toggles, history panel, extra blend modes."),
    ("how does a region generator get its region in leStudio",
     "Document._poly_gate(poly, feather) rasterises an inline polygon to a float gate; _path_gate(path, radius, feather) sweeps a dragged path; _resolve_gate(selection, sel_invert, feather, sel_mask) resolves a stored selection. scribble hatch textile scatter creature all take poly/path/selection. A lasso is a freehand drag turned into _poly_gate."),
    ("how should the lasso tool be built",
     "Add tool poly to Document._tool_field (select with prm.points -> _poly_gate), a tLasso button, key L, freehand drag collects points, polygon lasso clicks vertices and Enter/double-click closes. Move the creature brush to Shift+L. Pin with a test that a freehand ring selects its interior."),
    ("how should the gradient tool be built",
     "A journaled {op:gradient} record with two points, kind linear/radial/angle, and stops, applied by one shared function for the tool and for replay (the pwarp model). Respects the active selection gate. Uses the Gradient and Radial gradient node maths. Done when a replayed document reproduces it bit-exact."),
    ("how should dodge and burn be built",
     "As a recorded replayable non-pigment stroke, modelled on blend_stroke (POST /api/paint mode blend): a luminance shift along the path with mode dodge/burn/sponge and range shadows/mids/highlights. Never as an unrecorded pixel mutation like smudge, which demotes the layer's replay."),
    ("what is the never-flip rule",
     "New behaviour is opt-in and off by default; an existing stroke must stay byte-identical after any change. Pressure-to-opacity, brush tips, new blend modes and folds must not change today's output when unused."),
    ("what engineering rules bind leStudio changes",
     "Use the return value of add_layer never layers[-1]; _MUT_REV[0] += 1 before composite; composite_display uses canvas_layers(); validate NaN at endpoints; strata must not break replay determinism; tests pin intent not strings; measure before optimising; read back every file write; node --check passes on phantom JS functions so add a DOM-shim test (tests/domshim.js) for new client code."),
    ("how is leStudio tested",
     "PYTHONPATH=src /tmp/v5/bin/python tests/run.py, chunks --chunk N/4 (~330 s each), -k substring. Round files r58 r59 r60 r63 r64 are absent from the public checkout and are skipped. JS: node tests/test_toolbar.js, test_popups.js, test_creature_ui.js. Playwright tests need a browser and cannot run in the sandbox (cdn.playwright.dev blocked). 37 failures in the sandbox are environmental (browser, scipy, missing backlog docs, timing budgets, run.sh exec bit, leCore 0.2.22 drift)."),
    ("what is the creature brush",
     "R73: tool 🐞 key L, POST /api/creature. Up to ten CreatureMind agents (holographic_creature_mind, the UnifiedMind pattern) sense the layer egocentrically over five turn candidates and paint their walks as ordinary journaled strokes. Rules lines self others light color field wander solid; presets explorer knotter moth mole flock spinner mazer surprise; poly/selection makes it a fill; media/material/load carried; seed reproduces (60 steps per second, not wall clock)."),
    ("what did the creature brush teach about agent brushes",
     "The brain's learned values drown a single goal token, so the rules keep a 70% vote when they have a clear preference. Senses need a two-scale blurred scent or a goal 120px away is invisible. A seeker orbits its goal unless it has appetite that drains when fed. A 180-degree bump traps; a 90-degree bump slides. A pure vortex at the release point is a zero-radius circle; add 0.35 outward. Encoding senses is the whole cost: sense once per step and only encode senses a rule can act on."),
    ("what does the node editor lack compared to Blender or Nuke",
     "Tab/Shift+A quick-add at the cursor, drop a node on a wire to insert, drag a wire end to empty space to add, mute/bypass (M), Home frame all, minimap, frame nodes with labels and colours, favourites and recent in the add menu, and an Open-in-graph button on each layer row. It has ~110 ops, a searchable add menu (buildAddMenu), Ctrl+D duplicate, G group, Delete, per-node preview."),
    ("which leStudio hotkeys collide with tradition",
     "L is creature here but lasso everywhere (fix: lasso takes L, creature Shift+L). Y is blend here, history brush in Photoshop. Q is knife here, quick mask in Photoshop. U is stickers here, shapes in Photoshop. C clone matches Photoshop stamp tool if one squints; crop should be Shift+C."),
    ("what should the layer panel look like to a Photoshop user",
     "Opacity, blend mode, lock, mask, visibility at the top; the ~50 physics controls (tilt, IOR, dome, soak, cook, gravity, emit, relief) folded behind an Advanced/Physics disclosure remembered per session. Same for the brush panel: size opacity hardness spacing smoothing on top, jitter charge strata squeeze folded."),
    ("what are the cheap conveniences on the R74 list",
     "Lock transparency button (alpha_lock exists in the API), drag-to-reorder layers, Ctrl+T alias for transform, Ctrl+E merge down, Ctrl+Shift+E merge visible, Ctrl+Shift+N new layer, hotkeys printed on every tooltip as (L)."),
    ("in what order is the R74 work planned",
     "P0 first: lasso, gradient tool, stroke/fill selection, menu bar, selection modifier keys, fg/bg colours and X/D. Then P1: canvas rotation, blend modes, dodge/burn, crop, brush tips and pressure, history panel, panel folds, tool grouping. Then P2 node editor items. The order is stored as learn_plan r74-ux; ask step_at or precedes."),
    ("what is the status of R74 step 01 the lasso",
     "SHIPPED (R74a). _tool_field gained a poly branch on _poly_gate; /api/select validates params.points (junk dropped, <3 points is a 400); the client has tLasso (L) and tPolyLasso (Shift+L), freehand drag thinned to 2px, clicked corners with handles and a faint closing edge, Enter/double-click/first-corner close, Backspace undo, Esc abandon, and selModFromEvent so Shift adds, Alt subtracts, Shift+Alt intersects on every selection drag including the marquee. SHIFTTOOLKEY cycles L (lasso -> polygon -> creature) and M (rect -> ellipse). Tests: tests/test_r74.py 7 and tests/test_lasso_ui.js 44."),
    ("what did step 01 change about hotkeys",
     "L is the lasso now; the creature brush moved to Shift+L and its tooltip and tips say so. A Shift+letter table SHIFTTOOLKEY was added to the keydown handler, and the bare-letter TOOLKEY lookup now ignores a held Shift so the two cannot both fire. The R73 client pin that asserted l:'creature' was updated rather than loosened."),
    ("what traps did step 01 surface",
     "Two: a second select() on the SAME Document reuses the selection slot, so comparing a selection against its own replacement proves nothing -- use two Documents. And a wide feather on a small shape bleeds a hair into the centre (0.9999, not 1.0), so a feather test must assert the EDGE ramp, not an exact centre value."),
    ("what is the leStudio partition",
     "2d/lestudio/lecore_memory/, session lestudio-r74-plan, written by tools/lecore_plan.py teach. learning_save writes learning/state.lecore; autoboot(partition=...) mounts it in ~0.1 s. Re-run teach after editing docs/BACKLOG-r74-ux.md."),
]


def boot():
    import lecore
    m = lecore.autoboot(partition=PART, session=SESSION, llm=None)
    # The ordered plan is REBUILT FROM MEMORY at boot: learning_save keeps
    # taught facts (semantic, taught, goals ...) but not the sequence store,
    # so the steps are taught as facts "R74 plan step NN" and the sequence is
    # re-learned from what the partition recalls -- the order lives in leCore,
    # the script only asks for it.
    steps = []
    for i in range(1, 40):
        r = m.ask("R74 plan step %02d" % i)
        if r.get("tier") != "T0" or not r.get("answer"):
            break
        steps.append(r["answer"])
    if steps:
        m.learn_plan("r74-ux", steps)
        m._r74_steps = steps
    return m


def plan_steps(m):
    return getattr(m, "_r74_steps", None) or PLAN


def teach():
    m = boot()
    n = 0
    for q, a in FACTS:
        r = m.teach(q, a)
        if not r.get("taught"):
            print("REFUSED:", q, r.get("reason"))
        else:
            n += 1
    # every backlog item, as a fact, straight from the document
    txt = open(BACKLOG, encoding="utf-8").read()
    for head, body in re.findall(r"^### (\d+\. [^\n]+)\n(.*?)(?=^###|^## |\Z)", txt, re.S | re.M):
        body = " ".join(body.split())
        r = m.teach("R74 backlog item " + head.split(".")[0] + ": " + head.split(". ", 1)[1].split("  ")[0],
                    body[:900])
        n += bool(r.get("taught"))
    for i, step in enumerate(PLAN, 1):
        m.teach("R74 plan step %02d" % i, step)
    m.teach("what is the r74-ux plan in order", " ; ".join(PLAN))
    for a, b in zip(PLAN, PLAN[1:]):
        m.teach("what comes after " + a.split(" ", 1)[1][:50], b)
    m.learn_plan("r74-ux", PLAN)
    rep = m.learning_save(PART)
    print("taught %d facts, plan of %d steps -> %s (%d bytes)"
          % (n, len(PLAN), rep["path"], rep["bytes"]))


def show():
    m = boot()
    print("partition:", m._autoboot_report["mounted"], "| session:", SESSION)
    steps = plan_steps(m)
    print("\nR74 UX plan (learn_plan r74-ux, %d steps recalled from the partition):" % len(steps))
    for i in range(len(steps)):
        print("  ", m.step_at("r74-ux", i))
    print("\nprecedes(lasso, gradient):", m.precedes("r74-ux", steps[0], steps[1]))
    print("what comes after the gradient tool ->", m.ask("what comes after " + steps[1].split(" ", 1)[1][:50])["answer"])
    print("\nask('what is the R74 task for leStudio') ->")
    print("  ", m.ask("what is the R74 task for leStudio")["answer"])


def ask(q):
    m = boot()
    r = m.ask(q)
    print(r.get("tier"), r.get("via"), "\n", r.get("answer"))
    if r.get("tier") != "T0":
        for h in m.session_search(q, k=3)["hits"]:
            print("  near:", h["question"], "->", h["answer"][:160])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "teach":
        teach()
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        show()
