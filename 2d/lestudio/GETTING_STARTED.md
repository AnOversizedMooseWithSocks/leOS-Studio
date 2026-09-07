# leStudio v0.1.0 — build R54

A layered image editor with a non-destructive node graph, journal-first
documents (every stroke replays), generator brushes (scribble, hatch,
textile, perspective scatter), a live web UI, an agent/swarm API, and a
studio sage that remembers every lesson the painting rounds taught it.

## What's in this package

- `lestudio/` — the app source at commit 3f16c65 (R54), suite green 570.
- `lecore_memory/` — the studio sage's knowledge partition: 748 taught
  lessons spanning the whole painting arc (R8 → R54). Ship it next to
  where you run the server (or point LECORE_PARTITION at it) and
  /api/advise serves the house painting doctrine, paraphrases included.
- `lestudio-git-history.bundle` — the full 57-commit git history.
  `git clone lestudio-git-history.bundle lestudio-full` restores it.

## Run it

    cd lestudio
    pip install -e .            # pulls leos-core (leCore) from PyPI + deps
    lestudio                    # or: python -m lestudio.server

Open http://localhost:5050 — the canvas, layers, brushes (pencil, wet
media, textile 🧵, scribble 🌀, hatch ▤), fill tool (flood / generated /
scatter sources), node graph (incl. the Shade node), transform with
perspective warp, replay/timelapse, and the Sage panel.

If you develop against a local leCore checkout instead of the PyPI
package: PYTHONPATH=/path/to/lecore:src python -m lestudio.server

## Tests

    cd lestudio
    python tests/run.py             # full suite (570 green at this build)
    python tests/run.py --chunk 1/2 # halves, for CI time limits

## Agent quickstart

POST http://localhost:5050/api/…: paint, paint_batch, scribble,
hatchfill (line/hatch/both/weave/cross/stitch + depth), scatter
(grass/flowers/rocks/pebbles/reeds/ripples/custom, horizon +
perspective, inline poly gates), fill (generated sources), pwarp,
select, graph, replay/render, workspace.lews, advise (ask the sage /
teach it). Every generator emits journaled strokes: one undo per fill,
bit-faithful replay, timelapse for free.
