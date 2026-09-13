"""A deliberately tiny stand-in for the leCore engine.

WHY THIS EXISTS
---------------
`/api/photo_post` shipped broken: it called `_tonemap`, which exists only inside `photo()`, so every
request raised NameError -> 500. Nothing caught it. `node --check` passed, `py_compile` passed, the
browser harnesses passed -- because on the client a dead post request is indistinguishable from a
slider you have not touched yet. The only thing that would have caught it is *calling the route*.

Calling the route needs an engine, and the build machine has no network to install `leos-core`. So
this provides the smallest surface the render endpoints actually touch: enough to make the analytic
photo path run end to end, with the SAME call signatures the real engine uses.

WHAT IT IS NOT
--------------
It is NOT a renderer. `path_trace` returns a cheap deterministic gradient, not global illumination.
These tests prove PLUMBING -- parameter handling, NDJSON framing, the cancel latch, the post cache,
route wiring. They say nothing about image quality. `quality_gate.py` against the real engine is
still the gate that matters before shipping a render change, and nothing here substitutes for it.
"""
