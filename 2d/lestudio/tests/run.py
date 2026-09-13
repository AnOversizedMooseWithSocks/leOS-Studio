#!/usr/bin/env python3
"""Run the leStudio test suite.

The suite is plain functions (no pytest dependency at runtime), so this is the
entry point:

    python tests/run.py                 # everything
    python tests/run.py --slowest 10    # everything, then the 10 worst offenders
    python tests/run.py --chunk 1/2     # first half only
    python tests/run.py -k paint        # only tests whose name contains "paint"

--chunk exists because the full suite outgrew some CI/step time limits; running
it as `1/2` then `2/2` covers the same ground in two shorter passes. --slowest
is how the 38 s Flow-warp test got found: it was testing a cache bound by
computing twenty real noise fields.
"""
import argparse
import contextlib
import importlib
import os
import sys
import tempfile
import time
import traceback
import types
import warnings

# R57: the server now BACKS documents onto the live .lews workspace directory
# (restore at boot, publish on autosave). The suite must never read a
# developer's real shared workspace at import nor publish test documents into
# it, so it gets its own throwaway root -- set before ANY test imports the
# server module, which reads LESTUDIO_WS once.
os.environ.setdefault("LESTUDIO_WS",
                      tempfile.mkdtemp(prefix="lestudio_test_ws_"))


def _install_pytest_shim():
    """The tests use two pytest helpers; provide them without the dependency."""
    if "pytest" in sys.modules:
        return
    shim = types.ModuleType("pytest")
    shim.importorskip = importlib.import_module

    class _Skip(Exception):
        pass
    shim.skip = lambda why="": (_ for _ in ()).throw(_Skip(why))
    shim._Skip = _Skip

    @contextlib.contextmanager
    def raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError("expected %s" % getattr(exc, "__name__", exc))

    shim.raises = raises
    sys.modules["pytest"] = shim


def _reset_workspace():
    """Every test starts from the SAME shared-server state: one fresh default
    document, empty graph, empty presence. Two ordering bugs came from tests
    mutating the shared workspace (a 64x48 resize silently repositioned a
    later test's geometry; a doc-lineup shuffle made zombie references line
    up differently) -- and a census found 61 tests touching shared state.
    Rather than converting them one by one, the runner resets the workspace
    between tests, which kills the whole contamination class. Any test this
    breaks was latently order-dependent."""
    import sys
    SV = sys.modules.get("lestudio.server")
    if SV is None:
        return                               # server never imported: nothing shared
    from lestudio import Document, NodeGraph
    d = Document(768, 512)
    SV.WS.docs = {d.id: d}
    SV.WS.graphs = {d.id: NodeGraph(d)}
    SV.WS.active = d.id
    SV.WS._wire()
    for k in ("clients", "tabuser", "joined", "names", "viewing", "activity"):
        SV.SYNC.setdefault(k, {}).clear()
    SV.SYNC["kicked"].clear()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", dest="filter", default="",
                    help="substring match on the test name")
    ap.add_argument("--chunk", default="",
                    help="run one slice, e.g. 1/2 or 3/4")
    ap.add_argument("--slowest", type=int, default=0,
                    help="also print the N slowest tests")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    _install_pytest_shim()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # test_studio is the suite; test_r4 carries the R4 sweep pins (kept
    # separate so the sweep's fixtures stay next to its backlog doc)
    # test_r5 carries the R5 sweep pins, same separation reasoning as test_r4
    mods = [importlib.import_module(m)
            for m in ("test_studio", "test_r4", "test_r5", "test_r6",
                      "test_r7", "test_r8", "test_r9", "test_r10", "test_r16", "test_r17", "test_r18", "test_r19", "test_r20", "test_r21", "test_r22", "test_r23", "test_r24", "test_r25", "test_r26", "test_r27", "test_r28", "test_r29", "test_r30", "test_r31", "test_r32", "test_r33", "test_r34", "test_r35", "test_r36", "test_r37", "test_r47", "test_r48", "test_r49", "test_r50", "test_r53", "test_r58", "test_r59", "test_r60", "test_r62", "test_r63", "test_r64", "test_r65", "test_r66", "test_r67", "test_r68")]
    by_name = {}
    for m in mods:
        for n in dir(m):
            if n.startswith("test_"):
                by_name[n] = m
    names = sorted(by_name)

    # A test defined twice is silently lost: Python keeps only the last
    # definition, so the file claims more coverage than the suite runs. That
    # happened -- two identical copies of test_vector_masks_re_derive -- and
    # nothing noticed because the run was still all-green. Compare what the
    # SOURCE defines against what the MODULE exposes, here in the runner rather
    # than inside test_studio.py, so shadowing the check itself cannot hide it.
    import ast
    import collections
    src = open(os.path.join(here, "test_studio.py")).read()
    defined = [n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    dupes = {k: v for k, v in collections.Counter(defined).items() if v > 1}
    if dupes:
        print("ERROR: shadowed test definitions (only the last one runs): %s"
              % ", ".join("%s x%d" % kv for kv in sorted(dupes.items())))
        return 1

    if args.filter:
        names = [n for n in names if args.filter in n]
    if args.chunk:
        idx, total = (int(x) for x in args.chunk.split("/"))
        if not 1 <= idx <= total:
            ap.error("--chunk must be like 1/2 with 1 <= idx <= total")
        size = (len(names) + total - 1) // total
        names = names[(idx - 1) * size: idx * size]

    timings, failures = [], []
    t_start = time.time()
    for name in names:
        t0 = time.time()
        try:
            _reset_workspace()
            getattr(by_name[name], name)()
            ok = True
        except Exception as _e:
            import pytest as _pt
            if isinstance(_e, getattr(_pt, "_Skip", ())):
                ok = True                     # pytest.skip in the shim: not a failure
            else:
                ok = False
                failures.append((name, traceback.format_exc()))
        dt = time.time() - t0
        timings.append((dt, name))
        if not args.quiet:
            print("%s %-58s %6.2fs" % ("ok  " if ok else "FAIL", name, dt),
                  flush=True)

    total = time.time() - t_start
    print("\n%d passed, %d failed in %.0fs"
          % (len(names) - len(failures), len(failures), total))
    for name, tb in failures:
        print("\n--- FAIL %s ---\n%s" % (name, tb))
    if args.slowest:
        print("\nslowest %d:" % args.slowest)
        for dt, name in sorted(timings, reverse=True)[:args.slowest]:
            print("  %6.2fs  %s" % (dt, name))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
