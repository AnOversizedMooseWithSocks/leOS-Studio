"""C-kernel evaluation of an analytic SDF tree -- a SHIM over the engine (leCore sweep 163).

docs/POLYSTUDIO_AUDIT.md: "ccrun.py (C twin of zigrun, measured 4.0x at n=1e5) -> holographic_ccrun /
c_batch_eval: already upstreamed; the app should import the engine's and delete its copy." The C emission
and the content-addressed compile cache now live in holographic.io_and_interop.holographic_ccrun. What
stays here is the APP'S policy only: the size threshold below which the Python evaluator wins, and the
DSL -> C glue that names the kernel entry point.
"""
import os

_THRESH = int(os.environ.get("POLYSTUDIO_CC_MIN_POINTS", "20000"))


def cc_available():
    from holographic.io_and_interop.holographic_ccrun import cc_available as _cc
    return _cc()


def should_use(n):
    """The app's policy: below _THRESH points the vectorised Python SDF is faster than a compile."""
    return n >= _THRESH and cc_available()


class _Kernel:
    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def eval(self, P):
        import numpy as np
        P = np.asarray(P, float)
        return np.asarray(self._fn(P[:, 0], P[:, 1], P[:, 2]), float)


def get_sdf_kernel(dsl):
    """Compile the tree's own map(p) (engine emitter, dialect c_f64) through the engine's cached C
    compiler and return an object with .eval(P) -- the contract the field cache calls."""
    import lecore
    from holographic.io_and_interop.holographic_ccrun import build_batch_source, compile_cached
    from holographic.mesh_and_geometry.holographic_sdf import from_dsl
    m = lecore.UnifiedMind(dim=256, seed=0)
    src = m.sdf_dialect(from_dsl(dsl), "c_f64")
    fn = compile_cached(build_batch_source(src, dtype="f64"), opt="fast")
    return _Kernel(fn)
