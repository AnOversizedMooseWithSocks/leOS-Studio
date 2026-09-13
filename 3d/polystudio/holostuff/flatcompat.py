"""flatcompat -- make the PACKAGED leCore engine answer to the demos' FLAT import names.

The current leCore checkout ships the engine as a package (`holographic/` with topic subpackages:
rendering/, mesh_and_geometry/, simulation_and_physics/, ...). The demo backends were written against
the older FLAT layout (`import holographic_sparsefield`, `from holographic_raymarch import ...`).

Rather than edit every backend (and every future demo someone pastes in from an older gist), this module
installs a tiny import hook: any `holographic_<name>` import is resolved to the packaged module
`holographic.<subpackage>.holographic_<name>` and ALIASED in sys.modules, so both spellings are the SAME
module object (one instance, one state -- no split-brain between a demo's flat import and the engine's
own packaged-internal imports).

Lazy: the map (flat name -> packaged path) is built once by a cheap directory scan; nothing is imported
until a demo actually asks for it. Call install() once, before demo backends load.
"""
import importlib, importlib.abc, importlib.machinery, importlib.util, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))

def _engine_root():
    """PYPI-FIRST engine resolution: if a `holographic` package is INSTALLED (pip install lecore / the pypi
    release), that is the engine the app builds upon -- the bundled repo copy here is the DEVELOPMENT OVERLAY
    for building ahead of the next pypi release. Set LECORE_ENGINE=bundled to force the overlay, or
    LECORE_ENGINE=installed to force site-packages."""
    force = os.environ.get("LECORE_ENGINE", "").lower()
    bundled = os.path.join(_HERE, "holographic")
    if force == "bundled":
        return bundled
    try:
        # the pypi distribution is `leos-core`; accept any of its plausible import names so a
        # pip-installed engine is found no matter how the wheel lays out its top-level package
        spec = None
        for _name in ("lecore", "holographic", "leos_core", "leoscore"):   # `lecore` is the published import name
            spec = importlib.util.find_spec(_name)
            if spec is not None and spec.submodule_search_locations:
                break
        if spec is not None and spec.submodule_search_locations:
            installed = list(spec.submodule_search_locations)[0]
            if os.path.abspath(installed) != os.path.abspath(bundled):
                if force == "installed" or not os.path.isdir(bundled):
                    return installed
                # both exist: bundled overlay wins (it is AHEAD of the pypi release by design)
                return bundled
            return installed
    except Exception:
        pass
    return bundled

_PKG_ROOT = _engine_root()


def _build_map():
    """flat module name -> dotted packaged path, by scanning holographic/*/holographic_*.py once."""
    # A handful of basenames now exist in TWO subpackages (an engine reorg side-effect). "First sorted
    # subpackage wins" would silently pick the wrong one, so pin the demo-facing choice explicitly:
    #   creature -> misc  (holds GridWorld / the RL agent the maze demo imports; the mesh one is a rig helper
    #                      no demo imports by its flat name)
    #   snap     -> mesh_and_geometry (holds snap_transform_delta / mesh snapping; the caching one is unrelated)
    overrides = {
        "holographic_creature": "holographic.misc.holographic_creature",
        "holographic_snap": "holographic.mesh_and_geometry.holographic_snap",
    }
    out = {}
    if not os.path.isdir(_PKG_ROOT):
        raise ImportError(
            "leCore engine not found.\n"
            "  This app needs the engine installed:  pip install \"leos-core[ui]\"\n"
            "  (or run from a checkout that vendors it at %s)\n"
            "  Set LECORE_ENGINE=installed|bundled to force one or the other."
            % _PKG_ROOT)
    for sub in sorted(os.listdir(_PKG_ROOT)):
        d = os.path.join(_PKG_ROOT, sub)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "__init__.py")):
            continue
        for f in os.listdir(d):
            if f.startswith("holographic_") and f.endswith(".py"):
                name = f[:-3]
                # first subpackage wins deterministically (sorted); collisions are pinned in `overrides`
                out.setdefault(name, f"holographic.{sub}.{name}")
    out.update(overrides)
    return out


_MAP = _build_map()


class _FlatFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _MAP:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        real = importlib.import_module(_MAP[spec.name])
        sys.modules[spec.name] = real       # alias: flat name and packaged name are ONE module object
        return real

    def exec_module(self, module):          # already executed by the packaged import
        pass


def _bind_package_root():
    """Make the resolved engine importable as `holographic`, whatever the distribution calls it.

    The engine's own modules import each other as `holographic.<sub>.<mod>`. A pip wheel may expose the
    package under a different top-level name (`lecore`), in which case finding it is not enough -- every
    internal import still says `holographic` and fails. So: put the engine root on sys.path and, if the
    package is installed under another name, alias it (and its already-imported submodules) to
    `holographic` in sys.modules. Without this the pip-installed path resolves and then dies on the first
    internal import, which is exactly what it did before this was added."""
    root = _PKG_ROOT
    parent = os.path.dirname(os.path.abspath(root))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if "holographic" in sys.modules:
        return
    base = os.path.basename(os.path.abspath(root))
    if base == "holographic":
        return                                            # normal case: the directory is already the name
    try:
        pkg = importlib.import_module(base)               # e.g. `lecore`
    except Exception:
        return
    sys.modules["holographic"] = pkg                      # alias the root ...
    prefix = base + "."
    for name, mod in list(sys.modules.items()):           # ... and anything already imported under it
        if name.startswith(prefix):
            sys.modules["holographic." + name[len(prefix):]] = mod


def install():
    _bind_package_root()
    if not any(isinstance(f, _FlatFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FlatFinder())
    return len(_MAP)
