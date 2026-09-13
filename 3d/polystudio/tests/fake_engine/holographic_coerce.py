"""Coercion helpers: turn loose dicts into engine objects."""
from holographic_render import Camera

def as_camera(spec, **kw):
    if isinstance(spec, Camera):
        return spec
    d = dict(spec or {})
    return Camera(eye=d.get("eye", (0, 0, 3)), target=d.get("target", (0, 0, 0)),
                  up=d.get("up", (0, 1, 0)), fov_deg=d.get("fov_deg", 45.0))
def as_array(x):
    import numpy as np
    return np.asarray(x, float)
