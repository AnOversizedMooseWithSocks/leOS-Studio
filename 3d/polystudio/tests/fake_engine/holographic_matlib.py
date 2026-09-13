"""A handful of materials with the real attribute surface (glTF-PBR channels)."""
import numpy as np

class Material:
    def __init__(self, name, color, metallic=0.0, roughness=0.5, transmission=0.0, ior=1.5, emission=0.0):
        self.name = name
        self.base_color = tuple(color) + (1.0,) if len(color) == 3 else tuple(color)
        self.metallic = float(metallic); self.roughness = float(roughness)
        self.transmission = float(transmission); self.ior = float(ior)
        self.emission = float(emission); self.emissive = float(emission)   # engine exposes both spellings

_M = {
    "steel_brushed": Material("steel_brushed", (0.62, 0.64, 0.68), 1.0, 0.35),
    "plastic_white": Material("plastic_white", (0.86, 0.87, 0.88), 0.0, 0.45),
    "concrete":      Material("concrete",      (0.55, 0.54, 0.52), 0.0, 0.85),
    "glass_clear":   Material("glass_clear",   (0.92, 0.95, 0.97), 0.0, 0.08, 1.0, 1.52),
    "gold":          Material("gold",          (1.00, 0.77, 0.34), 1.0, 0.22),
}
_CLASSES = {"metal": ["steel_brushed", "gold"],
            "plastic": ["plastic_white"],
            "stone": ["concrete"],
            "glass": ["glass_clear"]}

def classes():
    """ml.classes() -> class NAMES (the backend iterates it, then calls by_class on each)."""
    return list(_CLASSES)
default = "plastic_white"

def names(): return list(_M)
def material(name):
    if name not in _M: raise KeyError(name)
    return _M[name]
def all_materials(): return dict(_M)

def by_class(cls):
    """Material NAMES in a class, empty list when unknown."""
    return list(_CLASSES.get(cls, []))
