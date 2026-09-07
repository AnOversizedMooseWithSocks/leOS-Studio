"""Analytic SDF trees: the shapes the exact render path unions together."""
import numpy as np

class Tree:
    def __init__(self, fn, desc="sdf"): self.fn = fn; self.desc = desc
    def eval(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.asarray(self.fn(P), float)
    def to_dsl(self):
        """A printable form of the tree -- the app shows this and exports from it."""
        return self.desc
    def union(self, other):
        return Tree(lambda P, a=self, b=other: np.minimum(a.eval(P), b.eval(P)), "union")
    def intersect(self, other):
        return Tree(lambda P, a=self, b=other: np.maximum(a.eval(P), b.eval(P)), "intersect")
    def subtract(self, other):
        return Tree(lambda P, a=self, b=other: np.maximum(a.eval(P), -b.eval(P)), "subtract")
    def scale(self, k):
        return Tree(lambda P, a=self, k=float(k): a.eval(P / k) * k, "scale")
    def translate(self, v):
        v = np.asarray(v, float)
        return Tree(lambda P, a=self, v=v: a.eval(P - v), "translate")

def sphere(r=1.0):
    return Tree(lambda P, r=float(r): np.linalg.norm(P, axis=1) - r, "sphere")
def box(hx=0.5, hy=None, hz=None):
    """S.box(hx, hy, hz) -- HALF extents, matching the engine."""
    if hasattr(hx, "__len__"):
        s = np.asarray(hx, float)
    else:
        hy = hx if hy is None else hy; hz = hx if hz is None else hz
        s = np.array([float(hx), float(hy), float(hz)])
    def f(P, s=s):
        q = np.abs(P) - s
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    return Tree(f, "box")
def rounded_box(hx=0.5, hy=None, hz=None, radius=0.05):
    b = box(hx, hy, hz)
    return Tree(lambda P, b=b, r=float(radius): b.eval(P) - r, "rounded_box")
def plane(y=0.0):
    return Tree(lambda P, y=float(y): P[:, 1] - y, "plane")
