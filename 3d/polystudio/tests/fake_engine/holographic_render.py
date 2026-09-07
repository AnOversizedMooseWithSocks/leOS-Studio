import numpy as np
class Camera:
    def __init__(self, eye=(0,0,3), target=(0,0,0), up=(0,1,0), fov_deg=45, **kw):
        self.eye=np.asarray(eye,float); self.target=np.asarray(target,float)
        self.up=np.asarray(up,float); self.fov_deg=float(fov_deg)

    def ray_dirs(self, W, H):
        """(eye, dirs) with dirs shaped (H*W, 3), unit length -- the real signature."""
        W, H = int(W), int(H)
        fwd = self.target - self.eye
        n = np.linalg.norm(fwd) or 1.0
        fwd = fwd / n
        right = np.cross(fwd, self.up)
        rn = np.linalg.norm(right) or 1.0
        right = right / rn
        up = np.cross(right, fwd)
        half = np.tan(np.radians(self.fov_deg) / 2.0)
        yy, xx = np.mgrid[0:H, 0:W]
        sx = ((xx + 0.5) / W * 2 - 1) * half * (W / max(H, 1))
        sy = (1 - (yy + 0.5) / H * 2) * half
        d = (fwd[None, None, :] + sx[..., None] * right[None, None, :] + sy[..., None] * up[None, None, :])
        d = d.reshape(-1, 3)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        return self.eye, d
def fit_camera(mesh_or_bounds, **kw): return Camera()
def rasterize_mesh(mesh, cam, width=64, height=48, **kw):
    W,H=int(width),int(height)
    yy,xx=np.mgrid[0:H,0:W]
    return np.clip(np.stack([xx/max(W-1,1), yy/max(H-1,1), np.full((H,W),0.5)],axis=-1),0,1)


class Light:
    """Light("directional", direction=..., color=..., intensity=...) / Light("ambient", intensity=...)"""
    def __init__(self, kind="directional", direction=(0, -1, 0), color=(1, 1, 1), intensity=1.0, **kw):
        self.kind = kind
        self.direction = np.asarray(direction, float)
        self.color = np.asarray(color, float)
        self.intensity = float(intensity)
