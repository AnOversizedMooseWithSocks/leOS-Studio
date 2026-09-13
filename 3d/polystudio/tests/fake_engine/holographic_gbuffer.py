import numpy as np
def primary_gbuffer(scene, cam, W, H, material=None, sky=None, **kw):
    W,H=int(W),int(H)
    yy,xx=np.mgrid[0:H,0:W]
    nrm=np.stack([xx/max(W-1,1)*2-1, yy/max(H-1,1)*2-1, np.ones((H,W))],axis=-1)
    alb=np.clip(np.stack([xx/max(W-1,1), yy/max(H-1,1), np.full((H,W),0.5)],axis=-1),0,1)
    dep=1.0+xx/max(W-1,1)*3.0
    return nrm, alb, dep
