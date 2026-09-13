"""HOLODIFFUSION -- one-shot image diffusion on the patch manifold.

Milanfar's thesis, straight from leCore's holographic_denoise: a
denoiser IS a map of the manifold clean signals live on. Here the
manifold is the TRAINING SET'S PATCHES, and generation is
holographic_diffuse's move -- walk from noise by annealed denoising --
run coarse-to-fine:

  for scale in coarsest..finest:
      for t in steps:                       # annealed schedule
          x <- x + sigma_t * noise          # diffuse
          x <- NN-project(x | patch bank)   # denoise onto the manifold
      x <- upsample(x)

The patch projector replaces every (overlapping) patch of x with a
softmin blend of its nearest training patches at that scale
(overlap-added under a Gaussian window). Local patches enforce real
shapes and textures; the pyramid enforces global composition; the
annealed noise is what makes it GENERATIVE rather than a copy --
different seeds take different walks and compose different images.

Init options: pure noise, a class-mean init, or a HoloGen gaussian
colour key (the deterministic gradient scaffold). Class conditioning
restricts the patch bank to one film family. Deterministic per seed.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/work")
sys.path.insert(0, "/root/work/lecore_main")

RW, RH = 240, 180
FILMS = {"labyrinth": (0, 1, 2), "fifth": (3, 4, 5), "night": (6,),
         "akira": (7, 8, 9), None: tuple(range(10))}


def load(i, w, h):
    im = Image.open("/root/work/refs_real/ref_%d.png" % i).convert("RGB")
    return np.asarray(im.resize((w, h), Image.LANCZOS), float) / 255.0


def _resize(a, w, h):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((w, h), Image.LANCZOS), float) / 255.0


def _patches(img, psize, stride):
    H, W = img.shape[:2]
    ys = np.arange(0, H - psize + 1, stride)
    xs = np.arange(0, W - psize + 1, stride)
    out = np.empty((len(ys) * len(xs), psize * psize * 3), np.float32)
    k = 0
    for y in ys:
        for x in xs:
            out[k] = img[y:y + psize, x:x + psize].ravel()
            k += 1
    return out, ys, xs


class PatchBank:
    """The manifold map at one scale: training patches + a PCA metric."""

    def __init__(self, imgs, psize, stride, pca_dim=24, seed=0,
                 metric="raw"):
        self.psize = psize
        self.metric = metric
        Ps = [_patches(im, psize, stride)[0] for im in imgs]
        self.P = np.concatenate(Ps, 0)
        # STRUCTURE metric: per-patch mean colour removed before PCA.
        # Raw-patch distance has a central attractor (mid-brown mild
        # texture wins every ambiguous query -- the wood-mud collapse,
        # measured); mean-free matching is structural, and the output
        # patch inherits the QUERY's mean colour, so the init's colour
        # composition survives while the bank supplies real structure.
        n3 = self.P.reshape(len(self.P), -1, 3)
        self.Pmean = n3.mean(1)                          # (N, 3)
        Pc = (n3 - self.Pmean[:, None, :]).reshape(len(self.P), -1)
        self.Pc = Pc
        feats = self.P if metric == "raw" else Pc
        self._feats = feats
        rng = np.random.RandomState(seed)
        sub = feats[rng.choice(len(feats), min(4000, len(feats)),
                               replace=False)]
        self.mu = sub.mean(0)
        _, _, Vt = np.linalg.svd(sub - self.mu, full_matrices=False)
        self.B = Vt[:min(pca_dim, Vt.shape[0])]
        self.PZ = (feats - self.mu) @ self.B.T           # (N, d)
        self.PZ_n2 = (self.PZ ** 2).sum(1)

    def project(self, x, stride, temp=0.05, topk=1, rng=None,
                chunk=2048):
        """Replace x's patches with softmin blends of nearest bank
        patches; Gaussian-window overlap-add."""
        psize = self.psize
        Q, ys, xs = _patches(x, psize, stride)
        q3 = Q.reshape(len(Q), -1, 3)
        Qmean = q3.mean(1)                               # (M, 3)
        Qc = (q3 - Qmean[:, None, :]).reshape(len(Q), -1)
        QZ = ((Q if self.metric == "raw" else Qc) - self.mu) @ self.B.T
        idx_all = np.empty((len(QZ), topk), np.int64)
        wts_all = np.empty((len(QZ), topk), np.float32)
        for c0 in range(0, len(QZ), chunk):
            qz = QZ[c0:c0 + chunk]
            d2 = (self.PZ_n2[None, :] - 2 * qz @ self.PZ.T
                  + (qz ** 2).sum(1)[:, None])
            if topk == 1:
                part = d2.argmin(1)[:, None]
                w = np.ones((len(qz), 1), np.float32)
            else:
                part = np.argpartition(d2, topk, 1)[:, :topk]
                dd = np.take_along_axis(d2, part, 1)
                w = np.exp(-dd / (2 * max(temp, 1e-4)
                                  * psize * psize * 3))
                w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
            idx_all[c0:c0 + chunk] = part
            wts_all[c0:c0 + chunk] = w
        acc = np.zeros_like(x)
        wac = np.zeros(x.shape[:2], np.float32)
        g1 = np.exp(-((np.arange(psize) - psize / 2 + 0.5) ** 2)
                    / (2 * (psize / 2.6) ** 2))
        win = np.outer(g1, g1).astype(np.float32)
        k = 0
        for y in ys:
            for x0 in xs:
                if self.metric == "raw":
                    blend = (self.P[idx_all[k]]
                             * wts_all[k][:, None]).sum(0)
                    pk = blend.reshape(psize, psize, 3)
                else:
                    struct = (self.Pc[idx_all[k]]
                              * wts_all[k][:, None]).sum(0)
                    bmean = (self.Pmean[idx_all[k]]
                             * wts_all[k][:, None]).sum(0)
                    mean = 0.7 * Qmean[k] + 0.3 * bmean
                    pk = (struct.reshape(psize, psize, 3)
                          + mean[None, None, :])
                acc[y:y + psize, x0:x0 + psize] += pk * win[..., None]
                wac[y:y + psize, x0:x0 + psize] += win
                k += 1
        wac = np.maximum(wac, 1e-6)[..., None]
        return acc / wac


# scale ladder: (width, psize, stride, n_steps, sigma0)
# sigma stays BELOW the patch-identity threshold: at 0.5+ the NN
# matches under noise are random and mud is a fixed point (measured);
# at these levels each step re-arranges real structure instead.
# noise ONLY at the coarsest scale (where composition is decided);
# every finer scale is pure EM refinement -- re-noising mid-pyramid
# pushes content off-manifold and the NN answers with mud (measured).
LADDER = [
    (30, 7, 1, 6, 0.30),
    (45, 7, 1, 4, 0.0),
    (68, 7, 2, 4, 0.0),
    (102, 7, 2, 4, 0.0),
    (152, 7, 2, 3, 0.0),
    (240, 7, 3, 3, 0.0),
]


class HoloDiffusion:
    def __init__(self, seed=0, cls=None, metric="raw"):
        self.seed = seed
        self.cls = cls
        self.banks = []
        idxs = FILMS[cls]
        for (w, psize, stride, _, _) in LADDER:
            h = int(w * RH / RW)
            imgs = [load(i, w, h) for i in idxs]
            self.banks.append(PatchBank(imgs, psize, stride,
                                        seed=seed, metric=metric))
        print("banks ready (%s): %s patches" %
              (cls, [len(b.P) for b in self.banks]), flush=True)

    def generate(self, seed, init=None, temp=0.04, keep=0.0,
                 host=None):
        rng = np.random.RandomState(seed)
        w0, _, _, _, _ = LADDER[0]
        h0 = int(w0 * RH / RW)
        if init is None:
            # a noised coarse REAL image seeds global structure (the
            # patch walk then diverges from it, seed by seed)
            i0 = host if host is not None else \
                FILMS[self.cls][rng.randint(len(FILMS[self.cls]))]
            base = load(i0, w0, h0)
            x = np.clip(base + rng.normal(0, 0.35, base.shape), 0, 1)
        else:
            x = _resize(init, w0, h0)
            x = np.clip(x + rng.normal(0, 0.4, x.shape), 0, 1)
        for si, (w, psize, stride, steps, sig0) in enumerate(LADDER):
            h = int(w * RH / RW)
            x = _resize(x, w, h)
            bank = self.banks[si]
            for t in range(steps):
                anneal = 1.0 - t / max(steps - 1, 1)
                x_n = x + rng.normal(0, sig0 * anneal, x.shape)
                proj = bank.project(np.clip(x_n, 0, 1), stride,
                                    temp=temp * (0.5 + anneal))
                x = (1 - keep) * proj + keep * x
            print("scale %d (%dpx) done" % (si, w), flush=True)
        return np.clip(x, 0, 1)


if __name__ == "__main__":
    hd = HoloDiffusion(seed=1, cls=None)
    tiles = []
    for s in (3, 14, 15, 92, 65, 35):
        tiles.append(hd.generate(s))
        print("generated", s, flush=True)
    grid = np.concatenate([np.concatenate(tiles[:3], 1),
                           np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(
        "/tmp/hd_grid.png")
    print("grid saved")
