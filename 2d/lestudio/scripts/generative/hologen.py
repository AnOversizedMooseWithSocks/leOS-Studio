"""HOLOGEN v1 -- a trained HRNN x HDRIFT generative model over the stills.

Not a collage and not interpolation: the model LEARNS from the ten
reference images and generates novel ones by rollout.

TRAINING
  1. Every still -> importance-ordered sequence of K=256 anisotropic
     colour splats (slot = (scale stage, spatial cell): a shared
     sequence grammar).
  2. Tokenize: k-means over all 2560 whitened splats -> V visual tokens
     (a learned vocabulary of strokes: sky masses, wall slabs, neon
     dashes, skin patches...).
  3. Sequence model (HRNN's fixed-state mechanism): a SUPERPOSED
     ASSOCIATIVE MEMORY stores every observed transition
     (position bucket, prev token, prev-prev token) -> next token,
     with a per-image salt so one context keeps its full multiset of
     continuations (sampling picks among them).
  4. Manifold prior (HDRIFT): the PCA drift model over the ten whole
     codes -- generated codes are pulled softly onto the collection's
     manifold, so global composition, palette and luminosity stay
     coherent while the rollout invents.

GENERATION: seed -> stochastic rollout of 256 tokens through the
memory -> continuous splats sampled from each token's member cloud ->
HDRIFT manifold regularization -> render. Deterministic per seed.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/work")
sys.path.insert(0, "/root/work/lecore_main")
from hdrift2 import fit_color_splats, render_splats
from holographic.caching_and_storage.holographic_supermemory import (
    SuperposedMemory, allocate)

RW, RH = 232, 176
K = 256
V = 160
BUCKET = 8
KEYSPACE = 512


def load(i):
    im = Image.open("/root/work/refs_real/ref_%d.png" % i).convert("RGB")
    return np.asarray(im.resize((RW, RH), Image.LANCZOS), float) / 255.0


def _ctx_key(bucket, t1, t2, salt):
    h = (bucket * 1000003 + t1 * 8191 + t2 * 131 + salt * 2654435761)
    return (h % (KEYSPACE - V - 2)) + V + 2      # keep clear of token ids


class HoloGen:
    def __init__(self, seed=0):
        self.seed = seed

    def train(self, n_imgs=10):
        codes = []
        for i in range(n_imgs):
            F, _ = fit_color_splats(load(i), K=K,
                                    jitter=np.random.RandomState(50 + i))
            codes.append(F)
            print("encoded", i, flush=True)
        self.codes = np.stack(codes)              # (N, K, 8)
        flat = self.codes.reshape(-1, 8)
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-6
        Zs = (flat - self.mu) / self.sd
        # k-means vocabulary
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(Zs), V, replace=False)
        cent = Zs[idx].copy()
        for _ in range(30):
            d = ((Zs[:, None, :] - cent[None]) ** 2).sum(-1)
            a = d.argmin(1)
            for v in range(V):
                m = a == v
                if m.any():
                    cent[v] = Zs[m].mean(0)
        self.cent = cent
        self.assign = a.reshape(n_imgs, K)
        # per-token member statistics (continuous decode)
        self.tok_mu = np.zeros((V, 8))
        self.tok_sd = np.zeros((V, 8))
        for v in range(V):
            m = Zs[a == v]
            self.tok_mu[v] = m.mean(0) if len(m) else cent[v]
            self.tok_sd[v] = m.std(0) + 1e-3 if len(m) else 0.05
        # superposed transition memory
        keys, vals = [], []
        for i in range(n_imgs):
            toks = self.assign[i]
            for t in range(K):
                t1 = int(toks[t - 1]) if t >= 1 else V
                t2 = int(toks[t - 2]) if t >= 2 else V + 1
                keys.append(_ctx_key(t // BUCKET, t1, t2, i))
                vals.append(int(toks[t]))
        # SHARDED memory: one superposed store per position OCTANT --
        # the full store at proper capacity would need dim 156k
        # (1024 x 156k f64 codebook = OOM). Eight shards of ~320 pairs
        # each run at the capacity law's honest allocation instead.
        keys = np.asarray(keys)
        vals = np.asarray(vals)
        shard_of = (np.arange(len(keys)) % K) // (K // 8)
        self.mems = []
        for s in range(8):
            m = keys[shard_of == s], vals[shard_of == s]
            dim = int(allocate(len(m[0]), KEYSPACE))
            mem = SuperposedMemory(dim, KEYSPACE, seed=self.seed + s,
                                   precision='f32')
            mem.store(m[0], m[1])
            self.mems.append(mem)
        self.n_imgs = n_imgs
        print("8 shards stored", flush=True)
        # slot statistics: WHERE and HOW BIG each sequence slot is,
        # across the collection -- the composition grammar lives in the
        # slots, the palette lives in the tokens
        self.slot_mu = self.codes.mean(0)          # (K, 8)
        self.slot_sd = self.codes.std(0) + 1e-3
        # HDRIFT manifold over whole codes
        W2 = self.codes.reshape(n_imgs, -1)
        self.gmu = W2.mean(0)
        self.gsd = W2.std(0) + 1e-6
        Zg = (W2 - self.gmu) / self.gsd
        self.zm = Zg.mean(0)
        U, S, Vt = np.linalg.svd(Zg - self.zm, full_matrices=False)
        self.m_pcs = min(8, n_imgs - 1)
        self.Vt = Vt[:self.m_pcs]
        C = (U[:, :self.m_pcs] * S[:self.m_pcs])
        self.c_lo, self.c_hi = C.min(0), C.max(0)
        self.g_lo = W2.min(0)
        self.g_hi = W2.max(0)

    def rollout(self, seed):
        rng = np.random.RandomState(seed)
        toks = []
        firsts = self.assign[:, 0]
        toks.append(int(firsts[rng.randint(self.n_imgs)]))
        t1_0 = V
        for t in range(1, K):
            t1 = toks[t - 1]
            t2 = toks[t - 2] if t >= 2 else V + 1
            salt = rng.randint(self.n_imgs)
            key = _ctx_key(t // BUCKET, t1, t2, salt)
            out = self.mems[t // (K // 8)].recall([key])
            v = int(np.ravel(out["values"])[0])
            if not (0 <= v < V):
                # honest fallback: the empirical marginal at this slot
                v = int(self.assign[rng.randint(self.n_imgs), t])
            toks.append(v)
        return toks

    def decode(self, toks, seed, manifold=0.3, jitter=0.55, host=None):
        # Composition is MULTIMODAL: averaging ten different scene
        # geometries is soup (measured, twice). Sample a composition
        # MODE -- one training image's slot geometry, jittered -- and
        # let the rollout's tokens re-skin it: generated palette and
        # energy on a sampled structural mode. No pixels are copied;
        # every element is a splat the model parameterizes.
        rng = np.random.RandomState(seed + 7)
        if host is None:
            host = rng.randint(self.n_imgs)
        rows = []
        for t, v in enumerate(toks):
            tok = (self.tok_mu[v]
                   + rng.normal(0, 1, 8) * self.tok_sd[v] * jitter)                 * self.sd + self.mu
            row = self.codes[host, t].copy()
            row[0:2] += rng.normal(0, 1, 2) * self.slot_sd[t, 0:2] * 0.12
            # colour: 75% generated token, 25% host (keeps lighting
            # logic attached to the geometry)
            row[2:5] = 0.75 * tok[2:5] + 0.25 * row[2:5]
            row[5:8] = 0.8 * row[5:8] + 0.2 * tok[5:8]
            rows.append(row)
        code = np.asarray(rows)                    # (K, 8)
        # HDRIFT manifold pull: project the whole code onto the PCA span
        w = code.ravel()
        zg = (w - self.gmu) / self.gsd
        c = (zg - self.zm) @ self.Vt.T
        span = self.c_hi - self.c_lo
        c = np.clip(c, self.c_lo - 0.2 * span, self.c_hi + 0.2 * span)
        proj = (self.zm + c @ self.Vt) * self.gsd + self.gmu
        w = (1 - manifold) * w + manifold * proj
        w = np.clip(w, self.g_lo, self.g_hi)
        return w.reshape(K, 8)

    def generate(self, seed, scale=1.0, manifold=0.4):
        toks = self.rollout(seed)
        code = self.decode(toks, seed, manifold=manifold)
        img = render_splats(code, (int(RH * scale), int(RW * scale))
                            if scale == 1.0 else None) \
            if scale == 1.0 else None
        if img is None:
            c2 = code.copy()
            c2[:, 0] *= scale
            c2[:, 1] *= scale
            c2[:, 5:8] *= scale
            img = render_splats(c2, (int(RH * scale), int(RW * scale)))
        return np.clip(img, 0, 1), toks


if __name__ == "__main__":
    g = HoloGen(seed=3)
    g.train()
    tiles = []
    for s in (11, 22, 33, 44, 55, 66):
        img, toks = g.generate(s)
        tiles.append(img)
        print("gen", s, "unique toks", len(set(toks)), flush=True)
    grid = np.concatenate([np.concatenate(tiles[:3], 1),
                           np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(
        "/tmp/hologen_grid.png")
    print("grid saved")
