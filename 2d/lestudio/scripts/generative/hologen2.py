"""HOLOGEN v2 -- fractal quadtree splat model, one-shot trained.

Devin's construction, implemented:
- ADAPTIVE SAMPLING: each image is encoded as a QUADTREE of splat
  blocks. A tile gets 14 colour splats fitted to the residual under its
  ancestors; if the tile's residual still carries energy, it splits
  into four children (detail exactly where needed, depth<=3).
- SELF-SIMILAR TOKENS: block splats are stored in TILE-NORMALISED
  coordinates, so a token is a scale-free stroke pattern -- the same
  vocabulary serves every depth (the inception property).
- TRIGRAM GENERATION: a superposed associative memory (HRNN's
  fixed-state mechanism) learns (depth, quadrant, parent token) ->
  child token over the whole collection; per-image salts keep each
  context's full multiset of continuations for stochastic sampling.
- HDRIFT PRIOR: a PCA drift manifold per depth regularises every
  generated block, and the root block is sampled from the depth-0
  manifold itself -- global composition from HDRIFT, recursive detail
  from the trigram memory.
- SPLIT LAW: p(split | depth, token) is learned from the data.

Generation: recursive rollout from the root. Rendering is
resolution-free. Colour statistics are finally imposed with leCore's
own color_transfer (covariance mode) toward the reference mosaic.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/work")
sys.path.insert(0, "/root/work/lecore_main")
from hdrift2 import _pursuit, _gauss_field
from holographic.caching_and_storage.holographic_supermemory import (
    SuperposedMemory, allocate)

RW, RH = 240, 180
KB = 15                 # splats per block (row 0 = DC)
DMAX = 3                # quadtree depth
V = 56                  # tokens per depth codebook
KEYSPACE = 512


def load(i):
    im = Image.open("/root/work/refs_real/ref_%d.png" % i).convert("RGB")
    return np.asarray(im.resize((RW, RH), Image.LANCZOS), float) / 255.0


def fit_block(tile, kk=KB):
    """Row 0 is the tile's DC (a huge splat carrying the mean colour);
    the rest are pursuit splats on the mean-subtracted tile. The DC
    carries the mass, so the corrections stay small and the block
    survives generative perturbation."""
    th, tw = tile.shape[:2]
    n = float(min(th, tw))
    mean = tile.reshape(-1, 3).mean(0)
    res = tile - mean[None, None, :]
    patch = max(4, int(n / 6))
    feats, _ = _pursuit(res, kk - 1, patch, n / 3.2, blur=2)
    F = np.zeros((kk, 8))
    F[0] = [0.5 * th, 0.5 * tw, mean[0] * 1.08, mean[1] * 1.08,
            mean[2] * 1.08, 0.95 * n, 0.0, 0.95 * n]
    F[1:] = np.asarray(feats, float)
    F[:, 0] /= th
    F[:, 1] /= tw
    F[:, 5:8] /= n
    return F


def render_block(F, y0, x0, th, tw, out, scale=1.0):
    """Accumulate a tile-normalised block into `out` at world scale."""
    H, W = out.shape[:2]
    n = float(min(th, tw)) * scale
    for row in F:
        cy = (y0 + row[0] * th) * scale
        cx = (x0 + row[1] * tw) * scale
        l11 = max(row[5] * n, 0.42)
        l22 = max(row[7] * n, 0.42)
        L = np.asarray([[l11, 0.0], [row[6] * n, l22]])
        try:
            S_inv = np.linalg.inv(L @ L.T)
        except np.linalg.LinAlgError:
            continue
        g = _gauss_field((H, W), cy, cx, S_inv)
        out += g[..., None] * np.asarray(row[2:5])[None, None, :]


def _feather(th, tw, frac=0.34):
    fy, fx = np.mgrid[0:th, 0:tw].astype(float)
    ey = np.clip(np.minimum(fy, th - 1 - fy) / max(th * frac, 1), 0, 1)
    ex = np.clip(np.minimum(fx, tw - 1 - fx) / max(tw * frac, 1), 0, 1)
    return (ey * ex)[..., None]


def composite_tree(nodes, H, W, scale=1.0):
    """Feathered replacement compositing: each depth REPLACES its tile
    region over the coarser levels (weight 0.78, feathered edges), so
    blocks stand alone and generation can perturb them safely."""
    H2, W2 = int(H * scale), int(W * scale)
    out = np.zeros((H2, W2, 3), np.float32)
    dmax = max(nd["depth"] for nd in nodes)
    for d in range(dmax + 1):
        for nd in nodes:
            if nd["depth"] != d:
                continue
            y0, x0, th, tw = nd["rect"]
            y0s, x0s = int(y0 * scale), int(x0 * scale)
            ths, tws = int(th * scale), int(tw * scale)
            local = np.zeros((ths, tws, 3), np.float32)
            render_block(nd["F"], 0, 0, th, tw, local, scale=scale)
            local = np.clip(np.maximum(local, 0)
                            / (1 + 0.12 * np.maximum(local, 0)), 0, 1)
            wgt = (1.0 if d == 0 else 0.78) \
                * (0.12 + 0.88 * _feather(ths, tws))
            sub = out[y0s:y0s + ths, x0s:x0s + tws]
            sub[:] = sub * (1 - wgt) + local * wgt
    return out


def _key(depth, quad, parent_tok, salt):
    h = (depth * 700001 + quad * 92821 + parent_tok * 269
         + salt * 2654435761)
    return (h % (KEYSPACE - V - 2)) + V + 2


class HoloGen2:
    def __init__(self, seed=5, tau=0.055):
        self.seed = seed
        self.tau = tau

    # ---------------------------------------------------------- encode
    def encode_image(self, img):
        """Quadtree encode; returns list of node dicts."""
        H, W = img.shape[:2]
        recon = np.zeros_like(img)
        nodes = []

        def visit(y0, x0, th, tw, depth, quad, parent_idx):
            # SELF-CONTAINED block: fit the tile's CONTENT. Residual
            # blocks are cancellation-coupled (the R21 failure, again):
            # they render only in concert with their ancestors, and any
            # generative perturbation breaks the cancellation into
            # confetti. A standalone block survives perturbation.
            tile = img[y0:y0 + th, x0:x0 + tw].copy()
            F = fit_block(tile)
            local = np.zeros((th, tw, 3), np.float32)
            render_block(F, 0, 0, th, tw, local)
            idx = len(nodes)
            nodes.append({"depth": depth, "quad": quad, "parent": parent_idx,
                          "rect": (y0, x0, th, tw), "F": F})
            rms = float(np.sqrt(((img[y0:y0 + th, x0:x0 + tw]
                                  - np.clip(local, 0, 1)) ** 2).mean()))
            nodes[idx]["split"] = bool(rms > self.tau and depth < DMAX)
            if nodes[idx]["split"]:
                hh, ww = th // 2, tw // 2
                for q, (dy, dx) in enumerate(((0, 0), (0, ww),
                                              (hh, 0), (hh, ww))):
                    visit(y0 + dy, x0 + dx,
                          th - hh if dy else hh, tw - ww if dx else ww,
                          depth + 1, q, idx)
            return idx

        visit(0, 0, H, W, 0, 4, -1)
        recon = composite_tree(nodes, H, W)
        return nodes, recon

    # ----------------------------------------------------------- train
    def train(self, n_imgs=10):
        self.trees = []
        blocks = {d: [] for d in range(DMAX + 1)}
        for i in range(n_imgs):
            nodes, recon = self.encode_image(load(i))
            self.trees.append(nodes)
            for nd in nodes:
                blocks[nd["depth"]].append(nd["F"].ravel())
            print("encoded", i, "nodes", len(nodes), flush=True)
        self.n_imgs = n_imgs
        # per-depth codebooks (k-means) + whitening + HDRIFT PCA manifold
        self.cb = {}
        rng = np.random.RandomState(self.seed)
        for d in range(DMAX + 1):
            X = np.stack(blocks[d])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Z = (X - mu) / sd
            v = min(V, max(4, len(Z) // 2))
            cent = Z[rng.choice(len(Z), v, replace=False)].copy()
            for _ in range(25):
                dist = ((Z[:, None] - cent[None]) ** 2).sum(-1)
                a = dist.argmin(1)
                for k in range(v):
                    m = a == k
                    if m.any():
                        cent[k] = Z[m].mean(0)
            tok_sd = np.stack([Z[a == k].std(0) + 1e-3 if (a == k).any()
                               else np.full(Z.shape[1], 0.05)
                               for k in range(v)])
            zm = Z.mean(0)
            U, S, Vt = np.linalg.svd(Z - zm, full_matrices=False)
            m_pcs = min(10, len(Z) - 1)
            C = U[:, :m_pcs] * S[:m_pcs]
            self.cb[d] = {"mu": mu, "sd": sd, "cent": cent, "tok_sd": tok_sd,
                          "assign": a, "v": v, "zm": zm,
                          "Vt": Vt[:m_pcs],
                          "c_lo": C.min(0), "c_hi": C.max(0),
                          "lo": X.min(0), "hi": X.max(0)}
        # token per node
        cursor = {d: 0 for d in range(DMAX + 1)}
        for tree in self.trees:
            for nd in tree:
                d = nd["depth"]
                nd["tok"] = int(self.cb[d]["assign"][cursor[d]])
                cursor[d] += 1
        # split law p(split | depth, token)
        self.p_split = {}
        for tree in self.trees:
            for nd in tree:
                k = (nd["depth"], nd["tok"])
                a, b = self.p_split.get(k, (0, 0))
                self.p_split[k] = (a + (1 if nd["split"] else 0), b + 1)
        # trigram memory: (depth, quad, parent token, salt) -> token
        keys, vals = [], []
        for i, tree in enumerate(self.trees):
            for nd in tree:
                pt = tree[nd["parent"]]["tok"] if nd["parent"] >= 0 else V
                keys.append(_key(nd["depth"], nd["quad"], pt, i))
                vals.append(nd["tok"])
        dim = int(allocate(len(keys), KEYSPACE))
        self.mem = SuperposedMemory(dim, KEYSPACE, seed=self.seed,
                                    precision="f32")
        self.mem.store(np.asarray(keys), np.asarray(vals))
        print("trained: %d transitions, mem dim %d" % (len(keys), dim),
              flush=True)

    # -------------------------------------------------------- generate
    def _sample_block(self, d, tok, rng, manifold=0.45, jitter=0.5):
        cb = self.cb[d]
        z = cb["cent"][tok] + rng.normal(0, 1, len(cb["mu"])) \
            * cb["tok_sd"][tok] * jitter
        # HDRIFT manifold pull at this depth
        c = (z - cb["zm"]) @ cb["Vt"].T
        span = cb["c_hi"] - cb["c_lo"]
        c = np.clip(c, cb["c_lo"] - 0.2 * span, cb["c_hi"] + 0.2 * span)
        proj = cb["zm"] + c @ cb["Vt"]
        z = (1 - manifold) * z + manifold * proj
        x = np.clip(z * cb["sd"] + cb["mu"], cb["lo"], cb["hi"])
        return x.reshape(KB, 8)

    def generate(self, seed, out_w=RW, out_h=RH, manifold=0.45):
        rng = np.random.RandomState(seed)
        out = np.zeros((out_h, out_w, 3), np.float32)
        scale = out_w / float(RW)

        def emit(y0, x0, th, tw, depth, quad, parent_tok):
            salt = rng.randint(self.n_imgs)
            key = _key(depth, quad, parent_tok, salt)
            v = int(np.ravel(self.mem.recall([key])["values"])[0])
            if not (0 <= v < self.cb[depth]["v"]):
                v = int(rng.randint(self.cb[depth]["v"]))
            F = self._sample_block(depth, v, rng, manifold=manifold)
            render_block(F, y0, x0, th, tw, out, scale=scale)
            a, b = self.p_split.get((depth, v), (0, 1))
            if depth < DMAX and rng.random() < (a / b if b else 0):
                hh, ww = th // 2, tw // 2
                for q, (dy, dx) in enumerate(((0, 0), (0, ww),
                                              (hh, 0), (hh, ww))):
                    emit(y0 + dy, x0 + dx,
                         th - hh if dy else hh, tw - ww if dx else ww,
                         depth + 1, q, v)

        emit(0, 0, RH, RW, 0, 4, V)
        out = np.maximum(out, 0)
        return out / (1.0 + 0.12 * out)


if __name__ == "__main__":
    g = HoloGen2(seed=5)
    g.train()
    tiles = []
    for s in (7, 21, 42, 63, 84, 105):
        img = g.generate(s)
        tiles.append(np.clip(img, 0, 1))
        print("gen", s, flush=True)
    grid = np.concatenate([np.concatenate(tiles[:3], 1),
                           np.concatenate(tiles[3:], 1)], 0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(
        "/tmp/hg2_grid.png")
    print("grid saved")
