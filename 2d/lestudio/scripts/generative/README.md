# Generative stack (R27-R32)

One-shot generative models trained on a user-supplied reference set
(`refs_real/ref_0..9.png` -- ten film stills; not committed). All pure
NumPy, deterministic per seed, built on leCore faculties.

- `holoscore.py` -- **HoloScore v2, the accepted generator.** Score-based
  one-shot diffusion: ridge+RFF patch denoisers per (scale, sigma),
  conditional cascade with residual targets, whole-image GlobalDenoiser
  stages through 54px, ancestral denoise-renoise sampling, conditioning
  augmentation (Ho & Saharia 2106.15282), edge-weighted ridge
  (Kadkhodaie & Simoncelli 2310.02557), final deterministic polish.
  Novelty-checked (gen-patch NN distance vs train-train baseline).
  Laws: iterated Langevin contracts to black -- sample ancestrally;
  patch models cannot coordinate composition -- global stages own the
  coarse scales; absolute cascade targets darken -- fit residuals;
  CFG guidance > 1 adds speckle here -- keep guidance=1.0.
- `holodiffusion.py` -- patch-manifold walker (GPNN-family; judged
  compositing rather than generation -- kept for the measured laws in
  its comments: no softmin averaging, noise only at the coarsest scale).
- `hologen.py` / `hologen2.py` -- HRNN x HDRIFT token-rollout and
  quadtree-splat models (the "stupid dots" era; kept for the
  SuperposedMemory sharding pattern and the cancellation-coupling law).
- `hdrift2.py` / `hdrift3.py` -- fast anisotropic splat pursuit fitter,
  relative ridge, PCA-latent drift, semantic slot fusion.
- `parallax_kit.py` -- depth-true parallax GIFs: z offsets from
  accumulated top-surface thickness, one shared palette, full-res fetch.

Run with `PYTHONPATH=/root/work/lecore_main` and the refs directory
beside them (paths at the top of each file).
