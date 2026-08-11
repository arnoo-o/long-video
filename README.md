# long-video

The repository implements one causal long-video architecture:

1. The pinned original Warp-as-History (WAH) path provides permanent source and
   temporal history plus the current rendered warp.
2. Frozen Pi3 predicts geometry only from causally available generated views.
3. `MemoryManager` validates candidates, preserves parent points, freezes shadow
   nodes, and activates them two chunks later.
4. Parent-First rendering prevents the newest delta from overwriting committed
   world points.
5. A small spatially aligned FiLM modulates only the Stage0 latent before the
   existing Helios patch embedding. Stage1 and Stage2 remain native Helios.

## Stage0 causal-world FiLM

For aligned Stage0 latent `Z0`, current-world latent `W0`, and continuous
renderer visibility `V0`, a per-position 1x1x1 MLP computes 16-channel gamma
and beta:

    Z0' = Z0 * (1 + V0 * gamma(W0)) + V0 * beta(W0)

The MLP is fixed at `16 -> 32 -> 32`; its final layer is zero initialized.
There is no spatial or temporal pooling. A pre-hook applies it immediately
before the existing Stage0 patch embedding, and exact shape matching prevents
it from running at Stage1 or Stage2.

Helios, original WAH, VAE, and Pi3 are frozen. Only
`stage0_causal_world_film.film.*` is trainable. Generic RGB-video manifests use
camera poses and intrinsics; future target RGB/depth are supervision only and
are rejected from causal-world construction.

## Pinned WAH

Apply the confidence patch only to WAH commit
`09aa6461355b298bfced51007bd709a251d6033a`:

    WAH_ROOT=/path/to/Warp-as-History bash scripts/apply_wah_patch.sh
    WAH_ROOT=/path/to/Warp-as-History bash scripts/check_wah_patch.sh

The conditioning route remains the baseline route:

    warp RGB -> official VAE -> patch_short/history -> Helios

`WAHAdapter` passes renderer RGB, visibility, and confidence without replacing
their temporal grouping or token support.

## Causal world

Only generated RGB is appended to `TransitionBuffer`. Pi3 candidate geometry
is built after the chunk, never from the supervised target. Accepted nodes are
immutable shadows until their scheduled activation. Rendering splits stable
parent and latest delta point sets and performs hard Parent-First composition.

The main entrypoints are:

- `scripts/infer_stage0_causal_world.py`
- `scripts/train_stage0_causal_world_film.py`

Configuration lives in `configs/stage0_causal_world_film.yaml`,
`configs/online_memory.yaml`, `configs/pi3.yaml`, and `configs/wah.yaml`.
