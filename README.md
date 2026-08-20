# long-video Sightline

The formal mainline is geometry-free Sightline: source RGB is resized with its
intrinsics, encoded to a source latent, and passed to pinned Helios. Subsequent
camera poses come from dataset c2w or WASD controls. Each 33-RGB-frame/9-latent
chunk uses native Helios source + 16/2/1 latent history, deterministic
Plücker-ray Q/K conditioning after QKNorm and native RoPE, and selected-layer
K/V-only long memory. The original six chunks, stride 32 and pyramid settings
`[2, 2, 2]` are fixed invariants.

Formal entrypoints:

```text
scripts/infer_sightline.py
scripts/train_sightline_dl3dv.py
scripts/build_sightline_correspondences.py
scripts/probe_sightline_layers.py
```

The Sightline process does not import or initialize WAH, PointWorld, ReCal3R,
Pi3X, depth or warp rendering. Existing WAH/PointWorld/ReCal3R/Pi3X files are
legacy/reference utilities only. Offline ReCal3R arrays may be read solely by
the sparse correspondence builder.

Use `configs/sightline.yaml` as the single configuration schema. Formal training
is not launched by repository setup. See `docs/sightline_architecture.md` for
the runtime contract and `third_party_versions.json` for pinned Helios/WAH
provenance.
