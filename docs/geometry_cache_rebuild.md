# Geometry Cache Rebuild

All geometry caches before `recal-full-teacher-world-v4-rgb-anchor` /
`recal-causal-teacher-world-v4-rgb-anchor` are stale by design. These builds
use persistent RGB appearance anchors instead of confidence-averaged colors.
Rebuild the full ReCal cache, Phase-B causal cache, then source-only Pi3X W0:

```bash
PYTHONPATH=. python scripts/build_recal3r_full_scene_dataset.py --recal3r-repo /ephemeral/mdu/recovery-20260807/source/ReCal3R --recal3r-checkpoint /ephemeral/mdu/recovery-20260807/source/ReCal3R/src/cut3r_512_dpt_4_64.pth --dataset-root <DATASET_ROOT> --output-root <RECAL_ROOT> --device cuda:0 --record-count 100
PYTHONPATH=. python scripts/cache_geotoken_phase_b_worlds.py --recal3r-root <RECAL_ROOT> --dataset-root <DATASET_ROOT> --output-root <CAUSAL_ROOT> --trajectory-ids-json <SELECTED_IDS_JSON> --voxel-size 0.02
PYTHONPATH=. python scripts/cache_dl3dv_initial_pi3x_worlds.py --dataset-root <DATASET_ROOT> --cache-root <PI3X_W0_ROOT> --pi3x-repo /ephemeral/mdu/recovery-20260807/source/Pi3 --pi3x-checkpoint /ephemeral/mdu/recovery-20260807/models/pi3x/model.safetensors --device cuda:0 --record-count 100
```

VAE latent caches are not geometry caches and remain valid.
