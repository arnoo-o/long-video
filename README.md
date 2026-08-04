# long-video

Confidence-aware, extensible spatial memory for Warp-as-History (WAH). The project keeps local 3D nodes outside Helios: each generation chunk renders the active point cloud into an incomplete future warp, then sends RGB, visibility, and confidence through the official WAH short-history path.

## Pipeline

1. Project 1-N user images (yaw, pitch, FOV) into eight canonical horizontal cameras.
2. Complete missing RGB with Holo oracle, precomputed views, or MVDiffusion.
3. Predict geometry with ground truth or the official Holo360D-finetuned Pi3 8-view checkpoint.
4. Fuse views into active node M0 with source-aware point confidence.
5. Integrate WASD/mouse controls into OpenCV camera-to-world poses.
6. GPU-render warp RGB, Z-depth, visibility, confidence, and source.
7. Encode warp RGB with the official WAH VAE and retain it in short history with future-aligned frame indices.
8. Pool spatial confidence with the exact VAE temporal sampling and Helios patch_short layout.
9. Add lambda*log(confidence) only to retained warp key columns in Helios self-attention.
10. When coverage falls, buffer generated keyframes, reconstruct and validate M1, and reactivate an archived node when it covers a revisited view better.

All project poses are float32 OpenCV c2w (+x right, +y down, +z forward). Holo360D mesh depth is RAY_DISTANCE; Pi3 and renderer depth are Z_DEPTH.

## Installation

Create a project environment without modifying WAH:

    pip install -r requirements-project.txt
    cp configs/paths.example.yaml configs/paths.yaml

External environments and weights remain on the H100 data disk. Apply the tracked WAH patch to the exact recorded upstream commit:

    WAH_ROOT=/ephemeral/mdu/long-video/third_party/Warp-as-History bash scripts/apply_wah_patch.sh
    WAH_ROOT=/ephemeral/mdu/long-video/third_party/Warp-as-History WAH_PYTHON=/ephemeral/mdu/venvs/wah/bin/python bash scripts/check_wah_patch.sh

## Initialization modes

- holo_oracle: complete eight views from a full Holo360D panorama and use mesh depth. This is the geometry upper bound.
- precomputed: load views_rgb.npy, views_depth.npy, view_poses.npy, intrinsics.npy, source_maps.npy, and image_confidence.npy.
- mvdiffusion_pi3: run official panorama outpainting in an isolated subprocess, overlay every observed pixel, then run official Pi3 8views geometry.

MVDiffusion setup:

    bash scripts/install_mvdiffusion.sh
    HF_TOKEN=... bash scripts/download_mvdiffusion_weights.sh
    /ephemeral/mdu/envs/longvideo-mvdiffusion/bin/python scripts/run_mvdiffusion_completion.py --manifest manifest.json

The released panorama checkpoint is installed, but Stable Diffusion 2 Inpainting requires accepting its Hugging Face license and a token.

## Geometry and M0

    PYTHONPATH=. python scripts/test_holo_geometry.py --zip /ephemeral/mdu/long-video-data/raw/holo360d/train/Indoor_013.zip --output outputs/geometry_debug --height 128 --width 128
    PYTHONPATH=. python scripts/test_pi3_8views.py --zip /ephemeral/mdu/long-video-data/raw/holo360d/test/Indoor_016.zip --checkpoint /ephemeral/mdu/long-video-data/raw/holo360d/ckpt/8views.bin --repo /ephemeral/mdu/long-video/third_party/Holo360D/Pi3_Finetuned_Holo360d --output outputs/pi3_test

SpatialNode stores its source views, world-space points, confidence, source class, observation count, schema version, and quality metrics. NodeStore uses an atomic directory replacement and verifies the NPZ SHA-256 on load.

## WAH confidence

The patch preserves the official sequence:

    warp RGB -> official VAE -> warp latent -> patch_short -> short history -> Helios

Visibility and confidence use identical chunk slicing, temporal indices, latent interpolation, patch kernel/stride, and token keep indices. Confidence one produces no mask and therefore preserves the official attention backend path. Lower confidence creates a negative additive key bias; ordinary history and target keys remain zero.

    PYTHONPATH=. python scripts/test_wah_confidence.py --wah-root /ephemeral/mdu/long-video/third_party/Warp-as-History

Canonical training samples contain first_frame.png, target_video.mp4, camera_poses.npy, warp_video.mp4, warp_visibility_mask.npy, warp_confidence.npy, warp_source.npy, prompt.txt, and metadata.json. Missing confidence remains backward-compatible and defaults to visibility.

## Online memory

OnlineSpatialHistoryPipeline returns generated_video, target_c2w, WarpBatch, and generation statistics. MemoryManager implements ACTIVE -> TRANSITION -> CANDIDATE -> VALIDATING -> ACTIVE_NEW_NODE and a coverage-based archived-node revisit check.

    PYTHONPATH=. python scripts/test_memory_transition.py

The deterministic test verifies the complete M0-to-active-M1 state transition. A real confidence-aware WAH/Helios chunk also ran on physical GPU 1 with full 33-frame warp conditioning and produced 33 frames at 384x640; GPU 0's vLLM was untouched.

## Habitat

    bash scripts/install_habitat.sh
    bash scripts/download_replicacad.sh
    PYTHONPATH=. /ephemeral/mdu/envs/longvideo-habitat/bin/python scripts/render_habitat_sequence.py --scene-dataset-config PATH --scene-id Baked_sc1_staging_00 --output outputs/habitat_sequence

The output follows the unified sequence format documented in docs/DATA_FORMATS.md. See docs/IMPLEMENTATION_STATUS.md for measured results and remaining external blockers.
