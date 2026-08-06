# long-video

Confidence-aware, extensible spatial memory for Warp-as-History (WAH). Local 3D nodes remain outside Helios: each chunk renders the active node into a future warp, then official WAH encodes warp RGB while visibility and confidence control the short-history keys Helios may trust.

All poses are float32 OpenCV camera-to-world matrices: +x right, +y down, +z forward.

## Pipeline

1. Validate 1-N same-optical-center images and yaw, pitch, FOV or pinhole intrinsics.
2. Project observations into a periodic 2048x1024 ERP.
3. Complete missing RGB with official DiT360 on FLUX.1-dev, then restore every observed pixel.
4. Project to eight 512x512, 90-degree views at yaw 0..315 degrees.
5. Predict geometry with the Holo360D-finetuned Pi3 8-view checkpoint.
6. Fuse RGB, depth, source, confidence and distinct-view provenance into M0.
7. Integrate scale-aware WASD/mouse controls into future c2w poses.
8. Render warp RGB, Z-depth, visibility, confidence and source.
9. Run official WAH VAE and patch_short with warp indices aligned to target frames.
10. Pool confidence over the exact latent/patch layout and bias only retained warp key logits.
11. On sustained coverage loss, buffer at least 12 frames: eight mapping plus four held-out.
12. Align Pi3 geometry to parent overlap, validate M1, promote it, and reactivate better archived nodes.

## H100 installation

Project and third-party environments remain separate:

    pip install -r requirements-project.txt
    cp configs/paths.example.yaml configs/paths.yaml
    bash scripts/install_dit360.sh
    HF_TOKEN=... bash scripts/download_dit360_weights.sh
    bash scripts/install_habitat.sh
    bash scripts/download_replicacad.sh

Every tested GPU command uses physical GPU 1:

    CUDA_VISIBLE_DEVICES=1 ...

Inside that process the configured device is cuda:0. GPU 0 is never selected.

## Initialization modes

- holo_oracle: full Holo360D panorama and mesh RAY_DISTANCE depth.
- precomputed: externally prepared canonical views and depth.
- sparse_images_pi3: configurable completion backend, currently DiT360, followed by Pi3 and M0 construction.

DiT360 uses a subprocess manifest. It supports non-square pinhole inputs, applies EXIF orientation, rejects explicit intrinsics invalidated by EXIF rotation, rejects distortion unless pre-undistorted, and verifies a common optical center when c2w translations exist. Output includes ERP RGB/valid/fusion/conflict maps and eight canonical views with per-pixel source/confidence.

    CUDA_VISIBLE_DEVICES=1 /ephemeral/mdu/envs/longvideo-dit360/bin/python       scripts/test_dit360_indoor016.py       --input outputs/pi3_8views_indoor016_fixed/rgb_00.png       --dit360-config configs/dit360.yaml --pi3-config configs/pi3.yaml       --output outputs/dit360_indoor016_final

## Scale policy

A same-center M0 has no translation parallax and cannot determine meters. Pi3 normalizes median valid depth to one node unit and stores ScaleMetadata mode relative; there is no default 3 m assumption. Motion speed, voxel size, near/far, coverage radius and transition baselines are interpreted in node scale.

Metric dataset or sensor depth produces dataset_calibrated or metric_anchor. M1 uses parent Z-depth overlap for robust median/MAD alignment. A relative parent stays relative; a metric parent propagates meters_per_world_unit. The known-depth convention is mandatory. known_mask affects scale/error fitting without deleting predictions outside the mask.

## SpatialNode and rendering

Schema v4 stores source views, view source/image/depth confidence, world points, normals, confidence, semantic source, distinct-view bit provenance, observation count, scale metadata and model versions. NodeStore checksums arrays, migrates older nodes, and rolls back node replacement if session-graph update fails.

The renderer has explicit device, near/far clipping, chunking, deterministic z-buffer ties and splatting. Coverage uses a fixed angular occupancy grid, independent of point radius and output resolution.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /ephemeral/mdu/venvs/wah/bin/python       scripts/test_holo_geometry.py       --zip /ephemeral/mdu/long-video-data/raw/holo360d/train/Indoor_013.zip       --output outputs/geometry_debug --height 128 --width 128

## WAH confidence patch

Apply only to the WAH commit recorded in third_party_versions.json:

    WAH_ROOT=/ephemeral/mdu/long-video/third_party/Warp-as-History       bash scripts/apply_wah_patch.sh
    WAH_ROOT=/ephemeral/mdu/long-video/third_party/Warp-as-History       WAH_PYTHON=/ephemeral/mdu/venvs/wah/bin/python       bash scripts/check_wah_patch.sh

The preserved path is:

    warp RGB -> official VAE -> warp latent -> short history / patch_short -> Helios

Visibility and confidence share chunk slicing, temporal sampling, latent interpolation and keep indices. Missing or all-one confidence takes the exact original path and creates no attention mask. Lower confidence adds a negative bias only to warp key columns; first-frame history, ordinary history and target keys stay at zero.

Training samples include warp_visibility_mask.npy, warp_confidence.npy and warp_source.npy. Legacy samples without confidence use one on visible pixels.

## Oracle-initialized single-scene WAH training

The P0 training path uses exactly one source frame per window. Its full 2048x1024 Holo360D ERP RGB, mesh RAY_DISTANCE depth, mask and world c2w are transformed into a source-relative frame and directly backprojected into a metric Oracle M0. It does not rebuild M0 from eight crops. WAH inputs are generated directly at 384x640 with one shared K and pixel-center convention.

target_rgb_for_loss is used only by the masked flow-matching loss. target_z_depth_for_eval is Z_DEPTH and is offline-only. Runtime contracts reject either target field from WAH history, TransitionBuffer, MemoryManager, candidate construction and promotion. M1 known pixels use parent-warp RGB/Z-depth while new pixels use WAH-generated RGB and Pi3 depth.

Machine paths are passed through --set or an untracked local override:

    python scripts/build_holo_oracle_sequences.py \
      --config configs/oracle_wah_training.yaml \
      --set holo_root=/path/to/Indoor_013.zip \
      --set wah_root=/path/to/Warp-as-History \
      --set wah_model=/path/to/helios-distilled \
      --set output_root=/path/to/oracle_sequences

    CUDA_VISIBLE_DEVICES=1 python scripts/train_oracle_wah_lora.py \
      --config configs/oracle_wah_training.yaml \
      --sequence /path/to/Indoor_013_train_000 \
      --set wah_root=/path/to/Warp-as-History \
      --set wah_model=/path/to/helios-distilled \
      --set checkpoint_root=/path/to/checkpoints

The official state reports 33 RGB frames per chunk and VAE temporal scale 4. Chunk stride is 32: chunk k>0 shares its first RGB boundary frame with chunk k-1, and the duplicate decoded frame is omitted when writing the long video. The one-frame source prefix is excluded from loss. The RGB primary mask is mapped by exact VAE temporal groups to nine latent frames; eight latent frames participate in loss.

Single-chunk training uses precomputed M0 warp. Four-chunk rollout initializes WAH state once and renders each chunk online from the node active at that chunk boundary. Candidate creation, validation, promotion or rejection happen only after a chunk. Only generated RGB enters memory. The current validation run completed four chunks and two Pi3 candidate validations; production thresholds rejected M1, so all four chunks correctly remained on M0. Multi-chunk optimizer training, M2 and reactivation are outside this P0.
## Online memory

OnlineSpatialHistoryPipeline reactivates archived nodes before rendering, creates the target trajectory, renders the warp, calls real WAH/Helios, and returns video, c2w, WarpBatch and statistics.

M1 promotion reports overlap RGB/depth, held-out RGB/depth, scale dispersion, pose error, valid depth ratio, new-point ratio and confidence-weighted coverage. A generated point becomes verified only after distinct translated-view support plus RGB, depth and occlusion agreement.

    PYTHONPATH=. python scripts/test_memory_transition.py
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /ephemeral/mdu/venvs/wah/bin/python       scripts/test_helios_memory_expansion.py       --video outputs/wah_full_chunk/bmx_17f.mp4       --pi3-repo third_party/Holo360D/Pi3_Finetuned_Holo360d       --checkpoint /ephemeral/mdu/long-video-data/raw/holo360d/ckpt/8views.bin       --output outputs/helios_memory_expansion --device cuda:0

## Habitat

ReplicaCAD output stores true RGB/depth sensor c2w and sensor extrinsics. Validation poses are constrained using snap_point and try_step on a loaded navmesh.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.       /ephemeral/mdu/envs/longvideo-habitat/bin/python       scripts/render_habitat_sequence.py       --scene-dataset-config /ephemeral/mdu/long-video-data/raw/replicacad/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json       --scene-id Baked_sc1_staging_00       --output outputs/habitat_sensor_pose --height 128 --width 128

See docs/IMPLEMENTATION_STATUS.md for measurements and remaining limitations.
## 24 FPS Practical-RIFE Oracle adaptation

`build_holo_oracle_24fps.py` scans Holo timestamps, rejects acquisition gaps, and allocates eight train windows, two diagnostic windows, and one disjoint four-chunk rollout window. Five real anchors become 33 model frames; seventeen anchors become 129 frames. Practical-RIFE 4.25 full generates exactly seven frames at 1/8 through 7/8 between anchors. Real anchors are byte-preserved.

Camera translation is linearly interpolated and rotation uses SLERP. Source ERP depth remains RAY_DISTANCE; rendered warp and anchor-only evaluation depth use Z_DEPTH. Interpolated evaluation depth is NaN. RGB supervision weights are 0 for the source prefix, 1 for real anchors, and 0.25 for RIFE-only frames, then averaged over the real VAE temporal groups.

Multi-window training is invoked through `scripts/train_oracle_wah_lora.py --mode smoke|train --manifest ...`. It uses the official WAH/Helios VAE, external short warp history, confidence patch, train-exact flow matching, and LoRA. Formal training uses four fresh random-window microsteps per optimizer step. Checkpoints contain LoRA, optimizer, scheduler, global step, RNG state, manifest SHA, Git SHA, and RIFE checkpoint SHA.

The four-chunk rollout renders each chunk from the active SpatialNode, initializes WAH state once, passes only generated RGB to MemoryManager, evaluates only real anchors, and writes 129 frames at 24 FPS. M1 uses parent warp evidence plus generated RGB and Pi3 geometry; target supervision never enters memory.