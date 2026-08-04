# Implementation Status

Last audited: 2026-08-04

## Completed and actually tested

- Holo360D Indoor_013/Indoor_016 and 8views.bin downloads completed with official byte sizes.
- RGB, EXR mesh depth, mask, and 3x4 panorama c2w are strictly matched by frame ID.
- Corrected equirectangular projection uses OpenCV axes and eight distinct canonical c2w matrices.
- RAY_DISTANCE/Z_DEPTH backprojection, confidence-weighted voxel fusion, atomic node storage, GPU scatter-reduce z-buffer, chunk_points, and point splatting are implemented.
- Real Indoor_013 geometry loop at 128x128: RGB MAE 0.005556, Z-depth MAE 0.004173 m; GPU/NumPy reference passed.
- Official Holo360D Pi3 8views.bin loaded and ran on Indoor_016. Median-scale-aligned Z-depth MAE is 0.075230 m, AbsRel 0.077395, valid ratio 0.85747. The released checkpoint has no confidence head; deterministic local-depth continuity is recorded as the fallback.
- Official MVDiffusion repository and pano_outpaint.ckpt are installed. Indoor_016 observed-view projection/overlay smoke passed with eight 128x128 views and observed ratio 0.24234.
- Official WAH at commit 09aa646... is patched for warp_confidence_mask and compiles. Unit tests pass for exact confidence-one baseline path, negative low-confidence key bias, token pooling, and future index preservation.
- Unified initialization and online WAH adapter are implemented.
- MemoryManager deterministic geometry test passed through M0 -> candidate M1 -> active M1 with coverage 1.0 and zero test-scene reprojection error.
- Habitat-Sim 0.3.3 is installed in an isolated environment.
- Real confidence-aware WAH/Helios ran on physical GPU 1 with a full 33-frame warp rollout and produced 33 frames at 384x640. GPU 0's vLLM remained untouched.
- ReplicaCAD baked-lighting v1.6 downloaded successfully (1.6 GB); an 85-frame 128x128 RGB/depth sequence rendered on GPU 1.

## External or runtime blockers

- MVDiffusion inference still needs licensed stabilityai/stable-diffusion-2-inpainting files. Hugging Face returned 401 until the user accepts the license and supplies HF_TOKEN.
- The WAH project training cache primarily renders online Pi3X warps. This repository provides canonical confidence/source sample loading, but a full training run is intentionally out of scope.
- M1's deterministic state-machine test is complete; a fully coupled M1 reconstruction from newly generated Helios frames remains an end-to-end follow-up.
