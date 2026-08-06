# Implementation Status

Last validated: 2026-08-06. H100 host: h100-demo. All GPU tests used physical GPU 1 through CUDA_VISIBLE_DEVICES=1; GPU 0 retained only the pre-existing vLLM process.

## Completed and measured

- Holo360D downloads are complete: Indoor_013.zip 8,935,677,935 bytes, Indoor_016.zip 7,637,161,165 bytes, 8views.bin 3,569,900,346 bytes.
- OpenCV c2w, panorama orientation, RAY_DISTANCE/Z_DEPTH conversion, intrinsics-aware backprojection, schema-v3 node fusion/storage and deterministic GPU rendering are implemented.
- Indoor_013 geometry loop at 128x128: 129,193 input points, 114,691 fused points, coverage 0.9445..0.9999, RGB MAE 0.005556, Z-depth MAE 0.004173 m; GPU/NumPy reference passed.
- Official DiT360 commit 3779fe7, FLUX.1-dev and official panorama LoRA ran on one Indoor_016 perspective input. Completion took 98.34 s. ERP observed ratio was 0.11617; canonical-view observed ratio was 0.24208. Peak allocated/reserved GPU memory was 46.18/54.22 GB.
- DiT360 then Pi3 then M0 completed: 132,181 points, mean point confidence 0.38021, Pi3 valid ratio 1.0, M0 scale relative, Pi3 time 17.28 s and peak allocated memory 7.40 GB.
- The Holo360D 8views checkpoint is official Pi3. It has no released confidence head; only those known missing confidence-head keys are accepted and deterministic local-depth continuity is recorded as fallback confidence.
- Final translated ReplicaCAD eight-view Pi3 validation passed: baseline 0.52677 m, depth MAE 0.06217 m, AbsRel 0.05058, predicted valid ratio 1.0, pose error 0.06606, pose scale dispersion 0.04678.
- WAH commit 09aa646 is patched for warp_confidence_mask in inference and training. Token mapping, negative key bias and training forward shape tests pass.
- Full WAH numerical equivalence passed with a fixed seed: original and patched confidence=1 outputs were both float32 [33,384,640,3], max/mean absolute error 0, exact array equality true. Patched peak GPU memory was 44.44 GB and inference took 5.42 s after model load.
- ReplicaCAD baked sc1 produced 85 RGB/depth frames at 128x128. Sensor-c2w test: valid depth 1.0, max pixel roundtrip 2.93e-5 px, max Z-depth roundtrip 4.77e-7, navmesh and collision checks true.
- Deterministic 12-frame M1 test passed with eight mapping and four held-out frames; verified point ratio 0.69091.
- Real Helios output drove M1 reconstruction and held-out validation; both M0 and overlap-aligned M1 remained relative-scale. Promotion passed with overlap RGB/depth 0.16604/0.001278, held-out RGB/depth 0.09583/0.003311, scale dispersion 0.10205, pose error 0.16654, valid depth ratio 1.0, confidence-weighted coverage 0.02133 and verified point ratio 0.94562. Returning to x=-1.5 reactivated node_000 with parent/active confidence coverage 0.01706/0.00544 and consistent RGB/depth errors 0.14052/0.00862.
- Local regression suite has eight tests covering ray/Z depth, partial known depth, no default 3 m, relative/metric scale, float RGB, distinct-view provenance, intrinsics resize, controls, ERP periodic recovery, EXIF fail-fast and formal HoloOracle initialization.

## Scale behavior

- Same-center sparse M0: relative units, median Pi3 depth normalized to one node unit.
- Metric Holo/Habitat depth: dataset_calibrated with meters_per_world_unit 1.
- M1: robust median/MAD alignment on parent high-confidence Z-depth overlap. Relative parent depth remains relative; metric parent metadata propagates. Scale diagnostics include ratio median, MAD, valid pixels, uncertainty and rejection reason.
- known_mask is used only for alignment/error fitting. Unknown regions retain Pi3 depth predictions.

## Removed after successful DiT360 closure

- H100: /ephemeral/mdu/long-video-third-party/MVDiffusion
- H100: /ephemeral/mdu/long-video-data/models/mvdiffusion
- Repository: configs/mvdiffusion.yaml, MVDiffusion backend and three install/download/run scripts.
- No matching MVDiffusion or SD2 entry existed in the shared Hugging Face cache, so no unrelated cache entry was deleted.

## Remaining limitations

- Real DiT360 inference was validated for one image. Multi-image ERP fusion, conflict confidence and observation restoration have deterministic tests, but not a second full diffusion run.
- The real Helios M1 test assigns a controlled translated trajectory to coherent frames from the official WAH demo; it is not yet a camera-conditioned production capture with ground-truth motion.
- ReplicaCAD package reports missing navmeshes for unrelated sc4 scenes and the Habitat environment lacks Bullet articulated-object support. The selected static baked sc1 scene and its navmesh work.
- Pi3 uses the slower PyTorch RoPE2D fallback because the optional CUDA extension is unavailable.
- No full training was run; only confidence-aware sample loading and a minimal training-forward shape test were required.