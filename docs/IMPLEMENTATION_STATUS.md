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

## Oracle WAH training P0 validation

- Indoor_013 produced independent train and diagnostic source windows. Each uses one 2048x1024 source ERP RGB-D frame, direct spherical point-cloud M0, dataset_calibrated scale and no future geometry. Perspective tensors are 384x640.
- Official WAH runtime contract is 33 RGB frames per chunk, VAE temporal scale 4, nine latent frames, one RGB source-prefix frame, chunk stride 32 and eight valid primary-loss latent frames.
- Real GPU-1 LoRA training completed two optimizer steps. Fixed-batch loss changed from 0.0262524653 to 0.0259536617. Gradient norms were 0.00676539 and 0.00723128. The checkpoint restored LoRA, optimizer, scheduler and global step, then continued training.
- Warp diagnostic losses were correct 0.0259536617, shuffled 0.0277263131 and empty 0.0240091328. Correct beat shuffled but not empty, so the two-step adapter is explicitly marked as not yet learning reliable warp use.
- Final fixed-seed four-chunk no-grad rollout completed in 85.14 s with one WAH state initialization. Online warp renderer node IDs were node_000 for chunks 0, 1, 2 and 3. Mean coverages were 0.63155, 0.06249, 0.06498 and 0.02235.
- Candidate construction and independent generated held-out Pi3 validation ran after chunks 2 and 3. Both candidates were rejected by production thresholds. The first rejection had confidence-weighted coverage 0.05462, held-out RGB/depth errors 0.11452/1.04853 over 565,223 new-region depth pixels, and pose error 0.47033. The second had 0.03303, 0.11722/0.61490 over 491,911 pixels, overlap depth error 1.07898 and pose error 0.44235. No promotion was fabricated, so next-chunk M1 switching was not exercised.
- Pi3 confidence_source was local_depth_continuity and confidence_type was heuristic because the released checkpoint lacks the confidence head.
- Candidate provenance was recorded in the real rollout: known RGB/depth origins were oracle_source with parent_warp evidence; new RGB was model_generated/current_generation; new depth was pi3_prediction/geometry_prediction. Generated image confidence was spatially varying (0.10144 to 0.24960, standard deviation 0.02664 in the first candidate), not a frame-wide constant.
- Training peak GPU-1 allocated/reserved memory was 50,096,549,888 / 52,361,691,136 bytes. Final rollout peak was 48,148,663,296 / 52,126,810,112 bytes. Latent state remained about 184 MB; accumulated decoded history was released to zero after each chunk while one fixed 2,949,120-byte boundary frame was retained.
- GPU 0 retained vLLM PID 675704 at about 42,450 MiB before, during and after both runs. All task CUDA contexts were restricted to physical GPU 1.
- Final CPU regression: 18/18 tests passed locally and 10/10 Oracle-specific tests passed on H100 with CUDA hidden. The clean WAH patch applied and all six modified Python files compiled.
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

- Indoor_013 has 338 stored frames but no 129-frame constant-rate acquisition run; the longest approximately 3 Hz uninterrupted run is 33 frames. The P0 uses consecutive stored frames and records acquisition_gap_count, maximum timestamp gap and assumed_fps instead of hiding this dataset limitation.
- Production thresholds rejected both generated M1 candidates, so a real M1-to-next-chunk renderer switch was not exercised. The boundary logic is implemented and unit-tested, but no promotion was forced.
- In the fixed two-step diagnostic, empty warp loss was lower than correct warp loss. The adapter therefore has not yet learned reliable warp use.

- Real DiT360 inference was validated for one image. Multi-image ERP fusion, conflict confidence and observation restoration have deterministic tests, but not a second full diffusion run.
- The real Helios M1 test assigns a controlled translated trajectory to coherent frames from the official WAH demo; it is not yet a camera-conditioned production capture with ground-truth motion.
- ReplicaCAD package reports missing navmeshes for unrelated sc4 scenes and the Habitat environment lacks Bullet articulated-object support. The selected static baked sc1 scene and its navmesh work.
- Pi3 uses the slower PyTorch RoPE2D fallback because the optional CUDA extension is unavailable.
- Single-chunk optimizer training and four-chunk no-grad rollout are complete. Multi-chunk optimizer training, M2, reactivation in this Oracle P0, and production threshold calibration remain out of scope.