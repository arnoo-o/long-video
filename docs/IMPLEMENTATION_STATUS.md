# Implementation Status

Last audited: 2026-08-04

## Completed and tested
- Holo360D reader matches RGB, mesh-depth EXR, mask and individual 3x4 pose files by timestamp.
- Official Holo360D code was audited at commit a54b75abc5d009aa98ec165f3a45ab19b48953e4. Individual pose files are treated as panorama c2w.
- Depth convention is RAY_DISTANCE. Invalid EXR values and invalid mask pixels become NaN.
- Equirectangular projection now follows OpenCV axes (+x right, +y down, +z forward); latitude uses -asin(y), fixing the vertical inversion.
- Eight canonical views have distinct c2w matrices and FOV-derived intrinsics.
- Backprojection applies inverse(K) and supports RAY_DISTANCE and Z_DEPTH.
- Node builder performs confidence-weighted voxel fusion. NodeStore writes atomically and verifies NPZ SHA-256.
- Point renderer has a PyTorch GPU scatter_reduce z-buffer implementation and a NumPy reference renderer.

## Actual tests
- REAL_HOLO_READER_OK: one official Indoor_013 RGB 1440x2880, mesh-depth 1440x2880, mask and pose read successfully; 3,878,121 valid depth pixels.
- Eight 128x128 perspective views and distinct canonical c2w matrices generated from the real sample.
- Existing synthetic smoke test passed before the geometry rewrite; it must be replaced with the real-data test suite.

## In progress
- Holo360D downloads: Indoor_013 is complete. Indoor_016 and 8views.bin are downloading in the background.
- A local sample contains a real RGB, EXR, mask, pose and corrected eight-view projection.

## Not yet complete
- Pi3X inference invocation is intentionally fail-fast until official runner/checkpoint wiring is configured.
- MVDiffusion has an external-backend interface only; the official environment and weights are not installed.
- WAH confidence is not yet injected into official WAH/Helios. No WAH patch has been applied.
- Online pipeline still needs real WAH generation and M1 reconstruction/verification.
- README, remaining configs, full integration tests and Habitat/ReplicaCAD are pending.
