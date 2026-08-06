# Architecture

## Coordinate and scale contracts

Every camera uses OpenCV c2w: +x right, +y down and +z forward. RAY_DISTANCE is Euclidean length along a normalized pixel ray; Z_DEPTH is camera-space z. Resolution changes always use pixel-center-preserving resize_intrinsics.

ScaleMetadata is attached to every node. Same-center M0 is relative because pure rotations contain no translational scale. Metric anchors propagate meters_per_world_unit. All online distances are node units; configuration rejects legacy keys ending in _m.

## Initialization

Sparse observations are validated, projected to a periodic ERP and completed by a configurable backend. DiT360 runs in its own environment and returns ERP validity, fusion weight, conflict and RGB. Observed pixels are restored after generation. Eight canonical cameras receive per-pixel observed/synthesized source and confidence. Pi3 predicts geometry and its known-depth fit never masks unknown predictions.

NodeBuilder backprojects through inverse K, transforms by c2w, computes source/image/depth/view confidence and performs confidence-weighted voxel fusion. point_view_mask records contributing view IDs; observation_count is its population count.

## Rendering and WAH

The active node is rendered with an explicit device into one synchronized WarpBatch. Z-buffer winners index RGB, Z-depth, visibility, confidence and source together. Coverage is measured on a fixed angular grid.

WAH retains the official VAE, short history, patch_short and future-aligned frame indices. Visibility and confidence follow identical temporal and spatial layouts. Dropped tokens use the same index for hidden state, RoPE and confidence. Retained warp key bias is lambda times log confidence. Confidence one bypasses the entire new transformer path, which preserves exact official output.

## Online expansion

MemoryManager states are ACTIVE, TRANSITION, CANDIDATE, VALIDATING and ACTIVE_NEW_NODE. TransitionBuffer enforces maximum length, age, duplicate suppression, cooldown and rejection records. At least eight mapping and four held-out frames are selected.

Parent visible pixels retain parent Z-depth/source/confidence; new pixels use Pi3 depth and pixel confidence. Parent overlap anchors a robust scale fit. Candidate construction uses one confidence-weighted voxel fusion. Promotion uses independent overlap and held-out RGB/depth errors, scale and pose diagnostics, valid-depth/new-point ratios and confidence-weighted coverage.

Generated points become verified only with at least two distinct views, configured translation baseline, RGB agreement, depth agreement and occlusion agreement. Before the next chunk render, graph-adjacent or spatially nearby archived nodes are compared using confidence coverage, RGB/depth consistency and hysteresis. Node and session state updates use NodeStore transactional replacement.
## Oracle training and rollout isolation

Each source window owns an independent source-relative world, M0, TransitionBuffer, MemoryManager and WAH state. Only the source ERP RGB-D contributes oracle geometry. ERP pixel-center rays are centralized: the center maps to OpenCV +z, horizontal right/left quarters to +x/-x, and vertical down/up to +y/-y with periodic horizontal wrap.

The single-chunk branch renders and stores M0 external warp before training. The target RGB supervises a masked flow-matching objective; target Z-depth is offline-only. The four-chunk branch stores only trajectory and supervision, then renders warp online from the active node before each chunk. Official short/mid/long latent history, chunk index and one bounded decoded boundary frame persist; accumulated decoded RGB is streamed and released.

At a chunk boundary, generated RGB plus parent warp evidence enters the shared MemoryManager. Eight mapping frames feed Pi3 and four independent generated held-out frames constrain validation. Parent-visible RGB/depth retains its content origin with evidence role parent_warp; new RGB is model_generated with current_generation, and new depth is pi3_prediction. No offline target metric is accepted by promotion.
