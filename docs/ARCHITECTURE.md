# Architecture

The active SpatialNode is the only 3D memory read during a chunk. PointRenderer projects it into future OpenCV cameras and returns one synchronized WarpBatch. WAH encodes only warp RGB; visibility deletes unsupported tokens and confidence changes only the retained warp key logits.

Spatial confidence is source_prior times image, depth, view-angle, and reprojection confidence. Pixel values follow WAH visibility through temporal sampling and latent resizing. patch_short weighted pooling produces token confidence and visible ratio. The same keep index filters hidden state, RoPE, visibility, confidence, and key bias.

MemoryManager observes per-frame coverage. Consecutive low-coverage chunks enter TRANSITION. Keyframes are selected using translation and viewing-angle diversity. The geometry backend reconstructs a candidate in the existing world frame, high-confidence parent points are inherited, and coverage/reprojection/overlap/new-point metrics control promotion. Archived neighboring nodes are rendered at low cost when revisiting an old region.
