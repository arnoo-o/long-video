# Data formats

Generic RGB training manifests contain a `records` list. Each record has:

- `video`: ordinary RGB video path;
- `camera_poses`: float32 OpenCV camera-to-world matrices;
- `intrinsics`: one matrix or one per frame;
- `prompt`;
- `conditioning_frame_end` and `target_frame_start`;
- `uses_future_gt: false`.

Conditioning must end strictly before the supervised target. Runtime world
nodes use the versioned `SpatialNode` schema; renderer outputs use `WarpBatch`
RGB, Z-depth, visibility, confidence, source, and provenance fields.
