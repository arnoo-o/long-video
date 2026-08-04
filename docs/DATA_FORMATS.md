# Data Formats

## Camera and depth

c2w is float32 [T,4,4] in OpenCV camera axes: +x right, +y down, +z forward. Intrinsics are float32 [T,3,3]. RAY_DISTANCE is Euclidean distance along a normalized pixel ray; Z_DEPTH is camera-coordinate z.

## Sequence

sequence_xxx contains rgb/, depth/, masks/, poses_c2w.npy, intrinsics.npy, controls.json, prompt.txt, and metadata.json. Invalid depth is NaN.

## Spatial node

Each node directory contains metadata.json and node_arrays.npz. Arrays include source views, camera matrices, points_xyz, points_rgb, points_confidence, points_source, and observation_count. metadata records schema_version, parent_id, status, bounds, depth convention, quality metrics, and the NPZ SHA-256.

Sources: 0 observed, 1 synthesized, 2 generated, 3 verified, 4 invalid.

## WAH sample

sample_xxx contains first_frame.png, target_video.mp4, camera_poses.npy, warp_video.mp4, warp_visibility_mask.npy, warp_confidence.npy, warp_source.npy, prompt.txt, and metadata.json. Visibility/confidence/source share [T,H,W]. Legacy samples without confidence use one on visible pixels and zero elsewhere.
