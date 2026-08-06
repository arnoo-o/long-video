# Data Formats

## Camera, depth and scale

c2w is float32 [T,4,4] in OpenCV axes: +x right, +y down, +z forward. Intrinsics are float32 [T,3,3]. RAY_DISTANCE is Euclidean distance on a normalized ray; Z_DEPTH is camera-coordinate z.

ScaleMetadata contains mode (relative, metric_anchor or dataset_calibrated), meters_per_world_unit, uncertainty, anchor_source and diagnostics. Relative mode must have null meters_per_world_unit.

## Unified sequence

sequence_xxx contains rgb/, depth/, masks/, poses_c2w.npy, intrinsics.npy, controls.json, prompt.txt and metadata.json. Invalid depth is NaN. Habitat metadata records RGB/depth sensor extrinsics, poses_are_sensor_c2w and navmesh/collision validation.

## DiT360 completion

The subprocess manifest lists prompt, observed image path, yaw/pitch, FOV or intrinsics, optional c2w, eight target yaws, ERP/output sizes, model paths and output directory.

Output contains views_rgb.npy [8,H,W,3], view_poses.npy [8,4,4], intrinsics.npy [8,3,3], observed_masks.npy, source_maps.npy, image_confidence.npy, erp_rgb.npy, erp_valid_mask.npy, erp_fusion_weight.npy, erp_conflict.npy, metadata.json and preview_panorama.png.

## Spatial node schema v4

metadata.json stores node ID/status/parent, center c2w, bounds, coverage radius, depth convention, ScaleMetadata, model versions, quality metrics and the NPZ SHA-256.

node_arrays.npz stores view RGB/depth/c2w/intrinsics, view source/image/depth confidence, points XYZ/RGB/confidence/source, optional normals, point_view_mask and observation_count. Source IDs are 0 observed, 1 synthesized, 2 generated, 3 verified and 4 invalid. observation_count is the number of distinct contributing views, not raw pixel count.

session.json stores node state and parent-child edges.

## Transition frame

Each buffered frame carries generated RGB, c2w, intrinsics, old-node warp RGB/Z-depth/source/visibility/confidence, coverage and global frame index. Old-node depth is explicitly Z_DEPTH.

## WAH sample

sample_xxx contains first_frame.png, target_video.mp4, camera_poses.npy, warp_video.mp4, warp_visibility_mask.npy, warp_confidence.npy, warp_source.npy, prompt.txt and metadata.json. Visibility/confidence/source share [T,H,W]. Legacy samples without confidence use one on visible pixels and zero elsewhere.
## Oracle WAH sequence

Each sequence_id contains:

    source/
      source_erp_rgb.png
      source_erp_depth_ray_distance.npy
      source_erp_mask.png
      source_c2w_world.npy
      source_perspective.png
    target/
      target_rgb_for_loss/
      target_z_depth_for_eval.npy
      target_ray_distance_for_reference.npy
      target_valid_mask.npy
      target_c2w_world.npy
      target_c2w_local.npy
      intrinsics.npy
    single_chunk_warp/
      warp_rgb/
      warp_z_depth.npy
      warp_visibility.npy
      warp_confidence.npy
      rgb_content_origin.npy
      depth_content_origin.npy
      evidence_role.npy
      rgb_evidence_role.npy
      depth_evidence_role.npy
    session/nodes/node_000/
    primary_loss_mask_rgb.npy
    primary_loss_mask_latent.npy
    prompt.txt
    metadata.json

The ERP is 2:1 and source depth is RAY_DISTANCE. Perspective target evaluation depth and renderer depth are Z_DEPTH at 384x640. geometry_source_frame_ids contains only the source frame and future_geometry_used is false. Content origins are separate from evidence roles: Oracle M0 is oracle_source/direct_source; parent-warp pixels inherit parent origins; candidate new RGB is model_generated/current_generation; candidate new depth is pi3_prediction.
