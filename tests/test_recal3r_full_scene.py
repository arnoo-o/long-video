import cv2
import numpy as np

from long_video.data.recal3r_full_scene import (
    apply_sim3_c2w,
    apply_sim3_points,
    estimate_camera_sim3,
    fuse_voxel_observations,
    official_resize_crop,
    remap_model_map,
)


def test_sim3_alignment_recovers_camera_and_points():
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    poses[:, :3, 3] = np.array([[i, i * i * .1, (-1) ** i * .2] for i in range(8)])
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    scale = 2.5
    translation = np.array([3., -2., .5])
    target = poses.copy()
    target[:, :3, :3] = rotation @ poses[:, :3, :3]
    target[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    alignment = estimate_camera_sim3(poses, target)
    assert np.allclose(alignment.scale, scale)
    assert np.allclose(alignment.rotation, rotation)
    assert np.allclose(apply_sim3_c2w(poses, alignment), target)
    points = np.array([[1., 2., 3.]])
    assert np.allclose(apply_sim3_points(points, alignment),
                       scale * (points @ rotation.T) + translation)


def test_remap_matches_official_384x640_crop():
    transform = official_resize_crop(384, 640)
    assert (transform["crop_height"], transform["crop_width"]) == (304, 512)
    model_map = np.ones((304, 512), np.float32)
    output, inside = remap_model_map(model_map, transform, cv2.INTER_LINEAR)
    assert output.shape == (384, 640)
    assert inside[:, 10:-10].mean() > .98
    assert not inside[0].any() and not inside[-1].any()


def test_voxel_fusion_counts_distinct_frames():
    frame0 = (np.array([[.001, 0, 0], [.002, 0, 0]], np.float32),
              np.array([[10, 20, 30], [30, 40, 50]], np.uint8),
              np.array([1., 3.], np.float32))
    frame1 = (np.array([[.003, 0, 0]], np.float32),
              np.array([[50, 60, 70]], np.uint8),
              np.array([2.], np.float32))
    fused = fuse_voxel_observations([frame0, frame1], .02)
    assert len(fused["points_xyz"]) == 1
    assert fused["observation_count"].tolist() == [2]
    assert fused["points_rgb"].shape == (1, 3)