from pathlib import Path
import tempfile
import unittest

import numpy as np

from long_video.data.camera import resize_intrinsics,rgb_to_uint8
from long_video.data.controls import integrate_controls
from long_video.geometry.backprojection import backproject_ray_distance,backproject_z_depth
from long_video.initialization.geometry_backend import (
    GeometryPrediction,GroundTruthGeometryBackend,MultiViewGeometryBackend,
    Pi3GeometryBackend,
)
from long_video.initialization.initial_node_pipeline import initialize_spatial_node
from long_video.memory.node_builder import build_from_views
from long_video.memory.node_store import NodeStore
from long_video.types import RAY_DISTANCE,Z_DEPTH,ScaleMetadata,ViewSet


class StaticGeometry(MultiViewGeometryBackend):
    def __init__(self,mode="relative",convention=Z_DEPTH): self.mode=mode; self.convention=convention
    def predict(self,view_rgb,view_c2w,intrinsics,known_depth=None,known_mask=None,
                known_depth_convention=None):
        depth=np.asarray(known_depth if known_depth is not None else
                         np.ones(view_rgb.shape[:3]),np.float32)
        valid=np.isfinite(depth)
        return GeometryPrediction(
            depth=depth,depth_confidence=valid.astype(np.float32),
            scale_info=({"mode":"relative","meters_per_world_unit":None,
                         "uncertainty":1.0,"anchor_source":"test"}
                        if self.mode=="relative" else
                        {"mode":"metric_anchor","meters_per_world_unit":1.0,
                         "uncertainty":0.0,"anchor_source":"test"}),
            depth_convention=self.convention,
        )


def tiny_views(float_rgb=False,two_views=1,convention=Z_DEPTH):
    rgb=np.zeros((two_views,4,4,3),np.float32 if float_rgb else np.uint8)
    rgb[...,0]=1.0 if float_rgb else 255
    depth=np.ones((two_views,4,4),np.float32)
    poses=np.repeat(np.eye(4,dtype=np.float32)[None],two_views,axis=0)
    k=np.repeat(np.array([[4,0,1.5],[0,4,1.5],[0,0,1]],np.float32)[None],two_views,axis=0)
    return ViewSet(rgb,depth,np.ones_like(depth),poses,k,
                   np.zeros_like(depth,dtype=np.int8),np.ones_like(depth),convention)


class FakePi3Model:
    def __call__(self,images):
        import torch
        b,n,_,h,w=images.shape
        points=torch.zeros((b,n,h,w,3),device=images.device)
        points[...,2]=2.0
        poses=torch.eye(4,device=images.device).reshape(1,1,4,4).repeat(b,n,1,1)
        return {"local_points":points,"camera_poses":poses}


class ScaleGeometryTests(unittest.TestCase):
    def test_ray_and_z_end_to_end(self):
        k=np.array([[2,0,0.5],[0,2,0.5],[0,0,1]],np.float32)
        z=np.ones((2,2),np.float32)
        xyz_z=backproject_z_depth(z,k)
        norm=np.linalg.norm(xyz_z,axis=-1).reshape(2,2)
        xyz_ray=backproject_ray_distance(norm,k)
        np.testing.assert_allclose(xyz_z,xyz_ray,rtol=1e-6,atol=1e-6)
        poses=np.eye(4,dtype=np.float32)[None]
        gt=GroundTruthGeometryBackend()
        ray=gt.predict(np.zeros((1,2,2,3),np.uint8),poses,k[None],
                       known_depth=norm[None],known_mask=np.ones((1,2,2),bool),
                       known_depth_convention=RAY_DISTANCE)
        self.assertEqual(ray.depth_convention,RAY_DISTANCE)
        np.testing.assert_allclose(ray.point_maps[0],xyz_z.reshape(2,2,3))

    def test_partial_known_depth_does_not_delete_prediction_and_no_3m(self):
        backend=Pi3GeometryBackend("unused","unused",device="cpu",input_size=8)
        backend._model=FakePi3Model(); backend._has_confidence_head=False
        rgb=np.zeros((8,4,6,3),np.uint8)
        poses=np.repeat(np.eye(4,dtype=np.float32)[None],8,axis=0)
        k=np.repeat(np.array([[4,0,2.5],[0,4,1.5],[0,0,1]],np.float32)[None],8,axis=0)
        mask=np.zeros((8,4,6),bool); mask[:,1:3,2:4]=True
        known=np.full((8,4,6),np.nan,np.float32); known[mask]=4
        anchored=backend.predict(rgb,poses,k,known,mask,Z_DEPTH)
        self.assertTrue(np.isfinite(anchored.depth[~mask]).all())
        self.assertEqual(anchored.scale_info["mode"],"relative")
        metric=backend.predict(
            rgb,poses,k,known,mask,Z_DEPTH,
            ScaleMetadata("dataset_calibrated",1.0,0.0,"test_metric_depth"),
        )
        self.assertEqual(metric.scale_info["mode"],"dataset_calibrated")
        self.assertEqual(metric.scale_info["meters_per_world_unit"],1.0)
        relative=backend.predict(rgb,poses,k)
        self.assertAlmostEqual(float(np.nanmedian(relative.depth)),1.0,places=5)
        self.assertEqual(relative.scale_info["mode"],"relative")
        self.assertNotIn(3.0,relative.depth)

    def test_relative_and_metric_m0(self):
        views=tiny_views(two_views=8)
        relative=initialize_spatial_node(
            views, StaticGeometry("relative"), {"voxel_size":0.1})
        self.assertEqual(relative.scale.mode,"relative")
        metric=initialize_spatial_node(
            views, StaticGeometry("metric_anchor"), {"voxel_size":0.1})
        self.assertEqual(metric.scale.mode,"metric_anchor")
        self.assertEqual(metric.scale.meters_per_world_unit,1.0)

    def test_rgb_and_distinct_views_and_store_migration(self):
        self.assertEqual(int(rgb_to_uint8(np.array([0.5],np.float32))[0]),128)
        one=build_from_views(tiny_views(float_rgb=True,two_views=1),voxel_size=0.001)
        two=build_from_views(tiny_views(float_rgb=True,two_views=2),voxel_size=0.001)
        self.assertTrue(np.all(one.points_rgb[:,0]==255))
        self.assertTrue(np.all(one.observation_count==1))
        self.assertTrue(np.all(two.observation_count==2))
        self.assertTrue(np.all(two.point_view_mask==3))
        with tempfile.TemporaryDirectory() as root:
            NodeStore(root).save(two)
            loaded=NodeStore(root).load(two.node_id)
            np.testing.assert_array_equal(loaded.point_view_mask,two.point_view_mask)
            self.assertEqual(loaded.schema_version,4)

    def test_intrinsics_resize_and_controls(self):
        k=np.array([[100,0,49.5],[0,120,39.5],[0,0,1]],np.float32)
        resized=resize_intrinsics(k,(80,100),(40,200))
        np.testing.assert_allclose(resized,[[200,0,99.5],[0,60,19.5],[0,0,1]])
        scale=ScaleMetadata("metric_anchor",0.5,0.0,"test")
        poses=integrate_controls(np.eye(4),[{"forward":1,"delta_time":0.1}],
                                 move_speed=1.0,scale=scale)
        self.assertAlmostEqual(float(poses[-1,2,3]),0.2)
        with self.assertRaises(ValueError):
            integrate_controls(np.eye(4),[{"delta_time":1.0}])
        r=integrate_controls(np.eye(4),[{"pitch_delta":10.0,"delta_time":0.1}])[-1,:3,:3]
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-5)

if __name__=="__main__":
    unittest.main()
