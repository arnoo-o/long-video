"""Lazy Habitat-Sim RGB/depth renderer using project OpenCV c2w poses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _habitat_quaternion(rotation):
    from habitat_sim.utils.common import quat_from_coeffs
    # Habitat quaternion coefficients use [x,y,z,w]. Convert from rotation matrix lazily.
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(rotation).as_quat().astype(np.float32)
    return quat_from_coeffs(xyzw)


class HabitatSequenceRenderer:
    def __init__(self, scene_dataset_config, scene_id, height=256, width=256, hfov=90.0):
        import habitat_sim
        simulator = habitat_sim.SimulatorConfiguration()
        simulator.scene_dataset_config_file = str(scene_dataset_config)
        simulator.scene_id = str(scene_id)
        simulator.enable_physics = False
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "color_sensor"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [int(height), int(width)]
        rgb.hfov = float(hfov)
        rgb.position = [0.0, 1.5, 0.0]
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth_sensor"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [int(height), int(width)]
        depth.hfov = float(hfov)
        depth.position = [0.0, 1.5, 0.0]
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [rgb, depth]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
        self.sim.seed(0)
        self.agent = self.sim.initialize_agent(0)
        self.height, self.width, self.hfov = int(height), int(width), float(hfov)

    def initial_c2w(self):
        pose = np.eye(4, dtype=np.float32)
        self.sensor_position=np.array([0.0,1.5,0.0],np.float32)
        self.cv_to_habitat=np.diag([1.0,-1.0,-1.0]).astype(np.float32)
        self.trajectory_validation={}
        if self.sim.pathfinder.is_loaded:
            pose[:3,:3]=self.cv_to_habitat
            pose[:3, 3] = self.sim.pathfinder.get_random_navigable_point()+self.sensor_position
        else:
            pose[:3,:3]=self.cv_to_habitat
            pose[:3,3]=np.asarray(self.agent.get_state().position,np.float32)+self.sensor_position
        return pose

    def close(self):
        self.sim.close()
    def constrain_to_navmesh(self,poses_c2w):
        if not self.sim.pathfinder.is_loaded:
            raise RuntimeError("Habitat trajectory validation requires a loaded navmesh")
        corrected=[]; corrections=[]; previous=None
        for index,pose in enumerate(np.asarray(poses_c2w,np.float32)):
            value=pose.copy(); agent_rotation=value[:3,:3]@self.cv_to_habitat
            desired=value[:3,3]-agent_rotation@self.sensor_position
            if previous is None:
                allowed=self.sim.pathfinder.snap_point(desired)
            else:
                allowed=self.sim.pathfinder.try_step(previous,desired)
            if not np.isfinite(allowed).all():
                raise ValueError(f"navmesh returned invalid point for frame {index}")
            correction=float(np.linalg.norm(allowed-desired))
            value[:3,3]=allowed+agent_rotation@self.sensor_position
            corrected.append(value); corrections.append(correction); previous=allowed
        self.trajectory_validation={
            "navmesh_loaded":True,"collision_checked":True,
            "max_position_correction":float(max(corrections,default=0)),
            "mean_position_correction":float(np.mean(corrections) if corrections else 0),
            "corrected_frame_count":int(np.count_nonzero(np.asarray(corrections)>1e-5)),
        }
        return np.stack(corrected)


    def render(self, poses_c2w, output_dir, controls=None, prompt="indoor room"):
        root = Path(output_dir)
        for name in ("rgb", "depth", "masks"):
            (root / name).mkdir(parents=True, exist_ok=True)
        for index, pose in enumerate(np.asarray(poses_c2w, np.float32)):
            state = self.agent.get_state()
            # OpenCV y-down/z-forward to Habitat y-up/-z-forward.
            agent_rotation=pose[:3,:3]@self.cv_to_habitat
            agent_position=pose[:3,3]-agent_rotation@self.sensor_position
            if self.sim.pathfinder.is_loaded:
                snapped=self.sim.pathfinder.snap_point(agent_position)
                if not np.isfinite(snapped).all() or np.linalg.norm(snapped-agent_position)>0.15:
                    raise ValueError(f"trajectory frame {index} leaves navmesh/collides")
            state.position = agent_position
            state.rotation = _habitat_quaternion(agent_rotation)
            self.agent.set_state(state)
            observations = self.sim.get_sensor_observations()
            rgb = np.asarray(observations["color_sensor"])[..., :3]
            depth = np.asarray(observations["depth_sensor"], np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            Image.fromarray(rgb).save(root / "rgb" / f"{index:06d}.png")
            np.save(root / "depth" / f"{index:06d}.npy", np.where(valid, depth, np.nan))
            Image.fromarray((valid * 255).astype(np.uint8)).save(root / "masks" / f"{index:06d}.png")
        focal = 0.5 * self.width / np.tan(np.deg2rad(self.hfov) / 2)
        intrinsics = np.repeat(
            np.array([[focal, 0, (self.width-1)/2], [0, focal, (self.height-1)/2], [0, 0, 1]],
                     np.float32)[None],
            len(poses_c2w), axis=0,
        )
        np.save(root / "poses_c2w.npy", np.asarray(poses_c2w, np.float32))
        np.save(root / "intrinsics.npy", intrinsics)
        (root / "controls.json").write_text(json.dumps(controls or [], indent=2), encoding="utf-8")
        (root / "prompt.txt").write_text(str(prompt), encoding="utf-8")
        (root / "metadata.json").write_text(json.dumps({
            "depth_convention": "Z_DEPTH", "coordinate_convention": "OpenCV_c2w",
            "height": self.height, "width": self.width, "hfov_degrees": self.hfov,
            "sensor_extrinsics": {
                "agent_to_rgb_translation": self.sensor_position.tolist(),
                "agent_to_depth_translation": self.sensor_position.tolist(),
                "agent_to_sensor_rotation": np.eye(3,dtype=np.float32).tolist(),
            },
            "poses_are_sensor_c2w": True,
            "trajectory_validation": self.trajectory_validation,
        }, indent=2), encoding="utf-8")
        return root
