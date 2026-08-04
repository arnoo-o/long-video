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
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth_sensor"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [int(height), int(width)]
        depth.hfov = float(hfov)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [rgb, depth]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
        self.agent = self.sim.initialize_agent(0)
        self.height, self.width, self.hfov = int(height), int(width), float(hfov)

    def close(self):
        self.sim.close()

    def render(self, poses_c2w, output_dir, controls=None, prompt="indoor room"):
        root = Path(output_dir)
        for name in ("rgb", "depth", "masks"):
            (root / name).mkdir(parents=True, exist_ok=True)
        for index, pose in enumerate(np.asarray(poses_c2w, np.float32)):
            state = self.agent.get_state()
            # OpenCV y-down/z-forward to Habitat y-up/-z-forward.
            basis = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            state.position = pose[:3, 3]
            state.rotation = _habitat_quaternion(pose[:3, :3] @ basis)
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
        }, indent=2), encoding="utf-8")
        return root
