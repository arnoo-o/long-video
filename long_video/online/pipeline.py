from ..data.controls import integrate_controls
from ..geometry.point_renderer import render
from ..types import CameraBatch
class OnlineSpatialHistoryPipeline:
    def __init__(self, active_node, memory_manager=None, prompt=''):
        self.active_node=active_node; self.memory_manager=memory_manager; self.prompt=prompt; self.current_camera_c2w=active_node.center_c2w.copy(); self.frame_index=0
    def generate_chunk(self, controls, intrinsics, height, width):
        poses=integrate_controls(self.current_camera_c2w,controls); cams=CameraBatch(poses,intrinsics,height,width); warp=render(self.active_node,cams); self.current_camera_c2w=poses[-1].copy() if len(poses) else self.current_camera_c2w; self.frame_index+=len(poses)
        return None, poses, warp
