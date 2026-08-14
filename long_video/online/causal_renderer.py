"""Causal committed-node renderer shared by training and inference."""
from dataclasses import dataclass
import numpy as np

from ..geometry.point_renderer import render
from ..memory.memory_manager import MemoryManager


def attach_warp_provenance(warp, node):
    visible, source = np.asarray(warp.visibility, bool), np.asarray(warp.source)
    parent, generated = visible & (source == 0), visible & np.isin(source, (2, 3))
    rgb_origin = np.full(source.shape, "", dtype="U24")
    depth_origin = np.full(source.shape, "", dtype="U24")
    role = np.full(source.shape, "", dtype="U24")
    rgb_origin[parent] = depth_origin[parent] = "oracle_source"
    rgb_origin[generated], depth_origin[generated] = "model_generated", "recal3r_prediction"
    role[visible] = "direct_source" if node.parent_id is None else "parent_warp"
    role[generated] = "current_generation"
    warp.rgb_content_origin, warp.depth_content_origin = rgb_origin, depth_origin
    warp.evidence_role = warp.rgb_evidence_role = role
    warp.depth_evidence_role = np.where(generated, "geometry_prediction", role)
    return warp


@dataclass
class CausalWarpResult:
    warp: object
    provenance: dict


class CausalActiveNodeRenderer:
    def __init__(self, active_node, *, node_id=None, renderer_kwargs=None, manager=None):
        if node_id is not None:
            active_node = active_node.load(node_id)
        self.active_node = active_node
        self.renderer_kwargs = dict(renderer_kwargs or {})
        self.manager = manager or MemoryManager(node_store=None)
        self.manager.register(active_node)
        self.events = []

    def render(self, cameras, *, frame_start, allow_reactivation=True):
        if int(frame_start) < int(self.active_node.created_frame):
            raise ValueError("causal renderer cannot render before node creation")
        before, reactivation = self.active_node.node_id, None
        if allow_reactivation:
            self.active_node, reactivation = self.manager.maybe_reactivate(self.active_node, cameras)
        warp = attach_warp_provenance(render(self.active_node, cameras, **self.renderer_kwargs), self.active_node)
        provenance = {"causal": True, "uses_future_gt": False, "active_node_at_start": before,
                      "active_node_id": self.active_node.node_id, "frame_start": int(frame_start),
                      "reactivation": reactivation,
                      "selection": "MemoryManager.maybe_reactivate" if allow_reactivation else "explicit_scheduled_render_node"}
        self.events.append(provenance)
        return CausalWarpResult(warp, provenance)
