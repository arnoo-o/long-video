"""Causal active-node rendering for Phase B history construction."""
from __future__ import annotations

from dataclasses import dataclass

from ..geometry.point_renderer import render
from ..memory.memory_manager import MemoryManager
from ..types import CameraBatch
from .dataset import attach_warp_provenance


@dataclass
class CausalWarpResult:
    warp: object
    provenance: dict


class CausalActiveNodeRenderer:
    """Render only from nodes established before the current frame.

    The current Oracle training dataset contains only M0. It is therefore
    deliberately recorded as M0-only; generated nodes are never inferred from
    future GT or silently loaded from a later session state.
    """

    def __init__(self, node_store, *, node_id="node_000", renderer_kwargs=None, manager=None):
        self.node_store = node_store
        self.renderer_kwargs = {
            "near": 0.05, "far": 100.0, "point_radius": 1, "chunk_points": 1000000,
            **dict(renderer_kwargs or {}),
        }
        self.active_node = node_store.load(node_id)
        self.manager = manager or MemoryManager(node_store=None)
        self.manager.register(self.active_node)
        self.node_mode = "M0-only"
        self.events = []

    def render(self, cameras: CameraBatch, *, frame_start: int, allow_reactivation: bool = True):
        if int(frame_start) < int(self.active_node.created_frame):
            raise ValueError("causal renderer cannot render before the active node was created")
        active_before = self.active_node.node_id
        reactivation = None
        if allow_reactivation:
            self.active_node, reactivation = self.manager.maybe_reactivate(
                self.active_node, cameras,
            )
        warp = attach_warp_provenance(render(
            self.active_node, cameras, **self.renderer_kwargs,
        ), self.active_node)
        provenance = {
            "causal": True,
            "uses_future_gt": False,
            "active_node_at_start": active_before,
            "active_node_id": self.active_node.node_id,
            "available_node_ids": sorted(self.manager.nodes),
            "selection": (
                "MemoryManager.maybe_reactivate" if allow_reactivation
                else "explicit_scheduled_render_node"
            ),
            "renderer": "long_video.geometry.point_renderer.render",
            "node_mode": self.node_mode,
            "frame_start": int(frame_start),
            "reactivation": reactivation,
        }
        self.events.append(provenance)
        return CausalWarpResult(warp=warp, provenance=provenance)
