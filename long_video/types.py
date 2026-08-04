from dataclasses import dataclass, field
from typing import Optional
import numpy as np

RAY_DISTANCE = "RAY_DISTANCE"
Z_DEPTH = "Z_DEPTH"

@dataclass
class CameraBatch:
    c2w: np.ndarray
    intrinsics: np.ndarray
    height: int
    width: int

@dataclass
class ViewSet:
    rgb: np.ndarray
    depth: np.ndarray
    depth_confidence: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    source: np.ndarray
    image_confidence: np.ndarray
    depth_convention: str = RAY_DISTANCE

@dataclass
class SpatialNode:
    node_id: str
    status: str
    parent_id: Optional[str]
    center_c2w: np.ndarray
    created_frame: int
    coverage_radius: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    view_rgb: np.ndarray
    view_depth: np.ndarray
    view_c2w: np.ndarray
    view_intrinsics: np.ndarray
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    points_confidence: np.ndarray
    points_source: np.ndarray
    observation_count: np.ndarray
    points_normal: Optional[np.ndarray] = None
    depth_convention: str = RAY_DISTANCE
    schema_version: int = 2
    quality_metrics: dict = field(default_factory=dict)

@dataclass
class WarpBatch:
    rgb: np.ndarray
    depth: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    coverage_per_frame: np.ndarray
