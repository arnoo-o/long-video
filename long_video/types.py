from dataclasses import dataclass, field
from typing import Optional
import numpy as np

RAY_DISTANCE = "RAY_DISTANCE"
Z_DEPTH = "Z_DEPTH"
SCALE_MODES = {"metric_anchor", "dataset_calibrated", "relative"}


@dataclass
class ScaleMetadata:
    mode: str = "relative"
    meters_per_world_unit: Optional[float] = None
    uncertainty: float = 1.0
    anchor_source: str = "unobservable_from_same-center_views"
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in SCALE_MODES:
            raise ValueError(f"Unsupported scale mode: {self.mode}")
        if self.mode == "relative" and self.meters_per_world_unit is not None:
            raise ValueError("relative scale must not claim meters_per_world_unit")
        if self.mode != "relative" and (
            self.meters_per_world_unit is None or self.meters_per_world_unit <= 0
        ):
            raise ValueError("metric scale requires positive meters_per_world_unit")
        if not 0 <= float(self.uncertainty) <= 1:
            raise ValueError("scale uncertainty must be in [0,1]")

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
    schema_version: int = 3
    quality_metrics: dict = field(default_factory=dict)
    view_source: Optional[np.ndarray] = None
    view_image_confidence: Optional[np.ndarray] = None
    view_depth_confidence: Optional[np.ndarray] = None
    point_view_mask: Optional[np.ndarray] = None
    scale: ScaleMetadata = field(default_factory=ScaleMetadata)
    model_versions: dict = field(default_factory=dict)

@dataclass
class WarpBatch:
    rgb: np.ndarray
    depth: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    coverage_per_frame: np.ndarray
