"""Unified geometry backends; legacy tuple-return depth interfaces were removed."""
from .geometry_backend import (
    GeometryPrediction,
    GroundTruthGeometryBackend,
    MultiViewGeometryBackend,
    Pi3GeometryBackend,
)
from .recal3r_geometry_backend import ReCal3RGeometryBackend

__all__ = [
    "GeometryPrediction",
    "GroundTruthGeometryBackend",
    "MultiViewGeometryBackend",
    "Pi3GeometryBackend",
    "ReCal3RGeometryBackend",
]
