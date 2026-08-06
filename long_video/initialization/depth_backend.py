"""Unified geometry backends; legacy tuple-return depth interfaces were removed."""
from .geometry_backend import (
    GeometryPrediction,
    GroundTruthGeometryBackend,
    MultiViewGeometryBackend,
    Pi3GeometryBackend,
)

__all__ = [
    "GeometryPrediction",
    "GroundTruthGeometryBackend",
    "MultiViewGeometryBackend",
    "Pi3GeometryBackend",
]