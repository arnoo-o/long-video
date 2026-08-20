"""Sightline mainline: camera-ray conditioned Helios with latent history.

This package deliberately has no imports from the legacy WAH/PointWorld stack.
"""
from .rays import plucker_rays
from .history import HistoryManager
from .memory import LongTermKVMemory
from .conditioning import SightlineConditioner

__all__ = ["plucker_rays", "HistoryManager", "LongTermKVMemory", "SightlineConditioner"]
