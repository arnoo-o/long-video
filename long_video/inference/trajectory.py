"""Build chunk-aligned, smoothly eased camera-control trajectories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import uuid


ROTATION_SIGNS = {"none": 0.0, "left": -1.0, "right": 1.0}
MOVEMENT_AXES = {
    "none": (0.0, 0.0),
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    "front_left": (math.sqrt(0.5), -math.sqrt(0.5)),
    "front_right": (math.sqrt(0.5), math.sqrt(0.5)),
    "back_left": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "back_right": (-math.sqrt(0.5), math.sqrt(0.5)),
}


def make_run_name() -> str:
    """Return a collision-proof name even for back-to-back GUI launches."""
    return f"gui_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class TrajectorySegment:
    rotation: str = "none"
    degrees: float = 0.0
    movement: str = "none"
    distance: float = 0.0
    chunks: int = 1

    def validate(self) -> None:
        if self.rotation not in ROTATION_SIGNS:
            raise ValueError(f"unsupported rotation direction: {self.rotation}")
        if self.movement not in MOVEMENT_AXES:
            raise ValueError(f"unsupported movement direction: {self.movement}")
        if not math.isfinite(float(self.degrees)) or float(self.degrees) < 0:
            raise ValueError("degrees must be finite and non-negative")
        if not math.isfinite(float(self.distance)) or float(self.distance) < 0:
            raise ValueError("distance must be finite and non-negative")
        if int(self.chunks) != self.chunks or int(self.chunks) <= 0:
            raise ValueError("chunks must be a positive integer")
        if self.rotation == "none" and self.degrees != 0:
            raise ValueError("degrees must be zero when rotation is none")
        if self.movement == "none" and self.distance != 0:
            raise ValueError("distance must be zero when movement is none")


def eased_increments(total: float, count: int) -> list[float]:
    """Cosine ease-in/out increments whose sum is exactly ``total``."""
    if count <= 0:
        raise ValueError("count must be positive")
    cumulative = [
        float(total) * (0.5 - 0.5 * math.cos(math.pi * index / count))
        for index in range(count + 1)
    ]
    return [right - left for left, right in zip(cumulative, cumulative[1:])]


def build_controls(
    segments: list[TrajectorySegment], *, fps: float = 24.0,
    controls_per_chunk: int = 32,
) -> list[list[dict[str, float]]]:
    """Convert editable relative-camera segments into the inference JSON layout."""
    if not segments:
        raise ValueError("at least one trajectory segment is required")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be positive")
    if controls_per_chunk <= 0:
        raise ValueError("controls_per_chunk must be positive")
    dt = 1.0 / float(fps)
    flat: list[dict[str, float]] = []
    for segment in segments:
        segment.validate()
        count = int(segment.chunks) * controls_per_chunk
        yaw = eased_increments(
            ROTATION_SIGNS[segment.rotation] * math.radians(float(segment.degrees)), count,
        )
        movement = eased_increments(float(segment.distance), count)
        forward_axis, right_axis = MOVEMENT_AXES[segment.movement]
        for yaw_delta, distance_delta in zip(yaw, movement):
            forward_delta = forward_axis * distance_delta
            right_delta = right_axis * distance_delta
            flat.append({
                "delta_time": dt,
                "yaw_delta": yaw_delta,
                "forward": max(forward_delta, 0.0) / dt,
                "backward": max(-forward_delta, 0.0) / dt,
                "strafe_left": max(-right_delta, 0.0) / dt,
                "strafe_right": max(right_delta, 0.0) / dt,
            })
    return [flat[index:index + controls_per_chunk]
            for index in range(0, len(flat), controls_per_chunk)]


def trajectory_document(segments: list[TrajectorySegment], *, fps: float = 24.0) -> dict:
    controls = build_controls(segments, fps=fps)
    return {
        "schema_version": 1,
        "fps": float(fps),
        "controls_per_chunk": 32,
        "segments": [asdict(segment) for segment in segments],
        "controls": controls,
    }
