"""Atomic, checksummed spatial-node storage and session graph."""
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from ..types import SpatialNode


class NodeStore:
    def __init__(self, root):
        self.root = Path(root)

    def _update_session(self, node):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "session.json"
        session = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "schema_version": 1, "nodes": {}, "edges": []
        }
        session["nodes"][node.node_id] = {
            "status": node.status,
            "parent_id": node.parent_id,
            "created_frame": int(node.created_frame),
            "coverage_radius": float(node.coverage_radius),
        }
        if node.parent_id is not None:
            edge = {"parent_id": node.parent_id, "child_id": node.node_id}
            if edge not in session["edges"]:
                session["edges"].append(edge)
        temporary = self.root / f".session.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(session, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def save(self, node):
        final = self.root / "nodes" / node.node_id
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=node.node_id + ".", dir=final.parent))
        arrays = {
            key: getattr(node, key)
            for key in (
                "view_rgb", "view_depth", "view_c2w", "view_intrinsics",
                "points_xyz", "points_rgb", "points_confidence", "points_source",
                "observation_count",
            )
        }
        if node.points_normal is not None:
            arrays["points_normal"] = node.points_normal
        array_path = temporary / "node_arrays.npz"
        np.savez_compressed(array_path, **arrays)
        digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": node.schema_version,
            "node_id": node.node_id,
            "status": node.status,
            "parent_id": node.parent_id,
            "created_frame": node.created_frame,
            "coverage_radius": node.coverage_radius,
            "center_c2w": node.center_c2w.tolist(),
            "bbox_min": node.bbox_min.tolist(),
            "bbox_max": node.bbox_max.tolist(),
            "depth_convention": node.depth_convention,
            "quality_metrics": node.quality_metrics,
            "node_arrays_sha256": digest,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        backup = None
        if final.exists():
            backup = final.with_name(f".{final.name}.backup.{uuid.uuid4().hex}")
            os.replace(final, backup)
        try:
            os.replace(temporary, final)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, final)
            raise
        else:
            if backup is not None:
                shutil.rmtree(backup)
        self._update_session(node)

    def load(self, node_id):
        path = self.root / "nodes" / node_id
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        blob = (path / "node_arrays.npz").read_bytes()
        if hashlib.sha256(blob).hexdigest() != metadata["node_arrays_sha256"]:
            raise IOError(f"Checksum mismatch for node {node_id}")
        with np.load(path / "node_arrays.npz") as arrays:
            values = {
                key: arrays[key].copy()
                for key in (
                    "view_rgb", "view_depth", "view_c2w", "view_intrinsics",
                    "points_xyz", "points_rgb", "points_confidence", "points_source",
                    "observation_count",
                )
            }
            normal = arrays["points_normal"].copy() if "points_normal" in arrays.files else None
        return SpatialNode(
            metadata["node_id"], metadata["status"], metadata.get("parent_id"),
            np.asarray(metadata["center_c2w"], np.float32), metadata["created_frame"],
            metadata["coverage_radius"], np.asarray(metadata["bbox_min"], np.float32),
            np.asarray(metadata["bbox_max"], np.float32),
            values["view_rgb"], values["view_depth"], values["view_c2w"],
            values["view_intrinsics"], values["points_xyz"], values["points_rgb"],
            values["points_confidence"], values["points_source"],
            values["observation_count"], normal,
            metadata.get("depth_convention", "RAY_DISTANCE"),
            metadata.get("schema_version", 1), metadata.get("quality_metrics", {}),
        )
