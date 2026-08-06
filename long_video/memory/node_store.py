"""Atomic, checksummed spatial-node storage and session graph."""
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from ..types import ScaleMetadata, SpatialNode


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
        for key in (
            "view_source", "view_image_confidence", "view_depth_confidence",
            "point_view_mask", "points_rgb_content_origin",
            "points_depth_content_origin", "points_evidence_role",
            "view_rgb_content_origin", "view_depth_content_origin", "view_evidence_role",
            "points_rgb_evidence_role", "points_depth_evidence_role",
            "view_rgb_evidence_role", "view_depth_evidence_role",
        ):
            value = getattr(node, key, None)
            if value is not None:
                arrays[key] = value
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
            "scale":vars(node.scale),
            "model_versions":node.model_versions,
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
        try:
            self._update_session(node)
        except Exception:
            if final.exists(): shutil.rmtree(final)
            if backup is not None and backup.exists(): os.replace(backup, final)
            raise
        if backup is not None:
            shutil.rmtree(backup)

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
            optional = {
                key: arrays[key].copy() if key in arrays.files else None
                for key in (
                    "view_source", "view_image_confidence", "view_depth_confidence",
                    "point_view_mask", "points_rgb_content_origin",
                    "points_depth_content_origin", "points_evidence_role",
                    "view_rgb_content_origin", "view_depth_content_origin", "view_evidence_role",
            "points_rgb_evidence_role", "points_depth_evidence_role",
            "view_rgb_evidence_role", "view_depth_evidence_role",
                )
            }
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
            max(4, metadata.get("schema_version", 1)), metadata.get("quality_metrics", {}),
            view_source=(optional["view_source"] if optional["view_source"] is not None
                         else np.full(values["view_depth"].shape, 4, np.int8)),
            view_image_confidence=(optional["view_image_confidence"]
                         if optional["view_image_confidence"] is not None
                         else np.ones(values["view_depth"].shape, np.float32)),
            view_depth_confidence=(optional["view_depth_confidence"]
                         if optional["view_depth_confidence"] is not None
                         else np.isfinite(values["view_depth"]).astype(np.float32)),
            point_view_mask=optional["point_view_mask"],
            scale=ScaleMetadata(**metadata.get("scale", {})),
            model_versions=metadata.get("model_versions", {}),
            points_rgb_content_origin=optional["points_rgb_content_origin"],
            points_depth_content_origin=optional["points_depth_content_origin"],
            points_evidence_role=optional["points_evidence_role"],
            view_rgb_content_origin=optional["view_rgb_content_origin"],
            view_depth_content_origin=optional["view_depth_content_origin"],
            view_evidence_role=optional["view_evidence_role"],
            points_rgb_evidence_role=optional["points_rgb_evidence_role"],
            points_depth_evidence_role=optional["points_depth_evidence_role"],
            view_rgb_evidence_role=optional["view_rgb_evidence_role"],
            view_depth_evidence_role=optional["view_depth_evidence_role"],
        )
