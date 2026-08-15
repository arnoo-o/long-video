"""Stable immutable identity for PointWorld render/cache ownership."""
from __future__ import annotations
import hashlib
import numpy as np

def point_world_snapshot_identity(node):
    digest=hashlib.sha256()
    digest.update(str(node.node_id).encode())
    digest.update(str(int(getattr(node,'quality_metrics',{}).get('recal3r_world_version',0))).encode())
    digest.update(str(int(getattr(node,'parent_point_count',0) or 0)).encode())
    xyz=np.asarray(node.points_xyz,np.float32)
    for value in (np.floor(xyz/.02).astype(np.int64), np.rint(xyz*1e5).astype(np.int64), xyz,
                  np.asarray(node.points_rgb,np.uint8), np.asarray(node.points_confidence,np.float32)):
        value=np.ascontiguousarray(value); digest.update(str(value.dtype).encode()); digest.update(np.asarray(value.shape,np.int64).tobytes()); digest.update(value.tobytes())
    return (str(node.node_id),int(len(xyz)),digest.hexdigest())
