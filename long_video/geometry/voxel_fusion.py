"""Canonical PointWorld fusion with stable per-voxel appearance anchors."""
from __future__ import annotations
import numpy as np


def _select_anchor_indices(inverse, anchor_confidence, source_locked, voxel_count):
    """Select anchors with the original causal rule in near-linear time."""
    inverse = np.asarray(inverse, np.int64)
    anchor_confidence = np.asarray(anchor_confidence, np.float32)
    source_locked = np.asarray(source_locked, bool)
    observation_indices = np.arange(len(inverse), dtype=np.int64)
    counts = np.bincount(inverse, minlength=voxel_count)

    first = np.full(voxel_count, len(inverse), np.int64)
    np.minimum.at(first, inverse, observation_indices)
    first_locked = np.full(voxel_count, len(inverse), np.int64)
    locked_indices = observation_indices[source_locked]
    if len(locked_indices):
        np.minimum.at(first_locked, inverse[source_locked], locked_indices)

    chosen = first.copy()
    has_locked = first_locked < len(inverse)
    chosen[has_locked] = first_locked[has_locked]
    repeated_unlocked = np.flatnonzero((counts > 1) & ~has_locked)
    if len(repeated_unlocked):
        order = np.argsort(inverse, kind="stable")
        starts = np.empty(voxel_count + 1, np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])
        for voxel_index in repeated_unlocked:
            members = order[starts[voxel_index]:starts[voxel_index + 1]]
            best = members[0]
            for candidate in members[1:]:
                if anchor_confidence[candidate] > 1.1 * anchor_confidence[best]:
                    best = candidate
            chosen[voxel_index] = best
    return chosen


def fuse_voxels(points_xyz, points_rgb, confidence, observation_count=None, voxel_size=0.02,
                *, anchor_confidence=None, anchor_frame=None, source_locked=None, return_anchors=False,
                rgb_mode="anchor"):
    if float(voxel_size) != .02: raise ValueError("persistent PointWorld voxel size is exactly 0.02")
    xyz=np.asarray(points_xyz,np.float32); rgb=np.asarray(points_rgb,np.uint8); conf=np.asarray(confidence,np.float32)
    obs=np.ones(len(xyz),np.int32) if observation_count is None else np.asarray(observation_count,np.int32).clip(1)
    anchor_conf=np.asarray(conf if anchor_confidence is None else anchor_confidence,np.float32)
    frame=np.full(len(xyz),-1,np.int32) if anchor_frame is None else np.asarray(anchor_frame,np.int32)
    locked=np.zeros(len(xyz),bool) if source_locked is None else np.asarray(source_locked,bool)
    valid=np.isfinite(xyz).all(1)&np.isfinite(conf)&(conf>0)&np.isfinite(anchor_conf)
    xyz,rgb,conf,obs,anchor_conf,frame,locked=(x[valid] for x in (xyz,rgb,conf,obs,anchor_conf,frame,locked))
    if not len(xyz):
        empty=(np.empty((0,3),np.float32),np.empty((0,3),np.uint8),np.empty(0,np.float32),np.empty(0,np.uint16),np.empty((0,3),np.int64))
        return (*empty, {"anchor_rgb":np.empty((0,3),np.uint8),"anchor_confidence":np.empty(0,np.float32),"anchor_frame":np.empty(0,np.int32),"source_locked":np.empty(0,bool)}) if return_anchors else empty
    keys=np.floor(xyz/.02).astype(np.int64); unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    weight=conf*obs; total=np.bincount(inverse,weights=weight,minlength=len(unique)).astype(np.float32); count=np.bincount(inverse,weights=obs,minlength=len(unique)).astype(np.int32)
    out_xyz=np.stack([np.bincount(inverse,weights=weight*xyz[:,i],minlength=len(unique))/total for i in range(3)],1).astype(np.float32)
    if rgb_mode == "weighted":
        out_rgb=np.stack([np.bincount(inverse,weights=weight*rgb[:,i],minlength=len(unique))/total for i in range(3)],1)
        result=(out_xyz,np.rint(np.clip(out_rgb,0,255)).astype(np.uint8),(total/count.clip(1)).astype(np.float32),np.minimum(count,65535).astype(np.uint16),unique)
        if return_anchors:
            anchors={"anchor_rgb":result[1].copy(),"anchor_confidence":result[2].copy(),"anchor_frame":np.zeros(len(unique),np.int32),"source_locked":np.ones(len(unique),bool)}
            return (*result,anchors)
        return result
    if rgb_mode != "anchor":
        raise ValueError(f"unsupported voxel RGB fusion mode: {rgb_mode}")
    # Appearance is an anchor selection, deliberately never a weighted RGB average.
    # Preserve the existing anchor unless a later observation exceeds it by
    # the explicit 10% margin. Input order remains causal: persisted fused
    # voxels precede new tail observations.
    chosen = _select_anchor_indices(inverse, anchor_conf, locked, len(unique))
    anchors={"anchor_rgb":rgb[chosen].astype(np.uint8),"anchor_confidence":anchor_conf[chosen].astype(np.float32),"anchor_frame":frame[chosen].astype(np.int32),"source_locked":locked[chosen].astype(bool)}
    result=(out_xyz,rgb[chosen].astype(np.uint8),(total/count.clip(1)).astype(np.float32),np.minimum(count,65535).astype(np.uint16),unique)
    return (*result,anchors) if return_anchors else result
