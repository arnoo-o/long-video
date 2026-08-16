"""Persistent-surface PointWorld ownership for causal ReCal3R observations."""
from __future__ import annotations

from dataclasses import replace
import time
import numpy as np

from ..geometry.point_renderer import render
from ..types import CameraBatch, ScaleMetadata


_NEIGHBORS = tuple((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1))


class ReCal3RWorldAccumulator:
    """Own immutable XYZ surfaces; only confirmed novel surfaces may be added."""

    def __init__(self, backend, initial_node, *, trajectory_id, voxel_size=0.02,
                 novel_min_frames=3, pending_expiry_chunks=8):
        self.backend, self.initial_node = backend, initial_node
        self.trajectory_id, self.voxel_size = str(trajectory_id), float(voxel_size)
        if self.voxel_size != 0.02:
            raise ValueError("ReCal3R world accumulation is fixed to voxel_size=0.02")
        self.novel_min_frames = int(novel_min_frames)
        self.pending_expiry_chunks = int(pending_expiry_chunks)
        self.reset()

    def reset(self):
        self.backend.reset(); node = self.initial_node
        self._xyz = np.asarray(node.points_xyz, np.float32).copy()
        self._rgb = np.asarray(node.points_rgb, np.uint8).copy()
        self._observations = np.asarray(node.observation_count, np.int32).clip(1).copy()
        self._weight = np.asarray(node.points_confidence, np.float32).clip(1e-6) * self._observations
        anchors = getattr(node, "appearance_anchors", {})
        self._anchor_confidence = np.asarray(anchors.get("anchor_confidence", node.points_confidence), np.float32).copy()
        self._anchor_frame = np.asarray(anchors.get("anchor_frame", np.zeros(len(self._xyz))), np.int32).copy()
        self._source_locked = np.asarray(anchors.get("source_locked", np.ones(len(self._xyz), bool)), bool).copy()
        self._source_point_count = len(self._xyz)
        self._seen_frame_ids, self._fused_frame_ids, self._pending_frame_ids = set(), set(), set()
        self._replay_rgb = [np.asarray(node.view_rgb[0], np.uint8).copy()]
        self._replay_c2w = [np.asarray(node.view_c2w[0], np.float32).copy()]
        self._replay_k = [np.asarray(node.view_intrinsics[0], np.float32).copy()]
        self._pending_surfaces, self._association_cache = {}, None
        self._pending_xyz=np.empty((0,3),np.float32); self._pending_rgb=np.empty((0,3),np.uint8)
        self._pending_conf=np.empty(0,np.float32); self._pending_depth=np.empty(0,np.float32)
        self._pending_frame=np.empty(0,np.int32); self._pending_last_chunk=np.empty(0,np.int32)
        self._chunk_serial = self._version = 0
        self._node = replace(node)
        self._build_source_projection(); self._rebuild_ownership(); self._publish()
        self._last_association_masks = []
        self.last_update_metrics = {"world_version": 0, "frames_submitted": 0,
            "frames_fused": 0, "frames_pending": 0, "existing_xyz_moved_count": 0}

    def _build_source_projection(self):
        rgb = np.asarray(self.initial_node.view_rgb[0], np.uint8); h, w = rgb.shape[:2]
        pose = np.asarray(self.initial_node.view_c2w[0], np.float32)
        k = np.asarray(self.initial_node.view_intrinsics[0], np.float32)
        local = (self._xyz[:self._source_point_count] - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]; uv = local @ k.T
        x = np.rint(uv[:, 0] / np.maximum(z, 1e-8)).astype(np.int64)
        y = np.rint(uv[:, 1] / np.maximum(z, 1e-8)).astype(np.int64)
        ok = np.isfinite(local).all(1) & (z > .05) & (z < 100) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        depth = np.full(h * w, np.inf, np.float32); np.minimum.at(depth, y[ok] * w + x[ok], z[ok])
        point_index = np.full(h * w, -1, np.int64); valid_indices = np.flatnonzero(ok)
        if len(valid_indices):
            flat = y[ok] * w + x[ok]; winners = valid_indices[z[ok] <= depth[flat] + 1e-5]
            point_index[y[winners] * w + x[winners]] = winners
        depth[~np.isfinite(depth)] = np.nan
        self._source_depth, self._source_point_index = depth.reshape(h, w), point_index.reshape(h, w)

    def _render_source_w0_depth(self): return self._source_depth.copy()

    def _rebuild_ownership(self):
        self._owned = {tuple(key): index for index, key in enumerate(np.floor(self._xyz / self.voxel_size).astype(np.int64))}

    def _publish(self):
        confidence = self._weight / self._observations.clip(1)
        bmin = self._xyz.min(0) if len(self._xyz) else np.zeros(3, np.float32)
        bmax = self._xyz.max(0) if len(self._xyz) else np.zeros(3, np.float32)
        self._node.points_xyz, self._node.points_rgb = self._xyz.copy(), self._rgb.copy()
        self._node.points_confidence = confidence.astype(np.float32)
        self._node.points_source = np.full(len(self._xyz), 2, np.int8)
        self._node.observation_count = np.minimum(self._observations, 65535).astype(np.uint16)
        self._node.bbox_min, self._node.bbox_max = bmin.astype(np.float32), bmax.astype(np.float32)
        self._node.coverage_radius = float(np.linalg.norm(bmax - bmin) * .5)
        self._node.scale = ScaleMetadata(mode="relative", meters_per_world_unit=None,
            uncertainty=1.0, anchor_source="pi3x_w0_source_geometry_commanded_pose")
        self._node.appearance_anchors = {"anchor_rgb": self._rgb.copy(),
            "anchor_confidence": self._anchor_confidence.copy(), "anchor_frame": self._anchor_frame.copy(),
            "source_locked": self._source_locked.copy()}
        self._node.quality_metrics.update({"recal3r_world_version": self._version,
            "voxel_size": self.voxel_size, "accumulator_points": int(len(self._xyz)),
            "surface_ownership": "immutable_xyz_v1"})

    def prepare_chunk_association(self, warp, cameras, *, render_seconds=0.0):
        """Reuse the one pre-generation WAH render as the W_k association cache."""
        if warp.point_index is None or warp.winning_xyz_world is None:
            raise RuntimeError("association render must expose point_index and winning_xyz_world")
        self._association_cache = {"depth": np.asarray(warp.depth).copy(),
            "visibility": np.asarray(warp.visibility, bool).copy(),
            "point_index": np.asarray(warp.point_index, np.int64).copy(),
            "winning_xyz_world": np.asarray(warp.winning_xyz_world, np.float32).copy(),
            "c2w": np.asarray(cameras.c2w, np.float32).copy(),
            "intrinsics": np.asarray(cameras.intrinsics, np.float32).copy(),
            "seconds": float(render_seconds), "reused_wah_warp": True}

    def _association_for(self, c2w, intrinsics, height, width):
        cache = self._association_cache
        if cache is not None and len(cache["depth"]) in (len(c2w), len(c2w) + 1):
            start = len(cache["depth"]) - len(c2w)
            if np.allclose(cache["c2w"][start:], c2w) and np.allclose(cache["intrinsics"][start:], intrinsics):
                return {key: (value[start:] if isinstance(value, np.ndarray) and value.ndim else value)
                        for key, value in cache.items()}
        started = time.perf_counter()
        warp = render(self._node, CameraBatch(c2w, intrinsics, int(height), int(width)),
                      device="cpu", point_radius=1)
        return {"depth": warp.depth, "visibility": warp.visibility, "point_index": warp.point_index,
            "winning_xyz_world": warp.winning_xyz_world, "c2w": c2w, "intrinsics": intrinsics,
            "seconds": time.perf_counter() - started, "reused_wah_warp": False}

    def _update_owned(self, indices, confidence, colors, frame_index):
        if not len(indices): return 0
        indices, confidence = np.asarray(indices, np.int64), np.asarray(confidence, np.float32)
        order = np.lexsort((-confidence, indices)); sorted_idx = indices[order]
        chosen = order[np.r_[True, sorted_idx[1:] != sorted_idx[:-1]]]
        indices, confidence, colors = indices[chosen], confidence[chosen], np.asarray(colors, np.uint8)[chosen]
        self._weight[indices] += confidence; self._observations[indices] += 1
        change = (~self._source_locked[indices]) & (confidence > 1.1 * self._anchor_confidence[indices])
        target = indices[change]; self._rgb[target] = colors[change]
        self._anchor_confidence[target] = confidence[change]; self._anchor_frame[target] = int(frame_index)
        return int(len(indices))

    def _project_source(self, xyz):
        pose = np.asarray(self.initial_node.view_c2w[0], np.float32)
        k = np.asarray(self.initial_node.view_intrinsics[0], np.float32)
        local = (xyz - pose[:3, 3]) @ pose[:3, :3]; z = local[:, 2]; uv = local @ k.T
        x = np.rint(uv[:, 0] / np.maximum(z, 1e-8)).astype(np.int64)
        y = np.rint(uv[:, 1] / np.maximum(z, 1e-8)).astype(np.int64)
        h, w = self._source_depth.shape
        inside = np.isfinite(local).all(1) & (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        depth = np.full(len(xyz), np.nan, np.float32); index = np.full(len(xyz), -1, np.int64)
        depth[inside], index[inside] = self._source_depth[y[inside], x[inside]], self._source_point_index[y[inside], x[inside]]
        return z, inside, depth, index

    def _pending_candidate(self, xyz, rgb, confidence, depth, frame_index):
        return self._pending_group(np.asarray([xyz]), np.asarray([rgb]), np.asarray([confidence]),
                                   np.asarray([depth]), np.asarray([frame_index]))

    def _pending_group(self, xyz, rgb, confidence, depth, frame_index):
        candidate_voxel_size = 2 * self.voxel_size
        center = np.median(xyz, axis=0)
        key0 = tuple(np.rint(center / candidate_voxel_size).astype(np.int64))
        tolerance = max(2*self.voxel_size, .02*float(np.median(depth)))
        found = None
        for delta in _NEIGHBORS:
            key = tuple(key0[i] + delta[i] for i in range(3)); candidate = self._pending_surfaces.get(key)
            if candidate is not None:
                old_center = np.median(np.stack([value[0] for value in candidate["supports"].values()]), 0)
                if np.linalg.norm(old_center - center) <= tolerance: found = candidate; break
        if found is None:
            found = {"key": key0, "supports": {}, "last_chunk": self._chunk_serial}
            self._pending_surfaces[key0] = found
        for point, color, conf, point_depth, frame in zip(xyz, rgb, confidence, depth, frame_index):
            old = found["supports"].get(int(frame))
            if old is None or conf > old[2]:
                found["supports"][int(frame)] = (point.copy(), color.copy(), float(conf), float(point_depth))
        found["last_chunk"] = self._chunk_serial
        return found

    def _owned_neighbor(self, xyz, depth):
        key0 = tuple(np.floor(xyz/self.voxel_size).astype(np.int64)); tolerance=max(2*self.voxel_size,.02*float(depth))
        for delta in _NEIGHBORS:
            index = self._owned.get(tuple(key0[i]+delta[i] for i in range(3)))
            if index is not None and np.linalg.norm(self._xyz[index]-xyz) <= tolerance: return index
        return None

    def update_frame(self, rgb, c2w, intrinsics, global_frame_index):
        return self.update_chunk(np.asarray(rgb)[None], np.asarray(c2w)[None], np.asarray(intrinsics)[None], [global_frame_index])

    def update_chunk(self, rgb, c2w, intrinsics, global_frame_indices):
        indices=[int(value) for value in global_frame_indices]; identities=[(self.trajectory_id,index) for index in indices]
        if any(identity in self._seen_frame_ids for identity in identities): raise RuntimeError("ReCal3R frame processed twice")
        if not indices or indices[0] != len(self._replay_rgb): raise RuntimeError("ReCal chunk must append after previous unique frame")
        rgb,c2w,intrinsics=np.asarray(rgb),np.asarray(c2w),np.asarray(intrinsics)
        self._replay_rgb.extend(np.asarray(x,np.uint8).copy() for x in rgb)
        self._replay_c2w.extend(np.asarray(x,np.float32).copy() for x in c2w)
        self._replay_k.extend(np.asarray(x,np.float32).copy() for x in intrinsics)
        before_xyz=self._xyz.copy(); before_points=len(self._xyz); before_obs=int(self._observations.sum())
        replay=self.backend.replay_prefix(np.stack(self._replay_rgb),np.stack(self._replay_c2w),np.stack(self._replay_k),
            trajectory_id=self.trajectory_id,global_frame_indices=range(len(self._replay_rgb)))
        if self.backend.get_state().get("alignment",{}).get("status") != "locked":
            replay=self.backend.lock_source_geometry_alignment(self._render_source_w0_depth())
        association=self._association_for(c2w,intrinsics,rgb.shape[1],rgb.shape[2])
        names=("association_match_pixels","association_conflict_pixels","association_novel_pixels",
            "source_free_space_rejected","source_surface_duplicate_pixels","matched_world_points",
            "novel_confirmed_points","novel_expired_count")
        metrics={name:0 for name in names}; residuals=[]; masks=[]; confirmed=[]; fused_frames=[]; pending_batches=[]
        candidate_started=time.perf_counter(); candidates=sorted(self._pending_frame_ids|set(indices))
        for index in candidates:
            if index==0 or index in self._fused_frame_ids: continue
            if index not in indices: self._pending_frame_ids.add(index); continue
            slot=indices.index(index); prediction=replay[index]
            xyz=np.asarray(prediction.point_maps[0],np.float32); conf=np.asarray(prediction.geometry_confidence[0],np.float32)
            image=np.asarray(self._replay_rgb[index],np.uint8); pose=np.asarray(c2w[slot],np.float32)
            recal_depth=((xyz-pose[:3,3])@pose[:3,:3])[...,2]
            valid=np.isfinite(xyz).all(-1)&np.isfinite(conf)&(conf>0)&np.isfinite(recal_depth)&(recal_depth>0)
            world_depth=np.asarray(association["depth"][slot]); visible=np.asarray(association["visibility"][slot],bool)
            point_index=np.asarray(association["point_index"][slot],np.int64)
            tolerance=np.maximum(2*self.voxel_size,.02*np.minimum(recal_depth,world_depth)); residual=np.abs(recal_depth-world_depth)
            match=valid&visible&np.isfinite(world_depth)&(residual<=tolerance)&(point_index>=0)
            conflict=valid&visible&~match; novel=valid&~visible
            metrics["association_match_pixels"]+=int(match.sum()); metrics["association_conflict_pixels"]+=int(conflict.sum())
            metrics["association_novel_pixels"]+=int(novel.sum()); residuals.extend(residual[match].tolist())
            metrics["matched_world_points"]+=self._update_owned(point_index[match],conf[match],image[match],index)
            mask=np.zeros((*valid.shape,3),np.uint8); mask[match]=(0,255,0); mask[conflict]=(255,0,0)
            ny,nx=np.nonzero(novel)
            if len(ny):
                nxyz,nconf,nrgb,ndepth=xyz[ny,nx],conf[ny,nx],image[ny,nx],recal_depth[ny,nx]
                source_z,inside,source_depth,source_idx=self._project_source(nxyz)
                source_valid=inside&np.isfinite(source_depth); free_tol=np.maximum(3*self.voxel_size,.03*source_depth)
                violation=source_valid&(source_z<source_depth-free_tol)
                duplicate=source_valid&(np.abs(source_z-source_depth)<=free_tol)&(source_idx>=0)
                metrics["source_free_space_rejected"]+=int(violation.sum()); metrics["source_surface_duplicate_pixels"]+=int(duplicate.sum())
                mask[ny[violation],nx[violation]]=(255,0,255); mask[ny[~violation],nx[~violation]]=(0,0,255)
                metrics["matched_world_points"]+=self._update_owned(source_idx[duplicate],nconf[duplicate],nrgb[duplicate],index)
                pending=~violation&~duplicate
                if pending.any():
                    pxyz,pconf,prgb,pdepth=nxyz[pending],nconf[pending],nrgb[pending],ndepth[pending]
                    pos=np.flatnonzero(pending)
                    pending_batches.append((pxyz,pconf,prgb,pdepth,
                        np.full(len(pxyz),index,np.int32),np.full(len(pxyz),len(masks),np.int32),ny[pos],nx[pos]))
            masks.append(mask); fused_frames.append(index); self._fused_frame_ids.add(index); self._pending_frame_ids.discard(index)
        confirmed_xyz=np.empty((0,3),np.float32); confirmed_rgb=np.empty((0,3),np.uint8)
        confirmed_conf_sum=np.empty(0,np.float32); confirmed_count=np.empty(0,np.int32)
        confirmed_best_conf=np.empty(0,np.float32); confirmed_best_depth=np.empty(0,np.float32)
        confirmed_last_frame=np.empty(0,np.int32)
        if pending_batches:
            px,pconf,prgb,pdepth,pframe,pslot,py,px_pixel = (
                np.concatenate(values) for values in zip(*pending_batches)
            )
            candidate_keys=np.rint(px/(2*self.voxel_size)).astype(np.int64)
            order=np.lexsort((-pconf,pframe,candidate_keys[:,2],candidate_keys[:,1],candidate_keys[:,0]))
            group_values=np.column_stack([candidate_keys,pframe])[order]
            first=np.r_[True,np.any(group_values[1:]!=group_values[:-1],axis=1)]
            selected=order[first]
            current_xyz,current_rgb,current_conf,current_depth,current_frame=(
                values[selected] for values in (px,prgb,pconf,pdepth,pframe)
            )
            all_xyz=np.concatenate([self._pending_xyz,current_xyz]); all_rgb=np.concatenate([self._pending_rgb,current_rgb])
            all_conf=np.concatenate([self._pending_conf,current_conf]); all_depth=np.concatenate([self._pending_depth,current_depth])
            all_frame=np.concatenate([self._pending_frame,current_frame]); all_last=np.concatenate([
                self._pending_last_chunk,np.full(len(current_xyz),self._chunk_serial,np.int32)])
            all_keys=np.rint(all_xyz/(2*self.voxel_size)).astype(np.int64)
            order=np.lexsort((-all_conf,all_frame,all_keys[:,2],all_keys[:,1],all_keys[:,0]))
            pairs=np.column_stack([all_keys,all_frame])[order]
            unique_support=np.r_[True,np.any(pairs[1:]!=pairs[:-1],axis=1)]
            support_indices=order[unique_support]
            support_keys=all_keys[support_indices]
            key_order=np.lexsort((support_keys[:,2],support_keys[:,1],support_keys[:,0]))
            support_indices=support_indices[key_order]; support_keys=support_keys[key_order]
            unique_keys,inverse=np.unique(support_keys,axis=0,return_inverse=True)
            counts=np.bincount(inverse,minlength=len(unique_keys)); starts=np.r_[0,np.cumsum(counts)]
            ranks=np.arange(len(support_indices))-np.repeat(starts[:-1],counts)
            first_three=ranks<3; grid=np.full((len(unique_keys),3,3),np.nan,np.float32)
            grid[inverse[first_three],ranks[first_three]]=all_xyz[support_indices[first_three]]
            canonical=np.nanmedian(grid,axis=1).astype(np.float32); confirmed_group=counts>=self.novel_min_frames
            confidence_order=np.lexsort((-all_conf[support_indices],inverse))
            sorted_groups=inverse[confidence_order]; best_first=np.r_[True,sorted_groups[1:]!=sorted_groups[:-1]]
            best_indices=support_indices[confidence_order[best_first]]
            best_by_group=np.empty(len(unique_keys),np.int64); best_by_group[inverse[confidence_order[best_first]]]=best_indices
            confirmed_indices=np.flatnonzero(confirmed_group); confirmed_best=best_by_group[confirmed_indices]
            confidence_sum=np.bincount(inverse,weights=all_conf[support_indices],minlength=len(unique_keys)).astype(np.float32)
            last_frame=np.full(len(unique_keys),-1,np.int32)
            np.maximum.at(last_frame,inverse,all_frame[support_indices])
            confirmed_xyz=canonical[confirmed_indices]; confirmed_rgb=all_rgb[confirmed_best]
            confirmed_conf_sum=confidence_sum[confirmed_indices]; confirmed_count=counts[confirmed_indices].astype(np.int32)
            confirmed_best_conf=all_conf[confirmed_best]; confirmed_best_depth=all_depth[confirmed_best]
            confirmed_last_frame=last_frame[confirmed_indices]
            combined=np.concatenate([unique_keys,candidate_keys])
            _,combined_inverse=np.unique(combined,axis=0,return_inverse=True)
            group_lookup=np.empty(int(combined_inverse.max())+1,np.int64)
            group_lookup[combined_inverse[:len(unique_keys)]]=np.arange(len(unique_keys))
            current_groups=group_lookup[combined_inverse[len(unique_keys):]]
            current_confirmed=confirmed_group[current_groups]
            for slot in np.unique(pslot[current_confirmed]):
                marked=current_confirmed&(pslot==slot)
                masks[int(slot)][py[marked],px_pixel[marked]]=(0,255,255)
            pending_group=~confirmed_group[inverse]
            kept=support_indices[pending_group]
            not_expired=(self._chunk_serial-all_last[kept])<self.pending_expiry_chunks
            metrics["novel_expired_count"]=int((~not_expired).sum())
            kept=kept[not_expired]
            self._pending_xyz,self._pending_rgb,self._pending_conf=all_xyz[kept],all_rgb[kept],all_conf[kept]
            self._pending_depth,self._pending_frame,self._pending_last_chunk=all_depth[kept],all_frame[kept],all_last[kept]
        if len(confirmed_xyz):
            from scipy.spatial import cKDTree
            tree=cKDTree(self._xyz); maximum_tolerance=float(np.max(np.maximum(2*self.voxel_size,.02*confirmed_best_depth)))
            distance,nearest=tree.query(confirmed_xyz,k=1,distance_upper_bound=maximum_tolerance,workers=-1)
            has_nearest=nearest<len(self._xyz); voxel_neighbor=np.zeros(len(confirmed_xyz),bool)
            if has_nearest.any():
                candidate_keys=np.floor(confirmed_xyz[has_nearest]/self.voxel_size).astype(np.int64)
                owned_keys=np.floor(self._xyz[nearest[has_nearest]]/self.voxel_size).astype(np.int64)
                voxel_neighbor[has_nearest]=np.max(np.abs(candidate_keys-owned_keys),axis=1)<=1
            tolerance=np.maximum(2*self.voxel_size,.02*confirmed_best_depth)
            duplicate=has_nearest&voxel_neighbor&(distance<=tolerance); novel=~duplicate
            xyz_batch=confirmed_xyz[novel]; start=len(self._xyz)
            self._xyz=np.concatenate([self._xyz,xyz_batch]); self._rgb=np.concatenate([self._rgb,confirmed_rgb[novel]])
            self._observations=np.concatenate([self._observations,confirmed_count[novel]])
            self._weight=np.concatenate([self._weight,confirmed_conf_sum[novel]])
            self._anchor_confidence=np.concatenate([self._anchor_confidence,confirmed_best_conf[novel]])
            self._anchor_frame=np.concatenate([self._anchor_frame,confirmed_last_frame[novel]])
            self._source_locked=np.concatenate([self._source_locked,np.zeros(int(novel.sum()),bool)])
            metrics["novel_confirmed_points"]=int(novel.sum())
            for offset,key in enumerate(np.floor(xyz_batch/self.voxel_size).astype(np.int64)):
                self._owned[tuple(key)]=start+offset
        self._seen_frame_ids.update(identities)
        self._version+=1; self._chunk_serial+=1; self._publish(); self._last_association_masks=masks; self._association_cache=None
        residuals=np.asarray(residuals,np.float32)
        self.last_update_metrics={"world_version":self._version,"frames_submitted":len(indices),"frames_fused":len(fused_frames),
            "fused_frame_indices":fused_frames,"frames_pending":len(self._pending_frame_ids),"pending_frame_indices":sorted(self._pending_frame_ids),
            "world_point_count_before":before_points,"world_point_count_after":len(self._xyz),
            "observation_count_before":before_obs,"observation_count_after":int(self._observations.sum()),**metrics,
            "novel_pending_count":int(len(np.unique(np.rint(self._pending_xyz/(2*self.voxel_size)).astype(np.int64),axis=0))),"match_depth_residual_median":float(np.median(residuals)) if len(residuals) else None,
            "match_depth_residual_p95":float(np.percentile(residuals,95)) if len(residuals) else None,
            "existing_xyz_moved_count":int(np.count_nonzero(np.any(self._xyz[:len(before_xyz)]!=before_xyz,axis=1))),
            "association_render_seconds":float(association["seconds"]),"candidate_processing_seconds":time.perf_counter()-candidate_started,
            "association_reused_wah_warp":bool(association["reused_wah_warp"]),"alignment":self.backend.get_state().get("alignment",{})}
        if self.last_update_metrics["existing_xyz_moved_count"] != 0: raise RuntimeError("persistent surface ownership moved existing XYZ")
        return self._node

    def get_point_world(self): return self._node
    def association_debug_masks(self): return [mask.copy() for mask in self._last_association_masks]
    def debug_geometry_for_frames(self, global_frame_indices):
        return [self.backend.raw_recal_debug(self.trajectory_id,int(index)) for index in global_frame_indices]
