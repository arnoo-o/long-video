"""Persistent-surface PointWorld ownership for causal ReCal3R observations."""
from __future__ import annotations

from dataclasses import replace
import time
import numpy as np

from ..geometry.point_renderer import render
from ..types import CameraBatch, ScaleMetadata


ONLINE_FUSION_VOXEL_SIZE = 0.05
MATCH_BASE_TOLERANCE = 0.04
SOURCE_FREE_SPACE_BASE_TOLERANCE = 0.06


class ReCal3RWorldAccumulator:
    """Own immutable XYZ surfaces and commit valid novel voxels once per chunk."""

    def __init__(self, backend, initial_node, *, trajectory_id,
                 voxel_size=ONLINE_FUSION_VOXEL_SIZE,
                 match_base_tolerance=MATCH_BASE_TOLERANCE,
                 source_free_space_base_tolerance=SOURCE_FREE_SPACE_BASE_TOLERANCE):
        self.backend, self.initial_node = backend, initial_node
        self.trajectory_id, self.voxel_size = str(trajectory_id), float(voxel_size)
        self.match_base_tolerance = float(match_base_tolerance)
        self.source_free_space_base_tolerance = float(source_free_space_base_tolerance)
        if self.voxel_size != ONLINE_FUSION_VOXEL_SIZE:
            raise ValueError(f"online ReCal3R fusion is fixed to voxel_size={ONLINE_FUSION_VOXEL_SIZE}")
        if self.match_base_tolerance != MATCH_BASE_TOLERANCE:
            raise ValueError(f"MATCH base tolerance is fixed to {MATCH_BASE_TOLERANCE}")
        if self.source_free_space_base_tolerance != SOURCE_FREE_SPACE_BASE_TOLERANCE:
            raise ValueError(f"source free-space base tolerance is fixed to {SOURCE_FREE_SPACE_BASE_TOLERANCE}")
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
        self._association_cache = None
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
            "match_base_tolerance": self.match_base_tolerance,
            "source_free_space_base_tolerance": self.source_free_space_base_tolerance,
            "surface_ownership": "immutable_xyz_chunk_local_immediate_commit_first_chunk_nonconflicting_v4"})

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

    @staticmethod
    def _row_keys(values):
        values = np.ascontiguousarray(values, dtype=np.int64)
        return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).reshape(-1)

    def _owned_indices_for_keys(self, keys):
        """Batch exact-voxel ownership lookup without per-point Python loops."""
        keys = np.asarray(keys, np.int64)
        result = np.full(len(keys), -1, np.int64)
        if not len(keys) or not len(self._xyz):
            return result
        owned_keys = np.floor(self._xyz / self.voxel_size).astype(np.int64)
        owned_rows, query_rows = self._row_keys(owned_keys), self._row_keys(keys)
        order = np.argsort(owned_rows)
        sorted_rows = owned_rows[order]
        positions = np.searchsorted(sorted_rows, query_rows)
        inside = positions < len(sorted_rows)
        matched = np.zeros(len(keys), bool)
        matched[inside] = sorted_rows[positions[inside]] == query_rows[inside]
        result[matched] = order[positions[matched]]
        return result

    @staticmethod
    def _grouped_median(keys, values):
        """Vectorized coordinate medians for rows grouped by integer voxel key."""
        keys, values = np.asarray(keys, np.int64), np.asarray(values, np.float32)
        unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        starts = np.r_[0, np.cumsum(counts[:-1])]
        lower = starts + (counts - 1) // 2
        upper = starts + counts // 2
        median = np.empty((len(unique_keys), values.shape[1]), np.float32)
        for axis in range(values.shape[1]):
            order = np.lexsort((values[:, axis], inverse))
            sorted_values = values[order, axis]
            median[:, axis] = (sorted_values[lower] + sorted_values[upper]) * .5
        return unique_keys, inverse, counts.astype(np.int32), median

    @staticmethod
    def _local_association(world_depth, visibility, point_index, recal_depth):
        """Find the closest-depth owned projection in a 3x3 pixel neighborhood."""
        world_depth = np.asarray(world_depth, np.float32)
        visibility = np.asarray(visibility, bool)
        point_index = np.asarray(point_index, np.int64)
        height, width = world_depth.shape
        best_residual = np.full((height, width), np.inf, np.float32)
        best_depth = np.full((height, width), np.nan, np.float32)
        best_index = np.full((height, width), -1, np.int64)
        for dy in (-1, 0, 1):
            dst_y = slice(max(0, -dy), min(height, height - dy))
            src_y = slice(max(0, dy), min(height, height + dy))
            for dx in (-1, 0, 1):
                dst_x = slice(max(0, -dx), min(width, width - dx))
                src_x = slice(max(0, dx), min(width, width + dx))
                depth = world_depth[src_y, src_x]
                valid = visibility[src_y, src_x] & np.isfinite(depth) & (point_index[src_y, src_x] >= 0)
                residual = np.abs(recal_depth[dst_y, dst_x] - depth)
                update = valid & (residual < best_residual[dst_y, dst_x])
                if update.any():
                    target_residual = best_residual[dst_y, dst_x]
                    target_depth = best_depth[dst_y, dst_x]
                    target_index = best_index[dst_y, dst_x]
                    target_residual[update] = residual[update]
                    target_depth[update] = depth[update]
                    target_index[update] = point_index[src_y, src_x][update]
        return np.isfinite(best_depth), best_depth, best_index, best_residual

    def update_frame(self, rgb, c2w, intrinsics, global_frame_index):
        return self.update_chunk(np.asarray(rgb)[None], np.asarray(c2w)[None], np.asarray(intrinsics)[None], [global_frame_index])

    def update_chunk(self, rgb, c2w, intrinsics, global_frame_indices):
        indices = [int(value) for value in global_frame_indices]
        identities = [(self.trajectory_id, index) for index in indices]
        if any(identity in self._seen_frame_ids for identity in identities):
            raise RuntimeError("ReCal3R frame processed twice")
        if not indices or indices[0] != len(self._replay_rgb):
            raise RuntimeError("ReCal chunk must append after previous unique frame")
        rgb, c2w, intrinsics = np.asarray(rgb), np.asarray(c2w), np.asarray(intrinsics)
        self._replay_rgb.extend(np.asarray(value, np.uint8).copy() for value in rgb)
        self._replay_c2w.extend(np.asarray(value, np.float32).copy() for value in c2w)
        self._replay_k.extend(np.asarray(value, np.float32).copy() for value in intrinsics)
        before_xyz = self._xyz.copy(); before_points = len(self._xyz); before_obs = int(self._observations.sum())
        replay = self.backend.replay_prefix(
            np.stack(self._replay_rgb), np.stack(self._replay_c2w), np.stack(self._replay_k),
            trajectory_id=self.trajectory_id, global_frame_indices=range(len(self._replay_rgb)),
        )
        if self.backend.get_state().get("alignment", {}).get("status") != "locked":
            replay = self.backend.lock_source_geometry_alignment(self._render_source_w0_depth())
        association = self._association_for(c2w, intrinsics, rgb.shape[1], rgb.shape[2])
        names = ("association_match_pixels", "association_conflict_pixels", "association_novel_pixels",
            "source_free_space_rejected", "source_surface_duplicate_pixels", "matched_world_points",
            "novel_committed_points")
        metrics = {name: 0 for name in names}; residuals = []; masks = []; fused_frames = []; novel_batches = []
        for slot, index in enumerate(indices):
            if index == 0 or index in self._fused_frame_ids:
                continue
            prediction = replay[index]
            xyz = np.asarray(prediction.point_maps[0], np.float32)
            conf = np.asarray(prediction.geometry_confidence[0], np.float32)
            image = np.asarray(self._replay_rgb[index], np.uint8)
            pose = np.asarray(c2w[slot], np.float32)
            recal_depth = ((xyz - pose[:3, 3]) @ pose[:3, :3])[..., 2]
            valid = np.isfinite(xyz).all(-1) & np.isfinite(conf) & (conf > 0) & np.isfinite(recal_depth) & (recal_depth > 0)
            has_world, world_depth, point_index, residual = self._local_association(
                association["depth"][slot], association["visibility"][slot],
                association["point_index"][slot], recal_depth,
            )
            tolerance = np.maximum(self.match_base_tolerance, .02 * np.minimum(recal_depth, world_depth))
            match = valid & has_world & (residual <= tolerance)
            # W0 is a source-only Pi3X initialization. During the first ReCal
            # chunk it is an alignment/free-space anchor, not a conflict veto:
            # unmatched ReCal surfaces must be allowed through the existing
            # source free-space test and chunk-local ownership commit. Once W1
            # is published, normal persistent-surface conflict rejection starts.
            initial_recal_chunk = self._chunk_serial == 0
            conflict = valid & has_world & ~match & (not initial_recal_chunk)
            novel = valid & (~has_world | (has_world & ~match & initial_recal_chunk))
            metrics["association_match_pixels"] += int(match.sum())
            metrics["association_conflict_pixels"] += int(conflict.sum())
            metrics["association_novel_pixels"] += int(novel.sum())
            residuals.extend(residual[match].tolist())
            metrics["matched_world_points"] += self._update_owned(point_index[match], conf[match], image[match], index)
            mask = np.zeros((*valid.shape, 3), np.uint8)
            mask[match] = (0, 255, 0); mask[conflict] = (255, 0, 0)
            ny, nx = np.nonzero(novel)
            if len(ny):
                nxyz, nconf, nrgb = xyz[ny, nx], conf[ny, nx], image[ny, nx]
                source_z, inside, source_depth, source_idx = self._project_source(nxyz)
                source_valid = inside & np.isfinite(source_depth)
                free_tol = np.maximum(self.source_free_space_base_tolerance, .03 * source_depth)
                violation = source_valid & (source_z < source_depth - free_tol)
                duplicate = source_valid & (np.abs(source_z - source_depth) <= free_tol) & (source_idx >= 0)
                metrics["source_free_space_rejected"] += int(violation.sum())
                metrics["source_surface_duplicate_pixels"] += int(duplicate.sum())
                mask[ny[violation], nx[violation]] = (255, 0, 255)
                mask[ny[duplicate], nx[duplicate]] = (0, 255, 0)
                metrics["matched_world_points"] += self._update_owned(source_idx[duplicate], nconf[duplicate], nrgb[duplicate], index)
                accepted = ~violation & ~duplicate
                mask[ny[accepted], nx[accepted]] = (0, 0, 255)
                if accepted.any():
                    novel_batches.append((nxyz[accepted], nrgb[accepted], nconf[accepted],
                        np.full(int(accepted.sum()), index, np.int32),
                        np.full(int(accepted.sum()), len(masks), np.int16), ny[accepted], nx[accepted]))
            masks.append(mask)
            fused_frames.append(index); self._fused_frame_ids.add(index); self._pending_frame_ids.discard(index)

        novel_fusion_started = time.perf_counter()
        if novel_batches:
            points, colors, confidence, frame_ids, mask_slots, pixel_y, pixel_x = (
                np.concatenate(values) for values in zip(*novel_batches)
            )
            voxel_keys = np.floor(points / self.voxel_size).astype(np.int64)
            unique_keys, inverse, raw_counts, canonical_xyz = self._grouped_median(voxel_keys, points)
            confidence_sum = np.bincount(inverse, weights=confidence, minlength=len(unique_keys)).astype(np.float32)
            confidence_mean = confidence_sum / raw_counts
            confidence_order = np.lexsort((-confidence, inverse))
            sorted_groups = inverse[confidence_order]
            first = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
            best_indices = confidence_order[first]
            best_by_group = np.empty(len(unique_keys), np.int64)
            best_by_group[inverse[best_indices]] = best_indices
            pairs = np.column_stack([inverse, frame_ids])
            observation_count = np.unique(pairs, axis=0, return_counts=False)
            distinct_count = np.bincount(observation_count[:, 0], minlength=len(unique_keys)).astype(np.int32)
            anchor_frame = np.full(len(unique_keys), -1, np.int32)
            np.maximum.at(anchor_frame, inverse, frame_ids)
            owner = self._owned_indices_for_keys(unique_keys)
            duplicate_group = owner >= 0
            if duplicate_group.any():
                duplicate_observation = duplicate_group[inverse]
                duplicate_indices = owner[inverse[duplicate_observation]]
                duplicate_frames = frame_ids[duplicate_observation]
                duplicate_conf = confidence[duplicate_observation]
                duplicate_rgb = colors[duplicate_observation]
                for frame_index in np.unique(duplicate_frames):
                    selected = duplicate_frames == frame_index
                    metrics["matched_world_points"] += self._update_owned(
                        duplicate_indices[selected], duplicate_conf[selected], duplicate_rgb[selected], int(frame_index))
            novel_group = ~duplicate_group
            committed = int(novel_group.sum())
            if committed:
                best = best_by_group[novel_group]
                self._xyz = np.concatenate([self._xyz, canonical_xyz[novel_group]])
                self._rgb = np.concatenate([self._rgb, colors[best]])
                self._observations = np.concatenate([self._observations, distinct_count[novel_group]])
                self._weight = np.concatenate([self._weight, confidence_mean[novel_group] * distinct_count[novel_group]])
                self._anchor_confidence = np.concatenate([self._anchor_confidence, confidence[best]])
                self._anchor_frame = np.concatenate([self._anchor_frame, anchor_frame[novel_group]])
                self._source_locked = np.concatenate([self._source_locked, np.zeros(committed, bool)])
            metrics["novel_committed_points"] = committed
            committed_observation = novel_group[inverse]
            duplicate_observation = duplicate_group[inverse]
            for mask_slot in np.unique(mask_slots):
                selected = mask_slots == mask_slot
                masks[int(mask_slot)][pixel_y[selected & committed_observation], pixel_x[selected & committed_observation]] = (0, 255, 255)
                masks[int(mask_slot)][pixel_y[selected & duplicate_observation], pixel_x[selected & duplicate_observation]] = (0, 255, 0)
        novel_fusion_seconds = time.perf_counter() - novel_fusion_started
        self._seen_frame_ids.update(identities)
        self._version += 1; self._chunk_serial += 1; self._rebuild_ownership(); self._publish()
        self._last_association_masks = masks; self._association_cache = None
        residuals = np.asarray(residuals, np.float32)
        self.last_update_metrics = {"world_version": self._version, "frames_submitted": len(indices),
            "frames_fused": len(fused_frames), "fused_frame_indices": fused_frames,
            "frames_pending": len(self._pending_frame_ids), "pending_frame_indices": sorted(self._pending_frame_ids),
            "world_point_count_before": before_points, "world_point_count_after": len(self._xyz),
            "observation_count_before": before_obs, "observation_count_after": int(self._observations.sum()), **metrics,
            "match_depth_residual_median": float(np.median(residuals)) if len(residuals) else None,
            "match_depth_residual_p95": float(np.percentile(residuals, 95)) if len(residuals) else None,
            "existing_xyz_moved_count": int(np.count_nonzero(np.any(self._xyz[:len(before_xyz)] != before_xyz, axis=1))),
            "association_render_seconds": float(association["seconds"]), "novel_fusion_seconds": novel_fusion_seconds,
            "association_reused_wah_warp": bool(association["reused_wah_warp"]),
            "alignment": self.backend.get_state().get("alignment", {})}
        if self.last_update_metrics["existing_xyz_moved_count"] != 0:
            raise RuntimeError("persistent surface ownership moved existing XYZ")
        return self._node

    def get_point_world(self): return self._node
    def association_debug_masks(self): return [mask.copy() for mask in self._last_association_masks]
    def debug_geometry_for_frames(self, global_frame_indices):
        return [self.backend.raw_recal_debug(self.trajectory_id,int(index)) for index in global_frame_indices]
