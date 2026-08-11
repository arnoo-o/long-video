#!/usr/bin/env python3
"""Build the 256-scene DL3DV Stage0-FiLM corpus from qualified official scenes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from long_video.data.dl3dv import (CHUNK_STRIDE, TARGET_FPS, TARGET_HW,
    center_crop_resize_geometry, chunk_real_indices, load_dl3dv_scene,
    select_revisit_trajectories, source_relative_opencv_c2w, validate_trajectory_record)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def assign_splits(records):
    """Exactly 224/16/16 with eight indoor and eight outdoor eval scenes."""
    by_env = {env: sorted((x for x in records if x["environment"] == env),
                          key=lambda x: x["scene_hash"]) for env in ("indoor", "outdoor")}
    if any(len(values) != 128 for values in by_env.values()):
        raise ValueError("qualified state must contain exactly 128 indoor and 128 outdoor scenes")
    result = {}
    for values in by_env.values():
        for item in values[:8]: result[item["scene_hash"]] = "diagnostic"
        for item in values[8:16]: result[item["scene_hash"]] = "val"
        for item in values[16:]: result[item["scene_hash"]] = "train"
    return result


def resize_frame(path, crop_box):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB").crop(crop_box).resize((TARGET_HW[1], TARGET_HW[0]), Image.Resampling.LANCZOS)


def make_visualization(rgb_dir, poses, destination):
    from PIL import Image, ImageDraw
    files = sorted(rgb_dir.glob("*.jpg"))
    picks = np.linspace(0, len(files) - 1, 8).round().astype(int)
    thumb_w, thumb_h = 240, 144
    canvas = Image.new("RGB", (4 * thumb_w, 3 * thumb_h), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(picks):
        with Image.open(files[index]) as frame:
            canvas.paste(frame.convert("RGB").resize((thumb_w, thumb_h)),
                         ((slot % 4) * thumb_w, (slot // 4) * thumb_h))
    xy = poses[:, [0, 2], 3]; lo, hi = xy.min(0), xy.max(0); span = np.maximum(hi - lo, 1e-6)
    points = (xy - lo) / span
    ox, oy = 0, 2 * thumb_h
    mapped = [(int(ox + p[0] * (4 * thumb_w - 20) + 10), int(oy + (1 - p[1]) * (thumb_h - 20) + 10)) for p in points]
    if len(mapped) > 1: draw.line(mapped, fill=(20, 80, 220), width=3)
    draw.ellipse((mapped[0][0]-5, mapped[0][1]-5, mapped[0][0]+5, mapped[0][1]+5), fill="red")
    destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination)


def build_trajectory(scene_record, split, scene, spec, output_root, ordinal, jpeg_quality):
    scene_hash = scene_record["scene_hash"]
    trajectory_id = f"{scene_hash}_{spec['sample_type']}_{ordinal:02d}_{spec['chunk_count']}chunk"
    trajectory_root = output_root / split / scene_hash / trajectory_id
    if trajectory_root.exists(): shutil.rmtree(trajectory_root)
    rgb_root = trajectory_root / "rgb"; source_root = trajectory_root / "source"
    rgb_root.mkdir(parents=True); source_root.mkdir(parents=True)
    indices = np.asarray(spec["real_frame_indices"], np.int64)
    crop_box, all_k = center_crop_resize_geometry(scene.source_hw, scene.intrinsics)
    selected_k = all_k[indices]
    selected_world = scene.c2w_opencv[indices]
    local = source_relative_opencv_c2w(selected_world, 0)
    cache = {}
    for output_index, real_index in enumerate(indices):
        key = int(real_index)
        if key not in cache: cache[key] = resize_frame(scene.image_paths[key], crop_box)
        cache[key].save(rgb_root / f"{output_index:06d}.jpg", quality=jpeg_quality, subsampling=0)
    cache[int(indices[0])].save(source_root / "source.png")
    np.save(trajectory_root / "target_c2w_local.npy", local)
    np.save(trajectory_root / "intrinsics.npy", selected_k.astype(np.float32))
    np.save(trajectory_root / "real_frame_indices.npy", indices)
    chunks = chunk_real_indices(indices, int(spec["chunk_count"]))
    (trajectory_root / "chunk_real_frame_indices.json").write_text(json.dumps(chunks, indent=2))
    prompt = (f"A stable {scene_record['environment']} {scene_record.get('poi','scene')} with clear "
              "three-dimensional structure, viewed by a smoothly moving camera.")
    rel = lambda p: str(p.relative_to(output_root)).replace("\\", "/")
    record = {
        "trajectory_id": trajectory_id, "scene_hash": scene_hash, "split": split,
        "sample_type": spec["sample_type"], "chunk_count": int(spec["chunk_count"]),
        "source_global_frame": int(indices[0]), "source": rel(source_root / "source.png"),
        "rgb_dir": rel(rgb_root), "target_c2w_local": rel(trajectory_root / "target_c2w_local.npy"),
        "intrinsics": rel(trajectory_root / "intrinsics.npy"),
        "real_frame_indices": rel(trajectory_root / "real_frame_indices.npy"),
        "chunk_real_frame_indices": chunks,
        "revisit_earlier_output_frame": int(spec["revisit_earlier_output_frame"]),
        "revisit_later_output_frame": int(spec["revisit_later_output_frame"]),
        "revisit_earlier_chunk": int(spec["revisit_earlier_chunk"]),
        "revisit_later_chunk": int(spec["revisit_later_chunk"]),
        "prompt": prompt, "fps": TARGET_FPS, "height": TARGET_HW[0], "width": TARGET_HW[1],
        "trainable_chunk_indices": list(range(int(spec["chunk_count"]))),
        "uses_future_gt": False,
        "causal_world_contract": {
            "initial_world_inputs": ["source/source.png"],
            "history_world_inputs": ["model_generated_rgb_before_current_chunk"],
            "forbidden_world_inputs": ["dl3dv_colmap_points", "future_gt_rgb", "future_depth"],
            "current_gt_role": "supervision_only",
        },
    }
    validate_trajectory_record(record, output_root)
    make_visualization(rgb_root, local, trajectory_root / "trajectory_preview.jpg")
    (trajectory_root / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def validate_manifest(manifest, root):
    records = manifest["records"]
    if len({x["scene_hash"] for x in records}) != 256: raise ValueError("manifest must contain 256 scenes")
    split_scenes = Counter()
    for record in records:
        validate_trajectory_record(record, root)
        split_scenes[(record["split"], record["scene_hash"])] = 1
        rgb_count = len(list((root / record["rgb_dir"]).glob("*.jpg")))
        if rgb_count != record["chunk_count"] * CHUNK_STRIDE + 1:
            raise ValueError("RGB count does not match chunk layout")
    counts = Counter(split for split, _ in split_scenes)
    if counts != Counter({"train": 224, "val": 16, "diagnostic": 16}):
        raise ValueError(f"wrong scene split counts: {counts}")
    return True


def main():
    args = parse_args()
    state = json.loads(args.selection_state.read_text(encoding="utf-8"))
    qualified = state.get("qualified", [])
    if len(qualified) != 256: raise ValueError("selection state must have exactly 256 qualified scenes")
    splits = assign_splits(qualified); args.output_root.mkdir(parents=True, exist_ok=True)
    records, scene_summary = [], []
    for scene_record in qualified:
        scene = load_dl3dv_scene(scene_record["raw_path"], source_fps=args.source_fps)
        specs = select_revisit_trajectories(scene, max_trajectories=2)
        if not specs: raise ValueError(f"qualified scene lost all trajectories: {scene_record['scene_hash']}")
        split = splits[scene_record["scene_hash"]]
        for ordinal, spec in enumerate(specs):
            records.append(build_trajectory(scene_record, split, scene, spec, args.output_root,
                                            ordinal, args.jpeg_quality))
        scene_summary.append({"scene_hash": scene_record["scene_hash"], "split": split,
                              "environment": scene_record["environment"],
                              "trajectory_count": len(specs)})
    manifest = {"schema_version": 1, "dataset": "DL3DV-10K official 480P images+poses",
                "target_fps": TARGET_FPS, "resolution": list(TARGET_HW),
                "chunk_frames": 33, "chunk_stride": 32, "uses_future_gt": False,
                "scene_count": 256, "records": records, "scenes": scene_summary}
    validate_manifest(manifest, args.output_root)
    target = args.output_root / "dl3dv_film_manifest.json"
    temp = target.with_suffix(".tmp"); temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8"); temp.replace(target)
    stats = {"manifest": str(target), "scenes": Counter(x["split"] for x in scene_summary),
             "environments": Counter(x["environment"] for x in scene_summary),
             "trajectories": len(records), "chunks": sum(x["chunk_count"] for x in records),
             "revisit_types": Counter(x["sample_type"] for x in records)}
    (args.output_root / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(stats, indent=2, default=dict))


if __name__ == "__main__":
    main()
