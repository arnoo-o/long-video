#!/usr/bin/env python3
"""Build balanced Phase A/B 24 FPS windows from qualified Holo360D archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rife-root", type=Path, required=True)
    parser.add_argument("--rife-checkpoint", type=Path, required=True)
    parser.add_argument("--rife-python", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    return parser.parse_args()


def _extract(archive, destination, start, count):
    with ZipFile(archive) as handle:
        rgb = sorted(
            [name for name in handle.namelist() if "/rgb/" in name and name.endswith(".jpg")],
            key=lambda name: float(Path(name).stem),
        )
        root = rgb[0].split("/", 1)[0]
        if start + count > len(rgb):
            raise IndexError("selected anchor window exceeds archive")
        consolidated_pose = f"{root}/poses/pose.txt"
        pose_rows = None
        if consolidated_pose in handle.namelist():
            lines = handle.read(consolidated_pose).decode("utf-8").splitlines()
            pose_rows = {Path(fields[0]).stem: fields[1:] for fields in map(str.split, lines[1:])}
        for index in range(start, start + count):
            stem = Path(rgb[index]).stem
            for relative in (
                f"rgb/{stem}.jpg", f"depth/mesh_depth/{stem}.exr",
                f"mask/{stem}.jpg",
            ):
                handle.extract(f"{root}/{relative}", destination)
            if pose_rows is None:
                handle.extract(f"{root}/poses/{stem}.txt", destination)
            else:
                values = pose_rows.get(stem)
                if values is None or len(values) != 12:
                    raise ValueError(f"missing consolidated pose for {stem}")
                pose_path = Path(destination) / root / "poses" / f"{stem}.txt"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                pose_path.write_text(" ".join(values) + "\n", encoding="utf-8")
    return Path(destination) / root


def _phase_a_starts(runs, count=5):
    candidates = []
    for run in sorted(runs, key=lambda item: item["anchor_count"], reverse=True):
        cursor = int(run["start"])
        while cursor + count <= int(run["end"]):
            candidates.append(cursor)
            cursor += count
    if len(candidates) < 5:
        raise ValueError("scene does not contain five disjoint Phase A windows")
    return {"train": candidates[:4], "diagnostic": candidates[4:5]}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _renderer_overlap_for_candidate(extracted, candidate):
    import numpy as np
    from long_video.data.erp_geometry import source_relative_c2w
    from long_video.data.holo360d import Holo360DReader
    from long_video.geometry.point_renderer import render
    from long_video.oracle_training.dataset import _perspective, _resize_erp
    from long_video.oracle_training.oracle_node import build_oracle_erp_node
    from long_video.oracle_training.revisit import bidirectional_depth_reprojection_overlap
    from long_video.types import CameraBatch

    reader = Holo360DReader(extracted, normalize_first_pose=False)
    source = reader.read(0)
    source_rgb, source_depth, source_mask = _resize_erp(
        source.rgb, source.depth, source.mask, (512, 1024),
    )
    node = build_oracle_erp_node(
        source_rgb, source_depth, source_mask,
        source_c2w_local=np.eye(4, dtype=np.float32), voxel_size=0.02, pixel_center=0.5,
        model_versions={"geometry": "Holo360D_mesh_depth", "builder": "selection_m0"},
    )
    offsets = [int(candidate["earlier_anchor_offset"]), int(candidate["later_anchor_offset"])]
    frames = [reader.read(offset) for offset in offsets]
    intrinsics = []
    poses = []
    for frame in frames:
        _, _, _, _, intrinsic = _perspective(frame, fov_degrees=90.0, height=192, width=320)
        intrinsics.append(intrinsic)
        poses.append(source_relative_c2w(source.raw_c2w, frame.raw_c2w))
    rendered = render(
        node, CameraBatch(np.asarray(poses), np.asarray(intrinsics), 192, 320),
        device="cpu", near=0.05, far=100.0, point_radius=1, chunk_points=1000000,
    )
    return bidirectional_depth_reprojection_overlap(
        rendered.depth, rendered.visibility, np.asarray(poses), np.asarray(intrinsics),
    )


def _finalize_scene_candidates(scene, archive, output):
    from long_video.oracle_training.revisit import (
        add_renderer_overlap, choose_independent_final_candidates,
    )

    pools = {
        "revisit": scene.get("phase_b_revisit_pose_candidates", []),
        "large_motion": scene.get("phase_b_large_motion_pose_candidates", []),
    }
    if not all(pools.values()):
        raise ValueError("scan report must contain pose-prefiltered Phase B candidate pools")
    evaluation_root = Path(output) / "_selection_eval" / scene["scene_id"]
    evaluated = {"revisit": [], "large_motion": []}
    for sample_type, candidates in pools.items():
        for ordinal, candidate in enumerate(candidates):
            destination = evaluation_root / f"{sample_type}_{ordinal:03d}"
            extracted = _extract(
                archive, destination, int(candidate["start"]), int(candidate["anchor_count"]),
            )
            try:
                overlap = _renderer_overlap_for_candidate(extracted, candidate)
            finally:
                resolved = destination.resolve()
                if evaluation_root.resolve() not in resolved.parents:
                    raise RuntimeError("selection cleanup escaped its temporary root")
                shutil.rmtree(resolved)
            evaluated[sample_type].append(
                add_renderer_overlap(candidate, overlap, sample_type=sample_type)
            )
    revisit, motion = choose_independent_final_candidates(
        evaluated["revisit"], evaluated["large_motion"],
    )
    if not revisit or not motion:
        raise ValueError("renderer scoring produced no independent Phase B selections")
    return revisit, motion


def main():
    args = _args()
    if args.physical_gpu != 1:
        raise ValueError("data preparation is restricted to physical GPU 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from long_video.oracle_training.dense24 import PracticalRIFE425
    from long_video.oracle_training.dense_dataset import build_dense_oracle_sequence

    report = json.loads(args.scan_report.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rife = PracticalRIFE425(args.rife_root, args.rife_checkpoint, args.rife_python)
    rife_revision = subprocess.check_output(
        ["git", "-C", str(args.rife_root), "rev-parse", "HEAD"], text=True
    ).strip()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    records = []
    for scene in report["scenes"]:
        archive = Path(scene["archive"])
        scene_id = scene["scene_id"]
        prompt = (
            "A stable outdoor environment viewed by a smoothly moving camera."
            if scene_id.lower().startswith("outdoor")
            else "A stable indoor environment viewed by a smoothly moving camera."
        )
        phase_a = _phase_a_starts(scene["continuous_runs"])
        specs = []
        for split, starts in phase_a.items():
            specs.extend({"phase": "A", "split": split, "start": start, "anchors": 5, "chunks": 1}
                         for start in starts)
        selected_revisit, selected_motion = _finalize_scene_candidates(
            scene, archive, args.output,
        )
        scene["selected_phase_b_windows"] = selected_revisit
        scene["selected_phase_b_large_motion_windows"] = selected_motion
        for window in selected_revisit:
            specs.append({
                "phase": "B", "split": "train", "start": int(window["start"]),
                "anchors": int(window["anchor_count"]), "chunks": int(window["chunks"]),
                "revisit": window,
                "sample_type": "revisit",
            })
        for window in selected_motion:
            specs.append({
                "phase": "B", "split": "train", "start": int(window["start"]),
                "anchors": int(window["anchor_count"]), "chunks": int(window["chunks"]),
                "revisit": window, "sample_type": "large_motion",
            })
        for ordinal, spec in enumerate(specs):
            sequence_id = (
                f"{scene_id}_phase{spec['phase']}_{spec['split']}_{ordinal:03d}_"
                f"{spec['chunks']}chunk_24fps"
            )
            extracted = _extract(
                archive, args.output / "_extracted" / sequence_id,
                spec["start"], spec["anchors"],
            )
            path, metadata = build_dense_oracle_sequence(
                extracted, args.output, sequence_id=sequence_id, split=spec["split"],
                anchor_count=spec["anchors"], rife=rife,
                erp_resolution=(1024, 2048), perspective_resolution=(384, 640),
                fov_degrees=90.0, pixel_center=0.5, prompt=prompt,
                voxel_size=0.01,
                renderer_kwargs={
                    "device": "cuda:0", "near": 0.05, "far": 100.0,
                    "point_radius": 1, "chunk_points": 1000000,
                },
                rife_revision=rife_revision,
                rife_checkpoint=args.rife_checkpoint / "flownet.pkl",
            )
            metadata["training_phase"] = spec["phase"]
            metadata["selected_chunks"] = spec["chunks"]
            metadata["scene_scale"] = 1.0
            metadata["scene_scale_source"] = "Holo360D_dataset_calibrated_metric"
            metadata["revisit"] = spec.get("revisit")
            metadata["sample_type"] = spec.get("sample_type", "single_chunk")
            (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            records.append({
                "sequence_id": sequence_id, "scene_id": scene_id,
                "phase": spec["phase"], "split": spec["split"],
                "path": str(path), "anchor_start": spec["start"],
                "anchor_count": spec["anchors"], "chunk_count": spec["chunks"],
                "metadata": metadata,
                "sample_type": metadata["sample_type"],
                "training_chunk_index": int(spec.get("revisit", {}).get("training_chunk_index", 0)),
            })
    manifest = {
        "schema_version": 3,
        "git_sha": git_sha,
        "scan_report": str(args.scan_report),
        "scan_report_sha256": _sha256(args.scan_report),
        "rife_revision": rife_revision,
        "rife_checkpoint_sha256": _sha256(args.rife_checkpoint / "flownet.pkl"),
        "sequences": records,
    }
    target = args.output / "spatial_reanchor_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({
        "manifest": str(target),
        "phase_a": sum(item["phase"] == "A" for item in records),
        "phase_b": sum(item["phase"] == "B" for item in records),
        "scenes": sorted({item["scene_id"] for item in records}),
    }, indent=2))


if __name__ == "__main__":
    main()
