"""Build resumable full v5 correspondence caches and a training manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


SCHEMA = "sightline-correspondence-v5"


def _records(payload):
    if isinstance(payload, list):
        return payload
    for key in ("records", "items", "trajectories"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("source manifest has no record list")


def _valid_cache(path: Path, trajectory_id: str) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        payload.get("schema_version") == SCHEMA
        and payload.get("trajectory_id") == trajectory_id
        and payload.get("probe_subset") is False
        and payload.get("token_grid") == {"token_height": 12, "token_width": 20}
        and payload.get("overlap_mining", {}).get("point_stride") == 4
        and payload.get("row_count") == len(payload.get("rows", ()))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--recal-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-records", type=int, default=95)
    args = parser.parse_args()

    source_payload = json.loads(args.source_manifest.read_text())
    source_root = args.source_manifest.parent
    source = {row["trajectory_id"]: row for row in _records(source_payload)}
    selected = json.loads(args.selection_manifest.read_text()).get("trajectory_ids", ())
    required_teacher = ("xyz_world.npy", "valid.npy", "confidence.npy")
    usable = [
        trajectory_id
        for trajectory_id in selected
        if trajectory_id in source
        and all((args.recal_root / trajectory_id / name).is_file() for name in required_teacher)
    ]
    if len(usable) != args.expected_records:
        raise RuntimeError(f"expected {args.expected_records} complete trajectories, found {len(usable)}")

    correspondence_root = args.out_root / "correspondence"
    stage_a_root = args.out_root / "stage_a"
    log_root = args.out_root / "logs"
    for directory in (correspondence_root, stage_a_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)

    builder = Path(__file__).with_name("build_sightline_correspondences.py")

    def build(trajectory_id):
        output = correspondence_root / f"{trajectory_id}.json"
        if _valid_cache(output, trajectory_id):
            return trajectory_id, "skip"
        record = source[trajectory_id]
        teacher = args.recal_root / trajectory_id
        c2w = source_root / record["target_c2w_local"]
        intrinsics = source_root / record["intrinsics"]
        command = [
            sys.executable,
            str(builder),
            "--xyz", str(teacher / "xyz_world.npy"),
            "--valid", str(teacher / "valid.npy"),
            "--confidence", str(teacher / "confidence.npy"),
            "--c2w", str(c2w),
            "--intrinsics", str(intrinsics),
            "--out", str(output),
            "--stage-a-cache", str(stage_a_root / f"{trajectory_id}.json"),
            "--trajectory-id", trajectory_id,
            "--token-height", "12", "--token-width", "20",
            "--screening-stride", "32",
            "--screening-distance-threshold", "0.05",
            "--min-overlap-count", "1", "--min-overlap-ratio", "0.01",
            "--point-stride", "4", "--projection-radius", "1",
            "--cycle-pixel-threshold", "2.0",
        ]
        log = log_root / f"{trajectory_id}.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        if not _valid_cache(output, trajectory_id):
            raise RuntimeError(f"builder produced an invalid cache: {output}")
        return trajectory_id, "generated"

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build, trajectory_id): trajectory_id for trajectory_id in usable}
        for future in as_completed(futures):
            trajectory_id = futures[future]
            try:
                _, status = future.result()
                print(json.dumps({"trajectory_id": trajectory_id, "status": status}), flush=True)
            except Exception as exc:
                failures.append((trajectory_id, repr(exc)))
                print(json.dumps({"trajectory_id": trajectory_id, "status": "failed", "error": repr(exc)}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} correspondence builds failed: {failures}")

    path_keys = (
        "source", "rgb_dir", "target_c2w_local", "intrinsics", "timestamps",
        "frame_sources", "source_frame_indices", "real_keyframe_indices",
        "initial_causal_rgb_dir", "initial_causal_c2w_local",
        "initial_causal_intrinsics", "initial_causal_real_frame_indices",
    )
    records = []
    for trajectory_id in usable:
        row = dict(source[trajectory_id])
        for key in path_keys:
            if key in row:
                row[key] = str((source_root / row[key]).resolve())
        teacher = args.recal_root / trajectory_id
        row.update({
            "gt_latent_cache": str((args.latent_root / trajectory_id / "continuous_49.pt").resolve()),
            "recal_xyz": str((teacher / "xyz_world.npy").resolve()),
            "recal_valid": str((teacher / "valid.npy").resolve()),
            "recal_confidence": str((teacher / "confidence.npy").resolve()),
            "correspondence_cache": str((correspondence_root / f"{trajectory_id}.json").resolve()),
        })
        records.append(row)
    manifest = {"schema_version": "sightline-training-manifest-v1", "record_count": len(records), "records": records}
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.out_manifest)


if __name__ == "__main__":
    main()
