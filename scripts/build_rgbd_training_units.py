"""Create the P1/P2 3-chunk view manifest from a mixed RGB-D manifest.

Six-chunk records are never resampled: their two views are exact [0,96] and
[96,192] source-frame slices.  Each view receives a separately rebased causal
cache, so it can be consumed independently without future identities.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest
from long_video.data.rgbd_memory import build_causal_correspondence_cache


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def unit_row(record, output_root: Path, offset: int) -> dict:
    row = dict(record.raw)
    row.update(record_id=f"{record.record_id}__frames_{offset:03d}_{offset + 96:03d}", parent_record_id=record.record_id,
               source_frame_start=offset, frame_count=97, chunk_count=3)
    cache_dir = output_root / "unit_correspondence"
    cache = cache_dir / f"{row['record_id'].replace(':', '_')}.npz"
    if not cache.is_file():
        rgb = record.rgb_paths(); depth = list(record.depth_paths())
        abs_c2w, K = record.load_cameras(local=False)
        build_causal_correspondence_cache(depth[offset:offset + 97], np.asarray(abs_c2w[offset:offset + 97]), np.asarray(K[offset:offset + 97]), cache, chunk_count=3)
    row["correspondence_cache"] = str(cache.relative_to(output_root))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="mixed full-record train manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_rgbd_memory_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        if record.chunk_count == 3:
            rows.append(dict(record.raw))
        elif record.chunk_count == 6:
            rows.extend((unit_row(record, args.out.parent, 0), unit_row(record, args.out.parent, 96)))
        else:
            raise ValueError(f"unsupported training record geometry: {record.record_id}")
    if len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError("training-unit ids must be unique")
    atomic_json(args.out, {"schema_version": "rgbd-memory-training-units-v2", "split": "train", "height": 480, "width": 832, "records": rows})
    print(json.dumps({"records": len(rows), "three_chunk_units": len(rows), "source_manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
