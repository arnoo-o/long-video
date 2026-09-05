"""Create the P1/P2 3-chunk view manifest from a mixed RGB-D manifest.

Six-chunk records are never resampled: their two views are exact [0,96] and
[96,192] source-frame slices.  Each view receives a separately rebased causal
cache, so it can be consumed independently without future identities.
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.training.rgbd_memory_data import load_rgbd_memory_manifest


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def unit_near_depth(record, offset: int, *, samples: int = 16) -> float:
    """Compute a unit-owned SI near-depth without touching parent metadata."""
    directory=Path(record.raw['depth_dir'])
    frames=sorted(path for path in directory.iterdir() if path.suffix.lower() in {'.png','.jpg','.jpeg'})
    parent_start=int(record.raw.get('source_frame_start',0))
    unit_frames=frames[parent_start+int(offset):parent_start+int(offset)+97]
    if len(unit_frames)!=97:
        raise ValueError(f'{record.record_id}: unit at {offset} has {len(unit_frames)} depth frames, expected 97')
    quantiles=[]
    for index in np.unique(np.linspace(0,96,min(int(samples),97),dtype=int)):
        depth_m=np.asarray(Image.open(unit_frames[int(index)]),dtype=np.float32)/1000.0
        valid=depth_m[np.isfinite(depth_m)&(depth_m>0)]
        if valid.size:
            quantiles.append(float(np.quantile(valid,.25)))
    value=float(np.median(quantiles)) if quantiles else float('nan')
    if not np.isfinite(value) or value<=0:
        raise ValueError(f'{record.record_id}: unit at {offset} has invalid near_depth')
    return value


def unit_row(record, output_root: Path, latent_root: Path, offset: int, *, rebuild: bool = False, arrays=None) -> dict:
    row = dict(record.raw)
    # A unit must never retain the parent's continuous_49 cache as a fallback.
    row.pop("latent_cache", None)
    row.pop("gt_latent_cache", None)
    # The overlapping views have independent depth statistics. They are kept
    # only in the unit row; never write them into shared parent metadata.
    row.pop('near_depth',None); row.pop('near_depth_unit',None); row.pop('near_depth_method',None)
    near_depth=unit_near_depth(record,offset)
    row.update(record_id=f"{record.record_id}__frames_{offset:03d}_{offset + 96:03d}", parent_record_id=record.record_id,
               source_frame_start=offset, frame_count=97, chunk_count=3,
               near_depth=near_depth, near_depth_unit='m', near_depth_method='median(frame_valid_depth_q25_m)')
    cache_dir = output_root / "unit_correspondence"
    cache = cache_dir / f"{row['record_id'].replace(':', '_')}.npz"
    if rebuild or not cache.is_file():
        arrays=record.load_correspondences() if arrays is None else arrays
        stop=offset+97; chunk_offset=offset//32
        keep=((arrays['query_frame']>=offset)&(arrays['query_frame']<stop)&
              (arrays['key_frame']>=offset)&(arrays['key_frame']<stop)&
              (arrays['query_chunk']>=chunk_offset)&(arrays['query_chunk']<chunk_offset+3)&
              (arrays['key_chunk']>=chunk_offset)&(arrays['key_chunk']<chunk_offset+3))
        sliced={key:value[keep].copy() for key,value in arrays.items()}
        sliced['query_frame']-=offset; sliced['key_frame']-=offset
        sliced['query_chunk']-=chunk_offset; sliced['key_chunk']-=chunk_offset
        cache.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(cache,**sliced)
    row["correspondence_cache"] = str(cache.relative_to(output_root))
    latent = latent_root / row["record_id"].replace(":", "_") / "continuous_25.pt"
    row["gt_latent_cache"] = str(latent.relative_to(output_root) if latent.is_relative_to(output_root) else latent)
    row["latent_schema"] = "continuous_25"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="mixed full-record train manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, help="unit-owned cache root; defaults beside the output manifest")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    records = load_rgbd_memory_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    latent_root = (args.latent_root or (args.out.parent / "unit_latents")).resolve()
    def convert(record):
        if record.chunk_count == 3:
            row=dict(record.raw)
            row.pop("latent_cache",None); row.pop("gt_latent_cache",None)
            latent=latent_root/record.record_id.replace(":","_")/"continuous_25.pt"
            row.update(gt_latent_cache=str(latent.relative_to(args.out.parent) if latent.is_relative_to(args.out.parent) else latent),latent_schema="continuous_25")
            return [row]
        elif record.chunk_count == 6:
            first_cache = args.out.parent / "unit_correspondence" / f"{record.record_id.replace(':', '_')}__frames_000_096.npz"
            second_cache = args.out.parent / "unit_correspondence" / f"{record.record_id.replace(':', '_')}__frames_096_192.npz"
            # Existing units already own their rebased causal caches.  Avoid
            # reading/decompressing the much larger parent cache unless at
            # least one unit actually has to be built.
            arrays = None
            if args.rebuild or not (first_cache.is_file() and second_cache.is_file()):
                arrays = record.load_correspondences()
            return [unit_row(record,args.out.parent,latent_root,0,rebuild=args.rebuild,arrays=arrays),
                    unit_row(record,args.out.parent,latent_root,96,rebuild=args.rebuild,arrays=arrays)]
        else:
            raise ValueError(f"unsupported training record geometry: {record.record_id}")
    if args.workers < 1: raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows=[row for converted in executor.map(convert,records) for row in converted]
    if len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError("training-unit ids must be unique")
    atomic_json(args.out, {"schema_version": "rgbd-memory-training-units-v2", "split": "train", "height": 480, "width": 832, "records": rows})
    print(json.dumps({"records": len(rows), "three_chunk_units": len(rows), "source_manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
