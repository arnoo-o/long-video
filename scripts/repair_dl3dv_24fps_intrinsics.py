#!/usr/bin/env python3
"""Repair dense DL3DV K coordinates and snap near-keyframe RGBs in place."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from long_video.data.dl3dv import TARGET_HW, center_crop_resize_geometry, load_dl3dv_scene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--selection-state", type=Path, required=True)
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()


def valid_k(k):
    fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
    return bool(np.isfinite(k).all() and (fx > 1).all() and (fy > 1).all() and
        (fx < 10 * TARGET_HW[1]).all() and (fy < 10 * TARGET_HW[0]).all() and
        (cx > 0).all() and (cx < TARGET_HW[1]).all() and
        (cy > 0).all() and (cy < TARGET_HW[0]).all())


def atomic_npy(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value)
    temp.replace(path)


def main():
    args = parse_args()
    manifest_path = args.dataset_root / "dl3dv_24fps_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state = json.loads(args.selection_state.read_text())
    metadata = {x["scene_hash"]: x for x in state["qualified"]}
    totals = {"records": 0, "snapped_left": 0, "snapped_right": 0,
              "rife": 0, "invalid": 0}
    for record in manifest["records"]:
        base = args.dataset_root / Path(record["intrinsics"]).parent
        item = metadata[record["scene_hash"]]
        scene = load_dl3dv_scene(item["raw_path"], duration=item["duration"])
        crop, resized_k = center_crop_resize_geometry(scene.source_hw, scene.intrinsics)
        sources = json.loads((args.dataset_root / record["frame_sources"]).read_text())
        left = np.array([x["left_real_index"] for x in sources], np.int64)
        right = np.array([x["right_real_index"] for x in sources], np.int64)
        alpha = np.array([x["alpha"] for x in sources], np.float64)
        dense_k = np.stack([(1-a)*resized_k[l] + a*resized_k[r]
                            for l, r, a in zip(left, right, alpha)]).astype(np.float32)
        pi3_indices = np.load(base / "pi3_initial_real_frame_indices.npy")
        pi3_k = resized_k[pi3_indices].astype(np.float32)
        if not valid_k(dense_k) or not valid_k(pi3_k):
            totals["invalid"] += 1
            raise ValueError(f"invalid 384x640 intrinsics: {record['trajectory_id']}")

        cache = {}
        def real_image(index):
            index = int(index)
            if index not in cache:
                with Image.open(scene.image_paths[index]) as image:
                    cache[index] = image.convert("RGB").crop(crop).resize(
                        (TARGET_HW[1], TARGET_HW[0]), Image.Resampling.LANCZOS)
            return cache[index]

        rgb_dir = args.dataset_root / record["rgb_dir"]
        for output_index, source in enumerate(sources):
            a = float(source["alpha"])
            use_left, use_right = a < .01, (1-a) < .01
            if use_left or use_right:
                real_index = int(source["left_real_index"] if use_left else source["right_real_index"])
                temp = rgb_dir / f"{output_index:06d}.jpg.tmp"
                real_image(real_index).save(temp, format="JPEG", quality=args.jpeg_quality, subsampling=0)
                temp.replace(rgb_dir / f"{output_index:06d}.jpg")
                source["source"] = "real"
                source["rgb_real_index"] = real_index
                totals["snapped_left" if use_left else "snapped_right"] += 1
            else:
                source["source"] = "rife"
                source["rgb_real_index"] = None
                totals["rife"] += 1

        atomic_npy(base / "intrinsics.npy", dense_k)
        atomic_npy(base / "pi3_initial_intrinsics.npy", pi3_k)
        frame_temp = base / "frame_sources.json.tmp"
        frame_temp.write_text(json.dumps(sources, indent=2)); frame_temp.replace(base / "frame_sources.json")
        validation_path = base / "validation.json"
        validation = json.loads(validation_path.read_text())
        fx, fy, cx, cy = dense_k[:,0,0], dense_k[:,1,1], dense_k[:,0,2], dense_k[:,1,2]
        validation.update({"valid": True, "intrinsics_valid_for_384x640": True,
            "renderer_intrinsics_are_384x640": True,
            "near_keyframe_rgb_snap_threshold": .01,
            "real_count": sum(x["source"] == "real" for x in sources),
            "rife_count": sum(x["source"] == "rife" for x in sources),
            "intrinsics_range": {"fx":[float(fx.min()),float(fx.max())],
                "fy":[float(fy.min()),float(fy.max())], "cx":[float(cx.min()),float(cx.max())],
                "cy":[float(cy.min()),float(cy.max())]}})
        temp = validation_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(validation, indent=2)); temp.replace(validation_path)
        totals["records"] += 1
        if totals["records"] % 10 == 0:
            print(json.dumps(totals), flush=True)
    manifest["intrinsics_coordinates"] = "384x640 crop-resize pixels"
    manifest["near_keyframe_rgb_snap_threshold"] = .01
    manifest["intrinsics_repair_valid_records"] = totals["records"]
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2)); temp.replace(manifest_path)
    (args.dataset_root / "intrinsics_repair_summary.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
