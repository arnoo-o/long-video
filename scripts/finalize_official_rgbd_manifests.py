#!/usr/bin/env python3
"""Select exact official additions and emit mixed 3/6-chunk manifests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


PATH_KEYS = ("rgb_dir", "depth_dir", "c2w_abs", "c2w_local", "intrinsics", "timestamps", "correspondence_cache", "metadata", "pointcloud")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def absolute_existing(row: dict, root: Path) -> dict:
    result = dict(row)
    for key in PATH_KEYS:
        if key in result and not Path(result[key]).is_absolute():
            result[key] = str((root / result[key]).resolve())
    return result


def addition_row(metadata_path: Path) -> dict:
    root = metadata_path.parent
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    row = dict(meta)
    row.update(
        rgb_dir=str(root / "rgb"), depth_dir=str(root / "depth"),
        c2w_abs=str(root / "c2w_abs.npy"), c2w_local=str(root / "c2w_local.npy"),
        intrinsics=str(root / "intrinsics.npy"), timestamps=str(root / "timestamps.npy"),
        correspondence_cache=str(root / "correspondence_cache.npz"),
        pointcloud=str(root / "pointcloud.npz"), metadata=str(metadata_path),
        memory_eligible=True, training_scope="rgbd_memory",
    )
    return row


def motion_score(row: dict) -> float:
    poses = np.load(row["c2w_abs"], mmap_mode="r")
    translation = float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())
    rotations = poses[:, :3, :3]
    relative = np.einsum("tji,tjk->tik", rotations[:-1], rotations[1:])
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)).sum()
    revisit = 1.0 / (0.05 + float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])))
    return translation + float(angle) + 0.1 * revisit


def ranked(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (-motion_score(row), row["record_id"]))


def select_tartan(rows: list[dict], train_count: int, val_count: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["sequence_id"].rsplit("/", 1)[0]].append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    # Exact subset-sum keeps every one of the 400 constructed records while
    # assigning whole trajectories to one split only.
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for key in ordered:
        count=len(groups[key])
        for total, selected in sorted(tuple(choices.items()),reverse=True):
            if total+count<=val_count and total+count not in choices:
                choices[total+count]=selected+(key,)
    if val_count not in choices:
        raise ValueError('TartanGround trajectory groups cannot realize the exact val count')
    val_groups=set(choices[val_count])
    val = ranked([row for key in val_groups for row in groups[key]])
    train = ranked([row for key in ordered if key not in val_groups for row in groups[key]])
    if len(train) != train_count or len(val) != val_count:
        raise ValueError(f'TartanGround capacity is insufficient: train={len(train)}, val={len(val)}')
    for row in train: row["split"] = "train"
    for row in val: row["split"] = "val"
    return train, val


def assert_disjoint(train: list[dict], val: list[dict]) -> None:
    def identity(row: dict) -> tuple[str, str]:
        sequence = row["sequence_id"]
        if row["dataset"] in {"tartanground", "arkitscenes"}: sequence = sequence.rsplit("/", 1)[0]
        return row["dataset"], sequence
    overlap = {identity(row) for row in train} & {identity(row) for row in val}
    if overlap:
        raise ValueError(f'train/val sequence leakage: {sorted(overlap)[:5]}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--additions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--latent-cache-root", type=Path, required=True)
    parser.add_argument("--ddad", type=int, default=132)
    parser.add_argument("--arkit-train", type=int, default=350)
    parser.add_argument("--arkit-val", type=int, default=50)
    parser.add_argument("--tartan-train", type=int, default=350)
    parser.add_argument("--tartan-val", type=int, default=50)
    args = parser.parse_args()

    payload = json.loads(args.existing_manifest.read_text(encoding="utf-8"))
    existing = [absolute_existing(row, args.existing_manifest.parent) for row in payload["records"]]
    additions = [addition_row(path) for path in args.additions_root.glob("records/*/*/metadata.json")]
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in additions: by_dataset[row["dataset"]].append(row)

    ddad = ranked(by_dataset["ddad"])[:args.ddad]
    if len(ddad) != args.ddad: raise ValueError(f'DDAD capacity is insufficient: {len(ddad)}')
    for row in ddad: row["split"] = "train"
    arkit_train = ranked([row for row in by_dataset["arkitscenes"] if row["split"] == "train"])[:args.arkit_train]
    arkit_val = ranked([row for row in by_dataset["arkitscenes"] if row["split"] == "val"])[:args.arkit_val]
    if len(arkit_train) != args.arkit_train or len(arkit_val) != args.arkit_val:
        raise ValueError(f'ARKit capacity is insufficient: train={len(arkit_train)}, val={len(arkit_val)}')
    tartan_train, tartan_val = select_tartan(by_dataset["tartanground"], args.tartan_train, args.tartan_val)

    train = [row for row in existing if row["split"] == "train"] + ddad + arkit_train + tartan_train
    val = [row for row in existing if row["split"] == "val"] + arkit_val + tartan_val
    for row in train + val:
        temporal=1+(int(row["frame_count"])-1)//4
        cache=args.latent_cache_root/row["record_id"]/f"continuous_{temporal}.pt"
        if not cache.is_file():
            raise FileNotFoundError(f'missing latent cache: {cache}')
        row["latent_cache"]=str(cache)
    assert_disjoint(train, val)
    if len({row["record_id"] for row in train + val}) != len(train) + len(val):
        raise ValueError("record ids are not unique")
    args.output_root.mkdir(parents=True, exist_ok=True)
    header = {"schema_version": "rgbd-memory-manifest-v2", "height": 480, "width": 832}
    atomic_json(args.output_root / "manifest_train.json", {**header, "split": "train", "records": train})
    atomic_json(args.output_root / "manifest_train_p3.json", {**header, "split": "train", "records": train})
    atomic_json(args.output_root / "manifest_val.json", {**header, "split": "val", "records": val})
    atomic_json(args.output_root / "manifest_all.json", {**header, "records": train + val})
    print(json.dumps({"train": len(train), "val": len(val), "all": len(train) + len(val),
                      "ddad": len(ddad), "arkit_train": len(arkit_train), "arkit_val": len(arkit_val),
                      "tartan_train": len(tartan_train), "tartan_val": len(tartan_val)}, indent=2))


if __name__ == "__main__":
    main()
