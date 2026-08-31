#!/usr/bin/env python3
"""Remove cache artifacts not referenced by any formal RGB-D manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


MANIFESTS = (
    "manifest_all.json", "manifest_train.json", "manifest_train_p3.json",
    "manifest_val.json", "manifest_train_units_3chunk.json",
)


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.unified_root.resolve()
    referenced: set[Path] = set()
    for name in MANIFESTS:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        for row in payload["records"]:
            for value in row.values():
                if not isinstance(value, str):
                    continue
                path = Path(value)
                referenced.add((path if path.is_absolute() else root / path).resolve())
    candidates: list[Path] = []
    for directory_name in ("full_latents", "unit_latents"):
        directory = (root / directory_name).resolve()
        if directory.parent != root:
            raise RuntimeError(f"unsafe cache root: {directory}")
        kept_children = set()
        for path in referenced:
            try:
                relative = path.relative_to(directory)
            except ValueError:
                continue
            if relative.parts:
                kept_children.add(relative.parts[0])
        for child in directory.iterdir():
            if child.name not in kept_children:
                candidates.append(child)
    correspondence = (root / "unit_correspondence").resolve()
    if correspondence.parent != root:
        raise RuntimeError(f"unsafe cache root: {correspondence}")
    for child in correspondence.iterdir():
        if child.resolve() not in referenced:
            candidates.append(child)
    total = sum(size(path) for path in candidates)
    print(json.dumps({"candidates": len(candidates), "bytes": total, "apply": args.apply}, indent=2))
    if args.apply:
        for path in candidates:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


if __name__ == "__main__":
    main()
