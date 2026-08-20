"""Pinned upstream identity for the legacy/reference WAH baseline only."""
from __future__ import annotations
import subprocess
from pathlib import Path

WAH_UPSTREAM_COMMIT = "09aa6461355b298bfced51007bd709a251d6033a"

def assert_wah_upstream(root):
    root = Path(root)
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception as error:
        raise RuntimeError(f"cannot verify pinned Warp-as-History checkout: {root}") from error
    if commit != WAH_UPSTREAM_COMMIT:
        raise RuntimeError(f"WAH upstream mismatch: {commit}, required {WAH_UPSTREAM_COMMIT}")
    return commit
