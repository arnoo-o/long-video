#!/usr/bin/env python3
"""Run dependency-light project tests; model/data tests remain explicit commands."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tests = (
        "scripts/run_smoke_test.py",
        "scripts/test_memory_transition.py",
    )
    for test in tests:
        subprocess.run([sys.executable, str(root / test)], cwd=root, check=True)
    print("dependency-light full smoke passed")


if __name__ == "__main__":
    main()
