#!/usr/bin/env python3
"""Run dependency-light project tests; model/data tests remain explicit commands."""
import subprocess
import sys
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tests = (
        "scripts/run_smoke_test.py",
        "scripts/test_memory_transition.py",
    )
    for test in tests:
        env=os.environ.copy()
        env["PYTHONPATH"]=str(root)+os.pathsep+env.get("PYTHONPATH","")
        subprocess.run([sys.executable,str(root/test)],cwd=root,env=env,check=True)
    print("dependency-light full smoke passed")


if __name__ == "__main__":
    main()
