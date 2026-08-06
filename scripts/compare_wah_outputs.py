#!/usr/bin/env python3
import argparse
import json
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("original")
parser.add_argument("patched")
args=parser.parse_args()
a=np.load(args.original)
b=np.load(args.patched)
delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
result={
    "shape_equal":a.shape==b.shape,
    "dtype_equal":str(a.dtype)==str(b.dtype),
    "max_abs":float(delta.max()),
    "mean_abs":float(delta.mean()),
    "all_equal":bool(np.array_equal(a,b)),
    "allclose_1e_6":bool(np.allclose(a,b,atol=1e-6,rtol=1e-6)),
}
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["allclose_1e_6"] else 1)