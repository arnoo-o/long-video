import json
from pathlib import Path
import numpy as np
from PIL import Image

def write_sequence(root, rgb, depth, masks, poses_c2w, intrinsics, controls=None, prompt="", metadata=None):
    root = Path(root)
    for name in ("rgb","depth","masks"): (root/name).mkdir(parents=True, exist_ok=True)
    for i, (image, dep, mask) in enumerate(zip(rgb, depth, masks)):
        stem = f"{i:06d}"
        Image.fromarray(np.asarray(image).astype(np.uint8)).save(root/"rgb"/f"{stem}.png")
        np.save(root/"depth"/f"{stem}.npy", np.asarray(dep, np.float32))
        Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255).save(root/"masks"/f"{stem}.png")
    np.save(root/"poses_c2w.npy", np.asarray(poses_c2w, np.float32))
    np.save(root/"intrinsics.npy", np.asarray(intrinsics, np.float32))
    (root/"controls.json").write_text(json.dumps(controls or [], indent=2))
    (root/"prompt.txt").write_text(prompt)
    (root/"metadata.json").write_text(json.dumps(metadata or {}, indent=2))
