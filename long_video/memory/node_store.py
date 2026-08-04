import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from ..types import SpatialNode

class NodeStore:
    def __init__(self, root): self.root=Path(root)
    def save(self,node):
        final=self.root/"nodes"/node.node_id; final.parent.mkdir(parents=True,exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix=node.node_id+".",dir=final.parent))
        arrays={k:getattr(node,k) for k in ("view_rgb","view_depth","view_c2w","view_intrinsics","points_xyz","points_rgb","points_confidence","points_source","observation_count")}
        if node.points_normal is not None: arrays["points_normal"]=node.points_normal
        array_path=temp/"node_arrays.npz"; np.savez_compressed(array_path,**arrays)
        digest=hashlib.sha256(array_path.read_bytes()).hexdigest()
        meta={"schema_version":node.schema_version,"node_id":node.node_id,"status":node.status,"parent_id":node.parent_id,"created_frame":node.created_frame,"coverage_radius":node.coverage_radius,"center_c2w":node.center_c2w.tolist(),"bbox_min":node.bbox_min.tolist(),"bbox_max":node.bbox_max.tolist(),"depth_convention":node.depth_convention,"quality_metrics":node.quality_metrics,"node_arrays_sha256":digest}
        (temp/"metadata.json").write_text(json.dumps(meta,indent=2))
        if final.exists(): backup=final.with_name(final.name+".previous"); os.replace(final,backup)
        os.replace(temp,final)
    def load(self,node_id):
        path=self.root/"nodes"/node_id; meta=json.loads((path/"metadata.json").read_text()); blob=(path/"node_arrays.npz").read_bytes()
        if hashlib.sha256(blob).hexdigest()!=meta["node_arrays_sha256"]: raise IOError(f"Checksum mismatch for node {node_id}")
        arrays=np.load(path/"node_arrays.npz"); normal=arrays["points_normal"] if "points_normal" in arrays.files else None
        return SpatialNode(meta["node_id"],meta["status"],meta.get("parent_id"),np.asarray(meta["center_c2w"],np.float32),meta["created_frame"],meta["coverage_radius"],np.asarray(meta["bbox_min"],np.float32),np.asarray(meta["bbox_max"],np.float32),*[arrays[k] for k in ("view_rgb","view_depth","view_c2w","view_intrinsics","points_xyz","points_rgb","points_confidence","points_source","observation_count")],normal,meta.get("depth_convention","RAY_DISTANCE"),meta.get("schema_version",1),meta.get("quality_metrics",{}))
