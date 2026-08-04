import json, os, numpy as np

class NodeStore:
    def __init__(self, root): self.root=os.fspath(root)
    def save(self,node):
        d=os.path.join(self.root,'nodes',node.node_id); os.makedirs(d,exist_ok=True)
        meta={k:getattr(node,k) for k in ('node_id','status','parent_id','created_frame','coverage_radius')}; meta.update({'bbox_min':node.bbox_min.tolist(),'bbox_max':node.bbox_max.tolist(),'center_c2w':node.center_c2w.tolist()})
        with open(os.path.join(d,'metadata.json'),'w') as f: json.dump(meta,f)
        np.savez_compressed(os.path.join(d,'node_arrays.npz'), **{k:getattr(node,k) for k in ('view_rgb','view_depth','view_c2w','view_intrinsics','points_xyz','points_rgb','points_confidence','points_source','observation_count')})
    def load(self,node_id):
        from ..types import SpatialNode
        d=os.path.join(self.root,'nodes',node_id); meta=json.load(open(os.path.join(d,'metadata.json'))); a=np.load(os.path.join(d,'node_arrays.npz'))
        fields=[a[k] for k in ('view_rgb','view_depth','view_c2w','view_intrinsics','points_xyz','points_rgb','points_confidence','points_source','observation_count')]
        return SpatialNode(meta['node_id'],meta['status'],meta['parent_id'],np.array(meta['center_c2w'],np.float32),meta['created_frame'],meta['coverage_radius'],np.array(meta['bbox_min'],np.float32),np.array(meta['bbox_max'],np.float32),*fields)
