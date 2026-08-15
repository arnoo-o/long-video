"""Regression: every rendered PointWorld payload contributes to identity."""
import numpy as np
from types import SimpleNamespace
from long_video.online.pipeline import point_world_snapshot_identity

def main():
    node = SimpleNamespace(node_id="node_000", quality_metrics={"recal3r_world_version": 1},
        points_xyz=np.array([[.001,.002,.003]],np.float32), points_rgb=np.array([[1,2,3]],np.uint8),
        points_confidence=np.array([.5],np.float32))
    baseline=point_world_snapshot_identity(node)
    node.points_rgb[0,0]+=1; assert point_world_snapshot_identity(node)!=baseline
    node.points_rgb[0,0]-=1; node.points_confidence[0]+=.1; assert point_world_snapshot_identity(node)!=baseline
    print('point-world-snapshot-identity-ok')
if __name__=='__main__': main()
