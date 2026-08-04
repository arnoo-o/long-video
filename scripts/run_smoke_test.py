import tempfile, numpy as np
from long_video.initialization.view_completion import HoloOracleCompletion
from long_video.memory.node_builder import build_from_views
from long_video.memory.node_store import NodeStore
from long_video.geometry.point_renderer import render
from long_video.types import CameraBatch
from long_video.wah.token_confidence import build_token_confidence
from long_video.wah.attention_bias import confidence_bias
from long_video.data.controls import integrate_controls

def main():
    h=w=32; pano=np.zeros((32,64,3),np.uint8); pano[...,0]=np.arange(64)[None,:]*4; dep=np.ones((32,64),np.float32)
    vs=HoloOracleCompletion(height=h,width=w).complete(pano,dep)
    node=build_from_views(vs,voxel_size=.02); assert len(node.points_xyz)>0
    with tempfile.TemporaryDirectory() as d: NodeStore(d).save(node); assert len(NodeStore(d).load(node.node_id).points_xyz)==len(node.points_xyz)
    cams=CameraBatch(integrate_controls(np.eye(4),[{'forward':1,'delta_time':.1}]*3),np.repeat(vs.intrinsics[:1],3,0),h,w); wb=render(node,cams); assert wb.rgb.shape==(3,h,w,3)
    tc,vr=build_token_confidence(wb.confidence,wb.visibility,(3,h,w),(3,4,4)); assert np.allclose(confidence_bias(np.ones_like(tc)),0)
    print('smoke test passed',len(node.points_xyz),wb.coverage_per_frame.tolist(),tc.shape)
if __name__=='__main__': main()
