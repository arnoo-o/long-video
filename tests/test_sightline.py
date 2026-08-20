import torch
from long_video.sightline.rays import plucker_rays
from long_video.sightline.conditioning import SightlineConditioner
from long_video.sightline.history import HistoryManager
from long_video.sightline.memory import LongTermKVMemory

def test_plucker_ray_geometry():
    K=torch.tensor([[[100.,0,2],[0,100,2],[0,0,1]]]); c=torch.eye(4).unsqueeze(0)
    r=plucker_rays(c,K,4,4); assert r.shape==(1,1,4,4,7); assert torch.allclose(r[0,0,1,1,:3],torch.tensor([0.,0.,1.]),atol=1e-5)

def test_scale_augmentation_gate_only():
    torch.manual_seed(1); m=SightlineConditioner(16); r=torch.randn(2,3,7); q,k=m(r,training=False); assert q.shape==k.shape==(2,3,16)
    m.alpha.data.zero_(); q,k=m(r,training=True); assert torch.count_nonzero(q)==0 and torch.count_nonzero(k)==0

def test_history_16_2_1_layout_and_causal():
    h=HistoryManager(); src=torch.zeros(1); h.set_source(src)
    h.append_chunk([torch.tensor(float(i)) for i in range(33)])
    slots=h.slots(); assert slots[0] is src and len(slots)==20
    assert max(h.indices()['long'])<=32

def test_memory_is_kv_only_and_eviction():
    m=LongTermKVMemory(budget=2,pool=1); x=torch.randn(1,3,4); r=torch.randn(1,3,7); m.capture(x,r,0); assert len(m)==2; k,v=m.get(); assert k.shape[-1]==4 and v.shape[-1]==7

