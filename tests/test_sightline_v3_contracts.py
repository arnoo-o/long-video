import pytest
import torch
import torch.nn.functional as F
from torch import nn
from long_video.sightline.conditioning import SightlineConditioner
from long_video.training.sightline import curriculum_phase, install_lora, LoRALinear

def _dense(c,rays,kind):
    proj=c.q_proj if kind=='q' else c.k_proj; norm=c.rms_norm_q if kind=='q' else c.rms_norm_k; alpha=c.alpha_q if kind=='q' else c.alpha_k
    flat=c._ordered_rays(rays,kind).reshape(-1,7); return (alpha*c.gate(flat[:,6:7]).sigmoid()*norm(F.linear(flat,proj.weight,proj.bias))).reshape(*rays.shape[:-1],c.inner_dim)

def test_zero_initialized_geometry_and_affine_rms_contract():
    c=SightlineConditioner(8,alpha_init=.7); rays=torch.randn(2,5,7)
    assert torch.equal(c.project(rays,kind='q'),torch.zeros(2,5,8))
    assert c.alpha_q.item()==pytest.approx(.7) and c.alpha_k.item()==pytest.approx(.7)
    assert c.q_proj.bias is not None and c.k_proj.bias is not None and c.gate.bias is not None
    assert torch.count_nonzero(c.q_proj.bias)==0 and torch.count_nonzero(c.gate.bias)==0
    assert c.rms_norm_q.eps==1e-6 and c.rms_norm_q.weight.requires_grad

@pytest.mark.parametrize('kind',['q','k'])
def test_blocked_affine_rms_matches_dense_forward_and_gradients(kind):
    torch.manual_seed(7); a=SightlineConditioner(9); b=SightlineConditioner(9); a.q_proj.weight.data.normal_(); a.q_proj.bias.data.normal_(); a.k_proj.weight.data.normal_(); a.k_proj.bias.data.normal_(); a.gate.weight.data.normal_(); a.gate.bias.data.normal_(); a.rms_norm_q.weight.data.uniform_(.5,1.5); a.rms_norm_k.weight.data.uniform_(.5,1.5); b.load_state_dict(a.state_dict()); a.token_tile=17
    ra=torch.randn(2,37,7,requires_grad=True); rb=ra.detach().clone().requires_grad_(); up=torch.randn(2,37,9)
    (a.project(ra,kind=kind)*up).sum().backward(); (_dense(b,rb,kind)*up).sum().backward()
    assert torch.allclose(a.project(ra.detach(),kind=kind),_dense(b,rb.detach(),kind),atol=3e-6,rtol=3e-6)
    assert torch.allclose(ra.grad,rb.grad,atol=5e-6,rtol=5e-6)
    for name in ('q_proj.weight','q_proj.bias','k_proj.weight','k_proj.bias','gate.weight','gate.bias','rms_norm_q.weight','rms_norm_k.weight','alpha_q','alpha_k'):
        pa=dict(a.named_parameters())[name]; pb=dict(b.named_parameters())[name]
        if pa.grad is not None: assert torch.allclose(pa.grad,pb.grad,atol=6e-6,rtol=6e-6)

class _Attention(nn.Module):
    def __init__(self): super().__init__(); self.to_q=nn.Linear(4,4); self.to_k=nn.Linear(4,4); self.to_v=nn.Linear(4,4); self.to_out=nn.ModuleList([nn.Linear(4,4),nn.Identity()])
    def unfuse_projections(self): pass
class _Block(nn.Module):
    def __init__(self): super().__init__(); self.attn1=_Attention()
class _Transformer(nn.Module):
    def __init__(self): super().__init__(); self.transformer_blocks=nn.ModuleList([_Block()])
def test_lora_wraps_qkvo():
    m=_Transformer(); install_lora(m,[0],rank=16); a=m.transformer_blocks[0].attn1
    assert all(isinstance(getattr(a,n),LoRALinear) for n in ('to_q','to_k','to_v')) and isinstance(a.to_out[0],LoRALinear)
def test_curriculum_boundaries():
    assert curriculum_phase(299)['max_chunks']==1 and curriculum_phase(300)['max_chunks']==2
    assert not curriculum_phase(399)['lora'] and curriculum_phase(400)['lora']
    assert curriculum_phase(999)['name']=='P2' and curriculum_phase(1000)['name']=='P3'
