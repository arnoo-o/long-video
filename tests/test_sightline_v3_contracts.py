import pytest
import torch
import torch.nn.functional as F
from torch import nn
from long_video.sightline.conditioning import SightlineConditioner
from long_video.training.sightline import curriculum_phase, install_lora, LoRALinear

def _dense(c,rays,native,kind):
    proj=c.q_proj if kind=='q' else c.k_proj; norm=c.rms_norm_q if kind=='q' else c.rms_norm_k; beta=c.beta_q if kind=='q' else c.beta_k
    ordered=c._ordered_rays(rays,kind); flat=ordered.reshape(-1,7); raw=F.linear(flat,proj.weight,proj.bias)
    u=(c.gate(flat[:,6:7]).sigmoid()*norm(raw)).reshape(*rays.shape[:-1],c.inner_dim)
    rho=beta.sigmoid()
    reduce_dims=tuple(range(1,u.ndim))
    nrms=(native.detach().float().square().mean(dim=tuple(range(1,native.ndim)),keepdim=True)+1e-6).sqrt()
    return (rho*nrms*u).to(native.dtype)

def test_zero_initialized_geometry_and_affine_rms_contract():
    c=SightlineConditioner(8,rho_init=.4); rays=torch.randn(2,5,7); native=torch.randn(2,5,8)
    assert torch.equal(c.project(rays,native,kind='q'),torch.zeros(2,5,8))
    assert c.rho_values()[0].item()==pytest.approx(.4) and c.rho_values()[1].item()==pytest.approx(.4)
    assert c.q_proj.bias is not None and c.k_proj.bias is not None and c.gate.bias is not None
    assert torch.count_nonzero(c.q_proj.bias)==0 and torch.count_nonzero(c.gate.bias)==0
    assert c.rms_norm_q.eps==1e-4 and c.rms_norm_q.weight.requires_grad

@pytest.mark.parametrize('kind',['q','k'])
def test_blocked_affine_rms_matches_dense_forward_and_gradients(kind):
    torch.manual_seed(7); a=SightlineConditioner(9); b=SightlineConditioner(9); a.q_proj.weight.data.normal_(); a.q_proj.bias.data.normal_(); a.k_proj.weight.data.normal_(); a.k_proj.bias.data.normal_(); a.gate.weight.data.normal_(); a.gate.bias.data.normal_(); a.rms_norm_q.weight.data.uniform_(.5,1.5); a.rms_norm_k.weight.data.uniform_(.5,1.5); b.load_state_dict(a.state_dict()); a.token_tile=17
    ra=torch.randn(2,37,7,requires_grad=True); rb=ra.detach().clone().requires_grad_(); na=torch.randn(2,37,9); nb=na.detach().clone(); up=torch.randn(2,37,9)
    (a.project(ra,na,kind=kind)*up).sum().backward(); (_dense(b,rb,nb,kind)*up).sum().backward()
    assert torch.allclose(a.project(ra.detach(),na,kind=kind),_dense(b,rb.detach(),nb,kind),atol=3e-6,rtol=3e-6)
    assert torch.allclose(ra.grad,rb.grad,atol=5e-6,rtol=5e-6)
    for name in ('q_proj.weight','q_proj.bias','k_proj.weight','k_proj.bias','gate.weight','gate.bias','rms_norm_q.weight','rms_norm_k.weight','beta_q','beta_k'):
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
    assert not curriculum_phase(399)['lora'] and not curriculum_phase(400)['lora'] and curriculum_phase(1000)['correspondence']
    assert curriculum_phase(999)['name']=='P2' and curriculum_phase(1000)['name']=='P3'

def test_relative_rms_is_sample_wide_and_batch_independent():
    torch.manual_seed(17)
    c=SightlineConditioner(8); c.q_proj.weight.data.normal_(); c.q_proj.bias.data.normal_()
    c.gate.weight.data.normal_(); c.gate.bias.data.normal_(); c.rms_norm_q.weight.data.uniform_(.5,1.5)
    rays=torch.randn(2,7,7); native=torch.randn(2,7,8)
    baseline=c.project(rays,native,kind='q')
    changed_native=native.clone(); changed_native[0,0].mul_(7.0)
    changed=c.project(rays,changed_native,kind='q')
    # Changing one token's native scale changes the single sample-wide scale,
    # hence every token in that sample, but never the other batch element.
    assert torch.all((baseline[0,1:]-changed[0,1:]).abs().sum(-1)>0)
    assert torch.allclose(baseline[1],changed[1])
    single=c.project(rays[:1],native[:1],kind='q')
    assert torch.allclose(single,baseline[:1],atol=2e-6,rtol=2e-6)

def test_relative_native_rms_scale_is_detached():
    torch.manual_seed(19)
    c=SightlineConditioner(8); c.q_proj.weight.data.normal_(); c.gate.weight.data.normal_()
    rays=torch.randn(2,9,7,requires_grad=True); native=torch.randn(2,9,8,requires_grad=True)
    c.project(rays,native,kind='q').square().mean().backward()
    assert native.grad is None
    assert rays.grad is not None and torch.isfinite(rays.grad).all()
