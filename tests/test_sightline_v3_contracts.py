import pytest
import torch
import torch.nn.functional as F
from torch import nn
from long_video.sightline.conditioning import SightlineConditioner
from long_video.sightline.geometry import geometry_sigma_schedule
from long_video.training.sightline import curriculum_phase, install_lora, LoRALinear

def _dense(c,rays,native,kind):
    proj=c.q_proj if kind=='q' else c.k_proj
    beta=c.beta_q if kind=='q' else c.beta_k
    raw=F.linear(rays.reshape(-1,7).float(),proj.weight.float())
    normalized=raw/torch.sqrt(raw.square().mean(-1,keepdim=True)+0.005**2)
    native_rms=native.detach().float().square().mean(-1,keepdim=True).sqrt()
    return (0.5*beta.float().sigmoid()*native_rms.reshape(-1,1)*normalized).to(native.dtype).reshape(*rays.shape[:-1],c.inner_dim)


def test_zero_initialized_geometry_and_affine_rms_contract():
    c=SightlineConditioner(8,rho_init=.4); rays=torch.randn(2,5,7); native=torch.randn(2,5,8)
    assert torch.equal(c.project(rays,native,kind='q'),torch.zeros(2,5,8))
    assert c.rho_values()[0].item()==pytest.approx(.4) and c.rho_values()[1].item()==pytest.approx(.4)
    assert c.q_proj.bias is None and c.k_proj.bias is None
    assert not hasattr(c,'gate') and not hasattr(c,'rms_norm_q') and not hasattr(c,'rms_norm_k')
    assert set(dict(c.named_parameters()))=={'q_proj.weight','k_proj.weight','beta_q','beta_k'}

@pytest.mark.parametrize('kind',['q','k'])
def test_blocked_affine_rms_matches_dense_forward_and_gradients(kind):
    torch.manual_seed(7); a=SightlineConditioner(9); b=SightlineConditioner(9)
    a.q_proj.weight.data.normal_(); a.k_proj.weight.data.normal_(); a.beta_q.data.fill_(-.3); a.beta_k.data.fill_(.2)
    b.load_state_dict(a.state_dict()); a.token_tile=17; b.token_tile=17
    ra=torch.randn(2,37,7,requires_grad=True); rb=ra.detach().clone().requires_grad_(); na=torch.randn(2,37,9); nb=na.detach().clone(); up=torch.randn(2,37,9)
    (a.project(ra,na,kind=kind)*up).sum().backward(); (_dense(b,rb,nb,kind)*up).sum().backward()
    assert torch.allclose(a.project(ra.detach(),na,kind=kind),_dense(b,rb.detach(),nb,kind),atol=3e-6,rtol=3e-6)
    assert torch.allclose(ra.grad,rb.grad,atol=5e-6,rtol=5e-6)
    for name in ('q_proj.weight','k_proj.weight','beta_q','beta_k'):
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

def test_relative_rms_is_token_local_and_batch_independent():
    torch.manual_seed(17)
    c=SightlineConditioner(8); c.q_proj.weight.data.normal_()
    rays=torch.randn(2,7,7); native=torch.randn(2,7,8)
    baseline=c.project(rays,native,kind='q')
    changed_native=native.clone(); changed_native[0,0].mul_(7.0)
    changed=c.project(rays,changed_native,kind='q')
    assert (baseline[0,0]-changed[0,0]).abs().sum()>0
    assert torch.allclose(baseline[0,1:],changed[0,1:])
    assert torch.allclose(baseline[1],changed[1])
    single=c.project(rays[:1],native[:1],kind='q')
    assert torch.allclose(single,baseline[:1],atol=2e-6,rtol=2e-6)


def test_relative_native_rms_scale_is_detached():
    torch.manual_seed(19)
    c=SightlineConditioner(8); c.q_proj.weight.data.normal_()
    rays=torch.randn(2,9,7,requires_grad=True); native=torch.randn(2,9,8,requires_grad=True)
    c.project(rays,native,kind='q').square().mean().backward()
    assert native.grad is None
    assert rays.grad is not None and torch.isfinite(rays.grad).all()

def test_geometry_sigma_contract_is_continuous_across_three_pyramid_stages():
    # Pinned Helios stages meet at their endpoints; Geometry must follow the
    # absolute sigma trajectory rather than restarting from each local stage.
    endpoints=((1.0,.7),(.7,.35),(.35,0.0))
    sigma_trace=[]; scale_trace=[]
    for sigma_start,sigma_end in endpoints:
        sigma_local=torch.linspace(1.0,0.0,17)
        sigma_abs,scale=geometry_sigma_schedule(sigma_local,sigma_start,sigma_end)
        sigma_trace.append(sigma_abs); scale_trace.append(scale)
    sigma_trace=torch.cat(sigma_trace); scale_trace=torch.cat(scale_trace)
    assert torch.all(torch.diff(sigma_trace)<=1e-6)
    assert torch.all(torch.diff(scale_trace)<=1e-6)
    assert sigma_trace[16].item()==pytest.approx(sigma_trace[17].item())
    assert sigma_trace[33].item()==pytest.approx(sigma_trace[34].item())
    assert scale_trace[0].item()==pytest.approx(1.0)
    assert scale_trace[-1].item()==pytest.approx(0.0)
