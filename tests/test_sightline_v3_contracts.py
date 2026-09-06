import pytest
import torch
import torch.nn.functional as F
from torch import nn

from long_video.sightline.conditioning import SightlineConditioner, fixed_rms_norm
from long_video.training.sightline import SightlineTrainable, curriculum_phase, install_lora, LoRALinear


def test_zero_linear_geometry_is_native_noop_and_alpha_is_sigmoid():
    c=SightlineConditioner(8,alpha_init=.7)
    rays=torch.randn(2,5,7)
    assert torch.equal(c.project(rays,kind='q'),torch.zeros(2,5,8))
    assert c.alpha_q.item()==pytest.approx(.7,abs=1e-6)
    assert c.alpha_k.item()==pytest.approx(.7,abs=1e-6)
    assert 0.<c.alpha_q.item()<1. and 0.<c.alpha_k.item()<1.


def test_fixed_rms_norm_has_no_trainable_gamma_and_gate_starts_half():
    c=SightlineConditioner(8)
    assert not any('rms' in name for name,_ in c.named_parameters())
    assert torch.equal(c.gate(torch.zeros(3,1)),torch.full((3,8),.5))
    value=torch.randn(2,8)
    assert torch.allclose(fixed_rms_norm(value,1e-3).float().square().mean(-1),value.square().mean(-1)/(value.square().mean(-1)+1e-3),atol=1e-5)


def _dense_project(c,rays,kind):
    projection=c.q_proj if kind=='q' else c.k_proj; beta=c.beta_q if kind=='q' else c.beta_k
    flat=c._ordered_rays(rays,kind).reshape(-1,7)
    raw=F.linear(flat,projection.weight)
    gate=c.gate(flat[:,6:7])
    return (beta.sigmoid()*gate*fixed_rms_norm(raw,c.geometry_rms_epsilon)).reshape(*rays.shape[:-1],c.inner_dim)


def test_geometry_rms_epsilon_known_value_and_zero_projector_are_exact():
    value=torch.tensor([[3.,4.]])
    expected=value/torch.sqrt(torch.tensor([[12.501]]))
    assert torch.allclose(fixed_rms_norm(value,1e-3),expected)
    c=SightlineConditioner(8,geometry_rms_epsilon=1e-3)
    assert c.geometry_rms_epsilon==pytest.approx(1e-3)
    assert torch.equal(c.project(torch.randn(3,9,7),kind='k'),torch.zeros(3,9,8))


@pytest.mark.parametrize('kind',['q','k'])
def test_token_blocked_projection_matches_dense_forward_and_gradients(kind):
    torch.manual_seed(4)
    blocked=SightlineConditioner(9,geometry_rms_epsilon=1e-3); dense=SightlineConditioner(9,geometry_rms_epsilon=1e-3)
    blocked.q_proj.weight.data.normal_(); blocked.k_proj.weight.data.normal_(); blocked.gate[0].weight.data.normal_(); blocked.gate[0].bias.data.normal_()
    dense.load_state_dict(blocked.state_dict()); blocked.token_tile=17
    rays_a=torch.randn(2,37,7,requires_grad=True); rays_b=rays_a.detach().clone().requires_grad_(); upstream=torch.randn(2,37,9)
    out_a=blocked.project(rays_a,kind=kind); out_b=_dense_project(dense,rays_b,kind)
    (out_a*upstream).sum().backward(); (out_b*upstream).sum().backward()
    assert torch.allclose(out_a,out_b,atol=3e-6,rtol=3e-6)
    assert torch.allclose(rays_a.grad,rays_b.grad,atol=4e-6,rtol=4e-6)
    for name in ('q_proj.weight','k_proj.weight','gate.0.weight','gate.0.bias','beta_q','beta_k'):
        pa=dict(blocked.named_parameters())[name]; pb=dict(dense.named_parameters())[name]
        if pa.grad is None: assert pb.grad is None
        else: assert torch.allclose(pa.grad,pb.grad,atol=5e-6,rtol=5e-6)


def test_token_blocked_projection_large_non_divisible_token_count():
    c=SightlineConditioner(7); c.q_proj.weight.data.normal_(); c.token_tile=127
    rays=torch.randn(1,1301,7,requires_grad=True)
    out=c.project(rays,kind='q')
    assert out.shape==(1,1301,7)
    out.square().mean().backward()
    assert rays.grad is not None and c.q_proj.weight.grad is not None


def test_geometry_has_no_timestep_route():
    c=SightlineConditioner(4); c.q_proj.weight.data.normal_()
    rays=torch.randn(1,3,7)
    # The public Geometry API contains no timestep/noise argument.
    assert torch.equal(c.project(rays,kind='q'),c.project(rays,kind='q'))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.to_q=nn.Linear(4,4); self.to_k=nn.Linear(4,4); self.to_v=nn.Linear(4,4); self.to_out=nn.ModuleList([nn.Linear(4,4),nn.Identity()])
    def unfuse_projections(self): pass
class _Block(nn.Module):
    def __init__(self): super().__init__(); self.attn1=_Attention()
class _Transformer(nn.Module):
    def __init__(self): super().__init__(); self.transformer_blocks=nn.ModuleList([_Block()])

def test_lora_wraps_only_v_and_output():
    model=_Transformer(); assert install_lora(model,[0],rank=16)==(0,)
    attn=model.transformer_blocks[0].attn1
    assert isinstance(attn.to_v,LoRALinear) and isinstance(attn.to_out[0],LoRALinear)
    assert isinstance(attn.to_q,nn.Linear) and isinstance(attn.to_k,nn.Linear)


def test_v3_curriculum_boundaries():
    assert curriculum_phase(0)['name']=='P1' and curriculum_phase(499)['max_chunks']==1
    assert curriculum_phase(500)['name']=='P2a' and curriculum_phase(699)['max_chunks']==1
    assert curriculum_phase(700)['name']=='P2b' and curriculum_phase(999)['max_chunks']==2
    assert [curriculum_phase(s)['max_chunks'] for s in (1000,1499,1500,1799,1800,2099,2100,2299,2300,2499)]==[2,2,3,3,4,4,5,5,6,6]
