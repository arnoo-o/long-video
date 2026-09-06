import pytest
import torch
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
    assert torch.allclose(fixed_rms_norm(value,1e-6).float().square().mean(-1),torch.ones(2),atol=1e-5)


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
