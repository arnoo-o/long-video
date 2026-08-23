import pytest
torch=pytest.importorskip('torch')

def test_camera_value_residual_is_frame_level_and_does_not_touch_memory_value():
    from long_video.sightline.conditioning import CameraValueResidual
    module=CameraValueResidual(4)
    rays=torch.randn(1,8,7,requires_grad=True)
    value=module(rays,temporal_tokens=2)
    assert value.shape==(1,8,4)
    # Tokens within a temporal frame share the camera feature; frames differ.
    assert torch.equal(value[:,0],value[:,3]) and not torch.equal(value[:,0],value[:,4])
    value.square().mean().backward()
    assert rays.grad is not None and all(p.grad is not None for p in module.parameters())

def test_all_layer_conditioner_has_independent_geometry_and_camera_only_1_to_6():
    from long_video.training.sightline import SightlineTrainable
    trainable=SightlineTrainable(8,layers=tuple(range(40)),camera_layers=tuple(range(1,7)))
    assert len(trainable.conditioner.layers)==40
    assert set(trainable.conditioner.camera_residuals)=={str(x) for x in range(1,7)}
    assert trainable.conditioner.for_layer(0).q_proj is not trainable.conditioner.for_layer(1).q_proj
    assert all(alpha.ndim==0 and float(alpha)==1.0 for alpha in trainable.conditioner.alpha_parameters())

def test_attention_install_prefers_nonempty_pinned_blocks_container():
    from long_video.training.sightline import install_lora
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.to_q=torch.nn.Linear(4,4); self.to_k=torch.nn.Linear(4,4); self.to_v=torch.nn.Linear(4,4)
            self.to_out=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Identity()]); self.fused_projections=False; self.to_qkv=None
        def unfuse_projections(self): self.fused_projections=False; self.to_qkv=None
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.attn1=Attention()
    class Transformer(torch.nn.Module):
        def __init__(self): super().__init__(); self.transformer_blocks=torch.nn.ModuleList(); self.blocks=torch.nn.ModuleList([Block()])
    model=Transformer()
    assert install_lora(model,(0,),rank=16)==(0,)
