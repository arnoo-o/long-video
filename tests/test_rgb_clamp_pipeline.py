import torch
from long_video.wah.rgb_clamp_pipeline import (
    PYRAMID_INFERENCE_STEPS, STAGE2_CLAMP_STATES,
    clamp_enabled, composite_renderer_rgb,
)

def test_pyramid_steps_and_stage2_schedule():
    assert PYRAMID_INFERENCE_STEPS == (2, 2, 4)
    assert [[i for i in range(n)] for n in PYRAMID_INFERENCE_STEPS] == [[0,1],[0,1],[0,1,2,3]]
    assert [int(clamp_enabled(2, i)) for i in range(4)] == list(STAGE2_CLAMP_STATES)
    assert not any(clamp_enabled(stage, step) for stage in (0,1) for step in range(4))

def test_rgb_clamp_uses_raw_binary_visibility():
    model=torch.zeros(1,3,1,2,2); warp=torch.ones_like(model)
    mask=torch.tensor([[[[[1,0],[0,1]]]]],dtype=torch.float32)
    mixed=composite_renderer_rgb(model,warp,mask)
    assert torch.equal(mixed,mask.expand_as(mixed))
    bad=mask.clone(); bad[...,0,0]=0.5
    try: composite_renderer_rgb(model,warp,bad)
    except ValueError: pass
    else: raise AssertionError('soft visibility must be rejected')
