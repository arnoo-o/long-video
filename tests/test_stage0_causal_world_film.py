from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from long_video.memory.memory_manager import MemoryManager
from long_video.training.stage0_causal_world import freeze_causal_world_training_stack
from long_video.types import WarpBatch
from long_video.wah.adapter import WAHAdapter
from long_video.wah.stage0_causal_world_film import (
    CausalTrainingContract, PointFiLMHead, fixed_source_scale,
    freeze_for_stage0_film_training, install_stage0_causal_world_film,
    scheduler_aligned_point_feature, world_xyz_to_fixed_source,
)


def test_zero_initialization_and_zero_visibility_are_exact_noops():
    head = PointFiLMHead()
    latent = torch.randn(1, 16, 9, 12, 20)
    feature = torch.randn_like(latent); sigma_feature = torch.randn_like(latent)
    visible = torch.rand(1, 1, 9, 12, 20)
    assert torch.equal(head(latent, feature, sigma_feature, visible, torch.tensor([.5])), latent)
    nn.init.normal_(head.output.weight); nn.init.normal_(head.output.bias)
    assert torch.equal(head(latent, feature, sigma_feature, torch.zeros_like(visible), .5), latent)


def test_scheduler_alignment_reuses_exact_native_stage0_start_point():
    feature = torch.randn(1, 16, 9, 12, 20)
    native_start = torch.randn_like(feature); sigma = torch.tensor([.7])
    aligned = scheduler_aligned_point_feature(feature, native_start, sigma, end_sigma=.25)
    endpoint = .25 * native_start + .75 * feature
    expected = .7 * native_start + .3 * endpoint
    torch.testing.assert_close(aligned, expected)


def test_only_point_encoder_and_film_head_train_and_stage1_stage2_do_not_apply():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = nn.Linear(4, 4); self.patch_embedding = nn.Identity()
    transformer = Transformer(); controller = install_stage0_causal_world_film(transformer)
    names = freeze_for_stage0_film_training(transformer)
    assert {n for n, p in transformer.named_parameters() if p.requires_grad} == set(names)
    assert all("point_encoder" in n or "film_head" in n for n in names)
    feature = torch.randn(1, 16, 9, 12, 20); controller.set_point_context(feature, torch.ones(1, 1, 9, 12, 20))
    controller.set_training_schedule({"stage_id": 0, "start_point": torch.randn_like(feature),
                                      "sigmas": torch.tensor([.5])}, .25)
    transformer.patch_embedding(feature)
    before = controller.applied_calls
    transformer.patch_embedding(torch.randn(1, 16, 9, 24, 40))
    transformer.patch_embedding(torch.randn(1, 16, 9, 48, 80))
    assert controller.applied_calls == before


def test_source_coordinate_frame_and_scale_are_fixed_values():
    pose = np.eye(4, dtype=np.float32); pose[:3, 3] = [1, 2, 3]
    scale = fixed_source_scale(np.array([[1, 2, np.nan], [3, 0, 4]], np.float32))
    assert scale == 2.5
    xyz = world_xyz_to_fixed_source(np.array([[3.5, 2, 3]], np.float32), pose, scale)
    np.testing.assert_allclose(xyz, [[1, 0, 0]])


def test_full_stack_freezes_helios_vae_and_pi3():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = nn.Linear(4, 4); self.patch_embedding = nn.Identity()
    pipe = SimpleNamespace(transformer=Transformer(), vae=nn.Linear(4, 4))
    pi3 = SimpleNamespace(_model=nn.Linear(4, 4))
    names, parameters = freeze_causal_world_training_stack(pipe, pi3)
    assert names and all(p.requires_grad for p in parameters)
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in pi3._model.parameters())


def test_original_wah_conditioning_is_unchanged():
    rng = np.random.default_rng(0); rgb = rng.random((33, 8, 12, 3), dtype=np.float32)
    visibility = rng.random((33, 8, 12)) > .4; confidence = rng.random((33, 8, 12), dtype=np.float32)
    warp = WarpBatch(rgb, np.ones((33, 8, 12), np.float32), visibility, confidence,
                     np.zeros((33, 8, 12), np.int8), visibility.reshape(33, -1).mean(1))
    inputs = WAHAdapter.warp_inputs(warp)
    np.testing.assert_array_equal(inputs["warp_video"], rgb)
    np.testing.assert_array_equal(inputs["warp_visibility_mask"], visibility[None, None])
    np.testing.assert_array_equal(inputs["warp_confidence_mask"], (confidence * visibility)[None, None])


def test_future_supervision_cannot_enter_world_or_history():
    CausalTrainingContract(31, 32, False).validate()
    with pytest.raises(ValueError, match="future GT"): CausalTrainingContract(31, 32, True).validate()
    manager = MemoryManager()
    with pytest.raises(ValueError, match="cannot receive supervision"):
        manager.process_chunk(SimpleNamespace(node_id="node_000"), np.zeros((1, 1, 1, 3), np.uint8),
                              None, SimpleNamespace(coverage_per_frame=np.zeros(1)), 0,
                              target_rgb_for_loss=np.zeros((1, 1, 1, 3), np.uint8))
