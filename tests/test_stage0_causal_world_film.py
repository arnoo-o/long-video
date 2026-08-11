from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from long_video.types import WarpBatch
from long_video.wah.adapter import WAHAdapter
from long_video.wah.stage0_causal_world_film import (
    CausalTrainingContract,
    Stage0CausalWorldFiLM,
    freeze_for_stage0_film_training,
    install_stage0_causal_world_film,
    renderer_visibility_to_latent,
)
from long_video.memory.memory_manager import MemoryManager
from long_video.training.stage0_causal_world import freeze_causal_world_training_stack


def test_zero_initialization_is_exact_identity():
    film = Stage0CausalWorldFiLM()
    latent = torch.randn(2, 16, 9, 3, 5)
    world = torch.randn_like(latent)
    visibility = torch.rand(2, 1, 9, 3, 5)
    assert torch.equal(film(latent, world, visibility), latent)


def test_zero_visibility_is_always_exact_identity():
    film = Stage0CausalWorldFiLM()
    nn.init.normal_(film.output.weight)
    nn.init.normal_(film.output.bias)
    latent = torch.randn(1, 16, 9, 3, 5)
    assert torch.equal(film(latent, torch.randn_like(latent), torch.zeros(1, 1, 9, 3, 5)), latent)


def test_only_film_parameters_require_grad_and_hook_is_stage0_only():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.patch_embedding = nn.Identity()

    transformer = Transformer()
    controller = install_stage0_causal_world_film(transformer)
    names = freeze_for_stage0_film_training(transformer)
    assert {name for name, value in transformer.named_parameters() if value.requires_grad} == set(names)
    world = torch.randn(1, 16, 9, 3, 5)
    controller.set_context(world, torch.ones(1, 1, 9, 3, 5))
    stage0 = torch.randn_like(world)
    assert torch.equal(transformer.patch_embedding(stage0), stage0)
    stage1 = torch.randn(1, 16, 9, 6, 10)
    assert torch.equal(transformer.patch_embedding(stage1), stage1)
    assert controller.applied_calls == 1


def test_full_training_stack_freezes_vae_and_pi3():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.patch_embedding = nn.Identity()

    pipe = SimpleNamespace(transformer=Transformer(), vae=nn.Linear(4, 4))
    pi3 = SimpleNamespace(_model=nn.Linear(4, 4))
    names, parameters = freeze_causal_world_training_stack(pipe, pi3)
    assert names
    assert all(parameter.requires_grad for parameter in parameters)
    assert not any(parameter.requires_grad for parameter in pipe.vae.parameters())
    assert not any(parameter.requires_grad for parameter in pi3._model.parameters())


def test_original_wah_conditioning_contract_is_unchanged():
    rgb = np.random.default_rng(0).random((33, 8, 12, 3), dtype=np.float32)
    visibility = np.random.default_rng(1).random((33, 8, 12)) > 0.4
    confidence = np.random.default_rng(2).random((33, 8, 12), dtype=np.float32)
    warp = WarpBatch(rgb, np.ones((33, 8, 12), np.float32), visibility, confidence,
                     np.zeros((33, 8, 12), np.int8), visibility.reshape(33, -1).mean(1))
    inputs = WAHAdapter.warp_inputs(warp)
    np.testing.assert_array_equal(inputs["warp_video"], rgb)
    np.testing.assert_array_equal(inputs["warp_visibility_mask"], visibility[None, None])
    np.testing.assert_array_equal(inputs["warp_confidence_mask"], (confidence * visibility)[None, None])


def test_real_vae_temporal_visibility_groups_are_continuous():
    visibility = np.zeros((33, 4, 4), np.float32)
    visibility[0] = 1
    visibility[1:3] = 1
    grouped = renderer_visibility_to_latent(
        visibility, latent_frames=9, latent_height=2, latent_width=2,
    )
    assert grouped.shape == (1, 1, 9, 2, 2)
    assert torch.equal(grouped[:, :, 0], torch.ones_like(grouped[:, :, 0]))
    assert torch.equal(grouped[:, :, 1], torch.full_like(grouped[:, :, 1], 0.5))


def test_future_supervision_is_rejected():
    CausalTrainingContract(31, 32, False).validate()
    with pytest.raises(ValueError, match="future GT"):
        CausalTrainingContract(31, 32, True).validate()
    with pytest.raises(ValueError, match="overlaps"):
        CausalTrainingContract(32, 32, False).validate()


def test_memory_manager_rejects_future_supervision_fields():
    manager = MemoryManager()
    with pytest.raises(ValueError, match="cannot receive supervision"):
        manager.process_chunk(
            SimpleNamespace(node_id="node_000"), np.zeros((1, 1, 1, 3), np.uint8),
            None, SimpleNamespace(coverage_per_frame=np.zeros(1)), 0,
            target_rgb_for_loss=np.zeros((1, 1, 1, 3), np.uint8),
        )


def test_pinned_patch_has_no_legacy_spatial_attention_call_path():
    from pathlib import Path
    patch = (Path(__file__).resolve().parents[1] / "patches" / "wah_confidence.patch").read_text(
        encoding="utf-8"
    )
    assert "_spatial_controller_ref" not in patch
    assert "controller.spatial_attention" not in patch
