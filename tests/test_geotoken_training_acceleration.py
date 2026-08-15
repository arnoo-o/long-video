import inspect
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from long_video.geometry.geotoken import GeoTokenConditioner
from long_video.online.pipeline import OnlineSpatialHistoryPipeline
from scripts import train_geotoken_dl3dv as training


class _VideoProcessor:
    def __init__(self):
        self.calls = 0

    def postprocess_video(self, value, output_type):
        assert output_type == "np"
        self.calls += 1
        return value.detach().cpu().permute(0, 2, 3, 4, 1).numpy()


def test_history_video_reuses_wah_decode_and_matches_legacy_path():
    torch.manual_seed(9)
    latents = torch.randn(1, 4, 9, 2, 3)
    decoded = torch.randn(1, 3, 33, 4, 5)

    class Pipe:
        def __init__(self):
            self.vae = SimpleNamespace(dtype=torch.float32)
            self.video_processor = _VideoProcessor()
            self.decode_calls = 0

        def _decode_autoregressive_latents(self, diffusion_latents, vae_latents):
            assert diffusion_latents is latents
            self.decode_calls += 1
            return decoded

    pipe = Pipe()
    mean = torch.zeros(1, 4, 1, 1, 1)
    std = torch.ones(1, 4, 1, 1, 1)
    legacy_decoded = pipe._decode_autoregressive_latents(latents, latents / std + mean)
    legacy = OnlineSpatialHistoryPipeline._video_array(
        pipe.video_processor.postprocess_video(legacy_decoded, output_type="np")
    )
    assert pipe.decode_calls == 1

    online = object.__new__(OnlineSpatialHistoryPipeline)
    online.wah_pipeline = pipe
    online.autoregressive_state = {
        "last_latents": latents,
        "latents_mean": mean,
        "latents_std": std,
        "history_video": decoded,
    }
    current = online._current_history_video_chunk()
    assert np.allclose(current, legacy, atol=1e-6)
    assert pipe.decode_calls == 1
    assert current.shape[0] == 33

    source = inspect.getsource(OnlineSpatialHistoryPipeline._generate_cameras)
    assert 'output_type="latent"' in source
    assert "_decode_autoregressive_latents" not in source


def test_source_rgb_identity_is_cached_per_trajectory(monkeypatch):
    calls = []
    monkeypatch.setattr(training, "file_sha256", lambda path: calls.append(path) or f"hash:{path}")
    cache = {"source_rgb_sha256": {}}
    assert training.cached_source_rgb_sha256(cache, "a", "source-a") == "hash:source-a"
    assert training.cached_source_rgb_sha256(cache, "a", "source-a") == "hash:source-a"
    assert training.cached_source_rgb_sha256(cache, "b", "source-b") == "hash:source-b"
    assert calls == ["source-a", "source-b"]


def test_prompt_embeddings_are_encoded_once_and_reused():
    class Pipe:
        _execution_device = torch.device("cpu")
        do_classifier_free_guidance = False
        transformer = SimpleNamespace(dtype=torch.float32)

        def __init__(self):
            self.calls = []

        def encode_prompt(self, *, prompt, **kwargs):
            self.calls.append(prompt)
            value = torch.full((1, 2, 3), float(len(self.calls)))
            return value, None

        @staticmethod
        def _add_prompt_trigger(prompt, trigger):
            return f"{trigger} {prompt}"

    pipe = Pipe()
    cache = training.build_prompt_embedding_cache(
        pipe, "fixed prompt", negative_prompt="negative", lora_prompt_trigger="trigger",
    )
    assert len(pipe.calls) == 2
    assert cache["prompt_embeds"].shape == cache["lora_prompt_embeds"].shape == (1, 2, 3)
    assert cache["negative_prompt_embeds"] is None
    assert all(not value.requires_grad for value in cache.values() if torch.is_tensor(value))


def test_unsampled_forward_does_not_publish_python_diagnostics():
    module = GeoTokenConditioner(16)
    module.set_diagnostics_enabled(False)
    assert module.diagnostics == {}
    assert not training.should_sample_diagnostics(2, (), smoke_only=False)
    assert training.should_sample_diagnostics(1, (), smoke_only=False)
    assert training.should_sample_diagnostics(10, (), smoke_only=False)
    assert training.should_sample_diagnostics(80, ("checkpoint_step_0080.pt",), smoke_only=False)
