#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_video.wah.token_confidence import build_token_confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.wah_root).resolve()))
    import helios.diffusers_version.transformer_helios_diffusers as helios

    confidence = torch.tensor(
        [[
            [[0.25, 0.9, 1.0, 1.0], [0.1, 0.8, 1.0, 1.0],
             [0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]
        ]],
        dtype=torch.float32,
    )
    visibility = torch.tensor(
        [[
            [[1, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ]],
        dtype=torch.float32,
    )
    ours, ours_visible = build_token_confidence(
        confidence, visibility, actual_vae_layout=(1, 4, 4),
        actual_patch_layout=(1, 2, 2), temporal_scale=1,
    )
    official = helios.pool_history_confidence(confidence, visibility, (1, 2, 2))
    official_visible = helios.pool_history_visible_mask(visibility, (1, 2, 2))
    torch.testing.assert_close(ours, official)
    torch.testing.assert_close(ours_visible, official_visible)

    captured = []
    def dispatch(query, key, value, attn_mask=None, **kwargs):
        captured.append(attn_mask)
        return torch.zeros_like(query)
    helios.dispatch_attention_fn = dispatch

    class Identity:
        def __call__(self, value):
            return value
    attention = SimpleNamespace(
        to_q=Identity(), to_k=Identity(), to_v=Identity(),
        norm_q=Identity(), norm_k=Identity(), to_out=[Identity(), Identity()],
        heads=2, fused_projections=False, is_cross_attention=False,
        is_amplify_history=False, history_key_boost_mask=None,
        history_key_boost_scale=1.0,
    )
    processor = helios.HeliosAttnProcessor()
    hidden = torch.randn(1, 3, 4)
    attention.history_key_bias = None
    processor(attention, hidden, original_context_length=1)
    assert captured[-1] is None
    attention.history_key_bias = torch.tensor([[0.0, 0.0, 2 * torch.log(torch.tensor(0.25))]])
    processor(attention, hidden, original_context_length=1)
    bias = captured[-1]
    assert bias.shape == (1, 1, 1, 3)
    assert bias[0, 0, 0, 0] == 0 and bias[0, 0, 0, 2] < 0
    print("token mapping: passed")
    print("confidence=1 baseline mask: None")
    print("low-confidence key bias:", bias.flatten().tolist())


if __name__ == "__main__":
    main()
