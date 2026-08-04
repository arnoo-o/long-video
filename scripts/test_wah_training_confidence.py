#!/usr/bin/env python3
"""Minimal training forward plumbing test for confidence-aware WAH."""
import argparse
import sys
from pathlib import Path

import torch


class CaptureTransformer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return (torch.zeros_like(kwargs["hidden_states"]),)


class FakePipe:
    def __init__(self):
        self.transformer = CaptureTransformer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wah-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wah_root))
    from warp_as_history.training.core import transformer_model_forward

    pipe = FakePipe()
    latent = torch.zeros(1, 16, 5, 4, 4)
    confidence = torch.ones(1, 1, 5, 4, 4)
    confidence[:, :, -2:] = 0.25
    histories = {
        "indices_hidden_states": torch.zeros(1, 5, 3, dtype=torch.long),
        "indices_latents_history_short": torch.zeros(1, 5, 3, dtype=torch.long),
        "indices_latents_history_mid": None,
        "indices_latents_history_long": None,
        "latents_history_short": latent,
        "latents_history_mid": None,
        "latents_history_long": None,
        "history_visible_mask_short": torch.ones_like(confidence),
        "history_visible_mask_mid": None,
        "history_visible_mask_long": None,
        "history_confidence_short": confidence,
    }
    output = transformer_model_forward(
        pipe,
        latent,
        torch.ones(1),
        torch.zeros(1, 2, 8),
        histories,
        {"history_confidence_lambda": 1.0},
    )
    forwarded = pipe.transformer.kwargs["history_confidence_short"]
    assert output.shape == latent.shape
    assert forwarded.shape == confidence.shape
    assert torch.equal(forwarded, confidence)
    print("WAH training confidence forward shape passed", tuple(forwarded.shape))


if __name__ == "__main__":
    main()
