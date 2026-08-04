#!/usr/bin/env python3
"""Validate an already materialized canonical WAH sample."""
import argparse

from long_video.wah.sample import load_wah_conditioning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir")
    args = parser.parse_args()
    sample = load_wah_conditioning(args.sample_dir)
    print({
        "camera_poses": sample["camera_poses"].shape,
        "visibility": sample["warp_visibility_mask"].shape,
        "confidence": sample["warp_confidence_mask"].shape,
        "source": sample["warp_source"].shape,
    })


if __name__ == "__main__":
    main()
