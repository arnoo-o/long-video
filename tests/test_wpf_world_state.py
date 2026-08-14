import torch

from long_video.wah.world_projected_pipeline import build_world_state_at_sigma


def test_world_state_interpolates_to_positive_clean_endpoint():
    stage_start = torch.tensor([2.0, -4.0])
    endpoint = torch.tensor([10.0, 6.0])

    torch.testing.assert_close(
        build_world_state_at_sigma(stage_start, endpoint, 1.0), stage_start,
    )
    torch.testing.assert_close(
        build_world_state_at_sigma(stage_start, endpoint, 0.0), endpoint,
    )
    torch.testing.assert_close(
        build_world_state_at_sigma(stage_start, endpoint, 0.25),
        0.25 * stage_start + 0.75 * endpoint,
    )
