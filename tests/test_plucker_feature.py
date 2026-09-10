import torch

from long_video.sightline.rays import encode_plucker_feature


def test_plucker_zero_moment_is_finite_and_has_zero_tail():
    origin = torch.zeros(2, 3, requires_grad=True)
    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    feature = encode_plucker_feature(origin, direction, 'q')
    assert torch.isfinite(feature).all()
    assert torch.equal(feature[:, 3:], torch.zeros_like(feature[:, 3:]))
    feature.sum().backward()
    assert torch.isfinite(origin.grad).all()
    assert torch.isfinite(direction.grad).all()


def test_plucker_small_moment_is_continuous_and_finite():
    direction = torch.tensor([[0.0, 0.0, 1.0]])
    origin_a = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    origin_b = torch.tensor([[1e-7, -2e-7, 0.0]], requires_grad=True)
    a = encode_plucker_feature(origin_a, direction, 'q')
    b = encode_plucker_feature(origin_b, direction, 'q')
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.linalg.vector_norm(a - b) < 1e-5
    (b.square().sum()).backward()
    assert torch.isfinite(origin_b.grad).all()
