import numpy as np

from long_video.oracle_training import causal_warp
from long_video.types import CameraBatch, WarpBatch


class _Node:
    node_id = "node_000"
    parent_id = None
    created_frame = 0


class _Store:
    def load(self, node_id):
        assert node_id == "node_000"
        return _Node()


class _Manager:
    def __init__(self):
        self.nodes = {}
        self.reactivation_calls = 0

    def register(self, node):
        self.nodes[node.node_id] = node

    def maybe_reactivate(self, node, cameras):
        self.reactivation_calls += 1
        assert len(cameras.c2w) == 2
        return node, None


def test_causal_warp_provenance_is_m0_only_and_no_future_gt(monkeypatch):
    def fake_render(node, cameras, **kwargs):
        shape = (len(cameras.c2w), 2, 2)
        return WarpBatch(
            rgb=np.zeros(shape + (3,), np.float32), depth=np.ones(shape, np.float32),
            visibility=np.ones(shape, bool), confidence=np.ones(shape, np.float32),
            source=np.zeros(shape, np.int8), coverage_per_frame=np.ones(len(cameras.c2w), np.float32),
        )

    monkeypatch.setattr(causal_warp, "render", fake_render)
    renderer = causal_warp.CausalActiveNodeRenderer(_Store(), manager=_Manager())
    result = renderer.render(
        CameraBatch(np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
                    np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0), 2, 2),
        frame_start=32,
    )
    assert result.provenance["causal"] is True
    assert result.provenance["uses_future_gt"] is False
    assert result.provenance["node_mode"] == "M0-only"
    assert result.provenance["available_node_ids"] == ["node_000"]
    assert result.warp.rgb_content_origin[0, 0, 0] == "oracle_source"


def test_explicit_scheduled_render_node_disables_reactivation(monkeypatch):
    monkeypatch.setattr(causal_warp, "render", lambda node, cameras, **kwargs: WarpBatch(
        rgb=np.zeros((1, 2, 2, 3), np.float32),
        depth=np.ones((1, 2, 2), np.float32),
        visibility=np.ones((1, 2, 2), bool),
        confidence=np.ones((1, 2, 2), np.float32),
        source=np.zeros((1, 2, 2), np.int8),
        coverage_per_frame=np.ones(1, np.float32),
    ))
    manager = _Manager()
    renderer = causal_warp.CausalActiveNodeRenderer(_Store(), manager=manager)
    result = renderer.render(
        CameraBatch(np.eye(4, dtype=np.float32)[None], np.eye(3, dtype=np.float32)[None], 2, 2),
        frame_start=32,
        allow_reactivation=False,
    )
    assert manager.reactivation_calls == 0
    assert result.provenance["selection"] == "explicit_scheduled_render_node"
