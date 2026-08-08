import pytest

from long_video.oracle_training.history_bank import HistoryBankKey, validate_history_bank_entry


def test_history_bank_key_changes_with_checkpoint_and_config():
    base = dict(
        checkpoint_sha="a", global_step=10, scene_id="Indoor_013",
        source_id="M0", trajectory_id="window", history_chunk_index=7,
        generation_config=(1, 1, 1), prompt="room", seed=42,
    )
    first = HistoryBankKey(**base).digest()
    assert first != HistoryBankKey(**{**base, "checkpoint_sha": "b"}).digest()
    assert first != HistoryBankKey(**{**base, "generation_config": (2, 2, 2)}).digest()


def test_history_bank_rejects_future_gt():
    entry = {
        "TEMP_LONG": None, "TEMP_MID": None, "TEMP_SHORT": None,
        "key": "x", "metadata": {"uses_gt_future": False},
    }
    assert validate_history_bank_entry(entry)
    with pytest.raises(ValueError, match="future GT"):
        validate_history_bank_entry({**entry, "future_rgb": object()})
