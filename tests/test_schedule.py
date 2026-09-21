import pytest

from main import DECAY_FRACTION, WARMUP_STEPS, lr_multiplier

AFTER_WARMUP = WARMUP_STEPS + 1


@pytest.mark.parametrize("schedule", ["cosine", "hold", "constant"])
def test_every_schedule_warms_up_to_the_peak(schedule):
    assert lr_multiplier(0, 0.0, schedule) == pytest.approx(1 / WARMUP_STEPS)
    assert lr_multiplier(WARMUP_STEPS - 1, 0.0, schedule) == pytest.approx(1.0)
    assert lr_multiplier(AFTER_WARMUP, 0.0, schedule) == pytest.approx(1.0)


def test_cosine_is_halfway_down_at_half_time_and_ends_at_zero():
    assert lr_multiplier(AFTER_WARMUP, 0.5, "cosine") == pytest.approx(0.5)
    assert lr_multiplier(AFTER_WARMUP, 1.0, "cosine") == pytest.approx(0.0)


def test_hold_stays_at_the_peak_until_the_decay_and_ends_at_zero():
    decay_start = 1 - DECAY_FRACTION

    assert lr_multiplier(AFTER_WARMUP, decay_start - 0.01, "hold") == 1.0
    assert lr_multiplier(AFTER_WARMUP, decay_start + DECAY_FRACTION / 2, "hold") == pytest.approx(0.5)
    assert lr_multiplier(AFTER_WARMUP, 1.0, "hold") == pytest.approx(0.0)


def test_constant_never_decays():
    assert lr_multiplier(AFTER_WARMUP, 1.0, "constant") == 1.0


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        lr_multiplier(AFTER_WARMUP, 0.5, "linear")
