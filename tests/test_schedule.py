import pytest

from training.train import DECAY_FRACTION, WARMUP_STEPS, lr_multiplier

AFTER_WARMUP = WARMUP_STEPS + 1


@pytest.mark.parametrize("schedule", ["cosine", "hold", "hold_sqrt", "constant"])
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


def test_hold_sqrt_stays_at_the_peak_then_falls_as_one_minus_the_root_of_the_decay_done():
    decay_start = 1 - DECAY_FRACTION

    assert lr_multiplier(AFTER_WARMUP, decay_start - 0.01, "hold_sqrt") == 1.0
    assert lr_multiplier(AFTER_WARMUP, decay_start + DECAY_FRACTION / 4, "hold_sqrt") == pytest.approx(0.5)
    assert lr_multiplier(AFTER_WARMUP, 1.0, "hold_sqrt") == pytest.approx(0.0)


@pytest.mark.parametrize("schedule", ["hold", "hold_sqrt"])
def test_the_decay_fraction_moves_where_the_decay_starts(schedule):
    assert lr_multiplier(AFTER_WARMUP, 0.7, schedule, decay_fraction=0.35) < 1.0
    assert lr_multiplier(AFTER_WARMUP, 0.7, schedule, decay_fraction=0.2) == 1.0
    assert lr_multiplier(AFTER_WARMUP, 1.0, schedule, decay_fraction=0.35) == pytest.approx(0.0)


def test_a_resumed_run_with_a_larger_budget_holds_until_its_new_decay():
    #the d768 continuation: the step a pre-decay state was saved at, in a run twice as long, is still at the peak
    from training.budget import Budget
    from training.train_ddp import make_scheduler
    import torch

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    budget = Budget(4104, positions=12_820_000_000)
    scheduler = make_scheduler(optimizer, budget, "hold_sqrt", 0.35)
    budget.steps = 1_015_000
    assert scheduler.lr_lambdas[0](budget.steps) == 1.0
    budget.steps = budget.max_steps
    assert scheduler.lr_lambdas[0](budget.steps) == pytest.approx(0.0)


def test_constant_never_decays():
    assert lr_multiplier(AFTER_WARMUP, 1.0, "constant") == 1.0


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        lr_multiplier(AFTER_WARMUP, 0.5, "linear")
