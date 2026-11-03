import pytest

from training import budget as budget_module
from training.budget import Budget


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 1000.0
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: Clock.now)
    return Clock


def test_positions_budget_counts_steps():
    budget = Budget(batch_size=100, positions=350)

    assert budget.max_steps == 3
    budget.step_done()
    budget.step_done()
    assert budget.progress() == pytest.approx(2 / 3) and not budget.finished()
    budget.step_done()
    assert budget.finished() and budget.positions == 300
    assert budget.describe() == "0.00035M"


def test_the_stopwatch_does_not_count_the_first_step(clock):
    budget = Budget(batch_size=100, positions=10_000)

    clock.now += 25
    budget.step_done()
    assert budget.seconds == 0
    clock.now += 30
    budget.step_done()
    assert budget.seconds == 30


def test_the_stopwatch_does_not_count_evaluation(clock):
    budget = Budget(batch_size=100, positions=10_000)
    budget.step_done()
    clock.now += 20
    budget.step_done()

    budget.pause()
    clock.now += 500
    budget.resume()

    clock.now += 40
    budget.step_done()
    assert budget.seconds == pytest.approx(60)
