import pytest

from tools import budget as budget_module
from tools.budget import Budget


#a clock the test moves by hand: clock.now += seconds
@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 1000.0
    monkeypatch.setattr(budget_module.time, "perf_counter", lambda: Clock.now)
    return Clock


def test_positions_budget_counts_steps():
    budget = Budget(batch_size=100, positions=350)  #3 whole batches

    assert not budget.timed and budget.total == 3
    assert [budget.step_done() for _ in range(2)] == [1, 1]
    assert budget.progress() == pytest.approx(2 / 3) and not budget.finished()
    budget.step_done()
    assert budget.finished() and budget.positions == 300
    assert budget.describe() == "0.00035M"


def test_minutes_budget_does_not_count_the_first_step(clock):
    budget = Budget(batch_size=100, minutes=1)

    clock.now += 25  #the first step is where torch.compile compiles
    assert budget.step_done() == 0
    clock.now += 30
    assert budget.step_done() == 30
    assert budget.progress() == pytest.approx(0.5) and budget.describe() == "1min"


def test_minutes_budget_does_not_count_evaluation(clock):
    budget = Budget(batch_size=100, minutes=1)
    budget.step_done()
    clock.now += 20
    budget.step_done()

    budget.pause()
    clock.now += 500  #a long evaluation
    budget.resume()

    clock.now += 40
    budget.step_done()
    assert budget.seconds == pytest.approx(60) and budget.finished()


def test_budget_needs_exactly_one_of_positions_and_minutes():
    with pytest.raises(ValueError):
        Budget(batch_size=100)
    with pytest.raises(ValueError):
        Budget(batch_size=100, positions=1000, minutes=1)
