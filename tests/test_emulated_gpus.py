"""tools/emulated_gpus.py: Searches' own round loop on fake gpus (selfplay_rs.fake_cuda_*), the harness that times the
loop on more gpus than the machine has."""
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import emulated_gpus  # noqa: E402
from engine import engine_search  # noqa: E402
from engine.engine_search import EngineSearch  # noqa: E402


#a search on two fake gpus runs its simulations through the loop, every gpu's batches answered (a loop that stalled,
#or answers the finish could not read, would end short or fail)
def test_a_search_on_fake_gpus_runs_its_simulations(monkeypatch):
    monkeypatch.setattr(engine_search, "cuda_runtime", emulated_gpus.FakeRuntime)
    networks = [emulated_gpus.FakeNetwork(index, engine_search.ROUNDS_IN_FLIGHT, 0.0, 0.0) for index in range(2)]
    search = EngineSearch(networks, rust_rounds=True, smart_pruning=False)
    search.new_game()
    report = search.search(chess.Board(emulated_gpus.POSITION), simulations=3000)
    assert report.simulations >= 3000
    batches, boards = search.rust_counts
    assert batches >= 2 and 0 < boards <= report.simulations
    assert chess.Board(emulated_gpus.POSITION).is_legal(report.line[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
