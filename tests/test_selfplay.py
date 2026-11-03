from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from training import data
from engine import selfplay
from training.data import SelfPlayDataset, load_training_set
from model.encoding import NO_MOVE, TOKENS_PER_BOARD
from engine.inference import cpu_batches, serve
from engine.selfplay import TOP_MOVES, Game, RemoteNetwork, SelfPlaySet, Slot, play_games_in_rust, slot_arrays
from fake_models import BoardModel

SETTINGS = {"simulations": 16}


def two_games() -> list[Game]:
    visit_moves = np.full((2, TOP_MOVES), NO_MOVE, dtype=np.int16)
    visit_moves[:, :2] = [[5, 9], [7, 3]]
    visit_shares = np.zeros((2, TOP_MOVES), dtype=np.float32)
    visit_shares[:, :2] = [[0.75, 0.25], [1.0, 0.0]]
    boards = np.full((2, TOKENS_PER_BOARD), 3, dtype=np.uint8)
    return [Game(0, boards, visit_moves, visit_shares, np.array([0.6, 0.3], dtype=np.float32), np.array([True, False]), 1.0),
            Game(1, boards, visit_moves, visit_shares, np.array([0.8, 0.1], dtype=np.float32), np.array([True, False]), 0.5)]


@pytest.fixture
def fresh_set(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    return SelfPlaySet(tmp_path, "played", SETTINGS)


#game 0 goes to val, game 1 to train
@pytest.fixture
def played_set(fresh_set):
    for game in two_games():
        fresh_set.add(game, 1.0)
    return fresh_set


def test_a_self_play_set_is_read_back_for_training(played_set):
    train, val = load_training_set("played", 0.02, result_weight=0.5)

    assert isinstance(train, SelfPlayDataset) and len(train) == len(val) == 2
    _, moves, shares, values, _ = val[[0, 1]]
    assert moves[:, :2].tolist() == [[5, 9], [7, 3]] and moves[:, 2:].eq(0).all()
    assert shares.tolist()[0][:2] == [0.75, 0.25]
    #white won game 0: white to move gets (1 + 0.6) / 2, black to move (0 + 0.3) / 2; game 1 was a draw
    assert values.tolist() == pytest.approx([0.8, 0.15])
    assert train.value_target_of([0, 1]).tolist() == pytest.approx([0.65, 0.3])


@pytest.mark.parametrize("temperature, expected", [(1.0, [0.75, 0.25]), (0.5, [0.9, 0.1]), (0.0, [1.0, 0.0])])
def test_the_visit_temperature_sharpens_the_policy_target(played_set, temperature, expected):
    _, val = load_training_set("played", 0.02, visit_temperature=temperature)

    _, moves, shares, _, _ = val[[0, 1]]
    assert shares[0, :2].tolist() == pytest.approx(expected) and shares[1, :2].tolist() == pytest.approx([1.0, 0.0])
    assert shares.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert moves[:, :2].tolist() == [[5, 9], [7, 3]]


#the part of a game written after the last record is cut off on reopening
def test_a_self_play_set_continues_after_an_interruption(played_set, tmp_path):
    with open(tmp_path / "played_train_boards.bin", "ab") as boards:
        boards.write(bytes(TOKENS_PER_BOARD * 3))

    reopened = SelfPlaySet(tmp_path, "played", SETTINGS)

    assert reopened.positions() == 4 and reopened.record["next_game"] == 2
    train, _ = load_training_set("played", 0.02)
    assert len(train) == 2
    with pytest.raises(ValueError):
        SelfPlaySet(tmp_path, "played", {"simulations": 64})


#windows refuses the move for a moment while another program holds the record open
def test_a_record_write_that_fails_for_a_moment_is_tried_again(fresh_set, tmp_path, monkeypatch):
    replace = Path.replace
    refusals = iter([True, True, False])
    def refuse_twice(path, target):
        if next(refusals):
            raise PermissionError("in use")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", refuse_twice)

    assert fresh_set.add(two_games()[0], 1.0)
    assert SelfPlaySet.read_record(tmp_path / "played_selfplay.json")["games"] == 1


#the .part record holds the game, so reopening keeps it
def test_a_game_whose_record_could_not_be_moved_into_place_is_kept(fresh_set, tmp_path, monkeypatch):
    first, second = two_games()
    fresh_set.add(first, 1.0)
    def refuse(path, target):
        raise PermissionError("in use")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", refuse)
        patch.setattr(selfplay, "RECORD_ATTEMPTS", 2)
        assert not fresh_set.add(second, 1.0)

    reopened = SelfPlaySet(tmp_path, "played", SETTINGS)
    assert reopened.record["games"] == 2 and reopened.positions() == 4
    train, val = load_training_set("played", 0.02)
    assert len(train) == len(val) == 2


def test_a_set_from_before_tree_reuse_carries_on_without_it(tmp_path):
    SelfPlaySet(tmp_path, "played", SETTINGS).add(two_games()[0], 1.0)

    with pytest.raises(ValueError, match="--no-tree-reuse"):
        SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": True})
    assert SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": False}).positions() == 2


#the inference process's loop in a thread, with a RemoteNetwork for each slot
@contextmanager
def serving(slots: list[Slot], network, games_per_process: int):
    stop = threading.Event()
    arrays = [slot_arrays(slot.boards.name, slot.answers.name, games_per_process) for slot in slots]
    counts = SimpleNamespace(value=0), SimpleNamespace(value=0)
    thread = threading.Thread(target=serve, args=(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays],
                                                  [slot.server_end for slot in slots], stop, *counts), daemon=True)
    thread.start()
    remotes = [RemoteNetwork(slot.boards.name, slot.answers.name, slot.worker_end, games_per_process) for slot in slots]
    try:
        yield remotes, counts
    finally:
        stop.set()
        thread.join()
        for memories in [remote.memories for remote in remotes] + [memories for memories, _, _ in arrays]:
            for memory in memories:
                memory.close()
        for slot in slots:
            slot.close()


#two workers asking at once are served in one batch, each getting the answers to its own boards
def test_every_worker_gets_the_answers_to_its_own_boards():
    network = BoardModel()
    asked = [np.random.default_rng(index).integers(0, 13, (count, TOKENS_PER_BOARD), dtype=np.uint8) for index, count in enumerate((5, 8))]
    with serving([Slot(8), Slot(8)], cpu_batches(network), 8) as (remotes, (calls, positions)):
        def ask(index: int) -> np.ndarray:
            for _ in range(20):
                answer = remotes[index](asked[index]).copy()
            return answer
        with ThreadPoolExecutor(2) as pool:
            answers = list(pool.map(ask, range(2)))

    for answer, boards in zip(answers, asked):
        assert np.allclose(answer, cpu_batches(network).evaluate_boards(boards), atol=1e-5)
    assert positions.value == 20 * (5 + 8) and calls.value <= 40


def test_games_through_the_inference_process_are_the_games_played_directly():
    network = cpu_batches(BoardModel())
    def play(evaluate) -> list[Game]:
        return sorted(play_games_in_rust(evaluate, 8, range(4), size=2, max_plies=10), key=lambda game: game.number)

    alone = play(network.evaluate_boards)
    with serving([Slot(2)], network, 2) as ([remote], _):
        served = play(remote)

    for direct, through in zip(alone, served):
        assert np.array_equal(direct.boards, through.boards)
        assert np.array_equal(direct.visit_moves, through.visit_moves)
        assert np.allclose(direct.root_scores, through.root_scores)


#torch would cost every worker about half a gigabyte
def test_a_worker_does_not_import_torch():
    imported = subprocess.check_output([sys.executable, "-c", "import sys, engine.selfplay; print('torch' in sys.modules)"], text=True,
                                       cwd=Path(selfplay.__file__).parent.parent)
    assert imported.strip() == "False"
