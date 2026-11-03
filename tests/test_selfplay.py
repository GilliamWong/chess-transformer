from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training import data
from engine import selfplay
from training.data import SelfPlayDataset, load_training_set
from model.encoding import NO_MOVE, NUM_MOVE_IDS
from engine.inference import cpu_batches, serve
from engine.selfplay import TOP_MOVES, Game, RemoteNetwork, SelfPlaySet, Slot, play_games_in_rust, slot_arrays

SETTINGS = {"simulations": 16}


def two_games() -> list[Game]:
    visit_moves = np.full((2, TOP_MOVES), NO_MOVE, dtype=np.int16)
    visit_moves[:, :2] = [[5, 9], [7, 3]]
    visit_shares = np.zeros((2, TOP_MOVES), dtype=np.float32)
    visit_shares[:, :2] = [[0.75, 0.25], [1.0, 0.0]]
    boards = np.full((2, 70), 3, dtype=np.uint8)
    return [Game(0, boards, visit_moves, visit_shares, np.array([0.6, 0.3], dtype=np.float32), np.array([True, False]), 1.0),
            Game(1, boards, visit_moves, visit_shares, np.array([0.8, 0.1], dtype=np.float32), np.array([True, False]), 0.5)]

#game 0 goes to val, game 1 to train, read back with the value target blended half and half
def test_a_self_play_set_is_read_back_for_training(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    for game in two_games():
        games.add(game, 1.0)

    train, val = load_training_set("played", 0.02, result_weight=0.5)

    assert isinstance(train, SelfPlayDataset) and len(train) == len(val) == 2
    _, moves, shares, values, _ = val[[0, 1]]
    assert moves[:, :2].tolist() == [[5, 9], [7, 3]] and moves[:, 2:].eq(0).all()
    assert shares.tolist()[0][:2] == [0.75, 0.25]
    #white won game 0: white to move gets (1 + 0.6) / 2, black to move (0 + 0.3) / 2
    assert values.tolist() == pytest.approx([0.8, 0.15])
    assert train.value_target_of([0, 1]).tolist() == pytest.approx([0.65, 0.3]) #game 1 was a draw


#0.75/0.25 becomes 0.9/0.1 at 0.5 and 1/0 at 0; a single visited move stays whole
def test_the_visit_temperature_sharpens_the_policy_target(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    for game in two_games():
        games.add(game, 1.0)

    for temperature, expected in ((1.0, [0.75, 0.25]), (0.5, [0.9, 0.1]), (0.0, [1.0, 0.0])):
        _, val = load_training_set("played", 0.02, visit_temperature=temperature)
        _, moves, shares, _, _ = val[[0, 1]]
        assert shares[0, :2].tolist() == pytest.approx(expected) and shares[1, :2].tolist() == pytest.approx([1.0, 0.0])
        assert shares.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
        assert moves[:, :2].tolist() == [[5, 9], [7, 3]]


#part of a game written after the last record is cut off on reopening
def test_a_self_play_set_continues_after_an_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    for game in two_games():
        games.add(game, 1.0)
    with open(tmp_path / "played_train_boards.bin", "ab") as boards:
        boards.write(bytes(70 * 3))

    reopened = SelfPlaySet(tmp_path, "played", SETTINGS)

    assert reopened.positions() == 4 and reopened.record["next_game"] == 2
    train, _ = load_training_set("played", 0.02)
    assert len(train) == 2
    with pytest.raises(ValueError):
        SelfPlaySet(tmp_path, "played", {"simulations": 64})


#windows refuses the move for a moment while another program holds the record open
def test_a_record_write_that_fails_for_a_moment_is_tried_again(tmp_path, monkeypatch):
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    replace = Path.replace
    refusals = iter([True, True, False])
    def refuse_twice(path, target):
        if next(refusals):
            raise PermissionError("in use")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", refuse_twice)

    assert games.add(two_games()[0], 1.0)
    assert SelfPlaySet.read_record(tmp_path / "played_selfplay.json")["games"] == 1


#what stopped a run on 2026-09-23: the .part record keeps the game instead of it being cut off
def test_a_game_whose_record_could_not_be_moved_into_place_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    first, second = two_games()
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    games.add(first, 1.0)
    def refuse(path, target):
        raise PermissionError("in use")
    monkeypatch.setattr(Path, "replace", refuse)
    monkeypatch.setattr(selfplay, "RECORD_ATTEMPTS", 2)

    assert not games.add(second, 1.0)
    monkeypatch.undo()
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)

    reopened = SelfPlaySet(tmp_path, "played", SETTINGS)
    assert reopened.record["games"] == 2 and reopened.positions() == 4
    train, val = load_training_set("played", 0.02)
    assert len(train) == len(val) == 2


def test_a_set_from_before_tree_reuse_carries_on_without_it(tmp_path):
    SelfPlaySet(tmp_path, "played", SETTINGS).add(two_games()[0], 1.0)

    with pytest.raises(ValueError, match="--no-tree-reuse"):
        SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": True})
    assert SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": False}).positions() == 2


#answers that depend on the board, so answers handed to the wrong worker show
class BoardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(70, NUM_MOVE_IDS, generator=torch.Generator().manual_seed(0))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = boards.float()
        return tokens @ self.weights / 100, tokens.sum(dim=1, keepdim=True) / 1000


#the inference process's loop in a thread
@contextmanager
def serving(slots: list[Slot], evaluate_boards, games_per_process: int):
    stop = threading.Event()
    arrays = [slot_arrays(slot.boards.name, slot.answers.name, games_per_process) for slot in slots]
    counts = SimpleNamespace(value=0), SimpleNamespace(value=0)
    thread = threading.Thread(target=serve, args=(evaluate_boards, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays],
                                                  [slot.server_end for slot in slots], stop, *counts), daemon=True)
    thread.start()
    try:
        yield counts
    finally:
        stop.set()
        thread.join()
        for memories, _, _ in arrays:
            for memory in memories:
                memory.close()
        for slot in slots:
            slot.close()


#two workers asking at once are served in one batch, each getting the answers to its own boards
def test_every_worker_gets_the_answers_to_its_own_boards():
    slots = [Slot(8), Slot(8)]
    network = BoardModel()
    with serving(slots, cpu_batches(network), 8) as (calls, positions):
        remotes = [RemoteNetwork(slot.boards.name, slot.answers.name, slot.worker_end, 8) for slot in slots]
        asked = [np.random.default_rng(index).integers(0, 13, (count, 70), dtype=np.uint8) for index, count in enumerate((5, 8))]
        answers = [None, None]
        def ask(index: int) -> None:
            for _ in range(20):
                answers[index] = remotes[index](asked[index]).copy()
        threads = [threading.Thread(target=ask, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for answer, boards in zip(answers, asked):
            assert np.allclose(answer, cpu_batches(network).evaluate_boards(boards), atol=1e-5)
        assert positions.value == 20 * (5 + 8) and calls.value <= 40
        for remote in remotes:
            for memory in remote.memories:
                memory.close()


def test_games_through_the_inference_process_are_the_games_played_directly():
    network = cpu_batches(BoardModel())
    alone = sorted(play_games_in_rust(network.evaluate_boards, 8, range(4), size=2, max_plies=10), key=lambda game: game.number)
    slot = Slot(2)
    with serving([slot], network, 2):
        remote = RemoteNetwork(slot.boards.name, slot.answers.name, slot.worker_end, 2)
        served = sorted(play_games_in_rust(remote, 8, range(4), size=2, max_plies=10), key=lambda game: game.number)
        for memory in remote.memories:
            memory.close()

    for direct, through in zip(alone, served):
        assert np.array_equal(direct.boards, through.boards)
        assert np.array_equal(direct.visit_moves, through.visit_moves)
        assert np.allclose(direct.root_scores, through.root_scores)


#torch would cost every worker about half a gigabyte
def test_a_worker_does_not_import_torch():
    imported = subprocess.run([sys.executable, "-c", "import sys, engine.selfplay; print('torch' in sys.modules)"], capture_output=True, text=True,
                              check=True, cwd=Path(selfplay.__file__).parent.parent)
    assert imported.stdout.strip() == "False"
