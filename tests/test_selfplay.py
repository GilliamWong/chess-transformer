from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch
from torch import nn

import data
import selfplay
from data import SelfPlayDataset, load_training_set
from encoding import NUM_MOVE_IDS, encode_move_to_id
from inference import cpu_batches, serve
from selfplay import TOP_MOVES, Game, RemoteNetwork, SelfPlaySet, Slot, play_game, play_games, play_games_in_rust, slot_arrays
from tools.label_with_stockfish import NO_MOVE

CPU = torch.device("cpu")
SETTINGS = {"simulations": 16}


#every move equally likely and every position even: whatever the search finds, it found by searching
class UniformModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), torch.zeros(len(boards), 1)


#black mates at once: the only position of the game is black's, and it is black's win, whatever the colour of the board
def test_a_game_is_recorded_for_the_side_to_move():
    board = chess.Board("7r/8/8/8/8/1k6/8/K7 b - - 0 1")

    game = play_game(UniformModel(), CPU, 800, number=0, board=board, sampled_plies=0)

    assert game.white_score == 0.0 and len(game.boards) == 1
    assert game.outcomes().tolist() == [1.0]
    assert game.root_scores[0] > 0.5
    #black's moves are stored in the mirrored frame the model sees them in
    assert game.visit_moves[0, 0] == encode_move_to_id(chess.Move.from_uci("h8h1"), chess.BLACK)
    assert game.visit_shares[0].sum() == pytest.approx(1.0)


#a game's random numbers come from the seed and its number alone, so the same game can be played again and two
#games of a run differ
def test_a_game_is_the_same_for_the_same_number():
    first, again, other = (play_game(UniformModel(), CPU, 16, number, max_plies=8) for number in (3, 3, 4))

    assert np.array_equal(first.boards, again.boards) and np.array_equal(first.visit_shares, again.visit_shares)
    assert not np.array_equal(first.boards, other.boards)
    assert first.outcomes().tolist() == [0.5] * 8 #cut off after 8 plies: a draw for both sides


def two_games() -> list[Game]:
    visit_moves = np.full((2, TOP_MOVES), NO_MOVE, dtype=np.int16)
    visit_moves[:, :2] = [[5, 9], [7, 3]]
    visit_shares = np.zeros((2, TOP_MOVES), dtype=np.float32)
    visit_shares[:, :2] = [[0.75, 0.25], [1.0, 0.0]]
    boards = np.full((2, 70), 3, dtype=np.uint8)
    return [Game(0, boards, visit_moves, visit_shares, np.array([0.6, 0.3], dtype=np.float32), np.array([True, False]), 1.0),
            Game(1, boards, visit_moves, visit_shares, np.array([0.8, 0.1], dtype=np.float32), np.array([True, False]), 0.5)]

#game 0 goes to val, game 1 to train, and training reads them back with the value target blended half and half
def test_a_self_play_set_is_read_back_for_training(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    for game in two_games():
        games.add(game, 1.0)

    train, val = load_training_set("played", 0.02, result_weight=0.5)

    assert isinstance(train, SelfPlayDataset) and len(train) == len(val) == 2
    _, moves, shares, values = val[[0, 1]]
    assert moves[:, :2].tolist() == [[5, 9], [7, 3]] and moves[:, 2:].eq(0).all()
    assert shares.tolist()[0][:2] == [0.75, 0.25]
    #white won game 0: white to move gets (1 + 0.6) / 2, black to move (0 + 0.3) / 2
    assert values.tolist() == pytest.approx([0.8, 0.15])
    assert train.value_target_of([0, 1]).tolist() == pytest.approx([0.65, 0.3]) #game 1 was a draw


#a lower visit temperature sharpens the policy target and keeps it a distribution: shares of 0.75 and 0.25 are 0.9 and 0.1
#at 0.5 (squared, 9 to 1), and the most visited move alone at 0. A position with one visited move keeps it whole
def test_the_visit_temperature_sharpens_the_policy_target(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    games = SelfPlaySet(tmp_path, "played", SETTINGS)
    for game in two_games():
        games.add(game, 1.0)

    for temperature, expected in ((1.0, [0.75, 0.25]), (0.5, [0.9, 0.1]), (0.0, [1.0, 0.0])):
        _, val = load_training_set("played", 0.02, visit_temperature=temperature)
        _, moves, shares, _ = val[[0, 1]]
        assert shares[0, :2].tolist() == pytest.approx(expected) and shares[1, :2].tolist() == pytest.approx([1.0, 0.0])
        assert shares.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
        assert moves[:, :2].tolist() == [[5, 9], [7, 3]]


#an interruption can leave part of a game in the files after the last record: opening the set again cuts it off
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


#games played together in one batch, their leaves evaluated in shared calls and finished games replaced by the next
#number, come out exactly as each game played on its own: nothing of one game leaks into another
def test_games_played_together_are_the_games_played_alone():
    alone = [play_game(UniformModel(), CPU, 8, number, max_plies=12) for number in range(5)]

    together = sorted(play_games(UniformModel(), CPU, 8, range(5), size=3, max_plies=12), key=lambda game: game.number)

    assert [game.number for game in together] == list(range(5))
    for single, batched in zip(alone, together):
        assert np.array_equal(single.boards, batched.boards)
        assert np.array_equal(single.visit_moves, batched.visit_moves)
        assert np.array_equal(single.visit_shares, batched.visit_shares)
        assert np.array_equal(single.root_scores, batched.root_scores)


#another program holding the record open makes windows refuse the move for a moment: the write is tried again
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


#the move of the record kept failing (what stopped a run on 2026-09-23): the game's positions and its record in the .part
#are both written, so opening the set again keeps that game instead of cutting it off
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


#counts the positions it evaluates, every move equally likely and every position even
class CountingModel(UniformModel):
    def __init__(self):
        super().__init__()
        self.positions = 0

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.positions += len(boards)
        return super().forward(boards)

#the move played keeps its subtree, so the next search already has visits and needs fewer evaluations for the same 64
def test_tree_reuse_needs_fewer_evaluations_for_the_same_visits():
    fresh, reused = CountingModel(), CountingModel()

    fresh_game = play_game(fresh, CPU, 64, 0, max_plies=10, tree_reuse=False)
    reused_game = play_game(reused, CPU, 64, 0, max_plies=10, tree_reuse=True)

    assert len(fresh_game.boards) == len(reused_game.boards) == 10
    assert reused.positions < fresh.positions
    assert np.allclose(reused_game.visit_shares.sum(axis=1), 1.0)


#a set played before tree reuse existed has no such setting: it counts as played without, and only carries on without
def test_a_set_from_before_tree_reuse_carries_on_without_it(tmp_path):
    SelfPlaySet(tmp_path, "played", SETTINGS).add(two_games()[0], 1.0)

    with pytest.raises(ValueError, match="--no-tree-reuse"):
        SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": True})
    assert SelfPlaySet(tmp_path, "played", {**SETTINGS, "tree_reuse": False}).positions() == 2


#an answer that depends on the board, so answers handed to the wrong worker, or games crossed between slots, show
class BoardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(70, NUM_MOVE_IDS, generator=torch.Generator().manual_seed(0))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = boards.float()
        return tokens @ self.weights / 100, tokens.sum(dim=1, keepdim=True) / 1000


#the inference process's loop in a thread, serving the given slots with evaluate_boards
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


#two workers asking at the same moment are served in one batch, and each gets back the answers to its own boards
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


#games whose leaves go through the inference process are the games played with the network in the same process
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


#every worker imports selfplay.py, and torch would take about half a gigabyte in each
def test_a_worker_does_not_import_torch():
    imported = subprocess.run([sys.executable, "-c", "import sys, selfplay; print('torch' in sys.modules)"], capture_output=True, text=True,
                              check=True, cwd=Path(selfplay.__file__).parent)
    assert imported.stdout.strip() == "False"
