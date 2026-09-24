from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from torch import nn

import data
import selfplay
from data import SelfPlayDataset, load_training_set
from encoding import NUM_MOVE_IDS, encode_move_to_id
from selfplay import TOP_MOVES, Game, SelfPlaySet, play_game, play_games
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
