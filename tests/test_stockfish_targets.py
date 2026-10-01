import math

import chess
import chess.engine
import numpy as np
import pytest
import torch
from torch import nn

from training import data
from training.data import EXPECTED_SCORE_PER_CENTIPAWN, load_training_set
from model.encoding import TOKENS_PER_BOARD, encode_move_to_id
from training.train import policy_loss
from tools import label_with_stockfish
from tools.label_with_stockfish import MATE_EVAL, NO_EVAL, NO_MOVE, score_to_eval, whole_games_prefix
from tools.play.download_engines import STOCKFISH

TEMPERATURE = 0.02

#two good moves and a blunder; a single legal move; a mate next to a good move
STOCKFISH_MOVES = [[5, 9, 2], [7, NO_MOVE, NO_MOVE], [1, 3, NO_MOVE]]
STOCKFISH_EVALS = [[50, 50, -300], [-120, NO_EVAL, NO_EVAL], [MATE_EVAL, 200, NO_EVAL]]


def write_labelled_set(folder, name, stockfish_moves=STOCKFISH_MOVES, stockfish_evals=STOCKFISH_EVALS) -> None:
    positions = len(stockfish_moves)
    for split in ("train", "val"):
        (folder / f"{name}_{split}_boards.bin").write_bytes(np.full((positions, TOKENS_PER_BOARD), 5, dtype=np.uint8).tobytes())
        (folder / f"{name}_{split}_stockfish_moves.bin").write_bytes(np.array(stockfish_moves, dtype=np.int16).tobytes())
        (folder / f"{name}_{split}_stockfish_evals.bin").write_bytes(np.array(stockfish_evals, dtype=np.int16).tobytes())


#the softmax written out with math instead of torch
def expected_probabilities(evals: list[int]) -> list[float]:
    weights = [math.exp(1 / (1 + math.exp(-EXPECTED_SCORE_PER_CENTIPAWN * centipawns)) / TEMPERATURE) for centipawns in evals]
    return [weight / sum(weights) for weight in weights]


def test_stockfish_targets_share_the_probability_by_their_evals(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_labelled_set(tmp_path, "labelled")

    train_dataset, _ = load_training_set("labelled", TEMPERATURE)
    _, target_moves, target_probabilities, _, _ = train_dataset[[0, 1, 2]]

    #places without a move are move 0 at probability 0
    assert target_moves.tolist() == [[5, 9, 2], [7, 0, 0], [1, 3, 0]]
    assert target_probabilities[0].tolist() == pytest.approx(expected_probabilities([50, 50, -300]))
    assert target_probabilities[0, 2] < 0.001  #the blunder
    assert target_probabilities[1].tolist() == [1.0, 0.0, 0.0]
    assert target_probabilities[2].tolist() == pytest.approx(expected_probabilities([MATE_EVAL, 200]) + [0.0])
    assert target_probabilities[2, 0] > 0.99  #a mate against two pawns up


#an unlabelled position (NO_MOVE) would otherwise train as "play move 0"
def test_a_set_that_is_not_labelled_to_the_end_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DATA_DIR", tmp_path)
    write_labelled_set(tmp_path, "unfinished", STOCKFISH_MOVES[:2] + [[NO_MOVE] * 3], STOCKFISH_EVALS[:2] + [[NO_EVAL] * 3])

    with pytest.raises(ValueError, match="not labelled"):
        load_training_set("unfinished", TEMPERATURE)


def test_policy_loss_with_one_move_is_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    moves = torch.randint(50, (6,))

    loss = policy_loss(logits, moves.unsqueeze(1), torch.ones(6, 1))

    assert loss.item() == pytest.approx(nn.CrossEntropyLoss()(logits, moves).item(), rel=1e-6)


def test_policy_loss_weighs_every_target_move_by_its_probability():
    torch.manual_seed(0)
    logits = torch.randn(2, 50)
    target_moves = torch.tensor([[3, 8, 0], [4, 0, 0]])
    target_probabilities = torch.tensor([[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]])

    loss = policy_loss(logits, target_moves, target_probabilities)

    log_probabilities = torch.log_softmax(logits, dim=-1)
    by_hand = -(0.7 * log_probabilities[0, 3] + 0.3 * log_probabilities[0, 8] + log_probabilities[1, 4]) / 2
    assert loss.item() == pytest.approx(by_hand.item(), rel=1e-6)


def test_a_copied_set_ends_at_the_end_of_a_game():
    plies = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.int16)  #games of 3, 2 and 4 positions

    assert whole_games_prefix(plies, 3) == 3  #already a whole game
    assert whole_games_prefix(plies, 4) == 5  #the second game is finished
    assert whole_games_prefix(plies, 6) == 9
    assert whole_games_prefix(plies, 100) == 9


#self play has no plies to cut at: each distinct board once, in first-seen order
def test_a_self_play_set_is_copied_as_its_distinct_boards(tmp_path):
    boards = np.repeat(np.array([[1], [2], [1], [3]], dtype=np.uint8), TOKENS_PER_BOARD, axis=1)
    boards.tofile(tmp_path / "played_train_boards.bin")
    np.zeros((4, 8), dtype=np.int16).tofile(tmp_path / "played_train_visit_moves.bin")

    label_with_stockfish.copy_prefix(tmp_path, "played", "played_sf", "train", 100)
    label_with_stockfish.copy_prefix(tmp_path, "played", "played_short_sf", "train", 2)

    copied = np.fromfile(tmp_path / "played_sf_train_boards.bin", dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
    assert copied[:, 0].tolist() == [1, 2, 3] and (copied == copied[:, :1]).all()
    assert np.fromfile(tmp_path / "played_short_sf_train_boards.bin", dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)[:, 0].tolist() == [1, 2]


def test_scores_are_stored_like_the_fishtest_evals():
    assert score_to_eval(chess.engine.Cp(-37)) == -37
    assert score_to_eval(chess.engine.Cp(31000)) == MATE_EVAL - 1  #never mistaken for a mate
    assert score_to_eval(chess.engine.Mate(3)) == MATE_EVAL
    assert score_to_eval(chess.engine.Mate(-1)) == -MATE_EVAL


#black to move: catches a move or score not turned to the side to move
@pytest.mark.skipif(not STOCKFISH.exists(), reason="stockfish is not downloaded (python -m tools.play.download_engines)")
def test_stockfish_finds_the_mate_for_black_and_scores_it_as_a_win():
    board = chess.Board()
    for san in ("f3", "e5", "g4"):
        board.push_san(san)

    label_with_stockfish.start_engine(str(STOCKFISH.resolve()), depth=6, top_moves=3)
    try:
        moves, evals = label_with_stockfish.label_position(board)
    finally:
        label_with_stockfish.ENGINE.quit()

    assert moves[0] == encode_move_to_id(board.parse_san("Qh4#"), chess.BLACK)
    assert evals[0] == MATE_EVAL
    assert len(moves) == len(evals) == 3 and evals[1] < MATE_EVAL
