import json
import math
from pathlib import Path

import chess
import chess.pgn
import pytest
import torch
from torch import nn

from data import EXPECTED_SCORE_PER_CENTIPAWN
from encoding import NUM_MOVE_IDS, encode_move_to_id
from tools.play.evaluate_elo import annotate_searches, elo_difference, model_engine, parse_score
from model import ChessTransformer
from tools.play.player import AHEAD_SCORE, choose_move, score_to_centipawns, search_move
from tools.play.uci import parse_position

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def untrained_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()


#an untrained model's favourite move is almost always illegal, so only the mask makes this pass
def test_chosen_move_is_always_legal(untrained_model, carlsen_games):
    for game in carlsen_games:
        board = game.board()
        for played_move in game.mainline_moves():
            assert choose_move(untrained_model, board, CPU) in board.legal_moves
            board.push(played_move)


@pytest.mark.parametrize(
    "fen, only_move",
    [
        pytest.param("6rk/8/8/8/8/8/8/r6K w - - 0 1", "h1h2", id="white-king-has-one-square"),
        pytest.param("R6k/8/8/8/8/8/8/6RK b - - 0 1", "h8h7", id="black-king-has-one-square"),
    ],
)
def test_the_only_legal_move_is_chosen(untrained_model, fen, only_move):
    board = chess.Board(fen)
    assert [move.uci() for move in board.legal_moves] == [only_move]

    assert choose_move(untrained_model, board, CPU).uci() == only_move


def test_promotion_moves_can_be_chosen(untrained_model):
    #every legal move is a promotion
    board = chess.Board("1n2k3/P7/8/8/8/8/8/8 w - - 0 1")

    assert choose_move(untrained_model, board, CPU).promotion is not None


def test_no_move_to_choose_when_the_game_is_over(untrained_model):
    checkmate = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30")

    with pytest.raises(ValueError):
        choose_move(untrained_model, checkmate, CPU)


#one or two simulations visit at most the policy's favourite
@pytest.mark.parametrize("simulations", [1, 2])
def test_a_search_without_room_to_search_plays_the_argmax_move(untrained_model, carlsen_games, simulations):
    board = carlsen_games[0].board()
    for played_move in list(carlsen_games[0].mainline_moves())[:30]:
        assert choose_move(untrained_model, board, CPU, "mcts", simulations=simulations) == choose_move(untrained_model, board, CPU)
        board.push(played_move)


def test_reported_centipawns_invert_the_training_curve():
    assert score_to_centipawns(0.5) == 0
    assert score_to_centipawns(1 / (1 + math.exp(-EXPECTED_SCORE_PER_CENTIPAWN * 300))) == 300
    assert score_to_centipawns(1.0) == -score_to_centipawns(0.0) > 2000 #capped rather than infinite


def test_a_search_reports_the_line_of_the_move_it_plays(untrained_model, carlsen_games):
    board = carlsen_games[2].board()
    for played_move in list(carlsen_games[2].mainline_moves())[:12]:
        board.push(played_move)

    move, report = search_move(untrained_model, board, CPU, 64)

    assert report.line[0] == move and report.simulations == 64
    assert 0 <= report.score <= 1
    line_board = board.copy()
    for line_move in report.line:
        assert line_move in line_board.legal_moves
        line_board.push(line_move)


#scores are logged for the mover, [%eval] is from white's side: black's good move is negative
def test_searches_are_written_next_to_their_moves_from_whites_side(tmp_path):
    pgn_path = tmp_path / "match.pgn"
    pgn_path.write_text('[Event "?"]\n\n1. e4 e5 2. Nf3 *\n')
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    record = {"fen": after_e4.fen(), "move": "e7e5", "score": 0.8, "simulations": 64, "line": ["e7e5", "g1f3"], "root_moves": [["e7e5", 50], ["c7c5", 10]]}
    (tmp_path / "1234.jsonl").write_text(json.dumps(record) + "\n")

    assert annotate_searches(pgn_path, tmp_path) == 1

    e4, e5, _ = chess.pgn.read_game(open(pgn_path)).mainline()
    assert e4.eval() is None
    assert e5.eval().white().score() == -score_to_centipawns(0.8) < 0
    assert "expects 1...e5 2. Nf3" in e5.comment and "searched e5 50, c5 10 of 64" in e5.comment


#a searching model would lose on time at 0.1 s a move
def test_only_a_searching_model_plays_without_a_clock():
    assert "tc=inf" in model_engine(Path("model.pt"), searches=True)
    assert "tc=inf" not in model_engine(Path("model.pt"))


def test_position_from_the_start_position_and_moves():
    board = parse_position("startpos moves e2e4 e7e5 g1f3".split())

    expected = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        expected.push_san(san)
    assert board == expected


def test_position_from_a_fen_with_and_without_moves():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

    assert parse_position(["fen", *fen.split()]).fen() == fen
    #castling arrives as the king's move
    castled = parse_position(["fen", *fen.split(), "moves", "e1g1"])
    assert castled.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


def test_unknown_position_is_rejected():
    with pytest.raises(ValueError):
        parse_position(["kiwipete"])


def test_elo_difference():
    assert elo_difference(0.5) == 0
    assert elo_difference(10 / 11) == pytest.approx(400)
    assert elo_difference(1 / 11) == pytest.approx(-400)


def test_score_is_read_from_the_last_score_line():
    output = """
Score of ChessTransformer vs Stockfish1320: 1 - 0 - 0  [1.000] 1
Finished game 2 (Stockfish1320 vs ChessTransformer): 1-0 {White mates}
Score of ChessTransformer vs Stockfish1320: 12 - 80 - 8  [0.160] 100
Elo difference: -288.1 +/- 80.2, LOS: 0.0 %, DrawRatio: 8.0 %
"""

    assert parse_score(output) == (12, 80, 8)


#the logits of the given moves (every other far below) and a fixed value
class FixedModel(nn.Module):
    def __init__(self, board: chess.Board, move_logits: dict[str, float], expected_score: float):
        super().__init__()
        self.move_logits = torch.full((1, NUM_MOVE_IDS), -20.0)
        for san, logit in move_logits.items():
            self.move_logits[0, encode_move_to_id(board.parse_san(san), board.turn)] = logit
        self.value_logits = torch.logit(torch.tensor([[expected_score]]))

    def forward(self, boards):
        return self.move_logits, self.value_logits


#lichess game dwCvJZkh: a drawn rook ending where Kf7 repeats and the only other move, Re8, hangs the rook
def rook_ending_before_the_repetition() -> chess.Board:
    board = chess.Board("2r5/5K2/4RP1k/8/8/8/8/8 w - - 0 1")
    for uci in ("e6e1", "c8c7", "f7g8", "c7c8"):
        board.push_uci(uci)
    return board


def test_a_repetition_is_kept_in_a_level_position():
    board = rook_ending_before_the_repetition()
    model = FixedModel(board, {"Kf7": 0.0, "Re8": -1.0}, expected_score=0.51)

    assert board.san(choose_move(model, board, CPU, "avoid-repetition")) == "Kf7"


def test_a_repetition_is_kept_rather_than_a_move_the_policy_rejects():
    board = rook_ending_before_the_repetition()
    model = FixedModel(board, {"Kf7": 4.0, "Re8": 0.0}, expected_score=0.9)

    assert board.san(choose_move(model, board, CPU, "avoid-repetition")) == "Kf7"


def test_a_repetition_is_avoided_when_clearly_ahead_with_a_move_the_policy_rates():
    board = rook_ending_before_the_repetition()
    model = FixedModel(board, {"Kf7": 0.0, "Re8": -1.0}, expected_score=AHEAD_SCORE + 0.05)

    assert board.san(choose_move(model, board, CPU, "avoid-repetition")) == "Re8"
    assert board.san(choose_move(model, board, CPU, "argmax")) == "Kf7"
