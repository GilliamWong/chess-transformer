import chess
import pytest
import torch

from tools.play.evaluate_elo import elo_difference, parse_score
from model import ChessTransformer
from tools.play.player import choose_move
from tools.play.uci import parse_position

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def untrained_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()


#an untrained model's favourite move is illegal almost every time, so only the mask can make this pass
def test_chosen_move_is_always_legal(untrained_model, carlsen_games):
    for game in carlsen_games:
        board = game.board()
        for played_move in game.mainline_moves():
            assert choose_move(untrained_model, board, CPU) in board.legal_moves
            board.push(played_move)


@pytest.mark.parametrize(
    "fen, only_move",
    [
        #the king is in check along the back rank and the g file is covered, so one square is left
        pytest.param("6rk/8/8/8/8/8/8/r6K w - - 0 1", "h1h2", id="white-king-has-one-square"),
        pytest.param("R6k/8/8/8/8/8/8/6RK b - - 0 1", "h8h7", id="black-king-has-one-square"),
    ],
)
def test_the_only_legal_move_is_chosen(untrained_model, fen, only_move):
    board = chess.Board(fen)
    assert [move.uci() for move in board.legal_moves] == [only_move]

    assert choose_move(untrained_model, board, CPU).uci() == only_move


def test_promotion_moves_can_be_chosen(untrained_model):
    #every legal move here is a promotion, so the promotion ids have to survive the mask
    board = chess.Board("1n2k3/P7/8/8/8/8/8/8 w - - 0 1")

    assert choose_move(untrained_model, board, CPU).promotion is not None


def test_no_move_to_choose_when_the_game_is_over(untrained_model):
    checkmate = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30")

    with pytest.raises(ValueError):
        choose_move(untrained_model, checkmate, CPU)


def test_position_from_the_start_position_and_moves():
    board = parse_position("startpos moves e2e4 e7e5 g1f3".split())

    expected = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        expected.push_san(san)
    assert board == expected


def test_position_from_a_fen_with_and_without_moves():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

    assert parse_position(["fen", *fen.split()]).fen() == fen
    #castling arrives as the king's move in uci
    castled = parse_position(["fen", *fen.split(), "moves", "e1g1"])
    assert castled.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


def test_unknown_position_is_rejected():
    with pytest.raises(ValueError):
        parse_position(["kiwipete"])


#known points of the elo curve: even score is no difference, 10 to 1 odds is 400 points
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
