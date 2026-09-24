import chess
import pytest
import torch
from torch import nn

import mcts
from encoding import NUM_MOVE_IDS, encode_move_to_id
from tools.label_with_stockfish import NO_EVAL, NO_MOVE
from tools.label_with_tablebase import TABLEBASE_WIN_EVAL, label_position
from selfplay import final_result
from tools.play.player import choose_move
from tools.tablebase import open_tablebase

CPU = torch.device("cpu")
TOP_MOVES = 4


#every move equally likely and every position even: only the tables tell positions apart
class UniformModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), torch.zeros(len(boards), 1)


@pytest.fixture(scope="module")
def tablebase():
    try:
        tables = open_tablebase()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    yield tables
    tables.close()


#nearly every rook move keeps the win, but a mate is the fastest progress there is, so it is the whole target
def test_a_mate_in_one_is_the_only_target(tablebase):
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")

    moves, evals, _ = label_position(tablebase, board, [], TOP_MOVES)

    assert moves == [encode_move_to_id(chess.Move.from_uci("h1h8"), chess.WHITE)] + [NO_MOVE] * 3
    assert evals == [TABLEBASE_WIN_EVAL] + [NO_EVAL] * 3


#a rook ending that is a draw until the white rook steps where the black one takes it: that move leaves the target, and
#the drawing move stockfish listed comes first
def test_a_move_that_gives_the_draw_away_leaves_the_target(tablebase):
    board = chess.Board("6k1/5r2/8/8/8/8/1R6/2K5 w - - 0 1")
    hangs_the_rook, draws = chess.Move.from_uci("b2f2"), chess.Move.from_uci("b2b3")
    assert tablebase.probe_wdl(board) == 0

    moves, evals, stockfish_wdls = label_position(tablebase, board, [hangs_the_rook, draws], TOP_MOVES)

    assert moves[0] == encode_move_to_id(draws, chess.WHITE)
    assert encode_move_to_id(hangs_the_rook, chess.WHITE) not in moves
    assert evals == [0] * TOP_MOVES
    assert stockfish_wdls == [-2, 0]


#black's moves are labelled in the mirrored frame the model sees them in, and a lost position gets a lost value target
def test_a_lost_position_for_black(tablebase):
    board = chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")

    moves, evals, _ = label_position(tablebase, board, [], TOP_MOVES)

    assert moves == [encode_move_to_id(chess.Move.from_uci("a8b8"), chess.BLACK)] + [NO_MOVE] * 3
    assert evals == [-TABLEBASE_WIN_EVAL] + [NO_EVAL] * 3


#6 pieces, so the tables do not know the root, but taking the rook leaves 5 and a won position for white. The network
#here thinks every position even, so only the tables can send the search there
def test_the_search_scores_the_positions_the_tables_cover(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)
    board = chess.Board("6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1")
    simulations = 200

    root = mcts.search(UniformModel(), board, CPU, simulations)

    assert root.children[chess.Move.from_uci("d1d5")].visit_count > simulations / 2
    with pytest.raises(ValueError):
        mcts.search(UniformModel(), chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1"), CPU)


#search refuses a position the tables know, so a game that reaches one plays their move instead of stopping there
def test_the_player_takes_the_tables_move_once_they_know_the_position(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)

    move = choose_move(UniformModel(), chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1"), CPU, "mcts", simulations=16)

    assert move == chess.Move.from_uci("h1h8")


#with the tables, a self play game ends once they know the result, which is scored from white's side
def test_self_play_ends_a_game_the_tables_know(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)

    assert final_result(chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")) == 1.0 #black to move, and lost
    assert final_result(chess.Board("6k1/5r2/8/8/8/8/1R6/2K5 w - - 0 1")) == 0.5
    assert final_result(chess.Board("6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1")) is None #6 pieces
