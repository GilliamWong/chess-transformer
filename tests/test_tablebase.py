import os
from pathlib import Path
import shutil

import chess
import numpy as np
import pytest
import torch

import mcts
import selfplay_rs
from encoding import TOKENS_PER_BOARD, encode_move_to_id
from fake_models import UniformModel
from inference import cpu_batches
from reference_selfplay import final_result
from selfplay import ANSWER_WIDTH, rust_search_settings
from tools.label_with_stockfish import NO_EVAL, NO_MOVE
from tools.label_with_tablebase import TABLEBASE_WIN_EVAL, label_position
from tools.play.engine_search import EngineSearch
from tools.play.player import choose_move
from tools.tablebase import TABLE_SETS, TableSet, covers, open_tablebase, ranked_moves, ranked_pieces

CPU = torch.device("cpu")
TOP_MOVES = 4
#white takes the rook (Qxd5+), which wins: from 7 pieces, 6 (a 6 piece table's) and 6 pieces, 5
SEVEN_PIECES = "6k1/1p5p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"
SIX_PIECES = "6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"


#nearly every rook move keeps the win, but a mate is the fastest progress, so it is the whole target
def test_a_mate_in_one_is_the_only_target(tablebase):
    board = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1")

    moves, evals, _ = label_position(tablebase, board, [], TOP_MOVES)

    assert moves == [encode_move_to_id(chess.Move.from_uci("h1h8"), chess.WHITE)] + [NO_MOVE] * 3
    assert evals == [TABLEBASE_WIN_EVAL] + [NO_EVAL] * 3


#a drawn rook ending: the move that hangs the rook leaves the target, and stockfish's drawing move comes first
def test_a_move_that_gives_the_draw_away_leaves_the_target(tablebase):
    board = chess.Board("6k1/5r2/8/8/8/8/1R6/2K5 w - - 0 1")
    hangs_the_rook, draws = chess.Move.from_uci("b2f2"), chess.Move.from_uci("b2b3")
    assert tablebase.probe_wdl(board) == 0

    moves, evals, stockfish_wdls = label_position(tablebase, board, [hangs_the_rook, draws], TOP_MOVES)

    assert moves[0] == encode_move_to_id(draws, chess.WHITE)
    assert encode_move_to_id(hangs_the_rook, chess.WHITE) not in moves
    assert evals == [0] * TOP_MOVES
    assert stockfish_wdls == [-2, 0]


#black's moves are labelled mirrored, and a lost position gets a lost value target
def test_a_lost_position_for_black(tablebase):
    board = chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")

    moves, evals, _ = label_position(tablebase, board, [], TOP_MOVES)

    assert moves == [encode_move_to_id(chess.Move.from_uci("a8b8"), chess.BLACK)] + [NO_MOVE] * 3
    assert evals == [-TABLEBASE_WIN_EVAL] + [NO_EVAL] * 3


#6 pieces at the root; taking the rook leaves 5 and a won position, which only the tables can tell the search
def test_the_search_scores_the_positions_the_tables_cover(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)
    board = chess.Board("6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1")
    simulations = 200

    root = mcts.search(UniformModel(), board, CPU, simulations)

    assert root.children[chess.Move.from_uci("d1d5")].visit_count > simulations / 2
    with pytest.raises(ValueError):
        mcts.search(UniformModel(), chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1"), CPU)


#search refuses a position the tables know, so the player plays their move instead
def test_the_player_takes_the_tables_move_once_they_know_the_position(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)

    move = choose_move(UniformModel(), chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1"), CPU, "mcts", simulations=16)

    assert move == chess.Move.from_uci("h1h8")


def test_self_play_ends_a_game_the_tables_know(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)

    assert final_result(chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")) == 1.0 #black to move, and lost
    assert final_result(chess.Board("6k1/5r2/8/8/8/8/1R6/2K5 w - - 0 1")) == 0.5
    assert final_result(chess.Board("6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1")) is None #6 pieces


#6 pieces are the 6 piece tables' (--tablebase-pieces 6), not the default's; castling rights are no one's
def test_the_tables_cover_the_pieces_asked_for():
    six, seven = chess.Board(SIX_PIECES), chess.Board(SEVEN_PIECES)

    assert covers(six, 6) and not covers(six)
    assert not covers(seven, 6)
    assert not covers(chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"), 6)


#table files of each half (empty: python-chess and shakmaty only take files named for a table), a folder for each size
def fake_table_sets(folder: Path, monkeypatch, tables: dict[int, tuple[int, int]]) -> tuple[TableSet, ...]:
    table_sets = tuple(TableSet(pieces, folder / str(pieces), str(pieces), 3) for pieces in (5, 6))
    for table_set in table_sets:
        table_set.folder.mkdir()
        wdl, dtz = tables[table_set.pieces]
        for name in [f"fake{index}.rtbw" for index in range(wdl)] + [f"fake{index}.rtbz" for index in range(dtz)]:
            (table_set.folder / name).touch()
    monkeypatch.setattr("tools.tablebase.TABLE_SETS", table_sets)
    return table_sets

#every wdl table must be there, and every dtz table up to 5 pieces; the 6 piece dtz tables may be left out as a whole
#(a pod's disk), and their positions are then searched rather than ranked. An unfinished download is refused
@pytest.mark.parametrize("five, six, ranked", [((3, 3), (3, 3), 6), ((3, 3), (3, 0), 5), ((3, 3), (2, 3), None),
                                               ((3, 3), (3, 1), None), ((3, 0), (3, 3), None)])
def test_which_tables_open(tmp_path, monkeypatch, five, six, ranked):
    fake_table_sets(tmp_path, monkeypatch, {5: five, 6: six})

    if ranked is None:
        with pytest.raises(FileNotFoundError):
            open_tablebase(6)
    else:
        open_tablebase(6).close()
        assert ranked_pieces(6) == ranked

#the default, 5 pieces, neither needs nor probes the 6 piece folder
def test_the_default_tables_are_the_5_piece_ones(tmp_path, monkeypatch):
    five, six = fake_table_sets(tmp_path, monkeypatch, {5: (3, 3), 6: (0, 0)})

    open_tablebase().close()
    assert ranked_pieces(5) == 5
    assert (rust_search_settings(True)["tablebase_folder"], rust_search_settings(True)["tablebase_pieces"]) == (str(five.folder), 5)
    assert rust_search_settings(True, 6)["tablebase_folder"] == os.pathsep.join([str(five.folder), str(six.folder)])


#one rust search to its end with a uniform network: its line and the visits of its root moves
def search_in_rust(fen: str, simulations: int, tablebase_pieces: int = 5) -> tuple[list[str], dict[str, int]]:
    searches = selfplay_rs.Searches(**rust_search_settings(True, tablebase_pieces))
    searches.start(0, fen, [], simulations)
    evaluate = cpu_batches(UniformModel()).evaluate_boards
    while True:
        leaves = np.frombuffer(bytearray(searches.descend()), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        reports = searches.finish(evaluate(leaves) if len(leaves) else np.zeros((0, ANSWER_WIDTH), np.float32))
        if reports:
            _, line, _, root_moves, _ = reports[0]
            return line, dict(root_moves)

#a root the tables cover is searched (the engine does, at a size it has no dtz tables for): below it only the moves
#after a capture or pawn move are the tables', the others are searched on. So the rook taken, or the mate, gets the
#visits rather than sharing them with every move the tables would call a win (lc0 probes so)
@pytest.mark.parametrize("fen, best", [("6k1/8/8/3r4/8/8/8/3Q2K1 w - - 0 1", "d1d5"), ("k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8")])
def test_rust_searches_a_root_the_tables_cover(tablebase, fen, best):
    simulations = 400

    line, root_moves = search_in_rust(fen, simulations)

    assert line[0] == best and root_moves[best] > simulations / 2


#the tables of up to 6 pieces, perhaps without the 6 piece dtz ones
@pytest.fixture(scope="module")
def tablebase6():
    try:
        tables = open_tablebase(6)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    yield tables
    tables.close()

#a 6 piece position from the 6 piece folder, and the 5 piece one after its capture from the 5 piece folder: both lost
#for black, as the lichess tablebase has it
def test_python_probes_both_folders(tablebase6):
    six, five = chess.Board("6k1/1p5p/8/3Q4/8/8/P7/6K1 b - - 0 1"), chess.Board("6k1/7p/8/3Q4/8/8/P7/6K1 b - - 0 1")

    assert tablebase6.probe_wdl(six) == tablebase6.probe_wdl(five) == -2

#from 7 pieces, taking the rook leaves a won 6 piece position that only the 6 piece tables can tell the search; rust
#probes them with the 5 piece ones (the positions after a capture)
@pytest.mark.parametrize("tablebase_pieces", [5, 6])
def test_rust_probes_the_6_piece_tables_when_asked(tablebase6, tablebase_pieces):
    simulations = 400

    line, root_moves = search_in_rust(SEVEN_PIECES, simulations, tablebase_pieces)

    assert (root_moves.get("d1d5", 0) > simulations / 2) == (tablebase_pieces == 6)
    line, root_moves = search_in_rust(SIX_PIECES, simulations, tablebase_pieces)
    assert line[0] == "d1d5" and root_moves["d1d5"] > simulations / 2

#with the 6 piece dtz tables the engine plays the tables' move at 6 pieces without a search; by default it searches
def test_the_engine_takes_the_tables_move_at_6_pieces(tablebase6):
    if ranked_pieces(6) < 6:
        pytest.skip("the 6 piece dtz tables are not downloaded")
    board = chess.Board(SIX_PIECES)

    report = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, tablebase_pieces=6).search(board, simulations=400)
    default = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True).search(board, simulations=400)

    assert report.simulations == 0 and report.line == [ranked_moves(tablebase6, board)[0].move]
    assert default.simulations > 0

#the table sets with the 6 piece one as its wdl tables alone: hard links in a folder of their own, removed afterwards (a
#link left in pytest's temporary folders would keep 73 GB on the disk after the tables are deleted)
@pytest.fixture(scope="module")
def wdl_only_table_sets(tablebase6, tmp_path_factory):
    five, six = TABLE_SETS
    wdl_only = TableSet(6, tmp_path_factory.mktemp("syzygy6"), six.lichess_name, six.tables_per_half)
    try:
        try:
            for table in six.folder.glob("*.rtbw"):
                os.link(table, wdl_only.folder / table.name)
        except OSError as error:
            pytest.skip(f"no hard links from {six.folder} to {wdl_only.folder}: {error}")
        yield five, wdl_only
    finally:
        shutil.rmtree(wdl_only.folder)

#with only the 6 piece wdl tables the engine searches a 6 piece root (probing no dtz table) and takes the rook, which the
#tables call a win; from 7 pieces its search sees that win in the 6 pieces after the capture
@pytest.mark.parametrize("fen", [SIX_PIECES, SEVEN_PIECES])
def test_the_engine_searches_6_pieces_without_their_dtz_tables(wdl_only_table_sets, monkeypatch, fen):
    monkeypatch.setattr("tools.tablebase.TABLE_SETS", wdl_only_table_sets)

    report = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, tablebase_pieces=6).search(chess.Board(fen), 400)

    #smart pruning stops the search once the move is decided
    takes_the_rook = chess.Move.from_uci("d1d5")
    assert report.line[0] == takes_the_rook and dict(report.root_moves)[takes_the_rook] > report.simulations / 2
