import os
from pathlib import Path
import shutil

import chess
import chess.syzygy
import numpy as np
import pytest
import torch

from engine import mcts
import selfplay_rs
from model.encoding import TOKENS_PER_BOARD
from fake_models import UniformModel
from engine.inference import cpu_batches
from reference_selfplay import final_result
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from engine.engine_search import EngineSearch
from engine.player import choose_move
from engine.tablebase import TABLE_SETS, TABLEBASE_PIECES, TableSet, complete_pieces, covers, open_tablebase, open_tablebase_path, ranked_moves, ranked_pieces
from engine.uci import Options

CPU = torch.device("cpu")
#Qxd5+ wins: from 7 pieces it leaves 6 (a 6 piece table's), from 6 pieces it leaves 5
SEVEN_PIECES = "6k1/1p5p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"
SIX_PIECES = "6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"


#a quiet best move whose win needs 101 plies to a capture or pawn move stays a cursed win, as the root probe says
@pytest.mark.parametrize("fen", ["8/8/4k3/4bb2/2K5/2N5/8/8 b - - 7 79", "8/8/8/2nk1K2/8/4B3/4B3/8 w - - 11 67", "6B1/3kn3/8/8/8/2B1K3/8/8 w - - 7 84"])
def test_the_best_move_keeps_the_roots_result_at_the_50_move_edge(tablebase, fen):
    board = chess.Board(fen)
    assert tablebase.probe_wdl(board) == 1

    assert ranked_moves(tablebase, board)[0].wdl == 1


#6 pieces at the root; taking the rook leaves 5 and a won position, which only the tables can tell the search
def test_the_search_scores_the_positions_the_tables_cover(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)
    simulations = 200

    root = mcts.search(UniformModel(), chess.Board(SIX_PIECES), CPU, simulations)

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
    assert final_result(chess.Board(SIX_PIECES)) is None #6 pieces


#6 pieces are the 6 piece tables' (--tablebase-pieces 6), not the default's; castling rights are no one's
def test_the_tables_cover_the_pieces_asked_for():
    six, seven = chess.Board(SIX_PIECES), chess.Board(SEVEN_PIECES)

    assert covers(six, 6) and not covers(six)
    assert not covers(seven, 6)
    assert not covers(chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"), 6)


#empty table files of each half (python-chess and shakmaty only take files named for a table), a folder for each size
def fake_table_sets(folder: Path, monkeypatch, tables: dict[int, tuple[int, int]]) -> tuple[TableSet, ...]:
    table_sets = tuple(TableSet(pieces, folder / str(pieces), str(pieces), 3) for pieces in (5, 6))
    for table_set in table_sets:
        table_set.folder.mkdir()
        wdl, dtz = tables[table_set.pieces]
        for name in [f"fake{index}.rtbw" for index in range(wdl)] + [f"fake{index}.rtbz" for index in range(dtz)]:
            (table_set.folder / name).touch()
    monkeypatch.setattr("engine.tablebase.TABLE_SETS", table_sets)
    return table_sets

#every wdl table and every 3-5 piece dtz table must be there; the 6 piece dtz tables may all be left out (then 6 pieces
#are searched, not ranked). An unfinished download is refused
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

#empty files named for every table of up to `pieces` pieces but those left out: the 3-5 piece ones in one folder, each
#bigger size's in its own, as a SyzygyPath
def fake_syzygy_path(folder: Path, left_out: tuple[str, ...] = (), pieces: int = 6) -> str:
    names = list(chess.syzygy.tablenames(piece_count=pieces))
    sizes = [(3, 4, 5), *((size,) for size in range(6, pieces + 1))]
    #as many as the downloaded sets have
    assert [sum(len(name) - 1 in group for name in names) for group in sizes] == [table_set.tables_per_half for table_set in TABLE_SETS[:len(sizes)]]
    folders = [folder / "".join(map(str, group)) for group in sizes]
    for size_folder in folders:
        size_folder.mkdir()
    for name in names:
        for extension in ("rtbw", "rtbz"):
            if f"{name}.{extension}" not in left_out:
                (folders[max(len(name) - 6, 0)] / f"{name}.{extension}").touch()
    return os.pathsep.join(str(size_folder) for size_folder in folders)

def tables_of(pieces: int) -> list[str]:
    return [name for name in chess.syzygy.tablenames(piece_count=pieces) if len(name) == pieces + 1]

SEVEN_PIECE_TABLES = tables_of(7)

#a SyzygyPath gives the most pieces it holds up to the limit, as stockfish does: a 7 piece table left out gives 6, the 7
#piece dtz tables left out as a whole still 7 (searched, not ranked), a 5 piece dtz table 4
@pytest.mark.parametrize("left_out, limit, pieces", [((), 7, 7), ((), 6, 6), ((f"{SEVEN_PIECE_TABLES[0]}.rtbw",), 7, 6),
                                                     (tuple(f"{name}.rtbz" for name in SEVEN_PIECE_TABLES), 7, 7), (("KQRvKR.rtbz",), 7, 4)])
def test_a_syzygy_path_gives_the_most_pieces_it_holds(tmp_path, left_out, limit, pieces):
    assert complete_pieces(fake_syzygy_path(tmp_path, left_out, pieces=7), limit) == pieces

#SyzygyProbeLimit defaults to 7, so TCEC's tables need no limit set. A folder without every table up to 5 pieces is refused
def test_the_engine_takes_the_7_piece_tables_it_is_given(tmp_path):
    for name in ("seven", "six", "none"):
        (tmp_path / name).mkdir()
    seven = fake_syzygy_path(tmp_path / "seven", pieces=7)
    six = fake_syzygy_path(tmp_path / "six", (f"{SEVEN_PIECE_TABLES[0]}.rtbw",), pieces=7)
    options = Options(False, TABLEBASE_PIECES)

    assert "option name SyzygyProbeLimit type spin default 7 min 5 max 7" in options.lines()
    assert options.set("SyzygyPath", seven, built=False) is None and options.search()[1]["tablebase_pieces"] == 7
    assert options.set("SyzygyPath", six, built=False) is None and options.search()[1]["tablebase_pieces"] == 6
    assert options.set("SyzygyProbeLimit", "5", built=False) is None and options.search()[1]["tablebase_pieces"] == 5
    assert options.set("SyzygyPath", seven, built=False) is None and options.search()[1]["tablebase_pieces"] == 5
    assert options.set("SyzygyPath", str(tmp_path / "none"), built=False).endswith("the tables stay as they were")
    assert options.search()[1]["syzygy_path"] == seven

#a SyzygyPath's folders are checked as test_which_tables_open does, by the tables' names across them
@pytest.mark.parametrize("left_out, ranked", [((), 6), (tuple(f"{name}.rtbz" for name in tables_of(6)), 5),
                                              (("KRPvKRP.rtbz",), None), (("KRPvKRP.rtbw",), None), (("KQvKR.rtbz",), None)])
def test_which_tables_open_from_a_syzygy_path(tmp_path, left_out, ranked):
    path = fake_syzygy_path(tmp_path, left_out)

    if ranked is None:
        with pytest.raises(FileNotFoundError):
            open_tablebase_path(path, 6)
    else:
        tables, ranked_by_path = open_tablebase_path(path, 6)
        tables.close()
        assert ranked_by_path == ranked

def test_a_syzygy_path_of_5_pieces(tmp_path):
    path = fake_syzygy_path(tmp_path)

    tables, ranked = open_tablebase_path(path.split(os.pathsep)[0], 5)
    tables.close()
    assert ranked == 5
    with pytest.raises(FileNotFoundError):
        open_tablebase_path(str(tmp_path / "nowhere"), 5)

def test_the_default_tables_are_the_5_piece_ones(tmp_path, monkeypatch):
    five, six = fake_table_sets(tmp_path, monkeypatch, {5: (3, 3), 6: (0, 0)})

    open_tablebase().close()
    assert ranked_pieces(5) == 5
    settings = rust_search_settings(True)
    assert (settings["tablebase_folder"], settings["tablebase_pieces"]) == (str(five.folder), 5)
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

#a root the tables cover is searched, and below it only the moves after a capture or pawn move are the tables': the rook
#taken, or the mate, gets the visits rather than sharing them with every move the tables call a win (lc0 probes so)
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

#a 6 piece position (6 piece folder) and the 5 piece one after its capture (5 piece folder), both lost for black per lichess
def test_python_probes_both_folders(tablebase6):
    six, five = chess.Board("6k1/1p5p/8/3Q4/8/8/P7/6K1 b - - 0 1"), chess.Board("6k1/7p/8/3Q4/8/8/P7/6K1 b - - 0 1")

    assert tablebase6.probe_wdl(six) == tablebase6.probe_wdl(five) == -2

#from 7 pieces, taking the rook leaves a won 6 piece position that only the 6 piece tables tell the search; rust probes
#them with the 5 piece ones (the positions after a capture)
@pytest.mark.parametrize("tablebase_pieces", [5, 6])
def test_rust_probes_the_6_piece_tables_when_asked(tablebase6, tablebase_pieces):
    simulations = 400

    _, root_moves = search_in_rust(SEVEN_PIECES, simulations, tablebase_pieces)

    assert (root_moves.get("d1d5", 0) > simulations / 2) == (tablebase_pieces == 6)
    line, root_moves = search_in_rust(SIX_PIECES, simulations, tablebase_pieces)
    assert line[0] == "d1d5" and root_moves["d1d5"] > simulations / 2

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
    five, six = TABLE_SETS[:2]
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
    monkeypatch.setattr("engine.tablebase.TABLE_SETS", wdl_only_table_sets)

    report = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, tablebase_pieces=6).search(chess.Board(fen), 400)

    #certainty proves the capture won by the tables, and the search stops once its move is proven
    takes_the_rook = chess.Move.from_uci("d1d5")
    assert report.simulations > 0 and report.line[0] == takes_the_rook and report.score == 1.0
