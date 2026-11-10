import os
from pathlib import Path
import shutil
import time

import chess
import chess.syzygy
import numpy as np
import pytest
import torch

from engine import engine_search, mcts
import selfplay_rs
from model.encoding import TOKENS_PER_BOARD
from fake_models import UniformModel
from engine.inference import cpu_batches
from reference_selfplay import final_result
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from engine.engine_search import EngineSearch
from engine.tablebase import (SYZYGY_DIR, TABLE_SETS, TABLEBASE_PIECES, TableSet, covers, found_tables, most_pieces, open_tablebase, open_tablebase_path,
                              path_folders, ranked_moves)
from engine.uci import Options

CPU = torch.device("cpu")
#Qxd5+ wins, leaving 6 pieces from SEVEN_PIECES and 5 from SIX_PIECES
SEVEN_PIECES = "6k1/1p5p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"
SIX_PIECES = "6k1/7p/8/3r4/8/8/P7/3Q2K1 w - - 0 1"


#a quiet best move whose win needs 101 plies to a capture or pawn move stays a cursed win, as the root probe says
@pytest.mark.parametrize("fen", ["8/8/4k3/4bb2/2K5/2N5/8/8 b - - 7 79", "8/8/8/2nk1K2/8/4B3/4B3/8 w - - 11 67", "6B1/3kn3/8/8/8/2B1K3/8/8 w - - 7 84"])
def test_the_best_move_keeps_the_roots_result_at_the_50_move_edge(tablebase, fen):
    board = chess.Board(fen)
    assert tablebase.probe_wdl(board) == 1

    assert ranked_moves(tablebase, board)[0].wdl == 1


#the tables assume a clock of 0: a quiet move's win or loss needs the clock plus its plies to reach the zeroing move within 100, else it is a draw
@pytest.mark.parametrize("fen", ["6K1/1B5N/8/8/8/8/8/3k4 w - - 0 1", "6K1/1B5N/8/8/8/8/8/3k4 b - - 0 1"])
def test_a_result_the_halfmove_clock_makes_a_draw_is_ranked_as_one(tablebase, fen):
    board = chess.Board(fen)
    best = ranked_moves(tablebase, board)[0]
    assert abs(best.wdl) == 2 and best.plies_to_zeroing > 1

    board.halfmove_clock = 100 - best.plies_to_zeroing
    assert ranked_moves(tablebase, board)[0].wdl == best.wdl
    board.halfmove_clock += 1
    assert ranked_moves(tablebase, board)[0].wdl == best.wdl // 2


def test_the_report_scores_a_win_the_halfmove_clock_makes_a_draw_as_a_draw(tablebase):
    board = chess.Board("6K1/1B5N/8/8/8/8/8/3k4 w - - 0 1")
    search = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True)

    assert search.search(board, simulations=10).score == 1.0
    board.halfmove_clock = 90
    assert search.search(board, simulations=10).score == 0.5


#tables that take longer than their share of the search's time (a slow disk) leave the root to the search
def test_a_root_the_tables_rank_too_slowly_is_searched(tablebase, monkeypatch):
    board = chess.Board("6K1/1B5N/8/8/8/8/8/3k4 w - - 0 1")
    search = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True)

    assert ranked_moves(tablebase, board, time.perf_counter()) is None
    assert ranked_moves(tablebase, board, time.perf_counter() + 60) == ranked_moves(tablebase, board)
    assert search.search(board, seconds=0.5).simulations == 0
    monkeypatch.setattr(engine_search, "TABLE_MOVE_SHARE", 0.0)
    report = search.search(board, seconds=0.5)
    assert report.simulations > 0 and report.line[0] in board.legal_moves


def test_the_search_scores_the_positions_the_tables_cover(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)
    simulations = 200

    root = mcts.search(UniformModel(), chess.Board(SIX_PIECES), CPU, simulations)

    assert root.children[chess.Move.from_uci("d1d5")].visit_count > simulations / 2
    with pytest.raises(ValueError):
        mcts.search(UniformModel(), chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1"), CPU)


def test_self_play_ends_a_game_the_tables_know(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "tablebase", tablebase)

    assert final_result(chess.Board("k7/8/1K6/8/8/8/8/7R b - - 0 1")) == 1.0 #black to move, and lost
    assert final_result(chess.Board("6k1/5r2/8/8/8/8/1R6/2K5 w - - 0 1")) == 0.5
    assert final_result(chess.Board(SIX_PIECES)) is None


#the tables have no positions with castling rights
def test_the_tables_cover_the_pieces_asked_for():
    six, seven = chess.Board(SIX_PIECES), chess.Board(SEVEN_PIECES)

    assert covers(six, 6) and not covers(six)
    assert not covers(seven, 6)
    assert not covers(chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"), 6)


#empty files for each half and a folder for each size; python-chess and shakmaty skip files not named for a table
def fake_table_sets(folder: Path, monkeypatch, tables: dict[int, tuple[int, int]]) -> tuple[TableSet, ...]:
    table_sets = tuple(TableSet(pieces, folder / str(pieces), str(pieces), 3) for pieces in tables)
    for table_set in table_sets:
        table_set.folder.mkdir()
        wdl, dtz = tables[table_set.pieces]
        for name in [f"fake{index}.rtbw" for index in range(wdl)] + [f"fake{index}.rtbz" for index in range(dtz)]:
            (table_set.folder / name).touch()
    monkeypatch.setattr("engine.tablebase.TABLE_SETS", table_sets)
    return table_sets

#all wdl tables and the 3-5 piece dtz ones are required; the 6 piece dtz ones may all be left out (then searched, not ranked), not partly
@pytest.mark.parametrize("five, six, opens", [((3, 3), (3, 3), True), ((3, 3), (3, 0), True), ((3, 3), (2, 3), False),
                                              ((3, 3), (3, 1), False), ((3, 0), (3, 3), False)])
def test_which_tables_open(tmp_path, monkeypatch, five, six, opens):
    fake_table_sets(tmp_path, monkeypatch, {5: five, 6: six})

    if opens:
        open_tablebase(6).close()
    else:
        with pytest.raises(FileNotFoundError):
            open_tablebase(6)

def fake_syzygy_path(folder: Path, left_out: tuple[str, ...] = (), pieces: int = 6) -> str:
    names = list(chess.syzygy.tablenames(piece_count=pieces))
    sizes = [(3, 4, 5), *((size,) for size in range(6, pieces + 1))]
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

#the most pieces a SyzygyPath holds up to the limit, as stockfish counts them
@pytest.mark.parametrize("left_out, limit, pieces", [((), 7, 7), ((), 6, 6), ((f"{SEVEN_PIECE_TABLES[0]}.rtbw",), 7, 7),
                                                     (tuple(f"{name}.rtbw" for name in SEVEN_PIECE_TABLES), 7, 6),
                                                     (tuple(f"{name}.rtbz" for name in SEVEN_PIECE_TABLES), 7, 7), (("KQRvKR.rtbz",), 7, 7)])
def test_a_syzygy_path_gives_the_most_pieces_it_holds(tmp_path, left_out, limit, pieces):
    assert most_pieces(fake_syzygy_path(tmp_path, left_out, pieces=7), limit) == pieces

#a junction on windows, where symlinks need admin rights
def dead_link(link: Path) -> None:
    gone = link.with_name(link.name + ".gone")
    gone.mkdir()
    try:
        os.symlink(gone, link)
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(gone), str(link))
    gone.rmdir()

def test_a_link_to_nothing_is_no_table(tmp_path):
    path = fake_syzygy_path(tmp_path, (f"{SEVEN_PIECE_TABLES[0]}.rtbw",), pieces=7)
    dead_link(Path(path.split(os.pathsep)[-1]) / f"{SEVEN_PIECE_TABLES[0]}.rtbw")

    assert most_pieces(path, 7) == 7
    assert SEVEN_PIECE_TABLES[0] not in found_tables(path_folders(path))["rtbw"]

#SyzygyProbeLimit defaults to 7 so TCEC's tables need no limit set; a folder without the tables is refused
def test_the_engine_takes_the_7_piece_tables_it_is_given(tmp_path):
    for name in ("seven", "six", "none"):
        (tmp_path / name).mkdir()
    seven = fake_syzygy_path(tmp_path / "seven", pieces=7)
    six = fake_syzygy_path(tmp_path / "six", tuple(f"{name}.rtbw" for name in SEVEN_PIECE_TABLES), pieces=7)
    options = Options(False, TABLEBASE_PIECES)

    assert "option name SyzygyProbeLimit type spin default 7 min 5 max 7" in options.lines()
    assert options.set("SyzygyPath", seven, built=False) is None and options.search()[1]["tablebase_pieces"] == 7
    assert options.set("SyzygyPath", six, built=False) is None and options.search()[1]["tablebase_pieces"] == 6
    assert options.set("SyzygyProbeLimit", "5", built=False) is None and options.search()[1]["tablebase_pieces"] == 5
    assert options.set("SyzygyPath", seven, built=False) is None and options.search()[1]["tablebase_pieces"] == 5
    assert options.set("SyzygyPath", str(tmp_path / "none"), built=False).endswith("holds no tables: the tables stay as they were")
    assert options.set("SyzygyPath", str(tmp_path / "nowhere"), built=False).endswith("the tables stay as they were")
    assert options.search()[1]["syzygy_path"] == seven

def partial_syzygy_path(folder: Path, left_out: tuple[str, ...], dead: tuple[str, ...] = ()) -> str:
    folder.mkdir()
    for table in SYZYGY_DIR.glob("*.rtb?"):
        if table.name in dead:
            dead_link(folder / table.name)
        elif table.name not in left_out:
            os.link(table, folder / table.name)
    return str(folder)

#a root missing a table is searched; the rust search must open past a link to nothing
@pytest.mark.parametrize("left_out, dead", [(("KRvK.rtbz",), ()), ((), ("KRvK.rtbw",))])
def test_a_root_missing_a_table_is_searched(tablebase, tmp_path, left_out, dead):
    path = partial_syzygy_path(tmp_path / "tables", left_out, dead)
    search = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, syzygy_path=path)
    rook, queen = chess.Board("k7/8/1K6/8/8/8/8/7R w - - 0 1"), chess.Board("k7/8/1K6/8/8/8/7Q/8 w - - 0 1")

    assert ranked_moves(open_tablebase_path(path), rook) is None and rook.fen() == "k7/8/1K6/8/8/8/8/7R w - - 0 1"
    assert search.search(rook, simulations=200).simulations > 0
    assert search.search(queen, simulations=200).simulations == 0

def test_a_syzygy_path_of_5_pieces(tmp_path):
    path = fake_syzygy_path(tmp_path)

    open_tablebase_path(path.split(os.pathsep)[0]).close()
    with pytest.raises(FileNotFoundError):
        open_tablebase_path(str(tmp_path / "nowhere"))

def test_the_default_tables_are_the_5_piece_ones(tmp_path, monkeypatch):
    five, six = fake_table_sets(tmp_path, monkeypatch, {5: (3, 3), 6: (0, 0)})

    open_tablebase().close()
    settings = rust_search_settings(True)
    assert (settings["tablebase_folder"], settings["tablebase_pieces"]) == (str(five.folder), 5)
    assert rust_search_settings(True, 6)["tablebase_folder"] == os.pathsep.join([str(five.folder), str(six.folder)])


#(line, visits by root move) of one rust search run to its end
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

#a covered root is searched; below it only positions after a capture or pawn move are probed, so the winning move takes the visits
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

def test_python_probes_both_folders(tablebase6):
    six, five = chess.Board("6k1/1p5p/8/3Q4/8/8/P7/6K1 b - - 0 1"), chess.Board("6k1/7p/8/3Q4/8/8/P7/6K1 b - - 0 1")

    assert tablebase6.probe_wdl(six) == tablebase6.probe_wdl(five) == -2

#from 7 pieces the rook taken leaves a won 6 piece position, which only the 6 piece tables tell the search
@pytest.mark.parametrize("tablebase_pieces", [5, 6])
def test_rust_probes_the_6_piece_tables_when_asked(tablebase6, tablebase_pieces):
    simulations = 400

    _, root_moves = search_in_rust(SEVEN_PIECES, simulations, tablebase_pieces)

    assert (root_moves.get("d1d5", 0) > simulations / 2) == (tablebase_pieces == 6)
    line, root_moves = search_in_rust(SIX_PIECES, simulations, tablebase_pieces)
    assert line[0] == "d1d5" and root_moves["d1d5"] > simulations / 2

def test_the_engine_takes_the_tables_move_at_6_pieces(tablebase6):
    if not TABLE_SETS[1].count("rtbz"):
        pytest.skip("the 6 piece dtz tables are not downloaded")
    board = chess.Board(SIX_PIECES)

    report = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, tablebase_pieces=6).search(board, simulations=400)
    default = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True).search(board, simulations=400)

    assert report.simulations == 0 and report.line == [ranked_moves(tablebase6, board)[0].move]
    assert default.simulations > 0

#the 6 piece table set as its wdl tables alone: hard links in a folder of their own, removed after (pytest's temporary folders would keep the disk space)
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

#without dtz tables a 6 piece root is searched, not ranked, and still takes the rook; from 7 pieces the search sees the win after the capture
@pytest.mark.parametrize("fen", [SIX_PIECES, SEVEN_PIECES])
def test_the_engine_searches_6_pieces_without_their_dtz_tables(wdl_only_table_sets, monkeypatch, fen):
    monkeypatch.setattr("engine.tablebase.TABLE_SETS", wdl_only_table_sets)

    report = EngineSearch(cpu_batches(UniformModel()), 16, use_tablebase=True, tablebase_pieces=6).search(chess.Board(fen), 400)

    #certainty proves the capture won, and the search stops once its move is proven
    takes_the_rook = chess.Move.from_uci("d1d5")
    assert report.simulations > 0 and report.line[0] == takes_the_rook and report.score == 1.0
