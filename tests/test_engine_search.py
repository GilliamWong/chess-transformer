#the uci engine's search (engine/engine_search.py): rounds of many leaves (selfplay_rs.Searches.descend), its
#deadline and node limit, smart pruning, the tree kept for the next move, the engine's own settings, and uci.py's go
#limits and clock, the commands it takes while it searches, its options and the positions it refuses
import io
import queue
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

import selfplay_rs
from torch import nn
from model.encoding import TOKENS_PER_BOARD, encode_move_to_id
from fake_models import BoardModel, CountingNetwork, UniformModel
from engine.inference import ImmediateBatches, cpu_batches
from model.model import ChessTransformer
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from tools.checkpoint import save_checkpoint
from tools.tablebase import SYZYGY_DIR
from engine import engine_search
from engine.engine_search import EngineSearch, engine_search_settings, parse_changes
from engine.player import SearchReport
from engine import uci
from engine.uci import MOST_EXTENSION, MOST_OF_THE_TIME_LEFT, MOVE_OVERHEAD_SECONDS, RESERVE_SECONDS, Clock

REPOSITORY = Path(__file__).parent.parent


#the searches here are mcts.py's, whose constants the expectations were set by; the engine's own changes over them are
#test_the_engines_settings'
@pytest.fixture(autouse=True)
def mcts_constants(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {})


def leaf_boards(leaves: bytes) -> np.ndarray:
    return np.frombuffer(bytearray(leaves), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)

#the first round finds only the root to evaluate; then a round spreads over the root's moves, and a round gathered
#while another is out takes none of its leaves again
def test_a_round_gathers_different_leaves():
    evaluate = cpu_batches(UniformModel()).evaluate_boards
    searches = selfplay_rs.Searches(**rust_search_settings(False))
    searches.start(0, chess.STARTING_FEN, [], 1000)

    root = leaf_boards(searches.descend(16, 16))
    searches.finish(evaluate(root))
    first, second = leaf_boards(searches.descend(8, 16)), leaf_boards(searches.descend(8, 16))
    searches.finish(evaluate(first))
    searches.finish(evaluate(second))

    assert len(root) == 1
    assert len(first) == len(second) == 8
    assert len({board.tobytes() for board in np.concatenate([first, second])}) == 16


#a position reached again by another order of moves takes its answer from the cache: the same search with fewer
#network calls
def test_the_cache_answers_transpositions_as_the_network_would():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    found = {}
    for cache_positions in (0, 100_000):
        searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=cache_positions)
        searches.start(0, chess.STARTING_FEN, [], 1000)
        evaluated, reports = 0, []
        while not reports:
            boards = leaf_boards(searches.descend())
            evaluated += len(boards)
            reports = searches.finish(evaluate(boards))
        found[cache_positions] = (reports[0], evaluated)

    (without, evaluated_without), (with_cache, evaluated_with_cache) = found[0], found[100_000]
    assert with_cache == without
    assert evaluated_with_cache < 0.9 * evaluated_without


#one leaf a round is mcts.py's search (Searches' defaults, test_selfplay_rs.py) at every move, as evaluate_elo's logged
#results were played, whatever the engine's own settings: a second round out on the network would change it. From a
#middlegame, where captures move the material count, so that an answer changes the next choice
def test_one_leaf_a_round_is_the_exact_search_at_every_move(monkeypatch, carlsen_games):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    search = EngineSearch(cpu_batches(BoardModel()), 1)
    board = carlsen_games[0].board()
    for move in list(carlsen_games[0].mainline_moves())[:20]:
        board.push(move)
    for _ in range(8):
        report = search.search(board, simulations=60)

        exact = selfplay_rs.Searches(**rust_search_settings(False))
        exact.start(0, board.root().fen(), [played.uci() for played in board.move_stack], 60)
        reports = []
        while not reports:
            reports = exact.finish(evaluate(leaf_boards(exact.descend())))
        _, line, score, root_moves, visits = reports[0]
        assert ([played.uci() for played in report.line], report.score, report.simulations) == (line, score, visits)
        assert [(played.uci(), move_visits) for played, move_visits in report.root_moves] == root_moves[:len(report.root_moves)]
        board.push(report.line[0])


#the engine's settings are mcts.py's with its own changes over them, and a caller's over those
def test_the_engines_settings(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "fpu_reduction": 0.25})
    assert engine_search_settings(False) == {**rust_search_settings(False), "fpu_reduction": 0.25, "c_puct": 0.5}
    assert engine_search_settings(False, {"c_puct": 0.7})["c_puct"] == 0.7
    assert engine_search_settings(False, own=False) == rust_search_settings(False)


#every position's answer: no move preferred, expected score 0.6
def answers_of_0_6(boards: np.ndarray) -> np.ndarray:
    answers = np.zeros((len(boards), ANSWER_WIDTH), dtype=np.float32)
    answers[:, -1] = 0.6
    return answers


#the network's value goes into the search and the cache sharpened (its logit times value_sharpness), so a position the
#cache answers is not sharpened again; 1 leaves the value as it was
@pytest.mark.parametrize("sharpness, expected", [(1.0, 0.6), (3.0, 1 / (1 + (0.4 / 0.6) ** 3))])
def test_the_value_is_sharpened_once(sharpness, expected):
    searches = selfplay_rs.Searches(**rust_search_settings(False), cache_positions=1000, value_sharpness=sharpness)
    scores = []
    for _ in range(2):
        searches.start(0, chess.STARTING_FEN, [], 1)
        scores.append(searches.finish(answers_of_0_6(leaf_boards(searches.descend())))[0][2])
    assert scores == pytest.approx([expected, expected])


#the engine reports its score on the network's own scale, as before the value was sharpened
def test_the_engine_reports_the_score_unsharpened():
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, settings={"value_sharpness": 3.0})
    assert search.search(chess.Board(), simulations=1).score == pytest.approx(0.6)


#with leaves out on the network, rounds stop at the search's simulations (without smart pruning, which stops sooner)
@pytest.mark.parametrize("simulations", [1, 37, 300])
def test_a_search_of_many_leaves_a_round_stops_at_its_simulations(carlsen_games, simulations):
    board = carlsen_games[1].board()
    for move in list(carlsen_games[1].mainline_moves())[:10]:
        board.push(move)
    search = EngineSearch(cpu_batches(BoardModel()), 16)
    search.smart_pruning = False

    report = search.search(board, simulations=simulations)

    assert report.simulations == simulations and report.line[0] in board.legal_moves


class SizesNetwork:
    def __init__(self, model):
        self.network = cpu_batches(model)
        self.sizes = []

    def launch(self, boards: np.ndarray) -> np.ndarray:
        self.sizes.append(len(boards))
        return self.network.launch(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return self.network.collect(launched)


#a fresh tree's rounds start at one leaf and grow to 64, still stopping at the simulations; fixed rounds are large at once
def test_a_rounds_leaves_grow_with_the_search(carlsen_games):
    board = carlsen_games[1].board()
    for move in list(carlsen_games[1].mainline_moves())[:10]:
        board.push(move)
    growing, fixed = SizesNetwork(BoardModel()), SizesNetwork(BoardModel())
    growing_search, fixed_search = EngineSearch(growing, 64, grow_leaves=16), EngineSearch(fixed, 64, grow_leaves=0)
    growing_search.smart_pruning = fixed_search.smart_pruning = False

    report = growing_search.search(board, simulations=3000)
    fixed_search.search(board, simulations=3000)

    assert report.simulations == 3000
    assert growing.sizes[:4] == [1, 1, 1, 1] and max(growing.sizes) == 64
    assert fixed.sizes[1] > 16 and len(growing.sizes) > len(fixed.sizes)


#a leaf per 16 simulations done or left (the fewer), one round out until a round takes all 64
@pytest.mark.parametrize("visits, simulations, expected", [(0, 1000, (1, 1)), (160, 1000, (10, 1)), (1024, None, (64, 2)), (995, 1000, (1, 1))])
def test_the_leaves_of_a_round(visits, simulations, expected):
    class Searches:
        def root_visits(self, game):
            return visits, 0, 0
    search = EngineSearch(cpu_batches(BoardModel()), 64, grow_leaves=16)
    search.searches = Searches()

    assert search.round_leaves(simulations) == expected


#a free queen, which the material count finds, and mates in one, which a flat eval finds by searching every move
@pytest.mark.parametrize("model, fen, best", [
    (BoardModel(), "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    (UniformModel(), "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
    (UniformModel(), "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
    (UniformModel(), "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),
])
@pytest.mark.parametrize("leaves", [1, 16, 64])
def test_many_leaves_a_round_find_what_one_finds(model, fen, best, leaves):
    report = EngineSearch(cpu_batches(model), leaves).search(chess.Board(fen), simulations=400)

    assert report.line[0].uci() == best


def test_a_timed_search_ends_by_its_deadline():
    search = EngineSearch(cpu_batches(BoardModel()), 16)

    started = time.perf_counter()
    report = search.search(chess.Board(), seconds=0.3)

    assert report.simulations > 16 and time.perf_counter() - started < 0.3 + 0.1


#a network with lanes, as a wrapper passes on GraphedBatches'
class Laned:
    def __init__(self, network, lanes: list):
        self.network, self.lanes = network, lanes

    def launch(self, boards: np.ndarray):
        return self.network.launch(boards)

    def collect(self, launched):
        return self.network.collect(launched)


#the deadline counts rounds as coming back together only when they run at once: many leaves a round on two lanes or more
def test_rounds_run_at_once_only_on_lanes_of_their_own():
    assert EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 4).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0, 0]), 1).rounds_at_once
    assert not EngineSearch(Laned(cpu_batches(BoardModel()), [0]), 4).rounds_at_once
    assert not EngineSearch(cpu_batches(BoardModel()), 4).rounds_at_once


#the same position again needs only the simulations it lacks; the reply to the move played is searched from the kept
#tree; another game's position gets a tree of its own
def test_the_next_search_carries_on_the_last_tree():
    network = CountingNetwork(BoardModel())
    search = EngineSearch(network, 16)
    search.smart_pruning = False
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    first = search.search(board, simulations=400)

    network.boards = 0
    assert search.search(board, simulations=600).simulations == 600
    assert 0 < network.boards <= 200

    board.push(first.line[0])
    board.push(first.line[1])
    network.boards = 0
    assert search.search(board, simulations=400).simulations == 400
    assert network.boards < 400

    other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    network.boards = 0
    report = search.search(other, simulations=100)
    assert report.line[0] in other.legal_moves and network.boards > 50


#the side to move's clock, milliseconds as seconds, and --simulations when go sets no limit (go infinite)
def test_go_limits():
    clock = Clock()
    white, black = chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert clock.limits("movetime 1000".split(), white, 800) == (None, pytest.approx(1 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("movetime 500 nodes 100".split(), white, 800) == (100, pytest.approx(0.5 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("nodes 5000".split(), black, 800) == (5000, None)
    assert clock.limits(["infinite"], black, 800) == (800, None)

    nodes, seconds = clock.limits("wtime 60000 btime 30000 winc 1000 binc 500".split(), black, 800)
    assert nodes is None and seconds == pytest.approx(Clock().seconds_for_move(30 - MOVE_OVERHEAD_SECONDS, 0.5, None, 32))
    assert 0.5 < seconds < 30 - RESERVE_SECONDS


#the (nodes, seconds) each of the gos from the start position gave a search, and the lines the engine sent
def searched_gos(monkeypatch, gos: list[str]) -> tuple[list, list[str]]:
    given, sent = [], []
    class Player:
        def search(self, board, nodes, seconds, extension, stop=None, info=None):
            given.append((nodes, seconds))
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["position startpos", *gos, "quit"])))
    uci.main(Path("unused.pt"), selection="mcts", simulations=800)
    return given, sent

#go nodes 0 (or less) plays at once, a search of one simulation, as lc0 answers at once (Searches refused 0, which
#ended the engine)
def test_go_nodes_0_plays_at_once(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes 0", "go nodes -5", "go nodes 7"])
    assert given == [(1, None), (1, None), (7, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3


#a go number that is not a whole number is left out, the gui told so (int() raised, which ended the engine): that go is
#searched as if it had not come with it
def test_go_leaves_out_numbers_that_are_not_whole(monkeypatch):
    given, sent = searched_gos(monkeypatch, ["go nodes x", "go movetime 1.5 nodes 30", "go wtime 1000.5 btime 1000"])
    assert given == [(800, None), (30, None), (800, None)] and len([line for line in sent if line.startswith("bestmove")]) == 3
    told = [line for line in sent if line.startswith("info string")]
    assert told == [f"info string go's {number} is not a whole number: left out" for number in ("nodes x", "movetime 1.5", "wtime 1000.5")]


#the start position with pieces taken off, the kings kept, until `men` are left
def board_with_men(men: int) -> chess.Board:
    board = chess.Board()
    for square in [square for square in chess.SQUARES if board.piece_type_at(square) not in (None, chess.KING)][:32 - men]:
        board.remove_piece_at(square)
    return board


#what a search is given and does not use (smart pruning stopped it early) is banked, and the next move gets all of it;
#a movetime search is not on the clock, and no move gets the reserve
def test_unused_time_goes_to_the_next_move():
    clock, board = Clock(), chess.Board()
    _, first = clock.limits("wtime 60000 btime 60000".split(), board, 800)
    clock.searched(first / 2, 0)
    _, second = clock.limits("wtime 59000 btime 60000".split(), board, 800)
    _, without_saving = Clock().limits("wtime 59000 btime 60000".split(), board, 800)
    assert second == pytest.approx(clock.share + first / 2) and second > 1.4 * without_saving

    #a search that takes all it is given leaves the bank empty
    clock.searched(second, 0)
    _, third = clock.limits("wtime 58000 btime 60000".split(), board, 800)
    assert third == pytest.approx(Clock().limits("wtime 58000 btime 60000".split(), board, 800)[1])

    clock.searched(third / 4, 0)
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.1, 0)
    assert clock.saved == pytest.approx(3 / 4 * third)
    assert clock.limits("wtime 1500 btime 60000".split(), board, 800)[1] == pytest.approx(1.5 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)


#a game at 60+0.6 losing material as the games against BT4 do (16 men by move 40, 10 by 60, 6 by 85), every search
#taking all it is given: the middlegame moves get more than the opening's, the late middlegame starts with time to
#spend, the endgame lives on the increment, and the clock never runs below the reserve
def test_the_clock_follows_the_material_and_never_runs_out():
    points = [(1, 32), (15, 32), (40, 16), (60, 10), (85, 6), (120, 6)]
    men_at = lambda move: next(round(men + (later_men - men) * (move - start) / (end - start))
                               for (start, men), (end, later_men) in zip(points, points[1:]) if start <= move <= end)
    clock, left, spent, low, late_middlegame_left = Clock(), 60.0, {}, 60.0, None
    for move in range(1, 121):
        men = men_at(move)
        if men <= 15 and late_middlegame_left is None:
            late_middlegame_left = left
        _, seconds = clock.limits(f"wtime {round(left * 1000)} btime 60000 winc 600 binc 600".split(), board_with_men(men), 800)
        clock.searched(seconds, 0)
        left += 0.6 - seconds - MOVE_OVERHEAD_SECONDS
        spent.setdefault("opening" if men >= 26 else "middle game" if men >= 16 else "late", []).append(seconds)
        low = min(low, left)
    #the last 30 moves are at 6 men
    opening, middle_game, endgame = (sum(moves) / len(moves) for moves in (spent["opening"], spent["middle game"], spent["late"][-30:]))
    assert middle_game > 1.1 * opening and endgame < 0.7
    assert late_middlegame_left > 12 and low > RESERVE_SECONDS / 2


#a clock planned over fewer moves a man spends more now, and a move's cap is the part of the time left given
def test_a_clocks_plan_and_cap_can_be_set():
    board = chess.Board()
    arguments = "wtime 60000 btime 60000 winc 600 binc 600".split()
    assert Clock(moves_per_man=1.0).limits(arguments, board, 800)[1] > 1.25 * Clock().limits(arguments, board, 800)[1]
    assert Clock(most_of_the_time_left=0.01).limits(arguments, board, 800)[1] == pytest.approx((60 - MOVE_OVERHEAD_SECONDS) * 0.01)
    #late in the game the least moves left and the reserve decide the share
    board = board_with_men(4)
    arguments = "wtime 10000 btime 10000 winc 600 binc 600".split()
    assert Clock(least_moves_left=8, reserve_seconds=0.0).limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS) / 8 + 0.6)
    assert Clock().limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS) / 10 + 0.6)


#ucinewgame starts a fresh clock: searches that used little of their time bank it for later moves (the 8th move here
#gets over twice the first's), but the next game's first move gets what the first game's did. A search after the first
#may run on if its score falls below the last one's
def test_a_new_game_starts_a_fresh_clock(monkeypatch):
    given = []
    class Player:
        def search(self, board, nodes, seconds, extension, stop, info=None):
            given.append((seconds, extension))
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    go = ["position startpos", "go wtime 60000 btime 60000 winc 600 binc 600"]
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", lambda line: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["uci", "isready", *go * 8, "ucinewgame", *go, "quit"])))
    uci.main(Path("unused.pt"), selection="mcts")
    seconds = [move_seconds for move_seconds, _ in given]
    assert seconds[7] > 2 * seconds[0] and seconds[8] == seconds[0]
    assert given[0][1] is None and given[8][1] is None
    last_score, extra = given[1][1]
    assert last_score == 0 and extra > 0


#the start position's men with `ply` plies before it, as a book line's fen counts them
def board_at_ply(ply: int) -> chess.Board:
    board = chess.Board()
    board.turn, board.fullmove_number = ply % 2 == 0, 1 + ply // 2
    return board

LC0_FIRST_MOVE = {"first_move_bonus": 1.8, "book_ply_bonus": 0.25}
CLOCK_60 = "wtime 60000 btime 60000 winc 600 binc 600".split()

#lc0's first move bonus: the game's first search on the clock gets 1 + 1.8 + 0.25 x its ply, at most 12, times what it
#gets without (x5.8 out of a UHO book: ~9.9 s at 60+0.6 for a share of 1.7 s)
@pytest.mark.parametrize("arguments", [CLOCK_60, "wtime 180000 btime 180000 winc 2000 binc 2000".split(), "wtime 60000 btime 60000 movestogo 40".split()])
@pytest.mark.parametrize("ply, bonus", [(0, 2.8), (1, 3.05), (8, 4.8), (16, 5.8), (40, 5.8)])
def test_the_games_first_search_gets_lc0s_bonus(arguments, ply, bonus):
    board = board_at_ply(ply)
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(bonus * Clock().limits(arguments, board, 800)[1])


#the bonus is the first search's own, out of the time left: what it leaves unused is banked as any search's is, and the
#next move gets that and no bonus, as from a clock without one that had banked as much. A search off the clock
#(movetime) leaves the bonus to the first on it, as lc0's does, and a falling score's extension counts in shares without it
def test_the_first_search_bonus_is_given_once_and_banked():
    clock, board = Clock(**LC0_FIRST_MOVE), board_at_ply(16)
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.98, 0)
    _, first = clock.limits(CLOCK_60, board, 800)
    plain = Clock()
    _, plain_first = plain.limits(CLOCK_60, board, 800)
    assert first == pytest.approx(5.8 * plain_first) and clock.saved == 0
    assert clock.extension == (0, pytest.approx(MOST_EXTENSION * plain.share))

    bank = plain_first / 2
    clock.searched(first - bank, 0)
    plain.searched(plain_first - bank, 0)
    assert clock.saved == pytest.approx(bank) == plain.saved
    arguments = f"wtime {round((60 - first + bank) * 1000)} btime 60000 winc 600 binc 600".split()
    _, second = clock.limits(arguments, board_at_ply(18), 800)
    assert second == pytest.approx(plain.limits(arguments, board_at_ply(18), 800)[1]) == plain.share + bank
    assert clock.saved == pytest.approx(plain.saved) and clock.extension == pytest.approx(plain.extension)


#however large the bonus, the first search never gets the reserve, nor more than the time left, nor past its cap
def test_the_first_search_bonus_keeps_to_the_reserve_and_the_time_left():
    board, arguments = board_at_ply(16), "wtime 3000 btime 60000 winc 1000 binc 1000".split()
    assert Clock().limits(arguments, board, 800)[1] < 1.1
    assert Clock(**LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS)
    assert Clock(reserve_seconds=0.0, **LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx(3 - MOVE_OVERHEAD_SECONDS)
    assert Clock(most_of_the_time_left=0.25, **LC0_FIRST_MOVE).limits(arguments, board, 800)[1] == pytest.approx((3 - MOVE_OVERHEAD_SECONDS) / 4)


#a search on the clock whose score has fallen FALL_FULL_CP below the last search's runs on past its deadline by all of
#its extension's seconds, once; one whose score held stops at its deadline
def test_a_falling_score_runs_the_search_on():
    search = EngineSearch(cpu_batches(UniformModel()), 16)
    timed = {}
    for last_score in (0, engine_search.FALL_FULL_CP):
        started = time.perf_counter()
        search.search(chess.Board(), seconds=0.1, extension=(last_score, 0.3))
        timed[last_score] = time.perf_counter() - started
    assert timed[0] < 0.1 + 0.1
    assert 0.1 + 0.25 < timed[engine_search.FALL_FULL_CP] < 0.1 + 0.3 + 0.15


#smart pruning stops once the simulations left could not change the most visited move: a mate in one is decided when
#it has about half the budget, and at the stop its lead is more than the simulations left; the start with a flat
#eval is never decided
def test_a_decided_search_stops_early():
    mate = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    (best, most), (_, second) = mate.root_moves[:2]
    assert best.uci() == "e1e8" and mate.simulations < 2200 and most - second > 4000 - mate.simulations

    start = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board(), simulations=400)
    assert start.simulations == 400


#uci.py --no-smart-pruning, for matches where both engines must spend all of their time: the decided mate in one
#still gets the whole budget
def test_a_search_without_smart_pruning_uses_its_whole_budget():
    mate = EngineSearch(cpu_batches(UniformModel()), 16, smart_pruning=False).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    assert mate.line[0].uci() == "e1e8" and mate.simulations == 4000


#with a number of simulations, the pruned search plays the move of the whole search (both from a new tree and cache)
def test_pruning_never_changes_the_move(carlsen_games):
    positions = []
    for game in carlsen_games:
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % 9 == 6 and not board.is_game_over():
                positions.append(chess.Board(board.fen()))

    stopped_early = 0
    for board in positions:
        pruned = EngineSearch(cpu_batches(BoardModel()), 16).search(board, simulations=300)
        whole_search = EngineSearch(cpu_batches(BoardModel()), 16)
        whole_search.smart_pruning = False
        whole = whole_search.search(board, simulations=300)
        assert pruned.line[0] == whole.line[0] and whole.simulations == 300, board.fen()
        stopped_early += pruned.simulations < 300
    assert len(positions) > 20 and stopped_early > 5


#Searches as smart pruning sees them, scripted: after every round the root has `visits`, its two most visited moves
#`most` and `second`, and is proven or not; the search is done after 30 rounds unless stopped first, when it notes the
#time
class ScriptedSearches:
    def __init__(self, visits: int, most: int, second: int, proven: bool = False):
        self.root, self.proven = (visits, most, second), proven
        self.finishes, self.stopped_at = 0, None

    def start(self, game, fen, moves, simulations):
        pass

    def descend(self, count, collisions, simulations):
        return bytes(TOKENS_PER_BOARD * count)

    def finish(self, answers):
        self.finishes += 1
        return [self.report()] if self.finishes == 30 and self.stopped_at is None else []

    def stop(self, game):
        self.stopped_at = time.perf_counter()
        return self.report()

    def root_visits(self, game):
        return self.root

    def decided(self, game):
        return self.proven

    def report(self):
        return engine_search.GAME, ["a1a2"], 0.5, [], self.root[0]


#lc0's factor 1.33: a search of 1,000 simulations with 600 done stops once the most visited move leads by more than
#400 / 1.33 = 300.75, where 1 waits for a lead of more than 400
@pytest.mark.parametrize("factor, lead, stops", [(1.0, 400, False), (1.0, 401, True), (1.33, 300, False), (1.33, 301, True)])
def test_smart_prunings_factor(factor, lead, stops):
    search = EngineSearch(ImmediateBatches(answers_of_0_6), 16, grow_leaves=0, smart_pruning_factor=factor)
    search.searches = ScriptedSearches(600, 500, 500 - lead)
    search.search(chess.Board(), simulations=1000)
    assert (search.searches.stopped_at is not None) == stops


#a network whose every round takes 1/16 s of the test's clock
class SixteenthsNetwork:
    def __init__(self, now: list[float]):
        self.now = now

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return boards

    def collect(self, boards: np.ndarray) -> np.ndarray:
        self.now[0] += 1 / 16
        return answers_of_0_6(boards)


#smart pruning stops a decided search no sooner than smart_pruning_wait after its first answers, at the first round
#since (lc0's 0.2 s; 0.25 is a round exactly); one legal move and a proven root stop at the first answers all the same
@pytest.mark.parametrize("fen, proven, wait, stopped_at", [
    (chess.STARTING_FEN, False, 0.0, 1 / 16), (chess.STARTING_FEN, False, 0.2, 5 / 16), (chess.STARTING_FEN, False, 0.25, 5 / 16),
    (chess.STARTING_FEN, True, 0.2, 1 / 16), ("k7/8/8/8/8/8/r7/K1q5 w - - 0 1", False, 0.2, 1 / 16)])
def test_smart_pruning_waits_after_the_first_answers(monkeypatch, fen, proven, wait, stopped_at):
    now = [0.0]
    monkeypatch.setattr(engine_search.time, "perf_counter", lambda: now[0])
    search = EngineSearch(SixteenthsNetwork(now), 16, grow_leaves=0, smart_pruning_wait=wait, settings={"certainty": True})
    search.searches = ScriptedSearches(600, 600, 0, proven)
    search.search(chess.Board(fen), simulations=1000)
    assert search.searches.stopped_at == stopped_at


#with certainty, a search whose proofs leave one root move not lost stops at once (lc0's), where smart pruning waits for
#the move's lead: every white move here but Rxf7 lets black mate at once
def test_a_search_with_one_move_left_stops():
    board = chess.Board("8/4Rq2/7K/5P2/5Prk/8/8/8 w - - 0 1")
    searched = {stop: EngineSearch(cpu_batches(UniformModel()), 16, settings={"certainty": True}, stop_at_one_move_left=stop).search(board, simulations=20_000)
                for stop in (False, True)}
    assert searched[True].line[0].uci() == searched[False].line[0].uci() == "e7f7"
    assert searched[True].simulations < 1000 and searched[False].simulations > 5000


#uci.py's --search-settings reach the search: the same search of the same network finds another line and score with
#c_puct 0.1 than with 10
def test_the_engine_takes_search_settings(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    found = []
    for changes in ("c_puct=0.1", "c_puct=10"):
        engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts",
                                                      "--search-settings", changes], cwd=REPOSITORY)
        try:
            played = engine.play(chess.Board(), chess.engine.Limit(nodes=400), info=chess.engine.INFO_ALL)
        finally:
            engine.quit()
        found.append((played.info["pv"], played.info["score"], played.info["nodes"]))
    assert found[0][2] == found[1][2] == 400 and found[0][:2] != found[1][:2]


#uci.py's --precision reaches the network the search runs, and is refused where no such network runs
def test_the_engine_takes_a_precision(monkeypatch, tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    built = []
    monkeypatch.setattr(uci, "engine_network", lambda checkpoint, leaves, precision: built.append(precision) or cpu_batches(BoardModel()))
    search = uci.load_player(tmp_path / "model.pt", "mcts", True, 4, False, {"smart_pruning": False}, "fp16")
    refused = subprocess.run([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts", "--precision", "fp16"],
                             cwd=REPOSITORY, capture_output=True, text=True, input="")

    assert built == ["fp16"] and search.search(chess.Board(), simulations=20).simulations == 20
    assert refused.returncode == 2 and "--precision needs --select mcts --gpu" in refused.stderr


#uci.py started without --precision builds the engine's network in int8 (self-play and labels stay fp16: they never
#build it)
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
def test_the_engine_plays_in_int8_unless_told_otherwise(monkeypatch):
    built = []
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision: built.append(precision) or cpu_batches(BoardModel()))
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("isready\nquit\n"))
    runpy.run_module("engine.uci", run_name="__main__")

    assert built == ["int8"]


#uci.py's lc0 time options reach the clock and the search: the first move bonus on each game's first move, smart
#pruning's factor, wait and one-move-left stop; a factor of 0 or less is refused
@pytest.mark.filterwarnings("ignore:'engine.uci' found in sys.modules")
def test_the_engine_takes_lc0s_time_options(monkeypatch):
    given = []
    def search(self, board, nodes, seconds, extension, stop, info=None):
        given.append((seconds, self.smart_pruning_factor, self.smart_pruning_wait, self.stop_at_one_move_left))
        return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    monkeypatch.setattr(engine_search, "engine_network", lambda checkpoint, leaves, precision: cpu_batches(BoardModel()))
    monkeypatch.setattr(EngineSearch, "search", search)
    go = "go " + " ".join(CLOCK_60)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["isready", "position startpos", go, "position startpos moves e2e4 e7e5", go,
                                                               "ucinewgame", "position startpos", go, "quit"])))
    options = ["--first-move-bonus", "1.8", "--book-ply-bonus", "0.25", "--smart-pruning-factor", "1.33", "--smart-pruning-wait", "0.2",
               "--stop-at-one-move-left"]
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", *options])
    runpy.run_module("engine.uci", run_name="__main__")

    first = pytest.approx(2.8 * Clock().limits(CLOCK_60, chess.Board(), 800)[1])
    assert given[0] == (first, 1.33, 0.2, True) and given[1][0] != first and given[2] == given[0]
    monkeypatch.setattr(sys, "argv", ["uci.py", "unused.pt", "--select", "mcts", "--gpu", "--smart-pruning-factor", "0"])
    with pytest.raises(SystemExit) as refused:
        runpy.run_module("engine.uci", run_name="__main__")
    assert refused.value.code == 2


#the engine as a gui drives it: a move within go movetime, go nodes' simulations (in a new game: a kept tree's visits
#count, as lc0 counts them), and on a clock more time for move 40 than for the first move
def test_the_engine_keeps_to_gos_limits(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "engine.uci", str(tmp_path / "model.pt"), "--select", "mcts"],
                                                 cwd=REPOSITORY)
    try:
        board = chess.Board()
        engine.play(board, chess.engine.Limit(nodes=10))

        started = time.perf_counter()
        timed = engine.play(board, chess.engine.Limit(time=0.5), info=chess.engine.INFO_BASIC)
        seconds = time.perf_counter() - started
        other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        counted = engine.play(other, chess.engine.Limit(nodes=50), info=chess.engine.INFO_BASIC)
        #on a 10+0.1 clock the first move gets ~0.27 s and a 9 man endgame ~1 s: the time is planned over the moves the
        #material has left. A game of their own (ucinewgame), so the last search's score is the first move's
        clock = chess.engine.Limit(white_clock=10, black_clock=10, white_inc=0.1, black_inc=0.1)
        started = time.perf_counter()
        engine.play(board, clock, game="clock")
        first_move = time.perf_counter() - started
        started = time.perf_counter()
        engine.play(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 40"), clock, game="clock")
        endgame_move = time.perf_counter() - started
    finally:
        engine.quit()

    assert timed.move in board.legal_moves and timed.info["nodes"] > 10 and 0.3 < seconds < 0.5 + 0.1
    assert counted.info["nodes"] == 50
    assert first_move < 0.5 and 0.8 < endgame_move < 1.3


#Searches' search of board to its simulations, one leaf a round; its report (id, line, score, root moves, visits)
def searched(searches: selfplay_rs.Searches, board: chess.Board, simulations: int, evaluate, search_id: int = 0):
    searches.start(search_id, board.root().fen(), [move.uci() for move in board.move_stack], simulations)
    reports = []
    while not reports:
        reports = searches.finish(evaluate(leaf_boards(searches.descend())))
    return reports[0]

#with tree reuse every game (search id) keeps its own tree: a game's next search carries on its tree although another
#game searched in between, and needs only the simulations it lacks
def test_every_game_keeps_its_tree():
    network = CountingNetwork(BoardModel())
    searches = selfplay_rs.Searches(**rust_search_settings(False), tree_reuse=True)
    boards = [chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"), chess.Board()]
    for search_id, board in enumerate(boards):
        searched(searches, board, 400, network.launch, search_id)

    for search_id, board in enumerate(boards):
        network.boards = 0
        assert searched(searches, board, 600, network.launch, search_id)[4] == 600
        assert 0 < network.boards <= 200


#certainty propagation proves a mate in two (the defender's one reply is mated) and the engine stops once it is
#proven: the mate's first move, scored as a certain win, long before its budget; without it the win is only likely.
#The tree kept for the next move keeps the proof of the mate in one
def test_certainty_proves_a_mate_and_the_search_stops():
    board = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, settings={"certainty": True})
    proven = search.search(board, simulations=20_000)
    assert proven.line[0].uci() == "c6b6" and proven.score == 1.0 and proven.simulations < 5_000

    unproven = EngineSearch(cpu_batches(UniformModel()), 16).search(board, simulations=proven.simulations)
    assert unproven.score < 1.0

    board.push_uci("c6b6")
    board.push_uci("a8b8")
    network.boards = 0
    mate = search.search(board, simulations=20_000)
    assert mate.line[0].uci() == "h1h8" and mate.score == 1.0 and network.boards <= 2 * 16


#the search goes no further than a proven node: with the whole budget searched (no smart pruning), the simulations
#after the proof take the proven score at once and the network sees few boards
def test_certainty_stops_descents_at_proven_nodes():
    network = CountingNetwork(UniformModel())
    search = EngineSearch(network, 16, smart_pruning=False, settings={"certainty": True})
    report = search.search(chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1"), simulations=6000)

    assert report.simulations == 6000 and report.score == 1.0
    assert network.boards < 3000


#the uniform model, but for the moves it likes (uci, side), e^logit times more likely than any other
class LikesMoves(nn.Module):
    def __init__(self, moves: list[tuple[str, chess.Color]], logit: float):
        super().__init__()
        self.move_ids = [encode_move_to_id(chess.Move.from_uci(move), side) for move, side in moves]
        self.logit = logit

    def forward(self, boards):
        logits, values = UniformModel()(boards)
        logits[:, self.move_ids] = self.logit
        return logits, values


#a proven win is played although another move has more visits (lc0 ranks proven wins first): the network likes a king
#move, the search finds the rook's mate
def test_certainty_plays_a_proven_win_over_the_most_visited_move():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    model = LikesMoves([("g1f1", chess.WHITE)], 5.0)
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() == "e1e8" and report.score == 1.0
    assert report.root_moves[0][0].uci() == "g1f1"


#the network all but insists on Ra2, after which Rd1 mates: the plain search spends its visits there and plays it, while
#with certainty the proven loss is passed over and its visits go to the other moves, one of which is played
def test_certainty_avoids_a_proven_loss_the_network_likes():
    board = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    model = LikesMoves([("a1a2", chess.WHITE)], 8.0)
    assert EngineSearch(cpu_batches(model), 16).search(board, simulations=4000).line[0].uci() == "a1a2"
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() != "a1a2" and dict(report.root_moves)[report.line[0]] > 100
    board.push(report.line[0])
    for reply in list(board.legal_moves):
        board.push(reply)
        assert not board.is_checkmate()
        board.pop()


#--search-settings takes a setting's type: true or false for certainty, a number otherwise
def test_search_settings_take_true_and_false():
    settings = {"certainty": True, "c_puct": 0.8}
    assert parse_changes("certainty=false,c_puct=1.5", settings) == {"certainty": False, "c_puct": 1.5}
    with pytest.raises(ValueError):
        parse_changes("certainty=0", settings)


#a network whose answers come back late once, as on a cuda graph's first capture or a gpu shared with another engine,
#on a round's given collect; it counts the boards it is asked for
class StallsOnce:
    def __init__(self, network, stall_seconds: float, on_collect: int):
        self.network, self.stall_seconds, self.on_collect = network, stall_seconds, on_collect
        self.collects = self.boards = 0

    def launch(self, boards: np.ndarray):
        self.boards += len(boards)
        return self.network.launch(boards)

    def collect(self, launched):
        self.collects += 1
        if self.collects == self.on_collect:
            time.sleep(self.stall_seconds)
        return self.network.collect(launched)


#one slow round at the end of a search does not cut the next searches on a clock short: a running mean of the round
#times carried it on, and the next searches stopped after one round each, whose time never updated the mean
#(2026-10-12: streaks of 3-5 moves at 6-41 nodes in 10 of 174 games). With growing rounds a cut search is one leaf
@pytest.mark.parametrize("grow_leaves", [0, 16])
def test_a_stalled_round_does_not_cut_the_next_searches_short(grow_leaves):
    network = StallsOnce(cpu_batches(BoardModel()), 1.5, on_collect=2)
    search = EngineSearch(network, 16, smart_pruning=False, grow_leaves=grow_leaves)
    search.search(chess.Board(), seconds=0.05)
    for fen in ["4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"]:
        network.boards = 0
        search.search(chess.Board(fen), seconds=0.3)
        assert network.boards > 5 * 16


#ucinewgame (new_game) drops the kept tree and the cache: the same position searched again is searched from nothing,
#as a fresh engine searches it, where without it the kept tree already has the simulations
def test_a_new_game_starts_a_fresh_tree():
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    network = CountingNetwork(BoardModel())
    search = EngineSearch(network, 16, smart_pruning=False)
    search.search(board, simulations=300)
    network.boards = 0
    assert search.search(board, simulations=300).simulations == 300 and network.boards == 0

    search.new_game()
    again = search.search(board, simulations=300)
    fresh = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False).search(board, simulations=300)
    assert network.boards > 100
    assert (again.line, again.root_moves, again.score) == (fresh.line, fresh.root_moves, fresh.score)


#the knights out and back, which the network all but insists on: the search scores the start position reached again
#inside it as a draw, most of its visits there. After Nf3 that repeat is of a position before the root, which the game
#needs a third time for: its draws come out of the kept tree, which keeps only the few visits elsewhere (a search of
#one simulation reports the kept tree's)
KNIGHTS_OUT_AND_BACK = [("g1f3", chess.WHITE), ("g8f6", chess.BLACK), ("f3g1", chess.WHITE), ("f6g8", chess.BLACK)]

def test_stale_draws_come_out_of_a_kept_tree():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    first = search.search(board, simulations=400)
    assert first.root_moves[0] == (chess.Move.from_uci("g1f3"), 399)
    board.push_uci("g1f3")
    assert search.search(board, simulations=1).simulations < 50


#once all four moves are played the root is the repeated start position, which the last search never expanded: a
#search of fewer simulations than it had there still searches its moves (it reported an empty line, a rust panic)
def test_a_root_the_last_search_never_expanded_is_searched():
    search = EngineSearch(cpu_batches(LikesMoves(KNIGHTS_OUT_AND_BACK, 8.0)), 16, smart_pruning=False, grow_leaves=0)
    board = chess.Board()
    search.search(board, simulations=100)
    for move, _ in KNIGHTS_OUT_AND_BACK:
        board.push_uci(move)
    report = search.search(board, simulations=50)
    assert report.line and report.line[0] in board.legal_moves and report.simulations >= 50


#a gui that leaves a draw to be claimed (fide's rules, lichess) asks for a move in a threefold repetition or at the fifty
#move limit: the engine searches it and plays a move (it raised that the game was over, which ended the engine)
@pytest.mark.parametrize("fen, moves", [(chess.STARTING_FEN, [move for move, _ in KNIGHTS_OUT_AND_BACK] * 2),
                                         ("8/8/4k3/8/2R5/4K3/8/8 w - - 100 120", [])])
@pytest.mark.parametrize("leaves", [1, 16])
def test_the_engine_plays_on_in_a_draw_left_to_be_claimed(fen, moves, leaves):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    assert board.is_repetition(3) or board.is_fifty_moves()

    report = EngineSearch(cpu_batches(BoardModel()), leaves).search(board, simulations=200)
    assert report.line[0] in board.legal_moves and report.simulations > 1


#an engine's game of `moves` searches from board, each answered by the reply its search expects (or the given replies),
#with uci.py's advance after each move or not, or with the kept tree dropped instead (an advance to another game; the
#cache keeps its answers): the reports, the replies, the network's boards and the advances that had a tree to carry on
def advancing_game(model, board: chess.Board, moves: int, way: str, replies: list[str] | None = None):
    network = CountingNetwork(model)
    search = EngineSearch(network, 16, smart_pruning=False)
    reports, played, advanced = [], [], 0
    for ply in range(moves):
        report = search.search(board, simulations=600)
        reports.append((report.line, report.score, report.simulations, report.root_moves))
        board.push(report.line[0])
        if way == "advance":
            advanced += search.advance(board)
        elif way == "drop":
            search.advance(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"))
        reply = chess.Move.from_uci(replies[ply]) if replies else report.line[1] if len(report.line) > 1 else next(iter(board.legal_moves))
        board.push(reply)
        played.append(reply.uci())
    return reports, played, network.boards, advanced, search

#uci.py's advance once its move is sent carries the kept tree on to the game after the move, while the opponent thinks,
#and the next search plays on from it exactly as from the tree the last search left: the same reports move for move, in
#a middlegame that reuses its trees (as dropping them shows) and in a game whose knights shuffle so that draws go stale.
#A position that does not follow the move is searched afresh
def test_advancing_the_tree_changes_no_search(carlsen_games):
    middlegame = carlsen_games[0].board()
    for move in list(carlsen_games[0].mainline_moves())[:16]:
        middlegame.push(move)
    reports, replies, boards, _, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "plain")
    advanced_reports, _, advanced_boards, advanced, _ = advancing_game(BoardModel(), middlegame.copy(), 8, "advance", replies)
    dropped_reports = advancing_game(BoardModel(), middlegame.copy(), 8, "drop", replies)[0]
    assert advanced_reports == reports != dropped_reports and advanced == 8 and advanced_boards == boards

    shuffles = ["g8f6", "f6g8", "a7a6", "g8f6", "f6g8", "a6a5", "b8c6", "c6b8", "h7h6", "h6h5"]
    model = LikesMoves(KNIGHTS_OUT_AND_BACK + [("b1c3", chess.WHITE), ("c3b1", chess.WHITE)], 4.0)
    reports = advancing_game(model, chess.Board(), len(shuffles), "plain", shuffles)[0]
    advanced_reports, _, _, advanced, search = advancing_game(model, chess.Board(), len(shuffles), "advance", shuffles)
    assert advanced_reports == reports and advanced == len(shuffles)

    board = chess.Board()
    report = search.search(board, simulations=50)
    played, other = board.copy(), board.copy()
    played.push(report.line[0])
    search.advance(played)
    other.push(next(move for move in board.legal_moves if move != report.line[0]))
    other.push(next(iter(other.legal_moves)))
    assert search.search(other, simulations=50).line[0] in other.legal_moves


#a search whose tree reaches Searches' memory_limit_mb ends there with its report, as at its simulations (the engine's
#rounds too, which would otherwise wait on a search that gathers nothing); one that stays below its limit is the search
#without a limit
def test_a_search_ends_at_its_memory_limit():
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    reports = {limit: searched(selfplay_rs.Searches(**rust_search_settings(False), memory_limit_mb=limit), chess.Board(), 3000, evaluate)
               for limit in (0, 1, 1000)}
    assert reports[1000] == reports[0] and reports[0][4] == 3000
    assert 100 < reports[1][4] < 3000

    search = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False)
    search.searches = selfplay_rs.Searches(**engine_search_settings(False), tree_reuse=True, memory_limit_mb=1)
    report = search.search(chess.Board(), simulations=100_000)
    assert 100 < report.simulations < 3000 and report.line[0] in chess.Board().legal_moves


#a gui driving uci.main on a thread of its own: the lines told are the engine's stdin, and the lines it sends are kept
#with when it sent them. load_player makes its player with make_player (unless None): on the engine's thread, as
#Searches must be used on the thread that made it; main_arguments are main's
class Gui:
    def __init__(self, monkeypatch, make_player, checkpoint: Path = Path("unused.pt"), **main_arguments):
        self.lines: queue.Queue = queue.Queue()
        self.sent: list[tuple[float, str]] = []
        self.errors: list[BaseException] = []
        monkeypatch.setattr(sys, "stdin", iter(self.lines.get, None))
        monkeypatch.setattr(uci, "send", lambda line: self.sent.append((time.perf_counter(), line)))
        if make_player is not None:
            monkeypatch.setattr(uci, "load_player", lambda *arguments: make_player())
        self.engine = threading.Thread(target=self.run, args=(checkpoint,), kwargs={"selection": "mcts", **main_arguments}, daemon=True)
        self.engine.start()

    def run(self, checkpoint: Path, **main_arguments) -> None:
        try:
            uci.main(checkpoint, **main_arguments)
        except BaseException as error:
            self.errors.append(error)

    #when the line was told
    def tell(self, line: str) -> float:
        told = time.perf_counter()
        self.lines.put(line + "\n")
        return told

    #when the engine sent its first line starting with `start` since `since`, waiting up to `seconds` for one; None if none
    def sent_at(self, start: str, since: float = 0.0, seconds: float = 0.0) -> float | None:
        give_up = time.perf_counter() + seconds
        while True:
            found = [at for at, line in list(self.sent) if at >= since and line.startswith(start)]
            if found or time.perf_counter() > give_up:
                return found[0] if found else None
            time.sleep(0.005)

    def lines_sent(self, start: str) -> list[str]:
        return [line for _, line in list(self.sent) if line.startswith(start)]

    #a number of the last info line, e.g. its nodes
    def last_info(self, name: str) -> int:
        words = self.lines_sent("info depth")[-1].split()
        return int(words[words.index(name) + 1])

    #the input ends after `last`; the engine must then be done within `seconds`
    def close(self, last: str | None = "quit", seconds: float = 5.0) -> None:
        if last is not None:
            self.tell(last)
        self.lines.put(None)
        self.engine.join(seconds)
        assert not self.engine.is_alive() and not self.errors, self.errors

def board_search() -> EngineSearch:
    return EngineSearch(cpu_batches(BoardModel()), 16)

#stop ends a running search as its deadline would (10 s here), and the engine answers with its move at once (stop was
#read only once the search had ended)
def test_stop_ends_a_running_search(monkeypatch):
    gui = Gui(monkeypatch, board_search)
    gui.tell("position startpos")
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    stopped = gui.tell("stop")
    answered = gui.sent_at("bestmove", seconds=3)
    assert answered is not None and stopped < answered < stopped + 1 and gui.last_info("nodes") > 16
    gui.close()

#isready during a search is answered at once, the search going on (UCI); one after a command the engine has not taken
#yet waits for it, so that readyok still means all before it is done
def test_isready_is_answered_while_the_engine_searches(monkeypatch):
    gui = Gui(monkeypatch, board_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    asked = gui.tell("isready")
    ready = gui.sent_at("readyok", seconds=1)
    assert ready is not None and ready < asked + 0.5 and gui.sent_at("bestmove") is None

    gui.tell("position startpos moves e2e4")
    asked = gui.tell("isready")
    time.sleep(0.3)
    assert gui.sent_at("readyok", since=asked) is None
    gui.tell("stop")
    answered, ready = gui.sent_at("bestmove", seconds=3), gui.sent_at("readyok", since=asked, seconds=3)
    assert answered is not None and ready is not None and answered < ready
    gui.close()

#quit during a search ends it, and the engine
def test_quit_ends_a_running_search(monkeypatch):
    gui = Gui(monkeypatch, board_search)
    gui.tell("go movetime 10000")
    time.sleep(0.3)
    gui.tell("quit")
    gui.engine.join(2)
    assert not gui.engine.is_alive() and not gui.errors
    gui.close(last=None)

#with --infinite-until-stop go infinite searches past --simulations until stop, and answers only then, also once its
#search has ended by itself (an only move); the end of the input ends one too
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "7k/8/8/8/8/8/8/K5R1 b - - 0 1"])
def test_go_infinite_searches_until_stop(monkeypatch, fen):
    gui = Gui(monkeypatch, board_search, simulations=50, infinite_until_stop=True)
    gui.tell(f"position fen {fen}")
    gui.tell("go infinite")
    time.sleep(0.5)
    assert gui.sent_at("bestmove") is None
    stopped = gui.tell("stop")
    assert gui.sent_at("bestmove", since=stopped, seconds=2) is not None
    assert gui.last_info("nodes") > 50 or chess.Board(fen).legal_moves.count() == 1

    gui.tell("go infinite")
    time.sleep(0.2)
    gui.close(last=None)
    assert len(gui.lines_sent("bestmove")) == 2

#without it go infinite runs --simulations, as cutechess-cli's tc=inf games need
def test_go_infinite_runs_the_simulations_by_default(monkeypatch):
    gui = Gui(monkeypatch, lambda: EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False), simulations=300)
    gui.tell("go infinite")
    assert gui.sent_at("bestmove", seconds=10) is not None and gui.last_info("nodes") == 300
    gui.close()

#go ponder searches the opponent's time until ponderhit, then searches as the go asks with its time counted from the
#ponderhit; at stop it answers at once, a move the gui throws away, and the clock never hears of it (it answered at
#its move's time, before ponderhit)
def test_go_ponder_searches_until_ponderhit(monkeypatch):
    limits = []
    class Recorded(EngineSearch):
        def search(self, board, simulations=None, seconds=None, extension=None, stop=None, info=None):
            limits.append((simulations, seconds))
            return super().search(board, simulations, seconds, extension, stop, info)
    gui = Gui(monkeypatch, lambda: Recorded(cpu_batches(BoardModel()), 16))
    board = chess.Board()
    board.push_uci("e2e4")
    gui.tell("position startpos moves e2e4")
    go = "go ponder wtime 10000 btime 10000"
    gui.tell(go)
    time.sleep(0.5)
    assert gui.sent_at("bestmove") is None
    hit = gui.tell("ponderhit")
    answered = gui.sent_at("bestmove", seconds=3)
    assert answered is not None and gui.last_info("time") < (answered - hit) * 1000 + 50
    assert limits == [(None, None), (None, pytest.approx(Clock().limits(go.split()[1:], board, 200)[1]))]

    gui.tell(go)
    time.sleep(0.3)
    stopped = gui.tell("stop")
    assert gui.sent_at("bestmove", since=stopped, seconds=2) is not None and len(limits) == 3
    gui.close()

#the options a gui sets reach the search when it is built (isready), MoveOverhead the next go's time. Set once the
#search is built, or to a value it could not take, they stay as they were and the gui is told why
def test_the_options_reach_the_search(monkeypatch):
    built, given = [], []
    class Player:
        def search(self, board, nodes, seconds, extension, stop, info=None):
            given.append(seconds)
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    script = ["uci", "setoption name cachepositions value 1000", "setoption name RamLimitMb value 64", "setoption name SyzygyProbeLimit value 6",
              "setoption name MoveOverhead value 500", "setoption name MoveOverhead value -1", "setoption name SyzygyProbeLimit value 7",
              "setoption name Hash value 16", "isready", "setoption name CachePositions value 5", "go movetime 1000",
              "setoption name MoveOverhead value 100", "go movetime 1000", "ucinewgame", "go movetime 1000", "quit"]
    sent = []
    monkeypatch.setattr(uci, "load_player", lambda *arguments: built.append(arguments) or Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(script)))
    uci.main(Path("unused.pt"), selection="mcts", search_options={"smart_pruning": False})

    assert [line.split()[2] for line in sent if line.startswith("option name")] == ["SyzygyPath", "SyzygyProbeLimit", "MoveOverhead", "CachePositions",
                                                                                     "RamLimitMb"]
    assert len(built) == 1 and built[0][4] is False
    assert built[0][5] == {"smart_pruning": False, "tablebase_pieces": 6, "syzygy_path": None, "cache_positions": 1000, "memory_limit_mb": 64}
    assert given == [pytest.approx(0.5), pytest.approx(0.9), pytest.approx(0.9)]
    told = [line for line in sent if line.startswith("info string")]
    assert len(told) == 4 and "-1" in told[0] and "7" in told[1] and "Hash" in told[2] and "CachePositions" in told[3]

#EngineSearch makes its Searches with the cache size and memory limit it is given, and a search ends at the limit
def test_the_search_takes_its_cache_size_and_memory_limit(monkeypatch):
    made = []
    searches = selfplay_rs.Searches
    monkeypatch.setattr(selfplay_rs, "Searches", lambda **settings: made.append(settings) or searches(**settings))
    EngineSearch(cpu_batches(BoardModel()), 16)
    limited = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False, cache_positions=1000, memory_limit_mb=1)
    assert [(settings["cache_positions"], settings["memory_limit_mb"]) for settings in made] == [(engine_search.CACHE_POSITIONS, 0), (1000, 1)]
    assert 100 < limited.search(chess.Board(), simulations=100_000).simulations < 3000

#a SyzygyPath the gui sets reaches the search: the root takes the tables' move (no simulations) from its folders. A
#folder without the tables is refused, and the engine plays on as it was
def test_a_syzygy_path_reaches_the_search(monkeypatch, tablebase, tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    (tmp_path / "no_tables").mkdir()
    gui = Gui(monkeypatch, None, tmp_path / "model.pt")
    gui.tell(f"setoption name SyzygyPath value {tmp_path / 'no_tables'}")
    assert gui.sent_at("info string", seconds=5) is not None and gui.lines_sent("info string")[0].endswith("the tables stay as they were")
    gui.tell(f"setoption name SyzygyPath value {SYZYGY_DIR}")
    gui.tell("position fen k7/8/1K6/8/8/8/8/7R w - - 0 1")
    gui.tell("go nodes 50")
    assert gui.sent_at("bestmove", seconds=30) is not None
    assert gui.lines_sent("bestmove") == ["bestmove h1h8"] and gui.last_info("nodes") == 0 and len(gui.lines_sent("info string")) == 1
    gui.close()

#a position the search could not start from leaves the last one, and the gui is told why (each of these ended the
#engine, the last two at go); in a game that is over go answers with the null move
def test_a_bad_position_leaves_the_last_one(monkeypatch):
    searched = []
    class Player:
        def search(self, board, nodes, seconds, extension, stop=None, info=None):
            searched.append(board.fen())
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    bad = ["startpos moves e2e4 e7e4", "startpos moves e2e4 e7", "fen not a fen", "fen 8/8/8/8/8/8/8/8 w - - 0 1", "", "kiwipete",
           "startpos moves e2e4 0000", "fen 4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1"]
    script = ["position startpos moves e2e4", *(f"position {position}" for position in bad), "go nodes 10",
              "position fen 4R1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", "go nodes 10", "quit"]
    sent = []
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", sent.append)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(script)))
    uci.main(Path("unused.pt"), selection="mcts")

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    assert searched == [after_e4.fen()]
    assert len([line for line in sent if line.startswith("info string") and line.endswith("the last position stays")]) == len(bad)
    assert [line for line in sent if line.startswith("bestmove")][1] == "bestmove 0000"

#the fens parse_position refuses are those the rust search refuses (python-chess's status, and shakmaty's promoted
#material, which that leaves out); a fen whose castling rights or en passant square python-chess cleans both take
@pytest.mark.parametrize("fen", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1",
    "4k3/8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/K3K3 w - - 0 1", "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1", "4k3/8/5N2/1B6/8/8/8/4R1K1 b - - 0 1", "4k3/8/8/8/1NNNNNNN/8/PPPPPPPP/RNB1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/1PPPPPPP/QQ2K3 w - - 0 1", "4k3/8/8/8/8/8/PPPPPPPP/B1B1K3 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/BB2K3 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
])
def test_the_positions_refused_are_the_searchs(fen):
    try:
        uci.parse_position(["fen", *fen.split()])
        refused = False
    except ValueError:
        refused = True
    try:
        selfplay_rs.Searches(**rust_search_settings(False)).start(0, chess.Board(fen).root().fen(), [], 10)
        searched = True
    except ValueError:
        searched = False
    assert refused != searched


#a search tells its report as it stands about every INFO_SECONDS while it runs, its nodes growing, and the gui gets each
#as an info line before bestmove, as lc0 sends one a second
def test_a_long_search_tells_the_gui_how_it_stands(monkeypatch):
    monkeypatch.setattr(engine_search, "INFO_SECONDS", 0.05)
    told = []
    report = EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False).search(chess.Board(), seconds=0.5, info=told.append)
    nodes = [running.simulations for running in told]
    assert len(told) >= 3 and nodes == sorted(nodes) and nodes[-1] <= report.simulations
    assert all(running.line and running.line[0] in chess.Board().legal_moves for running in told)

    gui = Gui(monkeypatch, lambda: EngineSearch(cpu_batches(BoardModel()), 16, smart_pruning=False))
    gui.tell("position startpos")
    went = gui.tell("go movetime 500")
    assert gui.sent_at("bestmove", since=went, seconds=5) is not None and len(gui.lines_sent("info depth")) >= 4
    gui.close()
