#the uci engine's search (engine/engine_search.py): rounds of many leaves (selfplay_rs.Searches.descend), its
#deadline and node limit, smart pruning, the tree kept for the next move, the engine's own settings, and uci.py's go
#limits and clock
import io
import runpy
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

import selfplay_rs
from model.encoding import TOKENS_PER_BOARD
from fake_models import BoardModel, CountingNetwork, UniformModel
from engine.inference import ImmediateBatches, cpu_batches
from model.model import ChessTransformer
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from tools.checkpoint import save_checkpoint
from engine import engine_search
from engine.engine_search import EngineSearch, engine_search_settings
from engine.player import SearchReport
from engine import uci
from engine.uci import MOST_OF_THE_TIME_LEFT, MOVE_OVERHEAD_SECONDS, RESERVE_SECONDS, Clock

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
    assert nodes is None and seconds == pytest.approx(clock.seconds_for_move(30 - MOVE_OVERHEAD_SECONDS, 0.5, None, 1))
    assert 0.5 < seconds < 30 * MOST_OF_THE_TIME_LEFT


#a search stopped early (smart pruning) leaves time on the clock, and the next moves get it; a movetime search is not
#on the clock
def test_time_saved_goes_to_later_moves():
    clock, board = Clock(), chess.Board()
    _, first = clock.limits("wtime 60000 btime 60000".split(), board, 800)
    clock.searched(first / 2)
    _, later = clock.limits("wtime 59000 btime 60000".split(), board, 800)
    _, without_saving = Clock().limits("wtime 59000 btime 60000".split(), board, 800)
    assert later > 1.05 * without_saving

    clock.searched(later / 4)
    _, after_more_saving = clock.limits("wtime 58000 btime 60000".split(), board, 800)
    assert after_more_saving > later
    clock.limits("movetime 1000".split(), board, 800)
    clock.searched(0.1)
    assert clock.limits("wtime 58000 btime 60000".split(), board, 800)[1] == pytest.approx(after_more_saving)
    for _ in range(50):
        clock.searched(0.0)
    assert clock.limits("wtime 3000 btime 60000".split(), board, 800)[1] <= 3 * MOST_OF_THE_TIME_LEFT


#at 60+0.6 the time is planned to last to about move 50, as lc0 spends its clock, and later moves live on the
#increment: searches that take all they are given keep the opening's pace through the middle game, have the clock low
#by move 60 (an even share over 30 more moves still had ~11 s there) and never run it below the reserve
def test_the_clock_is_spent_by_about_move_50_and_never_runs_out():
    clock, board, left, spent, low = Clock(), chess.Board(), 60.0, {}, 60.0
    for move in range(1, 121):
        board.fullmove_number = move
        _, seconds = clock.limits(f"wtime {round(left * 1000)} btime 60000 winc 600 binc 600".split(), board, 800)
        clock.searched(seconds)
        left += 0.6 - seconds - MOVE_OVERHEAD_SECONDS
        spent[move], low = seconds, min(low, left)
        if move == 60:
            left_at_60 = left
    opening, middle_game = sum(spent[m] for m in range(1, 11)) / 10, sum(spent[m] for m in range(20, 40)) / 20
    assert middle_game > 0.9 * opening
    assert left_at_60 < 5 and low > RESERVE_SECONDS / 2


#a clock planned to last to an earlier move spends more on the moves before it, and a move's cap is the part of the
#time left given
def test_a_clocks_plan_and_cap_can_be_set():
    board = chess.Board()
    board.fullmove_number = 10
    arguments = "wtime 60000 btime 60000 winc 600 binc 600".split()
    assert Clock(planned_moves=30).limits(arguments, board, 800)[1] > 1.5 * Clock().limits(arguments, board, 800)[1]
    assert Clock(most_of_the_time_left=0.01).limits(arguments, board, 800)[1] == pytest.approx((60 - MOVE_OVERHEAD_SECONDS) * 0.01)
    #late in the game the least moves left and the reserve decide the share
    board.fullmove_number = 70
    arguments = "wtime 10000 btime 10000 winc 600 binc 600".split()
    assert Clock(least_moves_left=8, reserve_seconds=0.0).limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS) / 8 + 0.6)
    assert Clock().limits(arguments, board, 800)[1] == pytest.approx((10 - MOVE_OVERHEAD_SECONDS - RESERVE_SECONDS) / 10 + 0.6)


#ucinewgame starts a fresh clock: searches that used little of their time make the clock give later moves more (the
#8th move here gets over twice the first's), but the next game's first move gets what the first game's did
def test_a_new_game_starts_a_fresh_clock(monkeypatch):
    given = []
    class Player:
        def search(self, board, nodes, seconds):
            given.append(seconds)
            return SearchReport(0.5, [next(iter(board.legal_moves))], 1, [])
    go = ["position startpos", "go wtime 60000 btime 60000 winc 600 binc 600"]
    monkeypatch.setattr(uci, "load_player", lambda *arguments: Player())
    monkeypatch.setattr(uci, "send", lambda line: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(["uci", "isready", *go * 8, "ucinewgame", *go, "quit"])))
    uci.main(Path("unused.pt"), selection="mcts")
    assert given[7] > 2 * given[0] and given[8] == given[0]


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
        #on a 10+0.1 clock the first move gets ~0.28 s and move 40 ~1 s: the time is planned to last to about move 50
        clock = chess.engine.Limit(white_clock=10, black_clock=10, white_inc=0.1, black_inc=0.1)
        started = time.perf_counter()
        engine.play(board, clock)
        first_move = time.perf_counter() - started
        started = time.perf_counter()
        engine.play(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 40"), clock)
        move_40 = time.perf_counter() - started
    finally:
        engine.quit()

    assert timed.move in board.legal_moves and timed.info["nodes"] > 10 and 0.3 < seconds < 0.5 + 0.1
    assert counted.info["nodes"] == 50
    assert first_move < 0.5 and 0.8 < move_40 < 1.3


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
