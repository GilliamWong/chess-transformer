#the uci engine's search (tools/play/engine_search.py): rounds of many leaves (selfplay_rs.Searches.descend), its
#deadline and node limit, smart pruning, the tree kept for the next move, and uci.py's go limits and clock
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

import selfplay_rs
from encoding import TOKENS_PER_BOARD
from fake_models import BoardModel, UniformModel
from inference import cpu_batches
from model import ChessTransformer
from selfplay import rust_search_settings
from tools.checkpoint import save_checkpoint
from tools.play.engine_search import EngineSearch
from tools.play.uci import MOST_OF_THE_TIME_LEFT, MOVE_OVERHEAD_SECONDS, Clock

REPOSITORY = Path(__file__).parent.parent


def leaf_boards(leaves: bytes) -> np.ndarray:
    return np.frombuffer(bytearray(leaves), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)

#the model on the cpu, counting the boards it evaluates
class CountingNetwork:
    def __init__(self, model):
        self.network = cpu_batches(model)
        self.boards = 0

    def launch(self, boards: np.ndarray) -> np.ndarray:
        self.boards += len(boards)
        return self.network.launch(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return self.network.collect(launched)


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
#results were played: a second round out on the network would change it. From a middlegame, where captures move the
#material count, so that an answer changes the next choice
def test_one_leaf_a_round_is_the_exact_search_at_every_move(carlsen_games):
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
    assert clock.limits("movetime 1000".split(), chess.WHITE, 800) == (None, pytest.approx(1 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("movetime 500 nodes 100".split(), chess.WHITE, 800) == (100, pytest.approx(0.5 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("nodes 5000".split(), chess.BLACK, 800) == (5000, None)
    assert clock.limits(["infinite"], chess.BLACK, 800) == (800, None)

    nodes, seconds = clock.limits("wtime 60000 btime 30000 winc 1000 binc 500".split(), chess.BLACK, 800)
    assert nodes is None and seconds == pytest.approx(clock.seconds_for_move(30 - MOVE_OVERHEAD_SECONDS, 0.5, None))
    assert 0.5 < seconds < 30 * MOST_OF_THE_TIME_LEFT


#a search stopped early (smart pruning) leaves time on the clock, and the next moves get it; a movetime search is not
#on the clock
def test_time_saved_goes_to_later_moves():
    clock = Clock()
    _, first = clock.limits("wtime 60000 btime 60000".split(), chess.WHITE, 800)
    clock.searched(first / 2)
    _, later = clock.limits("wtime 59000 btime 60000".split(), chess.WHITE, 800)
    _, without_saving = Clock().limits("wtime 59000 btime 60000".split(), chess.WHITE, 800)
    assert later > 1.05 * without_saving

    clock.searched(later / 4)
    _, after_more_saving = clock.limits("wtime 58000 btime 60000".split(), chess.WHITE, 800)
    assert after_more_saving > later
    clock.limits("movetime 1000".split(), chess.WHITE, 800)
    clock.searched(0.1)
    assert clock.limits("wtime 58000 btime 60000".split(), chess.WHITE, 800)[1] == pytest.approx(after_more_saving)
    for _ in range(50):
        clock.searched(0.0)
    assert clock.limits("wtime 3000 btime 60000".split(), chess.WHITE, 800)[1] <= 3 * MOST_OF_THE_TIME_LEFT


#smart pruning stops once the simulations left could not change the most visited move: a mate in one is decided when
#it has about half the budget, and at the stop its lead is more than the simulations left; the start with a flat
#eval is never decided
def test_a_decided_search_stops_early():
    mate = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"), simulations=4000)
    (best, most), (_, second) = mate.root_moves[:2]
    assert best.uci() == "e1e8" and mate.simulations < 2200 and most - second > 4000 - mate.simulations

    start = EngineSearch(cpu_batches(UniformModel()), 16).search(chess.Board(), simulations=400)
    assert start.simulations == 400


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


#the engine as a gui drives it: a move within go movetime, and go nodes' simulations (in a new game: a kept tree's
#visits count, as lc0 counts them)
def test_the_engine_keeps_to_gos_limits(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tools.play.uci", str(tmp_path / "model.pt"), "--select", "mcts"],
                                                 cwd=REPOSITORY)
    try:
        board = chess.Board()
        engine.play(board, chess.engine.Limit(nodes=10))

        started = time.perf_counter()
        timed = engine.play(board, chess.engine.Limit(time=0.5), info=chess.engine.INFO_BASIC)
        seconds = time.perf_counter() - started
        other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        counted = engine.play(other, chess.engine.Limit(nodes=50), info=chess.engine.INFO_BASIC)
    finally:
        engine.quit()

    assert timed.move in board.legal_moves and timed.info["nodes"] > 10 and 0.3 < seconds < 0.5 + 0.1
    assert counted.info["nodes"] == 50
