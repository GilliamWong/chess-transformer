#the uci engine's search (tools/play/engine_search.py): rounds of many leaves (selfplay_rs.Searches.descend), its
#deadline and node limit, smart pruning, the tree kept for the next move, the engine's own settings (policy
#temperature, certainty propagation, contempt), and uci.py's go limits and clock
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest
import torch
from torch import nn

import selfplay_rs
from encoding import TOKENS_PER_BOARD, encode_move_to_id
from fake_models import BoardModel, UniformModel
from inference import cpu_batches
from model import ChessTransformer
from selfplay import rust_search_settings
from tools.checkpoint import save_checkpoint
import tools.play.engine_search as engine_search
from tools.play.engine_search import CONTEMPT_FROM_SCORE, MCTS_EXTRAS, EngineSearch, LeaderChanges, TimeUse, engine_search_settings
from tools.play.uci import MOST_OF_THE_TIME_LEFT, MOVE_OVERHEAD_SECONDS, RESERVE_SECONDS, Clock

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
#results were played, whatever the engine's own settings: a second round out on the network would change it. From a
#middlegame, where captures move the material count, so that an answer changes the next choice
def test_one_leaf_a_round_is_the_exact_search_at_every_move(monkeypatch, carlsen_games):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "policy_temperature": 1.3})
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
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {"c_puct": 0.5, "certainty": True})
    assert engine_search_settings(False) == {**rust_search_settings(False), **MCTS_EXTRAS, "certainty": True, "c_puct": 0.5}
    assert engine_search_settings(False, {"c_puct": 0.7})["c_puct"] == 0.7
    assert engine_search_settings(False, own=False) == {**rust_search_settings(False), **MCTS_EXTRAS}
    assert not MCTS_EXTRAS["certainty"] and MCTS_EXTRAS["cpuct_stdev_scale"] == MCTS_EXTRAS["lcb_stdevs"] == 0.0


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
    white, black = chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert clock.limits("movetime 1000".split(), white, 800) == (None, pytest.approx(1 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("movetime 500 nodes 100".split(), white, 800) == (100, pytest.approx(0.5 - MOVE_OVERHEAD_SECONDS))
    assert clock.limits("nodes 5000".split(), black, 800) == (5000, None)
    assert clock.limits(["infinite"], black, 800) == (800, None) and clock.most is None

    nodes, seconds = clock.limits("wtime 60000 btime 30000 winc 1000 binc 500".split(), black, 800)
    assert nodes is None and seconds == pytest.approx(clock.seconds_for_move(30 - MOVE_OVERHEAD_SECONDS, 0.5, None, 1))
    assert 0.5 < seconds < 30 * MOST_OF_THE_TIME_LEFT
    #the most a search on the clock may take, when it runs past its time (engine_search.TimeUse)
    assert clock.most == pytest.approx((30 - MOVE_OVERHEAD_SECONDS) * MOST_OF_THE_TIME_LEFT)


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


#uci.py's --search-settings and --contempt reach the search: with certainty a mate in two (Kb6 or Kc7) is proven, and
#scored as a sure win
def test_the_engine_takes_search_settings(tmp_path):
    settings = {"d_model": 32, "num_layers": 1, "n_heads": 2, "mlp_hidden": 64, "value_bins": 16}
    save_checkpoint(ChessTransformer(**settings), settings, tmp_path / "model.pt")
    engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "tools.play.uci", str(tmp_path / "model.pt"), "--select", "mcts",
                                                  "--search-settings", "certainty=1,policy_temperature=1.2", "--contempt", "0.02"], cwd=REPOSITORY)
    try:
        played = engine.play(chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1"), chess.engine.Limit(nodes=20000), info=chess.engine.INFO_SCORE)
    finally:
        engine.quit()
    assert played.move.uci() in ("c6b6", "c6c7") and played.info["score"].relative.score() >= 2400


#the engine as a gui drives it: a move within go movetime, go nodes' simulations (in a new game: a kept tree's visits
#count, as lc0 counts them), and on a clock more time for move 40 than for the first move
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
def searched(searches: selfplay_rs.Searches, board: chess.Board, simulations: int, evaluate, search_id: int = 0, contempt: float = 0.0):
    searches.start(search_id, board.root().fen(), [move.uci() for move in board.move_stack], simulations, contempt)
    reports = []
    while not reports:
        reports = searches.finish(evaluate(leaf_boards(searches.descend())))
    return reports[0]

#BoardModel with sharper priors, its move logits multiplied by `scale` and divided by `divisor`
class SharpBoardModel(nn.Module):
    def __init__(self, scale: float, divisor: float = 1.0):
        super().__init__()
        self.model, self.scale, self.divisor = BoardModel(), scale, divisor

    def forward(self, boards):
        logits, value = self.model(boards)
        return logits * self.scale / self.divisor, value

#a policy temperature divides the logits before the softmax (lc0's policy softmax temperature): the search with it is
#the search of a network whose logits are divided, to the last bit, and not the search without it
def test_the_policy_temperature_divides_the_logits(carlsen_games):
    board = carlsen_games[2].board()
    found = {}
    for ply, move in enumerate(list(carlsen_games[2].mainline_moves())[:40]):
        board.push(move)
        if ply % 8 == 7:
            for name, temperature, model in (("tempered", 1.5, SharpBoardModel(30)), ("divided", 1.0, SharpBoardModel(30, 1.5)),
                                             ("untempered", 1.0, SharpBoardModel(30))):
                searches = selfplay_rs.Searches(**rust_search_settings(False), policy_temperature=temperature)
                found.setdefault(name, []).append(searched(searches, board, 200, cpu_batches(model).evaluate_boards))

    assert found["tempered"] == found["divided"]
    assert found["tempered"] != found["untempered"]


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
        logits, value = UniformModel()(boards)
        logits[:, self.move_ids] = self.logit
        return logits, value

#a proven win is played although another move has more visits (lc0 ranks proven wins first): the network likes a king
#move, the search finds the rook's mate
def test_certainty_plays_a_proven_win_over_the_most_visited_move():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    model = LikesMoves([("g1f1", chess.WHITE)], 5.0)
    report = EngineSearch(cpu_batches(model), 16, settings={"certainty": True}).search(board, simulations=4000)

    assert report.line[0].uci() == "e1e8" and report.score == 1.0
    assert report.root_moves[0][0].uci() == "g1f1"

#the position after these moves has white to move and Nh1 draws by a third repetition; two plies before, white to move
#could head there (Ng3 Ke8 Nh1)
REPETITION_START = "4k3/7p/8/8/8/8/P7/4K2N b - - 0 1"
REPETITION_MOVES = "e8d7 h1g3 d7e8 g3h1 e8d7 h1g3 d7e8".split()

#a tree is kept only for a search with its contempt: a draw proven with contempt below 0 (worth 0.7 to white) counts
#at the next search's contempt of 0.2 (worth 0.3), and white no longer plays it although the network likes it
@pytest.mark.parametrize("contempt, draws", [(-0.2, True), (0.2, False)])
def test_a_kept_trees_draws_count_at_the_next_searchs_contempt(contempt, draws):
    model = LikesMoves([("h1g3", chess.WHITE), ("g3h1", chess.WHITE), ("e8d7", chess.BLACK), ("d7e8", chess.BLACK)], 3.0)
    evaluate = cpu_batches(model).evaluate_boards
    searches = selfplay_rs.Searches(**rust_search_settings(False), tree_reuse=True, certainty=True)
    board = chess.Board(REPETITION_START)
    for move in REPETITION_MOVES[:-2]:
        board.push_uci(move)
    searched(searches, board, 500, evaluate, contempt=-0.2)

    for move in REPETITION_MOVES[-2:]:
        board.push_uci(move)
    _, line, score, _, _ = searched(searches, board, 3000, evaluate, contempt=contempt)
    assert (line[0] == "g3h1") == draws


#white can draw by a third repetition (Nh1). A draw is even for the uniform model, so without contempt the search
#visits the repetition like any move; contempt makes it worse for white, the root's side, and fewer visits go there;
#contempt below 0 (a draw worth more to white) sends the search to it
def test_contempt_steers_the_root_side_away_from_draws():
    board = chess.Board(REPETITION_START)
    for move in REPETITION_MOVES:
        board.push_uci(move)
    evaluate = cpu_batches(UniformModel()).evaluate_boards
    visits = {}
    for contempt in (0.0, 0.1, -0.1):
        _, line, _, root_moves, _ = searched(selfplay_rs.Searches(**rust_search_settings(False)), board, 600, evaluate, contempt=contempt)
        visits[contempt] = dict(root_moves)["g3h1"]
        if contempt < 0:
            assert line[0] == "g3h1"
    assert visits[0.1] < visits[0.0] / 2 < visits[-0.1] / 2

    with pytest.raises(ValueError):
        selfplay_rs.Searches(**rust_search_settings(False)).start(0, board.root().fen(), [], 10, 0.5)


#white is a rook ahead 90 halfmoves after the last pawn move. The material count scores every move the same, but with
#a fifty move horizon a pawn move keeps the whole score and the others are drawn towards a draw: the search moves a
#pawn. The same search again, all of it from the cache, finds the same
@pytest.mark.parametrize("horizon, pawn_moves", [(100.0, True), (0.0, False)])
def test_the_fifty_move_horizon_makes_the_side_ahead_move_a_pawn(horizon, pawn_moves):
    board = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 90 80")
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    searches = selfplay_rs.Searches(**rust_search_settings(False), fifty_move_horizon=horizon, cache_positions=10_000)
    report = searched(searches, board, 400, evaluate)
    _, _, _, root_moves, visits = report

    pawn_visits = sum(move_visits for move, move_visits in root_moves if move in ("a2a3", "a2a4"))
    assert (pawn_visits > visits / 2) == pawn_moves
    assert searched(searches, board, 400, evaluate) == report

#the engine's contempt applies once its last search of the same game found it ahead
def test_contempt_starts_once_the_engine_is_ahead():
    search = EngineSearch(cpu_batches(UniformModel()), 16, contempt=0.05)
    board = chess.Board()
    board.push_uci("e2e4")
    assert search.contempt_for(board, ["e2e4"]) == 0.0
    search.last_search = (chess.STARTING_FEN, ["e2e4"], CONTEMPT_FROM_SCORE)
    assert search.contempt_for(board, ["e2e4", "e7e5", "g1f3"]) == 0.05
    search.last_search = (chess.STARTING_FEN, ["d2d4"], CONTEMPT_FROM_SCORE)
    assert search.contempt_for(board, ["e2e4", "e7e5", "g1f3"]) == 0.0
    search.last_search = (chess.STARTING_FEN, ["e2e4"], CONTEMPT_FROM_SCORE - 0.01)
    assert search.contempt_for(board, ["e2e4", "e7e5", "g1f3"]) == 0.0


#positions of the replayed games, every few plies
def game_positions(carlsen_games, every: int) -> list[chess.Board]:
    positions = []
    for game in carlsen_games:
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % every == every - 1 and not board.is_game_over():
                positions.append(chess.Board(board.fen()))
    return positions

#the uniform model, but a position's score is a hash of its board, spread over 0 to 1
class NoisyScores(nn.Module):
    def forward(self, boards):
        logits, _ = UniformModel()(boards)
        hashed = (boards.long() * torch.arange(1, 71)).sum(dim=1, keepdim=True) % 97
        return logits, (hashed.float() - 48) / 8

#katago's c_puct scaled by the spread of a node's scores: where the scores never vary the scale shrinks exploration,
#and the moves visited first keep the visits (first play urgency holds back the others); where they vary widely it
#grows it, and more of the root's moves are tried
@pytest.mark.parametrize("model, explores_more", [(UniformModel(), False), (NoisyScores(), True)])
def test_c_puct_scaled_by_the_scores_spread(model, explores_more):
    evaluate = cpu_batches(model).evaluate_boards
    tried = {}
    for scale in (0.0, 0.9):
        searches = selfplay_rs.Searches(**rust_search_settings(False), cpuct_stdev_scale=scale)
        _, _, _, root_moves, _ = searched(searches, chess.Board(), 60, evaluate)
        tried[scale] = sum(visits > 0 for _, visits in root_moves)
    assert (tried[0.9] > tried[0.0]) == explores_more and tried[0.9] != tried[0.0]


#the move played by its lower confidence bound: with a bound of almost no standard errors it is the best scoring of
#the moves visited at least a tenth as often as the most visited, never worse scoring than the most visited, and in
#some positions another move
def test_the_lower_confidence_bound_picks_the_move(carlsen_games):
    evaluate = cpu_batches(BoardModel()).evaluate_boards
    different = 0
    for board in game_positions(carlsen_games, 11):
        reports = []
        for lcb_stdevs in (0.0, 1e-9):
            searches = selfplay_rs.Searches(**rust_search_settings(False), lcb_stdevs=lcb_stdevs, lcb_least_share=0.1)
            reports.append(searched(searches, board, 200, evaluate))
        (_, most_visited_line, most_visited_score, root_moves, _), (_, lcb_line, lcb_score, _, _) = reports
        assert lcb_score >= most_visited_score and dict(root_moves)[lcb_line[0]] >= 0.1 * root_moves[0][1]
        different += lcb_line[0] != most_visited_line[0]
    assert different > 0


#TimeUse's budget: the planned time by default, more for changes of the best move and a falling score, less when
#neither happened, never past the most
def test_time_use_budget():
    assert TimeUse().budget(1.0, 3.0, changes=4, fall=0.1) == 1.0
    time_use = TimeUse(change_weight=0.5, fall_weight=10.0, steady=0.7)
    assert time_use.budget(1.0, 3.0, changes=0, fall=0.0) == pytest.approx(0.7)
    assert time_use.budget(1.0, 3.0, changes=2, fall=0.0) == pytest.approx(2.0)
    assert time_use.budget(1.0, 3.0, changes=0, fall=0.05) == pytest.approx(1.5)
    assert time_use.budget(1.0, 3.0, changes=2, fall=0.05) == 3.0

    changes = LeaderChanges()
    for leader, counts in (("e2e4", False), ("d2d4", False), ("e2e4", True), ("e2e4", True), ("g1f3", True)):
        changes.observe(leader, counts)
    assert changes.changes == 2

#a search on a clock with TimeUse: the uniform model's leader keeps changing, so the search runs to the most it may
#take; a mate in one found at once is steady, and stops at the steady share of its time
def test_a_search_uses_its_time_as_time_use_says():
    time_use = TimeUse(change_weight=1.0, steady=0.5)
    timed = {}
    for name, fen in (("changing", chess.STARTING_FEN), ("steady", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")):
        search = EngineSearch(cpu_batches(UniformModel()), 16, smart_pruning=False, time_use=time_use)
        started = time.perf_counter()
        search.search(chess.Board(fen), seconds=0.2, most_seconds=0.5)
        timed[name] = time.perf_counter() - started
    assert 0.4 < timed["changing"] < 0.6 and timed["steady"] < 0.15
