#the rust games (selfplay_rs) against the python reference: the same rules, and without random numbers the same games
import random

import chess
import numpy as np
import pytest
import torch
from torch import nn

from engine import mcts, selfplay
import selfplay_rs
from model.encoding import TOKENS_PER_BOARD, encode_board, encode_move_to_id
from fake_models import BoardModel, UniformModel
from engine.inference import cpu_batches
from reference_selfplay import final_result, play_game
from engine.selfplay import ANSWER_WIDTH, MAX_PLIES, RustSelfPlayBatch, play_games_in_rust, rust_search_settings

CPU = torch.device("cpu")

#castling, en passant (one a pin makes illegal), promotions, check, fifty moves, same-coloured bishops, moves already played
STARTS = [
    chess.STARTING_FEN,
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 9",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/3p4/8/K3P2r/8/8/8/7k b - - 0 1",
    "8/PPP4k/8/8/8/8/4Kppp/8 w - - 0 1",
    "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1",
    "8/8/4k3/8/8/4K3/8/R7 w - - 96 80",
    "8/2b5/4k3/8/8/2B1K3/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",
]

def start_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    if fen == chess.STARTING_FEN:
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)
    return board

def rust_rules(board: chess.Board) -> selfplay_rs.SelfPlayGames:
    return RustSelfPlayBatch(lambda boards: boards, 1, iter(()), 0, 0, board, 0, MAX_PLIES, True, False).games

#a random game that often takes its last move back, so that positions come round again
def playout(board: chess.Board, plies: int, random_numbers: random.Random) -> list[chess.Board]:
    boards = [board.copy()]
    for _ in range(plies):
        if final_result(board) is not None:
            break
        moves = list(board.legal_moves)
        back = [move for move in moves if len(board.move_stack) >= 2 and move.from_square == board.move_stack[-2].to_square
                and move.to_square == board.move_stack[-2].from_square]
        board.push(random_numbers.choice(back if back and random_numbers.random() < 0.6 else moves))
        boards.append(board.copy())
    return boards

#every playout position and each one move on: encoding, legal moves in python-chess's order, game_result, final_result
def test_the_rust_rules_are_the_pythons():
    random_numbers = random.Random(0)
    checked = {"positions": 0, "over in the search": 0, "over in the game": 0}
    for fen in STARTS:
        start = start_board(fen)
        rules = rust_rules(start)
        for _ in range(10):
            for board in playout(start.copy(), 80, random_numbers):
                played = [move.uci() for move in board.move_stack[len(start.move_stack):]]
                positions = [(board, played)]
                for move in list(board.legal_moves)[:10]:
                    after = board.copy()
                    after.push(move)
                    positions.append((after, played + [move.uci()]))
                for position, moves in positions:
                    encoding, move_ids, in_search, over = rules.inspect(moves)
                    assert list(encoding) == encode_board(position), position.fen()
                    assert move_ids == [encode_move_to_id(move, position.turn) for move in position.legal_moves], position.fen()
                    assert in_search == mcts.game_result(position), position.fen()
                    assert over == final_result(position), position.fen()
                    checked["positions"] += 1
                    checked["over in the search"] += in_search is not None
                    checked["over in the game"] += over is not None

    assert checked["positions"] > 10000 and checked["over in the search"] > 200 and checked["over in the game"] > 400, checked


#mcts.evaluate_many with rust's softmax, whose last bits differ from numpy's
def evaluate_many_in_rust_numbers(model: nn.Module, boards: list[chess.Board], device: torch.device) -> list[tuple[dict[chess.Move, float], float]]:
    answers = cpu_batches(model).evaluate_boards(np.array([encode_board(board) for board in boards], dtype=np.uint8))
    evaluations = []
    for board, answer in zip(boards, answers):
        moves = list(board.legal_moves)
        priors = selfplay_rs.softmax(answer[[encode_move_to_id(move, board.turn) for move in moves]].tolist())
        evaluations.append((dict(zip(moves, priors)), float(answer[-1])))
    return evaluations

def assert_same_games(python_games, rust_games) -> None:
    by_number = lambda games: sorted(games, key=lambda game: game.number)
    for python, rust in zip(by_number(python_games), by_number(rust_games), strict=True):
        assert python.number == rust.number and python.white_score == rust.white_score
        assert np.array_equal(python.boards, rust.boards)
        assert np.array_equal(python.visit_moves, rust.visit_moves)
        assert np.array_equal(python.visit_shares, rust.visit_shares)
        assert np.array_equal(python.root_scores, rust.root_scores)
        assert np.array_equal(python.white_to_move, rust.white_to_move)

#game 0 of board in python and in rust, with nothing random, which must match to the last bit; returns the rust game
def play_and_compare(model: nn.Module, board: chess.Board, use_tablebase: bool = False, **options) -> selfplay.Game:
    python = play_game(model, CPU, 24, 0, board=board, sampled_plies=0, **options)
    rust, = play_games_in_rust(cpu_batches(model).evaluate_boards, 24, [0], 1, board=board, sampled_plies=0, use_tablebase=use_tablebase, **options)
    assert_same_games([python], [rust])
    return rust

#the cache and the rounds of several simulations it allows, both off
def disable_cache(monkeypatch) -> None:
    monkeypatch.setattr(selfplay, "CACHE_POSITIONS", 0)
    monkeypatch.setattr(selfplay, "SIMULATIONS_PER_LEAF", 1)

#without root noise and sampled plies nothing is random; the uniform model's ties go to the first child in python-chess's order
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("tree_reuse", [True, False])
@pytest.mark.parametrize("model", ["board", "uniform"])
def test_rust_games_are_the_python_games(monkeypatch, tree_reuse, model, cache):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)
    monkeypatch.setattr(mcts, "ROOT_NOISE_SHARE", 0.0)
    if not cache:
        disable_cache(monkeypatch)
    model = BoardModel() if model == "board" else UniformModel()

    results = [play_and_compare(model, start_board(fen), max_plies=160, tree_reuse=tree_reuse).white_score for fen in STARTS]

    if isinstance(model, BoardModel):
        assert any(result != 0.5 for result in results)


def test_rust_games_end_where_the_tablebase_knows_the_result(tablebase, monkeypatch):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)
    monkeypatch.setattr(mcts, "ROOT_NOISE_SHARE", 0.0)
    monkeypatch.setattr(mcts, "tablebase", tablebase)
    model = BoardModel()

    #6 and 7 pieces, then 4: a game the tables know before it begins has no positions
    fens = ("4k3/2p5/8/3P4/8/8/2R2K2/6r1 w - - 0 1", "8/5pk1/6p1/8/8/5PK1/6P1/2R5 b - - 0 1", "8/8/8/2k5/8/2K5/2P5/5n2 w - - 0 1")
    games = [play_and_compare(model, chess.Board(fen), use_tablebase=True) for fen in fens]

    assert all(len(game.boards) < MAX_PLIES for game in games)
    assert len(games[-1].boards) == 0


#selfplay_rs.Searches are mcts.search, one position per network call so that no bit depends on the batch
@pytest.mark.parametrize("model", ["board", "uniform"])
def test_rust_searches_are_mcts_search(monkeypatch, carlsen_games, model):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)
    model = BoardModel() if model == "board" else UniformModel()
    positions = []
    for game in carlsen_games:
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            if ply % 7 == 4 and mcts.game_result(board) is None:
                positions.append(board.copy())

    searches = selfplay_rs.Searches(**rust_search_settings(False))
    for number, board in enumerate(positions):
        searches.start(number, board.root().fen(), [move.uci() for move in board.move_stack], 40)
    evaluate = cpu_batches(model).evaluate_boards
    found = {}
    while len(searches):
        leaves = np.frombuffer(searches.descend(), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        answers = np.array([evaluate(leaf[None])[0] for leaf in leaves], dtype=np.float32).reshape(-1, ANSWER_WIDTH)
        for number, line, score, root_moves, visits in searches.finish(answers):
            found[number] = (line, score, root_moves, visits)

    assert len(positions) > 20 and sorted(found) == list(range(len(positions)))
    for number, board in enumerate(positions):
        root = mcts.search(model, board, CPU, 40)
        line = mcts.principal_variation(root)
        chosen = root.children[line[0]]
        score = chosen.value_sum / chosen.visit_count if chosen.visit_count > 0 else 1 - root.value_sum / root.visit_count
        by_visits = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        assert found[number] == ([move.uci() for move in line], score, [(move.uci(), child.visit_count) for move, child in by_visits],
                                 root.visit_count), board.fen()

    with pytest.raises(ValueError):
        searches.start(0, "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30", [], 40) #mated: nothing to search


def test_the_rust_numbers_are_numpys():
    logits = np.random.default_rng(0).normal(0, 4, 40).astype(np.float32)
    weights = np.exp(logits - logits.max())
    assert np.allclose(selfplay_rs.softmax(logits.tolist()), weights / weights.sum(), rtol=1e-5, atol=1e-7)


def uniform(boards: np.ndarray) -> np.ndarray:
    return cpu_batches(UniformModel()).evaluate_boards(boards)

#`count` games played together, and how many boards the network was asked for
def play_counting(count: int, **options) -> tuple[list[selfplay.Game], int]:
    evaluated = 0
    def counting(boards: np.ndarray) -> np.ndarray:
        nonlocal evaluated
        evaluated += len(boards)
        return uniform(boards)
    return list(play_games_in_rust(counting, 64, range(count), count, **options)), evaluated

#black mates at once: black's win, and black's move stored mirrored
def test_a_rust_game_is_recorded_for_the_side_to_move():
    board = chess.Board("7r/8/8/8/8/1k6/8/K7 b - - 0 1")

    game, = play_games_in_rust(uniform, 800, [0], 1, board=board, sampled_plies=0)

    assert game.white_score == 0.0 and len(game.boards) == 1
    assert game.outcomes().tolist() == [1.0]
    assert game.root_scores[0] > 0.5
    assert game.visit_moves[0, 0] == encode_move_to_id(chess.Move.from_uci("h8h1"), chess.BLACK)
    assert game.visit_shares[0].sum() == pytest.approx(1.0)


#with noise and sampled plies, a game's random numbers depend on the seed and its number alone
def test_rust_games_played_together_are_the_games_played_alone():
    alone = [game for number in range(5) for game in play_games_in_rust(uniform, 8, [number], 1, max_plies=40)]

    together = list(play_games_in_rust(uniform, 8, range(5), 3, max_plies=40))

    assert_same_games(alone, together)
    assert not np.array_equal(alone[0].boards, alone[1].boards)
    other_seed, = play_games_in_rust(uniform, 8, [0], 1, seed=7, max_plies=40)
    assert not np.array_equal(alone[0].boards, other_seed.boards)


#with noise too the cache changes no game (the noise comes right after the root's expansion); without tree reuse it knows many roots
def test_the_cache_changes_no_game_with_noise(monkeypatch):
    board_model = cpu_batches(BoardModel()).evaluate_boards
    cached = list(play_games_in_rust(board_model, 32, range(3), 3, max_plies=30, tree_reuse=False))
    disable_cache(monkeypatch)
    uncached = list(play_games_in_rust(board_model, 32, range(3), 3, max_plies=30, tree_reuse=False))

    assert_same_games(uncached, cached)


#the cache answers the openings games share: 8 games' first plies take fewer network calls
def test_the_cache_saves_network_calls(monkeypatch):
    cached_games, cached = play_counting(8, max_plies=4)
    disable_cache(monkeypatch)
    uncached_games, uncached = play_counting(8, max_plies=4)

    assert len(cached_games) == len(uncached_games) == 8
    assert cached < 0.85 * uncached


def test_rust_tree_reuse_needs_fewer_evaluations_for_the_same_visits():
    evaluated = {}
    for tree_reuse in (True, False):
        games, evaluated[tree_reuse] = play_counting(1, max_plies=10, tree_reuse=tree_reuse)
        game, = games
        assert len(game.boards) == 10 and np.allclose(game.visit_shares.sum(axis=1), 1.0)

    assert evaluated[True] < evaluated[False]
