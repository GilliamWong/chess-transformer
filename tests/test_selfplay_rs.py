#the rust games (selfplay_rs) against the python reference: the same rules, and without random numbers the same games
import random

import chess
import numpy as np
import pytest
import torch
from torch import nn

import mcts
import selfplay_rs
from encoding import NUM_MOVE_IDS, encode_board, encode_move_to_id
from fake_models import PIECE_VALUES, UniformModel
from inference import cpu_batches
from reference_selfplay import final_result, play_game
from selfplay import ANSWER_WIDTH, MAX_PLIES, RustSelfPlayBatch, play_games_in_rust, rust_search_settings
from tools.tablebase import SYZYGY_DIR, TABLES_PER_FOLDER, open_tablebase

CPU = torch.device("cpu")

#castling both ways, en passant (and one a pin makes illegal), promotions, check, the fifty move rule, same-coloured
#bishops, and a start that already has moves played
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

def rust_rules(board: chess.Board, use_tablebase: bool = False) -> selfplay_rs.SelfPlayGames:
    return RustSelfPlayBatch(lambda boards: boards, 1, iter(()), 0, 0, board, 0, MAX_PLIES, True, use_tablebase).games

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

#every playout position and every position one move on: the encoding, the legal moves in python-chess's order (a
#node's children's order), the search's game_result and the game's final_result
def test_the_rust_rules_are_the_pythons():
    random_numbers = random.Random(0)
    checked = {"positions": 0, "over in the search": 0, "over in the game": 0}
    for fen in STARTS:
        start = start_board(fen)
        rules = rust_rules(start)
        for _ in range(10):
            for board in playout(start.copy(), 80, random_numbers):
                played = [move.uci() for move in board.move_stack[len(start.move_stack):]]
                for position, moves in [(board, played)] + [(board.copy(), played + [move.uci()]) for move in board.legal_moves][:10]:
                    if len(moves) > len(played):
                        position.push_uci(moves[-1])
                    encoding, move_ids, in_search, over = rules.inspect(moves)
                    assert list(encoding) == encode_board(position), position.fen()
                    assert move_ids == [encode_move_to_id(move, position.turn) for move in position.legal_moves], position.fen()
                    assert in_search == mcts.game_result(position), position.fen()
                    assert over == final_result(position), position.fen()
                    checked["positions"] += 1
                    checked["over in the search"] += in_search is not None
                    checked["over in the game"] += over is not None

    #the corners were reached, not only positions where the game goes on
    assert checked["positions"] > 10000 and checked["over in the search"] > 200 and checked["over in the game"] > 400, checked


#priors of the board's own, and a position scored by its material, so that some games are won
class BoardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.randn(70, NUM_MOVE_IDS, generator=torch.Generator().manual_seed(0))

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return boards.float() @ self.weights / 100, PIECE_VALUES[boards[:, :64].long()].sum(dim=1, keepdim=True)

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
    for python, rust in zip(python_games, rust_games, strict=True):
        assert python.number == rust.number and python.white_score == rust.white_score
        assert np.array_equal(python.boards, rust.boards)
        assert np.array_equal(python.visit_moves, rust.visit_moves)
        assert np.array_equal(python.visit_shares, rust.visit_shares)
        assert np.array_equal(python.root_scores, rust.root_scores)
        assert np.array_equal(python.white_to_move, rust.white_to_move)

#without root noise and sampled plies nothing is random: every number must match to the last bit. The uniform model's
#searches are all ties, which go to the first child in python-chess's order
@pytest.mark.parametrize("tree_reuse", [True, False])
@pytest.mark.parametrize("model", ["board", "uniform"])
def test_rust_games_are_the_python_games(monkeypatch, tree_reuse, model):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)
    monkeypatch.setattr(mcts, "ROOT_NOISE_SHARE", 0.0)
    model = BoardModel() if model == "board" else UniformModel()

    results = []
    for fen in STARTS:
        board = start_board(fen)
        python = play_game(model, CPU, 24, 0, board=board, sampled_plies=0, max_plies=160, tree_reuse=tree_reuse)
        rust = list(play_games_in_rust(cpu_batches(model).evaluate_boards, 24, [0], 1, board=board, sampled_plies=0, max_plies=160,
                                       tree_reuse=tree_reuse))
        assert_same_games([python], rust)
        results.append(rust[0].white_score)
    if isinstance(model, BoardModel):
        assert any(result != 0.5 for result in results) #some were won, not all drawn


@pytest.mark.skipif(len(list(SYZYGY_DIR.glob("*.rtb*"))) != 2 * TABLES_PER_FOLDER, reason="the syzygy tables are not downloaded")
def test_rust_games_end_where_the_tablebase_knows_the_result(monkeypatch):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)
    monkeypatch.setattr(mcts, "ROOT_NOISE_SHARE", 0.0)
    monkeypatch.setattr(mcts, "tablebase", open_tablebase())
    model = BoardModel()

    #6 and 7 pieces, then 4: a game the tables know before it begins has no positions
    for fen in ("4k3/2p5/8/3P4/8/8/2R2K2/6r1 w - - 0 1", "8/5pk1/6p1/8/8/5PK1/6P1/2R5 b - - 0 1", "8/8/8/2k5/8/2K5/2P5/5n2 w - - 0 1"):
        board = chess.Board(fen)
        python = play_game(model, CPU, 24, 0, board=board, sampled_plies=0)
        rust = list(play_games_in_rust(cpu_batches(model).evaluate_boards, 24, [0], 1, board=board, sampled_plies=0, use_tablebase=True))
        assert_same_games([python], rust)
        assert len(rust[0].boards) < MAX_PLIES
    assert len(rust[0].boards) == 0


#a match's searches (selfplay_rs.Searches) are mcts.search, with the game's moves for the repetitions. One position per
#network call, as mcts.search asks, so that no bit depends on the batch
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
        leaves = np.frombuffer(searches.descend(), dtype=np.uint8).reshape(-1, 70)
        answers = np.concatenate([evaluate(leaves[row:row + 1]) for row in range(len(leaves))]) if len(leaves) else np.zeros((0, ANSWER_WIDTH), np.float32)
        for number, line, score, root_moves in searches.finish(answers):
            found[number] = (line, score, root_moves)

    assert len(positions) > 20 and sorted(found) == list(range(len(positions)))
    for number, board in enumerate(positions):
        root = mcts.search(model, board, CPU, 40)
        line = mcts.principal_variation(root)
        chosen = root.children[line[0]]
        score = chosen.value_sum / chosen.visit_count if chosen.visit_count > 0 else 1 - root.value_sum / root.visit_count
        by_visits = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        assert found[number] == ([move.uci() for move in line], score, [(move.uci(), child.visit_count) for move, child in by_visits]), board.fen()

    with pytest.raises(ValueError):
        searches.start(0, "R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30", [], 40) #mated: nothing to search


def test_the_rust_numbers_are_numpys():
    logits = np.random.default_rng(0).normal(0, 4, 40).astype(np.float32)
    weights = np.exp(logits - logits.max())
    assert np.allclose(selfplay_rs.softmax(logits.tolist()), weights / weights.sum(), rtol=1e-5, atol=1e-7)


def uniform(boards: np.ndarray) -> np.ndarray:
    return cpu_batches(UniformModel()).evaluate_boards(boards)

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

    together = sorted(play_games_in_rust(uniform, 8, range(5), 3, max_plies=40), key=lambda game: game.number)

    assert_same_games(alone, together)
    assert not np.array_equal(alone[0].boards, alone[1].boards)
    other_seed, = play_games_in_rust(uniform, 8, [0], 1, seed=7, max_plies=40)
    assert not np.array_equal(alone[0].boards, other_seed.boards)


def test_rust_tree_reuse_needs_fewer_evaluations_for_the_same_visits():
    evaluated = {True: 0, False: 0}
    for tree_reuse in evaluated:
        def counting(boards: np.ndarray) -> np.ndarray:
            evaluated[tree_reuse] += len(boards)
            return uniform(boards)
        game, = play_games_in_rust(counting, 64, [0], 1, max_plies=10, tree_reuse=tree_reuse)
        assert len(game.boards) == 10 and np.allclose(game.visit_shares.sum(axis=1), 1.0)

    assert evaluated[True] < evaluated[False]
