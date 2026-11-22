#the rust search with transpositions and move_means, one leaf a round, against reference_move_means.py's search of the same positions
#with the same network: after every simulation, not only at a search's end
from collections import Counter

import chess
import numpy as np
import pytest
import torch
from torch import nn

import selfplay_rs
import reference_move_means as reference
from engine import mcts
from engine.engine_search import EngineSearch
from engine.inference import cpu_batches
from engine.selfplay import ANSWER_WIDTH, rust_search_settings
from fake_models import PIECE_VALUES, BoardModel, UniformModel, value_logits
from model.encoding import NUM_MOVE_IDS, NUM_SQUARES, TOKENS_PER_BOARD, encode_board
from test_selfplay_rs import evaluate_many_in_rust_numbers

CPU = torch.device("cpu")
GAME = 0
NO_ANSWERS = np.zeros((0, ANSWER_WIDTH), dtype=np.float32)
SIMULATIONS = (50, 400, 3000)
#the two searches add up the same numbers in the same order, so their scores are equal to the bit; this much is allowed
SCORE_TOLERANCE = 1e-9
BLOCKED_PAWNS = "4k3/8/8/1p1p1p1p/1P1P1P1P/8/8/4K3 w - - 0 1"


#sharper move logits than BoardModel's and a score that changes from one position to the next: BoardModel's counts the material
#alone, a half wherever it is level, and a mean that never moves never drifts from what a move remembers of it
class PositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(1)
        self.move_weights = torch.randn(TOKENS_PER_BOARD, NUM_MOVE_IDS, generator=generator)
        self.value_weights = torch.randn(TOKENS_PER_BOARD, generator=generator)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = boards.float()
        material = PIECE_VALUES[boards[:, :NUM_SQUARES].long()].sum(dim=1)
        return tokens @ self.move_weights / 30, value_logits(0.3 * material + 1.5 * torch.sin(tokens @ self.value_weights))

#PositionModel for the rule's stops; UniformModel for ties, every one of which goes to the first move; BoardModel as the other tests' network
MODELS = {"position": PositionModel, "uniform": UniformModel, "board": BoardModel}


def played(fen: str, sans: str) -> chess.Board:
    board = chess.Board(fen)
    for san in sans.split():
        board.push_san(san)
    return board

#the last two have a game behind them: a position it had once is searched when a line comes to it again, one it had twice is a draw there
POSITIONS = {
    "start": chess.Board(),
    "middlegame": played(chess.STARTING_FEN, "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8"),
    "blocked pawns": chess.Board(BLOCKED_PAWNS),
    "rook ending near fifty moves": chess.Board("8/8/4k3/8/8/4K3/8/R7 w - - 92 80"),
    "pawn ending": chess.Board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
    "knights back home": played(chess.STARTING_FEN, "Nf3 Nf6 Ng1 Ng8 Nc3"),
    "blocked pawns after a shuffle": played(BLOCKED_PAWNS, "Kd1 Kd8 Ke1 Ke8 Kd2"),
}


#the reference's network is the rust search's to the bit: the same rows, and rust's softmax
@pytest.fixture(autouse=True)
def rust_numbers(monkeypatch):
    monkeypatch.setattr(mcts, "evaluate_many", evaluate_many_in_rust_numbers)

def move_means_searches(**options) -> selfplay_rs.Searches:
    return selfplay_rs.Searches(**rust_search_settings(False), transpositions=True, move_means=True, **options)

def moves_played(board: chess.Board) -> list[str]:
    return [move.uci() for move in board.move_stack]

#a search to its report, a round of one simulation at a time: after each, (the boards it asked the network about, the report as it stands)
def rust_simulations(searches: selfplay_rs.Searches, model: nn.Module, board: chess.Board, moves: list[str], simulations: int) -> list[tuple[list, tuple]]:
    evaluate = cpu_batches(model).evaluate_boards
    searches.start(GAME, board.root().fen(), moves, simulations)
    steps, reports = [], []
    while not reports:
        leaves = np.frombuffer(bytearray(searches.descend()), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        reports = searches.finish(evaluate(leaves) if len(leaves) else NO_ANSWERS)
        _, *report = reports[0] if reports else searches.report_now(GAME)
        steps.append((leaves.tolist(), tuple(report)))
    return steps

#Searches' report (lib.rs's Search::report) of the reference's node as a search's root: the expected line, the score of its first
#move's node, the moves by their own visits and the node's visits
def report_of(node: reference.Node) -> tuple[list[str], float, list[tuple[str, int]], int]:
    line = reference.principal_variation(node)
    chosen = reference.most_visited_link(node).child
    score = chosen.mean() if chosen is not None and chosen.visit_count > 0 else 1 - node.mean()
    by_visits = sorted(node.links, key=lambda link: link.visits, reverse=True)
    return [move.uci() for move in line], score, [(link.move.uci(), link.visits) for link in by_visits], node.visit_count

def assert_same_report(rust: tuple, expected: tuple, where: str) -> None:
    (rust_line, rust_score, rust_moves, rust_visits), (line, score, moves, visits) = rust, expected
    assert (rust_visits, rust_moves, rust_line) == (visits, moves, line), where
    assert abs(rust_score - score) <= SCORE_TOLERANCE, where


#every simulation asks the network about the same position or about none, and leaves the same report: every root move's visits, the
#line (which goes down through the most visited moves' nodes) and the score (a node's mean; the root's own only after the first
#simulation, the one time a report shows it)
@pytest.mark.parametrize("simulations", SIMULATIONS)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("model", MODELS)
def test_the_rust_search_is_the_reference_at_every_simulation(model, position, simulations):
    model, board = MODELS[model](), POSITIONS[position]
    steps = rust_simulations(move_means_searches(), model, board, moves_played(board), simulations)
    assert len(steps) == simulations

    search = reference.Search(model, board, CPU)
    for number, (rust_leaves, rust_report) in enumerate(steps, start=1):
        simulation = search.simulate()
        leaves = [] if simulation.evaluated is None else [encode_board(simulation.evaluated)]
        where = f"simulation {number}: in the reference {simulation.ended}, after {' '.join(move.uci() for move in simulation.line) or 'no move'}"
        assert rust_leaves == leaves, where
        assert_same_report(rust_report, report_of(search.root), where)


#the comparison above says little about a stop no search came to
def test_the_positions_come_to_every_way_a_simulation_ends():
    endings = {position: Counter() for position in POSITIONS}
    #a node that only repeats had ended at, until a line came to it that it does not repeat
    evaluated_after_repeats = 0
    for position, board in POSITIONS.items():
        search = reference.Search(PositionModel(), board, CPU)
        for _ in range(max(SIMULATIONS)):
            simulation = search.simulate()
            endings[position][simulation.ended] += 1
            evaluated_after_repeats += simulation.ended == reference.EVALUATED and simulation.visits_before > 0

    for ended in (reference.EVALUATED, reference.GAME_OVER, reference.REPEAT, reference.NEW_MOVE):
        assert sum(counts[ended] for counts in endings.values()) >= 100, (ended, endings)
    assert evaluated_after_repeats >= 10
    #the fifty move rule eight plies down, and the positions the game had twice before the search
    assert endings["rook ending near fifty moves"][reference.GAME_OVER] >= 100 and endings["blocked pawns after a shuffle"][reference.GAME_OVER] >= 100


#the nodes below the root: a tree kept for the next search (tree_reuse) and started again down a line, with no simulation left to
#make, reports the line's last node as its root. With transpositions nothing is taken out of a kept tree (search.rs's
#drop_stale_leaves), so the report is of the node as the first search left it: its visits from every path, its moves' own visits
@pytest.mark.parametrize("position", POSITIONS)
def test_the_nodes_below_the_root_are_the_references(position):
    model, board, simulations = PositionModel(), POSITIONS[position], max(SIMULATIONS)
    root = reference.search(model, board, CPU, simulations)

    by_visits = lambda node: sorted((link for link in node.links if link.child is not None and link.child.links), key=lambda link: -link.visits)
    lines = []
    for first in by_visits(root)[:3]:
        lines.append([first])
        lines.extend([first, reply] for reply in by_visits(first.child)[:2])
    assert len(lines) == 9

    for line in lines:
        searches = move_means_searches(tree_reuse=True)
        rust_simulations(searches, model, board, moves_played(board), simulations)
        line_moves = [link.move.uci() for link in line]
        steps = rust_simulations(searches, model, board, moves_played(board) + line_moves, 1)
        assert [leaves for leaves, _ in steps] == [[]], "the kept tree's root has its visits: the round has no simulation to make"
        assert_same_report(steps[0][1], report_of(line[-1].child), " ".join(line_moves))


#the engine's own way to a search of one leaf a round, which keeps the network's answers in a cache
@pytest.mark.parametrize("position", POSITIONS)
def test_the_engines_search_of_one_leaf_a_round_is_the_reference(position):
    model, board = PositionModel(), POSITIONS[position]
    report = EngineSearch(cpu_batches(model), 1, settings={"transpositions": True, "move_means": True}).search(board, simulations=400)
    line, score, root_moves, visits = report_of(reference.search(model, board, CPU, 400))
    found = ([move.uci() for move in report.line], report.score, [(move.uci(), move_visits) for move, move_visits in report.root_moves], report.simulations)
    assert_same_report(found, (line, score, root_moves[:len(report.root_moves)], visits), position)
