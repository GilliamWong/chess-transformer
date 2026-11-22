#the rust search with transpositions against reference_move_means.py's search of the same positions with the same network: one leaf a
#round after every simulation, and rounds of several leaves and collisions after every round, not only at a search's end
import math
from collections import Counter, deque

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
from model.encoding import NUM_MOVE_IDS, NUM_SQUARES, TOKENS_PER_BOARD, encode_board, encode_move_to_id
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
    return selfplay_rs.Searches(**rust_search_settings(False), transpositions=True, **options)

def moves_played(board: chess.Board) -> list[str]:
    return [move.uci() for move in board.move_stack]

#a search to its report, a round of one simulation at a time: after each, (the boards it asked the network about, the report as it stands)
def rust_simulations(searches: selfplay_rs.Searches, model: nn.Module, board: chess.Board, moves: list[str], visits: int) -> list[tuple[list, tuple]]:
    evaluate = cpu_batches(model).evaluate_boards
    searches.start(GAME, board.root().fen(), moves, visits)
    steps, reports = [], []
    while not reports:
        leaves = np.frombuffer(bytearray(searches.descend()), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        reports = searches.finish(evaluate(leaves) if len(leaves) else NO_ANSWERS)
        _, *report, _ = reports[0] if reports else searches.report_now(GAME)
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
#simulation, the one time a report shows it). The rust search, asked for `visits`, made a simulation for as long as the reference's
#root was short of them and no longer
def assert_the_reference_at_every_simulation(steps: list[tuple[list, tuple]], search: reference.Search, visits: int) -> None:
    for number, (rust_leaves, rust_report) in enumerate(steps, start=1):
        assert search.root.visit_count < visits, f"the rust search made simulation {number} with the visits it was asked for"
        simulation = search.simulate()
        leaves = [] if simulation.evaluated is None else [encode_board(simulation.evaluated)]
        where = f"simulation {number}: in the reference {simulation.ended} after {' '.join(move.uci() for move in simulation.line) or 'no move'}"
        assert rust_leaves == leaves, where
        assert_same_report(rust_report, report_of(search.root), where)
    assert search.root.visit_count >= visits, f"the rust search stopped after {len(steps)} simulations, short of its visits"

@pytest.mark.parametrize("simulations", SIMULATIONS)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("model", MODELS)
def test_the_rust_search_is_the_reference_at_every_simulation(model, position, simulations):
    model, board = MODELS[model](), POSITIONS[position]
    steps = rust_simulations(move_means_searches(), model, board, moves_played(board), simulations)
    assert len(steps) == simulations
    assert_the_reference_at_every_simulation(steps, reference.Search(model, board, CPU), simulations)


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

#nor about the nodes a node's pass gives its moves, if no node of these searches had a move to give one: the moves that were given a
#node, and those of them a later simulation picked for their first visit. The kings behind blocked pawns come to a position by many
#orders of moves
def test_the_positions_give_moves_known_nodes_and_come_to_visit_them():
    moves_given, first_visits = Counter(), Counter()
    for position, board in POSITIONS.items():
        search = reference.Search(PositionModel(), board, CPU)
        while search.root.visit_count < max(SIMULATIONS):
            simulation = search.simulate()
            moves_given[position] += len(simulation.passes_of_moves_given)
            first_visits[position] += simulation.first_visit_given_at_pass is not None

    assert sum(moves_given.values()) >= 50 and sum(first_visits.values()) >= 50, (moves_given, first_visits)


#a node's later passes over its moves. These positions come to a position by many orders of moves, and so many visits bring nodes
#below the root to second and third passes that give moves nodes: a search without the second pass leaves the rust search before
#1600 visits in both searches and one without the third before 18500. A fourth pass and a fifth (8192 visits) are made at a few
#nodes and give nothing
TRANSPOSING = {"queen and pawn ending": chess.Board("8/8/8/4k3/8/2p5/8/3QK3 w - - 0 1"), "blocked pawns": chess.Board(BLOCKED_PAWNS)}
MANY_VISITS = {"queen and pawn ending": 40000, "blocked pawns": 20000}

@pytest.mark.parametrize("position", TRANSPOSING)
def test_the_rust_search_of_many_visits_is_the_reference_at_every_simulation(position):
    model, board, visits = PositionModel(), TRANSPOSING[position], MANY_VISITS[position]
    steps = rust_simulations(move_means_searches(), model, board, moves_played(board), visits)
    assert_the_reference_at_every_simulation(steps, reference.Search(model, board, CPU), visits)

#which says little about the later passes unless they gave moves nodes and those moves came to be picked: by the pass (0 a node's
#first), the moves given a node and those of them that had their first visit, in the reference's searches of the comparison above
def test_the_searches_of_many_visits_give_moves_nodes_at_later_passes():
    moves_given, first_visits, passes_made_below_the_root = Counter(), Counter(), Counter()
    for position, board in TRANSPOSING.items():
        search = reference.Search(PositionModel(), board, CPU)
        while search.root.visit_count < MANY_VISITS[position]:
            simulation = search.simulate()
            moves_given.update(simulation.passes_of_moves_given)
            if simulation.first_visit_given_at_pass is not None:
                first_visits[simulation.first_visit_given_at_pass] += 1
        passes_made_below_the_root.update(node.passes_made for node in search.nodes.values() if node is not search.root)

    at_later_passes = lambda by_pass: sum(count for number, count in by_pass.items() if number > 0)
    assert at_later_passes(moves_given) >= 30 and at_later_passes(first_visits) >= 10, (moves_given, first_visits)
    assert moves_given[2] >= 20, moves_given
    assert passes_made_below_the_root[3] >= 50 and passes_made_below_the_root[4] >= 10 and passes_made_below_the_root[5] >= 3, passes_made_below_the_root


#a network that answers as it was told to for the boards it was told of, and evenly (every move the same, half a point) for the rest
class ScriptedModel(nn.Module):
    #a move without a weight has next to none
    NO_WEIGHT = 1e-6

    def __init__(self):
        super().__init__()
        self.answers: dict[tuple[int, ...], torch.Tensor] = {}

    #the board after `line` (moves from `fen`): its moves' weights, and its score for the side to move
    def tell(self, fen: str, line: list[str], weights: dict[str, float], score: float = 0.5) -> None:
        board = played_moves(fen, line)
        row = torch.full((ANSWER_WIDTH,), math.log(self.NO_WEIGHT))
        for uci, weight in weights.items():
            row[encode_move_to_id(chess.Move.from_uci(uci), board.turn)] = math.log(weight)
        row[ANSWER_WIDTH - 1] = score
        told = self.answers.setdefault(tuple(encode_board(board)), row)
        assert torch.equal(told, row), f"the board after {line} was told another answer"

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        even = torch.zeros(ANSWER_WIDTH)
        even[ANSWER_WIDTH - 1] = 0.5
        return torch.stack([self.answers.get(tuple(board), even) for board in boards.tolist()])

def played_moves(fen: str, line: list[str]) -> chess.Board:
    board = chess.Board(fen)
    for uci in line:
        board.push_uci(uci)
    return board

#a node that comes late to a pass over its moves, its visits past those of the next pass too: only repeats bring a node visits
#without a descent that picks a move from it, and no search above has a node that took three hundred of them between two picks.
#Here one does. The kings behind a blocked pawn pair, every score a half but one, and a network that knows a board and not its
#fifty-move count:
#- Y, the position after Kd2 Kd7, has a node two plies down and (LATE) one six plies down. The network gives both the same moves:
#  Kd1 mostly, Ke3 the rest.
#- REPEATING, which most visits go down, comes back to Y: a repeat, which ends at LATE and gives it a visit without picking a move
#  from it. So does Kd2 Kd7 Ke3 Kd8 Kd2 Kd7.
#- ARRIVING comes to LATE without a repeat, a visit in eighty or so. Its sixth visit evaluates LATE, which has hundreds of visits by
#  then; its seventh is the first to pick a move from LATE, and makes LATE's first pass with later ones due as well; its eighth
#  picks from LATE again.
#- KNOWING comes to the position Ke3 leads to from Y, seven plies down as from LATE, and first visits it between ARRIVING's seventh
#  visit and its eighth. It scores nothing for White there.
#At ARRIVING's eighth visit LATE's move Ke3 leads to a known position. A LATE that owed another pass would give Ke3 that
#position's node, which scores nothing, and pick Kd1; a LATE whose late pass counted for every pass it had come to picks Ke3, as a
#move nothing is known of
LATE_PASS_FEN = "4k3/8/8/3p4/3P4/8/8/4K3 w - - 0 1"
REPEATING = "e1d2 e8d7 d2d1 d7d8 d1d2 d8d7".split()
ARRIVING = "e1d1 e8d7 d1c1 d7c7 c1d2 c7d7".split()
KNOWING = "e1f2 e8d8 f2f3 d8c7 f3f2 c7d7 f2e3".split()
LATE_PASS_VISITS = 1000

def late_pass_model() -> ScriptedModel:
    model = ScriptedModel()
    model.tell(LATE_PASS_FEN, [], {REPEATING[0]: 1.0, ARRIVING[0]: 0.012, KNOWING[0]: 0.0112})
    for line in (REPEATING, ARRIVING, KNOWING):
        for plies in range(1, len(line)):
            model.tell(LATE_PASS_FEN, line[:plies], {"d2d1": 0.66, "d2e3": 0.34} if line[:plies] == REPEATING[:2] else {line[plies]: 1.0})
    model.tell(LATE_PASS_FEN, ARRIVING, {"d2d1": 0.66, "d2e3": 0.34})
    #from the position Ke3 leads to, back to Y by Kd8 Kd2 Kd7
    model.tell(LATE_PASS_FEN, KNOWING, {"d7d8": 1.0}, score=1.0)
    model.tell(LATE_PASS_FEN, KNOWING + ["d7d8"], {"e3d2": 1.0})
    return model

def test_the_rust_search_of_a_node_late_to_its_pass_is_the_reference_at_every_simulation():
    model, board = late_pass_model(), chess.Board(LATE_PASS_FEN)
    steps = rust_simulations(move_means_searches(), model, board, [], LATE_PASS_VISITS)
    assert_the_reference_at_every_simulation(steps, reference.Search(model, board, CPU), LATE_PASS_VISITS)

#which holds the rust search to one pass for a late node only if the reference's search went as the network was written to make it go
def test_the_node_late_to_its_pass_makes_one_pass_and_then_picks_a_move_it_would_have_given_a_node():
    search = reference.Search(late_pass_model(), chess.Board(LATE_PASS_FEN), CPU)
    #LATE's node, made here before the search comes to it: one without a visit is as none to the search
    late = search.node_of(played_moves(LATE_PASS_FEN, ARRIVING))
    #the lines of the simulations that picked a move from LATE, and where among them the two things happened
    picks, late_pass_at, known_at = [], None, None
    while search.root.visit_count < LATE_PASS_VISITS:
        visits_before, passes_before = late.visit_count, late.passes_made
        simulation = search.simulate()
        line = [move.uci() for move in simulation.line]
        if line[:len(ARRIVING)] == ARRIVING and len(line) > len(ARRIVING):
            picks.append(line[len(ARRIVING):])
            if late.passes_made > passes_before:
                #its first pass, late: it counts for every pass LATE's visits had come to, two at least
                assert (passes_before, late_pass_at) == (0, None) and late.passes_made == reference.passes_due(visits_before) >= 2
                late_pass_at = len(picks)
            if len(picks) == 2:
                #a move that was not given its node, picked as a move nothing is known of: its position's node is found then
                assert (simulation.ended, simulation.first_visit_given_at_pass, simulation.passes_of_moves_given) == (reference.NEW_MOVE, None, [])
                assert reference.passes_due(visits_before) == late.passes_made, "another pass was due by LATE's visits"
        if line == KNOWING and simulation.ended == reference.EVALUATED:
            known_at = len(picks)

    assert (late_pass_at, known_at) == (1, 1), "the first pick made the late pass, and Ke3's position was first visited before the second"
    assert picks[:2] == [["d2d1"], ["d2e3"]]


#rounds of several leaves: simulations wait for the network while others descend, and a descent may end at a leaf that waits
#(a collision). What a round gathers, what may collide, the simulations it may make in all (those backed up at once among them),
#and the rounds left unanswered while the next is gathered
ROUNDS = [pytest.param((8, 8, 16, 0), id="8 leaves"), pytest.param((16, 32, 48, 1), id="16 leaves, a round waiting"),
          pytest.param((4, 0, None, 2), id="4 leaves, no collision, two rounds waiting")]
ROUNDS_VISITS = 1500

#the answers of a round's boards, asked for one board a call as the reference asks: a board's row in a batch differs from its row
#alone in the last bits, and a prior on the edge between two stored ones is then stored as the other
def answers_a_board_a_call(model: nn.Module, boards: np.ndarray) -> np.ndarray:
    evaluate = cpu_batches(model).evaluate_boards
    return np.concatenate([evaluate(boards[index:index + 1]) for index in range(len(boards))]) if len(boards) else NO_ANSWERS

#what Searches shows of a running search, against the reference's: the report (none before the root has moves), and with certainty
#whether the root's moves decide it and whether one move is left
def assert_same_search(searches: selfplay_rs.Searches, search: reference.Search, where: str) -> None:
    report = searches.report_now(GAME)
    assert (report is None) == (not search.root.links), where
    if report is None:
        return
    assert_same_report(tuple(report[1:-1]), report_of(search.root), where)
    if search.certainty:
        assert searches.decided(GAME) == reference.root_decided(search.root), where
        assert searches.one_move_left(GAME) == reference.one_move_left(search.root), where

#a started rust search and the reference's, round by round to `visits`: every round asks the network about the same boards in the same
#order, and leaves the same search behind, when it is gathered (the descents backed up at once) and when it is answered
def assert_the_reference_at_every_round(searches: selfplay_rs.Searches, search: reference.Search, shape: tuple, visits: int) -> None:
    leaves, collisions, simulations, rounds_waiting = shape
    unanswered, number = deque(), 0
    while True:
        number += 1
        boards = np.frombuffer(bytearray(searches.descend(leaves, collisions, simulations)), dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD)
        made = search.gather(leaves, collisions, simulations, visits)
        where = f"round {number}: in the reference " + ", ".join(f"{simulation.ended} after {' '.join(move.uci() for move in simulation.line)}" for simulation in made)
        assert boards.tolist() == [encode_board(simulation.evaluated) for simulation in made if simulation.ended == reference.EVALUATED], where
        assert_same_search(searches, search, where)
        unanswered.append(boards)
        if len(unanswered) > rounds_waiting:
            reports = searches.finish(answers_a_board_a_call(search.model, unanswered.popleft()))
            search.finish()
            if reports:
                assert_same_report(tuple(reports[0][1:-1]), report_of(search.root), where)
                assert search.root.visit_count >= visits, f"the rust search stopped at round {number}, short of its visits"
                return
            assert_same_search(searches, search, where)

def started(searches: selfplay_rs.Searches, board: chess.Board, visits: int) -> selfplay_rs.Searches:
    searches.start(GAME, board.root().fen(), moves_played(board), visits)
    return searches

@pytest.mark.parametrize("shape", ROUNDS)
@pytest.mark.parametrize("position", POSITIONS)
def test_the_rust_search_is_the_reference_at_every_round_of_several_leaves(position, shape):
    model, board = PositionModel(), POSITIONS[position]
    searches = started(move_means_searches(), board, ROUNDS_VISITS)
    assert_the_reference_at_every_round(searches, reference.Search(model, board, CPU), shape, ROUNDS_VISITS)

#proofs (certainty): mates a move or two away, a rook that leaves the back rank to be mated, a position whose every move is mated
#at once, and draws by the rules: by the fifty moves two plies down, where a node's every move is soon a proven draw, and further off
PROVING = {
    "mate in one": chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"),
    "mate in two": chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1"),
    "a rook to lose": chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    "every move loses": chess.Board("k7/6p1/1K6/8/8/8/8/7R b - - 0 1"),
    "rook ending two plies from fifty moves": chess.Board("8/8/4k3/8/8/4K3/8/R7 w - - 98 80"),
    "rook ending near fifty moves": POSITIONS["rook ending near fifty moves"],
    "blocked pawns after a shuffle": POSITIONS["blocked pawns after a shuffle"],
}
#UniformModel as well: it finds the mates soonest, and so the positions where every move is lost
CERTAIN = {"position": PositionModel, "even priors": UniformModel}
ONE_LEAF = pytest.param((1, 0, None, 0), id="1 leaf")

@pytest.mark.parametrize("shape", [ONE_LEAF, *ROUNDS[:2]])
@pytest.mark.parametrize("position", PROVING)
@pytest.mark.parametrize("model", CERTAIN)
def test_the_rust_search_with_certainty_is_the_reference_at_every_round(model, position, shape):
    model, board = CERTAIN[model](), PROVING[position]
    searches = started(move_means_searches(certainty=True), board, ROUNDS_VISITS)
    assert_the_reference_at_every_round(searches, reference.Search(model, board, CPU, certainty=True), shape, ROUNDS_VISITS)

#which says little about a case of the rule that no search came to: what the reference counts in the searches compared above
def test_the_searches_in_rounds_come_to_the_rules_cases():
    counts = {False: Counter(), True: Counter()}
    for certainty, positions, models, shapes in ((False, POSITIONS, [PositionModel], ROUNDS), (True, PROVING, CERTAIN.values(), [ONE_LEAF, *ROUNDS[:2]])):
        for model in models:
            for board in positions.values():
                for (leaves, collisions, simulations, _), in (shape.values for shape in shapes):
                    search = reference.Search(model(), board, CPU, certainty=certainty)
                    while search.root.visit_count < ROUNDS_VISITS:
                        counts[certainty].update(simulation.ended for simulation in search.gather(leaves, collisions, simulations, ROUNDS_VISITS))
                        search.finish()
                    counts[certainty].update(search.counts)

    for certainty in counts:
        #pending counts, and a move that waits for its node's first evaluation or took a node that waited: no mean, whatever
        #repeats have given the node
        assert counts[certainty][reference.COLLISION] >= 1000 and counts[certainty][reference.NO_MEAN_OF_A_VISITED_NODE] >= 50, counts
        assert counts[certainty][reference.NO_MEAN_OF_A_REPEATED_NODE] >= 3, counts
        assert counts[certainty][reference.STALE_MEAN] >= 1000 and counts[certainty][reference.MEAN_REFRESHED] >= 1000, counts
    assert counts[True][reference.PROVEN] >= 1000 and counts[True][reference.LOST_PASSED_OVER] >= 1000, counts
    #another move to a node proven lost: picked by its mean, and lost once its own descent has met the node
    assert counts[True][reference.UNLEARNED_LOSS] >= 5 and counts[True][reference.LOSS_LEARNED] >= 10, counts
    assert counts[True][reference.EVERY_MOVE_LOST] >= 1000, counts
    #a move that learns of its node's proof, by the occasion, and those of them whose mean the proven score changed
    for learned, least in ((reference.ARRIVED_AT_A_PROOF, 1000), (reference.PROVED_ITS_LEAF, 500), (reference.PROVED_UP_THE_PATH, 100),
                           (reference.GIVEN_A_PROVEN_NODE, 10), (reference.PROVEN_SCORE_TAKEN, 100)):
        assert counts[True][learned] >= least, counts


#two moves to one node, which the descent of one of them proves a loss: the other is picked by the mean it holds once more, meets the
#proof there and is lost. White's king on b6 and rook on h1, Black's king on a8 and knight on d4, Black to move. Nd4b3 Rh2 Nb3a1 and
#Nd4c2 Rh2 Nc2a1 come to one position, which the network scores well for Black, and where Rh8 mates
TWO_WAYS_FEN = "k7/8/1K6/8/3n4/8/8/7R b - - 0 1"
TWO_WAYS = ("d4b3 h1h2 b3a1".split(), "d4c2 h1h2 c2a1".split())
TWO_WAYS_VISITS = 200

def two_ways_model() -> ScriptedModel:
    model = ScriptedModel()
    model.tell(TWO_WAYS_FEN, [], {TWO_WAYS[0][0]: 0.5, TWO_WAYS[1][0]: 0.5})
    for way in TWO_WAYS:
        for plies in (1, 2):
            model.tell(TWO_WAYS_FEN, way[:plies], {way[plies]: 1.0})
    model.tell(TWO_WAYS_FEN, TWO_WAYS[0], {"h2h1": 0.7, "h2h8": 0.3}, score=0.1)
    return model

def test_the_rust_search_of_two_moves_to_a_node_proven_lost_is_the_reference_at_every_simulation():
    model, board = two_ways_model(), chess.Board(TWO_WAYS_FEN)
    steps = rust_simulations(move_means_searches(certainty=True), model, board, [], TWO_WAYS_VISITS)
    assert_the_reference_at_every_simulation(steps, reference.Search(model, board, CPU, certainty=True), TWO_WAYS_VISITS)

#which holds the rust search to the other move's one more visit only if the reference's search went so
def test_the_other_move_to_a_node_proven_lost_is_picked_once_more_and_then_lost():
    search = reference.Search(two_ways_model(), chess.Board(TWO_WAYS_FEN), CPU, certainty=True)
    ended = []
    while search.root.visit_count < TWO_WAYS_VISITS:
        simulation = search.simulate()
        ended.append((simulation.ended, [move.uci() for move in simulation.line]))
    proved = ended.index((reference.GAME_OVER, TWO_WAYS[0] + ["h2h8"]))
    #the mate was found down the first way; the next simulation went down the second into the proven node, and none after it
    assert ended[proved + 1] == (reference.PROVEN, TWO_WAYS[1])
    assert all(line[:3] not in TWO_WAYS for _, line in ended[proved + 2:])
    assert search.counts[reference.UNLEARNED_LOSS] >= 1 and search.counts[reference.LOSS_LEARNED] >= 1


#a lost move that is lost no more at a new root: the game has had its node's position twice by then, and a third time is a draw.
#White's king on b6 and rook on h1, Black's king on a8 and a knight:
#- Q, the knight on a1 and White to move: Rh8 mates. The game began at Q and went Rh2 Nb3 Rh1 to the first root, Black to move.
#- Nb3a1 comes back to Q, searched as any position the game had once. Its node is a ply down.
#- Nb3d4 Rh2 Nd4c2 Rh1 Nc2a1 comes to Q five plies down, at a node (THIRD) that the first way comes to only as a repeat. There the
#  mate is found: THIRD is proven a loss for Black, and Nc2a1, the move into it, is lost.
#- The game goes on Nb3a1 Rh2 (the network likes Rh2 before it looks at the mate). Q has been had twice now. The new root's Nc2
#  Rh1 comes to the node before THIRD, whose Nc2a1 leads to a draw: the move is picked again, and visits a proven draw
THIRD_TIME_FEN = "k7/8/1K6/8/8/8/8/n6R w - - 0 1"
BEFORE_THE_ROOT = "h1h2 a1b3 h2h1".split()
TO_THE_THIRD_TIME = "b3d4 h1h2 d4c2 h2h1 c2a1".split()
TO_THE_NEW_ROOT = "b3a1 h1h2".split()
FROM_THE_NEW_ROOT = "a1c2 h2h1 c2a1".split()
THIRD_TIME_VISITS = (120, 300)

def third_time_model() -> ScriptedModel:
    model = ScriptedModel()
    model.tell(THIRD_TIME_FEN, [], {"h1h2": 0.7, "h1h8": 0.3})
    model.tell(THIRD_TIME_FEN, BEFORE_THE_ROOT, {TO_THE_NEW_ROOT[0]: 0.5, TO_THE_THIRD_TIME[0]: 0.5})
    #one move for each board on the two ways to THIRD, which meet after the knight's move to c2. The new root's board is the game's
    #after its first move
    first_way, second_way = BEFORE_THE_ROOT + TO_THE_THIRD_TIME, ["h1h2", *FROM_THE_NEW_ROOT]
    for way, boards in ((first_way, (4, 5)), (second_way, (1, 2, 3))):
        for plies in boards:
            model.tell(THIRD_TIME_FEN, way[:plies], {way[plies]: 1.0})
    return model

def test_the_rust_search_at_a_new_root_is_the_reference_where_the_third_time_takes_a_loss_off_a_move():
    model, board = third_time_model(), played_moves(THIRD_TIME_FEN, BEFORE_THE_ROOT)
    searches, search = move_means_searches(certainty=True, tree_reuse=True), reference.Search(model, board, CPU, certainty=True)
    first_visits, visits = THIRD_TIME_VISITS
    assert_the_reference_at_every_simulation(rust_simulations(searches, model, board, BEFORE_THE_ROOT, first_visits), search, first_visits)
    before_third = search.node_of(played_moves(THIRD_TIME_FEN, BEFORE_THE_ROOT + TO_THE_THIRD_TIME[:-1]))
    into_third = next(link for link in before_third.links if link.move.uci() == TO_THE_THIRD_TIME[-1])
    assert into_third.lost and into_third.child.proof.outcome == reference.LOSS and search.counts[reference.LOSS_CLEARED] == 0

    steps = rust_simulations(searches, model, board, BEFORE_THE_ROOT + TO_THE_NEW_ROOT, visits)
    search.new_root([chess.Move.from_uci(uci) for uci in TO_THE_NEW_ROOT])
    assert not into_third.lost and into_third.child.proof.outcome == reference.DRAWN and search.counts[reference.LOSS_CLEARED] == 1
    visits_while_lost = into_third.visits
    assert_the_reference_at_every_simulation(steps, search, visits)
    #a lost move is not picked: the move was, again and again
    assert into_third.visits >= visits_while_lost + 100


#a game played down the kept tree (tree_reuse), a search at each of its roots: the search keeps what the new root reaches and, for
#one line, what the root before the line reached. A line is the moves one start plays. (advance, the engine's way to play its own
#move, is not compared: it holds the root before our move, the reply's start then holds no other, and what an earlier root reached
#is dropped only if advance's thread has finished its copy when that start comes.)
GAMES = {position: (POSITIONS | TRANSPOSING)[position] for position in ("blocked pawns", "pawn ending", "queen and pawn ending", "middlegame")}
GAME_ROOTS = 5
GAME_VISITS = 600

#the game's next two moves by the reference's tree: the root's most visited move and a reply to it, the most visited one or
#(unexpected) the least visited one. Only moves whose nodes have moves: a kept tree is played down those
def next_moves(search: reference.Search, unexpected: bool) -> list[chess.Move]:
    with_moves = lambda node: sorted((link for link in node.links if link.child is not None and link.child.links), key=lambda link: -link.visits)
    ours = with_moves(search.root)[0]
    return [ours.move, with_moves(ours.child)[-1 if unexpected else 0].move]

#the reference's search of each root of a game, each of `visits` more than the root has: before a search, `at_a_new_root` is told
#the line or lines (a move each) the game went on by since the search before
def game_searches(search: reference.Search, a_line_a_move: bool, visits: int = GAME_VISITS):
    for root in range(GAME_ROOTS):
        yield search.root.visit_count + visits if root else visits
        if root == GAME_ROOTS - 1:
            return
        moves = next_moves(search, unexpected=root % 2 == 1)
        for line in ([moves[:1], moves[1:]] if a_line_a_move else [moves]):
            search.new_root(line)
            yield line

@pytest.mark.parametrize("a_line_a_move", [False, True], ids=["our move and the reply one line", "each move its own line"])
@pytest.mark.parametrize("position", GAMES)
def test_the_rust_search_down_a_game_is_the_reference_at_every_simulation(position, a_line_a_move):
    model, board = PositionModel(), GAMES[position]
    searches, search = move_means_searches(tree_reuse=True), reference.Search(model, board, CPU)
    moves, lines_played = moves_played(board), 0
    for visits_or_line in game_searches(search, a_line_a_move):
        if isinstance(visits_or_line, int):
            assert_the_reference_at_every_simulation(rust_simulations(searches, model, board, moves, visits_or_line), search, visits_or_line)
            lines_played = 0
            continue
        moves = moves + [move.uci() for move in visits_or_line]
        lines_played += 1
        if a_line_a_move and lines_played == 1:
            #our move, played by a start of its own: its root has its visits, there is no simulation to make
            steps = rust_simulations(searches, model, board, moves, 1)
            assert [leaves for leaves, _ in steps] == [[]]
            assert_same_report(steps[0][1], report_of(search.root), " ".join(moves))

#which says little about what a new root keeps unless the searches came to nodes that only the held root reached, and nodes were dropped
def test_the_searches_down_a_game_meet_held_nodes_and_drop_others():
    counts = Counter()
    for a_line_a_move in (False, True):
        for board in GAMES.values():
            search = reference.Search(PositionModel(), board, CPU)
            for visits_or_line in game_searches(search, a_line_a_move):
                while isinstance(visits_or_line, int) and search.root.visit_count < visits_or_line:
                    search.simulate()
            counts[a_line_a_move, reference.HELD_NODE_MET] += search.counts[reference.HELD_NODE_MET]
            counts[a_line_a_move, reference.NODES_DROPPED] += search.counts[reference.NODES_DROPPED]
    assert all(count >= 200 for count in counts.values()), counts


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
    report = EngineSearch(cpu_batches(model), 1, settings={"transpositions": True}).search(board, simulations=400)
    line, score, root_moves, visits = report_of(reference.search(model, board, CPU, 400))
    found = ([move.uci() for move in report.line], report.score, [(move.uci(), move_visits) for move, move_visits in report.root_moves], report.simulations)
    assert_same_report(found, (line, score, root_moves[:len(report.root_moves)], visits), position)
