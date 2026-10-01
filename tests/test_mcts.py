import numpy as np
import chess
import pytest
import torch
from torch import nn

from model.encoding import MOVE_TO_ID, NUM_MOVE_IDS, NUM_SQUARES
from fake_models import PIECE_VALUES, UniformModel
from engine.mcts import DRAW, Node, evaluate, evaluate_many, game_result, most_visited_move, search
from model.model import ChessTransformer

CPU = torch.device("cpu")


#every move equally likely, and a position scored by its material for the side to move: a pawn up is a logit of 1
class MaterialModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        material = PIECE_VALUES[boards[:, :NUM_SQUARES]].sum(dim=1, keepdim=True)
        return torch.zeros(len(boards), NUM_MOVE_IDS), material

#nearly all of the policy on the given move ids, shared among those that are legal
class FavouriteMoveModel(nn.Module):
    def __init__(self, *move_ids: int):
        super().__init__()
        self.move_ids = list(move_ids)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(len(boards), NUM_MOVE_IDS)
        logits[:, self.move_ids] = 100
        return logits, torch.zeros(len(boards), 1)


#black's moves reach the model mirrored: its e7e5 is the model's e2e4
def test_black_moves_get_the_priors_of_their_mirrored_ids():
    board = chess.Board()
    board.push_san("e4")

    priors, _ = evaluate(FavouriteMoveModel(MOVE_TO_ID[chess.Move.from_uci("e2e4")]), board, CPU)

    assert priors[chess.Move.from_uci("e7e5")] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fen, expected",
    [
        #the side to move is the one that got mated
        pytest.param("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30", 0.0, id="checkmate"),
        pytest.param("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", DRAW, id="stalemate"),
        pytest.param("8/8/8/4k3/8/8/8/R3K3 w - - 100 80", DRAW, id="fifty-moves"),
        pytest.param("8/8/8/4k3/8/8/8/4K3 w - - 0 1", DRAW, id="insufficient-material"),
        pytest.param(chess.STARTING_FEN, None, id="not-over"),
    ],
)
def test_game_result(fen, expected):
    assert game_result(chess.Board(fen)) == expected


def test_a_threefold_repetition_is_a_draw():
    board = chess.Board()
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)

    assert game_result(board) == DRAW


def node_after(root: Node, *moves: str) -> Node:
    node = root
    for move in moves:
        node = node.children[chess.Move.from_uci(move)]
    return node

#the knights go out and back: the root position seen a second time inside the search is a draw, never expanded
def test_a_position_repeated_inside_the_search_is_a_draw():
    model = FavouriteMoveModel(MOVE_TO_ID[chess.Move.from_uci("g1f3")], MOVE_TO_ID[chess.Move.from_uci("f3g1")])

    root = search(model, chess.Board(), CPU, 20)

    repeated = node_after(root, "g1f3", "g8f6", "f3g1", "f6g8")
    assert repeated.visit_count > 0
    assert repeated.children == {}

#its first occurrence was in the game before the root, not in a loop the search chose, so it is evaluated as usual
def test_a_single_repeat_of_the_game_before_the_root_is_not_a_draw():
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Nc3", "Nc6"):
        board.push_san(san)

    root = search(FavouriteMoveModel(MOVE_TO_ID[chess.Move.from_uci("c3b1")]), board, CPU, 20)

    repeated = node_after(root, "c3b1", "c6b8")
    assert repeated.visit_count > 0
    assert repeated.children != {}


#an expanded finished position would have no children and look unexpanded
def test_a_finished_game_is_neither_evaluated_nor_searched():
    checkmate = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30")

    with pytest.raises(ValueError):
        evaluate(UniformModel(), checkmate, CPU)
    with pytest.raises(ValueError):
        search(UniformModel(), checkmate, CPU)


#one simulation visits no move: the tie goes to the policy's favourite, not the first legal move
def test_a_search_too_short_to_visit_anything_plays_the_policys_move():
    favourite = chess.Move.from_uci("g1f3")

    root = search(FavouriteMoveModel(MOVE_TO_ID[favourite]), chess.Board(), CPU, 1)

    assert most_visited_move(root) == favourite


def test_every_simulation_passes_through_the_root():
    root = search(UniformModel(), chess.Board(), CPU, 50)

    assert root.visit_count == 50


def mates_on_the_spot(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mated = board.is_checkmate()
    board.pop()
    return mated

#nothing tells the search which move mates: it has to find it, and score the mate as a win for the side playing it
@pytest.mark.parametrize(
    "fen, mate",
    [
        pytest.param("k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8", id="white-mates"),
        pytest.param("7r/8/8/8/8/1k6/8/K7 b - - 0 1", "h8h1", id="black-mates"),
    ],
)
def test_finds_mate_in_one(fen, mate):
    board = chess.Board(fen)
    assert [move.uci() for move in board.legal_moves if mates_on_the_spot(board, move)] == [mate]

    root = search(UniformModel(), board, CPU, 800)

    mating_move = chess.Move.from_uci(mate)
    assert most_visited_move(root) == mating_move
    child = root.children[mating_move]
    assert child.value_sum / child.visit_count == pytest.approx(1.0)


#catches a sign error between levels; with one, the visits spread about evenly and the capture wins only by luck
def test_takes_a_hanging_queen():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    simulations = 200

    root = search(MaterialModel(), board, CPU, simulations)

    assert root.children[chess.Move.from_uci("d1d5")].visit_count > simulations / 2


def test_root_noise_moves_the_priors_the_same_way_for_the_same_seed():
    roots = [search(UniformModel(), chess.Board(), CPU, 2, root_noise=np.random.default_rng(seed)) for seed in (7, 7, 8)]
    priors = [[child.prior for child in root.children.values()] for root in roots]

    assert sum(priors[0]) == pytest.approx(1.0)
    assert priors[0] == priors[1] != priors[2]
    assert max(priors[0]) > 1 / 20 + 0.01 #20 moves at the start, equally likely without the noise


def test_positions_evaluated_together_get_what_they_get_alone(carlsen_games):
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=1, n_heads=2).eval()
    board = carlsen_games[0].board()
    boards = []
    for move in list(carlsen_games[0].mainline_moves())[:6]:
        board.push(move)
        boards.append(board.copy())

    together = evaluate_many(model, boards, CPU)

    for position, (priors, value) in zip(boards, together):
        alone_priors, alone_value = evaluate_many(model, [position], CPU)[0]
        assert value == pytest.approx(alone_value, abs=1e-5)
        assert list(priors) == list(alone_priors)
        assert list(priors.values()) == pytest.approx(list(alone_priors.values()), abs=1e-5)
