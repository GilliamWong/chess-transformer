import chess
import pytest
import torch
from torch import nn

from encoding import MOVE_TO_ID, NUM_MOVE_IDS, NUM_SQUARES
from mcts import DRAW, Node, evaluate, game_result, most_visited_move, search

CPU = torch.device("cpu")

#material in pawns by board token: the side to move's pieces count for it, the other side's against (see encode_board)
PIECE_VALUES = torch.tensor([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=torch.float)


#every move equally likely and every position even, so whatever the search finds, it found by searching
class UniformModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(len(boards), NUM_MOVE_IDS), torch.zeros(len(boards), 1)

#every move equally likely, and a position scored by its material for the side to move: a pawn up is a logit of 1
class MaterialModel(nn.Module):
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        material = PIECE_VALUES[boards[:, :NUM_SQUARES]].sum(dim=1, keepdim=True)
        return torch.zeros(len(boards), NUM_MOVE_IDS), material

#puts nearly all of the policy on the given move ids, shared out among those that are legal
class FavouriteMoveModel(nn.Module):
    def __init__(self, *move_ids: int):
        super().__init__()
        self.move_ids = list(move_ids)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(len(boards), NUM_MOVE_IDS)
        logits[:, self.move_ids] = 100
        return logits, torch.zeros(len(boards), 1)


#black's moves reach the model mirrored, as if black were white: its e7e5 is the model's e2e4
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

#the policy walks a knight out and back for each side, bringing the root position round again. Seen only twice, it is still
#a draw, so it is visited without ever being expanded
def test_a_position_repeated_inside_the_search_is_a_draw():
    model = FavouriteMoveModel(MOVE_TO_ID[chess.Move.from_uci("g1f3")], MOVE_TO_ID[chess.Move.from_uci("f3g1")])

    root = search(model, chess.Board(), CPU, 20)

    repeated = node_after(root, "g1f3", "g8f6", "f3g1", "f6g8")
    assert repeated.visit_count > 0
    assert repeated.children == {}

#the position after Nf3 Nf6 comes round once more inside the search, but its first time was in the game before the root,
#not in a loop the search chose, so it is evaluated like any other position
def test_a_single_repeat_of_the_game_before_the_root_is_not_a_draw():
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Nc3", "Nc6"):
        board.push_san(san)

    root = search(FavouriteMoveModel(MOVE_TO_ID[chess.Move.from_uci("c3b1")]), board, CPU, 20)

    repeated = node_after(root, "c3b1", "c6b8")
    assert repeated.visit_count > 0
    assert repeated.children != {}


#expanding a finished position would give a node without children, which looks like one that was never expanded
def test_a_finished_game_is_neither_evaluated_nor_searched():
    checkmate = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30")

    with pytest.raises(ValueError):
        evaluate(UniformModel(), checkmate, CPU)
    with pytest.raises(ValueError):
        search(UniformModel(), checkmate, CPU)


#one simulation only evaluates the root, so no move has a visit: the tie goes to the policy's favourite, not the first legal move
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

#one move mates and nothing tells the search which: it has to find it, and see that a mate wins for the side playing it
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


#scored by material, taking the queen is only the best move once the score after it (black to move, a queen down) is
#turned into white's score, and the replies deeper down are turned back again. A search with a sign wrong spreads its
#visits about evenly and can still pick the capture by luck, so most of them have to go there
def test_takes_a_hanging_queen():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    simulations = 200

    root = search(MaterialModel(), board, CPU, simulations)

    assert root.children[chess.Move.from_uci("d1d5")].visit_count > simulations / 2
