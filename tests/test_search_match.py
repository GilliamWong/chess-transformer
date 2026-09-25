#tools/play/search_match.py's match, with random movers standing in for stockfish
import random

import chess
import pytest

from fake_models import UniformModel
from inference import cpu_batches
from tools.play.search_match import OUR_NAME, play_match, result


class RandomMover:
    def __init__(self, seed: int):
        self.name = "RandomMover"
        self.random = random.Random(seed)

    def play(self, board: chess.Board, game: int) -> chess.Move:
        return self.random.choice(list(board.legal_moves))

def opening(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


#each opening played with both colours, every game played out, and each of our moves annotated with its search
def test_every_game_is_played_out_with_the_colours_swapped():
    openings = [opening("e4", "e5"), opening("d4", "d5")]

    pgns = play_match(cpu_batches(UniformModel()).evaluate_boards, [RandomMover(0), RandomMover(1)], openings, 4, simulations=8)

    assert [pgn.headers["White"] for pgn in pgns] == [OUR_NAME, "RandomMover", OUR_NAME, "RandomMover"]
    for number, pgn in enumerate(pgns):
        ours = chess.WHITE if number % 2 == 0 else chess.BLACK
        nodes = list(pgn.mainline())
        assert [node.move for node in nodes[:2]] == openings[number // 2].move_stack
        assert result(nodes[-1].board()) == pgn.headers["Result"]
        our_moves = [node for node in nodes[2:] if node.parent.board().turn == ours]
        assert our_moves and all(node.comment.startswith("expects") and "of 8" in node.comment and node.eval() is not None for node in our_moves)
        assert not any(node.comment for node in nodes[2:] if node.parent.board().turn != ours)


@pytest.mark.parametrize(
    "fen, expected",
    [
        pytest.param("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30", "1-0", id="checkmate"),
        pytest.param("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "1/2-1/2", id="stalemate"),
        pytest.param("8/8/8/4k3/8/8/8/R3K3 w - - 100 80", "1/2-1/2", id="fifty-moves"),
        pytest.param("8/8/8/4k3/8/8/8/4K3 w - - 0 1", "1/2-1/2", id="insufficient-material"),
        pytest.param("8/8/8/4k3/8/8/8/R3K3 w - - 0 301", "1/2-1/2", id="300-moves"),
        pytest.param("8/8/8/4k3/8/8/8/R3K3 w - - 0 300", None, id="going-on"),
    ],
)
def test_a_game_ends_as_cutechess_ends_it(fen, expected):
    assert result(chess.Board(fen)) == expected
