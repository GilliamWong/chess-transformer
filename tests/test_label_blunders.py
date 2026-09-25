#the blunder labeller (tools/label_blunders.py) against the real stockfish; skipped where it has not been downloaded
import chess
import numpy as np
import pytest

from encoding import encode_board, encode_move_to_id
from tools import label_blunders
from tools.label_blunders import MIN_LOSS, NO_MOVE, SLOTS, STOCKFISH, Engine, label_position

pytestmark = pytest.mark.skipif(not STOCKFISH.exists(), reason="stockfish is not downloaded (python -m tools.play.download_engines)")

#black to move, a queen up, the queen attacked by the rook: a king move leaves it to be taken, a queen move keeps it
QUEEN_ATTACKED = "4k3/8/8/8/3q4/8/8/3RK3 b - - 0 1"


def test_a_move_that_hangs_the_queen_is_a_blunder_and_a_safe_one_is_not():
    board = chess.Board(QUEEN_ATTACKED)
    engine = Engine(STOCKFISH)
    try:
        moves, losses = label_position(engine, np.array(encode_board(board), dtype=np.uint8))
    finally:
        engine.process.kill()

    labelled = {int(move): float(loss) for move, loss in zip(moves, losses) if move != NO_MOVE}
    assert encode_move_to_id(board.parse_san("Ke7"), board.turn) in labelled
    assert encode_move_to_id(board.parse_san("Qa4"), board.turn) not in labelled
    assert all(loss >= MIN_LOSS - 1e-3 for loss in labelled.values())
    #the mildest first, and the places past them empty
    assert list(losses[:len(labelled)]) == sorted(losses[:len(labelled)])
    assert (moves[len(labelled):] == NO_MOVE).all() and moves.shape == losses.shape == (SLOTS,)


#chunks of 2 rows: a file of 3 boards is 2 chunks, the second one row; merge joins them back in order
def test_labelled_chunks_merge_into_the_set_files_and_are_not_labelled_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(label_blunders, "CHUNK", 2)
    boards = np.array([encode_board(chess.Board(QUEEN_ATTACKED)), encode_board(chess.Board()), encode_board(chess.Board(QUEEN_ATTACKED))], dtype=np.uint8)
    boards_path = tmp_path / "tiny_boards.bin"
    boards.tofile(boards_path)

    label_blunders.label(boards_path, tmp_path / "chunks", workers=1, engine=STOCKFISH)
    first_chunk = label_blunders.chunk_paths(tmp_path / "chunks", boards_path, 0)[0]
    labelled_at = first_chunk.stat().st_mtime_ns
    label_blunders.label(boards_path, tmp_path / "chunks", workers=1, engine=STOCKFISH)  #everything is done: nothing is redone
    label_blunders.merge(tmp_path / "chunks", [boards_path], tmp_path / "tiny_train")

    assert first_chunk.stat().st_mtime_ns == labelled_at
    moves = np.fromfile(tmp_path / "tiny_train_worst_moves.bin", dtype=np.int16).reshape(-1, SLOTS)
    losses = np.fromfile(tmp_path / "tiny_train_worst_losses.bin", dtype=np.float16).reshape(-1, SLOTS)
    assert moves.shape == losses.shape == (3, SLOTS)
    #both copies of the queen position carry the move that hangs the queen (not the same rows move for move: a search
    #depends on what the engine's hash holds from the positions before), and the start position has no blunder
    hangs_the_queen = encode_move_to_id(chess.Board(QUEEN_ATTACKED).parse_san("Ke7"), chess.BLACK)
    assert hangs_the_queen in moves[0] and hangs_the_queen in moves[2]
    assert (moves[1] == NO_MOVE).all()
