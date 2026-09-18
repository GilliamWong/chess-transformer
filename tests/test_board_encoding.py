from itertools import islice
from pathlib import Path

import chess
import chess.pgn
import pytest

from main import (
    BLACK_KINGSIDE_INDEX,
    BLACK_QUEENSIDE_INDEX,
    EN_PASSANT_INDEX,
    ENCODING_SIZE,
    TURN_INDEX,
    WHITE_KINGSIDE_INDEX,
    WHITE_QUEENSIDE_INDEX,
    decode_board,
    encode_board,
)

CARLSEN_PGN = Path(__file__).parent.parent / "Carlsen" / "Carlsen.pgn"
NUM_PGN_GAMES_TO_REPLAY = 5

#written out by hand so the layout is checked against something other than encode_board itself
START_POSITION_ENCODING = (
    [4, 2, 3, 5, 6, 3, 2, 4]        #rank 1, a1..h1
    + [1] * 8                       #rank 2
    + [0] * 32                      #ranks 3-6
    + [7] * 8                       #rank 7
    + [10, 8, 9, 11, 12, 9, 8, 10]  #rank 8, a8..h8
    + [1]                           #white to move
    + [1, 1, 1, 1]                  #castling: white K, white Q, black k, black q
    + [0]                           #no en passant square
)

PIECE_CODES = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}


def board_after(*san_moves: str) -> chess.Board:
    board = chess.Board()
    for san in san_moves:
        board.push_san(san)
    return board


def castling_flags(encoding) -> tuple[int, int, int, int]:
    return (
        encoding[WHITE_KINGSIDE_INDEX],
        encoding[WHITE_QUEENSIDE_INDEX],
        encoding[BLACK_KINGSIDE_INDEX],
        encoding[BLACK_QUEENSIDE_INDEX],
    )


def assert_round_trip(board: chess.Board) -> None:
    decoded = decode_board(encode_board(board))
    #EPD is placement, turn, castling and en passant: exactly what the encoding carries.
    #The move counters are not encoded, so Board.__eq__ would be too strict here.
    #epd() only writes the en passant square when a capture is legal, matching encode_board.
    assert decoded.epd() == board.epd()


def test_start_position_layout():
    encoding = encode_board(chess.Board())

    assert len(encoding) == ENCODING_SIZE
    assert encoding == START_POSITION_ENCODING


def test_every_piece_has_its_own_code():
    board = chess.Board.empty()
    squares = dict(zip(PIECE_CODES, chess.SQUARES))
    for symbol, square in squares.items():
        board.set_piece_at(square, chess.Piece.from_symbol(symbol))

    encoding = encode_board(board)

    for symbol, square in squares.items():
        assert encoding[square] == PIECE_CODES[symbol], symbol
    assert_round_trip(board)


def test_black_to_move():
    board = board_after("e4")

    assert encode_board(board)[TURN_INDEX] == 0
    assert decode_board(encode_board(board)).turn == chess.BLACK


@pytest.mark.parametrize(
    "san_moves, expected_flags",
    [
        pytest.param(["e4", "e5", "Ke2"], (0, 0, 1, 1), id="white-king-moved"),
        pytest.param(["e4", "e5", "Ke2", "Ke7"], (0, 0, 0, 0), id="both-kings-moved"),
        pytest.param(["h4", "h5", "Rh3"], (0, 1, 1, 1), id="white-h-rook-moved"),
        pytest.param(["a4", "a5", "Ra3", "Ra6"], (1, 0, 1, 0), id="both-a-rooks-moved"),
        pytest.param(
            ["b3", "b6", "Bb2", "Bb7", "Bxg7", "Bxg2", "Bxh8"],
            (1, 1, 0, 1),
            id="black-h-rook-captured-at-home",
        ),
        pytest.param(
            ["b3", "b6", "Bb2", "Bb7", "Bxg7", "Bxg2", "Bxh8", "Bxh1"],
            (0, 1, 0, 1),
            id="both-h-rooks-captured-at-home",
        ),
        pytest.param(
            ["e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5", "O-O"],
            (0, 0, 1, 1),
            id="white-castled-kingside",
        ),
        pytest.param(
            ["e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5", "O-O", "O-O"],
            (0, 0, 0, 0),
            id="both-castled-kingside",
        ),
        pytest.param(
            ["d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7", "O-O-O"],
            (0, 0, 1, 1),
            id="white-castled-queenside",
        ),
    ],
)
def test_lost_castling_rights(san_moves, expected_flags):
    board = board_after(*san_moves)

    assert castling_flags(encode_board(board)) == expected_flags
    assert_round_trip(board)


def test_castling_rights_stay_lost_when_pieces_return_home():
    never_moved = board_after("e4", "e5")
    moved_and_returned = board_after("e4", "e5", "Ke2", "Ke7", "Ke1", "Ke8")
    assert never_moved.board_fen() == moved_and_returned.board_fen()

    never_moved_encoding = encode_board(never_moved)
    returned_encoding = encode_board(moved_and_returned)

    #same pieces on the same squares, so only the castling flags can tell these apart
    assert castling_flags(never_moved_encoding) == (1, 1, 1, 1)
    assert castling_flags(returned_encoding) == (0, 0, 0, 0)
    assert never_moved_encoding[:TURN_INDEX + 1] == returned_encoding[:TURN_INDEX + 1]

    decoded = decode_board(returned_encoding)
    assert not decoded.has_castling_rights(chess.WHITE)
    assert not decoded.has_castling_rights(chess.BLACK)


def test_decoded_board_can_still_castle():
    board = board_after("e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5")

    decoded = decode_board(encode_board(board))

    assert decoded.parse_san("O-O") in decoded.legal_moves


@pytest.mark.parametrize(
    "san_moves, expected_ep_square",
    [
        #a3 and h6 are the smallest and largest en passant squares, so 0 (a1) is a safe "none" value
        pytest.param(["h3", "b5", "h4", "b4", "a4"], chess.A3, id="a3-lowest-square"),
        pytest.param(["g4", "a6", "g5", "h5"], chess.H6, id="h6-highest-square"),
        #a double push that no enemy pawn can capture leaves the position playing the same, so it encodes as none
        pytest.param(["e4"], 0, id="no-capture-available"),
        #the capturing pawn is pinned to its king, so the capture is not legal
        pytest.param(["e4", "e5", "d4", "exd4", "e5", "Qe7", "a3", "d5"], 0, id="capturing-pawn-pinned"),
        pytest.param(["e4", "a6", "e5", "d5"], chess.D6, id="d6-white-can-capture"),
        pytest.param(["a3", "e5", "a4", "e4", "d4"], chess.D3, id="d3-black-can-capture"),
    ],
)
def test_en_passant_square(san_moves, expected_ep_square):
    board = board_after(*san_moves)

    assert encode_board(board)[EN_PASSANT_INDEX] == expected_ep_square
    assert_round_trip(board)


@pytest.mark.parametrize(
    "san_moves, capture_san",
    [
        pytest.param(["e4", "a6", "e5", "d5"], "exd6", id="white-captures"),
        pytest.param(["a3", "e5", "a4", "e4", "d4"], "exd3", id="black-captures"),
    ],
)
def test_decoded_board_allows_en_passant_capture(san_moves, capture_san):
    decoded = decode_board(encode_board(board_after(*san_moves)))

    capture = decoded.parse_san(capture_san)

    assert decoded.is_en_passant(capture)
    assert capture in decoded.legal_moves


def test_en_passant_square_expires_after_one_move():
    board = board_after("e4", "a6", "e5", "d5", "Nf3")

    encoding = encode_board(board)

    assert encoding[EN_PASSANT_INDEX] == 0
    assert decode_board(encoding).ep_square is None


def test_en_passant_capture_removes_the_captured_pawn():
    board = board_after("e4", "a6", "e5", "d5", "exd6")

    encoding = encode_board(board)

    assert encoding[chess.D5] == 0
    assert encoding[chess.D6] == PIECE_CODES["P"]
    assert_round_trip(board)


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            id="midgame-all-rights",
        ),
        pytest.param(
            "r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R b Kq - 4 9",
            id="midgame-mixed-rights-black-to-move",
        ),
        pytest.param("3Q4/5k2/8/8/2Q5/8/5K2/q2q4 b - - 0 60", id="promoted-queens"),
        pytest.param("6k1/8/8/8/8/2N5/1N3K2/N1N5 w - - 0 70", id="underpromoted-knights"),
        pytest.param("8/8/4k3/8/8/3K4/8/8 b - - 12 90", id="bare-kings"),
        pytest.param("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 30", id="checkmate"),
    ],
)
def test_round_trip_position(fen):
    assert_round_trip(chess.Board(fen))


def test_round_trip_every_position_of_real_games():
    with open(CARLSEN_PGN) as pgn:
        games = list(islice(iter(lambda: chess.pgn.read_game(pgn), None), NUM_PGN_GAMES_TO_REPLAY))
    assert len(games) == NUM_PGN_GAMES_TO_REPLAY

    for game in games:
        board = game.board()
        assert_round_trip(board)
        for move in game.mainline_moves():
            board.push(move)
            assert_round_trip(board)
