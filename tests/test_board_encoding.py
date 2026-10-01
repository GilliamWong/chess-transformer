from random import Random

import chess
import pytest

from model.encoding import (
    ALL_POSSIBLE_MOVES,
    CASTLING_TOKEN_OFFSET,
    EN_PASSANT_TOKEN_OFFSET,
    MOVE_TO_ID,
    NUM_PIECE_TYPES,
    TURN_TOKEN_OFFSET,
    BLACK_KINGSIDE_INDEX,
    BLACK_QUEENSIDE_INDEX,
    EN_PASSANT_INDEX,
    TOKENS_PER_BOARD,
    BOARD_VOCAB_SIZE,
    MOVE_FROM_SQUARES,
    MOVE_TO_SQUARES,
    PROMOTION_MOVE_IDS,
    PROMOTION_PIECES,
    PROMOTION_TO_SQUARES,
    NUM_MOVE_IDS,
    TURN_INDEX,
    WHITE_KINGSIDE_INDEX,
    WHITE_QUEENSIDE_INDEX,
    decode_board,
    decode_id_to_move,
    encode_board,
    encode_move_to_id,
)

#written out by hand so the layout is checked against something other than encode_board itself
START_POSITION_ENCODING = (
    [4, 2, 3, 5, 6, 3, 2, 4]        #rank 1, a1..h1
    + [1] * 8                       #rank 2
    + [0] * 32                      #ranks 3-6
    + [7] * 8                       #rank 7
    + [10, 8, 9, 11, 12, 9, 8, 10]  #rank 8, a8..h8
    + [14]                          #white to move
    + [16, 16, 16, 16]              #castling: white K, white Q, black k, black q
    + [17]                          #no en passant square
)

#after 1. e4, as black sees it: black's own pieces at the bottom and coded as white, the e4 pawn on e5
AFTER_E4_ENCODING = (
    [4, 2, 3, 5, 6, 3, 2, 4]
    + [1] * 8
    + [0] * 16
    + [0, 0, 0, 0, 7, 0, 0, 0]      #rank 5
    + [0] * 8
    + [7, 7, 7, 7, 0, 7, 7, 7]      #rank 7
    + [10, 8, 9, 11, 12, 9, 8, 10]
    + [13]                          #black to move
    + [16, 16, 16, 16]
    + [17]
)

BLACK_TO_MOVE, WHITE_TO_MOVE = 13, 14
NO_CASTLING_RIGHT, CASTLING_RIGHT = 15, 16
NO_EN_PASSANT = 17  #en passant square s is token 17 + s

PIECE_CODES = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}


def board_after(*san_moves: str) -> chess.Board:
    board = chess.Board()
    for san in san_moves:
        board.push_san(san)
    return board


#1 for a castling right, 0 for none, anything else for any other token
def castling_flags(encoding) -> tuple[int, int, int, int]:
    return (
        encoding[WHITE_KINGSIDE_INDEX] - NO_CASTLING_RIGHT,
        encoding[WHITE_QUEENSIDE_INDEX] - NO_CASTLING_RIGHT,
        encoding[BLACK_KINGSIDE_INDEX] - NO_CASTLING_RIGHT,
        encoding[BLACK_QUEENSIDE_INDEX] - NO_CASTLING_RIGHT,
    )


def assert_round_trip(board: chess.Board) -> None:
    decoded = decode_board(encode_board(board))
    #epd is exactly what the encoding carries (no move counters), and like encode_board it only writes a
    #capturable en passant square
    assert decoded.epd() == board.epd()


def test_start_position_layout():
    encoding = encode_board(chess.Board())

    assert len(encoding) == TOKENS_PER_BOARD
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

    assert encode_board(board) == AFTER_E4_ENCODING
    assert decode_board(encode_board(board)).turn == chess.BLACK


#the flags are the side to move's kingside and queenside, then the opponent's
@pytest.mark.parametrize(
    "san_moves, expected_flags",
    [
        pytest.param(["e4", "e5", "Ke2"], (1, 1, 0, 0), id="white-king-moved-black-to-move"),
        pytest.param(["e4", "e5", "Ke2", "Ke7"], (0, 0, 0, 0), id="both-kings-moved"),
        pytest.param(["h4", "h5", "Rh3"], (1, 1, 0, 1), id="white-h-rook-moved-black-to-move"),
        pytest.param(["a4", "a5", "Ra3", "Ra6"], (1, 0, 1, 0), id="both-a-rooks-moved"),
        pytest.param(
            ["b3", "b6", "Bb2", "Bb7", "Bxg7", "Bxg2", "Bxh8"],
            (0, 1, 1, 1),
            id="black-h-rook-captured-at-home-black-to-move",
        ),
        pytest.param(
            ["b3", "b6", "Bb2", "Bb7", "Bxg7", "Bxg2", "Bxh8", "Bxh1"],
            (0, 1, 0, 1),
            id="both-h-rooks-captured-at-home",
        ),
        pytest.param(
            ["e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5", "O-O"],
            (1, 1, 0, 0),
            id="white-castled-kingside-black-to-move",
        ),
        pytest.param(
            ["e4", "e5", "Nf3", "Nf6", "Bc4", "Bc5", "O-O", "O-O"],
            (0, 0, 0, 0),
            id="both-castled-kingside",
        ),
        pytest.param(
            ["d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7", "O-O-O"],
            (1, 1, 0, 0),
            id="white-castled-queenside-black-to-move",
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
        #the side to move captures up the board, so the square is on rank 6 and 0 (a1) is a safe "none"
        pytest.param(["h3", "b5", "h4", "b4", "a4"], chess.A6, id="a3-mirrored-for-black"),
        pytest.param(["g4", "a6", "g5", "h5"], chess.H6, id="h6-highest-square"),
        #a double push no enemy pawn can capture encodes as none
        pytest.param(["e4"], 0, id="no-capture-available"),
        #the capturing pawn is pinned to its king, so the capture is not legal
        pytest.param(["e4", "e5", "d4", "exd4", "e5", "Qe7", "a3", "d5"], 0, id="capturing-pawn-pinned"),
        pytest.param(["e4", "a6", "e5", "d5"], chess.D6, id="d6-white-can-capture"),
        pytest.param(["a3", "e5", "a4", "e4", "d4"], chess.D6, id="d3-mirrored-black-can-capture"),
    ],
)
def test_en_passant_square(san_moves, expected_ep_square):
    board = board_after(*san_moves)

    assert encode_board(board)[EN_PASSANT_INDEX] == NO_EN_PASSANT + expected_ep_square
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

    assert encoding[EN_PASSANT_INDEX] == NO_EN_PASSANT
    assert decode_board(encoding).ep_square is None


def test_en_passant_capture_removes_the_captured_pawn():
    board = board_after("e4", "a6", "e5", "d5", "exd6", "a5")

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


def test_round_trip_every_position_of_real_games(carlsen_games):
    for game in carlsen_games:
        board = game.board()
        assert_round_trip(board)
        for move in game.mainline_moves():
            board.push(move)
            assert_round_trip(board)


#counted on paper: 1456 queen-line pairs + 336 knight jumps, plus 44 promoting (from, to) pairs * 4 pieces
def test_move_vocabulary_size():
    assert NUM_MOVE_IDS == 1792 + 176
    assert len(ALL_POSSIBLE_MOVES) == NUM_MOVE_IDS
    assert len(set(ALL_POSSIBLE_MOVES)) == NUM_MOVE_IDS  #a duplicate would make two ids decode to one move


#the order of the table is the vocabulary: changing it invalidates every saved id and trained model
@pytest.mark.parametrize(
    "uci, turn, expected_id",
    [
        pytest.param("a1b1", chess.WHITE, 0, id="first-move"),
        pytest.param("a7a8n", chess.WHITE, 1792, id="first-promotion"),
        pytest.param("h2h1q", chess.WHITE, 1967, id="last-promotion"),
        #black's moves get the id of the same move seen from black's side
        pytest.param("a8b8", chess.BLACK, 0, id="black-first-move"),
        pytest.param("a2a1n", chess.BLACK, 1792, id="black-first-promotion"),
    ],
)
def test_move_id_layout(uci, turn, expected_id):
    move = chess.Move.from_uci(uci)

    assert encode_move_to_id(move, turn) == expected_id
    assert decode_id_to_move(expected_id, turn) == move


def test_black_and_white_share_ids_for_the_same_move():
    assert encode_move_to_id(chess.Move.from_uci("e7e5"), chess.BLACK) == encode_move_to_id(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert encode_move_to_id(chess.Move.from_uci("g8f6"), chess.BLACK) == encode_move_to_id(chess.Move.from_uci("g1f3"), chess.WHITE)


def test_every_id_round_trips():
    for turn in (chess.WHITE, chess.BLACK):
        for move_id in range(NUM_MOVE_IDS):
            assert encode_move_to_id(decode_id_to_move(move_id, turn), turn) == move_id


def test_impossible_move_is_rejected():
    #must fail loudly rather than become a wrong training label
    with pytest.raises(KeyError):
        encode_move_to_id(chess.Move(chess.A1, chess.B4), chess.WHITE)


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", id="white-castling"),
        pytest.param("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", id="black-castling"),
        pytest.param("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", id="en-passant"),
        pytest.param("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", id="white-edge-file-promotions"),
        pytest.param("4k3/8/8/8/8/8/7p/4K1N1 b - - 0 1", id="black-edge-file-promotions"),
    ],
)
def test_special_legal_moves_have_ids(fen):
    board = chess.Board(fen)

    for move in board.legal_moves:
        assert decode_id_to_move(encode_move_to_id(move, board.turn), board.turn) == move, move.uci()


#catches a wrong enumeration (a missing direction or piece), which the round trip over the table cannot see
def test_every_legal_move_of_real_games_has_an_id(carlsen_games):
    for game in carlsen_games:
        board = game.board()
        for played_move in game.mainline_moves():
            for move in board.legal_moves:
                assert decode_id_to_move(encode_move_to_id(move, board.turn), board.turn) == move, move.uci()
            board.push(played_move)


#a token never means two things. Blocks written out by hand
def test_every_slot_stays_inside_its_token_block(carlsen_games):
    assert BOARD_VOCAB_SIZE == 81

    for game in carlsen_games:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            encoding = encode_board(board)

            assert all(0 <= token <= 12 for token in encoding[:TURN_INDEX])
            assert encoding[TURN_INDEX] in (BLACK_TO_MOVE, WHITE_TO_MOVE)
            assert all(flag in (0, 1) for flag in castling_flags(encoding))
            assert NO_EN_PASSANT <= encoding[EN_PASSANT_INDEX] < BOARD_VOCAB_SIZE


#the policy head looks moves up by these instead of by id, so they have to describe the same move
def test_move_parts_describe_the_move_with_that_id():
    assert len(MOVE_FROM_SQUARES) == len(MOVE_TO_SQUARES) == NUM_MOVE_IDS

    for move_id in range(NUM_MOVE_IDS):
        move = decode_id_to_move(move_id, chess.WHITE)
        assert (move.from_square, move.to_square) == (MOVE_FROM_SQUARES[move_id], MOVE_TO_SQUARES[move_id])


def test_every_promotion_shares_its_squares_with_a_plain_move():
    #the head scores a promotion as its (from, to) pair plus something for the piece, so the pair has to exist
    plain_pairs = {(move.from_square, move.to_square) for move in ALL_POSSIBLE_MOVES if move.promotion is None}
    promotion_pairs = [(MOVE_FROM_SQUARES[move_id], MOVE_TO_SQUARES[move_id]) for move_id in PROMOTION_MOVE_IDS]

    assert len(promotion_pairs) == 176 and len(set(promotion_pairs)) == 44
    assert set(promotion_pairs) <= plain_pairs


#the policy head adds something to exactly these ids, picked by landing square and piece
def test_promotion_tables_describe_every_promotion_and_nothing_else():
    assert len(PROMOTION_MOVE_IDS) == len(PROMOTION_TO_SQUARES) == len(PROMOTION_PIECES) == 176
    assert set(PROMOTION_MOVE_IDS) == {move_id for move_id, move in enumerate(ALL_POSSIBLE_MOVES) if move.promotion is not None}

    pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for move_id, to_square, piece in zip(PROMOTION_MOVE_IDS, PROMOTION_TO_SQUARES, PROMOTION_PIECES):
        move = ALL_POSSIBLE_MOVES[move_id]
        assert move.to_square == to_square
        assert move.promotion == pieces[piece]


#the slow, obvious encoding (board.mirror(), then square by square): the reference the fast one must match exactly
def reference_encode_board(board: chess.Board) -> list[int]:
    turn = board.turn
    if turn == chess.BLACK:
        board = board.mirror()
    encoding = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        encoding.append(0 if piece is None else piece.piece_type + (0 if piece.color == chess.WHITE else NUM_PIECE_TYPES))
    encoding.append(TURN_TOKEN_OFFSET + (1 if turn == chess.WHITE else 0))
    for color in (chess.WHITE, chess.BLACK):
        encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(color) else 0))
        encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(color) else 0))
    encoding.append(EN_PASSANT_TOKEN_OFFSET + (board.ep_square if board.ep_square is not None and board.has_legal_en_passant() else 0))
    return encoding

def reference_move_id(move: chess.Move, turn: chess.Color) -> int:
    if turn == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)
    return MOVE_TO_ID[move]

#real games, and random ones: random play reaches promotions, en passant and castling rights lost in odd ways
def many_positions(carlsen_games) -> list[chess.Board]:
    boards = []
    for game in carlsen_games:
        board = game.board()
        for move in game.mainline_moves():
            boards.append(board.copy())
            board.push(move)
    random = Random(0)
    for _ in range(60):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 300:
            boards.append(board.copy())
            board.push(random.choice(list(board.legal_moves)))
    return boards

def test_the_encoding_matches_the_reference_everywhere(carlsen_games):
    boards = many_positions(carlsen_games)
    assert any(board.has_legal_en_passant() for board in boards) and any(board.turn == chess.BLACK for board in boards)

    for board in boards:
        assert encode_board(board) == reference_encode_board(board), board.fen()
        for move in board.legal_moves:
            assert encode_move_to_id(move, board.turn) == reference_move_id(move, board.turn), (board.fen(), move)
