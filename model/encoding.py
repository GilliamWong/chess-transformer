import chess

NUM_SQUARES = 64
NUM_PIECE_TYPES = 6

TURN_INDEX = 64
WHITE_KINGSIDE_INDEX = 65
WHITE_QUEENSIDE_INDEX = 66
BLACK_KINGSIDE_INDEX = 67
BLACK_QUEENSIDE_INDEX = 68
EN_PASSANT_INDEX = 69

TOKENS_PER_BOARD = 70

#a block of token ids per slot kind; rules.rs repeats the offsets
NUM_PIECE_CODES = 1 + 2 * NUM_PIECE_TYPES #0 empty, then the side to move's piece types, then the other side's
NUM_FLAG_VALUES = 2

TURN_TOKEN_OFFSET = NUM_PIECE_CODES
CASTLING_TOKEN_OFFSET = TURN_TOKEN_OFFSET + NUM_FLAG_VALUES
EN_PASSANT_TOKEN_OFFSET = CASTLING_TOKEN_OFFSET + NUM_FLAG_VALUES

BOARD_VOCAB_SIZE = EN_PASSANT_TOKEN_OFFSET + NUM_SQUARES #the en passant slot is 0 for none, otherwise the square

def _build_all_possible_moves() -> tuple[chess.Move, ...]:
    board = chess.Board.empty()
    moves = []
    queen = chess.Piece(chess.QUEEN, chess.WHITE)
    knight = chess.Piece(chess.KNIGHT, chess.WHITE)

    for square in range(NUM_SQUARES):
        board.set_piece_at(square, queen)
        destinations = board.attacks(square)
        board.set_piece_at(square, knight)
        destinations = destinations.union(board.attacks(square))
        board.remove_piece_at(square)

        moves.extend(chess.Move(square, destination) for destination in destinations)

    #both colours' promotions; the order fixes the move ids trained checkpoints use
    for from_rank, to_rank in [(6, 7), (1, 0)]:
        for from_file in range(8):
            for to_file in range(max(from_file - 1, 0), min(from_file + 2, 8)):
                from_square = chess.square(from_file, from_rank)
                to_square = chess.square(to_file, to_rank)
                for piece in (chess.KNIGHT, chess.ROOK, chess.BISHOP, chess.QUEEN):
                    moves.append(chess.Move(from_square, to_square, promotion=piece))

    return tuple(moves)

ALL_POSSIBLE_MOVES = _build_all_possible_moves()
NUM_MOVE_IDS = len(ALL_POSSIBLE_MOVES)

#no eval is the int16 minimum; no move also marks a position not labelled yet
NO_EVAL = -32768
NO_MOVE = -1
#expected score (0 lost, 1 won) is sigmoid(EXPECTED_SCORE_PER_CENTIPAWN * centipawns), lichess's fit
EXPECTED_SCORE_PER_CENTIPAWN = 0.00368208

MOVE_TO_ID = {move: move_id for move_id, move in enumerate(ALL_POSSIBLE_MOVES)}
MOVE_IDS_BY_SQUARES = {(move.from_square, move.to_square, move.promotion): move_id for move, move_id in MOVE_TO_ID.items()}

MOVE_FROM_SQUARES = tuple(move.from_square for move in ALL_POSSIBLE_MOVES)
MOVE_TO_SQUARES = tuple(move.to_square for move in ALL_POSSIBLE_MOVES)

#the promotion moves' ids and landing squares, and their piece as 0 knight, 1 bishop, 2 rook, 3 queen
PROMOTION_MOVE_IDS = tuple(move_id for move_id, move in enumerate(ALL_POSSIBLE_MOVES) if move.promotion is not None)
PROMOTION_TO_SQUARES = tuple(ALL_POSSIBLE_MOVES[move_id].to_square for move_id in PROMOTION_MOVE_IDS)
PROMOTION_PIECES = tuple(ALL_POSSIBLE_MOVES[move_id].promotion - chess.KNIGHT for move_id in PROMOTION_MOVE_IDS)

#encoded from the side to move: with black to move the ranks are flipped (square ^ 56) and the colours swapped; the turn token keeps the real colour
def encode_board(board: chess.Board) -> list[int]:
    turn = board.turn
    opponent = not turn
    flip = 0 if turn == chess.WHITE else 56

    encoding = [0] * NUM_SQUARES
    for piece_type in chess.PIECE_TYPES:
        for square in chess.scan_forward(board.pieces_mask(piece_type, turn)):
            encoding[square ^ flip] = piece_type
        for square in chess.scan_forward(board.pieces_mask(piece_type, opponent)):
            encoding[square ^ flip] = piece_type + NUM_PIECE_TYPES

    encoding += [
        TURN_TOKEN_OFFSET + (1 if turn == chess.WHITE else 0),
        CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(turn) else 0),
        CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(turn) else 0),
        CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(opponent) else 0),
        CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(opponent) else 0),
        EN_PASSANT_TOKEN_OFFSET + (board.ep_square ^ flip if board.ep_square is not None and board.has_legal_en_passant() else 0),
    ]
    return encoding

def decode_board(encoding: list[int]) -> chess.Board:
    board = chess.Board.empty()

    ep_square = encoding[EN_PASSANT_INDEX] - EN_PASSANT_TOKEN_OFFSET
    board.ep_square = None if ep_square == 0 else ep_square

    castling_slots = ((WHITE_KINGSIDE_INDEX, "K"), (WHITE_QUEENSIDE_INDEX, "Q"), (BLACK_KINGSIDE_INDEX, "k"), (BLACK_QUEENSIDE_INDEX, "q"))
    castling = "".join(letter for index, letter in castling_slots if encoding[index] == CASTLING_TOKEN_OFFSET + 1)
    board.set_castling_fen(castling or "-")

    for square in range(NUM_SQUARES):
        board.set_piece_at(square, integer_to_piece(encoding[square]))

    if encoding[TURN_INDEX] == TURN_TOKEN_OFFSET:
        board = board.mirror() #also hands the turn to black

    return board

#turn is the side playing the move: black's moves are mirrored to match encode_board
def encode_move_to_id(move: chess.Move, turn: chess.Color) -> int:
    flip = 0 if turn == chess.WHITE else 56
    return MOVE_IDS_BY_SQUARES[move.from_square ^ flip, move.to_square ^ flip, move.promotion]

def decode_id_to_move(move_id: int, turn: chess.Color) -> chess.Move:
    move = ALL_POSSIBLE_MOVES[move_id]
    if turn == chess.BLACK:
        return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)
    return move

def integer_to_piece(piece: int) -> chess.Piece | None:
    if piece == 0: return None
    if piece <= NUM_PIECE_TYPES: return chess.Piece(piece, chess.WHITE)
    return chess.Piece(piece - NUM_PIECE_TYPES, chess.BLACK)
