import chess

#constants
NUM_SQUARES = 64
NUM_PIECE_TYPES = 6

TURN_INDEX = 64
WHITE_KINGSIDE_INDEX = 65
WHITE_QUEENSIDE_INDEX = 66
BLACK_KINGSIDE_INDEX = 67
BLACK_QUEENSIDE_INDEX = 68
EN_PASSANT_INDEX = 69

TOKENS_PER_BOARD = 70

#each slot kind gets its own block of token ids, so a raw value never shares an embedding row across slots.
#encode_board adds the offsets
NUM_PIECE_CODES = 1 + 2 * NUM_PIECE_TYPES #13, empty + 6 white + 6 black
NUM_FLAG_VALUES = 2
NUM_EN_PASSANT_VALUES = NUM_SQUARES #0 for none, otherwise the square

PIECE_TOKEN_OFFSET = 0
TURN_TOKEN_OFFSET = PIECE_TOKEN_OFFSET + NUM_PIECE_CODES #13
CASTLING_TOKEN_OFFSET = TURN_TOKEN_OFFSET + NUM_FLAG_VALUES #15, shared by all four castling slots
EN_PASSANT_TOKEN_OFFSET = CASTLING_TOKEN_OFFSET + NUM_FLAG_VALUES #17

BOARD_VOCAB_SIZE = EN_PASSANT_TOKEN_OFFSET + NUM_EN_PASSANT_VALUES #81

def _build_all_possible_moves() -> tuple[chess.Move, ...]:
    board = chess.Board.empty()
    moves = []
    queen = chess.Piece(chess.QUEEN, chess.WHITE)
    knight = chess.Piece(chess.KNIGHT, chess.WHITE)

    for square in range(NUM_SQUARES):
        board.set_piece_at(square, queen)
        move_set = board.attacks(square)
        board.set_piece_at(square, knight)
        move_set = move_set.union(board.attacks(square))

        board.remove_piece_at(square)

        moves.extend(chess.Move(square, destination) for destination in move_set)

    #promotions
    for from_rank, to_rank in [(6, 7), (1, 0)]:
        for from_file in range(8):
            for shift in (-1, 0, 1):
                to_file = from_file + shift
                if to_file not in range(8):
                    continue
                from_square = chess.square(from_file, from_rank)
                to_square = chess.square(to_file, to_rank)

                for piece in (chess.KNIGHT, chess.ROOK, chess.BISHOP, chess.QUEEN):
                    moves.append(chess.Move(from_square, to_square, promotion=piece))


    return tuple(moves)

ALL_POSSIBLE_MOVES = _build_all_possible_moves()
NUM_MOVE_IDS = len(ALL_POSSIBLE_MOVES) #1968

def _build_move_dict() -> dict[chess.Move, int]:
    output = {}
    for move_id, move in enumerate(ALL_POSSIBLE_MOVES):
        output[move] = move_id

    return output

MOVE_TO_ID = _build_move_dict()
#the same ids keyed by (from, to, promotion), for encode_move_to_id
MOVE_IDS_BY_SQUARES = {(move.from_square, move.to_square, move.promotion): move_id for move, move_id in MOVE_TO_ID.items()}

MOVE_FROM_SQUARES = tuple(move.from_square for move in ALL_POSSIBLE_MOVES)
MOVE_TO_SQUARES = tuple(move.to_square for move in ALL_POSSIBLE_MOVES)

#the 176 promotions: move id, landing square, and piece as 0 knight, 1 bishop, 2 rook, 3 queen
PROMOTION_MOVE_IDS = tuple(move_id for move_id, move in enumerate(ALL_POSSIBLE_MOVES) if move.promotion is not None)
PROMOTION_TO_SQUARES = tuple(ALL_POSSIBLE_MOVES[move_id].to_square for move_id in PROMOTION_MOVE_IDS)
PROMOTION_PIECES = tuple(ALL_POSSIBLE_MOVES[move_id].promotion - chess.KNIGHT for move_id in PROMOTION_MOVE_IDS)


#the board is encoded from the side to move: with black to move it is mirrored (ranks flipped, colours swapped),
#so "white" means the side to move. The turn token keeps the real colour so decode_board can mirror back.
#square ^ 56 flips the rank; board.mirror() would copy the whole move stack, too slow for the search
def encode_board(board: chess.Board) -> list[int]:
    turn = board.turn
    opponent = not turn
    flip = 0 if turn == chess.WHITE else 56

    encoding = [0 for _ in range(NUM_SQUARES)] #0 is an empty square

    #the side to move's pieces are 1-6, the other side's 7-12
    for piece_type in chess.PIECE_TYPES:
        for square in chess.scan_forward(board.pieces_mask(piece_type, turn)):
            encoding[square ^ flip] = piece_type
        for square in chess.scan_forward(board.pieces_mask(piece_type, opponent)):
            encoding[square ^ flip] = piece_type + NUM_PIECE_TYPES

    encoding.append(TURN_TOKEN_OFFSET + (1 if turn == chess.WHITE else 0)) #64
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(turn) else 0)) #65
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(turn) else 0)) #66

    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(opponent) else 0)) #67
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(opponent) else 0)) #68

    encoding.append(EN_PASSANT_TOKEN_OFFSET + (board.ep_square ^ flip if board.ep_square is not None and board.has_legal_en_passant() else 0)) #69

    return encoding

def decode_board(encoding: list[int]) -> chess.Board:
    board = chess.Board.empty() #white to move, as every board is encoded

    ep_square = encoding[EN_PASSANT_INDEX] - EN_PASSANT_TOKEN_OFFSET
    board.ep_square = (None if ep_square == 0 else ep_square)

    castling = ""
    if encoding[WHITE_KINGSIDE_INDEX] == CASTLING_TOKEN_OFFSET + 1: castling += "K"
    if encoding[WHITE_QUEENSIDE_INDEX] == CASTLING_TOKEN_OFFSET + 1: castling += "Q"
    if encoding[BLACK_KINGSIDE_INDEX] == CASTLING_TOKEN_OFFSET + 1: castling += "k"
    if encoding[BLACK_QUEENSIDE_INDEX] == CASTLING_TOKEN_OFFSET + 1: castling += "q"
    if castling == "": castling = "-"

    board.set_castling_fen(castling)


    for square in range(NUM_SQUARES):
        board.set_piece_at(square, integer_to_piece(encoding[square]))

    if encoding[TURN_INDEX] == TURN_TOKEN_OFFSET:
        board = board.mirror() #also hands the turn to black

    return board

def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)

#turn is the side playing the move: black's moves are mirrored to match encode_board
def encode_move_to_id(move: chess.Move, turn: chess.Color) -> int:
    flip = 0 if turn == chess.WHITE else 56
    return MOVE_IDS_BY_SQUARES[move.from_square ^ flip, move.to_square ^ flip, move.promotion]


def decode_id_to_move(move_id: int, turn: chess.Color) -> chess.Move:
    move = ALL_POSSIBLE_MOVES[move_id]
    return mirror_move(move) if turn == chess.BLACK else move

def integer_to_piece(piece: int) -> chess.Piece | None:
    if piece == 0: return None
    if piece <= NUM_PIECE_TYPES:
        return chess.Piece(piece, chess.WHITE)
    else:
        return chess.Piece(piece - NUM_PIECE_TYPES, chess.BLACK)
