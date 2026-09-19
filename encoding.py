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

TOKENS_PER_BOARD = 70 #length of an encoded board, each token is one of BOARD_VOCAB_SIZE values

#the same raw value means different things in different slots (1 is a white pawn on a square,
#white to move in the turn slot, a castling right, or square b1 in the en passant slot).
#each kind of value gets its own block of token ids so they never share an embedding row.
#encode_board adds the offsets, so an encoded board is already a list of token ids
NUM_PIECE_CODES = 1 + 2 * NUM_PIECE_TYPES #13, empty + 6 white + 6 black
NUM_FLAG_VALUES = 2 #0 or 1
NUM_EN_PASSANT_VALUES = NUM_SQUARES #0 for no en passant square, otherwise the square itself

PIECE_TOKEN_OFFSET = 0
TURN_TOKEN_OFFSET = PIECE_TOKEN_OFFSET + NUM_PIECE_CODES #13
CASTLING_TOKEN_OFFSET = TURN_TOKEN_OFFSET + NUM_FLAG_VALUES #15, shared by all four castling slots
EN_PASSANT_TOKEN_OFFSET = CASTLING_TOKEN_OFFSET + NUM_FLAG_VALUES #17

BOARD_VOCAB_SIZE = EN_PASSANT_TOKEN_OFFSET + NUM_EN_PASSANT_VALUES #81, the size of the board vocabulary

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
NUM_MOVE_IDS = len(ALL_POSSIBLE_MOVES) #1968, the size of the move vocabulary

def _build_move_dict() -> dict[chess.Move, int]:
    output = {}
    for move_id, move in enumerate(ALL_POSSIBLE_MOVES):
        output[move] = move_id

    return output

MOVE_TO_ID = _build_move_dict()


#need to encode a board state into a tensor of some sort
#we have 6 unique pieces, so 12 per side. 
#we also need to encode castling and en passant somehow along with whose turn it is
def encode_board(board: chess.Board) -> list[int]:
    encoding = [0 for _ in range(NUM_SQUARES)]

    for square in chess.SQUARES: #integers 0-63
        piece_encoding = 0 #we use 0 to denote an empty square
        piece  = board.piece_at(square)

        if piece is not None:
            if piece.color == chess.WHITE:
                piece_encoding = piece.piece_type
            else:
                piece_encoding = piece.piece_type + NUM_PIECE_TYPES

        encoding[square] = piece_encoding

    encoding.append(TURN_TOKEN_OFFSET + (1 if board.turn == chess.WHITE else 0)) #64
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)) #65
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(chess.WHITE) else 0)) #66

    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_kingside_castling_rights(chess.BLACK) else 0)) #67
    encoding.append(CASTLING_TOKEN_OFFSET + (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)) #68

    encoding.append(EN_PASSANT_TOKEN_OFFSET + (board.ep_square if board.ep_square is not None and board.has_legal_en_passant() else 0)) #69

    return encoding

def decode_board(encoding: list[int]) -> chess.Board:
    board = chess.Board.empty()

    board.turn = (chess.WHITE if encoding[TURN_INDEX] == TURN_TOKEN_OFFSET + 1 else chess.BLACK)

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

    return board

def encode_move_to_id(move: chess.Move) -> int:
    return MOVE_TO_ID[move]


def decode_id_to_move(move_id: int) -> chess.Move:
    return ALL_POSSIBLE_MOVES[move_id]

# 1  # pawn white
# 2  # knight white
# 3  # bishop white
# 4  # rook white
# 5  # queen white
# 6  # king white
def integer_to_piece(piece: int) -> chess.Piece | None:
    if piece == 0: return None
    if piece <= NUM_PIECE_TYPES:
        return chess.Piece(piece, chess.WHITE)
    else:
        return chess.Piece(piece - NUM_PIECE_TYPES, chess.BLACK)

def piece_to_integer(piece: chess.Piece | None) -> int:
    if piece is None: return 0
    output_piece = 0
    if piece.color == chess.WHITE:
        output_piece += piece.piece_type
    else:
        output_piece += piece.piece_type + NUM_PIECE_TYPES

    return output_piece