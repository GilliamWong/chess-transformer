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

ENCODING_SIZE = 70 #for reference, not used

NUM_PROMOTION_CHOICES = 5 #no promotion, knight, bishop, rook, queen
NUM_MOVE_IDS = NUM_SQUARES * NUM_SQUARES * NUM_PROMOTION_CHOICES #20480, the size of the move vocabulary

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

    encoding.append(1 if board.turn == chess.WHITE else 0) #64
    encoding.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0) #65
    encoding.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0) #66

    encoding.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0) #67
    encoding.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0) #68

    encoding.append(board.ep_square if board.ep_square is not None and board.has_legal_en_passant() else 0) #69

    return encoding

def decode_board(encoding: list[int]) -> chess.Board:
    board = chess.Board.empty()

    board.turn = (chess.WHITE if encoding[TURN_INDEX] == 1 else chess.BLACK)
    board.ep_square = (None if encoding[EN_PASSANT_INDEX] == 0 else encoding[EN_PASSANT_INDEX])

    castling = ""
    if encoding[WHITE_KINGSIDE_INDEX] == 1: castling += "K"
    if encoding[WHITE_QUEENSIDE_INDEX] == 1: castling += "Q"
    if encoding[BLACK_KINGSIDE_INDEX] == 1: castling += "k"
    if encoding[BLACK_QUEENSIDE_INDEX] == 1: castling += "q"
    if castling == "": castling = "-"

    board.set_castling_fen(castling)


    for square in range(NUM_SQUARES):
        board.set_piece_at(square, integer_to_piece(encoding[square]))

    return board

def encode_move_to_id(move: chess.Move) -> int:
    promotion = move.promotion - 1 if move.promotion is not None else 0
    id = ((move.from_square * NUM_SQUARES) + move.to_square) * NUM_PROMOTION_CHOICES + promotion
    return id


def decode_id_to_move(move: int) -> chess.Move:
    promotion = move % NUM_PROMOTION_CHOICES
    destination = ((move - promotion) // NUM_PROMOTION_CHOICES) % NUM_SQUARES
    source = (((move - promotion) // NUM_PROMOTION_CHOICES) // NUM_SQUARES)

    promotion = None if promotion == 0 else promotion + 1

    return chess.Move(source, destination, promotion=promotion)

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