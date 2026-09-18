from data import CARLSEN_PGN, read_games
from encoding import decode_board, encode_board


def main():
    magnus_game = next(read_games(CARLSEN_PGN))

    board = magnus_game.board()

    encoded = encode_board(board)

    decoded = decode_board(encoded)


    print(decoded)

if __name__ == "__main__":
    main()
