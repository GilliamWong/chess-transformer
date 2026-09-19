import chess.pgn
import pytest

from data import CARLSEN_PGN

NUM_PGN_GAMES_TO_REPLAY = 5


@pytest.fixture(scope="session")
def carlsen_games() -> list[chess.pgn.Game]:
    """the first few real games, shared by every test that replays them"""
    games = []
    with open(CARLSEN_PGN) as pgn:
        #read_game returns None once the file is exhausted
        while len(games) < NUM_PGN_GAMES_TO_REPLAY and (game := chess.pgn.read_game(pgn)) is not None:
            games.append(game)

    assert len(games) == NUM_PGN_GAMES_TO_REPLAY
    return games
