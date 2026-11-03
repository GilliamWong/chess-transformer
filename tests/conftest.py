import chess.pgn
import pytest

from training.data import CARLSEN_PGN
from engine.tablebase import open_tablebase

NUM_PGN_GAMES_TO_REPLAY = 5


#the first few real games, shared by every test that replays them
@pytest.fixture(scope="session")
def carlsen_games() -> list[chess.pgn.Game]:
    games = []
    with open(CARLSEN_PGN) as pgn:
        while len(games) < NUM_PGN_GAMES_TO_REPLAY and (game := chess.pgn.read_game(pgn)) is not None:
            games.append(game)

    assert len(games) == NUM_PGN_GAMES_TO_REPLAY
    return games


#the syzygy tables of up to 5 pieces; a test that needs them is skipped without them
@pytest.fixture(scope="session")
def tablebase():
    try:
        tables = open_tablebase()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    yield tables
    tables.close()
