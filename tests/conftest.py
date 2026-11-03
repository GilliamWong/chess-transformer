import chess.pgn
import pytest

from engine.quantization import CARLSEN_PGN
from engine.tablebase import open_tablebase

NUM_PGN_GAMES_TO_REPLAY = 5


@pytest.fixture(scope="session")
def carlsen_games() -> list[chess.pgn.Game]:
    with open(CARLSEN_PGN) as pgn:
        games = [chess.pgn.read_game(pgn) for _ in range(NUM_PGN_GAMES_TO_REPLAY)]
    assert None not in games
    return games


@pytest.fixture(scope="session")
def tablebase():
    try:
        tables = open_tablebase()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    yield tables
    tables.close()
