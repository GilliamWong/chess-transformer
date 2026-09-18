from collections.abc import Iterator
from pathlib import Path
from dataclasses import dataclass

import chess.pgn

CARLSEN_PGN = Path(__file__).parent / "Carlsen" / "Carlsen.pgn"

def read_games(pgn_path: Path) -> Iterator[chess.pgn.Game]:
    with open(pgn_path) as pgn:
        #read_game returns None once the file is exhausted
        while (game := chess.pgn.read_game(pgn)) is not None:
            yield game
