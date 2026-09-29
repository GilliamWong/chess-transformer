#tools/play/variant_match.py: two variants of the engine's search play out every opening with both colours, on a
#number of nodes a move or a clock in nodes, and the elo of their pairs of games
import chess
import pytest

from fake_models import BoardModel
from inference import cpu_batches
from tools.play.variant_match import Variant, a_points, pair_elo, play, variant_settings

OPENINGS = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "8/5pk1/6p1/8/8/5PK1/6P1/2R5 b - - 0 1"]

def played(changes: tuple[str, str], nodes: int | None = None, clock: tuple[float, float] | None = None, node_rate: float = 0.0):
    variants = [Variant(name, variant_settings(change, False)) for name, change in zip("ab", changes)]
    games = [(2 * index + colour, opening, colour == 0) for index, opening in enumerate(OPENINGS) for colour in (0, 1)]
    finished = []
    play(cpu_batches(BoardModel()), variants, games, 3, 4, nodes, clock, node_rate, None, lambda record, game: finished.append((record, game)))
    return finished


#every opening with a as white and then as black, the variants moving in turn, every game to its end, and each move
#searched for its nodes (a search with one legal move stops at once)
def test_every_opening_is_played_out_with_both_colours():
    finished = played(("smart_pruning=0", "smart_pruning=0,policy_temperature=1.5"), nodes=30)

    assert sorted(record["game"] for record, _ in finished) == [0, 1, 2, 3]
    for record, game in finished:
        assert game.a_white == (record["game"] % 2 == 0) and game.opening == OPENINGS[record["game"] // 2]
        board = chess.Board(game.opening)
        for ply in game.plies:
            assert ply["variant"] == (0 if (board.turn == chess.WHITE) == game.a_white else 1)
            assert ply["nodes"] >= 30 or board.legal_moves.count() == 1
            board.push_uci(ply["move"])
        assert record["result"] in ("1-0", "0-1", "1/2-1/2") and record["plies"] == len(game.plies)


#on a clock the moves get a share of the time left, in nodes at node_rate a second: every move searches, and neither
#variant spends more nodes than its clock gave it
def test_a_clock_in_nodes_is_shared_over_the_moves():
    base, increment, node_rate = 3.0, 0.1, 50
    finished = played(("", ""), clock=(base, increment), node_rate=node_rate)

    for record, game in finished:
        assert all(seconds > 0 for seconds in record["seconds_left"])
        for variant in (0, 1):
            searched = [ply["nodes"] for ply in game.plies if ply["variant"] == variant]
            assert min(searched) > 0 and sum(searched) < node_rate * (base + increment * len(searched))


#a's points from both colours: 19 pairs of 1.5 of 2 and one of 0.5, a score of 0.7375 or 179 elo within its interval
def test_pair_elo():
    records = [{"game": game, "a_white": game % 2 == 0, "result": "1-0" if game % 2 == 0 else "1/2-1/2"} for game in range(40)]
    assert [a_points(record) for record in records[:2]] == [1.0, 0.5]

    elo, low, high, pairs = pair_elo(records[:-1] + [{**records[-1], "result": "1-0"}])
    assert pairs == 20 and low < elo < high and elo == pytest.approx(179.5, abs=0.5)


def test_variant_settings():
    settings = variant_settings("c_puct=1.2,certainty=1,smart_pruning=0", False)
    assert settings["c_puct"] == 1.2 and settings["certainty"] is True and settings["smart_pruning"] == 0.0
    with pytest.raises(ValueError):
        variant_settings("cpuct=1.2", False)
