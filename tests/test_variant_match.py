#tools/play/variant_match.py: two variants of the engine's search play out every opening with both colours, on a
#number of nodes a move or a clock in nodes, and the elo of their pairs of games
import math

import chess
import pytest

from fake_models import BoardModel
from inference import cpu_batches
import tools.play.engine_search as engine_search
from tools.play.variant_match import SPRT_BOUND, Variant, a_points, pair_elo, play, sprt_llr, variant_settings

OPENINGS = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "8/5pk1/6p1/8/8/5PK1/6P1/2R5 b - - 0 1"]


#the variants here change mcts.py's search, whose constants the expectations were set by
@pytest.fixture(autouse=True)
def mcts_constants(monkeypatch):
    monkeypatch.setattr(engine_search, "ENGINE_SETTINGS", {})

def played(changes: tuple[str, str], nodes: int | None = None, clock: tuple[float, float] | None = None, node_rate: float = 0.0):
    variants = [Variant(name, variant_settings(change, False)) for name, change in zip("ab", changes)]
    games = [(2 * index + colour, opening, colour == 0) for index, opening in enumerate(OPENINGS) for colour in (0, 1)]
    finished = []
    play(cpu_batches(BoardModel()), variants, games, 3, 4, nodes, clock, node_rate, None, lambda record, game: finished.append((record, game)))
    return finished


#every opening with a as white and then as black, the variants moving in turn, every game to its end, and each move
#searched for its nodes, a's twice as many (a search with one legal move stops at once)
def test_every_opening_is_played_out_with_both_colours():
    finished = played(("smart_pruning=0,node_scale=2", "smart_pruning=0,policy_temperature=1.5"), nodes=30)

    assert sorted(record["game"] for record, _ in finished) == [0, 1, 2, 3]
    for record, game in finished:
        assert game.a_white == (record["game"] % 2 == 0) and game.opening == OPENINGS[record["game"] // 2]
        board = chess.Board(game.opening)
        for ply in game.plies:
            assert ply["variant"] == (0 if (board.turn == chess.WHITE) == game.a_white else 1)
            assert ply["nodes"] >= 30 * (2 if ply["variant"] == 0 else 1) or board.legal_moves.count() == 1
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


#engine_search.TimeUse on the clock: a variant that stops at half its planned nodes when its search is steady spends
#fewer nodes a move than one that keeps to the plan
def test_time_use_on_a_clock_in_nodes():
    finished = played(("smart_pruning=0,time_steady=0.5", "smart_pruning=0"), clock=(3.0, 0.1), node_rate=50)

    searched = [sum(ply["nodes"] for _, game in finished for ply in game.plies if ply["variant"] == variant) for variant in (0, 1)]
    moves = [sum(ply["variant"] == variant for _, game in finished for ply in game.plies) for variant in (0, 1)]
    assert searched[0] / moves[0] < 0.8 * searched[1] / moves[1]


#a's points from both colours: 19 pairs of 1.5 of 2 and one of 0.5, a score of 0.7375 or 179 elo within its interval
def test_pair_elo():
    records = [{"game": game, "a_white": game % 2 == 0, "result": "1-0" if game % 2 == 0 else "1/2-1/2"} for game in range(40)]
    assert [a_points(record) for record in records[:2]] == [1.0, 0.5]

    elo, low, high, pairs = pair_elo(records[:-1] + [{**records[-1], "result": "1-0"}])
    assert pairs == 20 and low < elo < high and elo == pytest.approx(179.5, abs=0.5)


#records whose pairs score these, a as white first
def records_scoring(pair_points: list[float]) -> list[dict]:
    results = {1.0: ("1-0", "0-1"), 0.75: ("1-0", "1/2-1/2"), 0.5: ("1/2-1/2", "1/2-1/2"), 0.25: ("0-1", "1/2-1/2"), 0.0: ("0-1", "1-0")}
    return [{"game": 2 * pair + colour, "a_white": colour == 0, "result": results[points][colour]}
            for pair, points in enumerate(pair_points) for colour in (0, 1)]

#the SPRT's log likelihood ratio is that of two normal distributions of the pair scores with the pairs' own variance,
#centred on elo0's and elo1's scores: positive when a scores well above both, negative near elo0's
def test_sprt_llr():
    pair_points = [0.75, 0.5, 0.5, 1.0, 0.25, 0.75, 0.5, 0.75] * 8
    elo0, elo1 = 0.0, 15.0
    mean = sum(pair_points) / len(pair_points)
    variance = sum((points - mean) ** 2 for points in pair_points) / len(pair_points)
    score0, score1 = 1 / (1 + 10 ** (-elo0 / 400)), 1 / (1 + 10 ** (-elo1 / 400))
    log_density = lambda points, centre: -(points - centre) ** 2 / (2 * variance) - 0.5 * math.log(2 * math.pi * variance)
    expected = sum(log_density(points, score1) - log_density(points, score0) for points in pair_points)

    assert sprt_llr(records_scoring(pair_points), elo0, elo1) == pytest.approx(expected)
    assert sprt_llr(records_scoring(pair_points), elo0, elo1) > SPRT_BOUND
    assert sprt_llr(records_scoring([0.5, 0.25, 0.75, 0.5] * 150), elo0, elo1) < -SPRT_BOUND
    assert sprt_llr(records_scoring([0.5]), elo0, elo1) == 0.0


#on_result returning True stops the match: the first game to end is the only one recorded
def test_a_match_stops_when_on_result_says_so():
    variants = [Variant(name, variant_settings("", False)) for name in "ab"]
    games = [(2 * index + colour, opening, colour == 0) for index, opening in enumerate(OPENINGS) for colour in (0, 1)]
    finished = []
    play(cpu_batches(BoardModel()), variants, games, 3, 4, 20, None, 0.0, None, lambda record, game: finished.append(record) or True)
    assert len(finished) == 1


def test_variant_settings():
    settings = variant_settings("c_puct=1.2,certainty=1,smart_pruning=0", False)
    assert settings["c_puct"] == 1.2 and settings["certainty"] is True and settings["smart_pruning"] == 0.0
    with pytest.raises(ValueError):
        variant_settings("cpuct=1.2", False)
