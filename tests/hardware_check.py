#a check of the engine on real hardware in a few minutes, before it plays there: the real network on every visible gpu searches a middlegame
#and an endgame for a fixed time each, in the shared tree (concurrent descents) as in a game on several gpus, and the search's numbers are
#printed. With the extension built with its `audit` feature the tree each search leaves is audited too (search/shared/audit.rs: the references
#and the arenas, the counts of what waits, what the moves hold against their nodes), which stops the run with what it found. On a linux pod,
#from the repository's root:
#  MATURIN_PEP517_ARGS="--features audit" uv sync --frozen --inexact --reinstall-package selfplay-rs
#  uv run --no-sync python tests/hardware_check.py [model.pt] [--seconds 60] [--gpus all] [--memory-mb 0] [--search-settings name=value,...]
#  uv sync --frozen --inexact --reinstall-package selfplay-rs       (the engine's own build again, without the audit)
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
from engine.engine_search import ENGINE_FP16_BLOCKS, ENGINE_PRECISION, GAME, LEAVES, PRECISIONS, EngineSearch, engine_search_settings, parse_changes
from engine.uci import build_player, gpus_argument, load_networks
from model.checkpoint import latest_checkpoint

#a middlegame with every piece on, and a rook ending: few transpositions and many
POSITIONS = (("middlegame", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8"),
             ("endgame", "8/R4pk1/6p1/7p/7P/6P1/1r3PK1/8 w - - 0 40"))


#searches each position for `seconds` on the player's networks from a tree of its own, and gives `out` a line of numbers for each: the
#evaluations and visits a second, the leaves a round (of Searches' own round loop, which real networks run in), and with concurrent descents
#the tree's own numbers (Searches.tree_numbers) and, if the extension has it, its audit. Returns the numbers, a dict a position
def check(player: EngineSearch, seconds: float, positions=POSITIONS, out=print) -> list[dict]:
    results = []
    for name, fen in positions:
        player.new_game()
        batches, boards = player.rust_counts
        started = time.perf_counter()
        report = player.search(chess.Board(fen), seconds=seconds)
        elapsed = time.perf_counter() - started
        batches, boards = player.rust_counts[0] - batches, player.rust_counts[1] - boards
        numbers = {"position": name, "seconds": elapsed, "visits": report.simulations, "evaluations": report.evaluations,
                   "visits_per_second": report.simulations / elapsed, "evaluations_per_second": report.evaluations / elapsed,
                   "leaves_a_round": boards / batches if batches else None, "move": report.line[0].uci(), "score": report.score}
        if player.concurrent_descents:
            numbers.update(player.searches.tree_numbers(GAME))
            if hasattr(player.searches, "audit"):
                numbers["audited_nodes"], numbers["audited_moves"] = player.searches.audit(GAME)
        results.append(numbers)
        out(line(numbers))
    return results


def line(numbers: dict) -> str:
    leaves = "no count" if numbers["leaves_a_round"] is None else f"{numbers['leaves_a_round']:.1f}"
    text = (f"{numbers['position']}: {numbers['seconds']:.1f} s, {numbers['evaluations_per_second']:,.0f} evaluations/s, "
            f"{numbers['visits_per_second']:,.0f} visits/s, {numbers['visits']:,} visits, {leaves} leaves a round, "
            f"move {numbers['move']} score {numbers['score']:.3f}")
    if "allocated_bytes" in numbers:
        text += (f"; tree {numbers['allocated_bytes'] / 1e6:,.0f} MB in use ({numbers['allocated_bytes'] / max(numbers['visits'], 1):.0f} bytes a "
                 f"visit), arenas {numbers['footprint_bytes'] / 1e6:,.0f} MB; {numbers['collisions']:,} collisions "
                 f"({numbers['collisions'] / max(numbers['evaluations'], 1):.2f} an evaluation), gave way {numbers['gave_way']:,}, "
                 f"alone in the tree {numbers['exclusive']:,}, nodes made twice {numbers['orphans']:,}, freed for room {numbers['assists']:,}")
    if "audited_nodes" in numbers:
        text += f"; audit passed on {numbers['audited_nodes']:,} nodes and {numbers['audited_moves']:,} moves"
    elif "allocated_bytes" in numbers:
        text += "; no audit (the extension is built without --features audit)"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the engine's speed and its tree's audit on this machine's gpus")
    parser.add_argument("checkpoint", nargs="?", help="a model.pt; default the run that finished last")
    parser.add_argument("--seconds", type=float, default=60.0, help="a search's time for each of the two positions, default %(default)s")
    parser.add_argument("--gpus", type=gpus_argument, default=None, help="cuda devices as uci.py's --gpus; default all")
    parser.add_argument("--leaves", type=int, default=LEAVES, help="the most leaves a round, default %(default)s")
    parser.add_argument("--precision", choices=PRECISIONS, default=ENGINE_PRECISION)
    parser.add_argument("--memory-mb", type=int, default=0, help="the tree's limit (RamLimitMb), default none")
    parser.add_argument("--search-settings", default="", help="changes to the search's settings, as uci.py's")
    arguments = parser.parse_args()
    gpus = gpus_argument("all") if arguments.gpus is None else arguments.gpus
    if not gpus:
        parser.error("there is no cuda device: this check is for the machine the engine will play on")
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else latest_checkpoint()
    networks = load_networks(checkpoint, True, arguments.leaves, arguments.precision, ENGINE_FP16_BLOCKS, gpus)
    #the whole time is searched (no smart pruning), in the shared tree whatever the number of gpus
    options = {"smart_pruning": False, "concurrent_descents": True, "memory_limit_mb": arguments.memory_mb,
               "settings": parse_changes(arguments.search_settings, engine_search_settings(False))}
    check(build_player(networks, False, options), arguments.seconds)
