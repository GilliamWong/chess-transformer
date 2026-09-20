#plays the trained model against an opponent and reports the result.
#
#run from the repository root:
#   python -m play.evaluate_elo                    a match against stockfish limited to 1320 elo
#   python -m play.evaluate_elo --elo 1500         the same against a stronger setting
#   python -m play.evaluate_elo --opponent random  against random moves, a trained model should win nearly all of these
#   python -m play.evaluate_elo --opponent human   opens the cute chess gui so you can play it yourself
#
#the games are run by cutechess-cli rather than by our own loop. It already handles the things
#a match needs: alternating colours, a different opening for every pair of games, the draw rules,
#running several games at once, saving the pgns, and the elo difference with its error margin.
#The model talks to it through play/uci.py.
import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from data import CARLSEN_PGN
from play.download_engines import CUTECHESS_CLI, CUTECHESS_DIR, CUTECHESS_GUI, STOCKFISH, PROJECT_DIR
from main import CHECKPOINT_PATH


MATCH_PGN = PROJECT_DIR / "match.pgn" #every game of the last match, worth reading to see how it loses

MIN_STOCKFISH_ELO = 1320 #the range UCI_Elo accepts in stockfish 19
MAX_STOCKFISH_ELO = 3190

#kept fixed so that the numbers of different models can be compared with each other. UCI_Elo is
#calibrated at a longer time control than this, so treat the result as a yardstick, not a lichess rating
SECONDS_PER_MOVE = 0.1
OPENING_PLIES = 8 #both engines are close to deterministic, without different openings every game would be the same game
CONCURRENT_GAMES = 8
SEED = 42

def model_engine(checkpoint_path: Path) -> list[str]:
    return [
        "-engine", "name=ChessTransformer", f"cmd={sys.executable}", "arg=-m", "arg=play.uci", f"arg={checkpoint_path}",
        f"dir={PROJECT_DIR}", "proto=uci"
    ]

def stockfish_engine(elo: int) -> list[str]:
    return [
        "-engine", f"name=Stockfish{elo}", f"cmd={STOCKFISH}", "proto=uci",
        "option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}", "option.Threads=1"
    ]

def random_engine() -> list[str]:
    return ["-engine", "name=RandomMover", f"cmd={sys.executable}", "arg=-m", "arg=play.uci", "arg=random", f"dir={PROJECT_DIR}", "proto=uci"]

#runs the match and returns cutechess-cli's output
def play_match(checkpoint_path: Path, opponent: list[str], num_games: int) -> str:
    command = [
        str(CUTECHESS_CLI),
        *model_engine(checkpoint_path),
        *opponent,
        #timemargin because python answers a little later than a compiled engine would
        "-each", f"st={SECONDS_PER_MOVE}", "timemargin=2000",
        #the first moves of random carlsen games as openings, each one played twice with the colours swapped
        "-openings", f"file={CARLSEN_PGN}", "format=pgn", "order=random", f"plies={OPENING_PLIES}",
        "-srand", str(SEED),
        "-games", "2", "-rounds", str(num_games // 2), "-repeat",
        "-recover", #a crashed engine loses that game instead of ending the match
        "-concurrency", str(CONCURRENT_GAMES),
        #no -draw or -resign adjudication: those go by the engines' own evaluations, and the model has none.
        #repetition, the fifty move rule and stalemate still end games, this only stops a game that never would
        "-maxmoves", "300",
        "-pgnout", str(MATCH_PGN)
    ]

    #cutechess-cli appends to the pgn, so without this it would hold every match ever played
    MATCH_PGN.unlink(missing_ok=True)

    #stdout is streamed so the running score shows up as the games finish
    output_lines = []
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        if line.startswith("Score of") or line.startswith("Elo difference"):
            print(line.rstrip())
    process.wait()

    output = "".join(output_lines)
    if process.returncode != 0 or "Finished match" not in output:
        raise RuntimeError(f"cutechess-cli did not finish the match:\n{output[-2000:]}")

    return output

#wins, losses and draws from the last "Score of ChessTransformer vs ...: 12 - 80 - 8" line
def parse_score(output: str) -> tuple[int, int, int]:
    scores = re.findall(r"Score of ChessTransformer vs .*?: (\d+) - (\d+) - (\d+)", output)
    wins, losses, draws = scores[-1]
    return int(wins), int(losses), int(draws)

def elo_difference(score: float) -> float:
    return -400 * math.log10(1 / score - 1)

def report(wins: int, losses: int, draws: int, opponent_elo: int | None) -> None:
    num_games = wins + losses + draws
    score = (wins + 0.5 * draws) / num_games
    print(f"\n{wins} wins, {losses} losses, {draws} draws: {score:.1%} of the points over {num_games} games")

    if opponent_elo is None:
        return

    #the formula runs off to infinity at 0% and 100%, and a handful of points says very little anyway
    if score < 0.05:
        print(f"too one sided to measure, well below {opponent_elo}. Try a weaker opponent (--opponent random)")
    elif score > 0.95:
        print(f"too one sided to measure, well above {opponent_elo}. Try a higher --elo")
    else:
        print(f"estimated elo: {opponent_elo + elo_difference(score):.0f} on stockfish's UCI_Elo scale (cutechess's error margin is above)")

def play_human() -> None:
    print("opening cute chess. Game > New, set one side to Human and the other to CPU > ChessTransformer")
    subprocess.Popen([str(CUTECHESS_GUI)], cwd=CUTECHESS_DIR)

def main():
    parser = argparse.ArgumentParser(description="play the trained model against stockfish, random moves, or yourself")
    parser.add_argument("--opponent", choices=["stockfish", "random", "human"], default="stockfish")
    parser.add_argument("--elo", type=int, default=MIN_STOCKFISH_ELO, help="stockfish's UCI_Elo setting")
    parser.add_argument("--games", type=int, default=100, help="about +/-70 elo at 100 games, +/-35 at 400")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    arguments = parser.parse_args()

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(f"{arguments.checkpoint} does not exist, train a model with main.py first")
    for program in (STOCKFISH, CUTECHESS_CLI, CUTECHESS_GUI):
        if not program.exists():
            raise FileNotFoundError(f"{program} does not exist, run python -m play.download_engines first")
    if not MIN_STOCKFISH_ELO <= arguments.elo <= MAX_STOCKFISH_ELO:
        raise ValueError(f"--elo must be between {MIN_STOCKFISH_ELO} and {MAX_STOCKFISH_ELO}, got {arguments.elo}")

    if arguments.opponent == "human":
        play_human()
        return

    if arguments.opponent == "stockfish":
        opponent, opponent_elo = stockfish_engine(arguments.elo), arguments.elo
    else:
        opponent, opponent_elo = random_engine(), None

    output = play_match(arguments.checkpoint.resolve(), opponent, arguments.games)
    report(*parse_score(output), opponent_elo)
    print(f"games saved to {MATCH_PGN.name}")


if __name__ == "__main__":
    main()
