#monte carlo tree search over the model: the policy says which moves are worth a look, the eval head scores the positions
#the search reaches, and the move searched the most is played.
#
#run from the repository root to watch one search:
#   python mcts.py artifacts/runs/rl_start_d128_4min/model.pt
#   python mcts.py artifacts/runs/rl_start_d128_4min/model.pt --fen "<fen>" --simulations 800 --cpu
#
#a score is an expected score, 0 lost to 1 won, like the eval head's
import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
import chess
import chess.syzygy
import torch
from torch import nn
from encoding import encode_board, encode_move_to_id
from tools.checkpoint import latest_checkpoint
from tools.play.player import load_model
from tools.tablebase import covers, open_tablebase, ranked_moves, wdl_score

SIMULATIONS = 200
DRAW = 0.5
#the model's board has no history, so only the search can see a repetition coming
REPETITIONS_FOR_A_DRAW = 3
#leela's search constants (lc0's defaults), halved where they scale a score: its scores run from -1 to 1, ours from 0 to 1
C_PUCT = 1.745 / 2
#c_puct grows slowly with the parent's visits, by C_PUCT_FACTOR * log((visits + C_PUCT_BASE) / C_PUCT_BASE)
C_PUCT_FACTOR = 3.894 / 2
C_PUCT_BASE = 38739
FPU_REDUCTION = 0.33 / 2

#the endgame tablebase once main loads one (--tablebase). game_result scores the positions it covers, so the search stops
#there as it does at the end of a game
tablebase: chess.syzygy.Tablebase | None = None


@dataclass
class Node:
    prior: float #the policy's probability of the move that leads here
    visit_count: int = 0
    #the scores of all visits added up, for the side that played the move into this node: a parent compares its children
    #by what they are worth to its own side to move
    value_sum: float = 0.0
    #empty until the node is expanded
    children: dict[chess.Move, "Node"] = field(default_factory=dict)


#the network's view of a position that is not over: the prior of every legal move (adding up to 1), and the score of the
#side to move
def evaluate(model: nn.Module, board: chess.Board, device: torch.device) -> tuple[dict[chess.Move, float], float]:
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        raise ValueError(f"the game is over here, game_result scores it: {board.fen()}")

    legal_move_ids = [encode_move_to_id(move, board.turn) for move in legal_moves]
    boards = torch.tensor(encode_board(board), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        move_logits, value_logits = model(boards)

    #a softmax over the legal moves alone, the same as setting every illegal one to -inf first
    priors = torch.softmax(move_logits[0, legal_move_ids], dim=0).tolist()
    value = torch.sigmoid(value_logits[0, 0]).item()

    return dict(zip(legal_moves, priors)), value

#the score of the side to move if the game is over here or the tablebase knows how it ends, None while it goes on.
#Checkmate is checked first because a mate on the fifty move limit still wins
def game_result(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return 0.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(REPETITIONS_FOR_A_DRAW):
        return DRAW
    if tablebase is not None and covers(board):
        return wdl_score(tablebase.probe_wdl(board))
    return None


#the child to walk into next, by PUCT as leela does it. A child that was never visited has no mean score yet, so it gets its
#parent's for the side to move there, reduced more the more of the policy has been visited already (first play urgency)
def select_child(node: Node) -> tuple[chess.Move, Node]:
    c_puct = C_PUCT + C_PUCT_FACTOR * math.log((node.visit_count + C_PUCT_BASE) / C_PUCT_BASE)
    #the node's first visit evaluated it and every later one went on to a child. At least 1, so the first pick follows the priors
    children_visits = max(node.visit_count - 1, 1)
    visited_policy = sum(child.prior for child in node.children.values() if child.visit_count > 0)
    first_play_urgency = (1 - node.value_sum / node.visit_count) - FPU_REDUCTION * math.sqrt(visited_policy)

    children_U = []
    for move, child in node.children.items():
        P = child.prior
        N = child.visit_count
        if N > 0:
            Q = child.value_sum / N
        else:
            Q = first_play_urgency
        children_U.append((move, child, Q + (c_puct * P * (math.sqrt(children_visits) / (1 + N)))))

    best_move, best_child, _ = max(children_U, key=lambda item: item[2])
    return best_move, best_child


def expand(node: Node, priors: dict[chess.Move, float]) -> None:
    for move, prior in priors.items():
        node.children[move] = Node(prior=prior)

#path runs from the root down. value is the score of the side to move at the last node, and each node keeps its score for
#the side that played the move into it (see Node), so the score flips at every level up
def backup(path: list[Node], value: float) -> None:
    score = 1 - value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += score
        score = 1 - score

#one simulation: walk down to a node without children, score its position and back the score up. board starts at the root
#position and is a copy of its own, so push moves freely
def run_simulation(root: Node, board: chess.Board, model: nn.Module, device: torch.device) -> None:
    node = root
    path = [root]
    #the positions on the path, root first, keyed the way is_repetition compares them
    positions = [board._transposition_key()]

    while node.children:
        move, node = select_child(node)
        board.push(move)
        path.append(node)
        positions.append(board._transposition_key())

    value = game_result(board)
    if value is None:
        #a position already on the path came round because neither side left the loop, so like leela it is scored as the
        #draw the loop heads for. A repeat of the game before the root still waits for the third time (game_result)
        if positions[-1] in positions[:-1]:
            value = DRAW
        else:
            priors, value = evaluate(model, board, device)
            expand(node, priors)

    backup(path, value)


#runs the simulations from board, which is left as it was, and returns the root: its children hold what the search found
def search(model: nn.Module, board: chess.Board, device: torch.device, simulations: int = SIMULATIONS) -> Node:
    if game_result(board) is not None:
        raise ValueError(f"the game is over or the tablebase knows it, there is nothing to search: {board.fen()}")

    root = Node(prior=1.0)
    for _ in range(simulations):
        #copied with the moves played so far, which is how game_result sees repetitions
        run_simulation(root, board.copy(), model, device)

    return root

#the move searched the most. Its visit count is steadier than its mean score, which a few lucky visits can lift. Ties go to
#the better mean score, then to the policy (leela's order), so a search too short to visit anything plays the policy's move
def most_visited_move(root: Node) -> chess.Move:
    def preference(move: chess.Move) -> tuple[int, float, float]:
        child = root.children[move]
        mean_score = child.value_sum / child.visit_count if child.visit_count > 0 else 0.0
        return child.visit_count, mean_score, child.prior

    return max(root.children, key=preference)

#the line the search expects: its move, then the most visited reply, and so on for as long as the next move was visited
def principal_variation(root: Node) -> list[chess.Move]:
    line, node = [], root
    while node.children:
        move = most_visited_move(node)
        if line and node.children[move].visit_count == 0:
            break
        line.append(move)
        node = node.children[move]
    return line


#the root's moves, the most searched first, with the mean score each has for the side to move at the root
def print_search(root: Node, board: chess.Board, top: int = 10) -> None:
    print(f"{'move':<8}{'prior':>8}{'visits':>8}{'score':>8}")
    for move, child in sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)[:top]:
        score = f"{child.value_sum / child.visit_count:.3f}" if child.visit_count > 0 else "-"
        print(f"{board.san(move):<8}{child.prior:>8.1%}{child.visit_count:>8}{score:>8}")

def main(checkpoint_path: Path, fen: str, simulations: int, device: torch.device, use_tablebase: bool = False) -> None:
    global tablebase
    if use_tablebase:
        tablebase = open_tablebase()

    model = load_model(checkpoint_path, device)
    board = chess.Board(fen)
    print(board, "\n")

    #also warms the model up, so the first call's setup is not timed with the search
    priors, value = evaluate(model, board, device)
    favourite = max(priors, key=lambda move: priors[move])
    side = "white" if board.turn == chess.WHITE else "black"
    print(f"network alone: score {value:.3f} for {side}, favourite move {board.san(favourite)} ({priors[favourite]:.1%})")

    if tablebase is not None and covers(board):
        best = ranked_moves(tablebase, board)[0]
        print(f"tablebase: score {wdl_score(best.wdl)} for {side}, plays {board.san(best.move)}, nothing to search")
        return

    start = time.perf_counter()
    root = search(model, board, device, simulations)
    seconds = time.perf_counter() - start
    print(f"search: {simulations} simulations in {seconds:.2f} s ({simulations / seconds:.0f} a second), plays {board.san(most_visited_move(root))}\n")
    print_search(root, board)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="one search from a position, to watch what it does")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="a model.pt, default the run that finished last")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="the position to search, default the start")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS, help="default %(default)s")
    parser.add_argument("--cpu", action="store_true", help="leave the gpu alone, for when a training run is using it")
    parser.add_argument("--tablebase", action="store_true", help="score positions of at most 5 pieces from the syzygy tables (python -m tools.tablebase)")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    main(arguments.checkpoint or latest_checkpoint(), arguments.fen, arguments.simulations, device, arguments.tablebase)
