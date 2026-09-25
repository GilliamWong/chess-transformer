#monte carlo tree search over the model: the policy gives the priors, the eval head scores the leaves.
#scores are expected scores, 0 lost to 1 won. Watch one search:
#   python mcts.py artifacts/runs/rl_start_d128_4min/model.pt --fen "<fen>" --simulations 800 --cpu
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING
import chess
import chess.syzygy
import numpy as np
from encoding import encode_board, encode_move_to_id
from tools.tablebase import covers, open_tablebase, ranked_moves, wdl_score
#self play's workers import this file for its constants and must not import torch
if TYPE_CHECKING:
    import torch
    from torch import nn

SIMULATIONS = 200
DRAW = 0.5
#the board has no history, so only the search can see a repetition coming
REPETITIONS_FOR_A_DRAW = 3
#lc0's defaults, halved where they scale a score: its scores run from -1 to 1, ours from 0 to 1
C_PUCT = 1.745 / 2
C_PUCT_FACTOR = 3.894 / 2
C_PUCT_BASE = 38739
FPU_REDUCTION = 0.33 / 2
#alphazero's dirichlet noise for chess, mixed into the root's priors in self play
ROOT_NOISE_ALPHA = 0.3
ROOT_NOISE_SHARE = 0.25

#set by main (--tablebase): game_result then scores the positions the tables cover
tablebase: chess.syzygy.Tablebase | None = None


#slots: a search makes thousands of nodes
@dataclass(slots=True)
class Node:
    prior: float
    visit_count: int = 0
    #for the side that played the move into this node
    value_sum: float = 0.0
    #empty until expanded
    children: dict[chess.Move, "Node"] = field(default_factory=dict)


#the priors of the legal moves and the score of the side to move
def evaluate(model: nn.Module, board: chess.Board, device: torch.device) -> tuple[dict[chess.Move, float], float]:
    return evaluate_many(model, [board], device)[0]

def evaluate_many(model: nn.Module, boards: list[chess.Board], device: torch.device) -> list[tuple[dict[chess.Move, float], float]]:
    import torch
    from model import expected_score
    legal_moves = [list(board.legal_moves) for board in boards]
    for board, moves in zip(boards, legal_moves):
        if len(moves) == 0:
            raise ValueError(f"the game is over here, game_result scores it: {board.fen()}")

    encoded = torch.tensor([encode_board(board) for board in boards], dtype=torch.long, device=device)
    with torch.no_grad():
        move_logits, value_logits = model(encoded)
    move_logits = move_logits.float().cpu().numpy()
    values = expected_score(value_logits).cpu().tolist()

    evaluations = []
    for board, moves, logits, value in zip(boards, legal_moves, move_logits, values):
        #softmax over the legal moves alone
        legal_logits = logits[[encode_move_to_id(move, board.turn) for move in moves]]
        weights = np.exp(legal_logits - legal_logits.max())
        evaluations.append((dict(zip(moves, (weights / weights.sum()).tolist())), value))
    return evaluations

#the side to move's score if the game is over or the tablebase knows it, else None.
#checkmate first: a mate on the fifty move limit still wins
def game_result(board: chess.Board) -> float | None:
    if not any(board.generate_legal_moves()):
        return 0.0 if board.is_check() else DRAW
    if board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(REPETITIONS_FOR_A_DRAW):
        return DRAW
    if tablebase is not None and covers(board):
        return wdl_score(tablebase.probe_wdl(board))
    return None


#PUCT as leela does it, with first play urgency for unvisited children
def select_child(node: Node) -> tuple[chess.Move, Node]:
    c_puct = C_PUCT + C_PUCT_FACTOR * math.log((node.visit_count + C_PUCT_BASE) / C_PUCT_BASE)
    #at least 1, so the first pick follows the priors
    children_visits = max(node.visit_count - 1, 1)
    visited_policy = sum(child.prior for child in node.children.values() if child.visit_count > 0)
    first_play_urgency = (1 - node.value_sum / node.visit_count) - FPU_REDUCTION * math.sqrt(visited_policy)
    sqrt_children_visits = math.sqrt(children_visits)

    #keeps the first child with the highest score, as max() would; this loop is the search's hot spot
    best_move, best_child, best_U = None, None, -math.inf
    for move, child in node.children.items():
        P = child.prior
        N = child.visit_count
        if N > 0:
            Q = child.value_sum / N
        else:
            Q = first_play_urgency
        U = Q + (c_puct * P * (sqrt_children_visits / (1 + N)))
        if U > best_U:
            best_move, best_child, best_U = move, child, U
    return best_move, best_child


def expand(node: Node, priors: dict[chess.Move, float]) -> None:
    for move, prior in priors.items():
        node.children[move] = Node(prior=prior)

#value is for the side to move at the last node; the score flips at every level up
def backup(path: list[Node], value: float) -> None:
    score = 1 - value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += score
        score = 1 - score

#board is a copy at the root position
def run_simulation(root: Node, board: chess.Board, model: nn.Module, device: torch.device) -> None:
    leaf = descend(root, board)
    if leaf.value is None:
        priors, value = evaluate(model, board, device)
        finish(leaf, priors, value)
    else:
        finish(leaf, None, leaf.value)

#value is None when the network has to evaluate the leaf
@dataclass
class Leaf:
    path: list[Node]
    value: float | None

#walks down to a node without children, leaving board at the leaf
def descend(root: Node, board: chess.Board) -> Leaf:
    node = root
    path = [root]
    positions = [board._transposition_key()]

    while node.children:
        move, node = select_child(node)
        board.push(move)
        path.append(node)
        positions.append(board._transposition_key())

    value = game_result(board)
    #a repeat inside the search is scored as a draw (as leela does); a repeat of the game before the root waits for the third time
    if value is None and positions[-1] in positions[:-1]:
        value = DRAW
    return Leaf(path, value)

#priors is None when the leaf's value was decided without the network
def finish(leaf: Leaf, priors: dict[chess.Move, float] | None, value: float) -> None:
    if priors is not None:
        expand(leaf.path[-1], priors)
    backup(leaf.path, value)


#board is left as it was. root_noise is a self play game's random numbers
def search(model: nn.Module, board: chess.Board, device: torch.device, simulations: int = SIMULATIONS,
           root_noise: np.random.Generator | None = None) -> Node:
    if game_result(board) is not None:
        raise ValueError(f"the game is over or the tablebase knows it, there is nothing to search: {board.fen()}")

    root = Node(prior=1.0)
    for simulation in range(simulations):
        #copied with the move stack, which is how game_result sees repetitions
        run_simulation(root, board.copy(), model, device)
        #the first simulation expanded the root, and no child has been chosen yet
        if simulation == 0 and root_noise is not None:
            add_root_noise(root, root_noise)

    return root

def add_root_noise(root: Node, random: np.random.Generator) -> None:
    noise = random.dirichlet([ROOT_NOISE_ALPHA] * len(root.children))
    for child, amount in zip(root.children.values(), noise):
        child.prior = (1 - ROOT_NOISE_SHARE) * child.prior + ROOT_NOISE_SHARE * amount

#ties go to the better mean score, then the prior (leela's order), so a search too short to visit anything plays the policy's move
def most_visited_move(root: Node) -> chess.Move:
    def preference(move: chess.Move) -> tuple[int, float, float]:
        child = root.children[move]
        mean_score = child.value_sum / child.visit_count if child.visit_count > 0 else 0.0
        return child.visit_count, mean_score, child.prior

    return max(root.children, key=preference)

#the most visited move, then the most visited reply, for as long as the next move was visited
def principal_variation(root: Node) -> list[chess.Move]:
    line, node = [], root
    while node.children:
        move = most_visited_move(node)
        if line and node.children[move].visit_count == 0:
            break
        line.append(move)
        node = node.children[move]
    return line


def print_search(root: Node, board: chess.Board, top: int = 10) -> None:
    print(f"{'move':<8}{'prior':>8}{'visits':>8}{'score':>8}")
    for move, child in sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)[:top]:
        score = f"{child.value_sum / child.visit_count:.3f}" if child.visit_count > 0 else "-"
        print(f"{board.san(move):<8}{child.prior:>8.1%}{child.visit_count:>8}{score:>8}")

def main(checkpoint_path: Path, fen: str, simulations: int, device: torch.device, use_tablebase: bool = False) -> None:
    from tools.play.player import load_model
    global tablebase
    if use_tablebase:
        tablebase = open_tablebase()

    model = load_model(checkpoint_path, device)
    board = chess.Board(fen)
    print(board, "\n")

    #also warms the model up, so its setup is not timed with the search
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
    import torch
    from tools.checkpoint import latest_checkpoint
    parser = argparse.ArgumentParser(description="one search from a position, to watch what it does")
    parser.add_argument("checkpoint", nargs="?", type=Path, help="a model.pt, default the run that finished last")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="the position to search, default the start")
    parser.add_argument("--simulations", type=int, default=SIMULATIONS, help="default %(default)s")
    parser.add_argument("--cpu", action="store_true", help="leave the gpu alone, for when a training run is using it")
    parser.add_argument("--tablebase", action="store_true", help="score positions of at most 5 pieces from the syzygy tables (python -m tools.tablebase)")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    main(arguments.checkpoint or latest_checkpoint(), arguments.fen, arguments.simulations, device, arguments.tablebase)
