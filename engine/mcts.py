#scores are expected scores, 0 lost to 1 won
from __future__ import annotations
from dataclasses import dataclass, field
import math
import struct
from typing import TYPE_CHECKING
import chess
import chess.syzygy
import numpy as np
from model.encoding import encode_board, encode_move_to_id
from engine.tablebase import covers, wdl_score
#self play's workers import this file and must not import torch
if TYPE_CHECKING:
    import torch
    from torch import nn

SIMULATIONS = 200
DRAW = 0.5
REPETITIONS_FOR_A_DRAW = 3
#lc0's defaults, halved where they scale a score: its scores run from -1 to 1, ours from 0 to 1
C_PUCT = 1.745 / 2
C_PUCT_FACTOR = 3.894 / 2
C_PUCT_BASE = 38739
FPU_REDUCTION = 0.33 / 2
ROOT_NOISE_ALPHA = 0.3
ROOT_NOISE_SHARE = 0.25
#a node's first HEAD_MOVES moves keep 16 bit priors and the rest 8 bit ones, as the rust search stores them (search.rs's stored_prior)
HEAD_MOVES = 8

tablebase: chess.syzygy.Tablebase | None = None


@dataclass(slots=True)
class Node:
    prior: float
    visit_count: int = 0
    #for the side that played the move into this node
    value_sum: float = 0.0
    children: dict[chess.Move, Node] = field(default_factory=dict)


#the priors of the legal moves and the score of the side to move
def evaluate(model: nn.Module, board: chess.Board, device: torch.device) -> tuple[dict[chess.Move, float], float]:
    return evaluate_many(model, [board], device)[0]

def evaluate_many(model: nn.Module, boards: list[chess.Board], device: torch.device) -> list[tuple[dict[chess.Move, float], float]]:
    import torch
    from model.model import expected_score
    legal_moves = [list(board.legal_moves) for board in boards]
    for board, moves in zip(boards, legal_moves):
        if not moves:
            raise ValueError(f"the game is over here, game_result scores it: {board.fen()}")

    encoded = torch.tensor([encode_board(board) for board in boards], dtype=torch.long, device=device)
    with torch.no_grad():
        move_logits, value_logits = model(encoded)
    move_logits = move_logits.float().cpu().numpy()
    values = expected_score(value_logits).cpu().tolist()

    evaluations = []
    for board, moves, logits, value in zip(boards, legal_moves, move_logits, values):
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


def select_child(node: Node) -> tuple[chess.Move, Node]:
    c_puct = C_PUCT + C_PUCT_FACTOR * math.log((node.visit_count + C_PUCT_BASE) / C_PUCT_BASE)
    #at least 1, so the first pick follows the priors
    sqrt_children_visits = math.sqrt(max(node.visit_count - 1, 1))
    visited_policy = sum(child.prior for child in node.children.values() if child.visit_count > 0)
    first_play_urgency = (1 - node.value_sum / node.visit_count) - FPU_REDUCTION * math.sqrt(visited_policy)

    #the first child with the highest score wins ties, as max() would
    best_move, best_child, best_U = None, None, -math.inf
    for move, child in node.children.items():
        N = child.visit_count
        Q = child.value_sum / N if N > 0 else first_play_urgency
        U = Q + c_puct * child.prior * (sqrt_children_visits / (1 + N))
        if U > best_U:
            best_move, best_child, best_U = move, child, U
    return best_move, best_child


#value is for the side to move at the last node; the score flips at every level up
def backup(path: list[Node], value: float) -> None:
    score = 1 - value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += score
        score = 1 - score

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

def float_of(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


#the prior kept for the move of `rank` in the priors' order: the nearest 16 bit one (5 bits of exponent from 2^-31, 11 of significand)
#or 8 bit one (exponents 2^-18 to 2^-4, 4 bits of significand)
def stored_prior(rank: int, prior: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", prior))[0]
    if rank < HEAD_MOVES:
        code = min(max(((bits + (1 << 11)) >> 12) - (96 << 11), 0), 0xFFFF)
        return 0.0 if code == 0 else float_of((code + (96 << 11)) << 12)
    rounded = bits + (1 << 18)
    exponent = rounded >> 23
    if exponent <= 108:
        return 0.0
    code = 255 if exponent > 123 else (exponent - 108) << 4 | (rounded >> 19 & 15)
    return float_of(((code >> 4) + 108) << 23 | (code & 15) << 19)


def finish(leaf: Leaf, priors: dict[chess.Move, float] | None, value: float) -> None:
    if priors is not None:
        #the highest prior first, equal priors in the order given (python-chess's): the rust search stops at the first move without a node
        ordered = sorted(priors.items(), key=lambda item: -item[1])
        leaf.path[-1].children = {move: Node(prior=stored_prior(rank, prior)) for rank, (move, prior) in enumerate(ordered)}
    backup(leaf.path, value)


def search(model: nn.Module, board: chess.Board, device: torch.device, simulations: int = SIMULATIONS,
           root_noise: np.random.Generator | None = None) -> Node:
    if game_result(board) is not None:
        raise ValueError(f"the game is over or the tablebase knows it, there is nothing to search: {board.fen()}")

    root = Node(prior=1.0)
    for simulation in range(simulations):
        #copied with the move stack, which is how game_result sees repetitions
        leaf_board = board.copy()
        leaf = descend(root, leaf_board)
        if leaf.value is None:
            priors, value = evaluate(model, leaf_board, device)
            finish(leaf, priors, value)
        else:
            finish(leaf, None, leaf.value)
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

def principal_variation(root: Node) -> list[chess.Move]:
    line, node = [], root
    while node.children:
        move = most_visited_move(node)
        if line and node.children[move].visit_count == 0:
            break
        line.append(move)
        node = node.children[move]
    return line
