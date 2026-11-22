"""A slow search by the rule the rust search calls move_means (with transpositions), for tests to hold the rust search against.

Written from the rule's specification and from engine/mcts.py's plain search, not from the rust code that backs the rule up.

The rule, in this file's words:

- There is one node a position and fifty-move count, whichever moves led to it. A node's mean is the score of the side that moved
  into it, over every visit it had on any path.
- A move (Link) counts the visits that went through it, and remembers its node's mean as of the last of them, as a float32.
  The node a move starts from holds in its value_sum, for each of its moves, the move's visits times (1 - the remembered mean):
  a visit of a move takes the move's old share out, puts the new one in, and counts one visit of the node.
- A descent picks a move as the search with transpositions always did, and stops at the move's node without the network if
  1. the node's position repeats one of the descent's own path: the node takes a visit scored as a draw, or
  2. the node has moves already and the move is new to it: nothing changes in the node, the move only catches up with it.
  Otherwise it goes on through a node that has moves, the move first taking the node's mean where it stands (its visits' share
  of the starting node's value_sum changes with it, and no visit is counted), and a node without moves is its leaf: scored by the game's result if the
  game is over there, else by the network, which also gives it its moves.
- Every descent then walks back up its path, each move catching up with its node and each node counting one visit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Hashable
import math
import chess
import numpy as np
import torch
from torch import nn
from engine import mcts

#no position comes round in fewer plies
SHORTEST_REPEAT_PLIES = 4

#how a simulation ended: at a leaf (the first two), or at a node above one (the rule's stops)
EVALUATED = "the network evaluated the leaf"
GAME_OVER = "the game is over at the leaf"
REPEAT = "the position repeats the path"
NEW_MOVE = "a move's first visit found its node with moves"


@dataclass(eq=False, slots=True)
class Link:
    move: chess.Move
    prior: float
    #the node of the position after the move, found or made when the move is first picked
    child: Node | None = None
    visits: int = 0
    #the child's mean when the move was last visited, a float32's value
    mean: float = 0.0

@dataclass(eq=False, slots=True)
class Node:
    #the position without its counters, and its plies since a capture or pawn move
    key: Hashable
    halfmoves: int
    visit_count: int = 0
    #for the side that played a move into this node
    value_sum: float = 0.0
    #empty until the network has evaluated the node
    links: list[Link] = field(default_factory=list)

    def mean(self) -> float:
        return self.value_sum / self.visit_count


@dataclass
class Simulation:
    ended: str
    #the position the network evaluated, None if it was not asked
    evaluated: chess.Board | None
    #the moves from the root to the node the simulation ended at, and the visits that node had before
    line: list[chess.Move]
    visits_before: int


#whether the node's position was on the path already: path_keys are the keys from the root to the node the move into `node` is
#played from. Same side to move (every other ply), no capture or pawn move since (the node's count reaches back that far), and
#nothing before the root
def repeats(path_keys: list[Hashable], node: Node) -> bool:
    furthest = min(node.halfmoves, len(path_keys))
    return any(path_keys[-plies_back] == node.key for plies_back in range(SHORTEST_REPEAT_PLIES, furthest + 1, 2))


#mcts.select_child for links: a link's own visits explore, its node's mean is its Q; a link without a node, or whose node has no
#visit, is an unvisited child; a link whose node repeats the path is worth a draw
def select_link(node: Node, path_keys: list[Hashable]) -> Link:
    c_puct = mcts.C_PUCT + mcts.C_PUCT_FACTOR * math.log((node.visit_count + mcts.C_PUCT_BASE) / mcts.C_PUCT_BASE)
    sqrt_children_visits = math.sqrt(max(node.visit_count - 1, 1))
    #summed afresh at every pick, which is the same as refreshing it when a link has its first visit
    visited_policy = sum(link.prior for link in node.links if link.visits > 0)
    first_play_urgency = (1 - node.mean()) - mcts.FPU_REDUCTION * math.sqrt(visited_policy)

    best_link, best_U = None, -math.inf
    for link in node.links:
        child = link.child
        if child is not None and repeats(path_keys, child):
            Q, N = 1 - mcts.DRAW, link.visits
        elif child is not None and child.visit_count > 0:
            Q, N = child.mean(), link.visits
        else:
            Q, N = first_play_urgency, 0
        U = Q + c_puct * link.prior * (sqrt_children_visits / (1 + N))
        #the first link with the highest score wins ties
        if U > best_U:
            best_link, best_U = link, U
    return best_link


#each link of the path, the last first, catches up with its node, whose numbers are final by then; the node it starts from counts the visit
def back_up(path: list[tuple[Node, Link]]) -> None:
    for parent, link in reversed(path):
        new_mean = float(np.float32(link.child.mean()))
        visits, old_mean = link.visits, link.mean
        parent.visit_count += 1
        parent.value_sum += (visits + 1) * (1 - new_mean) - visits * (1 - old_mean)
        link.visits = visits + 1
        link.mean = new_mean


#mcts.finish's children: the highest prior first, equal priors in python-chess's order, each prior as the rust search stores it
def expand(node: Node, priors: dict[chess.Move, float]) -> None:
    ordered = sorted(priors.items(), key=lambda item: -item[1])
    node.links = [Link(move, mcts.stored_prior(rank, prior)) for rank, (move, prior) in enumerate(ordered)]


class Search:
    def __init__(self, model: nn.Module, board: chess.Board, device: torch.device):
        if mcts.game_result(board) is not None:
            raise ValueError(f"the game is over or the tablebase knows it, there is nothing to search: {board.fen()}")
        self.model, self.board, self.device = model, board, device
        self.nodes: dict[tuple[Hashable, int], Node] = {}
        self.root = self.node_of(board)

    def node_of(self, board: chess.Board) -> Node:
        identity = (board._transposition_key(), board.halfmove_clock)
        if identity not in self.nodes:
            self.nodes[identity] = Node(*identity)
        return self.nodes[identity]

    def simulate(self) -> Simulation:
        #copied with the move stack, which is how game_result sees repetitions
        board = self.board.copy()
        node = self.root
        path: list[tuple[Node, Link]] = []
        path_keys = [self.root.key]
        line = []

        while node.links:
            link = select_link(node, path_keys)
            board.push(link.move)
            line.append(link.move)
            if link.child is None:
                link.child = self.node_of(board)
            child = link.child
            path.append((node, link))

            visits_before = child.visit_count
            if repeats(path_keys, child):
                child.visit_count += 1
                child.value_sum += 1 - mcts.DRAW
                back_up(path)
                return Simulation(REPEAT, None, line, visits_before)
            if child.links and link.visits == 0:
                back_up(path)
                return Simulation(NEW_MOVE, None, line, visits_before)
            if child.links:
                #the move takes its node's mean in passing: its visits' share of the node it starts from goes with it
                new_mean = float(np.float32(child.mean()))
                node.value_sum += link.visits * (link.mean - new_mean)
                link.mean = new_mean
            path_keys.append(child.key)
            node = child

        #a node without moves: the root at first, a new position, one only repeats ended at so far, or a position where the game is over
        visits_before = node.visit_count
        value = mcts.game_result(board)
        ended = GAME_OVER
        if value is None:
            priors, value = mcts.evaluate(self.model, board, self.device)
            expand(node, priors)
            ended = EVALUATED
        node.visit_count += 1
        node.value_sum += 1 - value
        back_up(path)
        return Simulation(ended, board if ended == EVALUATED else None, line, visits_before)


def search(model: nn.Module, board: chess.Board, device: torch.device, simulations: int = mcts.SIMULATIONS) -> Node:
    running = Search(model, board, device)
    for _ in range(simulations):
        running.simulate()
    return running.root


#mcts.most_visited_move by the links' own visits: ties go to the better mean score of the link's node, then the prior, then the first
def most_visited_link(node: Node) -> Link:
    def preference(link: Link) -> tuple[int, float, float]:
        mean_score = link.child.mean() if link.visits > 0 else 0.0
        return link.visits, mean_score, link.prior

    return max(node.links, key=preference)

def principal_variation(root: Node) -> list[chess.Move]:
    line, node = [], root
    while node is not None and node.links:
        link = most_visited_link(node)
        if line and link.visits == 0:
            break
        line.append(link.move)
        node = link.child
    return line
