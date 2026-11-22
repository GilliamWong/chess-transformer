"""A slow search by the rule the rust search follows with transpositions, for tests to hold the rust search against.

Written from the rule's specification and from engine/mcts.py's plain search, not from the rust code that selects, descends, gathers
and backs up. Two things the specification leaves as they were are read off search.rs: how a node's proof follows from its moves'
nodes (proof_from_moves, prove) and which move a report prefers (most_visited_edge). What a gathering call may make is from the
comment on search.rs's Round and from watching the rust search.

The rule, in this file's words:

- There is one node a position and fifty-move count, whichever moves led to it. A node's mean is the score of the side that moved
  into it, over every visit it had on any path.
- A move (Link) that has a node holds what selection reads of it: the visits that went through it, the node's mean as the move last
  took it (a float32; none until it has taken one), the simulations that wait for the network through it (pending) and whether it
  is lost. The node a move starts from holds in its value_sum, for each of its moves, the move's visits times (1 - the held mean):
  a visit of a move takes the move's old share out, puts the new one in, and counts one visit of the node.
- A descent picks, at a node with moves, the first move with the highest score: its Q, plus the exploration of its prior over one
  more than its visits and its pending. Q is a draw's score if the move's node repeats the descent's path, else the held mean, else
  (no mean held, or no node) the node's first play urgency. A lost move is not picked; where every move is lost, the first is.
  Nothing of a move's node is read but its position, for the repeat.
- A move picked without a node takes its position's node, the one the search has or a new one, and holds no mean of it.
- The descent stops at the picked move's node without the network if
  1. the node's position repeats one of the descent's own path: the node takes a visit scored as a draw, or
  2. the node is proven (certainty): it takes a visit with its proven score, or
  3. the node has moves already and the move is new to it (no visit yet, whether it took the node now, earlier, or was given it):
     nothing changes in the node, the move only catches up with it.
  Otherwise it goes on through a node that has moves, the move first taking the node's mean where it stands (its visits' share
  of the starting node's value_sum changes with it, and no visit is counted), and a node without moves is its leaf: a collision if
  a simulation waits for the network there already, else scored by the game's result if the game is over there, else it waits for
  the network, which gives it its score and its moves.
- A descent that does not wait walks back up its path at once, and one that waits when its answer comes: each move catches up
  with its node (its held mean the node's as the node now stands) and each node counts one visit.
- A simulation that waits for the network adds one to the pending of every move of its path, and marks its leaf as waited at, until
  its answer is backed up. A collision adds the same until the gathering call that made it ends, and counts no visit. Nothing else
  adds: every other descent is backed up at once.
- A node that has moves makes passes over them as it grows: a pass gives every move of the node that has no node yet the node of
  the position it leads to, if the search has a node of that position and fifty-move count that has been visited. The node's pass k
  (0 the first, MAX_PASSES in all) is due once it has FIRST_PASS_VISITS * PASS_GROWTH^k visits, whichever way the visits came
  (walks up through it, or repeats that ended at it), and is made by the first descent that then comes to the node to pick one of
  its moves, before it picks. A pass counts as every pass the
  node's visits have come to, so a node that comes to its pass late, past the visits of the next one too, makes one pass and not
  two. A position not known at one pass is asked about again at the next. A move given a node has no visit of its own and holds the
  node's mean as it stood at the pass: it is picked by that mean wherever it stands among the node's moves, and its first visit is
  that of any move new to its node: stop 1, 2 or 3, or the node has no moves and is the leaf.
- With certainty, a node below the root can be proven: its result (for the side that moved into it), its score and the plies to the
  game's end. A leaf where the game is over is proven by the rules, a draw too. A descent that ends at a proven node (stop 2, or
  such a leaf) proves what it can of its path once it has walked back up it, from the end up and for as long as it can: a node of
  which a move's node is proven won is lost, the sooner the better for the side to move; a node whose every move has a proven node
  has the best of their results, turned round. A proven node's value_sum is its visits times its proven score, from then on: a
  later walk up through it adds its proven score. The walk up comes first, so the move into a node that the descent proves holds
  the node's mean from before the proof, and the node it starts from keeps the move's share by that mean.
- A move is lost once a descent of its own has met its node as a proven loss: it arrived there (stop 2), or it is on the path of a
  descent whose end proved its node a loss. A move given a node that is a proven loss is lost at once. Another move to the same
  node is picked by its held mean until its own descent meets the node: one more visit.
- A new root (the game went on down the tree by a line of moves, played together) keeps the nodes it reaches by its moves and
  theirs and, for this one line, those the root before the line reaches (the held root): a descent that comes to the position of
  one of them finds it with its visits. Every other node is dropped, and a move to its position makes a new node. A new root then
  proves a draw every node it kept whose position the game has had twice by then, and takes the loss off every move to such a
  node.
"""
from __future__ import annotations
from collections import Counter, deque
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
#the visits at which a node's first pass over its moves is due. Each later pass is due at PASS_GROWTH times the visits of the one
#before, and a node makes MAX_PASSES at most
FIRST_PASS_VISITS = 32
PASS_GROWTH = 4
MAX_PASSES = 13

#how a simulation ended: at a leaf (the first three), or at a node above one (the rule's stops)
EVALUATED = "the network evaluated the leaf"
GAME_OVER = "the game is over at the leaf"
COLLISION = "a simulation waits for the network at the leaf already"
REPEAT = "the position repeats the path"
PROVEN = "the node's result is proven"
NEW_MOVE = "a move's first visit found its node with moves"

#a proven result, for the side that played into the node: the better the higher
LOSS, DRAWN, WIN = 0, 1, 2

#what the tests count of a search (Search.counts): the rule's rarer cases
NO_MEAN_OF_A_VISITED_NODE = "a move weighed that holds no mean of a node that has visits"
NO_MEAN_OF_A_REPEATED_NODE = "a move weighed that holds no mean of a node that repeats have visited and the network not yet"
STALE_MEAN = "a move weighed by a mean that is not its node's any more"
MEAN_REFRESHED = "a move's mean changed as a descent passed it"
LOST_PASSED_OVER = "a lost move passed over"
UNLEARNED_LOSS = "a move to a node proven lost weighed by its mean, not having met the proof"
LOSS_LEARNED = "a move met its node's proof of a loss that another move's descent had made"
EVERY_MOVE_LOST = "a pick where every move is lost"
LOSS_CLEARED = "a lost move cleared at a new root"
HELD_NODE_MET = "a move took or was given a node that only the held root reached"
NODES_DROPPED = "nodes dropped at a new root"


@dataclass(frozen=True)
class Proof:
    outcome: int
    #the node's score under the proof
    value: float
    #to the game's end
    plies: int

    #the better for the side that played into the node the higher: a win the sooner, a loss the later
    def rank(self) -> tuple[int, int]:
        return self.outcome, {WIN: -self.plies, DRAWN: 0, LOSS: self.plies}[self.outcome]

@dataclass(eq=False, slots=True)
class Link:
    move: chess.Move
    prior: float
    #the node of the position after the move: found or made when the move is first picked, or given by a pass of the node it starts from
    child: Node | None = None
    #where the node was given (the move had it before it was ever picked), the pass of the node the move starts from that gave it
    given_at_pass: int | None = None
    visits: int = 0
    #the child's mean when the move last took it, a float32's value: at its last visit, as a descent passed it, or when it was given
    #the node. None until then
    mean: float | None = None
    #the simulations waiting for the network through the move, and a gathering call's collisions
    pending: int = 0
    #the move's own descent met its node proven a loss
    lost: bool = False

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
    #the passes over its moves it counts as having made
    passes_made: int = 0
    #the simulations that wait for the network at it, and a gathering call's collisions there
    waited_at: int = 0
    proof: Proof | None = None

    def mean(self) -> float:
        return self.value_sum / self.visit_count


@dataclass
class Simulation:
    ended: str
    #the position the network was asked about, None if it was not asked
    evaluated: chess.Board | None
    #the moves from the root to the node the simulation ended at, and the visits that node had before
    line: list[chess.Move]
    visits_before: int
    #for each move that was given a node on its way down, at the nodes whose pass it made, the pass that gave it: 0 a node's first
    passes_of_moves_given: list[int]
    #the pass that gave its last move its node, where this is that move's first visit: None for any other move
    first_visit_given_at_pass: int | None
    #the nodes it picked a move from, each with the move, the root's first
    path: list[tuple[Node, Link]] = field(repr=False, default_factory=list)


#whether the node's position was on the path already: path_keys are the keys from the root to the node the move into `node` is
#played from. Same side to move (every other ply), no capture or pawn move since (the node's count reaches back that far), and
#nothing before the root
def repeats(path_keys: list[Hashable], node: Node) -> bool:
    furthest = min(node.halfmoves, len(path_keys))
    return any(path_keys[-plies_back] == node.key for plies_back in range(SHORTEST_REPEAT_PLIES, furthest + 1, 2))


#mcts.select_child for links: a link's visits and pending explore; its Q is a draw if its node repeats the path, the mean it holds,
#or the node's first play urgency where it holds none; a lost link is not picked while another is not lost. Counts the rule's cases
def select_link(node: Node, path_keys: list[Hashable], counts: Counter) -> Link:
    c_puct = mcts.C_PUCT + mcts.C_PUCT_FACTOR * math.log((node.visit_count + mcts.C_PUCT_BASE) / mcts.C_PUCT_BASE)
    sqrt_children_visits = math.sqrt(max(node.visit_count - 1, 1))
    #summed afresh at every pick, which is the same as refreshing it when a link has its first visit
    visited_policy = sum(link.prior for link in node.links if link.visits > 0)
    first_play_urgency = (1 - node.mean()) - mcts.FPU_REDUCTION * math.sqrt(visited_policy)

    best_link, best_U = None, -math.inf
    for link in node.links:
        child = link.child
        if child is None:
            Q, N = first_play_urgency, 0
        elif link.lost:
            counts[LOST_PASSED_OVER] += 1
            continue
        else:
            N = link.visits + link.pending
            if repeats(path_keys, child):
                Q = 1 - mcts.DRAW
            elif link.mean is None:
                Q = first_play_urgency
                counts[NO_MEAN_OF_A_VISITED_NODE] += child.visit_count > 0
                counts[NO_MEAN_OF_A_REPEATED_NODE] += child.visit_count > 0 and not child.links
            else:
                Q = link.mean
                counts[STALE_MEAN] += link.mean != float(np.float32(child.mean()))
                counts[UNLEARNED_LOSS] += child.proof is not None and child.proof.outcome == LOSS
        U = Q + c_puct * link.prior * (sqrt_children_visits / (1 + N))
        #the first link with the highest score wins ties
        if U > best_U:
            best_link, best_U = link, U
    if best_link is None:
        counts[EVERY_MOVE_LOST] += 1
        return node.links[0]
    return best_link


#each link of the path, the last first, catches up with its node, whose numbers are final by then; the node it starts from counts the
#simulation's visit: by its moves' shares, or with its proven score if it is proven
def back_up(path: list[tuple[Node, Link]]) -> None:
    for parent, link in reversed(path):
        new_mean = float(np.float32(link.child.mean()))
        #a link without a mean has no visits, and no share to take out
        visits, old_mean = link.visits, link.mean if link.mean is not None else 0.0
        parent.visit_count += 1
        if parent.proof is None:
            parent.value_sum += (visits + 1) * (1 - new_mean) - visits * (1 - old_mean)
        else:
            parent.value_sum = parent.visit_count * parent.proof.value
        link.visits = visits + 1
        link.mean = new_mean


#mcts.finish's children: the highest prior first, equal priors in python-chess's order, each prior as the rust search stores it
def expand(node: Node, priors: dict[chess.Move, float]) -> None:
    ordered = sorted(priors.items(), key=lambda item: -item[1])
    node.links = [Link(move, mcts.stored_prior(rank, prior)) for rank, (move, prior) in enumerate(ordered)]


#what a node is of: the position without its counters, and its plies since a capture or pawn move
def identity_of(board: chess.Board) -> tuple[Hashable, int]:
    return board._transposition_key(), board.halfmove_clock

#the passes over its moves that a node with so many visits has come to
def passes_due(visits: int) -> int:
    passes = 0
    while passes < MAX_PASSES and visits >= FIRST_PASS_VISITS * PASS_GROWTH ** passes:
        passes += 1
    return passes

#a node's visits are those of its proven score from now on
def prove(node: Node, proof: Proof) -> None:
    node.proof = proof
    node.value_sum = node.visit_count * proof.value

#a node's result from its moves' nodes' proofs: lost for the side that moved into it if a move wins (the quickest), and otherwise
#known once every move has a proven node, the best of them turned round
def proof_from_moves(node: Node) -> Proof | None:
    proofs = [link.child.proof for link in node.links if link.child is not None and link.child.proof is not None]
    if not proofs:
        return None
    #the first of the best, as max() gives it
    best = max(proofs, key=Proof.rank)
    if best.outcome != WIN and len(proofs) < len(node.links):
        return None
    return Proof({WIN: LOSS, DRAWN: DRAWN, LOSS: WIN}[best.outcome], 1 - best.value, best.plies + 1)


#the nodes a node reaches by its moves and theirs, itself among them
def reached(node: Node) -> set[Node]:
    found, stack = {node}, [node]
    while stack:
        for link in stack.pop().links:
            if link.child is not None and link.child not in found:
                found.add(link.child)
                stack.append(link.child)
    return found


class Search:
    def __init__(self, model: nn.Module, board: chess.Board, device: torch.device, certainty: bool = False):
        if mcts.game_result(board) is not None:
            raise ValueError(f"the game is over or the tablebase knows it, there is nothing to search: {board.fen()}")
        self.model, self.board, self.device, self.certainty = model, board.copy(), device, certainty
        self.nodes: dict[tuple[Hashable, int], Node] = {}
        self.root = self.node_of(board)
        #the simulations that wait for the network, a gathering call's together, the oldest call first
        self.rounds: deque[list[Simulation]] = deque()
        self.counts = Counter()
        #the nodes kept for the root before the last line played, which the root did not reach when the line was played or since
        self.only_held: set[Node] = set()

    def node_of(self, board: chess.Board) -> Node:
        identity = identity_of(board)
        if identity not in self.nodes:
            self.nodes[identity] = Node(*identity)
        return self.nodes[identity]

    #a move took or was given `node`: the root reaches it now, and what it reaches
    def meet(self, node: Node) -> None:
        if node in self.only_held:
            self.counts[HELD_NODE_MET] += 1
            self.only_held -= reached(node)

    #the simulations waiting for the network
    def waiting(self) -> int:
        return sum(len(leaves) for leaves in self.rounds)

    #a node's pass: each of its moves without a node takes the node of the position it leads to, if the search has one that
    #has been visited, with the node's mean as it stands. `board` is the node's position. The pass's number, once for each move it
    #gave a node
    def give_known_nodes(self, node: Node, board: chess.Board) -> list[int]:
        this_pass = node.passes_made
        node.passes_made = passes_due(node.visit_count)
        passes_of_moves_given = []
        for link in node.links:
            if link.child is not None:
                continue
            board.push(link.move)
            known = self.nodes.get(identity_of(board))
            board.pop()
            if known is not None and known.visit_count > 0:
                link.child, link.given_at_pass, link.mean = known, this_pass, float(np.float32(known.mean()))
                link.lost = known.proof is not None and known.proof.outcome == LOSS
                self.meet(known)
                passes_of_moves_given.append(this_pass)
        return passes_of_moves_given

    #a descent has ended at a proven node: the nodes above it on its path are proven for as long as their moves' nodes prove them,
    #never the root; and from the end up, for as long as the path's nodes are proven, the move into a node proven a loss is lost
    def learn_proofs(self, path: list[tuple[Node, Link]]) -> None:
        for parent, _ in reversed(path[1:]):
            proof = proof_from_moves(parent)
            if proof is None:
                break
            prove(parent, proof)
        for _, link in reversed(path):
            if link.child.proof is None:
                break
            if link.child.proof.outcome == LOSS:
                link.lost = True

    #one descent from the root, backed up at once unless it ends at a leaf for the network or at one that waits for it already
    def descend(self) -> Simulation:
        #copied with the move stack, which is how game_result sees repetitions
        board = self.board.copy()
        node = self.root
        path: list[tuple[Node, Link]] = []
        path_keys = [self.root.key]
        line = []
        passes_of_moves_given, first_visit_given_at_pass = [], None
        ended_as = lambda ended, visits_before, evaluated=None: Simulation(
            ended, evaluated, line, visits_before, passes_of_moves_given, first_visit_given_at_pass, path)

        while node.links:
            if passes_due(node.visit_count) > node.passes_made:
                passes_of_moves_given += self.give_known_nodes(node, board)
            link = select_link(node, path_keys, self.counts)
            first_visit_given_at_pass = link.given_at_pass if link.visits == 0 else None
            board.push(link.move)
            line.append(link.move)
            if link.child is None:
                link.child = self.node_of(board)
                self.meet(link.child)
            child = link.child
            path.append((node, link))

            visits_before = child.visit_count
            if repeats(path_keys, child):
                child.visit_count += 1
                child.value_sum += 1 - mcts.DRAW
                back_up(path)
                return ended_as(REPEAT, visits_before)
            if child.proof is not None:
                child.visit_count += 1
                child.value_sum += child.proof.value
                self.counts[LOSS_LEARNED] += child.proof.outcome == LOSS and not link.lost
                back_up(path)
                self.learn_proofs(path)
                return ended_as(PROVEN, visits_before)
            if child.links and link.visits == 0:
                back_up(path)
                return ended_as(NEW_MOVE, visits_before)
            if child.links:
                #the move takes its node's mean in passing: its visits' share of the node it starts from goes with it
                new_mean = float(np.float32(child.mean()))
                self.counts[MEAN_REFRESHED] += new_mean != link.mean
                if node.proof is None:
                    node.value_sum += link.visits * (link.mean - new_mean)
                link.mean = new_mean
            path_keys.append(child.key)
            node = child

        #a node without moves: the root at first, a new position, one only repeats ended at so far, or a position where the game is over
        visits_before = node.visit_count
        if node.waited_at:
            self.wait(path, node)
            return ended_as(COLLISION, visits_before)
        value = mcts.game_result(board)
        if value is None:
            self.wait(path, node)
            return ended_as(EVALUATED, visits_before, board)
        node.visit_count += 1
        node.value_sum += 1 - value
        back_up(path)
        if self.certainty and path:
            prove(node, Proof(WIN if value == 0 else DRAWN, 1 - value, 0))
            self.learn_proofs(path)
        return ended_as(GAME_OVER, visits_before)

    #a simulation waits at `node` through the moves of `path`, or `waits` of -1 takes one off
    def wait(self, path: list[tuple[Node, Link]], node: Node, waits: int = 1) -> None:
        node.waited_at += waits
        for _, link in path:
            link.pending += waits

    #a gathering call (a round): descents until `leaves` of them wait for the network, or `simulations` are made (those backed up at
    #once among them, collisions not), or one collision more than `collisions`, or the root's visits with the simulations that wait
    #are `visits`. Every descent it made, in turn; the ones that wait are answered by finish
    def gather(self, leaves: int = 1, collisions: int = 0, simulations: int | None = None, visits: int | None = None) -> list[Simulation]:
        simulations = leaves if simulations is None else simulations
        made, waiting, collided = [], [], []
        while (len(waiting) < leaves and len(made) - len(collided) < simulations and len(collided) <= collisions
               and (visits is None or self.root.visit_count + self.waiting() + len(waiting) < visits)):
            simulation = self.descend()
            made.append(simulation)
            if simulation.ended == COLLISION:
                collided.append(simulation)
            elif simulation.ended == EVALUATED:
                waiting.append(simulation)
        for simulation in collided:
            self.wait(simulation.path, self.end_of(simulation), -1)
        self.rounds.append(waiting)
        return made

    def end_of(self, simulation: Simulation) -> Node:
        return simulation.path[-1][1].child if simulation.path else self.root

    #the network's answers for the oldest gathering call's leaves, each backed up in turn
    def finish(self) -> None:
        for simulation in self.rounds.popleft():
            node = self.end_of(simulation)
            priors, value = mcts.evaluate(self.model, simulation.evaluated, self.device)
            expand(node, priors)
            node.visit_count += 1
            node.value_sum += 1 - value
            self.wait(simulation.path, node, -1)
            back_up(simulation.path)

    #a round of one simulation, answered at once
    def simulate(self) -> Simulation:
        simulation, = self.gather()
        self.finish()
        return simulation

    #the game went on by `line`, moves of the tree from its root played together: the tree's node there is the root. The root before
    #the line is held until the next line: what it or the new root reaches stays, the rest is dropped. A node whose position the
    #game has had twice by now is a draw wherever a line comes to it, and a move to it is not lost
    def new_root(self, line: list[chess.Move]) -> None:
        held = self.root
        for move in line:
            self.root = next(link.child for link in self.root.links if link.move == move)
            self.board.push(move)
        kept = reached(self.root) | reached(held)
        self.only_held = kept - reached(self.root)
        self.counts[NODES_DROPPED] += len(self.nodes) - len(kept)
        self.nodes = {identity: node for identity, node in self.nodes.items() if node in kept}
        if not self.certainty:
            return
        #the game's positions back to its last capture or pawn move, the root's among them
        game, had = self.board.copy(), Counter()
        for _ in range(game.halfmove_clock + 1):
            had[game._transposition_key()] += 1
            if game.move_stack:
                game.pop()
        for node in self.nodes.values():
            if node is not self.root and had[node.key] >= 2 and node.halfmoves > self.root.halfmoves:
                prove(node, Proof(DRAWN, 1 - mcts.DRAW, 0))
        for node in self.nodes.values():
            for link in node.links:
                if link.lost and link.child.proof.outcome != LOSS:
                    link.lost = False
                    self.counts[LOSS_CLEARED] += 1


#until the root has `visits` visits
def search(model: nn.Module, board: chess.Board, device: torch.device, visits: int = mcts.SIMULATIONS) -> Node:
    running = Search(model, board, device)
    while running.root.visit_count < visits:
        running.simulate()
    return running.root


#mcts.most_visited_move by the links' own visits, after the nodes' proofs: a proven win first (the soonest), a proven loss last (the
#latest first); ties go to the better mean score of the link's node, then the prior, then the first
def most_visited_link(node: Node) -> Link:
    def preference(link: Link) -> tuple[tuple[int, int], int, float, float]:
        proof = link.child.proof if link.child is not None else None
        proven = proof.rank() if proof is not None and proof.outcome != DRAWN else (DRAWN, 0)
        mean_score = link.child.mean() if link.visits > 0 else 0.0
        return proven, link.visits, mean_score, link.prior

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

#whether the root's moves prove its result
def root_decided(root: Node) -> bool:
    return proof_from_moves(root) is not None

#whether at most one of the root's moves is not to a node proven a loss
def one_move_left(root: Node) -> bool:
    return bool(root.links) and sum(link.child is None or link.child.proof is None or link.child.proof.outcome != LOSS for link in root.links) < 2
