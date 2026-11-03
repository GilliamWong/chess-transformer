import chess
import numpy as np
import torch
from torch import nn
from engine import mcts
from model.encoding import NO_MOVE, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from engine.selfplay import MAX_PLIES, SAMPLED_PLIES, SEED, TOP_MOVES, TREE_REUSE, Game
from engine.tablebase import covers, wdl_score


#white's result if the game is over, None while it goes on
def final_result(board: chess.Board, max_plies: int = MAX_PLIES) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return 0.5 if outcome.winner is None else 1.0 if outcome.winner == chess.WHITE else 0.0
    if mcts.tablebase is not None and covers(board):
        score = wdl_score(mcts.tablebase.probe_wdl(board))
        return score if board.turn == chess.WHITE else 1 - score
    if len(board.move_stack) >= max_plies:
        return 0.5
    return None

class GameInProgress:
    def __init__(self, number: int, board: chess.Board, tree_reuse: bool):
        self.number = number
        self.random = np.random.default_rng([SEED, number])
        self.board = board.copy()
        self.tree_reuse = tree_reuse
        self.boards, self.visit_moves, self.visit_shares, self.root_scores, self.white_to_move = [], [], [], [], []
        self.new_search(mcts.Node(prior=1.0))

    #with tree reuse root is the node of the move just played, whose visits count towards this search's
    def new_search(self, root: mcts.Node) -> None:
        self.root = root
        #descents leave board at the leaf, for the network; play_game pops it back to root_plies
        self.root_plies = len(self.board.move_stack)
        self.noised = False
        self.add_noise_once_expanded()

    def add_noise_once_expanded(self) -> None:
        if not self.noised and self.root.children:
            mcts.add_root_noise(self.root, self.random)
            self.noised = True

    def play_move(self, sampled_plies: int) -> None:
        moves = list(self.root.children)
        visits = np.array([self.root.children[move].visit_count for move in moves], dtype=np.float64)

        most_visited = [index for index in np.argsort(-visits, kind="stable")[:TOP_MOVES] if visits[index] > 0]
        position_moves = np.full(TOP_MOVES, NO_MOVE, dtype=np.int16)
        position_shares = np.zeros(TOP_MOVES, dtype=np.float32)
        position_moves[:len(most_visited)] = [encode_move_to_id(moves[index], self.board.turn) for index in most_visited]
        position_shares[:len(most_visited)] = visits[most_visited] / visits[most_visited].sum()

        self.boards.append(encode_board(self.board))
        self.visit_moves.append(position_moves)
        self.visit_shares.append(position_shares)
        #the root keeps its score for the side that moved into it
        self.root_scores.append(1 - self.root.value_sum / self.root.visit_count)
        self.white_to_move.append(self.board.turn == chess.WHITE)

        if len(self.board.move_stack) < sampled_plies:
            move = moves[self.random.choice(len(moves), p=visits / visits.sum())]
        else:
            move = mcts.most_visited_move(self.root)
        next_root = self.root.children[move] if self.tree_reuse else mcts.Node(prior=1.0)
        self.board.push(move)
        self.new_search(next_root)

    def to_game(self, white_score: float) -> Game:
        return Game(self.number, np.array(self.boards, dtype=np.uint8).reshape(-1, TOKENS_PER_BOARD),
                    np.array(self.visit_moves, dtype=np.int16).reshape(-1, TOP_MOVES), np.array(self.visit_shares, dtype=np.float32).reshape(-1, TOP_MOVES),
                    np.array(self.root_scores, dtype=np.float32), np.array(self.white_to_move, dtype=bool), white_score)


def play_game(model: nn.Module, device: torch.device, simulations: int, number: int, board: chess.Board | None = None,
              sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES, tree_reuse: bool = TREE_REUSE) -> Game:
    game = GameInProgress(number, board if board is not None else chess.Board(), tree_reuse)
    white_score = final_result(game.board, max_plies)
    while white_score is None:
        leaf = mcts.descend(game.root, game.board)
        if leaf.value is None:
            (priors, value), = mcts.evaluate_many(model, [game.board], device)
            mcts.finish(leaf, priors, value)
        else:
            mcts.finish(leaf, None, leaf.value)

        while len(game.board.move_stack) > game.root_plies:
            game.board.pop()
        game.add_noise_once_expanded()
        if game.root.visit_count >= simulations:
            game.play_move(sampled_plies)
            white_score = final_result(game.board, max_plies)
    return game.to_game(white_score)
