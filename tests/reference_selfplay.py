#self play in python, with mcts.py's search: the reference the rust games (selfplay_rs) are held to in test_selfplay_rs.py.
#a change to the search or the rules goes here and into selfplay_rs/src alike
from collections.abc import Iterable, Iterator
import chess
import numpy as np
import torch
from torch import nn
import mcts
from encoding import TOKENS_PER_BOARD, encode_board, encode_move_to_id
from selfplay import MAX_PLIES, SAMPLED_PLIES, SEED, TOP_MOVES, TREE_REUSE, Game
from tools.label_with_stockfish import NO_MOVE
from tools.tablebase import covers, wdl_score


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

#its random numbers come from the seed and its number alone, so a game can be played again
class GameInProgress:
    def __init__(self, number: int, seed: int, board: chess.Board, tree_reuse: bool):
        self.number = number
        self.random = np.random.default_rng([seed, number])
        self.board = board.copy()
        self.tree_reuse = tree_reuse
        self.boards, self.visit_moves, self.visit_shares, self.root_scores, self.white_to_move = [], [], [], [], []
        self.new_search(mcts.Node(prior=1.0))

    #with tree reuse root is the node of the move just played, whose visits count towards this search's
    def new_search(self, root: mcts.Node) -> None:
        self.root = root
        #the search walks down on the game's own board and is taken back here after every simulation
        self.root_plies = len(self.board.move_stack)
        self.noised = False
        self.add_noise_once_expanded()

    #as soon as the root has children, before any is chosen: at once for a reused root, after the first simulation otherwise
    def add_noise_once_expanded(self) -> None:
        if not self.noised and self.root.children:
            mcts.add_root_noise(self.root, self.random)
            self.noised = True

    #records the position with what the search found, and plays a move
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


#up to `size` games at once, one simulation of each per round, their leaves through the network in one call.
#a move is played once its root has `simulations` visits, and a game that ends makes room for the next number
class SelfPlayBatch:
    def __init__(self, model: nn.Module, device: torch.device, simulations: int, numbers: Iterator[int], size: int, seed: int,
                 board: chess.Board, sampled_plies: int, max_plies: int, tree_reuse: bool):
        self.model, self.device, self.simulations = model, device, simulations
        self.numbers, self.seed, self.start_board = numbers, seed, board
        self.sampled_plies, self.max_plies, self.tree_reuse = sampled_plies, max_plies, tree_reuse
        self.games: list[GameInProgress] = []
        self.finished: list[Game] = []
        for _ in range(size):
            self.start_next_game()

    def start_next_game(self) -> None:
        for number in self.numbers:
            game = GameInProgress(number, self.seed, self.start_board, self.tree_reuse)
            white_score = final_result(game.board, self.max_plies)
            if white_score is None:
                self.games.append(game)
                return
            self.finished.append(game.to_game(white_score)) #over before it began: a game without positions

    def playing(self) -> bool:
        return bool(self.games or self.finished)

    #one round, returning the games it finished
    def step(self) -> list[Game]:
        waiting = []
        for game in self.games:
            leaf = mcts.descend(game.root, game.board)
            if leaf.value is None:
                waiting.append((game, leaf))
            else:
                mcts.finish(leaf, None, leaf.value)
        if waiting:
            #every waiting game's board is still at its leaf
            evaluations = mcts.evaluate_many(self.model, [game.board for game, _ in waiting], self.device)
            for (game, leaf), (priors, value) in zip(waiting, evaluations):
                mcts.finish(leaf, priors, value)

        playing = []
        for game in self.games:
            while len(game.board.move_stack) > game.root_plies:
                game.board.pop()
            game.add_noise_once_expanded()
            if game.root.visit_count >= self.simulations:
                game.play_move(self.sampled_plies)
                white_score = final_result(game.board, self.max_plies)
                if white_score is not None:
                    self.finished.append(game.to_game(white_score))
                    continue
            playing.append(game)

        replacements = len(self.games) - len(playing)
        self.games = playing
        for _ in range(replacements):
            self.start_next_game()
        finished, self.finished = self.finished, []
        return finished

#the games with these numbers, `size` at a time, each yielded as it ends
def play_games(model: nn.Module, device: torch.device, simulations: int, numbers: Iterable[int], size: int, seed: int = SEED,
               board: chess.Board | None = None, sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES,
               tree_reuse: bool = TREE_REUSE) -> Iterator[Game]:
    batch = SelfPlayBatch(model, device, simulations, iter(numbers), size, seed, board if board is not None else chess.Board(), sampled_plies,
                          max_plies, tree_reuse)
    while batch.playing():
        yield from batch.step()

def play_game(model: nn.Module, device: torch.device, simulations: int, number: int, seed: int = SEED, board: chess.Board | None = None,
              sampled_plies: int = SAMPLED_PLIES, max_plies: int = MAX_PLIES, tree_reuse: bool = TREE_REUSE) -> Game:
    return next(play_games(model, device, simulations, [number], 1, seed, board, sampled_plies, max_plies, tree_reuse))
