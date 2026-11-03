from dataclasses import dataclass
import math
import random
from pathlib import Path
import chess
import torch
from torch import nn
from training.data import EXPECTED_SCORE_PER_CENTIPAWN
from engine.checkpoint import load_checkpoint
from model.encoding import NUM_MOVE_IDS, encode_board, encode_move_to_id, decode_id_to_move
from model.model import ChessTransformer, expected_score
from engine.tablebase import covers, ranked_moves, wdl_score

#how a move is picked:
#   argmax            the most likely legal move
#   sample            drawn at --temperature for the first --sample-plies plies, argmax after
#   avoid-repetition  argmax, but when clearly ahead play the best move that does not repeat a position
#   mcts              the move mcts.py's search visits most
SELECTIONS = ("argmax", "sample", "avoid-repetition", "mcts")
TEMPERATURE = 0.5
SAMPLE_PLIES = 1000 #the whole game
#avoid-repetition needs both: the d640 bot lost lichess game dwCvJZkh avoiding a repetition in a drawn rook
#ending it rated a sliver ahead, with a move the policy gave 1.9% that hung the rook
AHEAD_SCORE = 0.6
ALTERNATIVE_SHARE = 0.1
SIMULATIONS = 200
ROOT_MOVES_REPORTED = 3
MAX_REPORTED_SCORE = 0.9999 #about 25 pawns

#the model in eval mode, at the size its checkpoint says
def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer:
    model, _ = load_checkpoint(checkpoint_path, device)
    model.eval()

    return model

#cublas keeps a workspace for each stream it runs on, made at its first matmul there: made while a graph records, it
#is in that graph's pool, and calls and graphs after on the stream used it past the graph's end (cuda error 700).
#Cleared around each recording (as inductor's cuda graph trees do), a graph records a workspace of its own in its pool
def clear_cublas_workspaces() -> None:
    #a private torch function, absent from some versions
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()

#one board at a time on the gpu as a replayed cuda graph: 1,700 simulations a second on the d128 against 440 on a cpu
#thread, but processes sharing the gpu get only about 2,400 between them, so many games at once are faster on the cpu
class CudaGraphModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda")

        #recording a graph needs a few warmup calls on a side stream
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream), torch.no_grad():
            for _ in range(3):
                model(self.boards)
        torch.cuda.current_stream().wait_stream(side_stream)

        self.graph = torch.cuda.CUDAGraph()
        clear_cublas_workspaces()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.outputs = torch.cat(model(self.boards), dim=1) #the move logits, then the eval head's
        clear_cublas_workspaces()

    #boards on the cpu in, the model's two outputs on the cpu out
    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(boards) > len(self.boards):
            raise ValueError(f"{len(boards)} boards for a graph recorded for {len(self.boards)}")
        self.boards[:len(boards)].copy_(boards)
        self.graph.replay()
        outputs = self.outputs[:len(boards)].cpu()
        return outputs[:, :NUM_MOVE_IDS], outputs[:, NUM_MOVE_IDS:]

#score is for the side to move (0 lost to 1 won), line starts with the move played
@dataclass
class SearchReport:
    score: float
    line: list[chess.Move]
    simulations: int
    root_moves: list[tuple[chess.Move, int]]

#mcts.py's search, or the tablebase's move where the tables know the position (search refuses those)
def search_move(model: nn.Module, board: chess.Board, device: torch.device, simulations: int) -> tuple[chess.Move, SearchReport]:
    #imported here because mcts.py imports this file
    from engine import mcts
    if mcts.tablebase is not None and covers(board):
        best = ranked_moves(mcts.tablebase, board)[0]
        return best.move, SearchReport(wdl_score(best.wdl), [best.move], 0, [])

    root = mcts.search(model, board, device, simulations)
    line = mcts.principal_variation(root)
    chosen = root.children[line[0]]
    #a move never visited (one simulation) has no mean score: the root's own, for its side to move
    score = chosen.value_sum / chosen.visit_count if chosen.visit_count > 0 else 1 - root.value_sum / root.visit_count
    by_visits = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)[:ROOT_MOVES_REPORTED]
    return line[0], SearchReport(score, line, simulations, [(move, child.visit_count) for move, child in by_visits])

#the inverse of data.py's centipawn curve, capped since a sure win is infinitely many centipawns
def score_to_centipawns(score: float) -> int:
    score = min(max(score, 1 - MAX_REPORTED_SCORE), MAX_REPORTED_SCORE)
    return round(math.log(score / (1 - score)) / EXPECTED_SCORE_PER_CENTIPAWN)

def choose_move(model: nn.Module, board: chess.Board, device: torch.device, selection: str = "argmax",
                temperature: float = TEMPERATURE, sample_plies: int = SAMPLE_PLIES, simulations: int = SIMULATIONS) -> chess.Move:
    legal_move_ids = [encode_move_to_id(move, board.turn) for move in board.legal_moves]
    if len(legal_move_ids) == 0:
        raise ValueError(f"no legal moves, the game is over: {board.fen()}")

    if selection == "mcts":
        return search_move(model, board, device, simulations)[0]

    boards = torch.tensor(encode_board(board), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        move_logits, value_logits = model(boards)
    logits = move_logits[0]

    #illegal moves at -inf
    masked_logits = torch.full((NUM_MOVE_IDS,), float("-inf"), device=device)
    masked_logits[legal_move_ids] = logits[legal_move_ids]

    if selection == "sample" and board.ply() < sample_plies:
        probabilities = torch.softmax(masked_logits[legal_move_ids] / temperature, dim=0).tolist()
        #python's random, not torch's: torch seeds every process the same, so every game would repeat
        return decode_id_to_move(random.choices(legal_move_ids, probabilities)[0], board.turn)

    if selection == "avoid-repetition":
        #moves are tried best first, stopping at the first below ALTERNATIVE_SHARE of the best
        if expected_score(value_logits)[0].item() >= AHEAD_SCORE:
            probabilities = torch.softmax(masked_logits, dim=0)
            for move_id in masked_logits.argsort(descending=True)[:len(legal_move_ids)].tolist():
                if probabilities[move_id] < ALTERNATIVE_SHARE * probabilities.max():
                    break
                move = decode_id_to_move(move_id, board.turn)
                board.push(move)
                repeats = board.is_repetition(2)
                board.pop()
                if not repeats:
                    return move

    return decode_id_to_move(int(masked_logits.argmax().item()), board.turn)
