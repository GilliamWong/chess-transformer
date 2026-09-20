from pathlib import Path
import chess
import torch
from torch import nn
from encoding import NUM_MOVE_IDS, encode_board, encode_move_to_id, decode_id_to_move
from model import ChessTransformer

#loads the weights that main.py saved. the model comes back in eval mode, ready to play
def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer:
    model = ChessTransformer()
    #map_location so a checkpoint saved on the gpu also loads on a machine without one
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model

#the model scores all 1968 possible moves, most of which are illegal in any given position.
#every illegal move is set to -inf so the argmax can only ever pick a legal one
def choose_move(model: nn.Module, board: chess.Board, device: torch.device) -> chess.Move:
    legal_move_ids = [encode_move_to_id(move) for move in board.legal_moves]
    if len(legal_move_ids) == 0:
        raise ValueError(f"no legal moves, the game is over: {board.fen()}")

    #the model always expects a batch, so one board is a batch of 1
    boards = torch.tensor(encode_board(board), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(boards)[0]

    masked_logits = torch.full((NUM_MOVE_IDS,), float("-inf"), device=device)
    masked_logits[legal_move_ids] = logits[legal_move_ids]

    return decode_id_to_move(int(masked_logits.argmax().item()))
