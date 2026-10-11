#the int8 weights of model/int8_blocks.py, rounded once per checkpoint by GPTQ (Frantar et al. 2022): each weight's
#rounding error is made up for by the weights not yet rounded, through the inverse Hessian of the matmul's inputs on
#CALIBRATION_POSITIONS positions. Computed in fp64, so that machines summing in other orders round them the same (the
#cpu and an RTX 5070 Ti agree on every weight of the d768), and kept in CACHE named by the checkpoint file's sha256
#and VERSION.
#   python -m engine.quantization <model.pt>    computes them ahead of a match (~1 min on an RTX 5070 Ti)
import argparse
import hashlib
import os
from pathlib import Path

import chess.pgn
import numpy as np
import torch

from engine.player import load_model
from model.encoding import encode_board
from model.int8_blocks import HADAMARD_BLOCK, down_inputs, hadamard
from model.model import ChessTransformer
from tools.play.evaluate_elo import CARLSEN_PGN

#derived from the checkpoint and remade when missing: kept in the home folder rather than beside the checkpoint, whose
#folder may be read-only, and found by every hyak node, whose home folders are shared
CACHE = Path.home() / ".cache" / "chess_transformer" / "int8"
#changed with the rounding, the calibration or the layers in int8, so that weights rounded the old way are not read
VERSION = 1
#every CALIBRATION_STEP plies of Carlsen's games from CALIBRATION_FIRST_GAME on: the first games are the tests' and the
#accuracy benchmark's
CALIBRATION_POSITIONS = 1024
CALIBRATION_FIRST_GAME = 1000
CALIBRATION_STEP = 9
#GPTQ's dampening of the Hessian, a share of its mean diagonal, and the columns it rounds before passing on their error
DAMPENING = 0.01
GPTQ_COLUMNS = 128
INT8_MAX = 127


def calibration_boards() -> torch.Tensor:
    boards = []
    with open(CARLSEN_PGN) as pgn:
        for _ in range(CALIBRATION_FIRST_GAME):
            chess.pgn.skip_game(pgn)
        while len(boards) < CALIBRATION_POSITIONS and (game := chess.pgn.read_game(pgn)) is not None:
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if board.is_game_over() or len(boards) == CALIBRATION_POSITIONS:
                    break
                if ply % CALIBRATION_STEP == CALIBRATION_STEP // 2:
                    boards.append(encode_board(board))
    if len(boards) < CALIBRATION_POSITIONS:
        raise ValueError(f"{CARLSEN_PGN} has {len(boards)} calibration positions past game {CALIBRATION_FIRST_GAME}, not {CALIBRATION_POSITIONS}")
    return torch.from_numpy(np.array(boards)).long()


#the weight (N x K) rounded to int8 with a scale per row, its absmax / INT8_MAX, by GPTQ on the Hessian (K x K) of its
#inputs: the columns are rounded in order, each one's error spread over the columns after it. Returns (int8, scale)
def gptq(weight: torch.Tensor, hessian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, hessian = weight.double().clone(), hessian.double().clone()
    #inputs that are always zero (the down matmul's padding) have no curvature: their weights are left at zero
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0
    hessian += DAMPENING * torch.diag(hessian).mean() * torch.eye(len(hessian), device=hessian.device, dtype=hessian.dtype)
    #the inverse's upper Cholesky factor: its row i spreads column i's error over the columns after it
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)
    scale = weight.abs().amax(dim=1).clamp(min=1e-20) / INT8_MAX
    integers = torch.zeros_like(weight)
    for first in range(0, weight.shape[1], GPTQ_COLUMNS):
        last = min(first + GPTQ_COLUMNS, weight.shape[1])
        columns, block_inverse = weight[:, first:last].clone(), inverse[first:last, first:last]
        errors = torch.zeros_like(columns)
        for i in range(last - first):
            rounded = (columns[:, i] / scale).round().clamp(-INT8_MAX, INT8_MAX)
            integers[:, first + i] = rounded
            errors[:, i] = (columns[:, i] - rounded * scale) / block_inverse[i, i]
            columns[:, i:] -= errors[:, i, None] * block_inverse[i, i:][None, :]
        weight[:, last:] -= errors @ inverse[first:last, last:]
    return integers.to(torch.int8), scale.float()


#each block's matmul inputs' x^T x summed over the calibration boards, in fp64: (projection, gate_up, down) per block
@torch.no_grad()
def input_hessians(model: ChessTransformer, boards: torch.Tensor, batch: int = 128) -> list[dict[str, torch.Tensor]]:
    hessians = [{} for _ in model.encoder]
    hooks = []
    for block, sums in zip(model.encoder, hessians):
        for name, layer in (("projection", block.Q), ("gate_up", block.mlp.gate_projection), ("down", block.mlp.down_projection)):
            def add(module, inputs, name=name, sums=sums):
                x = inputs[0].reshape(-1, inputs[0].shape[-1])
                sums[name] = sums.get(name, 0) + x.t() @ x
            hooks.append(layer.register_forward_pre_hook(add))
    try:
        for start in range(0, len(boards), batch):
            model(boards[start:start + batch].to(model.position_embed.device))
    finally:
        for hook in hooks:
            hook.remove()
    return hessians


#int8 weights of every block but the last (its down matmul alone costs the policy as much as all the others'),
#{layer: {name: (int8 N x K, scale per row)}}: projection is Q|K|V|template mix as fused_blocks joins them, gate_up the
#gate's rows then up's, down the mlp's down with its inputs padded to down_inputs and rotated as Int8Block rotates them
@torch.no_grad()
def compute_int8_weights(model: ChessTransformer, boards: torch.Tensor) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    hessians = input_hessians(model, boards)
    weights = {}
    for layer, (block, sums) in enumerate(list(zip(model.encoder, hessians))[:-1]):
        projection = torch.cat([layer_.weight for layer_ in (block.Q, block.K, block.V, block.template_mix)])
        gate_up = torch.cat([block.mlp.gate_projection.weight, block.mlp.up_projection.weight])
        hidden = block.mlp.down_projection.weight.shape[1]
        width = down_inputs(hidden)
        rotation = torch.block_diag(*[hadamard(HADAMARD_BLOCK)] * (width // HADAMARD_BLOCK)).to(sums["down"])
        padded = torch.zeros(width, width, device=rotation.device, dtype=rotation.dtype)
        padded[:hidden, :hidden] = sums["down"]
        down = torch.nn.functional.pad(block.mlp.down_projection.weight, (0, width - hidden)) @ rotation / HADAMARD_BLOCK
        weights[layer] = {"projection": gptq(projection, sums["projection"]), "gate_up": gptq(gate_up, sums["gate_up"]),
                          "down": gptq(down, rotation.t() @ padded @ rotation)}
    return weights


def checkpoint_digest(checkpoint_path: Path) -> str:
    with open(checkpoint_path, "rb") as checkpoint:
        return hashlib.file_digest(checkpoint, "sha256").hexdigest()


#the checkpoint's int8 weights (compute_int8_weights) on the cpu, from CACHE or computed on the gpu and cached
def int8_weights(checkpoint_path: Path) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    path = CACHE / f"{checkpoint_digest(checkpoint_path)}_v{VERSION}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    #fp64 but the heads, whose inputs the model casts to fp32
    model = load_model(checkpoint_path, torch.device("cuda")).half_except_heads(torch.float64)
    weights = {layer: {name: tuple(tensor.cpu() for tensor in pair) for name, pair in matmuls.items()}
               for layer, matmuls in compute_int8_weights(model, calibration_boards()).items()}
    del model
    torch.cuda.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    #written whole under another name first: engines starting together each compute the same weights
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(weights, partial)
    os.replace(partial, path)
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the int8 weights of a checkpoint, computed and cached")
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    int8_weights(arguments.checkpoint)
    print(CACHE / f"{checkpoint_digest(arguments.checkpoint)}_v{VERSION}.pt")
