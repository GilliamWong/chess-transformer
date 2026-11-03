import argparse
import hashlib
import os
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn

from engine.inference import FP16_RESIDUAL_SCALE, answer_rows, fp16_model
from model.checkpoint import load_model
from model.encoding import NUM_MOVE_IDS, encode_board, encode_move_to_id
from model.fused_blocks import link_input_mixes
from model.int8_blocks import DOWN_INPUT_MULTIPLE, HADAMARD_BLOCK, Int8Block, down_inputs, hadamard
from model.model import ChessTransformer

Int8Weights = dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]

CARLSEN_PGN = Path(__file__).resolve().parent / "Carlsen.pgn"
#not the checkpoint's folder: that may be read-only
CACHE = Path.home() / ".cache" / "chess_transformer" / "int8"
#bumped with the rounding code: it and the settings below name the cached file
VERSION = 1
CALIBRATION_POSITIONS = 1024
#the games before are the tests'
CALIBRATION_FIRST_GAME = 1000
CALIBRATION_STEP = 9
DAMPENING = 0.01
GPTQ_COLUMNS = 128
FP16_LAST_LAYERS = 1
INT8_MAX = 127
#bumped with block_errors, like VERSION
CHOICE_VERSION = 1
ERROR_BATCH = 256


def calibration_positions() -> list[chess.Board]:
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
                    boards.append(board.copy(stack=False))
    if len(boards) < CALIBRATION_POSITIONS:
        raise ValueError(f"{CARLSEN_PGN} has {len(boards)} calibration positions past game {CALIBRATION_FIRST_GAME}, not {CALIBRATION_POSITIONS}")
    return boards


def encode_positions(positions: list[chess.Board]) -> torch.Tensor:
    return torch.from_numpy(np.array([encode_board(board) for board in positions])).long()


#GPTQ (Frantar et al. 2022): (int8 weight, scale a row) of an N x K weight, each column's error spread over the later ones through the Hessian
def gptq(weight: torch.Tensor, hessian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #fp64: machines summing in other orders round every weight alike
    weight, hessian = weight.double().clone(), hessian.double().clone()
    diagonal = hessian.diagonal()
    #inputs that are always zero (the down matmul's padding) have no curvature: their weights are left at zero
    dead = diagonal == 0
    diagonal[dead] = 1
    weight[:, dead] = 0
    diagonal += DAMPENING * diagonal.mean()
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
            columns[:, i:] -= errors[:, i, None] * block_inverse[i, i:]
        weight[:, last:] -= errors @ inverse[first:last, last:]
    return integers.to(torch.int8), scale.float()


#per block, the sum over the boards of x^T x for the inputs x of its projection, gate_up and down matmuls
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


#all but the last FP16_LAST_LAYERS blocks, laid out as Int8Block reads them: projection is Q|K|V|template mix,
#gate_up the gate's rows then up's, down padded to down_inputs and rotated
@torch.no_grad()
def compute_int8_weights(model: ChessTransformer, boards: torch.Tensor) -> Int8Weights:
    hessians = input_hessians(model, boards)
    weights = {}
    for layer in range(len(hessians) - FP16_LAST_LAYERS):
        block, sums = model.encoder[layer], hessians[layer]
        projection = torch.cat([matmul.weight for matmul in (block.Q, block.K, block.V, block.template_mix)])
        gate_up = torch.cat([block.mlp.gate_projection.weight, block.mlp.up_projection.weight])
        hidden = block.mlp.down_projection.weight.shape[1]
        width = down_inputs(hidden)
        padding = width - hidden
        rotation = torch.block_diag(*[hadamard(HADAMARD_BLOCK)] * (width // HADAMARD_BLOCK)).to(sums["down"])
        padded = nn.functional.pad(sums["down"], (0, padding, 0, padding))
        down = nn.functional.pad(block.mlp.down_projection.weight, (0, padding)) @ rotation / HADAMARD_BLOCK
        weights[layer] = {"projection": gptq(projection, sums["projection"]), "gate_up": gptq(gate_up, sums["gate_up"]),
                          "down": gptq(down, rotation.t() @ padded @ rotation)}
    return weights


#per layer of weights, the total variation of the policy over the legal moves between the fp16 model and the fp16 model
#with only that block in int8, mean over the calibration boards
@torch.no_grad()
def block_errors(checkpoint_path: Path, weights: Int8Weights) -> dict[int, float]:
    positions = calibration_positions()
    boards = encode_positions(positions).cuda()
    legal = torch.zeros(len(positions), NUM_MOVE_IDS, dtype=torch.bool)
    for row, board in enumerate(positions):
        legal[row, [encode_move_to_id(move, board.turn) for move in board.legal_moves]] = True
    legal = legal.cuda()
    model = fp16_model(checkpoint_path)
    fp16_blocks = list(model.encoder)

    def priors(int8_blocks: dict[int, nn.Module]) -> torch.Tensor:
        blocks = [int8_blocks.get(layer, block) for layer, block in enumerate(fp16_blocks)]
        link_input_mixes(blocks)
        model.encoder = nn.ModuleList(blocks)
        logits = torch.cat([answer_rows(model(boards[start:start + ERROR_BATCH]))[:, :-1] for start in range(0, len(boards), ERROR_BATCH)])
        return torch.softmax(logits.double().masked_fill(~legal, -torch.inf), dim=-1)

    fp16_priors = priors({})
    errors = {}
    for layer, matmuls in weights.items():
        int8_priors = priors({layer: Int8Block(fp16_blocks[layer], matmuls, FP16_RESIDUAL_SCALE)})
        errors[layer] = 0.5 * (int8_priors - fp16_priors).abs().sum(dim=-1).mean().item()
    del model
    torch.cuda.empty_cache()
    return errors


#a count of 0 stays out of the name so the files cached without a count are still read
def cached_weights_path(checkpoint_path: Path, fp16_blocks: int) -> Path:
    with open(checkpoint_path, "rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    rounding = (VERSION, CALIBRATION_POSITIONS, CALIBRATION_FIRST_GAME, CALIBRATION_STEP, DAMPENING, GPTQ_COLUMNS, FP16_LAST_LAYERS,
                HADAMARD_BLOCK, DOWN_INPUT_MULTIPLE)
    if fp16_blocks:
        rounding += (CHOICE_VERSION, fp16_blocks)
    return CACHE / f"{digest}_{hashlib.sha256(repr(rounding).encode()).hexdigest()[:16]}.pt"


#the checkpoint's int8 weights on the cpu, from CACHE or computed on the gpu and cached, without the fp16_blocks blocks
#of the largest block_errors
def int8_weights(checkpoint_path: Path, fp16_blocks: int) -> Int8Weights:
    if fp16_blocks < 0:
        raise ValueError(f"a count of blocks to keep in fp16, not {fp16_blocks}")
    path = cached_weights_path(checkpoint_path, fp16_blocks)
    if path.exists():
        return torch.load(path, weights_only=True)
    if fp16_blocks:
        weights = int8_weights(checkpoint_path, 0)
        errors = block_errors(checkpoint_path, weights)
        kept = sorted(errors, key=errors.get, reverse=True)[:fp16_blocks]
        weights = {layer: matmuls for layer, matmuls in weights.items() if layer not in kept}
    else:
        #fp64 but the heads, whose inputs the model casts to fp32
        model = load_model(checkpoint_path, torch.device("cuda")).half_except_heads(torch.float64)
        weights = {layer: {name: tuple(tensor.cpu() for tensor in pair) for name, pair in matmuls.items()}
                   for layer, matmuls in compute_int8_weights(model, encode_positions(calibration_positions())).items()}
        del model
        torch.cuda.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    #written under another name first: engines starting together compute the same weights
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(weights, partial)
    os.replace(partial, path)
    return weights


if __name__ == "__main__":
    from engine.engine_search import ENGINE_FP16_BLOCKS
    parser = argparse.ArgumentParser(description="the int8 weights of a checkpoint, computed and cached")
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    weights = int8_weights(arguments.checkpoint, ENGINE_FP16_BLOCKS)
    print(cached_weights_path(arguments.checkpoint, ENGINE_FP16_BLOCKS))
    print("kept in fp16 besides the last:", sorted(set(int8_weights(arguments.checkpoint, 0)) - set(weights)))
