from typing import NamedTuple

import chess
import numpy as np
import torch

from model.encoding import (
    BLACK_QUEENSIDE_INDEX,
    BOARD_VOCAB_SIZE,
    CASTLING_TOKEN_OFFSET,
    EN_PASSANT_INDEX,
    EN_PASSANT_TOKEN_OFFSET,
    NO_MOVE,
    NUM_MOVE_IDS,
    NUM_SQUARES,
    TOKENS_PER_BOARD,
    WHITE_KINGSIDE_INDEX,
    decode_id_to_move,
    encode_move_to_id,
)

#left-right mirror (files a <-> h); castling is not symmetric, so only positions without a castling right are mirrored
#it keeps every square's rank, so it commutes with encode_board's rank flip for black: one table serves both colours
FILE_FLIP = 7 #square ^ 7 is the square on the other side of its rank

#the slot each slot of a mirrored board is copied from: squares by file, metadata unchanged
MIRRORED_SLOTS = torch.tensor([square ^ FILE_FLIP for square in range(NUM_SQUARES)] + list(range(NUM_SQUARES, TOKENS_PER_BOARD)))


#en passant squares are on the sixth rank (side to move as white); every other token maps to itself
def _build_mirrored_en_passant() -> torch.Tensor:
    table = torch.arange(BOARD_VOCAB_SIZE)
    for square in range(chess.A6, chess.H6 + 1):
        table[EN_PASSANT_TOKEN_OFFSET + square] = EN_PASSANT_TOKEN_OFFSET + (square ^ FILE_FLIP)
    return table

MIRRORED_EN_PASSANT = _build_mirrored_en_passant()


#move ids are moves of the side to move seen as white
def _build_mirrored_move_ids() -> torch.Tensor:
    moves = (decode_id_to_move(move_id, chess.WHITE) for move_id in range(NUM_MOVE_IDS))
    return torch.tensor([
        encode_move_to_id(chess.Move(move.from_square ^ FILE_FLIP, move.to_square ^ FILE_FLIP, move.promotion), chess.WHITE) for move in moves
    ])

MIRRORED_MOVE_IDS = _build_mirrored_move_ids()


#True where neither side has a castling right
def can_mirror(boards: torch.Tensor) -> torch.Tensor:
    return (boards[..., WHITE_KINGSIDE_INDEX:BLACK_QUEENSIDE_INDEX + 1] == CASTLING_TOKEN_OFFSET).all(dim=-1)

#flips the rows where mirrored is True; the caller checks can_mirror
def mirror_boards(boards: torch.Tensor, mirrored: torch.Tensor | None) -> torch.Tensor:
    if mirrored is None:
        return boards
    flipped = boards[..., MIRRORED_SLOTS]
    flipped[..., EN_PASSANT_INDEX] = MIRRORED_EN_PASSANT[flipped[..., EN_PASSANT_INDEX]]
    return torch.where(mirrored[..., None], flipped, boards)

#maps the ids of the rows where mirrored is True; NO_MOVE is not in the table, so it is masked and kept
def mirror_moves(moves: torch.Tensor, mirrored: torch.Tensor | None) -> torch.Tensor:
    if mirrored is None:
        return moves
    no_move = moves == NO_MOVE
    flipped = MIRRORED_MOVE_IDS[moves.masked_fill(no_move, 0)]
    return torch.where(mirrored[..., None] & ~no_move, flipped, moves)


#a batch's rows with a coin each: a row whose coin is True is shown mirrored if it can be
class MirrorRows(NamedTuple):
    rows: torch.Tensor
    coins: torch.Tensor

#a coin is a fixed hash of (seed, epoch, row), so it is the same for any process, batching or resume; int64 products stay below 2 ** 63
def draw_mirrors(rows: torch.Tensor, seed: int, epoch: int) -> MirrorRows:
    key = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
    hashed = rows.long() ^ key
    for _ in range(2):
        hashed = (((hashed >> 16) ^ hashed) * 0x45D9F3B) & 0xFFFFFFFF
    return MirrorRows(rows, (((hashed >> 16) ^ hashed) & 1).bool())

#(rows, boards as shown, mirrored mask); the mask is None for a plain index, which reads the set as it is
def read_boards(all_boards: torch.Tensor, index: int | list[int] | MirrorRows) -> tuple[int | list[int] | torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(index, MirrorRows):
        return index, all_boards[index].long(), None
    boards = all_boards[index.rows].long()
    mirrored = index.coins & can_mirror(boards)
    return index.rows, mirror_boards(boards, mirrored), mirrored
