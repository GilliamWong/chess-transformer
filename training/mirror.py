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
    NUM_MOVE_IDS,
    NUM_SQUARES,
    TOKENS_PER_BOARD,
    WHITE_KINGSIDE_INDEX,
    decode_id_to_move,
    encode_move_to_id,
)
from tools.label_with_stockfish import NO_MOVE

#left-right mirror augmentation (files a <-> h; encoding.mirror_move flips ranks, for black). Castling is the one rule
#that is not symmetric, so only positions without a castling right for either side are mirrored. A mirror keeps every
#square's rank, so it commutes with encode_board's rank flip for black: one table of each kind serves both colours
FILE_FLIP = 7 #square ^ 7 is the square on the other side of its rank

#the slot of the board each slot of a mirrored board is copied from: the squares by file, the metadata where it is
MIRRORED_SLOTS = torch.tensor([square ^ FILE_FLIP for square in range(NUM_SQUARES)] + list(range(NUM_SQUARES, TOKENS_PER_BOARD)))


#an en passant square is on the sixth rank, where the side to move captures: it goes to the other end of its rank. Every
#other token stays
def _build_mirrored_en_passant() -> torch.Tensor:
    table = torch.arange(BOARD_VOCAB_SIZE)
    for square in range(chess.A6, chess.H6 + 1):
        table[EN_PASSANT_TOKEN_OFFSET + square] = EN_PASSANT_TOKEN_OFFSET + (square ^ FILE_FLIP)
    return table

MIRRORED_EN_PASSANT = _build_mirrored_en_passant()


#the ids are moves of the side to move seen as white, so the mirror of an id is the id of the mirrored move
def _build_mirrored_move_ids() -> torch.Tensor:
    mirrored = []
    for move_id in range(NUM_MOVE_IDS):
        move = decode_id_to_move(move_id, chess.WHITE)
        mirrored.append(encode_move_to_id(chess.Move(move.from_square ^ FILE_FLIP, move.to_square ^ FILE_FLIP, move.promotion), chess.WHITE))
    return torch.tensor(mirrored)

MIRRORED_MOVE_IDS = _build_mirrored_move_ids()


#boards (..., TOKENS_PER_BOARD) with neither side holding a castling right, as a bool for each
def can_mirror(boards: torch.Tensor) -> torch.Tensor:
    return (boards[..., WHITE_KINGSIDE_INDEX:BLACK_QUEENSIDE_INDEX + 1] == CASTLING_TOKEN_OFFSET).all(dim=-1)

#long boards (..., TOKENS_PER_BOARD) with the rows where mirrored is True flipped. The caller checks can_mirror
def mirror_boards(boards: torch.Tensor, mirrored: torch.Tensor | None) -> torch.Tensor:
    if mirrored is None:
        return boards
    flipped = boards[..., MIRRORED_SLOTS]
    flipped[..., EN_PASSANT_INDEX] = MIRRORED_EN_PASSANT[flipped[..., EN_PASSANT_INDEX]]
    return torch.where(mirrored[..., None], flipped, boards)

#long move ids (..., places) with the rows where mirrored is True mapped to their mirror. NO_MOVE stays NO_MOVE
def mirror_moves(moves: torch.Tensor, mirrored: torch.Tensor | None) -> torch.Tensor:
    if mirrored is None:
        return moves
    no_move = moves == NO_MOVE
    flipped = MIRRORED_MOVE_IDS[moves.masked_fill(no_move, 0)]
    return torch.where(mirrored[..., None] & ~no_move, flipped, moves)


#the rows of one batch and a coin for each, which a sampler hands a dataset to show the batch mirrored at random:
#a row whose coin is True is mirrored if it can be
class MirrorRows(NamedTuple):
    rows: torch.Tensor
    coins: torch.Tensor

#The coin is a fixed function of (seed, epoch, row): a 32 bit integer hash of the row mixed with a key from the seed and
#epoch. So a row's coin does not depend on which process draws it, how the epoch is cut into batches, or whether the run
#was stopped and resumed. int64 arithmetic with products below 2 ** 63
def draw_mirrors(rows: torch.Tensor, seed: int, epoch: int) -> MirrorRows:
    key = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
    hashed = rows.long() ^ key
    for _ in range(2):
        hashed = (((hashed >> 16) ^ hashed) * 0x45D9F3B) & 0xFFFFFFFF
    return MirrorRows(rows, (((hashed >> 16) ^ hashed) & 1).bool())

#a dataset's index as its rows, the boards of those rows as they are shown, and which of them are mirrored (None for a
#plain index, which reads the set as it is)
def read_boards(all_boards: torch.Tensor, index: int | list[int] | MirrorRows) -> tuple[int | list[int] | torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(index, MirrorRows):
        return index, all_boards[index].long(), None
    boards = all_boards[index.rows].long()
    mirrored = index.coins & can_mirror(boards)
    return index.rows, mirror_boards(boards, mirrored), mirrored
