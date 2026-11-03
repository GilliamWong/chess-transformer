from random import Random

import chess
import numpy as np
import pytest
import torch

from model.encoding import (
    CASTLING_TOKEN_OFFSET,
    EN_PASSANT_INDEX,
    EN_PASSANT_TOKEN_OFFSET,
    EXPECTED_SCORE_PER_CENTIPAWN,
    NO_EVAL,
    NO_MOVE,
    NUM_MOVE_IDS,
    TOKENS_PER_BOARD,
    WHITE_KINGSIDE_INDEX,
    decode_id_to_move,
    encode_board,
    encode_move_to_id,
)
from training.data import BLUNDER_THRESHOLD, ChessGamesDataset, SelfPlayDataset, datasets_to_dataloaders
from training.mirror import MIRRORED_MOVE_IDS, MirrorRows, can_mirror, draw_mirrors, mirror_boards, mirror_moves
from training.train_ddp import TrainingSampler, parse_args

TEMPERATURE = 0.02


#python-chess's own mirror, to check the token and move tables against
def flip_square(square: int) -> int:
    return chess.square(7 - chess.square_file(square), chess.square_rank(square))

def flip_move(move: chess.Move) -> chess.Move:
    return chess.Move(flip_square(move.from_square), flip_square(move.to_square), move.promotion)

def flipped(board: chess.Board) -> chess.Board:
    return board.transform(chess.flip_horizontal)

def has_no_castling_right(board: chess.Board) -> bool:
    return not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK)


#en passant on the edge files and the middle for both colours, and promotions (pushing, capturing, on the edge files) for both
SPECIAL_FENS = [
    "4k3/8/8/pP6/8/8/8/4K3 w - a6 0 1",
    "4k3/8/8/6Pp/8/8/8/4K3 w - h6 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/8/Pp6/8/8/4K3 b - a3 0 1",
    "4k3/8/8/8/6pP/8/8/4K3 b - h3 0 1",
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/7p/4K1N1 b - - 0 1",
    "r1r2k2/1P1P4/8/8/8/8/1p1p4/R1R2K2 w - - 0 1",
    "r1r2k2/1P1P4/8/8/8/8/1p1p4/R1R2K2 b - - 0 1",
]

#each single castling right and all of them, for both colours to move
RIGHTS_FENS = [f"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R {turn} {rights} - 0 1" for turn in "wb" for rights in ("KQkq", "K", "Q", "k", "q")]


#the special positions, then random games' positions: random play reaches promotions, en passant and lost castling rights
@pytest.fixture(scope="module")
def positions() -> list[chess.Board]:
    boards = [chess.Board(fen) for fen in SPECIAL_FENS]
    random = Random(0)
    for _ in range(30):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 200:
            boards.append(board.copy(stack=False))
            board.push(random.choice(list(board.legal_moves)))
    return boards


@pytest.fixture(scope="module")
def castling_free(positions) -> list[chess.Board]:
    return [board for board in positions if has_no_castling_right(board)]


def test_the_positions_cover_what_a_mirror_must_get_right(castling_free):
    assert all(board.is_valid() for board in castling_free)
    assert {board.turn for board in castling_free} == {chess.WHITE, chess.BLACK}
    for turn in (chess.WHITE, chess.BLACK):
        assert any(board.turn == turn and board.has_legal_en_passant() for board in castling_free)
        promotions = {move.promotion for board in castling_free if board.turn == turn for move in board.legal_moves}
        assert promotions >= {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    #en passant on both edge files
    assert {encode_board(board)[EN_PASSANT_INDEX] - EN_PASSANT_TOKEN_OFFSET for board in castling_free} >= {chess.A6, chess.H6}


def test_mirrored_tokens_are_the_encoding_of_the_mirrored_board(castling_free):
    boards = torch.tensor([encode_board(board) for board in castling_free])
    expected = torch.tensor([encode_board(flipped(board)) for board in castling_free])

    mirrored = mirror_boards(boards, torch.ones(len(boards), dtype=torch.bool))

    wrong = (mirrored != expected).any(dim=1).nonzero().flatten().tolist()
    assert not wrong, [castling_free[row].fen() for row in wrong[:3]]
    assert (mirrored != boards).any(dim=1).float().mean() > 0.9 #the mirror does change positions: most are not symmetric


def test_mirrored_move_ids_are_the_mirrored_legal_moves_for_both_colours(castling_free):
    for board in castling_free[::3]:
        ids = [encode_move_to_id(move, board.turn) for move in board.legal_moves]

        mirrored_moves = {decode_id_to_move(MIRRORED_MOVE_IDS[move_id].item(), board.turn) for move_id in ids}

        #the whole set, so no legal move is missing a mirror and none maps outside the mirrored position's moves
        assert mirrored_moves == set(flipped(board).legal_moves), board.fen()


def test_every_move_id_mirrors_to_the_mirrored_move_and_back():
    for turn in (chess.WHITE, chess.BLACK):
        for move_id in range(NUM_MOVE_IDS):
            move = decode_id_to_move(move_id, turn)
            mirrored_id = MIRRORED_MOVE_IDS[move_id].item()
            assert decode_id_to_move(mirrored_id, turn) == flip_move(move) #promotion piece included
            assert MIRRORED_MOVE_IDS[mirrored_id].item() == move_id
    assert MIRRORED_MOVE_IDS.unique().numel() == NUM_MOVE_IDS #a permutation: no two moves share a mirror


def test_mirroring_twice_is_the_identity(castling_free):
    boards = torch.tensor([encode_board(board) for board in castling_free])
    legal_ids = [[encode_move_to_id(move, board.turn) for move in board.legal_moves][:3] for board in castling_free]
    moves = torch.tensor([ids + [NO_MOVE] * (3 - len(ids)) for ids in legal_ids]) #up to three moves, NO_MOVE after
    mirrored = torch.rand(len(boards), generator=torch.Generator().manual_seed(0)) < 0.5

    assert torch.equal(mirror_boards(mirror_boards(boards, mirrored), mirrored), boards)
    assert torch.equal(mirror_moves(mirror_moves(moves, mirrored), mirrored), moves)
    #and only the rows asked for change
    assert torch.equal(mirror_boards(boards, mirrored)[~mirrored], boards[~mirrored])
    assert torch.equal(mirror_moves(moves, mirrored)[~mirrored], moves[~mirrored])


def test_no_move_stays_no_move():
    moves = torch.tensor([[5, NO_MOVE, 1967], [NO_MOVE, NO_MOVE, NO_MOVE], [7, NO_MOVE, 0]])

    mirrored = mirror_moves(moves, torch.tensor([True, True, False]))

    assert mirrored[0].tolist() == [MIRRORED_MOVE_IDS[5].item(), NO_MOVE, MIRRORED_MOVE_IDS[1967].item()]
    assert mirrored[1].tolist() == [NO_MOVE] * 3
    assert mirrored[2].tolist() == [7, NO_MOVE, 0]


@pytest.mark.parametrize("fen", RIGHTS_FENS)
def test_a_castling_right_keeps_a_position_from_being_mirrored(fen):
    board = chess.Board(fen)
    without_rights = board.copy()
    without_rights.castling_rights = chess.BB_EMPTY

    assert not can_mirror(torch.tensor([encode_board(board)])).item()
    assert can_mirror(torch.tensor([encode_board(without_rights)])).item() #the right alone is what stops it


#N positions with three labelled moves, two blunders, two teacher moves; flip says which rows are written mirrored
def labelled_arrays(positions: list[chess.Board], chosen: dict[str, list[list[chess.Move]]], flip: list[bool]) -> dict[str, np.ndarray]:
    def ids(kind: str, places: int) -> np.ndarray:
        rows = [[encode_move_to_id(flip_move(move) if flip_row else move, board.turn) for move in moves] + [NO_MOVE] * (places - len(moves))
                for board, moves, flip_row in zip(positions, chosen[kind], flip)]
        return np.array(rows, dtype=np.int16)
    boards = np.array([encode_board(flipped(board) if flip_row else board) for board, flip_row in zip(positions, flip)], dtype=np.uint8)
    return dict(boards=boards, stockfish_moves=ids("stockfish", 3), worst_moves=ids("worst", 2), teacher_moves=ids("teacher", 2), visit_moves=ids("visits", 4))

def choose_moves(positions: list[chess.Board]) -> dict[str, list[list[chess.Move]]]:
    random = Random(1)
    legal = [list(board.legal_moves) for board in positions]
    return dict(stockfish=[random.sample(moves, 2) for moves in legal], worst=[random.sample(moves, 1) for moves in legal],
                teacher=[random.sample(moves, 2) for moves in legal], visits=[random.sample(moves, 3) for moves in legal])

#a stockfish labelled set and a self play set (and their mirror image) from positions that have 3 legal moves or more
def make_sets(positions: list[chess.Board], flip: list[bool]):
    chosen = choose_moves(positions)
    arrays = labelled_arrays(positions, chosen, flip)
    size = len(positions)
    rows = np.random.default_rng(3)
    losses = np.stack([np.full(size, 0.6), np.zeros(size)], axis=1).astype(np.float16)
    teacher = dict(teacher_moves=arrays["teacher_moves"], teacher_probabilities=np.tile(np.array([0.7, 0.3], dtype=np.float16), (size, 1)),
                   teacher_scores=rows.random(size).astype(np.float16))
    evals = np.stack([rows.integers(-200, 200, size), rows.integers(-200, 200, size), np.full(size, NO_EVAL)], axis=1).astype(np.int16)
    shares = np.tile(np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32), (size, 1))
    worst = (arrays["worst_moves"], losses)
    stockfish = ChessGamesDataset(arrays["boards"], arrays["stockfish_moves"], evals, TEMPERATURE, *worst, BLUNDER_THRESHOLD, teacher, teacher_weight=0.4)
    self_play = SelfPlayDataset(arrays["boards"], arrays["visit_moves"], shares, rows.random(size).astype(np.float32), rows.random(size).astype(np.float32),
                                result_weight=0.5, worst_moves=worst[0], worst_losses=worst[1])
    return stockfish, self_play

#the rows where the mirror coin is True take the mirrored set's tensors, the others the plain set's
def choose_rows(coins: torch.Tensor, mirrored_batch, plain_batch):
    return tuple(torch.where(coins.reshape(-1, *[1] * (plain.dim() - 1)), mirror, plain) for mirror, plain in zip(mirrored_batch, plain_batch))

@pytest.mark.parametrize("kind", [0, 1], ids=["stockfish labelled", "self play"])
def test_a_dataset_mirrors_the_board_and_every_kind_of_target_move(positions, kind):
    #rows with castling rights are in the set too: their mirror is the position itself
    rows_in_set = [board for board in positions if len(list(board.legal_moves)) >= 3]
    rows_in_set = rows_in_set[::max(1, len(rows_in_set) // 400)] + [chess.Board(fen) for fen in RIGHTS_FENS]
    eligible = [has_no_castling_right(board) for board in rows_in_set]
    assert 0 < sum(eligible) < len(eligible)
    dataset = make_sets(rows_in_set, [False] * len(rows_in_set))[kind]
    mirror_image = make_sets(rows_in_set, eligible)[kind] #written by python-chess's mirror, the independent expectation
    rows = torch.randperm(len(rows_in_set), generator=torch.Generator().manual_seed(0))[:200]
    coins = draw_mirrors(rows, 5, 0).coins
    assert coins.any() and not coins.all()

    got = dataset[MirrorRows(rows, coins)]

    wanted_when_mirrored = mirror_image[rows.tolist()]
    plain = dataset[rows.tolist()]
    wanted = choose_rows(coins, wanted_when_mirrored, plain) #for a row with a right the mirror image is the row itself
    for name, actual, expected in zip(("boards", "target moves", "target probabilities", "values", "blunders"), got, wanted):
        assert torch.equal(actual, expected), name
    #the labels moved with the boards: some rows really did change, and the values did not
    assert (got[0] != plain[0]).any() and (got[1] != plain[1]).any() and (got[4] != plain[4]).any()
    assert torch.equal(got[2], plain[2]) and torch.equal(got[3], plain[3])
    #no coin up, nothing changes: a plain index and a batch of False coins are the same read
    untouched = dataset[MirrorRows(rows, torch.zeros_like(coins))]
    assert all(torch.equal(actual, expected) for actual, expected in zip(untouched, plain))


def test_padding_places_stay_padding(positions):
    rows_in_set = [board for board in positions if has_no_castling_right(board) and len(list(board.legal_moves)) >= 3][:50]
    dataset, _ = make_sets(rows_in_set, [False] * len(rows_in_set))
    rows = torch.arange(len(rows_in_set))

    _, moves, probabilities, _, blunders = dataset[MirrorRows(rows, torch.ones(len(rows), dtype=torch.bool))]

    #the third stockfish place and the second blunder place are empty: move 0 at probability 0, and no blunder
    assert moves[:, 2].eq(0).all() and probabilities[:, 2].eq(0).all()
    assert blunders[:, 1].eq(NO_MOVE).all()


#the coins: fair, independent of each other, and fixed
def test_a_rows_coin_depends_on_its_seed_epoch_and_row_alone():
    rows = torch.arange(1000)
    whole = draw_mirrors(rows, 42, 3).coins

    shuffled = torch.randperm(1000, generator=torch.Generator().manual_seed(0))
    assert torch.equal(draw_mirrors(rows[shuffled], 42, 3).coins, whole[shuffled])
    assert torch.equal(torch.cat([draw_mirrors(rows[:300], 42, 3).coins, draw_mirrors(rows[300:], 42, 3).coins]), whole)
    assert not torch.equal(draw_mirrors(rows, 42, 4).coins, whole) and not torch.equal(draw_mirrors(rows, 43, 3).coins, whole)


def test_coins_are_fair_and_independent_across_rows_epochs_and_seeds():
    rows = torch.arange(200_000)
    coins = {(seed, epoch): draw_mirrors(rows, seed, epoch).coins for seed in (42, 43) for epoch in range(7)}
    #the largest gap from a half that fair coins would show is about 4 standard deviations (0.0011 each); agreement of two
    #independent fair coins is a half too
    def gap(agreement: torch.Tensor) -> float:
        return abs(agreement.float().mean().item() - 0.5)
    assert all(gap(flips) < 0.005 for flips in coins.values())
    first = coins[42, 0]
    assert all(gap(first == coins[42, epoch]) < 0.005 for epoch in range(1, 7))
    assert gap(first == coins[43, 0]) < 0.005
    assert gap(first[1:] == first[:-1]) < 0.005 and gap(first[2:] == first[:-2]) < 0.005
    #over the 7 epochs of a run most rows come up both ways, as 7 fair coins would have it (1 - 2 / 128)
    times_mirrored = torch.stack([coins[42, epoch] for epoch in range(7)]).sum(dim=0)
    assert abs(((times_mirrored > 0) & (times_mirrored < 7)).float().mean().item() - (1 - 2 / 128)) < 0.005


#a resumed run draws the coins this function gives: changing it changes the data of every saved run
def test_the_coins_are_pinned():
    def bits(rows: list[int], seed: int, epoch: int) -> str:
        return "".join("1" if coin else "0" for coin in draw_mirrors(torch.tensor(rows), seed, epoch).coins.tolist())
    assert bits(list(range(64)), 42, 3) == "0001101010110010000111100100110000111000110000101011001000100101"
    assert bits([0, 1, 1_000_000, 1_690_000_000, 4_000_000_000], 7, 0) == "10011"


#every (row, coin) pair of an epoch's steps, all ranks together
def epoch_steps(world_size: int, epoch: int, skip: int = 0, mirror: bool = True, size: int = 131, batch_size: int = 16, seed: int = 42):
    samplers = [TrainingSampler(list(range(size)), batch_size, rank, world_size, seed, mirror) for rank in range(world_size)]
    for sampler in samplers:
        sampler.epoch, sampler.skip = epoch, skip
    per_rank = [list(sampler) for sampler in samplers]
    if not mirror:
        return [dict.fromkeys(torch.cat([batches[step] for batches in per_rank]).tolist()) for step in range(len(per_rank[0]))]
    return [dict(zip(torch.cat([batches[step].rows for batches in per_rank]).tolist(), torch.cat([batches[step].coins for batches in per_rank]).tolist()))
            for step in range(len(per_rank[0]))]


def test_a_resumed_epoch_draws_the_coins_of_one_that_was_not_stopped():
    uninterrupted = epoch_steps(2, epoch=3)
    #the resume path restores the epoch and the batches of it already trained on
    assert epoch_steps(2, epoch=3, skip=5) == uninterrupted[5:]
    assert len(uninterrupted) == 8


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_a_resumed_run_with_another_process_count_draws_the_same_coins(world_size):
    #the same global batches and rows, cut between the processes another way
    assert epoch_steps(world_size, epoch=2) == epoch_steps(2, epoch=2)


def test_ranks_get_disjoint_rows_and_each_rows_coin_is_the_functions():
    steps = epoch_steps(4, epoch=1)
    rows = [row for step in steps for row in step]
    assert len(rows) == len(set(rows)) == 8 * 16 #no row twice, so no two ranks trained on the same one
    for step in steps:
        coins = draw_mirrors(torch.tensor(list(step)), 42, 1).coins.tolist()
        assert list(step.values()) == coins


def test_mirroring_leaves_the_shuffle_alone_and_changes_the_coins_every_epoch():
    assert [list(step) for step in epoch_steps(2, epoch=1)] == [list(step) for step in epoch_steps(2, epoch=1, mirror=False)]
    first, second = epoch_steps(2, epoch=0), epoch_steps(2, epoch=1)
    common = set(row for step in first for row in step) & set(row for step in second for row in step)
    coin_of = lambda steps: {row: coin for step in steps for row, coin in step.items()}
    assert any(coin_of(first)[row] != coin_of(second)[row] for row in common)


#rows 0..n-1 as boards that can be mirrored; the value target is the row's eval, so any batch says which rows it holds
def numbered_set(size: int, rights_token: int = CASTLING_TOKEN_OFFSET) -> ChessGamesDataset:
    boards = (np.arange(size)[:, None] + np.arange(TOKENS_PER_BOARD)[None, :]) % 13
    boards[:, WHITE_KINGSIDE_INDEX:WHITE_KINGSIDE_INDEX + 4] = rights_token
    boards[:, EN_PASSANT_INDEX] = EN_PASSANT_TOKEN_OFFSET
    moves = np.stack([np.arange(size), np.full(size, NO_MOVE)], axis=1).astype(np.int16)
    evals = np.stack([np.arange(size), np.full(size, NO_EVAL)], axis=1).astype(np.int16)
    return ChessGamesDataset(boards.astype(np.uint8), moves, evals, TEMPERATURE)

def rows_of(values: torch.Tensor) -> list[int]:
    return (torch.logit(values) / EXPECTED_SCORE_PER_CENTIPAWN).round().long().tolist()


def test_the_train_loader_mirrors_a_new_set_of_rows_every_pass_and_the_val_loader_never():
    dataset = numbered_set(100)
    train_loader, val_loader = datasets_to_dataloaders(dataset, dataset, train_batch_size=100, val_batch_size=100, seed=7, mirror=True)
    plain_boards = dataset[list(range(100))][0]
    mirrored_boards = mirror_boards(plain_boards, torch.ones(100, dtype=torch.bool))
    assert not torch.equal(plain_boards, mirrored_boards)

    passes = []
    for epoch in range(2):
        boards, _, _, values, _ = next(iter(train_loader))
        rows = rows_of(values)
        coins = draw_mirrors(torch.tensor(rows), 7, epoch).coins.tolist() #the pass number is the epoch
        for board, row, coin in zip(boards, rows, coins):
            assert torch.equal(board, mirrored_boards[row] if coin else plain_boards[row])
        passes.append(dict(zip(rows, coins)))
    assert passes[0] != passes[1]

    val_boards, _, _, _, _ = next(iter(val_loader))
    assert torch.equal(val_boards, plain_boards)


def test_rows_with_castling_rights_and_the_shuffle_come_out_as_without_the_flag():
    dataset = numbered_set(100, rights_token=CASTLING_TOKEN_OFFSET + 1) #every row holds a right
    plain_loader, _ = datasets_to_dataloaders(dataset, dataset, train_batch_size=16, seed=7)
    mirror_loader, _ = datasets_to_dataloaders(dataset, dataset, train_batch_size=16, seed=7, mirror=True)

    for plain, mirrored in zip(plain_loader, mirror_loader):
        assert all(torch.equal(first, second) for first, second in zip(plain, mirrored))
    assert len(plain_loader) == len(mirror_loader) == 7


def test_mirroring_without_a_seed_is_rejected():
    dataset = numbered_set(10)

    with pytest.raises(ValueError):
        datasets_to_dataloaders(dataset, dataset, mirror=True)


def test_mirroring_is_off_unless_asked_for():
    assert parse_args([]).mirror is False
    assert parse_args(["--mirror"]).mirror is True
    assert TrainingSampler(list(range(32)), 16, 0, 1, 0).mirror is False
