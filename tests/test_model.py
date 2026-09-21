import pytest
import torch
from torch import nn

from data import ChessGamesDataset, game_to_examples
from encoding import NUM_MOVE_IDS, PROMOTION_MOVE_IDS, TOKENS_PER_BOARD
from model import ChessTransformer

BATCH_SIZE = 16


def small_model() -> ChessTransformer:
    """small enough to train for a few hundred steps inside a test"""
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=2, n_heads=2, dropout=0.0)


@pytest.fixture(scope="module")
def batch(carlsen_games) -> tuple[torch.Tensor, torch.Tensor]:
    """the first positions of a real game, with the moves played from them"""
    dataset = ChessGamesDataset([game_to_examples(carlsen_games[0])])
    boards, moves = zip(*(dataset[index] for index in range(BATCH_SIZE)))
    return torch.stack(boards), torch.stack(moves)


def test_one_set_of_move_logits_per_board(batch):
    boards, _ = batch

    logits = small_model()(boards)

    #not (batch, TOKENS_PER_BOARD, NUM_MOVE_IDS): the tokens have to be pooled into one answer per board
    assert boards.shape == (BATCH_SIZE, TOKENS_PER_BOARD)
    assert logits.shape == (BATCH_SIZE, NUM_MOVE_IDS)


#without the positional embedding the model sees which pieces exist but not where they are
def test_moving_a_piece_changes_the_output(batch):
    model = small_model().eval()
    start_position = batch[0][:1]
    assert start_position[0, 1] != start_position[0, 12]

    swapped = start_position.clone()
    swapped[0, 1], swapped[0, 12] = start_position[0, 12], start_position[0, 1]  #knight b1 <-> pawn e2

    #with no position information the two outputs differ only by float rounding, around 1e-7
    difference = (model(start_position) - model(swapped)).abs().max().item()
    assert difference > 1e-4


#without batch_first=True attention runs across the boards of a batch instead of the tokens of a board
def test_a_board_gets_the_same_logits_alone_or_in_a_batch(batch):
    model = small_model().eval()
    boards, _ = batch

    alone = model(boards[:1])
    in_a_batch = model(boards)[:1]

    assert torch.allclose(alone, in_a_batch, atol=1e-5)


#something added to every move changes no probability but can grow without limit, and did: training in bf16
#collapsed once that constant passed 256. The promotion term may only reach the promotions
def test_promotion_term_only_changes_promotion_logits(batch):
    model = small_model().eval()
    boards, _ = batch
    before = model(boards)

    with torch.no_grad():
        model.promotion_projection.bias += 5.0
    change = model(boards) - before

    is_promotion = torch.zeros(NUM_MOVE_IDS, dtype=torch.bool)
    is_promotion[list(PROMOTION_MOVE_IDS)] = True
    assert torch.allclose(change[:, is_promotion], torch.full_like(change[:, is_promotion], 5.0), atol=1e-4)
    assert (change[:, ~is_promotion] == 0).all()


#a layer that forward never uses gets no gradient and would silently never train
def test_every_parameter_gets_a_gradient(batch):
    model = small_model()
    boards, moves = batch

    nn.CrossEntropyLoss()(model(boards), moves).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


#memorising one batch proves nothing about playing chess, only that the whole model can learn
def test_can_overfit_one_batch(batch):
    model = small_model()
    boards, moves = batch
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first_loss = loss_function(model(boards), moves).item()
    for _ in range(400):
        loss = loss_function(model(boards), moves)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert first_loss > 5  #an untrained guess over 1968 moves is about ln(1968) = 7.6
    assert loss.item() < 0.3  #the final layer norm bounds the pooled vector, so the loss no longer races to zero
    assert (model(boards).argmax(dim=1) == moves).all()
