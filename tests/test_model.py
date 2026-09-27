from itertools import islice

import pytest
import torch
from torch import nn

from encoding import NUM_MOVE_IDS, PROMOTION_MOVE_IDS, TOKENS_PER_BOARD, encode_board, encode_move_to_id
from main import value_loss
from model import ATTENTION_CAP, VALUE_BINS, ChessTransformer, mlp_hidden_width
from tools.attention_probe import attention_scores

BATCH_SIZE = 16


def small_model() -> ChessTransformer:
    """small enough to train for a few hundred steps inside a test"""
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=2, n_heads=2)


@pytest.fixture(scope="module")
def batch(carlsen_games) -> tuple[torch.Tensor, torch.Tensor]:
    """the first positions of a real game, with the moves played from them"""
    board = carlsen_games[0].board()
    boards, moves = [], []
    for move in islice(carlsen_games[0].mainline_moves(), BATCH_SIZE):
        boards.append(encode_board(board))
        moves.append(encode_move_to_id(move, board.turn))
        board.push(move)
    return torch.tensor(boards), torch.tensor(moves)


def test_one_set_of_move_logits_per_board(batch):
    boards, _ = batch

    logits, value_logits = small_model()(boards)

    #not (batch, TOKENS_PER_BOARD, NUM_MOVE_IDS): the tokens have to be pooled into one answer per board
    assert boards.shape == (BATCH_SIZE, TOKENS_PER_BOARD)
    assert logits.shape == (BATCH_SIZE, NUM_MOVE_IDS)
    assert value_logits.shape == (BATCH_SIZE, VALUE_BINS)  #a distribution over bins of the expected score


#without the positional embedding the model sees which pieces exist but not where they are
def test_moving_a_piece_changes_the_output(batch):
    model = small_model().eval()
    start_position = batch[0][:1]
    assert start_position[0, 1] != start_position[0, 12]

    swapped = start_position.clone()
    swapped[0, 1], swapped[0, 12] = start_position[0, 12], start_position[0, 1]  #knight b1 <-> pawn e2

    #with no position information the two outputs differ only by float rounding, around 1e-7
    difference = (model(start_position)[0] - model(swapped)[0]).abs().max().item()
    assert difference > 1e-4


#a board's output must not depend on the other boards of its batch
def test_a_board_gets_the_same_logits_alone_or_in_a_batch(batch):
    model = small_model().eval()
    boards, _ = batch

    for alone, in_a_batch in zip(model(boards[:1]), model(boards)):
        assert torch.allclose(alone, in_a_batch[:1], atol=1e-5)


#a term added to every move can drift without limit (bf16 training collapsed once it passed 256), so the promotion
#term may only reach the promotions
def test_promotion_term_only_changes_promotion_logits(batch):
    model = small_model().eval()
    boards, _ = batch
    before, _ = model(boards)

    with torch.no_grad():
        model.promotion_projection.bias += 5.0
    change = model(boards)[0] - before

    is_promotion = torch.zeros(NUM_MOVE_IDS, dtype=torch.bool)
    is_promotion[list(PROMOTION_MOVE_IDS)] = True
    assert torch.allclose(change[:, is_promotion], torch.full_like(change[:, is_promotion], 5.0), atol=1e-4)
    assert (change[:, ~is_promotion] == 0).all()


#the templates start at zero so that a new model is exactly the model without them, whatever template_mix holds
def test_attention_bias_starts_at_zero(batch):
    model = small_model().eval()
    boards, _ = batch
    before, _ = model(boards)

    with torch.no_grad():
        for block in model.encoder:
            block.template_mix.weight += 1.0

    assert torch.equal(model(boards)[0], before)


#the sink lets a head do nothing: with its score far above every real score the attention reads no token, and the block
#adds only combine_heads' bias before its mlp
def test_a_head_whose_sink_score_dominates_reads_nothing(batch):
    boards, _ = batch
    model = small_model().eval()
    block = model.encoder[0]
    x = model.embed(boards) + model.position_embed

    with torch.no_grad():
        block.sinks.fill_(1e4)
        after_attention = x + block.combine_heads.bias
        expected = after_attention + block.mlp(block.ln2(after_attention))
        assert torch.allclose(block(x, model.templates), expected, atol=1e-5)


#QK-norm: the scores depend on the directions of the queries and keys, not their size, so growing projections change
#nothing
def test_the_size_of_the_queries_and_keys_does_not_matter(batch):
    boards, _ = batch
    model = small_model().eval()
    block = model.encoder[0]
    x = model.embed(boards) + model.position_embed

    with torch.no_grad():
        before = block(x, model.templates)
        for projection in (block.Q, block.K):
            projection.weight *= 100
            projection.bias *= 100
        assert torch.allclose(block(x, model.templates), before, atol=1e-4)


#however large the template bias and the sink, every score stays inside the cap
def test_the_attention_scores_stay_inside_the_cap(batch):
    boards, _ = batch
    model = small_model().eval()
    with torch.no_grad():
        model.templates.fill_(1e4)
        for block in model.encoder:
            block.sinks.fill_(-1e4)

    largest = max(scores.abs().max().item() for scores in attention_scores(model, boards))
    assert 0.99 * ATTENTION_CAP < largest <= ATTENTION_CAP


#the search runs the model in fp16, whose largest number is 65,504: queries and keys this big made every score inf and
#every output NaN
def test_fp16_stays_finite_with_huge_queries_and_keys(batch):
    boards, _ = batch
    model = small_model().eval()
    with torch.no_grad():
        for block in model.encoder:
            block.Q.weight *= 1000
            block.K.weight *= 1000
    model.half_except_heads(torch.float16)

    logits, value_logits = model(boards)
    assert torch.isfinite(logits).all() and torch.isfinite(value_logits).all()


#a layer that forward never uses gets no gradient and would silently never train
def test_every_parameter_gets_a_gradient(batch):
    model = small_model()
    boards, moves = batch

    #while the templates are zero template_mix gets no gradient; in training they leave zero on the first step
    with torch.no_grad():
        model.templates.normal_(std=0.1)

    #both heads have to be in the loss, or the one left out gets no gradient
    logits, value_logits = model(boards)
    values = torch.full((len(boards),), 0.7)
    (nn.CrossEntropyLoss()(logits, moves) + value_loss(value_logits, values)).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


#looped depth runs the same blocks again: one block looped twice is two blocks with the same weights
def test_a_looped_block_is_the_same_block_twice(batch):
    boards, _ = batch
    torch.manual_seed(0)
    looped = ChessTransformer(d_model=32, num_layers=1, n_heads=2, loops=2).eval()
    with torch.no_grad():
        looped.templates.normal_(std=0.1)  #zero templates would leave the attention bias out of the comparison
    unrolled = ChessTransformer(d_model=32, num_layers=2, n_heads=2).eval()
    weights = looped.state_dict()
    weights.update({name.replace("encoder.0.", "encoder.1."): tensor for name, tensor in weights.items() if name.startswith("encoder.0.")})
    unrolled.load_state_dict(weights)

    for looped_output, unrolled_output in zip(looped(boards), unrolled(boards)):
        assert torch.equal(looped_output, unrolled_output)
    assert sum(parameter.numel() for parameter in looped.parameters()) < sum(parameter.numel() for parameter in unrolled.parameters())


def test_the_blocks_run_at_least_once():
    with pytest.raises(ValueError):
        ChessTransformer(d_model=32, num_layers=2, n_heads=2, loops=0)


#memorising one batch proves nothing about playing chess, only that the whole model can learn
def test_can_overfit_one_batch(batch):
    model = small_model()
    boards, moves = batch
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first_loss = loss_function(model(boards)[0], moves).item()
    for _ in range(400):
        loss = loss_function(model(boards)[0], moves)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert first_loss > 5  #an untrained guess over 1968 moves is about ln(1968) = 7.6
    assert loss.item() < 0.3  #the final layer norm bounds the pooled vector, so the loss no longer races to zero
    assert (model(boards)[0].argmax(dim=1) == moves).all()


def test_a_smaller_qk_ratio_narrows_only_the_queries_and_keys(batch):
    boards, _ = batch
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, num_layers=2, n_heads=2, qk_ratio=0.5)
    block = model.encoder[0]

    assert block.Q.weight.shape == block.K.weight.shape == (16, 32) #8 dims a head for the dot products
    assert block.V.weight.shape == (32, 32) #the values each head brings back keep their 16
    logits, _ = model(boards)
    assert logits.shape == (BATCH_SIZE, NUM_MOVE_IDS)


def test_a_qk_ratio_per_block_gives_each_block_its_own_width(batch):
    boards, _ = batch
    model = ChessTransformer(d_model=32, num_layers=2, n_heads=2, qk_ratio=[0.25, 1.0])

    assert [block.Q.weight.shape[0] for block in model.encoder] == [8, 32]
    assert model(boards)[0].shape == (BATCH_SIZE, NUM_MOVE_IDS)
    with pytest.raises(ValueError): #a list that does not match the blocks would otherwise quietly miss some
        ChessTransformer(d_model=32, num_layers=2, n_heads=2, qk_ratio=[0.25, 0.5, 1.0])


#widths the tensor cores take whole: the d512 recipe's ratio 1.333 is 672 wide, the default 8/3 of a d128 320
def test_the_mlp_width_is_rounded_down_to_a_multiple_of_32():
    assert mlp_hidden_width(512, 1.333) == 672
    assert mlp_hidden_width(128) == 320
    assert mlp_hidden_width(16) == 32
    assert ChessTransformer(d_model=128, num_layers=1).encoder[0].mlp.up_projection.out_features == 320


def test_grouped_query_attention_is_heads_sharing_their_keys_and_values(batch):
    #2 key/value heads for 4 heads must give what 4 give when heads 0, 1 and heads 2, 3 have the same key/value weights
    boards, _ = batch
    torch.manual_seed(0)
    grouped = ChessTransformer(d_model=32, num_layers=2, n_heads=4, kv_heads=2).eval()
    full = ChessTransformer(d_model=32, num_layers=2, n_heads=4).eval()
    weights = grouped.state_dict()
    for name in [f"encoder.{block}.{projection}.{kind}" for block in range(2) for projection in "KV" for kind in ("weight", "bias")]:
        #the rows are grouped by head, 8 each: every key/value head's rows again for the second head of its group
        weights[name] = weights[name].reshape(2, 8, *weights[name].shape[1:]).repeat_interleave(2, dim=0).flatten(0, 1)
    full.load_state_dict(weights)

    for grouped_output, full_output in zip(grouped(boards), full(boards)):
        assert torch.allclose(grouped_output, full_output, atol=1e-5)
    with pytest.raises(ValueError):
        ChessTransformer(d_model=32, num_layers=2, n_heads=4, kv_heads=3)
