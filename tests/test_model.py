from itertools import islice

import chess
import pytest
import torch
from torch import nn

from model.encoding import NUM_MOVE_IDS, PROMOTION_MOVE_IDS, encode_board, encode_move_to_id
from model.model import VALUE_BINS, ChessTransformer, mlp_hidden_width
from training.train import value_loss

BATCH_SIZE = 16


def small_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(d_model=32, num_layers=2, n_heads=2)


#moves the input skip's scales off their initial 1 and 0, where a missing input embedding would not show
def mix_in_the_input(model: ChessTransformer) -> None:
    with torch.no_grad():
        for block in model.encoder:
            block.residual_scale.fill_(0.8)
            block.input_scale.fill_(0.5)


@pytest.fixture(scope="module")
def batch(carlsen_games) -> tuple[torch.Tensor, torch.Tensor]:
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

    assert logits.shape == (BATCH_SIZE, NUM_MOVE_IDS)
    assert value_logits.shape == (BATCH_SIZE, VALUE_BINS)


def test_moving_a_piece_changes_the_output(batch):
    model = small_model().eval()
    start_position = batch[0][:1]
    knight, pawn = chess.B1, chess.E2
    assert start_position[0, knight] != start_position[0, pawn]

    swapped = start_position.clone()
    swapped[0, [knight, pawn]] = start_position[0, [pawn, knight]]

    #without position information the outputs differ only by float rounding, around 1e-7
    difference = (model(start_position)[0] - model(swapped)[0]).abs().max().item()
    assert difference > 1e-4


def test_a_board_gets_the_same_logits_alone_or_in_a_batch(batch):
    model = small_model().eval()
    boards, _ = batch

    for alone, in_a_batch in zip(model(boards[:1]), model(boards)):
        assert torch.allclose(alone, in_a_batch[:1], atol=1e-5)


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


def test_attention_bias_starts_at_zero(batch):
    model = small_model().eval()
    boards, _ = batch
    before, _ = model(boards)

    with torch.no_grad():
        for block in model.encoder:
            block.template_mix.weight += 1.0

    assert torch.equal(model(boards)[0], before)


def test_the_size_of_the_queries_and_keys_does_not_matter(batch):
    boards, _ = batch
    model = small_model().eval()
    block = model.encoder[0]
    x = model.embed(boards) + model.position_embed

    with torch.no_grad():
        before = block(x, model.templates, x)
        for projection in (block.Q, block.K):
            projection.weight *= 100
            projection.bias *= 100
        assert torch.allclose(block(x, model.templates, x), before, atol=1e-4)


#fp16 tops out at 65,504, so queries and keys this big overflow every score unless they are normalised
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


def test_scaling_the_residual_stream_changes_no_output(batch):
    boards, _ = batch
    model = small_model().eval()
    mix_in_the_input(model)
    for module in model.modules():
        if isinstance(module, nn.RMSNorm):
            module.eps = 0.0  #the norms' eps is the one thing that sees the scale

    with torch.no_grad():
        model.square_features.table.normal_()  #zeros at init would hide the table being missed
        expected = model(boards)
        scaled = model.scale_residual_stream(1 / 16)(boards)

    for output, expected_output in zip(scaled, expected, strict=True):
        assert torch.allclose(output, expected_output, atol=1e-4)


def test_a_new_block_does_not_read_the_input_embedding(batch):
    boards, _ = batch
    model = small_model().eval()
    block = model.encoder[1]
    x = model.embed(boards) + model.position_embed
    with torch.no_grad():
        assert torch.equal(block(x, model.templates, x), block(x, model.templates, torch.randn_like(x)))


def test_the_input_skip_mixes_in_the_input_embedding(batch):
    boards, _ = batch
    model = small_model().eval()
    with torch.no_grad():
        #the last block reads only the input embedding, so the first block's weights cannot reach the output
        for block in model.encoder:
            block.residual_scale.fill_(0.0)
            block.input_scale.fill_(1.0)
        before = model(boards)
        for parameter in model.encoder[0].parameters():
            parameter.normal_()
        for output, expected in zip(model(boards), before, strict=True):
            assert torch.equal(output, expected)


#white to move, so the board is not mirrored
def test_the_square_features_of_a_known_position():
    board = torch.tensor([encode_board(chess.Board("4k3/3p4/8/3q4/8/2N5/4P3/R3K3 w - - 0 1"))])
    features = small_model().square_features.indices(board)[0]
    e4, d5, a1 = chess.E4, chess.D5, chess.A1

    #from our king e1 3 ranks up, from their king e8 4 down, no queen of ours, 1 file right and 1 rank down from d5
    assert features[e4, :4].tolist() == [(3 + 7) * 15 + 7, (-4 + 7) * 15 + 7, 15 * 15, (-1 + 7) * 15 + 1 + 7]
    #neighbours up, down, right, left, then the diagonals: only d5 (up-left) holds a piece, their queen
    assert features[e4, 4:12].tolist() == [0, 0, 0, 0, 0, 5 + 6, 0, 0]
    #first piece up the file their king 4 away, down it our pawn 2 away, up-left their queen next door
    king, pawn, queen = 1 + (6 + 6 - 1) * 7 + 3, 1 + (1 - 1) * 7 + 1, 1 + (5 + 6 - 1) * 7 + 0
    assert features[e4, 12:20].tolist() == [king, pawn, 0, 0, 0, queen, 0, 0]
    #our knight on c3 attacks e4, none of theirs: 1 * 3 + 0
    assert features[e4, 20].item() == 3
    #no pawn up the e-file; our pawn 2 down it
    assert features[e4, 21:23].tolist() == [0, 1 + 1]
    #their pawn on d7, 2 up the d-file from d5
    assert features[d5, 21].item() == 1 + 7 + 1
    #left of a1 is off the board, and so is everything along that line
    assert features[a1, 4 + 3].item() == 13 and features[a1, 12 + 3].item() == 0


def test_the_square_features_start_at_zero_and_reach_the_output(batch):
    boards, _ = batch
    model = small_model().eval()
    before, _ = model(boards)
    assert (model.square_features(boards) == 0).all()

    with torch.no_grad():
        model.square_features.table.normal_()
    assert not torch.allclose(model(boards)[0], before)


def test_every_parameter_gets_a_gradient(batch):
    model = small_model()
    boards, moves = batch

    #template_mix gets no gradient while the templates are zero; they leave zero on the first training step
    with torch.no_grad():
        model.templates.normal_(std=0.1)

    logits, value_logits = model(boards)
    values = torch.full((len(boards),), 0.7)
    (nn.CrossEntropyLoss()(logits, moves) + value_loss(value_logits, values)).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


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

    assert first_loss > 5  #an untrained guess over NUM_MOVE_IDS moves is about ln(NUM_MOVE_IDS) = 7.6
    assert loss.item() < 0.3
    assert (model(boards)[0].argmax(dim=1) == moves).all()


@pytest.mark.parametrize("qk_ratio, qk_widths", [(0.5, [16, 16]), ([0.25, 1.0], [8, 32])], ids=["one ratio", "a ratio per block"])
def test_a_smaller_qk_ratio_narrows_only_the_queries_and_keys(batch, qk_ratio, qk_widths):
    boards, _ = batch
    model = ChessTransformer(d_model=32, num_layers=2, n_heads=2, qk_ratio=qk_ratio)

    for block, width in zip(model.encoder, qk_widths, strict=True):
        assert block.Q.weight.shape == block.K.weight.shape == (width, 32)
        assert block.V.weight.shape == (32, 32)
    assert model(boards)[0].shape == (BATCH_SIZE, NUM_MOVE_IDS)


@pytest.mark.parametrize("settings", [
    {"loops": 0},
    {"qk_ratio": [0.25, 0.5, 1.0]},
    {"n_heads": 4, "kv_heads": 3},
], ids=["no loops", "a ratio per block that does not match", "key/value heads that do not divide the heads"])
def test_sizes_that_do_not_fit_are_rejected(settings):
    with pytest.raises(ValueError):
        ChessTransformer(**{"d_model": 32, "num_layers": 2, "n_heads": 2, **settings})


def test_the_mlp_width_is_rounded_down_to_a_multiple_of_32():
    assert mlp_hidden_width(512, 1.333) == 672
    assert mlp_hidden_width(128) == 320
    assert mlp_hidden_width(16) == 32
    assert ChessTransformer(d_model=128, num_layers=1).encoder[0].mlp.up_projection.out_features == 320


def test_grouped_query_attention_is_heads_sharing_their_keys_and_values(batch):
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
