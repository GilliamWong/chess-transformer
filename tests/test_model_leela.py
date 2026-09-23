import chess
import pytest
import torch

from encoding import (NUM_HISTORY_CODES, NUM_MOVE_IDS, NUM_PIECE_CODES, NUM_SQUARES, TURN_INDEX, WHITE_KINGSIDE_INDEX, BLACK_QUEENSIDE_INDEX,
                      EN_PASSANT_INDEX, TURN_TOKEN_OFFSET, CASTLING_TOKEN_OFFSET, EN_PASSANT_TOKEN_OFFSET, encode_board)
from model import ChessTransformer, TransformerBlock
from model_leela import LeelaChessTransformer, SmolgenBlock, build_model

#model.py's settings of a small model, and the leela parts scaled down to match
SMALL = dict(d_model=32, num_layers=2, n_heads=2, history=1, future=2, dropout=0.0)
SMALL_LEELA = dict(dff=48, smolgen_channels=4, smolgen_hidden=16, smolgen_gen=8, embedding_dense=4, embedding_dff=48)
VARIANTS = [dict(block="smolgen"), dict(block="leela"), dict(block="leela", leela_embedding=True),
            dict(block="templates", leela_embedding=True), dict(block="templates", leela_embedding=True, embedding_maps=2)]


def boards() -> torch.Tensor:
    """the start position, one with castling rights gone on one side, and one with an en passant capture on"""
    castled = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
    en_passant = chess.Board()
    for san in ("e4", "c5", "e5", "d5"):
        en_passant.push_san(san)
    return torch.tensor([encode_board(board) for board in (chess.Board(), castled, en_passant)])


def small_variant(**variant) -> LeelaChessTransformer:
    torch.manual_seed(0)
    return LeelaChessTransformer(**variant, **SMALL_LEELA, **SMALL)


#the variants are trained and scored by the same code as model.py, so they have to give its outputs
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_gives_model_py_outputs(variant):
    board_state = boards()
    history = torch.zeros(len(board_state), 1, NUM_SQUARES, dtype=torch.long)

    logits, value_logits, future_logits = small_variant(**variant)(board_state, history)

    assert logits.shape == (len(board_state), NUM_MOVE_IDS)
    assert value_logits.shape == (len(board_state), 1)
    assert future_logits.shape == (len(board_state), NUM_SQUARES, NUM_HISTORY_CODES)


#DDP stops a run when a parameter gets no gradient, e.g. model.py's templates left behind unused
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_parameter_gets_a_gradient(variant):
    model = small_variant(**variant)
    board_state = boards()
    history = torch.randint(0, NUM_HISTORY_CODES, (len(board_state), 1, NUM_SQUARES))

    logits, value_logits, future_logits = model(board_state, history)
    (logits.mean() + value_logits.mean() + future_logits.mean()).backward()

    no_gradient = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert no_gradient == []


#block "templates" adds leela's embedding to model.py's model and nothing else, so an ablation of the embedding does
#not also swap the attention bias
def test_templates_block_keeps_model_py_blocks_and_templates():
    model = small_variant(block="templates", leela_embedding=True)
    reference = ChessTransformer(**SMALL)

    assert all(type(block) is TransformerBlock for block in model.encoder)
    assert model.templates.shape == reference.templates.shape


#SmolgenBlock copies TransformerBlock.forward, only the bias is different. With both biases zero they must agree,
#which catches a norm, residual or dropout that went missing in the copy
def test_smolgen_block_is_model_block_when_the_bias_is_zero():
    torch.manual_seed(0)
    block = TransformerBlock(32, 2, position_templates=1, dropout=0.0, swiglu=True, mlp_ratio=1.333)
    smolgen_block = SmolgenBlock(32, 2, 0.0, True, 1.333, 70, 4, 16, 8)
    missing, unexpected = smolgen_block.load_state_dict(block.state_dict(), strict=False)
    assert all(name.startswith("smolgen.") for name in missing)
    assert all(name.startswith("template_mix.") for name in unexpected)

    x = torch.randn(3, 70, 32)
    expected = block(x, torch.zeros(1, NUM_SQUARES, NUM_SQUARES))
    assert torch.allclose(smolgen_block(x, torch.zeros(8, NUM_SQUARES, NUM_SQUARES)), expected, atol=1e-6)


#the planes are leela's input, rebuilt from our tokens: a wrong offset would show the model another position
def test_embedding_planes_describe_the_board():
    board_state = boards()
    planes = small_variant(block="leela", leela_embedding=True).embed.planes(board_state)
    pieces, metadata = planes[..., :NUM_PIECE_CODES], planes[..., NUM_PIECE_CODES:]

    #pieces 1-12 first, then empty: code c is at c - 1, empty at 12
    expected_piece_channel = (board_state[:, :NUM_SQUARES] - 1) % NUM_PIECE_CODES
    assert torch.equal(pieces.argmax(-1), expected_piece_channel)
    assert torch.equal(pieces.sum(-1), torch.ones(len(board_state), NUM_SQUARES))

    #the same on every square: side to move (2), the four castling rights (2 each), the en passant square (64)
    assert torch.equal(metadata, metadata[:, :1].expand_as(metadata))
    turn, castling, en_passant = metadata[:, 0, :2], metadata[:, 0, 2:10].reshape(-1, 4, 2), metadata[:, 0, 10:]
    assert torch.equal(turn.argmax(-1), board_state[:, TURN_INDEX] - TURN_TOKEN_OFFSET)
    assert torch.equal(castling.argmax(-1), board_state[:, WHITE_KINGSIDE_INDEX:BLACK_QUEENSIDE_INDEX + 1] - CASTLING_TOKEN_OFFSET)
    assert torch.equal(en_passant.argmax(-1), board_state[:, EN_PASSANT_INDEX] - EN_PASSANT_TOKEN_OFFSET)
    assert en_passant[2].argmax() != 0  #the en passant position really has its square set


#a checkpoint stores the settings the model was built with; build_model has to rebuild the variant that made it
def test_build_model_rebuilds_a_saved_variant():
    settings = dict(block="leela", leela_embedding=True, **SMALL_LEELA, **SMALL)
    torch.manual_seed(0)
    model = build_model(settings).eval()
    rebuilt = build_model(settings).eval()
    rebuilt.load_state_dict(model.state_dict())

    board_state = boards()
    history = torch.zeros(len(board_state), 1, NUM_SQUARES, dtype=torch.long)
    assert type(rebuilt) is LeelaChessTransformer
    assert torch.equal(rebuilt(board_state, history)[0], model(board_state, history)[0])
    assert type(build_model(dict(SMALL))) is ChessTransformer
