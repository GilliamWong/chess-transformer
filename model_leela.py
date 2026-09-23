#Leela Chess Zero's transformer, ported from JAX (github.com/LeelaChessZero/lczero-training,
#src/lczero_training/model/{encoder,embedding,shared}.py at commit 7c5d756) to compare against model.py on our
#data, targets and trainer.
#
#LeelaChessTransformer is model.py's ChessTransformer (same heads, same forward) with parts swapped:
#   block="smolgen"         model.py's block, its templates mixed with smolgen's coefficients instead of per square ones
#   block="leela"           leela's block: post-norm with deepnorm scaling, mish ffn, smolgen
#   leela_embedding=True    leela's input embedding for the 64 squares
#
#smolgen is template mixing with global coefficients: leela's last smolgen layer, shared by every block, is a bank
#of gen 64x64 maps, and each block mixes them per head with coefficients computed from the whole board. The bank
#takes the place of model.py's templates, so ChessTransformer.forward hands it to every block unchanged.
#
#where this differs from leela: the input is our 70 tokens, so smolgen reads all 70 and its bias is padded to the
#metadata tokens like model.py's; leela's embedding is fed planes made from our tokens, and the history and
#model.py's learned position embedding are added after it; model.py's final layer norm and heads are kept
import math
from typing import cast
import torch
from torch import nn
import torch.nn.functional as F
from encoding import (NUM_SQUARES, NUM_PIECE_CODES, NUM_FLAG_VALUES, NUM_EN_PASSANT_VALUES, TURN_INDEX, WHITE_KINGSIDE_INDEX,
                      BLACK_QUEENSIDE_INDEX, EN_PASSANT_INDEX, TURN_TOKEN_OFFSET, CASTLING_TOKEN_OFFSET, EN_PASSANT_TOKEN_OFFSET, TOKENS_PER_BOARD)
from model import ChessTransformer, SwiGLU, TransformerBlock

LAYER_NORM_EPSILON = 1e-3 #leela's

#the planes leela's embedding reads per square: 12 piece planes first (its preprocess takes those), then empty,
#side to move, the four castling rights and the en passant square, the last ones the same on every square
NUM_PIECE_PLANES = NUM_PIECE_CODES - 1
NUM_CASTLING_SLOTS = BLACK_QUEENSIDE_INDEX - WHITE_KINGSIDE_INDEX + 1
NUM_INPUT_PLANES = NUM_PIECE_CODES + NUM_FLAG_VALUES + NUM_CASTLING_SLOTS * NUM_FLAG_VALUES + NUM_EN_PASSANT_VALUES #87

#flax's truncated normal: cut at 2 standard deviations, with the std corrected for the cut
def truncated_normal_(tensor: torch.Tensor, variance: float) -> torch.Tensor:
    std = math.sqrt(variance) / 0.87962566103423978
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)

#an nn.Linear initialised like flax's: lecun normal weights, or deepnorm's (variance beta / fan_avg) when
#deepnorm_beta is given, and a zero bias
def leela_linear(in_features: int, out_features: int, bias: bool = True, deepnorm_beta: float | None = None) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=bias)
    if deepnorm_beta is None:
        truncated_normal_(linear.weight, 1 / in_features)
    else:
        truncated_normal_(linear.weight, deepnorm_beta / ((in_features + out_features) / 2))
    if bias:
        nn.init.zeros_(linear.bias)
    return linear

#deepnorm: every sublayer's output is scaled by alpha, and the value, attention output and ffn weights start scaled by beta
def deepnorm_alpha(num_layers: int) -> float:
    return (2 * num_layers) ** -0.25

def deepnorm_beta(num_layers: int) -> float:
    return (8 * num_layers) ** -0.25

#B n_heads gen coefficients mixed over the gen 64x64 maps of the bank, zero padded to the metadata tokens
def smolgen_bias(coefficients: torch.Tensor, bank: torch.Tensor, tokens: int) -> torch.Tensor:
    bias = torch.einsum("bht,tij->bhij", coefficients, bank) #B n_heads 64 64
    metadata_tokens = tokens - NUM_SQUARES
    return F.pad(bias, (0, metadata_tokens, 0, metadata_tokens))


class Smolgen(nn.Module):
    #leela's smolgen up to its shared last layer: gen coefficients per head, from all tokens compressed and flattened
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tokens: int,
        channels: int,
        hidden: int,
        gen: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.gen = gen
        self.compress = leela_linear(d_model, channels, bias=False)
        self.dense1 = leela_linear(tokens * channels, hidden)
        self.ln1 = nn.LayerNorm(hidden, eps=LAYER_NORM_EPSILON)
        self.dense2 = leela_linear(hidden, gen * n_heads)
        self.ln2 = nn.LayerNorm(gen * n_heads, eps=LAYER_NORM_EPSILON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        compressed = self.compress(x).reshape(B, -1) #B tokens*channels
        hidden = self.ln1(F.silu(self.dense1(compressed)))
        coefficients = self.ln2(F.silu(self.dense2(hidden)))
        return coefficients.reshape(B, self.n_heads, self.gen)


class LeelaFfn(nn.Module):
    def __init__(self, d_model: int, dff: int, beta: float):
        super().__init__()
        self.linear1 = leela_linear(d_model, dff, deepnorm_beta=beta)
        self.linear2 = leela_linear(dff, d_model, deepnorm_beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.mish(self.linear1(x)))


class LeelaBlock(nn.Module):
    #leela's encoder block. Post-norm: x = ln1(x + attention(x) * alpha), then ln2(x + ffn(x) * alpha)
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dff: int,
        num_layers: int,
        tokens: int,
        smolgen_channels: int,
        smolgen_hidden: int,
        smolgen_gen: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.alpha = deepnorm_alpha(num_layers)
        beta = deepnorm_beta(num_layers)

        self.q = leela_linear(d_model, d_model)
        self.k = leela_linear(d_model, d_model)
        self.v = leela_linear(d_model, d_model, deepnorm_beta=beta)
        self.output_dense = leela_linear(d_model, d_model, deepnorm_beta=beta)
        self.smolgen = Smolgen(d_model, n_heads, tokens, smolgen_channels, smolgen_hidden, smolgen_gen)
        self.ln1 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPSILON)
        self.ffn = LeelaFfn(d_model, dff, beta)
        self.ln2 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPSILON)

    def attention(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2) #B n_heads T d_head
        k = self.k(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) #B n_heads T T
        logits = logits + smolgen_bias(self.smolgen(x), templates, T)
        weights = torch.softmax(logits, dim=-1)

        attended = (weights @ v).transpose(1, 2).reshape(B, T, d_model)
        return self.output_dense(attended)

    #templates is the model's template bank, handed to every block by ChessTransformer.forward
    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, templates) * self.alpha
        out1 = self.ln1(x)
        return self.ln2(out1 + self.ffn(out1) * self.alpha)


class SmolgenBlock(TransformerBlock):
    #model.py's block with the per square template coefficients replaced by smolgen's per head ones
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        swiglu: bool,
        mlp_ratio: float | None,
        tokens: int,
        smolgen_channels: int,
        smolgen_hidden: int,
        smolgen_gen: int
    ):
        super().__init__(d_model, n_heads, position_templates=1, dropout=dropout, swiglu=swiglu, mlp_ratio=mlp_ratio)
        del self.template_mix
        self.smolgen = Smolgen(d_model, n_heads, tokens, smolgen_channels, smolgen_hidden, smolgen_gen)

    #TransformerBlock.forward, the bias coming from smolgen
    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        residual = x
        x_norm = self.ln1(residual)

        Q_heads = self.Q(x_norm).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K_heads = self.K(x_norm).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V_heads = self.V(x_norm).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attention = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention = attention + smolgen_bias(self.smolgen(x_norm), templates, T)
        softmax_attention = self.attn_dropout(torch.softmax(attention, dim=-1))

        attention_weights = (softmax_attention @ V_heads).transpose(1, 2).reshape(B, T, d_model)
        residual = residual + self.head_mix_dropout(self.combine_heads(attention_weights))
        return residual + self.mlp_dropout(self.mlp(self.ln2(residual)))


class LeelaEmbedding(nn.Module):
    #leela's embedding of the 64 squares, from planes made of our tokens (see NUM_INPUT_PLANES). With metadata_embed the
    #6 metadata tokens follow as model.py embeds them (its heads read them); without, the 64 squares are all, as in leela
    def __init__(
        self,
        d_model: int,
        dense_size: int,
        dff: int,
        num_layers: int,
        metadata_embed: nn.Module | None,
        maps: int = 0
    ):
        super().__init__()
        self.alpha = deepnorm_alpha(num_layers)
        if maps:
            #the templates' way instead of the dense layer: each square's piece to dense_size / maps features per map,
            #spread over the board by that map's learned 64x64 weights. The same kind of board-wide features per square,
            #from far fewer parameters. Scaled so a square's sum over 64 others keeps the features' size
            if dense_size % maps:
                raise ValueError(f"{maps} maps have to divide the {dense_size} board-wide features")
            self.preprocess = None
            self.piece_features = nn.Linear(NUM_PIECE_PLANES, dense_size, bias=False)
            self.maps = nn.Parameter(torch.randn(maps, NUM_SQUARES, NUM_SQUARES) / NUM_SQUARES ** 0.5)
        else:
            #the whole board's pieces to dense_size features per square, leela's replacement for a position embedding
            self.preprocess = leela_linear(NUM_SQUARES * NUM_PIECE_PLANES, NUM_SQUARES * dense_size)
        self.embedding = leela_linear(NUM_INPUT_PLANES + dense_size, d_model)
        self.norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPSILON)
        #a multiplicative (through relu, starting at 1) and an additive (starting at 0) gate per square and channel
        self.mult_gate = nn.Parameter(torch.ones(NUM_SQUARES, d_model))
        self.add_gate = nn.Parameter(torch.zeros(NUM_SQUARES, d_model))
        self.ffn = LeelaFfn(d_model, dff, deepnorm_beta(num_layers))
        self.out_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPSILON)
        self.metadata_embed = metadata_embed

    #B 64 NUM_INPUT_PLANES of 0s and 1s
    def planes(self, board_state: torch.Tensor) -> torch.Tensor:
        B = board_state.shape[0]
        pieces = F.one_hot(board_state[:, :NUM_SQUARES], NUM_PIECE_CODES) #B 64 13, code 0 is an empty square
        turn = F.one_hot(board_state[:, TURN_INDEX] - TURN_TOKEN_OFFSET, NUM_FLAG_VALUES)
        castling = F.one_hot(board_state[:, WHITE_KINGSIDE_INDEX:BLACK_QUEENSIDE_INDEX + 1] - CASTLING_TOKEN_OFFSET, NUM_FLAG_VALUES).reshape(B, -1)
        en_passant = F.one_hot(board_state[:, EN_PASSANT_INDEX] - EN_PASSANT_TOKEN_OFFSET, NUM_EN_PASSANT_VALUES)
        metadata = torch.cat([turn, castling, en_passant], dim=-1).unsqueeze(1).expand(-1, NUM_SQUARES, -1)
        return torch.cat([pieces[..., 1:], pieces[..., :1], metadata], dim=-1).to(self.embedding.weight.dtype)

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        B = board_state.shape[0]
        planes = self.planes(board_state)
        if self.preprocess is None:
            #B 64 maps size: map m spreads its own features over the board, square to square
            features = self.piece_features(planes[..., :NUM_PIECE_PLANES]).reshape(B, NUM_SQUARES, len(self.maps), -1)
            position_info = torch.einsum("mij,bjmf->bimf", self.maps, features).reshape(B, NUM_SQUARES, -1)
        else:
            position_info = self.preprocess(planes[..., :NUM_PIECE_PLANES].reshape(B, -1)).reshape(B, NUM_SQUARES, -1)

        squares = self.norm(F.mish(self.embedding(torch.cat([planes, position_info], dim=-1))))
        squares = squares * F.relu(self.mult_gate) + self.add_gate
        squares = self.out_norm(squares + self.ffn(squares) * self.alpha)

        if self.metadata_embed is None:
            return squares
        metadata = self.metadata_embed(board_state[:, NUM_SQUARES:])
        return torch.cat([squares, metadata], dim=1) #B TOKENS_PER_BOARD d_model


class LeelaChessTransformer(ChessTransformer):
    #the defaults follow leela's BT4 (d_model 1024): ffn 1.5x, smolgen 32 / 256 / 256, embedding dense d_model / 2,
    #embedding ffn 1.5x. model_settings are ChessTransformer's; position_templates is ignored, smolgen's bank replaces the templates.
    #block "templates" keeps ChessTransformer's own blocks and templates, to try leela's embedding on model.py's model alone
    def __init__(
        self,
        block: str = "leela",
        leela_embedding: bool = False,
        dff: int | None = None,
        smolgen_channels: int = 32,
        smolgen_hidden: int = 256,
        smolgen_gen: int = 256,
        embedding_dense: int | None = None,
        embedding_dff: int | None = None,
        embedding_maps: int = 0,
        **model_settings
    ):
        if block not in ("leela", "smolgen", "templates"):
            raise ValueError(f"block must be leela, smolgen or templates, got {block}")
        if block != "templates":
            model_settings["position_templates"] = 1
        super().__init__(**model_settings)
        #read back from the blocks ChessTransformer built, so its defaults stay the single source
        first_block = cast(TransformerBlock, self.encoder[0])
        d_model, num_layers, n_heads = self.d_model, len(self.encoder), first_block.n_heads
        dropout, swiglu, mlp_ratio = first_block.dropout, isinstance(first_block.mlp, SwiGLU), model_settings.get("mlp_ratio")

        if block != "templates":
            #leela's shared weight_gen_dense as gen 64x64 maps, initialised the same way (lecun normal over gen inputs)
            self.templates = nn.Parameter(truncated_normal_(torch.empty(smolgen_gen, NUM_SQUARES, NUM_SQUARES), 1 / smolgen_gen))
            if block == "leela":
                blocks = [LeelaBlock(d_model, n_heads, dff or d_model * 3 // 2, num_layers, TOKENS_PER_BOARD, smolgen_channels, smolgen_hidden, smolgen_gen)
                          for _ in range(num_layers)]
            else:
                blocks = [SmolgenBlock(d_model, n_heads, dropout, swiglu, mlp_ratio, TOKENS_PER_BOARD, smolgen_channels, smolgen_hidden, smolgen_gen)
                          for _ in range(num_layers)]
            self.encoder = nn.ModuleList(blocks)

        if leela_embedding:
            self.embed = LeelaEmbedding(d_model, embedding_dense or d_model // 2, embedding_dff or d_model * 3 // 2, num_layers, self.embed,
                                        embedding_maps)

#the model a checkpoint's settings describe: a leela variant names its block, which ChessTransformer does not take
def build_model(model_settings: dict) -> ChessTransformer:
    if "block" in model_settings:
        return LeelaChessTransformer(**model_settings)
    return ChessTransformer(**model_settings)
