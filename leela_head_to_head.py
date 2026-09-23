#a temporary head to head: model.py's ChessTransformer trained the way main.py trains it, against leela's whole network
#trained with leela's losses, on the same positions in the same order for the same minutes of training. Both are then
#scored the same way: policy loss and accuracy against stockfish's moves on the whole val set, and DeepMind's puzzles.
#
#   python leela_head_to_head.py                      both, 4 minutes each
#   python leela_head_to_head.py --minutes 6 --seed 1
#   python leela_head_to_head.py --positions 24_000_000   the same positions for both instead of the same minutes: comparable
#                                                         across gpus, whose speed differs between the two models
#   python leela_head_to_head.py --pflops 2.2             the same compute for both: each trains on the positions that
#                                                         2.2 PFLOPs of forward passes buy it, measured on a val batch
#   python leela_head_to_head.py --only leela --lr 3e-3
#   python leela_head_to_head.py --only leela --leela-loss ours    leela's network with our loss, to tell architecture from loss
#   python leela_head_to_head.py --only leela --leela-size small   leela's network at about our parameters and compute
#   python leela_head_to_head.py --only ours_smolgen               our model with smolgen in place of the templates
#   python leela_head_to_head.py --only ours --templates 32        our model with 32 templates instead of 8
#   python leela_head_to_head.py --only ours_smolgen --smolgen-size 8/32/32   the paper's smolgen sizes instead of 4/32/8
#   python leela_head_to_head.py --only ours_leela_embedding       our model with one of leela's features, listed above LEELA_BLOCK_DFF
#   python leela_head_to_head.py --only ours_board_pooled          our model with a whole-board term in its template mix
#   python leela_head_to_head.py --only ours_embed_templates       our model with a board-wide term in its embedding
#
#they train one after the other, ours first, each with its own tensorboard run in artifacts/runs (h2h_ours_..., h2h_leela_...)
#
#leela's network: the embedding and blocks of model_leela.py (checked against leela's own JAX code) with leela's
#policy, value and moves left heads ported below, on the 64 square tokens as in leela. Its losses are those of
#lczero-training's reference config (docs/example.textproto): policy cross entropy and KL on the same head, win/draw/loss
#against the game's result, and moves left (huber), weight 1 each. With --leela-loss ours it trains with ours_loss instead:
#its value head gives one logit for stockfish's eval and there is no moves left head, the rest of the network is the same.
#
#what stays different from leela: no illegal move mask in the policy loss (our data has no legal move lists, so neither
#model gets one), stockfish's top 4 as the policy target instead of search visits, one ply of history instead of eight
#past positions, and main.py's AdamW and schedule for both instead of leela's NAdamW.
#The final results of every run are appended to artifacts/leela_head_to_head.jsonl, and each run's trained weights are
#saved as model.pt in its tensorboard run directory
import argparse
import csv
import json
import math
from datetime import datetime
from itertools import islice
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ChessGamesDataset, datasets_to_dataloaders, load_training_set
from encoding import MOVE_FROM_SQUARES, MOVE_TO_SQUARES, NUM_HISTORY_CODES, NUM_SQUARES, PROMOTION_MOVE_IDS, PROMOTION_PIECES, PROMOTION_TO_SQUARES
from main import (BATCH_SIZE, EVAL_BATCHES, EVAL_EVERY, PEAK_LEARN_RATE, SEED, TARGET_TEMPERATURE, TRAINING_SET, WARMUP_STEPS,
                  batches_on_device, endless_batches, future_loss, lr_multiplier, policy_loss, value_loss, warm_file_cache)
from model import ChessTransformer, SwiGLU
from model_leela import NUM_PIECE_PLANES, LeelaBlock, LeelaChessTransformer, LeelaEmbedding, leela_linear, truncated_normal_
from tools.budget import Budget
from tools.checkpoint import RUNS_DIR, load_checkpoint
from tools.puzzles import PUZZLES_CSV, solve

MINUTES = 4
HISTORY = 1 #both models see the same one ply of history

#model.py's model as the overnight d640 run was built, at the 4 minute size and without the future head for now, and its
#loss weights (the future weight only counts with future > 0)
OURS = dict(d_model=128, num_layers=4, n_heads=4, position_templates=8, history=HISTORY, future=0, swiglu=True, mlp_ratio=1.333,
            dropout=0.0, value_target="eval")
OURS_VALUE_WEIGHT = 1.0
OURS_FUTURE_WEIGHT = 0.5

#leela's network at the same width, depth and heads, in the proportions of leela's reference config (d_model 1024):
#ffn and embedding ffn 1.5x, embedding dense d/2, value head 128 channels (d/8), moves left 32 (d/32), smolgen as it is
LEELA = dict(d_model=128, num_layers=4, n_heads=4, history=HISTORY, dff=192, embedding_dense=64, embedding_dff=192,
             smolgen_channels=32, smolgen_hidden=256, smolgen_gen=256, value_channels=16, moves_left_channels=4)
#smolgen and the embedding's dense keep leela's sizes above whatever the width, so at d_model 128 they hold 92% of its
#8.1M parameters. Shrunk, leela's network is 1.0M parameters and 84.5 MFLOPs a position against ours' 0.77M and 91.9
LEELA_SMALL = {**LEELA, "embedding_dense": 4, "smolgen_channels": 4, "smolgen_hidden": 32, "smolgen_gen": 8}
#our model with smolgen making the attention bias (model_leela.py's "smolgen" block), everything else ours: 0.80M parameters
#and 90.2 MFLOPs a position. smolgen_gen 8 is a bank of 8 64x64 maps like our 8 templates; what differs is the mix, one
#per head from the whole board instead of one per head for each square from that square
OURS_SMOLGEN = dict(smolgen_channels=4, smolgen_hidden=32, smolgen_gen=8)
#smolgen's channels/hidden/gen for ours_smolgen, a bank of gen maps each: the default above, the sizes of the ablations in
#the Chessformer paper (arXiv 2605.19091, 8 layers of d_model 256) and those of every leela network from T1-256 to BT4
SMOLGEN_SIZES = {"4/32/8": OURS_SMOLGEN,
                 "8/32/32": dict(smolgen_channels=8, smolgen_hidden=32, smolgen_gen=32),
                 "32/256/256": dict(smolgen_channels=32, smolgen_hidden=256, smolgen_gen=256)}
#leela's features one at a time on our model, each against the model it changes:
#  ours_leela_embedding  leela's embedding (the board-wide preprocess dense among it) with leela's sizes, vs ours
#  ours_leela_block      leela's post-norm block (DeepNorm, mish ffn) with ours_smolgen's smolgen, vs ours_smolgen.
#                        dff 256 gives it as many ffn weights as ours' SwiGLU at 4/3 (2 x 128 x 256 against 3 x 128 x 168)
#  ours_leela_policy     leela's attention policy head in place of our from/to projections, vs ours
LEELA_BLOCK_DFF = 256
#our model with a whole-board term in its template coefficients, vs ours and ours_smolgen: ours_board_pooled summarises
#the board as the mean of its squares, ours_board_flattened as every square compressed to BOARD_CHANNELS channels
BOARD_CHANNELS = 4
#simpler board-wide embeddings on our model (BoardWideEmbedding), vs ours and ours_leela_embedding --embedding-dense 16:
#ours_embed_dense, ours_embed_dense_ffn and ours_embed_templates, the dense ones at leela's dense size 16, the template
#one as 8 maps of 16 features (d_model 128)
BOARD_EMBEDDING_SIZE = 16
BOARD_EMBEDDING_MAPS = 8
#leela's embedding combined with a whole-board bias on our model: ours_board_embedding with the pooled board mix (our
#templates kept), ours_smolgen_embedding with ours_smolgen's smolgen. --embedding-dense shrinks the embedding's
#board-wide dense layer (per square, leela's default d_model / 2) for every contender that uses it on our model
LEELA_POLICY_WEIGHT = 1.0
LEELA_KL_WEIGHT = 1.0
LEELA_VALUE_WEIGHT = 1.0
LEELA_MOVES_LEFT_WEIGHT = 1.0
MOVES_LEFT_SCALE = 20.0 #leela divides moves left by this before its huber loss, whose delta is 10 moves
MOVES_LEFT_HUBER_DELTA = 10.0

PUZZLE_CHUNK = 1000 #solve() sends every unfinished puzzle through the model at once
RESULTS_FILE = ARTIFACTS_DIR / "leela_head_to_head.jsonl"


class LeelaPolicyHead(nn.Module):
    #leela's attention policy: an embedding with mish, then a from (q) and a to (k) vector per square and their scaled
    #dot product. Promotions to queen, rook and bishop add an offset per destination square made from the keys of the
    #last rank; a knight promotion is the plain move, as in leela. Laid out over our move ids
    def __init__(self, d_model: int, embedding_size: int, d_policy: int):
        super().__init__()
        self.tokens = leela_linear(d_model, embedding_size)
        self.q = leela_linear(embedding_size, d_policy)
        self.k = leela_linear(embedding_size, d_policy)
        self.promotion_dense = leela_linear(d_policy, 4, bias=False)
        self.scale = math.sqrt(d_policy)
        self.register_buffer("from_squares", torch.tensor(MOVE_FROM_SQUARES, dtype=torch.long))
        self.register_buffer("to_squares", torch.tensor(MOVE_TO_SQUARES, dtype=torch.long))
        self.register_buffer("promotion_move_ids", torch.tensor(PROMOTION_MOVE_IDS, dtype=torch.long))
        self.register_buffer("promotion_to_squares", torch.tensor(PROMOTION_TO_SQUARES, dtype=torch.long))
        self.register_buffer("promotion_pieces", torch.tensor(PROMOTION_PIECES, dtype=torch.long))

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        x = F.mish(self.tokens(squares))
        q, k = self.q(x), self.k(x)
        logits = ((q @ k.transpose(1, 2)) / self.scale)[:, self.from_squares, self.to_squares] #B moves

        #leela: 4 numbers per last rank square from its key, the first three (queen, rook, bishop) each plus the fourth
        promotion = self.promotion_dense(k[:, -8:])
        queen_rook_bishop = promotion[..., :3] + promotion[..., 3:]
        #in our piece order, 0 knight 1 bishop 2 rook 3 queen, on all 64 squares (only the last rank is ever promoted to)
        offsets = torch.stack([torch.zeros_like(queen_rook_bishop[..., 0]), queen_rook_bishop[..., 2], queen_rook_bishop[..., 1], queen_rook_bishop[..., 0]], dim=-1)
        offsets = F.pad(offsets, (0, 0, NUM_SQUARES - 8, 0))
        logits[:, self.promotion_move_ids] += offsets[:, self.promotion_to_squares, self.promotion_pieces]
        return logits


class LeelaFlattenedHead(nn.Module):
    #leela's value and moves left heads: a few channels per square, flattened, then two dense layers with mish between
    def __init__(self, d_model: int, channels: int, outputs: int):
        super().__init__()
        self.embed = leela_linear(d_model, channels)
        self.dense1 = leela_linear(NUM_SQUARES * channels, 128)
        self.out = leela_linear(128, outputs)

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        x = F.mish(self.embed(squares).flatten(1))
        return self.out(F.mish(self.dense1(x)))


class LeelaNetwork(nn.Module):
    #leela's embedding (64 square tokens, the metadata in every square's planes), blocks and heads. Returns the policy over
    #our move ids, the value head's logits (value_outputs 3: win/draw/loss in model.py's order 0 lost, 1 draw, 2 won;
    #1: stockfish's eval like model.py's) and the plies left in the game (None without the moves left head)
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        history: int,
        dff: int,
        embedding_dense: int,
        embedding_dff: int,
        smolgen_channels: int,
        smolgen_hidden: int,
        smolgen_gen: int,
        value_channels: int,
        moves_left_channels: int,
        value_outputs: int = 3,
        moves_left: bool = True
    ):
        super().__init__()
        self.history = history
        self.embed = LeelaEmbedding(d_model, embedding_dense, embedding_dff, num_layers, metadata_embed=None)
        #leela reads past positions as input planes; here the history comes in the way model.py adds it
        self.learned_history = nn.ModuleList(nn.Embedding(NUM_HISTORY_CODES, d_model, padding_idx=0) for _ in range(history))
        for embedding in self.learned_history:
            nn.init.zeros_(embedding.weight)
        #leela's shared last smolgen layer as a bank of 64x64 maps, see model_leela.py
        self.templates = nn.Parameter(truncated_normal_(torch.empty(smolgen_gen, NUM_SQUARES, NUM_SQUARES), 1 / smolgen_gen))
        self.encoder = nn.ModuleList(LeelaBlock(d_model, n_heads, dff, num_layers, NUM_SQUARES, smolgen_channels, smolgen_hidden, smolgen_gen)
                                     for _ in range(num_layers))
        self.policy_head = LeelaPolicyHead(d_model, d_model, d_model)
        self.value_head = LeelaFlattenedHead(d_model, value_channels, value_outputs)
        self.moves_left_head = LeelaFlattenedHead(d_model, moves_left_channels, 1) if moves_left else None

    def forward(self, board_state: torch.Tensor, prev_boards: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        squares = self.embed(board_state)
        for i in range(self.history):
            squares = squares + self.learned_history[i](prev_boards[:, i, :])
        for block in self.encoder:
            squares = block(squares, self.templates)

        #the heads in float32 like model.py's: in bf16 the policy logits of close moves become the same number
        with torch.autocast(device_type=board_state.device.type, enabled=False):
            squares = squares.float()
            moves_left = F.relu(self.moves_left_head(squares)).squeeze(-1) if self.moves_left_head is not None else None
            return self.policy_head(squares), self.value_head(squares), moves_left


class OursWithLeelaPolicy(ChessTransformer):
    #model.py's model with leela's attention policy head in place of its from/to projections and promotion projection.
    #forward follows ChessTransformer.forward up to the final layernorm, then the policy comes from leela's head and the
    #value from model.py's
    def __init__(self, **model_settings):
        super().__init__(**model_settings)
        if self.future_head is not None:
            raise ValueError("the future head is not wired into this forward")
        del self.from_projection, self.to_projection, self.promotion_projection
        self.policy_head = LeelaPolicyHead(self.d_model, self.d_model, self.d_model)

    def forward(self, board_state: torch.Tensor, prev_boards: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, None]:
        B = board_state.shape[0]
        embed_state = self.embed(board_state) + self.position_embed
        for i in range(self.history):
            embed_state[:, :NUM_SQUARES] = embed_state[:, :NUM_SQUARES] + self.learned_history[i](prev_boards[:, i, :])
        for block in self.encoder:
            embed_state = block(embed_state, self.templates)
        model_output = self.ln(embed_state)

        #the heads in float32 like model.py's
        with torch.autocast(device_type=board_state.device.type, enabled=False):
            logits = self.policy_head(model_output[:, :NUM_SQUARES].float())
            compressed_board = self.board_down_project(model_output.float()).reshape(B, -1)
            return logits, self.eval_head(compressed_board), None


class BoardTemplateMix(nn.Module):
    #model.py's per-square template coefficients plus one set for the whole board, added to every square's. The board's
    #set mixes the same templates the way smolgen mixes its bank, one mix per head for the board. It starts at zero, so
    #the model starts out as model.py's. The whole board is summarised as
    #   pooled     the mean of the squares' states: what is on the board (material, game stage), not where
    #   flattened  each square compressed to a few channels, the 64 side by side: where things are too (pawn structure,
    #              open files), smolgen's front end without its hidden layer and norms
    def __init__(self, per_square: nn.Linear, summary: str, channels: int):
        super().__init__()
        self.per_square = per_square
        self.summary = summary
        d_model, coefficients = per_square.in_features, per_square.out_features
        if summary == "pooled":
            self.compress = None
            self.board = nn.Linear(d_model, coefficients)
        elif summary == "flattened":
            self.compress = nn.Linear(d_model, channels, bias=False)
            self.board = nn.Linear(NUM_SQUARES * channels, coefficients)
        else:
            raise ValueError(f"summary must be pooled or flattened, got {summary}")
        nn.init.zeros_(self.board.weight)
        nn.init.zeros_(self.board.bias)

    #squares is B 64 d_model, the block's layernormed square tokens. Returns B 64 coefficients like template_mix does
    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        summary = squares.mean(dim=1) if self.compress is None else self.compress(squares).flatten(1)
        return self.per_square(squares) + self.board(summary)[:, None, :]


class OursWithBoardMix(LeelaChessTransformer):
    #model.py's model with a whole-board term in every block's template coefficients, see BoardTemplateMix. The blocks
    #and their forward are model.py's; only template_mix is wrapped, keeping its per-square weights. Built on
    #LeelaChessTransformer's "templates" block so leela_embedding (and its sizes) can be asked for too; without it the
    #model has exactly model.py's modules
    def __init__(self, board_mix: str, board_channels: int, **model_settings):
        super().__init__(block="templates", **model_settings)
        for block in self.encoder:
            block.template_mix = BoardTemplateMix(block.template_mix, board_mix, board_channels)


class BoardWideEmbedding(nn.Module):
    #model.py's token embedding plus a board-wide term on the 64 square tokens, simpler takes on the part of leela's
    #embedding that seems to matter (every square seeing the whole board before the first block). The term starts at
    #zero, so the model starts out as model.py's:
    #   dense      leela's preprocess alone: the 768 piece planes of the whole board to `size` features per square, then
    #              to d_model. No mish, norms, gates or ffn
    #   dense_ffn  dense, then one residual pre-norm SwiGLU on the square tokens, as model.py's blocks have
    #   templates  the templates' way of mixing: every square's embedding to `size` features per map, spread over the
    #              board by `maps` learned 64x64 maps, one per group of features. A board-wide step with far fewer
    #              parameters than a dense from every square to every square
    def __init__(self, token_embed: nn.Embedding, kind: str, size: int, maps: int, mlp_ratio: float):
        super().__init__()
        self.token_embed = token_embed
        self.kind = kind
        d_model = token_embed.embedding_dim
        if kind in ("dense", "dense_ffn"):
            self.preprocess = nn.Linear(NUM_SQUARES * NUM_PIECE_PLANES, NUM_SQUARES * size)
            self.project = nn.Linear(size, d_model)
            nn.init.zeros_(self.project.weight)
            nn.init.zeros_(self.project.bias)
            if kind == "dense_ffn":
                self.norm = nn.LayerNorm(d_model)
                self.ffn = SwiGLU(d_model, max(8, 8 * round(mlp_ratio * d_model / 8)))
                nn.init.zeros_(self.ffn.down_projection.weight)
        elif kind == "templates":
            if maps * size != d_model:
                raise ValueError(f"{maps} maps of {size} features have to make up d_model {d_model}")
            self.features = nn.Linear(d_model, maps * size, bias=False)
            self.maps = nn.Parameter(torch.zeros(maps, NUM_SQUARES, NUM_SQUARES))
        else:
            raise ValueError(f"kind must be dense, dense_ffn or templates, got {kind}")

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embed(board_state) #B TOKENS_PER_BOARD d_model
        squares = tokens[:, :NUM_SQUARES]
        B = board_state.shape[0]
        if self.kind == "templates":
            #B 64 maps size: map m spreads its own features over the board, square to square
            features = self.features(squares).reshape(B, NUM_SQUARES, len(self.maps), -1)
            board = torch.einsum("mij,bjmf->bimf", self.maps, features).reshape(B, NUM_SQUARES, -1)
        else:
            #the 12 piece planes of every square (code 0, an empty square, has none), the whole board in one row
            pieces = F.one_hot(board_state[:, :NUM_SQUARES], NUM_PIECE_PLANES + 1)[..., 1:].reshape(B, -1).to(squares.dtype)
            board = self.project(self.preprocess(pieces).reshape(B, NUM_SQUARES, -1))
        squares = squares + board
        if self.kind == "dense_ffn":
            squares = squares + self.ffn(self.norm(squares))
        return torch.cat([squares, tokens[:, NUM_SQUARES:]], dim=1)


class OursWithBoardEmbedding(ChessTransformer):
    #model.py's model with a board-wide term in its token embedding, see BoardWideEmbedding; forward is model.py's.
    #With board_mix also OursWithBoardMix's whole-board term in every block's template coefficients
    def __init__(self, board_embedding: str, board_embedding_size: int, board_embedding_maps: int, board_mix: str | None = None,
                 board_channels: int = BOARD_CHANNELS, **model_settings):
        super().__init__(**model_settings)
        self.embed = BoardWideEmbedding(self.embed, board_embedding, board_embedding_size, board_embedding_maps, model_settings.get("mlp_ratio") or 4 / 3)
        if board_mix is not None:
            for block in self.encoder:
                block.template_mix = BoardTemplateMix(block.template_mix, board_mix, board_channels)


#the plies from each row to the end of its game, counting the row's own move. Rows are whole games one after
#another, and plies_bin counts the rows of the same game before each one
def plies_left(name: str, split: str) -> np.ndarray:
    plies = np.fromfile(PROCESSED_DATA_DIR / f"{name}_{split}_plies.bin", dtype=np.int16).astype(np.int64)
    starts = plies == 0
    #a game cut short or out of order would give wrong targets without any error
    if not starts[0] or np.any(np.diff(plies)[~starts[1:]] != 1):
        raise ValueError(f"{name}_{split}_plies.bin is not whole games one after another")
    game = np.cumsum(starts) - 1
    last_rows = np.append(np.flatnonzero(starts)[1:], len(plies)) - 1
    return (plies[last_rows][game] - plies + 1).astype(np.float32)


class WithLeelaTargets(Dataset):
    #the same items, then the game's result as a class (0 lost, 1 draw, 2 won) and the plies left in the game
    def __init__(self, dataset: ChessGamesDataset, moves_left: np.ndarray):
        if len(moves_left) != len(dataset):
            raise ValueError(f"{len(moves_left):,} moves left targets for {len(dataset):,} positions")
        self.dataset = dataset
        self.moves_left = torch.from_numpy(moves_left)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: list[int]) -> tuple[torch.Tensor, ...]:
        return (*self.dataset[index], self.dataset.results[index].long() + 1, self.moves_left[index])


#main.py's loss: stockfish's moves, stockfish's eval, and what the board will look like. Also leela's network's with
#--leela-loss ours, whose third output is then None like a model.py model without the future head
def ours_loss(outputs: tuple, batch: tuple) -> torch.Tensor:
    logits, eval_logits, future_logits = outputs
    _, target_moves, target_probabilities, values, _, future = batch
    loss = policy_loss(logits, target_moves, target_probabilities) + OURS_VALUE_WEIGHT * value_loss(eval_logits, values)
    if future_logits is not None:
        loss = loss + OURS_FUTURE_WEIGHT * future_loss(future_logits, future)
    return loss

#leela's reference config: policy cross entropy and KL, the game's result as win/draw/loss, and moves left
def leela_loss(outputs: tuple, batch: tuple) -> torch.Tensor:
    logits, result_logits, moves_left = outputs
    _, target_moves, target_probabilities, _, _, _, results, plies_left_targets = batch
    cross_entropy = policy_loss(logits, target_moves, target_probabilities)
    #KL is the cross entropy minus the target's entropy, which has no gradient: it doubles the policy's weight
    kl = cross_entropy + torch.special.xlogy(target_probabilities, target_probabilities).sum(-1).mean()
    value = F.cross_entropy(result_logits, results)
    moves_left_loss = F.huber_loss(moves_left / MOVES_LEFT_SCALE, plies_left_targets / MOVES_LEFT_SCALE, delta=MOVES_LEFT_HUBER_DELTA / MOVES_LEFT_SCALE)
    return (LEELA_POLICY_WEIGHT * cross_entropy + LEELA_KL_WEIGHT * kl + LEELA_VALUE_WEIGHT * value
            + LEELA_MOVES_LEFT_WEIGHT * moves_left_loss)


#main.py's training: compiled, bf16 (or fp16 with a loss scale), AdamW, warmup then cosine over the budget (minutes of pure training, or positions),
#gradients clipped at 1.
#Every EVAL_EVERY steps the train loss, learning rate and a look at the val set go to tensorboard, outside the budget
def train(
    model: nn.Module,
    loss_of,
    train_loader: DataLoader,
    val_loader: DataLoader,
    budget: Budget,
    peak_learn_rate: float,
    device: torch.device,
    writer: SummaryWriter,
    dtype: torch.dtype = torch.bfloat16
) -> Budget:
    compiled_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learn_rate, fused=(device.type == "cuda"))
    #fp16's narrow range needs the loss scaled up so small gradients do not round to zero; bf16 does not
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    warmup_end = 0.0
    def current_lr_multiplier(step: int) -> float:
        nonlocal warmup_end
        if step <= WARMUP_STEPS:
            warmup_end = budget.progress()
            return lr_multiplier(step, 0.0)
        return lr_multiplier(step, (budget.progress() - warmup_end) / (1 - warmup_end))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, current_lr_multiplier)

    #kept on the device: adding to it does not make the cpu wait for the gpu
    loss_total = torch.zeros((), device=device)
    steps_since_eval = 0
    model.train()
    progress = tqdm(total=budget.total, desc="train", unit="s" if budget.timed else "it", dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}{postfix}]")
    for batch in batches_on_device(endless_batches(train_loader), device):
        with torch.autocast(device_type=device.type, dtype=dtype):
            loss = loss_of(compiled_model(batch[0], batch[4]), batch)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        progress.update(budget.step_done())
        scheduler.step()
        loss_total += loss.detach()
        steps_since_eval += 1

        if budget.steps % EVAL_EVERY == 0 or budget.finished():
            train_loss = loss_total.item() / steps_since_eval
            budget.pause()
            val_loss, val_accuracy = validate(model, val_loader, device, EVAL_BATCHES, dtype)
            model.train()
            #the same names as main.py's runs, so the curves line up with them in tensorboard. The train loss is each
            #model's own total, not comparable between the two; the val loss is the policy's against stockfish's moves for both
            writer.add_scalar("loss/train", train_loss, budget.steps)
            writer.add_scalar("loss/val", val_loss, budget.steps)
            writer.add_scalar("accuracy/val", val_accuracy, budget.steps)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], budget.steps)
            progress.set_postfix(loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}", val_accuracy=f"{val_accuracy:.1%}")
            loss_total.zero_()
            steps_since_eval = 0
            budget.resume()
        if budget.finished():
            break
    progress.close()
    return budget


#the policy against stockfish's moves on the first max_batches val batches, or on all of them: loss and accuracy
@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int | None = None,
             dtype: torch.dtype = torch.bfloat16) -> tuple[float, float]:
    model.eval()
    loss_total, correct, positions = 0.0, 0, 0
    num_batches = len(val_loader) if max_batches is None else min(max_batches, len(val_loader))
    for batch in tqdm(islice(val_loader, num_batches), total=num_batches, desc="val", leave=False, dynamic_ncols=True):
        boards, target_moves, target_probabilities, _, history = (tensor.to(device) for tensor in batch[:5])
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model(boards, history)[0]
        loss_total += policy_loss(logits.float(), target_moves, target_probabilities).item() * len(boards)
        correct += (logits.argmax(dim=-1) == target_moves[:, 0]).sum().item()
        positions += len(boards)
    return loss_total / positions, correct / positions


#the matmul FLOPs of one forward pass, per position, on a val batch. The backward pass is about twice this for every
#model, so the forward alone ranks them the same as the whole training step
@torch.no_grad()
def forward_flops_per_position(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    boards, history = (next(iter(val_loader))[i].to(device) for i in (0, 4))
    with FlopCounterMode(display=False) as counter:
        model(boards, history)
    return counter.get_total_flops() / len(boards)


#the same final scoring for both: the policy on every val position, and the puzzles
@torch.no_grad()
def score(model: nn.Module, val_loader: DataLoader, puzzles: list[dict], device: torch.device, dtype: torch.dtype = torch.bfloat16) -> dict:
    val_loss, val_accuracy = validate(model, val_loader, device, dtype=dtype)
    solved = []
    with torch.autocast(device_type=device.type, dtype=dtype):
        for start in tqdm(range(0, len(puzzles), PUZZLE_CHUNK), desc="puzzles", leave=False, dynamic_ncols=True):
            solved += solve(model, puzzles[start:start + PUZZLE_CHUNK], device)
    return {"val_policy_loss": val_loss, "val_accuracy": val_accuracy, "puzzles": sum(solved) / len(solved)}


#reads back a model.pt that main() saved: model.py's model through tools/checkpoint.py, the others by the network class
#saved with them. main.py's checkpoints name no network, they are always model.py's. The same signature as
#load_checkpoint, so tools/play's player can load them in its place
def load_h2h_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("network", "ChessTransformer") == "ChessTransformer":
        return load_checkpoint(checkpoint_path, device)
    network = {"LeelaChessTransformer": LeelaChessTransformer, "LeelaNetwork": LeelaNetwork, "OursWithLeelaPolicy": OursWithLeelaPolicy,
               "OursWithBoardMix": OursWithBoardMix, "OursWithBoardEmbedding": OursWithBoardEmbedding}[checkpoint["network"]]
    model = network(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["model_settings"]


def main(
    minutes: float,
    positions: int | None,
    pflops: float | None,
    seed: int,
    peak_learn_rate: float,
    only: str | None,
    training_set: str,
    leela_loss_kind: str,
    leela_size: str,
    templates: int,
    smolgen_size: str,
    embedding_dense: int,
    qk_ratio: list[float],
    kv_heads: int | None,
    history: int,
    results_path: Path,
    target_temperature: float,
    embedding_maps: int = 0,
    fp16: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if fp16 else torch.bfloat16
    train_dataset, val_dataset = load_training_set(training_set, history, "stockfish", target_temperature, OURS["future"], OURS["value_target"])
    if not isinstance(train_dataset, ChessGamesDataset):
        raise ValueError("the head to head takes one training set, not a mix")
    warm_file_cache(train_dataset, val_dataset)
    leela_train_dataset = WithLeelaTargets(train_dataset, plies_left(training_set, "train"))
    with open(PUZZLES_CSV, newline="") as puzzles_file:
        puzzles = list(csv.DictReader(puzzles_file))
    print(f"{training_set}: {len(train_dataset):,} train positions, {len(val_dataset):,} val positions, {len(puzzles):,} puzzles")

    #every contender sees the history the data was loaded with
    base_settings = {**OURS, "history": history}
    leela_settings = {**(LEELA_SMALL if leela_size == "small" else LEELA), "history": history}
    #model.py's narrower queries/keys and grouped-query attention, for every contender built on our model's own blocks.
    #Only named when set, so a default run saves the same settings as before they existed
    ours_settings = {**base_settings, "position_templates": templates}
    if qk_ratio != [1.0]:
        ours_settings["qk_ratio"] = qk_ratio[0] if len(qk_ratio) == 1 else qk_ratio
    if kv_heads is not None:
        ours_settings["kv_heads"] = kv_heads
    smolgen_settings = SMOLGEN_SIZES[smolgen_size]
    #each contender's network, the settings it is built from (saved with its weights) and what it trains on
    if leela_loss_kind == "ours":
        leela = (LeelaNetwork, {**leela_settings, "value_outputs": 1, "moves_left": False}, ours_loss, train_dataset)
    else:
        leela = (LeelaNetwork, leela_settings, leela_loss, leela_train_dataset)
    embedding_settings = {"leela_embedding": True, "embedding_dense": embedding_dense, "embedding_dff": LEELA["embedding_dff"]}
    #only named when set, as with qk_ratio and kv_heads
    if embedding_maps:
        embedding_settings["embedding_maps"] = embedding_maps
    contenders = {"ours": (ChessTransformer, ours_settings, ours_loss, train_dataset), "leela": leela,
                  "ours_smolgen": (LeelaChessTransformer, {"block": "smolgen", **smolgen_settings, **base_settings}, ours_loss, train_dataset),
                  "ours_leela_embedding": (LeelaChessTransformer, {"block": "templates", **embedding_settings, **ours_settings}, ours_loss, train_dataset),
                  "ours_leela_block": (LeelaChessTransformer, {"block": "leela", "dff": LEELA_BLOCK_DFF, **smolgen_settings, **base_settings}, ours_loss, train_dataset),
                  "ours_leela_policy": (OursWithLeelaPolicy, ours_settings, ours_loss, train_dataset),
                  "ours_board_pooled": (OursWithBoardMix, {"board_mix": "pooled", "board_channels": BOARD_CHANNELS, **ours_settings}, ours_loss, train_dataset),
                  "ours_board_flattened": (OursWithBoardMix, {"board_mix": "flattened", "board_channels": BOARD_CHANNELS, **ours_settings}, ours_loss, train_dataset),
                  "ours_board_embedding": (OursWithBoardMix, {"board_mix": "pooled", "board_channels": BOARD_CHANNELS, **embedding_settings, **ours_settings},
                                           ours_loss, train_dataset),
                  "ours_smolgen_embedding": (LeelaChessTransformer, {"block": "smolgen", **embedding_settings, **smolgen_settings, **base_settings},
                                             ours_loss, train_dataset)}
    board_embedding_settings = {"board_embedding_size": BOARD_EMBEDDING_SIZE, "board_embedding_maps": BOARD_EMBEDDING_MAPS}
    for kind in ("dense", "dense_ffn", "templates"):
        contenders[f"ours_embed_{kind}"] = (OursWithBoardEmbedding, {"board_embedding": kind, **board_embedding_settings, **ours_settings}, ours_loss, train_dataset)
    contenders["ours_embed_templates_board"] = (OursWithBoardEmbedding, {"board_embedding": "templates", **board_embedding_settings, "board_mix": "pooled",
                                                                         "board_channels": BOARD_CHANNELS, **ours_settings}, ours_loss, train_dataset)
    #the ones after ours and leela only when asked for by name
    selected = [only] if only is not None else ["ours", "leela"]
    started = datetime.now()
    results = {}
    for name in selected:
        network, settings, loss_of, dataset = contenders[name]
        torch.manual_seed(seed)
        model = network(**settings).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        #any contender built differently from its default gets its own name, in the runs and in the results
        if name == "leela":
            name += ("_ourloss" if leela_loss_kind == "ours" else "") + ("_small" if leela_size == "small" else "")
        elif name == "ours" and templates != OURS["position_templates"]:
            name += f"_t{templates}"
        elif name in ("ours_smolgen", "ours_leela_block", "ours_smolgen_embedding") and smolgen_settings is not OURS_SMOLGEN:
            name += "_" + smolgen_size.replace("/", "-")
        if name.startswith(("ours_leela_embedding", "ours_board_embedding", "ours_smolgen_embedding")) and embedding_dense != LEELA["embedding_dense"]:
            name += f"_dense{embedding_dense}"
        if settings.get("embedding_maps"):
            name += f"_maps{embedding_maps}"
        if settings.get("qk_ratio", 1.0) != 1.0:
            name += "_qk" + "-".join(f"{ratio:g}" for ratio in qk_ratio)
        if settings.get("kv_heads") is not None:
            name += f"_kv{kv_heads}"
        if history != HISTORY:
            name += f"_h{history}"
        #the val loss is against targets of this temperature too, so it only compares within one temperature
        if target_temperature != TARGET_TEMPERATURE:
            name += f"_temp{target_temperature:g}"
        train_loader, val_loader = datasets_to_dataloaders(dataset, val_dataset, train_batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE)
        flops_per_position = forward_flops_per_position(model, val_loader, device)
        if pflops is not None:
            budget = Budget(BATCH_SIZE, positions=round(pflops * 1e15 / flops_per_position))
            budget_name = f"{pflops:g}PF"
        else:
            budget = Budget(BATCH_SIZE, positions=positions) if positions is not None else Budget(BATCH_SIZE, minutes=minutes)
            budget_name = budget.describe()
        run_dir = RUNS_DIR / f"h2h_{name}_{budget_name}_seed{seed}_lr{peak_learn_rate:g}_{started:%m-%d_%H-%M-%S}"
        print(f"\n{name}: {parameters:,} parameters, {flops_per_position / 1e6:.1f} MFLOPs a position, logging to {run_dir.relative_to(ARTIFACTS_DIR.parent)}")
        writer = SummaryWriter(log_dir=str(run_dir))
        #the same seed again right before the first batch, so both models see the same positions in the same order
        torch.manual_seed(seed)
        train(model, loss_of, train_loader, val_loader, budget, peak_learn_rate, device, writer, dtype)
        results[name] = {"parameters": parameters, "forward_mflops_per_position": round(flops_per_position / 1e6, 2), "positions": budget.positions,
                         "positions_per_second": round(budget.positions / budget.seconds), **score(model, val_loader, puzzles, device, dtype)}
        writer.add_scalar("final/val_loss", results[name]["val_policy_loss"], budget.steps)
        writer.add_scalar("final/val_accuracy", results[name]["val_accuracy"], budget.steps)
        writer.add_scalar("final/puzzles", results[name]["puzzles"], budget.steps)
        writer.close()
        #the trained weights next to the run's logs, in tools/checkpoint.py's format plus which network class they are for
        torch.save({"network": network.__name__, "model_settings": settings, "state_dict": model.state_dict()}, run_dir / "model.pt")
        print(f"{name}: {json.dumps(results[name])}")

    print(f"\n{'':20s} {'parameters':>11s} {'positions':>12s} {'val policy loss':>16s} {'val accuracy':>13s} {'puzzles':>8s}")
    for name, result in results.items():
        print(f"{name:20s} {result['parameters']:>11,} {result['positions']:>12,} {result['val_policy_loss']:>16.4f} {result['val_accuracy']:>13.2%} {result['puzzles']:>8.2%}")
    with open(results_path, "a") as results_file:
        results_file.write(json.dumps({"finished_at": datetime.now().isoformat(timespec="seconds"),
                                       "minutes": None if positions is not None or pflops is not None else minutes,
                                       "positions_budget": positions, "pflops_budget": pflops, "seed": seed, "history": history, "target_temperature": target_temperature,
                                       "peak_learn_rate": peak_learn_rate, "training_set": training_set, "leela_loss": leela_loss_kind,
                                       "precision": "fp16" if fp16 else "bf16",
                                       "leela_size": leela_size, "ours": ours_settings, "leela": leela_settings, "ours_smolgen": smolgen_settings,
                                       "results": results}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model.py's model against leela's, trained and scored the same way")
    budget_flags = parser.add_mutually_exclusive_group()
    budget_flags.add_argument("--minutes", type=float, default=MINUTES, help="minutes of pure training for each model, default %(default)s")
    budget_flags.add_argument("--positions", type=int, help="train each model on this many positions instead of for a number of minutes")
    budget_flags.add_argument("--pflops", type=float, help="train each model on the positions this many PFLOPs of forward passes buy it")
    parser.add_argument("--seed", type=int, default=SEED, help="of the weights and the order of the positions, the same for both. Default %(default)s")
    parser.add_argument("--lr", type=float, default=PEAK_LEARN_RATE, help="peak learning rate for both, default main.py's %(default)s")
    parser.add_argument("--only", choices=["ours", "leela", "ours_smolgen", "ours_leela_embedding", "ours_leela_block", "ours_leela_policy",
                                           "ours_board_pooled", "ours_board_flattened", "ours_board_embedding", "ours_smolgen_embedding",
                                           "ours_embed_dense", "ours_embed_dense_ffn", "ours_embed_templates", "ours_embed_templates_board"],
                        help="train and score just one of them, or one of the variants of our model")
    parser.add_argument("--training-set", default=TRAINING_SET, help="default main.py's %(default)s")
    parser.add_argument("--leela-loss", choices=["leela", "ours"], default="leela", help="what leela's network trains on: leela's losses, or ours (policy and stockfish's eval). Default %(default)s")
    parser.add_argument("--leela-size", choices=["full", "small"], default="full", help="leela's network with leela's sizes, or shrunk to about ours. Default %(default)s")
    parser.add_argument("--templates", type=int, default=OURS["position_templates"], help="our model's number of templates. Default %(default)s")
    parser.add_argument("--smolgen-size", choices=list(SMOLGEN_SIZES), default="4/32/8", help="ours_smolgen's and ours_leela_block's smolgen channels/hidden/gen. Default %(default)s")
    parser.add_argument("--embedding-dense", type=int, default=LEELA["embedding_dense"], help="per-square size of leela's board-wide embedding layer on our model. Default %(default)s")
    parser.add_argument("--qk-ratio", type=float, nargs="+", default=[1.0], help="model.py's query/key width for our model's blocks: one ratio or one per block. Default 1")
    parser.add_argument("--kv-heads", type=int, help="model.py's key/value heads for our model's blocks (grouped-query attention). Default one per head")
    parser.add_argument("--history", type=int, default=HISTORY, help="past plies every contender is shown. Default %(default)s")
    parser.add_argument("--target-temperature", type=float, default=TARGET_TEMPERATURE, help="how stockfish's top moves share the policy target, see main.py. Default %(default)s")
    parser.add_argument("--results-file", type=Path, default=RESULTS_FILE, help="the jsonl the results are appended to, one per job when many run at once. Default %(default)s")
    parser.add_argument("--embedding-maps", type=int, default=0, help="build leela's board-wide embedding features from this many learned 64x64 maps "
                        "(the templates' way) instead of its dense layer, --embedding-dense of them per square. Default 0, the dense layer")
    parser.add_argument("--fp16", action="store_true", help="train in fp16 with a loss scale instead of bf16, so that Turing GPUs (no bf16) "
                        "can run it and every run of a comparison uses the same precision")
    parser.add_argument("--no-tensorboard", action="store_true", help="do not start tensorboard or open the browser")
    arguments = parser.parse_args()

    if not arguments.no_tensorboard:
        #imported here like main.py does: tools/tensorboard_server.py imports from main
        from tools.tensorboard_server import start_tensorboard
        start_tensorboard()

    main(arguments.minutes, arguments.positions, arguments.pflops, arguments.seed, arguments.lr, arguments.only, arguments.training_set,
         arguments.leela_loss, arguments.leela_size, arguments.templates, arguments.smolgen_size, arguments.embedding_dense, arguments.qk_ratio,
         arguments.kv_heads, arguments.history, arguments.results_file, arguments.target_temperature, arguments.embedding_maps, arguments.fp16)
