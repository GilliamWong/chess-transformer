import torch
from torch import nn
import torch.nn.functional as F
import math
from model.encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, MOVE_FROM_SQUARES, MOVE_TO_SQUARES, NUM_SQUARES, NUM_PIECE_CODES, NUM_PIECE_TYPES, PROMOTION_MOVE_IDS, PROMOTION_TO_SQUARES, PROMOTION_PIECES

VALUE_TOKEN_DIM = 16
VALUE_BINS = 128
MLP_WIDTH_MULTIPLE = 32

def mlp_hidden_width(d_model: int, mlp_ratio: float | None = None) -> int:
    ratio = 8 / 3 if mlp_ratio is None else mlp_ratio
    return max(MLP_WIDTH_MULTIPLE, int(ratio * d_model) // MLP_WIDTH_MULTIPLE * MLP_WIDTH_MULTIPLE)

def value_bin_edges(bins: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, 1, bins + 1, device=device)

#0 to 1: a win scores 1, a draw 0.5
def expected_score(value_logits: torch.Tensor) -> torch.Tensor:
    bins = value_logits.shape[-1]
    if bins == 1:
        return torch.sigmoid(value_logits[..., 0].float())
    edges = value_bin_edges(bins, value_logits.device)
    centres = (edges[:-1] + edges[1:]) / 2
    return torch.softmax(value_logits.float(), dim=-1) @ centres

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()

        self.up_projection = nn.Linear(d_model, d_hidden, bias=False)
        self.gate_projection = nn.Linear(d_model, d_hidden, bias=False)
        self.down_projection = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.gate_projection(x)) * self.up_projection(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, position_templates: int, mlp_hidden: int, qk_ratio: float, kv_heads: int | None):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads #n_heads must divide d_model
        self.d_qk_head = max(1, round(qk_ratio * self.d_head))
        self.kv_heads = kv_heads or n_heads
        if n_heads % self.kv_heads != 0:
            raise ValueError(f"{n_heads} heads cannot be split into groups for {self.kv_heads} key/value heads")
        self.heads_per_kv = n_heads // self.kv_heads
        self.position_templates = position_templates

        self.Q = nn.Linear(d_model, n_heads * self.d_qk_head)
        self.K = nn.Linear(d_model, self.kv_heads * self.d_qk_head)
        self.V = nn.Linear(d_model, self.kv_heads * self.d_head)

        self.q_norm = nn.RMSNorm(self.d_qk_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.d_qk_head, eps=1e-5)

        #eps given: nn.RMSNorm's default is the dtype's, ~1e-3 in fp16, big next to the stream fp16 inference shrinks 16x
        self.ln1 = nn.RMSNorm(d_model, eps=1e-5)
        self.ln2 = nn.RMSNorm(d_model, eps=1e-5)

        self.combine_heads = nn.Linear(d_model, d_model)

        self.mlp = SwiGLU(d_model, mlp_hidden)

        self.template_mix = nn.Linear(d_model, position_templates * n_heads)

        self.residual_scale = nn.Parameter(torch.ones(1))
        self.input_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, templates: torch.Tensor, original_board_embed: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = self.residual_scale * x + self.input_scale * original_board_embed
        x_norm = self.ln1(residual)

        Q = self.Q(x_norm)
        K = self.K(x_norm)
        V = self.V(x_norm)

        Q_heads = self.q_norm(Q.reshape(B, T, self.n_heads, self.d_qk_head)).transpose(1, 2)
        K_heads = self.k_norm(K.reshape(B, T, self.kv_heads, self.d_qk_head)).transpose(1, 2)
        V_heads = V.reshape(B, T, self.kv_heads, self.d_head).transpose(1, 2)

        #head h reads key/value head h // heads_per_kv
        K_heads = K_heads.repeat_interleave(self.heads_per_kv, dim=1)
        V_heads = V_heads.repeat_interleave(self.heads_per_kv, dim=1)

        attention = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(self.d_qk_head)

        coefficients = self.template_mix(x_norm[:, :NUM_SQUARES]).reshape(B, NUM_SQUARES, self.n_heads, self.position_templates)
        bias = torch.einsum("biht,tij->bhij", coefficients, templates)
        metadata_tokens = T - NUM_SQUARES
        attention = attention + F.pad(bias, (0, metadata_tokens, 0, metadata_tokens))

        attention_weights = torch.softmax(attention, dim=-1) @ V_heads
        attention_weights = attention_weights.transpose(1, 2).reshape(B, T, self.d_model)

        residual = residual + self.combine_heads(attention_weights)
        return residual + self.mlp(self.ln2(residual))


#piece codes on the squares: 0 empty, 1-6 the side to move's pawn, knight, bishop, rook, queen, king, 7-12 the other side's
PAWN, KNIGHT, QUEEN, KING = 1, 2, 5, 6
THEIRS = NUM_PIECE_TYPES #added to a piece code for the other side's piece
EDGE = NUM_PIECE_CODES #the code past every edge of the board

#(file step, rank step); rank +1 is towards the other side, as every board is from the side to move
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
KNIGHT_JUMPS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

NO_PIECE = 15 * 15 #the offset feature's row when there is no such piece
OFFSET_ROWS = NO_PIECE + 1 #where the square is from a piece: (rank step + 7) * 15 + file step + 7
NEIGHBOUR_ROWS = NUM_PIECE_CODES + 1 #a neighbour's code, or the edge
LINE_ROWS = 1 + 12 * 7 #none, or the first piece along a line and how far
KNIGHT_ROWS = 3 * 3 #our and their knights attacking the square, 0, 1 or 2+ each
PAWN_ROWS = 1 + 2 * 7 #none, or the nearest pawn up or down the file, whose and how far

def step(square: int, file_step: int, rank_step: int) -> int:
    file, rank = square % 8 + file_step, square // 8 + rank_step
    return rank * 8 + file if 0 <= file < 8 and 0 <= rank < 8 else NUM_SQUARES #the square past the edge

#codes out along lines (... 7, nearest first): the code where found first holds (0 if it never does) and how far out
def first_along(along: torch.Tensor, found: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distance = found.int().argmax(dim=-1, keepdim=True)
    code = torch.where(found.any(dim=-1, keepdim=True), along.gather(-1, distance), 0)
    return code.squeeze(-1), distance.squeeze(-1)

class SquareFeatures(nn.Module):
    #per square: kings, queens, neighbours, the first piece along each line, attacking knights, nearest pawns on the
    #file. 23 learned vectors, summed
    def __init__(self, d_model: int):
        super().__init__()
        slot_rows = [OFFSET_ROWS] * 4 + [NEIGHBOUR_ROWS] * 8 + [LINE_ROWS] * 8 + [KNIGHT_ROWS] + [PAWN_ROWS] * 2
        self.register_buffer("row_offsets", torch.tensor([0] + slot_rows[:-1]).cumsum(0), persistent=False)
        #zeros: a new model starts without the features and draws no random numbers
        self.table = nn.Parameter(torch.zeros(sum(slot_rows), d_model))

        #[square, piece's square]: where the square is from the piece
        offsets = [[(s // 8 - p // 8 + 7) * 15 + s % 8 - p % 8 + 7 for p in range(NUM_SQUARES)] for s in range(NUM_SQUARES)]
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        #[square, direction, 7]: the squares out along each direction, nearest first
        rays = [[[step(s, d[0] * k, d[1] * k) for k in range(1, 8)] for d in DIRECTIONS] for s in range(NUM_SQUARES)]
        self.register_buffer("rays", torch.tensor(rays), persistent=False)
        self.register_buffer("knight_jumps", torch.tensor([[step(s, *j) for j in KNIGHT_JUMPS] for s in range(NUM_SQUARES)]), persistent=False)

    #B 64 23: each square's row within each feature's rows
    def indices(self, board_state: torch.Tensor) -> torch.Tensor:
        squares = board_state[:, :NUM_SQUARES]
        padded = F.pad(squares, (0, 1), value=EDGE) #index 64 is past the edge

        offsets = []
        for code in (KING, KING + THEIRS, QUEEN, QUEEN + THEIRS):
            present = squares == code
            square = present.int().argmax(dim=1) #the first one only
            offsets.append(torch.where(present.any(dim=1, keepdim=True), self.offsets[:, square].T, NO_PIECE))

        #neighbours: the first square out each way; lines: the first piece and how far
        along = padded[:, self.rays]
        first, distance = first_along(along, (along != 0) & (along != EDGE))
        lines = torch.where(first > 0, 1 + (first - 1) * 7 + distance, 0)

        jumps = padded[:, self.knight_jumps]
        knights = (jumps == KNIGHT).sum(dim=-1).clamp(max=2) * 3 + (jumps == KNIGHT + THEIRS).sum(dim=-1).clamp(max=2)

        file = along[:, :, :2] #up and down the file
        pawn, pawn_distance = first_along(file, (file == PAWN) | (file == PAWN + THEIRS))
        pawns = torch.where(pawn > 0, 1 + (pawn == PAWN + THEIRS).long() * 7 + pawn_distance, 0)

        return torch.cat([torch.stack(offsets, dim=-1), along[..., 0], lines, knights.unsqueeze(-1), pawns], dim=-1)

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        B = board_state.shape[0]
        rows = self.indices(board_state) + self.row_offsets
        return F.embedding_bag(rows.reshape(B * NUM_SQUARES, -1), self.table, mode="sum").reshape(B, NUM_SQUARES, -1)


class ChessTransformer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, n_heads: int = 4, position_templates: int = 8, mlp_hidden: int | None = None, qk_ratio: float | list[float] = 1.0, kv_heads: int | None = None, loops: int = 1, value_bins: int = VALUE_BINS):
        super().__init__()
        if loops < 1:
            raise ValueError(f"loops is how many times the blocks run, at least 1, got {loops}")
        if value_bins < 1:
            raise ValueError(f"value_bins is the eval head's bins over the expected score, at least 1, got {value_bins}")
        self.d_model = d_model
        self.loops = loops

        self.embed = nn.Embedding(BOARD_VOCAB_SIZE, d_model)

        self.position_embed = nn.Parameter(torch.randn(TOKENS_PER_BOARD, d_model))

        self.square_features = SquareFeatures(d_model)

        self.templates = nn.Parameter(torch.zeros(position_templates, NUM_SQUARES, NUM_SQUARES))

        qk_ratios = list(qk_ratio) if isinstance(qk_ratio, (list, tuple)) else [qk_ratio] * num_layers
        if len(qk_ratios) != num_layers:
            raise ValueError(f"{len(qk_ratios)} qk ratios for {num_layers} blocks, give one or one per block")

        if mlp_hidden is None:
            mlp_hidden = mlp_hidden_width(d_model)

        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, position_templates, mlp_hidden, ratio, kv_heads) for ratio in qk_ratios])

        self.board_down_project = nn.Sequential(nn.Linear(d_model, VALUE_TOKEN_DIM), nn.GELU())

        self.eval_head = nn.Sequential(
            nn.Linear(TOKENS_PER_BOARD * VALUE_TOKEN_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, value_bins)
        )

        self.register_buffer("from_squares", torch.tensor(MOVE_FROM_SQUARES, dtype=torch.long))
        self.register_buffer("to_squares", torch.tensor(MOVE_TO_SQUARES, dtype=torch.long))

        #per promotion move: its id, landing square and piece (0-3)
        self.register_buffer("promotion_move_ids", torch.tensor(PROMOTION_MOVE_IDS, dtype=torch.long))
        self.register_buffer("promotion_to_squares", torch.tensor(PROMOTION_TO_SQUARES, dtype=torch.long))
        self.register_buffer("promotion_pieces", torch.tensor(PROMOTION_PIECES, dtype=torch.long))

        self.from_projection = nn.Linear(d_model, d_model)
        self.to_projection = nn.Linear(d_model, d_model)
        self.promotion_projection = nn.Linear(d_model, 4) #knight, bishop, rook, queen

        self.ln = nn.RMSNorm(d_model, eps=1e-5)

    #all but the heads, which forward runs in fp32: weights rounded to half precision would stay rounded
    def half_except_heads(self, dtype: torch.dtype = torch.bfloat16) -> "ChessTransformer":
        heads = (self.from_projection, self.to_projection, self.promotion_projection, self.board_down_project, self.eval_head)
        for child in self.children():
            if child not in heads:
                child.to(dtype)
        for name, parameter in list(self.named_parameters(recurse=False)):
            setattr(self, name, nn.Parameter(parameter.detach().to(dtype)))
        return self

    #every read of the residual stream is an RMSNorm, which ignores its scale (all but eps), so scaling everything written
    #into it changes no output
    def scale_residual_stream(self, scale: float) -> "ChessTransformer":
        with torch.no_grad():
            for weight in (self.embed.weight, self.position_embed, self.square_features.table):
                weight *= scale
            for block in self.encoder:
                block.combine_heads.weight *= scale
                block.combine_heads.bias *= scale
                block.mlp.down_projection.weight *= scale
        return self

    def forward(self, board_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = board_state.shape[0]

        embed_state = self.embed(board_state) + self.position_embed + F.pad(self.square_features(board_state), (0, 0, 0, TOKENS_PER_BOARD - NUM_SQUARES))
        original_board_embed = embed_state

        for _ in range(self.loops):
            for block in self.encoder:
                embed_state = block(embed_state, self.templates, original_board_embed)
        model_output = self.ln(embed_state)

        with torch.autocast(device_type=board_state.device.type, enabled=False):
            from_vector = self.from_projection(model_output[:, :NUM_SQUARES].float())
            to_vector = self.to_projection(model_output[:, :NUM_SQUARES].float())
            promotion_logits = self.promotion_projection(model_output[:, :NUM_SQUARES].float())

            logits = (from_vector @ to_vector.transpose(1, 2)) / math.sqrt(self.d_model)

            combined_logits = logits[:, self.from_squares, self.to_squares]

            #promotions only: a term on every move can drift without limit
            combined_logits[:, self.promotion_move_ids] += promotion_logits[:, self.promotion_to_squares, self.promotion_pieces]

            compressed_board = self.board_down_project(model_output.float()).reshape(B, -1)
            board_eval_logits = self.eval_head(compressed_board)

        return combined_logits, board_eval_logits
