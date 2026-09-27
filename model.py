import torch
from torch import nn
import torch.nn.functional as F
import math
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, MOVE_FROM_SQUARES, MOVE_TO_SQUARES, NUM_SQUARES, PROMOTION_MOVE_IDS, PROMOTION_TO_SQUARES, PROMOTION_PIECES

VALUE_TOKEN_DIM = 16

#the eval head's output is a distribution over this many equal bins of the expected score (HL-Gauss, Farebrother et
#al. 2024); 1 is the older head, one logit of the score
VALUE_BINS = 128

#every attention score, sink included, is soft-capped to cap * tanh(score / cap): uncapped, block 1's ran past fp16's 65,504
ATTENTION_CAP = 50.0

#the swiglu mlp's hidden width is mlp_ratio (default 8/3) times d_model, rounded down to a multiple of this: whole
#tensor core tiles, 680 -> 672 made a d512 step 7% faster (a multiple of 16 is not enough)
MLP_WIDTH_MULTIPLE = 32

def mlp_hidden_width(d_model: int, mlp_ratio: float | None = None) -> int:
    ratio = 8 / 3 if mlp_ratio is None else mlp_ratio
    return max(MLP_WIDTH_MULTIPLE, int(ratio * d_model) // MLP_WIDTH_MULTIPLE * MLP_WIDTH_MULTIPLE)

#the bins' boundaries from 0 to 1
def value_bin_edges(bins: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.linspace(0, 1, bins + 1, device=device)

#the eval head's output as the expected score, 0 to 1: one logit is the score's logit, more are the bins' logits
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
        #B T d_model in and out
        up_projection = self.up_projection(x)
        gate = self.gate_projection(x)

        intermediate = F.silu(gate) * up_projection

        output = self.down_projection(intermediate)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, position_templates: int, mlp_hidden: int, qk_ratio: float = 1.0, kv_heads: int | None = None):
        super().__init__()
        #prenorm encoder block, no causal mask:
        #x = x + attention(ln1(x)), then x = x + mlp(ln2(x))

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads #these mmust be divisible
        #queries and keys only decide where a head looks, so they can be narrower than the values
        self.d_qk_head = max(1, round(qk_ratio * self.d_head))
        #grouped-query attention: each key/value head is shared by a group of neighbouring heads, None is one per head
        self.kv_heads = kv_heads or n_heads
        if n_heads % self.kv_heads != 0:
            raise ValueError(f"{n_heads} heads cannot be split into groups for {self.kv_heads} key/value heads")
        self.heads_per_kv = n_heads // self.kv_heads
        self.position_templates = position_templates

        self.Q = nn.Linear(self.d_model, self.n_heads * self.d_qk_head)
        self.K = nn.Linear(self.d_model, self.kv_heads * self.d_qk_head)
        self.V = nn.Linear(self.d_model, self.kv_heads * self.d_head)

        #QK-norm: each head's query and key at unit rms, so the scores cannot grow by the projections growing
        self.q_norm = nn.RMSNorm(self.d_qk_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.d_qk_head, eps=1e-5)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.combine_heads = nn.Linear(self.d_model, self.d_model)

        self.mlp = SwiGLU(self.d_model, mlp_hidden)

        self.template_mix = nn.Linear(self.d_model, position_templates * self.n_heads)

        #a learned score per head for attending to nothing; zeros draw no random numbers, so the other weights start the same
        self.sinks = nn.Parameter(torch.zeros(self.n_heads))


    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        #x is B T d_model, templates is position_templates 64 64 and shared by every block
        B, T, d_model = x.shape
        residual = x
        x_norm = self.ln1(residual)

        Q = self.Q(x_norm) #B T n_heads * d_qk_head
        K = self.K(x_norm) #B T kv_heads * d_qk_head
        V = self.V(x_norm) #B T kv_heads * d_head

        Q_heads = self.q_norm(Q.reshape(B, T, self.n_heads, self.d_qk_head)).transpose(1, 2) #B n_heads T d_qk_head
        K_heads = self.k_norm(K.reshape(B, T, self.kv_heads, self.d_qk_head)).transpose(1, 2)
        V_heads = V.reshape(B, T, self.kv_heads, self.d_head).transpose(1, 2) #B kv_heads T d_head

        #head h reads key/value head h // heads_per_kv
        K_heads = K_heads.repeat_interleave(self.heads_per_kv, dim=1)
        V_heads = V_heads.repeat_interleave(self.heads_per_kv, dim=1) #B n_heads T d_head

        attention = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(self.d_qk_head) #B n_heads T T

        #B 64 n_heads position_templates
        position_template_coeffs = self.template_mix(x_norm[:, :NUM_SQUARES, :]).reshape(B, NUM_SQUARES, self.n_heads, self.position_templates)
        bias = torch.einsum("biht,tij->bhij", position_template_coeffs, templates) #B n_heads 64 64

        #64 64 padded to T T with zeros for the metadata tokens
        metadata_tokens = T - NUM_SQUARES
        bias = F.pad(bias, (0, metadata_tokens, 0, metadata_tokens))
        attention = attention + bias

        sink = self.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1) #B n_heads T 1

        attention = torch.cat([attention, sink], dim=-1) #the sink as an extra last column, B n_heads T T+1
        attention = ATTENTION_CAP * torch.tanh(attention / ATTENTION_CAP)

        #the sink column is dropped after the softmax: what a head puts there reads nothing
        softmax_attention = torch.softmax(attention, dim=-1)[:, :, :, :-1]

        attention_weights = softmax_attention @ V_heads #B n_heads T d_head

        attention_weights = attention_weights.transpose(1, 2).reshape(B, T, self.d_model)

        combined_head_weights = self.combine_heads(attention_weights) #B T d_model

        residual = residual + combined_head_weights

        x_norm2 = self.ln2(residual)
        residual = residual + self.mlp(x_norm2)

        return residual


class ChessTransformer(nn.Module):
    #mlp_hidden None is mlp_hidden_width(d_model)
    def __init__(self, d_model: int = 128, num_layers: int = 4, n_heads: int = 4, position_templates: int = 8, mlp_hidden: int | None = None, qk_ratio: float | list[float] = 1.0, kv_heads: int | None = None, loops: int = 1, value_bins: int = VALUE_BINS):
        super().__init__()
        #the input is B x 70 board tokens, torch.long
        #the 6 metadata tokens are also attention's scratch space (registers): a 64 token board without them plays worse
        self.d_model = d_model
        #looped depth: the whole stack runs this many times with the same weights
        if loops < 1:
            raise ValueError(f"loops is how many times the blocks run, at least 1, got {loops}")
        self.loops = loops
        if value_bins < 1:
            raise ValueError(f"value_bins is the eval head's bins over the expected score, at least 1, got {value_bins}")

        self.embed = nn.Embedding(BOARD_VOCAB_SIZE, self.d_model)

        self.position_embed = nn.Parameter(torch.randn(TOKENS_PER_BOARD, self.d_model))

        #attention bias templates, each 64x64, shared by every block
        self.templates = nn.Parameter(torch.zeros(position_templates, NUM_SQUARES, NUM_SQUARES))

        #one q/k width for every block, or one per block
        qk_ratios = list(qk_ratio) if isinstance(qk_ratio, (list, tuple)) else [qk_ratio] * num_layers
        if len(qk_ratios) != num_layers:
            raise ValueError(f"{len(qk_ratios)} qk ratios for {num_layers} blocks, give one or one per block")

        if mlp_hidden is None:
            mlp_hidden = mlp_hidden_width(d_model)

        #a ModuleList since every block also needs the templates
        self.encoder = nn.ModuleList([TransformerBlock(self.d_model, n_heads, position_templates, mlp_hidden, qk_ratios[layer], kv_heads) for layer in range(num_layers)])

        self.board_down_project = nn.Sequential(
            nn.Linear(self.d_model, VALUE_TOKEN_DIM),
            nn.GELU()
        )

        #the board's expected score, see expected_score
        self.eval_head = nn.Sequential(
            nn.Linear(TOKENS_PER_BOARD * VALUE_TOKEN_DIM, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, value_bins)
        )

        self.register_buffer("from_squares", torch.tensor(MOVE_FROM_SQUARES, dtype=torch.long))
        self.register_buffer("to_squares", torch.tensor(MOVE_TO_SQUARES, dtype=torch.long))

        #the 176 promotions: move id, landing square and piece (0-3)
        self.register_buffer("promotion_move_ids", torch.tensor(PROMOTION_MOVE_IDS, dtype=torch.long))
        self.register_buffer("promotion_to_squares", torch.tensor(PROMOTION_TO_SQUARES, dtype=torch.long))
        self.register_buffer("promotion_pieces", torch.tensor(PROMOTION_PIECES, dtype=torch.long))

        self.from_projection = nn.Linear(self.d_model, self.d_model)
        self.to_projection = nn.Linear(self.d_model, self.d_model)
        self.promotion_projection = nn.Linear(self.d_model, 4) #one per piece a pawn can become

        self.ln = nn.LayerNorm(self.d_model)

    #bf16 (or fp16) weights for all but the heads, which forward runs in fp32 on fp32 inputs. The heads are left alone
    #rather than converted back, which would round their weights to the half precision
    def half_except_heads(self, dtype: torch.dtype = torch.bfloat16) -> "ChessTransformer":
        heads = {id(head) for head in (self.from_projection, self.to_projection, self.promotion_projection, self.board_down_project, self.eval_head)}
        for child in self.children():
            if id(child) not in heads:
                child.to(dtype)
        for name, parameter in list(self.named_parameters(recurse=False)):
            setattr(self, name, nn.Parameter(parameter.detach().to(dtype)))
        return self

    def forward(self, board_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #B x TOKENS_PER_BOARD
        B = board_state.shape[0]

        embed_state = self.embed(board_state) #B T d_model
        embed_state = embed_state + self.position_embed

        for _ in range(self.loops):
            for block in self.encoder:
                embed_state = block(embed_state, self.templates)
        model_output = self.ln(embed_state)

        with torch.autocast(device_type=board_state.device.type, enabled=False):

            #from and to vectors for the 64 squares, the metadata tokens left out
            from_vector = self.from_projection(model_output[:, :NUM_SQUARES, :].float()) #B 64 d_model
            to_vector = self.to_projection(model_output[:, :NUM_SQUARES, :].float()) #B 64 d_model
            promotion_logits = self.promotion_projection(model_output[:, :NUM_SQUARES, :].float()) #B 64 4

            #how much the piece on each square wants to go to each other square, like attention scores
            logits = (from_vector @ to_vector.transpose(1, 2)) / math.sqrt(self.d_model) #B 64 64

            combined_logits = logits[:, self.from_squares, self.to_squares] #B NUM_MOVE_IDS

            #only the promotions get the promotion term: adding it to every move let a constant drift up until bf16
            #could not tell logits apart
            combined_logits[:, self.promotion_move_ids] += promotion_logits[:, self.promotion_to_squares, self.promotion_pieces]

            compressed_board = self.board_down_project(model_output.float()).reshape(B, -1) #B T * VALUE_TOKEN_DIM

            board_eval_logits = self.eval_head(compressed_board) #B value_bins

        return combined_logits, board_eval_logits
