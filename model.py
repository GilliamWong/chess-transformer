import torch
from torch import nn
import torch.nn.functional as F
import math
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, NUM_MOVE_IDS, MOVE_FROM_SQUARES, MOVE_TO_SQUARES, NUM_SQUARES, PROMOTION_MOVE_IDS, PROMOTION_TO_SQUARES, PROMOTION_PIECES
from encoding import (NUM_PIECE_CODES, NUM_FLAG_VALUES, NUM_EN_PASSANT_VALUES, TURN_INDEX, WHITE_KINGSIDE_INDEX, BLACK_QUEENSIDE_INDEX,
                      EN_PASSANT_INDEX, TURN_TOKEN_OFFSET, CASTLING_TOKEN_OFFSET, EN_PASSANT_TOKEN_OFFSET)

VALUE_TOKEN_DIM = 16

#the planes the template embedding reads per square: the 12 piece planes first (the maps spread those), then empty, side to
#move, the four castling rights and the en passant square, the last ones the same on every square
NUM_PIECE_PLANES = NUM_PIECE_CODES - 1
NUM_CASTLING_SLOTS = BLACK_QUEENSIDE_INDEX - WHITE_KINGSIDE_INDEX + 1
NUM_INPUT_PLANES = NUM_PIECE_CODES + NUM_FLAG_VALUES + NUM_CASTLING_SLOTS * NUM_FLAG_VALUES + NUM_EN_PASSANT_VALUES #87
TEMPLATE_EMBEDDING_NORM_EPSILON = 1e-3 #leela's

#a truncated normal cut at 2 standard deviations, the std corrected for the cut: how leela's (flax's) weights start
def truncated_normal_(tensor: torch.Tensor, variance: float) -> torch.Tensor:
    std = math.sqrt(variance) / 0.87962566103423978
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)

#an nn.Linear started like leela's: lecun normal weights, or deepnorm's (variance beta / fan_avg) when deepnorm_beta is
#given, and a zero bias
def leela_linear(in_features: int, out_features: int, deepnorm_beta: float | None = None) -> nn.Linear:
    linear = nn.Linear(in_features, out_features)
    variance = 1 / in_features if deepnorm_beta is None else deepnorm_beta / ((in_features + out_features) / 2)
    truncated_normal_(linear.weight, variance)
    nn.init.zeros_(linear.bias)
    return linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()

        self.up_projection = nn.Linear(d_model, d_hidden, bias=False)
        self.gate_projection = nn.Linear(d_model, d_hidden, bias=False)
        self.down_projection = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #this would take something like B T d_model
        #and then return the same thing
        up_projection = self.up_projection(x)
        gate = self.gate_projection(x)

        intermediate = F.silu(gate) * up_projection

        output = self.down_projection(intermediate)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, position_templates: int, mlp_ratio: float | None = None, qk_ratio: float = 1.0, kv_heads: int | None = None):
        super().__init__()
        #this is all from memory so bear with me
        #the input is B TOKENS_PER_BOARD d_model

        #this is encoder attention so no causal masking needed, unless we want to predict more than one output position?
        #but I believe that's a future project. 

        #so the flow is: take input, create Q K and V
        #then split Q K V into separate heads in torch 
        #perform attention operation per head, then apply softmax
        #matmul with V, then recombine heads
        #matmul with head mixer weights and then output

        #this also contains the mlp and norms
        #so the entire flow is, with prenorm
        #x = x + attention(layernorm1(x))
        #project up with a linear layer into the mlp dim
        #then project back down into input dim with another linear layer
        #so x = x + mlp(layernorm2(x))

        #since this is prenorm we need to add a final norm in the chess transformer model right before
        #the output head. 

        #so in this init we need Q, K, V, Layernorm1, layernorm2, mlp up, mlp down, head combine 

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads #these mmust be divisible
        #the queries and keys only decide where a head looks, so they can be narrower than the values it brings back
        self.d_qk_head = max(1, round(qk_ratio * self.d_head))
        #grouped-query attention: fewer key/value heads than query heads, each shared by a group of neighbouring heads.
        #None is one for every head
        self.kv_heads = kv_heads or n_heads
        if n_heads % self.kv_heads != 0:
            raise ValueError(f"{n_heads} heads cannot be split into groups for {self.kv_heads} key/value heads")
        self.heads_per_kv = n_heads // self.kv_heads
        self.position_templates = position_templates

        self.Q = nn.Linear(self.d_model, self.n_heads * self.d_qk_head)
        self.K = nn.Linear(self.d_model, self.kv_heads * self.d_qk_head)
        self.V = nn.Linear(self.d_model, self.kv_heads * self.d_head)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.combine_heads = nn.Linear(self.d_model, self.d_model)

        if mlp_ratio is None:
            d_hidden = 8 * math.ceil(self.d_model / 3)
        else:
            d_hidden = max(8, 8 * round(mlp_ratio * self.d_model / 8))

        self.mlp = SwiGLU(self.d_model, d_hidden)

        self.template_mix = nn.Linear(self.d_model, position_templates * self.n_heads)


    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        #reminder that x is of shape B T d_model, templates is position_templates 64 64 and shared by every block
        #dims:
        B, T, d_model = x.shape
        residual = x
        x_norm = self.ln1(residual)
        
        #first create Q K V
        Q = self.Q(x_norm) #shape B T n_heads * d_qk_head
        K = self.K(x_norm) #shape B T kv_heads * d_qk_head
        V = self.V(x_norm) #shape B T kv_heads * d_head

        #reshape last dim into heads for q and k, we need to reshape first and then transpose to satisfy
        #the ever fickle nature of torch memory storage yay
        Q_heads = Q.reshape(B, T, self.n_heads, self.d_qk_head).transpose(1, 2) #shape of q and k is B n_heads T d_qk_head
        K_heads = K.reshape(B, T, self.kv_heads, self.d_qk_head).transpose(1, 2)
        V_heads = V.reshape(B, T, self.kv_heads, self.d_head).transpose(1, 2) #B kv_heads T d_head

        #every key/value head serves its group of query heads: head h reads key/value head h // heads_per_kv
        K_heads = K_heads.repeat_interleave(self.heads_per_kv, dim=1)
        V_heads = V_heads.repeat_interleave(self.heads_per_kv, dim=1) #B n_heads T d_head

        #perform attention then softmax
        attention = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(self.d_qk_head) #should be B n_heads T T

        #B 64 n_heads * position_templates, reshaped to B 64 n_heads position_templates
        position_template_coeffs = self.template_mix(x_norm[:, :NUM_SQUARES, :]).reshape(B, NUM_SQUARES, self.n_heads, self.position_templates)
        bias = torch.einsum("biht,tij->bhij", position_template_coeffs, templates) #B n_heads 64 64

        #pad takes (left, right) pairs starting from the last dim, so this grows 64 64 to T T with zeros for the metadata
        metadata_tokens = T - NUM_SQUARES
        bias = F.pad(bias, (0, metadata_tokens, 0, metadata_tokens))
        attention = attention + bias

        softmax_attention = torch.softmax(attention, dim=-1)

        attention_weights = softmax_attention @ V_heads #B n_heads T T @ B n_heads T d_head, output is B n_heads T d_head

        #reshape and transpose the heads back together
        attention_weights = attention_weights.transpose(1, 2).reshape(B, T, self.d_model)

        combined_head_weights = self.combine_heads(attention_weights) #B T d_model @ d_head d_model = B T d_model

        residual = residual + combined_head_weights

        #now for the mlp
        x_norm2 = self.ln2(residual)
        residual = residual + self.mlp(x_norm2)

        return residual


class TemplateEmbedding(nn.Module):
    #the square tokens built from the whole board before the first block, so every square starts out knowing more than its
    #own piece. Each piece gives one feature per map (piece_features), and map m, a learned 64x64 grid like the blocks'
    #templates, spreads that feature from the piece's square to every square. A square's own planes and those features
    #then go through leela chess zero's input embedding (lczero-training, embedding.py): a linear layer, mish, a norm, a
    #multiplicative and an additive gate per square and channel, and a small ffn.
    #Shallow models gain a lot from it, deep ones nothing: 192 maps against model.py's plain embedding, +0.039 val loss at
    #d128 with 4 layers, +0.015 at d256 with 4, +0.003 (noise) at d256 with 8 (README, "The current recipe")
    def __init__(self, d_model: int, maps: int, num_layers: int, token_embed: nn.Embedding):
        super().__init__()
        self.piece_features = nn.Linear(NUM_PIECE_PLANES, maps, bias=False)
        #scaled so a square's sum over the 64 others keeps the features' size
        self.maps = nn.Parameter(torch.randn(maps, NUM_SQUARES, NUM_SQUARES) / NUM_SQUARES ** 0.5)

        self.embedding = leela_linear(NUM_INPUT_PLANES + maps, d_model)
        self.norm = nn.LayerNorm(d_model, eps=TEMPLATE_EMBEDDING_NORM_EPSILON)
        #the multiplicative gate goes through relu and starts at 1, the additive one starts at 0
        self.mult_gate = nn.Parameter(torch.ones(NUM_SQUARES, d_model))
        self.add_gate = nn.Parameter(torch.zeros(NUM_SQUARES, d_model))

        #leela's deepnorm scaling for the ffn: its output times alpha, its weights starting at variance beta / fan_avg
        self.alpha = (2 * num_layers) ** -0.25
        beta = (8 * num_layers) ** -0.25
        d_hidden = d_model * 3 // 2
        self.ffn_in = leela_linear(d_model, d_hidden, deepnorm_beta=beta)
        self.ffn_out = leela_linear(d_hidden, d_model, deepnorm_beta=beta)
        self.out_norm = nn.LayerNorm(d_model, eps=TEMPLATE_EMBEDDING_NORM_EPSILON)

        #the six metadata tokens keep model.py's embedding
        self.token_embed = token_embed

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
        planes = self.planes(board_state)
        #B 64 maps: map m spreads every square's feature m over the board, square to square
        features = self.piece_features(planes[..., :NUM_PIECE_PLANES])
        board = torch.einsum("mij,bjm->bim", self.maps, features)

        squares = self.norm(F.mish(self.embedding(torch.cat([planes, board], dim=-1))))
        squares = squares * F.relu(self.mult_gate) + self.add_gate
        squares = self.out_norm(squares + self.ffn_out(F.mish(self.ffn_in(squares))) * self.alpha)

        metadata = self.token_embed(board_state[:, NUM_SQUARES:])
        return torch.cat([squares, metadata], dim=1) #B TOKENS_PER_BOARD d_model


class ChessTransformer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, n_heads: int = 4, position_templates: int = 8, mlp_ratio: float | None = None, qk_ratio: float | list[float] = 1.0, kv_heads: int | None = None, template_maps: int = 0):
        super().__init__()
        #the dim of the board input is 70, with a torch.long dtype
        self.d_model = d_model

        #one mebedding per possible board token, each of d_model size
        self.embed = nn.Embedding(BOARD_VOCAB_SIZE, self.d_model)
        #or with template_maps, the square tokens built from the whole board by that many template maps (TemplateEmbedding),
        #which pays off in shallow models
        if template_maps:
            self.embed = TemplateEmbedding(self.d_model, template_maps, num_layers, self.embed)

        #we'll have one positional encoding per token position on the board
        self.position_embed = nn.Parameter(torch.randn(TOKENS_PER_BOARD, self.d_model))

        #positional bias templates per piece, each the size of the entire board
        self.templates = nn.Parameter(torch.zeros(position_templates, NUM_SQUARES, NUM_SQUARES))

        #one q/k width for every block, or one per block: the first blocks of trained models use far fewer q/k dims
        qk_ratios = list(qk_ratio) if isinstance(qk_ratio, (list, tuple)) else [qk_ratio] * num_layers
        if len(qk_ratios) != num_layers:
            raise ValueError(f"{len(qk_ratios)} qk ratios for {num_layers} blocks, give one or one per block")

        #a ModuleList since Sequential can only hand a block one input, and they need the templates too
        self.encoder = nn.ModuleList([TransformerBlock(self.d_model, n_heads, position_templates, mlp_ratio, qk_ratios[layer], kv_heads) for layer in range(num_layers)])

        self.board_down_project = nn.Sequential(
            nn.Linear(self.d_model, VALUE_TOKEN_DIM),
            nn.GELU()
        ) 

        #predicts stockfish's eval of the current board, as the logit of the expected score it stands for
        self.eval_head = nn.Sequential(
            nn.Linear(TOKENS_PER_BOARD * VALUE_TOKEN_DIM, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1)
        )

        self.register_buffer("from_squares", torch.tensor(MOVE_FROM_SQUARES, dtype=torch.long))
        self.register_buffer("to_squares", torch.tensor(MOVE_TO_SQUARES, dtype=torch.long))

        #the 176 promotions: their move id, the square the pawn lands on and the piece (0-3)
        self.register_buffer("promotion_move_ids", torch.tensor(PROMOTION_MOVE_IDS, dtype=torch.long))
        self.register_buffer("promotion_to_squares", torch.tensor(PROMOTION_TO_SQUARES, dtype=torch.long))
        self.register_buffer("promotion_pieces", torch.tensor(PROMOTION_PIECES, dtype=torch.long))

        self.from_projection = nn.Linear(self.d_model, self.d_model)
        self.to_projection = nn.Linear(self.d_model, self.d_model)
        self.promotion_projection = nn.Linear(self.d_model, 4) #one per piece a pawn can become

        self.ln = nn.LayerNorm(self.d_model)

    def forward(self, board_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #input is B x TOKENS_PER_BOARD
        B = board_state.shape[0]

        #embed state is B tokens_per_board d_model now
        embed_state = self.embed(board_state)

        #add the position embed which is already the right dim
        embed_state = embed_state + self.position_embed

        #pass through the model layers
        #this is of dim B tokens_per_board d_model
        #we also apply layernorm as is standard with prenorm
        for block in self.encoder:
            embed_state = block(embed_state, self.templates)
        model_output = self.ln(embed_state)

        with torch.autocast(device_type=board_state.device.type, enabled=False):

            #take model output and make the from and to vectors for each board square
            #we take off the last six tokens since those are metadata
            from_vector = self.from_projection(model_output[:, :NUM_SQUARES, :].float()) #B T d_model
            to_vector = self.to_projection(model_output[:, :NUM_SQUARES, :].float()) #B T d_model
            promotion_logits = self.promotion_projection(model_output[:, :NUM_SQUARES, :].float()) #B T 4

            #take the from and to pairs and matmul them together
            #intuition is that for each square, the from vector stores smth like "what's on this square? Where do I want to go?"
            #and the to vector stores "what does this square want to receive"?
            #sorta like attention, when dot producted together, make a T T matrix that represents how much the piece on each square
            #wants to go to another square

            #B T T  
            logits = (from_vector @ to_vector.transpose(1, 2)) / math.sqrt(self.d_model)

            #B d_vocab
            combined_logits = logits[:, self.from_squares, self.to_squares]

            #the promotions of a pawn share their (from, to) pair with each other and with the plain move, so only they
            #get something added. Adding to every move let a constant drift up until bf16 could not tell logits apart
            combined_logits[:, self.promotion_move_ids] += promotion_logits[:, self.promotion_to_squares, self.promotion_pieces]

            #B T d_model turned to B T VALUE_TOKEN_DIM, then reshaped to B T * VALUE_TOKEN_DIM
            compressed_board = self.board_down_project(model_output.float()).reshape(B, -1)

            #shape B 1, the logit of the current board's expected score
            board_eval_logits = self.eval_head(compressed_board)

            #I wonder if we should instead take the model logits instead of the compressed board to predict evals
            #or keep the current system and then add another head that uses the logits to predict the eval of the next move
            #or, use the future state head somehow and roll this into that head?

        return combined_logits, board_eval_logits

        

