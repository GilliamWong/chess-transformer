import torch
from torch import nn
import torch.nn.functional as F
import math
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, NUM_MOVE_IDS, MOVE_FROM_SQUARES, MOVE_TO_SQUARES, NUM_SQUARES, NUM_HISTORY_CODES, PROMOTION_MOVE_IDS, PROMOTION_TO_SQUARES, PROMOTION_PIECES

VALUE_TOKEN_DIM = 16
NUM_RESULTS = 3

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
    def __init__(self, d_model: int, n_heads: int, position_templates: int, dropout: float, swiglu: bool = False):
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
        self.position_templates = position_templates
        self.dropout = dropout

        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.combine_heads = nn.Linear(self.d_model, self.d_model)

        #setting that models trained with the other one keep loading. 
        if swiglu:
            self.mlp = SwiGLU(self.d_model, 8 * math.ceil(self.d_model / 3))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, 4 * self.d_model),
                nn.GELU(),
                nn.Linear(4 * self.d_model, self.d_model)
            )

        self.template_mix = nn.Linear(self.d_model, position_templates * self.n_heads)

        self.attn_dropout = nn.Dropout(dropout)
        self.head_mix_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        #reminder that x is of shape B T d_model, templates is position_templates 64 64 and shared by every block
        #dims:
        B, T, d_model = x.shape
        residual = x
        x_norm = self.ln1(residual)
        
        #first create Q K V
        Q = self.Q(x_norm) #shape B T d_model
        K = self.K(x_norm) #shape B T d_model
        V = self.V(x_norm) #shape B T d_model

        #reshape last dim into heads for q and k, we need to reshape first and then transpose to satisfy 
        #the ever fickle nature of torch memory storage yay
        Q_heads = Q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2) #shape of all is is B n_heads T d_head
        K_heads = K.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V_heads = V.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        #perform attention then softmax
        attention = (Q_heads @ K_heads.transpose(-2, -1)) / math.sqrt(self.d_head) #should be B n_heads T T

        #B 64 n_heads * position_templates, reshaped to B 64 n_heads position_templates
        position_template_coeffs = self.template_mix(x_norm[:, :NUM_SQUARES, :]).reshape(B, NUM_SQUARES, self.n_heads, self.position_templates)
        bias = torch.einsum("biht,tij->bhij", position_template_coeffs, templates) #B n_heads 64 64

        #pad takes (left, right) pairs starting from the last dim, so this grows 64 64 to T T with zeros for the metadata
        metadata_tokens = T - NUM_SQUARES
        bias = F.pad(bias, (0, metadata_tokens, 0, metadata_tokens))
        attention = attention + bias

        softmax_attention = self.attn_dropout(torch.softmax(attention, dim=-1))

        attention_weights = softmax_attention @ V_heads #B n_heads T T @ B n_heads T d_head, output is B n_heads T d_head

        #reshape and transpose the heads back together
        attention_weights = attention_weights.transpose(1, 2).reshape(B, T, self.d_model)

        combined_head_weights = self.head_mix_dropout(self.combine_heads(attention_weights)) #B T d_model @ d_head d_model = B T d_model

        residual = residual + combined_head_weights

        #now for the mlp
        x_norm2 = self.ln2(residual)
        residual = residual + self.mlp_dropout(self.mlp(x_norm2))

        return residual


class ChessTransformer(nn.Module):
    def __init__(self, d_model: int = 128, num_layers: int = 4, n_heads: int = 4, position_templates: int = 16, history: int = 0, swiglu: bool = False, dropout: float = 0.1):
        super().__init__()
        #the dim of the board input is 70, with a torch.long dtype
        self.d_model = d_model

        self.history = history

        #one mebedding per possible board token, each of d_model size
        self.embed = nn.Embedding(BOARD_VOCAB_SIZE, self.d_model)

        #we'll have one positional encoding per token position on the board
        self.position_embed = nn.Parameter(torch.randn(TOKENS_PER_BOARD, self.d_model))

        #positional bias templates per piece, each the size of the entire board
        self.templates = nn.Parameter(torch.zeros(position_templates, NUM_SQUARES, NUM_SQUARES))

        #a ModuleList since Sequential can only hand a block one input, and they need the templates too
        self.encoder = nn.ModuleList([TransformerBlock(self.d_model, n_heads, position_templates, dropout, swiglu) for _ in range(num_layers)])

        self.board_down_project = nn.Sequential(
            nn.Linear(self.d_model, VALUE_TOKEN_DIM),
            nn.GELU()
        ) 

        #value head which predicts win loss or draw. Forces the model to learn some sense of long term planning from games. 
        self.value_head = nn.Sequential(
            nn.Linear(TOKENS_PER_BOARD * VALUE_TOKEN_DIM, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_RESULTS)
        )

        self.learned_history = nn.ModuleList(
            nn.Embedding(NUM_HISTORY_CODES, self.d_model, padding_idx=0)
            for _ in range(self.history)
        )

        for embedding in self.learned_history:
            nn.init.zeros_(embedding.weight)

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

    def forward(self, board_state: torch.Tensor, prev_boards: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        #input is B x TOKENS_PER_BOARD
        B = board_state.shape[0]

        #embed state is B tokens_per_board d_model now
        embed_state = self.embed(board_state)

        #add the position embed which is already the right dim
        embed_state = embed_state + self.position_embed

        #history is passed in B H 64, where H is self.history. Sparse board of mostly zeroes
        #where numbers 1-12 represent "this square changed, this is the piece that used to be here"
        for i in range(self.history):
            curr_history = self.learned_history[i](prev_boards[:, i, :])
            embed_state[:, :NUM_SQUARES] = embed_state[:, :NUM_SQUARES] + curr_history

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

            #shape B 3, stored as loss, draw, win
            board_value_logits = self.value_head(compressed_board)            
        return combined_logits, board_value_logits

        

