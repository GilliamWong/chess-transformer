import torch
from torch import nn
import math
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, NUM_MOVE_IDS

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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
        self.dropout = dropout

        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.combine_heads = nn.Linear(self.d_model, self.d_model)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.head_mix_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #reminder that x is of shape B T d_model
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
    def __init__(self, d_model: int = 128, num_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        #we take our inputs, which would just be current board state, and embed it
        #we take that embedding and feed it through the model
        #we then downproject that embedding back into whatever our output dim is
        #so the model needs layers for the embedding projection, adding positional encodings
        #the transformer itself, assuming we'll need encoders for now since no causal masking for chess
        #then an output head which downprojects the output of the encoder into 1968 output dimensions

        #the dim of the board input is 70, with a torch.long dtype

        #one mebedding per possible board token, each of d_model size
        self.embed = nn.Embedding(BOARD_VOCAB_SIZE, d_model)

        #we'll have one positional encoding per token position on the board
        self.position_embed = nn.Parameter(torch.randn(TOKENS_PER_BOARD, d_model))
        
        self.encoder = nn.Sequential(*[TransformerBlock(d_model, n_heads, dropout) for _ in range(num_layers)])

        self.output_projection = nn.Linear(d_model, NUM_MOVE_IDS)

        self.ln = nn.LayerNorm(d_model)

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        #input is B x TOKENS_PER_BOARD

        #take the input, embed it
        #take the embedded input and add the position embed as well
        #take that and pass it through our encoder layers
        #then take that, put it through the output dim, and return

        #embed state is B tokens_per_board d_model now
        embed_state = self.embed(board_state)

        #add the position embed which is already the right dim
        embed_state = embed_state + self.position_embed

        #pass through the model layers
        #this is of dim B tokens_per_board d_model, so we mean to get one token per board
        #we also apply layernorm as is standard with prenorm
        model_output = self.ln(self.encoder(embed_state)).mean(dim=1)

        #take the model output and project into output dim
        output_logits = self.output_projection(model_output)

        #return
        return output_logits

        

