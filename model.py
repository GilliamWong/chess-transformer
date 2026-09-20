import torch
from torch import nn
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, NUM_MOVE_IDS

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        #this is all from memory so bear with me
        #the input is B TOKENS_PER_BOARD d_model

        #this is encoder attention so no causal masking needed, unless we want to predict more than one output position?
        #but I believe that's a future project. 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        model_output = self.encoder(embed_state).mean(dim=1)

        #take the model output and project into output dim
        output_logits = self.output_projection(model_output)

        #return
        return output_logits

        

