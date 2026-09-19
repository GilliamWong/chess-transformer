import torch
from torch import nn
from encoding import BOARD_VOCAB_SIZE, TOKENS_PER_BOARD, NUM_MOVE_IDS

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

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )

        #all layers initialized to the same weights, but should be fine in this instance
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer, 
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, NUM_MOVE_IDS)



        

