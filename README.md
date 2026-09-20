# Chess Transformer

A small transformer that learns to play chess by imitating Magnus Carlsen's games. It looks at a
position and predicts the move that was played, with no search. The games are included in the
repository (`pgn_data/Carlsen.pgn`, 7,818 games, about 724,000 positions).

The model can also be trained on games from the [Lichess Elite Database](https://database.nikonoel.fr/)
(games between strong Lichess players, 27 GB of PGNs, about 30 million games). Those files are far too
big for git. They were encoded once into arrays in `processed_data/`, and a training set of 71 million
positions from rapid and classical games was written there as `baseline_*.bin`. The one-off scripts that
did this are not part of the codebase: they are in the git history (`preprocess.py`, `select_games.py`)
and a copy sits in `processed_data/scripts_used/` next to the data they produced.

`main.py` trains on the set named by `TRAINING_SET`. Set it to `"carlsen"` to train on the included
Carlsen games alone, which needs none of the above.

## Setup

The only thing to install yourself is [uv](https://docs.astral.sh/uv/getting-started/installation/).
It installs Python and every dependency from `pyproject.toml` and `uv.lock`.

```
git clone https://github.com/GilliamWong/chess-transformer.git
cd chess-transformer
uv sync
```

On Windows and Linux this installs the CUDA 13.0 build of PyTorch, which needs a recent NVIDIA driver.
Without an NVIDIA GPU everything still runs, on the CPU and much more slowly.

## Train

```
uv run main.py
```

This trains for 10 epochs and saves the weights of the epoch with the best validation loss to
`model.pt`. On the Carlsen games an epoch takes about 25 seconds on an RTX 5070 Ti.
The settings are the constants at the top of `main.py`.

## Measure its strength and play against it

This part is Windows only. It needs two programs that are too big for git, Stockfish and Cute Chess:

```
uv run python -m play.download_engines
```

Then:

```
uv run python -m play.evaluate_elo                     100 games against Stockfish limited to 1320 Elo
uv run python -m play.evaluate_elo --elo 1500          against a stronger setting
uv run python -m play.evaluate_elo --opponent random   against random moves, as a sanity check
uv run python -m play.evaluate_elo --opponent human    opens the Cute Chess board so you can play it yourself
```

The games of the last match are saved to `match.pgn`.

## Tests

```
uv run pytest
```

## Layout

| File | What it does |
|---|---|
| `encoding.py` | Boards to 70 tokens and back, moves to one of 1,968 ids and back |
| `data.py` | PGN to training examples, the train/validation split by game, the dataset and loaders |
| `model.py` | The transformer: board tokens in, a score for each of the 1,968 moves out |
| `main.py` | The training loop |
| `play/player.py` | Picks the model's move in a position, with illegal moves masked out |
| `play/uci.py` | Lets chess programs talk to the model over the UCI protocol |
| `play/evaluate_elo.py` | Matches against Stockfish, run by cutechess-cli |
| `play/download_engines.py` | Fetches Stockfish and Cute Chess into `engines/` |

Everything in `play/` is run from the repository root with `python -m play.<name>`, so that it can import
the training code next to it.
