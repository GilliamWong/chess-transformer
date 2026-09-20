# Chess Transformer

A small transformer that learns to play chess by imitating Magnus Carlsen's games. It looks at a
position and predicts the move that was played, with no search. The games are included in the
repository (`pgn_data/Carlsen.pgn`, 7,818 games, about 724,000 positions).

The much larger [Lichess Elite Database](https://database.nikonoel.fr/) (games between strong Lichess
players, one PGN per month, 27 GB unzipped) is not in the repository. To use it, download the monthly
archives and unzip the PGNs into `pgn_data/lichess_elite/`, then:

```
uv run preprocess.py      every game to arrays in processed_data/ (about 180 GB, a couple of hours, resumable)
uv run select_games.py    picks the games to train on and writes the "baseline" training set
```

`main.py` trains on the set named by `TRAINING_SET`. Set it to `"carlsen"` to train on the included
Carlsen games alone, which needs neither of the two steps above.

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
uv run download_engines.py
```

Then:

```
uv run evaluate_elo.py                     100 games against Stockfish limited to 1320 Elo
uv run evaluate_elo.py --elo 1500          against a stronger setting
uv run evaluate_elo.py --opponent random   against random moves, as a sanity check
uv run evaluate_elo.py --opponent human    opens the Cute Chess board so you can play it yourself
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
| `games.py` | PGN to training examples |
| `data.py` | The train/validation split by game, the dataset and loaders |
| `preprocess.py` | The Lichess PGNs to arrays on disk, one process per month |
| `select_games.py` | Chooses games from those arrays (time control, no time forfeits) and writes a training set |
| `model.py` | The transformer: board tokens in, a score for each of the 1,968 moves out |
| `main.py` | The training loop |
| `player.py` | Picks the model's move in a position, with illegal moves masked out |
| `uci.py` | Lets chess programs talk to the model over the UCI protocol |
| `evaluate_elo.py` | Matches against Stockfish, run by cutechess-cli |
| `download_engines.py` | Fetches Stockfish and Cute Chess into `engines/` |
