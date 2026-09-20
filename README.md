# Chess Transformer

A small transformer that plays chess by predicting the move a strong player would make in a position,
with no search.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```
git clone https://github.com/GilliamWong/chess-transformer.git
cd chess-transformer
uv sync
```

This installs Python and every dependency, including the CUDA 13.0 build of PyTorch. Training needs an
NVIDIA GPU with a recent driver.

## Train

```
uv run main.py                          TRAIN_POSITIONS_BUDGET positions, about 3 minutes per 10 million on an RTX 5070 Ti
uv run main.py --positions 12_300_000   any other amount
uv run main.py --name d256_lr3e-3       a name for the run's plots, default the date and time
```

Training starts TensorBoard and opens it in the browser (live plots of loss, accuracy and learning rate
at localhost:6006, every run in `runs/` overlaid). `--no-tensorboard` skips that, and
`uv run tensorboard --logdir runs` shows the plots without training.

Training runs for a fixed number of positions rather than epochs. A step is one batch of `BATCH_SIZE`
positions, so steps = positions / 1024. The other settings are constants at the top of `main.py`:

| Constant | Meaning |
|---|---|
| `TRAIN_POSITIONS_BUDGET` | Positions to train on when `--positions` is not given |
| `TRAINING_SET` | `"carlsen"` for the 7,818 Carlsen games in the repository (also set `DROPOUT = 0.1`), or the name of a set in `processed_data/` |
| `PEAK_LEARN_RATE`, `WARMUP_STEPS` | The learning rate warms up to the peak, then decays to zero at the last step |
| `EVAL_EVERY`, `EVAL_BATCHES` | How often, and on how much of the validation set, progress is reported |

The weights with the best validation loss are saved to `model.pt`.

`TRAINING_SET` defaults to `"baseline"`: 71 million positions from rapid and classical games of the
[Lichess Elite Database](https://database.nikonoel.fr/), which is too big for git. The one-off scripts
that built it are in the git history (`preprocess.py`, `select_games.py`). A fresh clone has only the
Carlsen games, so set `TRAINING_SET = "carlsen"`.

## Play against it and measure its strength

Windows only. Fetch Stockfish and Cute Chess once, then run a match:

```
uv run python -m play.download_engines
uv run python -m play.evaluate_elo                     100 games against Stockfish limited to 1320 Elo
uv run python -m play.evaluate_elo --elo 1500          a stronger setting
uv run python -m play.evaluate_elo --opponent random   random moves, as a sanity check
uv run python -m play.evaluate_elo --opponent human    opens a board so you can play it yourself
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
| `data.py` | PGN to training examples, the split by game, the dataset and loaders |
| `model.py` | The transformer: board tokens in, a score for each move out |
| `main.py` | The training loop |
| `play/` | Choosing the model's move, the UCI protocol, matches against Stockfish |
