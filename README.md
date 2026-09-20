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
uv run main.py                          4 minutes of training (TRAIN_MINUTES_BUDGET), the standard budget for comparing models
uv run main.py --note "final layer norm"    the same, with a note for the results page
uv run main.py --minutes 30             a longer run
uv run main.py --positions 10_000_000   a budget in positions instead, about 3 minutes per 10 million on an RTX 5070 Ti
uv run main.py --d-model 256 --layers 8 --heads 8 --lr 3e-3    a different model, see --help for every flag
uv run python -m tools.compare_runs     a table of every finished run, grouped by budget, best first
```

Training starts TensorBoard and opens it in the browser (live plots of loss, accuracy and learning rate
at localhost:6006, every run overlaid). `--no-tensorboard` skips that, and
`uv run tensorboard --logdir artifacts/runs` shows the plots without training.

Runs with the same budget are comparable: the same `--positions` for equal data, the same `--minutes` for
equal GPU time (compiling and evaluating are not counted, and a slower model simply gets fewer steps). Every
finished run is added to `artifacts/progress.html`, a table and a chart of validation accuracy run after
run. The page reloads itself, so it can stay open.

A step is one batch of `BATCH_SIZE` (1024) positions. The other settings are constants at the top of `main.py`:

| Constant | Meaning |
|---|---|
| `TRAIN_MINUTES_BUDGET` | Minutes to train for when neither `--minutes` nor `--positions` is given |
| `TRAINING_SET` | `"carlsen"` for the 7,818 Carlsen games in the repository (also set `DROPOUT = 0.1`), or the name of a set in `artifacts/processed_data/` |
| `PEAK_LEARN_RATE`, `WARMUP_STEPS` | The learning rate warms up to the peak, then decays to zero at the last step |
| `EVAL_EVERY`, `EVAL_BATCHES` | How often, and on how much of the validation set, progress is reported |

The checkpoint with the best validation loss is saved to `artifacts/model.pt` and to
`artifacts/runs/<name>/model.pt`. It holds the model's settings as well as its weights, so a model of any
size can be played (`--checkpoint artifacts/runs/<name>/model.pt` plays an older run).

`TRAINING_SET` defaults to `"baseline"`: 71 million positions from rapid and classical games of the
[Lichess Elite Database](https://database.nikonoel.fr/), which is too big for git. The one-off scripts
that built it are in the git history (`preprocess.py`, `select_games.py`). A fresh clone has only the
Carlsen games, so set `TRAINING_SET = "carlsen"`.

## Play against it and measure its strength

Windows only. Fetch Stockfish and Cute Chess once, then run a match:

```
uv run python -m tools.play.download_engines
uv run python -m tools.play.evaluate_elo                     100 games against Stockfish limited to 1320 Elo
uv run python -m tools.play.evaluate_elo --elo 1500          a stronger setting
uv run python -m tools.play.evaluate_elo --opponent random   random moves, as a sanity check
uv run python -m tools.play.evaluate_elo --opponent human    opens a board so you can play it yourself
```

The games of the last match are saved to `artifacts/match.pgn`.

## Tests

```
uv run pytest
```

## Layout

The four files in the root are the project itself:

| File | What it does |
|---|---|
| `encoding.py` | Boards to 70 tokens and back, moves to one of 1,968 ids and back |
| `data.py` | PGN to training examples, the split by game, the dataset and loaders |
| `model.py` | The transformer: board tokens in, a score for each move out |
| `main.py` | The training loop |

Everything else is kept out of the way:

| Folder | What is in it |
|---|---|
| `tools/` | Checkpoint saving and loading, the training budget, the run table and results page, the TensorBoard launcher, and `play/` (choosing the model's move, the UCI protocol, matches against Stockfish). Run as modules from the root: `python -m tools.<name>` |
| `artifacts/` | No code: the PGNs, preprocessed training sets, checkpoints, TensorBoard runs and chess engines. Only `pgn_data/Carlsen.pgn` is in git |
| `tests/` | `uv run pytest` |
