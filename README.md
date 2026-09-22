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

The checkpoint with the best validation loss is saved to `artifacts/runs/<name>/model.pt`, so every run keeps
its own and none is overwritten. The tools play and measure the run that finished last unless told otherwise. It holds the model's settings as well as its weights, so a model of any
size can be played (`--checkpoint artifacts/runs/<name>/model.pt` plays an older run).

`TRAINING_SET` defaults to `"baseline"`: 71 million positions from rapid and classical games of the
[Lichess Elite Database](https://database.nikonoel.fr/), which is too big for git. The one-off scripts
that built it are in the git history (`preprocess.py`, `select_games.py`). A fresh clone has only the
Carlsen games, so set `TRAINING_SET = "carlsen"`.

### Multiple GPUs

`train_ddp.py` is a separate Distributed Data Parallel trainer; `main.py` is unchanged.
Use a Linux machine with a CUDA/NCCL PyTorch build and launch one process per GPU:

```bash
uv run torchrun --standalone --nproc-per-node=2 train_ddp.py --minutes 30 --no-tensorboard
uv run torchrun --standalone --nproc-per-node=4 train_ddp.py --positions 10_000_000 --batch-size 1024 --no-tensorboard
```

It supports the model, history, future-head, Stockfish-target and mixture options from `main.py`.
`--batch-size` is the **global** batch: 1,024 means 512 positions per GPU with two GPUs, or 256 with four.
It must divide evenly across GPUs. For a mixture, each GPU's batch must also divide evenly among the
named shares (`baseline+baseline+baseline+fishtest` has four). Mixtures preserve those proportions and
sample with replacement using independent random streams. Single datasets are shuffled into disjoint
shards each epoch; incomplete global batches are dropped. Position budgets count all GPUs together
and round down to whole global batches. The existing 500-step warmup still applies.

`--minutes` measures training wall time, with the first step (including compilation), validation and
checkpoint writing excluded. All processes share the same clock, learning-rate schedule and stop decision.
Two GPUs for 30 minutes use 60 GPU-minutes: timed runs have a GPU-count suffix in the results table's
budget group, and settings record `world_size`, `per_gpu_batch_size` and `gpu_seconds`.
The future loss is normalized over all known future targets across GPUs, including batches where one
GPU has no known targets.

Validation runs on GPU 0 using the existing validation batch size and cohort; the other GPUs wait.
Only process 0 writes logs, checkpoints and the results page. `model.pt` stores ordinary unwrapped
weights and works with the existing playing and evaluation tools. A supplied `--name` must be a new run
folder. As in `main.py`, checkpoints are selected on periodic validation, while final metrics describe
the final weights on the full validation set. Training-resume checkpoints are not implemented.

`--no-compile` enables eager execution for debugging. Native Windows does not supply NCCL, so this
trainer's GPU mode requires Linux. `--device cpu --no-compile` uses Gloo for distributed smoke tests;
it is not a substitute for measuring the CUDA path. The DDP tests use two and four CPU processes and
temporary synthetic datasets.

### Stockfish's moves as the target

`baseline_sf` is a copy of the first 10 million positions of `baseline` (whole games, so `--history` works) in
which Stockfish has looked at every position: its 4 best moves and the score after each, at depth 10. The same
positions can then be trained on with either target, which is the A/B test of the labels:

```
uv run main.py --training-set baseline_sf                            the move the human played
uv run main.py --training-set baseline_sf --policy-target stockfish  Stockfish's best moves
```

With `stockfish` the target is a distribution over those moves: a softmax over the expected score after each
(0 lost to 1 won), divided by `--target-temperature` (0.02: a move that gives up 2% of the expected score gets 1/e
of the best move's share). Validation loss and accuracy are then against Stockfish's moves, so they are not
comparable with a `human` run's. Compare the two on puzzles and Elo, or measure a checkpoint against the other target:

```
uv run python -m tools.evaluate_checkpoint artifacts/runs/<run>/model.pt --training-set baseline_sf --policy-target stockfish
uv run python -m tools.label_with_stockfish     makes the set, about 4 hours for 10 million positions. Can be stopped and continued
```

In a mix (`baseline_sf+fishtest`) the sets without labels keep the move that was played.

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

The root training and model files are:

| File | What it does |
|---|---|
| `encoding.py` | Boards to 70 tokens and back, moves to one of 1,968 ids and back |
| `data.py` | PGN to training examples, the split by game, the dataset and loaders |
| `model.py` | The transformer: board tokens in, a score for each move out |
| `main.py` | The single-GPU training loop |
| `train_ddp.py` | The separate multi-GPU DDP training loop |

Everything else is kept out of the way:

| Folder | What is in it |
|---|---|
| `tools/` | Checkpoint saving and loading, the training budget, the run table and results page, the TensorBoard launcher, labelling a training set with Stockfish, and `play/` (choosing the model's move, the UCI protocol, matches against Stockfish). Run as modules from the root: `python -m tools.<name>` |
| `artifacts/` | No code: the PGNs, preprocessed training sets, checkpoints, TensorBoard runs and chess engines. Only `pgn_data/Carlsen.pgn` is in git |
| `tests/` | `uv run pytest` |
