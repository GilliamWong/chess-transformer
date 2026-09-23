# Chess Transformer

A small transformer that plays chess by predicting the moves Stockfish would pick in a position,
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
| `TRAINING_SET` | The name of a Stockfish-labelled set in `artifacts/processed_data/` |
| `PEAK_LEARN_RATE`, `WARMUP_STEPS` | The learning rate warms up to the peak, then decays to zero at the last step |
| `EVAL_EVERY`, `EVAL_BATCHES` | How often, and on how much of the validation set, progress is reported |

The checkpoint with the best validation loss is saved to `artifacts/runs/<name>/model.pt`, so every run keeps
its own and none is overwritten. The tools play and measure the run that finished last unless told otherwise. It holds the model's settings as well as its weights, so a model of any
size can be played (`--checkpoint artifacts/runs/<name>/model.pt` plays an older run).

Checkpoints trained before the 2026-09-23 cleanup, the d640 run's included, load only from a checkout of the tag
`pre-cleanup-2026-09-23` (for example `git worktree add ../chess-transformer-pre-cleanup pre-cleanup-2026-09-23`):
they name settings the model no longer has (history, the future head, dropout, the GELU MLP, the result head), and
loading one here stops with that advice.

`TRAINING_SET` defaults to `"full_sf8_30M_mix"`: 30 million positions of whole games from the labelled
`full_sf8_20260922` set, in its mix of [Lichess Elite Database](https://database.nikonoel.fr/) games (68%, a third of
them blitz), Fishtest games (29%) and over-the-board games (3%). Processed sets are far too big for git, so a fresh
clone has no training data. A set is its boards, `<name>_{train,val}_boards.bin`, from the processed-data pipeline
(the first one-off scripts, `preprocess.py` and `select_games.py`, are in the git history, later ones in
`artifacts/processed_data/scripts_used/`), plus Stockfish's labels of every position,
`<name>_{train,val}_stockfish_{moves,evals}.bin`, which `uv run python -m tools.label_with_stockfish` writes.

### Multiple GPUs

`multi_gpu/train_ddp.py` is a separate Distributed Data Parallel trainer; `main.py` is unchanged.
Use a Linux machine with a CUDA/NCCL PyTorch build and launch one process per GPU, as a module from the repository root:

```bash
uv run torchrun --standalone --nproc-per-node=2 -m multi_gpu.train_ddp --minutes 30 --no-tensorboard
uv run torchrun --standalone --nproc-per-node=4 -m multi_gpu.train_ddp --positions 10_000_000 --batch-size 1024 --no-tensorboard
```

It supports the model, training set and target temperature options from `main.py`.
`--batch-size` is the **global** batch: 1,024 means 512 positions per GPU with two GPUs, or 256 with four.
It must divide evenly across GPUs. The training set is shuffled into disjoint shards each epoch; incomplete global
batches are dropped. Position budgets count all GPUs together and round down to whole global batches. The existing
500-step warmup still applies.

`--minutes` measures training wall time, with the first step (including compilation), validation and
checkpoint writing excluded. All processes share the same clock, learning-rate schedule and stop decision.
Two GPUs for 30 minutes use 60 GPU-minutes: timed runs have a GPU-count suffix in the results table's
budget group, and settings record `world_size`, `per_gpu_batch_size` and `gpu_seconds`.

Validation runs on GPU 0 using the existing validation batch size and cohort; the other GPUs wait.
Only process 0 writes logs, checkpoints and the results page. `model.pt` stores ordinary unwrapped
weights and works with the existing playing and evaluation tools. A supplied `--name` must be a new run
folder. As in `main.py`, checkpoints are selected on periodic validation, while final metrics describe
the final weights on the full validation set. Training-resume checkpoints are not implemented.

`--no-compile` enables eager execution for debugging. Native Windows does not supply NCCL, so this
trainer's GPU mode requires Linux. `--device cpu --no-compile` uses Gloo for distributed smoke tests;
it is not a substitute for measuring the CUDA path. The DDP tests use two and four CPU processes and
temporary synthetic datasets.

### Stockfish's moves and eval as the targets

Every position of a training set carries Stockfish's best moves (4 in every set so far) and the score after each. The
policy learns a distribution over those moves: a softmax over the expected score after each (0 lost to 1 won), divided by
`--target-temperature` (0.02: a move that gives up 2% of the expected score gets 1/e of the best move's share).
Validation loss and accuracy are against Stockfish's moves. The eval head learns the score of the position itself (the
score after Stockfish's best move) as an expected score.

```
uv run python -m tools.evaluate_checkpoint artifacts/runs/<run>/model.pt --training-set full_sf8_30M    a checkpoint on another set's val positions
uv run python -m tools.label_with_stockfish     labels a copy of the first games of a set, about 4 hours for 10 million positions. Can be stopped and continued
```

## The current recipe

What the September 2026 experiments settled on for the next large model. The evidence is in
`artifacts/overnight_2026-09-23.md`: d128 models at 3x the standard compute, 2 seeds each, every one scored with
4000 games against Stockfish.

Some of it is now simply how the model works, and the code for the alternatives is gone (it is at tag
`pre-cleanup-2026-09-23`):
- It sees only the current board: earlier boards (history) did not help.
- It has no future head: it did not help, and cost puzzle Elo in short runs.
- Its value head is the eval head, Stockfish's eval as an expected score: neutral to +30-35 puzzle Elo against the game's
  result.
- Its targets are Stockfish's moves at `--target-temperature 0.02`: 0.01 tied it, 0.05 was worse.
- 8 position templates, which every recent experiment used, SwiGLU MLPs and no dropout.

| Choice | Setting | Why |
|---|---|---|
| Input embedding | Pending: a comparison between the small Leela embedding and the template-map embedding (both below) is running now. `model.py` has neither; their code, and the head-to-head trainer that compared them, are only at tag `pre-cleanup-2026-09-23` | At 3x compute the small Leela embedding had the best accuracy (+0.7 points over the plain model) and +13 Elo |
| Attention | Narrower queries and keys in the early layers, `--qk-ratio 0.25 0.5 0.5 0.75` at 4 layers or `0.25 0.25` + `0.5` x5 + `0.75` x13 at 20, and half as many key/value heads as heads, `--kv-heads` (grouped-query attention) | The same quality at 17-20% fewer FLOPs |
| Learning rate | `--lr 6e-3` at d128 | 1e-2 was slightly worse at the longer budget. Untested at d640 |

The two embeddings:
- The small Leela embedding for the 64 squares: the whole board's pieces through one dense layer to 16 features per
  square (Leela's own size is d_model / 2), then Leela's gates, norms and small FFN (1.5x d_model), with `model.py`'s
  blocks after it and templates in every layer. Every square sees the whole board before the first block. About 2% of
  the FLOPs at d640. The dense size of 16 was only tested at d128.
- The template-map embedding, a cheaper board-wide step before the first block: `model.py`'s embedding plus the square
  features spread over the board by 8 learned 64x64 maps, the templates' way of mixing. It is what the 2026-09-23 overnight
  run used. At 3x compute it was 0.5 accuracy points and 20 Elo behind the small Leela embedding; at the standard budget the
  two tied within seed noise.

Tried and left out:
- A whole-board term in every block's template mix (the board mix), and Leela's smolgen: little gain once the embedding
  is there. The template-map embedding plus the board mix tied the small Leela embedding in Elo (+14) but not in
  accuracy.
- Cut-down Leela embeddings: its dense layer alone, or the dense layer plus an FFN. Close behind at d128, and the FFN added
  nothing for its cost.

Still open: the full-size Leela embedding (a dense layer of d_model / 2 per square) averaged slightly better than 16 at d128
and equal compute (-0.006 loss, +0.25 accuracy points, the same Elo), though within seed noise: on one of the two seeds 16
was as good. It was never tried at 3x or with the narrower queries and keys. At d640 it would add 17M parameters (30%) for
only 0.7% more FLOPs than 16.

At d640 every one of these embeddings costs under 3% of the model's FLOPs (2-16% at d128), so the choice is on quality.

At scale, train synchronously (`multi_gpu/train_ddp.py`) at a global batch of about 4096, as the d640 run did. The measured gradient
noise scale of a d640 model is 2.5k-7k positions while the learning rate is high, so a much bigger batch mostly wastes positions. Averaging the weights of
150 preemptible GPUs (`train_hyak.py`, now only at the tag) ended about 100 Elo below that run on 5.9x its compute. Stay within about 4 passes
over the data: `full_sf8_20260922` has 290M positions, so about 1.2B per run, and a longer run needs more labelled positions.

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
| `data.py` | The Stockfish-labelled training sets as datasets, and their loaders |
| `model.py` | The transformer: board tokens in, a score for each move and an eval out |
| `main.py` | The single-GPU training loop |

Everything else is kept out of the way:

| Folder | What is in it |
|---|---|
| `multi_gpu/` | `train_ddp.py`, the separate multi-GPU DDP training loop. Run as a module from the root under `torchrun` |
| `tools/` | Checkpoint saving and loading, the training budget, the run table and results page, the TensorBoard launcher, labelling a training set with Stockfish, measuring a checkpoint (`evaluate_checkpoint`, `puzzles`), plotting its templates, and `play/` (choosing the model's move, the UCI protocol, matches against Stockfish). Run as modules from the root: `python -m tools.<name>` |
| `artifacts/` | No code: the PGNs, preprocessed training sets, checkpoints, TensorBoard runs and chess engines. Of the PGNs only `pgn_data/Carlsen.pgn` is in git: the openings of every match, and the games the tests replay |
| `tests/` | `uv run pytest` |
