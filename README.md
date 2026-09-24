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

Self-play's games are compiled from Rust (`selfplay_rs/`), so install [Rust](https://rustup.rs) first: uv builds them
on the first `uv sync` or `uv run`, and again whenever their source changes. A machine that only trains can leave them
out with `uv sync --no-group selfplay`.

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

Older runs are in `artifacts/runs_archive/`, moved out of `artifacts/runs` on 2026-09-23 so that TensorBoard stays quick
(runs kept: the d384 and first d640 overnight runs, `rl_start_d128_4min`, and the `elo` and `selfplay` charts).
`uv run tensorboard --logdir artifacts/runs_archive --port 6007` shows them, and their checkpoints load from there.

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
uv run python -m tools.label_with_stockfish --source selfplay_r1_reuse --name selfplay_r1_reuse_sf    the same for the positions of the model's own games (a self-play set), each position once
uv run python -m tools.tablebase                downloads the Syzygy endgame tables for up to 5 pieces, about 1 GB
uv run python -m tools.label_with_tablebase     a copy of a set whose positions of up to 5 pieces carry the tables' moves and results (--measure: only how far Stockfish's were off)
```

## The current recipe

What the September 2026 experiments settled on for the next large model. The evidence is in
`artifacts/overnight_2026-09-23.md`: d128 models at 3x the standard compute, 2-3 seeds each, every one scored with
4000 games against Stockfish.

Some of it is now simply how the model works, and the code for the alternatives is gone (it is at tag
`pre-cleanup-2026-09-23`):
- It sees only the current board: earlier boards (history) did not help.
- It has no future head: it did not help, and cost puzzle Elo in short runs.
- Its value head is the eval head, Stockfish's eval as an expected score: neutral to +30-35 puzzle Elo against the game's
  result.
- Its targets are Stockfish's moves at `--target-temperature 0.02`: 0.01 tied it, 0.05 was worse.
- 8 position templates, which every recent experiment used, SwiGLU MLPs and no dropout.
- No board-wide input embedding: the template maps (below) only helped shallow models, and the code came out on
  2026-09-24 (it is at commit `634f2e3`; `tools/checkpoint.py` points there for the two d128 recipe runs that used it).
- Muon trains the weight matrices inside the blocks and AdamW everything else, at one learning rate
  (`main.make_optimizers`): on d128 with 4 layers for 4 minutes, from the same start on the same data, val loss 1.9414
  against 1.9929 with AdamW for everything, with 8% fewer positions (its steps are slower); about what twice the training
  time gave. It reached AdamW's final loss 60% of the way through its run. Runs before 2026-09-24 used AdamW alone.

| Choice | Setting | Why |
|---|---|---|
| Input embedding | None, model.py's token embedding: a deep model learns the board's geometry itself | 192 template maps against no embedding: +0.039 loss and about +34 Elo at d128/4 layers, +0.015 and +15 at d256/4 layers, +0.003 and +3 (noise) at d256/8 layers |
| Attention | Narrower queries and keys in the early layers, `--qk-ratio 0.25 0.5 0.5 0.75` at 4 layers or `0.25 0.25` + `0.5` x5 + `0.75` x13 at 20, and half as many key/value heads as heads, `--kv-heads` (grouped-query attention) | The same quality at 17-20% fewer FLOPs |
| Learning rate | `--lr 6e-3` at d128 | 1e-2 was slightly worse at the longer budget. Untested at d640 |

The input embeddings compared (2026-09-23, in `artifacts/overnight_2026-09-23.md`): d128 with 4 layers, the narrower
queries and keys and grouped-query attention, 3x the standard compute, 1-3 seeds, fp16, every model scored with 4000 games
against Stockfish 1700:

| Board summary before the first block | Seeds | Parameters | Val loss | Elo |
|---|---|---|---|---|
| 192 template maps | 3 | 1.55M | **1.8171** | 1780 |
| 384 template maps | 1 | 2.36M | 1.8180 | **1794** |
| 64 template maps | 3 | 1.01M | 1.8256 | 1772 |
| Full Leela embedding | 3 | 3.89M | 1.8271 | 1773 |
| Small Leela embedding | 3 | 1.52M | 1.8337 | 1763 |
| 8 template maps (the overnight run's embedding) | 3 | 0.71M | 1.8458 | 1748 |

- `model.TemplateEmbedding` (removed 2026-09-24, at commit `634f2e3`) was the tested embedding exactly (checked output for
  output against the experiment code). The tested d128 models also had a narrower MLP than `model.py`'s default:
  `mlp_ratio=1.333` reproduces them to the parameter.
- Template maps here are learned 64x64 maps, shared by every piece and weighted by it, spreading the board's pieces to
  every square; one feature per map, then Leela's processing (a linear layer, mish, a norm, per-square gates and a small
  FFN). Leela's embedding does the same with a dense layer, which amounts to a separate map for every piece and feature.
  Its weights learn each piece's moves (pictures in `artifacts/embedding_plots/`).
- More maps help up to about 192-384. 768 cost 17% more FLOPs a position at d128 and do worse.
- In a deeper model none of it matters: at d256 with 8 layers, no embedding, both Leela embeddings and 192 maps all
  landed within 0.005 loss and 6 Elo (one seed).

The best template setting by model size, each against no board-wide embedding at equal compute (Elo against Stockfish 1700
at 0.1 s, 4000 games, about +-12):

| Model | Best `template_maps` | Val loss and Elo | Seeds | How it was trained |
|---|---|---|---|---|
| d128, 4 layers, 4 heads | **192** (384 as good: 1.8180, 1794 on one seed; 64 or fewer, and 768, are worse) | 1.8171 and 1780, against 1.8561 and 1746 without | 3 | `--qk-ratio 0.25 0.5 0.5 0.75 --kv-heads 2 --mlp-ratio 1.333 --lr 1e-2`, 6.6 PF (about 80M positions), fp16 |
| d256, 4 layers, 8 heads | **192** (the only count tried) | 1.7842 and 1824, against 1.7988 and 1809 without | 1 | `--qk-ratio 0.25 0.5 0.5 0.75 --kv-heads 4 --mlp-ratio 1.333 --lr 5e-3`, 13.2 PF (about 45M positions), fp16 |
| d256, 8 layers, 8 heads | **0**: 192 maps 1.7297 and 1904, no embedding 1.7328 and 1901, the Leela embeddings in between | noise | 1 | `--qk-ratio 0.25 0.25 0.5 0.5 0.75 0.75 0.75 0.75 --kv-heads 4 --mlp-ratio 1.333 --lr 5e-3`, 26.5 PF (about 45M positions), fp16 |
| d640, 20 layers | **0** expected: untested, extrapolated from the depth trend above | - | - | - |

The d128 numbers without an embedding are the earlier control runs (2 seeds), which also differ in precision (bf16) and
attention (full-width queries and keys, no grouped-query attention); every other number is fp16. Every run is in
`artifacts/leela_head_to_head.jsonl` and `artifacts/overnight_2026-09-23.md`.

Tried and left out:
- A whole-board term in every block's template mix (the board mix), and Leela's smolgen: little gain once the embedding
  is there. The template-map embedding plus the board mix tied the small Leela embedding in Elo (+14) but not in
  accuracy.
- Cut-down Leela embeddings: its dense layer alone, or the dense layer plus an FFN. Close behind at d128, and the FFN added
  nothing for its cost.

A 20-layer d640 should gain nothing measurable from any of them. For a shallow model, 192 template maps cost about 1.3M
parameters and 0.5% of the FLOPs at d640, against 15.7M parameters for the full Leela embedding.

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
uv run python -m tools.play.evaluate_elo --select mcts --simulations 256   with mcts.py's search and no clock
```

A match of the search against Stockfish (`tools/play/search_match.py`) plays every game in one process: the searches
run in Rust (the same search as `mcts.py`, visit for visit, like self-play's), and the leaves of all the games go through
the network together on the GPU, 32 games at a time by default. It replays the ladder's 64 simulation rung (83.2% against
the 81.8% of `artifacts/mcts_ladder_2026-09-23.md`) in 90 seconds, and plays 200 games at 800 simulations in about
6 minutes. At most 12 Stockfish think at once, each on a core of its own as in the ladder's matches. Its openings are drawn from the same Carlsen games, though not the same ones cutechess-cli draws. Every other
match runs under cutechess-cli with `tools/play/uci.py` as our engine.

Matches run at below-normal priority, so the computer stays usable. After a match with the search, every move of the
model in `artifacts/match.pgn` carries its score as `[%eval]`, the line it expected and its most searched moves: import
the PGN into a Lichess study to replay the games with an evaluation graph. In the Cute Chess GUI, the engine
"ChessTransformer MCTS" shows the same live while it plays (`python -m tools.play.download_engines` registers it).

The games of the last match are saved to `artifacts/match.pgn`, and the result against Stockfish is added to
`artifacts/elo_results.jsonl` and the `elo` chart in TensorBoard.

## Self-play (expert iteration)

The search is far stronger than the policy alone (`artifacts/mcts_ladder_2026-09-23.md`: +390 Elo at 64 simulations).
A round of expert iteration trains the policy towards the search's own choices:

```
uv run selfplay.py artifacts/runs/rl_start_d128_4min/model.pt --name selfplay_r1 --positions 1_000_000 --tablebase
uv run main.py --init-checkpoint artifacts/runs/rl_start_d128_4min/model.pt --training-set selfplay_r1 --mix-set full_sf8_30M_mix --positions 3_000_000 --name rl_r1
uv run python -m tools.play.evaluate_elo --checkpoint artifacts/runs/rl_r1/model.pt --elo 1700 --games 200 --concurrency 12
```

`selfplay.py` plays the model against itself with 64 simulations a move: noise in every root's priors and the first 30
plies drawn from the visit counts, so that games differ; a game ends by the rules, at 5 pieces with the tablebase's
result, or as a draw after 600 plies. Every position is saved with the search's visit shares (the policy target) and
with the game's result and the root's mean score, which training blends half and half (`--result-weight`). The move
played keeps its subtree for the next search (tree reuse: the same target quality, measured against an 800 simulation
search, for 38 new simulations a move instead of 64; `--no-tree-reuse` turns it off).

Its worker processes play 256 games each on the CPU, in Rust (`selfplay_rs/`): the search of `mcts.py` and the rules of
`selfplay.py`, move for move (`tests/test_selfplay_rs.py` plays games both ways and compares every number), about 65
times faster than the Python. A change to the search or to the rules has to be made in `selfplay_rs/src` as well, and
that test says when the two have drifted apart. One inference process (`inference.py`) runs the network for all the
workers on the GPU: every round it takes the boards of all the waiting workers in one batch, through CUDA graphs of a
compiled model in bf16 (`--no-compile`, `--precision tf32`). With the d128 and the default 2 workers the GPU is the limit:
about 12.5 million positions an hour over a whole run (1 million in under 5 minutes), with 10% of the CPU busy and 75 MB
of memory a worker. `+` and `-` add and remove a worker while it runs (up to 8), and everything runs at below-normal
priority. When it reaches
`--positions` it starts no new games and finishes the ones being played, so every game begun is saved; ctrl+c does the
same, and a second ctrl+c stops at once. Run again with the same flags (or a larger `--positions`), it carries on where it
stopped.

`main.py --init-checkpoint` fine-tunes the checkpoint, at a peak learning rate of 1e-3 unless `--lr` says otherwise (the
6e-3 of a run from scratch undid a good part of what the starting model knew during the warmup). `--visit-temperature`
sharpens the visit shares the policy learns (1 as recorded, 0 the most visited move alone): at 64 simulations the recorded
shares are broad, and on average a worse move than the policy's own (`VISIT_TEMPERATURE` in `data.py`). `--mix-set` keeps a
quarter of every batch (`--mix-share`) from a Stockfish-labelled set so that the model does not forget it, and the run
ends with the model's accuracy on that set's val positions next to the self-play ones, and the commands that measure it. The test of a round is the policy alone against the same Stockfish as before: the
starting model scored 1569.

TensorBoard (`uv run tensorboard --logdir artifacts/runs`) shows, next to the training runs, `selfplay/<name>` (positions
an hour, processes, game length and results as a run goes) and `elo`: every match `evaluate_elo` plays against Stockfish
is logged to `artifacts/elo_results.jsonl` and charted per lineage of models, the round of self play along the bottom
(round 0 is the model trained from scratch, each `--init-checkpoint` one round more) and a line per way of playing (the
policy alone, or the search at a number of simulations).

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
| `mcts.py` | Monte Carlo tree search over the model, `--tablebase` to score endgames from the tables |
| `selfplay.py` | The model playing itself with the search, saved as a training set for expert iteration |
| `inference.py` | Self-play's inference process: the network on the GPU for every self-play worker at once |

Everything else is kept out of the way:

| Folder | What is in it |
|---|---|
| `multi_gpu/` | `train_ddp.py`, the separate multi-GPU DDP training loop. Run as a module from the root under `torchrun` |
| `selfplay_rs/` | Self-play's games in Rust, which uv builds into the environment as the `selfplay_rs` module |
| `tools/` | Checkpoint saving and loading, the training budget, the run table and results page, the TensorBoard launcher, labelling a training set with Stockfish, measuring a checkpoint (`evaluate_checkpoint`, `puzzles`), plotting its templates, and `play/` (choosing the model's move, the UCI protocol, matches against Stockfish). Run as modules from the root: `python -m tools.<name>` |
| `artifacts/` | No code: the PGNs, preprocessed training sets, checkpoints, TensorBoard runs and chess engines. Of the PGNs only `pgn_data/Carlsen.pgn` is in git: the openings of every match, and the games the tests replay |
| `tests/` | `uv run pytest` |
