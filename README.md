# Chess Transformer

A transformer that plays chess. It is trained on Stockfish's moves and evals, plays on its own (the policy) or with
Monte Carlo tree search, and can be improved further by self-play (expert iteration).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Rust](https://rustup.rs), then:

```
git clone https://github.com/GilliamWong/chess-transformer.git
cd chess-transformer
uv sync
```

This installs Python, every dependency (PyTorch's CUDA 13.0 build) and compiles self-play's Rust module
(`selfplay_rs/`, rebuilt whenever its source changes). A machine that only trains can skip Rust with
`uv sync --no-group selfplay`. Training needs an NVIDIA GPU with a recent driver.

## Train

```
uv run main.py                              4 minutes of a d128 with 6 layers: the standard comparison
uv run main.py --note "final layer norm"    the same, with a note for the results page
uv run main.py --minutes 30                 a longer run
uv run main.py --positions 10_000_000       a budget in positions instead (about 3 minutes per 10M on an RTX 5070 Ti)
uv run main.py --d-model 256 --layers 8 --heads 8 --lr 3e-3    see --help for every flag
uv run python -m tools.compare_runs         every finished run, grouped by budget, best first
```

Training starts TensorBoard at localhost:6006 (`--no-tensorboard` skips it). Runs with the same budget are comparable:
`--positions` for equal data, `--minutes` for equal GPU time (compiling and evaluating are not counted). Each run gets
`artifacts/runs/<name>/` with its TensorBoard logs, `settings.json`, `model.pt` (best validation loss) and `final.pt`.
Tools play and measure the run that finished last unless given a checkpoint. `artifacts/progress.html` lists every
finished run.

Older runs are in `artifacts/runs_archive/` (`uv run tensorboard --logdir artifacts/runs_archive --port 6007`).
Checkpoints from before the 2026-09-23 cleanup load only from a checkout of tag `pre-cleanup-2026-09-23`, and the two
that used template maps from commit `634f2e3`; loading one here says so.

A training set is `artifacts/processed_data/<name>_{train,val}_boards.bin` plus Stockfish's labels of every position,
`<name>_{train,val}_stockfish_{moves,evals}.bin`. The default, `full_sf8_30M_mix`, is 30M positions of whole games from
`full_sf8_20260922` (Lichess Elite 68%, Fishtest 29%, over-the-board 3%). Processed sets are too big for git; the scripts
that made them are in `artifacts/processed_data/scripts_used/`.

The policy learns a softmax over Stockfish's top moves by the expected score after each, at `--target-temperature`
(0.02: a move that gives up 2% of the expected score gets 1/e of the best move's share). The eval head learns the
position's score as a distribution over `--value-bins` equal bins of the expected score (HL-Gauss: the target is a
narrow gaussian around the score), read out as its mean; `--value-bins 1` is the older single logit.

```
uv run python -m tools.label_with_stockfish        label the first games of a set (stoppable, carries on)
uv run python -m tools.label_blunders label <boards.bin> <folder>    blunder labels, then `merge` into the set
uv run python -m tools.tablebase                    download the 3-5 piece Syzygy tables (about 1 GB)
uv run python -m tools.label_with_tablebase         a copy of a set with tablebase labels for positions of <= 5 pieces
uv run python -m tools.evaluate_checkpoint <model.pt> --training-set <set>    a checkpoint on another set's val
```

### Multiple GPUs

`multi_gpu/train_ddp.py` is a Distributed Data Parallel version of `main.py` for Linux (NCCL), one process per GPU:

```
uv run torchrun --standalone --nproc-per-node=2 -m multi_gpu.train_ddp --minutes 30 --no-tensorboard
uv run torchrun --standalone --nproc-per-node=4 -m multi_gpu.train_ddp --positions 360_000_000 --name big --resume
```

- `--batch-size` and `--positions` are global, across all GPUs; the batch must divide evenly between them.
- `--minutes` follows GPU 0's clock, shared with the others so every process stops and schedules alike.
- Validation is split across the GPUs; only process 0 writes logs, checkpoints and the results page.
- `--resume` (with `--name` and `--positions`) saves the full training state every `--checkpoint-minutes` and carries
  on from it when the job is started again, for preemptible jobs. `--init-from` takes only the weights of a checkpoint.
- `--device cpu --no-compile` runs on Gloo, for the tests.

## The current recipe

What the September 2026 experiments settled on (evidence in `artifacts/overnight_2026-09-23.md`):

- The model sees only the current board; its value head learns Stockfish's eval; 8 position templates, SwiGLU, no
  dropout, no board-wide input embedding (template maps only helped shallow models).
- The eval head predicts a distribution over 128 bins of the expected score at `--value-weight 0.5`: value error
  0.044 against 0.054 for a single logit on a d128, the policy unchanged (at weight 1 it lost 0.02 val loss), and
  about +40 Elo in both a policy and a 64-simulation match, each inside its 200-game margin.
- Muon for the weight matrices inside the blocks, AdamW for the rest, one learning rate: d128 val loss 1.9414 against
  1.9929 with AdamW alone.
- The blunder penalty, `--blunder-weight 4`: +27 Elo on a d128. It needs blunder labels next to the set
  (`<set>_{split}_worst_{moves,losses}.bin`); a set without them trains as if the weight were 0.
- Narrower queries and keys in the early layers (`--qk-ratio 0.25 0.5 0.5 0.75` at 4 layers, `0.25 0.25` + `0.5` x5 +
  `0.75` x13 at 20) and half as many key/value heads (`--kv-heads`): the same quality at 17-20% fewer FLOPs.
- d128 with 6 layers as the quick model (`--lr 6e-3`). Looping the same layers (`--loops`) lost to real depth.
- At scale, train with `train_ddp.py` at a global batch of about 4096 (the gradient noise scale of a d640 is 2.5k-7k
  positions) and stay within about 4 passes over the data (`full_sf8_20260922` has 290M positions).

## Play against it and measure its strength

Windows only. Fetch Stockfish, Cute Chess and Leela once, then run a match:

```
uv run python -m tools.play.download_engines
uv run python -m tools.play.evaluate_elo                     100 games against Stockfish at 1320 Elo
uv run python -m tools.play.evaluate_elo --elo 1700 --games 200 --concurrency 12
uv run python -m tools.play.evaluate_elo --opponent random   a sanity check
uv run python -m tools.play.evaluate_elo --opponent human    play it yourself in Cute Chess
uv run python -m tools.play.evaluate_elo --select mcts --simulations 256   with the search, no clock
uv run python -m tools.play.analyze_match                    judge the last match's moves with Stockfish
```

The policy plays `avoid-repetition` by default (argmax, but no draw by repetition when clearly ahead). A search match
against Stockfish runs every game in one process (`tools/play/search_match.py`): the searches in Rust, their leaves
through the network together on the GPU; other matches run under cutechess-cli with `tools/play/uci.py` as the engine.
Games are saved to `artifacts/match.pgn` (with the search's scores as `[%eval]`), and results against Stockfish are
logged to `artifacts/elo_results.jsonl` and the `elo` chart in TensorBoard.

## Self-play (expert iteration)

```
uv run selfplay.py artifacts/runs/rl_start_d128_4min/model.pt --name selfplay_r1 --positions 1_000_000 --tablebase
uv run main.py --init-checkpoint artifacts/runs/rl_start_d128_4min/model.pt --training-set selfplay_r1 --mix-set full_sf8_30M_mix --positions 3_000_000 --name rl_r1
uv run python -m tools.play.evaluate_elo --checkpoint artifacts/runs/rl_r1/model.pt --elo 1700 --games 200 --concurrency 12
```

`selfplay.py` plays the model against itself with 64 simulations a move (root noise, the first 30 plies sampled from
the visits, tree reuse) and saves every position with the search's visit shares, the game's result and the root's
score. Worker processes play the games in Rust (`selfplay_rs/`); one inference process runs the network for all of them
on the GPU. `+` and `-` add or remove a worker while it runs; ctrl+c finishes the games being played, a second ctrl+c
stops at once, and running it again with the same flags carries on.

The Rust search and rules must match `mcts.py` and the Python reference games in `tests/reference_selfplay.py`
exactly; `tests/test_selfplay_rs.py` plays both and compares every number, so a change to the search goes in both.

`main.py --init-checkpoint` fine-tunes at a peak learning rate of 1e-3. `--result-weight` blends the result and the
root's score into the value target, `--visit-temperature` sharpens the visit shares, and `--mix-set` keeps a quarter of
every batch (`--mix-share`) from a Stockfish-labelled set.

## Tests

```
uv run pytest
```

## Layout

| Path | What it is |
|---|---|
| `encoding.py` | Boards to 70 tokens, moves to one of 1,968 ids, and back |
| `data.py` | Training sets as datasets and loaders |
| `model.py` | The transformer: board tokens in, move logits and an eval out |
| `main.py` | The single-GPU training loop |
| `mcts.py` | Monte Carlo tree search over the model |
| `selfplay.py`, `inference.py` | Self-play: the worker processes and the GPU inference process |
| `multi_gpu/` | The DDP trainer |
| `selfplay_rs/` | The search and self-play games in Rust |
| `tools/` | Checkpoints, run comparison, labelling, puzzles, and `play/` (UCI engine, matches, Elo log) |
| `artifacts/` | Data, checkpoints, runs, engines, results and reports; no code. Only small files are in git, among them `pgn_data/Carlsen.pgn` (match openings and test games) |
| `tests/` | `uv run pytest` |
