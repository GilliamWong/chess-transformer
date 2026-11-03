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
(`engine/selfplay_rs/`, rebuilt whenever its source changes). A machine that only trains can skip Rust with
`uv sync --no-group selfplay`. Training needs an NVIDIA GPU with a recent driver.

## Train

Every script runs as a module from the repository root (`uv run python -m <folder>.<script>`), so the folders import each other.

```
uv run python -m training.train                              4 minutes of a d128 with 6 layers: the standard comparison
uv run python -m training.train --note "final layer norm"    the same, with a note for the results page
uv run python -m training.train --minutes 30                 a longer run
uv run python -m training.train --positions 10_000_000       a budget in positions instead (about 3 minutes per 10M on an RTX 5070 Ti)
uv run python -m training.train --d-model 256 --layers 8 --heads 8 --lr 3e-3    see --help for every flag
uv run python -m training.compare_runs                          every finished run, grouped by budget, best first
```

Training starts TensorBoard at localhost:6006 (`--no-tensorboard` skips it). Runs with the same budget are comparable:
`--positions` for equal data, `--minutes` for equal GPU time (compiling and evaluating are not counted). Each run gets
`artifacts/runs/<name>/` with its TensorBoard logs, `settings.json`, `model.pt` (best validation loss) and `final.pt`.
Tools play and measure the run that finished last unless given a checkpoint. `artifacts/progress.html` lists every
finished run.

Older runs are in `artifacts/runs_archive/` (`uv run tensorboard --logdir artifacts/runs_archive --port 6007`).
Checkpoints from before the 2026-09-23 cleanup load only from a checkout of tag `pre-cleanup-2026-09-23`, and the two
that used template maps from commit `1a976d9`; loading one here says so.

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
uv run python -m engine.tablebase                    download the 3-5 piece Syzygy tables (about 1 GB)
uv run python -m tools.label_with_tablebase         a copy of a set with tablebase labels for positions of <= 5 pieces
uv run python -m tools.evaluate_checkpoint <model.pt> --training-set <set>    a checkpoint on another set's val
```

### Multiple GPUs

`training/train_ddp.py` is a Distributed Data Parallel version of `training/train.py` for Linux (NCCL), one process per GPU:

```
uv run torchrun --standalone --nproc-per-node=2 -m training.train_ddp --minutes 30 --no-tensorboard
uv run torchrun --standalone --nproc-per-node=4 -m training.train_ddp --positions 360_000_000 --name big --resume
```

- `--batch-size` and `--positions` are global, across all GPUs; the batch must divide evenly between them.
- `--minutes` follows GPU 0's clock, shared with the others so every process stops and schedules alike.
- Validation is split across the GPUs; only process 0 writes logs, checkpoints and the results page.
- `--resume` (with `--name` and `--positions`) saves the full training state every `--checkpoint-minutes` and carries
  on from it when the job is started again, for preemptible jobs. `--init-from` takes only the weights of a checkpoint.
- `--device cpu --no-compile` runs on Gloo, for the tests.

## The current recipe

What the September 2026 experiments settled on (evidence in `artifacts/reports/overnight_2026-09-23.md`):

- The model sees only the current board; its value head learns Stockfish's eval; 8 position templates, SwiGLU, no
  dropout, no board-wide input embedding (template maps only helped shallow models).
- The eval head predicts a distribution over 128 bins of the expected score at `--value-weight 0.5`: value error
  0.044 against 0.054 for a single logit on a d128, the policy unchanged (at weight 1 it lost 0.02 val loss), and
  about +40 Elo in both a policy and a 64-simulation match, each inside its 200-game margin.
- Muon for the weight matrices inside the blocks, AdamW for the rest, one learning rate: d128 val loss 1.9414 against
  1.9929 with AdamW alone. Learning rates of 4e-3 and 1e-2 and Muon momentum 0.9 all lost to 6e-3 and 0.95 (d128 L6,
  10M positions, 2026-09-25).
- Weight decay sized to the run (`--weight-decay`, both optimizers): 1/(lr x wd) should be 15-20% of the steps, so
  wd ≈ 1 / (0.17 x lr x steps). The default 0.1 won the short d128 sweep (0.007 better val loss than 0.01 on two
  seeds) but stalled an 850k-step d512, which wants about 0.002 (`artifacts/attention_lookups_20260927/`).
- Queries and keys are RMS-normalised per head (QK-norm), so block 1's scores can no longer run past fp16's range by the projections
  growing. Its gains alone set how sharp attention can be, so they are not weight-decayed: at 0.1 they fell to 0.3 and
  the heads went nearly uniform.
- Each square also gets 23 learned feature vectors, summed into its embedding (`SquareFeatures`), for what heads
  otherwise looked up: its offset from each king and queen, its 8 neighbours, the first piece along each of 8 lines
  and how far, the knights attacking it, and the nearest pawns up and down its file (`artifacts/attention_lookups_20260927/`).
- On the 6-minute d128 bed (10M positions, full val), each row adding to the last: old model 1.8840, + a learned
  attention sink 1.8812, + QK-norm 1.8781, + square features 1.8678 (45.8% against 45.3%). The features' lead shrank
  from 0.06 early to 0.010 at the end, a fixed head start of about 1,000 steps plus a lasting gain. With 5 layers the
  features gain 0.022 (1.8752 against 1.8973), matching the 6-layer model without them. Against Stockfish 1700 (1,000
  games) +9 ± 28 Elo at 6 layers. Cost per step: QK-norm +0.6% at d128, +3.4% at d512; the features +5% at d128,
  nothing measurable at d512. The sink, in the model from f47b28e, was taken out again: no gain, and it did not stop
  block 1's scores running away.
- Gated attention (Qiu et al. 2025, a sigmoid of the query's state on each head's output) was tried on the same bed and
  left out: one gate per head 1.8686 (nothing), one per channel 1.8509 but with 9% more block parameters, 1.8591 with
  the MLP narrowed to match them (-0.009, one seed). The gates stayed near 0.46 rather than going sparse as in the
  paper, and at d512 the per-channel gate would cost about 15% of a block's matrix work.
- Checkpoints from before the sink load from commit d969ada, with the sink but before QK-norm from ee34470, with the
  sink and QK-norm but before the square features from d1803d0 (which also soft-caps the scores).
- The blunder penalty, `--blunder-weight 4`: +27 Elo on a d128. It needs blunder labels next to the set
  (`<set>_{split}_worst_{moves,losses}.bin`); a set without them trains as if the weight were 0. A move counts as a
  blunder from 0.3 expected score lost (`--blunder-threshold`): thresholds 0.1, 0.2 and 0.3 played the same Elo
  (4000 games each against Stockfish 1700, two seeds) and 0.3 leaves the policy 0.033 val loss better than 0.2.
- Narrower queries and keys in the early layers (`--qk-ratio 0.25 0.5 0.5 0.75` at 4 layers, `0.25 0.25` + `0.5` x5 +
  `0.75` x13 at 20) and half as many key/value heads (`--kv-heads`): the same quality at 17-20% fewer FLOPs.
- d128 with 6 layers as the quick model (`--lr 6e-3`). Looping the same layers (`--loops`) lost to real depth.
- An MLP in every block. Fewer MLPs (tested 2026-10-01; the code is in `artifacts/patches/experiments_20261001.patch`) is a worse trade than fewer layers: on the
  d128 bed dense L5 beat MLPs in every 2nd block by 0.019 at nearly the same engine speed. In the trained d512 the
  first MLP (piece-on-square and attack facts) and the last (the readout, 8 times the residual's size) are essential,
  and the middle ten each cost 0.04-0.09 alone; a model trained without its first MLP mostly learns around it
  (`artifacts/mlp_layers_20261001/`).
- At scale, train with `train_ddp.py` at a global batch of about 4096 (the gradient noise scale of a d640 is 2.5k-7k
  positions) and stay within about 4 passes over the data (`full_sf8_20260922` has 290M positions).

## Where it stands

September 2026, details in `artifacts/reports/selfplay_and_leela_2026-09-26.md` and `artifacts/scaling_grid_20260925/`:

- The d512 with this recipe (22M parameters, 360M positions, about 3e18 training FLOPs) ties the d640 L20 against
  Stockfish (65.6% against UCI_Elo 2500) at about a quarter of its compute. Without search it loses to Leela's BT4 by
  383 Elo and to T1-512 by 189.
- Its policy is capped by its labels: judged by deeper Stockfish, depth 8's move is only slightly better than ours
  and worse than BT4's policy, while depth 10 and 12 labels are better than BT4's.
- Self-play at 800 simulations gives targets about 40% better than the depth-8 labels, but fine-tuning the d512 on
  19.8M such positions gained only +20 head-to-head and nothing against Stockfish: the net can fit the search's moves
  on positions it trained on, yet only about a tenth of those corrections transfer to new positions, with 500k or 19M
  positions alike.
- At equal training compute the bigger model won every comparison of the scaling grid, around 5-15 positions per
  parameter.
- With search, against lc0 with BT4 at a fixed number of nodes a move: -418 Elo at equal nodes, -255 with 5 times
  Leela's nodes (about her compute per move), -79 with 20 times. The engine search below does ~27k nodes/s on the
  5070 Ti against lc0's ~6k with BT4; a timed match has not been played yet.
- Depth-12 labels of every position (`full_sf12_20260926`, `selfplay_d512m_800_sf12` and `full_sf12_30M_mix`, the
  30M set's rows) were made on Hyak on 2026-09-26; they share the depth-8 sets' boards and blunder labels.
- Long runs on them (d512 from scratch and warm-started, and equal-parameter d448 L16 and d384 L22, 1.74B positions
  each) are under way on Hyak. At low weight decay block 1's attention turns into one-hot lookups with scores past
  fp16's limit (NaN in the search engine); a learned sink does not prevent it, and Leela's BT4 has no such heads.
  Across 8 models the heads look up the kings and queens, the neighbouring squares, the first piece along each line
  and the metadata: `artifacts/attention_lookups_20260927/`.

## To try next: distillation

On the 4-minute d128 test, a student trained on the 10-pass d512's move and value distributions (`training/train.py
--teacher-weight 1`, with targets from `tools/teacher_targets.py`) beat Stockfish-only targets by 0.019 val loss and
was the best on every per-move judge measure (2026-10-01). Still to find out:

1. [ ] **Does it last?** The same student for 100M positions, against the dense L6 run of the same length (1.7512).
2. [ ] **How much teacher?** Weights between 0.5 and 1 at 100M positions, and the teacher's probabilities softened or
   sharpened (a temperature) before mixing.
3. [ ] **Bigger students**: d256 and d384 from the d512. Distillation pays most when the teacher is about 2-4 times the
   student (Busbridge et al. 2025).
4. [ ] **Positions labelled by the teacher alone**: about 2.55B positions have no Stockfish labels, and the teacher
   labelled the 30M set in about 35 minutes on the 5070 Ti.
5. [ ] **More of the teacher in the targets**: its whole 128-bin value distribution instead of its expected score, and
   its top 32 moves instead of 16.
6. [ ] **A d512 student**, which needs a bigger teacher first (the d512 L24, or a wider model).

Judge them by val loss against the Stockfish labels, the per-move judge's big-mistake rate and Elo: a distilled
model's loss against the labels can rise while it plays better.

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
through the network together on the GPU; other matches run under cutechess-cli with `engine/uci.py` as the engine.
Since 2026-09-26 the search match's network runs in fp16 (tf32 before, so its numbers differ a little from results
logged earlier) and a cache answers positions reached again; the search is still `engine/mcts.py`'s, 4.5x faster for the
d512 at 32 games.
Games are saved to `artifacts/match.pgn` (with the search's scores as `[%eval]`), and results against Stockfish are
logged to `artifacts/elo_results.jsonl` and the `elo` chart in TensorBoard.

As a UCI engine with a clock (`go movetime`, `nodes`, `wtime`/`btime`), for timed matches against lc0 and others:

```
uv run python -m engine.uci artifacts/runs/<run>/model.pt --select mcts --gpu
```

The search (`engine/engine_search.py`) runs in Rust and gathers up to 64 leaves for each network call, a leaf per 16
simulations done or left as lc0 grows its batches with the tree (`--grow-leaves`; 64 from the first visit, two rounds
out, cost ~0.001 expected score a move at 1000 nodes, +26 Elo against BT4), with the next round gathered while one is
on the GPU once a round is full; the network runs in int8 as CUDA graphs, a cache answers positions
reached again by another order of moves, and the tree is kept for the next move (`ucinewgame` drops both, as lc0
does). Certainty propagation (lc0's sticky endgames) makes mates, draws by the rules and the tables' results proven in
the tree, plays a proven win over a move with more visits, passes over proven losses and stops a search whose move is
proven. The d512 does ~27k simulations a
second on the 5070 Ti in one search, lc0 with BT4 ~6k nodes. Smart pruning stops a search once the simulations left
cannot change its move (at 1 s a move 3 searches in 4 stop early, a quarter of the time saved), and on a clock the
time saved goes to the moves that are not decided. `--leaves 1` is `engine/mcts.py`'s search, without pruning. The first start
after an update compiles the network (~25 s, then ~7 s from torch's cache) and cutechess-cli gives an engine 15 s to
answer `isready`, so start the engine once by hand before a match.

`--gpus 0,1,...` searches on several GPUs at once, a thread driving each. That is the only setting: the rest is chosen
from the GPUs and the CPU. On several GPUs with lean lanes (sm_120: the 5070 Ti, the 5090) the rounds run in Rust with no
Python between them, the same search as the Python loop that one GPU (~1% faster there at 1 s a move), the A100 (sm_80)
and the CPU keep. On several GPUs the threads
descend in the one tree at once rather than in turn under its lock (2.3x the positions evaluated a second in an 8-GPU
emulation; a thread alone is 16% slower that way, so one GPU keeps the lock), each sleeps while its GPU works, and each
is pinned to a physical core of its own: all in one L3 if one has a core for each (a 9950X3D's CCD), else over the L3s
in turn (a 9175F's have one core each), SMT siblings only once every core has a thread. The search on several GPUs is
not the one-GPU search: more leaves are out at once, and it changes from run to run with the threads' timing.

The engine reads its input on a thread of its own, so `stop`, `quit` and `isready` reach a running search, and a
`position` it could not play (an illegal move, a bad FEN) is refused with an `info string`, the last position kept. Its
UCI options are `SyzygyPath` and `SyzygyProbeLimit` (the tables, as `--tablebase` and `--tablebase-pieces`),
`MoveOverhead` (ms, default 20), `CachePositions` (the answer cache, 5M positions and ~1-1.3 GB by default),
`RamLimitMb` (the tree's memory, 0 for no limit: a full tree ends the search, and while a kept tree stays full the next
searches move at once) and `AdvanceThreads` (the threads that carry the kept tree on to the next move once `bestmove`
is sent, default 4); all but `MoveOverhead` are read when the engine loads at `isready`, and need a restart after
that. `RamLimitMb` leaves out the cache and the copy a kept tree makes when the root moves, so set it to about (the
engine's memory - the cache) / 1.7. `go infinite` runs `--simulations`, as cutechess-cli's `tc=inf` games need; with
`--infinite-until-stop` it searches until `stop`, as analysis GUIs expect. `go ponder` searches until `ponderhit`
(GUIs do not ponder with the engine, which names no ponder move), and `searchmoves` is ignored.

The network runs every block but the last with its matmuls on normalised inputs (Q, K, V, the template mix, the mlp's
gate and up) and the mlp's down in int8 (`model/int8_blocks.py`): on the 5070 Ti it answers ~1.35x fp16's positions a
second at 64 boards and the search does ~1.4x the nodes at 1 s a move (1.33x at 800 nodes), and an A100, A40 and L40S
answer 1.09-1.3x fp16's positions. Its priors are further from fp32's (policy total variation 0.0030, fp16's 0.0011),
but at equal nodes it plays as fp16 does: -0.2 [-8.4, +8.0] Elo over 3,663 games on Hyak. One block more stays in fp16
by default: the one whose int8 alone moves the policy furthest from the fp16 model's on the calibration positions
(`engine/quantization.py`). A single block can cost as much as all the others: bt4prior_final's block 26 alone took its
total variation from fp16's 0.0012 to 0.0054 (all blocks 0.0061), and with it in fp16 the network is at 0.0030, for ~1%
of the network's speed (0.8% of the nodes a second at 1000 nodes).
`--fp16-blocks N` keeps N blocks so (0 for every other block in int8), `--precision fp16` runs the network in fp16. Self-play and labels stay in fp16: int8's answers depend on the rest of the batch. The int8 weights are
rounded once per checkpoint (GPTQ, ~1 min) and kept in `~/.cache/chess_transformer/int8`;
`python -m engine.quantization <model.pt>` rounds them ahead of a match, which `isready` would not wait for.

On a tournament box (TCEC's: 2x EPYC 9175F, 768 GB, 8x RTX 5090), run
`engine.uci <model.pt> --select mcts --gpu --gpus 0,1,2,3,4,5,6,7` (the Rust rounds, one tree for all eight, the sleeping
waits and the pins follow from it) with:

```
setoption name RamLimitMb value 350000        the tree's memory, ~1.7x at its peak with the kept tree's copy (~610 GB with the cache); 200000 on a shared box
setoption name CachePositions value 50000000  ~9.5 GB, more than a move's answers (~0.25-0.5 a node): one 10x smaller halved the speed, a bigger one gains nothing
setoption name AdvanceThreads value 16        the cores are idle while the opponent thinks, and a reply that comes before the kept tree is ready waits for it
setoption name MoveOverhead value 200         lc0's default: 0.2 s a move is nothing at TCEC's time controls, a loss on time is the game
setoption name SyzygyPath value <folders>     the box's syzygy tables, ':' between folders, with SyzygyProbeLimit 6 (the most the engine reads)
```
On the box itself before the first game, round the int8 weights (`python -m engine.quantization <model.pt>`, ~1 min) and start
the engine once to `isready` (the graphs compile, ~25 s): both are cached, and the GUI's own `isready` then takes ~7 s on
one GPU, an estimated ~35-40 s on eight (each GPU's network compiles in turn), so give the GUI a longer start timeout.

## Self-play (expert iteration)

```
uv run python -m engine.selfplay artifacts/runs/rl_start_d128_4min/model.pt --name selfplay_r1 --positions 1_000_000 --tablebase
uv run python -m training.train  --init-checkpoint artifacts/runs/rl_start_d128_4min/model.pt --training-set selfplay_r1 --mix-set full_sf8_30M_mix --positions 3_000_000 --name rl_r1
uv run python -m tools.play.evaluate_elo --checkpoint artifacts/runs/rl_r1/model.pt --elo 1700 --games 200 --concurrency 12
```

`engine/selfplay.py` plays the model against itself with 64 simulations a move (root noise, the first 30 plies sampled from
the visits, tree reuse) and saves every position with the search's visit shares, the game's result and the root's
score. Worker processes play the games in Rust (`engine/selfplay_rs/`); one inference process runs the network for all of them
on the GPU, in fp16 since 2026-09-26 (bf16 before: `--precision bf16`), and each worker keeps a cache of the network's
answers by position. The games are the same search either way; for the d512 at 64 simulations 423 positions/s instead
of 249. `+` and `-` add or remove a worker while it runs; ctrl+c finishes the games being played, a second ctrl+c
stops at once, and running it again with the same flags carries on.

The Rust search and rules must match `engine/mcts.py` and the Python reference games in `tests/reference_selfplay.py`
exactly; `tests/test_selfplay_rs.py` plays both and compares every number, so a change to the search goes in both.

`training/train.py --init-checkpoint` fine-tunes at a peak learning rate of 1e-3. `--result-weight` blends the result and the
root's score into the value target, `--visit-temperature` sharpens the visit shares, and `--mix-set` keeps a quarter of
every batch (`--mix-share`) from a Stockfish-labelled set.

## Tests

```
uv run pytest
```

## Layout

| Path | What it is |
|---|---|
| `model/` | The architecture: `encoding.py` (boards to 70 tokens, moves to one of 1,968 ids, and back), `model.py` (the transformer: board tokens in, move logits and an eval out) and `fused_blocks.py` (the same blocks as Triton kernels for the engine's GPU inference; a change to `model.py`'s blocks goes here too) |
| `training/` | `train.py` (the single-GPU training loop), `train_ddp.py` (the DDP trainer), `muon.py` (torch's Muon with each shape's matrices orthogonalised in batched matmuls, shared out between DDP ranks), `data.py` (training sets as datasets and loaders), `budget.py` (a run's time budget), `compare_runs.py` (every run compared, and the progress page) and `tensorboard_server.py` |
| `engine/` | Playing chess: `mcts.py` (Monte Carlo tree search over the model), `player.py` (move selection and model loading), `engine_search.py` and `uci.py` (the UCI engine and its batched search), `selfplay.py` and `inference.py` (self-play's worker processes and GPU inference process), `selfplay_rs/` (the search and self-play games in Rust), `checkpoint.py` (saving and loading checkpoints), `tablebase.py` (the Syzygy tables: download and probes) and `Carlsen.pgn` (match openings, test games and the int8 calibration positions) |
| `tools/` | Not in git, local only. Labelling, puzzles, attention probes and maps (`plot_attention.py`, like `plot_templates.py`), the scaling grid's and attention report's figures, and `play/` (matches against other engines, match analysis, the Elo log) |
| `artifacts/` | Data, checkpoints, runs, engines, results and reports. Code here is only the scripts and source snapshots an experiment or audit ran with, kept beside its results. Dated folders hold one experiment each, `plots/` the attention, template and embedding pictures, `reports/` the write-ups that span several, `patches/` code for experiments that were not merged, and `elo_results.jsonl` the match log. Not in git, local only. Among the write-ups are `scaling_grid_20260925/` (how val loss scales with model size and positions, with the size to pick for a compute budget), `reports/selfplay_and_leela_2026-09-26.md` (the d512, its self-play and fine-tunes, and how its moves compare with Leela's and with deeper Stockfish), `attention_lookups_20260927/` (weight decay for long runs, the block-1 attention collapse, and what every head looks up, with charts) and `mlp_layers_20261001/` (what each MLP layer does, and what dropping some costs in quality and buys in speed) |
| `tests/` | `uv run pytest` (with `tools/tests/` where `tools/` is there) |
| `notes.txt` | Not in git, local only. Running notes: ideas to try and small experiments' results |
