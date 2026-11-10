# Fry

## Description

Fry is a chess engine built on a transformer network: a 28-layer model trained on positions labelled by Stockfish and
the Syzygy tablebases, then on Leela Chess Zero's self-play games. It plays with Monte Carlo Tree Search.

The architecture, the experiments behind it and an in-browser explorer are on the
[Chess Transformer site](https://gilliamwong.github.io/chess-transformer-site/).

## Download and use

On Linux with an NVIDIA GPU (driver 570 or newer), git, curl and a C compiler (`build-essential`):

```
git clone https://github.com/GilliamWong/chess-transformer.git
chess-transformer/engine/linux/build.sh
chess-transformer/engine/linux/engine.sh
```

`build.sh` installs uv and Rust if they are missing, builds the Python environment (PyTorch's CUDA 12.9 build, which
runs on driver 570; `TORCH_CUDA=cu130` for CUDA 13.0 on drivers from 580) with the Rust search compiled for the
machine's CPU, downloads the network
[`fry-final-117`](https://github.com/GilliamWong/chess-transformer/releases/tag/fry-final-117) (453 MB, SHA-256
`fed2722da0085ef3299eeb72e9f06712a9426b38503aa8889bcf87affc5c0bed`) into `weights/` and prepares what a start would
otherwise make first: the network's int8 weights and the engine's compiled GPU code. With the GPU it makes them by
starting the engine once; without one (an image's build, as `engine/linux/fry.Dockerfile` for the CCC) it downloads
the ones published for this code. Run it again after a `git pull`. `engine/linux/update.sh [folder]` clones or pulls
and builds in one step.

`engine.sh` is the UCI engine, searching on every GPU the machine has. A start takes ~7 s on one GPU or ~35-40 s on
eight, so give the GUI that long to answer `isready`. The kernels are picked from the GPU's model at start-up;
`engine/linux/tcec.json` and `engine/linux/ccc.json` hold the options for TCEC's and the CCC's machines.

`engine.sh` fits `CachePositions` and `RamLimitMb` to the machine: the engine takes two fifths of its memory (of a
container's limit, if it has one), which leaves room for a tournament's other engine and the tables. The cache gets up
to 200 million positions and the tree's limit is what the rest allows, counting the copy made when a tree is kept. On
TCEC's 768 GiB machine that is a 179 GB tree. Setting either option overrides it.

| Option | Default | Description |
|---|---|---|
| `SyzygyPath` | `<empty>` | Syzygy table folders, separated by `:` |
| `SyzygyProbeLimit` | 7 | Most pieces to probe the tables for (5-7) |
| `MoveOverhead` | 20 | Milliseconds kept back each move for GUI lag |
| `CachePositions` | fitted | Network evaluations cached by position (5000000 without `engine.sh`) |
| `RamLimitMb` | fitted | Memory limit for the search tree in MB, 0 for none (0 without `engine.sh`) |
| `AdvanceThreads` | 4 | Threads that update the kept tree on the opponent's time (0 for none) |
| `GcThreads` | 2 | Threads that free the discarded tree after a move |
| `Leaves` | 64 | Positions evaluated at once on each GPU |
| `GPUs` | every GPU | `all`, or CUDA device numbers (`0,1,2,3`) |
| `SmartPruningFactor` | 1 | Stops early once the best move can't change (0 to use all the time) |
| `FirstMoveBonus` | 0 | Extra time for the game's first search |
| `BookPlyBonus` | 0 | Extra first-search time per book ply |
| `SearchSettings` | `<empty>` | Search constants as `name=value,...`; `value_sharpness` defaults to the network's own |

## Training data and credits

The network `fry-final-117` was trained on:

- positions from the [Lichess Elite Database](https://database.nikonoel.fr/), a selection of the
  [Lichess database](https://database.lichess.org/) (released under CC0), labelled with moves and evaluations by
  [Stockfish](https://stockfishchess.org/) and with results from the Syzygy tablebases (Ronald de Man);
- Leela Chess Zero's self-play games. The network contains information from the
  [Leela Chess Zero training data](https://storage.lczero.org/files/training_data/), which is made available under the
  [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/) (ODbL); its individual contents are under the
  [Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/).

The search follows the design of [Leela Chess Zero](https://lczero.org/)'s, and TCEC's settings use its search
constants. The rules and tablebase probing come from [python-chess](https://github.com/niklasf/python-chess),
[shakmaty](https://github.com/niklasf/shakmaty) and
[shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy) by Niklas Fiekas.

Fry is free software under the GNU General Public License, version 3 or later: see [LICENSE](LICENSE).
