# Fry

## Description

Fry is a chess engine built on a transformer network: a 28-layer model trained on positions labelled by Stockfish and
the Syzygy tablebases, then on Leela Chess Zero's self-play games. It plays with Monte Carlo Tree Search.

The architecture, the experiments behind it and an in-browser explorer are on the
[Chess Transformer site](https://gilliamwong.github.io/chess-transformer-site/).

## Download and use

On Linux with an NVIDIA GPU (driver 580 or newer), git, curl and a C compiler (`build-essential`):

```
git clone https://github.com/GilliamWong/chess-transformer.git
chess-transformer/engine/linux/build.sh
chess-transformer/engine/linux/engine.sh
```

`build.sh` installs uv and Rust if they are missing, builds the Python environment, downloads the network
[`fry-final-117`](https://github.com/GilliamWong/chess-transformer/releases/tag/fry-final-117) (453 MB, SHA-256
`fed2722da0085ef3299eeb72e9f06712a9426b38503aa8889bcf87affc5c0bed`) into `weights/` and prepares its int8 weights on
the GPU. Run it again after a `git pull`. `engine/linux/update.sh [folder]` clones or pulls and builds in one step.

`engine.sh` is the UCI engine, searching on every GPU the machine has. Its first start takes ~25 s and later starts
~7 s on one GPU or ~35-40 s on eight, so give the GUI that long to answer `isready`.

| Option | Default | What it sets |
|---|---|---|
| `SyzygyPath` | `<empty>` | Syzygy table folders, `:` between them |
| `SyzygyProbeLimit` | 7 | The most pieces the tables are used for, 5 to 7: `SyzygyPath` uses the most its folders hold up to it |
| `MoveOverhead` | 20 | Milliseconds kept back from each move's time for the GUI's lag |
| `CachePositions` | 5000000 | Network answers kept by position, ~190-260 bytes each |
| `RamLimitMb` | 0 | MB the search tree may take (0 for no limit); the cache and the kept tree's copy come on top, so set it to about (memory - cache) / 1.7 |
| `AdvanceThreads` | 4 | Threads that carry the tree on to the next move while the opponent thinks |
| `Leaves` | 64 | Positions sent to the network at once on each GPU |
| `GPUs` | every GPU | `all`, or CUDA devices by number (`0,1,2,3`) |
| `SmartPruningFactor` | 1 | Stops a search once its best move cannot be caught with the simulations left / this; 0 uses all of its time |
| `FirstMoveBonus` | 0 | Extra time for a game's first search: 1 + this + `BookPlyBonus` x its ply (at most 12) times its share |
| `BookPlyBonus` | 0 | See `FirstMoveBonus` |
| `SearchSettings` | `<empty>` | The search's constants, `name=value,...` (e.g. `c_puct=0.6`) |

`MoveOverhead`, `FirstMoveBonus` and `BookPlyBonus` take effect at once; the others when the engine loads at its first
`isready` or `go`. A value the engine cannot take is refused with an `info string`. There is no `Hash` or `Threads`
(the memory is `RamLimitMb`, the threads follow from the GPUs), and the engine does not ponder.

`engine/linux/tcec.json` is a tournament entry for a box like TCEC's (8x RTX 5090, at most 320 GiB of RAM for an engine):
`RamLimitMb` 180000, `CachePositions` 50000000, `AdvanceThreads` 16 and `MoveOverhead` 200,
with `SyzygyPath` set to the box's 7 piece tables.
