#!/usr/bin/env bash
#builds the engine on linux from inside its checkout: the python environment with the rust search, the network's
#weights, and the int8 rounding of those weights on the gpu. Safe to run again after a git pull: each step skips what is
#done. Needs git, curl, a C compiler (build-essential) and an NVIDIA driver of 570 or newer: torch's CUDA 12.9 build,
#which CUDA 12's minor version compatibility runs on TCEC's 570 (TORCH_CUDA=cu130 for CUDA 13.0, drivers 580 and newer).
#The rust search is compiled for this machine's cpu (AVX-512 on TCEC's EPYCs); a RUSTFLAGS already set replaces that.
#SKIP_INT8=1 leaves the rounding to the engine's first isready
set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS=weights/fry-final-117.pt
WEIGHTS_URL=https://github.com/GilliamWong/chess-transformer/releases/download/fry-final-117/fry-final-117.pt
WEIGHTS_SHA256=fed2722da0085ef3299eeb72e9f06712a9426b38503aa8889bcf87affc5c0bed
#the rust search is edition 2024
RUST_AT_LEAST=1.85.0
TORCH_CUDA=${TORCH_CUDA:-cu129}
export RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}

command -v cc >/dev/null || { echo "build.sh needs a C compiler: sudo apt install build-essential" >&2; exit 1; }

#uv builds the python environment from uv.lock, fetching python itself if need be
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
#rustup into CARGO_HOME (default ~/.cargo), the shell's profile left as it is
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
if [ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]; then
    . "${CARGO_HOME:-$HOME/.cargo}/env"
fi
rust=$(rustc --version | cut -d' ' -f2)
if [ "$(printf '%s\n' "$RUST_AT_LEAST" "$rust" | sort -V | head -1)" != "$RUST_AT_LEAST" ]; then
    echo "rust $rust is older than $RUST_AT_LEAST: rustup update stable" >&2
    exit 1
fi

uv sync --frozen --no-group dev --no-group cu130 --group "$TORCH_CUDA" --python 3.12

if ! echo "$WEIGHTS_SHA256  $WEIGHTS" | sha256sum --check --status 2>/dev/null; then
    mkdir -p weights
    curl -fL --retry 3 -o "$WEIGHTS.partial" "$WEIGHTS_URL"
    if ! echo "$WEIGHTS_SHA256  $WEIGHTS.partial" | sha256sum --check --status; then
        echo "the downloaded weights do not match their checksum" >&2
        exit 1
    fi
    mv "$WEIGHTS.partial" "$WEIGHTS"
fi

#cached under ~/.cache/chess_transformer by the weights' contents, where the engine looks first
if [ "${SKIP_INT8:-0}" != 1 ] && nvidia-smi >/dev/null 2>&1; then
    .venv/bin/python -m engine.quantization "$WEIGHTS"
else
    echo "the int8 weights are left to the engine's first isready (no gpu, or SKIP_INT8=1)"
fi
echo "built: engine/linux/engine.sh runs the engine"
