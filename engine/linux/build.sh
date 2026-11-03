#!/usr/bin/env bash
#builds the python environment, rust search, weights and their int8 rounding; safe to run again after a git pull
#the default cu129 torch runs on driver 570, TORCH_CUDA=cu130 needs 580 or newer; SKIP_INT8=1 leaves the rounding to the engine
set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS=weights/fry-final-117.pt
WEIGHTS_URL=https://github.com/GilliamWong/chess-transformer/releases/download/fry-final-117/fry-final-117.pt
WEIGHTS_SHA256=fed2722da0085ef3299eeb72e9f06712a9426b38503aa8889bcf87affc5c0bed
RUST_AT_LEAST=1.85.0
TORCH_CUDA=${TORCH_CUDA:-cu129}
export RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}

command -v cc >/dev/null || { echo "build.sh needs a C compiler: sudo apt install build-essential" >&2; exit 1; }

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
cargo_env=${CARGO_HOME:-$HOME/.cargo}/env
if [ -f "$cargo_env" ]; then
    . "$cargo_env"
fi
rust=$(rustc --version | cut -d' ' -f2)
if ! printf '%s\n' "$RUST_AT_LEAST" "$rust" | sort -VC; then
    echo "rust $rust is older than $RUST_AT_LEAST: rustup update stable" >&2
    exit 1
fi

uv sync --frozen --no-default-groups --group selfplay --group "$TORCH_CUDA" --python 3.12

weights_ok() { echo "$WEIGHTS_SHA256  $1" | sha256sum --check --status 2>/dev/null; }
if ! weights_ok "$WEIGHTS"; then
    mkdir -p weights
    curl -fL --retry 3 -o "$WEIGHTS.partial" "$WEIGHTS_URL"
    if ! weights_ok "$WEIGHTS.partial"; then
        echo "the downloaded weights do not match their checksum" >&2
        exit 1
    fi
    mv "$WEIGHTS.partial" "$WEIGHTS"
fi

#caches the int8 weights where the engine looks first
if [ "${SKIP_INT8:-0}" != 1 ] && nvidia-smi >/dev/null 2>&1; then
    .venv/bin/python -m engine.quantization "$WEIGHTS"
else
    echo "the int8 weights are left to the engine's first isready (no gpu, or SKIP_INT8=1)"
fi
echo "built: engine/linux/engine.sh runs the engine"
