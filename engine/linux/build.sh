#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS=weights/fry-final-117.pt
WEIGHTS_URL=https://github.com/GilliamWong/chess-transformer/releases/download/fry-final-117/fry-final-117.pt
WEIGHTS_SHA256=fed2722da0085ef3299eeb72e9f06712a9426b38503aa8889bcf87affc5c0bed
#the locked shakmaty 0.30 declares rust-version 1.95
RUST_AT_LEAST=1.95.0
#the default cu129 torch runs on driver 570, cu130 needs 580 or newer
TORCH_CUDA=${TORCH_CUDA:-cu129}
export RUSTFLAGS=${RUSTFLAGS:--C target-cpu=native}

command -v cc >/dev/null || { echo "build.sh needs a C compiler: sudo apt install build-essential" >&2; exit 1; }

#where the installers below put uv and cargo (rustup is told not to edit the shell's profile), so that a rerun in a new shell finds them
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi
rust=$(rustc --version | cut -d' ' -f2)
if ! printf '%s\n' "$RUST_AT_LEAST" "$rust" | sort -VC; then
    echo "rust $rust is older than $RUST_AT_LEAST: rustup update stable" >&2
    exit 1
fi

#bytecode compiled here: a start in a new container of an image would compile it again each time
uv sync --frozen --no-default-groups --group selfplay --group "$TORCH_CUDA" --python 3.12 --compile-bytecode
.venv/bin/python -m compileall -q engine model

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

#a start reads the network's int8 weights and its compiled code (caches.sh), and makes what is missing first: about a minute on one gpu.
#With a gpu the build makes both; without one (an image's build) it downloads the ones made for this code
PREPARED_URL=${PREPARED_URL:-https://github.com/GilliamWong/chess-transformer/releases/download/start-up-caches}
source engine/linux/caches.sh

#named by what the compiled code depends on besides the gpu: the code, the locked packages, the torch build and the caches' folders,
#which their entries name
compiled_name() {
    local code
    code=$(git ls-files -s -- model 'engine/*.py' uv.lock)
    echo "compiled-$(echo "$code $TORCH_CUDA $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR" | sha256sum | cut -c1-16).tar.gz"
}

fetch_prepared() {
    curl -fL --retry 3 -o "$2" "$PREPARED_URL/$1" && return
    echo "no prepared $1 for this code: build where the gpu is, or with SKIP_CACHES=1 to leave it to every start" >&2
    exit 1
}

if [ "${SKIP_CACHES:-0}" = 1 ]; then
    echo "the int8 weights and the compiled code are left to every start of the engine (SKIP_CACHES=1)"
elif nvidia-smi >/dev/null 2>&1; then
    .venv/bin/python -m engine.quantization "$WEIGHTS"
    #a start on every gpu: torch's compiler keeps its work by the gpu's number
    printf 'isready\nquit\n' | engine/linux/engine.sh >/dev/null
    #PACK_CACHES=folder: this build's caches as the files fetch_prepared downloads, on a machine whose caches hold nothing else
    if [ -n "${PACK_CACHES:-}" ]; then
        int8=$(.venv/bin/python -m engine.quantization "$WEIGHTS" --path)
        compiled=$(compiled_name)
        mkdir -p "$PACK_CACHES"
        cp "$int8" "$PACK_CACHES/"
        tar -czf "$PACK_CACHES/$compiled" -C / "${TRITON_CACHE_DIR#/}" "${TORCHINDUCTOR_CACHE_DIR#/}"
    fi
else
    int8=$(.venv/bin/python -m engine.quantization "$WEIGHTS" --path)
    compiled=$(compiled_name)
    mkdir -p "$(dirname "$int8")"
    fetch_prepared "$(basename "$int8")" "$int8.partial"
    mv "$int8.partial" "$int8"
    fetch_prepared "$compiled" "$WEIGHTS.compiled.partial"
    tar -xzf "$WEIGHTS.compiled.partial" -C /
    rm "$WEIGHTS.compiled.partial"
fi
echo "built: engine/linux/engine.sh runs the engine"
