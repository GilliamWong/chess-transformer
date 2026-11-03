#!/usr/bin/env bash
#clones the engine into FOLDER (default ./chess-transformer) or pulls it there, then builds it
set -euo pipefail
REPOSITORY=https://github.com/GilliamWong/chess-transformer.git
FOLDER=${1:-chess-transformer}

if [ -d "$FOLDER/.git" ]; then
    git -C "$FOLDER" pull --ff-only
else
    git clone --depth 1 "$REPOSITORY" "$FOLDER"
fi
"$FOLDER/engine/linux/build.sh"
