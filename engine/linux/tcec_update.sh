#!/usr/bin/env bash
#TCEC's update script for Fry (https://tcec-chess.com): run, or sourced, in an empty folder it downloads the engine's code at FRY_REF, builds
#it there and sets EXE to the file that runs the engine and NET to the network's weights, which the build downloads.
#
#Everything a start of the engine reads is kept under this folder (the checkout's `caches` folder: engine/linux/caches.sh), and EXE is a
#launcher holding the checkout's whole path, so the engine runs from a link to EXE or a copy of it, under another user than the builder.
#The build needs git, curl, a C compiler and the machine's gpus; it installs uv and Rust for the building user if they are missing.
FRY_REPOSITORY=${FRY_REPOSITORY:-https://github.com/GilliamWong/chess-transformer.git}
FRY_REF=${FRY_REF:-tcec-s30-gpu1}
FRY_FOLDER=$PWD/chess-transformer

#the build in a shell of its own: its settings and a failure's exit stay out of a shell that sources this file
if (
    set -euo pipefail
    if [ -d "$FRY_FOLDER/.git" ]; then
        git -C "$FRY_FOLDER" fetch --depth 1 origin "$FRY_REF"
        git -C "$FRY_FOLDER" checkout --quiet --detach FETCH_HEAD
    else
        git clone --depth 1 --branch "$FRY_REF" "$FRY_REPOSITORY" "$FRY_FOLDER"
    fi
    mkdir -p "$FRY_FOLDER/caches"
    "$FRY_FOLDER/engine/linux/build.sh"
    printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$FRY_FOLDER/engine/linux/engine.sh" > "$PWD/fry.sh"
    chmod +x "$PWD/fry.sh"
    #another user reads and runs all of it, and adds to the caches what a start of theirs still compiles
    chmod -R a+rX "$PWD/fry.sh" "$FRY_FOLDER"
    chmod -R a+w "$FRY_FOLDER/caches/triton" "$FRY_FOLDER/caches/compiler"
); then
    EXE=$PWD/fry.sh
    NET=$FRY_FOLDER/weights/fry-final-117.pt
    echo "Fry is built: EXE=$EXE NET=$NET"
else
    echo "Fry's build failed: EXE is not set" >&2
    return 1 2>/dev/null || exit 1
fi
