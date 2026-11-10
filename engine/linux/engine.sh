#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source engine/linux/caches.sh
#a rust panic's backtrace (the build keeps line tables) and a fatal signal's python stack go to stderr
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}" PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
exec .venv/bin/python -m engine.uci weights/fry-final-117.pt --gpu --gpus all --infinite-until-stop "$@"
