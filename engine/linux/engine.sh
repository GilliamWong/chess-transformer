#!/usr/bin/env bash
#the checkout is found from this file itself, not from the name it was started by: a gui may start it through a link in a folder of its own
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."
source engine/linux/caches.sh
#a rust panic's backtrace (the build keeps line tables) and a fatal signal's python stack go to stderr
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}" PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
exec .venv/bin/python -m engine.uci weights/fry-final-117.pt --gpu --gpus all --infinite-until-stop --fit-memory "$@"
