#!/usr/bin/env bash
#the engine as tournaments run it, after build.sh; further arguments go to engine/uci.py
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m engine.uci weights/fry-final-117.pt --select mcts --gpu --gpus all "$@"
