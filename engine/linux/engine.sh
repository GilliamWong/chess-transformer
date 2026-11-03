#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m engine.uci weights/fry-final-117.pt --select mcts --gpu --gpus all "$@"
