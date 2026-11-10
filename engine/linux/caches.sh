#sourced by build.sh and engine.sh: where a start's compiled code is kept, so that the build makes or fetches what every start reads
#triton's own default, named so that torch's compiler shares it: its kernels otherwise go to a folder a gpu under its own cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
#torch's compiler's cache, out of /tmp: a restart of the machine empties that
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/chess_transformer/compiler}"
