#sourced by build.sh and engine.sh from the checkout's folder: where a start's compiled code is kept, so that the build makes or fetches what
#every start reads

#a checkout with a `caches` folder keeps there all a start reads beside the code and the weights: the compiled code, the int8 weights and the
#python the environment runs on. The engine can then be built by one user and run by another, whose home folder is not the builder's
#(tcec_update.sh makes the folder before the build: TCEC builds its engines in one place and runs them from another)
if [ -d caches ]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/caches/triton}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/caches/compiler}"
    export CHESS_TRANSFORMER_INT8_CACHE="${CHESS_TRANSFORMER_INT8_CACHE:-$PWD/caches/int8}"
    #.venv's interpreter is a link to the python uv installs; one the machine already has may lie in the builder's home
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/caches/python}" UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"
fi
#triton's own default, named so that torch's compiler shares it: its kernels otherwise go to a folder a gpu under its own cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
#torch's compiler's cache, out of /tmp: a restart of the machine empties that
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/chess_transformer/compiler}"
