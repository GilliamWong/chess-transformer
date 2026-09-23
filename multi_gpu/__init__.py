#the multi-gpu trainer, kept apart so main.py stays a plain single-gpu loop. Run from the repository root as a
#module, so the training code there can be imported:
#   uv run torchrun --standalone --nproc-per-node=2 -m multi_gpu.train_ddp
