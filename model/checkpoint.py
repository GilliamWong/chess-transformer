#a checkpoint holds the weights and the settings the model was built with, so any size loads back
import json
import math
from pathlib import Path
import torch
from model.model import ChessTransformer

#no code lives here: pgns, training sets, checkpoints, runs, engines
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
#one folder per run: tensorboard logs, settings.json, model.pt (the best checkpoint) and final.pt
RUNS_DIR = ARTIFACTS_DIR / "runs"

#settings removed in the 2026-09-23 cleanup; every checkpoint saved before it names at least dropout
PRE_CLEANUP_SETTINGS = ("history", "future", "dropout", "swiglu", "value_target")

#the mlp's hidden width in checkpoints saved before mlp_hidden was a setting (2026-09-25): 8/3 of d_model or mlp_ratio
#times it, to a multiple of 8
def legacy_mlp_hidden(d_model: int, mlp_ratio: float | None) -> int:
    if mlp_ratio is None:
        return 8 * math.ceil(d_model / 3)
    return max(8, 8 * round(mlp_ratio * d_model / 8))

#the run that finished last. A run still training is skipped: its model.pt may be half written
def latest_checkpoint() -> Path:
    finished = {}
    for settings_path in RUNS_DIR.glob("*/settings.json"):
        finished_at = json.loads(settings_path.read_text()).get("finished_at")
        if finished_at is not None and (settings_path.parent / "model.pt").exists():
            finished[finished_at] = settings_path.parent / "model.pt"
    if not finished:
        raise FileNotFoundError(f"no finished run with a model.pt in {RUNS_DIR}, train a model with train.py first")
    return finished[max(finished)]

def save_checkpoint(model: ChessTransformer, model_settings: dict, checkpoint_path: Path) -> None:
    torch.save({"model_settings": model_settings, "state_dict": model.state_dict()}, checkpoint_path)

#the model (on device, still in training mode) and its settings
def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ChessTransformer, dict]:
    #map_location so a gpu checkpoint also loads without one
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_settings" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is bare weights without model settings, it was saved before checkpoints recorded them")

    model_settings = checkpoint["model_settings"]
    old_settings = [name for name in PRE_CLEANUP_SETTINGS if name in model_settings]
    if old_settings:
        raise ValueError(f"{checkpoint_path} predates the 2026-09-23 cleanup (its settings name {', '.join(old_settings)}), the model "
                         "here cannot load it. Load it from a checkout of tag pre-cleanup-2026-09-23, for example "
                         "git worktree add ../chess-transformer-pre-cleanup pre-cleanup-2026-09-23")
    #the template embedding was removed on 2026-09-24; checkpoints before that name it, 0 in all but two d128 runs
    template_maps = model_settings.pop("template_maps", 0)
    if template_maps:
        raise ValueError(f"{checkpoint_path} has a template embedding ({template_maps} template maps), which the model here no longer has. "
                         "Load it from a checkout of commit 1a976d9, for example git worktree add ../chess-transformer-template-maps 1a976d9")
    #before 2026-09-25 the eval head was one logit of the expected score
    model_settings.setdefault("value_bins", 1)
    if "mlp_hidden" not in model_settings:
        model_settings["mlp_hidden"] = legacy_mlp_hidden(model_settings["d_model"], model_settings.pop("mlp_ratio", None))
    #the input skip and RMSNorm were settings on 2026-10-07 before both became the model; that day's runs name them
    for name in ("input_skip", "rms_norm"):
        model_settings.pop(name, None)
    #LayerNorm (with a bias) and no input skip until then
    state_dict = checkpoint["state_dict"]
    if "ln.bias" in state_dict or "encoder.0.residual_scale" not in state_dict:
        raise ValueError(f"{checkpoint_path} has LayerNorms and no input skip, from before 2026-10-07; the model here has RMSNorms and the skip. "
                         "Load it from a checkout of commit 6742baf, for example git worktree add ../chess-transformer-layernorm 6742baf")
    model = ChessTransformer(**model_settings)
    model.load_state_dict(state_dict)

    return model.to(device), model_settings

#a checkpoint to train on from: its size and weights, but with value_bins eval head bins. With other bins than the
#checkpoint's, the eval head's last layer starts afresh and everything else carries over
def warm_start(checkpoint_path: Path, device: torch.device, value_bins: int) -> tuple[ChessTransformer, dict]:
    model, model_settings = load_checkpoint(checkpoint_path, device)
    if value_bins == model_settings["value_bins"]:
        return model, model_settings
    model_settings = {**model_settings, "value_bins": value_bins}
    warm = ChessTransformer(**model_settings).to(device)
    last_layer = f"eval_head.{len(warm.eval_head) - 1}."
    weights = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith(last_layer)}
    missing, unexpected = warm.load_state_dict(weights, strict=False)
    if set(missing) != {last_layer + "weight", last_layer + "bias"} or unexpected:
        raise ValueError(f"a warm start from {checkpoint_path} would leave {missing} new and {unexpected} unused, not just {last_layer}*")
    return warm, model_settings
