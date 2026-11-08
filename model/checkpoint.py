import json
from pathlib import Path
import torch
from model.model import ChessTransformer

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
#one folder per run: settings.json, model.pt (the best checkpoint), final.pt, tensorboard logs
RUNS_DIR = ARTIFACTS_DIR / "runs"

#a run still training is skipped, its model.pt may be half written
def latest_checkpoint() -> Path:
    finished = {}
    for settings_path in RUNS_DIR.glob("*/settings.json"):
        model_path = settings_path.with_name("model.pt")
        finished_at = json.loads(settings_path.read_text()).get("finished_at")
        if finished_at is not None and model_path.exists():
            finished[finished_at] = model_path
    if not finished:
        raise FileNotFoundError(f"no finished run with a model.pt in {RUNS_DIR}, train a model with train.py first")
    return finished[max(finished)]

def save_checkpoint(model: ChessTransformer, model_settings: dict, checkpoint_path: Path) -> None:
    torch.save({"model_settings": model_settings, "state_dict": model.state_dict()}, checkpoint_path)

#the value sharpness the engine's search serves the checkpoint at, None if it records none (the engine's default then)
def served_value_sharpness(checkpoint_path: Path) -> float | None:
    return torch.load(checkpoint_path, map_location="cpu", mmap=True).get("value_sharpness")

#the model is in training mode
def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ChessTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_settings = checkpoint["model_settings"]
    #the shipped checkpoint records loops: the times the blocks run, and 1 is the only count the model has
    if model_settings.pop("loops", 1) != 1:
        raise ValueError(f"{checkpoint_path} runs its blocks more than once (loops in its settings), which the model here cannot")
    model = ChessTransformer(**model_settings)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), model_settings

def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer:
    model, _ = load_checkpoint(checkpoint_path, device)
    return model.eval()
