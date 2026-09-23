#training on hyak's preemptible checkpoint GPUs, any number and any mix of them, each of which can disappear at any moment.
#Every GPU runs a worker that trains its own copy of the model, and the copies are averaged once a round through the
#shared filesystem (DiLoCo, arXiv 2311.08105): a worker starts the round from the round's global weights, runs AdamW on
#its own random positions until the round's deadline, and saves the change it made. The merger (one CPU core on a
#partition that is not preempted) averages the changes, weighted by the positions behind each, and takes an outer
#Nesterov step with them. Nothing needs a network between the jobs, a rendezvous, or the same GPU type or precision.
#
#   python train_hyak.py init   <run dir> --config config.json    the plan and round 0's weights
#   python train_hyak.py merger <run dir>                          one per run, on a CPU node
#   python train_hyak.py worker <run dir> --worker-id N            one per GPU; a restarted worker carries on
#
#Files in the run dir, each written to a temporary name and renamed into place so no reader sees half a file:
#   config.json                   the plan: model, data, inner and outer optimizer, round length, end time
#   global/<round>.pt, .json      the weights a round starts from, and its deadline (the json is written last)
#   deltas/<round>/<worker>.pt    a worker's change over the round (bf16) and its .json (positions, loss, speed)
#   workers/<worker>.pt           a worker's own AdamW state, so a preempted worker resumes it
#   val/<round>.json              the round's global weights on the first val batches, measured by a worker
#   merger.pt                     the outer momentum
#   checkpoints/, tensorboard/    for playing and for watching, in main.py's checkpoint format and tag names
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from torch import nn
from data import load_training_set
from main import VALUE_LOSS_WEIGHT, WARMUP_STEPS, lr_multiplier, policy_loss, value_loss, warm_file_cache
from model import ChessTransformer
from model_leela import LeelaChessTransformer
from leela_head_to_head import OursWithBoardEmbedding, OursWithBoardMix

NETWORKS = {"ChessTransformer": ChessTransformer, "LeelaChessTransformer": LeelaChessTransformer,
            "OursWithBoardEmbedding": OursWithBoardEmbedding, "OursWithBoardMix": OursWithBoardMix}
#tensorboard steps count batches of this many positions, so the curves line up with main.py's runs
STEPS_OF = 1024
#a worker stops this long before the deadline, to save its change in time, and does not start a round with less than
#MIN_ROUND_LEFT left (or a third of a round, for short test rounds)
STOP_BEFORE_DEADLINE = 15
MIN_ROUND_LEFT = 90
POLL_SECONDS = 5
#the merger waits past the grace while changes keep arriving, at most this long after the deadline, and stops
#early once none has arrived for MERGE_QUIET_SECONDS
MERGE_MAX_WAIT = 120
MERGE_QUIET_SECONDS = 10
#a worker's AdamW state is saved every this many rounds: it is only needed after a preemption, and 100 workers
#saving it every round (about 0.5 GB each) slowed gscratch enough that their changes missed the merge
STATE_EVERY_ROUNDS = 4


def save_atomically(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".json":
        temporary.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, temporary)
    os.replace(temporary, path)


def round_name(round_index: int) -> str:
    return f"{round_index:05d}"


#the newest round whose json has been published, or None
def latest_round(run_dir: Path) -> int | None:
    published = sorted(int(path.stem) for path in (run_dir / "global").glob("[0-9]*.json"))
    return published[-1] if published else None


def build_model(config: dict) -> nn.Module:
    return NETWORKS[config["network"]](**config["model_settings"])


#the learning rate multiplier of the whole run at a moment: cosine over the wall clock from start to end, as main.py's
#over its budget. A worker's own warmup comes on top, whenever its optimizer starts from nothing
def schedule(config: dict, now: float) -> float:
    progress = min(1.0, max(0.0, (now - config["start_time"]) / (config["end_time"] - config["start_time"])))
    return lr_multiplier(WARMUP_STEPS + 1, progress, "cosine")


def checkpoint(config: dict, state_dict: dict) -> dict:
    return {"network": config["network"], "model_settings": config["model_settings"], "state_dict": state_dict}


def init(run_dir: Path, config_path: Path) -> None:
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"{run_dir} already has a run")
    config = json.loads(config_path.read_text())
    config["start_time"] = time.time()
    config["end_time"] = config["start_time"] + config["hours"] * 3600
    torch.manual_seed(config["seed"])
    model = build_model(config)
    save_atomically(config, run_dir / "config.json")
    save_atomically(model.state_dict(), run_dir / "global" / "00000.pt")
    #round 0 is longer: every worker compiles and warms up in it
    save_atomically({"round": 0, "deadline": config["start_time"] + config["first_round_seconds"], "positions": 0, "done": False},
                    run_dir / "global" / "00000.json")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"{run_dir}: {config['network']} with {parameters:,} parameters, ends {time.ctime(config['end_time'])}")


def merger(run_dir: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter
    config = json.loads((run_dir / "config.json").read_text())
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    momentum = torch.load(run_dir / "merger.pt") if (run_dir / "merger.pt").exists() else None

    while True:
        #read again every round, so the outer step, round length and grace can be changed while the run goes on
        config = json.loads((run_dir / "config.json").read_text())
        outer_lr, momentum_factor = config["outer_lr"], config["outer_momentum"]
        round_index = latest_round(run_dir)
        info = json.loads((run_dir / "global" / f"{round_name(round_index)}.json").read_text())
        if info["done"]:
            print("the run is done")
            return
        #the workers stop a little before the deadline and need a moment to write their changes
        while time.time() < info["deadline"] + config["merge_grace_seconds"]:
            time.sleep(POLL_SECONDS)
        #and when gscratch is busy, more than a moment: wait on while changes keep coming in, until as many are in as
        #reached the last round (counting late ones), none has come for a while, or it gets too late
        delta_dir = run_dir / "deltas" / round_name(round_index)
        last_round_dir = run_dir / "deltas" / round_name(round_index - 1)
        expected = len(list(last_round_dir.glob("*.json"))) if last_round_dir.exists() else 0
        count, quiet_since = -1, time.time()
        while time.time() < info["deadline"] + MERGE_MAX_WAIT:
            arrived = len(list(delta_dir.glob("*.json"))) if delta_dir.exists() else 0
            if arrived != count:
                count, quiet_since = arrived, time.time()
            if arrived >= expected or time.time() - quiet_since >= MERGE_QUIET_SECONDS:
                break
            time.sleep(2)

        weights = torch.load(run_dir / "global" / f"{round_name(round_index)}.pt")
        reports = [json.loads(path.read_text()) for path in sorted(delta_dir.glob("*.json"))] if delta_dir.exists() else []
        positions = sum(report["positions"] for report in reports)
        if positions > 0:
            #the changes, weighted by the positions behind each: an H200's round counts for more than an RTX 6000's. Read
            #several at a time, since reading them one after the other from gscratch kept every GPU waiting
            average = {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in weights.items() if tensor.is_floating_point()}
            with ThreadPoolExecutor(max_workers=8) as pool:
                loads = {pool.submit(torch.load, delta_dir / f"{report['worker']}.pt"): report for report in reports}
                for loaded in as_completed(loads):
                    delta, share = loaded.result(), loads[loaded]["positions"] / positions
                    for name in average:
                        average[name] += delta[name].float() * share
            if momentum is None:
                momentum = {name: torch.zeros_like(tensor) for name, tensor in average.items()}
            #outer nesterov step on the averaged change (the change is the old weights minus the new, like a gradient)
            for name, change in average.items():
                momentum[name].mul_(momentum_factor).add_(change)
                weights[name] -= outer_lr * (change + momentum_factor * momentum[name])
        total_positions = info["positions"] + positions
        done = time.time() >= config["end_time"]
        next_index = round_index + 1
        save_atomically(weights, run_dir / "global" / f"{round_name(next_index)}.pt")
        round_seconds = config["round_seconds"]
        save_atomically({"round": next_index, "deadline": time.time() + round_seconds, "positions": total_positions, "done": done,
                         "workers": len(reports)}, run_dir / "global" / f"{round_name(next_index)}.json")
        if momentum is not None:
            save_atomically(momentum, run_dir / "merger.pt")

        #tensorboard, with main.py's tag names; steps are batches of STEPS_OF positions
        step = total_positions // STEPS_OF
        if reports:
            writer.add_scalar("loss/train", sum(report["train_loss"] * report["positions"] for report in reports) / positions, step)
            writer.add_scalar("positions_per_second", sum(report["positions_per_second"] for report in reports), step)
        writer.add_scalar("workers", len(reports), step)
        writer.add_scalar("outer_lr", outer_lr, step)
        writer.add_scalar("learning_rate", config["peak_lr"] * schedule(config, time.time()), step)
        val_path = run_dir / "val" / f"{round_name(round_index)}.json"
        if val_path.exists():
            val = json.loads(val_path.read_text())
            writer.add_scalar("loss/val", val["loss"], info["positions"] // STEPS_OF)
            writer.add_scalar("accuracy/val", val["accuracy"], info["positions"] // STEPS_OF)
        writer.flush()
        gpus = {}
        for report in reports:
            gpus[report["gpu"]] = gpus.get(report["gpu"], 0) + 1
        print(f"{time.strftime('%H:%M:%S')} round {round_index}: {len(reports)} workers {gpus}, {positions:,} positions "
              f"({total_positions:,} in all){' DONE' if done else ''}", flush=True)

        #weights to play and keep: every few rounds and at the end
        if done or next_index % config["checkpoint_every_rounds"] == 0:
            save_atomically(checkpoint(config, weights), run_dir / "checkpoints" / f"round_{round_name(next_index)}.pt")
            save_atomically(checkpoint(config, weights), run_dir / "checkpoints" / "latest.pt")
        #old rounds are not needed any more: a worker always starts from the newest
        for old in (run_dir / "global").glob("[0-9]*.pt"):
            if int(old.stem) < next_index - 2:
                old.unlink(missing_ok=True)
        for old in (run_dir / "deltas").glob("[0-9]*"):
            if int(old.name) < round_index:
                for path in old.iterdir():
                    path.unlink(missing_ok=True)
                old.rmdir()
        if done:
            save_atomically(checkpoint(config, weights), run_dir / "checkpoints" / "final.pt")
            print("the run is done")
            return


#the first val batches, the same every time: loss and accuracy against stockfish's best move
@torch.no_grad()
def validate(model: nn.Module, val_dataset, batches: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[float, float]:
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    for start in range(0, batches * batch_size, batch_size):
        boards, target_moves, target_probabilities, _, history, _ = (tensor.to(device) for tensor in val_dataset[list(range(start, start + batch_size))])
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(boards, history)[0]
        loss_total += policy_loss(logits.float(), target_moves, target_probabilities).item() * len(boards)
        correct += (logits.argmax(dim=-1) == target_moves[:, 0]).sum().item()
        count += len(boards)
    model.train()
    return loss_total / count, correct / count


def worker(run_dir: Path, worker_id: int) -> None:
    config = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    #Turing (RTX 6000, 2080 Ti) has no bf16 tensor cores: fp16 with a loss scale there. is_bf16_supported() says yes on
    #Turing too, counting emulation, so the capability decides. Only fp32 weights are ever averaged
    use_bf16 = properties.major >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    batch_size = next(size for memory, size in config["batch_by_gpu_memory"] if properties.total_memory >= memory * 1e9)
    name = f"w{worker_id:03d}"
    print(f"{name}: {properties.name}, {properties.total_memory / 1e9:.0f} GB, {'bf16' if use_bf16 else 'fp16'}, batch {batch_size}, "
          f"restart {os.environ.get('SLURM_RESTART_COUNT', 0)}", flush=True)

    model_settings = config["model_settings"]
    train_dataset, val_dataset = load_training_set(config["training_set"], model_settings["history"], "stockfish", config["target_temperature"],
                                                   model_settings.get("future", 0), model_settings.get("value_target", "eval"))
    #the files front to back once, so the random batches after are served from memory rather than from gscratch
    warm_file_cache(train_dataset, val_dataset)
    model = build_model(config).to(device)
    compiled_model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["peak_lr"], weight_decay=config["weight_decay"], fused=True)
    #steps since this worker's optimizer started from nothing: its own warmup
    local_steps = 0
    state_path = run_dir / "workers" / f"{name}.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        local_steps = state["local_steps"]
        print(f"{name}: resumed its optimizer after {local_steps:,} steps", flush=True)
    #different positions on every worker, and after every restart
    generator = torch.Generator().manual_seed(config["seed"] * 100003 + worker_id * 1009 + int(os.environ.get("SLURM_RESTART_COUNT", 0)))

    while True:
        round_index = latest_round(run_dir)
        info = json.loads((run_dir / "global" / f"{round_name(round_index)}.json").read_text())
        if info["done"]:
            print(f"{name}: the run is done", flush=True)
            return
        if info["deadline"] - time.time() < min(MIN_ROUND_LEFT, config["round_seconds"] / 3):
            #too little of this round left to be worth it: wait for the next one
            while latest_round(run_dir) == round_index:
                time.sleep(POLL_SECONDS)
            continue

        start_weights = torch.load(run_dir / "global" / f"{round_name(round_index)}.pt", map_location=device)
        model.load_state_dict(start_weights)
        val_path = run_dir / "val" / f"{round_name(round_index)}.json"
        if not val_path.exists():
            loss, accuracy = validate(model, val_dataset, config["val_batches"], batch_size, device, dtype)
            try:
                save_atomically({"loss": loss, "accuracy": accuracy, "worker": name}, val_path)
            except OSError:
                #another worker measured the same weights at the same moment and got its file in first
                pass

        round_start, positions, loss_total, steps = time.time(), 0, torch.zeros((), device=device), 0
        stop_at = info["deadline"] - STOP_BEFORE_DEADLINE
        while time.time() < stop_at:
            multiplier = schedule(config, time.time())
            if local_steps < WARMUP_STEPS:
                multiplier *= (local_steps + 1) / WARMUP_STEPS
            for group in optimizer.param_groups:
                group["lr"] = config["peak_lr"] * multiplier
            indices = torch.randint(len(train_dataset), (batch_size,), generator=generator).tolist()
            boards, target_moves, target_probabilities, values, history, _ = (tensor.to(device, non_blocking=True) for tensor in train_dataset[indices])
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, value_logits, _ = compiled_model(boards, history)
                loss = policy_loss(logits, target_moves, target_probabilities) + VALUE_LOSS_WEIGHT * value_loss(value_logits, values)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.detach()
            steps += 1
            local_steps += 1
            positions += batch_size

        #the change this worker made, as old minus new (a gradient's sign), in bf16 to halve the files
        with torch.no_grad():
            current = model.state_dict()
            delta = {key: (start_weights[key] - current[key]).to(torch.bfloat16).cpu() for key in start_weights if start_weights[key].is_floating_point()}
        seconds = time.time() - round_start
        delta_dir = run_dir / "deltas" / round_name(round_index)
        if latest_round(run_dir) == round_index:
            save_atomically(delta, delta_dir / f"{name}.pt")
            save_atomically({"worker": name, "positions": positions, "steps": steps, "train_loss": (loss_total / max(steps, 1)).item(),
                             "positions_per_second": positions / seconds, "gpu": properties.name}, delta_dir / f"{name}.json")
        if round_index % STATE_EVERY_ROUNDS == 0:
            save_atomically({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "local_steps": local_steps}, state_path)
        print(f"{time.strftime('%H:%M:%S')} {name} round {round_index}: {steps} steps, {positions / seconds:,.0f} positions/s, "
              f"loss {(loss_total / max(steps, 1)).item():.4f}", flush=True)
        while latest_round(run_dir) == round_index:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="many preemptible GPUs training one model by averaging their changes every round")
    parser.add_argument("role", choices=["init", "merger", "worker"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, help="for init: the plan, see the configs next to this file")
    parser.add_argument("--worker-id", type=int, help="for worker: a number that stays the same when the job is requeued")
    arguments = parser.parse_args()
    if arguments.role == "init":
        init(arguments.run_dir, arguments.config)
    elif arguments.role == "merger":
        merger(arguments.run_dir)
    else:
        worker(arguments.run_dir, arguments.worker_id)
