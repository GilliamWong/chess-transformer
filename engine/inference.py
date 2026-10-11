#selfplay.py's inference process: the only process on the gpu, running the network for every worker at once.
#a file of its own because it imports torch, which the workers must not
from collections.abc import Callable
from multiprocessing import connection
from pathlib import Path
import signal
import subprocess
import sys
import chess
import numpy as np
import torch
from torch import nn
from model.encoding import TOKENS_PER_BOARD, encode_board
from model.model import expected_score
from engine.selfplay import ANSWER_WIDTH, set_priority, slot_arrays
from engine.player import load_model

#a row of ANSWER_WIDTH numbers per board: the move logits, then the expected score
def answer_rows(outputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    move_logits, value_logits = outputs
    return torch.cat([move_logits.float(), expected_score(value_logits).unsqueeze(1)], dim=1)

#boards a forward takes at once, a bigger batch's graph running several: a d768's activations for 128 fit a 48 MB L2
CHUNK_POSITIONS = 128

#answer_rows for every board, CHUNK_POSITIONS boards at a time
def chunked_answer_rows(model: nn.Module, boards: torch.Tensor) -> torch.Tensor:
    rows = [answer_rows(model(chunk)) for chunk in boards.split(CHUNK_POSITIONS)]
    return rows[0] if len(rows) == 1 else torch.cat(rows)

CPU = torch.device("cpu")


#answers the workers' requests until stop is set. While one batch is on the gpu the next is gathered and launched and
#the last one's answers handed out, so the python overlaps the gpu
def serve(network, boards: list[np.ndarray], answers: list[np.ndarray], connections: list, stop, served_calls, served_positions) -> None:
    slot_of = {id(connection_): slot for slot, connection_ in enumerate(connections)}

    def hand_out(requests: list[tuple[int, int]], launched) -> None:
        evaluated = network.collect(launched)
        offset = 0
        for slot, count in requests:
            answers[slot][:count] = evaluated[offset:offset + count]
            offset += count
            connections[slot].send(True)
        served_calls.value += 1
        served_positions.value += offset

    in_flight = None
    while not stop.is_set():
        #with a batch on the gpu, only the requests already waiting: its answers must not wait for more
        waiting = connection.wait(connections, timeout=0 if in_flight else 0.1)
        requests = [(slot_of[id(ready)], ready.recv()) for ready in waiting]
        started = (requests, network.launch(np.concatenate([boards[slot][:count] for slot, count in requests]))) if requests else None
        if in_flight is not None:
            hand_out(*in_flight)
        in_flight = started

#graph sizes for batches from one up to max_batch (one search's rounds, a match's few games): past 16, 4 boards or a
#sixteenth apart. A round of the engine's 64 leaves often sends fewer boards (the cache answers some), and steps of 16
#rather than 4 cost it 3% of its nodes/s
def small_graph_sizes(max_batch: int) -> list[int]:
    sizes = [1, 2, 4, 8, 16]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + max(4, sizes[-1] // 16))
    return sizes

#every 64 up to 1024, then every 128, then every 256: padding wastes a few percent at most
def graph_sizes(max_batch: int) -> list[int]:
    sizes = [64]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + (64 if sizes[-1] < 1024 else 128 if sizes[-1] < 2048 else 256))
    return sizes

#a batch in flight's own stream, cuda graphs (one for every size in sizes, sharing a memory pool) and buffers
class Lane:
    def __init__(self, model: nn.Module, sizes: list[int]):
        self.stream = torch.cuda.Stream()
        largest = sizes[-1]
        #recorded on real boards: zero boards once hit a device-side assert
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda").repeat(largest, 1)
        self.boards_host = torch.zeros(largest, TOKENS_PER_BOARD, dtype=torch.long, pin_memory=True)
        self.answers_host = torch.zeros(largest, ANSWER_WIDTH, dtype=torch.float32, pin_memory=True)

        #warmup calls off the default stream, as recording a graph needs
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            for size in sizes:
                chunked_answer_rows(model, self.boards[:size])
        torch.cuda.current_stream().wait_stream(self.stream)

        #recorded on their own stream: lanes recorded on one stream into one pool wrote over each other's activations
        pool = torch.cuda.graph_pool_handle()
        self.graphs, self.answers = {}, {}
        for size in reversed(sizes):
            self.graphs[size] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[size], pool=pool, stream=self.stream), torch.no_grad():
                self.answers[size] = chunked_answer_rows(model, self.boards[:size])

#the model as cuda graphs in in_flight lanes that take the batches in turn and run them at once, a batch through the
#smallest graph that holds it. launch returns at once and collect waits: at most in_flight batches may be out
class GraphedBatches:
    def __init__(self, model: nn.Module, sizes: list[int], in_flight: int = 2):
        #the graphs read the model's weights where they were when recorded: freed, their memory went to the next
        #tensors made and every answer came back wrong
        self.model = model
        self.sizes = sizes
        self.lanes = [Lane(model, sizes) for _ in range(in_flight)]
        self.turn = 0

    def launch(self, boards: np.ndarray) -> tuple[Lane, int, int]:
        count = len(boards)
        size = next(size for size in self.sizes if size >= count)
        lane = self.lanes[self.turn]
        self.turn = (self.turn + 1) % len(self.lanes)
        lane.boards_host[:count] = torch.from_numpy(boards)
        with torch.cuda.stream(lane.stream):
            lane.boards[:count].copy_(lane.boards_host[:count], non_blocking=True)
            lane.graphs[size].replay()
        return lane, size, count

    #copied here, not at launch: the gpu's copies queue, and a copy at launch held the other lane's boards behind it
    def collect(self, launched: tuple[Lane, int, int]) -> np.ndarray:
        lane, size, count = launched
        with torch.cuda.stream(lane.stream):
            lane.answers_host[:count].copy_(lane.answers[size][:count])
        return lane.answers_host[:count].numpy()

#GraphedBatches' interface without a gpu (and in the tests): launch does the work
class ImmediateBatches:
    def __init__(self, evaluate_boards: Callable[[np.ndarray], np.ndarray]):
        self.evaluate_boards = evaluate_boards

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return self.evaluate_boards(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return launched

#the residual stream shrunk for fp16, whose largest number is 65,504: the d512_selfplay_sf12_10pass run's reaches ~127k
#on one channel and nearly every board came back NaN. 1/4 to 1/16 all match fp32 as well as fp16 can (policy total
#variation 0.0012 over 4,000 positions); 1/16 leaves 8x room
FP16_RESIDUAL_SCALE = 1 / 16

#the network in fp16 with fp16 accumulation, the heads in fp32: on a geforce card faster than bf16 (d512 18.4k
#positions/s at batch 64 against 13.6k; on data centre gpus fp16 accumulation is no faster) and nearer fp32's priors
#(total variation 0.0012 against 0.0049). It sets torch's matmul flags for the whole process
def fp16_model(checkpoint_path: Path) -> nn.Module:
    #imported here: it needs Triton, which the cpu-only paths do not
    from model.fused_blocks import FusedTransformer
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    #the heads' fp32 matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(checkpoint_path, torch.device("cuda")).scale_residual_stream(FP16_RESIDUAL_SCALE)
    return FusedTransformer(model.half_except_heads(torch.float16))

#compiled for exactly exact_size boards, the size most batches fill, and dynamically for the rest (all exact: minutes to start)
class ExactSizeCompiled(nn.Module):
    def __init__(self, model: nn.Module, exact_size: int):
        super().__init__()
        self.exact_size = exact_size
        self.exact, self.dynamic = torch.compile(model, dynamic=False), torch.compile(model, dynamic=True)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.exact if len(boards) == self.exact_size else self.dynamic)(boards)

#bf16 autocast, as train.py trains: targets as good as tf32's, and ~150k positions/s instead of 110k (d128, compiled)
class Bf16(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(boards)

def cpu_batches(model: nn.Module) -> ImmediateBatches:
    def evaluate_boards(boards: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return answer_rows(model(torch.from_numpy(boards).long())).numpy()
    return ImmediateBatches(evaluate_boards)

#ready is set once the graphs are recorded
def inference_process(checkpoint_path: str, compile_model: bool, precision: str, games_per_process: int, slot_names: list[tuple[str, str]], connections: list,
                      ready, stop, served_calls, served_positions) -> None:
    #ctrl+c reaches every process in the console, the main process decides what it means
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    #every worker waits on this process, so it runs at normal priority, above theirs
    set_priority(subprocess.NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)
    #a fifth faster, priors within 0.001 of full precision
    torch.backends.cuda.matmul.allow_tf32 = True
    arrays = [slot_arrays(boards_name, answers_name, games_per_process) for boards_name, answers_name in slot_names]
    if torch.cuda.is_available():
        #the cpu only copies: torch's threads spun on every core for large batches (60% of a 16 core cpu)
        torch.set_num_threads(1)
        if precision == "fp16":
            model = fp16_model(Path(checkpoint_path))
        else:
            model = load_model(Path(checkpoint_path), torch.device("cuda"))
            if precision == "bf16":
                model = Bf16(model)
        if compile_model:
            #most of a big batch's forwards are whole chunks
            model = ExactSizeCompiled(model, CHUNK_POSITIONS)
        network = GraphedBatches(model, graph_sizes(len(slot_names) * games_per_process))
    else:
        network = cpu_batches(load_model(Path(checkpoint_path), CPU))
    ready.set()
    serve(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays], connections, stop, served_calls, served_positions)
