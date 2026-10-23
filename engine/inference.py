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

#the L2 a position's activations take between a block's kernels, per unit of the model's width: a d768's 64 positions a
#lane fill the 5070 Ti's 48 MB with two lanes at once (forwards of 128 there cost two-lane batches 4-9%)
L2_BYTES_PER_POSITION_WIDTH = 512
#smaller forwards leave the matmuls too few tiles for a big gpu
MIN_CHUNK_POSITIONS = 64

#boards a forward takes at once, a bigger batch's graph running several: as many as fit the gpu's L2 with every lane's
#forward at once
def chunk_positions(width: int, lanes: int) -> int:
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(MIN_CHUNK_POSITIONS, l2_bytes // (lanes * width * L2_BYTES_PER_POSITION_WIDTH) // 16 * 16)

#answer_rows for every board, chunk boards at a time (None: all at once)
def chunked_answer_rows(model: nn.Module, boards: torch.Tensor, chunk: int | None) -> torch.Tensor:
    rows = [answer_rows(model(part)) for part in boards.split(chunk or len(boards))]
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
#rather than 4 cost it 3% of its nodes/s. Past a chunk (chunked_answer_rows), whole chunks and the last one in the same
#steps, so that no batch pads into a forward of its own (a graph of 133 ran 128 boards as forwards of 128 and 5)
def small_graph_sizes(max_batch: int, chunk: int | None = None) -> list[int]:
    chunk = min(chunk or max_batch, max_batch)
    steps = [1]
    while steps[-1] < chunk:
        steps.append(min(chunk, steps[-1] * 2 if steps[-1] < 16 else steps[-1] + max(4, steps[-1] // 16)))
    sizes = [whole + step for whole in range(0, max_batch, chunk) for step in steps]
    return sizes[:next(index for index, size in enumerate(sizes) if size >= max_batch) + 1]

#every 64 up to 1024, then every 128, then every 256: padding wastes a few percent at most
def graph_sizes(max_batch: int) -> list[int]:
    sizes = [64]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + (64 if sizes[-1] < 1024 else 128 if sizes[-1] < 2048 else 256))
    return sizes

#a batch in flight's own stream, cuda graphs (one for every size in sizes, sharing a memory pool, each running its boards
#chunk at a time) and buffers
class Lane:
    def __init__(self, model: nn.Module, sizes: list[int], chunk: int | None):
        self.stream = torch.cuda.Stream()
        #launched and not yet collected: another launch would write over its answers
        self.pending = False
        largest = sizes[-1]
        #recorded on real boards: zero boards once hit a device-side assert
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda").repeat(largest, 1)
        self.boards_host = torch.zeros(largest, TOKENS_PER_BOARD, dtype=torch.long, pin_memory=True)
        self.answers_host = torch.zeros(largest, ANSWER_WIDTH, dtype=torch.float32, pin_memory=True)
        #recorded after each batch for GraphedBatches' blocking_wait
        self.done = torch.cuda.Event(blocking=True)

        #warmup calls off the default stream, as recording a graph needs
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            for size in sizes:
                chunked_answer_rows(model, self.boards[:size], chunk)
        torch.cuda.current_stream().wait_stream(self.stream)

        #recorded on their own stream: lanes recorded on one stream into one pool wrote over each other's activations
        pool = torch.cuda.graph_pool_handle()
        self.graphs, self.answers = {}, {}
        for size in reversed(sizes):
            self.graphs[size] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[size], pool=pool, stream=self.stream), torch.no_grad():
                self.answers[size] = chunked_answer_rows(model, self.boards[:size], chunk)

#the model as cuda graphs in in_flight lanes that run their batches at once, a batch through the smallest graph that
#holds it on the first lane free. launch returns at once and collect waits, in any order: at most in_flight batches out.
#chunk: the boards a forward takes (chunk_positions), None for a batch's all at once.
#collect's copy spins a core until the batch is done; blocking_wait sleeps until then instead, a little slower to wake
class GraphedBatches:
    def __init__(self, model: nn.Module, sizes: list[int], in_flight: int = 2, chunk: int | None = None,
                 blocking_wait: bool = False):
        #the graphs read the model's weights where they were when recorded: freed, their memory went to the next
        #tensors made and every answer came back wrong
        self.model = model
        self.sizes = sizes
        self.lanes = [Lane(model, sizes, chunk) for _ in range(in_flight)]
        self.blocking_wait = blocking_wait

    def launch(self, boards: np.ndarray) -> tuple[Lane, int, int]:
        count = len(boards)
        size = next(size for size in self.sizes if size >= count)
        lane = next((lane for lane in self.lanes if not lane.pending), None)
        if lane is None:
            raise RuntimeError("more batches in flight than lanes")
        lane.pending = True
        lane.boards_host[:count] = torch.from_numpy(boards)
        with torch.cuda.stream(lane.stream):
            lane.boards[:count].copy_(lane.boards_host[:count], non_blocking=True)
            lane.graphs[size].replay()
            if self.blocking_wait:
                lane.done.record(lane.stream)
        return lane, size, count

    #copied here, not at launch: the gpu's copies queue, and a copy at launch held the other lane's boards behind it
    def collect(self, launched: tuple[Lane, int, int]) -> np.ndarray:
        lane, size, count = launched
        if self.blocking_wait:
            lane.done.synchronize()
        with torch.cuda.stream(lane.stream):
            lane.answers_host[:count].copy_(lane.answers[size][:count])
        lane.pending = False
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

#compiled for exactly exact_size boards, the size most batches fill, dynamically for the rest (all exact: slow to start)
class ExactSizeCompiled(nn.Module):
    def __init__(self, model: nn.Module, exact_size: int):
        super().__init__()
        self.exact_size = exact_size
        self.exact, self.dynamic = torch.compile(model, dynamic=False), torch.compile(model, dynamic=True)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.exact if len(boards) == self.exact_size else self.dynamic)(boards)

#fp16_model with every block but the last in int8 (model/int8_blocks.py), its weights rounded once per checkpoint
#(engine/quantization.py): needs an sm_80 gpu or newer
def int8_model(checkpoint_path: Path) -> nn.Module:
    from model.int8_blocks import quantize_blocks
    from engine.quantization import int8_weights
    weights = int8_weights(checkpoint_path)
    return quantize_blocks(fp16_model(checkpoint_path), weights, FP16_RESIDUAL_SCALE)

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
        #serve keeps two batches out
        chunk = chunk_positions(model.d_model, lanes=2)
        if precision == "bf16":
            model = Bf16(model)
        if compile_model:
            #most of a big batch's forwards are whole chunks
            model = ExactSizeCompiled(model, chunk)
        network = GraphedBatches(model, graph_sizes(len(slot_names) * games_per_process), in_flight=2, chunk=chunk)
    else:
        network = cpu_batches(load_model(Path(checkpoint_path), CPU))
    ready.set()
    serve(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays], connections, stop, served_calls, served_positions)
