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
from model.checkpoint import load_model
from engine.cuda_runtime import DEVICE_TO_HOST, HOST_TO_DEVICE, cuda_runtime
from selfplay_rs import LEAN_COLUMNS, LEAN_ROW

#a row of ANSWER_WIDTH numbers per board: the move logits, then the expected score (the fused model returns the rows itself)
def answer_rows(outputs: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    move_logits, value_logits = outputs
    return torch.cat([move_logits.float(), expected_score(value_logits).unsqueeze(1)], dim=1)

#the L2 a position's activations take between a block's kernels, per unit of the model's width
L2_BYTES_PER_POSITION_WIDTH = 1024
MIN_CHUNK_POSITIONS = 64

def chunk_positions(width: int, lanes: int) -> int:
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(MIN_CHUNK_POSITIONS, l2_bytes // (lanes * width * L2_BYTES_PER_POSITION_WIDTH) // 16 * 16)

def chunked_answer_rows(model: nn.Module, boards: torch.Tensor, chunk: int | None) -> torch.Tensor:
    rows = [answer_rows(model(part)) for part in boards.split(chunk or len(boards))]
    return rows[0] if len(rows) == 1 else torch.cat(rows)


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

#sizes from 1 to max_batch: doubling to 16, then sixteenth steps; past a chunk, whole chunks plus the same steps, so no batch pads into a forward of its own
def small_graph_sizes(max_batch: int, chunk: int | None = None) -> list[int]:
    chunk = min(chunk or max_batch, max_batch)
    steps = [1]
    while steps[-1] < chunk:
        steps.append(min(chunk, steps[-1] * 2 if steps[-1] < 16 else steps[-1] + max(4, steps[-1] // 16)))
    sizes = [whole + step for whole in range(0, max_batch, chunk) for step in steps]
    return sizes[:next(index for index, size in enumerate(sizes) if size >= max_batch) + 1]

def graph_sizes(max_batch: int) -> list[int]:
    sizes = [64]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + (64 if sizes[-1] < 1024 else 128 if sizes[-1] < 2048 else 256))
    return sizes

#a workspace cublas makes while a graph records lands in that graph's pool and breaks later calls (cuda error 700)
def clear_cublas_workspaces() -> None:
    #a private torch function, absent from some versions
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()

#owns everything its raw handles() point at, so nothing is freed or recorded again while it lives. warm_up: the model run once per size
#first, building every kernel (a network's later lanes skip it)
class Lane:
    def __init__(self, model: nn.Module, sizes: list[int], chunk: int | None, lean: bool = False, warm_up: bool = True):
        self.stream = torch.cuda.Stream()
        self.device = torch.cuda.current_device()
        #launched and not yet collected: another launch would write over its answers
        self.pending = False
        self.lean = lean
        largest = sizes[-1]
        #recorded on real boards: zero boards hit a device-side assert
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda").repeat(largest, 1)
        if lean:
            #a board is a row of bytes: its tokens, then the int16 indices of the answer columns to gather, so the answer has LEAN_COLUMNS
            #floats rather than ANSWER_WIDTH (padding gathers the expected score's)
            self.rows = torch.zeros(largest, LEAN_ROW, dtype=torch.uint8, device="cuda")
            self.rows[:, :TOKENS_PER_BOARD] = self.boards
            self.rows[:, TOKENS_PER_BOARD:].view(torch.int16).fill_(ANSWER_WIDTH - 1)
            self.rows_host = torch.zeros(largest, LEAN_ROW, dtype=torch.uint8, pin_memory=True)
            self.rows_array = self.rows_host.numpy()
        else:
            self.boards_host = torch.zeros(largest, TOKENS_PER_BOARD, dtype=torch.long, pin_memory=True)
            self.boards_array = self.boards_host.numpy()
        self.answers_host = torch.zeros(largest, LEAN_COLUMNS if lean else ANSWER_WIDTH, dtype=torch.float32, pin_memory=True)
        self.answers_array = self.answers_host.numpy()
        self.done = torch.cuda.Event(blocking=True)

        #recording a graph needs warm-up calls off the default stream
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            for size in sizes if warm_up else []:
                self.answer(model, size, chunk)
        torch.cuda.current_stream().wait_stream(self.stream)

        #a pool a lane: lanes recording into one pool wrote over each other's activations
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.answers = {}, {}
        for size in reversed(sizes):
            self.graphs[size] = torch.cuda.CUDAGraph()
            clear_cublas_workspaces()
            with torch.cuda.graph(self.graphs[size], pool=self.pool, stream=self.stream), torch.no_grad():
                self.answers[size] = self.answer(model, size, chunk)
            clear_cublas_workspaces()
        #the event's handle exists from its first record
        self.done.record(self.stream)
        self.done.synchronize()
        #a lean collect copies the answers' bytes as they are
        assert not lean or all(answers.dtype == torch.float32 and answers.is_contiguous() for answers in self.answers.values())

    def answer(self, model: nn.Module, size: int, chunk: int | None) -> torch.Tensor:
        if not self.lean:
            return chunked_answer_rows(model, self.boards[:size], chunk)
        self.boards[:size].copy_(self.rows[:size, :TOKENS_PER_BOARD])
        rows = chunked_answer_rows(model, self.boards[:size], chunk)
        return rows.gather(1, self.rows[:size, TOKENS_PER_BOARD:].view(torch.int16).long())

    #the raw handles launch and collect pass to the cuda runtime; graphs maps a size to (its exec, its answers on the device)
    def handles(self) -> dict:
        return {"device": self.device, "stream": self.stream.cuda_stream, "done": self.done.cuda_event,
                "rows_host": self.rows_host.data_ptr(), "rows": self.rows.data_ptr(), "answers_host": self.answers_host.data_ptr(),
                "graphs": {size: (graph.raw_cuda_graph_exec(), self.answers[size].data_ptr()) for size, graph in self.graphs.items()}}

#the model as cuda graphs in in_flight lanes: launch returns at once and collect waits, in any order, at most in_flight batches out. A batch
#takes the smallest graph that holds it, on the first free lane. chunk: boards per forward (chunk_positions; None: a batch's all at once).
#blocking_wait: collect sleeps until the batch is done instead of spinning a core. lean: lean lanes, which take launch_filled's boards only
class GraphedBatches:
    def __init__(self, model: nn.Module, sizes: list[int], in_flight: int = 2, chunk: int | None = None,
                 blocking_wait: bool = False, lean: bool = False):
        #kept alive: the graphs read its weights where they were recorded
        self.model = model
        self.sizes = sizes
        self.lanes = [Lane(model, sizes, chunk, lean, warm_up=number == 0) for number in range(in_flight)]
        self.lean = lean
        self.blocking_wait = blocking_wait
        self.runtime = cuda_runtime() if lean else None
        self.handles = {id(lane): lane.handles() for lane in self.lanes} if lean else {}

    def launch(self, boards: np.ndarray) -> tuple[Lane, int, int]:
        if self.lean:
            raise RuntimeError("lean lanes answer rows with their columns (launch_filled), not boards")
        lane = self.free_lane()
        lane.boards_host[:len(boards)] = torch.from_numpy(boards)
        return self.launch_filled(lane, len(boards))

    #the lane the next launch takes; a caller may fill its boards_array (a lean lane's rows_array) for launch_filled
    def free_lane(self) -> Lane:
        lane = next((lane for lane in self.lanes if not lane.pending), None)
        if lane is None:
            raise RuntimeError("more batches in flight than lanes")
        return lane

    #a lean lane's calls leave the thread's current device (torch's too) at the lane's
    def launch_filled(self, lane: Lane, count: int) -> tuple[Lane, int, int]:
        size = next(size for size in self.sizes if size >= count)
        if lane.lean:
            runtime, handles = self.runtime, self.handles[id(lane)]
            stream = handles["stream"]
            runtime.check(runtime.set_device(handles["device"]), "the lane's device")
            runtime.check(runtime.memcpy(handles["rows"], handles["rows_host"], count * LEAN_ROW, HOST_TO_DEVICE, stream), "the boards' copy")
            lane.pending = True
            runtime.check(runtime.graph_launch(handles["graphs"][size][0], stream), "the graph's launch")
            runtime.check(runtime.event_record(handles["done"], stream), "the batch's event")
            return lane, size, count
        lane.pending = True
        with torch.cuda.stream(lane.stream):
            lane.boards[:count].copy_(lane.boards_host[:count], non_blocking=True)
            lane.graphs[size].replay()
            if self.blocking_wait:
                lane.done.record(lane.stream)
        return lane, size, count

    #copied here, not at launch: the gpu's copies queue, so a copy at launch holds the other lane's boards behind it
    def collect(self, launched: tuple[Lane, int, int]) -> np.ndarray:
        lane, size, count = launched
        if lane.lean:
            runtime, handles = self.runtime, self.handles[id(lane)]
            stream = handles["stream"]
            runtime.check(runtime.set_device(handles["device"]), "the lane's device")
            if self.blocking_wait:
                runtime.check(runtime.event_synchronize(handles["done"]), "the batch")
            answers = handles["graphs"][size][1]
            runtime.check(runtime.memcpy(handles["answers_host"], answers, count * LEAN_COLUMNS * 4, DEVICE_TO_HOST, stream), "the answers' copy")
            runtime.check(runtime.stream_synchronize(stream), "the batch")
            lane.pending = False
            return lane.answers_array[:count]
        if self.blocking_wait:
            lane.done.synchronize()
        with torch.cuda.stream(lane.stream):
            lane.answers_host[:count].copy_(lane.answers[size][:count])
        lane.pending = False
        return lane.answers_host[:count].numpy()

    #Searches' round loop holds this object, so every handle's lane lives while it runs
    def rust_network(self) -> dict:
        if not self.lean:
            raise ValueError("Searches' round loop needs lean lanes")
        return {"blocking_wait": self.blocking_wait, "lanes": [self.handles[id(lane)] for lane in self.lanes]}

    def settle(self) -> None:
        for lane in self.lanes:
            if lane.pending:
                lane.stream.synchronize()
                lane.pending = False

#GraphedBatches' interface without a gpu: launch does the work
class ImmediateBatches:
    def __init__(self, evaluate_boards: Callable[[np.ndarray], np.ndarray]):
        self.evaluate_boards = evaluate_boards

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return self.evaluate_boards(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return launched

#scales the residual stream down so its channels stay below fp16's largest number, 65,504
FP16_RESIDUAL_SCALE = 1 / 16

#the network in fp16 with fp16 accumulation, the heads in fp32. It sets torch's matmul flags for the whole process
def fp16_model(checkpoint_path: Path) -> nn.Module:
    #imported here: it needs Triton, which the cpu-only paths do not
    from model.fused_blocks import FusedTransformer
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    #the heads' fp32 matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(checkpoint_path, torch.device("cuda")).scale_residual_stream(FP16_RESIDUAL_SCALE)
    return FusedTransformer(model.half_except_heads(torch.float16))

#compiled for exactly exact_size boards (the size most batches fill) and dynamically for the rest
class ExactSizeCompiled(nn.Module):
    def __init__(self, model: nn.Module, exact_size: int):
        super().__init__()
        self.exact_size = exact_size
        self.exact, self.dynamic = torch.compile(model, dynamic=False), torch.compile(model, dynamic=True)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #floats (the norms' eps) constant from the start, or dynamic traces them as symbols and compiles the model a second time
        #dynamo shares one frame's recompile_limit among all instances and each gpu's network adds 3, so it scales with the gpus
        limit = torch._dynamo.config.recompile_limit * max(1, torch.cuda.device_count())
        with torch._dynamo.config.patch(specialize_float=True, recompile_limit=limit):
            return (self.exact if len(boards) == self.exact_size else self.dynamic)(boards)

#fp16_model with every block in int8 but the last and the fp16_blocks whose int8 costs the policy most. Needs sm_80 or newer
#weights: the rounded weights (engine/quantization.py) if already read, else rounded once per checkpoint
def int8_model(checkpoint_path: Path, fp16_blocks: int, weights: dict | None = None) -> nn.Module:
    from model.int8_blocks import quantize_blocks
    from engine.quantization import int8_weights
    if weights is None:
        weights = int8_weights(checkpoint_path, fp16_blocks)
    return quantize_blocks(fp16_model(checkpoint_path), weights, FP16_RESIDUAL_SCALE)

def cpu_batches(model: nn.Module) -> ImmediateBatches:
    def evaluate_boards(boards: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return answer_rows(model(torch.from_numpy(boards).long())).numpy()
    return ImmediateBatches(evaluate_boards)

#ready is set once the graphs are recorded
def inference_process(checkpoint_path: str, compile_model: bool, games_per_process: int, slot_names: list[tuple[str, str]], connections: list,
                      ready, stop, served_calls, served_positions) -> None:
    #ctrl+c reaches every process in the console, the main process decides what it means
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    #every worker waits on this process, so it runs at normal priority, above theirs
    set_priority(subprocess.NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)
    checkpoint = Path(checkpoint_path)
    arrays = [slot_arrays(boards_name, answers_name, games_per_process) for boards_name, answers_name in slot_names]
    if torch.cuda.is_available():
        #the cpu only copies: torch's threads would spin on every core for large batches
        torch.set_num_threads(1)
        model = fp16_model(checkpoint)
        #serve keeps two batches out
        chunk = chunk_positions(model.d_model, lanes=2)
        if compile_model:
            #most of a big batch's forwards are whole chunks
            model = ExactSizeCompiled(model, chunk)
        network = GraphedBatches(model, graph_sizes(len(slot_names) * games_per_process), in_flight=2, chunk=chunk)
    else:
        network = cpu_batches(load_model(checkpoint, torch.device("cpu")))
    ready.set()
    serve(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays], connections, stop, served_calls, served_positions)
