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
from encoding import TOKENS_PER_BOARD, encode_board
from model import expected_score
from selfplay import ANSWER_WIDTH, set_priority, slot_arrays
from tools.play.player import load_model

#a row of ANSWER_WIDTH numbers per board: the move logits, then the expected score
def answer_rows(outputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    move_logits, value_logits = outputs
    return torch.cat([move_logits.float(), expected_score(value_logits).unsqueeze(1)], dim=1)

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

#every 64 up to 1024, then every 128, then every 256: padding wastes a few percent at most
def graph_sizes(max_batch: int) -> list[int]:
    sizes = [64]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + (64 if sizes[-1] < 1024 else 128 if sizes[-1] < 2048 else 256))
    return sizes

#a cuda graph for every batch size in sizes (ascending), sharing one memory pool; a batch runs through the smallest that
#holds it. launch returns at once and collect waits. Two pinned buffers each way let the next batch fill while the last
#one runs, so at most two batches may be launched and not yet collected
class GraphedBatches:
    def __init__(self, model: nn.Module, sizes: list[int]):
        self.sizes = sizes
        largest = self.sizes[-1]
        #recorded on real boards: zero boards once hit a device-side assert
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda").repeat(largest, 1)
        self.boards_host = [torch.zeros(largest, TOKENS_PER_BOARD, dtype=torch.long, pin_memory=True) for _ in range(2)]
        self.answers_host = [torch.zeros(largest, ANSWER_WIDTH, dtype=torch.float32, pin_memory=True) for _ in range(2)]
        self.turn = 0

        #warmup calls on a side stream, as recording a graph needs
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream), torch.no_grad():
            for size in self.sizes:
                model(self.boards[:size])
        torch.cuda.current_stream().wait_stream(side_stream)

        #one pool is safe: the graphs run one after another on one stream, and answers are copied out before the next
        pool = torch.cuda.graph_pool_handle()
        self.graphs, self.answers = {}, {}
        for size in reversed(self.sizes):
            self.graphs[size] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[size], pool=pool), torch.no_grad():
                self.answers[size] = answer_rows(model(self.boards[:size]))

    def launch(self, boards: np.ndarray) -> tuple[torch.cuda.Event, torch.Tensor]:
        count = len(boards)
        size = next(size for size in self.sizes if size >= count)
        #the buffers of the batch before last, whose answers serve has already handed out
        boards_host, answers_host = self.boards_host[self.turn], self.answers_host[self.turn]
        self.turn = 1 - self.turn
        boards_host[:count] = torch.from_numpy(boards)
        self.boards[:count].copy_(boards_host[:count], non_blocking=True)
        self.graphs[size].replay()
        answers_host[:count].copy_(self.answers[size][:count], non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return done, answers_host[:count]

    def collect(self, launched: tuple[torch.cuda.Event, torch.Tensor]) -> np.ndarray:
        done, answers = launched
        done.synchronize()
        return answers.numpy()

#GraphedBatches' interface without a gpu (and in the tests): launch does the work
class ImmediateBatches:
    def __init__(self, evaluate_boards: Callable[[np.ndarray], np.ndarray]):
        self.evaluate_boards = evaluate_boards

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return self.evaluate_boards(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return launched

#bf16 autocast, as main.py trains: targets as good as tf32's, and ~150k positions/s instead of 110k (d128, compiled)
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
        model = load_model(Path(checkpoint_path), torch.device("cuda"))
        if precision == "bf16":
            model = Bf16(model)
        if compile_model:
            model = torch.compile(model, dynamic=True)
        network = GraphedBatches(model, graph_sizes(len(slot_names) * games_per_process))
    else:
        network = cpu_batches(load_model(Path(checkpoint_path), CPU))
    ready.set()
    serve(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays], connections, stop, served_calls, served_positions)
