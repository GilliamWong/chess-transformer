#the inference process of selfplay.py: the only process on the gpu, running the network for every self play worker at
#once. A file of its own because it imports torch, which the workers do not (they import selfplay.py)
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
from selfplay import ANSWER_WIDTH, set_priority, slot_arrays
from tools.play.player import load_model

CPU = torch.device("cpu")


#answers the requests of the workers, until stop is set: every batch holds the boards of all the workers waiting at the
#moment. While one batch is on the gpu, the next is gathered and started and the last one's answers handed out, so the
#python here overlaps the gpu instead of adding to it (a worker has one request at a time, so the two never share one).
#served counts the batches and the positions
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

#the batch sizes the gpu has a cuda graph for: every 64 up to 1024, then every 128, then every 256, so padding a batch up
#to the next size wastes a few percent at most (doubling sizes wasted up to half)
def graph_sizes(max_batch: int) -> list[int]:
    sizes = [64]
    while sizes[-1] < max_batch:
        sizes.append(sizes[-1] + (64 if sizes[-1] < 1024 else 128 if sizes[-1] < 2048 else 256))
    return sizes

#the network on the gpu for batches of any size up to max_batch: a cuda graph recorded for every one of graph_sizes, all in
#one memory pool, and a batch runs through the smallest that holds it (the rows past it are computed and dropped). launch
#starts a batch and returns at once, collect waits for its answers. The boards go in and the answers come out through
#pinned memory, two buffers each way, so the next batch can be filled in while the last one is still on the gpu
class GraphedBatches:
    def __init__(self, model: nn.Module, max_batch: int):
        self.sizes = graph_sizes(max_batch)
        largest = self.sizes[-1]
        #real boards (the start position): the graphs are recorded on them and the rows past a batch keep them
        self.boards = torch.tensor([encode_board(chess.Board())], dtype=torch.long, device="cuda").repeat(largest, 1)
        self.boards_host = [torch.zeros(largest, TOKENS_PER_BOARD, dtype=torch.long, pin_memory=True) for _ in range(2)]
        self.answers_host = [torch.zeros(largest, ANSWER_WIDTH, dtype=torch.float32, pin_memory=True) for _ in range(2)]
        self.turn = 0

        #a few calls on a side stream first, as recording a graph needs
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream), torch.no_grad():
            for size in self.sizes:
                model(self.boards[:size])
        torch.cuda.current_stream().wait_stream(side_stream)

        #the graphs share one pool: they run one after another on one stream, and a batch's answers are copied out before
        #the next batch runs
        pool = torch.cuda.graph_pool_handle()
        self.graphs, self.answers = {}, {}
        for size in reversed(self.sizes):
            self.graphs[size] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[size], pool=pool), torch.no_grad():
                self.answers[size] = torch.cat(model(self.boards[:size]), dim=1).float()

    def launch(self, boards: np.ndarray) -> tuple[torch.cuda.Event, torch.Tensor]:
        count = len(boards)
        size = next(size for size in self.sizes if size >= count)
        #the buffers of the batch before last: serve has handed its answers out before starting this one
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

#the network evaluated at once where there is no gpu (and in the tests): launch does the work, collect hands it over
class ImmediateBatches:
    def __init__(self, evaluate_boards: Callable[[np.ndarray], np.ndarray]):
        self.evaluate_boards = evaluate_boards

    def launch(self, boards: np.ndarray) -> np.ndarray:
        return self.evaluate_boards(boards)

    def collect(self, launched: np.ndarray) -> np.ndarray:
        return launched

#the model under bf16 autocast, the precision main.py trains it in. Measured on 394 self play positions against an 800
#simulation search: the targets are as good as with tensor float 32 (visit shares 0.236 from the reference's against
#0.239, the most visited move the reference's 74.4% against 72.8%), and compiled the gpu does about 150,000 positions a
#second instead of 110,000 (d128)
class Bf16(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(boards)

#the network on the cpu, without graphs
def cpu_batches(model: nn.Module) -> ImmediateBatches:
    def evaluate_boards(boards: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.cat(model(torch.from_numpy(boards).long()), dim=1).float().numpy()
    return ImmediateBatches(evaluate_boards)

#the inference process: the only one on the gpu, so there is one cuda context instead of one per worker, and batches of
#all the waiting workers' games at once. ready is set once the graphs are recorded
def inference_process(checkpoint_path: str, compile_model: bool, precision: str, games_per_process: int, slot_names: list[tuple[str, str]], connections: list,
                      ready, stop, served_calls, served_positions) -> None:
    #ctrl+c reaches every process in the console: the main process decides what it means (see main)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    #every worker waits on this process, so it is the one that must not wait for a core: normal priority, above theirs.
    #It mostly waits for the gpu, so it takes little from the rest of the computer
    set_priority(subprocess.NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)
    #tensor float 32 for the matrix products: a fifth faster on the gpu, and the priors within 0.001 of full precision
    torch.backends.cuda.matmul.allow_tf32 = True
    arrays = [slot_arrays(boards_name, answers_name, games_per_process) for boards_name, answers_name in slot_names]
    if torch.cuda.is_available():
        #the cpu only copies boards in and answers out. With torch's threads a batch of more than 32,768 tokens is copied on
        #every core, whose threads then spin waiting for the next: 60% of a 16 core cpu at 4 rust workers, 11% with one
        torch.set_num_threads(1)
        model = load_model(Path(checkpoint_path), torch.device("cuda"))
        if precision == "bf16":
            model = Bf16(model)
        if compile_model:
            #one compile for every batch size, then recorded into the graphs
            model = torch.compile(model, dynamic=True)
        network = GraphedBatches(model, len(slot_names) * games_per_process)
    else:
        network = cpu_batches(load_model(Path(checkpoint_path), CPU))
    ready.set()
    serve(network, [boards for _, boards, _ in arrays], [answers for _, _, answers in arrays], connections, stop, served_calls, served_positions)
