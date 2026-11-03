import time

class Budget:
    def __init__(self, batch_size: int, positions: int):
        self.batch_size = batch_size
        self.positions_budget = positions
        self.max_steps = positions // batch_size

        self.steps = 0
        self.seconds = 0.0 #training only: the first step (torch.compile) and evaluation are not counted
        self._running_since = None

    @property
    def positions(self) -> int:
        return self.steps * self.batch_size

    def _catch_up(self) -> None:
        if self._running_since is not None:
            now = time.perf_counter()
            self.seconds += now - self._running_since
            self._running_since = now

    def step_done(self) -> None:
        self._catch_up()
        self.steps += 1
        if self._running_since is None:
            self._running_since = time.perf_counter()

    def pause(self) -> None:
        self._catch_up()
        self._running_since = None

    def resume(self) -> None:
        self._running_since = time.perf_counter()

    def progress(self) -> float:
        return min(1.0, self.steps / self.max_steps)

    def finished(self) -> bool:
        return self.progress() >= 1.0

    def describe(self) -> str:
        return f"{self.positions_budget / 1e6:g}M"
