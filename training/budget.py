#how long a training run lasts: a number of positions, or minutes of pure training (equal gpu time)
import time

class Budget:
    def __init__(self, batch_size: int, positions: int | None = None, minutes: float | None = None):
        if (positions is None) == (minutes is None):
            raise ValueError("give either positions or minutes")

        self.batch_size = batch_size
        self.positions_budget = positions
        self.max_steps = None if positions is None else positions // batch_size
        self.max_seconds = None if minutes is None else minutes * 60

        self.steps = 0
        self.seconds = 0.0 #training only: compiling (the first step) and evaluating are not counted
        self._running_since = None

    @property
    def timed(self) -> bool:
        return self.max_seconds is not None

    #the length of the progress bar, in steps or in seconds
    @property
    def total(self) -> float:
        return self.max_seconds if self.timed else self.max_steps

    @property
    def positions(self) -> int:
        return self.steps * self.batch_size

    def _catch_up(self) -> None:
        if self._running_since is not None:
            now = time.perf_counter()
            self.seconds += now - self._running_since
            self._running_since = now

    #call after every training step. Returns how far the progress bar moves
    def step_done(self) -> float:
        before = self.seconds
        self._catch_up()
        self.steps += 1
        if self._running_since is None:
            #the clock starts after the first step, where torch.compile compiles
            self._running_since = time.perf_counter()
        return self.seconds - before if self.timed else 1

    #around anything that is not training
    def pause(self) -> None:
        self._catch_up()
        self._running_since = None

    def resume(self) -> None:
        self._running_since = time.perf_counter()

    #0 at the start, 1 when the budget is used up
    def progress(self) -> float:
        used = self.seconds / self.max_seconds if self.timed else self.steps / self.max_steps
        return min(1.0, used)

    def finished(self) -> bool:
        return self.progress() >= 1.0

    def describe(self) -> str:
        return f"{self.max_seconds / 60:g}min" if self.timed else f"{self.positions_budget / 1e6:g}M"
