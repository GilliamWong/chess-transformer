#starts tensorboard for the training runs and opens it in the browser
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from engine.checkpoint import RUNS_DIR

TENSORBOARD_PORT = 6006

def tensorboard_is_running() -> bool:
    with socket.socket() as connection:
        return connection.connect_ex(("localhost", TENSORBOARD_PORT)) == 0

#a separate process, so it keeps serving after training has finished
def start_tensorboard() -> None:
    if not tensorboard_is_running():
        RUNS_DIR.mkdir(exist_ok=True)
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(RUNS_DIR), "--port", str(TENSORBOARD_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def open_when_ready() -> None:
        for _ in range(60):
            if tensorboard_is_running():
                webbrowser.open(f"http://localhost:{TENSORBOARD_PORT}")
                return
            time.sleep(0.5)

    #tensorboard takes a few seconds to come up, and training should not wait for it
    threading.Thread(target=open_when_ready, daemon=True).start()
