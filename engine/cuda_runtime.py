#the cuda runtime calls a lean lane makes (inference.Lane), straight from the library torch loaded through ctypes: none of
#torch's per-call work (its stream context, views and dispatch) around a copy or a graph's launch. The same calls'
#addresses are what Searches' own round loop calls
import ctypes
import os
import sys

import torch

HOST_TO_DEVICE, DEVICE_TO_HOST = 1, 2


class CudaError(RuntimeError):
    pass


class CudaRuntime:
    #the calls, by name, with their argument types; the waits release the gil, the others return at once and keep it
    CALLS = {
        "cudaSetDevice": [ctypes.c_int],
        "cudaMemcpyAsync": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
        "cudaGraphLaunch": [ctypes.c_void_p, ctypes.c_void_p],
        "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
        "cudaStreamWaitEvent": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint],
        "cudaEventSynchronize": [ctypes.c_void_p],
        "cudaStreamSynchronize": [ctypes.c_void_p],
    }
    WAITS = ("cudaEventSynchronize", "cudaStreamSynchronize")

    def __init__(self):
        torch.cuda.init()
        major = torch.version.cuda.split(".")[0]
        #already in the process: torch's own copy, not another runtime (on linux, none is loaded if torch's isn't)
        name = f"cudart64_{major}.dll" if sys.platform == "win32" else f"libcudart.so.{major}"
        mode = ctypes.DEFAULT_MODE if sys.platform == "win32" else os.RTLD_NOLOAD | os.RTLD_LAZY
        self.library, quick = ctypes.CDLL(name, mode=mode), ctypes.PyDLL(name, mode=mode)
        calls = {}
        for call_name, arguments in self.CALLS.items():
            call = getattr(self.library if call_name in self.WAITS else quick, call_name)
            call.argtypes, call.restype = arguments, ctypes.c_int
            calls[call_name] = call
        self.library.cudaGetErrorString.argtypes, self.library.cudaGetErrorString.restype = [ctypes.c_int], ctypes.c_char_p
        self.set_device, self.memcpy, self.graph_launch = calls["cudaSetDevice"], calls["cudaMemcpyAsync"], calls["cudaGraphLaunch"]
        self.event_record, self.stream_wait_event = calls["cudaEventRecord"], calls["cudaStreamWaitEvent"]
        self.event_synchronize, self.stream_synchronize = calls["cudaEventSynchronize"], calls["cudaStreamSynchronize"]

    #a call's error as an exception, never a hang or a quiet bad answer
    def check(self, error: int, what: str) -> None:
        if error:
            raise CudaError(f"{what}: cuda error {error}, {self.library.cudaGetErrorString(error).decode()}")

    #the calls' addresses, for Rust to make them itself
    def addresses(self) -> dict[str, int]:
        return {name: ctypes.cast(getattr(self.library, name), ctypes.c_void_p).value for name in self.CALLS}


RUNTIME: CudaRuntime | None = None


def cuda_runtime() -> CudaRuntime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = CudaRuntime()
    return RUNTIME
