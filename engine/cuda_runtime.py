#cuda runtime calls on torch's own library through ctypes, skipping torch's per-call work; Searches' round loop calls the same addresses
import ctypes
import functools
import os
import sys

import torch

HOST_TO_DEVICE, DEVICE_TO_HOST = 1, 2


class CudaError(RuntimeError):
    pass


class CudaRuntime:
    #the waits release the gil (CDLL), the other calls return at once and keep it (PyDLL)
    CALLS = {
        "cudaSetDevice": [ctypes.c_int],
        "cudaMemcpyAsync": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
        "cudaGraphLaunch": [ctypes.c_void_p, ctypes.c_void_p],
        "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
        "cudaEventSynchronize": [ctypes.c_void_p],
        "cudaStreamSynchronize": [ctypes.c_void_p],
    }
    WAITS = ("cudaEventSynchronize", "cudaStreamSynchronize")

    def __init__(self):
        torch.cuda.init()
        major = torch.version.cuda.split(".")[0]
        #torch's loaded copy, never a second runtime (on linux, RTLD_NOLOAD finds none if torch has none)
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
        self.event_record = calls["cudaEventRecord"]
        self.event_synchronize, self.stream_synchronize = calls["cudaEventSynchronize"], calls["cudaStreamSynchronize"]

    def check(self, error: int, what: str) -> None:
        if error:
            raise CudaError(f"{what}: cuda error {error}, {self.library.cudaGetErrorString(error).decode()}")

    def addresses(self) -> dict[str, int]:
        return {name: ctypes.cast(getattr(self.library, name), ctypes.c_void_p).value for name in [*self.CALLS, "cudaGetErrorString"]}


@functools.cache
def cuda_runtime() -> CudaRuntime:
    return CudaRuntime()
