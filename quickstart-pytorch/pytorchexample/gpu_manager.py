import json
import os
import fcntl

import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)

LOCK_FILE = "/tmp/fl_gpu_lock.json"


class GPUManager:
    def __init__(
        self,
        min_free_gb=10,
    ):
        self.min_free_gb = min_free_gb

        # optional manual override
        self.forced_gpu = os.getenv("FORCE_GPU")
        self.assigned_gpu = None
        nvmlInit()

        if not os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, "w") as f:
                json.dump({}, f)

    def _read_state(self):

        with open(LOCK_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        return data

    def _write_state(self, state):

        with open(LOCK_FILE, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            try:
                json.dump(state, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_best_gpu(self):
        # manual override
        if self.forced_gpu is not None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")

            print(f"[GPUManager] Forced physical GPU={self.forced_gpu}")
            print(f"[GPUManager] CUDA_VISIBLE_DEVICES={visible}")

            if visible is not None:
                visible_list = [x.strip() for x in visible.split(",")]

                if self.forced_gpu in visible_list:
                    local_id = visible_list.index(self.forced_gpu)

                    print(
                        f"[GPUManager] Physical GPU {self.forced_gpu} "
                        f"mapped to local cuda:{local_id}"
                    )

                    return torch.device(f"cuda:{local_id}")

                else:
                    raise RuntimeError(
                        f"Physical GPU {self.forced_gpu} "
                        f"is not visible to this Ray actor.\n"
                        f"Visible GPUs = {visible}"
                    )

            return torch.device(f"cuda:{self.forced_gpu}")

        state = self._read_state()
        best_gpu = None
        best_score = -1
        physical_count = nvmlDeviceGetCount()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        if visible:
            gpu_indices = [int(x) for x in visible.split(",")]
        else:
            gpu_indices = list(range(physical_count))

        for i in gpu_indices:
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / 1024**3
            num_clients = state.get(str(i), 0)
            score = free_gb - 5 * num_clients

            print(f"GPU{i}: free={free_gb:.1f}GB clients={num_clients}")

            if free_gb > self.min_free_gb and score > best_score:
                best_score = score
                best_gpu = i

        if best_gpu is None:
            best_gpu = max(
                gpu_indices,
                key=lambda i: (
                    nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
                ),
            )

        state[str(best_gpu)] = state.get(str(best_gpu), 0) + 1
        self._write_state(state)
        self.assigned_gpu = best_gpu
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        if visible:
            visible_list = [int(x) for x in visible.split(",")]

            if best_gpu in visible_list:
                local_gpu = visible_list.index(best_gpu)

                print(f"[GPUManager] Physical GPU {best_gpu} -> local cuda:{local_gpu}")

                return torch.device(f"cuda:{local_gpu}")

            else:
                print(
                    f"[GPUManager] GPU {best_gpu} not visible "
                    f"inside this actor. Using cuda:0"
                )

                return torch.device("cuda:0")

        return torch.device(f"cuda:{best_gpu}")

    def release_gpu(self):

        if self.assigned_gpu is None:
            return

        gpu_id = str(self.assigned_gpu)
        state = self._read_state()

        if gpu_id in state:
            state[gpu_id] = max(0, state[gpu_id] - 1)

            if state[gpu_id] == 0:
                del state[gpu_id]

        self._write_state(state)
        self.assigned_gpu = None
