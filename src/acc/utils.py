import json
import time
from pathlib import Path

import numpy as np
import torch


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Throughput:
    """Accumulates tokens / bytes over wall time for the reporting rules:
    fidelity numbers must ship with a throughput number."""

    def __init__(self):
        self.tokens = 0
        self.disk_bytes = 0
        self.seconds = 0.0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self, tokens=0, disk_bytes=0):
        self.seconds += time.perf_counter() - self._t0
        self.tokens += tokens
        self.disk_bytes += disk_bytes
        self._t0 = None

    def summary(self):
        s = max(self.seconds, 1e-9)
        return {
            "tokens": self.tokens,
            "disk_bytes": self.disk_bytes,
            "seconds": self.seconds,
            "tokens_per_s": self.tokens / s,
            "disk_MB_per_s": self.disk_bytes / s / 1e6,
        }
