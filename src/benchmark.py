"""Inference cost of trained models on this device (meant for the Jetson): latency, throughput, GPU use, memory, power.

    python3 run.py --task benchmark --tags final ablation_siso      # everything the paper's figures use
    python3 run.py --task benchmark --runs results/<run> [results/<run> ...]

For each run: rebuilds the model from config.json + model.pt and times only the forward pass on pre-built windows
(no window building or Python data handling). Inputs are random
windows of the right shape: these models have no data-dependent control flow, so cost doesn't depend on values.

Per batch size: untimed warm-up first (first-call CUDA setup and per-shape kernel compiles, ~30 s each for
Mamba-3 MIMO's TileLang kernels), then calls for --seconds while tegrastats samples power. Idle board power is
measured once (model loaded, nothing running) and subtracted, so energy per row is the model's own cost.
Writes <run>/benchmark.json.
"""
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import models

ROOT = Path(__file__).resolve().parent.parent


# --- Power: sample Jetson board power / GPU stats via tegrastats while a block of code runs. ----------------
# Off the Jetson (no tegrastats on PATH) PowerMonitor does nothing and summary() returns None.

TEGRASTATS = shutil.which("tegrastats")

_VDD_IN = re.compile(r"VDD_IN (\d+)mW")
_GPU_PCT = re.compile(r"GR3D_FREQ (\d+)%")
_RAM_MB = re.compile(r"RAM (\d+)/\d+MB")
_GPU_TEMP = re.compile(r"gpu@([\d.]+)C")


class PowerMonitor:
    """Context manager: polls tegrastats every `interval_ms` while the `with` block runs.

        with PowerMonitor() as mon:
            do_work()
        mon.summary(duration_s)  # None off-device or if no samples landed
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc = None
        self._thread = None
        self._samples: list[tuple[int, int | None, int | None, float | None]] = []

    def __enter__(self) -> "PowerMonitor":
        if TEGRASTATS is None:
            return self
        self._proc = subprocess.Popen(
            [TEGRASTATS, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            vdd = _VDD_IN.search(line)
            if not vdd:
                continue
            gpu, ram, temp = _GPU_PCT.search(line), _RAM_MB.search(line), _GPU_TEMP.search(line)
            self._samples.append((
                int(vdd.group(1)),
                int(gpu.group(1)) if gpu else None,
                int(ram.group(1)) if ram else None,
                float(temp.group(1)) if temp else None,
            ))

    def __exit__(self, *exc_info) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._thread.join(timeout=2)

    def summary(self, duration_s: float) -> dict | None:
        """None if tegrastats wasn't available or the block ran shorter than one sample interval."""
        if not self._samples:
            return None
        power = [s[0] for s in self._samples]
        gpu = [s[1] for s in self._samples if s[1] is not None]
        ram = [s[2] for s in self._samples if s[2] is not None]
        temp = [s[3] for s in self._samples if s[3] is not None]
        mean_power_mw = sum(power) / len(power)
        return {
            "n_samples": len(power),
            "mean_power_mw": round(mean_power_mw, 1),
            "max_power_mw": max(power),
            "energy_mj": round(mean_power_mw * duration_s, 1),  # mW * s = mJ
            "mean_gpu_util_pct": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "peak_ram_mb": max(ram) if ram else None,
            "mean_gpu_temp_c": round(sum(temp) / len(temp), 1) if temp else None,
        }


def load_model(run_dir: Path, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    meta = json.loads((ROOT / "data" / "processed" / cfg["dataset"] / "meta.json").read_text())
    return models.load(run_dir, len(meta["features"]), device), cfg


@torch.no_grad()
def time_calls(det, x: torch.Tensor, seconds: float) -> tuple[np.ndarray, float]:
    """Per-call latency (s) of det.net(x) for at least `seconds`; returns (latencies, wall time)."""
    times, t_start = [], time.perf_counter()
    while time.perf_counter() - t_start < seconds or len(times) < 20:
        t0 = time.perf_counter()
        with torch.autocast(**det.autocast):
            det.net(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times), time.perf_counter() - t_start


def measure_idle(seconds: float) -> float | None:
    with PowerMonitor() as mon:
        time.sleep(seconds)
    s = mon.summary(seconds)
    return s["mean_power_mw"] if s else None


def benchmark(run_dirs: list[Path], seconds: float = 20.0, batch_sizes: tuple[int, ...] = (1, 32, 512),
              idle_seconds: float = 15.0) -> None:
    """Benchmark each run and write its benchmark.json. `seconds` = timed duration per batch size."""
    device = "cuda"
    for run_dir in run_dirs:
        det, cfg = load_model(run_dir, device)
        torch.manual_seed(0)
        idle_mw = measure_idle(idle_seconds)
        out = {"run": run_dir.name, "model": cfg["model"], "hparams": cfg["hparams"], "n_params": det.n_params(),
               "device": torch.cuda.get_device_name(0), "idle_power_mw": idle_mw, "batch": {}}
        print(f"\n{run_dir.name}  ({det.n_params():,} params, idle {idle_mw} mW)")

        for bs in batch_sizes:
            x = torch.randn(bs, cfg["hparams"]["seq_len"], det.net.embed.in_features, device=device)
            time_calls(det, x, seconds=0)  # warm-up: >= 20 untimed calls, compiles happen here
            torch.cuda.reset_peak_memory_stats()
            with PowerMonitor() as mon:
                lat, wall = time_calls(det, x, seconds)
            p = mon.summary(wall)
            rows = bs * len(lat)
            us = lat * 1e6
            r = {
                "p50_us": round(float(np.percentile(us, 50)), 1),
                "p95_us": round(float(np.percentile(us, 95)), 1),
                "p99_us": round(float(np.percentile(us, 99)), 1),
                "throughput_rows_per_s": round(rows / wall, 1),
                "n_calls": len(lat),
                # Peak memory PyTorch allocated during the timed calls (weights + activations): the model's own
                # footprint. tegrastats' peak_ram_mb is the whole board (OS included), so it barely moves.
                "gpu_mem_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                "power": p,
            }
            if p and idle_mw is not None:
                above = p["mean_power_mw"] - idle_mw
                r["power_above_idle_mw"] = round(above, 1)
                r["energy_per_row_uj"] = round(above * wall / rows * 1000, 3)  # mW*s = mJ -> uJ
            out["batch"][str(bs)] = r
            print(f"  batch {bs:4d}: p50 {r['p50_us']:9.1f} us  p99 {r['p99_us']:9.1f} us  "
                  f"{r['throughput_rows_per_s']:12.1f} rows/s"
                  + (f"  power {p['mean_power_mw']:.0f} mW (+{r['power_above_idle_mw']:.0f})"
                     f"  {r['energy_per_row_uj']:.2f} uJ/row" if "energy_per_row_uj" in r else ""))

        (run_dir / "benchmark.json").write_text(json.dumps(out, indent=2))
        del det
        torch.cuda.empty_cache()
