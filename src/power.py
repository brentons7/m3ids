"""Sample Jetson power/thermal/utilization stats via tegrastats while a block of code runs.

Portable across runs on the Orin Nano and anywhere else: if tegrastats isn't on PATH (e.g. a
dev laptop), PowerMonitor is a no-op and summary() returns None, so the rest of the pipeline
doesn't need to know whether it's on the edge device.
"""
import re
import shutil
import subprocess
import threading

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
