"""
Shared helpers used by every script.

Design rules
  * Standard library only for the parts needed on the PARAM Rudra node
    (psutil / numpy / pandas are optional there).
  * Every measurement is wrapped by run_monitored(), which records peak RSS,
    swap activity and CPU frequency drift for that one process.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import config

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
GiB = 1024 ** 3


# --------------------------------------------------------------------------
# Locating binaries
# --------------------------------------------------------------------------
def find_bin(name: str) -> Path:
    """Find a llama.cpp executable (llama-bench, llama-perplexity, ...)."""
    exe = name + (".exe" if IS_WIN else "")
    env_dir = os.environ.get("LLAMA_BIN_DIR")
    cands: List[Path] = []
    if env_dir:
        cands.append(Path(env_dir) / exe)
    for base in (config.BUILD_DIR,):
        cands += [base / "bin" / exe, base / "bin" / "Release" / exe, base / exe]
    for c in cands:
        if c.is_file():
            return c
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        f"Cannot find '{exe}'. Build llama.cpp first: python scripts/00_build_llamacpp.py "
        f"(or set LLAMA_BIN_DIR)."
    )


def model_path(model_key: str, quant: str) -> Path:
    m = config.MODELS[model_key]
    if quant not in m["files"]:
        raise KeyError(f"{model_key} has no quant '{quant}' (project rules). Options: {list(m['files'])}")
    p = config.MODELS_DIR / m["files"][quant]
    if not p.is_file():
        raise FileNotFoundError(f"Model file missing: {p}. Run: python scripts/01_get_models.py")
    return p


def llama_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(config.LLAMA_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# CPU topology / threads
# --------------------------------------------------------------------------
def allowed_cpus() -> int:
    """Logical CPUs this process may use (respects SLURM cgroups / taskset)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    return os.cpu_count() or 1


def intel_pcore_list() -> Optional[List[int]]:
    """Logical CPU ids of Intel performance cores (hybrid CPUs, Linux only)."""
    p = Path("/sys/devices/cpu_core/cpus")
    if not p.exists():
        return None
    ids: List[int] = []
    for part in p.read_text().strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids or None


def pcore_mask_hex() -> Optional[str]:
    ids = intel_pcore_list()
    if not ids:
        return None
    mask = 0
    for i in ids:
        mask |= 1 << i
    return hex(mask)


def _smt_active() -> bool:
    try:
        return Path("/sys/devices/system/cpu/smt/active").read_text().strip() == "1"
    except Exception:
        return False


def linux_physical_cores_allowed() -> Optional[int]:
    """Distinct (package, core) pairs among the CPUs this process may use."""
    if not (IS_LINUX and hasattr(os, "sched_getaffinity")):
        return None
    seen = set()
    try:
        for cpu in os.sched_getaffinity(0):
            t = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            seen.add((t.joinpath("physical_package_id").read_text().strip(),
                      t.joinpath("core_id").read_text().strip()))
    except Exception:
        return None
    return len(seen) or None


def default_max_threads() -> int:
    """Threads for 'all performance cores'. Override with env BENCH_MAX_THREADS."""
    env = os.environ.get("BENCH_MAX_THREADS")
    if env:
        return max(1, int(env))
    ids = intel_pcore_list()
    if ids:                                   # Intel hybrid: P-cores only, one thread per core
        return max(1, len(ids) // 2) if _smt_active() else len(ids)
    n = linux_physical_cores_allowed()
    if n:
        return n
    if psutil:
        phys = psutil.cpu_count(logical=False)
        if phys:
            return max(1, min(phys, allowed_cpus()))
    return max(1, allowed_cpus())


def thread_list(max_threads: int) -> List[int]:
    out, t = [], 1
    while t < max_threads:
        out.append(t)
        t *= 2
    out.append(max_threads)
    return sorted(set(out))


# --------------------------------------------------------------------------
# Memory / swap / frequency
# --------------------------------------------------------------------------
def available_gb() -> float:
    if psutil:
        return psutil.virtual_memory().available / GiB
    if IS_LINUX:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GiB
    return float("nan")


def total_gb() -> float:
    if psutil:
        return psutil.virtual_memory().total / GiB
    if IS_LINUX:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / GiB
    return float("nan")


def swap_out_bytes() -> int:
    """Monotonic counter of bytes swapped OUT since boot (0 if unavailable)."""
    if IS_LINUX:
        try:
            for line in Path("/proc/vmstat").read_text().splitlines():
                if line.startswith("pswpout "):
                    return int(line.split()[1]) * 4096
        except Exception:
            pass
    if psutil:
        try:
            return int(psutil.swap_memory().sout)
        except Exception:
            pass
    return 0


def cpu_freq_mhz() -> Optional[float]:
    try:
        if psutil:
            f = psutil.cpu_freq()
            if f and f.current:
                return float(f.current)
        if IS_LINUX:
            vals = [float(l.split(":")[1]) for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("cpu MHz")]
            if vals:
                return sum(vals) / len(vals)
    except Exception:
        pass
    return None


def _peak_rss_bytes(pid: int) -> int:
    """Peak resident set of a live process (0 if it vanished)."""
    try:
        if IS_LINUX:
            # VmHWM is the kernel-tracked high-water mark, so coarse polling still sees the true peak.
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
            return 0
        if psutil:
            mi = psutil.Process(pid).memory_info()
            return int(getattr(mi, "peak_wset", 0) or mi.rss)   # Windows: peak_wset
    except Exception:
        return 0
    return 0


def run_monitored(cmd: List[str], env_extra: Optional[Dict[str, str]] = None,
                  timeout: Optional[float] = None, poll_s: float = 0.02) -> dict:
    """
    Run `cmd`, return dict(rc, stdout, stderr, wall_s, peak_rss_gb, swap_out_mb,
    freq_start_mhz, freq_end_mhz, timed_out).
    stdout/stderr go to temp files (no pipe deadlocks on long outputs).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""        # belt and braces: never touch the GPU
    env["GGML_CUDA_DISABLE"] = "1"
    if env_extra:
        env.update(env_extra)
    swap0 = swap_out_bytes()
    f0 = cpu_freq_mhz()
    t0 = time.time()
    peak = 0
    timed_out = False
    with tempfile.TemporaryFile("w+b") as fo, tempfile.TemporaryFile("w+b") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
        f_end = f0
        last_freq_t = t0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            peak = max(peak, _peak_rss_bytes(proc.pid))
            now = time.time()
            if now - last_freq_t > 1.0:          # sample frequency while the load is running
                f = cpu_freq_mhz()
                if f:
                    f_end = f
                last_freq_t = now
            if timeout and now - t0 > timeout:
                proc.kill()
                timed_out = True
                rc = proc.wait()
                break
            time.sleep(poll_s)
        wall = time.time() - t0
        fo.seek(0)
        fe.seek(0)
        out = fo.read().decode("utf-8", "replace")
        err = fe.read().decode("utf-8", "replace")
    return dict(rc=rc, stdout=out, stderr=err, wall_s=wall, peak_rss_gb=peak / GiB,
                swap_out_mb=max(0, swap_out_bytes() - swap0) / 1024 ** 2,
                freq_start_mhz=f0, freq_end_mhz=f_end, timed_out=timed_out)


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------
def append_csv(path: Path, row: dict, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def host_tag(default: str = "") -> str:
    return default or os.environ.get("BENCH_TAG") or platform.node().split(".")[0] or "machine"


def mean_std(xs: List[float]):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
