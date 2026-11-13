#!/usr/bin/env python3
"""
Step 1c - measure memory bandwidth with the official STREAM benchmark (McCalpin).

Downloads stream.c, compiles it with OpenMP, runs it at 1,2,4,... threads and stores
results/stream_<tag>.json.  The roofline script uses the Triad number.

    python scripts/02_stream.py --tag mylaptop
    python scripts/02_stream.py --tag rudra --max-threads 32 --proc-bind spread

Array size rule (STREAM docs): each array must be >= 4x the largest cache. We take
max(40M elements, 4 x L3 / 8) and cap the three arrays at 40% of RAM.

Compilers: gcc / clang (-fopenmp) or MSVC (cl /openmp). Apple clang has no OpenMP:
`brew install gcc` (then use gcc-14) or `brew install libomp`; without OpenMP the script
falls back to a single-thread build and says so (bandwidth is then a LOWER bound).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

STREAM_URL = "https://www.cs.virginia.edu/stream/FTP/Code/stream.c"
MIRROR_URL = "https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c"


def fetch_source(dst: Path):
    if dst.exists() and dst.stat().st_size > 1000:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for url in (STREAM_URL, MIRROR_URL):
        try:
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if b"STREAM" in data[:5000]:
                dst.write_bytes(data)
                return
        except Exception as e:
            print(f"  failed: {e}")
    C.die(f"could not download stream.c. Put it manually at {dst}")


def l3_bytes() -> int:
    try:
        if C.IS_LINUX:
            out = subprocess.run(["lscpu", "-B"], capture_output=True, text=True).stdout
            m = re.search(r"^L3 cache:\s*(\d+)", out, re.M)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 32 * 1024 ** 2


def compile_stream(src: Path, exe: Path, n_elem: int, ntimes: int):
    exe.parent.mkdir(parents=True, exist_ok=True)
    cands = []
    for cc in ("gcc", "gcc-15", "gcc-14", "gcc-13", "gcc-12", "clang", "cc"):
        p = shutil.which(cc)
        if p:
            cands.append(p)
    defs = [f"-DSTREAM_ARRAY_SIZE={n_elem}", f"-DNTIMES={ntimes}"]
    for cc in cands:
        for omp in (True, False):
            cmd = [cc, "-O3", "-march=native", *(["-fopenmp"] if omp else []), *defs, str(src), "-o", str(exe)]
            if cc.endswith("clang") and omp and C.IS_MAC:
                cmd = [cc, "-O3", "-Xpreprocessor", "-fopenmp", "-lomp", *defs, str(src), "-o", str(exe)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                if not omp:
                    print("[warn] compiled WITHOUT OpenMP: only single-thread bandwidth will be measured")
                return omp
    cl = shutil.which("cl")
    if cl:
        r = subprocess.run([cl, "/O2", "/openmp", *[d.replace("-D", "/D") for d in defs], str(src),
                            f"/Fe:{exe}"], capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(r.stdout[-800:], r.stderr[-800:])
    C.die("no working C compiler found (need gcc/clang, or run from a VS developer prompt). "
          "On Windows the easiest route is WSL2 Ubuntu.")


def parse(out: str):
    res = {}
    for name in ("Copy", "Scale", "Add", "Triad"):
        m = re.search(rf"^{name}:\s+([\d.]+)", out, re.M)
        if m:
            res[name] = float(m.group(1)) / 1000.0           # MB/s -> GB/s
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-threads", type=int, default=0)
    ap.add_argument("--ntimes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="runs per thread count (best kept)")
    ap.add_argument("--proc-bind", default="close", help="OMP_PROC_BIND value: close | spread | false")
    ap.add_argument("--array-elems", type=int, default=0, help="override STREAM_ARRAY_SIZE")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)

    src = config.DATA_DIR / "stream" / "stream.c"
    exe = config.DATA_DIR / "stream" / ("stream.exe" if C.IS_WIN else "stream_bin")
    fetch_source(src)

    ram = C.total_gb() * 1024 ** 3
    n = a.array_elems or max(40_000_000, 4 * l3_bytes() // 8)
    while 3 * 8 * n > 0.40 * ram and n > 10_000_000:
        n //= 2
    print(f"STREAM_ARRAY_SIZE={n} elements ({3 * 8 * n / 1024 ** 3:.2f} GiB for 3 arrays), NTIMES={a.ntimes}")
    omp = compile_stream(src, exe, n, a.ntimes)

    max_t = a.max_threads or C.default_max_threads()
    tl = C.thread_list(max_t) if omp else [1]
    per_thread = {}
    for t in tl:
        best = {}
        for _ in range(a.repeats):
            env = {"OMP_NUM_THREADS": str(t), "OMP_PROC_BIND": a.proc_bind, "OMP_PLACES": "cores"}
            res = C.run_monitored([str(exe)], env_extra=env, timeout=600)
            if res["rc"] != 0:
                print(res["stderr"][-400:])
                continue
            vals = parse(res["stdout"])
            if "Validation" in res["stdout"] and "failed" in res["stdout"].lower():
                C.die("STREAM validation failed")
            for k, v in vals.items():
                best[k] = max(best.get(k, 0.0), v)
        per_thread[t] = best
        print(f"threads={t:3d}  " + "  ".join(f"{k}={v:6.1f}" for k, v in best.items()) + "  GB/s")

    good = {t: v for t, v in per_thread.items() if "Triad" in v}
    if not good:
        C.die("STREAM produced no results")
    best_t = max(good, key=lambda t: good[t]["Triad"])
    out = {"tag": tag, "array_elems": n, "ntimes": a.ntimes, "openmp": omp, "proc_bind": a.proc_bind,
           "per_thread_gbs": per_thread, "best_threads": best_t,
           "peak_triad_gbs": good[best_t]["Triad"], "peak_copy_gbs": good[best_t].get("Copy")}
    C.json_dump(config.RESULTS_DIR / f"stream_{tag}.json", out)
    print(f"\nPeak Triad bandwidth: {out['peak_triad_gbs']:.1f} GB/s at {best_t} threads "
          f"-> {config.RESULTS_DIR / ('stream_' + tag + '.json')}")
    print("Note: use this number as the 'memory-bandwidth limit' in the roofline (05_roofline.py).")


if __name__ == "__main__":
    main()
