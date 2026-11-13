#!/usr/bin/env python3
"""
Step 6 - per-operator time breakdown for prefill and decode (project "Profiling").

Uses profiler/op_profile (C++ program built by 00_build_llamacpp.py) which times every ggml
graph node through llama.cpp's own scheduler callback. Works identically on Linux, macOS, Windows.
Optionally cross-checks with Linux `perf` (--perf).

    python scripts/07_profile_ops.py --tag mylaptop                       # all models, default depths
    python scripts/07_profile_ops.py --tag mylaptop --model llama-3.2-1b --quant Q4_K_M --depths 0,1024,8192,16256
    python scripts/07_profile_ops.py --tag mylaptop --perf                # + perf record on one decode run

Outputs
    results/ops_raw_<tag>.csv       every run (one row per operator bucket per rep)
    results/ops_summary_<tag>.csv   share of time per category (mean of reps)  -> operator-breakdown plot
    results/ops_top3_<tag>.txt      top-3 operators for prefill and for decode  -> report/slide text
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

RAW_FIELDS = ["tag", "model", "quant", "exp", "rep", "phase", "depth", "threads", "fa", "ctk", "ctv", "units",
              "op", "stem", "src0_type", "calls", "ms_total"]
VIEW_OPS = {"NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"}   # no-op nodes, excluded from shares


def category(op: str, stem: str) -> str:
    if op in ("MUL_MAT", "MUL_MAT_ID"):
        if stem in ("Qcur", "Kcur", "Vcur", "qkv"):
            return "attn QKV matmul"
        if stem in ("attn_out", "kqv_out"):
            return "attn output matmul"
        if stem.startswith("ffn"):
            return "FFN matmul (gate/up/down)"
        if stem == "result_output":
            return "output head (lm_head)"
        if stem.startswith("kq"):
            return "attention core (QK^T, softmax, AV)"
        return "other matmul"
    if op in ("FLASH_ATTN_EXT", "SOFT_MAX"):
        return "attention core (QK^T, softmax, AV)"
    if op == "ROPE":
        return "RoPE"
    if op in ("RMS_NORM", "NORM") or (op == "MUL" and "norm" in stem):
        return "RMSNorm"
    if op == "ADD":
        return "residual add"
    if op in ("GLU", "SILU", "GELU", "UNARY", "SWIGLU", "SCALE") or (op == "MUL" and stem.startswith("ffn")):
        return "FFN activation (SwiGLU)"
    if op == "GET_ROWS":
        return "embedding lookup"
    if op in ("SET_ROWS", "CPY", "CONT", "DUP"):
        return "KV-cache write / copies"
    if op in VIEW_OPS:
        return "views (no-op)"
    return "other"


def find_profiler() -> Path:
    exe = "op_profile.exe" if C.IS_WIN else "op_profile"
    for d in (config.ROOT / "profiler" / "build", config.ROOT / "profiler" / "build" / "Release"):
        if (d / exe).is_file():
            return d / exe
    C.die("profiler/build/op_profile not found - run scripts/00_build_llamacpp.py (without --no-profiler)")


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_profile(exe, mpath, phase, depth, threads, units, extra):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.csv"
        cmd = [str(exe), "-m", str(mpath), "-t", str(threads), "--phase", phase, "--depth", str(depth),
               "--out", str(out)]
        cmd += (["--n-gen", str(units)] if phase == "decode" else ["--n-prompt", str(units)])
        cmd += extra
        res = C.run_monitored(cmd, timeout=3 * 3600)
        if res["rc"] != 0 or not out.exists():
            print(res["stderr"][-800:])
            return None, res
        return read_rows(out), res


def summarize(raw_rows, out_csv: Path, top_txt: Path):
    """raw_rows: dicts with rep + the C++ columns. -> category shares and top-3."""
    # per (model,quant,exp,phase,depth,threads,rep): per-op ms
    runs = defaultdict(list)
    for r in raw_rows:
        k = (r["model"], r["quant"], r["exp"], r["phase"], int(r["depth"]), int(r["threads"]), r["rep"])
        runs[k].append(r)
    agg = defaultdict(lambda: defaultdict(list))        # config -> category -> [ms per token per rep]
    top = defaultdict(lambda: defaultdict(list))        # config -> "op/stem" -> [ms per token]
    wall = defaultdict(lambda: defaultdict(list))
    for k, rows in runs.items():
        cfg = k[:6]
        units = int(rows[0]["units"])
        cat_ms = defaultdict(float)
        for r in rows:
            if r["op"].startswith("_WALL"):
                wall[cfg][r["op"]].append(float(r["ms_total"]) / units)
                continue
            if r["op"] in VIEW_OPS:
                continue
            ms = float(r["ms_total"]) / units
            cat_ms[category(r["op"], r["stem"])] += ms
            label = f"{r['op']}:{r['stem'] or '-'}" + (f" [{r['src0_type']}]" if r["op"].startswith("MUL_MAT") else "")
            top[cfg][label].append(ms)
        for c, ms in cat_ms.items():
            agg[cfg][c].append(ms)

    def mean(xs):
        return sum(xs) / len(xs)

    out_rows, lines = [], []
    for cfg in sorted(agg):
        total = sum(mean(v) for v in agg[cfg].values())
        for c, v in sorted(agg[cfg].items(), key=lambda kv: -mean(kv[1])):
            out_rows.append(dict(model=cfg[0], quant=cfg[1], exp=cfg[2], phase=cfg[3], depth=cfg[4], threads=cfg[5],
                                 category=c, ms_per_token=round(mean(v), 5), share=round(mean(v) / total, 4)))
        plain = mean(wall[cfg]["_WALL_PLAIN"]) if wall[cfg]["_WALL_PLAIN"] else float("nan")
        prof = mean(wall[cfg]["_WALL_PROFILED"]) if wall[cfg]["_WALL_PROFILED"] else float("nan")
        lines.append(f"\n== {cfg[0]} {cfg[1]} [{cfg[2]}] {cfg[3]} depth={cfg[4]} threads={cfg[5]} ==  "
                     f"plain {plain:.3f} ms/token, instrumented {prof:.3f} ms/token "
                     f"(instrumentation overhead {(prof / plain - 1) * 100:+.1f} %)")
        tot_top = sum(mean(v) for v in top[cfg].values())
        for i, (lab, v) in enumerate(sorted(top[cfg].items(), key=lambda kv: -mean(kv[1]))[:5], 1):
            lines.append(f"  {i}. {lab:42s} {mean(v):8.4f} ms/token  {mean(v) / tot_top * 100:5.1f} %"
                         + ("   <- top-3" if i <= 3 else ""))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "quant", "exp", "phase", "depth", "threads", "category",
                                          "ms_per_token", "share"])
        w.writeheader()
        w.writerows(out_rows)
    top_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def perf_crosscheck(exe_bench: Path, mpath: Path, threads: int, tag: str):
    """Linux only: sample symbols with perf during one decode run (cross-check of the callback profile)."""
    if not C.IS_LINUX or not shutil.which("perf"):
        print("[perf] skipped (needs Linux + `perf`; in WSL2 perf needs a matching kernel - optional)")
        return
    data = config.RESULTS_DIR / f"perf_{tag}.data"
    cmd = ["perf", "record", "-F", "999", "-g", "-o", str(data), "--", str(exe_bench), "-m", str(mpath), "-ngl", "0",
           "-t", str(threads), "-p", "0", "-n", "64", "-r", "1", "-o", "csv"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[perf] record failed:", r.stderr[-300:], "\n  hint: sudo sysctl kernel.perf_event_paranoid=1")
        return
    rep = subprocess.run(["perf", "report", "-i", str(data), "--no-children", "--sort", "symbol", "--stdio",
                          "--percent-limit", "1"], capture_output=True, text=True).stdout
    out = config.RESULTS_DIR / f"perf_report_{tag}.txt"
    out.write_text(rep)
    print(f"[perf] top symbols -> {out}\n" + "\n".join(l for l in rep.splitlines() if "%" in l)[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--exp-name", default="baseline")
    ap.add_argument("--model", default="all")
    ap.add_argument("--quant", default="all")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--depths", default="", help="comma list of context depths for the decode profile")
    ap.add_argument("--n-gen", type=int, default=64, help="decode steps per profile")
    ap.add_argument("--n-prompt", type=int, default=512, help="prompt length for the prefill profile")
    ap.add_argument("--phases", default="decode,prefill")
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--fa", default=None)
    ap.add_argument("--ctk", default=None)
    ap.add_argument("--ctv", default=None)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)
    threads = a.threads or C.default_max_threads()
    raw_csv = config.RESULTS_DIR / f"ops_raw_{tag}.csv"
    sum_csv = config.RESULTS_DIR / f"ops_summary_{tag}.csv"
    top_txt = config.RESULTS_DIR / f"ops_top3_{tag}.txt"

    if not a.summarize_only:
        exe = find_profiler()
        extra = []
        for n, v in (("--fa", a.fa), ("--ctk", a.ctk), ("--ctv", a.ctv)):
            if v:
                extra += [n, v]
        env_mask = C.pcore_mask_hex()
        if env_mask:
            print(f"[info] hybrid CPU: run this script under `taskset -c <P-core ids>` for clean profiles "
                  f"(P-core mask {env_mask}).")
        phases = a.phases.split(",")
        for model, quant in [(m, q) for m in config.MODELS for q in config.MODELS[m]["files"]
                             if a.model in ("all", m) and a.quant in ("all", q)]:
            try:
                mpath = C.model_path(model, quant)
            except (FileNotFoundError, KeyError) as e:
                print(f"[skip] {e}")
                continue
            max_ctx = config.MODELS[model]["max_ctx"]
            deepest = max_ctx - a.n_gen - 8
            depths = [int(x) for x in a.depths.split(",") if x] or \
                sorted({d for d in (0, 1024, 4096, 8192) if d <= deepest} | {deepest})
            plan = []
            if "decode" in phases:
                plan += [("decode", d, a.n_gen) for d in depths]
            if "prefill" in phases:
                plan += [("prefill", 0, a.n_prompt)]
            for phase, depth, units in plan:
                for rep in range(1, a.reps + 1):
                    print(f"[prof] {model} {quant} {phase} depth={depth} threads={threads} rep={rep}/{a.reps}", flush=True)
                    rows, res = run_profile(exe, mpath, phase, depth, threads, units, extra)
                    if rows is None:
                        print("       FAILED")
                        continue
                    for r in rows:
                        r.update(model=model, quant=quant, exp=a.exp_name, rep=rep, tag=tag)
                    for r in rows:
                        C.append_csv(raw_csv, r, RAW_FIELDS)
                    print(f"       peak RSS {res['peak_rss_gb']:.2f} GB")
        if a.perf:
            m0 = next(((m, q) for m in config.MODELS for q in config.MODELS[m]["files"]
                       if a.model in ("all", m) and a.quant in ("all", q)), None)
            if m0:
                perf_crosscheck(C.find_bin("llama-bench"), C.model_path(*m0), threads, tag)

    if raw_csv.exists():
        summarize(read_rows(raw_csv), sum_csv, top_txt)
        print(f"\nsummary -> {sum_csv}\ntop-3   -> {top_txt}")


if __name__ == "__main__":
    main()
