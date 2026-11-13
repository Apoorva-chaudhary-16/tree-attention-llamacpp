#!/usr/bin/env python3
"""
Step 5 - scripted benchmark sweeps (prompt length x context depth x threads x quantization).

Every measurement = ONE llama-bench process with -r 3 (3 repetitions inside the process,
after llama-bench's own warm-up).  We wrap that process to record:
    * peak RSS (GB)         - project metric "Peak memory (RSS)"
    * swap-out during run   - if > 8 MB the run is flagged and automatically re-run
    * CPU MHz start/end     - cheap thermal-throttle indicator
    * the 3 individual samples -> mean and standard deviation (the "spread")

Examples
    python scripts/06_bench.py --quick                       # 10-minute smoke test
    python scripts/06_bench.py --tag mylaptop                # everything (hours)
    python scripts/06_bench.py --sweep decode --model llama-3.2-1b
    python scripts/06_bench.py --sweep threads --max-threads 8
    # your own experiments (step 8): flash-attn off, quantised KV cache, ...
    python scripts/06_bench.py --sweep decode --fa off --exp-name fa_off
    python scripts/06_bench.py --sweep decode --ctk q8_0 --ctv q8_0 --fa on --exp-name kv_q8

The script is resumable: re-running skips rows already in the CSV.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

FIELDS = ["timestamp", "tag", "exp", "model", "quant", "sweep", "test", "n_prompt", "n_gen", "depth",
          "threads", "reps", "ts_mean", "ts_std", "ts_min", "ts_max", "samples", "peak_rss_gb",
          "over_budget", "swap_out_mb", "swap_flag", "low_mem_flag", "throttle_suspect",
          "freq_start_mhz", "freq_end_mhz", "wall_s", "attempts", "status", "model_size_bytes",
          "model_params", "flags", "llama_commit", "cpu_info", "error"]

SWAP_FLAG_MB = 8.0


def build_plan(model_key, quant, sweeps, max_threads, threads_list):
    max_ctx = config.MODELS[model_key]["max_ctx"]
    n_gen = config.N_GEN
    plan = []
    if "prefill" in sweeps:
        lens = sorted({l for l in config.PREFILL_LENS if l <= max_ctx} | {max_ctx})
        for L in lens:
            plan.append(dict(sweep="prefill", n_prompt=L, n_gen=0, depth=0, threads=max_threads))
    if "decode" in sweeps:
        deepest = max_ctx - n_gen                       # the "largest allowed" context
        depths = sorted({d for d in config.DECODE_DEPTHS if d + n_gen <= max_ctx} | {deepest})
        for d in depths:
            plan.append(dict(sweep="decode", n_prompt=0, n_gen=n_gen, depth=d, threads=max_threads))
    if "threads" in sweeps:
        for t in threads_list:
            plan.append(dict(sweep="threads", n_prompt=config.THREAD_SWEEP_TESTS["pp"], n_gen=0, depth=0, threads=t))
            plan.append(dict(sweep="threads", n_prompt=0, n_gen=config.THREAD_SWEEP_TESTS["tg"], depth=0, threads=t))
    if "reference" in sweeps:     # the fixed PARAM Rudra test: 512-token prompt, 128-token decode
        r = config.REFERENCE
        plan.append(dict(sweep="reference", n_prompt=r["pp"], n_gen=0, depth=0, threads=max_threads))
        plan.append(dict(sweep="reference", n_prompt=0, n_gen=r["tg"], depth=0, threads=max_threads))
    return plan


def key_of(tag, exp, model, quant, p, flags):
    return (tag, exp, model, quant, str(p["n_prompt"]), str(p["n_gen"]), str(p["depth"]),
            str(p["threads"]), flags)


def parse_jsonl(stdout):
    rows = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="machine label, e.g. laptop / rudra (default: hostname)")
    ap.add_argument("--exp-name", default="baseline", help="label for non-baseline experiments")
    ap.add_argument("--model", default="all", help="llama-3.2-1b | llama-3.2-3b | all")
    ap.add_argument("--quant", default="all", help="Q4_K_M | Q8_0 | all")
    ap.add_argument("--sweep", default="all", help="prefill,decode,threads | all | reference (PARAM Rudra test)")
    ap.add_argument("--max-threads", type=int, default=0, help="0 = auto (performance/physical cores)")
    ap.add_argument("--threads-list", default="", help="comma list overriding the thread sweep")
    ap.add_argument("--reps", type=int, default=config.REPS)
    ap.add_argument("--out", default="", help="CSV path (default results/bench_<tag>.csv)")
    ap.add_argument("--quick", action="store_true", help="small subset to check the pipeline end to end")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--delay", type=int, default=3, help="cool-down seconds between runs")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="per-run timeout in seconds")
    ap.add_argument("--max-attempts", type=int, default=2)
    # llama-bench pass-through (used by experiments)
    ap.add_argument("--fa", choices=["on", "off", "auto"], default=None)
    ap.add_argument("--ctk", default=None, help="K-cache type: f16, q8_0, q4_0 ...")
    ap.add_argument("--ctv", default=None, help="V-cache type (needs --fa on for quantised types)")
    ap.add_argument("-b", "--batch", type=int, default=None)
    ap.add_argument("-ub", "--ubatch", type=int, default=None)
    ap.add_argument("--cpu-mask", default=None, help="hex mask, e.g. 0xFF (pin to performance cores)")
    ap.add_argument("--cpu-strict", choices=["0", "1"], default=None)
    ap.add_argument("--prio", default=None)
    ap.add_argument("--poll", default=None)
    ap.add_argument("--numa", default=None, choices=["distribute", "isolate", "numactl"])
    ap.add_argument("--extra", default="", help="raw extra llama-bench args, quoted")
    a = ap.parse_args()

    tag = C.host_tag(a.tag)
    out = Path(a.out) if a.out else config.RESULTS_DIR / f"bench_{tag}.csv"
    bench = C.find_bin("llama-bench")
    max_threads = a.max_threads or C.default_max_threads()
    tl = [int(x) for x in a.threads_list.split(",") if x] or C.thread_list(max_threads)
    sweeps = {"prefill", "decode", "threads"} if a.sweep == "all" else set(a.sweep.split(","))
    bad = sweeps - {"prefill", "decode", "threads", "reference"}
    if bad:
        C.die(f"unknown sweep(s) {bad}")

    # Intel hybrid CPU: pin to performance cores automatically unless told otherwise.
    cpu_mask = a.cpu_mask
    pm = C.pcore_mask_hex()
    if cpu_mask is None and pm:
        cpu_mask = pm
        print(f"[info] Intel hybrid CPU detected -> pinning to performance cores, mask {pm}, threads <= {max_threads}")

    flag_parts = []
    extra_args = []
    for name, val in (("-fa", a.fa), ("-ctk", a.ctk), ("-ctv", a.ctv), ("-b", a.batch), ("-ub", a.ubatch),
                      ("-C", cpu_mask), ("--cpu-strict", a.cpu_strict), ("--prio", a.prio),
                      ("--poll", a.poll), ("--numa", a.numa)):
        if val is not None:
            extra_args += [name, str(val)]
            flag_parts.append(f"{name}={val}")
    if a.extra:
        extra_args += a.extra.split()
        flag_parts.append(a.extra)
    flags = " ".join(flag_parts)

    models = [(m, q) for m in config.MODELS for q in config.MODELS[m]["files"]
              if a.model in ("all", m) and a.quant in ("all", q)]
    if not models:
        C.die("no model/quant matches your filters")

    if a.quick:
        a.reps = min(a.reps, 2)
        sweeps = {"prefill", "decode", "threads"} if a.sweep == "all" else sweeps
        models = models[:1]

    done = set()
    if not a.no_resume:
        for r in C.read_csv(out):
            if r.get("status") == "ok":
                done.add((r["tag"], r["exp"], r["model"], r["quant"], r["n_prompt"], r["n_gen"],
                          r["depth"], r["threads"], r["flags"]))

    commit = C.llama_commit()
    n_total = n_run = n_fail = 0
    t_begin = time.time()
    print(f"[info] results -> {out}\n[info] llama-bench: {bench}\n[info] max threads: {max_threads}, "
          f"thread sweep: {tl}, reps: {a.reps}, extra flags: '{flags}'")

    for model_key, quant in models:
        try:
            mpath = C.model_path(model_key, quant)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] {e}")
            continue
        plan = build_plan(model_key, quant, sweeps, max_threads, tl)
        if a.quick:
            plan = [p for p in plan if (p["sweep"] == "prefill" and p["n_prompt"] <= 512) or
                    (p["sweep"] == "decode" and p["depth"] <= 512) or
                    (p["sweep"] == "threads" and p["threads"] in (1, max_threads))]
        for p in plan:
            n_total += 1
            if key_of(tag, a.exp_name, model_key, quant, p, flags) in done:
                print(f"[resume] skip {model_key} {quant} {p['sweep']} p={p['n_prompt']} n={p['n_gen']} "
                      f"d={p['depth']} t={p['threads']}")
                continue
            cmd = [str(bench), "-m", str(mpath), "-ngl", "0", "-t", str(p["threads"]),
                   "-p", str(p["n_prompt"]), "-n", str(p["n_gen"]), "-d", str(p["depth"]),
                   "-r", str(a.reps), "-o", "jsonl"]
            if a.no_warmup:
                cmd.append("--no-warmup")
            cmd += extra_args

            row = None
            for attempt in range(1, a.max_attempts + 1):
                avail = C.available_gb()
                low_mem = avail == avail and avail < 1.5          # NaN-safe
                label = (f"{model_key} {quant} {p['sweep']:7s} p={p['n_prompt']:<6d} n={p['n_gen']:<4d} "
                         f"d={p['depth']:<6d} t={p['threads']:<3d}")
                print(f"[run ] {label} (attempt {attempt}, free RAM {avail:.1f} GB)", flush=True)
                res = C.run_monitored(cmd, timeout=a.timeout)
                parsed = parse_jsonl(res["stdout"]) if res["rc"] == 0 else []
                swapped = res["swap_out_mb"] > SWAP_FLAG_MB
                fs, fe = res["freq_start_mhz"], res["freq_end_mhz"]
                throttle = bool(fs and fe and fe < 0.85 * fs)
                row = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), tag=tag, exp=a.exp_name,
                           model=model_key, quant=quant, sweep=p["sweep"],
                           test=("pp%d" % p["n_prompt"]) if p["n_prompt"] else f"tg{p['n_gen']}",
                           n_prompt=p["n_prompt"], n_gen=p["n_gen"], depth=p["depth"],
                           threads=p["threads"], reps=a.reps,
                           peak_rss_gb=round(res["peak_rss_gb"], 3),
                           over_budget=res["peak_rss_gb"] > config.MEM_BUDGET_GB,
                           swap_out_mb=round(res["swap_out_mb"], 1), swap_flag=swapped,
                           low_mem_flag=low_mem, throttle_suspect=throttle,
                           freq_start_mhz=fs and round(fs), freq_end_mhz=fe and round(fe),
                           wall_s=round(res["wall_s"], 1), attempts=attempt, flags=flags,
                           llama_commit=commit)
                if res["rc"] != 0 or not parsed:
                    why = "timeout" if res["timed_out"] else f"rc={res['rc']}"
                    row.update(status="error", error=(why + ": " + res["stderr"][-400:]).replace("\n", " | "))
                    print(f"[FAIL] {why}\n{res['stderr'][-600:]}")
                    break                                   # a crash is not fixed by retrying
                j = parsed[-1]
                samples = [float(x) for x in j.get("samples_ts", [])] or [float(j["avg_ts"])]
                m, s = C.mean_std(samples)
                row.update(status="ok", ts_mean=round(m, 3), ts_std=round(s, 3),
                           ts_min=round(min(samples), 3), ts_max=round(max(samples), 3),
                           samples=json.dumps([round(x, 3) for x in samples]),
                           model_size_bytes=j.get("model_size"), model_params=j.get("model_n_params"),
                           cpu_info=j.get("cpu_info"))
                print(f"       -> {m:8.2f} +/- {s:5.2f} tok/s | peak RSS {res['peak_rss_gb']:.2f} GB"
                      f"{' | SWAPPED' if swapped else ''}{' | OVER 4.5GB' if row['over_budget'] else ''}"
                      f"{' | throttle?' if throttle else ''}", flush=True)
                if swapped and attempt < a.max_attempts:
                    print("       swap detected -> close programs; waiting 20 s and re-running", flush=True)
                    time.sleep(20)
                    continue
                break
            C.append_csv(out, row, FIELDS)
            n_run += 1
            n_fail += row["status"] != "ok"
            time.sleep(a.delay)

    print(f"\n[done] planned {n_total}, executed {n_run}, failed {n_fail}, "
          f"elapsed {(time.time() - t_begin) / 60:.1f} min -> {out}")
    if n_fail:
        print("[warn] failures are recorded with status=error in the CSV; fix and re-run (resume skips good rows).")
        sys.exit(2)


if __name__ == "__main__":
    main()
