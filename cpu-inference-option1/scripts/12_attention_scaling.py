#!/usr/bin/env python3
"""
Thread scaling of ATTENTION vs everything else in one decode step at a fixed context depth.
Answers: is the attention kernel limited by memory bandwidth (scales like the weight matmuls)
or by per-thread processing speed (scales better with threads)?

    python scripts/12_attention_scaling.py --tag laptop --model llama-3.2-1b --quant Q4_K_M --depth 8192
    python scripts/12_attention_scaling.py --tag rudra  --threads 1,2,4 --depth 4096

Writes results/ops_threads_d<depth>_<tag>.csv (raw operator rows, overwritten each run, so sums are
never doubled) and results/attention_scaling_<tag>.csv (the table printed below).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402


def find_profiler():
    exe = "op_profile.exe" if C.IS_WIN else "op_profile"
    for d in (config.ROOT / "profiler" / "build", config.ROOT / "profiler" / "build" / "Release"):
        if (d / exe).is_file():
            return d / exe
    C.die("profiler/build/op_profile not found - build it with scripts/00_build_llamacpp.py (without --no-profiler)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--depth", type=int, default=8192)
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--threads", default="", help="comma list (default: 1,2,4,.. up to the max)")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)
    tl = [int(x) for x in a.threads.split(",") if x] or C.thread_list(C.default_max_threads())
    exe, mpath = find_profiler(), C.model_path(a.model, a.quant)
    raw = config.RESULTS_DIR / f"ops_threads_d{a.depth}_{tag}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.unlink(missing_ok=True)                      # never append to an old file -> no doubled sums
    for t in tl:
        print(f"[run] threads={t} depth={a.depth}", flush=True)
        res = C.run_monitored([str(exe), "-m", str(mpath), "-t", str(t), "--phase", "decode",
                               "--depth", str(a.depth), "--n-gen", str(a.n_gen), "--tag", tag,
                               "--out", str(raw)], timeout=3 * 3600)
        if res["rc"] != 0:
            print(res["stderr"][-800:])
            C.die(f"op_profile failed at {t} threads")
        for line in res["stderr"].splitlines():
            if "[op_profile]" in line:
                print("      " + line)
    rows = list(csv.DictReader(open(raw, newline="")))
    out = []
    for t in tl:
        rs = [r for r in rows if int(r["threads"]) == t and not r["op"].startswith("_WALL")]
        wall = [float(r["ms_total"]) / int(r["units"]) for r in rows
                if int(r["threads"]) == t and r["op"] == "_WALL_PLAIN"]
        att = sum(float(r["ms_total"]) for r in rs if r["op"] == "FLASH_ATTN_EXT" or r["stem"] in ("kq", "kqv")
                  or r["op"] == "SOFT_MAX") / int(rs[0]["units"])
        tot = sum(float(r["ms_total"]) for r in rs if r["op"] not in ("RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE", "NONE")) / int(rs[0]["units"])
        out.append(dict(tag=tag, model=a.model, quant=a.quant, depth=a.depth, threads=t, attn_ms=att,
                        other_ms=tot - att, plain_tok_s=1000 / wall[0] if wall else float("nan")))
    base = out[0]
    for r in out:
        r["attn_speedup"] = base["attn_ms"] / r["attn_ms"]
        r["other_speedup"] = base["other_ms"] / r["other_ms"]
    res_csv = config.RESULTS_DIR / f"attention_scaling_{tag}.csv"
    with open(res_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in out)
    print(f"\n{'threads':>7} {'attn ms':>9} {'other ms':>9} {'attn x':>7} {'other x':>8} {'plain tok/s':>12}")
    for r in out:
        print(f"{r['threads']:7d} {r['attn_ms']:9.2f} {r['other_ms']:9.2f} {r['attn_speedup']:7.2f} "
              f"{r['other_speedup']:8.2f} {r['plain_tok_s']:12.2f}")
    print(f"\nsaved {res_csv}")


if __name__ == "__main__":
    main()
