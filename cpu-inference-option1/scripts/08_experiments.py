#!/usr/bin/env python3
"""
Step 8 - your own experiment: change ONE thing, measure its effect.
Each experiment re-runs part of the sweep with a different llama.cpp setting and a distinct --exp-name,
so 09_plots.py can compare it with the baseline.

    python scripts/08_experiments.py --tag mylaptop --list
    python scripts/08_experiments.py --tag mylaptop --which kv_q8,fa_off --model llama-3.2-1b --quant Q4_K_M

Recommended experiment for this project: kv_q8 (quantised KV cache). It attacks exactly the cost that
grows with context, so it tests the roofline explanation directly: if KV bytes shrink ~2x, deep-context
decode should speed up by about the predicted amount.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "fa_on":    ("decode",  ["--fa", "on"], "force flash attention on"),
    "fa_off":   ("decode",  ["--fa", "off"], "flash attention off (classic QK^T, softmax, AV)"),
    "kv_q8":    ("decode",  ["--fa", "on", "--ctk", "q8_0", "--ctv", "q8_0"], "8-bit KV cache"),
    "kv_q4":    ("decode",  ["--fa", "on", "--ctk", "q4_0", "--ctv", "q4_0"], "4-bit KV cache (quality risk)"),
    "strict":   ("threads", ["--cpu-strict", "1"], "strict thread pinning"),
    "poll0":    ("threads", ["--poll", "0"], "threads sleep instead of spin-wait"),
    "ub128":    ("prefill", ["-ub", "128"], "physical batch 128"),
    "ub1024":   ("prefill", ["-ub", "1024"], "physical batch 1024"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--which", default="kv_q8,fa_off")
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-threads", type=int, default=0)
    a, rest = ap.parse_known_args()
    if a.list:
        for k, (sw, fl, d) in EXPERIMENTS.items():
            print(f"{k:8s} sweep={sw:8s} {' '.join(fl):40s} {d}")
        return
    for name in a.which.split(","):
        if name not in EXPERIMENTS:
            sys.exit(f"unknown experiment {name}; use --list")
        sweep, flags, _ = EXPERIMENTS[name]
        cmd = [sys.executable, str(ROOT / "scripts" / "06_bench.py"), "--tag", a.tag, "--exp-name", name,
               "--model", a.model, "--quant", a.quant, "--sweep", sweep, *flags, *rest]
        if a.max_threads:
            cmd += ["--max-threads", str(a.max_threads)]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode not in (0, 2):
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
