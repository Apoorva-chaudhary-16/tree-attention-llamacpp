#!/usr/bin/env python3
"""
Steps 4 and 7 - roofline: predict decode speed BEFORE measuring, then compare.

    predicted tokens/s = measured memory bandwidth / bytes read per decoded token
    bytes per token    = weights streamed once + KV cache (grows with context)

Usage
    python scripts/05_roofline.py --tag mylaptop            # prediction table (needs stream_<tag>.json + models)
    python scripts/05_roofline.py --tag mylaptop --compare  # + join with results/bench_<tag>.csv

Outputs
    results/roofline_predict_<tag>.csv   prediction for every model/quant/context
    results/roofline_compare_<tag>.csv   predicted vs measured, efficiency = measured / predicted
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402
import gguf_meta as G  # noqa: E402


def load_bw(tag, source, bw_override):
    if bw_override:
        return {"peak": bw_override, "per_thread": {}}
    p = config.RESULTS_DIR / f"stream_{tag}.json"
    if not p.exists():
        C.die(f"{p} missing - run scripts/02_stream.py --tag {tag} (or pass --bw-gbs)")
    j = json.loads(p.read_text())
    key = "Triad" if source == "triad" else "Copy"
    per = {int(t): v[key] for t, v in j["per_thread_gbs"].items() if key in v}
    return {"peak": max(per.values()), "per_thread": per}


def parse_kv_types(flags: str):
    k = re.search(r"-ctk=(\S+)", flags or "")
    v = re.search(r"-ctv=(\S+)", flags or "")
    return (k.group(1) if k else "f16"), (v.group(1) if v else "f16")


def bw_for_threads(bw, t, use_threads):
    if not use_threads or not bw["per_thread"]:
        return bw["peak"]
    ks = sorted(bw["per_thread"])
    below = [k for k in ks if k <= t]
    return bw["per_thread"][below[-1]] if below else bw["per_thread"][ks[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--bw-gbs", type=float, default=0, help="override bandwidth (GB/s)")
    ap.add_argument("--bw-source", choices=["triad", "copy"], default="triad")
    ap.add_argument("--per-thread-bw", action="store_true",
                    help="use STREAM bandwidth at the same thread count as the run (default: machine peak)")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--bench-csv", default="")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)
    bw = load_bw(tag, a.bw_source, a.bw_gbs)
    print(f"memory bandwidth used: {bw['peak']:.1f} GB/s ({a.bw_source}, peak over thread counts)")

    pred_rows, metas = [], {}
    for model, m in config.MODELS.items():
        for quant in m["files"]:
            try:
                path = C.model_path(model, quant)
            except (FileNotFoundError, KeyError):
                continue
            meta = G.read_meta(path)
            metas[(model, quant)] = meta
            w = G.bytes_read_per_token(meta)
            kv_tok = G.kv_bytes_per_token(meta)
            cross = w["weights_read_bytes"] / kv_tok          # context where KV bytes == weight bytes
            deepest = m["max_ctx"] - config.N_GEN
            print(f"\n{model} {quant}: weights read/token = {w['weights_read_bytes'] / 1e9:.3f} GB "
                  f"(file {meta['file_bytes'] / 1e9:.3f} GB, tied embeddings={meta['tied_embeddings']}), "
                  f"KV = {kv_tok / 1024:.1f} KiB per context token, "
                  f"KV bytes == weight bytes at ~{cross:,.0f} tokens")
            for d in sorted({0, 1024, 2048, 4096, 8192, deepest}):
                if d > deepest:
                    continue
                avg_ctx = d + config.N_GEN / 2
                kv = kv_tok * avg_ctx
                tot = w["weights_read_bytes"] + kv
                pred = bw["peak"] * 1e9 / tot
                pred_rows.append(dict(model=model, quant=quant, depth=d, weights_gb=w["weights_read_bytes"] / 1e9,
                                      kv_gb=kv / 1e9, total_gb=tot / 1e9, kv_share=kv / tot,
                                      bw_gbs=bw["peak"], pred_tg=pred))
                print(f"   depth {d:6d}: {tot / 1e9:6.3f} GB/token  -> predicted {pred:7.1f} tok/s  (KV share {kv / tot * 100:4.1f} %)")
    out = config.RESULTS_DIR / f"roofline_predict_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if pred_rows:
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
            wr.writeheader()
            wr.writerows(pred_rows)
        print(f"\nprediction table -> {out}")

    if not a.compare:
        return
    bench = Path(a.bench_csv) if a.bench_csv else config.RESULTS_DIR / f"bench_{tag}.csv"
    rows = [r for r in C.read_csv(bench) if r["status"] == "ok" and r["n_gen"] != "0" and r["n_prompt"] == "0"]
    if not rows:
        C.die(f"no decode rows in {bench}")
    cmp_rows = []
    for r in rows:
        key = (r["model"], r["quant"])
        if key not in metas:
            continue
        meta = metas[key]
        kt, vt = parse_kv_types(r["flags"])
        w = G.bytes_read_per_token(meta)["weights_read_bytes"]
        n_gen, depth = int(r["n_gen"]), int(r["depth"])
        kv = G.kv_bytes_per_token(meta, kt, vt) * (depth + n_gen / 2)
        bwv = bw_for_threads(bw, int(r["threads"]), a.per_thread_bw)
        pred = bwv * 1e9 / (w + kv)
        meas = float(r["ts_mean"])
        cmp_rows.append(dict(tag=r["tag"], exp=r["exp"], model=r["model"], quant=r["quant"], sweep=r["sweep"],
                             depth=depth, threads=int(r["threads"]), flags=r["flags"], bytes_per_token_gb=(w + kv) / 1e9,
                             bw_gbs=bwv, pred_tg=round(pred, 2), meas_tg=meas, meas_std=r["ts_std"],
                             efficiency=round(meas / pred, 3),
                             achieved_gbs=round(meas * (w + kv) / 1e9, 2), peak_rss_gb=r["peak_rss_gb"]))
    cmp_out = config.RESULTS_DIR / f"roofline_compare_{tag}.csv"
    with open(cmp_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
        wr.writeheader()
        wr.writerows(cmp_rows)
    print(f"\npredicted vs measured -> {cmp_out}")
    print(f"{'model':14s} {'quant':7s} {'depth':>6s} {'thr':>3s}  {'pred':>8s} {'meas':>8s}  eff   achieved GB/s")
    for r in cmp_rows:
        if r["exp"] == "baseline" and r["sweep"] == "decode":
            print(f"{r['model']:14s} {r['quant']:7s} {r['depth']:6d} {r['threads']:3d}  {r['pred_tg']:8.1f} "
                  f"{r['meas_tg']:8.1f}  {r['efficiency']:.2f}  {r['achieved_gbs']:.1f}")
    print("\nHow to read it: efficiency = measured / predicted. ~0.7-0.9 means decode is bandwidth-bound; "
          "much lower means compute, threading or cache effects (see profiling).")


if __name__ == "__main__":
    main()
