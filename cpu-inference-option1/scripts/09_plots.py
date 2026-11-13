#!/usr/bin/env python3
"""
Required plots (project deliverables) from the CSVs written by the other scripts.

    python scripts/09_plots.py --tag mylaptop            # one machine
    python scripts/09_plots.py --tag mylaptop --compare rudra   # overlay a second machine where meaningful

Writes PNGs to results/plots/: decode_vs_context, prefill_vs_prompt, thread_scaling,
operator_breakdown, predicted_vs_measured, (+ experiment_<name> if experiment rows exist).
Rows flagged swap_flag or error are dropped; error bars = standard deviation of the 3 repetitions.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load(tag):
    p = config.RESULTS_DIR / f"bench_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d["status"] == "ok"].copy()
    d["swap_flag"] = d["swap_flag"].astype(str).str.lower() == "true"
    d = d[~d["swap_flag"]]
    d["flags"] = d["flags"].fillna("")
    return d


def save(fig, name):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PLOTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def series_label(r, tags):
    return f"{r['model']} {r['quant']}" + (f" ({r['tag']})" if len(tags) > 1 else "")


def line_plot(dfs, x, title, xlabel, name, filt, logx=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = 0
    for d in dfs:
        d = d[filt(d)]
        for (m, q, t), g in d.groupby(["model", "quant", "tag"]):
            g = g.sort_values(x)
            ax.errorbar(g[x], g["ts_mean"], yerr=g["ts_std"], marker="o", capsize=3, label=f"{m} {q} [{t}]")
            n += 1
    if not n:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel); ax.set_ylabel("tokens / s"); ax.set_title(title)
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", default="", help="second tag to overlay")
    a = ap.parse_args()
    import common as C
    tag = C.host_tag(a.tag)
    tags = [tag] + ([a.compare] if a.compare else [])
    dfs = [load(t) for t in tags]
    base = lambda d: d["exp"] == "baseline"

    line_plot(dfs, "depth", "Decode speed vs context length", "context length (tokens)", "decode_vs_context",
              lambda d: base(d) & (d["sweep"] == "decode"), logx=False)
    line_plot(dfs, "n_prompt", "Prefill speed vs prompt length", "prompt length (tokens)", "prefill_vs_prompt",
              lambda d: base(d) & (d["sweep"] == "prefill"))

    # thread scaling (pp512 and tg128 side by side)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    drawn = False
    for d in dfs:
        t = d[base(d) & ((d["sweep"] == "threads") | ((d["sweep"] == "prefill") & (d["n_prompt"] == 512)) | ((d["sweep"] == "decode") & (d["depth"] == 0)))]
        t = t.sort_values("timestamp").drop_duplicates(["model", "quant", "tag", "n_prompt", "n_gen", "threads"], keep="last")
        for ax, col, ttl in ((axs[0], t["n_prompt"] > 0, "prefill (pp512)"), (axs[1], t["n_gen"] > 0, "decode (tg128)")):
            for (m, q, tg), g in t[col].groupby(["model", "quant", "tag"]):
                g = g.sort_values("threads")
                ax.errorbar(g["threads"], g["ts_mean"], yerr=g["ts_std"], marker="o", capsize=3, label=f"{m} {q} [{tg}]")
                if len(g):
                    ideal = g["ts_mean"].iloc[0] * g["threads"] / g["threads"].iloc[0]
                    ax.plot(g["threads"], ideal, "k:", alpha=.4)
                    drawn = True
            ax.set_title(f"Thread scaling - {ttl}"); ax.set_xlabel("threads"); ax.set_ylabel("tokens / s")
            ax.grid(alpha=.3)
    if drawn:
        axs[0].legend(fontsize=8)
        save(fig, "thread_scaling")
    else:
        plt.close(fig)

    # operator breakdown (stacked share per decode depth + prefill)
    p = config.RESULTS_DIR / f"ops_summary_{tag}.csv"
    if p.exists():
        o = pd.read_csv(p)
        o = o[o["exp"] == "baseline"]
        for (m, q), g in o.groupby(["model", "quant"]):
            piv = g.assign(lab=g["phase"] + " d=" + g["depth"].astype(str)).pivot_table(
                index="lab", columns="category", values="share", aggfunc="sum").fillna(0)
            order = sorted(piv.index, key=lambda s: (s.startswith("prefill"), int(s.split("d=")[1])))
            piv = piv.loc[order]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_ylabel("share of operator time"); ax.set_xlabel(""); ax.set_title(f"Operator time breakdown - {m} {q}")
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            save(fig, f"operator_breakdown_{m}_{q}")

    # predicted vs measured
    p = config.RESULTS_DIR / f"roofline_compare_{tag}.csv"
    if p.exists():
        r = pd.read_csv(p)
        r = r[(r["exp"] == "baseline") & (r["sweep"] == "decode")]
        if len(r):
            fig, ax = plt.subplots(figsize=(5.8, 5.4))
            for (m, q), g in r.groupby(["model", "quant"]):
                ax.scatter(g["pred_tg"], g["meas_tg"], s=18 + g["depth"] / 400, label=f"{m} {q}")
            hi = max(r["pred_tg"].max(), r["meas_tg"].max()) * 1.1
            ax.plot([0, hi], [0, hi], "k--", lw=1, label="measured = predicted")
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
            ax.set_xlabel("predicted tokens/s (bandwidth / bytes per token)"); ax.set_ylabel("measured tokens/s")
            ax.set_title("Roofline: predicted vs measured decode"); ax.grid(alpha=.3); ax.legend(fontsize=8)
            save(fig, "predicted_vs_measured")

    # experiments vs baseline (decode)
    d = dfs[0]
    for exp in sorted(set(d["exp"]) - {"baseline", "reference", "quickcheck"}):
        e = d[d["exp"] == exp]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sw = e["sweep"].iloc[0]
        x = {"decode": "depth", "prefill": "n_prompt", "threads": "threads"}.get(sw, "depth")
        for label, df in (("baseline", d[(d["exp"] == "baseline") & (d["sweep"] == sw) & d["model"].isin(e["model"]) & d["quant"].isin(e["quant"])]), (exp, e)):
            df = df[df["n_gen"] > 0] if sw != "prefill" else df
            for (m, q), g in df.groupby(["model", "quant"]):
                g = g.sort_values(x)
                ax.errorbar(g[x], g["ts_mean"], yerr=g["ts_std"], marker="o", capsize=3, label=f"{label}: {m} {q}")
        ax.set_xlabel(x); ax.set_ylabel("tokens / s"); ax.set_title(f"Experiment: {exp}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        save(fig, f"experiment_{exp}")


if __name__ == "__main__":
    main()
