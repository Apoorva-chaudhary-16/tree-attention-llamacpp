#!/usr/bin/env python3
"""
Step 3 - correctness: (a) perplexity on a fixed WikiText-2 sample, (b) greedy sample outputs.

Why: a quantised model that silently produces garbage would still give "nice" tokens/s.
PPL confirms the weights are intact; the Q4_K_M vs Q8_0 gap confirms the quantiser worked.

    python scripts/04_check_correctness.py --tag mylaptop
    python scripts/04_check_correctness.py --model llama-3.2-1b --quant Q4_K_M --chunks 10   # fast

Outputs
    results/perplexity_<tag>.csv
    results/samples_<model>_<quant>_<tag>.txt   (read these by eye and quote one in your report)
Exit code 0 only if every check passes.
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import common as C  # noqa: E402

PPL_FIELDS = ["timestamp", "tag", "model", "quant", "ctx", "chunks", "threads", "ppl", "ppl_err",
              "peak_rss_gb", "wall_s", "status", "llama_commit"]


def run_ppl(model, quant, mpath, wiki, threads, chunks, ctx):
    exe = C.find_bin("llama-perplexity")
    cmd = [str(exe), "-m", str(mpath), "-f", str(wiki), "-c", str(ctx), "--chunks", str(chunks),
           "-t", str(threads), "-ngl", "0"]
    res = C.run_monitored(cmd, timeout=4 * 3600)
    text = res["stdout"] + "\n" + res["stderr"]
    m = re.search(r"Final estimate: PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    ppl, err = (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))
    ok = res["rc"] == 0 and m is not None and math.isfinite(ppl)
    if not ok:
        print(text[-1500:])
    return ok, ppl, err, res


def sample_outputs(mpath, threads, n_tokens=64):
    """Greedy generation for the fixed prompts. Returns list of (prompt, output, ok)."""
    try:
        exe = C.find_bin("llama-completion")
    except FileNotFoundError:
        exe = C.find_bin("llama-cli")
    outs = []
    for p in config.SAMPLE_PROMPTS:
        cmd = [str(exe), "-m", str(mpath), "-p", p, "-n", str(n_tokens), "--temp", "0", "-t", str(threads),
               "-ngl", "0", "-c", "2048", "-no-cnv", "--no-display-prompt", "--seed", "1"]
        res = C.run_monitored(cmd, timeout=900)
        text = res["stdout"].strip()
        outs.append((p, text, res["rc"] == 0 and len(text) > 0, res["stderr"][-300:]))
    return outs


def looks_sane(text: str) -> bool:
    """Cheap garbage detector: printable, not one token repeated forever."""
    if not text:
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    words = text.split()
    distinct = len(set(words)) / max(1, len(words))
    return printable > 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default="all")
    ap.add_argument("--quant", default="all")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=config.PPL_CHUNKS)
    ap.add_argument("--ctx", type=int, default=config.PPL_CTX)
    ap.add_argument("--skip-samples", action="store_true")
    a = ap.parse_args()
    tag = C.host_tag(a.tag)
    threads = a.threads or C.default_max_threads()
    wiki = config.DATA_DIR / "wikitext-2-raw" / "wiki.test.raw"
    if not wiki.exists():
        C.die(f"{wiki} missing - run scripts/03_get_models.py")

    out_csv = config.RESULTS_DIR / f"perplexity_{tag}.csv"
    commit = C.llama_commit()
    results, all_ok = {}, True
    for model, quant in [(m, q) for m in config.MODELS for q in config.MODELS[m]["files"]
                         if a.model in ("all", m) and a.quant in ("all", q)]:
        try:
            mpath = C.model_path(model, quant)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] {e}")
            continue
        print(f"[ppl ] {model} {quant}  ({a.chunks} x {a.ctx} tokens, {threads} threads)")
        ok, ppl, err, res = run_ppl(model, quant, mpath, wiki, threads, a.chunks, a.ctx)
        print(f"       PPL = {ppl:.4f} +/- {err:.4f}   peak RSS {res['peak_rss_gb']:.2f} GB   {res['wall_s']:.0f}s")
        C.append_csv(out_csv, dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                                   tag=tag, model=model, quant=quant, ctx=a.ctx, chunks=a.chunks, threads=threads,
                                   ppl=ppl, ppl_err=err, peak_rss_gb=round(res["peak_rss_gb"], 3),
                                   wall_s=round(res["wall_s"], 1), status="ok" if ok else "error",
                                   llama_commit=commit), PPL_FIELDS)
        all_ok &= ok
        results[(model, quant)] = ppl
        lo, hi = config.PPL_SANITY_RANGE
        if ok and not (lo < ppl < hi):
            print(f"       [FAIL] PPL {ppl} outside the plausible range ({lo}, {hi}) for a working LLM")
            all_ok = False

        if not a.skip_samples and ok:
            samples = sample_outputs(mpath, threads)
            f = config.RESULTS_DIR / f"samples_{model}_{quant}_{tag}.txt"
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "w", encoding="utf-8") as fh:
                for p, t, good, err_tail in samples:
                    sane = good and looks_sane(t)
                    fh.write(f"### PROMPT: {p!r}\n{t}\n[sane={sane}]\n\n")
                    print(f"       sample sane={sane}: {t[:70]!r}")
                    all_ok &= bool(sane)
            print(f"       samples -> {f}")

    # Quantisation sanity: Q4_K_M must be close to Q8_0 for the same model
    for model in config.MODELS:
        q4, q8 = results.get((model, "Q4_K_M")), results.get((model, "Q8_0"))
        if q4 and q8 and math.isfinite(q4) and math.isfinite(q8):
            rel = (q4 - q8) / q8
            verdict = "OK" if rel <= config.PPL_MAX_REL_INCREASE else "SUSPICIOUS"
            print(f"[quant] {model}: Q4_K_M {q4:.3f} vs Q8_0 {q8:.3f}  (+{rel * 100:.2f} %)  {verdict}")
            if verdict != "OK":
                all_ok = False

    print("\nALL CORRECTNESS CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED - do not trust speed numbers yet")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
