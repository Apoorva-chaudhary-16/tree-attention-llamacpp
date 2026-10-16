#!/usr/bin/env python3
"""Runs both tests on every sample document and writes results.md / results.csv for the slide.

For each sample (parsers/markdown/samples/index.json):
  1. parse it into doc.json  (tokens via llama-tokenize)
  2. TEST 1  change section B (an unrelated section), ask about section A: the greedy answer must be identical.
     CONTROL ask about B itself on both versions: the answers should DIFFER (proves the test can fail).
  3. TEST 2  evict B's whole subtree (all its sequence ids, by position range), ask about A again:
     the answer must be unchanged and the KV state must shrink.

Run from the llama.cpp folder:
  python3 tests/tree/run_tests.py --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Offline check of parser + statistics only (no model needed):
  python3 tests/tree/run_tests.py --dry-run --fake-tokenizer
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "parsers", "markdown"))

from parse_markdown import Tokenizer, parse_markdown, perturb, doc_title_of  # noqa: E402
from stats import doc_stats  # noqa: E402


def find_sid(doc, title):
    for nd in doc["nodes"]:
        if nd["title"] == title or title in nd["contains"]:
            return nd
    raise KeyError(title)


def related(doc, a):
    """sids that are a itself, an ancestor of a, or a descendant of a"""
    by = {nd["sid"]: nd for nd in doc["nodes"]}
    out = {a["sid"]}
    p = a["parent_sid"]
    while p in by:
        out.add(p)
        p = by[p]["parent_sid"]
    for nd in doc["nodes"]:
        q = nd["sid"]
        while q in by:
            if q == a["sid"]:
                out.add(nd["sid"])
                break
            q = by[q]["parent_sid"]
    return out


def subtree_sids(doc, b):
    by = {nd["sid"]: nd for nd in doc["nodes"]}
    out = []
    for nd in doc["nodes"]:
        q = nd["sid"]
        while q in by:
            if q == b["sid"]:
                out.append(nd["sid"])
                break
            q = by[q]["parent_sid"]
    return out


def core(a):
    """The actual answer = text before the first blank-line gap (the driver prints newlines as 2+ spaces)."""
    return re.split(r"\s{2,}", (a or "").strip())[0].strip()


def upto(answer, expected):
    """The answer text up to and including the first occurrence of the expected value (None if it is absent).
    Everything the model writes after the answer is filler and can differ by rounding."""
    a = answer or ""
    i = a.lower().find(expected.lower())
    return None if i < 0 else a[: i + len(expected)]


def same_through_answer(a, b, expected):
    ua, ub = upto(a, expected), upto(b, expected)
    return ua is not None and ua == ub


def verdict(exact, same_core):
    return "PASS (exact)" if exact else ("PASS (through answer)" if same_core else "FAIL")


def run_driver(args, path, sid, question, evict=None):
    cmd = [args.driver, "-m", args.model, "--doc", path, "--ask-seq", str(sid), "-q", question,
           "-n", str(args.n_predict), "-t", str(args.threads)]
    if evict:
        sids, p0 = evict
        cmd += ["--evict-seq", ",".join(map(str, sids)), "--evict-p0", str(p0)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("driver failed (%s)\n%s" % (" ".join(cmd), r.stderr[-1500:]))
    o = r.stdout
    g = lambda rx, cast=str: (cast(re.search(rx, o).group(1)) if re.search(rx, o) else None)
    return dict(
        answer1=g(r"ANSWER1:(.*)"), answer2=g(r"ANSWER2:(.*)"),
        prefill_s=g(r"PREFILL .*time=([\d.]+)s", float), answer_s=g(r"TIME answer1=([\d.]+)s", float),
        kv_before=g(r"state_bytes_before=(\d+)", int), kv_after=g(r"state_bytes_after=(\d+)", int),
        evict=g(r"EVICT_TEST (\w+)"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="build/bin/llama-treedriver")
    ap.add_argument("--tokenizer", default="build/bin/llama-tokenize")
    ap.add_argument("--model", default="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--samples", default=os.path.join(HERE, "..", "..", "parsers", "markdown", "samples"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--min-tokens", type=int, default=24)
    ap.add_argument("-n", "--n-predict", type=int, default=16)
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--fake-tokenizer", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="parse + statistics only, do not run the model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tk = Tokenizer(args.tokenizer, args.model, args.fake_tokenizer)
    items = json.load(open(os.path.join(args.samples, "index.json")))
    rows, ok_all = [], True

    for it in items:
        stem = os.path.splitext(it["file"])[0]
        md = open(os.path.join(args.samples, it["file"]), encoding="utf-8").read()
        title = doc_title_of(md, it["file"])
        doc = parse_markdown(md, tk, title, args.min_tokens)
        A = find_sid(doc, it["target"])
        cand = [nd for nd in doc["nodes"] if nd["sid"] not in related(doc, A) and nd["sid"] != 0]
        if not cand:
            print("%s: no unrelated section to change, skipping" % it["file"])
            continue
        B = max(cand, key=lambda nd: nd["own_tokens"])
        md2 = perturb(md, B["title"])
        doc2 = parse_markdown(md2, tk, title, args.min_tokens)
        A2, B2 = find_sid(doc2, it["target"]), find_sid(doc2, B["title"])
        p1, p2 = os.path.join(args.out, stem + ".json"), os.path.join(args.out, stem + "_changedB.json")
        json.dump(doc, open(p1, "w"))
        json.dump(doc2, open(p2, "w"))

        st = doc_stats(doc)
        ev = subtree_sids(doc, B)
        freed = B["subtree_tokens"]
        row = dict(doc=it["file"], A=it["target"], B=B["title"], tokens=st["tokens"], depth=st["depth"],
                   sequences=st["sequences"], sub_min=st["sub_min"], sub_med=st["sub_med"], sub_max=st["sub_max"],
                   markup_pct=round(st["markup_pct"], 1), pairs_saved_pct=round(st["pairs_saved_pct"], 1),
                   evicted_tokens=freed)
        print("\n== %s  (ask about '%s' [seq %d], change/evict '%s' [seqs %s])" % (it["file"], it["target"], A["sid"], B["title"], ev))
        if not args.dry_run:
            r1 = run_driver(args, p1, A["sid"], it["question"], evict=(ev, B["start"]))
            r2 = run_driver(args, p2, A2["sid"], it["question"])
            cq = "What is the last sentence of this section?"
            c1 = run_driver(args, p1, B["sid"], cq)
            c2 = run_driver(args, p2, B2["sid"], cq)
            t1_exact = r1["answer1"] == r2["answer1"]
            t1 = same_through_answer(r1["answer1"], r2["answer1"], it["expected"])
            t2_exact = r1["answer1"] == r1["answer2"]
            t2 = same_through_answer(r1["answer1"], r1["answer2"], it["expected"]) and r1["kv_after"] < r1["kv_before"]
            ctl = c1["answer1"] != c2["answer1"]
            correct = it["expected"].lower() in (r1["answer1"] or "").lower()
            drop = 100.0 * (1 - r1["kv_after"] / r1["kv_before"])
            row.update(test1=verdict(t1_exact, t1), test2=verdict(t2_exact, t2),
                       control="differs" if ctl else "SAME", answer_correct="yes" if correct else "no",
                       kv_before=r1["kv_before"], kv_after=r1["kv_after"], kv_drop_pct=round(drop, 1),
                       prefill_s=r1["prefill_s"], answer_s=r1["answer_s"])
            ok_all &= t1 and t2
            print("   answer A :", (r1["answer1"] or "").strip()[:90])
            print("   test 1 (change B, A unchanged): %s | control: %s | test 2 (evict B): %s | KV %d -> %d (-%.1f%%) | prefill %.2fs, answer %.2fs"
                  % (row["test1"], row["control"], row["test2"], r1["kv_before"], r1["kv_after"], drop,
                     r1["prefill_s"], r1["answer_s"]))
            if not t1_exact:
                print("   [test 1] A orig   :", r1["answer1"], "\n            A changed:", r2["answer1"])
            if not t2_exact:
                print("   [test 2] before evict:", r1["answer1"], "\n            after evict :", r1["answer2"])
            if not ctl:
                print("   [control] B orig   :", c1["answer1"], "\n             B changed:", c2["answer1"])
        else:
            print("   tokens=%d sequences=%d depth=%d markup=%.1f%% pairs saved=%.1f%%"
                  % (st["tokens"], st["sequences"], st["depth"], st["markup_pct"], st["pairs_saved_pct"]))
        rows.append(row)

    if not rows:
        return 1
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print("\nwrote %s/results.csv and results.md" % args.out)
    if not args.dry_run:
        print("ALL TESTS PASS" if ok_all else "SOME TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
