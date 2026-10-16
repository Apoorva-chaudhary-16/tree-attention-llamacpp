#!/usr/bin/env python3
"""Tree statistics for the one-slide report.

  python3 stats.py tests/tree/out/*.json

Per document: tokens, tree depth, number of sequences (merged subtrees), subtree sizes (min/median/max,
non-root nodes), share of tokens that are pure markup/syntax, and how many query-key pairs structured
attention needs compared with ordinary full causal attention.
"""
import json
import statistics
import sys
from bisect import bisect_right


def doc_stats(doc):
    n = len(doc["tokens"])
    nodes = doc["nodes"]
    subs = [nd["subtree_tokens"] for nd in nodes if nd["sid"] != 0] or [0]
    per = {}                                    # sid -> sorted positions of cells that carry it
    for pos, ids in zip(doc["positions"], doc["seq_ids"]):
        for s in ids:
            per.setdefault(s, []).append(pos)
    for v in per.values():
        v.sort()
    structured = 0
    for pos, ids in zip(doc["positions"], doc["seq_ids"]):
        structured += bisect_right(per[ids[0]], pos)      # earlier-or-equal cells visible to this token
    full = n * (n + 1) // 2
    return dict(
        tokens=n,
        depth=max(nd["depth"] for nd in nodes),
        heading_depth=doc["meta"]["max_heading_depth"],
        sequences=len(nodes),
        sub_min=min(subs), sub_med=int(statistics.median(subs)), sub_max=max(subs),
        markup_pct=100.0 * sum(doc["is_markup"]) / n,
        pairs_full=full, pairs_tree=structured,
        pairs_saved_pct=100.0 * (1 - structured / full),
    )


HEADER = ("| doc | tokens | depth | sequences | subtree sizes min/med/max | markup tokens | attention pairs saved |\n"
          "|---|---|---|---|---|---|---|")


def row(name, s):
    return "| %s | %d | %d | %d | %d / %d / %d | %.1f%% | %.1f%% |" % (
        name, s["tokens"], s["depth"], s["sequences"], s["sub_min"], s["sub_med"], s["sub_max"],
        s["markup_pct"], s["pairs_saved_pct"])


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return
    print(HEADER)
    for f in files:
        with open(f) as fh:
            print(row(f.split("/")[-1], doc_stats(json.load(fh))))


if __name__ == "__main__":
    main()
